# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Basin Size Study - Local Optimization Convergence Analysis
#
# This notebook investigates the **basin of attraction** of the lightcurve inversion
# objective function by systematically measuring the success rate of local optimization
# from starting points at various distances from the true solution.
#
# ## Goals
#
# 1. Implement a simple local optimization wrapper using L-BFGS-B
# 2. Generate random starting points within specified radii of the true solution
# 3. Measure success rate vs. initialization radius
# 4. Identify the critical radius where local optimization reliability drops
#
# This analysis informs whether global search is necessary or if local optimization
# suffices when initialized "close enough" to the true solution.

# %% [markdown]
# ---
# ## Setup
#
# Copy minimal setup from notebook 04 to generate the same test case.

# %%
import sys
from pathlib import Path
import time
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Project Root
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Import config and IO modules
from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader

# Import SPICE handler
from src.spice.spice_handler import SpiceHandler

# Import computation modules
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import compute_inertia_from_config

# Import articulation module for fixed component angles
from src.articulation import compute_rotation_matrices_from_angles

# Import dynamics and inversion modules
from src.dynamics import propagate_attitude
from src.inversion import (
    ObjectiveFunction,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    normalize_quaternion,
)

# Import for forward model
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

print(f"Project root: {PROJECT_ROOT}")
print("Imports successful!")

# %% [markdown]
# ---
# ## 1. Load Intelsat 901 Configuration (Same as Notebook 04)

# %%
# ============================================================================
# CONFIGURATION
# ============================================================================
config_path = "intelsat_901/intelsat_901_config.yaml"

# Number of observation points (same as notebook 04)
n_observations = 50

# Observer/Ground Station SPICE ID
OBSERVER_ID = 399999

# Load RSO configuration
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config(config_path)

# Get paths from configuration
metakernel_path = config_manager.get_metakernel_path(config)
output_dir = config_manager.get_output_directory(config)

# Use configuration values
satellite_id = config.spice_config.satellite_id
start_time_utc = config.simulation_defaults.start_time
end_time_utc = config.simulation_defaults.end_time

print("=" * 70)
print("BASIN SIZE STUDY - Local Optimization Convergence Analysis")
print("=" * 70)
print(f"\nConfiguration: {config.name}")
print(f"Observations: {n_observations}")

# %%
# Load satellite model from STL files
print(f"\nLoading {config.name} satellite model...")
satellite = STLLoader.create_satellite_from_stl_config(
    config=config,
    config_manager=config_manager
)

# Set up BRDF materials
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

print(f"Model loaded: {satellite.name}")
print(f"  Components: {len(satellite.components)}")
total_facets = sum(len(comp.facets) for comp in satellite.components if comp.facets)
print(f"  Total facets: {total_facets:,}")

# %% [markdown]
# ---
# ## 2. Fixed Articulation Configuration

# %%
# Same fixed angles as notebook 04
SOLAR_PANEL_ANGLE_DEG = 0.0
ANTENNA_DISH_ANGLE_DEG = 15.0

print("Fixed articulation configuration:")
print(f"  Solar panels (SP_North, SP_South): {SOLAR_PANEL_ANGLE_DEG}")
print(f"  Antenna dishes (AD_East, AD_West): {ANTENNA_DISH_ANGLE_DEG}")

# %% [markdown]
# ---
# ## 3. Calculate Inertia Tensor

# %%
# Same component masses as notebook 04
component_masses = {
    'Bus': 1532.0,
    'SP_North': 170.0,
    'SP_South': 170.0,
    'AD_East': 50.0,
    'AD_West': 50.0,
}

# Calculate inertia tensor
inertia_result = compute_inertia_from_config(
    config=config,
    config_manager=config_manager,
    masses=component_masses,
    articulation_angles={'SP_North': 0.0, 'SP_South': 0.0}
)

inertia_tensor = inertia_result.inertia_tensor
print(f"Inertia tensor computed:")
print(f"  Principal moments: {inertia_result.principal_moments}")

# %% [markdown]
# ---
# ## 4. Initialize SPICE and Compute Observation Geometry

# %%
# Initialize SPICE
print("Initializing SPICE...")
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))

# Generate time series
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, n_observations)

print(f"Time range: {start_time_utc} to {end_time_utc}")
print(f"Observations: {n_observations}")

# Relative observation times starting from 0
observation_times = epochs - epochs[0]

# %%
# Compute observation geometry using SPICE
print("Computing observation geometry...")
geometry_data = compute_observation_geometry(
    epochs=epochs,
    satellite_id=satellite_id,
    observer_id=OBSERVER_ID,
    spice_handler=spice_handler,
    config=config
)

# Extract J2000 positions
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

print(f"Geometry computed: {n_observations} observations")
print(f"Observer distance range: {observer_distances.min():.0f} - {observer_distances.max():.0f} km")

# %%
# Create fixed articulation matrices
fixed_articulation_angles = {
    'SP_North': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'SP_South': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'AD_East': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
    'AD_West': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
}

articulation_matrices = compute_rotation_matrices_from_angles(
    fixed_articulation_angles, satellite
)

print("Articulation matrices created")

# %% [markdown]
# ---
# ## 5. Define True Attitude Parameters (Same as Notebook 04)

# %%
# True initial quaternion (same as notebook 04)
true_axis = np.array([0.6, 0.3, 0.8])
true_axis /= np.linalg.norm(true_axis)
true_angle_deg = 45.0
true_angle_rad = np.deg2rad(true_angle_deg)

# Construct quaternion (scalar-first: w, x, y, z)
true_q0 = np.array([
    np.cos(true_angle_rad / 2),
    np.sin(true_angle_rad / 2) * true_axis[0],
    np.sin(true_angle_rad / 2) * true_axis[1],
    np.sin(true_angle_rad / 2) * true_axis[2],
])

# True initial angular velocity (rad/s in body frame)
true_omega_deg_per_s = np.array([0.005, -0.003, 0.05])
true_omega0 = np.deg2rad(true_omega_deg_per_s)

# Convert to axis-angle representation
true_axis_angle = quaternion_to_axis_angle(true_q0)

# Store as combined parameter vector for optimization
# Format: [axis_angle_x, axis_angle_y, axis_angle_z, omega_x, omega_y, omega_z]
true_params = np.concatenate([true_axis_angle, true_omega0])

print("True attitude parameters:")
print(f"  Axis-angle (rad): {true_axis_angle}")
print(f"  Axis-angle (deg): {np.rad2deg(true_axis_angle)}")
print(f"  Angular velocity (deg/s): {np.rad2deg(true_omega0)}")
print(f"\nCombined parameter vector:")
print(f"  true_params = {true_params}")

# %% [markdown]
# ---
# ## 6. Generate Synthetic Lightcurve

# %%
print("\nGenerating synthetic lightcurve with true parameters...")
print("  Mode: TUMBLING with inertia tensor")
print("  Shadows: ENABLED")

# Propagate true attitude using tumbling dynamics
true_quaternions, true_omega_history = propagate_attitude(
    q0=true_q0,
    omega0=true_omega0,
    times=observation_times,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
)

print(f"  Attitude propagated: {len(true_quaternions)} epochs")

# %%
# Create a temporary ObjectiveFunction to compute body-frame vectors
objective_temp = ObjectiveFunction(
    satellite=satellite,
    observation_times=observation_times,
    observed_lightcurve=np.zeros(n_observations),  # placeholder
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=articulation_matrices,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
)

# Get body-frame vectors from propagated attitude
k1_vectors, k2_vectors = objective_temp._compute_body_frame_vectors(true_quaternions)

# %%
# Compute shadows with ray tracing
print("Computing shadows...")
lit_status_dict = compute_shadows(
    satellite=satellite,
    k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices,
    show_progress=True,
)

# %%
# Generate true lightcurve
print("Generating lightcurve...")
true_lightcurve, total_flux, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict,
    k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors,
    observer_distances=observer_distances,
    satellite=satellite,
    epochs=epochs,
    pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False,
    animate=False,
    show_progress=True,
)

print(f"True lightcurve range: [{true_lightcurve.min():.2f}, {true_lightcurve.max():.2f}] mag")

# %%
# Add synthetic noise (same as notebook 04)
np.random.seed(42)
noise_sigma = 0.05
observed_lightcurve = true_lightcurve + np.random.normal(0, noise_sigma, n_observations)

print(f"\nAdded Gaussian noise (sigma = {noise_sigma} mag)")
print(f"Observed lightcurve range: [{observed_lightcurve.min():.2f}, {observed_lightcurve.max():.2f}] mag")

# %% [markdown]
# ---
# ## 7. Create ObjectiveFunction for Optimization

# %%
# Create the objective function for optimization
objective_fn = ObjectiveFunction(
    satellite=satellite,
    observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=True,  # Use full physics model
    articulation_matrices=articulation_matrices,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
)

# Verify the objective value at true parameters
obj_at_true = objective_fn.evaluate(true_params)
print(f"\nObjectiveFunction created for optimization")
print(f"  Objective value at true parameters: {obj_at_true:.6f}")

# Parameter names for reference
param_names = ['axis_angle_x', 'axis_angle_y', 'axis_angle_z', 'omega_x', 'omega_y', 'omega_z']

# %% [markdown]
# ---
# ## 8. Local Optimization Wrapper
#
# Implement `run_local_optimization(objective_fn, x0, bounds)` that uses L-BFGS-B
# with specified tolerances and returns optimization results including success status.

# %%
def run_local_optimization(
    objective_fn: ObjectiveFunction,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    ftol: float = 1e-8,
    gtol: float = 1e-6,
    maxiter: int = 1000,
) -> tuple[np.ndarray, float, bool, int]:
    """
    Run local optimization using L-BFGS-B.

    Parameters
    ----------
    objective_fn : ObjectiveFunction
        The objective function to minimize.
    x0 : np.ndarray
        Initial parameter guess (6 elements).
    bounds : list[tuple[float, float]]
        Parameter bounds for each of the 6 parameters.
    ftol : float
        Function tolerance for convergence.
    gtol : float
        Gradient tolerance for convergence.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    x_opt : np.ndarray
        Optimal parameters found.
    f_opt : float
        Final objective function value.
    success : bool
        Whether optimization converged successfully.
    n_evals : int
        Number of function evaluations.
    """
    # Track function evaluations
    n_evals = [0]

    def wrapped_objective(params: np.ndarray) -> float:
        n_evals[0] += 1

        # Normalize axis-angle via quaternion round-trip
        axis_angle = params[:3]
        omega = params[3:]

        q = axis_angle_to_quaternion(axis_angle)
        q_normalized = normalize_quaternion(q)
        axis_angle_norm = quaternion_to_axis_angle(q_normalized)

        params_normalized = np.concatenate([axis_angle_norm, omega])
        return objective_fn.evaluate(params_normalized)

    # Run L-BFGS-B optimization
    result = minimize(
        wrapped_objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "ftol": ftol,
            "gtol": gtol,
            "disp": False,
        },
    )

    # Normalize final parameters
    final_axis_angle = result.x[:3]
    final_omega = result.x[3:]

    q_final = axis_angle_to_quaternion(final_axis_angle)
    q_final_normalized = normalize_quaternion(q_final)
    final_axis_angle_normalized = quaternion_to_axis_angle(q_final_normalized)

    x_opt = np.concatenate([final_axis_angle_normalized, final_omega])

    return x_opt, result.fun, result.success, n_evals[0]


# %% [markdown]
# ---
# ## 9. Define Parameter Bounds and Success Criteria

# %%
# Define parameter bounds
omega_max_deg_per_s = 30.0
omega_max_rad_per_s = np.deg2rad(omega_max_deg_per_s)

# axis_angle bounds: [-pi, pi] for each component
# omega bounds: [-omega_max, omega_max] for each component
bounds = [
    (-np.pi, np.pi),  # axis_angle_x
    (-np.pi, np.pi),  # axis_angle_y
    (-np.pi, np.pi),  # axis_angle_z
    (-omega_max_rad_per_s, omega_max_rad_per_s),  # omega_x
    (-omega_max_rad_per_s, omega_max_rad_per_s),  # omega_y
    (-omega_max_rad_per_s, omega_max_rad_per_s),  # omega_z
]

print("Parameter bounds:")
for i, (name, (lb, ub)) in enumerate(zip(param_names, bounds)):
    if i < 3:
        print(f"  {name}: [{lb:.4f}, {ub:.4f}] rad = [{np.rad2deg(lb):.1f}, {np.rad2deg(ub):.1f}] deg")
    else:
        print(f"  {name}: [{lb:.6f}, {ub:.6f}] rad/s = [{np.rad2deg(lb):.2f}, {np.rad2deg(ub):.2f}] deg/s")

# %%
# Define success criteria
# Success requires BOTH conditions to be met:
# 1. Angular velocity error < 0.1 deg/s
# 2. RMS residual < 2 * noise_sigma

OMEGA_ERROR_THRESHOLD_DEG_PER_S = 0.1
RMS_THRESHOLD_FACTOR = 2.0


def evaluate_success(
    x_opt: np.ndarray,
    true_params: np.ndarray,
    objective_fn: ObjectiveFunction,
    noise_sigma: float,
) -> tuple[bool, float, float]:
    """
    Evaluate whether optimization was successful based on defined criteria.

    Parameters
    ----------
    x_opt : np.ndarray
        Optimized parameters.
    true_params : np.ndarray
        True parameters.
    objective_fn : ObjectiveFunction
        Objective function for computing residuals.
    noise_sigma : float
        Noise level in the observations.

    Returns
    -------
    success : bool
        True if both criteria are met.
    omega_error_deg : float
        Angular velocity error in deg/s.
    rms_residual : float
        RMS of residuals in magnitudes.
    """
    # Compute angular velocity error
    omega_opt = x_opt[3:]
    omega_true = true_params[3:]
    omega_error_rad = np.linalg.norm(omega_opt - omega_true)
    omega_error_deg = np.rad2deg(omega_error_rad)

    # Compute RMS residual
    # The objective function returns sum of squared residuals
    # RMS = sqrt(sum(residuals^2) / n)
    obj_value = objective_fn.evaluate(x_opt)
    n_obs = len(objective_fn.observed_lightcurve)
    rms_residual = np.sqrt(obj_value / n_obs)

    # Check success criteria
    omega_ok = omega_error_deg < OMEGA_ERROR_THRESHOLD_DEG_PER_S
    rms_ok = rms_residual < RMS_THRESHOLD_FACTOR * noise_sigma
    success = omega_ok and rms_ok

    return success, omega_error_deg, rms_residual


print("\nSuccess criteria:")
print(f"  1. Angular velocity error < {OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s")
print(f"  2. RMS residual < {RMS_THRESHOLD_FACTOR} x noise_sigma = {RMS_THRESHOLD_FACTOR * noise_sigma:.4f} mag")

# %% [markdown]
# ---
# ## 10. Test Local Optimization from True Parameters
#
# Verify that local optimization works correctly when started at the true solution.

# %%
print("\n" + "=" * 60)
print("TEST: Local optimization from true parameters")
print("=" * 60)

# Run local optimization starting from true parameters
t_start = time.time()
x_opt, f_opt, scipy_success, n_evals = run_local_optimization(
    objective_fn, true_params, bounds
)
t_elapsed = time.time() - t_start

# Evaluate success
success, omega_error_deg, rms_residual = evaluate_success(
    x_opt, true_params, objective_fn, noise_sigma
)

print(f"\nOptimization completed in {t_elapsed:.2f}s ({n_evals} evaluations)")
print(f"  Scipy success: {scipy_success}")
print(f"  Final objective: {f_opt:.6f}")
print(f"\nSuccess criteria evaluation:")
print(f"  Omega error: {omega_error_deg:.6f} deg/s (threshold: {OMEGA_ERROR_THRESHOLD_DEG_PER_S})")
print(f"  RMS residual: {rms_residual:.6f} mag (threshold: {RMS_THRESHOLD_FACTOR * noise_sigma:.4f})")
print(f"  SUCCESS: {success}")

# %% [markdown]
# ---
# ## 11. Constrained Random Initialization Sampler
#
# Generate random starting points uniformly distributed within a hypersphere
# of specified radius around the true solution. The radius is specified in
# scaled units where different parameter types (axis-angle vs omega) have
# different natural scales.

# %%
def sample_in_ball(
    true_params: np.ndarray,
    radius: float,
    n_samples: int,
    param_scales: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample points uniformly within a hypersphere around the true parameters.

    Uses the method of sampling a direction uniformly on the unit sphere and
    then sampling a radius with the appropriate distribution for uniform
    density in the ball.

    Parameters
    ----------
    true_params : np.ndarray
        Center of the ball (6 elements).
    radius : float
        Radius of the ball in scaled units.
    n_samples : int
        Number of samples to generate.
    param_scales : np.ndarray
        Scale factors for each parameter (6 elements). The radius is measured
        in units where each parameter is divided by its scale. For example,
        if param_scales = [1.0, 1.0, 1.0, 0.001, 0.001, 0.001], then a radius
        of 0.1 corresponds to 0.1 rad deviation in axis-angle and 0.0001 rad/s
        deviation in omega.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, 6) containing sampled parameters.
    """
    if seed is not None:
        np.random.seed(seed)

    n_dims = len(true_params)
    samples = np.zeros((n_samples, n_dims))

    for i in range(n_samples):
        # Sample direction uniformly on unit sphere
        # Using the method of normalizing a Gaussian vector
        direction = np.random.randn(n_dims)
        direction /= np.linalg.norm(direction)

        # Sample radius with r^(n-1) weighting for uniform density in ball
        # For n dimensions, CDF is (r/R)^n, so r = R * u^(1/n)
        u = np.random.random()
        r = radius * (u ** (1.0 / n_dims))

        # Compute scaled offset
        scaled_offset = r * direction

        # Convert to actual parameter offset by multiplying by scales
        param_offset = scaled_offset * param_scales

        # Add to true parameters
        samples[i] = true_params + param_offset

    return samples


def verify_samples_in_ball(
    samples: np.ndarray,
    true_params: np.ndarray,
    radius: float,
    param_scales: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """
    Verify that all samples are within the specified radius of true_params.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (n_samples, n_dims) containing samples.
    true_params : np.ndarray
        Center of the ball.
    radius : float
        Radius of the ball in scaled units.
    param_scales : np.ndarray
        Scale factors for each parameter.

    Returns
    -------
    all_inside : bool
        True if all samples are within the ball.
    scaled_distances : np.ndarray
        Array of scaled distances from true_params for each sample.
    """
    # Compute scaled distances
    offsets = samples - true_params
    scaled_offsets = offsets / param_scales
    scaled_distances = np.linalg.norm(scaled_offsets, axis=1)

    all_inside = np.all(scaled_distances <= radius * (1 + 1e-10))  # Small tolerance

    return all_inside, scaled_distances


# %%
# Define parameter scales
# Axis-angle: use 1 radian as natural scale
# Omega: use 1 deg/s = 0.01745 rad/s as natural scale
# This means radius is measured in degrees for both orientation and angular velocity

AXIS_ANGLE_SCALE = np.deg2rad(1.0)  # 1 degree in radians
OMEGA_SCALE = np.deg2rad(1.0)  # 1 deg/s in rad/s

param_scales = np.array([
    AXIS_ANGLE_SCALE,  # axis_angle_x: 1 deg
    AXIS_ANGLE_SCALE,  # axis_angle_y: 1 deg
    AXIS_ANGLE_SCALE,  # axis_angle_z: 1 deg
    OMEGA_SCALE,       # omega_x: 1 deg/s
    OMEGA_SCALE,       # omega_y: 1 deg/s
    OMEGA_SCALE,       # omega_z: 1 deg/s
])

print("Parameter scales (for uniform ball sampling):")
for name, scale in zip(param_names, param_scales):
    print(f"  {name}: {scale:.6f} (= 1 degree or 1 deg/s)")

# %% [markdown]
# ---
# ## 12. Test Random Initialization Sampler
#
# Verify that the sampler generates points uniformly within the specified radius.

# %%
# Test the sampler with a moderate radius
test_radius = 5.0  # 5 degrees
test_n_samples = 100
test_seed = 123

print(f"\nTesting sample_in_ball:")
print(f"  Radius: {test_radius} degrees")
print(f"  Samples: {test_n_samples}")
print(f"  Seed: {test_seed}")

test_samples = sample_in_ball(
    true_params=true_params,
    radius=test_radius,
    n_samples=test_n_samples,
    param_scales=param_scales,
    seed=test_seed,
)

print(f"\nSample array shape: {test_samples.shape}")

# Verify all samples are within the ball
all_inside, scaled_distances = verify_samples_in_ball(
    test_samples, true_params, test_radius, param_scales
)

print(f"\nVerification:")
print(f"  All samples inside ball: {all_inside}")
print(f"  Min scaled distance: {scaled_distances.min():.4f} deg")
print(f"  Max scaled distance: {scaled_distances.max():.4f} deg")
print(f"  Mean scaled distance: {scaled_distances.mean():.4f} deg")

# For uniform distribution in a ball, expected mean distance is n/(n+1) * R
# For n=6, this is 6/7 * R = 0.857 * R
expected_mean_ratio = 6.0 / 7.0
expected_mean = expected_mean_ratio * test_radius
print(f"\nExpected mean distance (for uniform ball): {expected_mean:.4f} deg")
print(f"Actual/Expected ratio: {scaled_distances.mean() / expected_mean:.3f}")

# %%
# Visualize the distribution of scaled distances
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram of scaled distances
ax1 = axes[0]
ax1.hist(scaled_distances, bins=20, density=True, alpha=0.7, edgecolor='black')
ax1.axvline(test_radius, color='red', linestyle='--', label=f'Radius = {test_radius}')
ax1.axvline(scaled_distances.mean(), color='green', linestyle='--', label=f'Mean = {scaled_distances.mean():.2f}')
ax1.set_xlabel('Scaled Distance (degrees)')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Sample Distances from True Parameters')
ax1.legend()

# Per-parameter offset distribution
ax2 = axes[1]
param_offsets_deg = np.rad2deg(test_samples - true_params)
for i, name in enumerate(param_names[:3]):  # Just axis-angle for clarity
    ax2.hist(param_offsets_deg[:, i], bins=15, alpha=0.5, label=name)
ax2.set_xlabel('Offset (degrees)')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Axis-Angle Offsets')
ax2.legend()

plt.tight_layout()
plt.savefig(output_dir / 'inversion_diagnostics' / 'sample_in_ball_test.png', dpi=150)
plt.show()

print(f"\nTest plot saved to: {output_dir / 'inversion_diagnostics' / 'sample_in_ball_test.png'}")

# %% [markdown]
# ---
# ## 13. Basin Size Sweep Experiment
#
# Systematically measure success rate vs. initialization radius.
# Test radii from 0.01 to 10 degrees with 20 trials per radius.

# %%
# Define the radii to test (in degrees)
test_radii = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
n_trials_per_radius = 20

print("=" * 70)
print("BASIN SIZE SWEEP EXPERIMENT")
print("=" * 70)
print(f"\nRadii to test: {test_radii} degrees")
print(f"Trials per radius: {n_trials_per_radius}")
print(f"Total optimizations: {len(test_radii) * n_trials_per_radius}")

# %%
# Data structure to store results
# For each radius, store: successes, omega_errors, rms_residuals, n_evals
sweep_results = {
    'radii': np.array(test_radii),
    'n_trials': n_trials_per_radius,
    # Per-trial results (n_radii x n_trials)
    'success': np.zeros((len(test_radii), n_trials_per_radius), dtype=bool),
    'omega_error': np.zeros((len(test_radii), n_trials_per_radius)),
    'rms_residual': np.zeros((len(test_radii), n_trials_per_radius)),
    'n_evals': np.zeros((len(test_radii), n_trials_per_radius), dtype=int),
    'final_objective': np.zeros((len(test_radii), n_trials_per_radius)),
}

# %%
# Run the sweep experiment
print("\nRunning basin size sweep...")
print("-" * 70)

total_start_time = time.time()

for r_idx, radius in enumerate(test_radii):
    print(f"\nRadius {radius:.2f} degrees ({r_idx + 1}/{len(test_radii)})")

    # Generate all starting points for this radius
    # Use different seeds for each radius to ensure diversity
    base_seed = 1000 * (r_idx + 1)
    starting_points = sample_in_ball(
        true_params=true_params,
        radius=radius,
        n_samples=n_trials_per_radius,
        param_scales=param_scales,
        seed=base_seed,
    )

    # Verify samples are within the ball
    all_inside, scaled_distances = verify_samples_in_ball(
        starting_points, true_params, radius, param_scales
    )
    if not all_inside:
        print(f"  WARNING: Some samples outside ball!")

    # Run optimization for each starting point
    radius_successes = 0
    for trial_idx in range(n_trials_per_radius):
        x0 = starting_points[trial_idx]

        # Run local optimization
        x_opt, f_opt, scipy_success, n_evals = run_local_optimization(
            objective_fn, x0, bounds
        )

        # Evaluate success
        success, omega_error_deg, rms_residual = evaluate_success(
            x_opt, true_params, objective_fn, noise_sigma
        )

        # Store results
        sweep_results['success'][r_idx, trial_idx] = success
        sweep_results['omega_error'][r_idx, trial_idx] = omega_error_deg
        sweep_results['rms_residual'][r_idx, trial_idx] = rms_residual
        sweep_results['n_evals'][r_idx, trial_idx] = n_evals
        sweep_results['final_objective'][r_idx, trial_idx] = f_opt

        if success:
            radius_successes += 1

    # Print summary for this radius
    success_rate = radius_successes / n_trials_per_radius * 100
    mean_omega_err = sweep_results['omega_error'][r_idx].mean()
    mean_evals = sweep_results['n_evals'][r_idx].mean()
    print(f"  Success rate: {radius_successes}/{n_trials_per_radius} = {success_rate:.0f}%")
    print(f"  Mean omega error: {mean_omega_err:.4f} deg/s")
    print(f"  Mean evaluations: {mean_evals:.0f}")

total_elapsed = time.time() - total_start_time
print("\n" + "-" * 70)
print(f"Sweep completed in {total_elapsed:.1f}s")
print(f"Total optimizations: {len(test_radii) * n_trials_per_radius}")

# %%
# Compute summary statistics for each radius
summary_stats = {
    'radii': test_radii,
    'success_rate': [],
    'mean_omega_error': [],
    'std_omega_error': [],
    'mean_rms_residual': [],
    'std_rms_residual': [],
    'mean_n_evals': [],
    'std_n_evals': [],
}

print("\n" + "=" * 70)
print("SUMMARY STATISTICS BY RADIUS")
print("=" * 70)
print(f"{'Radius':>8} | {'Success':>10} | {'Mean Omega Err':>15} | {'Mean RMS':>12} | {'Mean Evals':>12}")
print(f"{'(deg)':>8} | {'Rate (%)':>10} | {'(deg/s)':>15} | {'(mag)':>12} | {'':>12}")
print("-" * 70)

for r_idx, radius in enumerate(test_radii):
    successes = sweep_results['success'][r_idx]
    omega_errors = sweep_results['omega_error'][r_idx]
    rms_residuals = sweep_results['rms_residual'][r_idx]
    n_evals_arr = sweep_results['n_evals'][r_idx]

    success_rate = successes.sum() / len(successes) * 100
    mean_omega = omega_errors.mean()
    std_omega = omega_errors.std()
    mean_rms = rms_residuals.mean()
    std_rms = rms_residuals.std()
    mean_evals = n_evals_arr.mean()
    std_evals = n_evals_arr.std()

    summary_stats['success_rate'].append(success_rate)
    summary_stats['mean_omega_error'].append(mean_omega)
    summary_stats['std_omega_error'].append(std_omega)
    summary_stats['mean_rms_residual'].append(mean_rms)
    summary_stats['std_rms_residual'].append(std_rms)
    summary_stats['mean_n_evals'].append(mean_evals)
    summary_stats['std_n_evals'].append(std_evals)

    print(f"{radius:>8.2f} | {success_rate:>10.1f} | {mean_omega:>15.6f} | {mean_rms:>12.6f} | {mean_evals:>12.1f}")

# Convert to numpy arrays for easier manipulation
for key in summary_stats:
    if key != 'radii':
        summary_stats[key] = np.array(summary_stats[key])

# %% [markdown]
# ---
# ## Setup Complete
#
# We now have:
# - `objective_fn`: The ObjectiveFunction instance for optimization
# - `true_params`: The 6-parameter vector at the true solution
# - `bounds`: Parameter bounds for L-BFGS-B
# - `run_local_optimization()`: Function to run local optimization
# - `evaluate_success()`: Function to check if optimization succeeded
# - `noise_sigma`: The noise level (0.05 mag)
# - `sample_in_ball()`: Function to generate random samples in a hypersphere
# - `verify_samples_in_ball()`: Function to verify samples are within the ball
# - `param_scales`: Scale factors for proper distance measurement
# - `sweep_results`: Raw results from basin size sweep (US-009)
# - `summary_stats`: Aggregated statistics per radius (US-009)
#
# The next user story (US-010) will add:
# - Visualization and critical radius identification

# %%
# Summary of key objects for downstream analysis
print("\n" + "=" * 60)
print("SETUP COMPLETE - Objects available for basin size study:")
print("=" * 60)
print(f"\n  objective_fn: ObjectiveFunction instance")
print(f"  true_params: {true_params}")
print(f"  bounds: {len(bounds)} parameter bounds")
print(f"  noise_sigma: {noise_sigma}")
print(f"  n_observations: {n_observations}")
print(f"  param_scales: {param_scales}")
print(f"\n  Functions:")
print(f"    run_local_optimization(): L-BFGS-B wrapper")
print(f"    evaluate_success(): Success criteria checker")
print(f"    sample_in_ball(): Random sampling within hypersphere")
print(f"    verify_samples_in_ball(): Verify samples within radius")
print(f"\n  Basin sweep results:")
print(f"    sweep_results: Dict with per-trial data ({len(test_radii)} radii x {n_trials_per_radius} trials)")
print(f"    summary_stats: Dict with aggregated statistics per radius")
print(f"\n  Success criteria:")
print(f"    - Omega error < {OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s")
print(f"    - RMS < {RMS_THRESHOLD_FACTOR * noise_sigma:.4f} mag")

# %% [markdown]
# ---
# ## 14. Basin Size Visualization and Threshold Identification
#
# Visualize success rate vs. initialization radius and identify the critical
# radius where local optimization reliability drops below 80%.

# %%
# Compute binomial uncertainty for success rate
# For n trials with k successes, the standard error is sqrt(p*(1-p)/n)
# where p = k/n is the estimated success probability


def compute_binomial_error(success_rate: float, n_trials: int) -> float:
    """
    Compute standard error for binomial proportion.

    Parameters
    ----------
    success_rate : float
        Success rate as percentage (0-100).
    n_trials : int
        Number of trials.

    Returns
    -------
    error : float
        Standard error as percentage.
    """
    p = success_rate / 100.0
    if p <= 0 or p >= 1:
        # Use Wilson score interval approximation for edge cases
        return 100.0 / np.sqrt(n_trials + 4)
    return 100.0 * np.sqrt(p * (1 - p) / n_trials)


# Compute error bars for each radius
success_errors = np.array([
    compute_binomial_error(sr, n_trials_per_radius)
    for sr in summary_stats['success_rate']
])

print("Success rate with binomial uncertainty:")
print("-" * 50)
for radius, sr, err in zip(test_radii, summary_stats['success_rate'], success_errors):
    print(f"  {radius:>6.2f} deg: {sr:>5.1f}% +/- {err:>4.1f}%")

# %%
# Create basin size visualization figures
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Success rate vs radius with error bars
ax1 = axes[0]
ax1.errorbar(
    test_radii,
    summary_stats['success_rate'],
    yerr=success_errors,
    fmt='o-',
    capsize=4,
    capthick=1.5,
    markersize=8,
    linewidth=2,
    color='steelblue',
    ecolor='steelblue',
    label='Success rate',
)

# Add 80% threshold line
ax1.axhline(80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='80% threshold')

# Find and mark critical radius (first radius where success rate drops below 80%)
critical_idx = None
for i, sr in enumerate(summary_stats['success_rate']):
    if sr < 80:
        critical_idx = i
        break

if critical_idx is not None and critical_idx > 0:
    # Interpolate to find more precise critical radius
    sr_prev = summary_stats['success_rate'][critical_idx - 1]
    sr_curr = summary_stats['success_rate'][critical_idx]
    r_prev = test_radii[critical_idx - 1]
    r_curr = test_radii[critical_idx]

    # Linear interpolation in log space for radius
    if sr_prev != sr_curr:
        t = (80 - sr_curr) / (sr_prev - sr_curr)
        critical_radius = np.exp(t * np.log(r_prev) + (1 - t) * np.log(r_curr))
    else:
        critical_radius = r_curr

    ax1.axvline(
        critical_radius, color='green', linestyle=':', linewidth=2,
        label=f'Critical radius = {critical_radius:.2f} deg'
    )
elif critical_idx == 0:
    # First radius already below 80%
    critical_radius = test_radii[0]
    ax1.axvline(
        critical_radius, color='green', linestyle=':', linewidth=2,
        label=f'Critical radius < {critical_radius:.2f} deg'
    )
else:
    # All radii have >80% success rate
    critical_radius = test_radii[-1]
    ax1.text(
        0.5, 0.1, f'Success rate > 80% for all tested radii (up to {critical_radius:.1f} deg)',
        transform=ax1.transAxes, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )

ax1.set_xlabel('Initialization Radius (degrees)', fontsize=12)
ax1.set_ylabel('Success Rate (%)', fontsize=12)
ax1.set_title('Local Optimization Success Rate vs. Initialization Distance', fontsize=13)
ax1.set_xscale('log')
ax1.set_ylim(-5, 105)
ax1.set_xlim(test_radii[0] * 0.8, test_radii[-1] * 1.2)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower left', fontsize=10)

# Plot 2: Mean omega error vs radius
ax2 = axes[1]
ax2.errorbar(
    test_radii,
    summary_stats['mean_omega_error'],
    yerr=summary_stats['std_omega_error'],
    fmt='s-',
    capsize=4,
    capthick=1.5,
    markersize=8,
    linewidth=2,
    color='darkorange',
    ecolor='darkorange',
    label='Mean omega error',
)

# Add success threshold line
ax2.axhline(
    OMEGA_ERROR_THRESHOLD_DEG_PER_S, color='red', linestyle='--', linewidth=1.5,
    alpha=0.7, label=f'Success threshold ({OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s)'
)

ax2.set_xlabel('Initialization Radius (degrees)', fontsize=12)
ax2.set_ylabel('Angular Velocity Error (deg/s)', fontsize=12)
ax2.set_title('Final Angular Velocity Error vs. Initialization Distance', fontsize=13)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(test_radii[0] * 0.8, test_radii[-1] * 1.2)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='upper left', fontsize=10)

plt.tight_layout()

# Save figure
output_path = output_dir / 'inversion_diagnostics' / 'basin_size_success_rate.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to: {output_path}")

# %%
# Create additional visualization: Function evaluations and RMS residual
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Plot 3: Mean function evaluations vs radius
ax3 = axes2[0]
ax3.errorbar(
    test_radii,
    summary_stats['mean_n_evals'],
    yerr=summary_stats['std_n_evals'],
    fmt='D-',
    capsize=4,
    capthick=1.5,
    markersize=8,
    linewidth=2,
    color='forestgreen',
    ecolor='forestgreen',
    label='Mean function evaluations',
)

ax3.set_xlabel('Initialization Radius (degrees)', fontsize=12)
ax3.set_ylabel('Function Evaluations', fontsize=12)
ax3.set_title('Optimization Cost vs. Initialization Distance', fontsize=13)
ax3.set_xscale('log')
ax3.set_xlim(test_radii[0] * 0.8, test_radii[-1] * 1.2)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', fontsize=10)

# Plot 4: Mean RMS residual vs radius
ax4 = axes2[1]
ax4.errorbar(
    test_radii,
    summary_stats['mean_rms_residual'],
    yerr=summary_stats['std_rms_residual'],
    fmt='^-',
    capsize=4,
    capthick=1.5,
    markersize=8,
    linewidth=2,
    color='purple',
    ecolor='purple',
    label='Mean RMS residual',
)

# Add noise level reference
ax4.axhline(
    noise_sigma, color='gray', linestyle='--', linewidth=1.5,
    alpha=0.7, label=f'Noise sigma ({noise_sigma} mag)'
)
ax4.axhline(
    RMS_THRESHOLD_FACTOR * noise_sigma, color='red', linestyle='--', linewidth=1.5,
    alpha=0.7, label=f'Success threshold ({RMS_THRESHOLD_FACTOR}x sigma)'
)

ax4.set_xlabel('Initialization Radius (degrees)', fontsize=12)
ax4.set_ylabel('RMS Residual (mag)', fontsize=12)
ax4.set_title('Final RMS Residual vs. Initialization Distance', fontsize=13)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(test_radii[0] * 0.8, test_radii[-1] * 1.2)
ax4.grid(True, alpha=0.3, which='both')
ax4.legend(loc='upper left', fontsize=10)

plt.tight_layout()

# Save figure
output_path2 = output_dir / 'inversion_diagnostics' / 'basin_size_metrics.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFigure saved to: {output_path2}")

# %% [markdown]
# ---
# ## 15. Basin Size Study Summary
#
# This section summarizes the key findings from the basin size analysis.

# %%
# Generate summary findings


def find_critical_radius_80(radii: list, success_rates: np.ndarray) -> tuple[float | None, str]:
    """
    Find the critical radius where success rate drops below 80%.

    Returns
    -------
    critical_radius : float or None
        The interpolated critical radius, or None if all rates are above 80%.
    description : str
        Human-readable description of the finding.
    """
    # Find first index where success rate < 80%
    below_80_idx = None
    for i, sr in enumerate(success_rates):
        if sr < 80:
            below_80_idx = i
            break

    if below_80_idx is None:
        return None, f"Success rate remains above 80% for all tested radii (up to {radii[-1]:.1f} degrees)"

    if below_80_idx == 0:
        return radii[0], f"Success rate drops below 80% even at smallest tested radius ({radii[0]:.2f} degrees)"

    # Interpolate between the two points
    sr_prev = success_rates[below_80_idx - 1]
    sr_curr = success_rates[below_80_idx]
    r_prev = radii[below_80_idx - 1]
    r_curr = radii[below_80_idx]

    if sr_prev != sr_curr:
        t = (80 - sr_curr) / (sr_prev - sr_curr)
        critical_radius = np.exp(t * np.log(r_prev) + (1 - t) * np.log(r_curr))
    else:
        critical_radius = r_curr

    return critical_radius, f"Success rate drops below 80% at approximately {critical_radius:.2f} degrees"


# Find critical radius
critical_r, critical_description = find_critical_radius_80(test_radii, summary_stats['success_rate'])

# Generate summary
print("=" * 70)
print("BASIN SIZE STUDY - SUMMARY FINDINGS")
print("=" * 70)

print("\n1. CRITICAL RADIUS IDENTIFICATION")
print("-" * 40)
print(f"   {critical_description}")
if critical_r is not None:
    print(f"\n   Interpretation: Local optimization (L-BFGS-B) succeeds with >80%")
    print(f"   probability when initialized within {critical_r:.2f} degrees of the true solution.")

print("\n2. SUCCESS RATE PROFILE")
print("-" * 40)
high_success_radii = [r for r, sr in zip(test_radii, summary_stats['success_rate']) if sr >= 95]
moderate_success_radii = [r for r, sr in zip(test_radii, summary_stats['success_rate']) if 80 <= sr < 95]
low_success_radii = [r for r, sr in zip(test_radii, summary_stats['success_rate']) if sr < 80]

if high_success_radii:
    print(f"   Very high success (>=95%): radii <= {max(high_success_radii):.2f} degrees")
if moderate_success_radii:
    print(f"   Moderate success (80-95%): radii in [{min(moderate_success_radii):.2f}, {max(moderate_success_radii):.2f}] degrees")
if low_success_radii:
    print(f"   Low success (<80%): radii >= {min(low_success_radii):.2f} degrees")

print("\n3. OPTIMIZATION COST")
print("-" * 40)
min_evals = summary_stats['mean_n_evals'].min()
max_evals = summary_stats['mean_n_evals'].max()
print(f"   Function evaluations range: {min_evals:.0f} - {max_evals:.0f}")
print(f"   Cost increases with initialization distance as expected.")

print("\n4. PRACTICAL IMPLICATIONS")
print("-" * 40)
if critical_r is not None and critical_r < 1.0:
    print(f"   - Basin of attraction is NARROW ({critical_r:.2f} degrees)")
    print(f"   - Global search or multi-start strategies are RECOMMENDED")
    print(f"   - Single local optimization from random guess will likely fail")
elif critical_r is not None and critical_r < 5.0:
    print(f"   - Basin of attraction is MODERATE ({critical_r:.2f} degrees)")
    print(f"   - Multi-start local optimization may suffice")
    print(f"   - Consider guided initialization from prior knowledge")
else:
    print(f"   - Basin of attraction is WIDE (>{test_radii[-1]:.1f} degrees)")
    print(f"   - Local optimization from reasonable initial guess likely sufficient")
    print(f"   - Global search may not be necessary")

# %%
# Save summary to file
summary_text = f"""Basin Size Study - Summary Report
================================

Analysis Date: {time.strftime('%Y-%m-%d %H:%M')}
Configuration: {config.name}
Observations: {n_observations}
Noise Level: {noise_sigma} mag

Experiment Parameters
---------------------
- Test radii: {test_radii} degrees
- Trials per radius: {n_trials_per_radius}
- Total optimizations: {len(test_radii) * n_trials_per_radius}

Success Criteria
----------------
- Angular velocity error < {OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s
- RMS residual < {RMS_THRESHOLD_FACTOR * noise_sigma:.4f} mag

Results by Radius
-----------------
{'Radius (deg)':<12} | {'Success Rate':<12} | {'Mean Omega Err (deg/s)':<22} | {'Mean RMS (mag)':<15}
{'-' * 70}
"""

for i, radius in enumerate(test_radii):
    sr = summary_stats['success_rate'][i]
    omega_err = summary_stats['mean_omega_error'][i]
    rms = summary_stats['mean_rms_residual'][i]
    summary_text += f"{radius:<12.2f} | {sr:<12.1f} | {omega_err:<22.6f} | {rms:<15.6f}\n"

summary_text += f"""
Critical Radius Finding
-----------------------
{critical_description}

Recommendation
--------------
"""

if critical_r is not None and critical_r < 1.0:
    summary_text += f"""Local optimization succeeds with >80% probability when initialized within
{critical_r:.2f} degrees of the true solution. This is a NARROW basin of attraction,
indicating that global search or multi-start strategies are RECOMMENDED for
reliable inversion without prior knowledge of the true parameters.
"""
elif critical_r is not None and critical_r < 5.0:
    summary_text += f"""Local optimization succeeds with >80% probability when initialized within
{critical_r:.2f} degrees of the true solution. This is a MODERATE basin of attraction.
Multi-start local optimization with ~10-20 random initializations should provide
good convergence probability.
"""
else:
    summary_text += f"""Local optimization succeeds with >80% probability for all tested initialization
radii (up to {test_radii[-1]:.1f} degrees). This indicates a WIDE basin of attraction.
Local optimization from a reasonable initial guess should be sufficient for most cases.
"""

# Save to file
summary_path = output_dir / 'inversion_diagnostics' / 'basin_size_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary_text)

print(f"\nSummary saved to: {summary_path}")

# %% [markdown]
# ---
# ## Conclusions
#
# This notebook has systematically characterized the basin of attraction for
# lightcurve inversion by measuring local optimization success rate as a
# function of initialization distance from the true solution.
#
# Key outputs:
# - `basin_size_success_rate.png`: Success rate vs. radius with error bars
# - `basin_size_metrics.png`: Error and cost metrics vs. radius
# - `basin_size_summary.txt`: Quantitative summary of findings
#
# The critical radius where success rate drops below 80% indicates the
# effective size of the basin of attraction and informs whether global
# search strategies are necessary for reliable inversion.
