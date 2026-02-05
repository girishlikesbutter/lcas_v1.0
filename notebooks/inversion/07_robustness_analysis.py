# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Robustness Analysis
#
# This notebook investigates the robustness of lightcurve inversion across varying
# data quality conditions:
#
# 1. **Noise levels**: How does performance degrade with increasing measurement noise?
# 2. **Observation density**: What is the minimum number of observations needed?
# 3. **Combined effects**: Mapping success rate across noise/observation space
#
# ## Goals
#
# - Implement parameterized synthetic data generation
# - Run systematic sweep across noise levels and observation counts
# - Identify minimum data requirements for reliable inversion

# %% [markdown]
# ---
# ## Setup
#
# Copy minimal setup from notebook 04 to generate the same test case configuration.

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

# Maximum number of observations (we'll subsample from this)
max_n_observations = 100

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
print("ROBUSTNESS ANALYSIS")
print("=" * 70)
print(f"\nConfiguration: {config.name}")
print(f"Max observations: {max_n_observations}")

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
print(f"  Solar panels (SP_North, SP_South): {SOLAR_PANEL_ANGLE_DEG} deg")
print(f"  Antenna dishes (AD_East, AD_West): {ANTENNA_DISH_ANGLE_DEG} deg")

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
# ## 4. Initialize SPICE and Compute Full Observation Geometry
#
# We compute geometry for the maximum number of observations, then subsample
# for different test cases.

# %%
# Initialize SPICE
print("Initializing SPICE...")
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))

# Generate time series for maximum observations
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
full_epochs = np.linspace(start_et, end_et, max_n_observations)

print(f"Time range: {start_time_utc} to {end_time_utc}")
print(f"Max observations: {max_n_observations}")

# Relative observation times starting from 0
full_observation_times = full_epochs - full_epochs[0]

# %%
# Compute observation geometry using SPICE for full set
print("Computing observation geometry...")
full_geometry_data = compute_observation_geometry(
    epochs=full_epochs,
    satellite_id=satellite_id,
    observer_id=OBSERVER_ID,
    spice_handler=spice_handler,
    config=config
)

# Extract J2000 positions
full_sun_positions_j2000 = full_geometry_data['sun_positions']
full_observer_positions_j2000 = full_geometry_data['obs_positions']
full_satellite_positions_j2000 = full_geometry_data['sat_positions']
full_observer_distances = full_geometry_data['observer_distances']

print(f"Geometry computed: {max_n_observations} observations")
print(f"Observer distance range: {full_observer_distances.min():.0f} - {full_observer_distances.max():.0f} km")

# %%
# Create fixed articulation matrices for full observation set
full_articulation_angles = {
    'SP_North': np.full(max_n_observations, SOLAR_PANEL_ANGLE_DEG),
    'SP_South': np.full(max_n_observations, SOLAR_PANEL_ANGLE_DEG),
    'AD_East': np.full(max_n_observations, ANTENNA_DISH_ANGLE_DEG),
    'AD_West': np.full(max_n_observations, ANTENNA_DISH_ANGLE_DEG),
}

full_articulation_matrices = compute_rotation_matrices_from_angles(
    full_articulation_angles, satellite
)

print("Articulation matrices created for full observation set")

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

# Store as combined parameter vector for slicing
# Format: [axis_angle_x, axis_angle_y, axis_angle_z, omega_x, omega_y, omega_z]
true_params = np.concatenate([true_axis_angle, true_omega0])

print("True attitude parameters:")
print(f"  Axis-angle (rad): {true_axis_angle}")
print(f"  Axis-angle (deg): {np.rad2deg(true_axis_angle)}")
print(f"  Angular velocity (deg/s): {np.rad2deg(true_omega0)}")
print(f"\nCombined parameter vector:")
print(f"  true_params = {true_params}")

# Parameter names for reference
param_names = ['axis_angle_x', 'axis_angle_y', 'axis_angle_z', 'omega_x', 'omega_y', 'omega_z']

# %% [markdown]
# ---
# ## 6. Generate Full True Lightcurve
#
# We compute the true lightcurve for all observations first, then subsample.

# %%
print("\nGenerating full true lightcurve with true parameters...")
print("  Mode: TUMBLING with inertia tensor")
print("  Shadows: ENABLED")

# Propagate true attitude using tumbling dynamics
true_quaternions, true_omega_history = propagate_attitude(
    q0=true_q0,
    omega0=true_omega0,
    times=full_observation_times,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
)

print(f"  Attitude propagated: {len(true_quaternions)} epochs")

# %%
# Create a temporary ObjectiveFunction to compute body-frame vectors
objective_temp = ObjectiveFunction(
    satellite=satellite,
    observation_times=full_observation_times,
    observed_lightcurve=np.zeros(max_n_observations),  # placeholder
    sun_positions_j2000=full_sun_positions_j2000,
    observer_positions_j2000=full_observer_positions_j2000,
    satellite_positions_j2000=full_satellite_positions_j2000,
    observer_distances=full_observer_distances,
    compute_shadows_flag=True,
    articulation_matrices=full_articulation_matrices,
)

# Get body-frame vectors from propagated attitude
full_k1_vectors, full_k2_vectors = objective_temp._compute_body_frame_vectors(true_quaternions)

# %%
# Compute shadows with ray tracing
print("Computing shadows...")
full_lit_status_dict = compute_shadows(
    satellite=satellite,
    k1_vectors=full_k1_vectors,
    explicit_component_matrices=full_articulation_matrices,
    show_progress=True,
)

# %%
# Generate true lightcurve
print("Generating lightcurve...")
full_true_lightcurve, total_flux, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=full_lit_status_dict,
    k1_vectors_array=full_k1_vectors,
    k2_vectors_array=full_k2_vectors,
    observer_distances=full_observer_distances,
    satellite=satellite,
    epochs=full_epochs,
    pre_computed_matrices=full_articulation_matrices,
    generate_no_shadow=False,
    animate=False,
    show_progress=True,
)

print(f"Full true lightcurve range: [{full_true_lightcurve.min():.2f}, {full_true_lightcurve.max():.2f}] mag")

# %% [markdown]
# ---
# ## 7. Parameterized Test Case Generation
#
# This function generates test cases with varying observation counts and noise levels
# by subsampling the full geometry and adding noise to the true lightcurve.

# %%
def generate_test_case(
    n_observations: int,
    noise_sigma: float,
    seed: int,
) -> dict:
    """
    Generate a test case with specified observation count and noise level.

    Subsamples from the pre-computed full observation set and adds Gaussian
    noise to the true lightcurve. Uses the same true parameters across all
    test cases for comparability.

    Parameters
    ----------
    n_observations : int
        Number of observations to use. Must be <= max_n_observations.
    noise_sigma : float
        Standard deviation of Gaussian noise to add (in magnitudes).
    seed : int
        Random seed for reproducibility (controls both subsampling and noise).

    Returns
    -------
    dict
        Test case containing:
        - 'n_observations': Number of observations
        - 'noise_sigma': Noise level
        - 'seed': Random seed used
        - 'observation_times': Relative observation times
        - 'epochs': SPICE epoch times
        - 'sun_positions_j2000': Sun positions in J2000 frame
        - 'observer_positions_j2000': Observer positions in J2000 frame
        - 'satellite_positions_j2000': Satellite positions in J2000 frame
        - 'observer_distances': Observer distances
        - 'articulation_matrices': Pre-computed articulation matrices
        - 'true_lightcurve': True lightcurve (without noise)
        - 'observed_lightcurve': Observed lightcurve (with noise)
        - 'objective_fn': ObjectiveFunction for optimization
    """
    if n_observations > max_n_observations:
        raise ValueError(
            f"n_observations ({n_observations}) exceeds max_n_observations ({max_n_observations})"
        )

    # Set random state for reproducibility
    rng = np.random.default_rng(seed)

    # Select indices to subsample (uniformly spaced with some jitter)
    if n_observations == max_n_observations:
        indices = np.arange(max_n_observations)
    else:
        # Uniformly spaced indices to maintain temporal coverage
        indices = np.linspace(0, max_n_observations - 1, n_observations).astype(int)

    # Subsample all data arrays
    observation_times = full_observation_times[indices]
    epochs = full_epochs[indices]
    sun_positions_j2000 = full_sun_positions_j2000[indices]
    observer_positions_j2000 = full_observer_positions_j2000[indices]
    satellite_positions_j2000 = full_satellite_positions_j2000[indices]
    observer_distances = full_observer_distances[indices]

    # Subsample articulation matrices
    articulation_matrices = {}
    for comp_name, matrices in full_articulation_matrices.items():
        articulation_matrices[comp_name] = matrices[indices]

    # Subsample true lightcurve and add noise
    true_lightcurve = full_true_lightcurve[indices]
    noise = rng.normal(0, noise_sigma, n_observations)
    observed_lightcurve = true_lightcurve + noise

    # Create ObjectiveFunction for this test case
    objective_fn = ObjectiveFunction(
        satellite=satellite,
        observation_times=observation_times,
        observed_lightcurve=observed_lightcurve,
        sun_positions_j2000=sun_positions_j2000,
        observer_positions_j2000=observer_positions_j2000,
        satellite_positions_j2000=satellite_positions_j2000,
        observer_distances=observer_distances,
        compute_shadows_flag=True,
        articulation_matrices=articulation_matrices,
        mode="tumbling",
        inertia_tensor=inertia_tensor,
    )

    return {
        'n_observations': n_observations,
        'noise_sigma': noise_sigma,
        'seed': seed,
        'observation_times': observation_times,
        'epochs': epochs,
        'sun_positions_j2000': sun_positions_j2000,
        'observer_positions_j2000': observer_positions_j2000,
        'satellite_positions_j2000': satellite_positions_j2000,
        'observer_distances': observer_distances,
        'articulation_matrices': articulation_matrices,
        'true_lightcurve': true_lightcurve,
        'observed_lightcurve': observed_lightcurve,
        'objective_fn': objective_fn,
    }


print("generate_test_case() function defined")

# %% [markdown]
# ---
# ## 8. Test Case Verification
#
# Verify that the test case generation works correctly by generating a few examples.

# %%
print("\nTesting generate_test_case()...")
print("-" * 50)

# Test with different configurations
test_configs = [
    (50, 0.05, 42),   # Same as notebook 04 baseline
    (20, 0.1, 123),   # Sparse, noisy
    (100, 0.02, 456), # Dense, clean
]

for n_obs, sigma, seed in test_configs:
    test_case = generate_test_case(n_obs, sigma, seed)

    print(f"\nTest case: n_obs={n_obs}, sigma={sigma}, seed={seed}")
    print(f"  Observation times shape: {test_case['observation_times'].shape}")
    print(f"  True lightcurve range: [{test_case['true_lightcurve'].min():.2f}, {test_case['true_lightcurve'].max():.2f}] mag")
    print(f"  Observed lightcurve range: [{test_case['observed_lightcurve'].min():.2f}, {test_case['observed_lightcurve'].max():.2f}] mag")

    # Verify objective function works
    obj_val = test_case['objective_fn'](true_params)
    print(f"  Objective at true params: {obj_val:.6f}")

print("\nTest case generation PASSED!")

# %% [markdown]
# ---
# ## 9. Visualize Test Cases
#
# Visualize a few test cases to understand the effect of noise and observation density.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

test_cases_for_plot = [
    (50, 0.02, 42),   # Baseline noise
    (50, 0.1, 42),    # High noise
    (20, 0.05, 42),   # Sparse observations
    (100, 0.05, 42),  # Dense observations
]

titles = [
    f'n={50}, sigma={0.02} (low noise)',
    f'n={50}, sigma={0.1} (high noise)',
    f'n={20}, sigma={0.05} (sparse)',
    f'n={100}, sigma={0.05} (dense)',
]

for ax, (n_obs, sigma, seed), title in zip(axes.flat, test_cases_for_plot, titles):
    test_case = generate_test_case(n_obs, sigma, seed)

    times = test_case['observation_times'] / 3600  # Convert to hours
    ax.plot(times, test_case['true_lightcurve'], 'b-', alpha=0.7, label='True')
    ax.scatter(times, test_case['observed_lightcurve'], c='r', s=20, alpha=0.6, label='Observed')

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Magnitude')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/results/inversion_diagnostics/test_case_examples.png', dpi=150)
plt.show()

print("\nTest case visualization saved to data/results/inversion_diagnostics/test_case_examples.png")

# %% [markdown]
# ---
# ## 10. Define Parameter Bounds and Success Criteria
#
# Same definitions as notebook 06 for consistent evaluation.

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
# Define success criteria (same as notebook 06)
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
    true_omega = true_params[3:]
    opt_omega = x_opt[3:]
    omega_error_rad = np.linalg.norm(opt_omega - true_omega)
    omega_error_deg = np.rad2deg(omega_error_rad)

    # Compute RMS residual
    obj_value = objective_fn(x_opt)
    n_obs = len(objective_fn.observed_lightcurve)
    rms_residual = np.sqrt(obj_value / n_obs)

    # Check success criteria
    omega_ok = omega_error_deg < OMEGA_ERROR_THRESHOLD_DEG_PER_S
    rms_ok = rms_residual < RMS_THRESHOLD_FACTOR * noise_sigma
    success = omega_ok and rms_ok

    return success, omega_error_deg, rms_residual


print(f"\nSuccess criteria:")
print(f"  Angular velocity error < {OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s")
print(f"  RMS residual < {RMS_THRESHOLD_FACTOR} x noise_sigma")

# %% [markdown]
# ---
# ## 11. Summary
#
# This notebook has set up the infrastructure for robustness analysis:
#
# **Objects created:**
# - `satellite`: Satellite model loaded from STL files
# - `inertia_tensor`: Computed inertia tensor for tumbling dynamics
# - `true_params`: True parameter vector for comparability across test cases
# - `full_*` arrays: Pre-computed geometry and lightcurve for max_n_observations
#
# **Functions defined:**
# - `generate_test_case(n_observations, noise_sigma, seed)`: Generate test cases with
#   varying data quality. Returns dict with lightcurve, geometry, and ObjectiveFunction.
# - `evaluate_success(x_opt, true_params, objective_fn, noise_sigma)`: Evaluate
#   optimization success based on parameter error and residual quality.
#
# **Next steps (in subsequent cells):**
# - Implement robustness sweep across noise levels and observation counts (US-018)
# - Create heatmap visualization of success rates (US-019)
# - Add final recommendations (US-020)

# %%
print("\n" + "=" * 70)
print("NOTEBOOK 07 SETUP COMPLETE")
print("=" * 70)
print("\nReady for robustness analysis experiments.")
print(f"  Max observations: {max_n_observations}")
print(f"  True parameters: {true_params}")
print(f"  Full lightcurve range: [{full_true_lightcurve.min():.2f}, {full_true_lightcurve.max():.2f}] mag")

# %% [markdown]
# ---
# ## 12. CountedObjective Wrapper and Optimizer
#
# Implement the same CountedObjective wrapper and multi-start optimizer from notebook 06
# for fair comparison.

# %%
from scipy.stats.qmc import LatinHypercube


class CountedObjective:
    """
    Wrapper that counts function evaluations and enforces a budget limit.

    Parameters
    ----------
    objective_fn : ObjectiveFunction
        The underlying objective function to wrap.
    budget : int
        Maximum number of function evaluations allowed.
    penalty_value : float
        Value returned when budget is exhausted. Default is 1e10.
    """

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        budget: int,
        penalty_value: float = 1e10,
    ):
        self.objective_fn = objective_fn
        self.budget = budget
        self.penalty_value = penalty_value

        # Tracking state
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params = None

    def __call__(self, params: np.ndarray) -> float:
        """
        Evaluate the objective function with budget enforcement.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector to evaluate.

        Returns
        -------
        float
            Objective value, or penalty_value if budget is exhausted.
        """
        # Check budget
        if self.n_evals >= self.budget:
            return self.penalty_value

        self.n_evals += 1

        # Normalize axis-angle via quaternion round-trip
        axis_angle = params[:3]
        omega = params[3:]

        q = axis_angle_to_quaternion(axis_angle)
        q_normalized = normalize_quaternion(q)
        axis_angle_norm = quaternion_to_axis_angle(q_normalized)

        params_normalized = np.concatenate([axis_angle_norm, omega])

        # Evaluate objective
        value = self.objective_fn(params_normalized)

        # Track best result
        if value < self.best_value:
            self.best_value = value
            self.best_params = params_normalized.copy()

        return value

    def reset(self) -> None:
        """Reset the counter and tracking for a new optimization trial."""
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params = None

    def is_budget_exhausted(self) -> bool:
        """Check if the evaluation budget has been exhausted."""
        return self.n_evals >= self.budget


def multistart_local(
    counted_objective: CountedObjective,
    bounds: list[tuple[float, float]],
    n_starts: int,
    max_evals_per_start: int,
    seed: int | None = None,
) -> dict:
    """
    Multi-start local optimization using Latin Hypercube Sampling.

    Parameters
    ----------
    counted_objective : CountedObjective
        Budget-enforcing objective wrapper. Should already be reset before calling.
    bounds : list[tuple[float, float]]
        List of (lower, upper) bounds for each parameter.
    n_starts : int
        Number of random starting points to try.
    max_evals_per_start : int
        Maximum function evaluations per local optimization.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results containing x_best, f_best, n_evals, and more.
    """
    n_params = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # Generate Latin Hypercube samples in [0, 1]^n
    lhs = LatinHypercube(d=n_params, seed=seed)
    samples_unit = lhs.random(n=n_starts)

    # Scale to parameter bounds
    initial_points = lower_bounds + samples_unit * (upper_bounds - lower_bounds)

    # Track results
    x_best = None
    f_best = float('inf')

    for x0 in initial_points:
        # Check if budget is already exhausted
        if counted_objective.is_budget_exhausted():
            break

        # Run L-BFGS-B from this starting point
        result = minimize(
            counted_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxfun': max_evals_per_start,
                'ftol': 1e-8,
                'gtol': 1e-6,
            },
        )

        # Update best if this is better
        if result.fun < f_best:
            f_best = result.fun
            x_best = result.x.copy()

    return {
        'x_best': x_best if x_best is not None else counted_objective.best_params,
        'f_best': f_best if f_best < float('inf') else counted_objective.best_value,
        'n_evals': counted_objective.n_evals,
    }


print("CountedObjective and multistart_local() defined")

# %% [markdown]
# ---
# ## 13. Robustness Sweep Experiment
#
# Systematic sweep across noise levels and observation counts to map success rate.
#
# **Configuration:**
# - Noise levels: [0.01, 0.02, 0.05, 0.1, 0.2] mag
# - Observation counts: [20, 35, 50, 75, 100]
# - 5 trials per combination = 25 cells x 5 trials = 125 optimizations
# - Use multi-start local optimizer (best from notebook 06)

# %%
# Sweep configuration
NOISE_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2]  # magnitudes
N_OBSERVATIONS_LIST = [20, 35, 50, 75, 100]
N_TRIALS_PER_CELL = 5

# Optimizer settings (from notebook 06)
EVAL_BUDGET = 5000
N_STARTS = 50
EVALS_PER_START = 100

# Base seed for reproducibility
BASE_SEED = 42

print("=" * 70)
print("ROBUSTNESS SWEEP CONFIGURATION")
print("=" * 70)
print(f"\nNoise levels (mag): {NOISE_LEVELS}")
print(f"Observation counts: {N_OBSERVATIONS_LIST}")
print(f"Trials per cell: {N_TRIALS_PER_CELL}")
print(f"Total cells: {len(NOISE_LEVELS) * len(N_OBSERVATIONS_LIST)}")
print(f"Total optimizations: {len(NOISE_LEVELS) * len(N_OBSERVATIONS_LIST) * N_TRIALS_PER_CELL}")
print(f"\nOptimizer: Multi-start local (LHS)")
print(f"  Evaluation budget: {EVAL_BUDGET}")
print(f"  Number of starts: {N_STARTS}")
print(f"  Evals per start: {EVALS_PER_START}")

# %%
# Initialize result storage
# 2D arrays: [n_noise x n_observations]
n_noise = len(NOISE_LEVELS)
n_obs_configs = len(N_OBSERVATIONS_LIST)

# Success rate will be computed from trial results
success_matrix = np.zeros((n_noise, n_obs_configs))
mean_omega_error = np.zeros((n_noise, n_obs_configs))
mean_rms_residual = np.zeros((n_noise, n_obs_configs))

# Store all trial results for detailed analysis
all_trial_results = []

print("\nRunning robustness sweep...")
print("-" * 70)

sweep_start_time = time.time()

for i, noise_sigma in enumerate(NOISE_LEVELS):
    for j, n_obs in enumerate(N_OBSERVATIONS_LIST):
        cell_successes = 0
        cell_omega_errors = []
        cell_rms_residuals = []

        print(f"\nCell [{i},{j}]: noise={noise_sigma:.2f} mag, n_obs={n_obs}")

        for trial in range(N_TRIALS_PER_CELL):
            # Generate unique seed for this trial
            trial_seed = BASE_SEED + i * 1000 + j * 100 + trial

            # Generate test case
            test_case = generate_test_case(n_obs, noise_sigma, trial_seed)

            # Create counted objective for this trial
            counted_obj = CountedObjective(
                test_case['objective_fn'],
                budget=EVAL_BUDGET,
            )

            # Run optimizer
            trial_start = time.time()
            result = multistart_local(
                counted_objective=counted_obj,
                bounds=bounds,
                n_starts=N_STARTS,
                max_evals_per_start=EVALS_PER_START,
                seed=trial_seed + 10000,  # Different seed for optimizer
            )
            trial_time = time.time() - trial_start

            # Get best parameters
            x_opt = result['x_best']

            # Evaluate success
            if x_opt is not None:
                success, omega_error, rms_residual = evaluate_success(
                    x_opt, true_params, test_case['objective_fn'], noise_sigma
                )
            else:
                success = False
                omega_error = float('inf')
                rms_residual = float('inf')

            # Record trial result
            trial_result = {
                'noise_sigma': noise_sigma,
                'n_observations': n_obs,
                'trial': trial,
                'seed': trial_seed,
                'success': success,
                'omega_error_deg': omega_error,
                'rms_residual': rms_residual,
                'objective_value': result['f_best'],
                'n_evals': result['n_evals'],
                'wall_time': trial_time,
            }
            all_trial_results.append(trial_result)

            # Accumulate for cell statistics
            if success:
                cell_successes += 1
            cell_omega_errors.append(omega_error)
            cell_rms_residuals.append(rms_residual)

            print(f"  Trial {trial+1}: {'SUCCESS' if success else 'FAIL'}, "
                  f"omega_err={omega_error:.4f} deg/s, rms={rms_residual:.4f} mag, "
                  f"time={trial_time:.1f}s")

        # Compute cell statistics
        success_matrix[i, j] = cell_successes / N_TRIALS_PER_CELL
        mean_omega_error[i, j] = np.mean(cell_omega_errors)
        mean_rms_residual[i, j] = np.mean(cell_rms_residuals)

        print(f"  Cell success rate: {success_matrix[i, j]*100:.0f}%")

sweep_elapsed = time.time() - sweep_start_time

print("\n" + "=" * 70)
print(f"SWEEP COMPLETE in {sweep_elapsed/60:.1f} minutes")
print("=" * 70)

# %%
# Display success rate matrix
print("\nSuccess Rate Matrix (%):")
print("-" * 50)

# Header row
header = "          | " + " | ".join([f"n={n:3d}" for n in N_OBSERVATIONS_LIST]) + " |"
print(header)
print("-" * len(header))

# Data rows
for i, noise in enumerate(NOISE_LEVELS):
    row = f"σ={noise:.2f}  | " + " | ".join([f"{success_matrix[i,j]*100:5.0f}%" for j in range(n_obs_configs)]) + " |"
    print(row)

print("-" * len(header))

# %%
# Display omega error matrix
print("\nMean Omega Error Matrix (deg/s):")
print("-" * 50)

# Header row
header = "          | " + " | ".join([f"n={n:3d}" for n in N_OBSERVATIONS_LIST]) + " |"
print(header)
print("-" * len(header))

# Data rows
for i, noise in enumerate(NOISE_LEVELS):
    row = f"σ={noise:.2f}  | " + " | ".join([f"{mean_omega_error[i,j]:6.3f}" for j in range(n_obs_configs)]) + " |"
    print(row)

print("-" * len(header))
