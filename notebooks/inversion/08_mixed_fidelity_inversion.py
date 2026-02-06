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
# # Mixed-Fidelity Hierarchical Inversion
#
# This notebook implements a two-stage optimization strategy for lightcurve inversion:
#
# 1. **Stage 1 (Low-Fidelity Global Search)**: Run Differential Evolution on a
#    shadow-free objective function for fast global exploration.
# 2. **Stage 2 (High-Fidelity Local Refinement)**: Polish top candidates using
#    L-BFGS-B on the full shadow-enabled objective function.
#
# ## Goals
#
# - Quantify the speedup and accuracy trade-off of shadow-free evaluation
# - Determine whether the global landscape is preserved without shadows
# - Validate that the two-stage pipeline recovers accurate attitude parameters
# - Compare mixed-fidelity against full-fidelity baselines

# %% [markdown]
# ---
# ## Setup
#
# Reuse the Intelsat 901 test case from notebooks 04-07 with tumbling dynamics.

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
# ## 1. Load Intelsat 901 Configuration (Same as Notebooks 04-07)

# %%
# ============================================================================
# CONFIGURATION
# ============================================================================
config_path = "intelsat_901/intelsat_901_config.yaml"

# Number of observation points
n_observations = 50

# Observer/Ground Station SPICE ID
OBSERVER_ID = 399999

# Noise level for synthetic observations
noise_sigma = 0.05  # magnitudes

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
print("MIXED-FIDELITY HIERARCHICAL INVERSION")
print("=" * 70)
print(f"\nConfiguration: {config.name}")
print(f"Observations: {n_observations}")
print(f"Noise sigma: {noise_sigma} mag")

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
# Same fixed angles as notebooks 04-07
SOLAR_PANEL_ANGLE_DEG = 0.0
ANTENNA_DISH_ANGLE_DEG = 15.0

print("Fixed articulation configuration:")
print(f"  Solar panels (SP_North, SP_South): {SOLAR_PANEL_ANGLE_DEG} deg")
print(f"  Antenna dishes (AD_East, AD_West): {ANTENNA_DISH_ANGLE_DEG} deg")

# %% [markdown]
# ---
# ## 3. Calculate Inertia Tensor

# %%
# Same component masses as notebooks 04-07
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
# ## 5. Define True Attitude Parameters (Same as Notebooks 04-07)

# %%
# True initial quaternion (same as notebooks 04-07)
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
# ## 6. Generate Synthetic Lightcurve (Shadows Enabled)

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
# Add synthetic noise
np.random.seed(42)
observed_lightcurve = true_lightcurve + np.random.normal(0, noise_sigma, n_observations)

print(f"\nAdded Gaussian noise (sigma = {noise_sigma} mag)")
print(f"Observed lightcurve range: [{observed_lightcurve.min():.2f}, {observed_lightcurve.max():.2f}] mag")

# %% [markdown]
# ---
# ## 7. Create Dual-Fidelity Objective Functions
#
# Two ObjectiveFunction instances sharing the same observation data but differing
# in whether shadow ray tracing is performed:
#
# - **obj_hifi**: Full shadow computation (high fidelity, slow)
# - **obj_lofi**: No shadow computation (low fidelity, fast)

# %%
# High-fidelity objective: shadows enabled
obj_hifi = ObjectiveFunction(
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

# Low-fidelity objective: shadows disabled
obj_lofi = ObjectiveFunction(
    satellite=satellite,
    observation_times=observation_times,
    observed_lightcurve=observed_lightcurve,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=False,
    articulation_matrices=articulation_matrices,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
)

# Verify both evaluate correctly at true parameters
obj_hifi_at_true = obj_hifi.evaluate(true_params)
obj_lofi_at_true = obj_lofi.evaluate(true_params)

print("Dual-fidelity objective functions created")
print(f"\nHigh-fidelity (shadows ON):")
print(f"  Objective at true params: {obj_hifi_at_true:.6f}")
print(f"\nLow-fidelity (shadows OFF):")
print(f"  Objective at true params: {obj_lofi_at_true:.6f}")
print(f"\nDifference: {abs(obj_hifi_at_true - obj_lofi_at_true):.6f}")

# %% [markdown]
# ---
# ## 8. Define Parameter Bounds and Success Criteria

# %%
# Define parameter bounds (same as notebook 06)
omega_max_deg_per_s = 30.0
omega_max_rad_per_s = np.deg2rad(omega_max_deg_per_s)

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
    obj_value = objective_fn.evaluate(x_opt)
    n_obs = len(objective_fn.observed_lightcurve)
    rms_residual = np.sqrt(obj_value / n_obs)

    # Check success criteria
    omega_ok = omega_error_deg < OMEGA_ERROR_THRESHOLD_DEG_PER_S
    rms_ok = rms_residual < RMS_THRESHOLD_FACTOR * noise_sigma
    success = omega_ok and rms_ok

    return success, omega_error_deg, rms_residual


print(f"\nSuccess criteria:")
print(f"  Angular velocity error < {OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s")
print(f"  RMS residual < {RMS_THRESHOLD_FACTOR} x noise_sigma = {RMS_THRESHOLD_FACTOR * noise_sigma:.4f} mag")

# %% [markdown]
# ---
# ## 9. CountedObjective Wrapper (From Notebook 06)

# %%
class CountedObjective:
    """
    Wrapper that counts function evaluations and enforces a budget limit.

    Attributes
    ----------
    objective_fn : ObjectiveFunction
        The underlying objective function to wrap.
    budget : int
        Maximum number of evaluations allowed.
    n_evals : int
        Current count of evaluations.
    penalty_value : float
        Value returned when budget is exhausted.
    best_value : float
        Best (minimum) objective value seen so far.
    best_params : np.ndarray | None
        Parameters corresponding to best_value.
    """

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        budget: int,
        penalty_value: float = 1e10,
    ) -> None:
        self.objective_fn = objective_fn
        self.budget = budget
        self.penalty_value = penalty_value
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params: np.ndarray | None = None

    def __call__(self, params: np.ndarray) -> float:
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
        value = self.objective_fn.evaluate(params_normalized)

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

    def get_remaining_budget(self) -> int:
        """Get the number of evaluations remaining in the budget."""
        return max(0, self.budget - self.n_evals)

    def is_budget_exhausted(self) -> bool:
        """Check if the evaluation budget has been exhausted."""
        return self.n_evals >= self.budget


print("CountedObjective wrapper defined")

# %% [markdown]
# ---
# ## 10. Setup Verification
#
# Verify that all setup is correct by evaluating both objectives and checking success
# criteria at the true parameters.

# %%
print("\n" + "=" * 70)
print("SETUP VERIFICATION")
print("=" * 70)

# Test CountedObjective with both fidelity levels
counted_hifi = CountedObjective(obj_hifi, budget=100)
counted_lofi = CountedObjective(obj_lofi, budget=100)

val_hifi = counted_hifi(true_params)
val_lofi = counted_lofi(true_params)

print(f"\nCountedObjective test:")
print(f"  Hi-fi at true params: {val_hifi:.6f} (evals: {counted_hifi.n_evals})")
print(f"  Lo-fi at true params: {val_lofi:.6f} (evals: {counted_lofi.n_evals})")

# Evaluate success at true params (should succeed)
success_hifi, omega_err_hifi, rms_hifi = evaluate_success(
    true_params, true_params, obj_hifi, noise_sigma
)
success_lofi, omega_err_lofi, rms_lofi = evaluate_success(
    true_params, true_params, obj_lofi, noise_sigma
)

print(f"\nSuccess evaluation at true params:")
print(f"  Hi-fi: success={success_hifi}, omega_err={omega_err_hifi:.6f} deg/s, rms={rms_hifi:.6f}")
print(f"  Lo-fi: success={success_lofi}, omega_err={omega_err_lofi:.6f} deg/s, rms={rms_lofi:.6f}")

print(f"\nSetup objects available:")
print(f"  obj_hifi: ObjectiveFunction with compute_shadows_flag=True")
print(f"  obj_lofi: ObjectiveFunction with compute_shadows_flag=False")
print(f"  true_params: {true_params}")
print(f"  param_names: {param_names}")
print(f"  bounds: {len(bounds)} dimensions")
print(f"  noise_sigma: {noise_sigma}")
print(f"  evaluate_success(): success criteria function")
print(f"  CountedObjective: budget-enforcing wrapper")

print("\n" + "=" * 70)
print("NOTEBOOK 08 SETUP COMPLETE")
print("=" * 70)

# %% [markdown]
# ---
# ## Experiment 1: Fidelity Benchmarking
#
# Quantify the speedup and lightcurve discrepancy between shadow-enabled (hi-fi)
# and shadow-disabled (lo-fi) evaluation.

# %% [markdown]
# ### Experiment 1a: Evaluation Timing

# %%
# Time N evaluations of each fidelity level at true_params
N_timing = 20

# --- High-fidelity timing ---
times_hifi = []
for i in range(N_timing):
    t0 = time.perf_counter()
    obj_hifi.evaluate(true_params)
    t1 = time.perf_counter()
    times_hifi.append(t1 - t0)

times_hifi = np.array(times_hifi)

# --- Low-fidelity timing ---
times_lofi = []
for i in range(N_timing):
    t0 = time.perf_counter()
    obj_lofi.evaluate(true_params)
    t1 = time.perf_counter()
    times_lofi.append(t1 - t0)

times_lofi = np.array(times_lofi)

# --- Print results ---
mean_hifi = times_hifi.mean()
std_hifi = times_hifi.std()
mean_lofi = times_lofi.mean()
std_lofi = times_lofi.std()
speedup = mean_hifi / mean_lofi

print("=" * 70)
print("EXPERIMENT 1a: EVALUATION TIMING")
print("=" * 70)
print(f"\nHigh-fidelity (shadows ON):  {mean_hifi:.4f} ± {std_hifi:.4f} s")
print(f"Low-fidelity  (shadows OFF): {mean_lofi:.4f} ± {std_lofi:.4f} s")
print(f"\nSpeedup factor: {speedup:.1f}x")

# %% [markdown]
# ### Experiment 1b: Lightcurve Comparison at True Parameters

# %%
# Generate predicted lightcurves at true_params for both fidelity levels
# Replicate evaluate() pipeline steps 1-4 to extract predicted magnitudes

q0_true = axis_angle_to_quaternion(true_params[:3])
omega0_true = true_params[3:]

quaternions_true, _ = propagate_attitude(
    q0=q0_true,
    omega0=omega0_true,
    times=observation_times,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
)

# Body-frame vectors (same for both objectives since geometry is identical)
k1_true, k2_true = obj_hifi._compute_body_frame_vectors(quaternions_true)

# Predicted lightcurves from each fidelity level
mag_hifi = obj_hifi._generate_predicted_lightcurve(k1_true, k2_true)
mag_lofi = obj_lofi._generate_predicted_lightcurve(k1_true, k2_true)

# Compute residual between fidelity levels
mag_residual = mag_hifi - mag_lofi

print("Predicted lightcurves at true parameters:")
print(f"  Hi-fi range: [{mag_hifi.min():.2f}, {mag_hifi.max():.2f}] mag")
print(f"  Lo-fi range: [{mag_lofi.min():.2f}, {mag_lofi.max():.2f}] mag")
print(f"\nMagnitude residual (hifi - lofi):")
print(f"  Mean:  {mag_residual.mean():.4f} mag")
print(f"  Std:   {mag_residual.std():.4f} mag")
print(f"  Range: [{mag_residual.min():.4f}, {mag_residual.max():.4f}] mag")
print(f"  RMS:   {np.sqrt(np.mean(mag_residual**2)):.4f} mag")

# %% [markdown]
# ### Experiment 1c: Rank Correlation (Spearman)
#
# Evaluate both objectives on 100 random parameter samples to assess whether the
# low-fidelity landscape preserves the ranking of solutions.

# %%
from scipy.stats import spearmanr

# Evaluate both objectives on 100 random parameter samples within bounds
n_samples = 100
np.random.seed(123)

bounds_lower = np.array([b[0] for b in bounds])
bounds_upper = np.array([b[1] for b in bounds])

random_samples = np.random.uniform(bounds_lower, bounds_upper, size=(n_samples, len(bounds)))

print("=" * 70)
print("EXPERIMENT 1c: RANK CORRELATION")
print("=" * 70)
print(f"\nEvaluating {n_samples} random samples on both objectives...")

obj_values_hifi = np.empty(n_samples)
obj_values_lofi = np.empty(n_samples)

for i in range(n_samples):
    obj_values_hifi[i] = obj_hifi.evaluate(random_samples[i])
    obj_values_lofi[i] = obj_lofi.evaluate(random_samples[i])
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{n_samples} samples evaluated")

# Compute Spearman rank correlation
rho, p_value = spearmanr(obj_values_lofi, obj_values_hifi)

print(f"\nSpearman rank correlation:")
print(f"  rho = {rho:.4f}")
print(f"  p-value = {p_value:.2e}")

# %%
# Create 3-panel benchmark figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel (a): Lightcurve overlay
ax = axes[0]
ax.plot(observation_times, mag_hifi, 'b-', linewidth=1.5, label='Hi-fi (shadows ON)')
ax.plot(observation_times, mag_lofi, 'r--', linewidth=1.5, label='Lo-fi (shadows OFF)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Magnitude')
ax.set_title('(a) Lightcurve Comparison')
ax.invert_yaxis()
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (b): Magnitude residual vs time
ax = axes[1]
ax.plot(observation_times, mag_residual, 'k-', linewidth=1.0)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(observation_times, -noise_sigma, noise_sigma, alpha=0.15, color='orange',
                label=f'±noise_sigma ({noise_sigma} mag)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Residual (hi-fi − lo-fi) [mag]')
ax.set_title('(b) Fidelity Residual')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (c): Scatter of lofi vs hifi objectives with correlation annotation
ax = axes[2]
ax.scatter(obj_values_lofi, obj_values_hifi, s=15, alpha=0.6, color='steelblue', edgecolors='none')
ax.plot([obj_values_lofi.min(), obj_values_lofi.max()],
        [obj_values_lofi.min(), obj_values_lofi.max()],
        'k--', alpha=0.4, label='y = x')
ax.set_xlabel('Lo-fi Objective')
ax.set_ylabel('Hi-fi Objective')
ax.set_title('(c) Objective Correlation')
ax.annotate(f'Spearman ρ = {rho:.3f}', xy=(0.05, 0.92), xycoords='axes fraction',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('Mixed-Fidelity Benchmark', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
output_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "mixed_fidelity_benchmark.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {output_path}")
plt.show()

# %% [markdown]
# ### Experiment 1 Summary
#
# **Fidelity Benchmarking Results:**
#
# - **Speedup**: The lo-fi (shadow-free) evaluation is expected to be 5-20× faster
#   than hi-fi (shadow-enabled), depending on the satellite geometry complexity and
#   number of facets.
# - **Rank correlation**: A Spearman ρ > 0.9 indicates the lo-fi objective is a
#   good proxy for the hi-fi objective — the relative ranking of candidate solutions
#   is largely preserved even without shadow computation.
# - **Residual**: The magnitude residual between fidelity levels shows the lightcurve
#   discrepancy introduced by the all-lit approximation. If the residual RMS is
#   comparable to or smaller than the observation noise, the lo-fi approximation is
#   well-suited for global search.
#
# These results justify using the lo-fi objective for broad exploration (Stage 1)
# before refining with the hi-fi objective (Stage 2).

# %% [markdown]
# ---
# ## Experiment 2: Basin Shift Analysis
#
# Determine whether the global minimum shifts when shadows are disabled by comparing
# converged L-BFGS-B solutions from both fidelity levels.

# %% [markdown]
# ### Experiment 2a: Basin Shift from True Parameters

# %%
# Run L-BFGS-B from true_params on both fidelity levels

print("=" * 70)
print("EXPERIMENT 2a: BASIN SHIFT FROM TRUE PARAMETERS")
print("=" * 70)


def run_local_optimization(
    objective_fn: ObjectiveFunction,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    ftol: float = 1e-8,
    gtol: float = 1e-6,
    maxiter: int = 1000,
) -> tuple[np.ndarray, float, bool, int]:
    """
    Run local optimization using L-BFGS-B with axis-angle normalization.

    Returns (x_opt, f_opt, success, n_evals).
    """
    n_evals = [0]

    def wrapped_objective(params: np.ndarray) -> float:
        n_evals[0] += 1
        axis_angle = params[:3]
        omega = params[3:]
        q = axis_angle_to_quaternion(axis_angle)
        q_normalized = normalize_quaternion(q)
        axis_angle_norm = quaternion_to_axis_angle(q_normalized)
        params_normalized = np.concatenate([axis_angle_norm, omega])
        return objective_fn.evaluate(params_normalized)

    result = minimize(
        wrapped_objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol, "disp": False},
    )

    # Normalize final parameters
    final_aa = result.x[:3]
    final_omega = result.x[3:]
    q_final = axis_angle_to_quaternion(final_aa)
    q_final = normalize_quaternion(q_final)
    final_aa = quaternion_to_axis_angle(q_final)
    x_opt = np.concatenate([final_aa, final_omega])

    return x_opt, result.fun, result.success, n_evals[0]


# --- Run from true_params on both objectives ---
print("\nOptimizing from true_params...")

x_opt_hifi, f_opt_hifi, success_hifi, n_evals_hifi = run_local_optimization(
    obj_hifi, true_params, bounds
)
print(f"  Hi-fi: f={f_opt_hifi:.6f}, n_evals={n_evals_hifi}, converged={success_hifi}")

x_opt_lofi, f_opt_lofi, success_lofi, n_evals_lofi = run_local_optimization(
    obj_lofi, true_params, bounds
)
print(f"  Lo-fi: f={f_opt_lofi:.6f}, n_evals={n_evals_lofi}, converged={success_lofi}")

# --- Compare converged solutions ---
euclidean_dist = np.linalg.norm(x_opt_hifi - x_opt_lofi)
aa_displacement_rad = np.linalg.norm(x_opt_hifi[:3] - x_opt_lofi[:3])
aa_displacement_deg = np.rad2deg(aa_displacement_rad)
omega_displacement_rad = np.linalg.norm(x_opt_hifi[3:] - x_opt_lofi[3:])
omega_displacement_deg = np.rad2deg(omega_displacement_rad)

print(f"\nConverged solution comparison:")
print(f"  Euclidean distance:         {euclidean_dist:.6f}")
print(f"  Axis-angle displacement:    {aa_displacement_deg:.4f} deg ({aa_displacement_rad:.6f} rad)")
print(f"  Omega displacement:         {omega_displacement_deg:.4f} deg/s ({omega_displacement_rad:.6f} rad/s)")

print(f"\nHi-fi converged params: {x_opt_hifi}")
print(f"Lo-fi converged params: {x_opt_lofi}")
print(f"True params:            {true_params}")

# %% [markdown]
# ### Experiment 2b: Perturbed Starts Basin Shift

# %%
# Reuse sample_in_ball pattern from notebook 05
# Define parameter scales: 1 degree for both axis-angle and omega

AXIS_ANGLE_SCALE = np.deg2rad(1.0)  # 1 degree in radians
OMEGA_SCALE = np.deg2rad(1.0)       # 1 deg/s in rad/s

param_scales = np.array([
    AXIS_ANGLE_SCALE, AXIS_ANGLE_SCALE, AXIS_ANGLE_SCALE,
    OMEGA_SCALE, OMEGA_SCALE, OMEGA_SCALE,
])


def sample_in_ball(
    center: np.ndarray,
    radius: float,
    n_samples: int,
    param_scales: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample points uniformly within a hypersphere around center.

    Radius is measured in scaled units where each parameter is divided by its scale.
    """
    if seed is not None:
        np.random.seed(seed)

    n_dims = len(center)
    samples = np.zeros((n_samples, n_dims))

    for i in range(n_samples):
        direction = np.random.randn(n_dims)
        direction /= np.linalg.norm(direction)
        u = np.random.random()
        r = radius * (u ** (1.0 / n_dims))
        scaled_offset = r * direction
        param_offset = scaled_offset * param_scales
        samples[i] = center + param_offset

    return samples


# Basin radius from notebook 05 analysis: use a moderate radius that gives
# reasonable local optimization success (e.g., 2 degrees)
BASIN_RADIUS_DEG = 2.0
N_PERTURBED = 20

print("=" * 70)
print("EXPERIMENT 2b: PERTURBED STARTS BASIN SHIFT")
print("=" * 70)
print(f"\nBasin radius: {BASIN_RADIUS_DEG} degrees")
print(f"Number of perturbed starts: {N_PERTURBED}")

# Generate perturbed starting points
perturbed_starts = sample_in_ball(
    center=true_params,
    radius=BASIN_RADIUS_DEG,
    n_samples=N_PERTURBED,
    param_scales=param_scales,
    seed=42,
)

print(f"Generated {N_PERTURBED} perturbed starting points")

# %%
# Run L-BFGS-B from each perturbed start on both fidelity levels
print("\nRunning pairwise optimizations...")
print("-" * 50)

results_hifi = []
results_lofi = []
pairwise_displacements = {
    'euclidean': [],
    'aa_deg': [],
    'omega_deg_s': [],
}

for i in range(N_PERTURBED):
    x0 = perturbed_starts[i]

    # Optimize on hi-fi
    x_hifi_i, f_hifi_i, _, _ = run_local_optimization(obj_hifi, x0, bounds)
    results_hifi.append(x_hifi_i)

    # Optimize on lo-fi
    x_lofi_i, f_lofi_i, _, _ = run_local_optimization(obj_lofi, x0, bounds)
    results_lofi.append(x_lofi_i)

    # Compute pairwise displacement
    euc_dist = np.linalg.norm(x_hifi_i - x_lofi_i)
    aa_disp = np.rad2deg(np.linalg.norm(x_hifi_i[:3] - x_lofi_i[:3]))
    omega_disp = np.rad2deg(np.linalg.norm(x_hifi_i[3:] - x_lofi_i[3:]))

    pairwise_displacements['euclidean'].append(euc_dist)
    pairwise_displacements['aa_deg'].append(aa_disp)
    pairwise_displacements['omega_deg_s'].append(omega_disp)

    if (i + 1) % 5 == 0:
        print(f"  {i + 1}/{N_PERTURBED} starts completed")

results_hifi = np.array(results_hifi)
results_lofi = np.array(results_lofi)

for key in pairwise_displacements:
    pairwise_displacements[key] = np.array(pairwise_displacements[key])

print(f"\nAll {N_PERTURBED} pairwise optimizations completed")

# %%
# Report displacement statistics
print("\n" + "=" * 70)
print("BASIN SHIFT DISPLACEMENT STATISTICS")
print("=" * 70)

print(f"\n{'Metric':<30} | {'Mean':>10} | {'Std':>10} | {'Max':>10}")
print("-" * 70)
print(f"{'Euclidean distance':<30} | {pairwise_displacements['euclidean'].mean():>10.6f} | "
      f"{pairwise_displacements['euclidean'].std():>10.6f} | "
      f"{pairwise_displacements['euclidean'].max():>10.6f}")
print(f"{'Axis-angle displacement (deg)':<30} | {pairwise_displacements['aa_deg'].mean():>10.4f} | "
      f"{pairwise_displacements['aa_deg'].std():>10.4f} | "
      f"{pairwise_displacements['aa_deg'].max():>10.4f}")
print(f"{'Omega displacement (deg/s)':<30} | {pairwise_displacements['omega_deg_s'].mean():>10.4f} | "
      f"{pairwise_displacements['omega_deg_s'].std():>10.4f} | "
      f"{pairwise_displacements['omega_deg_s'].max():>10.4f}")

# Qualitative assessment based on basin width
# Basin width is the diameter = 2 * BASIN_RADIUS_DEG
basin_width_deg = 2 * BASIN_RADIUS_DEG
mean_aa_shift = pairwise_displacements['aa_deg'].mean()
shift_fraction = mean_aa_shift / basin_width_deg * 100

print(f"\nQualitative assessment:")
print(f"  Mean axis-angle shift: {mean_aa_shift:.4f} deg")
print(f"  Basin width (diameter): {basin_width_deg:.1f} deg")
print(f"  Shift as fraction of basin width: {shift_fraction:.2f}%")

if shift_fraction < 1.0:
    assessment = "NEGLIGIBLE"
    detail = "The lo-fi and hi-fi minima are practically co-located."
elif shift_fraction < 10.0:
    assessment = "MODERATE"
    detail = "The lo-fi minimum is shifted but remains within the hi-fi basin."
else:
    assessment = "SIGNIFICANT"
    detail = "The lo-fi minimum is substantially shifted; handoff refinement is critical."

print(f"\n  Basin shift: {assessment} (<1% = negligible, 1-10% = moderate, >10% = significant)")
print(f"  {detail}")

# %% [markdown]
# ### Experiment 2 Summary
#
# **Basin Shift Analysis Results:**
#
# - **From true parameters**: L-BFGS-B converges to slightly different minima
#   depending on fidelity level. The displacement quantifies how much the
#   all-lit approximation biases the optimum.
# - **From perturbed starts**: The pairwise displacements across 20 starting
#   points characterize the typical shift magnitude and its variability.
# - **Qualitative assessment**: The shift as a fraction of the basin width
#   determines whether the lo-fi Stage 1 solution is a good starting point
#   for hi-fi Stage 2 refinement.
#
# A negligible or moderate shift confirms that the two-stage mixed-fidelity
# approach is viable: the lo-fi global search identifies candidates close
# enough to the hi-fi minimum for local refinement to succeed.
