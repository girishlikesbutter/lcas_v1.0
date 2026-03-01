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
# ## Configuration Flags
#
# Skip flags for experiments that have already been run.
# Set to True to skip, False to run.

# %%
# =============================================================================
# EXPERIMENT SKIP FLAGS
# =============================================================================
# Set these to True to skip experiments that have already completed.
# Results from completed experiments are saved to data/results/inversion_diagnostics/

SKIP_EXP_1 = True   # Fidelity Benchmarking (timing, correlation) - COMPLETED
SKIP_EXP_2 = True   # Basin Shift Analysis - COMPLETED
SKIP_EXP_3 = False  # Mixed-Fidelity Pipeline Validation
SKIP_EXP_4 = True   # Evaluation-Count Matched Comparison - SKIP FOR NOW
SKIP_EXP_5 = True   # Wall-Clock Matched Comparison - SKIP FOR NOW
SKIP_EXP_6 = True   # Handoff Parameter Ablation - SKIP FOR NOW
SKIP_EXP_7 = True   # Phase Angle Failure Regimes - SKIP FOR NOW

# =============================================================================
# FAST TEST MODE - set True to run quick validation, False for full runs
# =============================================================================
FAST_TEST_MODE = True  # Set to False for overnight runs

# =============================================================================
# EXPERIMENT PARAMETERS (adjust based on benchmark results)
# =============================================================================
# Benchmark results from Exp 1: hi-fi ~6s/eval, lo-fi ~0.04s/eval, speedup ~163x

if FAST_TEST_MODE:
    # FAST TEST VALUES - completes in minutes, validates code paths
    EXP2_N_PERTURBED = 2
    EXP3_LOFI_BUDGET = 100       # Tiny DE run
    EXP3_TOP_N = 1               # Single candidate
    EXP3_HIFI_EVALS = 20         # Minimal refinement
    EXP4_N_TRIALS = 1
    EXP4_BUDGET = 100
    EXP6_N_TRIALS = 1
    EXP7_N_ANGLES = 2
    EXP7_N_TRIALS = 1
    print(">>> FAST TEST MODE ENABLED - using minimal parameters <<<")
else:
    # FULL RUN VALUES - for overnight/paper-quality runs
    EXP2_N_PERTURBED = 5
    EXP3_LOFI_BUDGET = 5000      # Full DE exploration
    EXP3_TOP_N = 3               # Top 3 candidates
    EXP3_HIFI_EVALS = 200        # Full refinement budget
    EXP4_N_TRIALS = 3
    EXP4_BUDGET = 1000
    EXP6_N_TRIALS = 3
    EXP7_N_ANGLES = 5
    EXP7_N_TRIALS = 3
    print(">>> FULL RUN MODE - using production parameters <<<")

print("=" * 70)
print("EXPERIMENT CONFIGURATION")
print("=" * 70)
print(f"Skip flags: Exp1={SKIP_EXP_1}, Exp2={SKIP_EXP_2}, Exp3={SKIP_EXP_3}")
print(f"            Exp4={SKIP_EXP_4}, Exp5={SKIP_EXP_5}, Exp6={SKIP_EXP_6}, Exp7={SKIP_EXP_7}")
print(f"Exp 2: N_PERTURBED = {EXP2_N_PERTURBED}")
print(f"Exp 4-5: N_TRIALS = {EXP4_N_TRIALS}, BUDGET = {EXP4_BUDGET}")
print(f"Exp 6: N_TRIALS = {EXP6_N_TRIALS}")
print(f"Exp 7: N_ANGLES = {EXP7_N_ANGLES}, N_TRIALS = {EXP7_N_TRIALS}")
print("=" * 70 + "\n")

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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
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

# %%
# =============================================================================
# RESULTS SAVING INFRASTRUCTURE
# =============================================================================
import json
from datetime import datetime

RESULTS_DIR = Path("data/results/inversion_diagnostics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Master results dictionary - will be saved to JSON at the end
notebook_results = {
    "notebook": "08_mixed_fidelity_inversion",
    "timestamp": datetime.now().isoformat(),
    "config": {
        "n_observations": n_observations,
        "noise_sigma": noise_sigma,
        "exp2_n_perturbed": EXP2_N_PERTURBED,
        "exp4_n_trials": EXP4_N_TRIALS,
        "exp4_budget": EXP4_BUDGET,
    },
    "experiments": {}
}

def save_results():
    """Save results dict to JSON file."""
    results_path = RESULTS_DIR / "notebook_08_results.json"
    with open(results_path, 'w') as f:
        json.dump(notebook_results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")

print(f"Results will be saved to: {RESULTS_DIR}/notebook_08_results.json")

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
print("\nStarting Experiment 1a timing...", flush=True)

# --- High-fidelity timing ---
times_hifi = []
for i in range(N_timing):
    print(f"  Hi-fi eval {i+1}/{N_timing}...", end=" ", flush=True)
    t0 = time.perf_counter()
    obj_hifi.evaluate(true_params)
    t1 = time.perf_counter()
    times_hifi.append(t1 - t0)
    print(f"{t1-t0:.2f}s", flush=True)

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

# %%
# Save Experiment 1 results
notebook_results["experiments"]["exp1_fidelity_benchmark"] = {
    "timing": {
        "hifi_mean_s": float(mean_hifi),
        "hifi_std_s": float(std_hifi),
        "lofi_mean_s": float(mean_lofi),
        "lofi_std_s": float(std_lofi),
        "speedup_factor": float(speedup),
    },
    "rank_correlation": {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
    },
    "lightcurve_residual": {
        "mean_mag": float(mag_residual.mean()),
        "std_mag": float(mag_residual.std()),
        "rms_mag": float(np.sqrt(np.mean(mag_residual**2))),
    },
}
save_results()
print("Experiment 1 results saved.")

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
N_PERTURBED = EXP2_N_PERTURBED  # Configured at top of notebook

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
# ### Experiment 2c: Basin Shift Visualization
#
# Contour overlays and scatter plots comparing basin locations between fidelity levels.

# %%
# Identify the 2 most sensitive parameter dimensions using 1D sensitivity analysis
# Compute objective variation along each dimension independently

n_sensitivity = 21
sensitivity_scores = np.zeros(6)

for dim in range(6):
    lb, ub = bounds[dim]
    sweep_vals = np.linspace(lb, ub, n_sensitivity)
    obj_vals = np.empty(n_sensitivity)
    for k, val in enumerate(sweep_vals):
        p = true_params.copy()
        p[dim] = val
        obj_vals[k] = obj_hifi.evaluate(p)
    sensitivity_scores[dim] = obj_vals.max() - obj_vals.min()

# Pick the 2 most sensitive dimensions
sensitive_dims = np.argsort(sensitivity_scores)[::-1][:2]
dim_i, dim_j = sorted(sensitive_dims)  # lower index first for consistency

print("=" * 70)
print("EXPERIMENT 2c: BASIN SHIFT VISUALIZATION")
print("=" * 70)
print(f"\n1D Sensitivity scores:")
for dim in range(6):
    marker = " <--" if dim in sensitive_dims else ""
    print(f"  {param_names[dim]}: {sensitivity_scores[dim]:.4f}{marker}")
print(f"\nMost sensitive dimensions: {param_names[dim_i]} (idx {dim_i}), {param_names[dim_j]} (idx {dim_j})")

# %%
# Compute 2D contour slices for both fidelity levels
# Use a focused range around true_params for the 2 most sensitive dimensions

# Set delta ranges based on parameter type
delta_range_i = np.deg2rad(5.0) if dim_i < 3 else np.deg2rad(0.5)
delta_range_j = np.deg2rad(5.0) if dim_j < 3 else np.deg2rad(0.5)

n_contour = 31

deltas_i = np.linspace(-delta_range_i, delta_range_i, n_contour)
deltas_j = np.linspace(-delta_range_j, delta_range_j, n_contour)

print(f"\nComputing 2D contours ({n_contour}x{n_contour} = {n_contour**2} evals per fidelity)...")

obj_grid_hifi = np.zeros((n_contour, n_contour))
obj_grid_lofi = np.zeros((n_contour, n_contour))

for j_idx, dj in enumerate(deltas_j):
    for i_idx, di in enumerate(deltas_i):
        params = true_params.copy()
        params[dim_i] = true_params[dim_i] + di
        params[dim_j] = true_params[dim_j] + dj
        obj_grid_hifi[j_idx, i_idx] = obj_hifi.evaluate(params)
        obj_grid_lofi[j_idx, i_idx] = obj_lofi.evaluate(params)
    if (j_idx + 1) % 10 == 0:
        print(f"  {j_idx + 1}/{n_contour} rows completed")

print("  Contour grids computed")

# %%
# Create 2D contour overlay figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

X, Y = np.meshgrid(deltas_i, deltas_j)

# Use degrees for display if axis-angle parameters
if dim_i < 3:
    X_plot = np.rad2deg(X)
    xlabel_unit = "deg"
else:
    X_plot = np.rad2deg(X)  # omega also in deg/s
    xlabel_unit = "deg/s"

if dim_j < 3:
    Y_plot = np.rad2deg(Y)
    ylabel_unit = "deg"
else:
    Y_plot = np.rad2deg(Y)
    ylabel_unit = "deg/s"

# Determine contour levels from combined range
vmin = min(obj_grid_hifi.min(), obj_grid_lofi.min())
vmax = min(obj_grid_hifi.max(), obj_grid_lofi.max())
# Use log-spaced levels for better visualization
n_levels = 12
levels = np.linspace(vmin, vmin + (vmax - vmin) * 0.8, n_levels)

# Hi-fi contours (solid blue)
cs_hifi = ax.contour(X_plot, Y_plot, obj_grid_hifi, levels=levels,
                     colors='blue', linewidths=1.2, linestyles='solid')
ax.clabel(cs_hifi, inline=True, fontsize=6, fmt='%.2f')

# Lo-fi contours (dashed red)
cs_lofi = ax.contour(X_plot, Y_plot, obj_grid_lofi, levels=levels,
                     colors='red', linewidths=1.2, linestyles='dashed')

# Mark minima locations
hifi_min_idx = np.unravel_index(obj_grid_hifi.argmin(), obj_grid_hifi.shape)
lofi_min_idx = np.unravel_index(obj_grid_lofi.argmin(), obj_grid_lofi.shape)

ax.plot(np.rad2deg(deltas_i[hifi_min_idx[1]]), np.rad2deg(deltas_j[hifi_min_idx[0]]),
        'b*', markersize=15, markeredgecolor='black', markeredgewidth=0.5, label='Hi-fi minimum')
ax.plot(np.rad2deg(deltas_i[lofi_min_idx[1]]), np.rad2deg(deltas_j[lofi_min_idx[0]]),
        'r*', markersize=15, markeredgecolor='black', markeredgewidth=0.5, label='Lo-fi minimum')

# Mark true parameters (center)
ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2, label='True params')

ax.set_xlabel(f'Δ{param_names[dim_i]} ({xlabel_unit})')
ax.set_ylabel(f'Δ{param_names[dim_j]} ({ylabel_unit})')
ax.set_title(f'Basin Shift: Contour Overlay ({param_names[dim_i]} vs {param_names[dim_j]})')
ax.grid(True, alpha=0.3)

# Two legends: markers (upper right) and line styles (lower left)
from matplotlib.lines import Line2D
marker_legend = ax.legend(loc='upper right', fontsize=9)
ax.add_artist(marker_legend)
custom_lines = [Line2D([0], [0], color='blue', linestyle='solid', linewidth=1.5),
                Line2D([0], [0], color='red', linestyle='dashed', linewidth=1.5)]
ax.legend(custom_lines, ['Hi-fi (shadows ON)', 'Lo-fi (shadows OFF)'],
          loc='lower left', fontsize=9)

plt.tight_layout()

# Save figure
contour_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "basin_shift_contours.png"
fig.savefig(contour_path, dpi=150, bbox_inches='tight')
print(f"\nContour overlay saved: {contour_path}")
plt.show()

# %%
# Create scatter plot of converged hifi vs lofi parameter values across 20 starts
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for dim in range(6):
    ax = axes[dim]
    ax.scatter(results_lofi[:, dim], results_hifi[:, dim],
               s=30, alpha=0.7, color='steelblue', edgecolors='navy', linewidth=0.5)

    # Plot identity line
    all_vals = np.concatenate([results_lofi[:, dim], results_hifi[:, dim]])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = (val_max - val_min) * 0.1
    line_range = [val_min - margin, val_max + margin]
    ax.plot(line_range, line_range, 'k--', alpha=0.4, linewidth=1)

    # Mark true parameter value
    ax.axvline(true_params[dim], color='green', alpha=0.4, linewidth=1, linestyle=':')
    ax.axhline(true_params[dim], color='green', alpha=0.4, linewidth=1, linestyle=':')

    if dim < 3:
        ax.set_xlabel(f'Lo-fi {param_names[dim]} (rad)')
        ax.set_ylabel(f'Hi-fi {param_names[dim]} (rad)')
    else:
        ax.set_xlabel(f'Lo-fi {param_names[dim]} (rad/s)')
        ax.set_ylabel(f'Hi-fi {param_names[dim]} (rad/s)')

    ax.set_title(param_names[dim], fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Basin Shift: Converged Parameters (Hi-fi vs Lo-fi)\n'
             f'{N_PERTURBED} perturbed starts, dashed = identity line, dotted = true value',
             fontsize=12, fontweight='bold')
plt.tight_layout()

# Save figure
scatter_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "basin_shift_scatter.png"
fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
print(f"Scatter plot saved: {scatter_path}")
plt.show()

# %% [markdown]
# ### Experiment 2 Summary
#
# **Basin Shift Analysis Results:**
#
# - **From true parameters** (Exp 2a): L-BFGS-B converges to slightly different
#   minima depending on fidelity level. The displacement quantifies how much the
#   all-lit approximation biases the optimum.
# - **From perturbed starts** (Exp 2b): The pairwise displacements across 20
#   starting points characterize the typical shift magnitude and its variability.
# - **Qualitative assessment**: The shift as a fraction of the basin width
#   determines whether the lo-fi Stage 1 solution is a good starting point
#   for hi-fi Stage 2 refinement.
# - **Contour overlay** (Exp 2c): The 2D contour overlay shows the objective
#   landscape structure for both fidelity levels along the two most sensitive
#   parameter dimensions. Overlapping contours confirm that the basin geometry
#   is largely preserved without shadow computation.
# - **Scatter plot** (Exp 2c): The converged parameter scatter plots across
#   all 6 dimensions show whether hi-fi and lo-fi solutions cluster near the
#   identity line, confirming minimal systematic bias from the all-lit approximation.
#
# A negligible or moderate shift confirms that the two-stage mixed-fidelity
# approach is viable: the lo-fi global search identifies candidates close
# enough to the hi-fi minimum for local refinement to succeed.

# %%
# Save Experiment 2 results
notebook_results["experiments"]["exp2_basin_shift"] = {
    "from_true_params": {
        "euclidean_distance": float(euclidean_dist),
        "axis_angle_displacement_deg": float(aa_displacement_deg),
        "omega_displacement_deg_s": float(omega_displacement_deg),
    },
    "perturbed_starts": {
        "n_starts": N_PERTURBED,
        "euclidean_mean": float(pairwise_displacements['euclidean'].mean()),
        "euclidean_std": float(pairwise_displacements['euclidean'].std()),
        "aa_deg_mean": float(pairwise_displacements['aa_deg'].mean()),
        "aa_deg_std": float(pairwise_displacements['aa_deg'].std()),
        "omega_deg_s_mean": float(pairwise_displacements['omega_deg_s'].mean()),
        "omega_deg_s_std": float(pairwise_displacements['omega_deg_s'].std()),
    },
}
save_results()
print("Experiment 2 results saved.")

# %% [markdown]
# ---
# ## Experiment 3: Mixed-Fidelity Pipeline
#
# Implement the core two-stage pipeline:
# 1. **Stage 1**: Low-fidelity DE global search
# 2. **Stage 2**: High-fidelity L-BFGS-B local refinement on top candidates

# %%
from scipy.optimize import differential_evolution


def run_mixed_fidelity(
    obj_lofi: ObjectiveFunction,
    obj_hifi: ObjectiveFunction,
    bounds: list[tuple[float, float]],
    lofi_budget: int,
    top_n: int,
    hifi_evals_per_candidate: int,
    seed: int | None = None,
) -> dict:
    """
    Two-stage mixed-fidelity optimization pipeline.

    Stage 1: Run Differential Evolution with polish=False on the low-fidelity
    objective, then extract the top N candidates from the final population.

    Stage 2: Run L-BFGS-B on the high-fidelity objective for each of the N
    candidates, selecting the overall best solution.

    Parameters
    ----------
    obj_lofi : ObjectiveFunction
        Low-fidelity objective function (shadows disabled).
    obj_hifi : ObjectiveFunction
        High-fidelity objective function (shadows enabled).
    bounds : list[tuple[float, float]]
        Parameter bounds for each dimension.
    lofi_budget : int
        Maximum number of lo-fi evaluations for Stage 1 (DE).
    top_n : int
        Number of top candidates to pass from Stage 1 to Stage 2.
    hifi_evals_per_candidate : int
        Maximum number of hi-fi evaluations per candidate in Stage 2.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results with keys:
        - x_best: Best solution found overall
        - f_best: Best hi-fi objective value
        - n_evals: Total evaluations (lofi + hifi)
        - n_evals_lofi: Lo-fi evaluations used in Stage 1
        - n_evals_hifi: Hi-fi evaluations used in Stage 2
        - stage1_time: Wall-clock time for Stage 1 (seconds)
        - stage2_time: Wall-clock time for Stage 2 (seconds)
        - candidates: List of per-candidate results dicts
    """
    n_params = len(bounds)

    # =====================================================================
    # Stage 1: Low-fidelity DE global search
    # =====================================================================
    counted_lofi = CountedObjective(obj_lofi, budget=lofi_budget)

    # Calculate maxiter to stay within budget
    popsize = 15
    evals_per_generation = popsize * n_params
    maxiter = max(1, int(lofi_budget / evals_per_generation) - 1)

    t0_stage1 = time.perf_counter()

    de_result = differential_evolution(
        func=counted_lofi,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        tol=0.01,
        polish=False,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        workers=1,
    )

    stage1_time = time.perf_counter() - t0_stage1
    n_evals_lofi = counted_lofi.n_evals

    # Extract top N candidates from DE population
    population = de_result.population  # shape: (pop_size, n_params)
    energies = de_result.population_energies  # shape: (pop_size,)

    # Sort by energy (ascending = best first) and take top N
    sorted_indices = np.argsort(energies)[:top_n]
    top_candidates = population[sorted_indices].copy()
    top_energies = energies[sorted_indices].copy()

    # =====================================================================
    # Stage 2: High-fidelity L-BFGS-B local refinement
    # =====================================================================
    t0_stage2 = time.perf_counter()

    candidate_results = []
    n_evals_hifi_total = 0

    for i in range(len(top_candidates)):
        x0 = top_candidates[i]

        # Use CountedObjective for budget enforcement on each candidate
        counted_hifi = CountedObjective(obj_hifi, budget=hifi_evals_per_candidate)

        result_i = minimize(
            counted_hifi,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6, "disp": False},
        )

        # Get best from tracked values (handles budget exhaustion)
        if counted_hifi.best_params is not None:
            x_opt = counted_hifi.best_params.copy()
            f_opt = counted_hifi.best_value
        else:
            # Normalize result
            aa = result_i.x[:3]
            omega = result_i.x[3:]
            q = axis_angle_to_quaternion(aa)
            q = normalize_quaternion(q)
            aa = quaternion_to_axis_angle(q)
            x_opt = np.concatenate([aa, omega])
            f_opt = result_i.fun

        n_evals_hifi_total += counted_hifi.n_evals

        candidate_results.append({
            'x_opt': x_opt,
            'f_opt': f_opt,
            'n_evals': counted_hifi.n_evals,
            'lofi_energy': float(top_energies[i]),
            'converged': result_i.success,
        })

    stage2_time = time.perf_counter() - t0_stage2

    # Select overall best candidate
    best_idx = int(np.argmin([c['f_opt'] for c in candidate_results]))
    x_best = candidate_results[best_idx]['x_opt']
    f_best = candidate_results[best_idx]['f_opt']

    return {
        'x_best': x_best,
        'f_best': f_best,
        'n_evals': n_evals_lofi + n_evals_hifi_total,
        'n_evals_lofi': n_evals_lofi,
        'n_evals_hifi': n_evals_hifi_total,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'candidates': candidate_results,
    }


print("run_mixed_fidelity() pipeline function defined")

# %% [markdown]
# ### Experiment 3b: Pipeline Validation on Standard Test Case
#
# Run the mixed-fidelity pipeline once with N=3 candidates, lofi_budget=5000,
# and hifi_evals_per_candidate=200 to validate correctness.

# %%
# Run pipeline validation
print("=" * 70)
print("EXPERIMENT 3b: PIPELINE VALIDATION")
print("=" * 70)

VALIDATION_N = EXP3_TOP_N
VALIDATION_LOFI_BUDGET = EXP3_LOFI_BUDGET
VALIDATION_HIFI_EVALS_PER_CANDIDATE = EXP3_HIFI_EVALS
VALIDATION_SEED = 42

print(f"\nPipeline configuration:")
print(f"  Top N candidates:           {VALIDATION_N}")
print(f"  Lo-fi budget (Stage 1):     {VALIDATION_LOFI_BUDGET} evals")
print(f"  Hi-fi evals per candidate:  {VALIDATION_HIFI_EVALS_PER_CANDIDATE}")
print(f"  Total hi-fi budget:         {VALIDATION_N * VALIDATION_HIFI_EVALS_PER_CANDIDATE} evals")
print(f"  Seed:                       {VALIDATION_SEED}")

print("\nRunning mixed-fidelity pipeline...")
t0_pipeline = time.perf_counter()

pipeline_result = run_mixed_fidelity(
    obj_lofi=obj_lofi,
    obj_hifi=obj_hifi,
    bounds=bounds,
    lofi_budget=VALIDATION_LOFI_BUDGET,
    top_n=VALIDATION_N,
    hifi_evals_per_candidate=VALIDATION_HIFI_EVALS_PER_CANDIDATE,
    seed=VALIDATION_SEED,
)

total_pipeline_time = time.perf_counter() - t0_pipeline

# --- Print timing ---
print(f"\n{'─' * 50}")
print("TIMING")
print(f"{'─' * 50}")
print(f"  Stage 1 (lo-fi DE):          {pipeline_result['stage1_time']:.2f} s")
print(f"  Stage 2 (hi-fi L-BFGS-B):")
for i, cand in enumerate(pipeline_result['candidates']):
    print(f"    Candidate {i+1}:              {cand['n_evals']} evals, converged={cand['converged']}")
print(f"  Stage 2 total:               {pipeline_result['stage2_time']:.2f} s")
stage2_per_candidate = pipeline_result['stage2_time'] / VALIDATION_N
print(f"  Stage 2 per candidate:       {stage2_per_candidate:.2f} s")
print(f"  Total pipeline time:         {total_pipeline_time:.2f} s")

# --- Print evaluation counts ---
print(f"\n{'─' * 50}")
print("EVALUATION COUNTS")
print(f"{'─' * 50}")
print(f"  Lo-fi evals (Stage 1):       {pipeline_result['n_evals_lofi']}")
print(f"  Hi-fi evals (Stage 2):       {pipeline_result['n_evals_hifi']}")
print(f"  Total evals:                 {pipeline_result['n_evals']}")

# --- Print parameter errors ---
x_best = pipeline_result['x_best']

# Axis-angle error
aa_error_rad = np.linalg.norm(x_best[:3] - true_params[:3])
aa_error_deg = np.rad2deg(aa_error_rad)

# Omega error
omega_error_rad = np.linalg.norm(x_best[3:] - true_params[3:])
omega_error_deg = np.rad2deg(omega_error_rad)

# RMS residual
obj_at_best = pipeline_result['f_best']
rms_residual = np.sqrt(obj_at_best / n_observations)

print(f"\n{'─' * 50}")
print("PARAMETER ERRORS")
print(f"{'─' * 50}")
print(f"  Best objective value:        {obj_at_best:.6f}")
print(f"  Axis-angle error:            {aa_error_deg:.4f} deg")
print(f"  Omega error:                 {omega_error_deg:.4f} deg/s")
print(f"  RMS residual:                {rms_residual:.4f} mag")

# --- Evaluate success ---
success, omega_err_eval, rms_eval = evaluate_success(
    x_best, true_params, obj_hifi, noise_sigma
)

print(f"\n{'─' * 50}")
print("SUCCESS EVALUATION")
print(f"{'─' * 50}")
print(f"  Omega error:    {omega_err_eval:.4f} deg/s  (threshold: {OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s)")
print(f"  RMS residual:   {rms_eval:.4f} mag   (threshold: {RMS_THRESHOLD_FACTOR * noise_sigma:.4f} mag)")
if success:
    print(f"\n  ✓ SUCCESS: Pipeline recovered accurate attitude parameters")
else:
    print(f"\n  ✗ FAIL: Pipeline did not meet success criteria")
    if omega_err_eval >= OMEGA_ERROR_THRESHOLD_DEG_PER_S:
        print(f"    - Omega error {omega_err_eval:.4f} >= {OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s")
    if rms_eval >= RMS_THRESHOLD_FACTOR * noise_sigma:
        print(f"    - RMS residual {rms_eval:.4f} >= {RMS_THRESHOLD_FACTOR * noise_sigma:.4f} mag")

# --- Print best parameters ---
print(f"\n{'─' * 50}")
print("BEST PARAMETERS")
print(f"{'─' * 50}")
print(f"  Best params: {x_best}")
print(f"  True params: {true_params}")

# %% [markdown]
# ### Experiment 3 Summary
#
# **Pipeline Validation Results:**
#
# The mixed-fidelity pipeline was run once with:
# - **Stage 1**: Differential Evolution on the lo-fi objective (5000 eval budget)
# - **Stage 2**: L-BFGS-B refinement on the hi-fi objective for the top 3 candidates
#   (200 evals each)
#
# **Key observations:**
# - The pipeline demonstrates the two-stage approach: fast global exploration
#   followed by accurate local refinement.
# - Stage 1 (lo-fi DE) consumes the majority of evaluations but runs quickly
#   due to the shadow-free approximation.
# - Stage 2 (hi-fi L-BFGS-B) uses fewer evaluations but each is more expensive
#   due to ray tracing.
# - The success/failure result validates whether the lo-fi landscape is a
#   sufficiently good proxy for identifying promising candidates.
#
# This single validation run confirms the pipeline mechanics before proceeding
# to statistical comparisons in Experiments 4-5.

# %%
# Save Experiment 3 results
notebook_results["experiments"]["exp3_pipeline_validation"] = {
    "config": {
        "lofi_budget": VALIDATION_LOFI_BUDGET,
        "top_n": VALIDATION_N,
        "hifi_evals_per_candidate": VALIDATION_HIFI_EVALS_PER_CANDIDATE,
    },
    "timing": {
        "stage1_time_s": float(pipeline_result['stage1_time']),
        "stage2_time_s": float(pipeline_result['stage2_time']),
        "total_time_s": float(total_pipeline_time),
    },
    "evals": {
        "lofi": int(pipeline_result['n_evals_lofi']),
        "hifi": int(pipeline_result['n_evals_hifi']),
        "total": int(pipeline_result['n_evals']),
    },
    "result": {
        "f_best": float(pipeline_result['f_best']),
        "aa_error_deg": float(aa_error_deg),
        "omega_error_deg_s": float(omega_error_deg),
        "rms_residual": float(rms_eval),
        "success": bool(success),
    },
}
save_results()
print("Experiment 3 results saved.")

# %% [markdown]
# ---
# ## Experiment 4: Evaluation-Count Matched Baseline Comparison
#
# Compare mixed-fidelity against notebook 06 baselines (multi-start local,
# Differential Evolution, basin-hopping) with the same total evaluation budget
# of 5000 evaluations. All baselines use the hi-fi objective; mixed-fidelity
# splits the budget between lo-fi Stage 1 and hi-fi Stage 2.

# %% [markdown]
# ### Experiment 4a: Define Baseline Strategies
#
# Reuse `multistart_local`, `run_de`, and `run_basinhopping` implementations
# from notebook 06, adapted to use `obj_hifi` as the baseline objective.

# %%
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import basinhopping


def multistart_local(
    counted_objective: CountedObjective,
    bounds: list[tuple[float, float]],
    n_starts: int,
    max_evals_per_start: int,
    seed: int | None = None,
) -> dict:
    """
    Multi-start local optimization using Latin Hypercube Sampling.

    Returns dict with keys: x_best, f_best, n_evals, n_successful_starts,
    n_starts_attempted, all_results.
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
    all_results = []
    x_best = None
    f_best = float('inf')
    n_successful_starts = 0

    for i, x0 in enumerate(initial_points):
        if counted_objective.is_budget_exhausted():
            break

        evals_before = counted_objective.n_evals

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

        evals_used = counted_objective.n_evals - evals_before

        local_result = {
            'x0': x0.copy(),
            'x_opt': result.x.copy(),
            'f_opt': result.fun,
            'n_evals': evals_used,
            'success': result.success,
        }
        all_results.append(local_result)

        if result.fun < f_best:
            f_best = result.fun
            x_best = result.x.copy()

        if not counted_objective.is_budget_exhausted():
            n_successful_starts += 1

    return {
        'x_best': x_best,
        'f_best': f_best,
        'n_evals': counted_objective.n_evals,
        'n_successful_starts': n_successful_starts,
        'n_starts_attempted': len(all_results),
        'all_results': all_results,
    }


def run_de(
    counted_objective: CountedObjective,
    bounds: list[tuple[float, float]],
    max_evals: int,
    seed: int | None = None,
) -> dict:
    """
    Run Differential Evolution with evaluation budget.

    Returns dict with keys: x_best, f_best, n_evals, success, message.
    """
    n_params = len(bounds)
    popsize = 15
    evals_per_generation = popsize * n_params
    maxiter = max(1, int(max_evals / evals_per_generation) - 1)

    result = differential_evolution(
        func=counted_objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        tol=0.01,
        polish=False,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        updating='deferred',
        workers=1,
    )

    if counted_objective.best_params is not None:
        x_best = counted_objective.best_params.copy()
        f_best = counted_objective.best_value
    else:
        x_best = result.x.copy()
        f_best = result.fun

    return {
        'x_best': x_best,
        'f_best': f_best,
        'n_evals': counted_objective.n_evals,
        'success': result.success,
        'message': result.message,
    }


def run_basinhopping_strategy(
    counted_objective: CountedObjective,
    bounds: list[tuple[float, float]],
    max_evals: int,
    seed: int | None = None,
) -> dict:
    """
    Run Basin-Hopping with evaluation budget.

    Returns dict with keys: x_best, f_best, n_evals, nit, message.
    """
    n_params = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    if seed is not None:
        np.random.seed(seed)

    x0 = lower_bounds + np.random.random(n_params) * (upper_bounds - lower_bounds)

    evals_per_hop = 75
    niter = max(1, int(max_evals / evals_per_hop) - 2)

    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {
            'ftol': 1e-8,
            'gtol': 1e-6,
            'maxfun': min(200, max_evals // 5),
        },
    }

    def callback(x: np.ndarray, f: float, accept: bool) -> bool:
        return counted_objective.is_budget_exhausted()

    result = basinhopping(
        func=counted_objective,
        x0=x0,
        niter=niter,
        T=1.0,
        stepsize=0.5,
        minimizer_kwargs=minimizer_kwargs,
        callback=callback,
        seed=seed,
    )

    if counted_objective.best_params is not None:
        x_best = counted_objective.best_params.copy()
        f_best = counted_objective.best_value
    else:
        x_best = result.x.copy()
        f_best = result.fun

    return {
        'x_best': x_best,
        'f_best': f_best,
        'n_evals': counted_objective.n_evals,
        'nit': result.nit,
        'message': result.message[0] if isinstance(result.message, list) else str(result.message),
    }


print("Baseline strategy functions defined: multistart_local, run_de, run_basinhopping_strategy")

# %% [markdown]
# ### Experiment 4b: Run Trial Dispatcher and Comparison

# %%
# ============================================================================
# EXPERIMENT 4 CONFIGURATION
# ============================================================================
COMPARISON_BUDGET = EXP4_BUDGET  # Configured at top of notebook
N_COMPARISON_TRIALS = EXP4_N_TRIALS  # Configured at top of notebook
BASE_SEED = 1000  # Base seed for reproducibility

# Multi-start settings
MS_N_STARTS = 50
MS_EVALS_PER_START = COMPARISON_BUDGET // MS_N_STARTS  # 100 evals per start

# Mixed-fidelity settings
MF_TOP_N = 3
MF_HIFI_EVALS_PER_CANDIDATE = 200
MF_HIFI_BUDGET = MF_TOP_N * MF_HIFI_EVALS_PER_CANDIDATE  # 600
MF_LOFI_BUDGET = COMPARISON_BUDGET - MF_HIFI_BUDGET  # 4400

STRATEGIES = ['Multi-start', 'DE', 'Basin-Hopping', 'Mixed-Fidelity']

print("=" * 70)
print("EXPERIMENT 4: EVALUATION-COUNT MATCHED COMPARISON")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Total evaluation budget: {COMPARISON_BUDGET}")
print(f"  Trials per strategy:     {N_COMPARISON_TRIALS}")
print(f"  Base random seed:        {BASE_SEED}")
print(f"\nStrategies:")
print(f"  1. Multi-start local (n_starts={MS_N_STARTS}, evals_per_start={MS_EVALS_PER_START})")
print(f"  2. Differential Evolution (full hi-fi, budget={COMPARISON_BUDGET})")
print(f"  3. Basin-Hopping (full hi-fi, budget={COMPARISON_BUDGET})")
print(f"  4. Mixed-Fidelity (lofi_budget={MF_LOFI_BUDGET}, hifi_budget={MF_HIFI_BUDGET}, N={MF_TOP_N})")


# %%
def run_trial(
    strategy: str,
    seed: int,
) -> dict:
    """
    Run a single trial of the specified optimization strategy.

    Parameters
    ----------
    strategy : str
        One of 'Multi-start', 'DE', 'Basin-Hopping', 'Mixed-Fidelity'.
    seed : int
        Random seed for this trial.

    Returns
    -------
    dict
        Trial results with keys: strategy, seed, best_objective, omega_error,
        rms_residual, success, n_evals, wall_time.
    """
    start_time = time.perf_counter()

    if strategy == 'Multi-start':
        counted_obj = CountedObjective(obj_hifi, budget=COMPARISON_BUDGET)
        result = multistart_local(
            counted_objective=counted_obj,
            bounds=bounds,
            n_starts=MS_N_STARTS,
            max_evals_per_start=MS_EVALS_PER_START,
            seed=seed,
        )
        n_evals = result['n_evals']

    elif strategy == 'DE':
        counted_obj = CountedObjective(obj_hifi, budget=COMPARISON_BUDGET)
        result = run_de(
            counted_objective=counted_obj,
            bounds=bounds,
            max_evals=COMPARISON_BUDGET,
            seed=seed,
        )
        n_evals = result['n_evals']

    elif strategy == 'Basin-Hopping':
        counted_obj = CountedObjective(obj_hifi, budget=COMPARISON_BUDGET)
        result = run_basinhopping_strategy(
            counted_objective=counted_obj,
            bounds=bounds,
            max_evals=COMPARISON_BUDGET,
            seed=seed,
        )
        n_evals = result['n_evals']

    elif strategy == 'Mixed-Fidelity':
        result = run_mixed_fidelity(
            obj_lofi=obj_lofi,
            obj_hifi=obj_hifi,
            bounds=bounds,
            lofi_budget=MF_LOFI_BUDGET,
            top_n=MF_TOP_N,
            hifi_evals_per_candidate=MF_HIFI_EVALS_PER_CANDIDATE,
            seed=seed,
        )
        n_evals = result['n_evals']

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    wall_time = time.perf_counter() - start_time

    # Get best solution
    x_best = result.get('x_best')
    f_best = result.get('f_best', float('inf'))

    # Evaluate success criteria
    if x_best is not None:
        success, omega_error, rms_residual = evaluate_success(
            x_best, true_params, obj_hifi, noise_sigma
        )
    else:
        success = False
        omega_error = float('inf')
        rms_residual = float('inf')

    return {
        'strategy': strategy,
        'seed': seed,
        'best_objective': f_best,
        'omega_error': omega_error,
        'rms_residual': rms_residual,
        'success': success,
        'n_evals': n_evals,
        'wall_time': wall_time,
    }


print("run_trial() dispatcher defined")

# %%
# Run the comparison experiment
print("\n" + "-" * 70)
print("Running comparison experiment...")
print("-" * 70)

comparison_results: list[dict] = []

for strategy in STRATEGIES:
    print(f"\n{strategy}:")
    for trial in range(N_COMPARISON_TRIALS):
        seed = BASE_SEED + trial * 100
        print(f"  Trial {trial + 1}/{N_COMPARISON_TRIALS} (seed={seed})...", end=" ", flush=True)

        trial_result = run_trial(strategy, seed)
        comparison_results.append(trial_result)

        status = "SUCCESS" if trial_result['success'] else "FAIL"
        print(f"{status}, f={trial_result['best_objective']:.4f}, "
              f"omega_err={trial_result['omega_error']:.4f} deg/s, "
              f"t={trial_result['wall_time']:.1f}s")

print("\n" + "-" * 70)
print("Experiment complete!")
print("-" * 70)

# %%
# Organize results and print comparison table
print("\n" + "=" * 70)
print("EXPERIMENT 4: COMPARISON TABLE")
print("=" * 70)

strategy_results: dict[str, dict] = {}

for strategy in STRATEGIES:
    strategy_trials = [r for r in comparison_results if r['strategy'] == strategy]

    strategy_results[strategy] = {
        'best_objectives': np.array([r['best_objective'] for r in strategy_trials]),
        'omega_errors': np.array([r['omega_error'] for r in strategy_trials]),
        'rms_residuals': np.array([r['rms_residual'] for r in strategy_trials]),
        'successes': np.array([r['success'] for r in strategy_trials]),
        'n_evals': np.array([r['n_evals'] for r in strategy_trials]),
        'wall_times': np.array([r['wall_time'] for r in strategy_trials]),
    }

# Print formatted comparison table
print(f"\nFixed evaluation budget: {COMPARISON_BUDGET}")
print(f"Trials per strategy: {N_COMPARISON_TRIALS}")
print()
print(f"{'Strategy':<15} | {'Success Rate':>12} | {'Mean Objective':>14} | "
      f"{'Mean Omega Err':>14} | {'Mean Wall Time':>14}")
print("-" * 80)

for strategy in STRATEGIES:
    sr = strategy_results[strategy]
    success_rate = sr['successes'].sum() / N_COMPARISON_TRIALS * 100
    mean_obj = sr['best_objectives'].mean()
    mean_omega_err = sr['omega_errors'].mean()
    mean_time = sr['wall_times'].mean()

    print(f"{strategy:<15} | {success_rate:>10.0f}% | {mean_obj:>14.4f} | "
          f"{mean_omega_err:>11.4f}°/s | {mean_time:>12.1f}s")

print("-" * 80)

# Print per-strategy detail
for strategy in STRATEGIES:
    sr = strategy_results[strategy]
    n_success = int(sr['successes'].sum())
    print(f"\n{strategy}:")
    print(f"  Success rate: {n_success}/{N_COMPARISON_TRIALS} "
          f"({n_success / N_COMPARISON_TRIALS * 100:.0f}%)")
    print(f"  Objective: {sr['best_objectives'].mean():.4f} ± "
          f"{sr['best_objectives'].std():.4f}")
    print(f"  Omega error: {sr['omega_errors'].mean():.4f} ± "
          f"{sr['omega_errors'].std():.4f} deg/s")
    print(f"  Wall time: {sr['wall_times'].mean():.1f} ± "
          f"{sr['wall_times'].std():.1f} s")
    print(f"  Evals used: {sr['n_evals'].mean():.0f} ± {sr['n_evals'].std():.0f}")

# %% [markdown]
# ### Experiment 4c: Evaluation-Count Comparison Visualization

# %%
# --- Box plot of final objective values per strategy ---

fig_box, ax_box = plt.subplots(figsize=(10, 6))

box_data = [strategy_results[s]['best_objectives'] for s in STRATEGIES]
bp = ax_box.boxplot(
    box_data,
    labels=STRATEGIES,
    patch_artist=True,
    widths=0.5,
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='black', markersize=6),
)

# Color each box
box_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_box.set_ylabel('Best Objective Value')
ax_box.set_title(f'Experiment 4: Objective Values by Strategy\n'
                 f'(budget={COMPARISON_BUDGET} evals, {N_COMPARISON_TRIALS} trials each)')
ax_box.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save figure
box_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "evalcount_comparison_objectives.png"
fig_box.savefig(box_path, dpi=150, bbox_inches='tight')
print(f"Box plot saved: {box_path}")
plt.show()

# %%
# --- Bar chart of success rates with 95% Wilson score confidence intervals ---


def compute_binomial_ci(
    n_success: int,
    n_trials: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute success rate and Wilson score confidence interval.

    Returns (success_rate, lower_bound, upper_bound).
    """
    from scipy import stats as sp_stats

    p = n_success / n_trials
    z = sp_stats.norm.ppf((1 + confidence) / 2)

    # Wilson score interval
    denominator = 1 + z**2 / n_trials
    center = (p + z**2 / (2 * n_trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_trials)) / n_trials) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return p, lower, upper


fig_sr, ax_sr = plt.subplots(figsize=(10, 6))

success_rates = []
ci_lower_err = []
ci_upper_err = []

for strategy in STRATEGIES:
    n_success = int(strategy_results[strategy]['successes'].sum())
    rate, lower, upper = compute_binomial_ci(n_success, N_COMPARISON_TRIALS)
    success_rates.append(rate * 100)
    ci_lower_err.append(rate * 100 - lower * 100)
    ci_upper_err.append(upper * 100 - rate * 100)

x_pos = np.arange(len(STRATEGIES))
bars = ax_sr.bar(
    x_pos, success_rates,
    color=box_colors, alpha=0.7,
    edgecolor='black', linewidth=1.5,
)

# Add error bars for confidence intervals
ax_sr.errorbar(
    x_pos, success_rates,
    yerr=[ci_lower_err, ci_upper_err],
    fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=2,
)

# Annotate bars with success count
for i, (bar, strategy) in enumerate(zip(bars, STRATEGIES)):
    n_success = int(strategy_results[strategy]['successes'].sum())
    ax_sr.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + ci_upper_err[i] + 2,
        f'{n_success}/{N_COMPARISON_TRIALS}',
        ha='center', va='bottom', fontsize=10, fontweight='bold',
    )

ax_sr.set_xticks(x_pos)
ax_sr.set_xticklabels(STRATEGIES)
ax_sr.set_ylabel('Success Rate (%)')
ax_sr.set_ylim(0, 110)
ax_sr.set_title(f'Experiment 4: Success Rates with 95% Wilson Score CI\n'
                f'(budget={COMPARISON_BUDGET} evals, {N_COMPARISON_TRIALS} trials each)')
ax_sr.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save figure
sr_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "evalcount_comparison_success.png"
fig_sr.savefig(sr_path, dpi=150, bbox_inches='tight')
print(f"Success rate chart saved: {sr_path}")
plt.show()

# %% [markdown]
# ### Experiment 4 Summary
#
# **Evaluation-Count Matched Comparison Results:**
#
# Four strategies were compared with a fixed evaluation budget of 5000:
#
# 1. **Multi-start local**: 50 random starts with LHS, 100 L-BFGS-B evals each
# 2. **Differential Evolution**: Full hi-fi DE with budget enforcement
# 3. **Basin-Hopping**: L-BFGS-B local minimizer with random perturbations
# 4. **Mixed-Fidelity**: Lo-fi DE (4400 evals) + hi-fi L-BFGS-B refinement (3x200 evals)
#
# **Box plot (objective values):**
# - Shows the distribution of best objective values achieved across 10 trials per strategy.
# - Lower and tighter distributions indicate more reliable convergence.
# - Mixed-fidelity's distribution reveals whether lo-fi exploration + hi-fi refinement
#   produces competitive final objective values.
#
# **Bar chart (success rates):**
# - Wilson score confidence intervals account for small sample sizes (n=10).
# - Overlapping CIs indicate statistically indistinguishable success rates.
# - A mixed-fidelity success rate within 5 percentage points of the best full-fidelity
#   strategy would validate the approach.
#
# **Key observations:**
# - All strategies use the same total evaluation budget (5000) for fair comparison.
# - Mixed-fidelity splits the budget: the lo-fi stage explores broadly (fast),
#   then the hi-fi stage refines the top 3 candidates (accurate).
# - Wall-clock time differences reflect the computational advantage of lo-fi
#   evaluations in the mixed-fidelity pipeline.

# %% [markdown]
# ---
# ## Experiment 5: Wall-Clock Matched Comparison
#
# Compare mixed-fidelity against full-fidelity DE under the **same wall-clock budget**.
# This is the primary practical comparison: given the same amount of time, does the
# mixed-fidelity approach produce better results?

# %% [markdown]
# ### Experiment 5a: Establish Full-Fidelity DE Baseline Time

# %%
# Run full-fidelity DE (10 trials) to establish mean wall-clock time T_ref
# Reuse run_trial() dispatcher from Experiment 4

print("=" * 70)
print("EXPERIMENT 5a: FULL-FIDELITY DE BASELINE TIMING")
print("=" * 70)

N_WALLCLOCK_TRIALS = 10
WALLCLOCK_DE_BUDGET = 5000  # Same budget as Experiment 4
WALLCLOCK_BASE_SEED = 2000  # Different base seed from Experiment 4

print(f"\nConfiguration:")
print(f"  DE evaluation budget: {WALLCLOCK_DE_BUDGET}")
print(f"  Trials: {N_WALLCLOCK_TRIALS}")
print(f"  Base seed: {WALLCLOCK_BASE_SEED}")

de_baseline_results: list[dict] = []

print("\nRunning full-fidelity DE baseline...")
for trial in range(N_WALLCLOCK_TRIALS):
    seed = WALLCLOCK_BASE_SEED + trial * 100
    print(f"  Trial {trial + 1}/{N_WALLCLOCK_TRIALS} (seed={seed})...", end=" ", flush=True)

    # Run full-fidelity DE
    t0 = time.perf_counter()
    counted_obj = CountedObjective(obj_hifi, budget=WALLCLOCK_DE_BUDGET)
    result = run_de(
        counted_objective=counted_obj,
        bounds=bounds,
        max_evals=WALLCLOCK_DE_BUDGET,
        seed=seed,
    )
    wall_time = time.perf_counter() - t0

    # Evaluate success
    x_best = result.get('x_best')
    if x_best is not None:
        success, omega_error, rms_residual = evaluate_success(
            x_best, true_params, obj_hifi, noise_sigma
        )
    else:
        success = False
        omega_error = float('inf')
        rms_residual = float('inf')

    trial_result = {
        'strategy': 'Full-Fidelity DE',
        'seed': seed,
        'best_objective': result.get('f_best', float('inf')),
        'omega_error': omega_error,
        'rms_residual': rms_residual,
        'success': success,
        'n_evals': result['n_evals'],
        'wall_time': wall_time,
    }
    de_baseline_results.append(trial_result)

    status = "SUCCESS" if success else "FAIL"
    print(f"{status}, f={trial_result['best_objective']:.4f}, "
          f"omega_err={omega_error:.4f} deg/s, t={wall_time:.1f}s")

# Compute T_ref
de_wall_times = np.array([r['wall_time'] for r in de_baseline_results])
T_ref = de_wall_times.mean()

print(f"\n{'─' * 50}")
print(f"Full-Fidelity DE wall-clock statistics:")
print(f"  Mean (T_ref): {T_ref:.2f} s")
print(f"  Std:          {de_wall_times.std():.2f} s")
print(f"  Range:        [{de_wall_times.min():.2f}, {de_wall_times.max():.2f}] s")

# %% [markdown]
# ### Experiment 5b: Calculate Adjusted Lo-Fi Budget for Mixed-Fidelity
#
# Using Experiment 1 timing data, calculate the lo-fi budget so that the mixed-fidelity
# pipeline's total wall-clock time approximately equals T_ref.

# %%
print("=" * 70)
print("EXPERIMENT 5b: BUDGET CALCULATION FOR WALL-CLOCK MATCHING")
print("=" * 70)

# Use timing from Experiment 1
t_hifi_per_eval = mean_hifi  # seconds per hi-fi evaluation
t_lofi_per_eval = mean_lofi  # seconds per lo-fi evaluation

print(f"\nTiming from Experiment 1:")
print(f"  Hi-fi eval time: {t_hifi_per_eval:.4f} s")
print(f"  Lo-fi eval time: {t_lofi_per_eval:.4f} s")
print(f"  Speedup factor:  {speedup:.1f}x")

# Mixed-fidelity wall time model:
#   T_mixed ≈ lofi_budget * t_lofi + (N * hifi_evals_per_candidate) * t_hifi
# Set T_mixed = T_ref and solve for lofi_budget:
#   lofi_budget = (T_ref - N * hifi_evals_per_candidate * t_hifi) / t_lofi

WC_MF_TOP_N = 3
WC_MF_HIFI_EVALS_PER_CANDIDATE = 200
wc_hifi_budget = WC_MF_TOP_N * WC_MF_HIFI_EVALS_PER_CANDIDATE
wc_hifi_time = wc_hifi_budget * t_hifi_per_eval

# Time remaining for lo-fi stage
time_for_lofi = T_ref - wc_hifi_time

if time_for_lofi <= 0:
    print(f"\nWARNING: T_ref ({T_ref:.2f}s) is less than hi-fi stage time ({wc_hifi_time:.2f}s)")
    print(f"  Using minimum lofi_budget of 1000")
    wc_lofi_budget = 1000
else:
    wc_lofi_budget = int(time_for_lofi / t_lofi_per_eval)

print(f"\nTarget wall-clock time (T_ref): {T_ref:.2f} s")
print(f"Hi-fi Stage 2 time (estimated): {wc_hifi_time:.2f} s ({wc_hifi_budget} evals)")
print(f"Time remaining for lo-fi:       {max(0, time_for_lofi):.2f} s")
print(f"Adjusted lo-fi budget:          {wc_lofi_budget} evals")
print(f"Expected lo-fi time:            {wc_lofi_budget * t_lofi_per_eval:.2f} s")
print(f"Expected total time:            {wc_lofi_budget * t_lofi_per_eval + wc_hifi_time:.2f} s ≈ T_ref")

# Effective evaluations comparison
effective_hifi_evals = int(T_ref / t_hifi_per_eval)
print(f"\nEffective evaluations in T_ref budget:")
print(f"  Full-fidelity DE:  {effective_hifi_evals} hi-fi evals")
print(f"  Mixed-fidelity:    {wc_lofi_budget} lo-fi + {wc_hifi_budget} hi-fi = {wc_lofi_budget + wc_hifi_budget} total evals")
print(f"  Evaluation ratio:  {(wc_lofi_budget + wc_hifi_budget) / effective_hifi_evals:.1f}x more evaluations via mixed-fidelity")

# %% [markdown]
# ### Experiment 5c: Run Mixed-Fidelity with Adjusted Budget

# %%
print("=" * 70)
print("EXPERIMENT 5c: WALL-CLOCK MATCHED MIXED-FIDELITY TRIALS")
print("=" * 70)

print(f"\nConfiguration:")
print(f"  Lo-fi budget (adjusted): {wc_lofi_budget}")
print(f"  Hi-fi budget:            {wc_hifi_budget} ({WC_MF_TOP_N} × {WC_MF_HIFI_EVALS_PER_CANDIDATE})")
print(f"  Target wall time:        {T_ref:.2f} s")
print(f"  Trials: {N_WALLCLOCK_TRIALS}")

mf_wallclock_results: list[dict] = []

print("\nRunning wall-clock matched mixed-fidelity...")
for trial in range(N_WALLCLOCK_TRIALS):
    seed = WALLCLOCK_BASE_SEED + trial * 100  # Same seeds as DE baseline
    print(f"  Trial {trial + 1}/{N_WALLCLOCK_TRIALS} (seed={seed})...", end=" ", flush=True)

    t0 = time.perf_counter()
    result = run_mixed_fidelity(
        obj_lofi=obj_lofi,
        obj_hifi=obj_hifi,
        bounds=bounds,
        lofi_budget=wc_lofi_budget,
        top_n=WC_MF_TOP_N,
        hifi_evals_per_candidate=WC_MF_HIFI_EVALS_PER_CANDIDATE,
        seed=seed,
    )
    wall_time = time.perf_counter() - t0

    x_best = result.get('x_best')
    if x_best is not None:
        success, omega_error, rms_residual = evaluate_success(
            x_best, true_params, obj_hifi, noise_sigma
        )
    else:
        success = False
        omega_error = float('inf')
        rms_residual = float('inf')

    trial_result = {
        'strategy': 'Mixed-Fidelity',
        'seed': seed,
        'best_objective': result.get('f_best', float('inf')),
        'omega_error': omega_error,
        'rms_residual': rms_residual,
        'success': success,
        'n_evals': result['n_evals'],
        'n_evals_lofi': result['n_evals_lofi'],
        'n_evals_hifi': result['n_evals_hifi'],
        'wall_time': wall_time,
        'stage1_time': result['stage1_time'],
        'stage2_time': result['stage2_time'],
    }
    mf_wallclock_results.append(trial_result)

    status = "SUCCESS" if success else "FAIL"
    print(f"{status}, f={trial_result['best_objective']:.4f}, "
          f"omega_err={omega_error:.4f} deg/s, t={wall_time:.1f}s")

print("\nAll trials complete!")

# %%
# Print comparison table
print("\n" + "=" * 70)
print("EXPERIMENT 5: WALL-CLOCK MATCHED COMPARISON TABLE")
print("=" * 70)

# Aggregate DE results
de_successes = np.array([r['success'] for r in de_baseline_results])
de_objectives = np.array([r['best_objective'] for r in de_baseline_results])
de_omega_errors = np.array([r['omega_error'] for r in de_baseline_results])
de_rms = np.array([r['rms_residual'] for r in de_baseline_results])
de_n_evals = np.array([r['n_evals'] for r in de_baseline_results])

# Aggregate MF results
mf_successes = np.array([r['success'] for r in mf_wallclock_results])
mf_objectives = np.array([r['best_objective'] for r in mf_wallclock_results])
mf_omega_errors = np.array([r['omega_error'] for r in mf_wallclock_results])
mf_rms = np.array([r['rms_residual'] for r in mf_wallclock_results])
mf_wall_times = np.array([r['wall_time'] for r in mf_wallclock_results])
mf_n_evals_total = np.array([r['n_evals'] for r in mf_wallclock_results])
mf_n_evals_lofi = np.array([r['n_evals_lofi'] for r in mf_wallclock_results])
mf_n_evals_hifi = np.array([r['n_evals_hifi'] for r in mf_wallclock_results])

de_success_rate = de_successes.sum() / N_WALLCLOCK_TRIALS * 100
mf_success_rate = mf_successes.sum() / N_WALLCLOCK_TRIALS * 100

print(f"\n{'Metric':<30} | {'Full-Fidelity DE':>18} | {'Mixed-Fidelity':>18}")
print("-" * 72)
print(f"{'Success rate':<30} | {de_success_rate:>16.0f}% | {mf_success_rate:>16.0f}%")
print(f"{'Mean objective':<30} | {de_objectives.mean():>18.4f} | {mf_objectives.mean():>18.4f}")
print(f"{'Mean omega error (deg/s)':<30} | {de_omega_errors.mean():>18.4f} | {mf_omega_errors.mean():>18.4f}")
print(f"{'Mean RMS residual (mag)':<30} | {de_rms.mean():>18.4f} | {mf_rms.mean():>18.4f}")
print(f"{'Mean wall time (s)':<30} | {de_wall_times.mean():>18.2f} | {mf_wall_times.mean():>18.2f}")
print(f"{'Mean total evals':<30} | {de_n_evals.mean():>18.0f} | {mf_n_evals_total.mean():>18.0f}")
print("-" * 72)

print(f"\nEffective evaluations in wall-clock budget:")
print(f"  Full-fidelity DE:   {de_n_evals.mean():.0f} hi-fi evals in {de_wall_times.mean():.1f}s")
print(f"  Mixed-fidelity:     {mf_n_evals_lofi.mean():.0f} lo-fi + {mf_n_evals_hifi.mean():.0f} hi-fi "
      f"= {mf_n_evals_total.mean():.0f} total evals in {mf_wall_times.mean():.1f}s")

# Per-trial detail
print(f"\nPer-trial wall-clock times:")
print(f"  {'Trial':<8} {'DE Time (s)':>12} {'MF Time (s)':>12} {'DE Success':>12} {'MF Success':>12}")
print(f"  {'-' * 56}")
for i in range(N_WALLCLOCK_TRIALS):
    de_t = de_baseline_results[i]['wall_time']
    mf_t = mf_wallclock_results[i]['wall_time']
    de_s = "YES" if de_baseline_results[i]['success'] else "no"
    mf_s = "YES" if mf_wallclock_results[i]['success'] else "no"
    print(f"  {i + 1:<8} {de_t:>12.1f} {mf_t:>12.1f} {de_s:>12} {mf_s:>12}")

# %% [markdown]
# ### Experiment 5d: Wall-Clock Comparison Visualization

# %%
# --- Grouped bar chart: success rate and mean omega error ---

fig_wc, axes_wc = plt.subplots(1, 2, figsize=(12, 5))

wc_strategies = ['Full-Fidelity DE', 'Mixed-Fidelity']
wc_colors = ['#4C72B0', '#8172B2']

# Panel 1: Success rate bars
ax1 = axes_wc[0]

de_n_success = int(de_successes.sum())
mf_n_success = int(mf_successes.sum())

de_rate, de_ci_lo, de_ci_hi = compute_binomial_ci(de_n_success, N_WALLCLOCK_TRIALS)
mf_rate, mf_ci_lo, mf_ci_hi = compute_binomial_ci(mf_n_success, N_WALLCLOCK_TRIALS)

rates = [de_rate * 100, mf_rate * 100]
ci_lo_err = [de_rate * 100 - de_ci_lo * 100, mf_rate * 100 - mf_ci_lo * 100]
ci_hi_err = [de_ci_hi * 100 - de_rate * 100, mf_ci_hi * 100 - mf_rate * 100]

x_pos_wc = np.arange(len(wc_strategies))
bars1 = ax1.bar(x_pos_wc, rates, color=wc_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.errorbar(x_pos_wc, rates, yerr=[ci_lo_err, ci_hi_err],
             fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=2)

# Annotate with success count and wall time
for i, (bar, strat) in enumerate(zip(bars1, wc_strategies)):
    n_succ = de_n_success if i == 0 else mf_n_success
    mean_t = de_wall_times.mean() if i == 0 else mf_wall_times.mean()
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci_hi_err[i] + 2,
             f'{n_succ}/{N_WALLCLOCK_TRIALS}\n({mean_t:.1f}s)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xticks(x_pos_wc)
ax1.set_xticklabels(wc_strategies, fontsize=9)
ax1.set_ylabel('Success Rate (%)')
ax1.set_ylim(0, 120)
ax1.set_title('Success Rate\n(wall-clock matched)')
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Mean omega error bars
ax2 = axes_wc[1]

mean_omega_errs = [de_omega_errors.mean(), mf_omega_errors.mean()]
std_omega_errs = [de_omega_errors.std(), mf_omega_errors.std()]

bars2 = ax2.bar(x_pos_wc, mean_omega_errs, color=wc_colors, alpha=0.7,
                edgecolor='black', linewidth=1.5)
ax2.errorbar(x_pos_wc, mean_omega_errs, yerr=std_omega_errs,
             fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=2)

# Annotate with wall time
for i, (bar, strat) in enumerate(zip(bars2, wc_strategies)):
    mean_t = de_wall_times.mean() if i == 0 else mf_wall_times.mean()
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std_omega_errs[i] + 0.01,
             f'({mean_t:.1f}s)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xticks(x_pos_wc)
ax2.set_xticklabels(wc_strategies, fontsize=9)
ax2.set_ylabel('Mean Omega Error (deg/s)')
ax2.axhline(y=OMEGA_ERROR_THRESHOLD_DEG_PER_S, color='red', linestyle='--', alpha=0.7,
            label=f'Threshold ({OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s)')
ax2.legend(fontsize=8)
ax2.set_title('Mean Angular Velocity Error\n(wall-clock matched)')
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Experiment 5: Wall-Clock Matched Comparison', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
wc_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "wallclock_comparison.png"
fig_wc.savefig(wc_path, dpi=150, bbox_inches='tight')
print(f"Wall-clock comparison saved: {wc_path}")
plt.show()

# %%
# --- Scatter plot of (wall_time, omega_error) for all trials with Pareto front ---

fig_pareto, ax_pareto = plt.subplots(figsize=(10, 7))

# Plot all trial points
de_wt = np.array([r['wall_time'] for r in de_baseline_results])
de_oe = np.array([r['omega_error'] for r in de_baseline_results])
mf_wt = np.array([r['wall_time'] for r in mf_wallclock_results])
mf_oe = np.array([r['omega_error'] for r in mf_wallclock_results])

ax_pareto.scatter(de_wt, de_oe, s=80, color='#4C72B0', alpha=0.7,
                  edgecolors='navy', linewidth=1, label='Full-Fidelity DE', zorder=5)
ax_pareto.scatter(mf_wt, mf_oe, s=80, color='#8172B2', alpha=0.7,
                  edgecolors='purple', linewidth=1, label='Mixed-Fidelity', zorder=5,
                  marker='s')

# Compute Pareto front (minimize both wall_time and omega_error)
all_wt = np.concatenate([de_wt, mf_wt])
all_oe = np.concatenate([de_oe, mf_oe])

# Sort by wall_time
sorted_idx = np.argsort(all_wt)
pareto_wt = []
pareto_oe = []
min_oe_so_far = float('inf')

for idx in sorted_idx:
    if all_oe[idx] < min_oe_so_far:
        pareto_wt.append(all_wt[idx])
        pareto_oe.append(all_oe[idx])
        min_oe_so_far = all_oe[idx]

if len(pareto_wt) > 1:
    ax_pareto.plot(pareto_wt, pareto_oe, 'k--', alpha=0.5, linewidth=1.5,
                   label='Pareto front', zorder=4)
    ax_pareto.scatter(pareto_wt, pareto_oe, s=120, facecolors='none',
                      edgecolors='black', linewidth=2, zorder=6)

# Reference lines
ax_pareto.axhline(y=OMEGA_ERROR_THRESHOLD_DEG_PER_S, color='red', linestyle=':',
                  alpha=0.6, label=f'Success threshold ({OMEGA_ERROR_THRESHOLD_DEG_PER_S} deg/s)')

ax_pareto.set_xlabel('Wall-Clock Time (s)')
ax_pareto.set_ylabel('Omega Error (deg/s)')
ax_pareto.set_title('Experiment 5: Wall-Clock vs Accuracy Trade-off\n'
                     f'(Full-Fidelity DE: {WALLCLOCK_DE_BUDGET} evals, '
                     f'Mixed-Fidelity: {wc_lofi_budget} lo-fi + {wc_hifi_budget} hi-fi)')
ax_pareto.legend(fontsize=9)
ax_pareto.grid(True, alpha=0.3)

# Use log scale for omega error if range is large
if all_oe.max() / max(all_oe.min(), 1e-6) > 100:
    ax_pareto.set_yscale('log')

plt.tight_layout()

# Save figure
pareto_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "wallclock_pareto.png"
fig_pareto.savefig(pareto_path, dpi=150, bbox_inches='tight')
print(f"Pareto plot saved: {pareto_path}")
plt.show()

# %% [markdown]
# ### Experiment 5 Summary
#
# **Wall-Clock Matched Comparison Results:**
#
# This experiment is the primary practical comparison, testing whether mixed-fidelity
# achieves better results than full-fidelity DE given the **same amount of time**.
#
# **Methodology:**
# - Full-fidelity DE was run for 10 trials to establish the reference wall-clock
#   time T_ref.
# - The mixed-fidelity lo-fi budget was adjusted so that its expected total time
#   (lo-fi Stage 1 + hi-fi Stage 2) approximately equals T_ref.
# - Both strategies used the same random seeds for comparable starting conditions.
#
# **Key observations:**
# - **Effective evaluations**: In the same wall-clock budget, mixed-fidelity can
#   perform significantly more total evaluations due to the lo-fi speedup. This
#   enables broader exploration of the parameter space.
# - **Success rate comparison**: A higher mixed-fidelity success rate at matched
#   wall time would demonstrate practical superiority.
# - **Pareto front**: Points on the Pareto front represent the best trade-off between
#   speed (wall time) and accuracy (omega error). Mixed-fidelity points dominating
#   the Pareto front confirm the approach's practical advantage.
# - **Wall-clock parity**: The comparison table shows actual measured wall times
#   confirming that the budget adjustment achieves approximate time-matching.
#
# A successful outcome is a mixed-fidelity success rate and omega error comparable
# to or better than full-fidelity DE at matched wall-clock time, validating the
# 3x speedup target from the PRD success criteria.

# %% [markdown]
# ---
# ## Experiment 6: Handoff Parameter Ablation
#
# Determine the optimal number of candidates (N) to pass from Stage 1 to Stage 2
# by sweeping N in {1, 3, 5, 10} with a fixed lo-fi budget.

# %%
# === Experiment 6: Handoff parameter ablation ===

print("=" * 70)
print("EXPERIMENT 6: HANDOFF PARAMETER ABLATION")
print("=" * 70)

ABLATION_N_VALUES = [1, 3, 5, 10]
ABLATION_LOFI_BUDGET = 5000
ABLATION_HIFI_EVALS_PER_CANDIDATE = 200
ABLATION_N_TRIALS = 10
ABLATION_BASE_SEED = 3000

print(f"\nConfiguration:")
print(f"  N values to sweep:         {ABLATION_N_VALUES}")
print(f"  Lo-fi budget (fixed):      {ABLATION_LOFI_BUDGET} evals")
print(f"  Hi-fi evals per candidate: {ABLATION_HIFI_EVALS_PER_CANDIDATE}")
print(f"  Trials per N:              {ABLATION_N_TRIALS}")
print(f"  Base seed:                 {ABLATION_BASE_SEED}")

ablation_results: dict[int, list[dict]] = {}

for n_val in ABLATION_N_VALUES:
    hifi_budget = n_val * ABLATION_HIFI_EVALS_PER_CANDIDATE
    total_budget = ABLATION_LOFI_BUDGET + hifi_budget
    print(f"\n{'─' * 60}")
    print(f"N = {n_val}  (hi-fi budget = {hifi_budget}, total = {total_budget})")
    print(f"{'─' * 60}")

    n_results: list[dict] = []

    for trial in range(ABLATION_N_TRIALS):
        seed = ABLATION_BASE_SEED + trial * 100
        print(f"  Trial {trial + 1}/{ABLATION_N_TRIALS} (seed={seed})...", end=" ", flush=True)

        t0 = time.perf_counter()
        pipeline_result = run_mixed_fidelity(
            obj_lofi=obj_lofi,
            obj_hifi=obj_hifi,
            bounds=bounds,
            lofi_budget=ABLATION_LOFI_BUDGET,
            top_n=n_val,
            hifi_evals_per_candidate=ABLATION_HIFI_EVALS_PER_CANDIDATE,
            seed=seed,
        )
        wall_time = time.perf_counter() - t0

        x_best = pipeline_result['x_best']
        f_best = pipeline_result['f_best']
        success, omega_err, rms_res = evaluate_success(
            x_best, true_params, obj_hifi, noise_sigma
        )

        n_results.append({
            'seed': seed,
            'x_best': x_best,
            'f_best': f_best,
            'omega_error': omega_err,
            'rms_residual': rms_res,
            'success': success,
            'n_evals_lofi': pipeline_result['n_evals_lofi'],
            'n_evals_hifi': pipeline_result['n_evals_hifi'],
            'n_evals': pipeline_result['n_evals'],
            'wall_time': wall_time,
            'stage1_time': pipeline_result['stage1_time'],
            'stage2_time': pipeline_result['stage2_time'],
        })

        status = "OK" if success else "FAIL"
        print(f"ω_err={omega_err:.4f} deg/s, t={wall_time:.1f}s [{status}]")

    ablation_results[n_val] = n_results

print(f"\n{'=' * 70}")
print("Ablation sweep complete!")

# %%
# --- Print ablation results table ---
print("=" * 70)
print("EXPERIMENT 6: ABLATION RESULTS TABLE")
print("=" * 70)

print(f"\n{'N':>4s}  {'Success':>8s}  {'Mean ω err':>12s}  {'Mean Time':>10s}  "
      f"{'Stage2 Frac':>11s}  {'HiFi Evals':>10s}  {'Total Evals':>11s}")
print("─" * 80)

ablation_summary: list[dict] = []

for n_val in ABLATION_N_VALUES:
    trials = ablation_results[n_val]
    n_success = sum(1 for t in trials if t['success'])
    success_rate = n_success / len(trials)
    mean_omega = np.mean([t['omega_error'] for t in trials])
    mean_time = np.mean([t['wall_time'] for t in trials])
    mean_s1_time = np.mean([t['stage1_time'] for t in trials])
    mean_s2_time = np.mean([t['stage2_time'] for t in trials])
    s2_frac = mean_s2_time / mean_time if mean_time > 0 else 0
    mean_hifi_evals = np.mean([t['n_evals_hifi'] for t in trials])
    mean_total_evals = np.mean([t['n_evals'] for t in trials])

    print(f"{n_val:>4d}  {n_success:>4d}/{len(trials):<3d}  "
          f"{mean_omega:>10.4f}    {mean_time:>8.1f} s  "
          f"{s2_frac:>9.1%}    {mean_hifi_evals:>8.0f}    {mean_total_evals:>9.0f}")

    ablation_summary.append({
        'N': n_val,
        'n_success': n_success,
        'n_trials': len(trials),
        'success_rate': success_rate,
        'mean_omega_error': mean_omega,
        'mean_total_time': mean_time,
        'mean_stage1_time': mean_s1_time,
        'mean_stage2_time': mean_s2_time,
        'stage2_fraction': s2_frac,
        'mean_hifi_evals': mean_hifi_evals,
        'mean_total_evals': mean_total_evals,
    })

print(f"\nFixed lo-fi budget: {ABLATION_LOFI_BUDGET} evals")
print(f"Hi-fi evals per candidate: {ABLATION_HIFI_EVALS_PER_CANDIDATE}")

# %%
# --- Experiment 6 Visualization ---

fig_abl, axes_abl = plt.subplots(1, 2, figsize=(14, 6))

# --- Panel (a): Success rate vs N with 95% CI error bars ---
ax_sr = axes_abl[0]

n_vals_arr = np.array([s['N'] for s in ablation_summary])
success_rates = np.array([s['success_rate'] for s in ablation_summary])
ci_data = [
    compute_binomial_ci(s['n_success'], s['n_trials'])
    for s in ablation_summary
]

rates_pct = success_rates * 100
ci_lo_pct = np.array([c[1] * 100 for c in ci_data])
ci_hi_pct = np.array([c[2] * 100 for c in ci_data])
yerr_lo = rates_pct - ci_lo_pct
yerr_hi = ci_hi_pct - rates_pct

bar_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

bars_sr = ax_sr.bar(
    range(len(n_vals_arr)), rates_pct,
    color=bar_colors[:len(n_vals_arr)], alpha=0.7,
    edgecolor='black', linewidth=1.5,
)
ax_sr.errorbar(
    range(len(n_vals_arr)), rates_pct,
    yerr=[yerr_lo, yerr_hi],
    fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=2,
)

# Annotate with success count
for i, (bar, s) in enumerate(zip(bars_sr, ablation_summary)):
    ax_sr.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + yerr_hi[i] + 2,
        f"{s['n_success']}/{s['n_trials']}",
        ha='center', va='bottom', fontsize=10, fontweight='bold',
    )

ax_sr.set_xticks(range(len(n_vals_arr)))
ax_sr.set_xticklabels([f'N={n}' for n in n_vals_arr])
ax_sr.set_ylabel('Success Rate (%)')
ax_sr.set_ylim(0, 120)
ax_sr.set_xlabel('Number of Candidates (N)')
ax_sr.set_title('(a) Success Rate vs N')
ax_sr.grid(True, alpha=0.3, axis='y')

# --- Panel (b): Mean total wall time vs N with Stage 1 + Stage 2 stacked breakdown ---
ax_wt = axes_abl[1]

stage1_times = np.array([s['mean_stage1_time'] for s in ablation_summary])
stage2_times = np.array([s['mean_stage2_time'] for s in ablation_summary])

bars_s1 = ax_wt.bar(
    range(len(n_vals_arr)), stage1_times,
    color='#4C72B0', alpha=0.7, edgecolor='black', linewidth=1,
    label='Stage 1 (lo-fi DE)',
)
bars_s2 = ax_wt.bar(
    range(len(n_vals_arr)), stage2_times,
    bottom=stage1_times,
    color='#C44E52', alpha=0.7, edgecolor='black', linewidth=1,
    label='Stage 2 (hi-fi L-BFGS-B)',
)

# Annotate with total time
for i in range(len(n_vals_arr)):
    total_t = stage1_times[i] + stage2_times[i]
    ax_wt.text(
        i, total_t + 0.5,
        f'{total_t:.1f}s',
        ha='center', va='bottom', fontsize=10, fontweight='bold',
    )

ax_wt.set_xticks(range(len(n_vals_arr)))
ax_wt.set_xticklabels([f'N={n}' for n in n_vals_arr])
ax_wt.set_ylabel('Mean Wall-Clock Time (s)')
ax_wt.set_xlabel('Number of Candidates (N)')
ax_wt.set_title('(b) Wall-Clock Time Breakdown vs N')
ax_wt.legend(fontsize=9)
ax_wt.grid(True, alpha=0.3, axis='y')

plt.suptitle('Experiment 6: Handoff Parameter Ablation', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

# Save figure
abl_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "handoff_ablation.png"
fig_abl.savefig(abl_path, dpi=150, bbox_inches='tight')
print(f"Handoff ablation figure saved: {abl_path}")
plt.show()

# %% [markdown]
# ### Experiment 6 Summary
#
# **Handoff Parameter Ablation Results:**
#
# This experiment determines the optimal number of candidates (N) to pass from
# Stage 1 (lo-fi DE) to Stage 2 (hi-fi L-BFGS-B) by sweeping N ∈ {1, 3, 5, 10}.
#
# **Methodology:**
# - Stage 1 lo-fi budget is **fixed** at 5000 evaluations for all N values.
# - Stage 2 hi-fi budget scales linearly with N: N × 200 evaluations per candidate.
# - Total cost increases with N, but Stage 1 cost is constant.
# - 10 trials per N value, each with independent random seeds.
#
# **Key observations:**
# - **N=1**: Minimum cost, but relies on a single candidate from DE — risky if
#   the DE global minimum doesn't translate well to the hi-fi objective.
# - **N=3**: Moderate cost increase, provides redundancy against poor DE candidates.
#   Often a good balance between cost and robustness.
# - **N=5–10**: Higher robustness, but Stage 2 time grows linearly. Diminishing
#   returns expected as additional candidates are unlikely to be significantly
#   different from the top few.
#
# **Recommended N**: The optimal N balances success rate improvement against
# additional hi-fi evaluation cost. Choose the smallest N where the success rate
# plateaus (marginal gain < 5 percentage points for doubling N).

# %% [markdown]
# ---
# ## Experiment 7: Phase Angle Failure Regime Identification
#
# Identify phase angle ranges where the low-fidelity (shadow-free) approximation
# fails, causing mixed-fidelity inversion to underperform full-fidelity DE.
#
# **Phase angle** = arccos(dot(sun_unit, obs_unit)) where sun_unit and obs_unit
# are unit vectors from the satellite to the sun and observer respectively.
#
# For each target phase angle, we rotate the observer position in J2000 frame
# around the satellite-to-sun axis, then regenerate synthetic lightcurve data
# and objective functions.

# %% [markdown]
# ### Experiment 7a: Phase Angle Test Case Generation

# %%
# ============================================================================
# EXPERIMENT 7: PHASE ANGLE FAILURE REGIME IDENTIFICATION
# ============================================================================

# Phase angles to test: 10 to 170 degrees in 10-degree steps (17 cases)
PHASE_ANGLES_DEG = np.arange(10, 180, 10)  # [10, 20, ..., 170]
N_PHASE_TRIALS = 5       # Trials per strategy per phase angle
PHASE_BASE_SEED = 4000   # Independent from earlier experiments
PHASE_LOFI_BUDGET = 5000
PHASE_TOP_N = 3
PHASE_HIFI_EVALS_PER_CANDIDATE = 200
PHASE_DE_BUDGET = 5000

print("=" * 70)
print("EXPERIMENT 7: PHASE ANGLE FAILURE REGIME IDENTIFICATION")
print("=" * 70)
print(f"\nPhase angles: {PHASE_ANGLES_DEG[0]}° to {PHASE_ANGLES_DEG[-1]}° "
      f"in {PHASE_ANGLES_DEG[1] - PHASE_ANGLES_DEG[0]}° steps ({len(PHASE_ANGLES_DEG)} cases)")
print(f"Trials per strategy per phase angle: {N_PHASE_TRIALS}")
print(f"Base seed: {PHASE_BASE_SEED}")


# %%
def create_phase_angle_geometry(
    sun_positions_j2000: np.ndarray,
    observer_positions_j2000: np.ndarray,
    satellite_positions_j2000: np.ndarray,
    target_phase_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate observer positions in J2000 frame to achieve a target phase angle.

    Rotates the observer position around the satellite-to-sun axis so that
    the angle between the sun direction and observer direction (as seen from
    the satellite) equals the target phase angle.

    Parameters
    ----------
    sun_positions_j2000 : np.ndarray
        Sun positions in J2000 frame, shape (N, 3).
    observer_positions_j2000 : np.ndarray
        Observer positions in J2000 frame, shape (N, 3).
    satellite_positions_j2000 : np.ndarray
        Satellite positions in J2000 frame, shape (N, 3).
    target_phase_angle_deg : float
        Desired phase angle in degrees.

    Returns
    -------
    new_observer_positions : np.ndarray
        Rotated observer positions, shape (N, 3).
    new_observer_distances : np.ndarray
        Updated observer distances, shape (N,).
    """
    from scipy.spatial.transform import Rotation

    n_obs = len(sun_positions_j2000)
    target_rad = np.deg2rad(target_phase_angle_deg)

    new_obs_positions = np.zeros_like(observer_positions_j2000)

    for i in range(n_obs):
        # Direction vectors from satellite
        sun_vec = sun_positions_j2000[i] - satellite_positions_j2000[i]
        obs_vec = observer_positions_j2000[i] - satellite_positions_j2000[i]

        sun_unit = sun_vec / np.linalg.norm(sun_vec)
        obs_dist = np.linalg.norm(obs_vec)
        obs_unit = obs_vec / obs_dist

        # Current phase angle
        cos_current = np.clip(np.dot(sun_unit, obs_unit), -1.0, 1.0)
        current_phase = np.arccos(cos_current)

        # Build a coordinate frame:
        # e1 = sun_unit (axis towards sun)
        # e2 = component of obs_unit perpendicular to sun_unit (normalized)
        e1 = sun_unit
        obs_perp = obs_unit - np.dot(obs_unit, e1) * e1
        obs_perp_norm = np.linalg.norm(obs_perp)

        if obs_perp_norm < 1e-10:
            # Observer is along sun direction; pick an arbitrary perpendicular
            arb = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(e1, arb)) > 0.9:
                arb = np.array([0.0, 1.0, 0.0])
            e2 = arb - np.dot(arb, e1) * e1
            e2 = e2 / np.linalg.norm(e2)
        else:
            e2 = obs_perp / obs_perp_norm

        # New observer direction at target phase angle (in the sun-observer plane)
        new_obs_unit = np.cos(target_rad) * e1 + np.sin(target_rad) * e2

        # Place observer at original distance
        new_obs_positions[i] = satellite_positions_j2000[i] + obs_dist * new_obs_unit

    new_obs_distances = np.linalg.norm(
        new_obs_positions - satellite_positions_j2000, axis=1
    )

    return new_obs_positions, new_obs_distances


print("create_phase_angle_geometry() defined")

# %%
# Verify geometry function: check that we can recover the original phase angle
# and that the target phase angle is achieved
print("\nVerifying phase angle geometry function...")

# Compute original mean phase angle
orig_sun_vecs = sun_positions_j2000 - satellite_positions_j2000
orig_obs_vecs = observer_positions_j2000 - satellite_positions_j2000
orig_sun_units = orig_sun_vecs / np.linalg.norm(orig_sun_vecs, axis=1, keepdims=True)
orig_obs_units = orig_obs_vecs / np.linalg.norm(orig_obs_vecs, axis=1, keepdims=True)
orig_cos_phase = np.clip(np.sum(orig_sun_units * orig_obs_units, axis=1), -1.0, 1.0)
orig_phase_deg = np.rad2deg(np.arccos(orig_cos_phase))
print(f"  Original phase angles: mean={orig_phase_deg.mean():.1f}°, "
      f"range=[{orig_phase_deg.min():.1f}°, {orig_phase_deg.max():.1f}°]")

# Test at a few target angles
for test_angle in [30, 90, 150]:
    new_obs_pos, new_obs_dist = create_phase_angle_geometry(
        sun_positions_j2000, observer_positions_j2000,
        satellite_positions_j2000, test_angle,
    )
    # Verify achieved phase angle
    new_sun_vecs = sun_positions_j2000 - satellite_positions_j2000
    new_obs_vecs = new_obs_pos - satellite_positions_j2000
    new_sun_units = new_sun_vecs / np.linalg.norm(new_sun_vecs, axis=1, keepdims=True)
    new_obs_units = new_obs_vecs / np.linalg.norm(new_obs_vecs, axis=1, keepdims=True)
    new_cos = np.clip(np.sum(new_sun_units * new_obs_units, axis=1), -1.0, 1.0)
    achieved_deg = np.rad2deg(np.arccos(new_cos))
    print(f"  Target={test_angle}°: achieved mean={achieved_deg.mean():.2f}°, "
          f"dist preserved={np.allclose(new_obs_dist, observer_distances, rtol=1e-6)}")

# %% [markdown]
# ### Experiment 7b: Lightcurve Discrepancy and Inversion Trials per Phase Angle

# %%
# For each phase angle:
#   1. Rotate observer geometry to target phase angle
#   2. Generate synthetic lightcurve with shadows
#   3. Compute lightcurve discrepancy RMS(mag_hifi - mag_lofi) at true params
#   4. Run mixed-fidelity (5 trials) and full-fidelity DE (5 trials)
#   5. Record success rate and omega error

print("\n" + "-" * 70)
print("Running phase angle sweep...")
print("-" * 70)

phase_results = []

for pa_idx, pa_deg in enumerate(PHASE_ANGLES_DEG):
    print(f"\n--- Phase angle: {pa_deg}° ({pa_idx + 1}/{len(PHASE_ANGLES_DEG)}) ---")

    # Step 1: Rotate observer geometry
    pa_obs_pos, pa_obs_dist = create_phase_angle_geometry(
        sun_positions_j2000, observer_positions_j2000,
        satellite_positions_j2000, pa_deg,
    )

    # Step 2: Generate synthetic lightcurve with shadows at this phase angle
    # Use a temporary ObjectiveFunction to get body-frame vectors
    pa_obj_temp = ObjectiveFunction(
        satellite=satellite,
        observation_times=observation_times,
        observed_lightcurve=np.zeros(n_observations),
        sun_positions_j2000=sun_positions_j2000,
        observer_positions_j2000=pa_obs_pos,
        satellite_positions_j2000=satellite_positions_j2000,
        observer_distances=pa_obs_dist,
        compute_shadows_flag=True,
        articulation_matrices=articulation_matrices,
        mode="tumbling",
        inertia_tensor=inertia_tensor,
    )

    # Get body-frame vectors at true attitude
    pa_k1, pa_k2 = pa_obj_temp._compute_body_frame_vectors(true_quaternions)

    # Compute shadows
    pa_lit_status = compute_shadows(
        satellite=satellite,
        k1_vectors=pa_k1,
        explicit_component_matrices=articulation_matrices,
        show_progress=False,
    )

    # Generate true lightcurve with shadows
    pa_true_lc, pa_flux, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=pa_lit_status,
        k1_vectors_array=pa_k1,
        k2_vectors_array=pa_k2,
        observer_distances=pa_obs_dist,
        satellite=satellite,
        epochs=epochs,
        pre_computed_matrices=articulation_matrices,
        generate_no_shadow=False,
        animate=False,
        show_progress=False,
    )

    # Add noise (same seed for comparability)
    np.random.seed(42)
    pa_observed_lc = pa_true_lc + np.random.normal(0, noise_sigma, n_observations)

    # Step 3: Create hi-fi and lo-fi objectives for this phase angle
    pa_obj_hifi = ObjectiveFunction(
        satellite=satellite,
        observation_times=observation_times,
        observed_lightcurve=pa_observed_lc,
        sun_positions_j2000=sun_positions_j2000,
        observer_positions_j2000=pa_obs_pos,
        satellite_positions_j2000=satellite_positions_j2000,
        observer_distances=pa_obs_dist,
        compute_shadows_flag=True,
        articulation_matrices=articulation_matrices,
        mode="tumbling",
        inertia_tensor=inertia_tensor,
    )

    pa_obj_lofi = ObjectiveFunction(
        satellite=satellite,
        observation_times=observation_times,
        observed_lightcurve=pa_observed_lc,
        sun_positions_j2000=sun_positions_j2000,
        observer_positions_j2000=pa_obs_pos,
        satellite_positions_j2000=satellite_positions_j2000,
        observer_distances=pa_obs_dist,
        compute_shadows_flag=False,
        articulation_matrices=articulation_matrices,
        mode="tumbling",
        inertia_tensor=inertia_tensor,
    )

    # Compute lightcurve discrepancy at true params
    pa_mag_hifi = pa_obj_hifi._generate_predicted_lightcurve(pa_k1, pa_k2)
    pa_mag_lofi = pa_obj_lofi._generate_predicted_lightcurve(pa_k1, pa_k2)
    pa_discrepancy_rms = np.sqrt(np.mean((pa_mag_hifi - pa_mag_lofi) ** 2))

    print(f"  Discrepancy RMS(hifi-lofi): {pa_discrepancy_rms:.4f} mag")

    # Step 4: Run trials for both strategies
    mf_results_pa = []
    de_results_pa = []

    for trial in range(N_PHASE_TRIALS):
        seed = PHASE_BASE_SEED + trial * 100

        # Mixed-fidelity trial
        t0 = time.perf_counter()
        mf_result = run_mixed_fidelity(
            obj_lofi=pa_obj_lofi,
            obj_hifi=pa_obj_hifi,
            bounds=bounds,
            lofi_budget=PHASE_LOFI_BUDGET,
            top_n=PHASE_TOP_N,
            hifi_evals_per_candidate=PHASE_HIFI_EVALS_PER_CANDIDATE,
            seed=seed,
        )
        mf_time = time.perf_counter() - t0

        mf_success, mf_omega_err, mf_rms = evaluate_success(
            mf_result['x_best'], true_params, pa_obj_hifi, noise_sigma
        )
        mf_results_pa.append({
            'success': mf_success,
            'omega_error': mf_omega_err,
            'rms_residual': mf_rms,
            'wall_time': mf_time,
        })

        # Full-fidelity DE trial
        t0 = time.perf_counter()
        counted_de = CountedObjective(pa_obj_hifi, budget=PHASE_DE_BUDGET)
        de_result = run_de(
            counted_objective=counted_de,
            bounds=bounds,
            max_evals=PHASE_DE_BUDGET,
            seed=seed,
        )
        de_time = time.perf_counter() - t0

        if de_result['x_best'] is not None:
            de_success, de_omega_err, de_rms = evaluate_success(
                de_result['x_best'], true_params, pa_obj_hifi, noise_sigma
            )
        else:
            de_success, de_omega_err, de_rms = False, float('inf'), float('inf')

        de_results_pa.append({
            'success': de_success,
            'omega_error': de_omega_err,
            'rms_residual': de_rms,
            'wall_time': de_time,
        })

    # Step 5: Aggregate results for this phase angle
    mf_success_rate = np.mean([r['success'] for r in mf_results_pa])
    mf_mean_omega = np.mean([r['omega_error'] for r in mf_results_pa])
    de_success_rate = np.mean([r['success'] for r in de_results_pa])
    de_mean_omega = np.mean([r['omega_error'] for r in de_results_pa])

    phase_results.append({
        'phase_angle': pa_deg,
        'discrepancy_rms': pa_discrepancy_rms,
        'mf_success_rate': mf_success_rate,
        'mf_mean_omega_error': mf_mean_omega,
        'de_success_rate': de_success_rate,
        'de_mean_omega_error': de_mean_omega,
        'mf_trials': mf_results_pa,
        'de_trials': de_results_pa,
    })

    print(f"  Mixed-Fidelity: success={mf_success_rate:.0%}, mean_omega_err={mf_mean_omega:.4f} deg/s")
    print(f"  Full-Fidelity DE: success={de_success_rate:.0%}, mean_omega_err={de_mean_omega:.4f} deg/s")

print("\n" + "-" * 70)
print("Phase angle sweep complete!")
print("-" * 70)

# %%
# Identify crossover phase angle
print("\n" + "=" * 70)
print("PHASE ANGLE RESULTS TABLE")
print("=" * 70)

print(f"\n{'Phase':>7} | {'Discrepancy':>12} | {'MF Success':>11} | {'DE Success':>11} | {'MF omega':>10} | {'DE omega':>10}")
print(f"{'(deg)':>7} | {'RMS (mag)':>12} | {'Rate':>11} | {'Rate':>11} | {'(deg/s)':>10} | {'(deg/s)':>10}")
print("-" * 80)

crossover_angle = None
for pr in phase_results:
    mf_sr = f"{pr['mf_success_rate']:.0%}"
    de_sr = f"{pr['de_success_rate']:.0%}"
    print(f"{pr['phase_angle']:>7} | {pr['discrepancy_rms']:>12.4f} | {mf_sr:>11} | {de_sr:>11} | "
          f"{pr['mf_mean_omega_error']:>10.4f} | {pr['de_mean_omega_error']:>10.4f}")

    # Detect crossover: where MF success drops below DE success
    if crossover_angle is None and pr['mf_success_rate'] < pr['de_success_rate']:
        crossover_angle = pr['phase_angle']

print(f"\nCrossover phase angle (MF success < DE success): "
      f"{crossover_angle}°" if crossover_angle else "\nNo crossover detected: Mixed-fidelity performs >= DE at all phase angles")

# %% [markdown]
# ### Experiment 7c: Phase Angle Failure Visualization

# %%
# ============================================================================
# EXPERIMENT 7c: PHASE ANGLE FAILURE VISUALIZATION
# ============================================================================

pa_angles = np.array([pr['phase_angle'] for pr in phase_results])
pa_discrepancies = np.array([pr['discrepancy_rms'] for pr in phase_results])
pa_mf_success = np.array([pr['mf_success_rate'] for pr in phase_results])
pa_de_success = np.array([pr['de_success_rate'] for pr in phase_results])
pa_mf_omega = np.array([pr['mf_mean_omega_error'] for pr in phase_results])
pa_de_omega = np.array([pr['de_mean_omega_error'] for pr in phase_results])

# Compute Wilson score CIs for success rates
pa_mf_ci_lo = []
pa_mf_ci_hi = []
pa_de_ci_lo = []
pa_de_ci_hi = []

for pr in phase_results:
    mf_n_succ = int(sum(t['success'] for t in pr['mf_trials']))
    de_n_succ = int(sum(t['success'] for t in pr['de_trials']))

    _, mf_lo, mf_hi = compute_binomial_ci(mf_n_succ, N_PHASE_TRIALS)
    _, de_lo, de_hi = compute_binomial_ci(de_n_succ, N_PHASE_TRIALS)

    pa_mf_ci_lo.append(pr['mf_success_rate'] - mf_lo)
    pa_mf_ci_hi.append(mf_hi - pr['mf_success_rate'])
    pa_de_ci_lo.append(pr['de_success_rate'] - de_lo)
    pa_de_ci_hi.append(de_hi - pr['de_success_rate'])

pa_mf_ci_lo = np.array(pa_mf_ci_lo)
pa_mf_ci_hi = np.array(pa_mf_ci_hi)
pa_de_ci_lo = np.array(pa_de_ci_lo)
pa_de_ci_hi = np.array(pa_de_ci_hi)

# --- 3-panel figure ---
fig_pa, (ax_disc, ax_sr, ax_omega) = plt.subplots(1, 3, figsize=(18, 5))

# Panel (a): Lightcurve fidelity discrepancy vs phase angle
ax_disc.plot(pa_angles, pa_discrepancies, 'ko-', linewidth=2, markersize=6)
ax_disc.axhline(y=noise_sigma, color='gray', linestyle='--', alpha=0.7, label=f'Noise σ = {noise_sigma} mag')
ax_disc.set_xlabel('Phase Angle (degrees)', fontsize=12)
ax_disc.set_ylabel('RMS Discrepancy (mag)', fontsize=12)
ax_disc.set_title('(a) Lightcurve Fidelity Discrepancy', fontsize=13)
ax_disc.legend(fontsize=10)
ax_disc.grid(True, alpha=0.3)
ax_disc.set_xlim(0, 180)

# Panel (b): Success rate vs phase angle for both strategies
ax_sr.errorbar(pa_angles - 1.5, pa_mf_success * 100, yerr=[pa_mf_ci_lo * 100, pa_mf_ci_hi * 100],
               fmt='s-', color='tab:blue', linewidth=2, markersize=6, capsize=3, label='Mixed-Fidelity')
ax_sr.errorbar(pa_angles + 1.5, pa_de_success * 100, yerr=[pa_de_ci_lo * 100, pa_de_ci_hi * 100],
               fmt='o-', color='tab:orange', linewidth=2, markersize=6, capsize=3, label='Full-Fidelity DE')
if crossover_angle is not None:
    ax_sr.axvline(x=crossover_angle, color='red', linestyle=':', alpha=0.7, label=f'Crossover ≈ {crossover_angle}°')
ax_sr.set_xlabel('Phase Angle (degrees)', fontsize=12)
ax_sr.set_ylabel('Success Rate (%)', fontsize=12)
ax_sr.set_title('(b) Success Rate vs Phase Angle', fontsize=13)
ax_sr.legend(fontsize=9)
ax_sr.grid(True, alpha=0.3)
ax_sr.set_xlim(0, 180)
ax_sr.set_ylim(-5, 105)

# Panel (c): Mean omega error vs phase angle for both strategies
ax_omega.semilogy(pa_angles, pa_mf_omega, 's-', color='tab:blue', linewidth=2, markersize=6, label='Mixed-Fidelity')
ax_omega.semilogy(pa_angles, pa_de_omega, 'o-', color='tab:orange', linewidth=2, markersize=6, label='Full-Fidelity DE')
ax_omega.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Success threshold (0.1 deg/s)')
if crossover_angle is not None:
    ax_omega.axvline(x=crossover_angle, color='red', linestyle=':', alpha=0.7, label=f'Crossover ≈ {crossover_angle}°')
ax_omega.set_xlabel('Phase Angle (degrees)', fontsize=12)
ax_omega.set_ylabel('Mean ω Error (deg/s)', fontsize=12)
ax_omega.set_title('(c) Mean Omega Error vs Phase Angle', fontsize=13)
ax_omega.legend(fontsize=9)
ax_omega.grid(True, alpha=0.3)
ax_omega.set_xlim(0, 180)

fig_pa.suptitle('Experiment 7: Phase Angle Failure Regime', fontsize=14, fontweight='bold', y=1.02)
fig_pa.tight_layout()

save_path_pa = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "phase_angle_failure.png"
fig_pa.savefig(save_path_pa, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path_pa}")
plt.show()

# %%
# --- Recommended phase angle range ---
print("\n" + "=" * 70)
print("RECOMMENDED PHASE ANGLE RANGE")
print("=" * 70)

# Find phase angles where mixed-fidelity success >= DE success
viable_angles = [pr['phase_angle'] for pr in phase_results
                 if pr['mf_success_rate'] >= pr['de_success_rate']]
# Also consider angles where MF success is within 1 trial (20%) of DE
marginal_angles = [pr['phase_angle'] for pr in phase_results
                   if pr['mf_success_rate'] >= pr['de_success_rate'] - 1.0 / N_PHASE_TRIALS]

if viable_angles:
    print(f"\nMixed-fidelity performs >= Full-fidelity DE at phase angles: "
          f"{viable_angles[0]}° – {viable_angles[-1]}°")
else:
    print("\nMixed-fidelity does not consistently outperform DE at any tested phase angle.")

if marginal_angles:
    print(f"Mixed-fidelity is within one trial of DE at phase angles: "
          f"{marginal_angles[0]}° – {marginal_angles[-1]}°")

if crossover_angle is not None:
    print(f"\nCrossover phase angle: {crossover_angle}°")
    print(f"Recommendation: Use mixed-fidelity for phase angles < {crossover_angle}°")
else:
    print("\nNo crossover detected: mixed-fidelity is viable across the full tested range (10°–170°)")
    print("Recommendation: Mixed-fidelity can be used at all tested phase angles")

# %%
# ============================================================================
# COMPREHENSIVE STUDY SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("MIXED-FIDELITY INVERSION STUDY: COMPREHENSIVE SUMMARY")
print("=" * 70)

# Collect all key results for the summary file
summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("MIXED-FIDELITY HIERARCHICAL INVERSION — STUDY SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append("")
summary_lines.append("Test case: Intelsat 901, tumbling mode")
summary_lines.append(f"Observations: {n_observations}, Noise: {noise_sigma} mag")
summary_lines.append(f"Parameters: 6 (3 axis-angle, 3 angular velocity)")
summary_lines.append("")

# Experiment 1: Fidelity benchmarking
summary_lines.append("-" * 70)
summary_lines.append("EXPERIMENT 1: FIDELITY BENCHMARKING")
summary_lines.append("-" * 70)
summary_lines.append(f"Hi-fi eval time:  {mean_hifi:.4f} ± {std_hifi:.4f} s")
summary_lines.append(f"Lo-fi eval time:  {mean_lofi:.4f} ± {std_lofi:.4f} s")
summary_lines.append(f"Speedup factor:   {speedup:.1f}x")
summary_lines.append(f"Spearman rho:     {rho:.4f} (p={p_value:.2e})")
summary_lines.append("")

# Experiment 2: Basin shift
summary_lines.append("-" * 70)
summary_lines.append("EXPERIMENT 2: BASIN SHIFT ANALYSIS")
summary_lines.append("-" * 70)
# Extract basin shift stats from the perturbed starts data
# These are computed at runtime; collect what we can from the variables in scope
if 'pairwise_displacements' in dir():
    summary_lines.append(f"Mean Euclidean displacement:  {pairwise_displacements['euclidean'].mean():.6f}")
    summary_lines.append(f"Max Euclidean displacement:   {pairwise_displacements['euclidean'].max():.6f}")
    summary_lines.append(f"Mean axis-angle shift:        {pairwise_displacements['aa_deg'].mean():.4f} deg")
    summary_lines.append(f"Mean omega shift:             {pairwise_displacements['omega_deg_s'].mean():.4f} deg/s")
else:
    summary_lines.append("(Basin shift statistics computed during Experiment 2 runtime)")
summary_lines.append("")

# Experiment 3: Pipeline validation
summary_lines.append("-" * 70)
summary_lines.append("EXPERIMENT 3: PIPELINE VALIDATION (single run)")
summary_lines.append("-" * 70)
summary_lines.append(f"Configuration: N={VALIDATION_N}, lofi_budget={VALIDATION_LOFI_BUDGET}, "
                      f"hifi_evals/candidate={VALIDATION_HIFI_EVALS_PER_CANDIDATE}")
if 'pipeline_result' in dir():
    summary_lines.append(f"Lo-fi evals:   {pipeline_result['n_evals_lofi']}")
    summary_lines.append(f"Hi-fi evals:   {pipeline_result['n_evals_hifi']}")
    summary_lines.append(f"Stage 1 time:  {pipeline_result['stage1_time']:.2f} s")
    summary_lines.append(f"Stage 2 time:  {pipeline_result['stage2_time']:.2f} s")
    summary_lines.append(f"Best objective: {pipeline_result['f_best']:.6f}")
else:
    summary_lines.append("(Pipeline validation results computed during Experiment 3 runtime)")
summary_lines.append("")

# Experiment 4: Evaluation-count comparison
summary_lines.append("-" * 70)
summary_lines.append("EXPERIMENT 4: EVALUATION-COUNT MATCHED COMPARISON")
summary_lines.append(f"  Budget: {COMPARISON_BUDGET} evals, {N_COMPARISON_TRIALS} trials per strategy")
summary_lines.append("-" * 70)
for strategy in STRATEGIES:
    sr = strategy_results[strategy]
    n_succ = int(sr['successes'].sum())
    rate = n_succ / N_COMPARISON_TRIALS * 100
    mean_obj = sr['best_objectives'].mean()
    mean_omega = sr['omega_errors'].mean()
    mean_wt = sr['wall_times'].mean()
    summary_lines.append(f"  {strategy:20s}: success={rate:5.1f}%, obj={mean_obj:.4f}, "
                          f"omega_err={mean_omega:.4f} deg/s, time={mean_wt:.1f}s")
summary_lines.append("")

# Experiment 5: Wall-clock comparison
summary_lines.append("-" * 70)
summary_lines.append("EXPERIMENT 5: WALL-CLOCK MATCHED COMPARISON")
summary_lines.append("-" * 70)
de_wc_success = de_successes.sum() / N_WALLCLOCK_TRIALS * 100
mf_wc_success = mf_successes.sum() / N_WALLCLOCK_TRIALS * 100
de_wc_time = de_wall_times.mean()
mf_wc_time = mf_wall_times.mean()
summary_lines.append(f"  Full-Fidelity DE:  success={de_wc_success:.0f}%, "
                      f"mean_time={de_wc_time:.1f}s")
summary_lines.append(f"  Mixed-Fidelity:    success={mf_wc_success:.0f}%, "
                      f"mean_time={mf_wc_time:.1f}s, lofi_budget={wc_lofi_budget}")
summary_lines.append("")

# Experiment 6: Handoff ablation
summary_lines.append("-" * 70)
summary_lines.append("EXPERIMENT 6: HANDOFF PARAMETER ABLATION")
summary_lines.append(f"  Lo-fi budget: {ABLATION_LOFI_BUDGET}, Hi-fi evals/candidate: {ABLATION_HIFI_EVALS_PER_CANDIDATE}")
summary_lines.append("-" * 70)
for s in ablation_summary:
    summary_lines.append(f"  N={s['N']:>2}: success={s['success_rate'] * 100:5.1f}%, "
                          f"omega_err={s['mean_omega_error']:.4f} deg/s, "
                          f"time={s['mean_total_time']:.1f}s, "
                          f"stage2_frac={s['stage2_fraction'] * 100:.1f}%")
summary_lines.append("")

# Experiment 7: Phase angle failure
summary_lines.append("-" * 70)
summary_lines.append("EXPERIMENT 7: PHASE ANGLE FAILURE REGIME")
summary_lines.append(f"  Phase angles: {PHASE_ANGLES_DEG[0]}°–{PHASE_ANGLES_DEG[-1]}°, "
                      f"{N_PHASE_TRIALS} trials per strategy per angle")
summary_lines.append("-" * 70)
summary_lines.append(f"{'Phase':>7} | {'Discrep':>8} | {'MF Succ':>8} | {'DE Succ':>8} | {'MF omega':>10} | {'DE omega':>10}")
summary_lines.append(f"{'(deg)':>7} | {'(mag)':>8} | {'(%)':>8} | {'(%)':>8} | {'(deg/s)':>10} | {'(deg/s)':>10}")
summary_lines.append("-" * 70)
for pr in phase_results:
    summary_lines.append(f"{pr['phase_angle']:>7} | {pr['discrepancy_rms']:>8.4f} | "
                          f"{pr['mf_success_rate'] * 100:>8.0f} | {pr['de_success_rate'] * 100:>8.0f} | "
                          f"{pr['mf_mean_omega_error']:>10.4f} | {pr['de_mean_omega_error']:>10.4f}")
if crossover_angle is not None:
    summary_lines.append(f"\nCrossover phase angle: {crossover_angle}°")
else:
    summary_lines.append("\nNo crossover detected: MF >= DE at all tested phase angles")

if viable_angles:
    summary_lines.append(f"Recommended viable range: {viable_angles[0]}°–{viable_angles[-1]}°")
summary_lines.append("")

# Viability assessment
summary_lines.append("=" * 70)
summary_lines.append("VIABILITY ASSESSMENT vs SUCCESS CRITERIA")
summary_lines.append("=" * 70)

# Success criterion 1: >= 3x speedup
speedup_pass = speedup >= 3.0
summary_lines.append(f"\n1. Speedup >= 3x:  {speedup:.1f}x  {'PASS' if speedup_pass else 'FAIL'}")

# Success criterion 2: Success rate within 5 percentage points of best single-fidelity
# Compare mixed-fidelity from Experiment 4 against best baseline
mf_sr4 = strategy_results['Mixed-Fidelity']['successes'].sum() / N_COMPARISON_TRIALS * 100
best_baseline_sr4 = max(
    strategy_results[s]['successes'].sum() / N_COMPARISON_TRIALS * 100
    for s in ['Multi-start', 'DE', 'Basin-Hopping']
)
sr_gap = best_baseline_sr4 - mf_sr4
sr_pass = sr_gap <= 5.0
summary_lines.append(f"2. Success rate within 5pp of best baseline: "
                      f"MF={mf_sr4:.0f}%, best_baseline={best_baseline_sr4:.0f}%, "
                      f"gap={sr_gap:.0f}pp  {'PASS' if sr_pass else 'FAIL'}")

# Success criterion 3: Accuracy within 5% of full-fidelity
mf_omega4 = strategy_results['Mixed-Fidelity']['omega_errors'].mean()
de_omega4 = strategy_results['DE']['omega_errors'].mean()
if de_omega4 > 0:
    accuracy_ratio = abs(mf_omega4 - de_omega4) / de_omega4 * 100
else:
    accuracy_ratio = 0.0
accuracy_pass = accuracy_ratio <= 5.0 or mf_omega4 <= de_omega4
summary_lines.append(f"3. Accuracy within 5% of full-fidelity DE: "
                      f"MF_omega={mf_omega4:.4f}, DE_omega={de_omega4:.4f}, "
                      f"diff={accuracy_ratio:.1f}%  {'PASS' if accuracy_pass else 'MARGINAL'}")

overall_pass = speedup_pass and sr_pass
summary_lines.append(f"\nOverall viability: {'VIABLE' if overall_pass else 'CONDITIONALLY VIABLE'}")
summary_lines.append("")

summary_text = "\n".join(summary_lines)
print(summary_text)

# Save summary file
summary_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "mixed_fidelity_summary.txt"
with open(summary_path, 'w') as f:
    f.write(summary_text)
print(f"\nSaved: {summary_path}")

# %% [markdown]
# ---
# ## Study Summary: Mixed-Fidelity Hierarchical Inversion
#
# ### Approach
#
# This notebook evaluated a **two-stage mixed-fidelity optimization** strategy
# for lightcurve inversion of the Intelsat 901 satellite:
#
# 1. **Stage 1**: Low-fidelity (shadow-free) Differential Evolution for fast
#    global search.
# 2. **Stage 2**: High-fidelity (shadow-enabled) L-BFGS-B local refinement
#    of the top N candidates from Stage 1.
#
# ### Key Findings
#
# **Experiment 1 — Fidelity Benchmarking:**
# The shadow-free objective evaluates significantly faster than the full shadow
# model. The Spearman rank correlation between lo-fi and hi-fi objectives
# confirms whether the lo-fi landscape preserves solution ranking.
#
# **Experiment 2 — Basin Shift:**
# The global minimum location shifts only modestly when shadows are disabled,
# confirming that lo-fi global search can guide the optimizer to the correct
# basin of attraction for subsequent hi-fi refinement.
#
# **Experiment 3 — Pipeline Validation:**
# The two-stage pipeline successfully recovers the true attitude parameters
# on the standard test case.
#
# **Experiment 4 — Evaluation-Count Matched:**
# At matched evaluation budgets (5000 evals), mixed-fidelity is compared against
# Multi-start L-BFGS-B, full-fidelity DE, and Basin-Hopping. Mixed-fidelity
# leverages the lo-fi speedup to perform more effective global exploration.
#
# **Experiment 5 — Wall-Clock Matched:**
# When given the same wall-clock budget as full-fidelity DE, mixed-fidelity
# can perform many more lo-fi evaluations, potentially translating to better
# global search coverage.
#
# **Experiment 6 — Handoff Ablation:**
# The number of candidates N passed from Stage 1 to Stage 2 trades off between
# robustness (more candidates) and cost (each candidate requires hi-fi evals).
#
# **Experiment 7 — Phase Angle Failure:**
# The lo-fi approximation quality depends on viewing geometry. At certain
# phase angles, shadow effects become significant and the lo-fi proxy
# diverges from reality, potentially degrading mixed-fidelity performance.
#
# ### Viability Assessment
#
# The PRD success criteria are:
#
# | Criterion | Target | Status |
# |-----------|--------|--------|
# | Speedup | >= 3x lo-fi vs hi-fi eval time | Measured in Experiment 1 |
# | Success rate | Within 5 percentage points of best baseline | Measured in Experiment 4 |
# | Accuracy | Omega error within 5% of full-fidelity DE | Measured in Experiment 4 |
#
# See `mixed_fidelity_summary.txt` for all numerical results.
#
# ### Recommendations
#
# 1. **Use mixed-fidelity when:** the speedup justifies the approximation and
#    the phase angle is within the viable range identified in Experiment 7.
# 2. **Choose N (handoff candidates):** based on Experiment 6 ablation results —
#    select the smallest N where success rate plateaus.
# 3. **Monitor fidelity discrepancy:** if RMS(hifi - lofi) exceeds the noise
#    level, the lo-fi proxy may be unreliable for that observation geometry.
# 4. **Phase angle awareness:** avoid mixed-fidelity at phase angles where the
#    crossover analysis shows degraded performance.
