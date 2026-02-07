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

# ======================================================================
# SKIPPING EXP 1 AND 2 - GOING DIRECTLY TO EXP 3
# ======================================================================


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
        updating='deferred',
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
