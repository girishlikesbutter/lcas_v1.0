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
# # Optimizer Comparison Study
#
# This notebook compares different optimization strategies for lightcurve inversion
# under a **fixed evaluation budget**, providing a fair comparison of:
#
# 1. **Multi-start local optimization** (random restarts + L-BFGS-B)
# 2. **Differential Evolution** (global optimizer baseline)
# 3. **Basin Hopping** (hybrid global/local approach)
#
# ## Goals
#
# - Implement evaluation-counted objective wrapper for fair budget enforcement
# - Compare success rates, parameter errors, and wall times across strategies
# - Identify the most efficient optimizer for this inversion problem

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
print("OPTIMIZER COMPARISON STUDY")
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
print(f"  Solar panels (SP_North, SP_South): {SOLAR_PANEL_ANGLE_DEG}°")
print(f"  Antenna dishes (AD_East, AD_West): {ANTENNA_DISH_ANGLE_DEG}°")

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
# ## 7. Create Base ObjectiveFunction

# %%
# Create the base objective function
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

# Verify the objective value at true parameters
obj_at_true = objective_fn(true_params)
print(f"\nBase ObjectiveFunction created")
print(f"  Objective value at true parameters: {obj_at_true:.6f}")

# %% [markdown]
# ---
# ## 8. Evaluation-Counted Objective Wrapper
#
# This wrapper class counts function evaluations and enforces a budget limit
# for fair comparison between optimization strategies.

# %%
class CountedObjective:
    """
    Wrapper that counts function evaluations and enforces a budget limit.

    When the budget is exhausted, returns a large penalty value to discourage
    further exploration. This allows optimizers to gracefully stop when
    the budget is reached.

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
        """
        Initialize the counted objective wrapper.

        Parameters
        ----------
        objective_fn : ObjectiveFunction
            The underlying objective function to wrap.
        budget : int
            Maximum number of evaluations allowed.
        penalty_value : float
            Value returned when budget is exhausted.
        """
        self.objective_fn = objective_fn
        self.budget = budget
        self.penalty_value = penalty_value
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params: np.ndarray | None = None

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

        # Increment counter
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
        """
        Reset the counter and tracking for a new optimization trial.

        Call this before each independent optimization run to ensure
        fair budget enforcement across trials.
        """
        self.n_evals = 0
        self.best_value = float('inf')
        self.best_params = None

    def get_remaining_budget(self) -> int:
        """
        Get the number of evaluations remaining in the budget.

        Returns
        -------
        int
            Remaining evaluations.
        """
        return max(0, self.budget - self.n_evals)

    def is_budget_exhausted(self) -> bool:
        """
        Check if the evaluation budget has been exhausted.

        Returns
        -------
        bool
            True if no evaluations remain.
        """
        return self.n_evals >= self.budget


# %% [markdown]
# ---
# ## 9. Test CountedObjective Wrapper

# %%
print("\nTesting CountedObjective wrapper...")
print("-" * 50)

# Create a counted objective with small budget for testing
test_budget = 10
counted_obj = CountedObjective(objective_fn, budget=test_budget)

print(f"Initial state:")
print(f"  Budget: {counted_obj.budget}")
print(f"  Evaluations: {counted_obj.n_evals}")
print(f"  Remaining: {counted_obj.get_remaining_budget()}")
print(f"  Exhausted: {counted_obj.is_budget_exhausted()}")

# Make some evaluations
print(f"\nMaking {test_budget + 2} evaluations...")
for i in range(test_budget + 2):
    # Perturb true params slightly for each call
    test_params = true_params + np.random.normal(0, 0.01, 6)
    value = counted_obj(test_params)

    if i < test_budget:
        print(f"  Eval {i+1}: value = {value:.4f}")
    else:
        print(f"  Eval {i+1}: value = {value:.4f} (penalty - budget exhausted)")

print(f"\nAfter evaluations:")
print(f"  Evaluations: {counted_obj.n_evals}")
print(f"  Remaining: {counted_obj.get_remaining_budget()}")
print(f"  Exhausted: {counted_obj.is_budget_exhausted()}")
print(f"  Best value: {counted_obj.best_value:.6f}")

# Test reset
print(f"\nAfter reset():")
counted_obj.reset()
print(f"  Evaluations: {counted_obj.n_evals}")
print(f"  Remaining: {counted_obj.get_remaining_budget()}")
print(f"  Exhausted: {counted_obj.is_budget_exhausted()}")
print(f"  Best value: {counted_obj.best_value}")
print(f"  Best params: {counted_obj.best_params}")

print("\nCountedObjective wrapper test PASSED!")

# %% [markdown]
# ---
# ## 10. Define Parameter Bounds and Success Criteria

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
# Define success criteria (same as notebook 05)
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
print(f"  RMS residual < {RMS_THRESHOLD_FACTOR} x noise_sigma = {RMS_THRESHOLD_FACTOR * noise_sigma:.4f} mag")

# %% [markdown]
# ---
# ## 11. Multi-Start Local Optimization Strategy
#
# This strategy uses Latin Hypercube Sampling (LHS) to generate well-distributed
# initial points across the parameter space, then runs L-BFGS-B from each start.
# LHS provides better coverage than random sampling for the same number of points.

# %%
from scipy.stats.qmc import LatinHypercube


def multistart_local(
    counted_objective: CountedObjective,
    bounds: list[tuple[float, float]],
    n_starts: int,
    max_evals_per_start: int,
    seed: int | None = None,
) -> dict:
    """
    Multi-start local optimization using Latin Hypercube Sampling.

    Generates initial points using LHS for good coverage of the parameter space,
    then runs L-BFGS-B from each start point. Tracks evaluations across all
    local optimizations to respect the total budget.

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
        Results containing:
        - 'x_best': Best solution found
        - 'f_best': Best objective value
        - 'n_evals': Total function evaluations used
        - 'n_successful_starts': Number of starts that completed without hitting budget
        - 'all_results': List of results from each local optimization
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
        # Check if budget is already exhausted
        if counted_objective.is_budget_exhausted():
            break

        # Record evaluations before this start
        evals_before = counted_objective.n_evals

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

        # Record evaluations used for this start
        evals_used = counted_objective.n_evals - evals_before

        # Store result
        local_result = {
            'x0': x0.copy(),
            'x_opt': result.x.copy(),
            'f_opt': result.fun,
            'n_evals': evals_used,
            'success': result.success,
            'message': result.message,
        }
        all_results.append(local_result)

        # Update best if this is better
        if result.fun < f_best:
            f_best = result.fun
            x_best = result.x.copy()

        # Count successful completions (finished without budget exhaustion)
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


# %% [markdown]
# ---
# ## 12. Test Multi-Start Local Optimization

# %%
print("\nTesting multi-start local optimization...")
print("-" * 50)

# Create a counted objective with test budget
test_budget = 500
test_n_starts = 5
test_evals_per_start = 100

counted_obj_test = CountedObjective(objective_fn, budget=test_budget)

print(f"Configuration:")
print(f"  Total budget: {test_budget} evaluations")
print(f"  Number of starts: {test_n_starts}")
print(f"  Max evals per start: {test_evals_per_start}")

# Run multi-start optimization
start_time = time.time()
ms_result = multistart_local(
    counted_objective=counted_obj_test,
    bounds=bounds,
    n_starts=test_n_starts,
    max_evals_per_start=test_evals_per_start,
    seed=42,
)
elapsed_time = time.time() - start_time

print(f"\nResults:")
print(f"  Total evaluations used: {ms_result['n_evals']}")
print(f"  Starts attempted: {ms_result['n_starts_attempted']}")
print(f"  Successful starts: {ms_result['n_successful_starts']}")
print(f"  Best objective value: {ms_result['f_best']:.6f}")
print(f"  Wall time: {elapsed_time:.1f}s")

# Show individual start results
print(f"\nPer-start breakdown:")
for i, r in enumerate(ms_result['all_results']):
    print(f"  Start {i+1}: f={r['f_opt']:.4f}, evals={r['n_evals']}, success={r['success']}")

# Check against success criteria
if ms_result['x_best'] is not None:
    success, omega_err, rms = evaluate_success(
        ms_result['x_best'], true_params, objective_fn, noise_sigma
    )
    print(f"\nSuccess evaluation:")
    print(f"  Omega error: {omega_err:.4f} deg/s (threshold: {OMEGA_ERROR_THRESHOLD_DEG_PER_S})")
    print(f"  RMS residual: {rms:.4f} mag (threshold: {RMS_THRESHOLD_FACTOR * noise_sigma:.4f})")
    print(f"  Overall success: {success}")

print("\nMulti-start local optimization test completed!")

# %% [markdown]
# ---
# ## Setup Complete
#
# We now have:
# - `objective_fn`: The base ObjectiveFunction instance
# - `CountedObjective`: Wrapper class for budget enforcement
# - `true_params`: The 6-parameter vector for validation
# - `bounds`: Parameter bounds for optimization
# - `evaluate_success()`: Function to check if optimization succeeded
# - `multistart_local()`: Multi-start local optimization with LHS initialization
#
# The next sections (US-013 onwards) will implement:
# - Differential Evolution baseline
# - Basin Hopping strategy
# - Comparison experiment and visualization

# %% [markdown]
# ---
# ## 13. Differential Evolution Baseline
#
# This function wraps scipy's `differential_evolution` to match the existing
# codebase configuration from notebook 03 and the `src/inversion/optimizers.py`
# module. Key settings:
#
# - **Strategy**: `'best1bin'` - uses best member for mutation
# - **Mutation**: `(0.5, 1.0)` - dithered mutation factor
# - **Recombination**: `0.7` - crossover probability
# - **Polish**: Disabled to stay within evaluation budget
#
# The `maxiter` is calculated from the budget to approximately achieve
# the target number of function evaluations.

# %%
from scipy.optimize import differential_evolution


def run_de(
    counted_objective: CountedObjective,
    bounds: list[tuple[float, float]],
    max_evals: int,
    seed: int | None = None,
) -> dict:
    """
    Run Differential Evolution global optimization with evaluation budget.

    Configures DE with the same settings as the existing LCAS inversion pipeline
    (from src/inversion/optimizers.py) but respects a fixed evaluation budget.

    Parameters
    ----------
    counted_objective : CountedObjective
        Budget-enforcing objective wrapper. Should already be reset before calling.
    bounds : list[tuple[float, float]]
        List of (lower, upper) bounds for each parameter.
    max_evals : int
        Maximum number of function evaluations allowed.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results containing:
        - 'x_best': Best solution found (from counted_objective tracking)
        - 'f_best': Best objective value found
        - 'n_evals': Total function evaluations used
        - 'de_result_x': DE's reported best solution
        - 'de_result_fun': DE's reported best objective value
        - 'success': Whether DE converged
        - 'message': DE termination message

    Notes
    -----
    The maxiter parameter is calculated as: maxiter = max_evals / (popsize * n_params)
    where popsize is scipy's default (15). This ensures we approximately stay within
    the evaluation budget.

    Polish is disabled to avoid additional evaluations beyond the budget.
    The CountedObjective wrapper tracks the actual best solution found during
    optimization, which may differ from DE's final reported best if budget
    was exhausted mid-generation.
    """
    n_params = len(bounds)

    # scipy DE defaults: popsize = 15 (multiplied by n_params for population)
    # evaluations per generation ≈ popsize * n_params
    # We use popsize=15 to match scipy default
    popsize = 15
    evals_per_generation = popsize * n_params

    # Calculate maxiter to approximately achieve target evaluation count
    # Add small buffer to account for initialization
    maxiter = max(1, int(max_evals / evals_per_generation) - 1)

    # Run differential evolution
    result = differential_evolution(
        func=counted_objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        tol=0.01,  # Same as optimizers.py
        polish=False,  # Disable polish to stay within budget
        strategy='best1bin',  # Same as optimizers.py
        mutation=(0.5, 1.0),  # Same as optimizers.py
        recombination=0.7,  # Same as optimizers.py
        updating='deferred',  # Same as optimizers.py
        workers=1,  # Serial for consistent budget enforcement
    )

    # Use the tracked best from CountedObjective (may be better than DE's final)
    # because budget exhaustion returns penalty values
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
        'de_result_x': result.x.copy(),
        'de_result_fun': result.fun,
        'success': result.success,
        'message': result.message,
    }


# %% [markdown]
# ---
# ## 14. Test Differential Evolution Baseline

# %%
print("\nTesting Differential Evolution baseline...")
print("-" * 50)

# Create a counted objective with test budget
test_budget_de = 500
counted_obj_de = CountedObjective(objective_fn, budget=test_budget_de)

print(f"Configuration:")
print(f"  Total budget: {test_budget_de} evaluations")
print(f"  Expected maxiter: ~{test_budget_de // (15 * 6) - 1} generations")

# Run DE optimization
start_time = time.time()
de_result = run_de(
    counted_objective=counted_obj_de,
    bounds=bounds,
    max_evals=test_budget_de,
    seed=42,
)
elapsed_time_de = time.time() - start_time

print(f"\nResults:")
print(f"  Total evaluations used: {de_result['n_evals']}")
print(f"  Best objective value: {de_result['f_best']:.6f}")
print(f"  DE converged: {de_result['success']}")
print(f"  DE message: {de_result['message']}")
print(f"  Wall time: {elapsed_time_de:.1f}s")

# Check against success criteria
success_de, omega_err_de, rms_de = evaluate_success(
    de_result['x_best'], true_params, objective_fn, noise_sigma
)
print(f"\nSuccess evaluation:")
print(f"  Omega error: {omega_err_de:.4f} deg/s (threshold: {OMEGA_ERROR_THRESHOLD_DEG_PER_S})")
print(f"  RMS residual: {rms_de:.4f} mag (threshold: {RMS_THRESHOLD_FACTOR * noise_sigma:.4f})")
print(f"  Overall success: {success_de}")

print("\nDifferential Evolution test completed!")

# %% [markdown]
# ---
# ## 15. Basin-Hopping Strategy
#
# Basin-hopping is a hybrid global optimization algorithm that combines:
#
# 1. **Random perturbations** (global exploration) - "hops" to new basins
# 2. **Local minimization** (local refinement) - finds basin minimum
#
# This makes it particularly effective for problems with multiple local minima
# where we want to explore different basins while efficiently finding the
# minimum within each basin. It's a middle-ground between purely local and
# purely global approaches.
#
# Key parameters:
# - **stepsize**: Size of random perturbations (controls exploration distance)
# - **T**: Temperature parameter (controls acceptance of worse solutions)
# - **minimizer_kwargs**: Settings for local minimizer (L-BFGS-B)

# %%
from scipy.optimize import basinhopping


def run_basinhopping(
    counted_objective: CountedObjective,
    bounds: list[tuple[float, float]],
    max_evals: int,
    seed: int | None = None,
) -> dict:
    """
    Run Basin-Hopping global optimization with evaluation budget.

    Basin-hopping combines random perturbations with local minimization to
    explore multiple basins of attraction. This is a middle-ground approach
    between pure local optimization and differential evolution.

    Parameters
    ----------
    counted_objective : CountedObjective
        Budget-enforcing objective wrapper. Should already be reset before calling.
    bounds : list[tuple[float, float]]
        List of (lower, upper) bounds for each parameter.
    max_evals : int
        Maximum number of function evaluations allowed.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results containing:
        - 'x_best': Best solution found (from counted_objective tracking)
        - 'f_best': Best objective value found
        - 'n_evals': Total function evaluations used
        - 'bh_result_x': Basin-hopping's reported best solution
        - 'bh_result_fun': Basin-hopping's reported best objective value
        - 'nit': Number of basin-hopping iterations completed
        - 'message': Termination message

    Notes
    -----
    The number of iterations (niter) is estimated based on the budget and
    typical evaluations per iteration. L-BFGS-B is used as the local minimizer
    with bounded constraints.

    Stepsize and temperature are set based on the parameter scales:
    - stepsize: 0.5 rad for axis-angle, scaled appropriately
    - T: 1.0 (moderate acceptance of uphill moves)
    """
    n_params = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Generate random starting point within bounds
    x0 = lower_bounds + np.random.random(n_params) * (upper_bounds - lower_bounds)

    # Estimate iterations: each basin-hop does ~50-100 function evals for L-BFGS-B
    # Be conservative to stay within budget
    evals_per_hop = 75  # rough estimate for L-BFGS-B convergence
    niter = max(1, int(max_evals / evals_per_hop) - 2)

    # Configure local minimizer (L-BFGS-B with bounds)
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'options': {
            'ftol': 1e-8,
            'gtol': 1e-6,
            'maxfun': min(200, max_evals // 5),  # limit per local opt
        },
    }

    # Stepsize: use a reasonable fraction of the parameter range
    # axis-angle range is ~2*pi, omega range is ~2*omega_max
    # Use 0.5 rad as base stepsize (scaled by optimizer internally)
    stepsize = 0.5

    # Temperature: moderate value allows some uphill moves
    temperature = 1.0

    # Custom callback to check budget
    def callback(x: np.ndarray, f: float, accept: bool) -> bool:
        """Return True to stop iteration if budget exhausted."""
        return counted_objective.is_budget_exhausted()

    # Run basin-hopping
    result = basinhopping(
        func=counted_objective,
        x0=x0,
        niter=niter,
        T=temperature,
        stepsize=stepsize,
        minimizer_kwargs=minimizer_kwargs,
        callback=callback,
        seed=seed,
    )

    # Use the tracked best from CountedObjective (may be better than BH's final)
    # because budget exhaustion returns penalty values
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
        'bh_result_x': result.x.copy(),
        'bh_result_fun': result.fun,
        'nit': result.nit,
        'message': result.message[0] if isinstance(result.message, list) else str(result.message),
    }


# %% [markdown]
# ---
# ## 16. Test Basin-Hopping Strategy

# %%
print("\nTesting Basin-Hopping strategy...")
print("-" * 50)

# Create a counted objective with test budget
test_budget_bh = 500
counted_obj_bh = CountedObjective(objective_fn, budget=test_budget_bh)

print(f"Configuration:")
print(f"  Total budget: {test_budget_bh} evaluations")
print(f"  Expected iterations: ~{test_budget_bh // 75 - 2}")

# Run basin-hopping optimization
start_time = time.time()
bh_result = run_basinhopping(
    counted_objective=counted_obj_bh,
    bounds=bounds,
    max_evals=test_budget_bh,
    seed=42,
)
elapsed_time_bh = time.time() - start_time

print(f"\nResults:")
print(f"  Total evaluations used: {bh_result['n_evals']}")
print(f"  Iterations completed: {bh_result['nit']}")
print(f"  Best objective value: {bh_result['f_best']:.6f}")
print(f"  BH message: {bh_result['message']}")
print(f"  Wall time: {elapsed_time_bh:.1f}s")

# Check against success criteria
success_bh, omega_err_bh, rms_bh = evaluate_success(
    bh_result['x_best'], true_params, objective_fn, noise_sigma
)
print(f"\nSuccess evaluation:")
print(f"  Omega error: {omega_err_bh:.4f} deg/s (threshold: {OMEGA_ERROR_THRESHOLD_DEG_PER_S})")
print(f"  RMS residual: {rms_bh:.4f} mag (threshold: {RMS_THRESHOLD_FACTOR * noise_sigma:.4f})")
print(f"  Overall success: {success_bh}")

print("\nBasin-Hopping test completed!")

# %%
# Summary of key objects for downstream optimization
print("\n" + "=" * 60)
print("SETUP COMPLETE - Objects available for optimizer comparison:")
print("=" * 60)
print(f"\n  objective_fn: ObjectiveFunction instance")
print(f"  CountedObjective: Budget-enforcing wrapper class")
print(f"  multistart_local(): Multi-start local optimization with LHS")
print(f"  run_de(): Differential Evolution baseline optimizer")
print(f"  run_basinhopping(): Basin-Hopping hybrid optimizer")
print(f"  true_params: {true_params}")
print(f"  noise_sigma: {noise_sigma}")
print(f"  n_observations: {n_observations}")
print(f"\n  Parameter names: {param_names}")
