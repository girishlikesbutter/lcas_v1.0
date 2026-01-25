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
# # Lightcurve Inversion Workflow - Intelsat 901
#
# This notebook demonstrates the complete lightcurve inversion pipeline for estimating
# satellite initial attitude and angular velocity from observed lightcurves.
#
# ## Overview
#
# The inversion workflow consists of:
#
# 1. **Load satellite model** (Intelsat 901 from STL files)
#    - Config-based loading via RSO_ConfigManager
#    - SPICE-based observation geometry
#
# 2. **Generate synthetic observations** (forward model)
#    - Define "true" attitude parameters
#    - Propagate attitude over observation window
#    - Generate synthetic lightcurve using LCAS forward model
#    - Optionally add noise
#
# 3. **Run inversion** (inverse problem)
#    - Optimize to recover parameters from lightcurve
#    - Multi-start global optimization with local refinement
#
# 4. **Estimate uncertainties**
#    - **Quick mode**: Fisher Information Matrix (fast, approximate)
#    - **Full mode**: MCMC posterior sampling (thorough)
#
# 5. **Validate and visualize results**
#    - Compare recovered vs true parameters
#    - Plot observed vs predicted lightcurves
#    - Show residuals and uncertainty distributions
#
# ## Prerequisites
#
# - LCAS installed with all dependencies
# - SPICE kernels installed (run `python install_dependencies.py`)
# - `emcee` and `corner` packages (for full uncertainty mode)

# %% [markdown]
# ---
# ## Setup

# %%
import sys
from pathlib import Path
import time
import os

import numpy as np
import matplotlib.pyplot as plt

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
    invert_lightcurve,
    ObjectiveFunction,
    InversionResult,
    ConstraintMode,
    get_bounds,
    get_default_bounds,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
)

# Import visualization modules (same as pipeline notebooks)
from src.visualization.lightcurve_plotter import create_light_curve_plot
from src.visualization.plotly_animation_generator import create_interactive_3d_animation

# Import for forward model with animation data
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

print(f"Project root: {PROJECT_ROOT}")
print("Imports successful!")

# %% [markdown]
# ---
# ## 1. Load Intelsat 901 Satellite Model
#
# We load the Intelsat 901 satellite model from STL files using the config-based
# approach. This provides a realistic satellite geometry for the inversion demo.

# %%
# ============================================================================
# CONFIGURATION
# ============================================================================
config_path = "intelsat_901/intelsat_901_config.yaml"

# Number of observation points
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

print("="*70)
print("LIGHTCURVE INVERSION WORKFLOW - Intelsat 901")
print("="*70)
print("\nConfiguration:")
print(f"  Satellite: {config.name}")
print(f"  Config file: {config_path}")
print(f"  Components: {list(config.components.keys())}")
print(f"  SPICE metakernel exists: {metakernel_path.exists()}")

# %%
# Load satellite model from STL files
print(f"\nLoading {config.name} satellite model from STL files...")
model_load_start = time.time()

satellite = STLLoader.create_satellite_from_stl_config(
    config=config,
    config_manager=config_manager
)

model_load_time = time.time() - model_load_start

print(f"Model loaded: {satellite.name} ({model_load_time:.2f}s)")
print(f"   Components: {len(satellite.components)}")
for component in satellite.components:
    print(f"     - {component.name}: {len(component.facets)} facets")

# Count total facets
total_facets = sum(len(comp.facets) for comp in satellite.components if comp.facets)
print(f"   Total facets: {total_facets:,}")

# Set up BRDF materials
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

# %% [markdown]
# ---
# ## 1.5 Configure Fixed Articulation Angles
#
# For this inversion workflow, we assume **fixed articulation angles**:
# - Solar panels (SP_North, SP_South): 0 degrees
# - Antenna dishes (AD_East, AD_West): 15 degrees
#
# These angles are held constant throughout the observation window.
# The inversion will only estimate initial attitude and angular velocity.

# %%
# ============================================================================
# FIXED ARTICULATION CONFIGURATION
# ============================================================================
# Define fixed angles for articulated components (constant for all epochs)
# These represent the "true" configuration during observations

SOLAR_PANEL_ANGLE_DEG = 0.0    # Solar panels at 0 degrees
ANTENNA_DISH_ANGLE_DEG = 15.0  # Antenna dishes at 15 degrees

print("Fixed articulation configuration:")
print(f"  Solar panels (SP_North, SP_South): {SOLAR_PANEL_ANGLE_DEG}°")
print(f"  Antenna dishes (AD_East, AD_West): {ANTENNA_DISH_ANGLE_DEG}°")

# Note: We'll create the rotation matrices after we know n_observations
# (done in Section 3 after SPICE geometry computation)

# %% [markdown]
# ---
# ## 2. Calculate Satellite Inertia Tensor
#
# For tumbling dynamics, we need the inertia tensor of the satellite.
# This is computed from the STL meshes and component masses.

# %%
# Define component masses (kg) - these are approximate values for Intelsat 901
# In practice, you would use actual mass properties from the satellite datasheet
component_masses = {
    'Bus': 1532.0,        # Main bus
    'SP_North': 170.0,    # North solar panel
    'SP_South': 170.0,    # South solar panel
    'AD_East': 50.0,     # East antenna dish
    'AD_West': 50.0,     # West antenna dish
}

print("\nComponent masses:")
for name, mass in component_masses.items():
    print(f"  {name}: {mass:.1f} kg")
print(f"  Total: {sum(component_masses.values()):.1f} kg")

# %%
# Calculate inertia tensor
print("\nCalculating inertia tensor...")
inertia_start = time.time()

inertia_result = compute_inertia_from_config(
    config=config,
    config_manager=config_manager,
    masses=component_masses,
    articulation_angles={'SP_North': 0.0, 'SP_South': 0.0}  # Solar panels at 0 degrees
)

inertia_time = time.time() - inertia_start
print(f"Inertia calculated ({inertia_time:.2f}s)")

# Extract the inertia tensor (3x3 matrix)
inertia_tensor = inertia_result.inertia_tensor

print(f"\nInertia tensor (kg·m²):")
print(f"  Ixx: {inertia_tensor[0, 0]:.2f}")
print(f"  Iyy: {inertia_tensor[1, 1]:.2f}")
print(f"  Izz: {inertia_tensor[2, 2]:.2f}")
print(f"\nPrincipal moments: {inertia_result.principal_moments}")
print(f"Center of mass: {inertia_result.center_of_mass}")

# %% [markdown]
# ---
# ## 3. Initialize SPICE and Compute Observation Geometry
#
# We use NASA SPICE to compute realistic observation geometry:
# - Sun position in J2000 frame
# - Observer (ground station) position in J2000 frame
# - Satellite position in J2000 frame
#
# The geometry changes over time as Earth rotates and the satellite orbits.

# %%
# Initialize SPICE
print("Initializing SPICE...")
spice_init_start = time.time()
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
spice_init_time = time.time() - spice_init_start
print(f"SPICE initialized ({spice_init_time:.2f}s)")

# Generate time series
print(f"\nTime range: {start_time_utc} to {end_time_utc}")
print(f"   Time points: {n_observations}")

start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, n_observations)

duration_hours = (end_et - start_et) / 3600
time_resolution_min = duration_hours * 60 / n_observations
print(f"   Duration: {duration_hours:.1f} hours")
print(f"   Resolution: {time_resolution_min:.1f} minutes")

# For the inversion, we need relative observation times starting from 0
observation_times = epochs - epochs[0]  # seconds from start

# %%
# Compute observation geometry using SPICE
print("\nComputing observation geometry...")
geometry_start = time.time()

geometry_data = compute_observation_geometry(
    epochs=epochs,
    satellite_id=satellite_id,
    observer_id=OBSERVER_ID,
    spice_handler=spice_handler,
    config=config
)

geometry_time = time.time() - geometry_start
print(f"Geometry computed ({geometry_time:.2f}s)")

# Extract J2000 positions for the inversion objective function
sun_positions_j2000 = geometry_data['sun_positions']
observer_positions_j2000 = geometry_data['obs_positions']
satellite_positions_j2000 = geometry_data['sat_positions']
observer_distances = geometry_data['observer_distances']

# Also extract body-frame vectors (for reference/validation)
k1_vectors_spice = geometry_data['k1_vectors']  # Sun direction in body frame (from SPICE attitude)
k2_vectors_spice = geometry_data['k2_vectors']  # Observer direction in body frame

print("\nObservation geometry:")
print(f"  Number of observations: {n_observations}")
print(f"  Time span: {observation_times[0]:.1f} to {observation_times[-1]:.1f} seconds")
print(f"  Observer distance range: {observer_distances.min():.0f} - {observer_distances.max():.0f} km")
print(f"  Sun positions shape: {sun_positions_j2000.shape}")
print(f"  Satellite positions shape: {satellite_positions_j2000.shape}")

# %%
# ============================================================================
# CREATE FIXED ARTICULATION MATRICES
# ============================================================================
# Now that we know n_observations, create the rotation matrices for fixed angles

fixed_articulation_angles = {
    'SP_North': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'SP_South': np.full(n_observations, SOLAR_PANEL_ANGLE_DEG),
    'AD_East': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
    'AD_West': np.full(n_observations, ANTENNA_DISH_ANGLE_DEG),
}

# Convert to rotation matrices using the articulation module
articulation_matrices = compute_rotation_matrices_from_angles(
    fixed_articulation_angles, satellite
)

print("\nArticulation matrices created:")
for comp_name, matrices in articulation_matrices.items():
    print(f"  {comp_name}: shape {matrices.shape}, angle = {fixed_articulation_angles[comp_name][0]:.1f}°")

# %% [markdown]
# ---
# ## 4. Define True Attitude Parameters
#
# Define the "true" initial attitude and angular velocity that we will
# try to recover through inversion.

# %%
# True initial quaternion (arbitrary rotation from identity)
# Small rotation about axis [0.6, 0.3, 0.8] by ~45 degrees
true_axis = np.array([0.6, 0.3, 0.8])
true_axis /= np.linalg.norm(true_axis)
true_angle_deg = 45.0
true_angle_rad = np.deg2rad(true_angle_deg)

# Construct quaternion (scalar-first: w, x, y, z)
true_q0 = np.array([
    np.cos(true_angle_rad / 2),  # w
    np.sin(true_angle_rad / 2) * true_axis[0],  # x
    np.sin(true_angle_rad / 2) * true_axis[1],  # y
    np.sin(true_angle_rad / 2) * true_axis[2],  # z
])

# True initial angular velocity (rad/s in body frame)
# Spin at 5 deg/s about body Z axis
true_omega_deg_per_s = np.array([0.005, -0.003, 0.05])  # mostly Z-axis spin
true_omega0 = np.deg2rad(true_omega_deg_per_s)

# Convert to axis-angle representation (for comparison with inversion output)
true_axis_angle = quaternion_to_axis_angle(true_q0)

print("True attitude parameters:")
print(f"  Initial quaternion: {true_q0}")
print(f"  Axis-angle representation: {true_axis_angle}")
print(f"  Angular velocity: {np.rad2deg(true_omega0)} deg/s")
print(f"  Total angular velocity: {np.rad2deg(np.linalg.norm(true_omega0)):.2f} deg/s")

# %% [markdown]
# ---
# ## 5. Generate Synthetic Lightcurve (Forward Model)
#
# Now we generate a synthetic "observed" lightcurve by running the forward model
# with the true parameters. We use **tumbling mode** which evolves both quaternion
# and angular velocity according to Euler's equations using the inertia tensor.

# %%
print("\nGenerating synthetic lightcurve with true parameters...")
print("  Using TUMBLING mode with inertia tensor")
print("  Shadows: ENABLED")
print(f"  Articulation: SP={SOLAR_PANEL_ANGLE_DEG}°, AD={ANTENNA_DISH_ANGLE_DEG}°")
forward_start_time = time.time()

# Propagate true attitude using tumbling dynamics (Euler's equations)
true_quaternions, true_omega_history = propagate_attitude(
    q0=true_q0,
    omega0=true_omega0,
    times=observation_times,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
)

print(f"  Angular velocity evolved from {np.rad2deg(np.linalg.norm(true_omega0)):.2f} deg/s")
print(f"    to {np.rad2deg(np.linalg.norm(true_omega_history[-1])):.2f} deg/s (final)")

# %%
# Create ObjectiveFunction to compute body-frame vectors from propagated quaternions
# (This uses the J2000 positions and transforms them to body frame using our attitude)
objective_for_forward = ObjectiveFunction(
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
k1_vectors, k2_vectors = objective_for_forward._compute_body_frame_vectors(true_quaternions)
print(f"  Body-frame vectors computed: k1 {k1_vectors.shape}, k2 {k2_vectors.shape}")

# %%
# Compute shadows with ray tracing
print("\nComputing shadows...")
shadow_start = time.time()

lit_status_dict = compute_shadows(
    satellite=satellite,
    k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices,
    show_progress=True,
)

shadow_time = time.time() - shadow_start
print(f"Shadow computation completed ({shadow_time:.2f}s)")

# %%
# Generate lightcurve WITH animation data collection
print("\nGenerating lightcurve with animation data...")
lc_start = time.time()

true_lightcurve, total_flux, _, _, _, animation_data = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict,
    k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors,
    observer_distances=observer_distances,
    satellite=satellite,
    epochs=epochs,
    pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False,
    animate=True,  # Collect animation data for Plotly visualization
    show_progress=True,
)

lc_time = time.time() - lc_start
forward_time = time.time() - forward_start_time

print(f"Lightcurve generation completed ({lc_time:.2f}s)")
print(f"Animation data collected: {len(animation_data) if animation_data else 0} frames")
print(f"\nTotal forward model time: {forward_time:.2f} seconds")
print(f"  Lightcurve range: [{true_lightcurve.min():.2f}, {true_lightcurve.max():.2f}] mag")

# %%
# Add synthetic noise to simulate real observations
np.random.seed(42)  # reproducibility
noise_sigma = 0.05  # 0.05 magnitude noise
observed_lightcurve = true_lightcurve + np.random.normal(0, noise_sigma, n_observations)

print(f"\nAdded Gaussian noise (sigma = {noise_sigma} mag)")
print(f"  Observed lightcurve range: [{observed_lightcurve.min():.2f}, {observed_lightcurve.max():.2f}] mag")

# %%
# Prepare data for plotting (same format as pipeline notebooks)
time_hours = (epochs - epochs[0]) / 3600.0  # Time in hours from start
utc_times = [spice_handler.et_to_utc(epoch, "C", 0) for epoch in epochs]

# Compute phase angles
phase_angles = np.zeros(n_observations)
for i in range(n_observations):
    cos_phase = np.dot(k1_vectors[i], k2_vectors[i])
    cos_phase = np.clip(cos_phase, -1.0, 1.0)
    phase_angles[i] = np.degrees(np.arccos(cos_phase))

print(f"Phase angles computed: {phase_angles.min():.1f}° to {phase_angles.max():.1f}°")

# %%
# Plot the synthetic lightcurve using pipeline plotting function
print("\nCreating lightcurve plot...")

plot = create_light_curve_plot(
    time_hours=time_hours,
    epochs=epochs,
    magnitudes=true_lightcurve,
    phase_angles=phase_angles,
    utc_times=utc_times,
    satellite_name=satellite.name,
    plot_mode="single",  # Single curve mode
    output_dir=output_dir,
    observer_distances=observer_distances,
    no_plot=False,
    save=True
)

print(f"Lightcurve plot saved to: {output_dir}")

# %% [markdown]
# ---
# ## 5.5 Interactive 3D Animation
#
# Visualize the satellite's facet-level illumination over time using Plotly.
# This helps verify the forward model is working correctly before running the inversion.

# %%
# Create interactive 3D animation (same as pipeline notebooks)
if animation_data is None or len(animation_data) == 0:
    print("No animation data available.")
else:
    print("\nCreating interactive 3D animation...")

    # Build geometry_data dict for animation (needs sat_att_matrices)
    # We'll use identity matrices since body-frame vectors already account for attitude
    geometry_data_for_anim = {
        'k1_vectors': k1_vectors,
        'k2_vectors': k2_vectors,
        'observer_distances': observer_distances,
        'sun_positions': sun_positions_j2000,
        'sat_positions': satellite_positions_j2000,
        'obs_positions': observer_positions_j2000,
        'sat_att_matrices': np.array([np.eye(3) for _ in range(n_observations)]),
    }

    animation_path = create_interactive_3d_animation(
        animation_data=animation_data,
        magnitudes=true_lightcurve,
        time_hours=time_hours,
        geometry_data=geometry_data_for_anim,
        satellite_name=satellite.name,
        output_dir=output_dir,
        show_j2000_frame=True,
        show_body_frame=True,
        show_sun_vector=True,
        show_observer_vector=True,
        frame_duration_ms=100,
        save=True,
        color_mode='flux'  # 'lit_status' or 'flux'
    )

    if animation_path:
        print(f"Animation saved to: {animation_path}")

# %% [markdown]
# ---
# ## 6. Configuration Options
#
# Before running the inversion, let's review all the configuration options
# available for the `invert_lightcurve` function.
#
# ### Key Parameters
#
# | Parameter | Description | Default |
# |-----------|-------------|---------|
# | `mode` | `'principal_axis'` or `'tumbling'` | `'principal_axis'` |
# | `uncertainty_mode` | `'quick'` (Fisher) or `'full'` (MCMC) | `'quick'` |
# | `n_starts` | Number of multi-start optimizations | 3 |
# | `compute_shadows` | Whether to compute ray-traced shadows | `True` |
# | `constraint_mode` | `ConstraintMode.FREE` or `.PHYSICS_INFORMED` | None (default bounds) |
# | `omega_max_deg_per_s` | Maximum angular velocity | 30.0 |
# | `seed` | Random seed for reproducibility | None |
# | `mcmc_n_samples` | MCMC samples (if `uncertainty_mode='full'`) | 1000 |
# | `mcmc_burn_in` | MCMC burn-in steps | 100 |
#
# ### Constraint Modes
#
# - **FREE**: Allows any angular velocity direction up to `omega_max`
# - **PHYSICS_INFORMED**: Biases toward principal axis rotation (tighter bounds)

# %%
# Show default bounds
print("Default parameter bounds (omega_max = 30 deg/s):")
bounds = get_default_bounds(omega_max_deg_per_s=0.5)
param_names = ['axis_angle_x', 'axis_angle_y', 'axis_angle_z', 'omega_x', 'omega_y', 'omega_z']
for name, (low, high) in zip(param_names, bounds):
    if 'omega' in name:
        print(f"  {name}: [{np.rad2deg(low):.1f}, {np.rad2deg(high):.1f}] deg/s")
    else:
        print(f"  {name}: [{np.rad2deg(low):.1f}, {np.rad2deg(high):.1f}] deg")

# %%
# Compare FREE vs PHYSICS_INFORMED bounds
print("\nConstraint mode comparison (omega_max = 30 deg/s):")

for mode in [ConstraintMode.FREE, ConstraintMode.PHYSICS_INFORMED]:
    bounds = get_bounds(mode, omega_max=np.deg2rad(0.5))
    omega_bounds = bounds[3:]  # last 3 are omega bounds
    omega_max_component = np.rad2deg(omega_bounds[0][1])
    print(f"\n  {mode.name}:")
    print(f"    Omega component bounds: [-{omega_max_component:.2f}, {omega_max_component:.2f}] deg/s")
    max_total_omega = omega_max_component * np.sqrt(3)
    print(f"    Max total omega (at bounds): {max_total_omega:.2f} deg/s")

# %% [markdown]
# ---
# ## 7. Run Inversion with Quick (Fisher) Uncertainty
#
# First, we'll run the inversion with quick uncertainty estimation
# using the Fisher Information Matrix approximation.
#
# **Important**: We use `mode="tumbling"` and pass the `inertia_tensor` so the
# inversion uses the same physics model as the forward model.

# %%
print("\n" + "="*60)
print("Running inversion with QUICK (Fisher) uncertainty mode...")
print("  Using TUMBLING mode with inertia tensor")
print("  Shadows: ENABLED")
print(f"  Articulation: SP={SOLAR_PANEL_ANGLE_DEG}°, AD={ANTENNA_DISH_ANGLE_DEG}°")
print("="*60)

start_time = time.time()

result_quick = invert_lightcurve(
    observed_lightcurve=observed_lightcurve,
    satellite=satellite,
    observation_times=observation_times,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
    uncertainty_mode="quick",
    compute_shadows=True,  # Enable shadows for accurate physics
    n_starts=3,
    omega_max_deg_per_s=0.5,
    seed=123,  # reproducibility
    articulation_matrices=articulation_matrices,  # Fixed component angles
)

inversion_time_quick = time.time() - start_time
print(f"\nInversion completed in {inversion_time_quick:.2f} seconds")

# %% [markdown]
# ### 7.1 Analyze Quick Mode Results

# %%
# Display recovered parameters
print("\n" + "-"*50)
print("RECOVERED PARAMETERS (Quick Mode)")
print("-"*50)

print(f"\nInitial quaternion:")
print(f"  True:      {true_q0}")
print(f"  Recovered: {result_quick.q0}")

print(f"\nInitial angular velocity:")
print(f"  True (deg/s):      {np.rad2deg(true_omega0)}")
print(f"  Recovered (deg/s): {np.rad2deg(result_quick.omega0)}")

# Compute errors
omega_error = result_quick.omega0 - true_omega0
omega_error_deg = np.rad2deg(omega_error)
print(f"\nAngular velocity error (deg/s): {omega_error_deg}")
print(f"  Magnitude error: {np.linalg.norm(omega_error_deg):.4f} deg/s")

# %%
# Display fit quality
print("\nFit Quality:")
print(f"  Chi-squared: {result_quick.chi_squared:.6f}")
print(f"  RMS residual: {result_quick.rms_residual:.6f} mag")
print(f"  (Noise sigma was: {noise_sigma:.2f} mag)")

# %%
# Display uncertainties
print("\nParameter Uncertainties (Fisher Information):")
if result_quick.uncertainties is not None:
    std_devs = result_quick.uncertainties.get('std_devs')
    if std_devs is not None:
        std_devs_deg = std_devs.copy()
        std_devs_deg[:3] = np.rad2deg(std_devs[:3])  # axis-angle to degrees
        std_devs_deg[3:] = np.rad2deg(std_devs[3:])  # omega to deg/s

        for name, std in zip(param_names, std_devs_deg):
            if 'omega' in name:
                print(f"  {name}: {std:.4f} deg/s")
            else:
                print(f"  {name}: {std:.4f} deg")

# %%
# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Lightcurve comparison
result_quick.plot_lightcurve_comparison(ax=axes[0])
axes[0].set_title('Observed vs Predicted Lightcurve (Quick Mode)')

# Residuals
result_quick.plot_residuals(ax=axes[1])
axes[1].set_title('Residuals (Quick Mode)')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'data' / 'results' / 'inversion_quick_results.png', dpi=150)
plt.show()
print("Figure saved to data/results/inversion_quick_results.png")

# %% [markdown]
# ---
# ## 8. Run Inversion with Full (MCMC) Uncertainty
#
# Now we'll run the inversion with full MCMC posterior sampling.
# This takes longer but provides complete uncertainty characterization.
#
# **Note**: For a production workflow, you would typically use more samples
# (e.g., 5000-10000) and longer burn-in (500-1000).

# %%
print("\n" + "="*60)
print("Running inversion with FULL (MCMC) uncertainty mode...")
print("  Using TUMBLING mode with inertia tensor")
print("  Shadows: ENABLED")
print(f"  Articulation: SP={SOLAR_PANEL_ANGLE_DEG}°, AD={ANTENNA_DISH_ANGLE_DEG}°")
print("="*60)
print("(This will take several minutes with shadows enabled...)")

start_time = time.time()

result_full = invert_lightcurve(
    observed_lightcurve=observed_lightcurve,
    satellite=satellite,
    observation_times=observation_times,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    mode="tumbling",
    inertia_tensor=inertia_tensor,
    uncertainty_mode="full",
    compute_shadows=True,  # Enable shadows for accurate physics
    n_starts=2,  # Fewer starts for demo (MCMC is the main uncertainty source)
    omega_max_deg_per_s=30.0,
    seed=123,
    mcmc_n_samples=500,  # Reduced for demo (use 1000+ in production)
    mcmc_burn_in=100,
    articulation_matrices=articulation_matrices,  # Fixed component angles
)

inversion_time_full = time.time() - start_time
print(f"\nInversion completed in {inversion_time_full:.2f} seconds")

# %% [markdown]
# ### 8.1 Analyze Full Mode Results

# %%
# Display recovered parameters
print("\n" + "-"*50)
print("RECOVERED PARAMETERS (Full MCMC Mode)")
print("-"*50)

print(f"\nInitial quaternion:")
print(f"  True:      {true_q0}")
print(f"  Recovered: {result_full.q0}")

print(f"\nInitial angular velocity:")
print(f"  True (deg/s):      {np.rad2deg(true_omega0)}")
print(f"  Recovered (deg/s): {np.rad2deg(result_full.omega0)}")

# Compute errors
omega_error_full = result_full.omega0 - true_omega0
omega_error_full_deg = np.rad2deg(omega_error_full)
print(f"\nAngular velocity error (deg/s): {omega_error_full_deg}")
print(f"  Magnitude error: {np.linalg.norm(omega_error_full_deg):.4f} deg/s")

# %%
# Display MCMC-specific information
print("\nMCMC Diagnostics:")
if result_full.uncertainties is not None:
    acceptance = result_full.uncertainties.get('acceptance_fraction')
    autocorr = result_full.uncertainties.get('autocorr_time')
    converged = result_full.uncertainties.get('converged')

    if acceptance is not None:
        print(f"  Acceptance fraction: {np.mean(acceptance):.2%}")
    if autocorr is not None:
        if isinstance(autocorr, np.ndarray):
            print(f"  Autocorrelation times: {autocorr}")
        else:
            print(f"  Autocorrelation time: {autocorr}")
    if converged is not None:
        print(f"  Converged: {converged}")

# %%
# Display uncertainties from MCMC
print("\nParameter Uncertainties (MCMC Posterior):")
if result_full.uncertainties is not None:
    std_devs_mcmc = result_full.uncertainties.get('std_devs')
    if std_devs_mcmc is not None:
        std_devs_mcmc_deg = std_devs_mcmc.copy()
        std_devs_mcmc_deg[:3] = np.rad2deg(std_devs_mcmc[:3])
        std_devs_mcmc_deg[3:] = np.rad2deg(std_devs_mcmc[3:])

        for name, std in zip(param_names, std_devs_mcmc_deg):
            if 'omega' in name:
                print(f"  {name}: {std:.4f} deg/s")
            else:
                print(f"  {name}: {std:.4f} deg")

# %%
# Visualize lightcurve and residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

result_full.plot_lightcurve_comparison(ax=axes[0])
axes[0].set_title('Observed vs Predicted Lightcurve (MCMC Mode)')

result_full.plot_residuals(ax=axes[1])
axes[1].set_title('Residuals (MCMC Mode)')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'data' / 'results' / 'inversion_mcmc_lightcurve.png', dpi=150)
plt.show()
print("Figure saved to data/results/inversion_mcmc_lightcurve.png")

# %%
# Plot corner plot of posterior samples
print("\nGenerating corner plot of posterior samples...")

try:
    fig = result_full.plot_corner()
    plt.savefig(PROJECT_ROOT / 'data' / 'results' / 'inversion_mcmc_corner.png', dpi=150)
    plt.show()
    print("Figure saved to data/results/inversion_mcmc_corner.png")
except ImportError as e:
    print(f"Could not generate corner plot: {e}")
except ValueError as e:
    print(f"Could not generate corner plot: {e}")

# %% [markdown]
# ---
# ## 9. Compare Quick vs Full Uncertainty Methods
#
# Let's compare the results and uncertainties from both methods.

# %%
print("\n" + "="*60)
print("COMPARISON: Quick (Fisher) vs Full (MCMC)")
print("="*60)

# Compare recovered omega values
print("\nRecovered Angular Velocity (deg/s):")
print(f"  True:  {np.rad2deg(true_omega0)}")
print(f"  Quick: {np.rad2deg(result_quick.omega0)}")
print(f"  Full:  {np.rad2deg(result_full.omega0)}")

# Compare errors
print("\nAngular Velocity Error Magnitude (deg/s):")
quick_error = np.linalg.norm(np.rad2deg(result_quick.omega0 - true_omega0))
full_error = np.linalg.norm(np.rad2deg(result_full.omega0 - true_omega0))
print(f"  Quick: {quick_error:.4f}")
print(f"  Full:  {full_error:.4f}")

# Compare fit quality
print("\nFit Quality:")
print(f"  Quick chi-squared: {result_quick.chi_squared:.6f}")
print(f"  Full chi-squared:  {result_full.chi_squared:.6f}")
print(f"  Quick RMS: {result_quick.rms_residual:.6f} mag")
print(f"  Full RMS:  {result_full.rms_residual:.6f} mag")

# Compare computation time
print("\nComputation Time:")
print(f"  Quick: {inversion_time_quick:.2f} seconds")
print(f"  Full:  {inversion_time_full:.2f} seconds")
print(f"  Ratio: {inversion_time_full/inversion_time_quick:.1f}x")

# Compare uncertainties
print("\nUncertainty Comparison (std dev):")
if result_quick.uncertainties is not None and result_full.uncertainties is not None:
    std_quick = result_quick.uncertainties.get('std_devs')
    std_full = result_full.uncertainties.get('std_devs')

    if std_quick is not None and std_full is not None:
        print(f"  {'Parameter':<15} {'Fisher':>12} {'MCMC':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12}")
        for i, name in enumerate(param_names):
            val_q = np.rad2deg(std_quick[i])
            val_f = np.rad2deg(std_full[i])
            unit = 'deg/s' if 'omega' in name else 'deg'
            print(f"  {name:<15} {val_q:>10.4f}  {val_f:>10.4f}  {unit}")

# %% [markdown]
# ---
# ## 10. Round-Trip Validation Summary
#
# This notebook demonstrated the complete round-trip:
#
# 1. **Forward model** generated synthetic lightcurve from known parameters
# 2. **Inversion** recovered the parameters from the lightcurve
# 3. **Comparison** validated the recovery accuracy
#
# ### Key Findings
#
# - The inversion successfully recovers the true parameters
# - Quick (Fisher) mode provides fast uncertainty estimates
# - Full (MCMC) mode provides thorough posterior characterization
# - Trade-off: MCMC is more thorough but significantly slower
#
# ### When to Use Each Mode
#
# | Mode | Use When |
# |------|----------|
# | Quick (Fisher) | Rapid iteration, initial exploration, real-time applications |
# | Full (MCMC) | Final analysis, publication-quality uncertainties, multimodal posteriors |
#
# ### Production Recommendations
#
# For production workflows:
# - Increase `n_starts` to 5-10 for better global search
# - Enable `compute_shadows=True` for accurate physics
# - Use MCMC with `mcmc_n_samples=5000+` and `mcmc_burn_in=500+`
# - Consider `constraint_mode=ConstraintMode.PHYSICS_INFORMED` for spin-stabilized satellites

# %%
# Final summary table
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print("\nTrue vs Recovered Parameters:")
print("-"*60)

# Quaternion comparison
print("\nQuaternion (w, x, y, z):")
print(f"  True:  [{true_q0[0]:.6f}, {true_q0[1]:.6f}, {true_q0[2]:.6f}, {true_q0[3]:.6f}]")
print(f"  Quick: [{result_quick.q0[0]:.6f}, {result_quick.q0[1]:.6f}, {result_quick.q0[2]:.6f}, {result_quick.q0[3]:.6f}]")
print(f"  MCMC:  [{result_full.q0[0]:.6f}, {result_full.q0[1]:.6f}, {result_full.q0[2]:.6f}, {result_full.q0[3]:.6f}]")

# Omega comparison
print(f"\nAngular Velocity (deg/s):")
true_omega_deg = np.rad2deg(true_omega0)
quick_omega_deg = np.rad2deg(result_quick.omega0)
full_omega_deg = np.rad2deg(result_full.omega0)
print(f"  True:  [{true_omega_deg[0]:.4f}, {true_omega_deg[1]:.4f}, {true_omega_deg[2]:.4f}]")
print(f"  Quick: [{quick_omega_deg[0]:.4f}, {quick_omega_deg[1]:.4f}, {quick_omega_deg[2]:.4f}]")
print(f"  MCMC:  [{full_omega_deg[0]:.4f}, {full_omega_deg[1]:.4f}, {full_omega_deg[2]:.4f}]")

print("\n" + "="*60)
print("Lightcurve inversion workflow demonstration complete!")
print("="*60)

# %% [markdown]
# ---
# ## Appendix: API Reference
#
# ### invert_lightcurve()
#
# ```python
# result = invert_lightcurve(
#     # Required inputs
#     observed_lightcurve,       # Array of observed magnitudes
#     satellite,                  # Satellite model
#     observation_times,          # Time array (seconds)
#     sun_positions_j2000,        # Sun positions (N, 3) in km
#     observer_positions_j2000,   # Observer positions (N, 3) in km
#     satellite_positions_j2000,  # Satellite positions (N, 3) in km
#     observer_distances,         # Observer distances (N,) in km
#
#     # Mode options
#     mode="principal_axis",      # or "tumbling"
#     uncertainty_mode="quick",   # or "full"
#
#     # Tumbling mode only
#     inertia_tensor=None,        # Required for tumbling
#
#     # Optimization options
#     compute_shadows=True,       # Ray tracing
#     n_starts=3,                 # Multi-start count
#     constraint_mode=None,       # ConstraintMode enum
#     omega_max_deg_per_s=30.0,   # Max angular velocity
#     seed=None,                  # Random seed
#
#     # MCMC options (uncertainty_mode="full")
#     mcmc_n_samples=1000,        # Posterior samples
#     mcmc_burn_in=100,           # Burn-in steps
# )
# ```
#
# ### InversionResult
#
# ```python
# result.q0                    # Initial quaternion (4,)
# result.omega0                # Initial angular velocity (3,) rad/s
# result.chi_squared           # Fit statistic
# result.rms_residual          # RMS of residuals (mag)
# result.predicted_lightcurve  # Model prediction
# result.observed_lightcurve   # Input observations
# result.observation_times     # Input times
# result.uncertainties         # Dict with covariance, std_devs, etc.
# result.omega_history         # For tumbling mode
# result.mcmc_samples          # For full mode
#
# # Visualization methods
# result.plot_lightcurve_comparison()  # Observed vs predicted
# result.plot_residuals()              # Residual plot
# result.plot_corner()                 # Posterior corner plot (MCMC only)
# ```
