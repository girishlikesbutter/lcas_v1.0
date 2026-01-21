# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lightcurve Inversion Workflow
#
# This notebook demonstrates the complete lightcurve inversion pipeline for estimating
# satellite initial attitude and angular velocity from observed lightcurves.
#
# ## Overview
#
# The inversion workflow consists of:
#
# 1. **Generate synthetic observations** (forward model)
#    - Define "true" attitude parameters
#    - Propagate attitude over observation window
#    - Generate synthetic lightcurve using LCAS forward model
#    - Optionally add noise
#
# 2. **Run inversion** (inverse problem)
#    - Optimize to recover parameters from lightcurve
#    - Multi-start global optimization with local refinement
#
# 3. **Estimate uncertainties**
#    - **Quick mode**: Fisher Information Matrix (fast, approximate)
#    - **Full mode**: MCMC posterior sampling (thorough)
#
# 4. **Validate and visualize results**
#    - Compare recovered vs true parameters
#    - Plot observed vs predicted lightcurves
#    - Show residuals and uncertainty distributions
#
# ## Prerequisites
#
# - LCAS installed with all dependencies
# - `emcee` and `corner` packages (for full uncertainty mode)

# %% [markdown]
# ---
# ## Setup

# %%
import sys
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

# Project Root
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import LCAS modules
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

print(f"Project root: {PROJECT_ROOT}")
print("Imports successful!")

# %% [markdown]
# ---
# ## 1. Create a Simple Satellite Model
#
# For this demonstration, we'll create a simple satellite model using trimesh
# primitives. This avoids the complexity of loading SPICE kernels while still
# demonstrating the full inversion workflow.
#
# In practice, you would load a real satellite model with:
# ```python
# from src.io.stl_loader import STLLoader
# satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)
# ```

# %%
import trimesh
from src.io.stl_loader import Satellite, SatelliteComponent, Facet

def create_simple_satellite():
    """
    Create a simple satellite model with a box body and two solar panels.

    Returns
    -------
    Satellite
        A Satellite object with three components.
    """
    # Create box body (2m x 2m x 3m)
    body_mesh = trimesh.creation.box(extents=[2.0, 2.0, 3.0])
    body_facets = _mesh_to_facets(body_mesh, brdf_name="body_brdf")
    body_component = SatelliteComponent(
        name="Body",
        facets=body_facets,
        mesh=body_mesh,
    )

    # Create North solar panel (thin box: 4m x 0.1m x 2m)
    # Positioned at y = +3m (offset from body center)
    sp_n_mesh = trimesh.creation.box(extents=[4.0, 0.1, 2.0])
    sp_n_mesh.apply_translation([0.0, 3.0, 0.0])
    sp_n_facets = _mesh_to_facets(sp_n_mesh, brdf_name="panel_brdf")
    sp_n_component = SatelliteComponent(
        name="SP_North",
        facets=sp_n_facets,
        mesh=sp_n_mesh,
    )

    # Create South solar panel (thin box: 4m x 0.1m x 2m)
    # Positioned at y = -3m (offset from body center)
    sp_s_mesh = trimesh.creation.box(extents=[4.0, 0.1, 2.0])
    sp_s_mesh.apply_translation([0.0, -3.0, 0.0])
    sp_s_facets = _mesh_to_facets(sp_s_mesh, brdf_name="panel_brdf")
    sp_s_component = SatelliteComponent(
        name="SP_South",
        facets=sp_s_facets,
        mesh=sp_s_mesh,
    )

    # Create satellite
    satellite = Satellite(
        name="SimpleTestSat",
        components=[body_component, sp_n_component, sp_s_component],
    )

    # Set up BRDF parameters
    # Simple diffuse BRDF for testing
    satellite.brdf_mappings = {
        "body_brdf": {
            "r_d": 0.3,  # diffuse reflectance
            "r_s": 0.1,  # specular reflectance
            "n_phong": 10.0,  # Phong exponent
        },
        "panel_brdf": {
            "r_d": 0.1,
            "r_s": 0.6,  # panels are more specular
            "n_phong": 50.0,
        },
    }

    return satellite


def _mesh_to_facets(mesh: trimesh.Trimesh, brdf_name: str):
    """Convert trimesh to list of Facet objects."""
    facets = []
    vertices = mesh.vertices

    for face_idx, face in enumerate(mesh.faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Compute centroid
        centroid = (v0 + v1 + v2) / 3.0

        # Compute normal (cross product)
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(normal)
        normal = normal / (2.0 * area)  # normalize

        facet = Facet(
            vertices=np.array([v0, v1, v2]),
            normal=normal,
            area=area,
            centroid=centroid,
            brdf_name=brdf_name,
        )
        facets.append(facet)

    return facets


# Create the satellite
satellite = create_simple_satellite()
print(f"Created satellite: {satellite.name}")
print(f"Components: {[c.name for c in satellite.components]}")
total_facets = sum(len(c.facets) for c in satellite.components)
print(f"Total facets: {total_facets}")

# %% [markdown]
# ---
# ## 2. Define Observation Geometry
#
# We need to define the observation geometry:
# - Sun position (J2000)
# - Observer position (J2000)
# - Satellite position (J2000)
#
# For simplicity, we use a fixed geometry where:
# - Satellite is at origin
# - Sun is at a fixed position (1 AU along +X, slightly offset in Y)
# - Observer is at a fixed position (40,000 km along +Z)

# %%
# Number of observation points
n_observations = 50

# Time array: 0 to 120 seconds (2 minutes of observations)
observation_times = np.linspace(0, 120, n_observations)

# Fixed geometry in J2000 frame
# Sun: ~1 AU away, mostly along +X
sun_distance_km = 1.496e8  # 1 AU in km
sun_direction = np.array([0.98, 0.17, 0.1])  # mostly +X
sun_direction /= np.linalg.norm(sun_direction)
sun_position = sun_distance_km * sun_direction

# Observer: ground station ~40,000 km away, mostly along +Z
observer_distance_km = 40000.0
observer_direction = np.array([0.1, 0.2, 0.97])  # mostly +Z
observer_direction /= np.linalg.norm(observer_direction)
observer_position = observer_distance_km * observer_direction

# Satellite at origin (for simplicity)
satellite_position = np.array([0.0, 0.0, 0.0])

# Create arrays for all observation times (geometry is constant for this demo)
sun_positions_j2000 = np.tile(sun_position, (n_observations, 1))
observer_positions_j2000 = np.tile(observer_position, (n_observations, 1))
satellite_positions_j2000 = np.tile(satellite_position, (n_observations, 1))
observer_distances = np.full(n_observations, observer_distance_km)

print("Observation geometry defined:")
print(f"  Number of observations: {n_observations}")
print(f"  Time span: {observation_times[0]} to {observation_times[-1]} seconds")
print(f"  Sun position: {sun_position / sun_distance_km} * {sun_distance_km:.2e} km")
print(f"  Observer position: {observer_position / observer_distance_km} * {observer_distance_km:.0f} km")

# %% [markdown]
# ---
# ## 3. Define True Attitude Parameters
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
true_omega_deg_per_s = np.array([0.5, -0.3, 5.0])  # mostly Z-axis spin
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
# ## 4. Generate Synthetic Lightcurve (Forward Model)
#
# Now we generate a synthetic "observed" lightcurve by running the forward model
# with the true parameters. This creates our test data for inversion.

# %%
print("\nGenerating synthetic lightcurve with true parameters...")
start_time = time.time()

# Propagate true attitude
true_quaternions, true_omega_history = propagate_attitude(
    q0=true_q0,
    omega0=true_omega0,
    times=observation_times,
    mode="principal_axis",
)

# Create objective function (we'll use it to generate the forward model)
objective_for_forward = ObjectiveFunction(
    satellite=satellite,
    observation_times=observation_times,
    observed_lightcurve=np.zeros(n_observations),  # placeholder
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    compute_shadows_flag=False,  # Skip shadows for speed in this demo
)

# Compute body-frame vectors and generate lightcurve
k1_vectors, k2_vectors = objective_for_forward._compute_body_frame_vectors(true_quaternions)
true_lightcurve = objective_for_forward._generate_predicted_lightcurve(k1_vectors, k2_vectors)

forward_time = time.time() - start_time
print(f"Forward model completed in {forward_time:.2f} seconds")
print(f"  Lightcurve range: [{true_lightcurve.min():.2f}, {true_lightcurve.max():.2f}] mag")

# %%
# Add synthetic noise to simulate real observations
np.random.seed(42)  # reproducibility
noise_sigma = 0.05  # 0.05 magnitude noise
observed_lightcurve = true_lightcurve + np.random.normal(0, noise_sigma, n_observations)

print(f"\nAdded Gaussian noise (sigma = {noise_sigma} mag)")
print(f"  Observed lightcurve range: [{observed_lightcurve.min():.2f}, {observed_lightcurve.max():.2f}] mag")

# %%
# Plot the synthetic observations
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(observation_times, true_lightcurve, 'b-', linewidth=1.5, label='True lightcurve')
ax.scatter(observation_times, observed_lightcurve, c='red', s=20, alpha=0.7, label='Observed (with noise)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Magnitude')
ax.set_title('Synthetic Lightcurve (Forward Model)')
ax.legend()
ax.invert_yaxis()  # magnitudes are brighter when smaller
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'data' / 'results' / 'inversion_synthetic_lightcurve.png', dpi=150)
plt.show()
print("Figure saved to data/results/inversion_synthetic_lightcurve.png")

# %% [markdown]
# ---
# ## 5. Configuration Options
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
bounds = get_default_bounds(omega_max_deg_per_s=30.0)
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
    bounds = get_bounds(mode, omega_max=np.deg2rad(30.0))
    omega_bounds = bounds[3:]  # last 3 are omega bounds
    omega_max_component = np.rad2deg(omega_bounds[0][1])
    print(f"\n  {mode.name}:")
    print(f"    Omega component bounds: [-{omega_max_component:.2f}, {omega_max_component:.2f}] deg/s")
    max_total_omega = omega_max_component * np.sqrt(3)
    print(f"    Max total omega (at bounds): {max_total_omega:.2f} deg/s")

# %% [markdown]
# ---
# ## 6. Run Inversion with Quick (Fisher) Uncertainty
#
# First, we'll run the inversion with quick uncertainty estimation
# using the Fisher Information Matrix approximation.

# %%
print("\n" + "="*60)
print("Running inversion with QUICK (Fisher) uncertainty mode...")
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
    mode="principal_axis",
    uncertainty_mode="quick",
    compute_shadows=False,  # Skip shadows for speed
    n_starts=3,
    omega_max_deg_per_s=30.0,
    seed=123,  # reproducibility
)

inversion_time_quick = time.time() - start_time
print(f"\nInversion completed in {inversion_time_quick:.2f} seconds")

# %% [markdown]
# ### 6.1 Analyze Quick Mode Results

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
# ## 7. Run Inversion with Full (MCMC) Uncertainty
#
# Now we'll run the inversion with full MCMC posterior sampling.
# This takes longer but provides complete uncertainty characterization.
#
# **Note**: For a production workflow, you would typically use more samples
# (e.g., 5000-10000) and longer burn-in (500-1000).

# %%
print("\n" + "="*60)
print("Running inversion with FULL (MCMC) uncertainty mode...")
print("="*60)
print("(This may take a while...)")

start_time = time.time()

result_full = invert_lightcurve(
    observed_lightcurve=observed_lightcurve,
    satellite=satellite,
    observation_times=observation_times,
    sun_positions_j2000=sun_positions_j2000,
    observer_positions_j2000=observer_positions_j2000,
    satellite_positions_j2000=satellite_positions_j2000,
    observer_distances=observer_distances,
    mode="principal_axis",
    uncertainty_mode="full",
    compute_shadows=False,  # Skip shadows for speed
    n_starts=2,  # Fewer starts for demo (MCMC is the main uncertainty source)
    omega_max_deg_per_s=30.0,
    seed=123,
    mcmc_n_samples=500,  # Reduced for demo (use 1000+ in production)
    mcmc_burn_in=100,
)

inversion_time_full = time.time() - start_time
print(f"\nInversion completed in {inversion_time_full:.2f} seconds")

# %% [markdown]
# ### 7.1 Analyze Full Mode Results

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
# ## 8. Compare Quick vs Full Uncertainty Methods
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
# ## 9. Round-Trip Validation Summary
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
