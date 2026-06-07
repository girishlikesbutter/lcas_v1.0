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
# # Inertia Tensor Calculation from STL Files
#
# This notebook demonstrates how to compute the inertia tensor of a satellite
# from its STL mesh components. We cover:
#
# 1. **Config-based workflow**: Loading component positions from config files
# 2. **Articulation support**: Computing inertia at different articulation angles
# 3. **Low-level API**: Direct mesh loading and inertia calculation
# 4. **Validation**: Comparing against analytical solutions
#
# **Prerequisites:** Ensure you have trimesh installed (`pip install trimesh`).

# %% [markdown]
# ---
# ## Setup

# %%
import sys
from pathlib import Path

import numpy as np
import trimesh

# Project Root
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.computation import (
    compute_mesh_volume,
    is_mesh_watertight,
    compute_component_inertia,
    translate_inertia,
    compute_inertia_from_stl,
    compute_inertia_from_config,
    load_components_from_config,
    InertiaResult,
    STLComponent,
)
from src.config.rso_config_manager import RSO_ConfigManager

print(f"Project root: {PROJECT_ROOT}")
print("Imports successful!")

# %% [markdown]
# ---
# ## 1. Config-Based Workflow (Recommended)
#
# The simplest way to compute satellite inertia is using the config-based workflow.
# This approach:
# - Loads component positions directly from the satellite config file
# - Handles articulation capabilities automatically
# - Applies default articulation angles when not specified
#
# ### 1.1 Load the Satellite Configuration

# %%
# Initialize config manager with project root
config_manager = RSO_ConfigManager(PROJECT_ROOT)

# Load Intelsat 901 configuration
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")

print(f"Loaded configuration for: {config.name}")
print(f"\nComponents defined in config:")
for comp_name, comp_def in config.components.items():
    print(f"  {comp_name}:")
    print(f"    STL file: {comp_def.stl_file}")
    print(f"    Position: {comp_def.position}")

# %%
# Show articulation capabilities from config
print("Articulation capabilities:")
for comp_name, art_cap in config.articulation_capabilities.items():
    print(f"  {comp_name}:")
    print(f"    Rotation center: {art_cap.rotation_center}")
    print(f"    Rotation axis: {art_cap.rotation_axis}")
    print(f"    Limits: {art_cap.limits['min_angle']}° to {art_cap.limits['max_angle']}°")

# %% [markdown]
# ### 1.2 Define Component Masses
#
# Masses are defined in the notebook, not in the config file. This allows flexibility
# to adjust masses without modifying the satellite configuration.

# %%
# Define component masses (kg)
# These are example values for demonstration purposes
masses = {
    "Bus": 1532.0,       # Main spacecraft bus
    "SP_North": 170.0,    # North solar panel
    "SP_South": 170.0,    # South solar panel
    "AD_West": 50.0,     # West antenna dish
    "AD_East": 50.0,     # East antenna dish
}

total_mass = sum(masses.values())
print(f"Component masses (kg):")
for name, mass in masses.items():
    print(f"  {name}: {mass:.1f} kg")
print(f"\nTotal mass: {total_mass:.1f} kg")

# %% [markdown]
# ### 1.3 Compute Inertia with Default Articulation Angles
#
# When calling `compute_inertia_from_config()` without specifying articulation angles,
# default values are applied:
# - **Antenna dishes** (components with 'AD' prefix): 15 degrees
# - **Other articulable components** (e.g., solar panels): 0 degrees
#
# A warning is logged when default angles are used.

# %%
# Compute inertia with default articulation angles
# Note: Warnings will be logged for components using default angles
import logging
logging.basicConfig(level=logging.WARNING)

result_default = compute_inertia_from_config(config, config_manager, masses)

print("Inertia with DEFAULT articulation angles:")
print("=" * 55)
print(f"Total mass: {result_default.total_mass:.2f} kg")
print(f"\nCenter of mass (body frame):")
print(f"  [{result_default.center_of_mass[0]:.4f}, {result_default.center_of_mass[1]:.4f}, {result_default.center_of_mass[2]:.4f}] m")

# %%
# Display the inertia tensor
print("Inertia tensor about satellite CoM (kg*m^2):")
I = result_default.inertia_tensor
print(f"  [{I[0,0]:12.2f}  {I[0,1]:12.2f}  {I[0,2]:12.2f}]")
print(f"  [{I[1,0]:12.2f}  {I[1,1]:12.2f}  {I[1,2]:12.2f}]")
print(f"  [{I[2,0]:12.2f}  {I[2,1]:12.2f}  {I[2,2]:12.2f}]")

print("\nPrincipal moments of inertia:")
print(f"  I_1 = {result_default.principal_moments[0]:.2f} kg*m^2")
print(f"  I_2 = {result_default.principal_moments[1]:.2f} kg*m^2")
print(f"  I_3 = {result_default.principal_moments[2]:.2f} kg*m^2")

# %% [markdown]
# ### 1.4 Compute Inertia with Custom Articulation Angles
#
# You can specify explicit articulation angles for any articulable component.
# Angles are validated against the limits defined in the config.

# %%
# Define custom articulation angles (degrees)
# Solar panels rotated, antenna dishes at their limits
custom_angles = {
    "SP_North": 45.0,   # Solar panel rotated 45 degrees
    "SP_South": -30.0,  # Solar panel rotated -30 degrees
    "AD_West": 45.0,    # Antenna dish at 45 degrees
    "AD_East": 90.0,    # Antenna dish at maximum angle
}

result_custom = compute_inertia_from_config(
    config, config_manager, masses,
    articulation_angles=custom_angles
)

print("Inertia with CUSTOM articulation angles:")
print("=" * 55)
print(f"Articulation angles used:")
for name, angle in custom_angles.items():
    print(f"  {name}: {angle}°")

print(f"\nCenter of mass (body frame):")
print(f"  [{result_custom.center_of_mass[0]:.4f}, {result_custom.center_of_mass[1]:.4f}, {result_custom.center_of_mass[2]:.4f}] m")

# %%
# Display the inertia tensor with custom angles
print("Inertia tensor about satellite CoM (kg*m^2):")
I = result_custom.inertia_tensor
print(f"  [{I[0,0]:12.2f}  {I[0,1]:12.2f}  {I[0,2]:12.2f}]")
print(f"  [{I[1,0]:12.2f}  {I[1,1]:12.2f}  {I[1,2]:12.2f}]")
print(f"  [{I[2,0]:12.2f}  {I[2,1]:12.2f}  {I[2,2]:12.2f}]")

print("\nPrincipal moments of inertia:")
print(f"  I_1 = {result_custom.principal_moments[0]:.2f} kg*m^2")
print(f"  I_2 = {result_custom.principal_moments[1]:.2f} kg*m^2")
print(f"  I_3 = {result_custom.principal_moments[2]:.2f} kg*m^2")

# %% [markdown]
# ### 1.5 Compare Inertia at Different Articulation States
#
# The articulation state can significantly affect the satellite's inertia tensor,
# especially for large articulating components like solar panels.

# %%
# Compute inertia at several articulation states
articulation_states = {
    "Stowed (0°)": {"SP_North": 0.0, "SP_South": 0.0, "AD_West": 0.0, "AD_East": 0.0},
    "Panels +45°": {"SP_North": 45.0, "SP_South": 45.0, "AD_West": 15.0, "AD_East": 15.0},
    "Panels +90°": {"SP_North": 90.0, "SP_South": 90.0, "AD_West": 15.0, "AD_East": 15.0},
    "Panels -45°": {"SP_North": -45.0, "SP_South": -45.0, "AD_West": 15.0, "AD_East": 15.0},
}

results_by_state = {}
print("Inertia Comparison at Different Articulation States")
print("=" * 70)
print(f"{'State':<15} {'I_1 (kg*m^2)':<15} {'I_2 (kg*m^2)':<15} {'I_3 (kg*m^2)':<15}")
print("-" * 70)

for state_name, angles in articulation_states.items():
    result = compute_inertia_from_config(
        config, config_manager, masses,
        articulation_angles=angles
    )
    results_by_state[state_name] = result

    pm = result.principal_moments
    print(f"{state_name:<15} {pm[0]:>14.2f} {pm[1]:>14.2f} {pm[2]:>14.2f}")

# %%
# Show how center of mass shifts with articulation
print("\nCenter of Mass at Different Articulation States")
print("=" * 70)
print(f"{'State':<15} {'X (m)':<15} {'Y (m)':<15} {'Z (m)':<15}")
print("-" * 70)

for state_name, result in results_by_state.items():
    com = result.center_of_mass
    print(f"{state_name:<15} {com[0]:>14.4f} {com[1]:>14.4f} {com[2]:>14.4f}")

# %% [markdown]
# ---
# ## 2. Low-Level API: Direct Mesh Loading
#
# For more control, you can load meshes directly and use the lower-level functions.
# This approach is useful when you need to:
# - Work with meshes not defined in a config file
# - Customize the loading process
# - Validate individual mesh properties

# %%
# Path to available STL files
models_dir = PROJECT_ROOT / "data" / "models"

# List available models
print("Available satellite models:")
for model_folder in models_dir.iterdir():
    if model_folder.is_dir():
        stl_files = list(model_folder.glob("*.stl"))
        if stl_files:
            print(f"  {model_folder.name}/: {[f.name for f in stl_files]}")

# %%
# Load Intelsat 901 components manually
intelsat_dir = models_dir / "intelsat_901"

bus_mesh = trimesh.load(str(intelsat_dir / "bus.stl"))
sp_mesh = trimesh.load(str(intelsat_dir / "sp.stl"))
ad_mesh = trimesh.load(str(intelsat_dir / "ad.stl"))

print("Loaded meshes:")
print(f"  Bus: {len(bus_mesh.vertices)} vertices, {len(bus_mesh.faces)} faces")
print(f"  Solar Panel: {len(sp_mesh.vertices)} vertices, {len(sp_mesh.faces)} faces")
print(f"  Antenna Dish: {len(ad_mesh.vertices)} vertices, {len(ad_mesh.faces)} faces")

# %% [markdown]
# ### 2.1 Mesh Watertightness Validation
#
# For accurate volume and inertia calculations, meshes should be "watertight"
# (closed, with no holes). The `is_mesh_watertight()` function checks this property.

# %%
# Check watertightness of each mesh
print("Watertightness check:")
print(f"  Bus: {'watertight' if is_mesh_watertight(bus_mesh) else 'NOT watertight'}")
print(f"  Solar Panel: {'watertight' if is_mesh_watertight(sp_mesh) else 'NOT watertight'}")
print(f"  Antenna Dish: {'watertight' if is_mesh_watertight(ad_mesh) else 'NOT watertight'}")

# %%
# Compute volumes (will warn if not watertight)
print("\nMesh volumes:")
print(f"  Bus volume: {compute_mesh_volume(bus_mesh):.6f} units^3")
print(f"  Solar Panel volume: {compute_mesh_volume(sp_mesh):.6f} units^3")
print(f"  Antenna Dish volume: {compute_mesh_volume(ad_mesh):.6f} units^3")

# %% [markdown]
# ### 2.2 Single Component Inertia Calculation
#
# For a single component, `compute_component_inertia()` computes the inertia tensor
# about its center of mass, assuming homogeneous mass distribution.

# %%
# Assign a mass to the bus component (in kg)
bus_mass = 1500.0  # kg

# Compute inertia tensor about the component's center of mass
I_bus, com_bus = compute_component_inertia(bus_mesh, bus_mass)

print("Bus component:")
print(f"  Mass: {bus_mass} kg")
print(f"  Center of mass: [{com_bus[0]:.4f}, {com_bus[1]:.4f}, {com_bus[2]:.4f}]")
print(f"\n  Inertia tensor about CoM (kg*m^2):")
print(f"    [{I_bus[0,0]:12.4f}  {I_bus[0,1]:12.4f}  {I_bus[0,2]:12.4f}]")
print(f"    [{I_bus[1,0]:12.4f}  {I_bus[1,1]:12.4f}  {I_bus[1,2]:12.4f}]")
print(f"    [{I_bus[2,0]:12.4f}  {I_bus[2,1]:12.4f}  {I_bus[2,2]:12.4f}]")

# %% [markdown]
# ### 2.3 Combining Components with STLComponent
#
# The `compute_inertia_from_stl()` function combines multiple components using
# either `STLComponent` dataclass or tuples.

# %%
# Define satellite components with positions and masses
# Positions match those in the Intelsat 901 config
components = [
    STLComponent(mesh=bus_mesh, position=np.array([0.0, 0.0, 0.0]), mass=1500.0),
    STLComponent(mesh=sp_mesh, position=np.array([0.0, 0.0, 9.2]), mass=50.0),
    STLComponent(mesh=sp_mesh, position=np.array([0.0, 0.0, -9.2]), mass=50.0),
    STLComponent(mesh=ad_mesh, position=np.array([-1.5, 4.0, 0.0]), mass=25.0),
    STLComponent(mesh=ad_mesh, position=np.array([-1.5, -4.0, 0.0]), mass=25.0),
]

# Compute total satellite inertia
result_manual = compute_inertia_from_stl(components)

print("Full Satellite Inertia (manually assembled):")
print("=" * 50)
print(f"Total mass: {result_manual.total_mass:.2f} kg")
print(f"\nCenter of mass (body frame):")
print(f"  [{result_manual.center_of_mass[0]:.4f}, {result_manual.center_of_mass[1]:.4f}, {result_manual.center_of_mass[2]:.4f}]")

# %%
# Display principal moments
print("Principal moments of inertia:")
print(f"  I_1 = {result_manual.principal_moments[0]:.2f} kg*m^2")
print(f"  I_2 = {result_manual.principal_moments[1]:.2f} kg*m^2")
print(f"  I_3 = {result_manual.principal_moments[2]:.2f} kg*m^2")

# %% [markdown]
# ---
# ## 3. Validation with Analytical Solutions
#
# To verify the inertia calculation is correct, we compare against the
# analytical solution for simple shapes: unit cube and rectangular box.
#
# For a solid box with dimensions $a \times b \times c$ and mass $m$,
# the principal moments of inertia about the center of mass are:
#
# $$I_{xx} = \frac{m}{12}(b^2 + c^2)$$
# $$I_{yy} = \frac{m}{12}(a^2 + c^2)$$
# $$I_{zz} = \frac{m}{12}(a^2 + b^2)$$

# %%
# Create a unit cube mesh centered at origin for validation
cube_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
cube_mass = 1.0  # kg

print("Unit cube validation mesh:")
print(f"  Dimensions: 1.0 x 1.0 x 1.0 m")
print(f"  Mass: {cube_mass} kg")
print(f"  Watertight: {is_mesh_watertight(cube_mesh)}")
print(f"  Volume: {compute_mesh_volume(cube_mesh):.6f} m^3 (expected: 1.0)")

# %%
# Compute inertia tensor for the unit cube
I_cube, com_cube = compute_component_inertia(cube_mesh, cube_mass)

print(f"\nCenter of mass: [{com_cube[0]:.6f}, {com_cube[1]:.6f}, {com_cube[2]:.6f}]")
print("  (expected: [0, 0, 0] for centered cube)")

# %%
# Compare with analytical solution
# For a cube with side s and mass m: I_xx = I_yy = I_zz = (m/12) * 2*s^2 = m*s^2/6
s = 1.0  # side length
analytical_I = cube_mass * (2 * s**2) / 12.0  # = 1/6 for unit cube with unit mass

print("\nInertia tensor comparison:")
print(f"  Computed I_xx: {I_cube[0,0]:.6f}")
print(f"  Computed I_yy: {I_cube[1,1]:.6f}")
print(f"  Computed I_zz: {I_cube[2,2]:.6f}")
print(f"  Analytical:    {analytical_I:.6f}")

# Check relative error
errors = [
    abs(I_cube[0,0] - analytical_I) / analytical_I,
    abs(I_cube[1,1] - analytical_I) / analytical_I,
    abs(I_cube[2,2] - analytical_I) / analytical_I,
]
print(f"\nRelative errors: {[f'{e*100:.4f}%' for e in errors]}")

# Off-diagonal elements should be zero (or nearly zero)
print(f"\nOff-diagonal elements (should be ~0):")
print(f"  I_xy: {I_cube[0,1]:.2e}")
print(f"  I_xz: {I_cube[0,2]:.2e}")
print(f"  I_yz: {I_cube[1,2]:.2e}")

# %%
# Validate with an asymmetric box (different dimensions)
box_dims = np.array([2.0, 3.0, 4.0])  # a, b, c in meters
box_mass = 10.0  # kg

box_mesh = trimesh.creation.box(extents=box_dims)
I_box, com_box = compute_component_inertia(box_mesh, box_mass)

# Analytical principal moments for a rectangular box
a, b, c = box_dims
I_xx_analytical = (box_mass / 12.0) * (b**2 + c**2)
I_yy_analytical = (box_mass / 12.0) * (a**2 + c**2)
I_zz_analytical = (box_mass / 12.0) * (a**2 + b**2)

print(f"\nAsymmetric box validation ({a} x {b} x {c} m, {box_mass} kg):")
print(f"  Watertight: {is_mesh_watertight(box_mesh)}")
print(f"  Volume: {compute_mesh_volume(box_mesh):.6f} m^3 (expected: {a*b*c:.1f})")
print(f"\nComputed vs Analytical principal moments:")
print(f"  I_xx: {I_box[0,0]:10.4f} vs {I_xx_analytical:10.4f} (error: {abs(I_box[0,0]-I_xx_analytical)/I_xx_analytical*100:.4f}%)")
print(f"  I_yy: {I_box[1,1]:10.4f} vs {I_yy_analytical:10.4f} (error: {abs(I_box[1,1]-I_yy_analytical)/I_yy_analytical*100:.4f}%)")
print(f"  I_zz: {I_box[2,2]:10.4f} vs {I_zz_analytical:10.4f} (error: {abs(I_box[2,2]-I_zz_analytical)/I_zz_analytical*100:.4f}%)")

# %% [markdown]
# ### 3.1 Symmetric Two-Component Validation
#
# Validate the multi-component calculation using two identical cubes placed
# symmetrically about the origin. The parallel axis theorem should give
# predictable results.

# %%
# Two unit cubes, each offset by 2m from the origin along the X axis
cube_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
cube_mass = 1.0  # kg each

components_symmetric = [
    STLComponent(mesh=cube_mesh, position=np.array([2.0, 0.0, 0.0]), mass=cube_mass),
    STLComponent(mesh=cube_mesh, position=np.array([-2.0, 0.0, 0.0]), mass=cube_mass),
]

result_symmetric = compute_inertia_from_stl(components_symmetric)

print("Symmetric two-cube system:")
print(f"  Total mass: {result_symmetric.total_mass} kg (expected: 2.0)")
print(f"  Center of mass: {result_symmetric.center_of_mass}")
print(f"    (expected: [0, 0, 0] due to symmetry)")

# %%
# Analytical solution for two point masses at x = +/-d:
# The CoM is at origin, each mass is at distance d from CoM
# I_yy = I_zz = 2 * m * d^2 (for rotation about y or z axis)
# I_xx = 0 (for rotation about x axis, masses are on the axis)
# But since these are cubes with finite size, we also add the cube's own inertia

d = 2.0  # offset distance
I_cube_principal = cube_mass * (2 * 1.0**2) / 12.0  # = 1/6 for unit cube

# For I_yy and I_zz, each cube contributes its own inertia plus parallel axis term
# Parallel axis: m * d^2 for each cube (rotation about axis through CoM)
I_yy_expected = 2 * (I_cube_principal + cube_mass * d**2)
I_zz_expected = 2 * (I_cube_principal + cube_mass * d**2)
# For I_xx, cubes are aligned along x, so parallel axis adds nothing for x
I_xx_expected = 2 * I_cube_principal

print("\nInertia tensor about satellite CoM:")
I = result_symmetric.inertia_tensor
print(f"  Computed I_xx: {I[0,0]:.6f}  Expected: {I_xx_expected:.6f}")
print(f"  Computed I_yy: {I[1,1]:.6f}  Expected: {I_yy_expected:.6f}")
print(f"  Computed I_zz: {I[2,2]:.6f}  Expected: {I_zz_expected:.6f}")

# Verify
print("\nValidation:")
print(f"  I_xx error: {abs(I[0,0] - I_xx_expected):.6e}")
print(f"  I_yy error: {abs(I[1,1] - I_yy_expected):.6e}")
print(f"  I_zz error: {abs(I[2,2] - I_zz_expected):.6e}")

# %% [markdown]
# ---
# ## Summary
#
# This notebook demonstrated the complete workflow for computing satellite inertia
# from STL mesh files:
#
# ### Config-Based Workflow (Recommended)
# 1. **Load configuration** using `RSO_ConfigManager`
# 2. **Define masses** in the notebook (not in config)
# 3. **Compute inertia** with `compute_inertia_from_config()`
# 4. **Articulation support** - default and custom angles
#
# ### Low-Level API
# 1. **Loading meshes** directly using trimesh
# 2. **Validating watertightness** with `is_mesh_watertight()`
# 3. **Computing single component inertia** with `compute_component_inertia()`
# 4. **Combining multiple components** with `compute_inertia_from_stl()`
#
# ### Validation
# - Unit cube and asymmetric box match analytical formulas
# - Symmetric two-cube test validates parallel axis theorem
#
# The computed inertia tensor can be used for:
# - Attitude dynamics simulation (Euler equations)
# - Lightcurve inversion (estimating rotation state from observations)
# - Spacecraft design and analysis

# %%
