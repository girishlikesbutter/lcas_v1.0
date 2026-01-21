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
# # Inertia Tensor Calculation from STL Files
#
# This notebook demonstrates how to compute the inertia tensor of a satellite
# from its STL mesh components. We cover:
#
# 1. Loading STL meshes using trimesh
# 2. Validating mesh watertightness
# 3. Computing component inertia tensors
# 4. Combining multiple components into a full satellite inertia
# 5. Validating results against analytical solutions
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
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
else:
    PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.computation import (
    compute_mesh_volume,
    is_mesh_watertight,
    compute_component_inertia,
    translate_inertia,
    compute_inertia_from_stl,
    InertiaResult,
    STLComponent,
)

print(f"Project root: {PROJECT_ROOT}")
print("Imports successful!")

# %% [markdown]
# ---
# ## 1. Loading STL Meshes
#
# We'll demonstrate loading STL files from the `data/models/` directory.
# This project includes several satellite models, including the Intelsat 901.

# %%
# Path to available STL files
models_dir = PROJECT_ROOT / "data" / "models"

# List available models
print("Available satellite models:")
for model_folder in models_dir.iterdir():
    if model_folder.is_dir():
        stl_files = list(model_folder.glob("*.stl"))
        print(f"  {model_folder.name}/: {[f.name for f in stl_files]}")

# %%
# Load Intelsat 901 components
intelsat_dir = models_dir / "intelsat_901"

bus_mesh = trimesh.load(str(intelsat_dir / "bus.stl"))
sp_mesh = trimesh.load(str(intelsat_dir / "sp.stl"))
ad_mesh = trimesh.load(str(intelsat_dir / "ad.stl"))

print("Loaded meshes:")
print(f"  Bus: {len(bus_mesh.vertices)} vertices, {len(bus_mesh.faces)} faces")
print(f"  Solar Panel: {len(sp_mesh.vertices)} vertices, {len(sp_mesh.faces)} faces")
print(f"  Antenna Dish: {len(ad_mesh.vertices)} vertices, {len(ad_mesh.faces)} faces")

# %% [markdown]
# ---
# ## 2. Mesh Watertightness Validation
#
# For accurate volume and inertia calculations, meshes should be "watertight"
# (closed, with no holes). This means each edge is shared by exactly two faces.
#
# The `is_mesh_watertight()` function checks this property.

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
# ---
# ## 3. Single Component Inertia Calculation
#
# For a single component, we can compute its inertia tensor about its center of mass
# using `compute_component_inertia()`. This assumes homogeneous mass distribution.
#
# The function returns both the inertia tensor and the center of mass location.

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
# ---
# ## 4. Parallel Axis Theorem
#
# When combining multiple components, we need to translate each component's inertia
# tensor from its center of mass to a common reference point (usually the body frame origin).
#
# The `translate_inertia()` function applies the parallel axis theorem:
#
# $$I_{body} = I_{cm} + m \cdot (d^2 \cdot \mathbf{I} - \mathbf{d} \otimes \mathbf{d})$$
#
# where $\mathbf{d}$ is the displacement from the body origin to the component CoM.

# %%
# Example: translate bus inertia to body origin (assuming bus is at origin)
# If the bus center of mass is offset from body origin, we translate:
I_bus_at_origin = translate_inertia(I_bus, bus_mass, com_bus)

print("Bus inertia translated to body origin:")
print(f"  [{I_bus_at_origin[0,0]:12.4f}  {I_bus_at_origin[0,1]:12.4f}  {I_bus_at_origin[0,2]:12.4f}]")
print(f"  [{I_bus_at_origin[1,0]:12.4f}  {I_bus_at_origin[1,1]:12.4f}  {I_bus_at_origin[1,2]:12.4f}]")
print(f"  [{I_bus_at_origin[2,0]:12.4f}  {I_bus_at_origin[2,1]:12.4f}  {I_bus_at_origin[2,2]:12.4f}]")

# %% [markdown]
# ---
# ## 5. Full Satellite Inertia from Multiple Components
#
# The `compute_inertia_from_stl()` function combines multiple STL components
# (each with its own position and mass) into a total satellite inertia.
#
# We can define components using either:
# - `STLComponent` dataclass: `STLComponent(mesh, position, mass)`
# - Tuple format: `(mesh, position, mass)`

# %%
# Define satellite components with positions and masses
# Positions are in the body frame (meters)

components = [
    # Main bus at origin
    STLComponent(
        mesh=bus_mesh,
        position=np.array([0.0, 0.0, 0.0]),
        mass=1500.0  # kg
    ),
    # North solar panel (offset in +Y direction)
    STLComponent(
        mesh=sp_mesh,
        position=np.array([0.0, 5.0, 0.0]),
        mass=50.0  # kg
    ),
    # South solar panel (offset in -Y direction)
    STLComponent(
        mesh=sp_mesh,
        position=np.array([0.0, -5.0, 0.0]),
        mass=50.0  # kg
    ),
    # Antenna dish (offset in +X direction)
    STLComponent(
        mesh=ad_mesh,
        position=np.array([3.0, 0.0, 0.0]),
        mass=100.0  # kg
    ),
]

# Compute total satellite inertia
result = compute_inertia_from_stl(components)

print("Full Satellite Inertia Results:")
print("=" * 50)
print(f"Total mass: {result.total_mass:.2f} kg")
print(f"\nCenter of mass (body frame):")
print(f"  [{result.center_of_mass[0]:.4f}, {result.center_of_mass[1]:.4f}, {result.center_of_mass[2]:.4f}]")

# %%
# Display the full inertia tensor
print("Inertia tensor about satellite CoM (kg*m^2):")
I = result.inertia_tensor
print(f"  [{I[0,0]:12.4f}  {I[0,1]:12.4f}  {I[0,2]:12.4f}]")
print(f"  [{I[1,0]:12.4f}  {I[1,1]:12.4f}  {I[1,2]:12.4f}]")
print(f"  [{I[2,0]:12.4f}  {I[2,1]:12.4f}  {I[2,2]:12.4f}]")

# %%
# Display principal moments and axes
print("\nPrincipal moments of inertia (ascending order):")
print(f"  I_1 = {result.principal_moments[0]:.4f} kg*m^2")
print(f"  I_2 = {result.principal_moments[1]:.4f} kg*m^2")
print(f"  I_3 = {result.principal_moments[2]:.4f} kg*m^2")

print("\nPrincipal axes (as column vectors):")
axes = result.principal_axes
for i in range(3):
    print(f"  Axis {i+1}: [{axes[0,i]:.4f}, {axes[1,i]:.4f}, {axes[2,i]:.4f}]")

# %% [markdown]
# ---
# ## 6. Validation with Analytical Solution
#
# To verify the inertia calculation is correct, we compare against the
# analytical solution for a simple shape: a **solid rectangular box**.
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
# ---
# ## 7. Symmetric Two-Component Validation
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
# 1. **Loading meshes** using trimesh
# 2. **Validating watertightness** to ensure accurate volume calculations
# 3. **Computing single component inertia** with `compute_component_inertia()`
# 4. **Translating inertia tensors** using the parallel axis theorem
# 5. **Combining multiple components** with `compute_inertia_from_stl()`
# 6. **Validating against analytical solutions** for simple shapes
#
# The computed inertia tensor can be used for:
# - Attitude dynamics simulation (Euler equations)
# - Lightcurve inversion (estimating rotation state from observations)
# - Spacecraft design and analysis
