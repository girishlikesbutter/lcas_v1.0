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
# # LCAS STL-Based Satellite Light Curve Generation Pipeline
#
# ## Overview
#
# This notebook demonstrates the STL-based satellite light curve generation pipeline with runtime articulation assignment. Unlike the YAML-based approach, this uses pre-tessellated STL files and allows users to assign articulation behaviors programmatically.
#
# ## This Notebook: Torus-Plate Test Model
#
# This uses the simple **torus_plate** model - ideal for:
# - Learning the basic pipeline
# - Fast iteration during development
# - Testing new features
#
# For a more complex example with multiple articulated components, see `lcas_stl_pipeline-is901-3.py`.
#
# ## Pipeline Steps
#
# 1. Load configuration and STL model
# 2. Assign articulation behaviors (optional sun-tracking for torus)
# 3. Initialize SPICE for orbital mechanics
# 4. Compute observation geometry (sun/observer positions)
# 5. Calculate shadows via ray tracing
# 6. Generate light curves
# 7. Create 3D animation

# %% [markdown]
# ---
# ## Cell 1: Configuration and Imports
#
# Set up all necessary imports and load the STL-based configuration.

# %%
# File system operations
import sys
import time
from pathlib import Path
import os

# Project Root - works whether run as script or in Jupyter
if '__file__' in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path.cwd().parent  # Jupyter notebook fallback
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Arrays and data tables
import numpy as np
import pandas as pd

# Import config & brdf manager
from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager

# Import io modules for loading
from src.io.stl_loader import STLLoader

# Import spice_handler for spice tools
from src.spice.spice_handler import SpiceHandler

# Import shadow engine
from src.computation.shadow_engine import get_shadow_engine

# Import articulation modules for moving parts
from src.articulation import (
    ArticulationEngine,
    SunTrackingBehavior,
    calculate_angles_from_behaviors,
    compute_rotation_matrices_from_angles)
from src.interpolation import create_angle_interpolator

# Import computation modules
from src.computation.brdf import BRDFCalculator
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import generate_lightcurves
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status

# Import Visualization modules
from src.visualization.plot_styling import setup_matplotlib_backend
from src.visualization.lightcurve_plotter import create_light_curve_plot
from src.visualization.plotly_animation_generator import create_interactive_3d_animation

# ============================================================================
# LOAD STL CONFIGURATION
# ============================================================================
config_path = "torus_plate/torus_plate_config.yaml"

# Choose a number of points
num_points = 100

# Observer/Ground Station SPICE ID (e.g., 399999 for DST, or your custom kernel ID)
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

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================
print("="*70)
print("LCAS STL-BASED LIGHT CURVE GENERATION PIPELINE")
print("With Runtime Articulation Assignment")
print("="*70)
print("\nConfiguration:")
print(f"  Satellite: {config.name}")
print(f"  Config file: {config_path}")
print(f"  Components: {list(config.components.keys())}")
print(f"  SPICE metakernel exists: {metakernel_path.exists()}")
print(f"  Output directory exists: {output_dir.exists()}")
print(f"\nTime range: {start_time_utc} to {end_time_utc}")
print(f"  Time points: {num_points}")

# %% [markdown]
# ---
# ## Cell 2: Load Satellite from STL Files
#
# Load the satellite geometry directly from STL files instead of YAML definitions.

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

# %% [markdown]
# ---
# ## Cell 3: Runtime Articulation Assignment
#
# This is where users assign articulation behaviors to components. The config file only defines the physical capabilities (rotation axes, limits), not the control behaviors.

# %%
# Create articulation engine
articulation_engine = ArticulationEngine()

# Get articulation capabilities from config
articulation_capabilities = config.articulation_capabilities
print("\nComponents with articulation capability:")
for comp_name, capability in articulation_capabilities.items():
    print(f"  - {comp_name}: axis={capability.rotation_axis}, limits={capability.limits}")

# ============================================================================
# USER-DEFINED ARTICULATION ASSIGNMENT
# ============================================================================
# Example: Assign sun-tracking behavior to the Torus component
if 'Torus' in articulation_capabilities:
    torus_cap = articulation_capabilities['Torus']

    # Create sun-tracking behavior
    sun_tracking = SunTrackingBehavior({
        'rotation_center': torus_cap.rotation_center,
        'rotation_axis': torus_cap.rotation_axis,
        'reference_normal': [1.0, 0.0, 0.0],  # User chooses reference normal
        'limits': torus_cap.limits
    })

    # Register the behavior
    articulation_engine.register_component_behavior('Torus', sun_tracking)
    print("\nAssigned sun-tracking behavior to Torus component")

# Alternative examples using create_angle_interpolator (see Cell 7 for usage):
#
# # Fixed angle example - use 'constant' transition with same value at start/end
# # fixed_angle = 45.0
# # torus_angles = create_angle_interpolator(
# #     keyframe_times_utc=[start_time_utc, end_time_utc],
# #     keyframe_values=[fixed_angle, fixed_angle],
# #     transitions=['constant'],
# #     epochs=epochs,
# #     utc_to_et=spice_handler.utc_to_et
# # )
#
# # Linearly varying angle example
# # torus_angles = create_angle_interpolator(
# #     keyframe_times_utc=[start_time_utc, end_time_utc],
# #     keyframe_values=[0.0, 90.0],  # Rotate from 0 to 90 degrees
# #     transitions=['linear'],
# #     epochs=epochs,
# #     utc_to_et=spice_handler.utc_to_et
# # )
#
# # Oscillating example using numpy (sine wave)
# # amplitude_deg = 30.0
# # period_seconds = 3600.0
# # torus_angles = amplitude_deg * np.sin(2 * np.pi * (epochs - epochs[0]) / period_seconds)

# Create BRDF manager
brdf_manager = BRDFManager(config)

# %% [markdown]
# ---
# ## Cell 4: SPICE Setup
#
# Initialize SPICE kernels for orbital mechanics.

# %%
# Initialize SPICE
print("Initializing SPICE...")
spice_init_start = time.time()
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
spice_init_time = time.time() - spice_init_start
print(f"SPICE initialized ({spice_init_time:.2f}s)")
print()

# Generate time series
print(f"Time range: {start_time_utc} to {end_time_utc}")
print(f"   Time points: {num_points}")

start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, num_points)

duration_hours = (end_et - start_et) / 3600
time_resolution_min = duration_hours * 60 / num_points
print(f"   Duration: {duration_hours:.1f} hours")
print(f"   Resolution: {time_resolution_min:.1f} minutes")

# Calculate time_array in hours for data saving and plotting
time_hours = (epochs - epochs[0]) / 3600.

# Pre-compute UTC times for downstream functions (plotting, saving)
# This avoids passing spice_handler to functions that only need time strings
utc_times = [spice_handler.et_to_utc(epoch, "C", 0) for epoch in epochs]

# %% [markdown]
# ---
# ## Cell 5: Compute Observation Geometry
#
# Calculate satellite positions, orientations, and sun/observer vectors.

# %%
# Compute observation geometry
geometry_start = time.time()

print("Using SPICE-based attitude kernels...")

geometry_data = compute_observation_geometry(
    epochs=epochs,
    satellite_id=satellite_id,
    observer_id=OBSERVER_ID,
    spice_handler=spice_handler,
    config=config
)

geometry_time = time.time() - geometry_start
print(f"Geometry data computed in ({geometry_time:.2f}s)")
print()

# Extract arrays for compatibility
k1_vectors_array = geometry_data['k1_vectors']
k2_vectors_array = geometry_data['k2_vectors']
observer_distances = geometry_data['observer_distances']

print("Geometry data (first 3 entries):")
for item in ['k1_vectors', 'k2_vectors', 'observer_distances']:
    print(f"  {item}: shape={geometry_data[item].shape}")

# %% [markdown]
# ---
# ## Cell 6: Calculate Articulation Angles
#
# Calculate explicit angles for articulated components using the user-assigned behaviors.
# You can also use `create_angle_interpolator` to create custom angle profiles.

# %%
# Step 1: Get sun-tracking angles from behaviors (ONLY angles, no matrices)
sun_angles_dict = calculate_angles_from_behaviors(
    satellite=satellite,
    k1_vectors=k1_vectors_array,
    articulation_engine=articulation_engine,
    articulation_offset=0  # Don't apply offset here, we'll add it separately if needed
)

# Extract sun angles for Torus (the only articulating component in this model)
if 'Torus' in sun_angles_dict:
    torus_angles = sun_angles_dict['Torus']
else:
    # If no behavior was registered, create a fixed angle array
    torus_angles = create_angle_interpolator(
        keyframe_times_utc=[start_time_utc, end_time_utc],
        keyframe_values=[0.0, 0.0],
        transitions=['constant'],
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )

# Step 2: (Optional) Add a time-varying offset using interpolator
# Uncomment to add an offset that varies over time:
# offset_angles = create_angle_interpolator(
#     keyframe_times_utc=[start_time_utc, end_time_utc],
#     keyframe_values=[0.0, 10.0],  # Offset from 0 to 10 degrees
#     transitions=['linear'],
#     epochs=epochs,
#     utc_to_et=spice_handler.utc_to_et
# )
# torus_angles = torus_angles + offset_angles

# Step 3: Build the explicit component angles dictionary
explicit_component_angles = {}
if 'Torus' in articulation_capabilities:
    explicit_component_angles['Torus'] = torus_angles

# Step 4: Convert angles to rotation matrices
explicit_component_matrices = compute_rotation_matrices_from_angles(
    explicit_component_angles,
    satellite
)

print(f"\nSuccessfully created rotation matrices for {len(explicit_component_matrices)} components")

print('Angles for each articulated component (first 3 time-points):')
print()
for component in explicit_component_angles:
    angles = explicit_component_angles[component][:3]
    print(f"{component}: {angles} degrees")

if not explicit_component_angles:
    print("No articulated components found.")

# %% [markdown]
# ---
# ## Cell 7: Shadow Computation
#
# Compute facet-level shadows using ray tracing.

# %%
shadow_start = time.time()

# Compute shadows via ray tracing
lit_status_dict = compute_shadows(
    satellite=satellite,
    k1_vectors=k1_vectors_array,
    explicit_component_angles=explicit_component_angles,
    explicit_component_matrices=explicit_component_matrices
)

shadow_time = time.time() - shadow_start
print(f"Shadows computed ({shadow_time:.2f}s)")
print()

# Also create no-shadows version for comparison
lit_status_dict_unshadowed = create_no_shadow_lit_status(satellite, len(epochs))
print("No-shadow mode initialized")
print()

# %% [markdown]
# ---
# ## Cell 8: Light Curve Generation
#
# Calculate BRDF and generate light curves.

# %%
# Initialize BRDF calculator
brdf_calc = BRDFCalculator()

# Apply BRDF parameters using flexible system
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

lightcurve_shadowed_start = time.time()

# Generate shadowed LC with animation data
magnitudes_shadowed, total_flux_shadowed, _, _, observer_distances_out, animation_data = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict,
    k1_vectors_array=k1_vectors_array,
    k2_vectors_array=k2_vectors_array,
    observer_distances=observer_distances,
    satellite=satellite,
    epochs=epochs,
    brdf_calculator=brdf_calc,
    pre_computed_matrices=explicit_component_matrices,
    generate_no_shadow=False,
    animate=True  # ENABLE ANIMATION DATA COLLECTION
)
lc_dur_shad = time.time() - lightcurve_shadowed_start
print(f"Light curve (shadowed) generated ({lc_dur_shad:.2f}s)")
print(f"Animation data collected: {len(animation_data) if animation_data else 0} frames")

lightcurve_unshadowed_start = time.time()

# Generate unshadowed LC (no animation for this one)
magnitudes_unshadowed, total_flux_unshadowed, _, _, observer_distances_out, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict_unshadowed,
    k1_vectors_array=k1_vectors_array,
    k2_vectors_array=k2_vectors_array,
    observer_distances=observer_distances,
    satellite=satellite,
    epochs=epochs,
    brdf_calculator=brdf_calc,
    pre_computed_matrices=explicit_component_matrices,
    generate_no_shadow=False,
    animate=False  # No animation for unshadowed
)

lc_dur_unshad = time.time() - lightcurve_unshadowed_start
print(f"Light curve (unshadowed) generated ({lc_dur_unshad:.2f}s)")

# %% [markdown]
# ---
# ## Cell 9: Comparison Plot
#
# Create comparison plot of shadowed vs unshadowed light curves.

# %%
# Calculate phase angles for plotting
print("Computing phase angles...")
phase_angles = np.zeros(len(epochs))

for i in range(len(epochs)):
    sun_direction = k1_vectors_array[i]
    observer_direction = k2_vectors_array[i]
    cos_phase = np.dot(sun_direction, observer_direction)
    cos_phase = np.clip(cos_phase, -1.0, 1.0)
    phase_angles[i] = np.degrees(np.arccos(cos_phase))

print(f"Phase angles computed")
print()

# Generate comparison plot (automatically saves PNG and CSV to date-based folder)
print("Creating comparison plot...")

plot_time = create_light_curve_plot(
    time_hours=time_hours,
    epochs=epochs,
    magnitudes=magnitudes_shadowed,  # Primary curve (shadowed)
    phase_angles=phase_angles,
    utc_times=utc_times,
    satellite_name=satellite.name,
    plot_mode="comparison",  # This triggers comparison plot style
    output_dir=output_dir,   # Base output dir - date folder created automatically
    magnitudes_no_shadow=magnitudes_unshadowed,  # Secondary curve (unshadowed)
    observer_distances=observer_distances_out,
    no_plot=False,  # Set to True if you don't want to display inline
    save=True       # Set to False to display without saving
)

print(f"Plot and data saved to: {output_dir}/<today's date>/")

# %% [markdown]
# ---
# ## Cell 10: Interactive 3D Animation (Optional)
#
# Create an interactive 3D animation using Plotly to visualize the satellite's facet-level illumination over time.
#
# **Color Modes:**
# - `'lit_status'`: Discrete coloring - yellow (lit), dark blue (shadowed), purple (back-culled)
# - `'flux'`: Continuous coloring based on facet brightness using a warm gradient (dark red → orange → yellow → white). Uses log scaling with global normalization to reveal both specular glints and diffuse variations.

# %%
# ============================================================================
# INTERACTIVE 3D ANIMATION USING PLOTLY
# ============================================================================
# Check if animation data was collected
if animation_data is None or len(animation_data) == 0:
    print("No animation data available. Re-run light curve generation with animate=True")
else:
    # Choose color mode:
    # - 'lit_status': Discrete coloring (yellow=lit, blue=shadowed, purple=back-culled)
    # - 'flux': Continuous coloring based on facet brightness (warm gradient)
    ANIMATION_COLOR_MODE = 'flux'

    # Create the interactive animation (automatically saves to date-based folder)
    animation_path = create_interactive_3d_animation(
        animation_data=animation_data,
        magnitudes=magnitudes_shadowed,
        time_hours=time_hours,
        geometry_data=geometry_data,
        satellite_name=satellite.name,
        output_dir=output_dir,  # Base output dir - date folder created automatically
        show_j2000_frame=True,
        show_body_frame=True,
        show_sun_vector=True,
        show_observer_vector=True,
        frame_duration_ms=100,
        save=True,  # Set to False to display without saving
        color_mode=ANIMATION_COLOR_MODE  # 'lit_status' or 'flux'
    )

    if animation_path:
        print(f"Animation successfully created at: {animation_path}")

# %% [markdown]
# ---
# ## Summary
#
# This notebook demonstrates the key features of the STL-based pipeline:
#
# 1. **STL Model Loading**: Satellite geometry loaded from pre-tessellated STL files
# 2. **Runtime Articulation**: Behaviors assigned programmatically, not in config files
# 3. **Angle Interpolation**: Use `create_angle_interpolator` for custom angle profiles
# 4. **Clean Separation**: Config defines physical capabilities, user defines control logic
# 5. **Full Compatibility**: Works with existing LCAS pipeline components
#
# ### Articulation Options
#
# You can control articulation in two ways:
#
# **Using Behaviors (Cell 3):**
# - `SunTrackingBehavior`: Automatically track the sun direction
#
# **Using Angle Interpolation (Cell 7):**
# - **Fixed Angle**: Use `transitions=['constant']` with same start/end values
# - **Linear Ramp**: Use `transitions=['linear']` between keyframe values
# - **Step Changes**: Use `transitions=['step']` for discrete jumps
# - **Oscillating**: Use numpy to create sine wave arrays
# - **Composable**: Add offset arrays to behavior-computed angles
#
# The same satellite model can be used with different articulation strategies without modifying any configuration files!
