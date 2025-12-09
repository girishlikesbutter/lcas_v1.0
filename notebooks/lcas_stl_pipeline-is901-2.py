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
# l
# This notebook demonstrates the STL-based satellite light curve generation pipeline with runtime articulation assignment. Unlike the YAML-based approach, this uses pre-tessellated STL files and allows users to assign articulation behaviors programmatically.

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
import quaternion  # For custom attitude interpolation

# Import config & brdf manager
from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager

# Import io modules for loading
from src.io.stl_loader import STLLoader

# Import spice_handler for spice tools
from src.spice.spice_handler import SpiceHandler

# Import shadow engine
from src.computation.shadow_engine import get_shadow_engine

# Import articulation modules and interpolator for moving parts
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
config_path = "intelsat_901/intelsat_901_config.yaml"

# Choose a number of points
num_points = 100

# Set fixed antenna dish angles
dish_angle = 15

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
# Get articulation capabilities of articulating components
sp_cap = articulation_capabilities['SP_North'] # Same for both panels, so obtained just once
ad_east_cap = articulation_capabilities['AD_East']
ad_west_cap = articulation_capabilities['AD_West']

# Create behaviours for each articulating component
sun_tracking = SunTrackingBehavior({
    'rotation_center': sp_cap.rotation_center,
    'rotation_axis': sp_cap.rotation_axis,
    'reference_normal': [1.0, 0.0, 0.0],  # User chooses reference normal
    'limits': sp_cap.limits
    })

# Register the behaviors
articulation_engine.register_component_behavior('SP_North', sun_tracking)
articulation_engine.register_component_behavior('SP_South', sun_tracking)

print("\nAssigned sun-tracking behavior to Solar Panel components")

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
# ## Cell 5: Compute Observation Geometry with Optional Custom Attitudes
#
# Calculate satellite positions, orientations, and sun/observer vectors.
#
# **NEW FEATURE**: Toggle between SPICE-based attitudes and custom SLERP-interpolated quaternions.
# - Set `CUSTOM_QUATS = False` to use attitude kernels from SPICE (default)
# - Set `CUSTOM_QUATS = True` to use manually defined quaternions with SLERP interpolation
#
# This is useful for:
# - Testing attitude variations without creating SPICE kernels
# - Interpolating sparse SPICE data more densely
# - Prototyping attitude control scenarios

# %%
# ============================================================================
# CUSTOM ATTITUDE CONTROL
# ============================================================================
# Toggle between custom attitude interpolation and SPICE-based attitudes
CUSTOM_QUATS = True

# Define custom attitude keyframes (quaternions in scalar-first format: w, x, y, z)
# Example: Two keyframes defining satellite rotation over the time period
custom_quaternions = [
    quaternion.quaternion(0.707, 0.707, 0.0, 0.0),  # 90 degree rotation about body X at start time
    quaternion.quaternion(0.707, 0.0, 0.707, 0.0)  # 90 degree rotation about body Y by end time
]

custom_attitude_times = [start_time_utc, end_time_utc]

# ============================================================================
# COMPUTE OBSERVATION GEOMETRY
# ============================================================================
geometry_start = time.time()

if CUSTOM_QUATS:
    print("Using CUSTOM attitude interpolation (SLERP)...")

    # Build attitude keyframes dictionary for interpolation
    attitude_keyframes = {
        'times': custom_attitude_times,
        'time_format': 'utc',
        'attitudes': custom_quaternions,
        'format': 'quaternion'
    }

    # Compute geometry with custom attitudes
    geometry_data = compute_observation_geometry(
        epochs=epochs,
        satellite_id=satellite_id,
        observer_id=OBSERVER_ID,
        spice_handler=spice_handler,
        config=config,
        attitude_keyframes=attitude_keyframes  # This triggers SLERP interpolation
    )
    print("Custom attitudes interpolated via SLERP")

else:
    print("Using SPICE-based attitude kernels...")

    # Standard SPICE path (requires attitude kernel in metakernel)
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
# ## Cell 6: [OPTIONAL] Simple Angle Interpolation
#
# This cell demonstrates the simplified angle interpolation system.
# You can directly create angle arrays from keyframes and compose them as needed.

# %%
# Step 1: Get sun-tracking angles from behaviors (ONLY angles, no matrices)
sun_angles_dict = calculate_angles_from_behaviors(
    satellite=satellite,
    k1_vectors=k1_vectors_array,
    articulation_engine=articulation_engine,
    articulation_offset=0  # Don't apply offset here, we'll add it separately
)

# Extract sun angles for solar panels (they both get the same angles)
sun_angles = sun_angles_dict['SP_North']  # Same for both panels

# Step 2: Create time-varying offset using interpolator
offset_angles = create_angle_interpolator(
    keyframe_times_utc=[
        '2020-02-05T10:00:00',
        '2020-02-05T10:30:00',
        '2020-02-05T14:30:00'
    ],
    keyframe_values=[5.5, 8.2, 13.5],  # Offset varies over time
    transitions=['linear', 'step'],
    step_params=[None, 0.99],
    epochs=epochs,
    utc_to_et=spice_handler.utc_to_et
)

# Step 3: Compose final solar panel angles
sp_angles = sun_angles + offset_angles

# Step 4: Create fixed angle for antenna dishes
ad_angles = create_angle_interpolator(
    keyframe_times_utc=[start_time_utc, end_time_utc],
    keyframe_values=[dish_angle, dish_angle],
    transitions=['constant'],
    epochs=epochs,
    utc_to_et=spice_handler.utc_to_et
)

# Step 5: Build the explicit component angles dictionary
explicit_component_angles = {}
explicit_component_angles['SP_North'] = sp_angles
explicit_component_angles['SP_South'] = sp_angles
explicit_component_angles['AD_West'] = ad_angles
explicit_component_angles['AD_East'] = ad_angles

# Step 6: Convert angles to rotation matrices
explicit_component_matrices = compute_rotation_matrices_from_angles(
    explicit_component_angles,
    satellite
)

print(f"\nCreated variable offset array: {offset_angles[0]:.1f}° to {offset_angles[-1]:.1f}°")
print(f"\nThe offset angles are {offset_angles}")
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

plot = create_light_curve_plot(
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
