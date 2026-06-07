#!/usr/bin/env python3
"""
LCAS Example 5: Interactive 3D Animation
========================================

This script demonstrates how to create interactive 3D animations
showing satellite illumination over time.

Features demonstrated:
- Two color modes: 'lit_status' and 'flux'
- Reference frame visualization
- Sun and observer vectors
- Frame duration control

Run from project root:
    python examples/05_3d_animation.py

Expected output:
    - Two HTML animation files (one for each color mode)
    - Files saved to data/results/<satellite>_results/<date>/

Time to run: ~1-2 minutes

To view: Open the generated .html files in a web browser
"""

import sys
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# LCAS imports
from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import generate_lightcurves
from src.computation.shadow_engine import compute_shadows
from src.computation.brdf import BRDFCalculator, BRDFManager
from src.visualization.plotly_animation_generator import create_interactive_3d_animation


def main():
    print("=" * 60)
    print("LCAS Example 5: Interactive 3D Animation")
    print("=" * 60)

    # =========================================================================
    # SETUP
    # =========================================================================
    print("\n[1/4] Loading configuration and model...")

    config_manager = RSO_ConfigManager(PROJECT_ROOT)

    # Use torus_plate for simpler visualization
    config = config_manager.load_config("torus_plate/torus_plate_config.yaml")
    satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

    print(f"  Satellite: {config.name}")
    total_facets = sum(len(c.facets) for c in satellite.components if c.facets)
    print(f"  Total facets: {total_facets}")

    # =========================================================================
    # COMPUTE GEOMETRY AND SHADOWS
    # =========================================================================
    print("\n[2/4] Computing observation geometry...")

    spice_handler = SpiceHandler()
    metakernel_path = config_manager.get_metakernel_path(config)
    spice_handler.load_metakernel_programmatically(str(metakernel_path))

    # Use fewer points for faster animation
    num_points = 20
    start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
    end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
    epochs = np.linspace(start_et, end_et, num_points)
    time_hours = (epochs - epochs[0]) / 3600

    geometry_data = compute_observation_geometry(
        epochs=epochs,
        satellite_id=config.spice_config.satellite_id,
        observer_id=399999,
        spice_handler=spice_handler,
        config=config
    )

    print(f"  Time points: {num_points}")
    print(f"  Duration: {time_hours[-1]:.1f} hours")

    # =========================================================================
    # GENERATE LIGHT CURVE WITH ANIMATION DATA
    # =========================================================================
    print("\n[3/4] Generating light curve with animation data...")

    # Compute shadows
    lit_status_dict = compute_shadows(
        satellite=satellite,
        k1_vectors=geometry_data['k1_vectors']
    )

    # Setup BRDF
    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(satellite, BRDFManager(config))

    # Generate light curve - IMPORTANT: animate=True to collect animation data
    magnitudes, flux, _, _, _, animation_data = generate_lightcurves(
        facet_lit_status_dict=lit_status_dict,
        k1_vectors_array=geometry_data['k1_vectors'],
        k2_vectors_array=geometry_data['k2_vectors'],
        observer_distances=geometry_data['observer_distances'],
        satellite=satellite,
        epochs=epochs,
        brdf_calculator=brdf_calc,
        animate=True  # ← This is crucial for collecting animation data!
    )

    if animation_data is None or len(animation_data) == 0:
        print("  ERROR: No animation data collected!")
        print("  Make sure animate=True in generate_lightcurves()")
        return

    print(f"  Animation frames collected: {len(animation_data)}")
    print(f"  Magnitude range: {magnitudes.min():.2f} to {magnitudes.max():.2f}")

    # =========================================================================
    # CREATE ANIMATIONS
    # =========================================================================
    print("\n[4/4] Creating animations...")
    output_dir = config_manager.get_output_directory(config)

    # -------------------------------------------------------------------------
    # Animation 1: Lit Status Mode (Discrete Colors)
    # -------------------------------------------------------------------------
    print("\n  Creating 'lit_status' animation...")
    print("    Colors: Yellow=Lit, Blue=Shadowed, Purple=Back-culled")

    path_lit_status = create_interactive_3d_animation(
        animation_data=animation_data,
        magnitudes=magnitudes,
        time_hours=time_hours,
        geometry_data=geometry_data,
        satellite_name=f"{satellite.name}_lit_status",
        output_dir=output_dir,
        show_j2000_frame=True,       # Show inertial reference frame
        show_body_frame=True,        # Show satellite body frame
        show_sun_vector=True,        # Show sun direction arrow
        show_observer_vector=True,   # Show observer direction arrow
        frame_duration_ms=200,       # 200ms per frame (5 fps)
        color_mode='lit_status',     # Discrete coloring
        save=True
    )

    if path_lit_status:
        print(f"    Saved: {path_lit_status}")

    # -------------------------------------------------------------------------
    # Animation 2: Flux Mode (Continuous Brightness)
    # -------------------------------------------------------------------------
    print("\n  Creating 'flux' animation...")
    print("    Colors: Dark Red → Orange → Yellow → White (brightness)")

    path_flux = create_interactive_3d_animation(
        animation_data=animation_data,
        magnitudes=magnitudes,
        time_hours=time_hours,
        geometry_data=geometry_data,
        satellite_name=f"{satellite.name}_flux",
        output_dir=output_dir,
        show_j2000_frame=True,
        show_body_frame=True,
        show_sun_vector=True,
        show_observer_vector=True,
        frame_duration_ms=200,
        color_mode='flux',           # Continuous brightness coloring
        save=True
    )

    if path_flux:
        print(f"    Saved: {path_flux}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("ANIMATIONS CREATED")
    print("=" * 60)

    print("""
Two animation files have been created:

1. *_lit_status_*.html
   - Shows discrete illumination states
   - Yellow: Facet receives direct sunlight
   - Dark Blue: Facet is shadowed by another part
   - Purple: Facet faces away from sun (back-culled)
   - Best for understanding shadow patterns

2. *_flux_*.html
   - Shows continuous brightness per facet
   - Uses warm gradient (red → orange → yellow → white)
   - Brighter colors = higher flux contribution
   - Best for seeing specular highlights and BRDF effects

How to view:
   Open the .html files in any modern web browser

Controls:
   - Left-click + drag: Rotate view
   - Right-click + drag: Pan view
   - Scroll wheel: Zoom in/out
   - Play/Pause: Animation controls at bottom
   - Hover: See component names and values

Reference frames shown:
   - Red/Green/Blue axes (J2000): Inertial reference
   - Cyan/Magenta/Yellow axes (Body): Satellite-fixed frame
   - Yellow arrow: Sun direction
   - Green arrow: Observer direction
""")

    print(f"Output directory: {output_dir}/<today's date>/")
    print("\nExample complete!")


if __name__ == "__main__":
    main()
