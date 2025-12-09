#!/usr/bin/env python3
"""
LCAS Example 2: Sun-Tracking Solar Panels
==========================================

This script demonstrates the SunTrackingBehavior for automatic solar panel
articulation using the Intelsat 901 model.

The solar panels will automatically rotate to face the sun, and you'll see
how this affects the light curve compared to fixed panels.

Run from project root:
    python examples/02_sun_tracking_panels.py

Expected output:
    - Console output showing articulation angles over time
    - Light curve plot with shadowing effects

Time to run: ~1 minute
"""

import sys
import time
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
from src.articulation import (
    ArticulationEngine,
    SunTrackingBehavior,
    calculate_angles_from_behaviors,
    compute_rotation_matrices_from_angles
)
from src.visualization.lightcurve_plotter import create_light_curve_plot


def main():
    print("=" * 60)
    print("LCAS Example 2: Sun-Tracking Solar Panels")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Load Configuration
    # =========================================================================
    print("\n[1/7] Loading Intelsat 901 configuration...")

    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")

    print(f"  Satellite: {config.name}")
    print(f"  Components: {list(config.components.keys())}")

    # =========================================================================
    # STEP 2: Load Satellite Model
    # =========================================================================
    print("\n[2/7] Loading satellite model...")

    satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

    for comp in satellite.components:
        print(f"  {comp.name}: {len(comp.facets)} facets")

    # =========================================================================
    # STEP 3: Setup Articulation Behaviors
    # =========================================================================
    print("\n[3/7] Setting up sun-tracking behaviors...")

    articulation_engine = ArticulationEngine()
    articulation_capabilities = config.articulation_capabilities

    # Setup sun-tracking for both solar panels
    for panel_name in ['SP_North', 'SP_South']:
        if panel_name in articulation_capabilities:
            cap = articulation_capabilities[panel_name]

            sun_tracking = SunTrackingBehavior({
                'rotation_center': cap.rotation_center,
                'rotation_axis': cap.rotation_axis,
                'reference_normal': [1.0, 0.0, 0.0],  # Panel normal when angle=0
                'limits': cap.limits
            })

            articulation_engine.register_component_behavior(panel_name, sun_tracking)
            print(f"  {panel_name}: Sun-tracking enabled")
            print(f"    Rotation axis: {cap.rotation_axis}")
            print(f"    Limits: {cap.limits['min_angle']}° to {cap.limits['max_angle']}°")

    # =========================================================================
    # STEP 4: Initialize SPICE and Compute Geometry
    # =========================================================================
    print("\n[4/7] Initializing SPICE and computing geometry...")

    spice_handler = SpiceHandler()
    metakernel_path = config_manager.get_metakernel_path(config)
    spice_handler.load_metakernel_programmatically(str(metakernel_path))

    # Time setup
    num_points = 15
    start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
    end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
    epochs = np.linspace(start_et, end_et, num_points)

    geometry_data = compute_observation_geometry(
        epochs=epochs,
        satellite_id=config.spice_config.satellite_id,
        observer_id=399999,
        spice_handler=spice_handler,
        config=config
    )

    k1_vectors = geometry_data['k1_vectors']
    k2_vectors = geometry_data['k2_vectors']
    observer_distances = geometry_data['observer_distances']

    print(f"  Time points: {num_points}")
    print(f"  Duration: {(end_et - start_et) / 3600:.1f} hours")

    # =========================================================================
    # STEP 5: Calculate Articulation Angles
    # =========================================================================
    print("\n[5/7] Calculating sun-tracking angles...")

    # Get angles from sun-tracking behavior
    angles_dict = calculate_angles_from_behaviors(
        satellite=satellite,
        k1_vectors=k1_vectors,
        articulation_engine=articulation_engine,
        articulation_offset=0
    )

    # Show angle evolution
    print("\n  Articulation angles over time:")
    print("  " + "-" * 50)

    time_hours = (epochs - epochs[0]) / 3600
    for i in [0, num_points // 2, num_points - 1]:
        print(f"  t = {time_hours[i]:.1f} hours:")
        for comp_name, angles in angles_dict.items():
            print(f"    {comp_name}: {angles[i]:.1f}°")

    # Build explicit angles dict for articulated components only
    explicit_component_angles = {}
    for comp_name in angles_dict:
        if comp_name in articulation_capabilities:
            explicit_component_angles[comp_name] = angles_dict[comp_name]

    # Convert to rotation matrices
    explicit_component_matrices = compute_rotation_matrices_from_angles(
        explicit_component_angles,
        satellite
    )

    print(f"\n  Rotation matrices computed for {len(explicit_component_matrices)} components")

    # =========================================================================
    # STEP 6: Compute Shadows with Articulation
    # =========================================================================
    print("\n[6/7] Computing shadows with articulated panels...")

    shadow_start = time.time()
    lit_status_dict = compute_shadows(
        satellite=satellite,
        k1_vectors=k1_vectors,
        explicit_component_angles=explicit_component_angles,
        explicit_component_matrices=explicit_component_matrices
    )
    shadow_time = time.time() - shadow_start

    print(f"  Shadow computation: {shadow_time:.2f}s")

    # =========================================================================
    # STEP 7: Generate Light Curve
    # =========================================================================
    print("\n[7/7] Generating light curve...")

    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(satellite, BRDFManager(config))

    magnitudes, flux, _, _, distances_out, _ = generate_lightcurves(
        facet_lit_status_dict=lit_status_dict,
        k1_vectors_array=k1_vectors,
        k2_vectors_array=k2_vectors,
        observer_distances=observer_distances,
        satellite=satellite,
        epochs=epochs,
        brdf_calculator=brdf_calc,
        pre_computed_matrices=explicit_component_matrices,
        animate=False
    )

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nMagnitude range: {magnitudes.min():.2f} to {magnitudes.max():.2f}")

    # Phase angles
    phase_angles = np.array([
        np.degrees(np.arccos(np.clip(np.dot(k1_vectors[i], k2_vectors[i]), -1, 1)))
        for i in range(len(epochs))
    ])

    # Save plot
    utc_times = [spice_handler.et_to_utc(e, "C", 0) for e in epochs]
    output_dir = config_manager.get_output_directory(config)

    create_light_curve_plot(
        time_hours=time_hours,
        epochs=epochs,
        magnitudes=magnitudes,
        phase_angles=phase_angles,
        utc_times=utc_times,
        satellite_name=f"{satellite.name} (Sun-Tracking)",
        output_dir=output_dir,
        observer_distances=distances_out,
        save=True,
        plot_mode='shadowed',
        no_plot=True
    )

    print(f"\nPlot saved to: {output_dir}/<today's date>/")
    print("\nExample complete!")
    print("\nKey takeaway: The solar panels automatically computed their")
    print("rotation angles to face the sun at each time point.")


if __name__ == "__main__":
    main()
