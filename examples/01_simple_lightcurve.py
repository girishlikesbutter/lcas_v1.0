#!/usr/bin/env python3
"""
LCAS Example 1: Simple Light Curve Generation
==============================================

This script demonstrates the minimal pipeline for generating a satellite
light curve using the torus_plate test model.

Run from project root:
    python examples/01_simple_lightcurve.py

Expected output:
    - Console output showing magnitudes
    - Light curve plot saved to data/results/torus_plate_results/<date>/

Time to run: ~30 seconds
"""

import sys
import time
from pathlib import Path

# =============================================================================
# SETUP PROJECT ROOT
# =============================================================================
# This ensures imports work whether run from project root or examples/
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
from src.visualization.lightcurve_plotter import create_light_curve_plot


def main():
    print("=" * 60)
    print("LCAS Example 1: Simple Light Curve Generation")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Load Configuration
    # =========================================================================
    print("\n[1/6] Loading configuration...")

    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config("torus_plate/torus_plate_config.yaml")

    print(f"  Satellite: {config.name}")
    print(f"  Time range: {config.simulation_defaults.start_time}")
    print(f"           to {config.simulation_defaults.end_time}")

    # =========================================================================
    # STEP 2: Load Satellite Model
    # =========================================================================
    print("\n[2/6] Loading satellite model from STL files...")

    satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

    total_facets = sum(len(c.facets) for c in satellite.components if c.facets)
    print(f"  Components: {len(satellite.components)}")
    print(f"  Total facets: {total_facets:,}")

    # =========================================================================
    # STEP 3: Initialize SPICE
    # =========================================================================
    print("\n[3/6] Initializing SPICE orbital mechanics...")

    spice_handler = SpiceHandler()
    metakernel_path = config_manager.get_metakernel_path(config)
    spice_handler.load_metakernel_programmatically(str(metakernel_path))

    # Generate time points
    num_points = 25  # Increase for smoother curves
    start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
    end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
    epochs = np.linspace(start_et, end_et, num_points)

    print(f"  Time points: {num_points}")

    # =========================================================================
    # STEP 4: Compute Observation Geometry
    # =========================================================================
    print("\n[4/6] Computing observation geometry...")

    geometry_data = compute_observation_geometry(
        epochs=epochs,
        satellite_id=config.spice_config.satellite_id,
        observer_id=399999,  # Ground station ID
        spice_handler=spice_handler,
        config=config
    )

    k1_vectors = geometry_data['k1_vectors']  # Sun direction
    k2_vectors = geometry_data['k2_vectors']  # Observer direction
    observer_distances = geometry_data['observer_distances']

    print(f"  Sun vectors shape: {k1_vectors.shape}")
    print(f"  Observer vectors shape: {k2_vectors.shape}")

    # =========================================================================
    # STEP 5: Compute Shadows
    # =========================================================================
    print("\n[5/6] Computing shadows...")

    shadow_start = time.time()
    lit_status_dict = compute_shadows(
        satellite=satellite,
        k1_vectors=k1_vectors
    )
    shadow_time = time.time() - shadow_start

    print(f"  Shadow computation: {shadow_time:.2f}s")

    # =========================================================================
    # STEP 6: Generate Light Curve
    # =========================================================================
    print("\n[6/6] Generating light curve...")

    # Setup BRDF
    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(satellite, BRDFManager(config))

    # Generate
    lc_start = time.time()
    magnitudes, flux, _, _, distances_out, _ = generate_lightcurves(
        facet_lit_status_dict=lit_status_dict,
        k1_vectors_array=k1_vectors,
        k2_vectors_array=k2_vectors,
        observer_distances=observer_distances,
        satellite=satellite,
        epochs=epochs,
        brdf_calculator=brdf_calc,
        animate=False
    )
    lc_time = time.time() - lc_start

    print(f"  Light curve generation: {lc_time:.2f}s")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nMagnitude range: {magnitudes.min():.2f} to {magnitudes.max():.2f}")
    print(f"Mean magnitude: {magnitudes.mean():.2f}")

    # Calculate phase angles
    phase_angles = np.array([
        np.degrees(np.arccos(np.clip(np.dot(k1_vectors[i], k2_vectors[i]), -1, 1)))
        for i in range(len(epochs))
    ])
    print(f"Phase angle range: {phase_angles.min():.1f}° to {phase_angles.max():.1f}°")

    # Create plot
    print("\nSaving light curve plot...")
    time_hours = (epochs - epochs[0]) / 3600
    utc_times = [spice_handler.et_to_utc(e, "C", 0) for e in epochs]
    output_dir = config_manager.get_output_directory(config)

    create_light_curve_plot(
        time_hours=time_hours,
        epochs=epochs,
        magnitudes=magnitudes,
        phase_angles=phase_angles,
        utc_times=utc_times,
        satellite_name=satellite.name,
        output_dir=output_dir,
        observer_distances=distances_out,
        save=True,
        plot_mode='shadowed',
        no_plot=True  # Don't display, just save
    )

    print(f"Plot saved to: {output_dir}/<today's date>/")
    print("\nExample complete!")


if __name__ == "__main__":
    main()
