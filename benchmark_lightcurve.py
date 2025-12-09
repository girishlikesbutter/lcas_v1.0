#!/usr/bin/env python3
"""
LCAS Light Curve Benchmark Script
=================================

Generates the Intelsat 901 shadowed light curve as fast as possible.
Used to benchmark the vectorized shadow ray batching and BRDF computation.

Usage:
    python benchmark_lightcurve.py [num_points]

    num_points: Number of time points (default: 100)

Example:
    python benchmark_lightcurve.py 200
"""

import sys
import time
from pathlib import Path

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def benchmark_lightcurve(num_points: int = 100, show_progress: bool = True):
    """
    Generate Intelsat 901 shadowed light curve and report timing.

    Args:
        num_points: Number of time epochs
        show_progress: Show progress bars during computation

    Returns:
        Dict with timing breakdown and results
    """
    timings = {}
    total_start = time.time()

    # =========================================================================
    # PHASE 1: Configuration and Model Loading
    # =========================================================================
    phase1_start = time.time()

    from src.config.rso_config_manager import RSO_ConfigManager
    from src.io.stl_loader import STLLoader
    from src.computation.brdf import BRDFManager, BRDFCalculator

    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")

    satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)
    total_facets = sum(len(c.facets) for c in satellite.components if c.facets)

    brdf_manager = BRDFManager(config)
    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

    timings['model_loading'] = time.time() - phase1_start

    # =========================================================================
    # PHASE 2: SPICE Initialization
    # =========================================================================
    phase2_start = time.time()

    from src.spice.spice_handler import SpiceHandler

    metakernel_path = config_manager.get_metakernel_path(config)
    spice_handler = SpiceHandler()
    spice_handler.load_metakernel_programmatically(str(metakernel_path))

    # Generate epochs
    start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
    end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
    epochs = np.linspace(start_et, end_et, num_points)

    timings['spice_init'] = time.time() - phase2_start

    # =========================================================================
    # PHASE 3: Observation Geometry
    # =========================================================================
    phase3_start = time.time()

    from src.computation.observation_geometry import compute_observation_geometry

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

    timings['geometry'] = time.time() - phase3_start

    # =========================================================================
    # PHASE 4: Articulation (angles + matrices)
    # =========================================================================
    phase4_start = time.time()

    from src.articulation import (
        ArticulationEngine, SunTrackingBehavior,
        calculate_angles_from_behaviors, compute_rotation_matrices_from_angles
    )
    from src.interpolation import create_angle_interpolator

    # Create articulation engine with sun tracking
    articulation_engine = ArticulationEngine()
    sp_cap = config.articulation_capabilities['SP_North']

    sun_tracking = SunTrackingBehavior({
        'rotation_center': sp_cap.rotation_center,
        'rotation_axis': sp_cap.rotation_axis,
        'reference_normal': [1.0, 0.0, 0.0],
        'limits': sp_cap.limits
    })
    articulation_engine.register_component_behavior('SP_North', sun_tracking)
    articulation_engine.register_component_behavior('SP_South', sun_tracking)

    # Get sun-tracking angles
    sun_angles_dict = calculate_angles_from_behaviors(
        satellite=satellite,
        k1_vectors=k1_vectors,
        articulation_engine=articulation_engine,
        articulation_offset=0
    )

    # Create offset angles (EXACTLY as in notebook 3)
    offset_angles = create_angle_interpolator(
        keyframe_times_utc=[
            '2020-02-05T10:00:00',
            '2020-02-05T10:30:00',
            '2020-02-05T14:30:00'
        ],
        keyframe_values=[5.5, 8.2, 13.5],
        transitions=['linear', 'step'],
        step_params=[None, 0.99],
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )

    sp_angles = sun_angles_dict['SP_North'] + offset_angles

    # Fixed dish angles (same as notebook)
    dish_angle = 15
    ad_angles = create_angle_interpolator(
        keyframe_times_utc=[
            config.simulation_defaults.start_time,
            config.simulation_defaults.end_time
        ],
        keyframe_values=[dish_angle, dish_angle],
        transitions=['constant'],
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )

    # Build explicit angles dict
    explicit_component_angles = {
        'SP_North': sp_angles,
        'SP_South': sp_angles,
        'AD_West': ad_angles,
        'AD_East': ad_angles
    }

    # Convert to rotation matrices
    explicit_component_matrices = compute_rotation_matrices_from_angles(
        explicit_component_angles, satellite
    )

    timings['articulation'] = time.time() - phase4_start

    # =========================================================================
    # PHASE 5: Shadow Computation (MAIN BENCHMARK TARGET)
    # =========================================================================
    phase5_start = time.time()

    from src.computation.shadow_engine import compute_shadows

    lit_status_dict = compute_shadows(
        satellite=satellite,
        k1_vectors=k1_vectors,
        explicit_component_angles=explicit_component_angles,
        explicit_component_matrices=explicit_component_matrices,
        show_progress=show_progress
    )

    timings['shadows'] = time.time() - phase5_start

    # =========================================================================
    # PHASE 6: Light Curve Generation
    # =========================================================================
    phase6_start = time.time()

    from src.computation import generate_lightcurves

    magnitudes, total_flux, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit_status_dict,
        k1_vectors_array=k1_vectors,
        k2_vectors_array=k2_vectors,
        observer_distances=observer_distances,
        satellite=satellite,
        epochs=epochs,
        brdf_calculator=brdf_calc,
        pre_computed_matrices=explicit_component_matrices,
        generate_no_shadow=False,
        animate=False,  # No animation for speed
        show_progress=show_progress
    )

    timings['lightcurve'] = time.time() - phase6_start

    # =========================================================================
    # PHASE 7: Plot Generation and Save
    # =========================================================================
    phase7_start = time.time()

    from src.visualization.lightcurve_plotter import create_light_curve_plot
    import matplotlib.pyplot as plt

    # Calculate time array and phase angles
    time_hours = (epochs - epochs[0]) / 3600.0
    utc_times = [spice_handler.et_to_utc(epoch, "C", 0) for epoch in epochs]

    phase_angles = np.zeros(num_points)
    for i in range(num_points):
        cos_phase = np.dot(k1_vectors[i], k2_vectors[i])
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase_angles[i] = np.degrees(np.arccos(cos_phase))

    output_dir = config_manager.get_output_directory(config)

    # Create and save the plot (no_plot=True to avoid blocking)
    create_light_curve_plot(
        time_hours=time_hours,
        epochs=epochs,
        magnitudes=magnitudes,
        phase_angles=phase_angles,
        utc_times=utc_times,
        satellite_name=satellite.name,
        plot_mode="single",
        output_dir=output_dir,
        observer_distances=observer_distances,
        no_plot=True,  # Don't display yet - just create and save
        save=True
    )

    timings['plotting'] = time.time() - phase7_start

    # Display plot AFTER timing ends (blocking call)
    plt.show()

    # =========================================================================
    # TOTAL
    # =========================================================================
    timings['total'] = time.time() - total_start

    return {
        'timings': timings,
        'magnitudes': magnitudes,
        'total_flux': total_flux,
        'num_points': num_points,
        'total_facets': total_facets,
        'total_rays': num_points * total_facets
    }


def print_results(results: dict):
    """Print benchmark results in a nice format."""
    timings = results['timings']

    print()
    print("=" * 70)
    print("LCAS LIGHT CURVE BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Time points:   {results['num_points']:,}")
    print(f"  Total facets:  {results['total_facets']:,}")
    print(f"  Total rays:    {results['total_rays']:,}")
    print()
    print("Timing Breakdown:")
    print("-" * 50)
    print(f"  Model loading:     {timings['model_loading']:>8.3f}s")
    print(f"  SPICE init:        {timings['spice_init']:>8.3f}s")
    print(f"  Geometry:          {timings['geometry']:>8.3f}s")
    print(f"  Articulation:      {timings['articulation']:>8.3f}s")
    print(f"  Shadow rays:       {timings['shadows']:>8.3f}s  <-- Main benchmark")
    print(f"  Light curve:       {timings['lightcurve']:>8.3f}s")
    print(f"  Plotting:          {timings['plotting']:>8.3f}s")
    print("-" * 50)
    print(f"  TOTAL:             {timings['total']:>8.3f}s")
    print()

    # Performance metrics
    rays_per_sec = results['total_rays'] / timings['shadows']
    facets_per_sec = results['total_facets'] / (timings['shadows'] / results['num_points'])

    print("Performance Metrics:")
    print(f"  Shadow rays/sec:   {rays_per_sec:,.0f}")
    print(f"  Facets/epoch:      {facets_per_sec:,.0f}")
    print(f"  Time per epoch:    {timings['shadows'] / results['num_points'] * 1000:.1f}ms")
    print()

    # Magnitude range
    valid_mags = results['magnitudes'][np.isfinite(results['magnitudes'])]
    if len(valid_mags) > 0:
        print(f"Light Curve Results:")
        print(f"  Magnitude range:   {valid_mags.min():.2f} to {valid_mags.max():.2f}")
        print(f"  Mean magnitude:    {valid_mags.mean():.2f}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    # Parse command line args
    num_points = 50
    if len(sys.argv) > 1:
        try:
            num_points = int(sys.argv[1])
        except ValueError:
            print(f"Invalid num_points: {sys.argv[1]}, using default 100")

    print(f"Running benchmark with {num_points} time points...")
    print()

    results = benchmark_lightcurve(num_points=num_points, show_progress=True)
    print_results(results)
