#!/usr/bin/env python3
"""
LCAS Example 3: Custom Angle Profiles
=====================================

This script demonstrates the create_angle_interpolator function for
explicit control over component articulation angles.

Shows three transition types:
- constant: Hold a fixed value
- linear: Smooth interpolation between keyframes
- step: Discrete jumps at specified times

Run from project root:
    python examples/03_custom_angle_profiles.py

Expected output:
    - Console output showing angle profiles
    - Demonstration of different interpolation methods

Time to run: ~1 minute
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
from src.articulation import compute_rotation_matrices_from_angles
from src.interpolation import create_angle_interpolator


def main():
    print("=" * 60)
    print("LCAS Example 3: Custom Angle Profiles")
    print("=" * 60)

    # =========================================================================
    # SETUP
    # =========================================================================
    print("\n[Setup] Loading configuration and initializing SPICE...")

    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
    satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

    spice_handler = SpiceHandler()
    metakernel_path = config_manager.get_metakernel_path(config)
    spice_handler.load_metakernel_programmatically(str(metakernel_path))

    # Time setup - use 30 points to see smooth transitions
    num_points = 30
    start_time_utc = config.simulation_defaults.start_time
    end_time_utc = config.simulation_defaults.end_time

    start_et = spice_handler.utc_to_et(start_time_utc)
    end_et = spice_handler.utc_to_et(end_time_utc)
    epochs = np.linspace(start_et, end_et, num_points)
    time_hours = (epochs - epochs[0]) / 3600

    print(f"  Satellite: {config.name}")
    print(f"  Time points: {num_points}")
    print(f"  Duration: {time_hours[-1]:.1f} hours")

    # =========================================================================
    # EXAMPLE 1: Constant Angle (Fixed Position)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Constant Angle")
    print("=" * 60)

    # Solar panels fixed at 45 degrees
    constant_angle = create_angle_interpolator(
        keyframe_times_utc=[start_time_utc, end_time_utc],
        keyframe_values=[45.0, 45.0],
        transitions=['constant'],
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )

    print("\n  Constant angle at 45°:")
    print(f"  First value:  {constant_angle[0]:.1f}°")
    print(f"  Middle value: {constant_angle[num_points//2]:.1f}°")
    print(f"  Last value:   {constant_angle[-1]:.1f}°")
    print("  → All values are identical (constant)")

    # =========================================================================
    # EXAMPLE 2: Linear Interpolation (Smooth Rotation)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Linear Interpolation")
    print("=" * 60)

    # Solar panels rotate from 0° to 90° smoothly
    linear_angle = create_angle_interpolator(
        keyframe_times_utc=[start_time_utc, end_time_utc],
        keyframe_values=[0.0, 90.0],
        transitions=['linear'],
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )

    print("\n  Linear interpolation from 0° to 90°:")
    print(f"  First value:  {linear_angle[0]:.1f}°")
    print(f"  Middle value: {linear_angle[num_points//2]:.1f}°")
    print(f"  Last value:   {linear_angle[-1]:.1f}°")
    print("  → Smooth, continuous change")

    # =========================================================================
    # EXAMPLE 3: Step Transition (Discrete Jump)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Step Transition")
    print("=" * 60)

    # Panel jumps from 0° to 60° at the midpoint
    mid_time = spice_handler.et_to_utc((start_et + end_et) / 2, "ISOC", 0)

    step_angle = create_angle_interpolator(
        keyframe_times_utc=[start_time_utc, mid_time, end_time_utc],
        keyframe_values=[0.0, 60.0, 60.0],
        transitions=['step', 'constant'],
        step_params=[0.5, None],  # Step at 50% through first interval
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )

    print("\n  Step transition: 0° → 60° at midpoint:")
    print(f"  t=0h:    {step_angle[0]:.1f}°")
    print(f"  t={time_hours[num_points//4]:.1f}h:  {step_angle[num_points//4]:.1f}° (before step)")
    print(f"  t={time_hours[num_points//2]:.1f}h:  {step_angle[num_points//2]:.1f}° (after step)")
    print(f"  t={time_hours[-1]:.1f}h:  {step_angle[-1]:.1f}°")
    print("  → Instant jump, then holds")

    # =========================================================================
    # EXAMPLE 4: Multi-Segment Profile
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multi-Segment Profile")
    print("=" * 60)

    # Complex profile: ramp up, hold, ramp down
    # Time: 0h → 2h → 4h → 6h
    duration_hours = time_hours[-1]
    t1 = spice_handler.et_to_utc(start_et + (end_et - start_et) * 0.33, "ISOC", 0)
    t2 = spice_handler.et_to_utc(start_et + (end_et - start_et) * 0.67, "ISOC", 0)

    complex_angle = create_angle_interpolator(
        keyframe_times_utc=[start_time_utc, t1, t2, end_time_utc],
        keyframe_values=[0.0, 60.0, 60.0, -30.0],
        transitions=['linear', 'constant', 'linear'],
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )

    print("\n  Complex profile: 0° → 60° → hold → -30°")
    print("  Segment 1: Linear ramp 0° to 60°")
    print("  Segment 2: Hold at 60°")
    print("  Segment 3: Linear ramp 60° to -30°")
    print("\n  Values at key times:")
    for i in [0, num_points//3, 2*num_points//3, num_points-1]:
        print(f"    t={time_hours[i]:.1f}h: {complex_angle[i]:.1f}°")

    # =========================================================================
    # EXAMPLE 5: Composing Angles (Sun-Tracking + Offset)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Composing Angles")
    print("=" * 60)

    # Base angle: simple linear rotation
    base_angle = create_angle_interpolator(
        keyframe_times_utc=[start_time_utc, end_time_utc],
        keyframe_values=[0.0, 45.0],
        transitions=['linear'],
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )

    # Offset: sinusoidal variation (simulating oscillation)
    offset_angle = 10.0 * np.sin(2 * np.pi * time_hours / duration_hours)

    # Combined angle
    combined_angle = base_angle + offset_angle

    print("\n  Base angle (linear 0° → 45°) + offset (±10° sine wave)")
    print("\n  Values at key times:")
    for i in [0, num_points//4, num_points//2, 3*num_points//4, num_points-1]:
        print(f"    t={time_hours[i]:.1f}h: base={base_angle[i]:.1f}° + "
              f"offset={offset_angle[i]:.1f}° = {combined_angle[i]:.1f}°")

    # =========================================================================
    # RUNNING A LIGHT CURVE WITH CUSTOM ANGLES
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING LIGHT CURVE WITH CUSTOM ANGLES")
    print("=" * 60)

    # Use the complex angle profile for the solar panels
    explicit_component_angles = {
        'SP_North': complex_angle,
        'SP_South': complex_angle
    }

    # Compute geometry
    geometry_data = compute_observation_geometry(
        epochs=epochs,
        satellite_id=config.spice_config.satellite_id,
        observer_id=399999,
        spice_handler=spice_handler,
        config=config
    )

    # Convert angles to matrices
    explicit_component_matrices = compute_rotation_matrices_from_angles(
        explicit_component_angles,
        satellite
    )

    # Compute shadows
    print("\n  Computing shadows...")
    lit_status_dict = compute_shadows(
        satellite=satellite,
        k1_vectors=geometry_data['k1_vectors'],
        explicit_component_angles=explicit_component_angles,
        explicit_component_matrices=explicit_component_matrices
    )

    # Generate light curve
    print("  Generating light curve...")
    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(satellite, BRDFManager(config))

    magnitudes, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit_status_dict,
        k1_vectors_array=geometry_data['k1_vectors'],
        k2_vectors_array=geometry_data['k2_vectors'],
        observer_distances=geometry_data['observer_distances'],
        satellite=satellite,
        epochs=epochs,
        brdf_calculator=brdf_calc,
        pre_computed_matrices=explicit_component_matrices,
        animate=False
    )

    print(f"\n  Magnitude range: {magnitudes.min():.2f} to {magnitudes.max():.2f}")
    print("\n  The light curve reflects the custom angle profile,")
    print("  showing how articulation affects brightness.")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. 'constant' - holds a fixed angle")
    print("  2. 'linear' - smooth interpolation between values")
    print("  3. 'step' - discrete jumps at specified times")
    print("  4. Angles can be composed (added together)")
    print("  5. Use numpy for complex patterns (sine, etc.)")


if __name__ == "__main__":
    main()
