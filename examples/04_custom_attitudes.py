#!/usr/bin/env python3
"""
LCAS Example 4: Custom Quaternion Attitudes
===========================================

This script demonstrates how to define custom satellite attitudes using
quaternion keyframes instead of SPICE attitude kernels.

Useful for:
- Prototyping without real attitude data
- What-if scenarios
- Sensitivity analysis
- Testing different pointing strategies

Run from project root:
    python examples/04_custom_attitudes.py

Expected output:
    - Light curves comparing different attitude modes
    - Demonstration of SLERP interpolation

Time to run: ~1-2 minutes
"""

import sys
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import quaternion  # numpy-quaternion library

# LCAS imports
from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import generate_lightcurves
from src.computation.shadow_engine import compute_shadows
from src.computation.brdf import BRDFCalculator, BRDFManager


def main():
    print("=" * 60)
    print("LCAS Example 4: Custom Quaternion Attitudes")
    print("=" * 60)

    # =========================================================================
    # SETUP
    # =========================================================================
    print("\n[Setup] Loading configuration...")

    config_manager = RSO_ConfigManager(PROJECT_ROOT)
    config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
    satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

    spice_handler = SpiceHandler()
    metakernel_path = config_manager.get_metakernel_path(config)
    spice_handler.load_metakernel_programmatically(str(metakernel_path))

    # Time setup
    num_points = 20
    start_time_utc = config.simulation_defaults.start_time
    end_time_utc = config.simulation_defaults.end_time

    start_et = spice_handler.utc_to_et(start_time_utc)
    end_et = spice_handler.utc_to_et(end_time_utc)
    epochs = np.linspace(start_et, end_et, num_points)
    time_hours = (epochs - epochs[0]) / 3600

    print(f"  Satellite: {config.name}")
    print(f"  Duration: {time_hours[-1]:.1f} hours")

    # Setup BRDF
    brdf_calc = BRDFCalculator()
    brdf_calc.update_satellite_brdf_with_manager(satellite, BRDFManager(config))

    # =========================================================================
    # UNDERSTANDING QUATERNIONS
    # =========================================================================
    print("\n" + "=" * 60)
    print("UNDERSTANDING QUATERNIONS")
    print("=" * 60)

    print("""
    Quaternions represent 3D rotations using 4 numbers: (w, x, y, z)

    - w is the scalar (real) part
    - x, y, z are the vector (imaginary) parts
    - Unit quaternions (||q|| = 1) represent rotations

    Common quaternions:
    - (1, 0, 0, 0): Identity (no rotation)
    - (0.707, 0.707, 0, 0): 90° rotation about X axis
    - (0.707, 0, 0.707, 0): 90° rotation about Y axis
    - (0.707, 0, 0, 0.707): 90° rotation about Z axis

    LCAS uses scalar-first format: quaternion.quaternion(w, x, y, z)
    """)

    # =========================================================================
    # EXAMPLE 1: Fixed Attitude (Identity)
    # =========================================================================
    print("=" * 60)
    print("EXAMPLE 1: Fixed Attitude (No Rotation)")
    print("=" * 60)

    # Identity quaternion - satellite maintains initial orientation
    identity_quat = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

    attitude_keyframes_fixed = {
        'times': [start_time_utc, end_time_utc],
        'time_format': 'utc',
        'attitudes': [identity_quat, identity_quat],
        'format': 'quaternion'
    }

    print("\n  Attitude: Identity (no rotation)")
    print(f"  Start quaternion: {identity_quat}")
    print(f"  End quaternion:   {identity_quat}")

    # Compute geometry with custom attitude
    geometry_fixed = compute_observation_geometry(
        epochs=epochs,
        satellite_id=config.spice_config.satellite_id,
        observer_id=399999,
        spice_handler=spice_handler,
        config=config,
        attitude_keyframes=attitude_keyframes_fixed
    )

    # Generate light curve
    lit_status_fixed = compute_shadows(satellite, geometry_fixed['k1_vectors'])
    magnitudes_fixed, *_ = generate_lightcurves(
        lit_status_fixed, geometry_fixed['k1_vectors'], geometry_fixed['k2_vectors'],
        geometry_fixed['observer_distances'], satellite, epochs, brdf_calc, animate=False
    )

    print(f"\n  Magnitude range: {magnitudes_fixed.min():.2f} to {magnitudes_fixed.max():.2f}")

    # =========================================================================
    # EXAMPLE 2: Rotating Attitude (SLERP)
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Rotating Attitude (90° about Z)")
    print("=" * 60)

    # Start at identity, end rotated 90° about Z axis
    start_quat = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    end_quat = quaternion.quaternion(0.707, 0.0, 0.0, 0.707)  # 90° about Z

    attitude_keyframes_rotating = {
        'times': [start_time_utc, end_time_utc],
        'time_format': 'utc',
        'attitudes': [start_quat, end_quat],
        'format': 'quaternion'
    }

    print("\n  SLERP interpolation between:")
    print(f"    Start: {start_quat} (identity)")
    print(f"    End:   {end_quat} (90° about Z)")
    print("\n  SLERP provides constant angular velocity rotation")

    geometry_rotating = compute_observation_geometry(
        epochs=epochs,
        satellite_id=config.spice_config.satellite_id,
        observer_id=399999,
        spice_handler=spice_handler,
        config=config,
        attitude_keyframes=attitude_keyframes_rotating
    )

    lit_status_rotating = compute_shadows(satellite, geometry_rotating['k1_vectors'])
    magnitudes_rotating, *_ = generate_lightcurves(
        lit_status_rotating, geometry_rotating['k1_vectors'], geometry_rotating['k2_vectors'],
        geometry_rotating['observer_distances'], satellite, epochs, brdf_calc, animate=False
    )

    print(f"\n  Magnitude range: {magnitudes_rotating.min():.2f} to {magnitudes_rotating.max():.2f}")

    # =========================================================================
    # EXAMPLE 3: Multi-Keyframe Attitude Sequence
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multi-Keyframe Attitude Sequence")
    print("=" * 60)

    # Create intermediate times
    mid_time = spice_handler.et_to_utc((start_et + end_et) / 2, "ISOC", 0)

    # Sequence: Identity → 45° X → 90° X
    q0 = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)            # Identity
    q1 = quaternion.quaternion(0.924, 0.383, 0.0, 0.0)        # 45° about X
    q2 = quaternion.quaternion(0.707, 0.707, 0.0, 0.0)        # 90° about X

    attitude_keyframes_sequence = {
        'times': [start_time_utc, mid_time, end_time_utc],
        'time_format': 'utc',
        'attitudes': [q0, q1, q2],
        'format': 'quaternion'
    }

    print("\n  Three-keyframe sequence:")
    print(f"    t=0h:   {q0} (identity)")
    print(f"    t=3h:   {q1} (45° about X)")
    print(f"    t=6h:   {q2} (90° about X)")

    geometry_sequence = compute_observation_geometry(
        epochs=epochs,
        satellite_id=config.spice_config.satellite_id,
        observer_id=399999,
        spice_handler=spice_handler,
        config=config,
        attitude_keyframes=attitude_keyframes_sequence
    )

    lit_status_sequence = compute_shadows(satellite, geometry_sequence['k1_vectors'])
    magnitudes_sequence, *_ = generate_lightcurves(
        lit_status_sequence, geometry_sequence['k1_vectors'], geometry_sequence['k2_vectors'],
        geometry_sequence['observer_distances'], satellite, epochs, brdf_calc, animate=False
    )

    print(f"\n  Magnitude range: {magnitudes_sequence.min():.2f} to {magnitudes_sequence.max():.2f}")

    # =========================================================================
    # COMPARISON: SPICE vs Custom Attitudes
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON: SPICE vs Custom Attitudes")
    print("=" * 60)

    # Generate with SPICE attitudes (no attitude_keyframes)
    geometry_spice = compute_observation_geometry(
        epochs=epochs,
        satellite_id=config.spice_config.satellite_id,
        observer_id=399999,
        spice_handler=spice_handler,
        config=config
        # No attitude_keyframes = use SPICE
    )

    lit_status_spice = compute_shadows(satellite, geometry_spice['k1_vectors'])
    magnitudes_spice, *_ = generate_lightcurves(
        lit_status_spice, geometry_spice['k1_vectors'], geometry_spice['k2_vectors'],
        geometry_spice['observer_distances'], satellite, epochs, brdf_calc, animate=False
    )

    print("\n  Magnitude Summary:")
    print("  " + "-" * 50)
    print(f"  {'Mode':<25} {'Min':>8} {'Max':>8} {'Range':>8}")
    print("  " + "-" * 50)
    print(f"  {'SPICE attitudes':<25} {magnitudes_spice.min():>8.2f} {magnitudes_spice.max():>8.2f} "
          f"{magnitudes_spice.max() - magnitudes_spice.min():>8.2f}")
    print(f"  {'Fixed (identity)':<25} {magnitudes_fixed.min():>8.2f} {magnitudes_fixed.max():>8.2f} "
          f"{magnitudes_fixed.max() - magnitudes_fixed.min():>8.2f}")
    print(f"  {'Rotating (90° Z)':<25} {magnitudes_rotating.min():>8.2f} {magnitudes_rotating.max():>8.2f} "
          f"{magnitudes_rotating.max() - magnitudes_rotating.min():>8.2f}")
    print(f"  {'Sequence (X rotation)':<25} {magnitudes_sequence.min():>8.2f} {magnitudes_sequence.max():>8.2f} "
          f"{magnitudes_sequence.max() - magnitudes_sequence.min():>8.2f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("""
Key takeaways:

1. Custom attitudes override SPICE kernels
2. Quaternions use scalar-first format: (w, x, y, z)
3. SLERP interpolation ensures smooth rotation
4. Multiple keyframes create complex attitude sequences
5. Different attitudes significantly affect brightness

Common quaternion rotations:
  - 90° about X: (0.707, 0.707, 0, 0)
  - 90° about Y: (0.707, 0, 0.707, 0)
  - 90° about Z: (0.707, 0, 0, 0.707)
  - 180° about Z: (0, 0, 0, 1)
""")


if __name__ == "__main__":
    main()
