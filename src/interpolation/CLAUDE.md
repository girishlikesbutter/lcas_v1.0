# src/interpolation - Quaternion and Angle Interpolation

This module provides interpolation utilities for attitudes (quaternion SLERP) and articulation angles (step, linear, constant transitions).

## Module Overview

| File | Purpose |
|------|---------|
| `quaternion_interpolator.py` | SLERP, angle interpolation, time parsing |
| `attitude_interpolator.py` | High-level attitude interpolation helper |

## Key Exports

```python
from src.interpolation import (
    slerp,                        # Quaternion SLERP
    create_quaternion_interpolator,  # Build SLERP interpolator function
    create_angle_interpolator,    # Build angle interpolator (step/linear/constant)
    parse_keyframe_times,         # Convert UTC/ET times
    interpolate_attitudes_only    # Attitude-only interpolation
)
```

## Angle Interpolation

### create_angle_interpolator
Create angle arrays with complex transition patterns.

```python
from src.interpolation import create_angle_interpolator

angles = create_angle_interpolator(
    keyframe_times_utc=['2020-02-05T10:00:00', '2020-02-05T10:30:00', '2020-02-05T14:30:00'],
    keyframe_values=[5.5, 8.2, 13.5],    # Angles in degrees
    transitions=['linear', 'step'],       # Transition types between keyframes
    step_params=[None, 0.99],            # step_at fraction (0.99 = hold until 99%)
    epochs=epochs,                        # Target time array (ET)
    utc_to_et=spice_handler.utc_to_et    # Time conversion function
)
# Returns: np.ndarray(N,) of interpolated angles
```

### Transition Types
- `'linear'`: Linear interpolation between values
- `'step'`: Hold first value, jump to second at `step_at` fraction
- `'constant'`: Always return first value (ignore second)

### Use Case: Sun Tracking + Offset
```python
# Get base sun-tracking angles from behavior
sun_angles = calculate_angles_from_behaviors(satellite, k1_vectors, engine)['SP_North']

# Create time-varying offset
offset = create_angle_interpolator(
    keyframe_times_utc=[start_time, mid_time, end_time],
    keyframe_values=[0, 10, 5],
    transitions=['linear', 'step'],
    step_params=[None, 0.9],
    epochs=epochs,
    utc_to_et=spice_handler.utc_to_et
)

# Compose final angles
final_angles = sun_angles + offset
```

## Quaternion Interpolation

### slerp
Low-level Spherical Linear Interpolation.

```python
from src.interpolation import slerp
import quaternion

q1 = quaternion.quaternion(0.707, 0.707, 0, 0)
q2 = quaternion.quaternion(0.707, 0, 0.707, 0)

# Interpolate at t=0.5 (midpoint)
q_mid = slerp(q1, q2, t=0.5)  # Returns [w, x, y, z] array
```

### create_quaternion_interpolator
Build interpolator function for attitude quaternions.

```python
from src.interpolation import create_quaternion_interpolator

interpolator = create_quaternion_interpolator(
    sample_times=['2020-02-05T10:00:00', '2020-02-05T16:00:00'],
    sample_quats=[quat1, quat2],
    time_format='utc',
    spice_handler=spice_handler
)

# Interpolate at any time
quat_at_t = interpolator(et_time)        # Single time
quats = interpolator(epochs)              # Array of times
```

## Attitude Interpolation

### interpolate_attitudes_only
High-level helper for custom attitude interpolation in observation geometry.

```python
from src.interpolation import interpolate_attitudes_only

attitude_keyframes = {
    'times': ['2020-02-05T10:00:00', '2020-02-05T16:00:00'],
    'time_format': 'utc',
    'attitudes': [quat1, quat2],
    'format': 'quaternion'  # or 'matrix'
}

att_matrices = interpolate_attitudes_only(
    attitude_keyframes=attitude_keyframes,
    epochs=epochs,
    spice_handler=spice_handler
)
# Returns: np.ndarray(N, 3, 3) rotation matrices
```

### Supported Formats
- `'quaternion'`: numpy-quaternion objects or [w,x,y,z] arrays
- `'matrix'`: 3x3 rotation matrices (converted to quaternions internally)

## Time Parsing

```python
from src.interpolation import parse_keyframe_times

# UTC to ET
times_et = parse_keyframe_times(
    times=['2020-02-05T10:00:00', '2020-02-05T16:00:00'],
    time_format='utc',
    spice_handler=spice_handler
)

# Already ET
times_et = parse_keyframe_times(
    times=[634168800.0, 634190400.0],
    time_format='et',
    spice_handler=None
)
```

## Usage in Pipeline

### Custom Attitude in Observation Geometry
```python
from src.computation.observation_geometry import compute_observation_geometry

attitude_keyframes = {
    'times': [start_utc, end_utc],
    'time_format': 'utc',
    'attitudes': [quaternion.quaternion(0.707, 0.707, 0, 0),
                  quaternion.quaternion(0.707, 0, 0.707, 0)],
    'format': 'quaternion'
}

geometry_data = compute_observation_geometry(
    epochs, satellite_id, spice_handler, config,
    attitude_keyframes=attitude_keyframes  # Triggers SLERP
)
```

### Explicit Angle Control
```python
from src.interpolation import create_angle_interpolator
from src.articulation import compute_rotation_matrices_from_angles

# Create angles for each component
sp_angles = create_angle_interpolator(...)
ad_angles = create_angle_interpolator(...)

explicit_component_angles = {
    'SP_North': sp_angles,
    'SP_South': sp_angles,
    'AD_East': ad_angles,
    'AD_West': ad_angles
}

component_matrices = compute_rotation_matrices_from_angles(
    explicit_component_angles, satellite
)
```

## Cross-References

- **Observation geometry**: `src/computation/observation_geometry.py` uses attitude interpolation
- **Articulation**: Angle arrays used by `src/articulation/articulation_engine.py`
- **SPICE**: `spice_handler.utc_to_et` for time conversion
