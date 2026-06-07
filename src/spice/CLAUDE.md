# src/spice - SPICE Orbital Mechanics Integration

This module wraps NASA's SPICE toolkit (via spiceypy) for orbital mechanics calculations.

## Module Overview

| File | Purpose |
|------|---------|
| `spice_handler.py` | SpiceHandler class for kernel loading and SPICE operations |

## SpiceHandler Class

### Initialization and Kernel Loading

```python
from src.spice.spice_handler import SpiceHandler

spice_handler = SpiceHandler()

# Load metakernel (recommended)
spice_handler.load_metakernel_programmatically(str(metakernel_path))

# Or load with explicit project root (for v2 metakernels)
spice_handler.load_metakernel(metakernel_path, project_root)

# Load individual kernels
spice_handler.load_kernel("path/to/kernel.bsp")
```

### Time Conversions

```python
# UTC to Ephemeris Time (ET)
et = spice_handler.utc_to_et("2020-02-05T10:00:00")

# ET to UTC
utc_str = spice_handler.et_to_utc(et, "C", 0)      # Calendar format
utc_iso = spice_handler.et_to_utc(et, "ISOC", 3)  # ISO format with 3 decimal places
```

### Position Calculations

```python
# Get body position relative to observer
position, light_time = spice_handler.get_body_position(
    target="SUN",           # Target body (name or ID)
    et=epoch,               # Ephemeris time
    frame="J2000",          # Reference frame
    observer="EARTH",       # Observer body
    aberration_correction="NONE"  # or "LT+S"
)
# position: np.ndarray [x, y, z] in km
# light_time: float in seconds
```

### Orientation/Attitude

```python
# Get rotation matrix from one frame to another
rotation_matrix = spice_handler.get_target_orientation(
    from_frame="J2000",
    to_frame="IS901_BUS_FRAME",
    et=epoch
)
# rotation_matrix: 3x3 np.ndarray
```

### Frame Information

```python
# Frame name from ID
frame_name = spice_handler.get_frame_name_from_id(-126824)

# Frame ID from name
frame_id = spice_handler.get_frame_id_from_name("IS901_BUS_FRAME")

# Detailed frame info
name, center, frclass, clssid_list = spice_handler.get_frame_info_by_id(frame_id)
```

### Kernel Management

```python
# Unload specific kernel
spice_handler.unload_kernel("path/to/kernel.bsp")

# Unload all kernels
spice_handler.unload_all_kernels()

# Context manager (auto-unloads on exit)
with SpiceHandler() as sh:
    sh.load_metakernel_programmatically(metakernel_path)
    # ... use spice ...
# Kernels automatically unloaded
```

## SPICE Concepts

### Bodies and IDs
- Sun: "SUN" or "10"
- Earth: "EARTH" or "399"
- Satellites: Negative integers (e.g., -126824 for Intelsat 901)
- Ground stations: Custom IDs (e.g., 399999 for DST observatory)

### Reference Frames
- **J2000**: Earth-centered inertial frame (standard for SPICE)
- **Body frames**: Satellite-fixed frames (e.g., "IS901_BUS_FRAME")

### Time Systems
- **UTC**: Human-readable time strings
- **ET (Ephemeris Time)**: Seconds past J2000 epoch, used internally

## Usage in Pipeline

```python
from src.spice.spice_handler import SpiceHandler

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))

# Generate time array
start_et = spice_handler.utc_to_et(start_time_utc)
end_et = spice_handler.utc_to_et(end_time_utc)
epochs = np.linspace(start_et, end_et, num_points)

# Used by observation_geometry.py for:
# - Sun position: get_body_position("SUN", et, "J2000", "EARTH")
# - Satellite position: get_body_position(satellite_id, et, "J2000", "EARTH")
# - Observer position: get_body_position(observer_id, et, "J2000", "EARTH")
# - Attitude: get_target_orientation("J2000", body_frame, et)
```

## Cross-References

- **Observation geometry**: `src/computation/observation_geometry.py` uses SpiceHandler
- **Metakernels**: Located in `data/spice_kernels/missions/`
- **Config**: SPICE settings in config's `spice_config` section
