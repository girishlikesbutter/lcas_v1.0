# src/articulation - Component Movement and Transformation

This module handles articulation (rotation) of satellite components like solar panels and antenna dishes.

## Module Overview

| File | Purpose |
|------|---------|
| `articulation_engine.py` | Engine class + batch angle/matrix computation functions |
| `behaviors/` | Behavior implementations (sun tracking, etc.) |

## Key Exports

```python
from src.articulation import (
    ArticulationEngine,
    SunTrackingBehavior,
    calculate_angles_from_behaviors,
    compute_rotation_matrices_from_angles,
)
```

## ArticulationEngine Class

Manages component behavior registration and lookup.

```python
from src.articulation import ArticulationEngine, SunTrackingBehavior

# Create engine
engine = ArticulationEngine()

# Register behavior for a component
sun_tracking = SunTrackingBehavior({
    'rotation_center': [0.0, 5.0, 0.0],
    'rotation_axis': [0.0, 1.0, 0.0],
    'reference_normal': [1.0, 0.0, 0.0],
    'limits': {'min_angle': -180.0, 'max_angle': 180.0}
})
engine.register_component_behavior('SP_North', sun_tracking)
engine.register_component_behavior('SP_South', sun_tracking)

# Check if component is articulated
is_articulated = engine.is_component_articulated(component)

# Get behavior for component
behavior = engine.get_component_behavior(component)
```

## Workflow Functions

### calculate_angles_from_behaviors
Compute angles from registered behaviors for all epochs.

```python
from src.articulation import calculate_angles_from_behaviors

component_angles = calculate_angles_from_behaviors(
    satellite=satellite,
    k1_vectors=k1_vectors_array,    # (N, 3) sun vectors in body frame
    articulation_engine=engine,
    articulation_offset=0.0          # Additional offset in degrees
)
# Returns: Dict[component_name, np.ndarray(N,)] angles in degrees
```

### compute_rotation_matrices_from_angles
Convert angle arrays to rotation matrices.

```python
from src.articulation import compute_rotation_matrices_from_angles

component_matrices = compute_rotation_matrices_from_angles(
    component_angles=component_angles,  # Dict[name, angles_array]
    satellite=satellite,
    articulation_engine=engine          # Optional, for fallback
)
# Returns: Dict[component_name, np.ndarray(N, 4, 4)] rotation matrices
```

## Complete Workflow

### Using Behaviors (automatic angle calculation)

```python
# 1. Create engine and register behaviors
engine = ArticulationEngine()
engine.register_component_behavior('SP_North', sun_tracking)
engine.register_component_behavior('SP_South', sun_tracking)

# 2. Calculate angles from behaviors
component_angles = calculate_angles_from_behaviors(
    satellite, k1_vectors, engine
)

# 3. Convert to matrices
component_matrices = compute_rotation_matrices_from_angles(
    component_angles, satellite
)

# 4. Use in shadow computation and light curve generation
lit_status = compute_shadows(
    satellite, k1_vectors,
    explicit_component_matrices=component_matrices
)
```

### Using Explicit Angles (manual control)

```python
from src.interpolation import create_angle_interpolator

# 1. Create angle arrays manually
sp_angles = create_angle_interpolator(
    keyframe_times_utc=['...', '...'],
    keyframe_values=[0, 45],
    transitions=['linear'],
    epochs=epochs,
    utc_to_et=spice_handler.utc_to_et
)

# 2. Build explicit angles dict
explicit_component_angles = {
    'SP_North': sp_angles,
    'SP_South': sp_angles,
    'AD_East': ad_angles,
    'AD_West': ad_angles
}

# 3. Convert to matrices (no engine needed)
component_matrices = compute_rotation_matrices_from_angles(
    explicit_component_angles, satellite
)
```

## Rotation Axis Source

The rotation axis for matrix computation comes from:
1. `component.articulation_parameters.rotation_axis` (preferred)
2. Fallback: `behavior.rotation_axis` via engine

## Cross-References

- **Behaviors**: See `src/articulation/behaviors/CLAUDE.md`
- **Geometry utils**: `src/utils/geometry_utils.py` provides `build_rotation_matrix`
- **Shadow computation**: `src/computation/shadow_engine.py` uses pre-computed matrices
- **Config**: Articulation capabilities defined in YAML config
