# src/utils - Geometry Utilities

This module provides geometry utility functions for rotation matrix construction and sun-pointing calculations.

## Module Overview

| File | Purpose |
|------|---------|
| `geometry_utils.py` | Rotation matrix construction, sun-pointing angle calculation |

## Key Functions

### build_rotation_matrix
Create 4x4 homogeneous rotation matrix for articulation.

```python
from src.utils.geometry_utils import build_rotation_matrix

# Rotate 45 degrees around Y axis
rotation_matrix = build_rotation_matrix(
    angle_deg=45.0,
    rotation_axis=np.array([0.0, 1.0, 0.0])
)
# Returns: np.ndarray (4, 4) homogeneous rotation matrix
```

**Implementation:**
- Uses Rodrigues' rotation formula
- Accepts angle in degrees
- Normalizes rotation axis internally
- Returns 4x4 matrix (3x3 rotation + identity translation)

### calculate_sun_pointing_rotation
Calculate rotation angle to align a panel normal toward the sun.

```python
from src.utils.geometry_utils import calculate_sun_pointing_rotation

angle_deg = calculate_sun_pointing_rotation(
    sun_vector=k1_vector,                # Sun direction in body frame
    panel_axis=np.array([0, 1, 0]),      # Rotation axis
    panel_normal=np.array([1, 0, 0])     # Reference normal to point at sun
)
# Returns: float, rotation angle in degrees
```

**Algorithm:**
1. Project sun vector onto plane perpendicular to rotation axis
2. Project panel normal onto same plane
3. Calculate signed angle between projections using `atan2`
4. Return angle needed to align normal toward sun

## Usage in Articulation

```python
from src.utils.geometry_utils import build_rotation_matrix
from src.articulation import ArticulationEngine

# Build rotation matrices for all epochs
rotation_matrices = np.zeros((num_epochs, 4, 4))
for i, angle in enumerate(angles_array):
    rotation_matrices[i] = build_rotation_matrix(
        angle_deg=angle,
        rotation_axis=component.articulation_parameters.rotation_axis
    )
```

## Rodrigues' Rotation Formula

The rotation matrix R is computed as:

```
R = I + sin(θ) K + (1 - cos(θ)) K²
```

Where:
- `I` is the 3x3 identity matrix
- `θ` is the rotation angle in radians
- `K` is the skew-symmetric cross-product matrix of the unit rotation axis

```python
K = [[ 0,   -k_z,  k_y],
     [ k_z,  0,   -k_x],
     [-k_y,  k_x,  0  ]]
```

## Cross-References

- **Articulation engine**: Uses `build_rotation_matrix` in `compute_rotation_matrices_from_angles`
- **Sun tracking behavior**: Uses similar logic in `SunTrackingBehavior.calculate_rotation_angle`
- **Shadow engine**: Rotation matrices applied during mesh transformation
