"""
Attitude interpolation module for LCAS.

Provides SLERP-based interpolation for satellite attitudes with support
for both quaternion and rotation matrix inputs, and UTC/ET time formats.
Always uses SLERP for physically correct attitude interpolation.
"""

import numpy as np
import quaternion
from typing import Dict, Any

from .quaternion_interpolator import create_quaternion_interpolator, parse_keyframe_times


def interpolate_attitudes_only(
    attitude_keyframes: Dict[str, Any],
    epochs: np.ndarray,
    spice_handler: Any
) -> np.ndarray:
    """
    Interpolate attitude matrices from sparse keyframes using SLERP.

    This is a minimal helper that ONLY handles attitude interpolation,
    without recomputing positions or transforming vectors.

    Parameters:
    -----------
    attitude_keyframes : dict
        Dictionary containing:
        - 'times': List of keyframe times (UTC strings or ET floats)
        - 'time_format': 'utc' or 'et' (optional, auto-detected)
        - 'attitudes': List of attitudes (quaternions or rotation matrices)
        - 'format': 'quaternion' or 'matrix'
    epochs : np.ndarray
        Full array of ephemeris times to interpolate to
    spice_handler : SpiceHandler
        For UTC to ET conversion (if needed)

    Returns:
    --------
    np.ndarray
        Attitude matrices of shape (N, 3, 3) representing body frame
        orientation at each epoch

    Example:
    --------
    attitude_keyframes = {
        'times': ['2024-01-01T12:00:00', '2024-01-01T13:00:00'],
        'time_format': 'utc',
        'attitudes': [quat1, quat2],
        'format': 'quaternion'
    }
    att_matrices = interpolate_attitudes_only(attitude_keyframes, epochs, spice_handler)
    """
    # Parse keyframe times to ET
    time_format = attitude_keyframes.get('time_format', 'et')
    keyframe_times_et = parse_keyframe_times(
        attitude_keyframes['times'],
        time_format,
        spice_handler
    )

    # Convert attitudes to quaternions if needed
    attitudes = attitude_keyframes['attitudes']
    att_format = attitude_keyframes.get('format', 'quaternion')

    if att_format == 'matrix':
        # Convert rotation matrices to quaternions
        keyframe_quats = [
            quaternion.from_rotation_matrix(R) for R in attitudes
        ]
    elif att_format == 'quaternion':
        # Ensure we have quaternion objects
        if not isinstance(attitudes[0], quaternion.quaternion):
            # Convert from [w, x, y, z] arrays if needed
            keyframe_quats = [
                quaternion.quaternion(q[0], q[1], q[2], q[3]) if isinstance(q, (list, np.ndarray))
                else q for q in attitudes
            ]
        else:
            keyframe_quats = attitudes
    else:
        raise ValueError(f"Unknown attitude format: {att_format}")

    # Create SLERP interpolator (always SLERP for attitudes)
    attitude_interpolator = create_quaternion_interpolator(
        keyframe_times_et,
        keyframe_quats,
        time_format='et'  # Already converted to ET
    )

    # Interpolate attitudes at all epochs
    interpolated_quats = attitude_interpolator(epochs)

    # Convert to rotation matrices
    if np.isscalar(epochs):
        sat_att_matrices = np.array([quaternion.as_rotation_matrix(interpolated_quats)])
    else:
        sat_att_matrices = np.array([
            quaternion.as_rotation_matrix(q) for q in interpolated_quats
        ])

    return sat_att_matrices
