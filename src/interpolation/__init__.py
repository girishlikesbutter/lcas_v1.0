"""
Interpolation module for LCAS.

Provides interpolation utilities for:
- Attitudes (quaternion SLERP)
- Articulation angles (step, linear, constant)
"""

from .quaternion_interpolator import (
    slerp,
    create_quaternion_interpolator,
    create_angle_interpolator,
    parse_keyframe_times
)

from .attitude_interpolator import interpolate_attitudes_only

__all__ = [
    # Quaternion/attitude interpolation
    'slerp',
    'create_quaternion_interpolator',
    'interpolate_attitudes_only',
    # Angle interpolation
    'create_angle_interpolator',
    # Utilities
    'parse_keyframe_times',
]