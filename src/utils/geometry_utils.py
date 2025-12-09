"""
Geometry utilities for satellite articulation and light curve computation.

Provides:
- build_rotation_matrix: Create 4x4 rotation matrices for articulating components
- calculate_sun_pointing_rotation: Calculate angles for sun-tracking solar panels
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_rotation_matrix(angle_deg: float, rotation_axis: np.ndarray) -> np.ndarray:
    """
    Build 4x4 homogeneous rotation matrix for the given angle around an axis.

    Uses Rodrigues' rotation formula to create a rotation matrix. The rotation
    is about the origin - component geometry should be defined with the rotation
    axis passing through local origin.

    Args:
        angle_deg: Rotation angle in degrees.
        rotation_axis: Axis of rotation (3D vector, will be normalized).

    Returns:
        4x4 homogeneous rotation matrix.
    """
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Normalize rotation axis
    axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Create rotation matrix using Rodrigues' formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)

    # Convert to 4x4 homogeneous matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R

    return rotation_matrix


def calculate_sun_pointing_rotation(sun_vector: np.ndarray, panel_axis: np.ndarray, panel_normal: np.ndarray) -> float:
    """
    Calculate the rotation angle to align a panel normal with the sun direction.

    Projects both vectors onto the plane perpendicular to the rotation axis,
    then computes the signed angle between them.

    Args:
        sun_vector: Direction to the sun in body frame (3D vector).
        panel_axis: Axis of rotation for the panel in body frame (3D vector).
        panel_normal: Current normal direction of the panel in body frame (3D vector).

    Returns:
        Rotation angle in degrees needed to align panel_normal toward sun_vector.
    """
    if not isinstance(sun_vector, np.ndarray) or not isinstance(panel_axis, np.ndarray) or not isinstance(panel_normal, np.ndarray):
        raise TypeError("All input vectors must be NumPy arrays.")

    if sun_vector.shape != (3,) or panel_axis.shape != (3,) or panel_normal.shape != (3,):
        raise ValueError("All input vectors must be 3-element vectors.")

    # Normalize input vectors
    sun_norm = np.linalg.norm(sun_vector)
    axis_norm = np.linalg.norm(panel_axis)
    normal_norm = np.linalg.norm(panel_normal)

    if sun_norm < 1e-9 or axis_norm < 1e-9 or normal_norm < 1e-9:
        raise ValueError("Input vectors must not be zero vectors.")

    sun_unit = sun_vector / sun_norm
    axis_unit = panel_axis / axis_norm
    normal_unit = panel_normal / normal_norm

    # Project sun vector and panel normal onto the plane perpendicular to the rotation axis
    sun_projected = sun_unit - np.dot(sun_unit, axis_unit) * axis_unit
    normal_projected = normal_unit - np.dot(normal_unit, axis_unit) * axis_unit

    # Normalize projected vectors
    sun_proj_norm = np.linalg.norm(sun_projected)
    normal_proj_norm = np.linalg.norm(normal_projected)

    if sun_proj_norm < 1e-9 or normal_proj_norm < 1e-9:
        # Sun vector or panel normal is parallel to rotation axis, no rotation needed
        return 0.0

    sun_proj_unit = sun_projected / sun_proj_norm
    normal_proj_unit = normal_projected / normal_proj_norm

    # Calculate signed angle between projected vectors using atan2
    cos_angle = np.dot(normal_proj_unit, sun_proj_unit)
    sin_angle = np.dot(np.cross(normal_proj_unit, sun_proj_unit), axis_unit)

    angle_rad = np.arctan2(sin_angle, cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg
