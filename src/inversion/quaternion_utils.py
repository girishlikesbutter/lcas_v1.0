"""
Quaternion utilities for optimization.

This module provides quaternion manipulation functions specifically
designed for use in the lightcurve inversion optimization loop.

All functions use the scalar-first (w, x, y, z) convention.
"""

import numpy as np
from numpy.typing import NDArray


def normalize_quaternion(
    q: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Normalize a quaternion to unit length.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Quaternion (w, x, y, z) - scalar first convention.

    Returns
    -------
    ndarray, shape (4,)
        Unit quaternion.
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        # Return identity quaternion for near-zero input
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def axis_angle_to_quaternion(
    axis_angle: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Convert axis-angle representation to quaternion.

    The axis-angle representation uses a 3-parameter encoding where
    the direction is the rotation axis and the magnitude is the
    rotation angle in radians.

    Parameters
    ----------
    axis_angle : ndarray, shape (3,)
        Rotation vector where ||axis_angle|| is the rotation angle
        and axis_angle / ||axis_angle|| is the rotation axis.

    Returns
    -------
    ndarray, shape (4,)
        Quaternion (w, x, y, z) - scalar first convention.

    Notes
    -----
    The conversion formula is:
        angle = ||axis_angle||
        axis = axis_angle / angle (if angle > 0)
        q = [cos(angle/2), sin(angle/2) * axis]
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-12:
        # No rotation - return identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])

    # Normalize axis
    axis = axis_angle / angle
    half_angle = 0.5 * angle

    # Build quaternion: [cos(θ/2), sin(θ/2) * axis]
    sin_half = np.sin(half_angle)
    return np.array([
        np.cos(half_angle),
        sin_half * axis[0],
        sin_half * axis[1],
        sin_half * axis[2],
    ])


def quaternion_to_axis_angle(
    q: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Convert quaternion to axis-angle representation.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Quaternion (w, x, y, z) - scalar first convention.

    Returns
    -------
    ndarray, shape (3,)
        Rotation vector where ||result|| is the rotation angle
        and result / ||result|| is the rotation axis.

    Notes
    -----
    The conversion formula is:
        angle = 2 * acos(w)
        axis = [x, y, z] / sin(angle/2) (if sin(angle/2) > 0)
        axis_angle = angle * axis
    """
    q = np.asarray(q, dtype=np.float64)

    # Normalize the quaternion first
    q = q / np.linalg.norm(q)

    # Ensure w is positive (use the shorter rotation path)
    if q[0] < 0:
        q = -q

    w, x, y, z = q

    # Clamp w to [-1, 1] to handle numerical errors
    w = np.clip(w, -1.0, 1.0)

    # Compute the half-angle
    half_angle = np.arccos(w)
    angle = 2.0 * half_angle
    sin_half = np.sin(half_angle)

    if sin_half < 1e-12:
        # No rotation - return zero vector
        return np.array([0.0, 0.0, 0.0])

    # Compute axis from vector part
    axis = np.array([x, y, z]) / sin_half

    return angle * axis


def quaternion_multiply(
    q1: NDArray[np.floating],
    q2: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Multiply two quaternions.

    Parameters
    ----------
    q1 : ndarray, shape (4,)
        First quaternion (w, x, y, z) - scalar first convention.
    q2 : ndarray, shape (4,)
        Second quaternion (w, x, y, z) - scalar first convention.

    Returns
    -------
    ndarray, shape (4,)
        Product quaternion q1 * q2.

    Notes
    -----
    Uses Hamilton product with scalar-first convention:
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
