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
    raise NotImplementedError("normalize_quaternion not yet implemented")


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
    raise NotImplementedError("axis_angle_to_quaternion not yet implemented")


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
    raise NotImplementedError("quaternion_to_axis_angle not yet implemented")


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
    raise NotImplementedError("quaternion_multiply not yet implemented")
