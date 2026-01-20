"""
Attitude propagation functions for satellite dynamics.

This module provides functions to propagate satellite attitude (orientation)
over time using different dynamical models:

1. Principal axis rotation: Assumes constant angular velocity in body frame.
   Uses closed-form quaternion solution for efficient computation.

2. Euler dynamics: Propagates coupled attitude and angular velocity using
   Euler's equations of motion. Suitable for tumbling debris analysis.

All functions use scalar-first quaternion convention (w, x, y, z).
"""

from typing import Literal, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


def propagate_attitude(
    q0: NDArray[np.floating],
    omega0: NDArray[np.floating],
    times: NDArray[np.floating],
    mode: Literal["principal_axis", "tumbling"],
    inertia_tensor: Optional[NDArray[np.floating]] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Propagate satellite attitude using the specified mode.

    Unified interface that dispatches to the appropriate propagation method
    based on the selected mode.

    Parameters
    ----------
    q0 : ndarray, shape (4,)
        Initial quaternion (w, x, y, z) - scalar first convention.
    omega0 : ndarray, shape (3,)
        Initial angular velocity in body frame (rad/s).
    times : ndarray, shape (N,)
        Array of times at which to compute attitude (seconds).
    mode : {'principal_axis', 'tumbling'}
        Propagation mode:
        - 'principal_axis': Constant angular velocity (closed-form).
        - 'tumbling': Evolving angular velocity using Euler equations.
    inertia_tensor : ndarray, shape (3, 3), optional
        Inertia tensor in body frame (kg*m^2). Required for tumbling mode.

    Returns
    -------
    quaternions : ndarray, shape (N, 4)
        Quaternion array at each time step (w, x, y, z).
    omega_history : ndarray, shape (N, 3)
        Angular velocity history at each time step (rad/s).
        For principal_axis mode, this is constant across all times.

    Raises
    ------
    ValueError
        If mode is 'tumbling' and inertia_tensor is not provided.
        If mode is not recognized.
    """
    raise NotImplementedError("propagate_attitude not yet implemented")


def propagate_principal_axis(
    q0: NDArray[np.floating],
    omega: NDArray[np.floating],
    times: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Propagate attitude assuming constant angular velocity in body frame.

    Uses the closed-form solution for rotation about a fixed axis:
        q(t) = q0 * quaternion_exp(0.5 * omega * dt)

    This is valid when the satellite rotates about a principal axis
    with no external torques.

    Parameters
    ----------
    q0 : ndarray, shape (4,)
        Initial quaternion (w, x, y, z) - scalar first convention.
    omega : ndarray, shape (3,)
        Constant angular velocity vector in body frame (rad/s).
    times : ndarray, shape (N,)
        Array of times at which to compute attitude (seconds).
        Times are relative to the initial state (t=0 corresponds to q0).

    Returns
    -------
    quaternions : ndarray, shape (N, 4)
        Quaternion array at each time step (w, x, y, z).
        All output quaternions are normalized.

    Notes
    -----
    The quaternion exponential for a rotation vector v = omega * dt is:
        quat_exp(v) = [cos(|v|/2), sin(|v|/2) * v/|v|]
    """
    q0 = np.asarray(q0, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    # Normalize initial quaternion
    q0 = q0 / np.linalg.norm(q0)

    n_times = len(times)
    quaternions = np.zeros((n_times, 4), dtype=np.float64)

    # Angular velocity magnitude
    omega_mag = np.linalg.norm(omega)

    for i, t in enumerate(times):
        if omega_mag < 1e-12:
            # No rotation - return initial quaternion
            quaternions[i] = q0
        else:
            # Rotation angle: theta = |omega| * t
            theta = omega_mag * t

            # Half angle for quaternion
            half_theta = 0.5 * theta

            # Quaternion exponential: exp(0.5 * omega * t)
            # For rotation vector v, quat_exp(v) = [cos(|v|/2), sin(|v|/2) * v/|v|]
            # Here v = omega * t, |v| = omega_mag * t = theta
            # So quat_exp(0.5 * omega * t) = [cos(theta/2), sin(theta/2) * omega/|omega|]
            axis = omega / omega_mag
            q_rot = np.array([
                np.cos(half_theta),
                np.sin(half_theta) * axis[0],
                np.sin(half_theta) * axis[1],
                np.sin(half_theta) * axis[2],
            ])

            # Quaternion multiplication: q(t) = q0 * q_rot
            # Using scalar-first convention (w, x, y, z)
            quaternions[i] = _quaternion_multiply(q0, q_rot)

    # Normalize all quaternions
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / norms

    return quaternions


def _quaternion_multiply(
    q1: NDArray[np.floating], q2: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Multiply two quaternions using scalar-first convention (w, x, y, z).

    Parameters
    ----------
    q1 : ndarray, shape (4,)
        First quaternion.
    q2 : ndarray, shape (4,)
        Second quaternion.

    Returns
    -------
    ndarray, shape (4,)
        Product quaternion q1 * q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def propagate_euler(
    q0: NDArray[np.floating],
    omega0: NDArray[np.floating],
    inertia_tensor: NDArray[np.floating],
    times: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Propagate attitude with evolving angular velocity using Euler equations.

    Integrates the coupled system of Euler's equations (angular momentum
    conservation) and quaternion kinematics to propagate both attitude
    and angular velocity over time.

    Parameters
    ----------
    q0 : ndarray, shape (4,)
        Initial quaternion (w, x, y, z) - scalar first convention.
    omega0 : ndarray, shape (3,)
        Initial angular velocity in body frame (rad/s).
    inertia_tensor : ndarray, shape (3, 3)
        Inertia tensor in body frame (kg*m^2).
    times : ndarray, shape (N,)
        Array of times at which to compute attitude (seconds).

    Returns
    -------
    quaternions : ndarray, shape (N, 4)
        Quaternion array at each time step (w, x, y, z).
        Quaternions are renormalized during integration.
    omega_history : ndarray, shape (N, 3)
        Angular velocity history at each time step (rad/s).

    Notes
    -----
    Euler's equations of motion (torque-free):
        I @ omega_dot = -omega x (I @ omega)

    Quaternion kinematics:
        q_dot = 0.5 * q * omega_quat

    where omega_quat = [0, omega_x, omega_y, omega_z]

    Uses scipy.integrate.solve_ivp with RK45 or DOP853 solver.
    """
    raise NotImplementedError("propagate_euler not yet implemented")
