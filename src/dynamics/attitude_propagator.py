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
    raise NotImplementedError("propagate_principal_axis not yet implemented")


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
