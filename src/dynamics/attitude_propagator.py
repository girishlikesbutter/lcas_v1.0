"""
Attitude propagation functions for satellite dynamics.

This module provides functions to propagate satellite attitude (orientation)
over time using different dynamical models:

1. Principal axis rotation: Assumes constant angular velocity in body frame.
   Uses closed-form quaternion solution for efficient computation.

2. Euler dynamics: Propagates coupled attitude and angular velocity using
   Euler's equations of motion. Suitable for tumbling debris analysis.

All functions use scalar-first quaternion convention (w, x, y, z).

Convention (a) — passive J2000→body interpretation:
    R(q) = scipy.from_quat(q[xyzw]).as_matrix() == passive J2000→body matrix
    Kinematic ODE (body-frame omega): dq/dt = -0.5 omega_quat ⊗ q (LEFT)
    Closed-form principal axis: q(t) = exp(-0.5 omega t) ⊗ q0  (LEFT)

This is what every downstream renderer assumes via `k1_body = R @ k1_J2000`.
SPICE pxform validation: feeding `propagate_euler(+omega_textbook)` matches
`pxform('J2000', body, et)` at ~1e-10 per second.
L_J2000 = R(t).T @ I @ omega(t) is conserved to 1e-12 on the toy and ~1e-8
on the m048 cohort over a 60-min LC (DOP853 noise floor).

Validation: `notebooks/inversion/survey/experiments/s067_postfix_propagator_validation.py`.

History — pre-2026-05-12 the kinematic ODE used `+0.5` (opposite sign). Combined
with the textbook-correct Euler equation, that produced trajectories with
L_J2000 drifting 36–137% over a 60-min LC (s066 finding). Fix lands here on
2026-05-12 (one-character flip on each propagator); m048 cohort regenerated.
See `notebooks/inversion/survey/concepts/quaternion_convention.md` for the
full historical narrative.
"""

from typing import Literal, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


def propagate_attitude(
    q0: NDArray[np.floating],
    omega0: NDArray[np.floating],
    times: NDArray[np.floating],
    mode: Literal["principal_axis", "tumbling"],
    inertia_tensor: Optional[NDArray[np.floating]] = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
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
    rtol : float, optional
        Relative tolerance for ODE solver (default 1e-10).
    atol : float, optional
        Absolute tolerance for ODE solver (default 1e-12).

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
    if mode == "principal_axis":
        # Use closed-form solution for constant angular velocity
        quaternions = propagate_principal_axis(q0, omega0, times)
        # For principal axis mode, angular velocity is constant
        n_times = len(times)
        omega_history = np.tile(omega0.reshape(1, 3), (n_times, 1))
        return quaternions, omega_history

    elif mode == "tumbling":
        # Check that inertia tensor is provided
        if inertia_tensor is None:
            raise ValueError(
                "inertia_tensor is required for tumbling mode propagation"
            )
        # Use Euler dynamics integration
        return propagate_euler(q0, omega0, inertia_tensor, times,
                               rtol=rtol, atol=atol)

    else:
        raise ValueError(
            f"Unrecognized mode '{mode}'. Must be 'principal_axis' or 'tumbling'."
        )


def propagate_principal_axis(
    q0: NDArray[np.floating],
    omega: NDArray[np.floating],
    times: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Propagate attitude assuming constant angular velocity in body frame.

    Convention: q is in convention (a), i.e.
        as_rotation_matrix(q) = passive J2000 → body matrix
        (equivalently, q is the active inertial → body rotation)

    Closed-form solution under convention (a):
        q(t) = quaternion_exp(-0.5 * omega * dt) * q0     (LEFT multiply)

    This is valid when the satellite rotates about a principal axis with
    no external torques (so omega is constant in the body frame).

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

            # Quaternion exponential: exp(-0.5 * omega * t)
            # For rotation vector v, quat_exp(v) = [cos(|v|/2), sin(|v|/2) * v/|v|]
            # Here v = -omega * t, |v| = omega_mag * t = theta
            # So quat_exp(-0.5 * omega * t) = [cos(theta/2), -sin(theta/2) * omega/|omega|]
            axis = omega / omega_mag
            q_rot = np.array([
                np.cos(half_theta),
                -np.sin(half_theta) * axis[0],
                -np.sin(half_theta) * axis[1],
                -np.sin(half_theta) * axis[2],
            ])

            # Quaternion multiplication under convention (a): q(t) = q_rot * q0
            # (LEFT multiply — composing a body-frame rotation onto the existing
            # attitude is a left-multiply when q is the active inertial→body rotation).
            # Using scalar-first convention (w, x, y, z).
            quaternions[i] = _quaternion_multiply(q_rot, q0)

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
    rtol: float = 1e-10,
    atol: float = 1e-12,
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
    Convention: q is in convention (a), i.e.
        as_rotation_matrix(q) = passive J2000 → body matrix
        (equivalently, q is the active inertial → body rotation)

    Euler's equations of motion (torque-free, body frame, convention-independent):
        I @ omega_dot = -omega x (I @ omega)

    Quaternion kinematics under convention (a) with body-frame omega:
        q_dot = -0.5 * omega_quat * q                      (LEFT multiply, Hamilton)

    where omega_quat = [0, omega_x, omega_y, omega_z].

    Uses scipy.integrate.solve_ivp with RK45 or DOP853 solver.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    omega0 = np.asarray(omega0, dtype=np.float64)
    inertia_tensor = np.asarray(inertia_tensor, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    # Normalize initial quaternion
    q0 = q0 / np.linalg.norm(q0)

    # Precompute inverse inertia tensor for efficiency
    I_inv = np.linalg.inv(inertia_tensor)

    def dynamics(t: float, state: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute state derivatives for the coupled Euler + quaternion system.

        State vector: [q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]
        """
        # Extract quaternion and angular velocity from state
        q = state[0:4]
        omega = state[4:7]

        # Renormalize quaternion to prevent drift
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        # Euler's equations (torque-free): I @ omega_dot = -omega x (I @ omega)
        # omega_dot = I_inv @ (-omega x (I @ omega))
        I_omega = inertia_tensor @ omega
        omega_cross_I_omega = np.cross(omega, I_omega)
        omega_dot = I_inv @ (-omega_cross_I_omega)

        # Quaternion kinematics under convention (a) with body-frame omega:
        #   q_dot = -0.5 * omega_quat * q    (LEFT multiply, Hamilton)
        # where omega_quat = [0, omega_x, omega_y, omega_z].
        # Using scalar-first convention (w, x, y, z).
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot = -0.5 * _quaternion_multiply(omega_quat, q)

        return np.concatenate([q_dot, omega_dot])

    # Initial state vector: [q(4), omega(3)]
    y0 = np.concatenate([q0, omega0])

    # Determine time span
    t_span = (times[0], times[-1])

    # Solve the ODE using DOP853 (8th order Dormand-Prince, good for smooth problems)
    solution = solve_ivp(
        dynamics,
        t_span,
        y0,
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )

    # Extract results
    n_times = len(times)
    quaternions = np.zeros((n_times, 4), dtype=np.float64)
    omega_history = np.zeros((n_times, 3), dtype=np.float64)

    for i in range(n_times):
        q = solution.y[0:4, i]
        omega = solution.y[4:7, i]

        # Renormalize quaternion
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        quaternions[i] = q
        omega_history[i] = omega

    return quaternions, omega_history
