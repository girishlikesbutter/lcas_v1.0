"""Forward propagation helpers — q0 + ω → body-frame sun/observer vectors.

Uses the post-fix `src.dynamics.attitude_propagator.propagate_attitude` plus
scipy `Rotation.from_quat` for the q→matrix step. Mirrors the canonical
trajectory-generator pattern in `notebooks/inversion/09_glint_analysis/
m048_generate_trajectories_v2.py:157-173` exactly so that round-tripping
(truth-q0, truth-ω) reproduces the cached `k1_body / k2_body / mag_*`
arrays at machine precision.

This is the only place in the survey that converts a candidate (q0, ω)
state into body-frame coordinates. Other experiments should call
`propagate_to_body_frame` rather than re-rolling the loop.

Why scipy Rotation.from_quat([qx, qy, qz, qw]).as_matrix() (xyzw order):
the propagator returns (w, x, y, z) — scalar-first; scipy expects xyzw
(scalar-last). The `[q[1], q[2], q[3], q[0]]` reshuffle is required.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from src.dynamics.attitude_propagator import propagate_attitude


def propagate_to_body_frame(
    q0_wxyz: np.ndarray,
    omega0_rad: np.ndarray,
    observation_times: np.ndarray,
    sun_pos: np.ndarray,
    obs_pos: np.ndarray,
    sat_pos: np.ndarray,
    inertia_tensor: np.ndarray,
    mode: str = "tumbling",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate (q0, ω0) and project sun/observer into body frame.

    Args:
      q0_wxyz: (4,) initial quaternion (w, x, y, z).
      omega0_rad: (3,) initial body-frame angular velocity (rad/s).
      observation_times: (N,) seconds since start_et (from cached NPZ).
      sun_pos / obs_pos / sat_pos: (N, 3) J2000 positions (km).
      inertia_tensor: (3, 3) body-frame inertia (kg·m²) — only used in
        `tumbling` mode.
      mode: 'tumbling' (Euler ODE) or 'principal_axis' (closed-form).

    Returns:
      k1_body: (N, 3) unit body-frame sun direction.
      k2_body: (N, 3) unit body-frame observer direction.
      quaternions: (N, 4) propagated attitude (w, x, y, z).
    """
    quats, _ = propagate_attitude(
        q0=q0_wxyz, omega0=omega0_rad,
        times=observation_times, mode=mode,
        inertia_tensor=inertia_tensor,
    )
    N = observation_times.shape[0]
    k1_body = np.empty((N, 3))
    k2_body = np.empty((N, 3))
    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    for i in range(N):
        q = quats[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        k1_body[i] = R @ sun_vec[i] / np.linalg.norm(sun_vec[i])
        k2_body[i] = R @ obs_vec[i] / np.linalg.norm(obs_vec[i])
    return k1_body, k2_body, quats


def quat_geodesic_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    """Geodesic angle in degrees between two unit quaternions on SO(3).

    Antipode-aware: q and -q represent the same rotation, so this returns
    the minimum of d(q1, q2) and d(q1, -q2), giving values in [0, 180].
    """
    dot = float(np.abs(np.dot(q1_wxyz, q2_wxyz)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def quat_geodesic_deg_batch(q_grid: np.ndarray, q_ref_wxyz: np.ndarray) -> np.ndarray:
    """Batched geodesic angle (deg) from each row of q_grid to q_ref."""
    dots = np.abs(q_grid @ q_ref_wxyz)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))
