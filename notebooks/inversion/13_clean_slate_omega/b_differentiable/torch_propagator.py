"""Torch differentiable attitude propagator (Euler rigid-body + quaternion kinematics).

Dynamics (torque-free rigid body):
    q_dot   = 0.5 * q (x) [0, omega]           (quaternion kinematics, scalar-first)
    I omega_dot = -omega x (I omega)           (Euler's equation)

Integrator: classical RK4 on the 7-vector state [q_w,q_x,q_y,q_z,ox,oy,oz].
  Quaternion is renormalised after each step (lightly) to control numerical
  drift without breaking gradients.

We use a fine internal grid (dt ~ 0.25 s by default) and sample at the given
observation_times via the closest internal index. Over 3600 s this is ~14400
steps; a single 500-step LC evaluation is O(seconds) on CPU.

Output: quaternions at the requested times (N, 4).

All ops are differentiable; gradients flow back to q0 and omega0.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two scalar-first quaternions (..., 4) x (..., 4)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def rot_matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) -> (..., 3, 3) inertial->body rotation matrix R_ib.

    Matches lib/forward.body_vectors_from_attitude.
    """
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y], dim=-1),
        torch.stack([2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x], dim=-1),
        torch.stack([2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y], dim=-1),
    ], dim=-2)
    return R


def _state_dot(state: torch.Tensor, inertia_tensor: torch.Tensor,
               I_inv: torch.Tensor) -> torch.Tensor:
    """State derivative for RK4.

    state: (..., 7) = [q_w, q_x, q_y, q_z, ox, oy, oz]
    inertia_tensor, I_inv: (3, 3) (shared across the batch)
    returns: (..., 7)

    NOTE: we do NOT renormalise the quaternion inside the derivative — that
    would break RK4 consistency (the derivative must be a pure function of
    state). We renormalise between integrator steps instead.
    """
    q = state[..., 0:4]
    omega = state[..., 4:7]

    omega_quat = torch.cat([torch.zeros_like(omega[..., :1]), omega], dim=-1)
    q_dot = 0.5 * quat_mul(q, omega_quat)

    # I @ omega  and omega x (I omega)
    I_omega = omega @ inertia_tensor.T
    omega_cross = torch.cross(omega, I_omega, dim=-1)
    omega_dot = (-omega_cross) @ I_inv.T
    return torch.cat([q_dot, omega_dot], dim=-1)


def propagate_euler_torch_batched(
    q0: torch.Tensor,
    omega0: torch.Tensor,
    inertia_tensor: torch.Tensor,
    times: torch.Tensor,
    substeps_per_obs: int = 16,
    renormalise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched RK4 propagation over B parallel initial conditions.

    Args:
        q0: (B, 4) batch of initial quaternions.
        omega0: (B, 3) batch of initial angular velocities (rad/s).
        inertia_tensor: (3, 3) shared inertia (torque-free dynamics).
        times: (N,) observation times (seconds).
        substeps_per_obs: RK4 substeps per observation interval.

    Returns:
        q_out: (B, N, 4) quaternions at each obs time.
        omega_out: (B, N, 3).
    """
    dtype = q0.dtype
    device = q0.device
    inertia_tensor = inertia_tensor.to(dtype=dtype, device=device)
    I_inv = torch.linalg.inv(inertia_tensor)

    if q0.dim() == 1:
        q0 = q0.unsqueeze(0)
    if omega0.dim() == 1:
        omega0 = omega0.unsqueeze(0)
    B = q0.shape[0]

    q0 = q0 / torch.clamp(q0.norm(dim=-1, keepdim=True), min=1e-12)
    state = torch.cat([q0, omega0], dim=-1)  # (B, 7)

    N = int(times.shape[0])
    q_snapshots = [state[:, 0:4]]
    omega_snapshots = [state[:, 4:7]]
    times_np = times.detach().cpu().numpy()

    def rk4_step(s, h):
        k1 = _state_dot(s, inertia_tensor, I_inv)
        k2 = _state_dot(s + 0.5 * h * k1, inertia_tensor, I_inv)
        k3 = _state_dot(s + 0.5 * h * k2, inertia_tensor, I_inv)
        k4 = _state_dot(s + h * k3, inertia_tensor, I_inv)
        return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    for j in range(1, N):
        dt_obs = float(times_np[j] - times_np[j - 1])
        h = dt_obs / substeps_per_obs
        for _ in range(substeps_per_obs):
            state = rk4_step(state, h)
            if renormalise:
                q_part = state[:, 0:4]
                q_part = q_part / torch.clamp(q_part.norm(dim=-1, keepdim=True), min=1e-12)
                state = torch.cat([q_part, state[:, 4:7]], dim=-1)
        q_snapshots.append(state[:, 0:4])
        omega_snapshots.append(state[:, 4:7])

    q_out = torch.stack(q_snapshots, dim=1)  # (B, N, 4)
    omega_out = torch.stack(omega_snapshots, dim=1)
    return q_out, omega_out


def body_vectors_from_quats_batched(
    quats: torch.Tensor,
    sun_j2k: torch.Tensor,
    obs_j2k: torch.Tensor,
    sat_j2k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched rotation. quats (B,N,4); sun/obs/sat (N,3). Returns (B,N,3) each."""
    R = rot_matrix_from_quat(quats)  # (B, N, 3, 3)
    sun_rel = sun_j2k - sat_j2k  # (N, 3)
    obs_rel = obs_j2k - sat_j2k
    k1 = torch.einsum("bnij,nj->bni", R, sun_rel)
    k2 = torch.einsum("bnij,nj->bni", R, obs_rel)
    k1 = k1 / torch.clamp(k1.norm(dim=-1, keepdim=True), min=1e-30)
    k2 = k2 / torch.clamp(k2.norm(dim=-1, keepdim=True), min=1e-30)
    return k1, k2


def propagate_euler_torch(
    q0: torch.Tensor,
    omega0: torch.Tensor,
    inertia_tensor: torch.Tensor,
    times: torch.Tensor,
    substeps_per_obs: int = 16,
    renormalise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RK4 Euler propagation aligned to observation times.

    Takes `substeps_per_obs` RK4 steps between each pair of observation
    times, then snapshots the state exactly at each observation time. This
    eliminates the O(h) "snap to nearest internal grid index" error.

    Args:
        q0: (4,) scalar-first quaternion at t=times[0] (requires_grad supported).
        omega0: (3,) rad/s at t=times[0].
        inertia_tensor: (3, 3) body-frame inertia.
        times: (N,) monotonically increasing times (seconds).
        substeps_per_obs: number of RK4 substeps between consecutive obs
            times. m048 obs dt ~ 7.2 s, so 16 substeps => h ≈ 0.45 s;
            this gives k1/k2 errors well under 1e-6 vs scipy DOP853.
        renormalise: if True, renormalise the quaternion after each substep
            (stabilises long propagation).

    Returns:
        q_out: (N, 4) quaternion at each requested time.
        omega_out: (N, 3) angular velocity at each requested time.
    """
    dtype = q0.dtype
    device = q0.device
    inertia_tensor = inertia_tensor.to(dtype=dtype, device=device)
    I_inv = torch.linalg.inv(inertia_tensor)

    q0 = q0 / torch.clamp(q0.norm(), min=1e-12)
    state = torch.cat([q0, omega0], dim=0)  # (7,)

    N = int(times.shape[0])
    q_out = [None] * N
    omega_out = [None] * N
    q_out[0] = state[0:4]
    omega_out[0] = state[4:7]

    # dt per obs interval, then substep within.
    times_np = times.detach().cpu().numpy()

    def rk4_step(s, h):
        k1 = _state_dot(s, inertia_tensor, I_inv)
        k2 = _state_dot(s + 0.5 * h * k1, inertia_tensor, I_inv)
        k3 = _state_dot(s + 0.5 * h * k2, inertia_tensor, I_inv)
        k4 = _state_dot(s + h * k3, inertia_tensor, I_inv)
        return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    for j in range(1, N):
        dt_obs = float(times_np[j] - times_np[j - 1])
        h = dt_obs / substeps_per_obs
        h_t = torch.as_tensor(h, dtype=dtype, device=device)
        for _ in range(substeps_per_obs):
            state = rk4_step(state, h_t)
            if renormalise:
                q_part = state[0:4]
                q_part = q_part / torch.clamp(q_part.norm(), min=1e-12)
                state = torch.cat([q_part, state[4:7]], dim=0)
        q_out[j] = state[0:4]
        omega_out[j] = state[4:7]

    q_out_t = torch.stack(q_out, dim=0)
    omega_out_t = torch.stack(omega_out, dim=0)
    return q_out_t, omega_out_t


def body_vectors_from_quats(
    quats: torch.Tensor,
    sun_j2k: torch.Tensor,
    obs_j2k: torch.Tensor,
    sat_j2k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate J2000 sun/obs relative vectors into body frame using q_body_from_inertial.

    Inputs are (N, 3) or (N, 4). Returns (k1_body, k2_body), both (N, 3) unit.
    """
    R = rot_matrix_from_quat(quats)  # (N, 3, 3)
    sun_rel = sun_j2k - sat_j2k  # (N, 3)
    obs_rel = obs_j2k - sat_j2k
    k1 = torch.einsum("nij,nj->ni", R, sun_rel)
    k2 = torch.einsum("nij,nj->ni", R, obs_rel)
    k1 = k1 / torch.clamp(k1.norm(dim=1, keepdim=True), min=1e-30)
    k2 = k2 / torch.clamp(k2.norm(dim=1, keepdim=True), min=1e-30)
    return k1, k2


__all__ = [
    "propagate_euler_torch",
    "propagate_euler_torch_batched",
    "body_vectors_from_quats",
    "body_vectors_from_quats_batched",
    "quat_mul",
    "rot_matrix_from_quat",
]
