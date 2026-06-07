"""
Jacobi-elliptic closed-form torque-free attitude propagation.

Implements analytical omega(t) via scipy.special.ellipj (Landau-Lifshitz §37).

Public API:
- omega_jacobi(times, omega0, inertia) -> (omega_hist, info)
- propagate_jacobi(q0, omega0, inertia, times) -> (q_hist, omega_hist)
    Hybrid (analytical omega + DOP853 on q). Matches the propagator's q
    exactly because they share the same q ODE and integrator family.
- propagate_jacobi_path2(q0, omega0, inertia, times, method='elliprj') -> (q_hist, omega_hist)
    Path 2 closed-form q(t) via 3-1-3 precession+nutation decomposition.
    s072 (2026-05-13) re-derived under the post-fix textbook convention
    (no phi sign flip). s074 (2026-05-20) replaced the 1-D phi ODE with
    a closed-form Pi(n; am(tau) | m) evaluation via scipy.special.elliprj
    + elliprf (Carlson R_J + R_F per DLMF 19.25.14); ~5x faster, retaining
    the strict 1e-9 q-vs-DOP853 + 1e-12 conservation gates of s072.
    Pass `method='ode'` for the s072 solve_ivp path (kept as fallback).
    The legacy attempt `_propagate_jacobi_path2_incomplete` is preserved
    below as a historical record of the pre-fix puzzle (sign-flipped phi).

Convention (matches src/dynamics/attitude_propagator.py post-2026-05-12 fix):
scalar-first quaternions, as_rotation_matrix(q) = passive J2000 -> body,
q_dot = -0.5 * omega_quat * q (textbook conv-(a) kinematic).
"""

from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.special import ellipj, ellipkinc, ellipk, elliprf, elliprj


def _quat_multiply(q1: NDArray, q2: NDArray) -> NDArray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q: NDArray) -> NDArray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_from_matrix(R: NDArray) -> NDArray:
    """Convert a 3x3 proper rotation matrix to scalar-first quaternion.

    Uses Shepperd's method (numerically stable across all cases).
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _eigendecompose_inertia(inertia: NDArray) -> Tuple[NDArray, NDArray]:
    """Return (I_principal_ascending, R_pa) with R_pa a proper rotation
    such that R_pa @ diag(I_principal) @ R_pa.T == inertia.

    Columns of R_pa are the principal axes expressed in the body frame.
    """
    I_vals, V = np.linalg.eigh(inertia)
    if np.linalg.det(V) < 0:
        V = V.copy()
        V[:, 0] *= -1.0
    return I_vals, V


def _build_omega_func(
    I_pa: NDArray, omega0_pa: NDArray
) -> Tuple[Callable[[NDArray], NDArray], dict]:
    """Build a closed-form omega(t) function in the PA frame.

    Returns (omega_at_t, info_dict). omega_at_t accepts a scalar or array of times
    and returns shape (..., 3).
    """
    I_1, I_2, I_3 = I_pa
    w1_0, w2_0, w3_0 = omega0_pa

    twoT = I_1 * w1_0**2 + I_2 * w2_0**2 + I_3 * w3_0**2
    L2 = (I_1 * w1_0)**2 + (I_2 * w2_0)**2 + (I_3 * w3_0)**2

    disc = twoT * I_2 - L2

    info = {"twoT_0": twoT, "L2_0": L2, "disc_0": disc}

    if disc >= 0:
        # Case A: 2T·I_2 > L², polhode encloses I_1 (small axis).
        # ω_1 = sgn_1·a1·dn(τ), ω_2 = a2·sn(τ), ω_3 = a3·cn(τ).
        # k² = (I_3-I_2)(L²-2T·I_1) / [(I_2-I_1)(2T·I_3-L²)] ∈ (0, 1).
        # Sign convention: ω_1 doesn't change sign in this regime; sgn_1 carries it.
        # Time evolution sign: from Euler eq, dτ/dt = sgn_1 * |tau_dot|.
        a1 = np.sqrt((twoT*I_3 - L2) / (I_1 * (I_3 - I_1)))
        a2 = np.sqrt((L2 - twoT*I_1) / (I_2 * (I_2 - I_1)))
        a3 = np.sqrt((L2 - twoT*I_1) / (I_3 * (I_3 - I_1)))
        tau_dot_mag = np.sqrt((I_2 - I_1) * (twoT*I_3 - L2) / (I_1 * I_2 * I_3))
        m = (I_3 - I_2) * (L2 - twoT*I_1) / ((I_2 - I_1) * (twoT*I_3 - L2))

        sgn_1 = 1.0 if w1_0 >= 0 else -1.0
        tau_dot = sgn_1 * tau_dot_mag

        sn_0 = np.clip(w2_0 / a2, -1.0, 1.0)
        cn_0_signed = w3_0 / a3
        phi_0 = np.arcsin(sn_0)
        u_0 = ellipkinc(phi_0, m)
        K_val = ellipk(m)
        tau_0 = (2.0 * K_val - u_0) if cn_0_signed < 0 else u_0

        info.update(dict(regime="A", a=(a1, a2, a3), tau_dot=tau_dot,
                         m=m, tau_0=tau_0, sgn_1=sgn_1))

        def omega_at_t(t):
            scalar = np.isscalar(t)
            t_arr = np.atleast_1d(t).astype(np.float64)
            tau = tau_dot * t_arr + tau_0
            sn, cn, dn, _ = ellipj(tau, m)
            out = np.column_stack([sgn_1 * a1 * dn, a2 * sn, a3 * cn])
            return out[0] if scalar else out

    else:
        # Case B: 2T·I_2 < L², polhode encloses I_3 (large axis).
        # ω_1 = a1·cn(τ), ω_2 = a2·sn(τ), ω_3 = sgn_3·a3·dn(τ).
        # k² = (I_2-I_1)(2T·I_3-L²) / [(I_3-I_2)(L²-2T·I_1)] ∈ (0, 1).
        # Sign convention: ω_3 doesn't change sign; sgn_3 carries it.
        # Time evolution sign: from Euler eq, dτ/dt = sgn_3 * |tau_dot|.
        a1 = np.sqrt((twoT*I_3 - L2) / (I_1 * (I_3 - I_1)))
        a2 = np.sqrt((twoT*I_3 - L2) / (I_2 * (I_3 - I_2)))
        a3 = np.sqrt((L2 - twoT*I_1) / (I_3 * (I_3 - I_1)))
        tau_dot_mag = np.sqrt((I_3 - I_2) * (L2 - twoT*I_1) / (I_1 * I_2 * I_3))
        m = (I_2 - I_1) * (twoT*I_3 - L2) / ((I_3 - I_2) * (L2 - twoT*I_1))

        sgn_3 = 1.0 if w3_0 >= 0 else -1.0
        tau_dot = sgn_3 * tau_dot_mag

        sn_0 = np.clip(w2_0 / a2, -1.0, 1.0)
        cn_0_signed = w1_0 / a1
        phi_0 = np.arcsin(sn_0)
        u_0 = ellipkinc(phi_0, m)
        K_val = ellipk(m)
        tau_0 = (2.0 * K_val - u_0) if cn_0_signed < 0 else u_0

        info.update(dict(regime="B", a=(a1, a2, a3), tau_dot=tau_dot,
                         m=m, tau_0=tau_0, sgn_3=sgn_3))

        def omega_at_t(t):
            scalar = np.isscalar(t)
            t_arr = np.atleast_1d(t).astype(np.float64)
            tau = tau_dot * t_arr + tau_0
            sn, cn, dn, _ = ellipj(tau, m)
            out = np.column_stack([a1 * cn, a2 * sn, sgn_3 * a3 * dn])
            return out[0] if scalar else out

    return omega_at_t, info


def omega_jacobi(
    times: NDArray, omega0: NDArray, inertia: NDArray
) -> Tuple[NDArray, dict]:
    """Closed-form omega(t) in the BODY frame for torque-free motion.

    Parameters
    ----------
    times : (N,) array of times relative to t=0 (seconds).
    omega0 : (3,) initial angular velocity in body frame (rad/s).
    inertia : (3,3) symmetric positive-definite inertia tensor in body frame.

    Returns
    -------
    omega_hist : (N, 3) angular velocity in body frame at each time.
    info : dict with regime, k², tau_0, etc. for diagnostics.
    """
    times = np.asarray(times, dtype=np.float64)
    omega0 = np.asarray(omega0, dtype=np.float64)
    inertia = np.asarray(inertia, dtype=np.float64)

    I_pa, R_pa = _eigendecompose_inertia(inertia)
    omega0_pa = R_pa.T @ omega0

    omega_func_pa, info = _build_omega_func(I_pa, omega0_pa)
    omega_pa_hist = omega_func_pa(times)              # (N, 3) in PA frame
    omega_body_hist = omega_pa_hist @ R_pa.T          # PA -> body
    return omega_body_hist, info


def _Rz_passive(theta: float) -> NDArray:
    """Passive rotation matrix about z by angle theta.

    Rotates the COORDINATE FRAME counterclockwise by theta; the same physical
    vector's components transform as v_new = R_z(theta) @ v_old.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _Rx_passive(theta: float) -> NDArray:
    """Passive rotation matrix about x by angle theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  s],
        [0.0, -s,  c],
    ])


def _quat_to_matrix(q: NDArray) -> NDArray:
    """Convert scalar-first (w, x, y, z) quaternion to 3x3 rotation matrix.

    Returns the convention-(a) matrix R such that v_body = R @ v_J2000
    (i.e., the same matrix scipy returns from `Rotation.from_quat(xyzw).as_matrix()`,
    interpreted as passive J2000->body under conv-(a)).
    """
    w, x, y, z = q
    # Standard Hamilton quaternion-to-matrix; agrees with scipy's Rotation.as_matrix
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),       2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ])


def _propagate_jacobi_path2_incomplete(
    q0: NDArray, omega0: NDArray, inertia: NDArray, times: NDArray,
    phi_rtol: float = 1e-13, phi_atol: float = 1e-15,
) -> Tuple[NDArray, NDArray]:
    """INCOMPLETE attempt at closed-form q(t) via precession+nutation (Path 2).

    s062b finding (2026-05-12): the codebase's q ODE `dq/dt = +0.5 omega ⊗ q`
    (LEFT, Hamilton) does NOT correspond to the conventional passive J2000->body
    kinematic equation `dR/dt = -[omega]_x R`. Empirically, `L_J2000 = R.T @ L_body`
    is NOT conserved by the propagator output, on either a real seed or a toy
    asymmetric tumbler — even though physics demands L_J2000 conservation under
    torque-free motion. Neither this nor `R @ L_body` is constant.

    s065 (2026-05-12, partial resolution): the SPICE comparison test
    (`experiments/s065_propagator_convention_spice_test.py`) traced the t=0
    behaviour to an omega-sign convention. The q-interpretation is correct
    (`scipy.as_matrix(q) == passive J2000->body`, matches SPICE `pxform` to
    3e-16 at t=0); the kinematic ODE has the OPPOSITE sign of the textbook
    passive-J2000->body kinematic, so feeding `-omega_textbook` produces R
    matching SPICE to ~1e-10/s at 1-second horizon.

    s066 (2026-05-12 later, post-s065): the s065 framing is UNDERSTATED for
    LC durations. L_J2000 = R(t).T @ I @ omega(t) is NOT conserved by the
    coupled ODE: 36-137% drift on m048 cohort over a 60-min LC. Real torque-
    free physics requires exact conservation. The codebase pairs textbook
    Euler (correct, RHS even in ω) with an opposite-sign Hamilton kinematic.
    omega → -omega only fixes the kinematic LOCALLY; over an LC the Euler
    equation's sign-asymmetry in time means no global sign substitution
    recovers a textbook trajectory. See `experiments/s066_lj2000_nonconservation.md`.

    Consequence for Path 2: the textbook precession+nutation decomposition
    decomposes the motion around the (constant) L_J2000 direction. In the
    codebase's data L_J2000 is NOT constant — there is no fixed axis to
    precess around. The "derive under codebase convention (omega -> -omega)"
    recipe DOES NOT WORK because the codebase's trajectory is not a textbook
    torque-free trajectory at all. A future closed-form q(t) implementation
    would need to derive directly for the codebase's specific non-physical
    coupled ODE (meaningfully harder than the textbook case); alternatively,
    the propagator could be fixed and the cohort regenerated (which makes the
    textbook Path 2 work cleanly).

    The hybrid path below (closed-form omega + DOP853 on q-quaternion ODE)
    remains the production path because it bypasses both issues: it integrates
    the same q ODE the propagator uses, so the q output matches whatever the
    propagator would have produced (faithful to the codebase's forward model),
    and the closed-form ω is correct in either physics interpretation.

    This function is kept as a record of the Path 2 attempt. At t=0 it
    produces R matching the propagator to 1e-16; at t > 0 it drifts at the
    precession rate because the derivation assumed textbook signs.

    Parameters
    ----------
    q0 : (4,) initial quaternion (w, x, y, z), scalar-first.
    omega0 : (3,) initial angular velocity in body frame (rad/s).
    inertia : (3,3) inertia tensor in body frame (kg m^2).
    times : (N,) times relative to t=0 (seconds). times[0] should be 0.
    phi_rtol, phi_atol : tolerances for the 1D scalar phi-dot ODE.

    Returns
    -------
    q_hist : (N, 4) quaternions in BODY frame at each time (INCORRECT — see above).
    omega_hist : (N, 3) angular velocities in body frame at each time (correct).
    """
    q0 = np.asarray(q0, dtype=np.float64)
    omega0 = np.asarray(omega0, dtype=np.float64)
    inertia = np.asarray(inertia, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    q0 = q0 / np.linalg.norm(q0)

    # PA decomposition
    I_pa, R_pa = _eigendecompose_inertia(inertia)
    I_1, I_2, I_3 = I_pa

    omega0_pa = R_pa.T @ omega0

    # Closed-form omega(t) in PA frame
    omega_func_pa, info = _build_omega_func(I_pa, omega0_pa)
    omega_pa_hist = omega_func_pa(times)               # (N, 3)
    L2 = info["L2_0"]
    twoT = info["twoT_0"]
    L_mag = np.sqrt(L2)

    # Closed-form theta, psi from omega(t) in PA frame
    cos_theta = np.clip((I_3 * omega_pa_hist[:, 2]) / L_mag, -1.0, 1.0)
    theta_hist = np.arccos(cos_theta)
    psi_hist = np.arctan2(I_1 * omega_pa_hist[:, 0], I_2 * omega_pa_hist[:, 1])

    # phi(t) via 1D scalar ODE: dphi/dt = |L| (2T - I_3 w3^2) / (L^2 - I_3^2 w3^2)
    def phi_dot(t, _phi):
        w_pa = omega_func_pa(np.float64(t))
        if w_pa.ndim == 2:
            w_pa = w_pa[0]
        w3 = w_pa[2]
        num = twoT - I_3 * w3 * w3
        den = L2 - (I_3 * w3) ** 2
        return [L_mag * num / den]

    sol = solve_ivp(
        phi_dot, (times[0], times[-1]), [0.0],
        method="DOP853", t_eval=times,
        rtol=phi_rtol, atol=phi_atol,
    )
    phi_hist = -sol.y[0]                               # (N,)
    # Sign flip: the codebase's q ODE dq/dt = +0.5 omega ⊗ q (LEFT) gives a
    # passive-J2000->body matrix evolving as dR/dt = +[omega]_x R, opposite
    # to the conventional right-handed dR/dt = -[omega]_x R. The Goldstein
    # 3-1-3 phi-dot formula assumes right-handed kinematics; under this
    # codebase's convention the precession runs the other way, so phi -> -phi.

    # Determine R_J2000_to_L by enforcing phi(0) = 0 and matching to q0_PA.
    # q0 -> R_J2000_to_body_0 (passive); R_J2000_to_PA_0 = R_pa.T @ R_J2000_to_body_0.
    # Demand: R_z(psi_0) @ R_x(theta_0) @ R_z(0) @ R_J2000_to_L = R_J2000_to_PA_0
    # => R_J2000_to_L = R_x(-theta_0) @ R_z(-psi_0) @ R_J2000_to_PA_0
    R_J2000_to_body_0 = _quat_to_matrix(q0)
    R_J2000_to_PA_0 = R_pa.T @ R_J2000_to_body_0
    R_J2000_to_L = _Rx_passive(-theta_hist[0]) @ _Rz_passive(-psi_hist[0]) @ R_J2000_to_PA_0

    # Reconstruct R(t) and convert to quaternion at every t.
    N = times.shape[0]
    q_body_hist = np.empty((N, 4))
    for i in range(N):
        R_zxz = _Rz_passive(psi_hist[i]) @ _Rx_passive(theta_hist[i]) @ _Rz_passive(phi_hist[i])
        R_J2000_to_PA = R_zxz @ R_J2000_to_L
        R_J2000_to_body = R_pa @ R_J2000_to_PA
        q_body_hist[i] = _quat_from_matrix(R_J2000_to_body)

    omega_body_hist = omega_pa_hist @ R_pa.T
    return q_body_hist, omega_body_hist


def propagate_jacobi(
    q0: NDArray, omega0: NDArray, inertia: NDArray, times: NDArray,
    q_rtol: float = 1e-12, q_atol: float = 1e-14,
) -> Tuple[NDArray, NDArray]:
    """Hybrid propagator: closed-form omega + DOP853 on quaternion ODE.

    This is the PRODUCTION path post-s062b. The Path 2 closed-form q(t)
    attempt (precession+nutation Euler decomposition) is held in
    `_propagate_jacobi_path2_incomplete` pending resolution of a codebase
    convention puzzle — see that function's docstring for details.

    Convention (a): scalar-first quaternion, as_rotation_matrix(q) = J2000 -> body.

    Parameters
    ----------
    q0 : (4,) initial quaternion (w, x, y, z), scalar-first.
    omega0 : (3,) initial angular velocity in body frame (rad/s).
    inertia : (3,3) inertia tensor in body frame (kg m^2).
    times : (N,) times relative to t=0 (seconds). times[0] should be 0.

    Returns
    -------
    q_hist : (N, 4) quaternions in BODY frame at each time, normalized.
    omega_hist : (N, 3) angular velocities in body frame at each time.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    omega0 = np.asarray(omega0, dtype=np.float64)
    inertia = np.asarray(inertia, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    q0 = q0 / np.linalg.norm(q0)

    # PA decomposition
    I_pa, R_pa = _eigendecompose_inertia(inertia)
    omega0_pa = R_pa.T @ omega0

    # PA-frame quaternion: q_body_from_inertial = q_pa_to_body ⊗ q_PA_from_inertial
    # ⟹ q_PA_from_inertial = q_pa_to_body.conj ⊗ q_body_from_inertial
    q_pa_to_body = _quat_from_matrix(R_pa)        # represents R_pa
    q_body_to_pa = _quat_conj(q_pa_to_body)
    q0_pa = _quat_multiply(q_body_to_pa, q0)
    q0_pa = q0_pa / np.linalg.norm(q0_pa)

    # Closed-form omega(t) in PA frame
    omega_func_pa, _info = _build_omega_func(I_pa, omega0_pa)

    # Integrate q̇ = -0.5 * omega_quat * q with closed-form omega
    # (textbook conv-(a) kinematic; matches src/dynamics/attitude_propagator.py post-2026-05-12 fix).
    def q_dot(t, q):
        omega_t = omega_func_pa(t)
        omega_quat = np.array([0.0, omega_t[0], omega_t[1], omega_t[2]])
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm
        return -0.5 * _quat_multiply(omega_quat, q)

    sol = solve_ivp(
        q_dot, (times[0], times[-1]), q0_pa,
        method="DOP853", t_eval=times,
        rtol=q_rtol, atol=q_atol,
    )
    q_pa_hist = sol.y.T   # (N, 4) in PA frame
    q_pa_hist = q_pa_hist / np.linalg.norm(q_pa_hist, axis=1, keepdims=True)

    # PA -> body for both q and omega
    q_body_hist = np.array([_quat_multiply(q_pa_to_body, q) for q in q_pa_hist])
    q_body_hist = q_body_hist / np.linalg.norm(q_body_hist, axis=1, keepdims=True)

    omega_pa_hist = omega_func_pa(times)
    omega_body_hist = omega_pa_hist @ R_pa.T

    return q_body_hist, omega_body_hist


def _phi_closed_form_elliprj(times: NDArray, info: dict, I_pa: NDArray) -> NDArray:
    """Closed-form phi(t) via incomplete elliptic integral of the third kind Pi(n; am(tau) | m).

    Replaces the solve_ivp(phi_dot, ...) call inside propagate_jacobi_path2.
    s073b (2026-05-14) profiled the ODE path at 81% of Path 2 wall; this
    function targets that bottleneck.

    The phi ODE is

        dphi/dt = |L| (2T − I_3 omega_3^2) / (L^2 − I_3^2 omega_3^2)

    Using the regime-agnostic identity I_3 N − D = 2T·I_3 − L^2 (where N is
    the numerator and D is the denominator), this rearranges to

        dphi/dt = |L|/I_3 + |L|·(2T·I_3 − L²)/(I_3 · D)

    integrating to

        phi(t) − phi(t0) = (|L|/I_3)·(t − t0)
                        + (|L|·(2T·I_3 − L²) / (I_3·tau_dot·(L² − I_3·P)))
                          · [J(tau(t)) − J(tau(t0))]

    where J(tau) = ∫₀^tau dτ'/(1 − n·sn²(τ', m)) and n = I_3·Q/(L² − I_3·P)
    with the constants P, Q defined below per regime. J(tau) is itself the
    incomplete Pi function: J(tau) = Pi(n; am(tau) | m), evaluated via
    Carlson's R_F + R_J per DLMF 19.25.14.

    Periodicity: sn²(τ, m) has period 2K(m), so J(τ + 2K) = J(τ) + 2·Pi_K
    where Pi_K = Pi(n; π/2 | m) is the complete elliptic integral of the
    third kind. We reduce τ to the principal interval [-K, K] using
    `j = floor((τ + K)/(2K))` and τ_red = τ − j·2K ∈ [-K, K], then
    am(τ_red) ∈ [-π/2, π/2] is in DLMF's principal range.

    Gauge: phi(times[0]) = 0 (consistent with `_propagate_jacobi_path2`'s
    R_J→L pin step).

    Returns
    -------
    phi_hist : (N,) phi values at each time.
    """
    times = np.asarray(times, dtype=np.float64)
    I_1, I_2, I_3 = I_pa
    twoT = info["twoT_0"]
    L2 = info["L2_0"]
    L_mag = np.sqrt(L2)
    regime = info["regime"]
    a1, a2, a3 = info["a"]
    tau_dot = info["tau_dot"]
    tau_0 = info["tau_0"]
    m = info["m"]

    # In both regimes ω_3(t)² = a_3² + β·sn²(τ, m) with
    #   regime A: ω_3 = ±a_3·cn(τ),     β = −a_3²       (cn² = 1 − sn²)
    #   regime B: ω_3 = ±a_3·dn(τ),     β = −m·a_3²     (dn² = 1 − m·sn²)
    # so D(τ) = L² − I_3²·ω_3² = (L² − I_3·P) − I_3·Q·sn²(τ),
    # with P = I_3·a_3², Q = I_3·β, and we factor D = (L² − I_3·P)·(1 − n·sn²).
    P = I_3 * a3 * a3
    if regime == "A":
        beta = -a3 * a3
    else:  # regime B
        beta = -m * a3 * a3
    Q = I_3 * beta
    L2_minus_I3P = L2 - I_3 * P
    n_param = (I_3 * Q) / L2_minus_I3P  # may be negative; that's fine for Π

    A_coeff = L_mag / I_3
    B_coeff = L_mag * (twoT * I_3 - L2) / (I_3 * tau_dot * L2_minus_I3P)

    K_val = ellipk(m)
    two_K = 2.0 * K_val

    # Vectorized tau at t0 prepended so we can subtract J(tau_0).
    tau_t = tau_dot * times + tau_0
    tau_all = np.empty(tau_t.size + 1, dtype=np.float64)
    tau_all[0] = tau_0
    tau_all[1:] = tau_t

    # Period reduction.
    j = np.floor((tau_all + K_val) / two_K)
    tau_red = tau_all - j * two_K  # in [-K, K]

    sn, cn, dn, _ = ellipj(tau_red, m)
    sn2 = sn * sn
    cn2 = cn * cn
    dn2 = dn * dn

    ones = np.ones_like(sn2)
    # DLMF 19.25.14: Π(n; am(τ) | m) = sn·R_F(cn², dn², 1)
    #                                + (n/3)·sn³·R_J(cn², dn², 1, 1 − n·sn²)
    R_F_val = elliprf(cn2, dn2, ones)
    R_J_val = elliprj(cn2, dn2, ones, 1.0 - n_param * sn2)
    Pi_inc = sn * R_F_val + (n_param / 3.0) * sn * sn2 * R_J_val

    # Complete Π over [0, K]: Π(n; π/2 | m). Same formula with sn = 1, cn² = 0, dn² = 1 − m.
    one_minus_m = 1.0 - m
    R_F_c = elliprf(0.0, one_minus_m, 1.0)
    R_J_c = elliprj(0.0, one_minus_m, 1.0, 1.0 - n_param)
    Pi_K_val = 1.0 * R_F_c + (n_param / 3.0) * 1.0 * R_J_c

    J_all = j * (2.0 * Pi_K_val) + Pi_inc
    J_0 = J_all[0]
    J_t = J_all[1:]

    phi = A_coeff * times + B_coeff * (J_t - J_0)
    # Pin gauge: phi(times[0]) = 0.
    phi = phi - phi[0]
    return phi


def propagate_jacobi_path2(
    q0: NDArray, omega0: NDArray, inertia: NDArray, times: NDArray,
    phi_rtol: float = 1e-13, phi_atol: float = 1e-15,
    method: str = "elliprj",
) -> Tuple[NDArray, NDArray]:
    """Closed-form q(t) via Path 2 (3-1-3 precession+nutation decomposition).

    Under post-fix textbook convention (-0.5 * omega ⊗ q LEFT, dR/dt = -[ω]× R)
    L_J2000 is constant, so the motion decomposes uniquely into precession of
    the principal-axis (PA) frame about the fixed L_J2000 direction at rate
    phi_dot, nutation by angle theta(t) (instantaneous angle between PA z and
    L), and spin by psi(t) about PA z. Closed-form omega(t) drives theta and
    psi algebraically; phi requires a 1-D scalar ODE.

    Math (Goldstein 4.87, passive 3-1-3 with PA z as third axis):
        cos(theta)  = I_3 * omega_3 / |L|
        psi         = atan2(I_1 * omega_1, I_2 * omega_2)
        d(phi)/dt   = |L| * (2T - I_3 * omega_3^2) / (L^2 - I_3^2 * omega_3^2)
        R_J→PA(t)   = R_z(psi) @ R_x(theta) @ R_z(phi) @ R_J→L
        R_J→body(t) = R_pa @ R_J→PA(t)

    R_J→L is the constant frame rotation that aligns J2000's z with L_J2000;
    it's recovered at t=0 from (q_0, theta_0, psi_0) under the gauge phi(0)=0.

    This formula is regime-agnostic: in regime A (rotation about I_1) it gives
    theta ≈ pi/2 with oscillations; in regime B (rotation about I_3) theta
    stays small. The atan2 for psi handles wraparound automatically.

    Parameters
    ----------
    q0 : (4,) initial quaternion (w, x, y, z), scalar-first.
    omega0 : (3,) initial body-frame angular velocity (rad/s).
    inertia : (3,3) body-frame inertia tensor (kg·m²); need not be diagonal.
    times : (N,) times relative to t=0 (seconds). times[0] should be 0.
    phi_rtol, phi_atol : tolerances for the 1-D phi ODE
        (only used when ``method='ode'``).
    method : str, default 'elliprj'
        - 'elliprj' (default, s074 2026-05-20): closed-form phi(t) via the
          incomplete Pi function evaluated through scipy.special.elliprj +
          elliprf (Carlson R_J + R_F per DLMF 19.25.14). Removes the 1-D
          ODE entirely; ~5x faster than the ODE path (full-function median
          ~3 ms vs ~16 ms on a 500-epoch trajectory).
        - 'ode' (legacy, kept as fallback): scipy.integrate.solve_ivp with
          DOP853 on the dphi/dt closed-form integrand. Retained for the
          k² → 1 separatrix limit where elliprj's parameter region becomes
          sensitive.

    Returns
    -------
    q_hist : (N, 4) quaternions in BODY frame, scalar-first, normalized.
    omega_hist : (N, 3) angular velocities in body frame (rad/s).
    """
    if method not in ("elliprj", "ode"):
        raise ValueError(f"method must be 'elliprj' or 'ode', got {method!r}")

    q0 = np.asarray(q0, dtype=np.float64)
    omega0 = np.asarray(omega0, dtype=np.float64)
    inertia = np.asarray(inertia, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    q0 = q0 / np.linalg.norm(q0)

    # PA decomposition: inertia = R_pa @ diag(I_pa) @ R_pa.T (R_pa is proper).
    I_pa, R_pa = _eigendecompose_inertia(inertia)
    I_1, I_2, I_3 = I_pa

    omega0_pa = R_pa.T @ omega0

    # Closed-form omega(t) in PA frame.
    omega_func_pa, info = _build_omega_func(I_pa, omega0_pa)
    omega_pa_hist = omega_func_pa(times)
    L2 = info["L2_0"]
    twoT = info["twoT_0"]
    L_mag = np.sqrt(L2)

    # theta(t), psi(t) — algebraic from omega.
    cos_theta = np.clip(I_3 * omega_pa_hist[:, 2] / L_mag, -1.0, 1.0)
    theta_hist = np.arccos(cos_theta)
    psi_hist = np.arctan2(I_1 * omega_pa_hist[:, 0], I_2 * omega_pa_hist[:, 1])

    # phi(t) — closed-form via elliprj (default) or solve_ivp (fallback).
    if method == "elliprj":
        phi_hist = _phi_closed_form_elliprj(times, info, I_pa)
    else:
        # Legacy ODE path. Textbook 3-1-3 passive form, NO sign flip:
        # the post-fix propagator's q ODE matches textbook conv-(a) kinematic.
        def phi_dot(t, _phi):
            w_pa = omega_func_pa(np.float64(t))
            if w_pa.ndim == 2:
                w_pa = w_pa[0]
            w3 = w_pa[2]
            num = twoT - I_3 * w3 * w3
            den = L2 - (I_3 * w3) ** 2
            return [L_mag * num / den]

        sol = solve_ivp(
            phi_dot, (times[0], times[-1]), [0.0],
            method="DOP853", t_eval=times,
            rtol=phi_rtol, atol=phi_atol,
        )
        phi_hist = sol.y[0]

    # Pin R_J→L from initial condition (gauge: phi(0) = 0).
    R_J2000_to_body_0 = _quat_to_matrix(q0)
    R_J2000_to_PA_0 = R_pa.T @ R_J2000_to_body_0
    R_J2000_to_L = (
        _Rx_passive(-theta_hist[0]) @ _Rz_passive(-psi_hist[0]) @ R_J2000_to_PA_0
    )

    # Reconstruct R(t) → quaternion at every t.
    N = times.shape[0]
    q_body_hist = np.empty((N, 4))
    for i in range(N):
        R_zxz = (
            _Rz_passive(psi_hist[i])
            @ _Rx_passive(theta_hist[i])
            @ _Rz_passive(phi_hist[i])
        )
        R_J2000_to_PA = R_zxz @ R_J2000_to_L
        R_J2000_to_body = R_pa @ R_J2000_to_PA
        q_body_hist[i] = _quat_from_matrix(R_J2000_to_body)

    omega_body_hist = omega_pa_hist @ R_pa.T
    return q_body_hist, omega_body_hist
