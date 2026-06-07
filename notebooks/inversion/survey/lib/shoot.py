"""Two-point attitude BVP solver (Newton/LM shoot) on the closed-form propagator.

Given two body-frame orientations q_a (at t=0) and q_b (at t=Δt) and the inertia
tensor, solve for the body-frame angular velocity omega_a at t=0 such that the
torque-free trajectory from (q_a, omega_a) reaches q_b at Δt.

3 unknowns (omega_a) <-> 3 constraints (q_b in SO(3)) => generically isolated
solutions (regular-value theorem). A single solve returns ONE root (the basin
of the init); use ``multi_shoot`` to enumerate the family by multi-start.

Convention matches lib.jacobi_propagator: scalar-first quaternions, q maps
J2000 -> body (passive), q_dot = -0.5 * omega ⊗ q, dR/dt = -[omega]_x R, so the
body-frame relative rotation over dt is q_b ⊗ q_a^{-1} (validated against truth
in s088 Gate A1).
"""
from pathlib import Path
import json
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import ellipk

from lib.jacobi_propagator import (
    propagate_jacobi_path2,
    omega_jacobi,
    _quat_multiply,
    _quat_conj,
)

_INERTIA_JSON = (
    Path(__file__).resolve().parent.parent / "results" / "s062" / "inertia_principal_axes.json"
)


def m048_inertia() -> NDArray:
    """m048 body-frame inertia tensor (kg·m²), cached by s062."""
    with open(_INERTIA_JSON) as f:
        return np.array(json.load(f)["inertia_body"], dtype=np.float64)


# ---------------------------------------------------------------------------
# quaternion / SO(3) helpers
# ---------------------------------------------------------------------------
def geodesic_angle(q1: NDArray, q2: NDArray) -> float:
    """Shortest rotation angle (rad) between two scalar-first unit quaternions."""
    d = abs(float(np.dot(q1, q2)))
    return 2.0 * np.arccos(min(1.0, d))


def _quat_log_residual(q_pred: NDArray, q_b: NDArray) -> NDArray:
    """Rotation-vector (3,) of the relative rotation q_b^{-1} ⊗ q_pred.

    Zero iff q_pred == ±q_b. This is the LM residual: a 3-vector whose norm is
    the geodesic angle, smooth away from angle=pi.
    """
    r = _quat_multiply(_quat_conj(q_b), q_pred)
    if r[0] < 0:  # shortest arc (quaternion double cover)
        r = -r
    w = min(1.0, abs(r[0]))
    angle = 2.0 * np.arccos(w)
    v = r[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-15:
        return np.zeros(3)
    return (angle / nv) * v


def finite_diff_omega(q_a: NDArray, q_b: NDArray, dt: float) -> NDArray:
    """Constant-omega finite-diff estimate of body-frame omega_a from (q_a,q_b,dt).

    Inverts q(dt) = q_delta ⊗ q_a with q_delta = exp(-0.5 [0,omega] dt), i.e.
    q_delta = q_b ⊗ q_a^{-1} has vector part -sin(|omega|dt/2) * omega_hat.
    The minus sign + body-frame interpretation are validated against truth in
    s088 Gate A1. Crude (s057c: 6-30° dir error) but a usable LM init.
    """
    r = _quat_multiply(q_b, _quat_conj(q_a))
    if r[0] < 0:
        r = -r
    w = min(1.0, abs(r[0]))
    angle = 2.0 * np.arccos(w)  # = |omega| * dt (principal branch)
    v = r[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-15 or dt == 0:
        return np.zeros(3)
    axis = v / nv
    return -(angle / dt) * axis


def omega_dir_err_deg(w: NDArray, w_ref: NDArray) -> float:
    """Angle (deg) between two angular-velocity directions."""
    nw, nr = np.linalg.norm(w), np.linalg.norm(w_ref)
    if nw < 1e-18 or nr < 1e-18:
        return np.nan
    c = float(np.dot(w, w_ref) / (nw * nr))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def omega_mag_err_frac(w: NDArray, w_ref: NDArray) -> float:
    """Relative |omega| error (signed fraction) vs a reference."""
    nr = np.linalg.norm(w_ref)
    return float((np.linalg.norm(w) - nr) / nr) if nr > 0 else np.nan


def polhode_period(omega0: NDArray, inertia: NDArray) -> float:
    """Polhode period T_pol (s): full cycle of body-frame omega = 4K(m)/|tau_dot|."""
    _, info = omega_jacobi(np.array([0.0]), omega0, inertia)
    m = info["m"]
    tau_dot = abs(info["tau_dot"])
    if tau_dot < 1e-30:
        return np.inf
    return float(4.0 * ellipk(m) / tau_dot)


# ---------------------------------------------------------------------------
# the shoot
# ---------------------------------------------------------------------------
def shoot(
    q_a: NDArray,
    q_b: NDArray,
    dt: float,
    inertia: NDArray,
    omega_init: NDArray,
    method: str = "elliprj",
    xtol: float = 1e-12,
    ftol: float = 1e-12,
) -> dict:
    """Solve for omega_a connecting q_a (t=0) -> q_b (t=dt) via LM (Levenberg-Marquardt).

    Returns a dict with the converged omega, the residual/geodesic floor (the
    connectability signal), the Jacobian condition number at the solution
    (high => degenerate locus / separatrix), and solver bookkeeping.
    """
    q_a = np.asarray(q_a, float)
    q_b = np.asarray(q_b, float)
    times = np.array([0.0, float(dt)])

    def resid(w):
        qh, _ = propagate_jacobi_path2(q_a, w, inertia, times, method=method)
        return _quat_log_residual(qh[-1], q_b)

    sol = least_squares(
        resid, np.asarray(omega_init, float), method="lm", xtol=xtol, ftol=ftol
    )
    w = sol.x
    qh, _ = propagate_jacobi_path2(q_a, w, inertia, times, method=method)
    geo = geodesic_angle(qh[-1], q_b)
    try:
        sv = np.linalg.svd(np.atleast_2d(sol.jac), compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    except Exception:
        cond = np.nan
    return dict(
        omega=w,
        residual_norm=float(np.linalg.norm(sol.fun)),
        geo_err_rad=float(geo),
        geo_err_deg=float(np.degrees(geo)),
        n_eval=int(sol.nfev),
        cost=float(sol.cost),
        jac_cond=cond,
        connected=bool(geo < 1e-6),
    )


def _apply_rotvec(q: NDArray, r: NDArray) -> NDArray:
    """Left-multiply q (scalar-first) by the rotation exp([0, r]) given a
    rotation-vector r (rad). Used to let the base orientation float."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-15:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        dq = np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * (r / theta)])
    out = _quat_multiply(dq, q)
    return out / np.linalg.norm(out)


def shoot_multianchor_freebase(
    q_a_init: NDArray,
    anchors: list,
    inertia: NDArray,
    omega_init: NDArray,
    base_resid=None,
    w_geo: float = 1.0,
    method: str = "elliprj",
    xtol: float = 1e-12,
    ftol: float = 1e-12,
) -> dict:
    """Over-determined BVP with the BASE orientation q_a FLOATING.

    Unknowns = [omega_a (3), r (3)] where the working base is
    q_a = exp([0, r]) ⊗ q_a_init (r a rotation-vector correction, init 0).
    Residual stack = w_geo * [per-anchor geodesic log-residuals] (3K of them)
    PLUS, if ``base_resid`` is given, the scalar it returns for the current q_a
    appended at the end (already weighted by the caller).

    Rationale (s090): with K=2 target anchors, freeing q_a makes the geometry-only
    system SQUARE (6 unknowns vs 6 constraints) — it drives the residual to zero by
    threading the *noisy* targets exactly, over-fitting cloud noise rather than
    averaging it. A scalar ``base_resid`` (e.g. a brightness-isophote residual,
    (mag(q_a) - mag_obs)/sigma_mag) re-constrains q_a to its observed-brightness
    2-surface, so it may slide ALONG the isophote (the shared-base noise component
    that can average out) but not off it.
    """
    q_a_init = np.asarray(q_a_init, float)
    targets = sorted(
        ((np.asarray(q, float), float(dt)) for q, dt in anchors), key=lambda t: t[1]
    )
    times = np.concatenate([[0.0], np.array([dt for _, dt in targets], float)])

    def resid(x):
        w, r = x[:3], x[3:]
        q_a = _apply_rotvec(q_a_init, r)
        qh, _ = propagate_jacobi_path2(q_a, w, inertia, times, method=method)
        stack = [
            w_geo * _quat_log_residual(qh[k + 1], targets[k][0])
            for k in range(len(targets))
        ]
        if base_resid is not None:
            stack.append(np.array([float(base_resid(q_a))]))
        return np.concatenate(stack)

    x0 = np.concatenate([np.asarray(omega_init, float), np.zeros(3)])
    sol = least_squares(resid, x0, method="lm", xtol=xtol, ftol=ftol)
    w, r = sol.x[:3], sol.x[3:]
    q_a = _apply_rotvec(q_a_init, r)
    qh, _ = propagate_jacobi_path2(q_a, w, inertia, times, method=method)
    geos = [geodesic_angle(qh[k + 1], targets[k][0]) for k in range(len(targets))]
    return dict(
        omega=w,
        q_a=q_a,
        base_shift_deg=float(np.degrees(np.linalg.norm(r))),
        residual_norm=float(np.linalg.norm(sol.fun)),
        geo_err_deg=[float(np.degrees(g)) for g in geos],
        geo_err_max_deg=float(np.degrees(max(geos))),
        n_eval=int(sol.nfev),
        cost=float(sol.cost),
        connected=bool(max(geos) < 1e-6),
    )


def shoot_multianchor(
    q_a: NDArray,
    anchors: list,
    inertia: NDArray,
    omega_init: NDArray,
    method: str = "elliprj",
    xtol: float = 1e-12,
    ftol: float = 1e-12,
) -> dict:
    """Over-determined two-point BVP: omega_a at t=0 whose torque-free trajectory
    from q_a passes through MULTIPLE target orientations.

    ``anchors`` = list of (q_k, dt_k) with dt_k > 0 (the t=0 anchor is q_a). With
    K>=2 targets this is over-determined: 3 unknowns (omega_a) vs 3K geodesic
    constraints. Joint LM least-squares (a) averages independent endpoint (q-cloud)
    noise down and (b) pins the trajectory winding via the intermediate anchor, so
    a LONG-baseline (well-conditioned) solve no longer aliases to a neighbouring
    winding root — the s088 Phase-B conditioning<->multiplicity crossover. s089.

    Returns the converged omega plus per-anchor geodesic residuals and the worst
    residual across anchors (the multi-anchor connectability signal).
    """
    q_a = np.asarray(q_a, float)
    targets = sorted(
        ((np.asarray(q, float), float(dt)) for q, dt in anchors), key=lambda t: t[1]
    )
    times = np.concatenate([[0.0], np.array([dt for _, dt in targets], float)])

    def resid(w):
        qh, _ = propagate_jacobi_path2(q_a, w, inertia, times, method=method)
        return np.concatenate(
            [_quat_log_residual(qh[k + 1], targets[k][0]) for k in range(len(targets))]
        )

    sol = least_squares(
        resid, np.asarray(omega_init, float), method="lm", xtol=xtol, ftol=ftol
    )
    w = sol.x
    qh, _ = propagate_jacobi_path2(q_a, w, inertia, times, method=method)
    geos = [geodesic_angle(qh[k + 1], targets[k][0]) for k in range(len(targets))]
    try:
        sv = np.linalg.svd(np.atleast_2d(sol.jac), compute_uv=False)
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    except Exception:
        cond = np.nan
    return dict(
        omega=w,
        residual_norm=float(np.linalg.norm(sol.fun)),
        geo_err_deg=[float(np.degrees(g)) for g in geos],
        geo_err_max_deg=float(np.degrees(max(geos))),
        n_eval=int(sol.nfev),
        cost=float(sol.cost),
        jac_cond=cond,
        connected=bool(max(geos) < 1e-6),
    )
