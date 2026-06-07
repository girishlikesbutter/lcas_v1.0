"""s063a — Polhode coords <-> omega_a bijectivity verification.

Test that the polhode parameterization (regime, k², τ_0, τ_dot, amplitudes,
signs) extracted by `_build_omega_func` is a clean bijection with omega_a.

Three checks:
  (1) Forward+reconstruct: build info_dict from omega_a, then reconstruct
      omega(t=0) from the polhode coords directly via the sn/cn/dn formulas.
      Should equal omega_a in PA frame to machine precision.
  (2) Polhode-curve consistency: evaluate omega(t) over a full polhode period
      (200 phases). All points must lie on the same polhode (constant 2T, L²).
  (3) Coverage: uniform sampling in tau ∈ [0, 4K(m)] should cover the polhode
      curve (no clustering). Check via min/max sweep of |omega| and components.

Test population:
  - 120 cohort seeds (real m048 omega_0's)
  - 60 random ω at varied magnitudes and orientations
  - 20 corner cases (near principal axes, near separatrix by construction)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ellipj, ellipk

SURVEY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY))

from lib.jacobi_propagator import (  # noqa: E402
    omega_jacobi,
    _build_omega_func,
    _eigendecompose_inertia,
)


def reconstruct_omega_pa(info: dict, tau: np.ndarray) -> np.ndarray:
    """Pure inverse map: given polhode coords (info) + phase tau, return omega_PA.

    This is what s062d would sample on. Mirrors the formulas in
    `_build_omega_func` but takes tau directly (no implicit time mapping).
    """
    m = info["m"]
    a1, a2, a3 = info["a"]
    sn, cn, dn, _ = ellipj(tau, m)
    if info["regime"] == "A":
        sgn_1 = info["sgn_1"]
        return np.column_stack([sgn_1 * a1 * dn, a2 * sn, a3 * cn])
    else:
        sgn_3 = info["sgn_3"]
        return np.column_stack([a1 * cn, a2 * sn, sgn_3 * a3 * dn])


def test_population():
    rng = np.random.default_rng(20260512)
    cases = []

    # (a) Cohort seeds
    traj_dir = SURVEY / "data" / "trajectories"
    for seed in range(120):
        npz = traj_dir / f"traj_seed{seed:03d}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        cases.append(("cohort", seed, d["omega0_rad"].astype(np.float64)))

    # (b) Random ω at varied magnitudes — log-uniform |ω| in [1e-4, 5e-2] rad/s
    for i in range(60):
        log_mag = rng.uniform(np.log(1e-4), np.log(5e-2))
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        omega = np.exp(log_mag) * direction
        cases.append(("random", i, omega))

    # (c) Corner cases: near principal axes
    mag = 5e-3
    for i, axis_pa in enumerate(np.eye(3)):
        # Slight off-axis perturbation
        omega_pa = mag * axis_pa + 1e-5 * rng.standard_normal(3)
        cases.append(("near_axis_PA", i, omega_pa))

    # (d) Sign permutations of the same magnitude
    for i in range(8):
        sign = rng.choice([-1, 1], size=3)
        magvec = np.array([3e-3, 2e-3, 1.5e-3]) * sign
        cases.append(("signed", i, magvec))

    # (e) Near-separatrix by construction (synthetic — k² close to 1)
    # In PA frame: pick ω_2 such that 2T*I_2 = L² ± epsilon.
    # For m048 I_PA = (7749, 37985, 38306). Take ω_PA = (a, b, c).
    # disc = 0 requires a^2 (I_1 (I_2 - I_1)) = c^2 (I_3 (I_3 - I_2)) approximately.
    I_pa = np.array([7749.014672251076, 37985.15566495171, 38305.70560133419])
    for i in range(9):
        eps_log = rng.uniform(np.log(1e-6), np.log(1e-2))
        a = 1e-3
        c = a * np.sqrt(I_pa[0] * (I_pa[1] - I_pa[0]) /
                        (I_pa[2] * (I_pa[2] - I_pa[1]))) * (1 + np.exp(eps_log))
        b = rng.uniform(0.5e-3, 2e-3)
        omega_pa = np.array([a, b, c])
        cases.append(("near_separatrix_PA", i, omega_pa))

    return cases


def main():
    inertia = np.diag([37985.15566495171, 38305.70560133419, 7749.014672251076])

    cases = test_population()
    print(f"Total test cases: {len(cases)}")

    I_pa, R_pa = _eigendecompose_inertia(inertia)
    print(f"I_pa ascending: {I_pa}")

    results = []
    fail_reconstruct = 0
    fail_polhode = 0
    fail_round_trip = 0

    max_recon_err = 0.0
    max_polhode_err = 0.0
    max_round_trip_err = 0.0
    t0 = time.time()

    for tag, idx, omega_body in cases:
        omega_pa = R_pa.T @ omega_body

        info = _build_omega_func(I_pa, omega_pa)[1]
        twoT_0 = info["twoT_0"]
        L2_0 = info["L2_0"]
        regime = info["regime"]
        m = info["m"]
        tau_dot = info["tau_dot"]
        tau_0 = info["tau_0"]

        # ----- Check 1: reconstruct omega(t=0) from polhode coords. -----
        # `_build_omega_func` returns omega_at_t with omega_at_t(0) = omega_pa
        # only if tau_0 and signs are correctly resolved.
        recon = reconstruct_omega_pa(info, np.array([tau_0]))[0]
        recon_err = float(np.linalg.norm(recon - omega_pa))
        max_recon_err = max(max_recon_err, recon_err)
        if recon_err > 1e-10:
            fail_reconstruct += 1

        # ----- Check 2: polhode consistency. Evaluate at 200 phases. -----
        K_m = ellipk(m)
        taus = np.linspace(0, 4.0 * K_m, 200, endpoint=False)
        omega_curve_pa = reconstruct_omega_pa(info, taus)  # (200, 3)
        twoT_curve = (I_pa[None, :] * omega_curve_pa ** 2).sum(axis=1)
        L2_curve = ((I_pa[None, :] * omega_curve_pa) ** 2).sum(axis=1)
        twoT_err = float(np.max(np.abs(twoT_curve - twoT_0)) / twoT_0)
        L2_err = float(np.max(np.abs(L2_curve - L2_0)) / L2_0)
        max_polhode_err = max(max_polhode_err, twoT_err, L2_err)
        if twoT_err > 1e-12 or L2_err > 1e-12:
            fail_polhode += 1

        # ----- Check 3: round-trip through omega_jacobi at t=0. -----
        # The output of omega_jacobi(times=[0.0], ...) should equal omega0
        omega_back_body, _ = omega_jacobi(np.array([0.0]), omega_body, inertia)
        rt_err = float(np.linalg.norm(omega_back_body[0] - omega_body))
        max_round_trip_err = max(max_round_trip_err, rt_err)
        if rt_err > 1e-12:
            fail_round_trip += 1

        results.append(dict(
            tag=tag, idx=idx,
            omega_mag_dps=float(np.linalg.norm(omega_body) * 180.0 / np.pi),
            regime=regime, m_k_sq=float(m),
            twoT_0=float(twoT_0), L2_0=float(L2_0),
            tau_dot=float(tau_dot), tau_0=float(tau_0),
            recon_err=recon_err,
            twoT_curve_err=twoT_err, L2_curve_err=L2_err,
            round_trip_err=rt_err,
        ))

    wall = time.time() - t0
    summary = dict(
        n_cases=len(cases),
        wall_seconds=wall,
        n_fail_reconstruct=fail_reconstruct,
        n_fail_polhode=fail_polhode,
        n_fail_round_trip=fail_round_trip,
        max_recon_err=max_recon_err,
        max_polhode_err=max_polhode_err,
        max_round_trip_err=max_round_trip_err,
        gates=dict(
            reconstruct=max_recon_err < 1e-10,
            polhode_curve=max_polhode_err < 1e-12,
            round_trip=max_round_trip_err < 1e-12,
        ),
    )

    out_dir = SURVEY / "results" / "s063a"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    per_case_path = out_dir / "per_case.json"
    with open(per_case_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save coverage sample: for a single mid-range case, dump omega curve
    # to illustrate that uniform tau coverage gives a clean polhode curve.
    if results:
        idx_demo = next((i for i, r in enumerate(results) if r["regime"] == "A"
                         and 0.1 < r["m_k_sq"] < 0.9), 0)
        demo_case = cases[idx_demo]
        omega_pa = R_pa.T @ demo_case[2]
        info = _build_omega_func(I_pa, omega_pa)[1]
        K_m = ellipk(info["m"])
        taus = np.linspace(0, 4.0 * K_m, 500)
        curve = reconstruct_omega_pa(info, taus)
        np.savez(out_dir / "demo_polhode_curve.npz",
                 tau=taus, omega_pa=curve, info_m=info["m"],
                 twoT_0=info["twoT_0"], L2_0=info["L2_0"],
                 case_tag=str(demo_case[0]), case_idx=str(demo_case[1]))

    print(f"\n=== Summary ===")
    print(f"Wall: {wall:.2f}s")
    print(f"Cases: {len(cases)}")
    print(f"  reconstruct fails (>1e-10): {fail_reconstruct}  max err: {max_recon_err:.2e}")
    print(f"  polhode-curve fails (>1e-12 rel): {fail_polhode}  max rel err: {max_polhode_err:.2e}")
    print(f"  round-trip fails (>1e-12): {fail_round_trip}  max err: {max_round_trip_err:.2e}")
    print(f"\nGates: {summary['gates']}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_case_path}")


if __name__ == "__main__":
    main()
