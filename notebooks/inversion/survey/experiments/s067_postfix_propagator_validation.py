"""s067 — Post-fix propagator validation: SPICE pxform + L_J2000 + Jacobi cross-check.

Canonical regression test for the 2026-05-12 propagator sign fix.

Bundles four gates:

  Gate A1 — SPICE direction-aware comparison.
    Feed `propagate_euler` the textbook body-frame omega (finite-diff of
    pxform). PASS: ||R_prop - R_SPICE|| at dt=1s < 1e-9.

  Gate A2 — L_J2000 conservation on a controlled toy (tight tolerances).
    I=diag(1,2,3), omega0=(0.5,0.3,0.7), q0=identity, dt=50s,
    rtol=1e-12, atol=1e-14.
    PASS: max ||L_J2000(t) - L_J2000(0)|| < 1e-10.

  Gate A3 — L_J2000 conservation on m048 cohort (3 representative seeds).
    Seeds 89 (slow), 28 (fast), 14 (near-separatrix), full LC,
    default tolerances (rtol=1e-10, atol=1e-12).
    PASS: per-seed drift < 1e-6 (DOP853 noise floor over 60 min at the
    cohort's |omega| range; pre-fix drift was on order 1e+2 / 36-137% rel).

  Gate A4 — Jacobi (closed-form omega + DOP853 on q) ↔ propagate_euler parity.
    Both share the same q ODE post-fix; should match to integration noise.
    PASS: max |q_jacobi - q_codebase| (antipode-aware) < 1e-7 across 3 seeds.

Exit code: 0 if all gates pass, 1 otherwise. Becomes the canonical pre-merge
check for any future propagator change. Wall ~10s.

Usage:
    python experiments/s067_postfix_propagator_validation.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import spiceypy as spice
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

from lib.traj_load import load_truth  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402
from lib.jacobi_propagator import propagate_jacobi  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude, propagate_euler  # noqa: E402
from src.spice.spice_handler import SpiceHandler  # noqa: E402

OUT = SURVEY / "results" / "s067_postfix_validation"
OUT.mkdir(parents=True, exist_ok=True)


def R_of_q(q_wxyz):
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def antipode_aware_q_diff(q_a, q_b):
    """Largest antipode-aware difference across a (N, 4) trajectory pair."""
    sign = np.where(np.sum(q_a * q_b, axis=1) < 0, -1.0, 1.0)
    return float(np.max(np.linalg.norm(q_a - sign[:, None] * q_b, axis=1)))


def gate_a1_spice():
    """Gate A1 — SPICE direction-aware comparison via IS-901 kernel."""
    metakernel = SURVEY.parent.parent.parent / "data" / "spice_kernels" / \
        "missions" / "dst-is901" / "INTELSAT_901-metakernel.tm"
    sh = SpiceHandler()
    sh.load_metakernel_programmatically(str(metakernel))

    t0 = sh.utc_to_et("2020-02-05T10:00:00")
    dt_omega = 0.01
    R0 = np.asarray(spice.pxform("J2000", "IS901_BUS_FRAME", t0))
    R0p = np.asarray(spice.pxform("J2000", "IS901_BUS_FRAME", t0 + dt_omega))

    q_xyzw = Rotation.from_matrix(R0).as_quat()
    q0_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    # Textbook body-frame omega from SPICE: dR/dt = -[omega]_x R for passive J2000->body
    om_skew = -((R0p - R0) / dt_omega) @ R0.T
    omega = np.array([om_skew[2, 1], om_skew[0, 2], om_skew[1, 0]])

    I_test = np.eye(3) * 1000.0
    test_dts = np.array([0.0, 1.0, 10.0, 60.0])

    q_hist, _ = propagate_euler(q0_wxyz, +omega, I_test, test_dts,
                                rtol=1e-12, atol=1e-14)

    err_table = []
    err_at_1s = None
    for i, dt in enumerate(test_dts):
        R_ref = np.asarray(spice.pxform("J2000", "IS901_BUS_FRAME", t0 + dt))
        q = q_hist[i]
        R_prop = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        d = float(np.linalg.norm(R_prop - R_ref))
        err_table.append({"dt_s": float(dt), "err": d})
        if dt == 1.0:
            err_at_1s = d

    spice.kclear()
    passed = err_at_1s is not None and err_at_1s < 1e-9
    return passed, {"err_at_1s": err_at_1s, "err_table": err_table,
                    "criterion": "err@1s < 1e-9", "omega_textbook_dps":
                    (omega * 180.0 / np.pi).tolist()}


def gate_a2_toy():
    """Gate A2 — L_J2000 toy."""
    I = np.diag([1.0, 2.0, 3.0])
    omega0 = np.array([0.5, 0.3, 0.7])
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    times = np.linspace(0.0, 50.0, 51)

    q_hist, om_hist = propagate_attitude(
        q0=q0, omega0=omega0, times=times, mode="tumbling",
        inertia_tensor=I, rtol=1e-12, atol=1e-14,
    )

    L_J = np.array([R_of_q(q_hist[i]).T @ (I @ om_hist[i]) for i in range(len(times))])
    drift = np.linalg.norm(L_J - L_J[0], axis=1)
    drift_max = float(drift.max())
    L0_norm = float(np.linalg.norm(L_J[0]))
    drift_rel = drift_max / L0_norm * 100.0

    passed = drift_max < 1e-10
    return passed, {"drift_abs_max": drift_max, "drift_rel_pct": drift_rel,
                    "L_body_mag": L0_norm, "criterion": "drift_abs_max < 1e-10"}


def gate_a3_cohort(seeds=(89, 28, 14)):
    """Gate A3 — L_J2000 on m048 cohort (3 seeds)."""
    ctx = build_context(seed=int(seeds[0]))
    inertia = np.asarray(ctx["inertia_tensor"])

    per_seed = {}
    all_pass = True
    for seed in seeds:
        d = load_truth(seed)
        q0 = np.asarray(d["q0_wxyz"])
        om0 = np.asarray(d["omega0_rad"])
        times = np.asarray(d["observation_times"])

        q_hist, om_hist = propagate_attitude(
            q0=q0, omega0=om0, times=times, mode="tumbling",
            inertia_tensor=inertia,
        )

        sample_idx = [0, 1, len(times)//4, len(times)//2,
                      3*len(times)//4, len(times)-1]
        L_J = np.array([R_of_q(q_hist[i]).T @ (inertia @ om_hist[i])
                        for i in sample_idx])
        drift = np.linalg.norm(L_J - L_J[0], axis=1)
        drift_max = float(drift.max())
        L0_norm = float(np.linalg.norm(L_J[0]))
        drift_rel = drift_max / L0_norm * 100.0
        seed_pass = drift_max < 1e-6
        all_pass = all_pass and seed_pass
        per_seed[int(seed)] = {
            "omega_mag_dps": float(np.linalg.norm(om0) * 180.0 / np.pi),
            "duration_min": float((times[-1] - times[0]) / 60.0),
            "L_body_mag": L0_norm,
            "drift_abs_max": drift_max,
            "drift_rel_pct": drift_rel,
            "passed": seed_pass,
        }

    return all_pass, {"per_seed": per_seed, "criterion": "drift_abs_max < 1e-6 per seed"}


def gate_a4_jacobi(seeds=(89, 28, 14)):
    """Gate A4 — Jacobi hybrid ↔ propagate_euler parity."""
    ctx = build_context(seed=int(seeds[0]))
    inertia = np.asarray(ctx["inertia_tensor"])

    per_seed = {}
    all_pass = True
    for seed in seeds:
        d = load_truth(seed)
        q0 = np.asarray(d["q0_wxyz"])
        om0 = np.asarray(d["omega0_rad"])
        times = np.asarray(d["observation_times"])

        q_codebase, _ = propagate_attitude(
            q0=q0, omega0=om0, times=times, mode="tumbling",
            inertia_tensor=inertia,
        )
        q_jacobi, _ = propagate_jacobi(q0, om0, inertia, times)

        diff = antipode_aware_q_diff(q_codebase, q_jacobi)
        seed_pass = diff < 1e-7
        all_pass = all_pass and seed_pass
        per_seed[int(seed)] = {"q_diff_max": diff, "passed": seed_pass}

    return all_pass, {"per_seed": per_seed, "criterion": "q_diff_max < 1e-7 per seed"}


def main():
    print("=" * 70)
    print("s067 — Post-fix propagator validation")
    print("=" * 70)

    results = {}
    overall_pass = True

    print("\n[Gate A1] SPICE direction-aware comparison ...")
    p1, r1 = gate_a1_spice()
    results["A1_spice_pxform"] = {"passed": p1, **r1}
    overall_pass = overall_pass and p1
    print(f"  err@1s = {r1['err_at_1s']:.3e}  ({r1['criterion']})  ->  "
          f"{'PASS' if p1 else 'FAIL'}")
    for row in r1["err_table"]:
        print(f"    dt={row['dt_s']:5.1f}s  err={row['err']:.3e}")

    print("\n[Gate A2] L_J2000 conservation on toy ...")
    p2, r2 = gate_a2_toy()
    results["A2_toy_LJ2000"] = {"passed": p2, **r2}
    overall_pass = overall_pass and p2
    print(f"  drift_abs_max = {r2['drift_abs_max']:.3e}  "
          f"({r2['drift_rel_pct']:.4f}% of |L_body|)  ->  "
          f"{'PASS' if p2 else 'FAIL'}")

    print("\n[Gate A3] L_J2000 conservation on m048 cohort (3 seeds) ...")
    p3, r3 = gate_a3_cohort()
    results["A3_cohort_LJ2000"] = {"passed": p3, **r3}
    overall_pass = overall_pass and p3
    for s, r in r3["per_seed"].items():
        print(f"  seed {s:03d}: drift {r['drift_abs_max']:.3e} "
              f"({r['drift_rel_pct']:.4f}%)  ->  "
              f"{'PASS' if r['passed'] else 'FAIL'}")

    print("\n[Gate A4] Jacobi ↔ propagate_euler parity (3 seeds) ...")
    p4, r4 = gate_a4_jacobi()
    results["A4_jacobi_parity"] = {"passed": p4, **r4}
    overall_pass = overall_pass and p4
    for s, r in r4["per_seed"].items():
        print(f"  seed {s:03d}: q_diff_max = {r['q_diff_max']:.3e}  ->  "
              f"{'PASS' if r['passed'] else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)

    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(
        {"overall_pass": overall_pass, "gates": results}, indent=2))
    print(f"Saved: {summary_path}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
