"""s067b — Gate B: cohort-wide L_J2000 audit on the post-fix m048 cohort.

After the propagator fix (commit d5705ff) the cohort was regenerated with the
same RNG seeds. This script propagates each cached (q0, omega0) under the
fixed propagator and verifies L_J2000 = R(t).T @ I @ omega(t) is conserved
across the full LC.

Pass criterion: per-seed max ||L_J2000(t) - L_J2000(0)|| < 5e-6 (DOP853 noise
floor for default tolerances over 60 min at the cohort's |omega| range).

Pre-fix this drifted 36-137% of |L_body| — see s066. Post-fix should be
~0% relative drift across all 120 seeds.

Usage:
    python experiments/s067b_cohort_lj2000_audit.py [N_SEEDS]   # default 120
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
from multiprocessing import Pool
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

from lib.traj_load import load_truth  # noqa: E402
from lib.hifi_render import build_context  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402

OUT = SURVEY / "results" / "s067b_cohort_audit"
OUT.mkdir(parents=True, exist_ok=True)


def R_of_q(q_wxyz):
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def audit_one(args):
    seed, inertia = args
    try:
        d = load_truth(seed)
    except Exception as e:
        return seed, None, f"load failed: {e}"
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
    return seed, {
        "omega_mag_dps": float(np.linalg.norm(om0) * 180.0 / np.pi),
        "duration_min": float((times[-1] - times[0]) / 60.0),
        "L_body_mag": L0_norm,
        "drift_abs_max": drift_max,
        "drift_rel_pct": drift_rel,
        "passed": drift_max < 5e-6,
    }, None


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    print(f"=== s067b — cohort L_J2000 audit on {n_seeds} seeds ===\n")

    ctx = build_context(seed=0)
    inertia = np.asarray(ctx["inertia_tensor"])
    print(f"  I_body diag (kg m^2): {np.diag(inertia)}")
    print(f"  Pass criterion: drift_abs_max < 5e-6 per seed\n")

    args = [(s, inertia) for s in range(n_seeds)]
    per_seed = {}
    failures = []
    with Pool(8) as pool:
        for seed, result, err in pool.imap_unordered(audit_one, args):
            if err is not None:
                failures.append((seed, err))
                continue
            per_seed[seed] = result
            tag = "PASS" if result["passed"] else "FAIL"
            print(f"  seed {seed:03d}  drift {result['drift_abs_max']:.3e}  "
                  f"({result['drift_rel_pct']:.5f}%)  |w|={result['omega_mag_dps']:.3f}  -> {tag}")

    n_pass = sum(1 for r in per_seed.values() if r["passed"])
    n_fail = sum(1 for r in per_seed.values() if not r["passed"])
    drifts = sorted([r["drift_abs_max"] for r in per_seed.values()])
    drift_max = max(drifts) if drifts else 0.0
    drift_p99 = drifts[int(0.99 * len(drifts))] if drifts else 0.0
    drift_med = drifts[len(drifts)//2] if drifts else 0.0

    print(f"\n{'='*60}")
    print(f"COHORT AUDIT SUMMARY ({n_seeds} seeds):")
    print(f"  PASS: {n_pass}/{n_seeds}    FAIL: {n_fail}/{n_seeds}    LOAD-ERR: {len(failures)}")
    print(f"  drift_abs distribution: median={drift_med:.3e}, p99={drift_p99:.3e}, max={drift_max:.3e}")
    if failures:
        print(f"  Load failures:")
        for s, e in failures[:5]:
            print(f"    seed {s}: {e}")
    overall_pass = (n_fail == 0) and (len(failures) == 0)
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*60}")

    summary = {
        "n_seeds": n_seeds, "n_pass": n_pass, "n_fail": n_fail,
        "n_load_err": len(failures),
        "drift_median": drift_med, "drift_p99": drift_p99, "drift_max": drift_max,
        "overall_pass": overall_pass,
        "criterion": "per-seed drift_abs_max < 5e-6",
        "per_seed": {int(s): r for s, r in per_seed.items()},
        "failures": [{"seed": s, "err": e} for s, e in failures],
    }
    out = OUT / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
