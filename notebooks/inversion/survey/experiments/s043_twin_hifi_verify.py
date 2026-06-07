"""s043 — verify body-twin LC equivalence via hi-fi forward rendering.

For each seed in {23, 28, 89}: render hi-fi LC at (q0_truth, ω_truth) and at
(q_180x ⊗ q0_truth, R_180x · ω_truth). Compute per-epoch difference. Report
max/RMS against the 0.05 mag photometric noise floor.

If max |Δmag| < 0.05 mag, the body-twin is operationally exact under the
hi-fi forward model and search-space deduplication is safe. If max
|Δmag| > 0.05 mag, the small inertia asymmetry (I_xx 37985 vs I_yy 38306
≈ 0.84% off) produces measurable LC drift over the obs window and we'd
need to track twin separately.

Outputs results/s043_twin_hifi_verify/{seed}_diff.npz + summary.json.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from lib.hifi_render import build_context, render_hifi, rho_from_hifi

SEEDS = [23, 28, 89]
OUT_DIR = SURVEY_DIR / "results" / "s043_twin_hifi_verify"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def main():
    print(f"=== s043 twin hi-fi verification | seeds={SEEDS} ===", flush=True)
    summary = {"seeds": {}, "noise_floor_mag": 0.05}
    q_180x = np.array([0.0, 1.0, 0.0, 0.0])
    R_180x = np.diag([1.0, -1.0, -1.0])

    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        ctx = build_context(seed)
        traj = np.load(SURVEY_DIR / "data" / "trajectories"
                       / f"traj_seed{seed:03d}.npz")
        q0_truth = traj["q0_wxyz"].astype(np.float64)
        omega_truth = traj["omega0_rad"].astype(np.float64)
        truth_cached = traj["mag_hifi"].astype(np.float64)

        q0_twin = quat_mul(q_180x, q0_truth)
        omega_twin = R_180x @ omega_truth

        t0 = time.time()
        mag_truth = render_hifi(q0_truth, omega_truth, ctx)
        t_truth = time.time() - t0
        t0 = time.time()
        mag_twin = render_hifi(q0_twin, omega_twin, ctx)
        t_twin = time.time() - t0

        # cached round-trip check (truth vs cached)
        valid = np.isfinite(truth_cached) & np.isfinite(mag_truth) & np.isfinite(mag_twin)
        n_valid = int(valid.sum())
        truth_vs_cached = mag_truth[valid] - truth_cached[valid]
        truth_vs_twin = mag_truth[valid] - mag_twin[valid]

        max_truth_cached = float(np.max(np.abs(truth_vs_cached)))
        rms_truth_cached = float(np.sqrt(np.mean(truth_vs_cached**2)))
        max_diff = float(np.max(np.abs(truth_vs_twin)))
        rms_diff = float(np.sqrt(np.mean(truth_vs_twin**2)))
        rho = float(np.sqrt(np.mean(truth_vs_twin**2)) / 0.05)

        print(f"  render times: truth {t_truth:.1f}s, twin {t_twin:.1f}s", flush=True)
        print(f"  truth vs cached: max |Δ|={max_truth_cached:.2e}, "
              f"RMS={rms_truth_cached:.2e} (round-trip sanity)", flush=True)
        print(f"  TRUTH vs TWIN  : max |Δ|={max_diff:.2e}, RMS={rms_diff:.2e}, "
              f"ρ={rho:.4f} (target ρ < 1)", flush=True)
        # Per-epoch worst offenders
        order = np.argsort(np.abs(truth_vs_twin))[-5:][::-1]
        print(f"  worst 5 epochs: " + " ".join(
            f"e={int(np.where(valid)[0][i])} Δ={truth_vs_twin[i]:+.4e}" for i in order),
            flush=True)

        np.savez(OUT_DIR / f"seed{seed:03d}_diff.npz",
                 mag_truth=mag_truth, mag_twin=mag_twin,
                 truth_vs_cached=truth_vs_cached, truth_vs_twin=truth_vs_twin,
                 valid_mask=valid)

        summary["seeds"][f"seed{seed:03d}"] = {
            "n_valid": n_valid,
            "max_truth_vs_cached_mag": max_truth_cached,
            "rms_truth_vs_cached_mag": rms_truth_cached,
            "max_truth_vs_twin_mag": max_diff,
            "rms_truth_vs_twin_mag": rms_diff,
            "rho_truth_vs_twin": rho,
            "render_truth_s": t_truth,
            "render_twin_s": t_twin,
            "verdict_op_exact": max_diff < 0.05,
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[s043] Saved: {OUT_DIR / 'summary.json'}", flush=True)
    # Headline
    all_op_exact = all(s["verdict_op_exact"] for s in summary["seeds"].values())
    max_diff_cohort = max(s["max_truth_vs_twin_mag"] for s in summary["seeds"].values())
    print(f"\n=== VERDICT ===", flush=True)
    print(f"  Cohort max |truth - twin| = {max_diff_cohort:.2e} mag", flush=True)
    print(f"  Photometric noise floor    = 0.05 mag", flush=True)
    print(f"  Operationally exact (all seeds)? {all_op_exact}", flush=True)


if __name__ == "__main__":
    main()
