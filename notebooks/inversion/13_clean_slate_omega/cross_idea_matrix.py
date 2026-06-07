"""Cross-idea head-to-head matrix.

Reads candidate NPZs from the three sub-experiments and builds a per-seed
comparison table. No re-compute; pure aggregation.

Outputs:
    - FINDINGS-appended table + per-seed summary
    - data/results/inversion_diagnostics/13_clean_slate_omega/cross_idea_matrix.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.data import load_seed  # noqa: E402
from lib.scoring import RESIDUAL_MSE_GATE, RESIDUAL_MSE_TIGHT, classify  # noqa: E402

RESULTS_ROOT = (
    HERE.parents[2] / "data" / "results" / "inversion_diagnostics"
    / "13_clean_slate_omega"
)


def idea1_oracle(seed: int) -> dict | None:
    p = RESULTS_ROOT / "a_spectral" / f"seed{seed:03d}" / "pilot_oracleq0.npz"
    if not p.exists():
        return None
    d = np.load(p)
    mse = d["mse"]
    dir_deg = d["dir_err_deg"]
    best = int(mse.argmin())
    return {
        "path": str(p),
        "n_candidates_under_gate": int((mse < RESIDUAL_MSE_GATE).sum()),
        "n_candidates_under_tight": int((mse < RESIDUAL_MSE_TIGHT).sum()),
        "best_mse": float(mse[best]),
        "best_dir_err_deg": float(dir_deg[best]),
        "n_dirs": int(len(mse)),
    }


def idea1_joint(seed: int) -> dict | None:
    p = RESULTS_ROOT / "a_spectral" / f"seed{seed:03d}" / "pilot_jointq0.npz"
    if not p.exists():
        return None
    d = np.load(p)
    mse = d["mse_joint"]
    dir_deg = d["dir_err_deg"]
    best = int(mse.argmin())
    return {
        "path": str(p),
        "n_candidates_under_gate": int((mse < RESIDUAL_MSE_GATE).sum()),
        "best_mse": float(mse[best]),
        "best_dir_err_deg": float(dir_deg[best]),
        "n_dirs": int(len(mse)),
        "n_q0": int(d["q_grid"].shape[0]),
    }


def idea2_adam(seed: int) -> dict | None:
    p = RESULTS_ROOT / "b_differentiable" / f"seed{seed:03d}" / "candidates.npz"
    r = RESULTS_ROOT / "b_differentiable" / f"seed{seed:03d}" / "result.json"
    if not p.exists():
        return None
    d = np.load(p)
    mse = d["mse"]
    q_err = d["q_err_deg"]
    w_dir = d["w_dir_err_deg"]
    w_mag = d["w_mag_pct"]
    best = int(mse.argmin())
    out = {
        "path": str(p),
        "n_starts": int(len(mse)),
        "n_under_gate": int((mse < RESIDUAL_MSE_GATE).sum()),
        "n_under_tight": int((mse < RESIDUAL_MSE_TIGHT).sum()),
        "best_mse": float(mse[best]),
        "best_dir_err_deg": float(w_dir[best]),
        "best_mag_pct": float(w_mag[best]),
        "best_q_err_deg": float(q_err[best]),
        "truth_init_mse": float(mse[0]),  # start 0 is truth-init by convention
        "truth_init_q_err_deg": float(q_err[0]),
    }
    if r.exists():
        with open(r) as f:
            out["elapsed_s"] = float(json.load(f).get("elapsed_s", -1))
    return out


def idea3_mdn(seed: int) -> dict | None:
    p = RESULTS_ROOT / "c_learned_inverse" / f"seed{seed:03d}" / "candidates.npz"
    if not p.exists():
        return None
    d = np.load(p)
    mse = d["mse"]
    best = int(mse.argmin())
    return {
        "path": str(p),
        "n_candidates": int(len(mse)),
        "n_under_gate": int((mse < RESIDUAL_MSE_GATE).sum()),
        "best_mse": float(mse[best]),
    }


def main():
    seeds = [0, 23, 49, 69, 81]
    rows = []
    for s in seeds:
        b = load_seed(s)
        row = {
            "seed": s,
            "omega_mag_dps": float(b["omega_mag_dps"]),
            "phase_mean_deg": float(np.mean(b["phase_angle_3d"])),
            "idea1_oracle": idea1_oracle(s),
            "idea1_joint":  idea1_joint(s),
            "idea2_adam":   idea2_adam(s),
            "idea3_mdn":    idea3_mdn(s),
        }
        rows.append(row)

    out = {"seeds": seeds, "rows": rows}
    out_path = RESULTS_ROOT / "cross_idea_matrix.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")

    # Human-readable table
    print("\n=== CROSS-IDEA MATRIX ===")
    print(f"{'seed':>4} {'|ω|dps':>7} {'phase':>6} "
          f"{'I1.oracle.best':>14} {'I1.joint.best':>13} "
          f"{'I2.best':>8} {'I2.gate':>8} {'I2.tight':>9} {'I2.dir°':>8}")
    for r in rows:
        def fmt_idea(x, field):
            if x is None:
                return "—"
            return f"{x.get(field, '—'):.3f}" if isinstance(x.get(field), float) else str(x.get(field, "—"))

        o = r["idea1_oracle"]; j = r["idea1_joint"]; a = r["idea2_adam"]
        oracle_str = f"{o['best_mse']:.2f}@{o['best_dir_err_deg']:.1f}°" if o else "—"
        joint_str = f"{j['best_mse']:.2f}@{j['best_dir_err_deg']:.1f}°" if j else "—"
        a_best = f"{a['best_mse']:.4f}" if a else "—"
        a_gate = f"{a['n_under_gate']}/{a['n_starts']}" if a else "—"
        a_tight = f"{a['n_under_tight']}/{a['n_starts']}" if a else "—"
        a_dir = f"{a['best_dir_err_deg']:.1f}°" if a else "—"
        print(f"{r['seed']:>4} {r['omega_mag_dps']:>7.3f} {r['phase_mean_deg']:>6.1f} "
              f"{oracle_str:>14} {joint_str:>13} {a_best:>8} {a_gate:>8} {a_tight:>9} {a_dir:>8}")


if __name__ == "__main__":
    main()
