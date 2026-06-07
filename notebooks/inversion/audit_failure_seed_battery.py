#!/usr/bin/env python3
"""Compact audit of the 2026-04-28 failure-seed battery.

Reads m103 lofi_ckpt.npz + lofi_surr_ckpt.npz + constrained_anchor_ckpt(_oracle/_honest).npz
for seeds 47, 51, 79, 84, 89 on m048 and prints a single table.

For each seed:
  - lofi pool: closest ω in pool to truth (sets the bridging-radius ceiling)
  - rank-1 ω-direction error under: align_cost (m103 default), lofi_mse,
    surrogate full-LC MSE
  - constrained anchor: rank-1 ω-direction error in oracle and honest modes
  - "rescuable?": does any rank-1 land within m115's 5° bridging radius?
"""
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAG = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
SEEDS = [47, 51, 79, 84, 89]


def signed_dir_err(w_arr, w_truth_dir):
    w_dir = w_arr / np.linalg.norm(w_arr, axis=-1, keepdims=True)
    cos = np.clip(w_dir @ w_truth_dir, -1, 1)
    return np.degrees(np.arccos(cos))


def safe_load(path):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def audit():
    master = np.load(DIAG / "m048_trajectories" / "m048_trajectories.npz",
                     allow_pickle=True)
    rows = []
    for seed in SEEDS:
        sd = DIAG / "m103_hybrid_m048" / f"seed_{seed:03d}"
        true_omega = master["omega0s"][seed]
        true_w_dir = true_omega / np.linalg.norm(true_omega)

        lofi = safe_load(sd / "lofi_ckpt.npz")
        lofi_surr = safe_load(sd / "lofi_surr_ckpt.npz")
        ca_oracle = safe_load(sd / "constrained_anchor_ckpt_oracle.npz")
        ca_honest = safe_load(sd / "constrained_anchor_ckpt_honest.npz")

        row = {"seed": seed}
        if lofi is not None:
            err = signed_dir_err(lofi["w0"], true_w_dir)
            row["pool_min_deg"] = float(err.min())
            row["align_top1_deg"] = float(err[np.argmin(lofi["align_cost"])])
            row["lofimse_top1_deg"] = float(err[np.argmin(lofi["lofi_mse"])])
            if lofi_surr is not None:
                surr_mse = lofi_surr["surr_mse"]
                if np.isfinite(surr_mse).any():
                    masked = np.where(np.isfinite(surr_mse), surr_mse, np.inf)
                    row["surr_top1_deg"] = float(err[np.argmin(masked)])

        for key, ck in [("anchor_oracle_top1_deg", ca_oracle),
                        ("anchor_honest_top1_deg", ca_honest)]:
            if ck is None:
                continue
            err = signed_dir_err(ck["omega_dirs"], true_w_dir)
            mse = ck["min_surr_mse"]
            if np.isfinite(mse).any():
                masked = np.where(np.isfinite(mse), mse, np.inf)
                row[key] = float(err[np.argmin(masked)])

        rows.append(row)

    # Print
    cols = ["seed", "pool_min_deg", "align_top1_deg", "lofimse_top1_deg",
            "surr_top1_deg", "anchor_oracle_top1_deg", "anchor_honest_top1_deg"]
    header = f"{'seed':>5}  {'pool_min':>9}  {'align_t1':>9}  {'lofimse_t1':>11}  " \
             f"{'surr_t1':>9}  {'anchor_orc':>11}  {'anchor_hst':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['seed']:>5}  "
        for c in cols[1:]:
            v = r.get(c)
            line += (f"{v:>9.2f}  " if c == "pool_min_deg" else
                     f"{v:>9.2f}  " if c.endswith("top1_deg") and "anchor" not in c else
                     f"{v:>11.2f}  ") if v is not None else (f"{'--':>9}  " if "anchor" not in c else f"{'--':>11}  ")
        print(line)

    # Rescue tally
    print()
    bridging = 5.0  # m115 DE bridging radius
    print(f"Within m115 bridging radius (≤{bridging}°):")
    for col, label in [("align_top1_deg", "  m103 align cost (current)"),
                       ("lofimse_top1_deg", "  lofi_mse rank"),
                       ("surr_top1_deg", "  surrogate-LC rank (Finding 2)"),
                       ("anchor_oracle_top1_deg", "  constrained-anchor (ORACLE |ω|)"),
                       ("anchor_honest_top1_deg", "  constrained-anchor (HONEST)")]:
        hits = sum(1 for r in rows
                   if r.get(col) is not None and r[col] <= bridging)
        n_have = sum(1 for r in rows if r.get(col) is not None)
        print(f"{label:<40s}: {hits}/{n_have}")

    # Save JSON
    out = DIAG / "failure_seed_battery_2026_04_28" / "audit_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    audit()
