#!/usr/bin/env python3
"""m119 multi-seed aggregator.

Reads each seed's summary.json and target_scores.npz, prints a table, and
saves a consolidated JSON. Answers the key question: does mean_L1 rank truth
at or near 0 across all 10 baseline seeds?
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)

SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]
BASE = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m119"

VARIANTS_OF_INTEREST = [
    "mean_L1",
    "mean_L2",
    "max_abs",
    "count_pass_020",
    "count_pass_050",
    "soft_pass_050",
]


def summarise_seed(seed: int):
    sdir = BASE / f"seed_{seed:03d}"
    summary_path = sdir / "summary.json"
    target_path = sdir / "target_scores.npz"
    if not summary_path.exists():
        return {"seed": seed, "status": "MISSING summary.json"}
    with summary_path.open() as f:
        summary = json.load(f)
    row = {
        "seed": seed,
        "poc_verdict": summary.get("poc_verdict"),
        "truth_resid_median": summary.get("truth_residual_median"),
        "truth_resid_p90": summary.get("truth_residual_p90"),
        "truth_resid_max": summary.get("truth_residual_max"),
        "truth_ranks": {v: summary["truth_ranks"].get(v)
                         for v in VARIANTS_OF_INTEREST
                         if v in summary.get("truth_ranks", {})},
        "runtime_s": summary.get("total_runtime_s"),
    }
    return row


def main():
    rows = [summarise_seed(s) for s in SEEDS]

    # Print table
    print("=" * 90)
    print(f"{'seed':>5} {'verdict':>9} {'med|r|':>8} {'p90|r|':>8} "
          f"{'L1rk':>7} {'L2rk':>7} {'maxrk':>7} {'cp020':>7} {'cp050':>7} {'sp050':>7}")
    print("-" * 90)
    for r in rows:
        if "status" in r:
            print(f"{r['seed']:>5}  {r['status']}")
            continue
        tr = r["truth_ranks"]
        print(f"{r['seed']:>5} {r['poc_verdict']:>9} "
              f"{r['truth_resid_median']:>8.3f} {r['truth_resid_p90']:>8.3f} "
              f"{tr.get('mean_L1', -1):>7} {tr.get('mean_L2', -1):>7} "
              f"{tr.get('max_abs', -1):>7} {tr.get('count_pass_020', -1):>7} "
              f"{tr.get('count_pass_050', -1):>7} {tr.get('soft_pass_050', -1):>7}")
    print("=" * 90)

    # Classification
    l1_ranks = [r["truth_ranks"].get("mean_L1", 10**9) for r in rows if "truth_ranks" in r]
    l2_ranks = [r["truth_ranks"].get("mean_L2", 10**9) for r in rows if "truth_ranks" in r]
    n_valid = sum(1 for x in l1_ranks if x < 10**9)
    n_l1_rank0 = sum(1 for x in l1_ranks if x == 0)
    n_l1_top100 = sum(1 for x in l1_ranks if x < 100)
    n_l1_top1000 = sum(1 for x in l1_ranks if x < 1000)

    print("\nmean_L1 truth-rank distribution across seeds:")
    print(f"  rank == 0:      {n_l1_rank0}/{n_valid}")
    print(f"  rank < 100:     {n_l1_top100}/{n_valid}")
    print(f"  rank < 1000:    {n_l1_top1000}/{n_valid}")
    if l1_ranks and min(l1_ranks) < 10**9:
        print(f"  min = {min(x for x in l1_ranks if x < 10**9)}, "
              f"max = {max(x for x in l1_ranks if x < 10**9)}")

    # Save consolidated JSON
    out = {
        "seeds": SEEDS,
        "rows": rows,
        "summary": {
            "n_seeds_run": n_valid,
            "n_L1_rank0": n_l1_rank0,
            "n_L1_top100": n_l1_top100,
            "n_L1_top1000": n_l1_top1000,
        },
    }
    out_path = BASE / "multiseed_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
