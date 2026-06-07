#!/usr/bin/env python3
"""Identify Band-D failed seeds whose top-3 changes under surr_q0polish_mse.

For each candidate Band-D failed seed:
- geo_ckpt top-3 by geo_cost (current pipeline)
- geo_ckpt top-3 by surr_q0polish_mse (proposed replacement)
- whether either contains a truth-close (<20°) omega

Prints a table and writes target_seeds.json listing the seeds to rerun.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DIAG = ROOT / "data/results/inversion_diagnostics"
M103 = DIAG / "m103_hybrid_m048"
RR = DIAG / "rerank_experiment"

BAND_D_FAILED = [7, 8, 11, 16, 34, 47, 48, 51, 57, 59, 64, 67, 71, 78, 84, 89, 99]
TRUTH_CLOSE = 20.0


def pick_top3(scores):
    return list(map(int, np.argsort(scores)[:3]))


def seed_report(seed):
    ckpt = np.load(M103 / f"seed_{seed:03d}/geo_ckpt.npz")
    geo_costs = ckpt["geo_costs"]
    w0_ref_errs = ckpt["w0_ref_errs"]

    q0p_path = RR / f"seed_{seed:03d}_q0polish.json"
    if not q0p_path.exists():
        return {"seed": seed, "skip_reason": "no_q0polish"}
    q0p = json.load(open(q0p_path))
    q0polish_mse = np.asarray(q0p["surr_q0polish_mse"])

    n = min(len(geo_costs), len(w0_ref_errs), len(q0polish_mse))
    geo_costs = geo_costs[:n]
    w0_ref_errs = w0_ref_errs[:n]
    q0polish_mse = q0polish_mse[:n]

    top3_geo = pick_top3(geo_costs)
    top3_q0p = pick_top3(q0polish_mse)

    has_truth_in_pool = bool((w0_ref_errs < TRUTH_CLOSE).any())
    best_in_pool = float(w0_ref_errs.min())

    geo_errs = [float(w0_ref_errs[i]) for i in top3_geo]
    q0p_errs = [float(w0_ref_errs[i]) for i in top3_q0p]

    geo_has = min(geo_errs) < TRUTH_CLOSE
    q0p_has = min(q0p_errs) < TRUTH_CLOSE

    changed = set(top3_geo) != set(top3_q0p)

    return {
        "seed": seed,
        "n_cand": int(n),
        "has_truth_in_pool": has_truth_in_pool,
        "best_truth_err_in_pool": best_in_pool,
        "top3_by_geo_cost": top3_geo,
        "top3_geo_errs_deg": geo_errs,
        "geo_has_truth_close": geo_has,
        "top3_by_q0polish_mse": top3_q0p,
        "top3_q0p_errs_deg": q0p_errs,
        "q0p_has_truth_close": q0p_has,
        "top3_changed": changed,
        "predicted_rescue": (not geo_has) and q0p_has,
    }


def main():
    rows = []
    for s in BAND_D_FAILED:
        try:
            r = seed_report(s)
        except FileNotFoundError as e:
            r = {"seed": s, "skip_reason": str(e)}
        rows.append(r)

    print(f"{'seed':>5} {'nc':>3} {'pool':>5} {'tpool':>7}  "
          f"{'geo_top3_errs':>30} {'q0p_top3_errs':>30}  "
          f"{'chg':>4} {'resc':>5}")
    print("-" * 120)
    for r in rows:
        if "skip_reason" in r:
            print(f"{r['seed']:>5}  SKIP ({r['skip_reason']})")
            continue
        geo_str = ",".join(f"{x:5.1f}" for x in r["top3_geo_errs_deg"])
        q0p_str = ",".join(f"{x:5.1f}" for x in r["top3_q0p_errs_deg"])
        print(f"{r['seed']:>5} {r['n_cand']:>3} "
              f"{'Y' if r['has_truth_in_pool'] else 'n':>5} "
              f"{r['best_truth_err_in_pool']:>7.2f}  "
              f"{geo_str:>30} {q0p_str:>30}  "
              f"{'Y' if r['top3_changed'] else '.':>4} "
              f"{'YES' if r['predicted_rescue'] else '.':>5}")

    rescue = [r["seed"] for r in rows if r.get("predicted_rescue")]
    changed_but_not_rescue = [r["seed"] for r in rows
                              if r.get("top3_changed") and not r.get("predicted_rescue")]
    unchanged = [r["seed"] for r in rows
                 if r.get("top3_changed") is False]
    print()
    print(f"Predicted rescue (top-3 now has truth-close ω): {rescue}")
    print(f"Top-3 changed but no rescue predicted:           {changed_but_not_rescue}")
    print(f"Top-3 unchanged:                                  {unchanged}")

    out = RR / "target_seeds.json"
    with open(out, "w") as f:
        json.dump({
            "band_d_failed_seeds": BAND_D_FAILED,
            "truth_close_threshold_deg": TRUTH_CLOSE,
            "per_seed": rows,
            "rescue": rescue,
            "changed_but_not_rescue": changed_but_not_rescue,
            "unchanged": unchanged,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
