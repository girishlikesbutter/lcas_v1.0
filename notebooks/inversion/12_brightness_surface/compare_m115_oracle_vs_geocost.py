#!/usr/bin/env python3
"""
Compare m115 Phase-B (m048) outputs under oracle-sort (old) vs geo_cost sort
(fixed 2026-04-21) for the 6-seed cohort {23, 24, 49, 81, 90, 91}.

Reads:
  data/results/inversion_diagnostics/m115_surrogate_pipeline_m048_oracle_baseline/seed_NNN/result.json
  data/results/inversion_diagnostics/m115_surrogate_pipeline_m048/seed_NNN/result.json

Prints a table + writes JSON summary to
  data/results/inversion_diagnostics/m115_oracle_bug_audit_m048.json
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
ORACLE = RESULTS / "m115_surrogate_pipeline_m048_oracle_baseline"
FIXED = RESULTS / "m115_surrogate_pipeline_m048"
OUT = RESULTS / "m115_oracle_bug_audit_m048.json"

SEEDS = [23, 24, 49, 81, 90, 91]


def load(seed, base):
    p = base / f"seed_{seed:03d}" / "result.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def classify(hifi):
    if hifi is None:
        return "NO_DATA"
    if hifi < 0.01:
        return "OK"
    if hifi < 0.1:
        return "PARTIAL"
    return "FAIL"


def summary(d):
    if d is None:
        return {"cls": "NO_DATA"}
    hifi = d.get("best_hifi_mse")
    return {
        "cls_m115_internal": d.get("classification_old"),
        "cls_invert_scale": classify(hifi),
        "best_hifi_mse": hifi,
        "best_q0_err": d.get("best_q0_err"),
        "best_is_twin": d.get("best_is_twin"),
        "omegas_w_dir_err": [o.get("w_dir_err") for o in d.get("omegas", [])],
        "omegas_geo_cost": [o.get("geo_cost") for o in d.get("omegas", [])],
        "omegas_cand_idx": [o.get("cand_idx") for o in d.get("omegas", [])],
        "sort_by": d.get("omega_sort_by") or (
            d.get("omegas", [{}])[0].get("sort_by") if d.get("omegas") else None),
    }


def main():
    rows = []
    print("seed | oracle-sort                       | geo_cost-sort                     | DELTA")
    print("     |  cls  hifi_mse   q0°  idx   w_err |  cls  hifi_mse   q0°  idx   w_err |")
    print("-" * 110)
    for seed in SEEDS:
        d_o = load(seed, ORACLE)
        d_g = load(seed, FIXED)
        so = summary(d_o)
        sg = summary(d_g)

        def fmt(s):
            cls = s.get("cls_invert_scale", "?")[:4]
            hm = s.get("best_hifi_mse")
            q0 = s.get("best_q0_err")
            idx = s.get("omegas_cand_idx", []) or []
            we = s.get("omegas_w_dir_err", []) or []
            idx_s = ",".join(str(i) for i in idx[:3])
            we_s = ",".join(f"{x:.1f}" for x in we[:3] if x is not None)
            return (f"{cls:>4} "
                    f"{hm:>8.5f} " if hm is not None else f"{cls:>4} {'?':>8} ") + \
                   (f"{q0:>5.1f} " if q0 is not None else f"{'?':>5} ") + \
                   f"{idx_s:>8} {we_s:>16}"

        o_cls = so.get("cls_invert_scale", "?")
        g_cls = sg.get("cls_invert_scale", "?")
        delta = "SAME" if o_cls == g_cls else f"{o_cls}→{g_cls}"
        print(f" {seed:3d} | {fmt(so)} | {fmt(sg)} | {delta}")
        rows.append({
            "seed": seed,
            "oracle": so,
            "geo_cost": sg,
            "verdict_changed": (o_cls != g_cls),
        })

    cohort = {
        "oracle_sort": {
            "OK": sum(1 for r in rows if r["oracle"].get("cls_invert_scale") == "OK"),
            "PARTIAL": sum(1 for r in rows if r["oracle"].get("cls_invert_scale") == "PARTIAL"),
            "FAIL": sum(1 for r in rows if r["oracle"].get("cls_invert_scale") == "FAIL"),
        },
        "geo_cost_sort": {
            "OK": sum(1 for r in rows if r["geo_cost"].get("cls_invert_scale") == "OK"),
            "PARTIAL": sum(1 for r in rows if r["geo_cost"].get("cls_invert_scale") == "PARTIAL"),
            "FAIL": sum(1 for r in rows if r["geo_cost"].get("cls_invert_scale") == "FAIL"),
        },
    }
    print()
    print(f"Cohort (invert.py scale: <0.01 OK, <0.1 PARTIAL, else FAIL):")
    print(f"  oracle-sort:  {cohort['oracle_sort']}")
    print(f"  geo_cost-sort: {cohort['geo_cost_sort']}")

    with open(OUT, "w") as f:
        json.dump({"per_seed": rows, "cohort": cohort}, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
