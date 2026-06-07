#!/usr/bin/env python3
"""
For each m103 geo_ckpt.npz in the project (m046 and m048 baselines), compare
the oracle-sort top-3 vs the geo_cost-sort top-3 candidate indices. Seeds
with identical top-3 sets were *untouched* by the m115 oracle bug even
before the 2026-04-21 fix. Seeds with divergent top-3 sets had their
m115 output contaminated — for those seeds historical Phase-B/Phase-A
OK/PARTIAL/FAIL classifications need re-running (or at least asterisking).
"""
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RES = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
OUT = RES / "m103_sort_divergence_audit.json"

BASES = {
    "m046": RES / "m103_hybrid",
    "m048": RES / "m103_hybrid_m048",
}

def audit_base(base: Path, n_top=3):
    rows = []
    for p in sorted(base.glob("seed_*/geo_ckpt.npz")):
        seed = int(p.parent.name.split("_")[-1])
        z = np.load(str(p), allow_pickle=True)
        if "geo_costs" not in z or "w0_ref_errs" not in z:
            rows.append({"seed": seed, "error": "missing keys"})
            continue
        gc = z["geo_costs"]
        we = z["w0_ref_errs"]
        order_geo = np.argsort(gc)[:n_top]
        order_or = np.argsort(we)[:n_top]
        set_geo = set(int(i) for i in order_geo)
        set_or = set(int(i) for i in order_or)
        overlap = set_geo & set_or
        rows.append({
            "seed": seed,
            "oracle_top3_idx": [int(i) for i in order_or],
            "oracle_top3_w_err": [float(we[i]) for i in order_or],
            "geo_cost_top3_idx": [int(i) for i in order_geo],
            "geo_cost_top3_w_err": [float(we[i]) for i in order_geo],
            "geo_cost_top3_cost": [float(gc[i]) for i in order_geo],
            "overlap_count": len(overlap),
            "set_identical": set_geo == set_or,
            "min_we_oracle": float(we[order_or[0]]),
            "min_we_geo_cost": float(we[order_geo[0]]),
        })
    return rows


def main():
    summary = {}
    for label, base in BASES.items():
        if not base.exists():
            summary[label] = {"error": f"no {base}"}
            continue
        rows = audit_base(base)
        identical = sum(1 for r in rows if r.get("set_identical"))
        partial = sum(1 for r in rows if not r.get("set_identical") and r.get("overlap_count", 0) > 0)
        disjoint = sum(1 for r in rows if r.get("overlap_count") == 0)
        print(f"\n== {label} — {len(rows)} seeds ==")
        print(f"  identical top-3:   {identical}")
        print(f"  partial overlap:   {partial}")
        print(f"  disjoint top-3:    {disjoint}")
        print(f"  seeds affected (not identical): "
              f"{sorted(r['seed'] for r in rows if not r.get('set_identical'))}")
        summary[label] = {
            "n_total": len(rows),
            "n_identical": identical,
            "n_partial_overlap": partial,
            "n_disjoint": disjoint,
            "affected_seeds": sorted(r["seed"] for r in rows if not r.get("set_identical")),
            "per_seed": rows,
        }

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
