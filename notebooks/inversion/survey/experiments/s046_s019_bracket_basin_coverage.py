"""s046 — re-score s019's five LS-bracket strategies against the s042 basin rule.

s019 measured per-seed nearest-cell-pct for five ω-mag bracket strategies on
the 100 post-fix m048 cohort. The original headline (`bracket: 98/100 within
5%`) was scored against the m138-era ±5% bar, which is too lenient for the
post-fix basin reality (s042 cohort scaling rule says required basin can be
0.5–3% depending on |ω|).

This script re-scores all five strategies against the s042 conservative rule
(c=1% / a=0.5) and, for context, the lenient (c=2% / a=1.0) and tight
(c=0.5% / a=1.0) variants. Pure-math on cached data, no compute.

Closes the question s045 left open: does the s019 LS-bracket post-fix prior
deliver per-seed accuracy good enough that the s045 per-seed adaptive bracket
architecture is feasible?

Outputs:
    results/s046_s019_bracket_basin_coverage/summary.json
    results/s046_s019_bracket_basin_coverage/per_seed.csv
    results/s046_s019_bracket_basin_coverage/run.log
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_DIR))

S019_PATH = SURVEY_DIR / "results" / "s019" / "summary.json"
OUT_DIR = SURVEY_DIR / "results" / "s046_s019_bracket_basin_coverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Five bracket strategies measured by s019.
STRATEGIES = ["pc", "ls", "bracket", "multi", "harm"]
GRID_SIZE_KEY = {
    "pc": None,  # 20 by construction; s019 doesn't store it per-seed
    "ls": None,  # 20 by construction
    "bracket": "bracket_grid_size",
    "multi": "multi_grid_size",
    "harm": "harm_grid_size",
}
PCT_KEY = {s: f"{s}_nearest_pct" for s in STRATEGIES}
PCT_KEY["ls"] = "ls_nearest_pct"  # explicit
WITHIN_5_KEY = {s: f"{s}_n_within_5pct" for s in STRATEGIES}

# Cohort scaling rules from s042 / wind-down (c% / |ω|^a). Lower number = tighter rule.
RULES = {
    "tight (c=0.5%/a=1.0)": (0.5, 1.0),
    "conservative (c=1%/a=0.5)": (1.0, 0.5),
    "lenient (c=2%/a=1.0)": (2.0, 1.0),
}


def basin_pct(omega_dps: float, c: float, a: float) -> float:
    """Required basin half-width as % of |ω|."""
    return c / (omega_dps ** a)


def main():
    print("=== s046 s019-bracket basin-coverage re-score ===", flush=True)
    s019 = json.loads(S019_PATH.read_text())
    per_seed_in = s019["per_seed"]
    print(
        f"Loaded s019: {len(per_seed_in)} seeds, "
        f"strategies={STRATEGIES}",
        flush=True,
    )

    # --------------------------------------------------------------- #
    # Build per-seed table.
    # --------------------------------------------------------------- #
    rows = []
    for sk, rec in per_seed_in.items():
        seed = int(rec["seed"])
        omega_dps = float(rec["truth_mag_dps"])
        row = {
            "seed": seed,
            "omega_mag_dps": omega_dps,
            "n_significant_peaks": int(rec.get("n_significant_peaks", -1)),
        }
        for strat in STRATEGIES:
            row[f"{strat}_nearest_pct"] = float(rec[PCT_KEY[strat]])
            row[f"{strat}_n_within_5pct"] = int(rec[WITHIN_5_KEY[strat]])
            gk = GRID_SIZE_KEY[strat]
            row[f"{strat}_grid_size"] = (
                int(rec[gk]) if gk and gk in rec else (20 if strat in ("pc", "ls") else None)
            )
        # Basin per scaling rule.
        for rule_name, (c, a) in RULES.items():
            row[f"basin_{rule_name}"] = basin_pct(omega_dps, c, a)
            for strat in STRATEGIES:
                row[f"{strat}_pass_{rule_name}"] = (
                    row[f"{strat}_nearest_pct"] <= row[f"basin_{rule_name}"]
                )
        rows.append(row)

    rows.sort(key=lambda r: r["seed"])
    n_total = len(rows)

    # --------------------------------------------------------------- #
    # Aggregate: pass count and offset distribution per strategy × rule.
    # --------------------------------------------------------------- #
    print("\n[1] Pass count per strategy × scaling rule:", flush=True)
    print(
        f"   {'strategy':>10} {'5% bar':>8} "
        + "".join(f"{rn:>30}" for rn in RULES.keys()),
        flush=True,
    )
    aggregate = {}
    for strat in STRATEGIES:
        five_pct_pass = sum(1 for r in rows if r[f"{strat}_nearest_pct"] <= 5.0)
        aggregate[strat] = {"5pct_bar": five_pct_pass}
        line = f"   {strat:>10} {five_pct_pass:>8d}"
        for rule_name in RULES.keys():
            n_pass = sum(1 for r in rows if r[f"{strat}_pass_{rule_name}"])
            aggregate[strat][rule_name] = n_pass
            line += f"{n_pass:>30d}"
        print(line, flush=True)

    # --------------------------------------------------------------- #
    # Distribution of nearest_offset_pct per strategy.
    # --------------------------------------------------------------- #
    print("\n[2] Nearest-cell-offset distribution per strategy (% of truth-|ω|):", flush=True)
    print(
        f"   {'strategy':>10} {'p10':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8}",
        flush=True,
    )
    distribution = {}
    for strat in STRATEGIES:
        offsets = sorted(r[f"{strat}_nearest_pct"] for r in rows)
        pcts = {p: float(np.percentile(offsets, p)) for p in (10, 25, 50, 75, 90, 95)}
        distribution[strat] = pcts
        print(
            f"   {strat:>10} "
            + " ".join(f"{pcts[p]:8.2f}" for p in (10, 25, 50, 75, 90, 95)),
            flush=True,
        )

    # --------------------------------------------------------------- #
    # |ω|-stratified pass rate for the strongest strategy ('bracket').
    # --------------------------------------------------------------- #
    print("\n[3] |ω|-stratified pass rate for 'bracket' under conservative rule (c=1%/a=0.5):", flush=True)
    omegas = sorted(r["omega_mag_dps"] for r in rows)
    quartiles = [
        ("Q1 (slowest)", float(np.percentile(omegas, 25))),
        ("Q2", float(np.percentile(omegas, 50))),
        ("Q3", float(np.percentile(omegas, 75))),
    ]
    bins = [
        (0.0, quartiles[0][1], "Q1 (|ω|≤p25)"),
        (quartiles[0][1], quartiles[1][1], "Q2 (p25-p50)"),
        (quartiles[1][1], quartiles[2][1], "Q3 (p50-p75)"),
        (quartiles[2][1], float("inf"), "Q4 (fastest, |ω|>p75)"),
    ]
    rule_name = "conservative (c=1%/a=0.5)"
    omega_strat = {}
    for lo, hi, label in bins:
        in_bin = [r for r in rows if lo < r["omega_mag_dps"] <= hi]
        if not in_bin:
            continue
        n_pass = sum(1 for r in in_bin if r[f"bracket_pass_{rule_name}"])
        omega_strat[label] = {
            "n_total": len(in_bin),
            "n_pass": n_pass,
            "median_offset_pct": float(
                np.median([r["bracket_nearest_pct"] for r in in_bin])
            ),
            "median_basin_pct": float(
                np.median([r[f"basin_{rule_name}"] for r in in_bin])
            ),
        }
        print(
            f"   {label:>26}: pass {n_pass:>3d}/{len(in_bin):<3d}  "
            f"median offset={omega_strat[label]['median_offset_pct']:.2f}%  "
            f"median basin={omega_strat[label]['median_basin_pct']:.2f}%",
            flush=True,
        )

    # --------------------------------------------------------------- #
    # Per-strategy 'cells within basin' counts (multi-cell coverage).
    # Useful because top-K cell ranking benefits from multiple cells in basin.
    # --------------------------------------------------------------- #
    print("\n[4] How many bracket cells fall within the basin (per seed median, conservative rule)?", flush=True)
    n_in_basin = {}
    for strat in STRATEGIES:
        # We don't have per-cell list in s019; use: if nearest_pct <= basin, then
        # n_cells_within = max(1, n_within_5pct × basin/5%) -- under-estimate.
        # Better: use n_within_5pct as proxy, scaled by basin/5%.
        per_seed_count = []
        for r in rows:
            basin = r[f"basin_{rule_name}"]
            n5 = r[f"{strat}_n_within_5pct"]
            # If nearest_pct > basin, then no cells in basin.
            if r[f"{strat}_nearest_pct"] > basin:
                per_seed_count.append(0)
            else:
                # Approximate: assume cells uniform within ±5%, so
                # n_in_basin ≈ n_within_5pct × basin / 5.
                per_seed_count.append(max(1, int(n5 * basin / 5.0)))
        med = float(np.median(per_seed_count))
        n_in_basin[strat] = med
        print(f"   {strat:>10}: median cells in basin per seed = {med:.1f}", flush=True)
    print(
        "   (estimate; assumes cells are uniform within the ±5% window — over-counts on the\n"
        "    grid edges and under-counts where multiple cells cluster around truth.)",
        flush=True,
    )

    # --------------------------------------------------------------- #
    # Save artefacts.
    # --------------------------------------------------------------- #
    out_csv = OUT_DIR / "per_seed.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_csv}", flush=True)

    summary = {
        "n_seeds": n_total,
        "strategies": STRATEGIES,
        "rules": {k: {"const_pct": c, "exp": a} for k, (c, a) in RULES.items()},
        "aggregate_pass_count": aggregate,
        "offset_distribution_pct": distribution,
        "omega_stratified_bracket_conservative": omega_strat,
        "median_cells_in_basin_per_seed_conservative": n_in_basin,
    }
    out_json = OUT_DIR / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}", flush=True)

    # Headlines.
    print("\n=== HEADLINES ===", flush=True)
    rule_name = "conservative (c=1%/a=0.5)"
    print(
        f"  s019 'bracket' strategy under conservative rule: "
        f"{aggregate['bracket'][rule_name]}/{n_total} within basin",
        flush=True,
    )
    print(
        f"  s019 'bracket' strategy under lenient rule (c=2%/a=1.0): "
        f"{aggregate['bracket']['lenient (c=2%/a=1.0)']}/{n_total} within basin",
        flush=True,
    )
    print(
        f"  vs s032 fast-path (5 cells/seed): 0/78 within basin under any rule (s045)",
        flush=True,
    )
    print(
        f"  Per-seed bracket grid size (median): "
        f"{int(np.median([r['bracket_grid_size'] for r in rows if r['bracket_grid_size']]))}",
        flush=True,
    )
    print(
        f"  Cohort total (seed,cell) pairs (s019-bracket × 100 seeds): "
        f"{sum(r['bracket_grid_size'] for r in rows if r['bracket_grid_size']):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
