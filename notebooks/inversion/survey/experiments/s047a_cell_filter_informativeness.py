"""s047a — does the cell-filter score rank cells by closeness to truth |ω|?

The s047 hybrid pipeline architecture rests on this assumption: rank the
bracket's ω-cells by cell-filter survivor count; the top-K cells are the
ones we densify around. If the filter score doesn't correlate with cell-
to-truth distance, this top-K-then-densify architecture is broken.

Test on cached s032 data: each of 78 OK seeds has 5 ω-cells × 1056
q-targets = 5,280 candidates per cell scored by `geo_score` and
`align_score`. Aggregate per cell (max/mean/survivor count) and check if
the cell with the highest score is the closest to truth |ω|.

s032's 5 cells are typically outside basin on every seed under the c=1%/a=0.5
rule (s045), so this isn't testing "filter picks within-basin cell" — it's
testing "filter picks the *least-distant* of an out-of-basin set." If the
filter is informative even at this level, then with a wider bracket the
closest cell would also be the highest-scoring.

Outputs:
    results/s047a_cell_filter_informativeness/per_seed.csv
    results/s047a_cell_filter_informativeness/summary.json
    results/s047a_cell_filter_informativeness/run.log
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

S032_DIR = SURVEY_DIR / "results" / "s032_cohort_fast"
OUT_DIR = SURVEY_DIR / "results" / "s047a_cell_filter_informativeness"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_per_cell(seed_dir: Path) -> dict | None:
    """Aggregate filter scores per cell for a single seed.

    For each (omega_mag_idx ∈ {0..4}), compute across all dir × q_target:
      - cell_truth_dist_pct: % offset from truth |ω| (signed)
      - n_pass_both, n_pass_geo, n_pass_align: pass counts
      - max_geo, max_align, mean_geo, mean_align: filter score aggregates

    Returns None if files missing.
    """
    bracket_path = seed_dir / "bracket.npz"
    candidates_path = seed_dir / "candidates_meta.npz"
    omega_grid_path = seed_dir / "omega_grid.npz"
    if not all(p.exists() for p in (bracket_path, candidates_path, omega_grid_path)):
        return None

    bracket = np.load(bracket_path)
    candidates = np.load(candidates_path)
    grid = np.load(omega_grid_path)

    # 5 ω-mag cells (selected). bracket_dist_to_truth_pct is signed.
    cells_dps = bracket["bracket_cells"] * 180.0 / np.pi  # rad/s → dps
    truth_dps = float(bracket["truth_omega_mag_dps"])
    cell_offsets_pct = bracket["bracket_dist_to_truth_pct"]  # signed
    n_cells = len(cells_dps)

    # Each candidate has omega_cell_idx in [0, 1500). The 1500 cells are
    # 300 dirs × 5 mags. Map omega_cell_idx → ω-mag-idx via integer
    # division. (s032 layout: idx = dir_idx × 5 + mag_idx? OR dir_idx +
    # mag_idx × 300? Check via the cell_idx of any candidate.)
    omega_cell_idx = candidates["omega_cell_idx"]
    n_dirs = grid["omega_dirs"].shape[0]
    n_mags = grid["omega_mags"].shape[0]
    assert n_mags == n_cells, f"n_mags={n_mags} vs n_cells={n_cells}"
    # Assumption: omega_vectors flattens as [dir × mag] in row-major.
    # Verify: cell idx 0 should be (dir 0, mag 0). cell idx 5 should be
    # (dir 0, mag 5) NOT (dir 1, mag 0). Use the omega_vectors content.
    # For mag_idx assignment we compare cell magnitudes.
    cell_mags = np.linalg.norm(grid["omega_vectors"], axis=1)  # (1500,)
    # Map each cell to a mag-bin by argmin over the 5 magnitudes.
    mag_bin_per_cell = np.argmin(
        np.abs(cell_mags[:, None] - grid["omega_mags"][None, :]), axis=1
    )
    cand_mag_bin = mag_bin_per_cell[omega_cell_idx]

    geo = candidates["geo_score"]
    align = candidates["align_score"]
    pass_both = candidates["cat_both"]

    per_cell = {}
    for mi in range(n_cells):
        mask = cand_mag_bin == mi
        n = int(mask.sum())
        per_cell[mi] = {
            "cell_dps": float(cells_dps[mi]),
            "cell_offset_pct": float(cell_offsets_pct[mi]),
            "abs_offset_pct": float(abs(cell_offsets_pct[mi])),
            "n_candidates": n,
            "n_pass_both": int(pass_both[mask].sum()),
            "max_geo": float(geo[mask].max()) if n > 0 else float("nan"),
            "mean_geo": float(geo[mask].mean()) if n > 0 else float("nan"),
            "max_align": float(align[mask].max()) if n > 0 else float("nan"),
            "mean_align": float(align[mask].mean()) if n > 0 else float("nan"),
        }
    return {
        "truth_dps": truth_dps,
        "cells": per_cell,
    }


def main():
    print("=== s047a cell-filter informativeness ===", flush=True)
    seeds = sorted(
        int(p.name.replace("seed", ""))
        for p in S032_DIR.glob("seed[0-9][0-9][0-9]")
        if (p / "candidates_meta.npz").exists()
    )
    print(f"Found {len(seeds)} seeds with candidates_meta.npz", flush=True)

    rows = []
    rank_match = {
        "max_geo": {"matches": 0, "total": 0},
        "max_align": {"matches": 0, "total": 0},
        "n_pass_both": {"matches": 0, "total": 0},
        "max_geo_align_sum": {"matches": 0, "total": 0},
    }
    rank2_match = {k: 0 for k in rank_match}  # top-K=2 match
    for seed in seeds:
        seed_dir = S032_DIR / f"seed{seed:03d}"
        agg = aggregate_per_cell(seed_dir)
        if agg is None:
            continue
        cells = agg["cells"]
        # Find the cell-index closest to truth.
        sorted_by_dist = sorted(cells.items(), key=lambda kv: kv[1]["abs_offset_pct"])
        closest_idx = sorted_by_dist[0][0]
        # Rank cells by each filter metric.
        for metric in rank_match:
            if metric == "max_geo_align_sum":
                # Combined score: max_geo + max_align (both bounded [0, 1]).
                ranked = sorted(
                    cells.items(),
                    key=lambda kv: (kv[1]["max_geo"] + kv[1]["max_align"]),
                    reverse=True,
                )
            else:
                ranked = sorted(
                    cells.items(),
                    key=lambda kv: kv[1][metric],
                    reverse=True,
                )
            top1_idx = ranked[0][0]
            top2_indices = {ranked[0][0], ranked[1][0]}
            rank_match[metric]["total"] += 1
            if top1_idx == closest_idx:
                rank_match[metric]["matches"] += 1
            if closest_idx in top2_indices:
                rank2_match[metric] += 1

        # Per-seed row
        rows.append(
            {
                "seed": seed,
                "truth_dps": agg["truth_dps"],
                "closest_cell_idx": closest_idx,
                "closest_cell_offset_pct": cells[closest_idx]["abs_offset_pct"],
                "max_geo_top1_idx": sorted(
                    cells.items(), key=lambda kv: kv[1]["max_geo"], reverse=True
                )[0][0],
                "max_align_top1_idx": sorted(
                    cells.items(), key=lambda kv: kv[1]["max_align"], reverse=True
                )[0][0],
                "n_pass_both_top1_idx": sorted(
                    cells.items(), key=lambda kv: kv[1]["n_pass_both"], reverse=True
                )[0][0],
                "max_geo_at_closest": cells[closest_idx]["max_geo"],
                "max_align_at_closest": cells[closest_idx]["max_align"],
                "n_pass_both_at_closest": cells[closest_idx]["n_pass_both"],
            }
        )

    print(f"\n[1] Top-1 rank match: which filter metric picks the closest cell?", flush=True)
    print(f"   {'metric':>20} {'matches':>10} {'total':>8} {'rate':>8}", flush=True)
    aggregate = {}
    for metric, d in rank_match.items():
        n_match = d["matches"]
        n_tot = d["total"]
        rate = n_match / n_tot if n_tot else 0.0
        # Random baseline = 1/5 = 20%.
        aggregate[metric] = {"top1": n_match, "total": n_tot, "rate": rate}
        print(f"   {metric:>20} {n_match:>10d} {n_tot:>8d} {rate:>7.1%}", flush=True)
    print(f"   (random baseline = 1/5 = 20%)", flush=True)

    print(f"\n[2] Top-2 rank match: closest cell appears in top-2 by filter metric?", flush=True)
    print(f"   {'metric':>20} {'matches':>10} {'rate':>8}", flush=True)
    for metric, n_match in rank2_match.items():
        n_tot = rank_match[metric]["total"]
        rate = n_match / n_tot if n_tot else 0.0
        aggregate[metric]["top2"] = n_match
        aggregate[metric]["top2_rate"] = rate
        print(f"   {metric:>20} {n_match:>10d} {rate:>7.1%}", flush=True)
    print(f"   (random baseline = 2/5 = 40%)", flush=True)

    # |ω|-stratified: does the filter rank work better on slow vs fast seeds?
    print(f"\n[3] |ω|-stratified top-1 rate for max_geo_align_sum:", flush=True)
    sorted_seeds = sorted(rows, key=lambda r: r["truth_dps"])
    n = len(sorted_seeds)
    for label, lo_pct, hi_pct in [
        ("Q1 (slowest)", 0, 25),
        ("Q2", 25, 50),
        ("Q3", 50, 75),
        ("Q4 (fastest)", 75, 100),
    ]:
        lo = int(lo_pct / 100 * n)
        hi = int(hi_pct / 100 * n)
        if lo == hi:
            continue
        bin_rows = sorted_seeds[lo:hi]
        bin_rate = 0
        for r in bin_rows:
            # Recompute top-1 by max_geo_align_sum
            cells_in_seed = {
                idx: {"max_geo_at_closest": r["max_geo_at_closest"],
                      "max_align_at_closest": r["max_align_at_closest"]}
                for idx in [r["closest_cell_idx"]]
            }
            # Just count: did max_geo_top1 OR max_align_top1 OR n_pass_both_top1 match?
            if (r["max_geo_top1_idx"] == r["closest_cell_idx"]
                or r["max_align_top1_idx"] == r["closest_cell_idx"]):
                bin_rate += 1
        # But what we really want is the geo+align combined rank match
        # — which is rank_match["max_geo_align_sum"] aggregated. We computed
        # that globally; here we want stratified. Re-do by picking metric
        # to use:
        n_match_max_geo = sum(
            1 for r in bin_rows if r["max_geo_top1_idx"] == r["closest_cell_idx"]
        )
        n_match_max_align = sum(
            1 for r in bin_rows if r["max_align_top1_idx"] == r["closest_cell_idx"]
        )
        n_match_n_pass = sum(
            1 for r in bin_rows if r["n_pass_both_top1_idx"] == r["closest_cell_idx"]
        )
        print(
            f"   {label:>16}: n={len(bin_rows):>3d}  "
            f"max_geo {n_match_max_geo:>3d}/{len(bin_rows):<3d} "
            f" max_align {n_match_max_align:>3d}/{len(bin_rows):<3d} "
            f" n_pass {n_match_n_pass:>3d}/{len(bin_rows):<3d}",
            flush=True,
        )

    # Save artefacts
    out_csv = OUT_DIR / "per_seed.csv"
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nSaved: {out_csv}", flush=True)

    summary = {
        "n_seeds": len(rows),
        "rank_match_aggregate": aggregate,
        "random_baselines": {"top1": 0.2, "top2": 0.4},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR / 'summary.json'}", flush=True)

    print("\n=== HEADLINES ===", flush=True)
    best_metric = max(aggregate.keys(), key=lambda k: aggregate[k]["rate"])
    print(
        f"  Best top-1 metric: '{best_metric}' picks closest cell on "
        f"{aggregate[best_metric]['top1']}/{aggregate[best_metric]['total']} "
        f"({aggregate[best_metric]['rate']:.1%}) seeds",
        flush=True,
    )
    print(
        f"  Random baseline (top-1): 20%. Lift = "
        f"{(aggregate[best_metric]['rate'] / 0.2 - 1) * 100:+.0f}%",
        flush=True,
    )
    print(
        f"  '{best_metric}' top-2 rate: {aggregate[best_metric]['top2_rate']:.1%} "
        f"(random baseline 40%)",
        flush=True,
    )


if __name__ == "__main__":
    main()
