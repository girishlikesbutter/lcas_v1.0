#!/usr/bin/env python3
"""Count valid solutions per seed by hi-fi error band.

Under multi-solution philosophy, every basin below a hi-fi threshold is a
valid solution (truth-adjacent, ±X twin, flipped-ω — all equally valid LC
fits). This script aggregates per-basin hi-fi values from m126_wrapped (6
seeds) + m124/m125 (5 seeds) and plots the distribution by band.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

M126_SEEDS = [0, 6, 12, 24, 33, 36]
M124_SEEDS = [14, 27, 46, 74, 93]
ALL_SEEDS = sorted(M126_SEEDS + M124_SEEDS)

BANDS = [
    ("< 0.005 (tight)",     0.0,    0.005,  "#08519c"),
    ("0.005 – 0.01 (OK)",   0.005,  0.01,   "#3182bd"),
    ("0.01 – 0.05",         0.01,   0.05,   "#6baed6"),
    ("0.05 – 0.1 (PARTIAL)",0.05,   0.1,    "#c6dbef"),
    ("0.1 – 0.3",           0.1,    0.3,    "#fcae91"),
    ("≥ 0.3 (FAIL)",        0.3,    np.inf, "#cb181d"),
]


def collect_per_basin():
    m125 = json.load(open(RESULTS_DIR / "m125_keep_better" / "summary.json"))
    per_seed = {s["seed"]: s for s in m125["per_seed"]}

    records = []
    for seed in M126_SEEDS:
        res = json.load(open(RESULTS_DIR / "m126_wrapped" / f"seed_{seed:03d}"
                             / "result.json"))
        for b in res["basins"]:
            is_twin_q0 = b["q0_err_after"] > 170.0
            is_flipped_w = b["w_dir_err_after"] > 90.0
            records.append({
                "seed": seed,
                "basin": b["basin_idx"],
                "hifi": b["hifi_wrapped"],
                "q0_err": b["q0_err_after"],
                "w_dir_err": b["w_dir_err_after"],
                "is_twin": is_twin_q0 and not is_flipped_w,
                "is_flipped_w": is_flipped_w,
                "is_truth_adj": b["q0_err_after"] < 10.0 and not is_flipped_w,
            })

    for seed in M124_SEEDS:
        for b in per_seed[seed]["per_basin"]:
            records.append({
                "seed": seed,
                "basin": int(b["label"].split("_")[1]),
                "hifi": b["hifi_best_wrapped"],
                "q0_err": None,
                "w_dir_err": None,
                "is_twin": None,
                "is_flipped_w": None,
                "is_truth_adj": None,
            })
    return records


def band_of(hifi):
    for i, (_, lo, hi, _) in enumerate(BANDS):
        if lo <= hifi < hi:
            return i
    return len(BANDS) - 1


def main():
    records = collect_per_basin()
    assert len(records) == 33, f"expected 33 basins, got {len(records)}"

    counts = np.zeros((len(ALL_SEEDS), len(BANDS)), dtype=int)
    for r in records:
        si = ALL_SEEDS.index(r["seed"])
        bi = band_of(r["hifi"])
        counts[si, bi] += 1

    print("\nPer-seed basin distribution by hi-fi band:")
    print(f"{'seed':>5} " + " ".join(f"{lbl:>22s}" for lbl, *_ in BANDS) + "  total")
    for si, seed in enumerate(ALL_SEEDS):
        row = " ".join(f"{counts[si, bi]:>22d}" for bi in range(len(BANDS)))
        print(f"{seed:>5d} {row}  {counts[si].sum():>5d}")
    total_row = counts.sum(axis=0)
    print(f"{'all':>5s} " + " ".join(f"{total_row[bi]:>22d}"
          for bi in range(len(BANDS))) + f"  {total_row.sum():>5d}")

    print(f"\nBasins below 0.1 (valid solutions): "
          f"{counts[:, :4].sum()}/33  "
          f"({100.0*counts[:, :4].sum()/33:.0f}%)")
    print(f"Basins below 0.01 (tight OK): "
          f"{counts[:, :2].sum()}/33  "
          f"({100.0*counts[:, :2].sum()/33:.0f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6),
                                    gridspec_kw={'width_ratios': [2, 1]})

    x = np.arange(len(ALL_SEEDS))
    bottom = np.zeros(len(ALL_SEEDS), dtype=int)
    for bi, (lbl, lo, hi, color) in enumerate(BANDS):
        ax1.bar(x, counts[:, bi], bottom=bottom, color=color, edgecolor='white',
                linewidth=0.6, label=lbl)
        for si in range(len(ALL_SEEDS)):
            c = counts[si, bi]
            if c > 0:
                ax1.text(x[si], bottom[si] + c / 2, str(c),
                         ha='center', va='center', fontsize=9,
                         color='white' if bi in (0, 1, 5) else 'black')
        bottom += counts[:, bi]

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}" for s in ALL_SEEDS])
    ax1.set_xlabel("seed")
    ax1.set_ylabel("basin count (out of 3)")
    ax1.set_ylim(0, 3.2)
    ax1.set_title("Valid solutions per seed, stratified by hi-fi MSE band\n"
                  "wrapped pipeline (m126 + m124/m125), post-fix 2026-04-17",
                  fontsize=10)
    ax1.legend(loc='upper right', fontsize=8, ncol=1, framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y')

    cumulative = counts.sum(axis=0)
    y = np.arange(len(BANDS))
    colors = [b[3] for b in BANDS]
    ax2.barh(y, cumulative, color=colors, edgecolor='white', linewidth=0.8)
    for bi, c in enumerate(cumulative):
        if c > 0:
            ax2.text(c + 0.3, bi, f" {c}  ({100*c/33:.0f}%)",
                     ha='left', va='center', fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([b[0] for b in BANDS], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("basin count across all 11 seeds (33 total)")
    ax2.set_title("Aggregate solution count by band", fontsize=10)
    ax2.set_xlim(0, max(cumulative) * 1.25)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    out = RESULTS_DIR / "m132_solution_count_by_band.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()

    out_json = RESULTS_DIR / "m132_solution_count.json"
    with open(out_json, "w") as f:
        json.dump({
            "bands": [{"label": b[0], "lo": b[1], "hi": b[2]} for b in BANDS],
            "seeds": ALL_SEEDS,
            "counts_by_seed_band": counts.tolist(),
            "totals_by_band": total_row.tolist(),
            "per_basin": records,
        }, f, indent=2, default=float)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    sys.exit(main())
