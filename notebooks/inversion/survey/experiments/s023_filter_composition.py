"""s023 — Filter composition + class stratification (post-processing).

Pure analysis on cached s021 output. No new computation.

Questions:
  1. Orthogonality: are alignment & geo scores correlated on random
     candidates? Independent filters multiply harder.
  2. Cumulative rejection: at thresholds calibrated for 100% truth retention,
     what fraction of random candidates is discarded?
  3. Stratification: does filter strength depend on
       (a) n_rotations class (n_rot<2 cohort tail, n_rot ∈ [2,5), n_rot ≥5)
       (b) tier coverage class (T1-rich, multi-tier, sub-3-peak, zero-classifiable)
       (c) multi-solution-rich vs not (s014 classes)?

Outputs:
  results/s023/composition.json
  results/s023/scatter_align_vs_geo.png
  results/s023/cumulative_rejection.png
  results/s023/per_class_stratification.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib.traj_load import load_truth, list_seeds  # noqa: E402

RESULTS = SURVEY_DIR / "results" / "s023"
RESULTS.mkdir(parents=True, exist_ok=True)

# Load s021 output
S021 = SURVEY_DIR / "results" / "s021" / "score_distributions.npz"
S018B = SURVEY_DIR / "results" / "s018b" / "face_tiers.npz"

if not S021.exists():
    print(f"FATAL: {S021} not found. Run s021 first.")
    sys.exit(1)


def main():
    print("=" * 72)
    print("s023 — Filter composition + class stratification")
    print("=" * 72)

    d21 = np.load(S021)
    seeds = d21["seeds"]
    n_seeds = len(seeds)
    truth_align = d21["truth_align"]
    truth_geo = d21["truth_geo"]
    rand_align = d21["rand_align"]    # (100, N_RANDOM)
    rand_geo = d21["rand_geo"]
    twin_align = d21["twin_align"]
    twin_geo = d21["twin_geo"]
    n_random = int(d21["n_random"])
    print(f"  s021 cohort: {n_seeds} seeds × {n_random} random candidates")

    # ---- Cohort metadata for stratification ----
    # n_rotations from cached truth (omega_mag_dps × duration_s / 360°)
    # Use s014b's analysis if cached, else compute on the fly
    n_rot = np.zeros(n_seeds)
    for i, s in enumerate(seeds):
        truth = load_truth(int(s))
        omega_mag = float(truth["omega_mag_dps"])
        # Duration ~3600s per m048 generator
        n_rot[i] = omega_mag * 3600.0 / 360.0

    # Tier coverage from s018b
    s18b = np.load(S018B)
    s18b_seeds = s18b["seeds"]
    n_tiers_hit = s18b["per_seed_distinct_tiers_hit"]  # 0..4
    n_t1 = s18b["per_seed_t1_count"]
    n_total_classifiable = s18b["per_seed_total_classifiable"]
    # Reorder to match s021 seed ordering
    seed_to_s18b_idx = {int(s): i for i, s in enumerate(s18b_seeds)}
    tier_class = np.empty(n_seeds, dtype=object)
    for i, s in enumerate(seeds):
        idx = seed_to_s18b_idx[int(s)]
        if n_total_classifiable[idx] == 0:
            tier_class[i] = "zero_classifiable"
        elif n_t1[idx] >= 1:
            tier_class[i] = "T1_rich"
        elif n_tiers_hit[idx] >= 2:
            tier_class[i] = "multi_tier"
        else:
            tier_class[i] = "sub3_peak"

    # ---- 1. Orthogonality on random ----
    flat_a = rand_align.flatten()
    flat_g = rand_geo.flatten()
    finite = np.isfinite(flat_a) & np.isfinite(flat_g)
    n_pair = int(finite.sum())
    pearson_r, pearson_p = pearsonr(flat_a[finite], flat_g[finite])
    spearman_r, spearman_p = spearmanr(flat_a[finite], flat_g[finite])
    print(f"\n  Random-candidate orthogonality (n={n_pair}):")
    print(f"    Pearson:  {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"    Spearman: {spearman_r:.4f} (p={spearman_p:.2e})")

    # ---- 2. Cumulative rejection at calibrated thresholds ----
    # Calibrate per-seed thresholds at "truth retention 100%" = retain truth
    # AND known-multi-solution.
    # Per-seed alignment threshold = truth_align (everything < truth gets rejected).
    # Per-seed geo threshold = truth_geo (or NaN if undefined).
    rej_align_per_seed = np.zeros(n_seeds)
    rej_geo_per_seed = np.zeros(n_seeds)
    rej_either_per_seed = np.zeros(n_seeds)
    rej_both_per_seed = np.zeros(n_seeds)

    for i in range(n_seeds):
        ra = rand_align[i]
        rg = rand_geo[i]
        thresh_a = truth_align[i] if np.isfinite(truth_align[i]) else 0.999
        thresh_g = truth_geo[i] if np.isfinite(truth_geo[i]) else np.nan
        n = ra.size
        # Pass alignment: ra >= thresh_a; otherwise rejected
        pass_a = ra >= thresh_a
        if np.isfinite(thresh_g):
            pass_g = rg >= thresh_g
        else:
            # Geo is undefined — every candidate "passes" trivially (no info)
            pass_g = np.ones(n, dtype=bool)
        rej_align_per_seed[i] = 1.0 - np.sum(pass_a) / n
        rej_geo_per_seed[i] = 1.0 - np.sum(pass_g) / n
        # Either filter: pass = pass_a AND pass_g (intersection)
        pass_either = pass_a & pass_g
        rej_either_per_seed[i] = 1.0 - np.sum(pass_either) / n
        # Both reject: any one filter rejects (union of rejections)
        rej_both_per_seed[i] = 1.0 - np.sum(pass_a | pass_g) / n  # Union of passing

    # Cohort-aggregate rejection (mean across seeds)
    print(f"\n  Cumulative rejection at truth=100%-retention threshold:")
    print(f"    Alignment alone (cohort mean): {np.mean(rej_align_per_seed):.3f}")
    print(f"    Geo alone (cohort mean):       {np.nanmean(rej_geo_per_seed):.3f}")
    print(f"    Intersection (both required):  {np.mean(rej_either_per_seed):.3f}")
    # NB: median is more robust than mean for cohort summary
    print(f"    Alignment alone (median):      {np.median(rej_align_per_seed):.3f}")
    print(f"    Geo alone (median):            {np.median(rej_geo_per_seed):.3f}")
    print(f"    Intersection (median):         {np.median(rej_either_per_seed):.3f}")

    # ---- 3. Stratification ----
    # By tier class
    by_tier = {}
    for tc in ["zero_classifiable", "sub3_peak", "multi_tier", "T1_rich"]:
        mask = tier_class == tc
        if mask.sum() == 0:
            continue
        by_tier[tc] = {
            "n_seeds": int(mask.sum()),
            "rej_align_median": float(np.median(rej_align_per_seed[mask])),
            "rej_geo_median": float(np.nanmedian(rej_geo_per_seed[mask])),
            "rej_intersection_median": float(np.median(rej_either_per_seed[mask])),
        }

    # By n_rot class
    by_rot = {}
    for label, mask in [
        ("low_rot_lt2", n_rot < 2),
        ("mid_rot_2_to_5", (n_rot >= 2) & (n_rot < 5)),
        ("high_rot_ge5", n_rot >= 5),
    ]:
        if mask.sum() == 0:
            continue
        by_rot[label] = {
            "n_seeds": int(mask.sum()),
            "rej_align_median": float(np.median(rej_align_per_seed[mask])),
            "rej_geo_median": float(np.nanmedian(rej_geo_per_seed[mask])),
            "rej_intersection_median": float(np.median(rej_either_per_seed[mask])),
        }

    # ---- 4. Truth percentile rank in random distribution ----
    truth_align_rank = np.full(n_seeds, np.nan)
    truth_geo_rank = np.full(n_seeds, np.nan)
    for i in range(n_seeds):
        ra = rand_align[i]
        rg = rand_geo[i]
        ra_f = ra[np.isfinite(ra)]
        rg_f = rg[np.isfinite(rg)]
        if np.isfinite(truth_align[i]) and ra_f.size > 0:
            truth_align_rank[i] = (np.sum(ra_f >= truth_align[i]) + 1) / (ra_f.size + 1)
        if np.isfinite(truth_geo[i]) and rg_f.size > 0:
            truth_geo_rank[i] = (np.sum(rg_f >= truth_geo[i]) + 1) / (rg_f.size + 1)

    # ---- 5. Save summary JSON ----
    summary = {
        "n_seeds": int(n_seeds),
        "n_random_per_seed": int(n_random),
        "orthogonality": {
            "pearson": float(pearson_r),
            "spearman": float(spearman_r),
            "n_pairs": n_pair,
        },
        "rejection_at_truth_retention_100pct": {
            "alignment_alone_median": float(np.median(rej_align_per_seed)),
            "alignment_alone_mean": float(np.mean(rej_align_per_seed)),
            "geo_alone_median": float(np.nanmedian(rej_geo_per_seed)),
            "geo_alone_mean": float(np.nanmean(rej_geo_per_seed)),
            "intersection_median": float(np.median(rej_either_per_seed)),
            "intersection_mean": float(np.mean(rej_either_per_seed)),
        },
        "by_tier_class": by_tier,
        "by_n_rot": by_rot,
        "truth_percentile_rank": {
            "align_median": float(np.nanmedian(truth_align_rank)),
            "align_p10": float(np.nanpercentile(truth_align_rank, 10)),
            "align_p90": float(np.nanpercentile(truth_align_rank, 90)),
            "align_n_top1pct": int(np.sum(truth_align_rank <= 0.01)),
            "align_n_top10pct": int(np.sum(truth_align_rank <= 0.10)),
            "geo_median": float(np.nanmedian(truth_geo_rank)),
            "geo_n_top1pct": int(np.sum(truth_geo_rank <= 0.01)),
            "geo_n_top10pct": int(np.sum(truth_geo_rank <= 0.10)),
        },
        "twin_filter_pass": {
            "twin_align_median": float(np.nanmedian(twin_align)),
            "twin_align_n_at_1.0": int(np.sum(twin_align >= 0.999)),
            "twin_geo_median": float(np.nanmedian(twin_geo)),
            "twin_geo_n_at_1.0": int(np.sum(twin_geo >= 0.999)),
        },
    }
    with open(RESULTS / "composition.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS / 'composition.json'}")

    # ---- 6. Plots ----
    # 6a. Scatter random align vs geo
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sub = np.random.default_rng(0).choice(flat_a[finite].size, size=min(20000, flat_a[finite].size), replace=False)
    ax.scatter(flat_a[finite][sub], flat_g[finite][sub], s=1, alpha=0.2, label="random")
    ax.scatter(truth_align, truth_geo, s=80, c="red", marker="*", label="truth", zorder=5)
    ax.scatter(twin_align, twin_geo, s=60, c="blue", marker="^", label="body-twin", zorder=5)
    ax.set_xlabel("alignment cost (high=good)")
    ax.set_ylabel("geo cost (high=good)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title(f"Filter scores: random (n={n_pair}) vs truth/twin per seed\n"
                 f"Pearson={pearson_r:.3f} Spearman={spearman_r:.3f}")
    fig.tight_layout()
    fig.savefig(RESULTS / "scatter_align_vs_geo.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {RESULTS / 'scatter_align_vs_geo.png'}")

    # 6b. Per-seed rejection distribution
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(rej_align_per_seed, bins=30, color="C0", edgecolor="k")
    axs[0].axvline(np.median(rej_align_per_seed), c="red", linestyle="--",
                   label=f"median {np.median(rej_align_per_seed):.2f}")
    axs[0].set_xlabel("alignment-only rejection fraction")
    axs[0].set_ylabel("seeds")
    axs[0].legend()
    axs[1].hist(rej_geo_per_seed[np.isfinite(rej_geo_per_seed)], bins=30,
                color="C1", edgecolor="k")
    axs[1].axvline(np.nanmedian(rej_geo_per_seed), c="red", linestyle="--",
                   label=f"median {np.nanmedian(rej_geo_per_seed):.2f}")
    axs[1].set_xlabel("geo-only rejection fraction")
    axs[1].legend()
    axs[2].hist(rej_either_per_seed, bins=30, color="C2", edgecolor="k")
    axs[2].axvline(np.median(rej_either_per_seed), c="red", linestyle="--",
                   label=f"median {np.median(rej_either_per_seed):.2f}")
    axs[2].set_xlabel("intersection rejection fraction")
    axs[2].legend()
    fig.suptitle("Per-seed rejection at truth=100%-retention threshold")
    fig.tight_layout()
    fig.savefig(RESULTS / "cumulative_rejection.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {RESULTS / 'cumulative_rejection.png'}")

    # 6c. Per-class stratification
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    # By tier class
    classes = list(by_tier.keys())
    align_med = [by_tier[c]["rej_align_median"] for c in classes]
    geo_med = [by_tier[c]["rej_geo_median"] for c in classes]
    inter_med = [by_tier[c]["rej_intersection_median"] for c in classes]
    n_per = [by_tier[c]["n_seeds"] for c in classes]
    x = np.arange(len(classes))
    axs[0].bar(x - 0.25, align_med, width=0.25, label="alignment", color="C0")
    axs[0].bar(x, geo_med, width=0.25, label="geo", color="C1")
    axs[0].bar(x + 0.25, inter_med, width=0.25, label="intersection", color="C2")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, n_per)],
                           rotation=0, fontsize=8)
    axs[0].set_ylabel("median rejection fraction")
    axs[0].set_title("By tier-coverage class")
    axs[0].set_ylim(0, 1.05)
    axs[0].legend()
    # By n_rot class
    classes2 = list(by_rot.keys())
    a = [by_rot[c]["rej_align_median"] for c in classes2]
    g = [by_rot[c]["rej_geo_median"] for c in classes2]
    i_ = [by_rot[c]["rej_intersection_median"] for c in classes2]
    n_per2 = [by_rot[c]["n_seeds"] for c in classes2]
    x2 = np.arange(len(classes2))
    axs[1].bar(x2 - 0.25, a, width=0.25, label="alignment", color="C0")
    axs[1].bar(x2, g, width=0.25, label="geo", color="C1")
    axs[1].bar(x2 + 0.25, i_, width=0.25, label="intersection", color="C2")
    axs[1].set_xticks(x2)
    axs[1].set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes2, n_per2)])
    axs[1].set_ylabel("median rejection fraction")
    axs[1].set_title("By n_rotations class")
    axs[1].set_ylim(0, 1.05)
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "per_class_stratification.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {RESULTS / 'per_class_stratification.png'}")

    # ---- Print headline summary ----
    print()
    print("=" * 72)
    print("HEADLINE — filter framework strength on cohort:")
    print(f"  truth percentile rank in random:")
    print(f"    alignment top-1%: {summary['truth_percentile_rank']['align_n_top1pct']}/100")
    print(f"    alignment top-10%: {summary['truth_percentile_rank']['align_n_top10pct']}/100")
    print(f"    geo top-1%: {summary['truth_percentile_rank']['geo_n_top1pct']}/100")
    print(f"  cumulative rejection (median across cohort):")
    print(f"    alignment alone: {np.median(rej_align_per_seed):.3f}")
    print(f"    geo alone:       {np.nanmedian(rej_geo_per_seed):.3f}")
    print(f"    intersection:    {np.median(rej_either_per_seed):.3f}")
    print(f"  random-pair filter orthogonality:")
    print(f"    Pearson  = {pearson_r:.3f}  (independent if ~0)")
    print(f"    Spearman = {spearman_r:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
