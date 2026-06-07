"""s014 — post-hoc analysis after the main batch lands.

Adds analyses NOT in the main script:

  (1) hi-fi / surrogate MSE ratio distribution per cohort + per seed.
      Tests whether the surrogate is ABSOLUTELY (not just rank-) faithful
      across the cohort. s013 measured median ratio 1.003 / p10 0.987 /
      p90 1.041 on seed 10. If similar bounds hold across cohort A,
      surrogate MSE can be used as a direct hi-fi proxy.

  (2) Cohort-wide Spearman across all 540 s011 non-basin candidates
      (in addition to per-seed Spearman in summary.json).

  (3) Multi-solution candidates outside in-basin: dump all Band A∪B
      candidates with their (q0_err, ω_dir, ω_mag, ρ) for inspection.

  (4) hi-fi/surr ratio histogram + cohort A scatter plot annotated with
      multi-solution candidates.

Reads results/s014/{rho_s011_nb.npz, rho_s005.npz, summary.json},
results/s013/rho_in_basin.npz, results/s013/rho_seed10.npz.

Output: results/s014/{ratio_distribution.png, multi_sol_candidates.json,
analysis_summary.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))

OUT_DIR = SURVEY_DIR / "results" / "s014"
S013_DIR = SURVEY_DIR / "results" / "s013"


def _rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(x, y):
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def main():
    print("Loading s014 + s013 NPZs ...", flush=True)
    a_nb = np.load(OUT_DIR / "rho_s011_nb.npz")
    a_ib = np.load(S013_DIR / "rho_in_basin.npz")
    b = np.load(OUT_DIR / "rho_s005.npz")
    seed10 = np.load(S013_DIR / "rho_seed10.npz")

    # ---------------- hi-fi / surrogate ratio distribution ------------------
    def ratio_stats(rho, surr_mse, label):
        hifi_mse = (rho * 0.05) ** 2
        ratio = hifi_mse / np.maximum(surr_mse, 1e-12)
        return {
            "label": label,
            "n": int(len(ratio)),
            "median": float(np.median(ratio)),
            "p10": float(np.percentile(ratio, 10)),
            "p90": float(np.percentile(ratio, 90)),
            "mean": float(np.mean(ratio)),
            "std": float(np.std(ratio)),
            "ratio": ratio,
        }

    stats_a_ib = ratio_stats(a_ib["rho"], a_ib["surrogate_mse"], "s011 in-basin (s013)")
    stats_a_nb = ratio_stats(a_nb["rho"], a_nb["surrogate_mse"], "s011 non-basin (s014)")
    stats_b = ratio_stats(b["rho"], b["surrogate_mse"], "s005 all (s014)")
    stats_seed10 = ratio_stats(seed10["rho"], seed10["surrogate_mse"], "seed 10 N=256 (s013)")

    print()
    print("hi-fi / surrogate MSE ratio (median, p10-p90):")
    for s in (stats_a_ib, stats_a_nb, stats_b, stats_seed10):
        print(f"  {s['label']:<32} n={s['n']:>4}  median={s['median']:.3f}  "
              f"p10={s['p10']:.3f}  p90={s['p90']:.3f}")

    # ---------------- Cohort-wide Spearman across cohort A ------------------
    rho_a_all = np.concatenate([a_ib["rho"], a_nb["rho"]])
    surr_a_all = np.concatenate([a_ib["surrogate_mse"], a_nb["surrogate_mse"]])
    seed_a_all = np.concatenate([a_ib["seed"], a_nb["seed"]])
    cohort_wide_spearman = spearman(surr_a_all, rho_a_all)
    print(f"\nCohort A overall Spearman(surrogate_mse, rho) on n={len(rho_a_all)}: "
          f"{cohort_wide_spearman:.4f}")

    # ---------------- Multi-solution candidates outside in-basin ------------------
    # (any Band A∪B candidate from cohort A non-basin OR cohort B non-basin)
    msol_A_mask = (a_nb["rho"] < 4.0)
    msol_A = []
    if msol_A_mask.any():
        for i in np.where(msol_A_mask)[0]:
            msol_A.append({
                "cohort": "s011_non_basin",
                "s011_row": int(a_nb["s011_row"][i]),
                "seed": int(a_nb["seed"][i]),
                "ic_idx": int(a_nb["ic_idx"][i]),
                "rho": float(a_nb["rho"][i]),
                "band": str(a_nb["band"][i]),
                "hifi_mse": float(a_nb["hifi_mse"][i]),
                "surrogate_mse": float(a_nb["surrogate_mse"][i]),
                "q0_err_deg": float(a_nb["q0_err_deg"][i]),
                "omega_dir_err_deg": float(a_nb["omega_dir_err_deg"][i]),
                "omega_mag_err_pct": float(a_nb["omega_mag_err_pct"][i]),
            })

    msol_B_mask = (b["rho"] < 4.0) & (~b["truth_basin_strict"])
    msol_B = []
    if msol_B_mask.any():
        for i in np.where(msol_B_mask)[0]:
            msol_B.append({
                "cohort": "s005_non_basin",
                "s005_row": int(b["s005_row"][i]),
                "seed": int(b["seed"][i]),
                "ic_idx": int(b["ic_idx"][i]),
                "label": str(b["label"][i]),
                "kind": str(b["kind"][i]),
                "tier": int(b["tier"][i]),
                "rho": float(b["rho"][i]),
                "band": str(b["band"][i]),
                "hifi_mse": float(b["hifi_mse"][i]),
                "surrogate_mse": float(b["surrogate_mse"][i]),
                "q0_err_deg": float(b["q0_err_deg"][i]),
                "omega_dir_err_deg": float(b["omega_dir_err_deg"][i]),
                "omega_mag_err_pct": float(b["omega_mag_err_pct"][i]),
            })

    msol_combined = msol_A + msol_B
    seeds_with_msol = sorted(set(c["seed"] for c in msol_combined))
    print()
    print(f"Multi-solution candidates outside in-basin (ρ < 4):")
    print(f"  Cohort A (s011 non-basin): {len(msol_A)} across {len(set(c['seed'] for c in msol_A))} seeds")
    print(f"  Cohort B (s005 non-basin): {len(msol_B)} across {len(set(c['seed'] for c in msol_B))} seeds")
    print(f"  Combined: {len(msol_combined)} across {len(seeds_with_msol)} seeds")
    print()
    if msol_combined:
        msol_combined.sort(key=lambda c: c["rho"])
        print(f"  Top {min(20, len(msol_combined))} by ρ:")
        print(f"  {'cohort':<18} {'seed':>4} {'rho':>5} {'band':>4} {'q0_err':>7} {'ωd':>5} {'ωm':>6} {'surr_mse':>9}")
        for c in msol_combined[:20]:
            print(f"  {c['cohort']:<18} {c['seed']:>4} {c['rho']:>5.2f} {c['band']:>4} "
                  f"{c['q0_err_deg']:>7.1f} {c['omega_dir_err_deg']:>5.1f} "
                  f"{c['omega_mag_err_pct']:>+6.1f} {c['surrogate_mse']:>9.4f}")

    # ---------------- Save analysis summary ------------------
    out = {
        "ratio_distribution": {
            k: {kk: v for kk, v in vv.items() if kk != "ratio"}
            for k, vv in [
                ("s011_in_basin", stats_a_ib),
                ("s011_non_basin", stats_a_nb),
                ("s005_all", stats_b),
                ("seed_10_n256", stats_seed10),
            ]
        },
        "cohort_a_overall_spearman": cohort_wide_spearman,
        "multi_solution_outside_basin": {
            "n_total": len(msol_combined),
            "n_seeds": len(seeds_with_msol),
            "seeds": seeds_with_msol,
            "candidates": msol_combined,
        },
    }
    with open(OUT_DIR / "analysis_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'analysis_summary.json'}")

    # ---------------- Plot: ratio distribution ------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    cohorts = [stats_a_ib, stats_a_nb, stats_b, stats_seed10]
    for ax, st in zip(axes, cohorts):
        # Bin in log-space for symmetry
        ratio = st["ratio"]
        clip = np.clip(ratio, 0.01, 100.0)
        ax.hist(clip, bins=np.logspace(-2, 2, 50), color="C0", alpha=0.7, edgecolor="k")
        ax.axvline(1.0, color="green", linestyle="--", linewidth=1.0, label="ratio=1")
        ax.axvline(st["median"], color="orange", linestyle="-", linewidth=1.0,
                   label=f"median={st['median']:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("hi-fi MSE / surrogate MSE")
        ax.set_title(f"{st['label']}\nn={st['n']}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Hi-fi / surrogate MSE ratio across cohorts", fontsize=12)
    fig.tight_layout()
    p = OUT_DIR / "ratio_distribution.png"
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
