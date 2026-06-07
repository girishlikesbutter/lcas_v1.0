"""s017 — Hi-fi ρ-band validation of s016c_prime cohort-selector winners.

Decisive question:
  How many of the 7 cohort-selector winners (one per seed, argmin of
  surrogate MSE in s016c_prime) hi-fi-validate as Band A∪B (ρ < 4)?
  - ≥4/7 → C' multi-solution architecture viable for production
  - <4/7 → multi-solution reframe killed; pivot to S016-A (ω-grid)

Render strategy:
  Top-3 lowest-surrogate-MSE candidates per seed × 7 seeds = 21 hi-fi
  renders. Top-3 is cheap insurance: if #1 is Band C/D but #2 or #3 is
  Band A∪B, the cohort-selector is narrowly missing rather than
  structurally broken.

Forward model: `lib/hifi_render.py` (smoke-tested in s013 to round-trip
mag_hifi to bit-identical on seeds 6/10/91).

Predictions (using s014's "non-basin: surrogate ≈ hi-fi to <1%" proxy):

  | seed | min surr MSE | predicted ρ | predicted band |
  |------|--------------|-------------|----------------|
  | 6    | 1.250        | 22.4        | D              |
  | 13   | 0.0045       | 1.34        | A              |
  | 28   | 5.388        | 46.4        | D              |
  | 41   | 0.00587      | 1.53        | A              |
  | 49   | 3.153        | 35.5        | D              |
  | 79   | 0.0107       | 2.07        | B (boundary)   |
  | 91   | 0.0825       | 5.74        | C              |

  Predicted yield: 3/7 Band A∪B (seeds 13, 41, 79).

Output:
  results/s017/{rho_top3.npz, summary.json, rho_top3_per_seed.png,
                surrogate_vs_hifi_mse.png}
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib.hifi_render import build_context, render_hifi, rho_band  # noqa: E402

OUT_DIR = SURVEY_DIR / "results" / "s017"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S016C_RUNS = SURVEY_DIR / "results" / "s016c_prime" / "runs.npz"

N_WORKERS = 8
TOP_K = 3
NOISE_SIGMA = 0.05
RHO_THRESHOLDS = {"A": 2.0, "B": 4.0, "C": 8.0}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_CTX_CACHE: dict = {}


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _get_ctx(seed: int) -> dict:
    if seed not in _CTX_CACHE:
        _CTX_CACHE[seed] = build_context(int(seed))
    return _CTX_CACHE[seed]


def render_one(args) -> dict:
    seed, row_idx, q0_final, omega_final = args
    ctx = _get_ctx(int(seed))
    pred = render_hifi(q0_final, omega_final, ctx)
    truth = ctx["mag_hifi_truth"]
    diff = pred - truth
    finite = np.isfinite(diff)
    if finite.all():
        hifi_mse = float(np.mean(diff ** 2))
    elif finite.any():
        hifi_mse = float(np.mean(diff[finite] ** 2))
    else:
        hifi_mse = float("inf")
    rho = float(np.sqrt(hifi_mse) / NOISE_SIGMA) if np.isfinite(hifi_mse) else float("inf")
    band = rho_band(rho) if np.isfinite(rho) else "D"
    return {
        "seed": int(seed),
        "row_idx": int(row_idx),
        "hifi_mse": hifi_mse,
        "rho": rho,
        "band": band,
        "n_finite": int(finite.sum()),
        "n_total": int(diff.size),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Loading s016c_prime/runs.npz ...", flush=True)
    d = np.load(str(S016C_RUNS), allow_pickle=True)
    seeds = sorted(np.unique(d["seed"]).tolist())
    print(f"  {len(seeds)} seeds: {seeds}", flush=True)

    # Per seed: select top-K lowest final_mse; record overall row indices
    work = []
    selection_meta = []  # (seed, overall_row, rank_within_seed, surrogate_mse, ...)
    for s in seeds:
        seed_mask = d["seed"] == s
        seed_rows = np.where(seed_mask)[0]
        seed_fmse = d["final_mse"][seed_rows]
        order = np.argsort(seed_fmse)
        top_rows = seed_rows[order[:TOP_K]]
        for rank, r in enumerate(top_rows, start=1):
            work.append(
                (
                    int(s),
                    int(r),
                    d["q0_final_wxyz"][r].copy(),
                    d["omega_final_rad"][r].copy(),
                )
            )
            selection_meta.append({
                "seed": int(s),
                "row_idx": int(r),
                "rank": int(rank),
                "surrogate_mse": float(d["final_mse"][r]),
                "q0_err_deg": float(d["q0_err_deg"][r]),
                "twin_err_deg": float(d["twin_err_deg"][r]),
                "omega_dir_err_deg": float(d["omega_dir_err_deg"][r]),
                "omega_mag_err_pct": float(d["omega_mag_err_pct"][r]),
                "harvest_idx": int(d["harvest_idx"][r]),
                "sobol_idx": int(d["sobol_idx"][r]),
                "omega_mag_fixed_err_pct": float(d["omega_mag_fixed_err_pct"][r]),
            })

    n_renders = len(work)
    print(f"Total hi-fi renders: {n_renders} ({len(seeds)} seeds × {TOP_K} top)",
          flush=True)
    print(f"Pool: {N_WORKERS}", flush=True)

    # ----------- Parallel render -----------
    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(render_one, work)
    wall = time.time() - t0
    avg_s = wall / n_renders
    print(f"Wall: {wall:.1f} s  ({avg_s:.1f} s/render avg)", flush=True)

    # ----------- Assemble + save (BEFORE plotting, per checkpoint design) -----
    rho_arr = np.array([r["rho"] for r in results], dtype=np.float64)
    hifi_mse_arr = np.array([r["hifi_mse"] for r in results], dtype=np.float64)
    band_arr = np.array([r["band"] for r in results])
    seed_arr = np.array([m["seed"] for m in selection_meta], dtype=np.int64)
    rank_arr = np.array([m["rank"] for m in selection_meta], dtype=np.int64)
    row_idx_arr = np.array([m["row_idx"] for m in selection_meta], dtype=np.int64)
    surr_mse_arr = np.array([m["surrogate_mse"] for m in selection_meta], dtype=np.float64)
    q0_err_arr = np.array([m["q0_err_deg"] for m in selection_meta], dtype=np.float64)
    twin_err_arr = np.array([m["twin_err_deg"] for m in selection_meta], dtype=np.float64)
    wd_err_arr = np.array([m["omega_dir_err_deg"] for m in selection_meta], dtype=np.float64)
    wm_err_arr = np.array([m["omega_mag_err_pct"] for m in selection_meta], dtype=np.float64)
    n_finite_arr = np.array([r["n_finite"] for r in results], dtype=np.int64)

    np.savez(
        str(OUT_DIR / "rho_top3.npz"),
        seed=seed_arr,
        rank=rank_arr,
        row_idx=row_idx_arr,
        surrogate_mse=surr_mse_arr,
        hifi_mse=hifi_mse_arr,
        rho=rho_arr,
        band=band_arr,
        q0_err_deg=q0_err_arr,
        twin_err_deg=twin_err_arr,
        omega_dir_err_deg=wd_err_arr,
        omega_mag_err_pct=wm_err_arr,
        n_finite=n_finite_arr,
        n_total=np.array([r["n_total"] for r in results], dtype=np.int64),
        q0_final_wxyz=np.array([w[2] for w in work], dtype=np.float64),
        omega_final_rad=np.array([w[3] for w in work], dtype=np.float64),
        harvest_idx=np.array([m["harvest_idx"] for m in selection_meta], dtype=np.int64),
        sobol_idx=np.array([m["sobol_idx"] for m in selection_meta], dtype=np.int64),
        omega_mag_fixed_err_pct=np.array(
            [m["omega_mag_fixed_err_pct"] for m in selection_meta], dtype=np.float64
        ),
    )
    print(f"Saved: {OUT_DIR / 'rho_top3.npz'}", flush=True)

    # ----------- Per-seed structured summary -----------
    per_seed_summary = {}
    winners = []  # rank-1 only, for the decisive question
    for s in seeds:
        sm = seed_arr == s
        ranks = rank_arr[sm]
        rhos = rho_arr[sm]
        bands = band_arr[sm]
        surrs = surr_mse_arr[sm]
        hifis = hifi_mse_arr[sm]
        q0es = q0_err_arr[sm]
        twins = twin_err_arr[sm]
        wdes = wd_err_arr[sm]
        wmes = wm_err_arr[sm]

        # rank-1 = cohort-selector winner
        i1 = int(np.where(ranks == 1)[0][0])
        winners.append({
            "seed": int(s),
            "rank": 1,
            "surrogate_mse": float(surrs[i1]),
            "hifi_mse": float(hifis[i1]),
            "rho": float(rhos[i1]),
            "band": str(bands[i1]),
            "q0_err_deg": float(q0es[i1]),
            "twin_err_deg": float(twins[i1]),
            "omega_dir_err_deg": float(wdes[i1]),
            "omega_mag_err_pct": float(wmes[i1]),
        })

        per_seed_summary[f"seed_{int(s):03d}"] = {
            "n_top": int(sm.sum()),
            "top_candidates": [
                {
                    "rank": int(ranks[k]),
                    "row_idx": int(row_idx_arr[sm][k]),
                    "surrogate_mse": float(surrs[k]),
                    "hifi_mse": float(hifis[k]),
                    "rho": float(rhos[k]),
                    "band": str(bands[k]),
                    "q0_err_deg": float(q0es[k]),
                    "twin_err_deg": float(twins[k]),
                    "omega_dir_err_deg": float(wdes[k]),
                    "omega_mag_err_pct": float(wmes[k]),
                    "surrogate_to_hifi_ratio": (
                        float(hifis[k] / surrs[k]) if surrs[k] > 0 and np.isfinite(hifis[k]) else None
                    ),
                }
                for k in range(int(sm.sum()))
            ],
            # any band A∪B in top-K?
            "best_band_in_topK": min(bands.tolist(), key=lambda b: "ABCD".index(b)),
            "best_rho_in_topK": float(np.min(rhos)),
            "best_rank_band_in_topK": int(
                ranks[int(np.argmin(rhos))]
            ),
        }

    # ----------- Decisive question -----------
    n_winners_AB = int(sum(1 for w in winners if w["band"] in ("A", "B")))
    n_winners_A = int(sum(1 for w in winners if w["band"] == "A"))
    n_seeds_with_AB_in_topK = int(sum(
        1 for s in seeds if per_seed_summary[f"seed_{int(s):03d}"]["best_band_in_topK"] in ("A", "B")
    ))

    decision = (
        "C_PRIME_VIABLE: ≥4/7 winners hi-fi-validate as Band A∪B → "
        "multi-solution architecture viable for production cohort scan."
        if n_winners_AB >= 4
        else
        "C_PRIME_KILLED: <4/7 winners hi-fi-validate as Band A∪B → "
        "multi-solution reframe rejected; pivot to S016-A (ω-grid) "
        "or design hierarchical architecture."
    )

    # surrogate-vs-hifi diagnostic across top-K
    finite_mask = np.isfinite(hifi_mse_arr) & (surr_mse_arr > 0)
    if finite_mask.sum() > 0:
        ratios = hifi_mse_arr[finite_mask] / surr_mse_arr[finite_mask]
        ratio_stats = {
            "n": int(finite_mask.sum()),
            "median": float(np.median(ratios)),
            "p10": float(np.percentile(ratios, 10)),
            "p90": float(np.percentile(ratios, 90)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
        }
    else:
        ratio_stats = None

    summary = {
        "n_workers": N_WORKERS,
        "wall_s": wall,
        "n_renders": n_renders,
        "noise_sigma": NOISE_SIGMA,
        "rho_thresholds": RHO_THRESHOLDS,
        "top_k_per_seed": TOP_K,
        "seeds": seeds,
        "winners_rank1": winners,
        "n_winners_band_A": n_winners_A,
        "n_winners_band_AB": n_winners_AB,
        "n_seeds_with_AB_in_topK": n_seeds_with_AB_in_topK,
        "n_seeds": len(seeds),
        "hifi_to_surrogate_mse_ratio_topK": ratio_stats,
        "per_seed": per_seed_summary,
        "decision": decision,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR / 'summary.json'}", flush=True)

    # ----------- Print human-readable summary -----------
    print()
    print("=" * 78)
    print("PER-SEED COHORT-SELECTOR WINNER (rank=1)")
    print("=" * 78)
    print(
        f"{'seed':>4}  {'surr_mse':>10}  {'hifi_mse':>10}  {'ratio':>6}  "
        f"{'rho':>7}  {'band':>4}  {'q0_err':>7}  {'wd_err':>7}  {'wm_err':>7}"
    )
    for w in winners:
        ratio = (w["hifi_mse"] / w["surrogate_mse"]
                 if w["surrogate_mse"] > 0 and np.isfinite(w["hifi_mse"]) else float("nan"))
        print(
            f"{w['seed']:>4}  {w['surrogate_mse']:>10.4e}  "
            f"{w['hifi_mse']:>10.4e}  {ratio:>6.3f}  "
            f"{w['rho']:>7.3f}  {w['band']:>4}  "
            f"{w['q0_err_deg']:>7.2f}  {w['omega_dir_err_deg']:>7.2f}  "
            f"{w['omega_mag_err_pct']:>+7.2f}"
        )
    print()
    print(f"  WINNERS Band A: {n_winners_A}/{len(seeds)}")
    print(f"  WINNERS Band A∪B: {n_winners_AB}/{len(seeds)}  "
          f"(threshold ≥4/7 for C' viability)")
    print(f"  SEEDS with ANY top-{TOP_K} candidate Band A∪B: "
          f"{n_seeds_with_AB_in_topK}/{len(seeds)}")
    print()
    print(f"  >>> DECISION: {decision} <<<")
    print()

    print("=" * 78)
    print(f"PER-SEED TOP-{TOP_K} CANDIDATES (sorted by surrogate MSE)")
    print("=" * 78)
    for s in seeds:
        ps = per_seed_summary[f"seed_{int(s):03d}"]
        print(f"\nseed {s} (best ρ in top-{TOP_K} = {ps['best_rho_in_topK']:.3f}, "
              f"band {ps['best_band_in_topK']}, at rank {ps['best_rank_band_in_topK']}):")
        for c in ps["top_candidates"]:
            ratio_str = (
                f"{c['surrogate_to_hifi_ratio']:.3f}"
                if c["surrogate_to_hifi_ratio"] is not None else "  inf"
            )
            print(
                f"  #{c['rank']}: surr_mse={c['surrogate_mse']:.4e}  "
                f"hifi_mse={c['hifi_mse']:.4e}  hi/surr={ratio_str}  "
                f"ρ={c['rho']:7.3f} {c['band']}  "
                f"q0_err={c['q0_err_deg']:6.2f}°  ωd={c['omega_dir_err_deg']:6.2f}°  "
                f"ωm={c['omega_mag_err_pct']:+6.2f}%"
            )

    if ratio_stats is not None:
        print()
        print("=" * 78)
        print("HI-FI / SURROGATE MSE RATIO (top-K, finite, surr>0)")
        print("=" * 78)
        print(
            f"  n={ratio_stats['n']}  "
            f"median={ratio_stats['median']:.3f}  "
            f"p10={ratio_stats['p10']:.3f}  "
            f"p90={ratio_stats['p90']:.3f}  "
            f"min={ratio_stats['min']:.3f}  "
            f"max={ratio_stats['max']:.3f}"
        )
        print(
            "  (s014 cohort non-basin reference: median 1.000, "
            "p10-p90 [0.998, 1.006])"
        )

    # ----------- Plots -----------
    # Plot 1: per-seed top-3 ρ as bars
    fig, ax = plt.subplots(figsize=(11, 5))
    x_seed = np.arange(len(seeds))
    width = 0.25
    for k in range(TOP_K):
        rhos_k = []
        for s in seeds:
            sm = (seed_arr == s) & (rank_arr == (k + 1))
            rhos_k.append(float(rho_arr[sm][0]) if sm.any() else float("nan"))
        # clip inf for plot
        rhos_clipped = np.clip(np.array(rhos_k), 1e-2, 200.0)
        ax.bar(x_seed + (k - 1) * width, rhos_clipped, width=width,
               label=f"rank {k+1}",
               color=["C0", "C1", "C2"][k], edgecolor="k", alpha=0.85)
    for thr_name, thr_val in RHO_THRESHOLDS.items():
        ax.axhline(thr_val, linestyle="--", linewidth=1,
                   color={"A": "green", "B": "orange", "C": "red"}[thr_name],
                   label=f"ρ={thr_val:.0f} ({thr_name}↔next)", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xticks(x_seed)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("ρ = √(hi-fi MSE) / 0.05  (log)")
    ax.set_title(
        f"s017 — hi-fi ρ for top-{TOP_K} surrogate-MSE candidates per seed "
        f"(s016c_prime, n={n_renders})"
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, which="both", axis="y")
    fig.tight_layout()
    p1 = OUT_DIR / "rho_top3_per_seed.png"
    fig.savefig(str(p1), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}", flush=True)

    # Plot 2: surrogate vs hi-fi MSE scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = {"A": "green", "B": "orange", "C": "red", "D": "k"}
    plotted_bands = set()
    for k in range(len(work)):
        if not np.isfinite(hifi_mse_arr[k]) or surr_mse_arr[k] <= 0:
            continue
        b = band_arr[k]
        lbl = f"Band {b}" if b not in plotted_bands else None
        plotted_bands.add(b)
        ax.scatter(
            surr_mse_arr[k], hifi_mse_arr[k],
            c=cmap.get(b, "k"), s=70 - 20 * (rank_arr[k] - 1),
            edgecolor="k", alpha=0.8, label=lbl, marker="o",
        )
    if surr_mse_arr.size > 0:
        finite_pts = np.isfinite(hifi_mse_arr) & (surr_mse_arr > 0)
        all_vals = np.concatenate([surr_mse_arr[finite_pts], hifi_mse_arr[finite_pts]])
        if all_vals.size > 0:
            lo, hi = float(np.min(all_vals)) * 0.5, float(np.max(all_vals)) * 2.0
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x (surrogate = hi-fi)")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("surrogate MSE (mag²)")
    ax.set_ylabel("hi-fi MSE (mag²)")
    ax.set_title(f"s017 — surrogate ↔ hi-fi MSE for top-{TOP_K} per seed (n={n_renders})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p2 = OUT_DIR / "surrogate_vs_hifi_mse.png"
    fig.savefig(str(p2), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}", flush=True)


if __name__ == "__main__":
    main()
