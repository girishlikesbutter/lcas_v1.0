"""s014 — Cohort-scale hi-fi ρ-band of s011 non-in-basin + s005.

Extension of s013. Two cohorts:
  (A) s011 NOT-in-basin AND NOT-seed-10 final states (540 LM landings
      across 9 recoverable seeds: 6/21/28/41/44/48/60/84/91, 56-63 ICs
      per seed).
  (B) s005 all 50 LM landings (5 seeds × 10 deterministic+random ICs).
      Bundles the in-basin sanity check (30/50) with the basin-edge /
      escape cases (20/50).

Decisive question: does the surrogate ↔ hi-fi rank correlation observed
on s012a seed 10 (Spearman 0.999, hi-fi-best == surrogate-best) generalise
to the s011 9-recoverable-seed cohort? If YES, the cohort architecture's
"select lowest surrogate-MSE per seed" selector is directly trustworthy
without a hi-fi rerank step. If NO, identify the seeds where surrogate
mis-ranks and whether multi-solution candidates (ρ < 4) exist in the
non-best surrogate landings.

ρ = √(hi-fi MSE) / 0.05 per concepts/rho_band.md. Bands A/B/C/D as in s013.
Survey acceptance bar: ρ < 4 (Band A∪B = multi-solution / valid recovery).

Forward model: lib/hifi_render.py (already smoke-tested on seeds 6/10/91).

Per s011 wall-time observation: ~73 s per render, Pool(8). Predicted wall
for 590 renders: ~5400 s = 90 min. Kill criterion: 2× = 180 min.

Output:
  results/s014/{rho_s011_nb.npz, rho_s005.npz, summary.json,
                surrogate_vs_hifi_scatter.png, rho_per_seed.png}
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
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))  # for src.*

from lib.hifi_render import build_context, render_hifi, rho_band  # noqa: E402

OUT_DIR = SURVEY_DIR / "results" / "s014"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S011_RUNS = SURVEY_DIR / "results" / "s011" / "runs.npz"
S005_RUNS = SURVEY_DIR / "results" / "s005" / "runs.npz"

N_WORKERS = 8

NOISE_SIGMA = 0.05
RHO_THRESHOLDS = {"A": 2.0, "B": 4.0, "C": 8.0}

# ---------------------------------------------------------------------------
# Worker — lazy per-process ctx cache. Same pattern as s013.
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
    cohort, src_row, seed, ic_idx, q0_final, omega_final = args
    ctx = _get_ctx(int(seed))
    pred = render_hifi(q0_final, omega_final, ctx)
    truth = ctx["mag_hifi_truth"]
    diff = pred - truth
    hifi_mse = float(np.mean(diff ** 2))
    rho = float(np.sqrt(hifi_mse) / NOISE_SIGMA)
    band = rho_band(rho)
    return {
        "cohort": cohort,
        "src_row": int(src_row),
        "seed": int(seed),
        "ic_idx": int(ic_idx),
        "hifi_mse": hifi_mse,
        "rho": rho,
        "band": band,
    }


# ---------------------------------------------------------------------------
# Spearman without scipy (rank-correlation; ties broken by average rank).
# ---------------------------------------------------------------------------


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-ranks rank function (matches scipy.stats.rankdata default)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-indexed average
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading s011 cohort A: not-in-basin AND not-seed-10 ...", flush=True)
    d11 = np.load(str(S011_RUNS))
    s11_target_mask = (~d11["truth_basin_strict"]) & (d11["seed"] != 10)
    s11_rows = np.where(s11_target_mask)[0]
    cohort_a = []
    for i in s11_rows:
        cohort_a.append(
            (
                "s011",
                int(i),
                int(d11["seed"][i]),
                int(d11["ic_idx"][i]),
                d11["q0_final_wxyz"][i].copy(),
                d11["omega_final_rad"][i].copy(),
            )
        )
    print(f"  cohort A (s011 non-basin non-seed10): {len(cohort_a)}", flush=True)

    print("Loading s005 cohort B: all 50 LM landings ...", flush=True)
    d5 = np.load(str(S005_RUNS))
    cohort_b = []
    for i in range(len(d5["seed"])):
        cohort_b.append(
            (
                "s005",
                int(i),
                int(d5["seed"][i]),
                int(d5["ic_idx"][i]),
                d5["q0_final_wxyz"][i].copy(),
                d5["omega_final_rad"][i].copy(),
            )
        )
    print(f"  cohort B (s005 all): {len(cohort_b)}", flush=True)

    work = cohort_a + cohort_b
    n_a = len(cohort_a)
    n_b = len(cohort_b)
    print(f"Total renders: {len(work)} (Pool={N_WORKERS})", flush=True)
    print(f"Predicted wall: ~{len(work) / N_WORKERS * 73 / 60:.0f} min "
          f"(at 73 s/render observed in s013)", flush=True)

    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(render_one, work)
    wall = time.time() - t0
    avg_ms = wall / len(work) * 1000.0
    print(f"Wall: {wall:.1f} s ({wall/60:.1f} min)  "
          f"({avg_ms:.0f} ms/render avg, {wall * N_WORKERS / len(work):.1f} s/render per worker)",
          flush=True)

    res_a = results[:n_a]
    res_b = results[n_a:]

    # ---------------- Save NPZs (aligned with source row indices) ------------------

    rho_a = np.array([r["rho"] for r in res_a])
    bands_a = np.array([r["band"] for r in res_a])
    np.savez(
        str(OUT_DIR / "rho_s011_nb.npz"),
        s011_row=np.array([r["src_row"] for r in res_a], dtype=np.int64),
        seed=np.array([r["seed"] for r in res_a], dtype=np.int64),
        ic_idx=np.array([r["ic_idx"] for r in res_a], dtype=np.int64),
        hifi_mse=np.array([r["hifi_mse"] for r in res_a], dtype=np.float64),
        rho=rho_a,
        band=bands_a,
        surrogate_mse=d11["final_mse"][s11_rows].copy(),
        q0_err_deg=d11["q0_err_deg"][s11_rows].copy(),
        omega_dir_err_deg=d11["omega_dir_err_deg"][s11_rows].copy(),
        omega_mag_err_pct=d11["omega_mag_err_pct"][s11_rows].copy(),
        q0_final_wxyz=d11["q0_final_wxyz"][s11_rows].copy(),
        omega_final_rad=d11["omega_final_rad"][s11_rows].copy(),
    )
    print(f"Saved: {OUT_DIR / 'rho_s011_nb.npz'}", flush=True)

    rho_b = np.array([r["rho"] for r in res_b])
    bands_b = np.array([r["band"] for r in res_b])
    np.savez(
        str(OUT_DIR / "rho_s005.npz"),
        s005_row=np.array([r["src_row"] for r in res_b], dtype=np.int64),
        seed=np.array([r["seed"] for r in res_b], dtype=np.int64),
        ic_idx=np.array([r["ic_idx"] for r in res_b], dtype=np.int64),
        label=d5["label"].copy(),
        kind=d5["kind"].copy(),
        tier=d5["tier"].copy(),
        hifi_mse=np.array([r["hifi_mse"] for r in res_b], dtype=np.float64),
        rho=rho_b,
        band=bands_b,
        surrogate_mse=d5["final_mse"].copy(),
        q0_err_deg=d5["q0_err_deg"].copy(),
        omega_dir_err_deg=d5["omega_dir_err_deg"].copy(),
        omega_mag_err_pct=d5["omega_mag_err_pct"].copy(),
        truth_basin_strict=d5["truth_basin_strict"].copy(),
        q0_final_wxyz=d5["q0_final_wxyz"].copy(),
        omega_final_rad=d5["omega_final_rad"].copy(),
    )
    print(f"Saved: {OUT_DIR / 'rho_s005.npz'}", flush=True)

    # ---------------- Per-seed analysis: cohort A (s011 non-basin) ------------------
    # Need to combine with s011 in-basin (already in s013/rho_in_basin.npz) to get
    # the per-seed FULL rank correlation across all 64 ICs per seed (in-basin + not).

    s013_in_basin = np.load(SURVEY_DIR / "results" / "s013" / "rho_in_basin.npz")
    # Build a mapping s011_row -> rho for both (in-basin + non-basin)
    full_rho_by_row = {}
    full_seed_by_row = {}
    full_surr_by_row = {}
    for i, row in enumerate(s013_in_basin["s011_runs_row"]):
        full_rho_by_row[int(row)] = float(s013_in_basin["rho"][i])
        full_seed_by_row[int(row)] = int(s013_in_basin["seed"][i])
        full_surr_by_row[int(row)] = float(s013_in_basin["surrogate_mse"][i])
    for i, row in enumerate(np.array([r["src_row"] for r in res_a])):
        full_rho_by_row[int(row)] = float(rho_a[i])
        full_seed_by_row[int(row)] = int(res_a[i]["seed"])
        full_surr_by_row[int(row)] = float(d11["final_mse"][row])

    # Per-seed across all 64 ICs (excluding seed 10)
    per_seed_a = {}
    for s in [6, 21, 28, 41, 44, 48, 60, 84, 91]:
        rows_s = [r for r in full_rho_by_row if full_seed_by_row[r] == s]
        if len(rows_s) < 5:
            continue
        rhos = np.array([full_rho_by_row[r] for r in rows_s])
        surrs = np.array([full_surr_by_row[r] for r in rows_s])

        # Find s011 row that is the surrogate-best for this seed
        surr_argmin_idx_local = int(np.argmin(surrs))
        hifi_argmin_idx_local = int(np.argmin(rhos))
        surr_best_row = rows_s[surr_argmin_idx_local]
        hifi_best_row = rows_s[hifi_argmin_idx_local]

        # Surrogate-best rank in hi-fi space
        rho_ranks = _rankdata(rhos)  # 1-indexed; smaller rho = lower rank
        rank_of_surr_best_in_rho = float(rho_ranks[surr_argmin_idx_local])

        sp = spearman(surrs, rhos)
        # Top-K agreement
        k = min(5, len(rows_s))
        top_k_surr = set(np.argsort(surrs)[:k].tolist())
        top_k_rho = set(np.argsort(rhos)[:k].tolist())
        topk_overlap = int(len(top_k_surr & top_k_rho))

        bands_s = [s013_in_basin["band"][i].decode() if isinstance(s013_in_basin["band"][i], bytes) else str(s013_in_basin["band"][i])
                   for i in range(len(s013_in_basin["band"])) if int(s013_in_basin["seed"][i]) == s]
        bands_s += [r["band"] for r in res_a if r["seed"] == s]
        band_count = {b: bands_s.count(b) for b in "ABCD"}

        per_seed_a[s] = {
            "n_total": len(rows_s),
            "spearman_surr_vs_rho": sp,
            "topk_overlap": topk_overlap,
            "topk_k": k,
            "rho_min": float(np.min(rhos)),
            "rho_median": float(np.median(rhos)),
            "rho_max": float(np.max(rhos)),
            "surrogate_mse_min": float(np.min(surrs)),
            "rho_of_surrogate_best": float(rhos[surr_argmin_idx_local]),
            "rho_best": float(rhos[hifi_argmin_idx_local]),
            "rank_of_surrogate_best_in_rho": rank_of_surr_best_in_rho,
            "surrogate_best_row": int(surr_best_row),
            "hifi_best_row": int(hifi_best_row),
            "surrogate_best_eq_hifi_best": bool(surr_best_row == hifi_best_row),
            "band_counts": band_count,
            "n_band_A_or_B": band_count.get("A", 0) + band_count.get("B", 0),
        }

    # Cohort A overall band counts (s011 non-basin only — in-basin is in s013)
    band_count_a = {b: int((bands_a == b).sum()) for b in "ABCD"}

    # Cohort B (s005)
    band_count_b = {b: int((bands_b == b).sum()) for b in "ABCD"}
    per_seed_b = {}
    for s in [6, 18, 28, 41, 91]:
        mask_s = np.array([r["seed"] == s for r in res_b])
        if not mask_s.any():
            continue
        rhos_s = rho_b[mask_s]
        surrs_s = d5["final_mse"][mask_s]
        in_basin_s = d5["truth_basin_strict"][mask_s]
        bands_sub = bands_b[mask_s]
        sp = spearman(surrs_s, rhos_s)
        n_band_a = int((bands_sub == "A").sum())
        n_band_b = int((bands_sub == "B").sum())
        per_seed_b[s] = {
            "n_total": int(mask_s.sum()),
            "n_in_basin": int(in_basin_s.sum()),
            "rho_min": float(np.min(rhos_s)),
            "rho_median": float(np.median(rhos_s)),
            "rho_max": float(np.max(rhos_s)),
            "spearman_surr_vs_rho": sp,
            "band_A": n_band_a,
            "band_B": n_band_b,
            "band_C": int((bands_sub == "C").sum()),
            "band_D": int((bands_sub == "D").sum()),
        }

    # ---------------- Decisive aggregates ------------------
    # Cohort generalisation: how many seeds pass Spearman >= 0.95 and
    # surrogate-best == hi-fi-best?
    n_seeds_pass_spearman = sum(
        1 for s, ps in per_seed_a.items() if ps["spearman_surr_vs_rho"] >= 0.95
    )
    n_seeds_surr_best_eq_hifi = sum(
        1 for s, ps in per_seed_a.items() if ps["surrogate_best_eq_hifi_best"]
    )

    # Multi-solution candidates outside in-basin per seed
    multi_sol_per_seed = {}
    for s, ps in per_seed_a.items():
        # n_band_A_or_B includes the 36 in-basin (which are all Band A from s013).
        # Subtract in-basin count (= 4 per seed for the 9 recoverable seeds, except seed 10 = 0)
        # Actually s011 in-basin distribution: 6→8, 21→3, 28→7, 41→2, 44→5, 48→1, 60→4, 84→3, 91→3.
        in_basin_count_per_seed = {6: 8, 21: 3, 28: 7, 41: 2, 44: 5, 48: 1, 60: 4, 84: 3, 91: 3}
        nb_band_AB = ps["n_band_A_or_B"] - in_basin_count_per_seed.get(s, 0)
        multi_sol_per_seed[s] = max(0, nb_band_AB)

    n_seeds_with_multi_sol = sum(1 for v in multi_sol_per_seed.values() if v > 0)

    summary = {
        "n_workers": N_WORKERS,
        "wall_s": wall,
        "noise_sigma": NOISE_SIGMA,
        "rho_thresholds": RHO_THRESHOLDS,
        "cohort_a_s011_non_basin": {
            "description": "s011 non-in-basin AND non-seed-10 final states",
            "n": int(n_a),
            "rho_min": float(np.min(rho_a)),
            "rho_median": float(np.median(rho_a)),
            "rho_max": float(np.max(rho_a)),
            "band_counts": band_count_a,
            "per_seed_full_64": per_seed_a,  # combined w/ s013 in-basin
            "n_seeds_spearman_ge_0p95": n_seeds_pass_spearman,
            "n_seeds_surr_best_eq_hifi_best": n_seeds_surr_best_eq_hifi,
            "n_seeds_with_multi_sol_outside_basin": n_seeds_with_multi_sol,
            "multi_sol_per_seed_outside_basin": multi_sol_per_seed,
        },
        "cohort_b_s005": {
            "description": "s005 all 50 LM landings",
            "n": int(n_b),
            "rho_min": float(np.min(rho_b)),
            "rho_median": float(np.median(rho_b)),
            "rho_max": float(np.max(rho_b)),
            "band_counts": band_count_b,
            "per_seed": per_seed_b,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR / 'summary.json'}", flush=True)

    # ---------------- Print summary ------------------
    print()
    print("=== COHORT A: s011 non-in-basin non-seed-10 ===")
    print(f"  n={n_a}  ρ-min={np.min(rho_a):.3f}  ρ-median={np.median(rho_a):.3f}  ρ-max={np.max(rho_a):.3f}")
    print(f"  Bands: A={band_count_a['A']}  B={band_count_a['B']}  C={band_count_a['C']}  D={band_count_a['D']}")
    print()
    print(f"  Per-seed combined full-64 (s011 in-basin from s013 + non-basin from s014):")
    print(f"    {'seed':>5} {'n':>4} {'spearman':>9} {'topK':>4} {'rho_min':>8} {'rho_surrB':>10} {'surrB==hfB':>12} {'A+B(out_basin)':>15}")
    for s in sorted(per_seed_a):
        ps = per_seed_a[s]
        nb_AB = multi_sol_per_seed.get(s, 0)
        print(f"    {s:>5} {ps['n_total']:>4} {ps['spearman_surr_vs_rho']:>9.3f} "
              f"{ps['topk_overlap']:>2}/{ps['topk_k']:<2} {ps['rho_min']:>8.3f} "
              f"{ps['rho_of_surrogate_best']:>10.3f} {str(ps['surrogate_best_eq_hifi_best']):>12} {nb_AB:>15}")
    print(f"  Cohort generalisation: {n_seeds_pass_spearman}/9 pass Spearman ≥ 0.95")
    print(f"                         {n_seeds_surr_best_eq_hifi}/9 have surrogate-best == hi-fi-best")
    print(f"                         {n_seeds_with_multi_sol}/9 have multi-solution candidates outside basin (Band A∪B)")
    print()
    print("=== COHORT B: s005 all ===")
    print(f"  n={n_b}  ρ-min={np.min(rho_b):.3f}  ρ-median={np.median(rho_b):.3f}  ρ-max={np.max(rho_b):.3f}")
    print(f"  Bands: A={band_count_b['A']}  B={band_count_b['B']}  C={band_count_b['C']}  D={band_count_b['D']}")
    print(f"  Per-seed:")
    for s, ps in sorted(per_seed_b.items()):
        print(f"    seed {s:3d}: n={ps['n_total']:2d} (n_basin={ps['n_in_basin']})  "
              f"ρ_min={ps['rho_min']:.3f}  spearman={ps['spearman_surr_vs_rho']:>6.3f}  "
              f"A={ps['band_A']} B={ps['band_B']} C={ps['band_C']} D={ps['band_D']}")
    print()

    # ---------------- Plots ------------------

    # Plot 1: surrogate vs hi-fi scatter, all s011 non-basin candidates, colour by seed.
    fig, ax = plt.subplots(figsize=(8, 6))
    seeds_a = np.array([r["seed"] for r in res_a])
    surrs_a = d11["final_mse"][s11_rows]
    cmap = plt.cm.tab10
    seed_list = [6, 21, 28, 41, 44, 48, 60, 84, 91]
    for i, s in enumerate(seed_list):
        m = seeds_a == s
        if not m.any():
            continue
        ax.scatter(surrs_a[m], rho_a[m] ** 2 * NOISE_SIGMA ** 2,
                   color=cmap(i % 10), label=f"seed {s}", s=18, alpha=0.65,
                   edgecolor="k", linewidth=0.3)
    # y = x reference (perfect surrogate-MSE = hi-fi-MSE)
    ax_lo = max(min(surrs_a.min(), (rho_a ** 2 * NOISE_SIGMA ** 2).min()) * 0.5, 1e-4)
    ax_hi = max(surrs_a.max(), (rho_a ** 2 * NOISE_SIGMA ** 2).max()) * 1.5
    ax.plot([ax_lo, ax_hi], [ax_lo, ax_hi], "k--", linewidth=0.8, label="surrogate = hi-fi")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("surrogate MSE [mag²]")
    ax.set_ylabel("hi-fi MSE [mag²]")
    ax.set_title(f"s014 cohort A: surrogate vs hi-fi MSE on {n_a} s011 non-basin candidates")
    ax.legend(fontsize=8, loc="lower right", ncol=3)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = OUT_DIR / "surrogate_vs_hifi_scatter.png"
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}", flush=True)

    # Plot 2: ρ histogram per seed (combined full 64 from s013 + s014).
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for ax, s in zip(axes.flatten(), seed_list):
        rows_s = [r for r in full_rho_by_row if full_seed_by_row[r] == s]
        rhos = np.array([full_rho_by_row[r] for r in rows_s])
        clipped = np.clip(rhos, 1e-3, None)
        ax.hist(clipped, bins=np.logspace(-3, 3, 50), color="C0", alpha=0.7, edgecolor="k")
        ax.axvline(2.0, color="green", linestyle="--", linewidth=0.8)
        ax.axvline(4.0, color="orange", linestyle="--", linewidth=0.8)
        ax.axvline(8.0, color="red", linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_title(f"seed {s} (n={len(rhos)})", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("ρ")
    for ax in axes[:, 0]:
        ax.set_ylabel("count")
    fig.suptitle("s014 — full-64 ρ distribution per seed (s011 in-basin from s013 + non-basin from s014)",
                 fontsize=12)
    fig.tight_layout()
    p = OUT_DIR / "rho_per_seed.png"
    fig.savefig(str(p), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}", flush=True)


if __name__ == "__main__":
    main()
