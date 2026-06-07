"""s013 — Hi-fi ρ-band classification of LM-converged candidates.

Two cohorts:
  (a) s011 in-basin landings — 36 across 9 recoverable seeds.
      Sanity check: should all be Band A (ρ ≤ 1).
  (b) s012a seed-10 N=256 final states (includes s011's first 64).
      Decisive: are the seed-10 competing basins observationally
      indistinguishable from truth (Band A∪B → MULTI-SOLUTION) or
      visibly worse (Band C∪D → GENUINE FAILURE)?

ρ = √(hi-fi MSE) / 0.05 per `concepts/rho_band.md`. Bands:
  A: ρ < 2  (truth-grade fit, below noise)
  B: 2 ≤ ρ < 4 (acceptable, within ~2× noise)
  C: 4 ≤ ρ < 8 (marginal — visually presentable, not publishable)
  D: ρ ≥ 8  (failure)
Survey acceptance bar: ρ < 4 (Band A∪B = multi-solution / valid recovery).

Forward model: `lib/hifi_render.py` (smoke-tested round-trip on seeds
6/10/91 to machine precision before this run).

Output:
  results/s013/{rho_in_basin.npz, rho_seed10.npz, summary.json,
                rho_distribution.png}
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

OUT_DIR = SURVEY_DIR / "results" / "s013"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S011_RUNS = SURVEY_DIR / "results" / "s011" / "runs.npz"
S012A_RUNS = SURVEY_DIR / "results" / "s012a" / "runs_combined.npz"

N_WORKERS = 8

NOISE_SIGMA = 0.05
RHO_THRESHOLDS = {"A": 2.0, "B": 4.0, "C": 8.0}

# ---------------------------------------------------------------------------
# Worker — lazy per-process ctx cache
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
    seed, ic_idx, q0_final, omega_final = args
    ctx = _get_ctx(int(seed))
    pred = render_hifi(q0_final, omega_final, ctx)
    truth = ctx["mag_hifi_truth"]
    diff = pred - truth
    hifi_mse = float(np.mean(diff ** 2))
    rho = float(np.sqrt(hifi_mse) / NOISE_SIGMA)
    band = rho_band(rho)
    return {
        "seed": int(seed),
        "ic_idx": int(ic_idx),
        "hifi_mse": hifi_mse,
        "rho": rho,
        "band": band,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading s011 in-basin landings (cohort A) ...", flush=True)
    d11 = np.load(str(S011_RUNS))
    in_basin_mask = d11["truth_basin_strict"]
    cohort_a_rows = np.where(in_basin_mask)[0]
    cohort_a = []
    for i in cohort_a_rows:
        cohort_a.append(
            (
                int(d11["seed"][i]),
                int(d11["ic_idx"][i]),
                d11["q0_final_wxyz"][i].copy(),
                d11["omega_final_rad"][i].copy(),
            )
        )
    print(f"  cohort A: {len(cohort_a)} in-basin landings", flush=True)

    print("Loading s012a seed-10 N=256 final states (cohort B) ...", flush=True)
    d12 = np.load(str(S012A_RUNS))
    cohort_b = []
    for i in range(len(d12["ic_idx"])):
        cohort_b.append(
            (
                10,
                int(d12["ic_idx"][i]),
                d12["q0_final_wxyz"][i].copy(),
                d12["omega_final_rad"][i].copy(),
            )
        )
    print(f"  cohort B: {len(cohort_b)} seed-10 final states", flush=True)

    work = cohort_a + cohort_b
    n_a = len(cohort_a)
    n_b = len(cohort_b)
    print(f"Total renders: {len(work)} (Pool={N_WORKERS})", flush=True)

    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(render_one, work)
    wall = time.time() - t0
    avg_ms = wall / len(work) * 1000.0
    print(f"Wall: {wall:.1f} s  ({avg_ms:.0f} ms/render avg)", flush=True)

    res_a = results[:n_a]
    res_b = results[n_a:]

    # Save cohort A
    bands_a = np.array([r["band"] for r in res_a])
    np.savez(
        str(OUT_DIR / "rho_in_basin.npz"),
        seed=np.array([r["seed"] for r in res_a], dtype=np.int64),
        ic_idx=np.array([r["ic_idx"] for r in res_a], dtype=np.int64),
        hifi_mse=np.array([r["hifi_mse"] for r in res_a], dtype=np.float64),
        rho=np.array([r["rho"] for r in res_a], dtype=np.float64),
        band=bands_a,
        s011_runs_row=cohort_a_rows.astype(np.int64),
        surrogate_mse=d11["final_mse"][cohort_a_rows].copy(),
        q0_err_deg=d11["q0_err_deg"][cohort_a_rows].copy(),
        omega_dir_err_deg=d11["omega_dir_err_deg"][cohort_a_rows].copy(),
        omega_mag_err_pct=d11["omega_mag_err_pct"][cohort_a_rows].copy(),
        q0_final_wxyz=d11["q0_final_wxyz"][cohort_a_rows].copy(),
        omega_final_rad=d11["omega_final_rad"][cohort_a_rows].copy(),
    )
    print(f"Saved: {OUT_DIR / 'rho_in_basin.npz'}", flush=True)

    # Save cohort B
    bands_b = np.array([r["band"] for r in res_b])
    np.savez(
        str(OUT_DIR / "rho_seed10.npz"),
        seed=np.full(n_b, 10, dtype=np.int64),
        ic_idx=np.array([r["ic_idx"] for r in res_b], dtype=np.int64),
        hifi_mse=np.array([r["hifi_mse"] for r in res_b], dtype=np.float64),
        rho=np.array([r["rho"] for r in res_b], dtype=np.float64),
        band=bands_b,
        surrogate_mse=d12["final_mse"].copy(),
        q0_err_deg=d12["q0_err_deg"].copy(),
        omega_dir_err_deg=d12["omega_dir_err_deg"].copy(),
        omega_mag_err_pct=d12["omega_mag_err_pct"].copy(),
        truth_basin_strict=d12["truth_basin_strict"].copy(),
        twin_basin_strict=d12["twin_basin_strict"].copy(),
        q0_final_wxyz=d12["q0_final_wxyz"].copy(),
        omega_final_rad=d12["omega_final_rad"].copy(),
    )
    print(f"Saved: {OUT_DIR / 'rho_seed10.npz'}", flush=True)

    # ---------------- Summaries ------------------
    rho_a = np.array([r["rho"] for r in res_a])
    rho_b = np.array([r["rho"] for r in res_b])

    band_count_a = {b: int((bands_a == b).sum()) for b in "ABCD"}
    band_count_b = {b: int((bands_b == b).sum()) for b in "ABCD"}

    # Per-seed cohort-A summary
    per_seed_a = {}
    for r in res_a:
        per_seed_a.setdefault(r["seed"], []).append(r)
    per_seed_a_summary = {}
    for s, rs in sorted(per_seed_a.items()):
        bs = [r["band"] for r in rs]
        per_seed_a_summary[int(s)] = {
            "n": len(rs),
            "rho_min": float(np.min([r["rho"] for r in rs])),
            "rho_median": float(np.median([r["rho"] for r in rs])),
            "rho_max": float(np.max([r["rho"] for r in rs])),
            "band_A": int(sum(1 for b in bs if b == "A")),
            "band_B": int(sum(1 for b in bs if b == "B")),
            "band_C": int(sum(1 for b in bs if b == "C")),
            "band_D": int(sum(1 for b in bs if b == "D")),
        }

    # Cohort-B decision: any ρ ≤ 4 → multi-solution candidate exists
    n_band_a_or_b = int((rho_b <= RHO_THRESHOLDS["B"]).sum())
    decision_b = "MULTI_SOLUTION" if n_band_a_or_b > 0 else "GENUINE_FAILURE"

    # Lowest-ρ representative per cohort B (most "indistinguishable" candidate)
    best_b_idx = int(np.argmin(rho_b))
    best_b = {
        "ic_idx": int(d12["ic_idx"][best_b_idx]),
        "rho": float(rho_b[best_b_idx]),
        "hifi_mse": float(res_b[best_b_idx]["hifi_mse"]),
        "surrogate_mse": float(d12["final_mse"][best_b_idx]),
        "q0_err_deg": float(d12["q0_err_deg"][best_b_idx]),
        "omega_dir_err_deg": float(d12["omega_dir_err_deg"][best_b_idx]),
        "omega_mag_err_pct": float(d12["omega_mag_err_pct"][best_b_idx]),
        "band": str(bands_b[best_b_idx]),
    }

    # Surrogate-best (lowest final_mse) per cohort B
    surr_best_idx = int(np.argmin(d12["final_mse"]))
    surr_best_b = {
        "ic_idx": int(d12["ic_idx"][surr_best_idx]),
        "rho": float(rho_b[surr_best_idx]),
        "hifi_mse": float(res_b[surr_best_idx]["hifi_mse"]),
        "surrogate_mse": float(d12["final_mse"][surr_best_idx]),
        "q0_err_deg": float(d12["q0_err_deg"][surr_best_idx]),
        "omega_dir_err_deg": float(d12["omega_dir_err_deg"][surr_best_idx]),
        "omega_mag_err_pct": float(d12["omega_mag_err_pct"][surr_best_idx]),
        "band": str(bands_b[surr_best_idx]),
    }

    summary = {
        "n_workers": N_WORKERS,
        "wall_s": wall,
        "noise_sigma": NOISE_SIGMA,
        "rho_thresholds": RHO_THRESHOLDS,
        "cohort_a": {
            "description": "s011 in-basin landings (truth_basin_strict=True)",
            "n": int(n_a),
            "rho_min": float(np.min(rho_a)),
            "rho_median": float(np.median(rho_a)),
            "rho_max": float(np.max(rho_a)),
            "band_counts": band_count_a,
            "per_seed": per_seed_a_summary,
        },
        "cohort_b_seed10": {
            "description": "s012a seed-10 N=256 final states",
            "n": int(n_b),
            "rho_min": float(np.min(rho_b)),
            "rho_median": float(np.median(rho_b)),
            "rho_max": float(np.max(rho_b)),
            "band_counts": band_count_b,
            "n_band_a_or_b": n_band_a_or_b,
            "best_rho_landing": best_b,
            "surrogate_best_landing": surr_best_b,
            "decision": decision_b,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR / 'summary.json'}", flush=True)

    # ---------------- Print summary ------------------
    print()
    print("=== COHORT A: s011 in-basin landings ===")
    print(
        f"  n={n_a}  ρ-min={np.min(rho_a):.3f}  ρ-median={np.median(rho_a):.3f}  "
        f"ρ-max={np.max(rho_a):.3f}"
    )
    print(
        f"  Bands: A={band_count_a['A']}  B={band_count_a['B']}  "
        f"C={band_count_a['C']}  D={band_count_a['D']}"
    )
    print()
    for s, ps in sorted(per_seed_a_summary.items()):
        print(
            f"  seed {s:3d}: n={ps['n']:2d}  ρ-max={ps['rho_max']:.3f}  "
            f"A={ps['band_A']} B={ps['band_B']} C={ps['band_C']} D={ps['band_D']}"
        )
    print()
    print("=== COHORT B: seed-10 N=256 ===")
    print(
        f"  n={n_b}  ρ-min={np.min(rho_b):.3f}  ρ-median={np.median(rho_b):.3f}  "
        f"ρ-max={np.max(rho_b):.3f}"
    )
    print(
        f"  Bands: A={band_count_b['A']}  B={band_count_b['B']}  "
        f"C={band_count_b['C']}  D={band_count_b['D']}"
    )
    print(f"  n with ρ ≤ 4 (Band A∪B): {n_band_a_or_b}")
    print(f"  best-ρ landing:         IC#{best_b['ic_idx']:3d}  ρ={best_b['rho']:.3f}  "
          f"band={best_b['band']}  surr_mse={best_b['surrogate_mse']:.4f}  "
          f"q0_err={best_b['q0_err_deg']:.1f}°  ωd={best_b['omega_dir_err_deg']:.1f}°  "
          f"ωm={best_b['omega_mag_err_pct']:+.1f}%")
    print(f"  surrogate-best landing: IC#{surr_best_b['ic_idx']:3d}  ρ={surr_best_b['rho']:.3f}  "
          f"band={surr_best_b['band']}  surr_mse={surr_best_b['surrogate_mse']:.4f}  "
          f"q0_err={surr_best_b['q0_err_deg']:.1f}°")
    print(f"  >>> DECISION: {decision_b} <<<")
    print()

    # ---------------- Plot ------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rho, name, n in [
        (axes[0], rho_a, "Cohort A: in-basin", n_a),
        (axes[1], rho_b, "Cohort B: seed-10 N=256", n_b),
    ]:
        clipped_rho = np.clip(rho, 1e-3, None)
        ax.hist(clipped_rho, bins=np.logspace(-3, 3, 60), color="C0", alpha=0.75,
                edgecolor="k")
        ax.axvline(RHO_THRESHOLDS["A"], color="green", linestyle="--",
                   label="ρ=2 (A↔B)")
        ax.axvline(RHO_THRESHOLDS["B"], color="orange", linestyle="--",
                   label="ρ=4 (B↔C)")
        ax.axvline(RHO_THRESHOLDS["C"], color="red", linestyle="--",
                   label="ρ=8 (C↔D)")
        ax.set_xscale("log")
        ax.set_xlabel("ρ = √(hi-fi MSE) / 0.05")
        ax.set_ylabel("count")
        ax.set_title(f"{name}  (n={n})")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plot_path = OUT_DIR / "rho_distribution.png"
    fig.savefig(str(plot_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
