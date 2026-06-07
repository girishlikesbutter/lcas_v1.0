"""s035: Hi-fi confirmation of s034 polished candidates on seed 89.

Renders the 24 surrogate-Band-A polished candidates from s034 through the
full hi-fi forward model (shadow + BRDF), computes ρ vs truth, and reports
per-candidate hi-fi band classification.

Expected: surrogate ρ≈0.37-0.39 → hi-fi ρ≈1-2 (still A∪B per s017 ratio).
If confirmed, framework is end-to-end validated on seed 89.

Usage:
    cd notebooks/inversion/survey
    python experiments/s035_hifi_confirm_seed89.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "false"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

SURVEY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURVEY_DIR.parent.parent.parent))

from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

SEED = 89
N_WORKERS = 8
RESULTS_DIR = SURVEY_DIR / "results" / "s035_hifi_confirm_seed089"
POLISH_JSON = SURVEY_DIR / "results" / "s034_lm_polish_seed089" / "polish_summary.json"

RHO_SURROGATE_THRESHOLD = 2.0  # Band A cutoff


def _render_one(args):
    """Worker: render a single (q0, ω) candidate and compute hi-fi ρ."""
    idx, q0_wxyz, omega_rad, ctx_seed = args
    ctx = _WORKER_CTX
    mag_pred = render_hifi(np.array(q0_wxyz), np.array(omega_rad), ctx)
    rho = rho_from_hifi(mag_pred, ctx["mag_hifi_truth"])
    return idx, rho, mag_pred


_WORKER_CTX = None


def _init_worker(ctx):
    global _WORKER_CTX
    _WORKER_CTX = ctx


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load s034 polished candidates
    with open(POLISH_JSON) as f:
        summary = json.load(f)
    polished = summary["polished"]

    # Select surrogate Band A (ρ_pred_final < 2.0)
    band_a = [c for c in polished if c["rho_pred_final"] < RHO_SURROGATE_THRESHOLD]
    band_a.sort(key=lambda x: x["rho_pred_final"])
    print(f"s035: {len(band_a)} surrogate-Band-A candidates from s034 (seed {SEED})")
    print(f"      surrogate ρ range: [{band_a[0]['rho_pred_final']:.3f}, {band_a[-1]['rho_pred_final']:.3f}]")

    # Build render context (heavy — STL + BRDF + SPICE cached geometry)
    print(f"Building hi-fi context for seed {SEED} ...")
    t0 = time.time()
    ctx = build_context(SEED)
    print(f"  context built in {time.time()-t0:.1f}s")

    # Prepare work items
    work = []
    for i, c in enumerate(band_a):
        work.append((i, c["q0_final"], c["omega_final"], SEED))

    # Render in Pool(8) with fork context (workers inherit ctx)
    print(f"Rendering {len(work)} candidates with Pool({N_WORKERS}) fork-context ...")
    t0 = time.time()
    fork_ctx = mp.get_context("fork")
    with fork_ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(ctx,)) as pool:
        results_raw = pool.map(_render_one, work)
    wall = time.time() - t0
    print(f"  done in {wall:.1f}s ({wall/len(work):.1f}s/candidate)")

    # Collate results
    results_raw.sort(key=lambda x: x[0])
    hifi_rhos = np.array([r[1] for r in results_raw])
    hifi_mags = np.array([r[2] for r in results_raw])

    # Report
    print(f"\n{'='*70}")
    print(f"  Hi-fi confirmation: seed {SEED}, {len(band_a)} candidates")
    print(f"{'='*70}")
    print(f"  {'Rank':>4} {'Surr ρ':>7} {'Hi-fi ρ':>8} {'Band':>4} {'q0→truth°':>10} {'ω-dir°':>7} {'ω-mag%':>7}")
    print(f"  {'-'*4} {'-'*7} {'-'*8} {'-'*4} {'-'*10} {'-'*7} {'-'*7}")

    n_a, n_b, n_c, n_d = 0, 0, 0, 0
    out_records = []
    for i, c in enumerate(band_a):
        rho_hifi = hifi_rhos[i]
        band = rho_band(rho_hifi)
        if band == "A": n_a += 1
        elif band == "B": n_b += 1
        elif band == "C": n_c += 1
        else: n_d += 1

        print(f"  {c['rank']:>4} {c['rho_pred_final']:>7.3f} {rho_hifi:>8.3f} {band:>4}"
              f" {c['q0_to_truth_deg']:>10.2f} {c['omega_dir_to_truth_deg']:>7.2f}"
              f" {c['omega_mag_pct']:>7.3f}")

        out_records.append({
            "rank": c["rank"],
            "global_idx": c["global_idx"],
            "rho_surrogate": c["rho_pred_final"],
            "rho_hifi": float(rho_hifi),
            "band_hifi": band,
            "q0_to_truth_deg": c["q0_to_truth_deg"],
            "q0_to_twin_deg": c["q0_to_twin_deg"],
            "omega_dir_to_truth_deg": c["omega_dir_to_truth_deg"],
            "omega_mag_pct": c["omega_mag_pct"],
            "q0_wxyz": c["q0_final"],
            "omega_rad": c["omega_final"],
        })

    print(f"\n  Band summary: A={n_a}, B={n_b}, C={n_c}, D={n_d}")
    print(f"  Hi-fi ρ: min={hifi_rhos.min():.3f}, median={np.median(hifi_rhos):.3f}, max={hifi_rhos.max():.3f}")
    print(f"  A∪B yield: {n_a + n_b}/{len(band_a)}")

    # Save
    out = {
        "seed": SEED,
        "n_candidates": len(band_a),
        "wall_s": wall,
        "n_band_a": n_a,
        "n_band_b": n_b,
        "n_band_c": n_c,
        "n_band_d": n_d,
        "rho_hifi_min": float(hifi_rhos.min()),
        "rho_hifi_median": float(np.median(hifi_rhos)),
        "rho_hifi_max": float(hifi_rhos.max()),
        "candidates": out_records,
    }
    json_path = RESULTS_DIR / "hifi_confirm.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    npz_path = RESULTS_DIR / "hifi_mags.npz"
    np.savez_compressed(npz_path, hifi_mags=hifi_mags, hifi_rhos=hifi_rhos,
                        truth_mag=ctx["mag_hifi_truth"])
    print(f"Saved: {npz_path}")


if __name__ == "__main__":
    main()
