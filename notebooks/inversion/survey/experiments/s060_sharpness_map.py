"""s060_sharpness_map — measure |C_t|(t) across the full LC for a seed.

Diagnostic for the multi-anchor architecture spitball: how many sharp anchors
exist, and where in the LC do they live? Single seed, single pool, full LC.

For each epoch t in 0..N-1:
    project pool to body frame at t,
    survive @ epoch t (|surrogate_pred - measured_mag| < TOL_MAG),
    record |C_t|.

Then identify the K sharpest epochs (smallest |C_t|), report alongside their
LC magnitude, and plot |C_t|(t) vs mag(t).

Usage:
    python experiments/s060_sharpness_map.py --seed 28
    python experiments/s060_sharpness_map.py --seed 28 --n-pool 100000 --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.c_t_pipeline import (  # noqa: E402
    sample_so3_pool, compute_j2000_units, project_directions, survive_at_epoch,
)
from lib.hifi_render import build_context  # noqa: E402
from lib.surrogate_eval import get_model  # noqa: E402

from experiments.s059_pilot import TOL_MAG, SP_DEG, AD_DEG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-pool", type=int, default=50_000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rng-seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        SURVEY / "results" / "s060_sharpness_map" / f"seed{args.seed:03d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== s060_sharpness_map — seed {args.seed} ===\n")

    # Load context
    print("loading context...")
    t0 = time.time()
    ctx = build_context(seed=args.seed)
    n_ep = int(ctx["observation_times"].shape[0])
    mag_truth = np.asarray(ctx["mag_hifi_truth"])
    obs_dist = np.asarray(ctx["obs_dist"])
    sun_unit, obs_unit = compute_j2000_units(ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"])
    print(f"  loaded in {time.time()-t0:.1f}s; n_epochs={n_ep}")

    # Build pool
    print(f"building Sobol pool N={args.n_pool}...")
    t0 = time.time()
    pool = sample_so3_pool(args.n_pool, sample_seed=args.rng_seed)
    R_cache = pool["R_cache"]
    print(f"  built in {time.time()-t0:.1f}s")

    # Warm surrogate
    model = get_model()

    # Sweep epochs
    print(f"sweeping {n_ep} epochs (BLAS=1, single-thread)...")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

    t0 = time.time()
    Ct = np.zeros(n_ep, dtype=np.int64)
    for t in range(n_ep):
        k1_b, k2_b = project_directions(R_cache, sun_unit[t], obs_unit[t])
        _, keep = survive_at_epoch(
            model, k1_b, k2_b, float(obs_dist[t]),
            SP_DEG, AD_DEG, float(mag_truth[t]), TOL_MAG,
        )
        Ct[t] = int(keep.sum())
        if (t + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (t + 1) / elapsed
            eta = (n_ep - t - 1) / rate
            print(f"  epoch {t+1}/{n_ep}  |C_t|={Ct[t]:6d}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    wall = time.time() - t0
    print(f"  swept in {wall:.1f}s")

    # Identify K sharpest
    K = int(args.top_k)
    sharp_idx = np.argsort(Ct)[:K]
    sharp_idx_sorted = np.sort(sharp_idx)

    # Stats
    Ct_min = int(Ct.min())
    Ct_med = float(np.median(Ct))
    Ct_max = int(Ct.max())
    n_below_1pct = int((Ct < 0.01 * args.n_pool).sum())  # |C_t| < 1% of pool
    n_below_01pct = int((Ct < 0.001 * args.n_pool).sum())
    n_below_500 = int((Ct < 500).sum())
    n_below_100 = int((Ct < 100).sum())

    print(f"\n=== |C_t| distribution (seed {args.seed}, pool N={args.n_pool}) ===")
    print(f"  min       = {Ct_min}")
    print(f"  median    = {Ct_med:.0f}")
    print(f"  max       = {Ct_max}")
    print(f"  <100      = {n_below_100} epochs")
    print(f"  <500      = {n_below_500} epochs")
    print(f"  <1% pool  = {n_below_1pct} epochs (={int(0.01*args.n_pool)})")
    print(f"  <0.1% pool= {n_below_01pct} epochs (={int(0.001*args.n_pool)})")

    print(f"\n=== top-{K} sharpest anchors (chronological) ===")
    print(f"  {'idx':>4}  {'t':>5}  {'|C_t|':>6}  {'mag':>7}  {'mag_pct':>8}")
    mag_pcts = (mag_truth - mag_truth.min()) / (mag_truth.max() - mag_truth.min()) * 100
    for rank, t in enumerate(sharp_idx_sorted, 1):
        print(f"  {rank:>4}  {t:>5d}  {Ct[t]:>6d}  {mag_truth[t]:>7.2f}  {mag_pcts[t]:>7.1f}%")

    # Save NPZ
    np.savez(
        out_dir / "sharpness_map.npz",
        seed=np.int64(args.seed),
        n_pool=np.int64(args.n_pool),
        rng_seed=np.int64(args.rng_seed),
        tol_mag=np.float64(TOL_MAG),
        sp_deg=np.float64(SP_DEG),
        ad_deg=np.float64(AD_DEG),
        Ct=Ct,
        mag_truth=mag_truth,
        observation_times=ctx["observation_times"],
        sharp_idx_sorted=sharp_idx_sorted.astype(np.int64),
        sharp_Ct=Ct[sharp_idx_sorted],
        sharp_mag=mag_truth[sharp_idx_sorted],
        wall_s=np.float64(wall),
    )
    print(f"\nSaved: {out_dir/'sharpness_map.npz'}")

    # Save JSON summary
    summary = {
        "seed": int(args.seed),
        "n_pool": int(args.n_pool),
        "n_epochs": int(n_ep),
        "tol_mag": float(TOL_MAG),
        "Ct_min": Ct_min,
        "Ct_median": Ct_med,
        "Ct_max": Ct_max,
        "n_below_100": n_below_100,
        "n_below_500": n_below_500,
        "n_below_1pct_pool": n_below_1pct,
        "n_below_01pct_pool": n_below_01pct,
        "top_k": K,
        "sharp_anchors": [
            {
                "rank_in_chrono": i,
                "t": int(t),
                "Ct": int(Ct[t]),
                "mag": float(mag_truth[t]),
                "mag_normalized_pct": float(mag_pcts[t]),
            }
            for i, t in enumerate(sharp_idx_sorted, 1)
        ],
        "wall_s": float(wall),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_dir/'summary.json'}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t_arr = np.arange(n_ep)

    ax_lc = axes[0]
    ax_lc.plot(t_arr, mag_truth, "k-", lw=0.7, label="mag truth")
    ax_lc.scatter(sharp_idx_sorted, mag_truth[sharp_idx_sorted],
                  c="red", s=40, zorder=5, label=f"top-{K} sharpest |C_t|")
    ax_lc.invert_yaxis()
    ax_lc.set_ylabel("magnitude")
    ax_lc.legend(loc="lower right")
    ax_lc.set_title(f"seed {args.seed} — sharp anchors and where they sit in the LC")
    ax_lc.grid(True, alpha=0.3)

    ax_ct = axes[1]
    ax_ct.semilogy(t_arr, np.maximum(Ct, 1), "b-", lw=0.7, label="|C_t|")
    ax_ct.scatter(sharp_idx_sorted, np.maximum(Ct[sharp_idx_sorted], 1),
                  c="red", s=40, zorder=5)
    ax_ct.axhline(0.01 * args.n_pool, color="orange", ls="--", lw=1,
                  label=f"1% pool ({int(0.01*args.n_pool)})")
    ax_ct.axhline(0.001 * args.n_pool, color="green", ls="--", lw=1,
                  label=f"0.1% pool ({int(0.001*args.n_pool)})")
    ax_ct.set_xlabel("epoch index")
    ax_ct.set_ylabel("|C_t|  (log)")
    ax_ct.legend(loc="upper right")
    ax_ct.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    png_path = out_dir / "sharpness_map.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
