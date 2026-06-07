"""s059k_full_lc_rescue — re-polish s059j local-window-polished states on full-LC.

Phase 1 of s059k surfaced that local-window polish on seed 28 converges to
PHANTOM local minima (ρ_local=1.98 passes gate but hi-fi ρ=70 Band D). The
local-window cost surface has multiple basins; the polish lands in the
NEAREST one which may not be truth's basin.

This script tests whether re-polishing on FULL-LC residual rescues states
stuck in a phantom local-window minimum. If full-LC has a wider effective
basin (or the truth basin extends further in full-LC than local-window),
the re-polish should move phantom-basin states toward truth or a genuine
multi-solution alternate.

Usage:
    python experiments/s059k_full_lc_rescue.py --in-summary results/s059k_nd800_seed89/seed089/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s058_lm_polish_clusters import lm_polish  # noqa: E402
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402

_CTX_GLOBAL = None
_TARGET_GLOBAL = None


def _worker_init(ctx_pkl, target_pkl):
    """Set BLAS=1 and warm caches in worker."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _CTX_GLOBAL, _TARGET_GLOBAL
    _CTX_GLOBAL = ctx_pkl
    _TARGET_GLOBAL = target_pkl


def _polish_worker(args):
    idx, q0_seed, om0_seed, label = args
    t0 = time.time()
    res = lm_polish(np.asarray(q0_seed), np.asarray(om0_seed),
                    _CTX_GLOBAL, _TARGET_GLOBAL, label=label)
    res["polish_wall_s"] = time.time() - t0
    res["idx"] = idx
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="s059j-style results dir, e.g. results/s059k_nd800_seed89/seed089")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "rescue_full_lc")
    out_dir.mkdir(parents=True, exist_ok=True)

    pol = np.load(in_dir / "polished_states.npz")
    q0_pol = pol["q0_pol_wxyz"]   # (N, 4)
    om0_pol = pol["om0_pol_rad"]  # (N, 3)
    cluster_id = pol["cluster_id"]
    cluster_rank = pol["cluster_rank"]
    rho_local_pol = pol["surrogate_rho_local_polished"]
    rho_hifi_pol = pol["rho_polished_hifi"]
    band_pol = pol["band"]

    n = len(q0_pol)
    print(f"=== s059k_full_lc_rescue — seed {args.seed}, {n} polished states ===\n")

    print(f"building hifi context for seed {args.seed}...")
    t0 = time.time()
    ctx = build_context(seed=args.seed)
    target = ctx["mag_hifi_truth"]
    print(f"  built in {time.time()-t0:.1f}s; target len={len(target)}")

    print(f"\nre-polishing each state on FULL-LC residual...")
    args_list = [(i, q0_pol[i].tolist(), om0_pol[i].tolist(),
                  f"rank{int(cluster_rank[i])}_id{int(cluster_id[i])}")
                 for i in range(n)]

    ctx_pool = get_context("fork")
    rescued = []
    t_start = time.time()
    if args.n_workers <= 1:
        # single-threaded fallback for debug
        _worker_init(ctx, target)
        for a in args_list:
            r = _polish_worker(a)
            r_print(r, rho_local_pol, rho_hifi_pol, band_pol, cluster_rank, cluster_id)
            rescued.append(r)
    else:
        with ctx_pool.Pool(args.n_workers,
                           initializer=_worker_init,
                           initargs=(ctx, target)) as pool:
            for r in pool.imap_unordered(_polish_worker, args_list):
                r_print(r, rho_local_pol, rho_hifi_pol, band_pol,
                        cluster_rank, cluster_id)
                rescued.append(r)

    rescued_sorted = sorted(rescued, key=lambda r: r["idx"])

    # Hi-fi gate any with ρ_polished < 4
    print(f"\nhi-fi gating + classification (parent process)...")
    for r in rescued_sorted:
        if r["surrogate_rho_polished"] < 4.0:
            t0 = time.time()
            try:
                pred = render_hifi(np.asarray(r["q0_pol_wxyz"]),
                                   np.asarray(r["om0_pol_rad"]), ctx)
                rho_h = float(rho_from_hifi(pred, target))
                band = rho_band(rho_h)
            except Exception as e:
                rho_h, band = float("nan"), "ERR"
            r["rho_polished_hifi"] = rho_h
            r["band_polished_hifi"] = band
            r["hifi_render_s"] = time.time() - t0
        else:
            r["rho_polished_hifi"] = float("nan")
            r["band_polished_hifi"] = "GATED"
            r["hifi_render_s"] = 0.0

    # Compute errors vs truth at t=0
    om_truth = ctx["omega0_truth_rad"]
    om_truth_mag = float(np.linalg.norm(om_truth))
    for r in rescued_sorted:
        q0p = np.asarray(r["q0_pol_wxyz"])
        om0p = np.asarray(r["om0_pol_rad"])
        r["q0_err_deg"] = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(q0p, ctx["q0_truth"]))))))
        r["om_mag_err_pct"] = float((np.linalg.norm(om0p) - om_truth_mag) / om_truth_mag * 100)
        r["om_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)),
                       om_truth / om_truth_mag)), 0, 1))))

    # Tally
    band_counts_before = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for b in band_pol: band_counts_before[str(b)] = band_counts_before.get(str(b), 0) + 1
    band_counts_after = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for r in rescued_sorted: band_counts_after[r["band_polished_hifi"]] = band_counts_after.get(r["band_polished_hifi"], 0) + 1

    n_AB_before = band_counts_before.get("A", 0) + band_counts_before.get("B", 0)
    n_AB_after = band_counts_after.get("A", 0) + band_counts_after.get("B", 0)

    print(f"\n{'='*60}")
    print(f"BANDS BEFORE (s059j local-window polish): {band_counts_before}")
    print(f"BANDS AFTER  (full-LC re-polish):          {band_counts_after}")
    print(f"Headline yield: A∪B before={n_AB_before}, after={n_AB_after}")
    print(f"{'='*60}")

    # Save
    summary = {
        "experiment": "s059k_full_lc_rescue",
        "seed": int(args.seed),
        "n_states_rescued": n,
        "wall_total_s": time.time() - t_start,
        "band_counts_before": band_counts_before,
        "band_counts_after": band_counts_after,
        "n_band_AB_before": n_AB_before,
        "n_band_AB_after": n_AB_after,
        "rescued": [
            {
                "idx": int(r["idx"]),
                "cluster_rank": int(cluster_rank[r["idx"]]),
                "cluster_id": int(cluster_id[r["idx"]]),
                "rho_local_polished_s059j": float(rho_local_pol[r["idx"]]),
                "rho_hifi_polished_s059j": float(rho_hifi_pol[r["idx"]]) if not np.isnan(rho_hifi_pol[r["idx"]]) else None,
                "band_s059j": str(band_pol[r["idx"]]),
                "rho_seed_full_lc": float(r["surrogate_rho_seed"]),
                "rho_polished_full_lc": float(r["surrogate_rho_polished"]),
                "rho_hifi_polished_full_lc": float(r["rho_polished_hifi"]) if not np.isnan(r["rho_polished_hifi"]) else None,
                "band_full_lc": r["band_polished_hifi"],
                "q0_err_deg": r["q0_err_deg"],
                "om_mag_err_pct": r["om_mag_err_pct"],
                "om_dir_err_deg": r["om_dir_err_deg"],
                "q0_pol_wxyz_full_lc": r["q0_pol_wxyz"],
                "om0_pol_rad_full_lc": r["om0_pol_rad"],
                "polish_wall_s": float(r["polish_wall_s"]),
            }
            for r in rescued_sorted
        ],
    }
    out_path = out_dir / "rescue_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


def r_print(r, rho_local_pol, rho_hifi_pol, band_pol, cluster_rank, cluster_id):
    i = r["idx"]
    print(f"  idx={i:3d} (rank={int(cluster_rank[i]):3d} id={int(cluster_id[i]):3d})  "
          f"s059j ρ_local={rho_local_pol[i]:.3f} ρ_hifi={rho_hifi_pol[i]:.3f} band={str(band_pol[i]):<5}  →  "
          f"full-LC ρ_seed={r['surrogate_rho_seed']:.3f} → ρ_pol={r['surrogate_rho_polished']:.3f}")


if __name__ == "__main__":
    main()
