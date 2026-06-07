"""s059k_full_lc_from_seeds — full-LC LM polish from cluster best-member seeds.

PHASE 4 architectural fix. The smoke test (s059k_smoke_seed89) showed:
- Local-window polish has PHANTOM BASINS regardless of seed quality
  (ρ_local=0.009 looks "perfect" locally but hi-fi ρ=51).
- Full-LC polish from N=1600-grade noise (q_a=0.85°, ω=1.75°, |ω|+0%)
  lands ρ=0.18 Band A on seed 89.

Fix: same ω-grid + cluster architecture, but swap local-window polish →
full-LC polish. This script reads cached s059j-style score_grid.npz +
clusters.npz, re-runs the cluster step to recover best_member_idx per
cluster, polishes top-K cluster reps via FULL-LC LM, and hi-fi gates.

Usage:
    python experiments/s059k_full_lc_from_seeds.py \
        --in-dir results/s059k_nd800_seed89/seed089 \
        --seed 89 \
        --top-k-polish 50 \
        --n-workers 8
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

from experiments.s059_pilot import (  # noqa: E402
    stage_cluster, back_propagate, quat_ang_deg_batch, ang_to_axis,
)
from experiments.s058_lm_polish_clusters import lm_polish  # noqa: E402
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402

_CTX = None
_TARGET = None


def _worker_init(ctx, target):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _CTX, _TARGET
    _CTX = ctx
    _TARGET = target


def _polish_worker(args):
    cluster_rank, cluster_id, q_a_seed, om_a_seed, t_a_seconds, inertia_tensor, mag_pct_offset = args
    t0 = time.time()
    om_a_arr = np.asarray(om_a_seed)
    if mag_pct_offset != 0.0:
        om_a_arr = om_a_arr * (1.0 + mag_pct_offset / 100.0)
    q0_seed, om0_seed = back_propagate(np.asarray(q_a_seed), om_a_arr,
                                        t_a_seconds, np.asarray(inertia_tensor))
    res = lm_polish(q0_seed, om0_seed, _CTX, _TARGET,
                    label=f"rank{cluster_rank}_id{cluster_id}_mag{mag_pct_offset:+.0f}")
    res["polish_wall_s"] = time.time() - t0
    res["cluster_rank"] = int(cluster_rank)
    res["cluster_id"] = int(cluster_id)
    res["mag_pct_offset"] = float(mag_pct_offset)
    res["q_a_seed_wxyz"] = np.asarray(q_a_seed).tolist()
    res["om_a_seed_rad"] = om_a_arr.tolist()
    res["q0_seed_wxyz"] = np.asarray(q0_seed).tolist()
    res["om0_seed_rad"] = np.asarray(om0_seed).tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="s059j-style dir, e.g. results/s059k_nd800_seed89/seed089")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--top-k-polish", type=int, default=50)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--mag-starts", default="0.0",
                    help="Comma-separated list of |omega| pct offsets for multi-start polish. "
                         "Default '0.0' = single-start. Try '0,3,-3' for 3 starts per cluster, "
                         "'-9,-6,-3,0,3,6,9' for finer multi-start.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    mag_pct_offsets = [float(x) for x in args.mag_starts.split(",")]

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "full_lc_seeds")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== s059k_full_lc_from_seeds — seed {args.seed} ===\n")

    # Load cached
    cl_npz = np.load(in_dir / "clusters.npz")
    sg_npz = np.load(in_dir / "score_grid.npz")

    Q_top = cl_npz["Q_top"]               # (5000, 4)
    Om_top = cl_npz["Om_top"]             # (5000, 3)
    mse_top = cl_npz["mse_top"]           # (5000,)
    truth_idx = int(cl_npz["truth_idx_diagnostic"])

    q_a_truth = sg_npz["q_a_truth"]       # (4,)
    om_a_truth = sg_npz["om_a_truth"]     # (3,)
    T_A = int(sg_npz["T_A"])

    # Reconstruct fp for stage_cluster
    qa_dist_diag = quat_ang_deg_batch(Q_top, q_a_truth)
    om_dist_diag = ang_to_axis(Om_top, om_a_truth)

    fp = {
        "Q_A_pass": Q_top,
        "om_pass": Om_top,
        "scores": -mse_top,  # higher=better
        "qa_dist_to_truth": qa_dist_diag,
        "om_dist_to_truth": om_dist_diag,
        "truth_idx": truth_idx,
    }

    def log(msg):
        print(msg)

    print(f"re-running stage_cluster on cached top-K=5000...")
    cl = stage_cluster(fp, log)
    truth_cluster_rank = cl["truth_cluster_rank"]
    n_clusters = len(cl["clusters_sorted"])
    print(f"  truth cluster rank {truth_cluster_rank}/{n_clusters}")

    print(f"\nbuilding hifi context for seed {args.seed}...")
    t0 = time.time()
    ctx = build_context(seed=args.seed)
    target = ctx["mag_hifi_truth"]
    inertia_tensor = ctx["inertia_tensor"]
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])
    print(f"  built in {time.time()-t0:.1f}s; t_a_seconds={t_a_seconds:.1f}s")

    # Build polish list — top-K cluster reps × mag_pct_offsets multi-start. NO truth injection.
    n_polish = min(args.top_k_polish, n_clusters)
    polish_list = []
    for rank0, c in enumerate(cl["clusters_sorted"][:n_polish]):
        bm = c["best_member_idx"]
        q_a_seed = Q_top[bm]
        om_a_seed = Om_top[bm]
        for mag_off in mag_pct_offsets:
            polish_list.append((rank0 + 1, c["cluster_id"], q_a_seed, om_a_seed,
                                t_a_seconds, inertia_tensor, mag_off))
    print(f"\npolishing {n_polish} cluster reps × {len(mag_pct_offsets)} mag-starts = "
          f"{len(polish_list)} polishes via full-LC LM, NO truth injection")
    print(f"  mag_pct_offsets: {mag_pct_offsets}")

    t_polish = time.time()
    rescued = []
    if args.n_workers <= 1:
        _worker_init(ctx, target)
        for a in polish_list:
            r = _polish_worker(a)
            print_polish(r)
            rescued.append(r)
    else:
        ctx_pool = get_context("fork")
        with ctx_pool.Pool(args.n_workers,
                           initializer=_worker_init,
                           initargs=(ctx, target)) as pool:
            for r in pool.imap_unordered(_polish_worker, polish_list):
                print_polish(r)
                rescued.append(r)
    rescued.sort(key=lambda r: r["cluster_rank"])
    polish_wall = time.time() - t_polish
    print(f"\npolish wall: {polish_wall:.1f}s ({polish_wall/60:.2f} min)")

    # Hi-fi gate (parent process)
    print(f"\nhi-fi gating + classification...")
    t_hifi = time.time()
    om_truth_t0 = ctx["omega0_truth_rad"]
    om_truth_t0_mag = float(np.linalg.norm(om_truth_t0))
    band_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for r in rescued:
        if r["surrogate_rho_polished"] < 4.0:
            try:
                pred = render_hifi(np.asarray(r["q0_pol_wxyz"]),
                                   np.asarray(r["om0_pol_rad"]), ctx)
                rho_h = float(rho_from_hifi(pred, target))
                band = rho_band(rho_h)
            except Exception:
                rho_h, band = float("nan"), "ERR"
            r["rho_polished_hifi"] = rho_h
            r["band_polished_hifi"] = band
        else:
            r["rho_polished_hifi"] = float("nan")
            r["band_polished_hifi"] = "GATED"
        band_counts[r["band_polished_hifi"]] = band_counts.get(r["band_polished_hifi"], 0) + 1

        # errors vs truth at t=0
        q0p = np.asarray(r["q0_pol_wxyz"])
        om0p = np.asarray(r["om0_pol_rad"])
        r["q0_err_deg"] = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(q0p, ctx["q0_truth"]))))))
        r["om_mag_err_pct"] = float((np.linalg.norm(om0p) - om_truth_t0_mag) / om_truth_t0_mag * 100)
        r["om_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)),
                       om_truth_t0 / om_truth_t0_mag)), 0, 1))))
    print(f"hi-fi wall: {time.time()-t_hifi:.1f}s")

    n_AB_polishes = band_counts.get("A", 0) + band_counts.get("B", 0)

    # Per-cluster yield: how many unique cluster_ids landed at least one Band A∪B
    cluster_best = {}
    for r in rescued:
        cid = int(r["cluster_id"])
        if cid not in cluster_best or (
            not np.isnan(r["rho_polished_hifi"]) and
            (np.isnan(cluster_best[cid]["rho_polished_hifi"]) or
             r["rho_polished_hifi"] < cluster_best[cid]["rho_polished_hifi"])
        ):
            cluster_best[cid] = r
    n_AB_clusters = sum(1 for r in cluster_best.values()
                        if r["band_polished_hifi"] in {"A", "B"})

    # Print sorted by hi-fi ρ
    print(f"\nResults (sorted by hi-fi ρ, all polishes shown):")
    rescued_byhifi = sorted(
        rescued,
        key=lambda r: (np.inf if np.isnan(r["rho_polished_hifi"]) else r["rho_polished_hifi"]),
    )
    for r in rescued_byhifi[:30]:
        print(f"  rank={r['cluster_rank']:3d} id={r['cluster_id']:3d} mag{r['mag_pct_offset']:+5.1f}%  "
              f"ρ_seed={r['surrogate_rho_seed']:6.2f} → ρ_pol={r['surrogate_rho_polished']:6.3f} → "
              f"hi-fi={r['rho_polished_hifi']:6.3f} band={r['band_polished_hifi']:<5}  "
              f"q0_err={r['q0_err_deg']:5.2f}° |ω|err={r['om_mag_err_pct']:+6.2f}% ω_dir={r['om_dir_err_deg']:5.2f}°")

    print(f"\n{'='*60}")
    print(f"BAND COUNTS (all polishes): {band_counts}")
    print(f"  total polishes = {len(rescued)} ({n_polish} clusters × {len(mag_pct_offsets)} mag-starts)")
    print(f"  Band A∪B polishes: {n_AB_polishes}")
    print(f"  Band A∪B unique clusters (best polish per cluster): {n_AB_clusters}")
    print(f"HEADLINE YIELD (cluster A∪B): {n_AB_clusters}/{n_polish}")
    print(f"truth cluster rank: {truth_cluster_rank}/{n_clusters} (NOT polished if > {n_polish})")
    print(f"{'='*60}")

    summary = {
        "experiment": "s059k_full_lc_from_seeds",
        "seed": int(args.seed), "T_A": T_A,
        "top_k_polish": int(n_polish),
        "mag_pct_offsets": mag_pct_offsets,
        "n_clusters": int(n_clusters),
        "truth_cluster_rank": int(truth_cluster_rank),
        "band_counts_all_polishes": band_counts,
        "n_band_AB_polishes": int(n_AB_polishes),
        "n_band_AB_clusters": int(n_AB_clusters),
        "headline_yield": int(n_AB_clusters),
        "wall_polish_s": polish_wall,
        "polished": [
            {"cluster_rank": int(r["cluster_rank"]), "cluster_id": int(r["cluster_id"]),
             "mag_pct_offset": float(r["mag_pct_offset"]),
             "q_a_seed_wxyz": r["q_a_seed_wxyz"],
             "om_a_seed_rad": r["om_a_seed_rad"],
             "rho_seed_full_lc": float(r["surrogate_rho_seed"]),
             "rho_polished_full_lc": float(r["surrogate_rho_polished"]),
             "rho_polished_hifi": float(r["rho_polished_hifi"]) if not np.isnan(r["rho_polished_hifi"]) else None,
             "band": r["band_polished_hifi"],
             "q0_err_deg": float(r["q0_err_deg"]),
             "om_mag_err_pct": float(r["om_mag_err_pct"]),
             "om_dir_err_deg": float(r["om_dir_err_deg"]),
             "q0_pol_wxyz": r["q0_pol_wxyz"],
             "om0_pol_rad": r["om0_pol_rad"],
             "polish_wall_s": float(r["polish_wall_s"]),
             "n_eval": int(r["n_eval"])}
            for r in rescued
        ],
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


def print_polish(r):
    print(f"  rank={r['cluster_rank']:3d} id={r['cluster_id']:3d} mag{r['mag_pct_offset']:+5.1f}%  "
          f"ρ_seed={r['surrogate_rho_seed']:6.2f} → ρ_pol={r['surrogate_rho_polished']:6.3f}  "
          f"(n_eval={r['n_eval']:3d}, wall={r['polish_wall_s']:.1f}s)")


if __name__ == "__main__":
    main()
