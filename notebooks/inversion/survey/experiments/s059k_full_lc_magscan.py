"""s059k_full_lc_magscan — full-LC polish with PER-CLUSTER fine |ω|-mag scan.

Backup to s059k_full_lc_from_seeds.py (which uses fixed multi-mag-start
offsets). For each cluster best-member at (q_a, ω_dir, ω_mag_grid),
SCAN |ω|-mag at fine resolution (default 21 cells from -10% to +10%
relative to grid mag), pick the smallest-MSE cell on local-window cost,
polish from THAT seed via full-LC LM. One polish per cluster instead of
N (multi-start), so 5× fewer polishes.

Use only if multi-mag-start architecture (s059k_full_lc_from_seeds.py)
underperforms.

Usage:
    python experiments/s059k_full_lc_magscan.py \
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
from experiments.s059e_local_window import propagate_local_window  # noqa: E402
from experiments.s058_lm_polish_clusters import lm_polish  # noqa: E402
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402
from lib.surrogate_eval import predict as surrogate_predict  # noqa: E402

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


def mag_scan_local(q_a_wxyz, om_dir, om_mag_grid, T_A, W, ctx, target,
                   pct_lo, pct_hi, n_scan):
    """Scan |ω|-mag at the given (q_a, ω_dir); return best-MSE mag and seed."""
    factors = np.linspace(1.0 + pct_lo / 100, 1.0 + pct_hi / 100, n_scan)
    best_mse = np.inf
    best_om_a = None
    best_factor = None
    for f in factors:
        om_a = om_dir * (om_mag_grid * f)
        try:
            k1, k2, lo, hi = propagate_local_window(q_a_wxyz, om_a, T_A, W, ctx)
            pred = surrogate_predict(k1, k2, ctx["obs_dist"][lo:hi])
            r = pred - target[lo:hi]
            r = np.where(np.isfinite(r), r, 5.0)
            r = np.clip(r, -5.0, 5.0)
            mse = float(np.mean(r ** 2))
        except Exception:
            mse = float(np.inf)
        if mse < best_mse:
            best_mse = mse
            best_om_a = om_a
            best_factor = float(f)
    return best_mse, best_om_a, best_factor


def _polish_worker(args):
    (cluster_rank, cluster_id, q_a_seed, om_a_seed_grid,
     T_A, W, t_a_seconds, inertia_tensor, pct_lo, pct_hi, n_scan) = args
    t0 = time.time()

    om_mag_grid = float(np.linalg.norm(om_a_seed_grid))
    om_dir = np.asarray(om_a_seed_grid) / max(1e-12, om_mag_grid)
    best_mse, best_om_a, best_factor = mag_scan_local(
        np.asarray(q_a_seed), om_dir, om_mag_grid, T_A, W, _CTX, _TARGET,
        pct_lo, pct_hi, n_scan)
    scan_wall = time.time() - t0

    q0_seed, om0_seed = back_propagate(
        np.asarray(q_a_seed), best_om_a, t_a_seconds, np.asarray(inertia_tensor))
    res = lm_polish(q0_seed, om0_seed, _CTX, _TARGET,
                    label=f"rank{cluster_rank}_id{cluster_id}_factor{best_factor:.4f}")
    res["polish_wall_s"] = time.time() - t0 - scan_wall
    res["scan_wall_s"] = scan_wall
    res["cluster_rank"] = int(cluster_rank)
    res["cluster_id"] = int(cluster_id)
    res["mag_scan_best_factor"] = best_factor
    res["mag_scan_best_mse_local"] = float(best_mse)
    res["q_a_seed_wxyz"] = np.asarray(q_a_seed).tolist()
    res["om_a_seed_rad"] = best_om_a.tolist()
    res["q0_seed_wxyz"] = np.asarray(q0_seed).tolist()
    res["om0_seed_rad"] = np.asarray(om0_seed).tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--top-k-polish", type=int, default=50)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--scan-pct-lo", type=float, default=-10.0,
                    help="lower bound (pct) for fine mag scan")
    ap.add_argument("--scan-pct-hi", type=float, default=10.0)
    ap.add_argument("--scan-n", type=int, default=21)
    ap.add_argument("--w", type=int, default=10, help="local-window radius for mag scan")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "full_lc_magscan")
    out_dir.mkdir(parents=True, exist_ok=True)

    cl_npz = np.load(in_dir / "clusters.npz")
    sg_npz = np.load(in_dir / "score_grid.npz")
    Q_top = cl_npz["Q_top"]; Om_top = cl_npz["Om_top"]; mse_top = cl_npz["mse_top"]
    truth_idx = int(cl_npz["truth_idx_diagnostic"])
    q_a_truth = sg_npz["q_a_truth"]; om_a_truth = sg_npz["om_a_truth"]
    T_A = int(sg_npz["T_A"])

    qa_dist_diag = quat_ang_deg_batch(Q_top, q_a_truth)
    om_dist_diag = ang_to_axis(Om_top, om_a_truth)
    fp = {"Q_A_pass": Q_top, "om_pass": Om_top, "scores": -mse_top,
          "qa_dist_to_truth": qa_dist_diag, "om_dist_to_truth": om_dist_diag,
          "truth_idx": truth_idx}
    cl = stage_cluster(fp, lambda m: print(m))

    print(f"\nbuilding hifi context for seed {args.seed}...")
    ctx = build_context(seed=args.seed)
    target = ctx["mag_hifi_truth"]
    inertia_tensor = ctx["inertia_tensor"]
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])
    print(f"t_a_seconds={t_a_seconds:.1f}")

    n_polish = min(args.top_k_polish, len(cl["clusters_sorted"]))
    polish_list = []
    for rank0, c in enumerate(cl["clusters_sorted"][:n_polish]):
        bm = c["best_member_idx"]
        polish_list.append((rank0 + 1, c["cluster_id"], Q_top[bm], Om_top[bm],
                            T_A, args.w, t_a_seconds, inertia_tensor,
                            args.scan_pct_lo, args.scan_pct_hi, args.scan_n))
    print(f"\npolishing {n_polish} cluster reps (mag-scan {args.scan_pct_lo:.1f}% to "
          f"{args.scan_pct_hi:.1f}%, n={args.scan_n}) → full-LC LM")

    t_start = time.time()
    rescued = []
    if args.n_workers <= 1:
        _worker_init(ctx, target)
        for a in polish_list:
            r = _polish_worker(a)
            print(f"  rank={r['cluster_rank']:3d} id={r['cluster_id']:3d} mag×{r['mag_scan_best_factor']:.4f}  "
                  f"ρ_seed={r['surrogate_rho_seed']:6.2f} → ρ_pol={r['surrogate_rho_polished']:6.3f}")
            rescued.append(r)
    else:
        ctx_pool = get_context("fork")
        with ctx_pool.Pool(args.n_workers, initializer=_worker_init,
                           initargs=(ctx, target)) as pool:
            for r in pool.imap_unordered(_polish_worker, polish_list):
                print(f"  rank={r['cluster_rank']:3d} id={r['cluster_id']:3d} mag×{r['mag_scan_best_factor']:.4f}  "
                      f"ρ_seed={r['surrogate_rho_seed']:6.2f} → ρ_pol={r['surrogate_rho_polished']:6.3f}")
                rescued.append(r)
    rescued.sort(key=lambda r: r["cluster_rank"])
    polish_wall = time.time() - t_start

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
            r["rho_polished_hifi"] = rho_h; r["band_polished_hifi"] = band
        else:
            r["rho_polished_hifi"] = float("nan"); r["band_polished_hifi"] = "GATED"
        band_counts[r["band_polished_hifi"]] = band_counts.get(r["band_polished_hifi"], 0) + 1
        q0p = np.asarray(r["q0_pol_wxyz"]); om0p = np.asarray(r["om0_pol_rad"])
        r["q0_err_deg"] = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(q0p, ctx["q0_truth"]))))))
        r["om_mag_err_pct"] = float((np.linalg.norm(om0p) - om_truth_t0_mag) / om_truth_t0_mag * 100)
        r["om_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)),
                       om_truth_t0 / om_truth_t0_mag)), 0, 1))))

    n_AB = band_counts.get("A", 0) + band_counts.get("B", 0)

    print(f"\nResults (sorted by hi-fi ρ):")
    rescued_byhifi = sorted(rescued, key=lambda r: (np.inf if np.isnan(r["rho_polished_hifi"]) else r["rho_polished_hifi"]))
    for r in rescued_byhifi[:30]:
        print(f"  rank={r['cluster_rank']:3d} id={r['cluster_id']:3d} mag×{r['mag_scan_best_factor']:.4f}  "
              f"ρ_pol={r['surrogate_rho_polished']:6.3f} hi-fi={r['rho_polished_hifi']:6.3f} "
              f"band={r['band_polished_hifi']:<5}  q0_err={r['q0_err_deg']:5.2f}° "
              f"|ω|err={r['om_mag_err_pct']:+6.2f}% ω_dir={r['om_dir_err_deg']:5.2f}°")

    print(f"\n{'='*60}\nBAND COUNTS: {band_counts}\nHEADLINE YIELD (Band A∪B): {n_AB}/{n_polish}\n{'='*60}")
    summary = {
        "experiment": "s059k_full_lc_magscan", "seed": int(args.seed), "T_A": T_A,
        "scan_pct_lo": args.scan_pct_lo, "scan_pct_hi": args.scan_pct_hi, "scan_n": args.scan_n,
        "top_k_polish": int(n_polish), "n_clusters": int(len(cl["clusters_sorted"])),
        "truth_cluster_rank": int(cl["truth_cluster_rank"]),
        "band_counts": band_counts, "n_band_AB": int(n_AB), "headline_yield": int(n_AB),
        "wall_polish_s": polish_wall,
        "polished": [{"cluster_rank": int(r["cluster_rank"]), "cluster_id": int(r["cluster_id"]),
                      "mag_scan_best_factor": float(r["mag_scan_best_factor"]),
                      "rho_seed_full_lc": float(r["surrogate_rho_seed"]),
                      "rho_polished_full_lc": float(r["surrogate_rho_polished"]),
                      "rho_polished_hifi": float(r["rho_polished_hifi"]) if not np.isnan(r["rho_polished_hifi"]) else None,
                      "band": r["band_polished_hifi"],
                      "q0_err_deg": float(r["q0_err_deg"]),
                      "om_mag_err_pct": float(r["om_mag_err_pct"]),
                      "om_dir_err_deg": float(r["om_dir_err_deg"]),
                      "q0_pol_wxyz": r["q0_pol_wxyz"], "om0_pol_rad": r["om0_pol_rad"],
                      "polish_wall_s": float(r["polish_wall_s"]),
                      "scan_wall_s": float(r["scan_wall_s"]),
                      "n_eval": int(r["n_eval"])}
                     for r in rescued]
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
