"""s060_cohort_anchor_topology — per-seed sharpest bright + dim anchor topology.

For each cohort seed:
  - Build 25k Sobol pool.
  - Restrict to top-5% dimmest epochs (mag_pct ≥ 95%) and top-5% brightest
    epochs (mag_pct ≤ 5%). For each restricted set, measure |C_t|.
  - Pick the argmin-|C_t| anchor in each set: the sharpest dim and sharpest
    bright anchor.
  - At each anchor, run greedy cluster of survivors at 40° threshold.
  - Report: |C_t|, n_clusters, top-3 sizes, top-3 mass fraction, intra_max_deg,
    between_gap_deg.

Aggregates the per-seed table to test the user's claim:
  - Dim-extreme: 3-5 clusters with ≥85% mass in top 3 (clean enumeration).
  - Bright-extreme: 5-10 small clusters with low top-3 mass (passage validator).

Usage:
    python experiments/s060_cohort_anchor_topology.py --seed 6 --n-pool 25000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

from experiments.s060_anchor_topology import (  # noqa: E402
    greedy_cluster_quats, classify_topology,
)


def measure_anchor(model, pool, sun_unit, obs_unit, obs_dist, mag_truth,
                   t, threshold_deg=40.0):
    """Measure topology at one epoch. Returns dict with anchor row."""
    k1_b, k2_b = project_directions(pool["R_cache"], sun_unit[t], obs_unit[t])
    _, keep = survive_at_epoch(
        model, k1_b, k2_b, float(obs_dist[t]),
        SP_DEG, AD_DEG, float(mag_truth[t]), TOL_MAG,
    )
    survivors_q = pool["q_pool_wxyz"][keep]
    if len(survivors_q) < 2:
        return None  # too sparse to cluster
    clusters = greedy_cluster_quats(survivors_q, threshold_deg)
    topo = classify_topology(clusters, survivors_q)
    topo["t"] = int(t)
    topo["mag"] = float(mag_truth[t])
    return topo


def find_sharpest_in_region(model, pool, sun_unit, obs_unit, obs_dist,
                            mag_truth, region_idx):
    """Among the epochs in region_idx, find the one with smallest |C_t|."""
    best_t = -1
    best_Ct = 10**9
    for t in region_idx:
        k1_b, k2_b = project_directions(pool["R_cache"], sun_unit[t], obs_unit[t])
        _, keep = survive_at_epoch(
            model, k1_b, k2_b, float(obs_dist[t]),
            SP_DEG, AD_DEG, float(mag_truth[t]), TOL_MAG,
        )
        Ct = int(keep.sum())
        if Ct < best_Ct:
            best_Ct = Ct
            best_t = int(t)
    return best_t, best_Ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-pool", type=int, default=25_000)
    ap.add_argument("--cluster-threshold-deg", type=float, default=40.0)
    ap.add_argument("--region-pct", type=float, default=5.0,
                    help="Region width: top-X%% dim, bottom-X%% bright")
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== s060_cohort_anchor_topology — seed {args.seed} ===\n")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

    t0 = time.time()
    ctx = build_context(seed=args.seed)
    obs_dist = np.asarray(ctx["obs_dist"])
    sun_unit, obs_unit = compute_j2000_units(ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"])
    mag_truth = np.asarray(ctx["mag_hifi_truth"])
    omega0 = np.asarray(ctx["omega0_truth_rad"])
    omega_mag_dps = float(np.degrees(np.linalg.norm(omega0)))
    n_ep = len(mag_truth)

    pool = sample_so3_pool(args.n_pool, sample_seed=args.rng_seed)
    model = get_model()

    # Define dim and bright regions by magnitude percentile
    n_region = int(np.ceil(args.region_pct / 100 * n_ep))
    sorted_idx = np.argsort(mag_truth)  # ascending: brightest first
    bright_idx = sorted_idx[:n_region]            # smallest mag = brightest
    dim_idx = sorted_idx[-n_region:]              # largest mag = dimmest

    print(f"  |ω|={omega_mag_dps:.3f} dps, n_ep={n_ep}, "
          f"region={args.region_pct}% → {n_region} epochs each")
    print(f"  bright epochs span mag [{mag_truth[bright_idx].min():.2f}, "
          f"{mag_truth[bright_idx].max():.2f}]")
    print(f"  dim epochs    span mag [{mag_truth[dim_idx].min():.2f}, "
          f"{mag_truth[dim_idx].max():.2f}]")

    # Find sharpest in each region
    print(f"\nfinding sharpest in bright region ({n_region} epochs)...")
    t_bright, Ct_bright = find_sharpest_in_region(
        model, pool, sun_unit, obs_unit, obs_dist, mag_truth, bright_idx)
    print(f"  bright-sharpest: t={t_bright}, |C_t|={Ct_bright}")

    print(f"finding sharpest in dim region ({n_region} epochs)...")
    t_dim, Ct_dim = find_sharpest_in_region(
        model, pool, sun_unit, obs_unit, obs_dist, mag_truth, dim_idx)
    print(f"  dim-sharpest: t={t_dim}, |C_t|={Ct_dim}")

    # Topology measurements
    bright_topo = measure_anchor(
        model, pool, sun_unit, obs_unit, obs_dist, mag_truth,
        t_bright, args.cluster_threshold_deg)
    dim_topo = measure_anchor(
        model, pool, sun_unit, obs_unit, obs_dist, mag_truth,
        t_dim, args.cluster_threshold_deg)

    # Report
    print(f"\n=== seed {args.seed} (|ω|={omega_mag_dps:.3f}) cluster threshold {args.cluster_threshold_deg}° ===")
    print(f"{'kind':>10} {'t':>4} {'mag':>6} {'|C|':>5} {'#clst':>5} {'top3sz':>15} {'top3frac':>9} {'intra_max':>10} {'gap':>6}")
    for kind, topo in [("bright", bright_topo), ("dim", dim_topo)]:
        if topo is None:
            print(f"{kind:>10}    n/a (|C|<2)")
            continue
        sz3 = "/".join(str(s) for s in topo["cluster_sizes"][:3])
        top3_frac = sum(topo["cluster_sizes"][:3]) / topo["n_total"] if topo["n_total"] > 0 else 0
        print(f"{kind:>10} {topo['t']:>4d} {topo['mag']:>6.2f} "
              f"{topo['n_total']:>5d} {topo['n_clusters']:>5d} {sz3:>15} "
              f"{top3_frac:>9.2f} {topo['intra_max_deg']:>10.1f} "
              f"{topo['between_gap_deg']:>6.1f}")

    out_dir = SURVEY / "results" / "s060_cohort_topology"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": int(args.seed),
        "omega_mag_dps": omega_mag_dps,
        "n_pool": int(args.n_pool),
        "region_pct": float(args.region_pct),
        "cluster_threshold_deg": float(args.cluster_threshold_deg),
        "bright": {"t": int(t_bright), "Ct": int(Ct_bright), "topology": bright_topo},
        "dim": {"t": int(t_dim), "Ct": int(Ct_dim), "topology": dim_topo},
        "wall_s": float(time.time() - t0),
    }
    with open(out_dir / f"seed{args.seed:03d}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / f'seed{args.seed:03d}.json'}")


if __name__ == "__main__":
    main()
