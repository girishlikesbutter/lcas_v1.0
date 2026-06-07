"""s060_anchor_topology — classify each chosen sharp anchor by survivor topology.

For each of the cluster-independent sharp anchors (from s060_sharpness_map):
  - Project pool at the anchor epoch.
  - Get survivor quaternions (|surrogate_pred - measured| < TOL_MAG).
  - Greedy-cluster survivors via geodesic distance (threshold = 20°).
  - Report: n_clusters, cluster sizes, intra-cluster spread, max pairwise.
  - Classify topology:
        single tight cluster   (1 cluster, intra < 30°)
        multimodal             (≥2 clusters, between-cluster gap > 30°)
        continuous spread      (1 "cluster" with intra-spread > 60°, low cluster contrast)

Usage:
    python experiments/s060_anchor_topology.py --seed 28 --n-pool 25000
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


def quat_geodesic_deg_batch(q1_wxyz, q_pool_wxyz):
    """Geodesic angle in degrees between q1 and each q in pool, antipode-aware."""
    dots = np.abs(q_pool_wxyz @ np.asarray(q1_wxyz, float))
    dots = np.clip(dots, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def greedy_cluster_quats(q_wxyz: np.ndarray, threshold_deg: float = 20.0):
    """Greedy clustering: pick remaining element, group neighbors within
    threshold geodesic, repeat. Returns list of cluster index arrays.
    """
    n = q_wxyz.shape[0]
    unassigned = np.ones(n, dtype=bool)
    clusters = []
    while unassigned.any():
        idx_pool = np.where(unassigned)[0]
        seed_idx = idx_pool[0]
        d = quat_geodesic_deg_batch(q_wxyz[seed_idx], q_wxyz[idx_pool])
        members_local = idx_pool[d < threshold_deg]
        clusters.append(members_local)
        unassigned[members_local] = False
    clusters.sort(key=lambda c: -len(c))
    return clusters


def classify_topology(clusters, q_wxyz):
    """Heuristic topology label."""
    sizes = np.array([len(c) for c in clusters])
    n_total = sizes.sum()
    n_clusters = len(clusters)

    # Intra-cluster spread for the largest cluster
    big = clusters[0]
    if len(big) >= 2:
        # Sample up to 200 pairs for speed
        rng = np.random.default_rng(0)
        idx = rng.choice(len(big), size=min(len(big), 200), replace=False)
        sub = q_wxyz[big[idx]]
        pairwise = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pairwise.append(quat_geodesic_deg_batch(sub[i], sub[j:j+1])[0])
        intra_max = float(np.max(pairwise)) if pairwise else 0.0
        intra_med = float(np.median(pairwise)) if pairwise else 0.0
    else:
        intra_max = 0.0
        intra_med = 0.0

    # Between-cluster gap (largest two cluster centroids if applicable)
    if n_clusters >= 2 and len(clusters[0]) >= 1 and len(clusters[1]) >= 1:
        # Use first elements as proxies for centroid (greedy seeds)
        gap = float(quat_geodesic_deg_batch(q_wxyz[clusters[0][0]],
                                            q_wxyz[clusters[1][0]:clusters[1][0]+1])[0])
    else:
        gap = 0.0

    # Largest cluster fraction
    big_frac = float(sizes[0] / n_total) if n_total > 0 else 0.0

    # Topology label
    if n_clusters == 1 and intra_max < 30.0:
        topology = "tight"
    elif n_clusters == 1 and intra_max >= 60.0:
        topology = "spread"  # 1 "cluster" but huge intra-spread → continuous family
    elif n_clusters >= 2 and gap > 30.0 and big_frac < 0.7:
        topology = "multimodal"
    elif n_clusters >= 2 and big_frac >= 0.7:
        topology = "main+stragglers"
    else:
        topology = "mixed"

    return {
        "n_total": int(n_total),
        "n_clusters": int(n_clusters),
        "cluster_sizes": sizes.tolist()[:6],  # top-6
        "biggest_cluster_frac": big_frac,
        "intra_max_deg": intra_max,
        "intra_med_deg": intra_med,
        "between_gap_deg": gap,
        "topology": topology,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-pool", type=int, default=25_000)
    ap.add_argument("--cluster-threshold-deg", type=float, default=20.0)
    ap.add_argument("--max-anchors", type=int, default=10)
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== s060_anchor_topology — seed {args.seed} ===\n")

    # Find chosen anchors (greedy K from cached sharpness_map)
    sharp_npz = SURVEY / "results" / "s060_sharpness_map" / f"seed{args.seed:03d}" / "sharpness_map.npz"
    if not sharp_npz.exists():
        print(f"ERROR: {sharp_npz} not found. Run s060_sharpness_map first.")
        sys.exit(1)
    smap = np.load(sharp_npz)
    Ct = smap["Ct"]
    mag = smap["mag_truth"]

    chosen = []
    Ct_work = Ct.astype(float).copy()
    while len(chosen) < args.max_anchors:
        idx = int(np.argmin(Ct_work))
        if Ct_work[idx] > 500:
            break
        chosen.append(idx)
        lo, hi = max(0, idx - 30), min(len(Ct), idx + 31)
        Ct_work[lo:hi] = np.inf
    print(f"chosen anchors (Δt≥30, |C_t|<500): {chosen}\n")

    # Build context
    print("loading context...")
    ctx = build_context(seed=args.seed)
    obs_dist = np.asarray(ctx["obs_dist"])
    sun_unit, obs_unit = compute_j2000_units(ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"])
    mag_truth = np.asarray(ctx["mag_hifi_truth"])

    # Build pool
    print(f"building Sobol pool N={args.n_pool}...")
    pool = sample_so3_pool(args.n_pool, sample_seed=args.rng_seed)

    model = get_model()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl; threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

    # Per-anchor topology measurement
    rows = []
    for rank, t in enumerate(chosen, 1):
        t0 = time.time()
        k1_b, k2_b = project_directions(pool["R_cache"], sun_unit[t], obs_unit[t])
        _, keep = survive_at_epoch(
            model, k1_b, k2_b, float(obs_dist[t]),
            SP_DEG, AD_DEG, float(mag_truth[t]), TOL_MAG,
        )
        survivors_q = pool["q_pool_wxyz"][keep]
        clusters = greedy_cluster_quats(survivors_q, args.cluster_threshold_deg)
        topo = classify_topology(clusters, survivors_q)
        topo["t"] = int(t)
        topo["rank"] = rank
        topo["mag"] = float(mag_truth[t])
        topo["mag_pct"] = float((mag_truth[t] - mag_truth.min()) /
                                (mag_truth.max() - mag_truth.min()) * 100)
        topo["wall_s"] = time.time() - t0
        rows.append(topo)

    print(f"\n=== seed {args.seed} anchor topology (cluster threshold {args.cluster_threshold_deg}°) ===")
    print(f"{'rank':>4} {'t':>4} {'mag':>6} {'pct':>5} {'|C|':>5} {'#clst':>5} {'top3sz':>15} {'big_frac':>9} {'intra_max':>10} {'gap':>6} {'topology':>16}")
    for r in rows:
        sz3 = "/".join(str(s) for s in r["cluster_sizes"][:3])
        print(f"{r['rank']:>4} {r['t']:>4d} {r['mag']:>6.2f} {r['mag_pct']:>4.0f}% "
              f"{r['n_total']:>5d} {r['n_clusters']:>5d} {sz3:>15} "
              f"{r['biggest_cluster_frac']:>9.2f} {r['intra_max_deg']:>10.1f} "
              f"{r['between_gap_deg']:>6.1f} {r['topology']:>16}")

    out_dir = SURVEY / "results" / "s060_sharpness_map" / f"seed{args.seed:03d}"
    with open(out_dir / "anchor_topology.json", "w") as f:
        json.dump({
            "seed": int(args.seed),
            "n_pool": int(args.n_pool),
            "cluster_threshold_deg": float(args.cluster_threshold_deg),
            "anchors": rows,
        }, f, indent=2)
    print(f"\nSaved: {out_dir / 'anchor_topology.json'}")


if __name__ == "__main__":
    main()
