"""s061b — N-step cloud threading on seed 89.

Extends s061a from 1 step to multi-step. At each step k:
  1. For each survivor at step k-1, sample N_per perturbations in geodesic
     ball of radius ω_max·Δt.
  2. Brightness-filter at epoch t_k via v2 surrogate.
  3. Cap to MAX_PER_STEP via random subsample (keeps wall bounded; tracks
     predecessor for cluster lineage).
  4. Track which ANCHOR cluster each survivor descends from (lineage).

Output per step: total survivor count, per-anchor-cluster lineage count,
per-cluster centroid drift from truth, distance to truth at that epoch.

Hypothesis: at adjacent epochs brightness is weakly discriminative
(s061a showed 54-72% retention); over many steps, cumulative brightness
constraints kill non-truth clusters. Different clusters thread different
polhodes; only the truth cluster threads consistently.

Usage:
    python experiments/s061b_multi_step_thread.py --seed 89 \\
        --anchor-t 208 --n-steps 20 --omega-max-dps 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from lib.c_t_pipeline import (  # noqa: E402
    sample_so3_pool, compute_j2000_units, project_directions, survive_at_epoch,
)
from lib.hifi_render import build_context  # noqa: E402
from lib.surrogate_eval import get_model  # noqa: E402
from lib.traj_load import load_truth  # noqa: E402

# Reuse helpers from s061a
sys.path.insert(0, str(SURVEY / "experiments"))
from s061a_thread_smoke import (  # noqa: E402
    quat_mul_batch, quat_conj, quat_log,
    quat_geodesic_deg_batch,
    sample_perturbations_in_geodesic_ball,
    body_twin_canonicalize_q,
    greedy_cluster,
)

TOL_MAG = 0.10
SP_DEG = 0.0
AD_DEG = 15.0


def thread_step(
    anchor_q: np.ndarray,            # (M, 4) current cloud
    anchor_lineage: np.ndarray,      # (M,) original anchor cluster id per survivor
    target_t: int,
    delta_t_sec: float,
    omega_max_rad_per_sec: float,
    n_per_anchor: int,
    max_total: int,
    model,
    sun_unit, obs_unit, obs_dist, mag_truth,
    rng: np.random.Generator,
):
    """One threading step. Sample tubes, filter brightness, cap to max_total
    via random subsample, return (q_survivors, lineage)."""
    M = anchor_q.shape[0]
    if M == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64), 0, 0
    r_max = omega_max_rad_per_sec * delta_t_sec

    # Cap per-step candidate budget by adapting n_per_anchor
    target_n_candidates = max(M * n_per_anchor, 5_000)
    if target_n_candidates > 200_000:  # hard cap
        n_per = max(1, 200_000 // M)
    else:
        n_per = n_per_anchor

    q_anchor_tiled = np.repeat(anchor_q, n_per, axis=0)  # (M*n_per, 4)
    lineage_tiled = np.repeat(anchor_lineage, n_per)

    delta_q = sample_perturbations_in_geodesic_ball(M * n_per, r_max, rng)
    q_child = quat_mul_batch(q_anchor_tiled, delta_q)

    R_child = Rotation.from_quat(q_child[:, [1, 2, 3, 0]]).as_matrix()
    k1_b = R_child @ sun_unit[target_t]
    k2_b = R_child @ obs_unit[target_t]
    pred_mag = model.predict_magnitude(
        k1_b, k2_b, SP_DEG, AD_DEG,
        np.full(R_child.shape[0], float(obs_dist[target_t]))
    )
    keep = np.abs(pred_mag - float(mag_truth[target_t])) < TOL_MAG

    q_surv = q_child[keep]
    lineage_surv = lineage_tiled[keep]

    n_total_candidates = q_child.shape[0]
    n_surv_unfiltered = q_surv.shape[0]

    if q_surv.shape[0] > max_total:
        idx = rng.choice(q_surv.shape[0], size=max_total, replace=False)
        q_surv = q_surv[idx]
        lineage_surv = lineage_surv[idx]

    return q_surv, lineage_surv, n_total_candidates, n_surv_unfiltered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=89)
    ap.add_argument("--anchor-t", type=int, default=208)
    ap.add_argument("--n-pool", type=int, default=25_000)
    ap.add_argument("--cluster-threshold-deg", type=float, default=40.0)
    ap.add_argument("--omega-max-dps", type=float, default=0.5,
                    help="Tighter than s061a (1.5) — seed 89 |ω|=0.24 dps so 0.5 = 2× margin")
    ap.add_argument("--n-per-anchor", type=int, default=20)
    ap.add_argument("--max-per-step", type=int, default=2000)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== s061b — multi-step threading ===")
    print(f"seed={args.seed}, anchor_t={args.anchor_t}, ω_max={args.omega_max_dps} dps")
    print(f"n_steps={args.n_steps}, n_per_anchor={args.n_per_anchor}, max_per_step={args.max_per_step}\n")

    out_dir = SURVEY / "results" / "s061b_multi_step_thread" / f"seed{args.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading context + v2 surrogate...")
    ctx = build_context(seed=args.seed)
    obs_dist = np.asarray(ctx["obs_dist"])
    sun_unit, obs_unit = compute_j2000_units(ctx["sun_pos"], ctx["obs_pos"], ctx["sat_pos"])
    mag_truth = np.asarray(ctx["mag_hifi_truth"])
    obs_times = np.asarray(ctx["observation_times"])
    full_truth = load_truth(args.seed)
    truth_quaternions = np.asarray(full_truth["quaternions"])
    omega0_truth = np.asarray(ctx["omega0_truth_rad"])

    n_obs = obs_times.shape[0]
    omega_max_rad = args.omega_max_dps * np.pi / 180.0

    # === Build anchor cloud ===================================================
    t_a = args.anchor_t
    print(f"\n--- ANCHOR cloud at t={t_a} ---")
    pool = sample_so3_pool(args.n_pool, sample_seed=args.rng_seed)
    model = get_model()
    k1_b, k2_b = project_directions(pool["R_cache"], sun_unit[t_a], obs_unit[t_a])
    _, keep = survive_at_epoch(
        model, k1_b, k2_b, float(obs_dist[t_a]),
        SP_DEG, AD_DEG, float(mag_truth[t_a]), TOL_MAG,
    )
    anchor_q = body_twin_canonicalize_q(pool["q_pool_wxyz"][keep])
    print(f"  |C_anchor|={anchor_q.shape[0]}")

    clusters = greedy_cluster(anchor_q, args.cluster_threshold_deg)
    cluster_id = np.full(anchor_q.shape[0], -1, dtype=np.int64)
    for cid, members in enumerate(clusters):
        cluster_id[members] = cid
    cluster_sizes = [len(c) for c in clusters]
    n_anchor_clusters = len(clusters)
    print(f"  n_clusters={n_anchor_clusters}, sizes top6={cluster_sizes[:6]}")

    # Truth cluster
    q_truth_a = truth_quaternions[t_a]
    q_truth_a_canon = body_twin_canonicalize_q(q_truth_a[None, :])[0]
    truth_dist_anchor = quat_geodesic_deg_batch(q_truth_a_canon, anchor_q)
    truth_idx_anchor = int(np.argmin(truth_dist_anchor))
    truth_cluster = int(cluster_id[truth_idx_anchor])
    print(f"  truth in cluster {truth_cluster} (size {cluster_sizes[truth_cluster]}), "
          f"closest anchor {truth_dist_anchor[truth_idx_anchor]:.2f}°")

    # === Multi-step thread (right) ============================================
    def run_thread(direction: str):
        sign = +1 if direction == "right" else -1
        steps_data = []
        current_q = anchor_q.copy()
        current_lineage = cluster_id.copy()
        rng = np.random.default_rng(args.rng_seed + (1 if direction == "right" else 2))

        for k in range(1, args.n_steps + 1):
            target_t = t_a + sign * k
            if target_t < 0 or target_t >= n_obs:
                break
            delta_t = abs(obs_times[target_t] - obs_times[target_t - sign])

            t0 = time.time()
            new_q, new_lineage, n_cand, n_surv_raw = thread_step(
                anchor_q=current_q,
                anchor_lineage=current_lineage,
                target_t=target_t,
                delta_t_sec=delta_t,
                omega_max_rad_per_sec=omega_max_rad,
                n_per_anchor=args.n_per_anchor,
                max_total=args.max_per_step,
                model=model,
                sun_unit=sun_unit, obs_unit=obs_unit,
                obs_dist=obs_dist, mag_truth=mag_truth,
                rng=rng,
            )

            # Lineage counts
            lineage_counts = {}
            for cid in range(n_anchor_clusters):
                lineage_counts[cid] = int((new_lineage == cid).sum())

            # Distance to truth at target_t
            q_truth_target = body_twin_canonicalize_q(
                truth_quaternions[target_t][None, :]
            )[0]
            if new_q.shape[0] > 0:
                dist_to_truth = quat_geodesic_deg_batch(q_truth_target, new_q)
                truth_min_dist = float(dist_to_truth.min())
                # Truth-cluster lineage subset
                truth_lineage_mask = new_lineage == truth_cluster
                if truth_lineage_mask.sum() > 0:
                    truth_lineage_dist = float(dist_to_truth[truth_lineage_mask].min())
                else:
                    truth_lineage_dist = float("inf")
            else:
                truth_min_dist = float("inf")
                truth_lineage_dist = float("inf")

            wall = time.time() - t0
            steps_data.append({
                "step": k,
                "target_t": int(target_t),
                "delta_t_sec": float(delta_t),
                "n_candidates_total": int(n_cand),
                "n_survivors_raw": int(n_surv_raw),
                "n_survivors_subsampled": int(new_q.shape[0]),
                "lineage_counts": lineage_counts,
                "n_anchor_clusters_alive": int(sum(1 for c in lineage_counts.values() if c > 0)),
                "truth_cluster_alive": bool(lineage_counts[truth_cluster] > 0),
                "truth_min_dist_deg": truth_min_dist,
                "truth_cluster_min_dist_deg": truth_lineage_dist,
                "wall_s": wall,
            })

            # Print summary
            alive_marker = " 🟢" if lineage_counts[truth_cluster] > 0 else " 🔴"
            line = (f"  step {k:2d} t={target_t:3d}: cand={n_cand:6d} surv={n_surv_raw:5d} "
                    f"sub={new_q.shape[0]:4d} clusters_alive={steps_data[-1]['n_anchor_clusters_alive']}/"
                    f"{n_anchor_clusters} truth_cl={lineage_counts[truth_cluster]}{alive_marker} "
                    f"d_truth={truth_min_dist:5.1f}° (truth_cl_min={truth_lineage_dist:5.1f}°) "
                    f"wall={wall:.1f}s")
            print(line)

            current_q = new_q
            current_lineage = new_lineage
            if new_q.shape[0] == 0:
                print(f"    DEAD at step {k} — all candidates failed brightness filter.")
                break

        return steps_data

    print(f"\n--- THREADING RIGHT ---")
    right_data = run_thread("right")
    print(f"\n--- THREADING LEFT ---")
    left_data = run_thread("left")

    # === Save ==================================================================
    summary = {
        "seed": int(args.seed),
        "anchor_t": int(args.anchor_t),
        "omega_max_dps": float(args.omega_max_dps),
        "n_per_anchor": int(args.n_per_anchor),
        "max_per_step": int(args.max_per_step),
        "anchor_n_clusters": int(n_anchor_clusters),
        "anchor_cluster_sizes": cluster_sizes,
        "truth_cluster_id": int(truth_cluster),
        "right_steps": right_data,
        "left_steps": left_data,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")

    # Final verdict
    print(f"\n=== VERDICT ===")
    rstats = right_data[-1] if right_data else None
    lstats = left_data[-1] if left_data else None
    if rstats:
        print(f"  RIGHT @ step {rstats['step']} (t={rstats['target_t']}): "
              f"{rstats['n_anchor_clusters_alive']}/{n_anchor_clusters} clusters alive, "
              f"truth_cluster_alive={rstats['truth_cluster_alive']}")
    if lstats:
        print(f"  LEFT  @ step {lstats['step']} (t={lstats['target_t']}): "
              f"{lstats['n_anchor_clusters_alive']}/{n_anchor_clusters} clusters alive, "
              f"truth_cluster_alive={lstats['truth_cluster_alive']}")


if __name__ == "__main__":
    main()
