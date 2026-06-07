"""s061c — multi-step threading on seed 28 (high |ω|=1.43 dps, narrow basin).

Seed 28 is where s059j FAILED (ω-grid quantization 7° → 5° truth offset →
ρ=38 at W=10 with EXACT q_a). The threading architecture replaces the
ω-grid with the connectability tube — ω falls out of the per-step
geodesic step, no quantization. With ω_max=1.5 dps and Δt≈7.2 s, tube
radius is ~10.8°; truth step is |ω|·Δt = 1.43 × 7.2 / 57.3 ≈ 10.3°.
Tube is BARELY big enough — meaning the connectability filter is much
more discriminative here than on slow seed 89.

Anchor candidates from s060_sharpness_map seed 28:
  t=312, |C_t|=5  (mag=20.6, dimmest in LC, dim-extreme anchor)
  t=224, |C_t|=11 (mag=19.5, secondary dim)
  t=390, |C_t|=14 (mag=5.0,  bright peak)

Default: t=312 with N=100k Sobol (10× s061b) to get a slightly larger
anchor cloud since |C_t| is so small. Body-twin canon merges further.

Test:
  1. Does truth track on seed 28 (where s059j ω-grid failed)?
  2. Do clusters DIE on seed 28 (where they didn't on seed 89)?
  3. Does ω derived from threaded survivors match truth |ω|?

Usage:
    python experiments/s061c_seed28_thread.py --seed 28 \\
        --anchor-t 312 --n-steps 20 --omega-max-dps 1.5 --n-pool 100000
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

sys.path.insert(0, str(SURVEY / "experiments"))
from s061a_thread_smoke import (  # noqa: E402
    quat_mul_batch, quat_conj, quat_log,
    quat_geodesic_deg_batch,
    sample_perturbations_in_geodesic_ball,
    body_twin_canonicalize_q,
    greedy_cluster,
)
from s061b_multi_step_thread import thread_step  # noqa: E402

TOL_MAG = 0.10
SP_DEG = 0.0
AD_DEG = 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--anchor-t", type=int, default=312)
    ap.add_argument("--n-pool", type=int, default=100_000)
    ap.add_argument("--cluster-threshold-deg", type=float, default=40.0)
    ap.add_argument("--omega-max-dps", type=float, default=1.5)
    ap.add_argument("--n-per-anchor", type=int, default=100)
    ap.add_argument("--max-per-step", type=int, default=2000)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== s061c — multi-step threading, seed {args.seed} ===")
    print(f"anchor_t={args.anchor_t}, ω_max={args.omega_max_dps} dps "
          f"(truth |ω| seed28 ≈ 1.43 dps → tube ≈ 1.05× truth)")
    print(f"n_steps={args.n_steps}, n_per_anchor={args.n_per_anchor}, "
          f"max_per_step={args.max_per_step}, N_pool={args.n_pool}\n")

    out_dir = SURVEY / "results" / "s061c_seed28_thread" / f"seed{args.seed:03d}"
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
    truth_omega_dps = float(np.linalg.norm(omega0_truth) * 180 / np.pi)
    print(f"truth |ω|={truth_omega_dps:.3f} dps")
    delta_t_typical = float(obs_times[1] - obs_times[0])
    truth_step_deg = truth_omega_dps * delta_t_typical
    tube_radius_deg = args.omega_max_dps * delta_t_typical
    print(f"Δt={delta_t_typical:.3f}s, truth step={truth_step_deg:.2f}°, "
          f"tube radius={tube_radius_deg:.2f}° (ratio={tube_radius_deg/truth_step_deg:.2f})")

    # === Anchor cloud ===
    t_a = args.anchor_t
    print(f"\n--- ANCHOR cloud at t={t_a} (N_pool={args.n_pool}) ---")
    t0 = time.time()
    pool = sample_so3_pool(args.n_pool, sample_seed=args.rng_seed)
    model = get_model()
    k1_b, k2_b = project_directions(pool["R_cache"], sun_unit[t_a], obs_unit[t_a])
    _, keep = survive_at_epoch(
        model, k1_b, k2_b, float(obs_dist[t_a]),
        SP_DEG, AD_DEG, float(mag_truth[t_a]), TOL_MAG,
    )
    raw_survivors = pool["q_pool_wxyz"][keep]
    print(f"  raw |C_t={t_a}|={raw_survivors.shape[0]} (wall {time.time()-t0:.1f}s)")

    # Body-twin canonicalize
    anchor_q = body_twin_canonicalize_q(raw_survivors)

    # Cluster
    clusters = greedy_cluster(anchor_q, args.cluster_threshold_deg)
    cluster_id = np.full(anchor_q.shape[0], -1, dtype=np.int64)
    for cid, members in enumerate(clusters):
        cluster_id[members] = cid
    cluster_sizes = [len(c) for c in clusters]
    n_anchor_clusters = len(clusters)
    print(f"  n_clusters={n_anchor_clusters}, sizes={cluster_sizes}")

    # Truth cluster
    q_truth_a = truth_quaternions[t_a]
    q_truth_a_canon = body_twin_canonicalize_q(q_truth_a[None, :])[0]
    truth_dist_anchor = quat_geodesic_deg_batch(q_truth_a_canon, anchor_q)
    truth_idx_anchor = int(np.argmin(truth_dist_anchor))
    truth_cluster = int(cluster_id[truth_idx_anchor])
    print(f"  truth in cluster {truth_cluster}, closest anchor {truth_dist_anchor[truth_idx_anchor]:.2f}°")

    # === Threading ===
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

            lineage_counts = {}
            for cid in range(n_anchor_clusters):
                lineage_counts[cid] = int((new_lineage == cid).sum())

            q_truth_target = body_twin_canonicalize_q(
                truth_quaternions[target_t][None, :]
            )[0]
            if new_q.shape[0] > 0:
                dist_to_truth = quat_geodesic_deg_batch(q_truth_target, new_q)
                truth_min_dist = float(dist_to_truth.min())
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
                "step": k, "target_t": int(target_t), "delta_t_sec": float(delta_t),
                "n_candidates": int(n_cand), "n_survivors_raw": int(n_surv_raw),
                "n_survivors_subsampled": int(new_q.shape[0]),
                "lineage_counts": lineage_counts,
                "n_anchor_clusters_alive": int(sum(1 for c in lineage_counts.values() if c > 0)),
                "truth_cluster_alive": bool(lineage_counts[truth_cluster] > 0),
                "truth_min_dist_deg": truth_min_dist,
                "truth_cluster_min_dist_deg": truth_lineage_dist,
                "wall_s": wall,
            })

            alive_marker = " 🟢" if lineage_counts[truth_cluster] > 0 else " 🔴"
            line = (f"  step {k:2d} t={target_t:3d}: cand={n_cand:6d} surv={n_surv_raw:6d} "
                    f"sub={new_q.shape[0]:5d} cl_alive={steps_data[-1]['n_anchor_clusters_alive']}/"
                    f"{n_anchor_clusters} truth_cl={lineage_counts[truth_cluster]:4d}{alive_marker} "
                    f"d_truth={truth_min_dist:6.1f}° (truth_cl={truth_lineage_dist:6.1f}°) "
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

    # === Headline yield ===
    print(f"\n=== HEADLINE ===")
    if right_data:
        last = right_data[-1]
        print(f"  RIGHT @ step {last['step']} (t={last['target_t']}): "
              f"{last['n_anchor_clusters_alive']}/{n_anchor_clusters} clusters alive, "
              f"truth alive={last['truth_cluster_alive']}")
    if left_data:
        last = left_data[-1]
        print(f"  LEFT  @ step {last['step']} (t={last['target_t']}): "
              f"{last['n_anchor_clusters_alive']}/{n_anchor_clusters} clusters alive, "
              f"truth alive={last['truth_cluster_alive']}")

    # ω-derived from final right step (compare derived ω to truth)
    # We use only the last step's δq to estimate per-step ω via |ω|=|log|/Δt
    # (Note: this is per-step instantaneous, not a long-baseline integrated ω)
    if right_data and right_data[-1]["n_survivors_subsampled"] > 0:
        # Re-build the full pipeline state to get final delta_q stats... too
        # complex; just report per-step truth-cluster lineage tracking quality
        truth_cl_dists = [s["truth_cluster_min_dist_deg"] for s in right_data
                          if s["truth_cluster_alive"]]
        if truth_cl_dists:
            print(f"  RIGHT truth-cluster lineage min dist: "
                  f"first={truth_cl_dists[0]:.1f}° final={truth_cl_dists[-1]:.1f}° "
                  f"max={max(truth_cl_dists):.1f}°")

    # === Save ===
    summary = {
        "seed": int(args.seed),
        "anchor_t": int(args.anchor_t),
        "n_pool": int(args.n_pool),
        "omega_max_dps": float(args.omega_max_dps),
        "truth_omega_dps": float(truth_omega_dps),
        "tube_to_truth_ratio": float(tube_radius_deg / truth_step_deg),
        "anchor_n_clusters": int(n_anchor_clusters),
        "anchor_cluster_sizes": cluster_sizes,
        "anchor_raw_Ct": int(raw_survivors.shape[0]),
        "truth_cluster_id": int(truth_cluster),
        "right_steps": right_data,
        "left_steps": left_data,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
