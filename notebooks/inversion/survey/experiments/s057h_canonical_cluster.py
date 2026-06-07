"""s057h — body-twin canonicalisation + cluster-rank candidates from s057g.

Hypothesis: s057g found the truth-candidate at rank 55/548. The top
candidates were dominated by body-twins of truth (qa_d≈180°, ω at
twin-direction). Per `lib.twin.canonical_batch`, the body-twin and truth
canonicalise to the same hemisphere representative — so de-duplication
should collapse them into a single cluster, raising the rank of the
"truth canonical" cluster significantly.

This script:
1. Regenerates candidates and forward-propagation scores (same as s057g)
2. Canonicalises each (q_a, ω) via lib.twin.canonical_batch
3. Greedy-clusters canonicalised candidates by (q_threshold, ω_threshold)
4. Per cluster: cumulative score (sum of member scores), best member,
   distance of cluster centroid to (truth_q_canon, truth_ω_canon)
5. Re-ranks clusters by cumulative score; reports truth cluster's rank
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
from lib.twin import canonical_batch

DENSE_RUN = SURVEY / "results" / "s048c_cloud_viewer" / "seed089" / "8bb9b81f1602" / "spread.npz"
TRAJ089 = SURVEY / "data" / "trajectories" / "traj_seed089.npz"
OUT = SURVEY / "results" / "s057h_canonical_cluster"
OUT.mkdir(parents=True, exist_ok=True)

T_A = 411
DELTA_GEN = 15
PRIOR_BRACKET = (0.75, 1.25)
HIT_THRESHOLD_DEG = 5.0
CONST_OMEGA_RELIABLE_MAX_DEG = 20.0
SMOKE_DELTAS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80]

# Cluster thresholds (in canonical space)
CLUSTER_Q_DEG = 8.0
CLUSTER_OM_DEG = 15.0
CLUSTER_OM_MAG_PCT = 25.0


def wxyz_to_xyzw(q): return q[..., [1, 2, 3, 0]]
def xyzw_to_wxyz(q): return q[..., [3, 0, 1, 2]]


def fd_omega_passive(q_a, q_b, dt):
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    R_b = Rotation.from_quat(wxyz_to_xyzw(q_b))
    return (R_b * R_a.inv()).as_rotvec() / dt


def propagate_const_omega(q_a, omega_rad_s, dt):
    rotvec = omega_rad_s * dt
    R_om = Rotation.from_rotvec(rotvec)
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    return xyzw_to_wxyz((R_om * R_a).as_quat())


def quat_ang_deg(q1, q2):
    return float(2 * np.degrees(np.arccos(
        np.clip(abs(np.dot(q1, q2)), 0, 1))))


def quat_ang_deg_batch(q_arr, q_ref):
    """Pairwise geodesic angle from each row of q_arr to q_ref. Returns deg."""
    dots = np.abs(q_arr @ q_ref)
    return 2 * np.degrees(np.arccos(np.clip(dots, 0, 1)))


def ang_to_axis(om_arr, ref):
    om_norm = np.linalg.norm(om_arr, axis=-1, keepdims=True)
    safe = np.where(om_norm > 1e-12, om_norm, 1.0)
    om_hat = om_arr / safe
    r_hat = ref / np.linalg.norm(ref)
    cos_a = np.abs(np.einsum("...i,i->...", om_hat, r_hat))
    return np.degrees(np.arccos(np.clip(cos_a, 0, 1)))


def main() -> dict:
    z = np.load(DENSE_RUN)
    survive_all = z["survive_all"]
    q_pool = z["q_pool_wxyz"]
    obs_times = z["obs_times"]

    traj = np.load(TRAJ089)
    q_truth_t = traj["quaternions"]
    dt_epoch = float(np.median(np.diff(obs_times)))

    # === STEP 1: smoke test (same as s057g) ===
    om_inst = fd_omega_passive(q_truth_t[T_A:T_A+1],
                                q_truth_t[T_A+1:T_A+2], dt_epoch)[0]
    valid_max_delta = 1
    for Δ in SMOKE_DELTAS:
        if T_A + Δ >= len(q_truth_t):
            continue
        q_pred = propagate_const_omega(q_truth_t[T_A], om_inst, Δ * dt_epoch)
        err = quat_ang_deg(q_pred, q_truth_t[T_A + Δ])
        if err <= CONST_OMEGA_RELIABLE_MAX_DEG:
            valid_max_delta = Δ
        else:
            break

    # === STEP 2: validators ===
    validators = []
    lo_v = max(0, T_A - valid_max_delta)
    hi_v = min(len(survive_all), T_A + valid_max_delta + 1)
    for t_v in range(lo_v, hi_v):
        if t_v == T_A:
            continue
        idx_v = np.where(survive_all[t_v])[0]
        if len(idx_v) == 0:
            continue
        validators.append({
            "t_v": t_v, "delta": t_v - T_A, "C_v": q_pool[idx_v],
            "n_surv": int(len(idx_v)),
            "weight": float(np.log(100000.0 / len(idx_v))),
        })

    # === STEP 3: candidates ===
    idx_a = np.where(survive_all[T_A])[0]
    C_a = q_pool[idx_a]
    n_a = len(C_a)
    Δgen_t_b = T_A + DELTA_GEN
    idx_b = np.where(survive_all[Δgen_t_b])[0]
    C_b = q_pool[idx_b]
    n_b = len(C_b)
    Δt_gen = DELTA_GEN * dt_epoch
    om_truth = fd_omega_passive(q_truth_t[T_A:T_A+1],
                                 q_truth_t[Δgen_t_b:Δgen_t_b+1], Δt_gen)[0]
    om_truth_mag = float(np.linalg.norm(om_truth))

    Q_A = np.repeat(C_a, n_b, axis=0)
    Q_B = np.tile(C_b, (n_a, 1))
    om_all = fd_omega_passive(Q_A, Q_B, Δt_gen)
    om_mag_all = np.linalg.norm(om_all, axis=1)
    target = om_truth_mag
    mask = (om_mag_all >= target * PRIOR_BRACKET[0]) & \
           (om_mag_all <= target * PRIOR_BRACKET[1])
    Q_A_pass = Q_A[mask]
    om_pass = om_all[mask]
    n_cand = len(Q_A_pass)

    # === STEP 4: scoring (same as s057g) ===
    scores = np.zeros(n_cand)
    hit_counts = np.zeros(n_cand, dtype=int)
    for v in validators:
        Δ = v["delta"]
        Δt = Δ * dt_epoch
        q_pred = propagate_const_omega(Q_A_pass, om_pass, Δt)
        dots = np.abs(q_pred @ v["C_v"].T)
        max_dot = dots.max(axis=1)
        min_ang = 2 * np.degrees(np.arccos(np.clip(max_dot, 0, 1)))
        hit_mask = min_ang < HIT_THRESHOLD_DEG
        scores += v["weight"] * hit_mask
        hit_counts += hit_mask.astype(int)

    # diagnostic distances to truth
    qa_dist = quat_ang_deg_batch(Q_A_pass, q_truth_t[T_A])
    om_dist = ang_to_axis(om_pass, om_truth)
    truth_idx = int(np.argmin(qa_dist + om_dist))
    rank_truth_orig = int((scores > scores[truth_idx]).sum())
    print(f"=== regenerated s057g ===")
    print(f"  {n_cand} candidates, truth at rank {rank_truth_orig+1}/{n_cand} "
          f"({(rank_truth_orig+1)/n_cand*100:.2f}%-ile), score={scores[truth_idx]:.2f}")

    # === STEP 5: canonicalise everything ===
    print(f"\n=== body-twin canonicalisation ===")
    Q_A_canon, om_canon = canonical_batch(Q_A_pass, om_pass)
    # Truth canon (single point in canonical space)
    q_truth_canon, om_truth_canon = canonical_batch(
        q_truth_t[T_A][None, :], om_truth[None, :]
    )
    q_truth_canon = q_truth_canon[0]
    om_truth_canon = om_truth_canon[0]

    qa_dist_canon = quat_ang_deg_batch(Q_A_canon, q_truth_canon)
    om_dist_canon = ang_to_axis(om_canon, om_truth_canon)
    print(f"  truth's canonical (q_a, ω): qa_dist {qa_dist_canon[truth_idx]:.2f}°, "
          f"ω_dist {om_dist_canon[truth_idx]:.2f}°")
    # Now check: how many candidates ARE body-twins of truth?
    # A candidate is "twin-of-truth-class" iff its canonical form is close to truth canonical.
    truth_class_mask = (qa_dist_canon < CLUSTER_Q_DEG) & (om_dist_canon < CLUSTER_OM_DEG)
    n_truth_class = int(truth_class_mask.sum())
    print(f"  candidates within ({CLUSTER_Q_DEG}°q, {CLUSTER_OM_DEG}°ω) "
          f"of truth canonical: {n_truth_class}/{n_cand}")
    if n_truth_class:
        print(f"  scores in truth class: max={scores[truth_class_mask].max():.2f}, "
              f"sum={scores[truth_class_mask].sum():.2f}")

    # === STEP 6: greedy clustering on canonical candidates by score ===
    print(f"\n=== greedy clustering: q≤{CLUSTER_Q_DEG}°, ω≤{CLUSTER_OM_DEG}°, "
          f"|ω|≤{CLUSTER_OM_MAG_PCT}% ===")
    om_mag_canon = np.linalg.norm(om_canon, axis=1)

    order = np.argsort(-scores)  # descending score
    assigned = np.full(n_cand, -1, dtype=int)
    clusters = []
    for ci, seed_i in enumerate(order):
        if assigned[seed_i] != -1:
            continue
        # Find unassigned candidates within cluster thresholds
        unassigned = np.where(assigned == -1)[0]
        # angular dist of unassigned to seed in canonical (q, ω)
        d_q = quat_ang_deg_batch(Q_A_canon[unassigned], Q_A_canon[seed_i])
        d_om = ang_to_axis(om_canon[unassigned], om_canon[seed_i])
        d_om_mag_pct = np.abs(om_mag_canon[unassigned] - om_mag_canon[seed_i]) / \
                       om_mag_canon[seed_i] * 100
        in_cluster = (d_q < CLUSTER_Q_DEG) & (d_om < CLUSTER_OM_DEG) & \
                     (d_om_mag_pct < CLUSTER_OM_MAG_PCT)
        members = unassigned[in_cluster]
        cluster_id = len(clusters)
        assigned[members] = cluster_id
        cluster_score = float(scores[members].sum())
        cluster_max = float(scores[members].max())
        cluster_qa_dist = float(np.min(qa_dist_canon[members]))
        cluster_om_dist = float(np.min(om_dist_canon[members]))
        clusters.append({
            "cluster_id": cluster_id,
            "n_members": int(len(members)),
            "members": members.tolist(),
            "seed_idx": int(seed_i),
            "score_sum": cluster_score,
            "score_max": cluster_max,
            "min_qa_dist_to_truth_deg": cluster_qa_dist,
            "min_om_dist_to_truth_deg": cluster_om_dist,
            "centroid_q": Q_A_canon[seed_i].tolist(),
            "centroid_om": om_canon[seed_i].tolist(),
            "centroid_om_mag_dps": float(om_mag_canon[seed_i] * 180 / np.pi),
        })

    # Sort clusters by cumulative score
    clusters_sorted = sorted(clusters, key=lambda c: -c["score_sum"])
    # Where does truth cluster sit?
    truth_cluster_id = int(assigned[truth_idx])
    truth_cluster = clusters[truth_cluster_id]
    truth_rank_in_clusters = next(i for i, c in enumerate(clusters_sorted)
                                   if c["cluster_id"] == truth_cluster_id)
    print(f"\nClustering result:")
    print(f"  {n_cand} candidates → {len(clusters)} clusters")
    print(f"  truth cluster: id={truth_cluster_id}, n={truth_cluster['n_members']} members, "
          f"score_sum={truth_cluster['score_sum']:.2f}, score_max={truth_cluster['score_max']:.2f}")
    print(f"  truth cluster ranks {truth_rank_in_clusters+1}/{len(clusters)} by sum-score "
          f"({(truth_rank_in_clusters+1)/len(clusters)*100:.2f}%-ile)")

    # Top clusters
    print(f"\n  Top 15 clusters by sum-score:")
    for r, c in enumerate(clusters_sorted[:15]):
        is_truth = "← TRUTH" if c["cluster_id"] == truth_cluster_id else ""
        print(f"    rank {r+1:2d}: id={c['cluster_id']:3d}  n={c['n_members']:3d}  "
              f"sum={c['score_sum']:6.1f}  max={c['score_max']:5.1f}  "
              f"qa_d={c['min_qa_dist_to_truth_deg']:5.2f}°  "
              f"ω_d={c['min_om_dist_to_truth_deg']:5.2f}°  "
              f"|ω|={c['centroid_om_mag_dps']:.3f}dps  {is_truth}")

    # === FIGURE ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.hist(scores, bins=40, color="lightblue", edgecolor="grey", alpha=0.7,
            label="raw candidates (n={})".format(n_cand))
    cluster_sums = [c["score_sum"] for c in clusters]
    ax.hist(cluster_sums, bins=40, color="red", alpha=0.4,
            label=f"cluster sums (n={len(clusters)})")
    ax.axvline(scores[truth_idx], color="red", lw=2, ls="--",
               label=f"truth raw ({scores[truth_idx]:.1f})")
    ax.axvline(truth_cluster["score_sum"], color="purple", lw=2,
               label=f"truth cluster sum ({truth_cluster['score_sum']:.1f})")
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title(f"score distribution before/after clustering\n"
                 f"truth raw rank {rank_truth_orig+1}/{n_cand} → "
                 f"truth cluster rank {truth_rank_in_clusters+1}/{len(clusters)}")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    rank_x = np.arange(min(50, len(clusters_sorted)))
    cs_top = clusters_sorted[:len(rank_x)]
    sums_top = [c["score_sum"] for c in cs_top]
    is_t = [c["cluster_id"] == truth_cluster_id for c in cs_top]
    colors = ["red" if t else "lightblue" for t in is_t]
    edgecolors = ["black" if t else "grey" for t in is_t]
    ax.bar(rank_x, sums_top, color=colors, edgecolor=edgecolors)
    ax.set_xlabel("cluster rank")
    ax.set_ylabel("cluster sum score")
    ax.set_title(f"top {len(rank_x)} clusters: truth (red) at rank {truth_rank_in_clusters+1}")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    cluster_qa = [c["min_qa_dist_to_truth_deg"] for c in clusters]
    ax.scatter(cluster_qa, cluster_sums, s=8, alpha=0.4, color="b")
    ax.scatter([truth_cluster["min_qa_dist_to_truth_deg"]],
               [truth_cluster["score_sum"]],
               s=200, marker="*", color="red", edgecolor="black", zorder=10,
               label=f"truth cluster (rank {truth_rank_in_clusters+1})")
    ax.set_xlabel("cluster min qa-dist to truth_canon (deg)")
    ax.set_ylabel("cluster sum-score")
    ax.set_title("cluster sum-score vs canonicalised qa-distance to truth")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    cluster_om = [c["min_om_dist_to_truth_deg"] for c in clusters]
    ax.scatter(cluster_om, cluster_sums, s=8, alpha=0.4, color="b")
    ax.scatter([truth_cluster["min_om_dist_to_truth_deg"]],
               [truth_cluster["score_sum"]],
               s=200, marker="*", color="red", edgecolor="black", zorder=10,
               label=f"truth cluster")
    ax.set_xlabel("cluster min ω-dist to truth_canon (deg)")
    ax.set_ylabel("cluster sum-score")
    ax.set_title("cluster sum-score vs canonicalised ω-distance to truth")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_p = OUT / "canonical_cluster.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_p}")

    summary = {
        "seed": 89,
        "T_A": T_A,
        "n_candidates": n_cand,
        "n_clusters": len(clusters),
        "cluster_thresholds": {
            "q_deg": CLUSTER_Q_DEG, "om_deg": CLUSTER_OM_DEG,
            "om_mag_pct": CLUSTER_OM_MAG_PCT,
        },
        "truth_raw_rank": rank_truth_orig + 1,
        "truth_raw_score": float(scores[truth_idx]),
        "truth_cluster_id": truth_cluster_id,
        "truth_cluster_n_members": int(truth_cluster["n_members"]),
        "truth_cluster_score_sum": float(truth_cluster["score_sum"]),
        "truth_cluster_score_max": float(truth_cluster["score_max"]),
        "truth_cluster_rank": truth_rank_in_clusters + 1,
        "truth_cluster_rank_pct": float((truth_rank_in_clusters + 1) / len(clusters) * 100),
        "truth_class_count": n_truth_class,
        "top_15_clusters": [
            {"rank": r+1, "cluster_id": c["cluster_id"],
             "n_members": c["n_members"], "score_sum": c["score_sum"],
             "score_max": c["score_max"],
             "min_qa_dist_to_truth_deg": c["min_qa_dist_to_truth_deg"],
             "min_om_dist_to_truth_deg": c["min_om_dist_to_truth_deg"],
             "centroid_om_mag_dps": c["centroid_om_mag_dps"],
             "is_truth": c["cluster_id"] == truth_cluster_id}
            for r, c in enumerate(clusters_sorted[:15])
        ],
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
