"""s057f — per-q_a multi-Δt consistency test on seed 89.

Architecture: for each q_a ∈ C_{t_a=411}, walk forward to t_a+Δ for
Δ ∈ {3, 6, 9, 12, 15}. At each Δ, compute ω(q_a, q_b) finite-diff for
all q_b ∈ C_{t_a+Δ} that pass the |ω|-prior ±25% bracket. Centroid the
surviving ω's to get one estimate per (q_a, Δ).

Truth-q_a's per-Δ centroids should be ANGULARLY CONSISTENT (small spread)
because truth-ω̂(t_a) is a single direction over the polhode-slow time-
scale. Random q_a's should give per-Δ centroids in random directions.

Two diagnostics per q_a:
1. OPERATIONAL — across-Δ angular spread of |ω|-prior-filtered centroids
2. ORACLE — ω(q_a, closest-survivor-to-truth_q_b at Δ); across-Δ spread.

The oracle test isolates whether per-q_a consistency works WITH the right
q_b on the other side; the operational test asks whether the |ω|-prior
filter recovers it WITHOUT truth-q_b knowledge.

Headline: rank q_a's by consistency; does qa_rank=0 (closest-to-truth
survivor) rank #1?
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
DENSE_RUN = SURVEY / "results" / "s048c_cloud_viewer" / "seed089" / "8bb9b81f1602" / "spread.npz"
TRAJ089 = SURVEY / "data" / "trajectories" / "traj_seed089.npz"
OUT = SURVEY / "results" / "s057f_per_qa_consistency"
OUT.mkdir(parents=True, exist_ok=True)

T_A = 411
DELTAS = [3, 6, 9, 12, 15]
PRIOR_BRACKET = (0.75, 1.25)


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def fd_omega_passive(q_a, q_b, dt):
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    R_b = Rotation.from_quat(wxyz_to_xyzw(q_b))
    return (R_b * R_a.inv()).as_rotvec() / dt


def angular_dist_axes(om_3vec, ref):
    """Geodesic angle (deg) to axis ref, antipodal-aware (|cos|)."""
    om_norm = np.linalg.norm(om_3vec, axis=-1, keepdims=True)
    safe = np.where(om_norm > 1e-12, om_norm, 1.0)
    om_hat = om_3vec / safe
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

    # Anchor cloud + sort by distance to truth
    idx_a = np.where(survive_all[T_A])[0]
    C_a = q_pool[idx_a]
    n_a = len(idx_a)
    dots_a = np.abs(C_a @ q_truth_t[T_A])
    order = np.argsort(dots_a)[::-1]
    C_a_sorted = C_a[order]
    deg_to_truth_a = np.degrees(2 * np.arccos(np.clip(dots_a[order], 0, 1)))
    print(f"anchor t_a={T_A}, |C_a|={n_a}; closest-survivor distance to truth: {deg_to_truth_a[0]:.3f}°")

    # Pre-compute per-Δ data: truth-ω, C_b, closest-survivor-to-truth_q_b
    per_delta = {}
    for Δ in DELTAS:
        t_b = T_A + Δ
        Δt_s = Δ * dt_epoch
        idx_b = np.where(survive_all[t_b])[0]
        C_b = q_pool[idx_b]
        # truth ω over [t_a, t_a+Δ]
        truth_ω = fd_omega_passive(q_truth_t[T_A:T_A+1],
                                   q_truth_t[t_b:t_b+1], Δt_s)[0]
        truth_ω_mag = float(np.linalg.norm(truth_ω))
        # closest-survivor-to-truth at t_b
        if len(C_b):
            dots_b = np.abs(C_b @ q_truth_t[t_b])
            best_b = int(np.argmax(dots_b))
            qb_truth_in_cloud = C_b[best_b]
            qb_dist_truth = float(np.degrees(2 * np.arccos(np.clip(dots_b[best_b], 0, 1))))
        else:
            qb_truth_in_cloud = None
            qb_dist_truth = float("nan")
        per_delta[Δ] = {
            "t_b": t_b, "Δt_s": Δt_s, "C_b": C_b,
            "truth_ω": truth_ω, "truth_ω_mag": truth_ω_mag,
            "qb_truth_in_cloud": qb_truth_in_cloud,
            "qb_dist_truth": qb_dist_truth,
        }
        print(f"  Δ={Δ:3d}: t_b={t_b}, |C_b|={len(C_b):4d}, truth-ω={truth_ω_mag*180/np.pi:.4f} dps, "
              f"qb-truth-survivor at {qb_dist_truth:.2f}°")

    # Per-q_a analysis
    per_qa = []
    for qa_rank in range(n_a):
        qa = C_a_sorted[qa_rank]

        # OPERATIONAL: |ω|-prior-filtered centroid per Δ
        op_centroids = []
        op_n_pass = []
        for Δ in DELTAS:
            d = per_delta[Δ]
            C_b = d["C_b"]
            target = d["truth_ω_mag"]
            qa_arr = np.tile(qa[None, :], (len(C_b), 1))
            ω_set = fd_omega_passive(qa_arr, C_b, d["Δt_s"])
            ω_mag = np.linalg.norm(ω_set, axis=1)
            mask = (ω_mag >= target * PRIOR_BRACKET[0]) & \
                   (ω_mag <= target * PRIOR_BRACKET[1])
            n_pass = int(mask.sum())
            op_n_pass.append(n_pass)
            if n_pass == 0:
                op_centroids.append(None)
                continue
            ω_pass = ω_set[mask]
            ω_hat = ω_pass / np.linalg.norm(ω_pass, axis=1, keepdims=True)
            # Antipodal fold to first sample
            flip = (ω_hat @ ω_hat[0]) < 0
            ω_hat_f = np.where(flip[:, None], -ω_hat, ω_hat)
            cent = ω_hat_f.mean(axis=0)
            cent /= np.linalg.norm(cent)
            op_centroids.append(cent)

        # Across-Δ consistency of operational centroids
        valid_op = [c for c in op_centroids if c is not None]
        if len(valid_op) >= 2:
            stack = np.array(valid_op)
            flip = (stack @ stack[0]) < 0
            stack_f = np.where(flip[:, None], -stack, stack)
            mean_dir = stack_f.mean(axis=0)
            mean_dir /= np.linalg.norm(mean_dir)
            spreads = np.degrees(np.arccos(np.clip(np.abs(stack_f @ mean_dir), 0, 1)))
            op_spread_deg = float(spreads.mean())
            op_dist_to_truth = float(angular_dist_axes(mean_dir[None, :],
                                                       per_delta[DELTAS[0]]["truth_ω"])[0])
        else:
            op_spread_deg = float("nan")
            op_dist_to_truth = float("nan")

        # ORACLE: ω(q_a, truth-q_b-survivor) per Δ
        oracle_oms = []
        for Δ in DELTAS:
            d = per_delta[Δ]
            if d["qb_truth_in_cloud"] is None:
                oracle_oms.append(None)
                continue
            ω = fd_omega_passive(qa[None, :], d["qb_truth_in_cloud"][None, :],
                                  d["Δt_s"])[0]
            ω_hat = ω / np.linalg.norm(ω)
            oracle_oms.append(ω_hat)
        valid_or = [c for c in oracle_oms if c is not None]
        if len(valid_or) >= 2:
            stack = np.array(valid_or)
            flip = (stack @ stack[0]) < 0
            stack_f = np.where(flip[:, None], -stack, stack)
            mean_dir = stack_f.mean(axis=0)
            mean_dir /= np.linalg.norm(mean_dir)
            spreads = np.degrees(np.arccos(np.clip(np.abs(stack_f @ mean_dir), 0, 1)))
            or_spread_deg = float(spreads.mean())
            or_dist_to_truth = float(angular_dist_axes(mean_dir[None, :],
                                                       per_delta[DELTAS[0]]["truth_ω"])[0])
        else:
            or_spread_deg = float("nan")
            or_dist_to_truth = float("nan")

        per_qa.append({
            "qa_rank": qa_rank,
            "qa_dist_to_truth_deg": float(deg_to_truth_a[qa_rank]),
            "op_n_pass_per_delta": op_n_pass,
            "op_spread_deg": op_spread_deg,
            "op_dist_to_truth_omega_deg": op_dist_to_truth,
            "oracle_spread_deg": or_spread_deg,
            "oracle_dist_to_truth_omega_deg": or_dist_to_truth,
        })

    # Rank q_a's by operational consistency (smaller spread = better)
    op_valid = [r for r in per_qa if not np.isnan(r["op_spread_deg"])]
    op_valid.sort(key=lambda r: r["op_spread_deg"])
    truth_qa = next(r for r in op_valid if r["qa_rank"] == 0)
    truth_op_rank = op_valid.index(truth_qa)
    print(f"\n=== OPERATIONAL: |ω|-prior-filtered centroid consistency ===")
    print(f"Truth-q_a (qa_rank=0, dist {truth_qa['qa_dist_to_truth_deg']:.2f}°):")
    print(f"  spread = {truth_qa['op_spread_deg']:.2f}°")
    print(f"  centroid → truth-ω = {truth_qa['op_dist_to_truth_omega_deg']:.2f}°")
    print(f"  ranked {truth_op_rank+1} of {len(op_valid)} valid q_a's by consistency")
    # show top-5
    print(f"\n  Top 5 by consistency:")
    for r in op_valid[:5]:
        print(f"    qa_rank={r['qa_rank']:2d} (dist={r['qa_dist_to_truth_deg']:5.2f}°)  "
              f"spread={r['op_spread_deg']:5.2f}°  cent→truth={r['op_dist_to_truth_omega_deg']:5.2f}°")

    or_valid = [r for r in per_qa if not np.isnan(r["oracle_spread_deg"])]
    or_valid.sort(key=lambda r: r["oracle_spread_deg"])
    truth_qa_or = next(r for r in or_valid if r["qa_rank"] == 0)
    truth_or_rank = or_valid.index(truth_qa_or)
    print(f"\n=== ORACLE: (q_a, truth-q_b-survivor) per Δ ===")
    print(f"Truth-q_a (qa_rank=0):")
    print(f"  spread = {truth_qa_or['oracle_spread_deg']:.2f}°")
    print(f"  centroid → truth-ω = {truth_qa_or['oracle_dist_to_truth_omega_deg']:.2f}°")
    print(f"  ranked {truth_or_rank+1} of {len(or_valid)} valid q_a's by oracle consistency")
    print(f"\n  Top 5 by oracle consistency:")
    for r in or_valid[:5]:
        print(f"    qa_rank={r['qa_rank']:2d} (dist={r['qa_dist_to_truth_deg']:5.2f}°)  "
              f"oracle_spread={r['oracle_spread_deg']:5.2f}°  cent→truth={r['oracle_dist_to_truth_omega_deg']:5.2f}°")

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) qa_rank (closeness-to-truth) vs operational spread
    ax = axes[0, 0]
    ranks = [r["qa_rank"] for r in per_qa]
    op_spr = [r["op_spread_deg"] for r in per_qa]
    or_spr = [r["oracle_spread_deg"] for r in per_qa]
    qa_dists = [r["qa_dist_to_truth_deg"] for r in per_qa]
    ax.scatter(qa_dists, op_spr, color="b", label="OPERATIONAL")
    ax.scatter(qa_dists, or_spr, color="g", marker="^", label="ORACLE (truth-q_b)")
    ax.set_xlabel("q_a distance to truth at t_a (deg)")
    ax.set_ylabel("across-Δ ω-direction spread (deg)")
    ax.set_title(f"per-q_a multi-Δ consistency: qa-truth distance vs spread\n"
                 f"truth-q_a (qa_rank=0, leftmost) should give SMALLEST spread")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    # mark truth-q_a
    truth_op = truth_qa["op_spread_deg"]
    truth_or = truth_qa_or["oracle_spread_deg"]
    ax.scatter([truth_qa["qa_dist_to_truth_deg"]], [truth_op],
               color="red", s=200, marker="*", edgecolor="black", zorder=10,
               label="truth-q_a (op)")
    ax.scatter([truth_qa_or["qa_dist_to_truth_deg"]], [truth_or],
               color="red", s=200, marker="X", edgecolor="black", zorder=10,
               label="truth-q_a (oracle)")
    ax.legend(loc="best", fontsize=9)

    # (b) qa_dist vs centroid-to-truth-ω
    ax = axes[0, 1]
    op_d = [r["op_dist_to_truth_omega_deg"] for r in per_qa]
    or_d = [r["oracle_dist_to_truth_omega_deg"] for r in per_qa]
    ax.scatter(qa_dists, op_d, color="b", label="OPERATIONAL")
    ax.scatter(qa_dists, or_d, color="g", marker="^", label="ORACLE")
    ax.scatter([truth_qa["qa_dist_to_truth_deg"]], [truth_qa["op_dist_to_truth_omega_deg"]],
               color="red", s=200, marker="*", edgecolor="black", zorder=10)
    ax.scatter([truth_qa_or["qa_dist_to_truth_deg"]], [truth_qa_or["oracle_dist_to_truth_omega_deg"]],
               color="red", s=200, marker="X", edgecolor="black", zorder=10)
    ax.set_xlabel("q_a distance to truth at t_a (deg)")
    ax.set_ylabel("centroid distance to truth-ω̂ (deg)")
    ax.set_title("per-q_a: centroid alignment with truth-ω̂")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # (c) ranking-by-consistency: where does truth-q_a sit?
    ax = axes[1, 0]
    sorted_op = sorted(op_valid, key=lambda r: r["op_spread_deg"])
    op_spread_sorted = [r["op_spread_deg"] for r in sorted_op]
    op_qadist_sorted = [r["qa_dist_to_truth_deg"] for r in sorted_op]
    sc = ax.scatter(range(len(sorted_op)), op_spread_sorted,
                    c=op_qadist_sorted, cmap="viridis_r", s=40)
    plt.colorbar(sc, ax=ax, label="qa→truth at t_a (deg)")
    ax.scatter([truth_op_rank], [truth_op],
               color="red", s=300, marker="*", edgecolor="black", zorder=10,
               label=f"truth-q_a (rank {truth_op_rank+1}/{len(op_valid)})")
    ax.set_xlabel("rank by operational consistency")
    ax.set_ylabel("across-Δ spread (deg)")
    ax.set_title("OPERATIONAL ranking: smaller spread = better")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    sorted_or = sorted(or_valid, key=lambda r: r["oracle_spread_deg"])
    or_spread_sorted = [r["oracle_spread_deg"] for r in sorted_or]
    or_qadist_sorted = [r["qa_dist_to_truth_deg"] for r in sorted_or]
    sc = ax.scatter(range(len(sorted_or)), or_spread_sorted,
                    c=or_qadist_sorted, cmap="viridis_r", s=40)
    plt.colorbar(sc, ax=ax, label="qa→truth at t_a (deg)")
    ax.scatter([truth_or_rank], [truth_or],
               color="red", s=300, marker="X", edgecolor="black", zorder=10,
               label=f"truth-q_a (rank {truth_or_rank+1}/{len(or_valid)})")
    ax.set_xlabel("rank by oracle consistency")
    ax.set_ylabel("across-Δ spread (deg)")
    ax.set_title("ORACLE ranking (q_a paired with truth-q_b at each Δ)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_p = OUT / "per_qa_consistency.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_p}")

    summary = {
        "seed": 89,
        "T_A": T_A,
        "n_C_a": n_a,
        "deltas": DELTAS,
        "prior_bracket": list(PRIOR_BRACKET),
        "operational": {
            "truth_qa_rank": truth_op_rank + 1,
            "truth_qa_total": len(op_valid),
            "truth_qa_spread_deg": truth_qa["op_spread_deg"],
            "truth_qa_centroid_to_truth_omega_deg": truth_qa["op_dist_to_truth_omega_deg"],
        },
        "oracle": {
            "truth_qa_rank": truth_or_rank + 1,
            "truth_qa_total": len(or_valid),
            "truth_qa_spread_deg": truth_qa_or["oracle_spread_deg"],
            "truth_qa_centroid_to_truth_omega_deg": truth_qa_or["oracle_dist_to_truth_omega_deg"],
        },
        "per_qa": [
            {k: v for k, v in r.items() if k != "op_n_pass_per_delta"}
            | {"op_n_pass_per_delta": r["op_n_pass_per_delta"]}
            for r in per_qa
        ],
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"Saved: {OUT / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
