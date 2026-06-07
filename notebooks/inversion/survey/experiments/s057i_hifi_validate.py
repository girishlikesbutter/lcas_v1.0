"""s057i — hi-fi LC validation of top forward-prop candidate clusters.

Architecture: s057g identified 548 (q_a, ω) candidates at t_a=411 with
strong forward-propagation discrimination (4400× over null). s057h
clustered them into 325 groups; truth cluster ranks 107/325 by sum-score.
Open question: are the top-scoring clusters GENUINE multi-solution
alternates (Band A∪B per ρ-band gating) or false positives from
const-ω validator looseness?

This script:
1. Regenerates candidate scoring (same as s057g/h)
2. Identifies top-K clusters by sum-score + the truth cluster
3. For each cluster representative (top-scoring member), back-propagates
   (q_a, ω_a) at t_a=411 to (q_0, ω_0) at t=0 via Euler-integrated
   reverse dynamics
4. Renders hi-fi LC via lib.hifi_render
5. Computes ρ vs cached truth LC
6. Reports per-cluster Band A/B/C/D classification

Truth cluster's representative SHOULD be Band A (small ρ — by construction
truth-q_a is 2.43° from truth, polish-amenable). Top-scoring non-truth
clusters: outcome is the answer to the architecture's operational value.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

from lib.twin import canonical_batch
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

DENSE_RUN = SURVEY / "results" / "s048c_cloud_viewer" / "seed089" / "8bb9b81f1602" / "spread.npz"
TRAJ089 = SURVEY / "data" / "trajectories" / "traj_seed089.npz"
OUT = SURVEY / "results" / "s057i_hifi_validate"
OUT.mkdir(parents=True, exist_ok=True)

T_A = 411
DELTA_GEN = 15
PRIOR_BRACKET = (0.75, 1.25)
HIT_THRESHOLD_DEG = 5.0
CONST_OMEGA_RELIABLE_MAX_DEG = 20.0
SMOKE_DELTAS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80]
CLUSTER_Q_DEG = 8.0
CLUSTER_OM_DEG = 15.0
CLUSTER_OM_MAG_PCT = 25.0
TOP_K_CLUSTERS = 15


def wxyz_to_xyzw(q): return q[..., [1, 2, 3, 0]]
def xyzw_to_wxyz(q): return q[..., [3, 0, 1, 2]]


def fd_omega_passive(q_a, q_b, dt):
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    R_b = Rotation.from_quat(wxyz_to_xyzw(q_b))
    return (R_b * R_a.inv()).as_rotvec() / dt


def propagate_const_omega(q_a, omega, dt):
    rotvec = omega * dt
    R_om = Rotation.from_rotvec(rotvec)
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    return xyzw_to_wxyz((R_om * R_a).as_quat())


def quat_ang_deg_batch(q_arr, q_ref):
    dots = np.abs(q_arr @ q_ref)
    return 2 * np.degrees(np.arccos(np.clip(dots, 0, 1)))


def ang_to_axis(om_arr, ref):
    om_norm = np.linalg.norm(om_arr, axis=-1, keepdims=True)
    safe = np.where(om_norm > 1e-12, om_norm, 1.0)
    om_hat = om_arr / safe
    r_hat = ref / np.linalg.norm(ref)
    cos_a = np.abs(np.einsum("...i,i->...", om_hat, r_hat))
    return np.degrees(np.arccos(np.clip(cos_a, 0, 1)))


def back_propagate(q_a_wxyz, omega_a_rad, t_back_s, inertia_tensor):
    """Back-propagate (q_a, ω_a) at t=t_back_s to (q_0, ω_0) at t=0.

    Uses time-reversal symmetry: (q(-t), -ω(-t)) is also a solution of the
    free-rigid-body equations. So forward-integrate (q_a, -ω_a) over
    [0, t_back_s]; resulting (q', ω') has q'=q_0, -ω'=ω_0.
    """
    from src.dynamics.attitude_propagator import propagate_attitude
    times = np.array([0.0, t_back_s])
    quats, omegas = propagate_attitude(
        q0=q_a_wxyz, omega0=-omega_a_rad, times=times,
        mode="tumbling", inertia_tensor=inertia_tensor,
    )
    q_0 = quats[-1]
    omega_0 = -omegas[-1]
    return q_0, omega_0


def main() -> dict:
    t_overall = time.time()

    z = np.load(DENSE_RUN)
    survive_all = z["survive_all"]
    q_pool = z["q_pool_wxyz"]
    obs_times = z["obs_times"]

    traj = np.load(TRAJ089)
    q_truth_t = traj["quaternions"]
    dt_epoch = float(np.median(np.diff(obs_times)))
    t_a_seconds = float(obs_times[T_A] - obs_times[0])
    print(f"t_a={T_A}, t_a_seconds={t_a_seconds:.2f}s")

    # === Build hifi context (loads satellite + inertia + per-seed SPICE) ===
    print("\n=== building hifi context ===")
    t0 = time.time()
    ctx = build_context(seed=89)
    print(f"  built in {time.time()-t0:.1f}s")

    # === Smoke test: round-trip truth ===
    print("\n=== smoke test: render truth (q0, ω0) ===")
    t0 = time.time()
    pred_truth = render_hifi(ctx["q0_truth"], ctx["omega0_truth_rad"], ctx)
    rho_truth = rho_from_hifi(pred_truth, ctx["mag_hifi_truth"])
    print(f"  ρ_truth(self) = {rho_truth:.6f}  ({time.time()-t0:.1f}s)")
    assert rho_truth < 1e-6, "truth round-trip should be machine-precision exact"

    # === Smoke test: back-propagate truth ===
    print("\n=== smoke test: back-propagate truth-(q_a, ω_a) → (q_0, ω_0) ===")
    om_truth_at_ta_inst = fd_omega_passive(
        q_truth_t[T_A:T_A+1], q_truth_t[T_A+1:T_A+2], dt_epoch
    )[0]
    q0_back, om0_back = back_propagate(
        q_truth_t[T_A], om_truth_at_ta_inst, t_a_seconds, ctx["inertia_tensor"]
    )
    q0_err = float(2 * np.degrees(np.arccos(
        np.clip(abs(np.dot(q0_back, ctx["q0_truth"])), 0, 1))))
    om0_err = float(np.linalg.norm(om0_back - ctx["omega0_truth_rad"]) /
                    np.linalg.norm(ctx["omega0_truth_rad"]) * 100)
    print(f"  truth back-prop q0_err = {q0_err:.4f}°, ω0_err = {om0_err:.3f}%")
    pred_back = render_hifi(q0_back, om0_back, ctx)
    rho_back = rho_from_hifi(pred_back, ctx["mag_hifi_truth"])
    print(f"  ρ from back-propagated truth = {rho_back:.4f}")

    # === Regenerate candidates (s057g/h logic) ===
    print("\n=== regenerating candidates ===")
    om_inst = fd_omega_passive(q_truth_t[T_A:T_A+1],
                                q_truth_t[T_A+1:T_A+2], dt_epoch)[0]
    valid_max_delta = 1
    for Δ in SMOKE_DELTAS:
        if T_A + Δ >= len(q_truth_t):
            continue
        q_pred = propagate_const_omega(q_truth_t[T_A], om_inst, Δ * dt_epoch)
        err = float(2 * np.degrees(np.arccos(
            np.clip(abs(np.dot(q_pred, q_truth_t[T_A + Δ])), 0, 1))))
        if err <= CONST_OMEGA_RELIABLE_MAX_DEG:
            valid_max_delta = Δ
        else:
            break

    validators = []
    lo_v = max(0, T_A - valid_max_delta)
    hi_v = min(len(survive_all), T_A + valid_max_delta + 1)
    for t_v in range(lo_v, hi_v):
        if t_v == T_A: continue
        idx_v = np.where(survive_all[t_v])[0]
        if len(idx_v) == 0: continue
        validators.append({
            "delta": t_v - T_A, "C_v": q_pool[idx_v],
            "n_surv": int(len(idx_v)),
            "weight": float(np.log(100000.0 / len(idx_v))),
        })

    idx_a = np.where(survive_all[T_A])[0]
    C_a = q_pool[idx_a]
    n_a = len(C_a)
    Δgen_t_b = T_A + DELTA_GEN
    idx_b = np.where(survive_all[Δgen_t_b])[0]
    C_b = q_pool[idx_b]
    n_b = len(C_b)
    Δt_gen = DELTA_GEN * dt_epoch

    om_truth_gen = fd_omega_passive(q_truth_t[T_A:T_A+1],
                                     q_truth_t[Δgen_t_b:Δgen_t_b+1], Δt_gen)[0]
    om_truth_mag = float(np.linalg.norm(om_truth_gen))

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

    # Score candidates
    scores = np.zeros(n_cand)
    for v in validators:
        Δ = v["delta"]
        Δt = Δ * dt_epoch
        q_pred = propagate_const_omega(Q_A_pass, om_pass, Δt)
        dots = np.abs(q_pred @ v["C_v"].T)
        max_dot = dots.max(axis=1)
        min_ang = 2 * np.degrees(np.arccos(np.clip(max_dot, 0, 1)))
        scores += v["weight"] * (min_ang < HIT_THRESHOLD_DEG)

    qa_dist_orig = quat_ang_deg_batch(Q_A_pass, q_truth_t[T_A])
    om_dist_orig = ang_to_axis(om_pass, om_truth_gen)
    truth_idx = int(np.argmin(qa_dist_orig + om_dist_orig))

    # Canonicalise + cluster
    Q_A_canon, om_canon = canonical_batch(Q_A_pass, om_pass)
    om_mag_canon = np.linalg.norm(om_canon, axis=1)
    order = np.argsort(-scores)
    assigned = np.full(n_cand, -1, dtype=int)
    clusters = []
    for seed_i in order:
        if assigned[seed_i] != -1: continue
        unassigned = np.where(assigned == -1)[0]
        d_q = quat_ang_deg_batch(Q_A_canon[unassigned], Q_A_canon[seed_i])
        d_om = ang_to_axis(om_canon[unassigned], om_canon[seed_i])
        d_om_mag_pct = np.abs(om_mag_canon[unassigned] - om_mag_canon[seed_i]) / \
                       om_mag_canon[seed_i] * 100
        in_cluster = (d_q < CLUSTER_Q_DEG) & (d_om < CLUSTER_OM_DEG) & \
                     (d_om_mag_pct < CLUSTER_OM_MAG_PCT)
        members = unassigned[in_cluster]
        cluster_id = len(clusters)
        assigned[members] = cluster_id
        clusters.append({
            "cluster_id": cluster_id,
            "n_members": int(len(members)),
            "members": members.tolist(),
            "seed_idx": int(seed_i),
            "score_sum": float(scores[members].sum()),
            "score_max": float(scores[members].max()),
            "best_member_idx": int(members[np.argmax(scores[members])]),
        })
    clusters_sorted = sorted(clusters, key=lambda c: -c["score_sum"])
    truth_cluster_id = int(assigned[truth_idx])
    truth_rank = next(i for i, c in enumerate(clusters_sorted)
                      if c["cluster_id"] == truth_cluster_id)
    print(f"  {n_cand} candidates → {len(clusters)} clusters; "
          f"truth cluster ranks {truth_rank+1}/{len(clusters)}")

    # === Render hi-fi for top-K clusters + truth cluster ===
    to_render = list(clusters_sorted[:TOP_K_CLUSTERS])
    truth_cluster_obj = clusters[truth_cluster_id]
    if truth_cluster_obj not in to_render:
        to_render.append(truth_cluster_obj)
    print(f"\n=== rendering hi-fi for {len(to_render)} clusters ===")

    rendered = []
    for ri, c in enumerate(to_render):
        cluster_rank = next(i for i, cc in enumerate(clusters_sorted)
                            if cc["cluster_id"] == c["cluster_id"])
        is_truth = c["cluster_id"] == truth_cluster_id
        bm = c["best_member_idx"]
        q_a = Q_A_pass[bm]
        om_a = om_pass[bm]
        # back-propagate to t=0
        try:
            t0 = time.time()
            q_0, om_0 = back_propagate(q_a, om_a, t_a_seconds, ctx["inertia_tensor"])
            backprop_s = time.time() - t0
            t0 = time.time()
            pred = render_hifi(q_0, om_0, ctx)
            render_s = time.time() - t0
            rho = rho_from_hifi(pred, ctx["mag_hifi_truth"])
            band = rho_band(rho)
        except Exception as e:
            rho = float("nan")
            band = "ERR"
            backprop_s = render_s = 0.0
            print(f"    ERROR rendering cluster_id={c['cluster_id']}: {e}")
            pred = np.full_like(ctx["mag_hifi_truth"], np.nan)
            q_0 = q_a; om_0 = om_a

        rendered.append({
            "render_order": ri,
            "cluster_rank": cluster_rank + 1,
            "cluster_id": c["cluster_id"],
            "is_truth_cluster": is_truth,
            "score_sum": c["score_sum"],
            "score_max": c["score_max"],
            "n_members": c["n_members"],
            "qa_dist_t_a_to_truth_deg": float(qa_dist_orig[bm]),
            "om_dist_at_t_a_to_truth_deg": float(om_dist_orig[bm]),
            "om_mag_dps_at_t_a": float(np.linalg.norm(om_a) * 180 / np.pi),
            "q0_back_wxyz": q_0.tolist(),
            "om0_back_rad": om_0.tolist(),
            "rho": float(rho),
            "band": band,
            "backprop_s": backprop_s,
            "render_s": render_s,
        })
        marker = "← TRUTH" if is_truth else ""
        print(f"  rank {cluster_rank+1:3d}: id={c['cluster_id']:3d}  score_sum={c['score_sum']:6.1f}  "
              f"qa_d={qa_dist_orig[bm]:5.2f}°  ω_d={om_dist_orig[bm]:5.2f}°  "
              f"|ω|={np.linalg.norm(om_a)*180/np.pi:.3f}dps  ρ={rho:6.2f} band={band}  {marker}")

    # === FIGURE ===
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) cluster rank vs ρ
    ax = axes[0, 0]
    ranks = [r["cluster_rank"] for r in rendered]
    rhos = [r["rho"] for r in rendered]
    bands = [r["band"] for r in rendered]
    color_by_band = {"A": "green", "B": "lime", "C": "orange", "D": "red", "ERR": "grey"}
    truth_marker = [r["is_truth_cluster"] for r in rendered]
    for r, ρ, b, t in zip(ranks, rhos, bands, truth_marker):
        ax.scatter([r], [ρ], color=color_by_band.get(b, "k"),
                   marker="*" if t else "o",
                   s=300 if t else 80,
                   edgecolor="black", linewidth=1.5 if t else 0.5,
                   zorder=10 if t else 5,
                   label=f"truth cluster (ρ={ρ:.2f}, {b})" if t else None)
    ax.axhline(2, ls="--", color="green", lw=0.8, label="A/B boundary (ρ=2)")
    ax.axhline(4, ls="--", color="orange", lw=0.8, label="B/C boundary (ρ=4)")
    ax.axhline(8, ls="--", color="red", lw=0.8, label="C/D boundary (ρ=8)")
    ax.set_xlabel("cluster rank by sum-score")
    ax.set_ylabel("ρ = √MSE / 0.05 vs truth LC")
    ax.set_title(f"hi-fi ρ for top {TOP_K_CLUSTERS} clusters + truth cluster")
    ax.set_yscale("symlog")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) ρ vs qa_dist
    ax = axes[0, 1]
    qadists = [r["qa_dist_t_a_to_truth_deg"] for r in rendered]
    for q, ρ, b, t in zip(qadists, rhos, bands, truth_marker):
        ax.scatter([q], [ρ], color=color_by_band.get(b, "k"),
                   marker="*" if t else "o",
                   s=300 if t else 80,
                   edgecolor="black", linewidth=1.5 if t else 0.5,
                   zorder=10 if t else 5)
    ax.axhline(2, ls="--", color="green", lw=0.8)
    ax.axhline(4, ls="--", color="orange", lw=0.8)
    ax.axhline(8, ls="--", color="red", lw=0.8)
    ax.set_xlabel("qa distance to truth at t_a (deg)")
    ax.set_ylabel("ρ")
    ax.set_yscale("symlog")
    ax.set_title("ρ vs qa-distance: cluster types")
    ax.grid(alpha=0.3)

    # (c) ρ vs ω_dist
    ax = axes[1, 0]
    omdists = [r["om_dist_at_t_a_to_truth_deg"] for r in rendered]
    for o, ρ, b, t in zip(omdists, rhos, bands, truth_marker):
        ax.scatter([o], [ρ], color=color_by_band.get(b, "k"),
                   marker="*" if t else "o",
                   s=300 if t else 80,
                   edgecolor="black", linewidth=1.5 if t else 0.5,
                   zorder=10 if t else 5)
    ax.axhline(2, ls="--", color="green", lw=0.8)
    ax.axhline(4, ls="--", color="orange", lw=0.8)
    ax.axhline(8, ls="--", color="red", lw=0.8)
    ax.set_xlabel("ω axis dist to truth-ω at t_a (deg)")
    ax.set_ylabel("ρ")
    ax.set_yscale("symlog")
    ax.set_title("ρ vs ω-direction distance")
    ax.grid(alpha=0.3)

    # (d) band counts bar
    ax = axes[1, 1]
    band_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "ERR": 0}
    for b in bands:
        band_counts[b] = band_counts.get(b, 0) + 1
    cs = ["green", "lime", "orange", "red", "grey"]
    bs = ["A", "B", "C", "D", "ERR"]
    counts = [band_counts[b] for b in bs]
    ax.bar(bs, counts, color=cs, edgecolor="black")
    ax.set_xlabel("ρ-band")
    ax.set_ylabel("# clusters")
    ax.set_title(f"band distribution among top {TOP_K_CLUSTERS} (+ truth) clusters")
    for i, v in enumerate(counts):
        if v > 0: ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig_p = OUT / "hifi_validate.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_p}")

    # Aggregate
    n_band_A = band_counts.get("A", 0)
    n_band_B = band_counts.get("B", 0)
    n_band_AB = n_band_A + n_band_B
    n_band_C = band_counts.get("C", 0)
    n_band_D = band_counts.get("D", 0)
    truth_render = next((r for r in rendered if r["is_truth_cluster"]), None)
    print(f"\n=== summary ===")
    print(f"  rendered: {len(rendered)} clusters total ({TOP_K_CLUSTERS} top + 1 truth)")
    print(f"  Band A:  {n_band_A}")
    print(f"  Band B:  {n_band_B}")
    print(f"  Band C:  {n_band_C}")
    print(f"  Band D:  {n_band_D}")
    print(f"  Band A∪B (acceptable): {n_band_AB}/{len(rendered)}")
    if truth_render:
        print(f"  truth cluster: ρ={truth_render['rho']:.2f}, band={truth_render['band']}")
    print(f"\nTotal wall: {time.time()-t_overall:.1f}s")

    summary = {
        "seed": 89,
        "T_A": T_A,
        "n_candidates": n_cand,
        "n_clusters": len(clusters),
        "truth_cluster_rank": truth_rank + 1,
        "smoke_test": {
            "rho_truth_self": float(rho_truth),
            "back_prop_q0_err_deg": q0_err,
            "back_prop_om0_err_pct": om0_err,
            "rho_back_propagated_truth": float(rho_back),
        },
        "rendered": rendered,
        "band_counts": band_counts,
        "n_band_AB_top_K": n_band_AB,
        "truth_cluster_render": truth_render,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"Saved: {OUT / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
