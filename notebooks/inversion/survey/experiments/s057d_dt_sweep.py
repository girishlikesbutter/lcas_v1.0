"""s057d — Δt sweep at single anchor t_a=411 on seed 89.

Critique of s057: Δt=10 (72s) was chosen by SNR analytic, not empirical
sweep. For seed 89 with cone_max=155° and polhode period ~ rotation period
(25 min), ω̂(t) drifts ~7° over 72s — comparable to pool noise. Smaller
Δt may give better SNR despite reduced |ω|·Δt geometric signal.

Sweep Δt ∈ {1, 2, 3, 5, 7, 10, 15, 20} at fixed t_a=411 (deepest |C_t|=44).
For each Δt:
  - check closest-in-cloud at t_b
  - all-pair finite-diff ω
  - top K×K=25 truth-adjacent pair: min/median ang error to truth-ω
  - ±25% |ω|-prior survival counts
  - concentration f<10° vs uniform baseline
  - MODAL cluster: find densest 10°-radius patch on the sphere among
    survivors; report angular distance of patch centroid to truth-ω̂

Identifies the best Δt empirically, before any multi-anchor experiment.
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
OUT = SURVEY / "results" / "s057d_dt_sweep"
OUT.mkdir(parents=True, exist_ok=True)

T_A = 411
DELTAS = [1, 2, 3, 5, 7, 10, 15, 20]
K_TRUTH = 5  # top-K nearest survivors per side for truth-adjacent stats
PRIOR_BRACKET = (0.75, 1.25)  # ±25%, matches s057's loosest setting


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def fd_omega_passive(q_a, q_b, dt):
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    R_b = Rotation.from_quat(wxyz_to_xyzw(q_b))
    return (R_b * R_a.inv()).as_rotvec() / dt


def angular_dist(om_3vec, om_truth):
    om_norm = np.linalg.norm(om_3vec, axis=-1, keepdims=True)
    safe = np.where(om_norm > 1e-12, om_norm, 1.0)
    om_hat = om_3vec / safe
    t_hat = om_truth / np.linalg.norm(om_truth)
    cos_a = np.abs(np.einsum("...i,i->...", om_hat, t_hat))
    return np.degrees(np.arccos(np.clip(cos_a, 0, 1)))


def find_modal_cluster(om_3vec, patch_radius_deg=10.0):
    """Densest unit-sphere patch by greedy cap-cover. Returns (centroid_hat, count_in_patch).

    Antipodal-folded: each ω treated as ±ω. Approximation: pick the survivor
    with the most other survivors within `patch_radius_deg` (axial distance);
    that's the patch centre.
    """
    om_norm = np.linalg.norm(om_3vec, axis=1, keepdims=True)
    om_hat = om_3vec / np.where(om_norm > 1e-12, om_norm, 1.0)
    # |dot| matrix
    M = np.abs(om_hat @ om_hat.T)
    cos_thresh = np.cos(np.radians(patch_radius_deg))
    in_patch = (M >= cos_thresh).astype(np.int32)
    counts = in_patch.sum(axis=1)
    best = int(np.argmax(counts))
    centre = om_hat[best]
    # refine: take mean of all in-patch members (axial-aware: flip if needed)
    members_mask = in_patch[best].astype(bool)
    members = om_hat[members_mask]
    flip = (members @ centre) < 0
    members = np.where(flip[:, None], -members, members)
    centroid = members.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return centroid, int(counts[best])


def main() -> dict:
    z = np.load(DENSE_RUN)
    survive_all = z["survive_all"]
    q_pool = z["q_pool_wxyz"]
    obs_times = z["obs_times"]

    traj = np.load(TRAJ089)
    q_truth_t = traj["quaternions"]

    closest_in_cloud_data = np.load(SURVEY / "results/s057b_anchor_scan/closest_in_cloud.npz")
    closest_in_cloud = closest_in_cloud_data["closest_in_cloud"]

    dt_epoch = float(np.median(np.diff(obs_times)))
    n_epochs = len(survive_all)

    idx_a = np.where(survive_all[T_A])[0]
    C_a = q_pool[idx_a]
    n_a = len(idx_a)
    print(f"anchor t_a={T_A}, |C_a|={n_a}, closest-in-cloud={closest_in_cloud[T_A]:.2f}°")

    # top-K nearest survivors to truth on a-side (computed once)
    dots_a = np.abs(C_a @ q_truth_t[T_A])
    order_a = np.argsort(dots_a)[::-1][:K_TRUTH]
    deg_a_topk = np.degrees(2 * np.arccos(np.clip(dots_a[order_a], 0, 1)))

    rows = []
    for Δ in DELTAS:
        t_b = T_A + Δ
        if t_b >= n_epochs:
            continue
        idx_b = np.where(survive_all[t_b])[0]
        if len(idx_b) == 0:
            rows.append({"delta_epochs": Δ, "skipped": "empty cloud"})
            continue
        C_b = q_pool[idx_b]
        n_b = len(idx_b)
        Δt_s = Δ * dt_epoch

        # truth ω at this Δt: finite-diff cached q(t_a)→q(t_b)
        om_truth = fd_omega_passive(q_truth_t[T_A:T_A+1],
                                    q_truth_t[t_b:t_b+1], Δt_s)[0]
        om_truth_mag = float(np.linalg.norm(om_truth))
        om_truth_hat = om_truth / om_truth_mag
        om_truth_dps = om_truth_mag * 180 / np.pi

        # truth-adjacent K×K pairs
        dots_b = np.abs(C_b @ q_truth_t[t_b])
        order_b = np.argsort(dots_b)[::-1][:K_TRUTH]
        deg_b_topk = np.degrees(2 * np.arccos(np.clip(dots_b[order_b], 0, 1)))
        Q_A = np.repeat(C_a[order_a], K_TRUTH, axis=0)
        Q_B = np.tile(C_b[order_b], (K_TRUTH, 1))
        om_kk = fd_omega_passive(Q_A, Q_B, Δt_s)
        om_kk_mag = np.linalg.norm(om_kk, axis=1)
        ang_kk = angular_dist(om_kk, om_truth)
        mag_err_kk_pct = (om_kk_mag - om_truth_mag) / om_truth_mag * 100

        # all-pair (subsample if huge: cap at 50k pairs to keep compute bounded)
        max_pairs = 50000
        n_total = n_a * n_b
        if n_total > max_pairs:
            rng = np.random.default_rng(42)
            sub_a = rng.integers(0, n_a, size=max_pairs)
            sub_b = rng.integers(0, n_b, size=max_pairs)
            QA_all = C_a[sub_a]
            QB_all = C_b[sub_b]
        else:
            QA_all = np.repeat(C_a, n_b, axis=0)
            QB_all = np.tile(C_b, (n_a, 1))
        om_all = fd_omega_passive(QA_all, QB_all, Δt_s)
        om_all_mag = np.linalg.norm(om_all, axis=1)

        # |ω|-prior at ±25%
        target = om_truth_mag
        pass_mask = (om_all_mag >= target * PRIOR_BRACKET[0]) & \
                    (om_all_mag <= target * PRIOR_BRACKET[1])
        n_pass = int(pass_mask.sum())
        if n_pass < 2:
            rows.append({
                "delta_epochs": Δ, "delta_t_s": Δt_s,
                "n_C_b": n_b, "closest_in_cloud_b_deg": float(closest_in_cloud[t_b]),
                "om_truth_dps": om_truth_dps,
                "kk_min_ang_deg": float(ang_kk.min()),
                "kk_median_ang_deg": float(np.median(ang_kk)),
                "kk_pass_25pct_brkt": int(((om_kk_mag >= target * 0.75) &
                                            (om_kk_mag <= target * 1.25)).sum()),
                "n_pass_total": n_pass,
                "skipped": "too few survivors after prior",
            })
            continue
        ang_pass = angular_dist(om_all[pass_mask], om_truth)
        baseline_10 = 1.0 - np.cos(np.radians(10.0))
        baseline_30 = 1.0 - np.cos(np.radians(30.0))
        frac_10 = float((ang_pass < 10).mean())
        frac_30 = float((ang_pass < 30).mean())
        ratio_10 = frac_10 / baseline_10 if baseline_10 > 0 else float("nan")
        ratio_30 = frac_30 / baseline_30 if baseline_30 > 0 else float("nan")

        # modal cluster on the survived-prior set
        if n_pass <= 5000:
            centroid, count = find_modal_cluster(om_all[pass_mask], 10.0)
            modal_dist = float(angular_dist(centroid[None, :], om_truth)[0])
            modal_count = count
        else:
            # subsample for tractability
            rng = np.random.default_rng(0)
            sub = rng.choice(n_pass, 5000, replace=False)
            centroid, count = find_modal_cluster(om_all[pass_mask][sub], 10.0)
            modal_dist = float(angular_dist(centroid[None, :], om_truth)[0])
            modal_count = int(count * n_pass / 5000)

        rows.append({
            "delta_epochs": Δ,
            "delta_t_s": Δt_s,
            "n_C_b": n_b,
            "closest_in_cloud_b_deg": float(closest_in_cloud[t_b]),
            "om_truth_dps": om_truth_dps,
            "om_x_Δt_deg": om_truth_mag * Δt_s * 180 / np.pi,
            "kk_min_ang_deg": float(ang_kk.min()),
            "kk_median_ang_deg": float(np.median(ang_kk)),
            "kk_pass_25pct_brkt": int(((om_kk_mag >= target * 0.75) &
                                        (om_kk_mag <= target * 1.25)).sum()),
            "n_pass_total": n_pass,
            "frac_pass_within_10deg_pct": frac_10 * 100,
            "frac_pass_within_30deg_pct": frac_30 * 100,
            "ratio_10deg_vs_baseline": ratio_10,
            "ratio_30deg_vs_baseline": ratio_30,
            "modal_cluster_dist_to_truth_deg": modal_dist,
            "modal_cluster_count": modal_count,
        })

    # print table
    print()
    print(f"{'Δt':>4} | {'s':>5} | {'|C_b|':>5} | {'cinC_b°':>7} | {'|ω|·Δt°':>7} | "
          f"{'kk_min°':>7} | {'kk_med°':>7} | {'pass':>5} | {'×base@10°':>9} | {'modal°':>7}")
    print("-" * 100)
    for r in rows:
        if "skipped" in r:
            print(f"{r['delta_epochs']:>4} | SKIPPED: {r['skipped']}")
            continue
        print(f"{r['delta_epochs']:>4} | {r['delta_t_s']:>5.1f} | "
              f"{r['n_C_b']:>5d} | {r['closest_in_cloud_b_deg']:>7.2f} | "
              f"{r['om_x_Δt_deg']:>7.2f} | "
              f"{r['kk_min_ang_deg']:>7.2f} | {r['kk_median_ang_deg']:>7.2f} | "
              f"{r['n_pass_total']:>5d} | {r['ratio_10deg_vs_baseline']:>9.2f} | "
              f"{r['modal_cluster_dist_to_truth_deg']:>7.2f}")

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    Δs_ok = [r["delta_epochs"] for r in rows if "skipped" not in r]
    om_x_Δt = [r["om_x_Δt_deg"] for r in rows if "skipped" not in r]
    kk_min = [r["kk_min_ang_deg"] for r in rows if "skipped" not in r]
    kk_med = [r["kk_median_ang_deg"] for r in rows if "skipped" not in r]
    ratio_10 = [r["ratio_10deg_vs_baseline"] for r in rows if "skipped" not in r]
    modal = [r["modal_cluster_dist_to_truth_deg"] for r in rows if "skipped" not in r]
    pass_n = [r["n_pass_total"] for r in rows if "skipped" not in r]

    ax = axes[0, 0]
    ax.plot(Δs_ok, kk_min, "go-", label="K×K best (min)")
    ax.plot(Δs_ok, kk_med, "ro-", label="K×K median")
    ax.set_xlabel("Δt (epochs)")
    ax.set_ylabel("ω-axis err to truth (deg)")
    ax.set_title("truth-adjacent K×K=25 pairs: ω-direction error vs Δt")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(Δs_ok, om_x_Δt, "b-", label="|ω|·Δt (deg)")
    ax2 = ax.twinx()
    ax2.plot(Δs_ok, [r["closest_in_cloud_b_deg"] for r in rows if "skipped" not in r],
             "r--", label="closest-in-cloud at t_b")
    ax.set_xlabel("Δt (epochs)")
    ax.set_ylabel("|ω|·Δt geometric signal (deg)", color="b")
    ax2.set_ylabel("pool→truth at t_b (deg)", color="r")
    ax.set_title("geometric SNR: signal (|ω|·Δt) vs noise (pool→truth)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(Δs_ok, ratio_10, "ko-", label="frac<10° / baseline")
    ax.axhline(1.0, ls=":", color="grey", label="random uniform")
    ax.set_xlabel("Δt (epochs)")
    ax.set_ylabel("concentration ratio at 10° vs baseline")
    ax.set_title("|ω|-prior ±25% concentration vs Δt")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(Δs_ok, modal, "mo-", label="modal cluster dist to truth")
    ax.set_xlabel("Δt (epochs)")
    ax.set_ylabel("modal cluster centroid → truth-ω̂ (deg)")
    ax.set_title("modal-cluster diagnostic: where does the densest patch sit?")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_p = OUT / "dt_sweep.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_p}")

    out_p = OUT / "summary.json"
    with open(out_p, "w") as f:
        json.dump({"t_a": T_A, "rows": rows}, f, indent=2)
    print(f"Saved: {out_p}")
    return {"t_a": T_A, "rows": rows}


if __name__ == "__main__":
    main()
