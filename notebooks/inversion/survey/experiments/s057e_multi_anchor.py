"""s057e — multi-anchor aggregation at Δt=3 on seed 89.

Δt=3 was the best concentration ratio in the s057d sweep (3.87× over
uniform baseline at f<10°), but the modal hypothesis cluster sits ~42°
from truth — single-anchor architecture has too much "geometric noise"
mode contamination. Multi-anchor hypothesis: different anchors produce
DIFFERENT geometric noise modes (anchor-specific pair sampling) while
truth-ω̂ stays consistent across nearby-in-time anchors (polhode drift
small over ~1 min for seed 89).

Strategy:
- Filter anchors at Δt=3 where closest-in-cloud < 5° at BOTH t_a and t_b
- Pick top-10 anchors by (cinC_a + cinC_b) sum
- Run all-pair architecture at each anchor with |ω|-prior ±25%
- Per-anchor: compute angular distance of each hypothesis to THAT
  anchor's truth-ω̂(t_a). Stack across anchors.
- Aggregate visual: plot all hypotheses on body-frame sphere with
  truth-ω̂(t_anchor) trajectory overlaid

Key diagnostic: does the aggregated angular-distance-to-truth distribution
peak more sharply at 0° than any single anchor's? If yes, multi-anchor
aggregation is the right architectural pivot.
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
CLOSEST = SURVEY / "results" / "s057b_anchor_scan" / "closest_in_cloud.npz"
OUT = SURVEY / "results" / "s057e_multi_anchor"
OUT.mkdir(parents=True, exist_ok=True)

DELTA_EPOCH = 3
N_ANCHORS = 10
CIC_THRESH_DEG = 5.0
PRIOR_BRACKET = (0.75, 1.25)


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def fd_omega_passive(q_a, q_b, dt):
    R_a = Rotation.from_quat(wxyz_to_xyzw(q_a))
    R_b = Rotation.from_quat(wxyz_to_xyzw(q_b))
    return (R_b * R_a.inv()).as_rotvec() / dt


def angular_dist(om_3vec, truth):
    om_norm = np.linalg.norm(om_3vec, axis=-1, keepdims=True)
    safe = np.where(om_norm > 1e-12, om_norm, 1.0)
    om_hat = om_3vec / safe
    t_hat = truth / np.linalg.norm(truth)
    cos_a = np.abs(np.einsum("...i,i->...", om_hat, t_hat))
    return np.degrees(np.arccos(np.clip(cos_a, 0, 1)))


def find_modal_cluster(om_hat, patch_radius_deg=10.0):
    """Densest sphere-cap on antipodal-folded directions."""
    M = np.abs(om_hat @ om_hat.T)
    cos_t = np.cos(np.radians(patch_radius_deg))
    counts = (M >= cos_t).sum(axis=1)
    best = int(np.argmax(counts))
    centre = om_hat[best]
    in_patch = M[best] >= cos_t
    members = om_hat[in_patch]
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

    closest_in_cloud = np.load(CLOSEST)["closest_in_cloud"]
    dt_epoch = float(np.median(np.diff(obs_times)))
    Δt_pair = DELTA_EPOCH * dt_epoch
    n_epochs = len(survive_all)

    # --- anchor selection ---
    candidates = []
    for t_a in range(n_epochs - DELTA_EPOCH):
        t_b = t_a + DELTA_EPOCH
        cic_a = closest_in_cloud[t_a]
        cic_b = closest_in_cloud[t_b]
        if np.isnan(cic_a) or np.isnan(cic_b):
            continue
        if cic_a >= CIC_THRESH_DEG or cic_b >= CIC_THRESH_DEG:
            continue
        candidates.append((t_a, t_b, cic_a, cic_b, cic_a + cic_b))
    print(f"candidates with cinC<{CIC_THRESH_DEG}° at both sides: {len(candidates)}")
    candidates.sort(key=lambda x: x[4])
    selected = candidates[:N_ANCHORS]
    print(f"top-{N_ANCHORS} by (cinC_a + cinC_b):")
    for r, (t_a, t_b, ca, cb, _) in enumerate(selected):
        n_a = int(survive_all[t_a].sum())
        n_b = int(survive_all[t_b].sum())
        print(f"  rank {r}: t_a={t_a} t_b={t_b}  |C_a|={n_a:5d} |C_b|={n_b:5d}  "
              f"cinC=({ca:.2f}°/{cb:.2f}°)")

    # --- per-anchor architecture ---
    all_hypotheses_om = []        # rad/s vectors
    all_hypotheses_anchor = []    # anchor index
    all_ang_to_truth = []         # angular distance to that anchor's truth
    per_anchor_results = []
    per_anchor_truth = []

    for ai, (t_a, t_b, ca, cb, _) in enumerate(selected):
        idx_a = np.where(survive_all[t_a])[0]
        idx_b = np.where(survive_all[t_b])[0]
        C_a = q_pool[idx_a]
        C_b = q_pool[idx_b]
        n_a, n_b = len(C_a), len(C_b)

        # truth ω at t_a from cached q(t) finite-diff over the pair window
        om_truth = fd_omega_passive(q_truth_t[t_a:t_a+1],
                                    q_truth_t[t_b:t_b+1], Δt_pair)[0]
        om_truth_mag = float(np.linalg.norm(om_truth))
        om_truth_hat = om_truth / om_truth_mag

        # all pair finite-diff (cap at 50k pairs by random subsample)
        max_pairs = 50000
        n_total = n_a * n_b
        if n_total > max_pairs:
            rng = np.random.default_rng(42 + ai)
            sa = rng.integers(0, n_a, size=max_pairs)
            sb = rng.integers(0, n_b, size=max_pairs)
            QA = C_a[sa]
            QB = C_b[sb]
        else:
            QA = np.repeat(C_a, n_b, axis=0)
            QB = np.tile(C_b, (n_a, 1))
        om_all = fd_omega_passive(QA, QB, Δt_pair)
        om_mag = np.linalg.norm(om_all, axis=1)

        # |ω|-prior ±25%
        target = om_truth_mag
        mask = (om_mag >= target * PRIOR_BRACKET[0]) & \
               (om_mag <= target * PRIOR_BRACKET[1])
        n_pass = int(mask.sum())
        if n_pass == 0:
            per_anchor_results.append({
                "anchor_idx": ai, "t_a": t_a, "t_b": t_b,
                "n_C_a": n_a, "n_C_b": n_b,
                "n_pass": 0, "skipped": "no survivors after prior",
            })
            continue

        om_pass = om_all[mask]
        ang_to_truth = angular_dist(om_pass, om_truth)

        baseline_10 = 1.0 - np.cos(np.radians(10.0))
        baseline_30 = 1.0 - np.cos(np.radians(30.0))
        frac_10 = float((ang_to_truth < 10).mean())
        frac_30 = float((ang_to_truth < 30).mean())

        # modal-cluster of this anchor's hypotheses
        om_hat_pass = om_pass / np.linalg.norm(om_pass, axis=1, keepdims=True)
        if n_pass <= 5000:
            modal_centroid, modal_count = find_modal_cluster(om_hat_pass, 10.0)
        else:
            sub = np.random.default_rng(0).choice(n_pass, 5000, replace=False)
            modal_centroid, modal_count = find_modal_cluster(om_hat_pass[sub], 10.0)
        modal_dist_to_truth = float(angular_dist(modal_centroid[None, :], om_truth)[0])

        per_anchor_results.append({
            "anchor_idx": ai, "t_a": t_a, "t_b": t_b,
            "n_C_a": n_a, "n_C_b": n_b,
            "n_pass": n_pass,
            "om_truth_dps": om_truth_mag * 180 / np.pi,
            "om_truth_hat": om_truth_hat.tolist(),
            "frac_within_10deg": frac_10,
            "frac_within_30deg": frac_30,
            "ratio_10deg_vs_baseline": frac_10 / baseline_10,
            "ratio_30deg_vs_baseline": frac_30 / baseline_30,
            "median_ang_deg": float(np.median(ang_to_truth)),
            "p10_ang_deg": float(np.percentile(ang_to_truth, 10)),
            "min_ang_deg": float(ang_to_truth.min()),
            "modal_dist_to_truth_deg": modal_dist_to_truth,
            "modal_count": modal_count,
        })
        all_hypotheses_om.append(om_pass)
        all_hypotheses_anchor.append(np.full(n_pass, ai))
        all_ang_to_truth.append(ang_to_truth)
        per_anchor_truth.append({
            "ai": ai, "t_a": t_a, "om_truth": om_truth, "om_truth_hat": om_truth_hat
        })

    # --- aggregated stats ---
    om_agg = np.concatenate(all_hypotheses_om, axis=0)
    anchor_tag = np.concatenate(all_hypotheses_anchor, axis=0)
    ang_to_own_truth = np.concatenate(all_ang_to_truth, axis=0)
    n_total_pass = len(om_agg)
    print(f"\ntotal aggregated hypotheses: {n_total_pass}")

    # aggregated angular-distance-to-OWN-anchor-truth distribution
    baseline_10 = 1.0 - np.cos(np.radians(10.0))
    baseline_30 = 1.0 - np.cos(np.radians(30.0))
    agg_frac_10 = float((ang_to_own_truth < 10).mean())
    agg_frac_30 = float((ang_to_own_truth < 30).mean())
    agg_ratio_10 = agg_frac_10 / baseline_10
    agg_ratio_30 = agg_frac_30 / baseline_30
    print(f"aggregated f<10°: {agg_frac_10*100:.2f}% (×{agg_ratio_10:.2f} baseline)")
    print(f"aggregated f<30°: {agg_frac_30*100:.2f}% (×{agg_ratio_30:.2f} baseline)")

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) per-anchor concentration ratios
    ax = axes[0, 0]
    okay = [r for r in per_anchor_results if "skipped" not in r]
    ax.bar(range(len(okay)), [r["ratio_10deg_vs_baseline"] for r in okay],
           color="b", alpha=0.6, label="frac<10° / baseline")
    ax.bar(range(len(okay)),
           [r["ratio_30deg_vs_baseline"] for r in okay],
           color="g", alpha=0.4, label="frac<30° / baseline")
    ax.axhline(1.0, ls=":", color="grey", label="random uniform")
    ax.axhline(agg_ratio_10, color="b", lw=2, ls="--",
               label=f"aggregated 10° ratio = {agg_ratio_10:.2f}")
    ax.axhline(agg_ratio_30, color="g", lw=2, ls="--",
               label=f"aggregated 30° ratio = {agg_ratio_30:.2f}")
    ax.set_xlabel("anchor index")
    ax.set_ylabel("concentration ratio")
    ax.set_title(f"per-anchor concentration vs aggregated (n={N_ANCHORS} anchors)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) histogram: per-anchor vs aggregated angular distance to truth
    ax = axes[0, 1]
    bins = np.linspace(0, 90, 46)
    for ai, r in enumerate(okay[:5]):
        mask_ai = anchor_tag == r["anchor_idx"]
        ax.hist(ang_to_own_truth[mask_ai], bins=bins, alpha=0.3,
                density=True, label=f"anchor {r['anchor_idx']} (t_a={r['t_a']})")
    ax.hist(ang_to_own_truth, bins=bins, color="k", alpha=0.6,
            density=True, label="AGGREGATED", histtype="step", lw=2)
    theta = np.linspace(0, 90, 200)
    ax.plot(theta, np.sin(np.radians(theta)) * (np.pi / 180), "k--",
            lw=1, label="uniform-sphere baseline")
    ax.set_xlabel("angular distance to anchor's truth-ω̂  (deg)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"f<10° aggregated = {agg_frac_10*100:.2f}% (×{agg_ratio_10:.2f}); "
                 f"f<30° = {agg_frac_30*100:.2f}% (×{agg_ratio_30:.2f})")

    # (c) per-anchor modal cluster vs truth
    ax = axes[1, 0]
    modals = [r["modal_dist_to_truth_deg"] for r in okay]
    ax.bar(range(len(okay)), modals, color="m", alpha=0.6)
    ax.axhline(np.mean(modals), color="k", ls="--", label=f"mean {np.mean(modals):.1f}°")
    ax.set_xlabel("anchor index")
    ax.set_ylabel("modal cluster → truth-ω̂ (deg)")
    ax.set_title("per-anchor modal cluster: still drifting if architecture noisy")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # (d) aggregated hypothesis density on sphere (lon/lat) with truth trace
    ax = axes[1, 1]
    # Antipodal-fold each anchor's hypotheses to its OWN truth-ω̂
    om_hat_agg = om_agg / np.linalg.norm(om_agg, axis=1, keepdims=True)
    flipped = np.zeros_like(om_hat_agg)
    for ai, ptruth in enumerate(per_anchor_truth):
        mask = anchor_tag == ptruth["ai"]
        oh = om_hat_agg[mask]
        flip = (oh @ ptruth["om_truth_hat"]) < 0
        flipped[mask] = np.where(flip[:, None], -oh, oh)
    lon = np.degrees(np.arctan2(flipped[:, 1], flipped[:, 0]))
    lat = np.degrees(np.arcsin(flipped[:, 2]))
    # subsample for plotting
    if len(lon) > 5000:
        sub = np.random.default_rng(0).choice(len(lon), 5000, replace=False)
        ax.scatter(lon[sub], lat[sub], s=1, alpha=0.2, color="b", label="hypotheses")
    else:
        ax.scatter(lon, lat, s=1, alpha=0.2, color="b", label="hypotheses")
    # Plot each anchor's truth-ω̂ at sphere
    for ai, ptruth in enumerate(per_anchor_truth):
        h = ptruth["om_truth_hat"]
        lon_t = np.degrees(np.arctan2(h[1], h[0]))
        lat_t = np.degrees(np.arcsin(h[2]))
        ax.scatter(lon_t, lat_t, s=120, marker="*", color="red",
                   edgecolor="black", linewidth=1, zorder=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("body-frame longitude (deg)")
    ax.set_ylabel("body-frame latitude (deg)")
    ax.set_title("aggregated hypothesis cloud (antipodal-folded to truth);  red ★ = truth-ω̂(t_a) per anchor")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_p = OUT / "multi_anchor_aggregation.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_p}")

    summary = {
        "seed": 89,
        "delta_epoch": DELTA_EPOCH,
        "delta_t_s": Δt_pair,
        "cic_threshold_deg": CIC_THRESH_DEG,
        "n_candidates": len(candidates),
        "n_anchors_selected": len(selected),
        "anchors": [{"t_a": t[0], "t_b": t[1], "cic_a": t[2], "cic_b": t[3]}
                    for t in selected],
        "per_anchor": per_anchor_results,
        "aggregated": {
            "n_total_pass": n_total_pass,
            "frac_within_10deg": agg_frac_10,
            "frac_within_30deg": agg_frac_30,
            "ratio_10deg_vs_baseline": agg_ratio_10,
            "ratio_30deg_vs_baseline": agg_ratio_30,
            "median_ang_deg": float(np.median(ang_to_own_truth)),
            "p10_ang_deg": float(np.percentile(ang_to_own_truth, 10)),
        },
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"Saved: {OUT / 'summary.json'}")
    return summary


if __name__ == "__main__":
    main()
