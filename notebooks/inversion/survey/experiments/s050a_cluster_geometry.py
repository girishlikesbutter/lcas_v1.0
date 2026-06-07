"""s050a — Phase 1a cluster geometry on s049 cascade survivors (seed 14, tol=0.10/K=3).

Reconstructs the 35,835 survivor set, computes per-survivor geometric error
metrics (q-distance to truth, ω direction/magnitude/vector errors), plots the
joint distribution, and greedy-clusters the survivors at multiple radii.

Output: results/s050a_cluster_geometry/ with cluster.npz, summary.json, and
PNG plots.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SURVEY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY_ROOT))

from lib.twin import canonical_batch  # noqa: E402

CASCADE_NPZ = SURVEY_ROOT / "results" / "s049_cascade_seed14" / "cascade.npz"
SUMMARY_JSON = SURVEY_ROOT / "results" / "s049_cascade_seed14" / "summary.json"
TRAJ_NPZ = (
    PROJECT_ROOT
    / "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed014.npz"
)

OUT_DIR = SURVEY_ROOT / "results" / "s050a_cluster_geometry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOL_MAG = 0.10
K_REQUIRED = 3
CLUSTER_RADII = [
    {"label": "tight (3°/3%)", "q_deg": 3.0, "om_rel": 0.03},
    {"label": "default (5°/5%)", "q_deg": 5.0, "om_rel": 0.05},
    {"label": "loose (10°/10%)", "q_deg": 10.0, "om_rel": 0.10},
    {"label": "very loose (20°/20%)", "q_deg": 20.0, "om_rel": 0.20},
]


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------


def quat_angular_dist_deg(q_pool: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Angular distance in degrees between each q in q_pool (N,4) and q_ref (4,).

    Treats q and -q as the same rotation by taking |dot|.
    """
    cos_half = np.abs(q_pool @ q_ref)
    cos_half = np.clip(cos_half, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cos_half))


# ---------------------------------------------------------------------------
# Greedy cluster
# ---------------------------------------------------------------------------


def greedy_cluster(
    qA: np.ndarray,
    om: np.ndarray,
    om_truth_mag: float,
    priority: np.ndarray,
    q_radius_deg: float,
    om_radius_rel: float,
    max_centers: int = 50000,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Greedy clustering: lowest-priority points become centers; absorb neighbours.

    Returns (assignment, member_lists). assignment[i] = cluster index of point i.
    """
    n = len(qA)
    assignment = -np.ones(n, dtype=np.int32)
    sort_idx = np.argsort(priority, kind="stable")
    centers: list[int] = []
    members: list[np.ndarray] = []

    for next_seed in sort_idx:
        if assignment[next_seed] >= 0:
            continue
        if len(centers) >= max_centers:
            break

        qA_c = qA[next_seed]
        om_c = om[next_seed]

        unassigned_mask = assignment < 0
        idx_un = np.where(unassigned_mask)[0]
        if len(idx_un) == 0:
            break

        cos_half = np.abs(qA[idx_un] @ qA_c)
        cos_half = np.clip(cos_half, 0.0, 1.0)
        q_dist = np.degrees(2.0 * np.arccos(cos_half))

        om_dist = np.linalg.norm(om[idx_un] - om_c, axis=1) / om_truth_mag

        within = (q_dist < q_radius_deg) & (om_dist < om_radius_rel)
        member_idx = idx_un[within]
        if not within.any():
            # Should not happen — the seed is always within radius of itself.
            member_idx = np.array([next_seed], dtype=np.int64)

        assignment[member_idx] = len(centers)
        centers.append(int(next_seed))
        members.append(member_idx)

    return assignment, [np.asarray(m, dtype=np.int64) for m in members]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()

    print(f"[load] cascade: {CASCADE_NPZ}")
    casc = np.load(CASCADE_NPZ)
    qA_kept = casc["qA_kept"]
    om_kept = casc["om_kept"]
    delta_mag = casc["delta_mag"]
    val_eps = casc["val_eps"]
    t0_ep = int(casc["t0_ep"])
    t1_ep = int(casc["t1_ep"])
    delta_t = float(casc["delta_t"])
    truth_q_a_discrete = casc["truth_q_a"]
    truth_q_b_discrete = casc["truth_q_b"]
    om_truth_cascade = casc["om_truth_cascade"]
    om_truth_at_t0 = casc["om_truth_at_t0"]
    print(
        f"[load] N_total_hyp={len(qA_kept)}, val_eps={val_eps.tolist()}, "
        f"t0={t0_ep}, t1={t1_ep}, dt={delta_t:.3f}s"
    )

    print(f"[load] traj: {TRAJ_NPZ}")
    traj = np.load(TRAJ_NPZ)
    quats_truth = traj["quaternions"]  # (T, 4)
    omega0_rad = traj["omega0_rad"]  # (3,)
    omega_mag_dps = float(traj["omega_mag_dps"])
    truth_q_actual_at_t0 = quats_truth[t0_ep]  # (4,)
    om_truth_dps = np.degrees(omega0_rad)  # (3,) in dps
    om_truth_mag_dps = float(np.linalg.norm(om_truth_dps))
    assert (
        abs(om_truth_mag_dps - omega_mag_dps) < 1e-9
    ), f"om mag mismatch: {om_truth_mag_dps} vs {omega_mag_dps}"
    print(
        f"[truth] omega_mag_dps={om_truth_mag_dps:.4f}, "
        f"om_truth_dps={om_truth_dps}, |om_truth_at_t0_summary|={float(np.linalg.norm(om_truth_at_t0)):.4f}"
    )

    # ------------------------------------------------------------------
    # Stage 1: rebuild survivor mask at tol=0.10/K=3
    # ------------------------------------------------------------------
    pass_per_epoch = delta_mag < TOL_MAG  # (N, 5)
    survive_mask = pass_per_epoch[:, :K_REQUIRED].all(axis=1)
    n_surv = int(survive_mask.sum())
    print(f"[stage1] survivors @ tol={TOL_MAG}/K={K_REQUIRED}: {n_surv}")

    qA_surv = qA_kept[survive_mask]
    om_surv = om_kept[survive_mask]  # rad/s
    delta_mag_surv = delta_mag[survive_mask]  # (n_surv, 5)
    sum_delta = delta_mag_surv[:, :K_REQUIRED].sum(axis=1)  # priority for greedy

    # Convert ω to dps for human-readable plotting
    om_surv_dps = np.degrees(om_surv)

    # ------------------------------------------------------------------
    # Stage 2: per-survivor error metrics
    # ------------------------------------------------------------------
    print("[stage2] computing per-survivor errors")

    q_dist_truth_actual = quat_angular_dist_deg(qA_surv, truth_q_actual_at_t0)
    q_dist_truth_qa_disc = quat_angular_dist_deg(qA_surv, truth_q_a_discrete)

    # Reference ω at the anchor epoch — torque-free rigid-body dynamics gives a
    # ~1% body-frame ω wobble vs omega0_rad, so use the cascade-cached snapshot.
    om_truth_at_t0_dps = np.degrees(om_truth_at_t0)  # (3,)
    om_truth_at_t0_mag_dps = float(np.linalg.norm(om_truth_at_t0_dps))
    print(
        f"[truth] om at t0 (rigid-body): {om_truth_at_t0_dps}  |·|={om_truth_at_t0_mag_dps:.4f} dps  "
        f"(vs omega0 |·|={om_truth_mag_dps:.4f}, drift={(om_truth_at_t0_mag_dps-om_truth_mag_dps)/om_truth_mag_dps*100:+.2f}%)"
    )

    om_truth_unit = om_truth_at_t0_dps / om_truth_at_t0_mag_dps
    om_surv_mag = np.linalg.norm(om_surv_dps, axis=1)
    cos_om = np.clip(
        (om_surv_dps @ om_truth_unit) / np.maximum(om_surv_mag, 1e-12), -1.0, 1.0
    )
    om_dir_err_deg = np.degrees(np.arccos(cos_om))
    om_mag_err_rel = (om_surv_mag - om_truth_at_t0_mag_dps) / om_truth_at_t0_mag_dps
    om_vec_err_rel = (
        np.linalg.norm(om_surv_dps - om_truth_at_t0_dps, axis=1)
        / om_truth_at_t0_mag_dps
    )

    print(
        f"[stage2] q_dist_truth_actual:    p10/p50/p90 = "
        f"{np.percentile(q_dist_truth_actual, 10):.2f}° / "
        f"{np.percentile(q_dist_truth_actual, 50):.2f}° / "
        f"{np.percentile(q_dist_truth_actual, 90):.2f}°  "
        f"min/max = {q_dist_truth_actual.min():.3f}° / {q_dist_truth_actual.max():.2f}°"
    )
    print(
        f"[stage2] q_dist_truth_qa_disc:   p10/p50/p90 = "
        f"{np.percentile(q_dist_truth_qa_disc, 10):.2f}° / "
        f"{np.percentile(q_dist_truth_qa_disc, 50):.2f}° / "
        f"{np.percentile(q_dist_truth_qa_disc, 90):.2f}°  "
        f"min = {q_dist_truth_qa_disc.min():.3f}°"
    )
    print(
        f"[stage2] om_dir_err_deg:         p10/p50/p90 = "
        f"{np.percentile(om_dir_err_deg, 10):.2f}° / "
        f"{np.percentile(om_dir_err_deg, 50):.2f}° / "
        f"{np.percentile(om_dir_err_deg, 90):.2f}°"
    )
    print(
        f"[stage2] om_mag_err_rel:         p10/p50/p90 = "
        f"{np.percentile(om_mag_err_rel, 10)*100:.2f}% / "
        f"{np.percentile(om_mag_err_rel, 50)*100:.2f}% / "
        f"{np.percentile(om_mag_err_rel, 90)*100:.2f}%"
    )
    print(
        f"[stage2] om_vec_err_rel:         p10/p50/p90 = "
        f"{np.percentile(om_vec_err_rel, 10)*100:.2f}% / "
        f"{np.percentile(om_vec_err_rel, 50)*100:.2f}% / "
        f"{np.percentile(om_vec_err_rel, 90)*100:.2f}%"
    )

    # Sanity: how many survivors land in each "interesting" bucket?
    near_truth_q = q_dist_truth_actual < 5.0
    near_truth_om_vec = om_vec_err_rel < 0.30
    near_truth_om_mag = np.abs(om_mag_err_rel) < 0.10
    print(
        f"[stage2] survivors with q_dist<5°: {near_truth_q.sum()} "
        f"({near_truth_q.mean()*100:.2f}%)"
    )
    print(
        f"[stage2] survivors with om_vec<30%: {near_truth_om_vec.sum()} "
        f"({near_truth_om_vec.mean()*100:.2f}%)"
    )
    print(
        f"[stage2] survivors with both q<5° AND om_vec<30%: "
        f"{(near_truth_q & near_truth_om_vec).sum()}"
    )
    # Truth-q_a enrichment: among the 153 hypotheses with qA == truth_q_a (s049
    # summary), how many survive at this filter? Random would give 25.3%.
    truth_qa_hypotheses_mask = q_dist_truth_qa_disc < 1e-6  # exact match in survivor
    n_truth_qa_surv = int(truth_qa_hypotheses_mask.sum())
    n_truth_qa_total = 153  # from s049 summary.json (n_hypotheses_at_truth_qa)
    expected_random = n_surv / 141706 * n_truth_qa_total
    print(
        f"[stage2] truth-q_a survivors (qa exactly == truth_q_a discrete): "
        f"{n_truth_qa_surv} / {n_truth_qa_total} (expected if random "
        f"throughput {n_surv/141706*100:.1f}%: {expected_random:.1f})"
    )

    # ------------------------------------------------------------------
    # Stage 3: canonicalise + greedy cluster
    # ------------------------------------------------------------------
    print("[stage3] canonicalising survivors via lib.twin.canonical_batch")
    qA_canon, om_canon = canonical_batch(qA_surv, om_surv)
    om_canon_dps = np.degrees(om_canon)
    print(
        f"[stage3] canon-flipped fraction: "
        f"{(qA_canon[:, 0] != qA_surv[:, 0]).mean()*100:.1f}% (rough sign-flip indicator)"
    )

    # Compute (q_a, ω) joint duplicates after canon (within tight bins) — purely informational
    # Use 0.5° + 0.005 ω-rel bins
    bin_q = (qA_canon * 200).round().astype(np.int64)  # very coarse
    bin_om = (om_canon_dps * 200).round().astype(np.int64)
    keys = np.concatenate([bin_q, bin_om], axis=1)
    keys_view = np.ascontiguousarray(keys).view(
        [("", keys.dtype)] * keys.shape[1]
    )
    n_unique_canon = int(len(np.unique(keys_view)))
    print(
        f"[stage3] n_unique (q_canon, om_canon) at coarse bins: {n_unique_canon}"
    )
    print(f"[stage3] n_unique q_a: {len(np.unique(qA_surv, axis=0))}")

    cluster_results: list[dict] = []
    for cfg in CLUSTER_RADII:
        t_c = time.time()
        # Cluster on pre-canon survivors first (the LM polish stage will run
        # on each cluster rep; canon merges twins — we want both numbers).
        assign_pre, members_pre = greedy_cluster(
            qA_surv,
            om_surv_dps,
            om_truth_mag_dps,
            sum_delta,
            cfg["q_deg"],
            cfg["om_rel"],
        )
        assign_canon, members_canon = greedy_cluster(
            qA_canon,
            om_canon_dps,
            om_truth_mag_dps,
            sum_delta,
            cfg["q_deg"],
            cfg["om_rel"],
        )
        sizes_pre = np.array([len(m) for m in members_pre])
        sizes_canon = np.array([len(m) for m in members_canon])
        print(
            f"[stage3] cluster {cfg['label']}:  "
            f"n_clusters_pre={len(members_pre)} (largest {sizes_pre.max() if len(sizes_pre) else 0}, "
            f"top-5 sizes {np.sort(sizes_pre)[::-1][:5].tolist()})  "
            f"n_clusters_canon={len(members_canon)}  wall={time.time()-t_c:.1f}s"
        )
        cluster_results.append(
            {
                "label": cfg["label"],
                "q_radius_deg": cfg["q_deg"],
                "om_radius_rel": cfg["om_rel"],
                "n_clusters_pre_canon": int(len(members_pre)),
                "n_clusters_canon": int(len(members_canon)),
                "largest_cluster_size_pre_canon": int(sizes_pre.max())
                if len(sizes_pre)
                else 0,
                "largest_cluster_size_canon": int(sizes_canon.max())
                if len(sizes_canon)
                else 0,
                "top5_sizes_pre_canon": np.sort(sizes_pre)[::-1][:5].tolist(),
                "top5_sizes_canon": np.sort(sizes_canon)[::-1][:5].tolist(),
                "n_clusters_with_min_size_2_canon": int(
                    (sizes_canon >= 2).sum()
                ),
                "n_singletons_canon": int((sizes_canon == 1).sum()),
            }
        )

    # ------------------------------------------------------------------
    # Stage 4: cluster decomposition at the chosen default radius — find which
    # clusters land near truth.
    # ------------------------------------------------------------------
    chosen = CLUSTER_RADII[1]  # 5°/5%
    assign_canon, members_canon = greedy_cluster(
        qA_canon,
        om_canon_dps,
        om_truth_mag_dps,
        sum_delta,
        chosen["q_deg"],
        chosen["om_rel"],
    )
    cluster_summary = []
    for ci, mem in enumerate(members_canon):
        if ci >= 50:
            break  # report top 50 by size only
        member_q_dist = q_dist_truth_actual[mem]
        member_om_vec = om_vec_err_rel[mem]
        cluster_summary.append(
            {
                "rank": ci,
                "size": int(len(mem)),
                "rep_idx": int(mem[0]),
                "q_dist_to_truth_min_deg": float(member_q_dist.min()),
                "q_dist_to_truth_med_deg": float(np.median(member_q_dist)),
                "om_vec_err_rel_min": float(member_om_vec.min()),
                "om_vec_err_rel_med": float(np.median(member_om_vec)),
            }
        )

    # Sort by size desc to surface the dominant clusters
    cluster_summary_by_size = sorted(
        cluster_summary, key=lambda c: -c["size"]
    )[:20]
    print("[stage4] top 20 clusters by size at canon (q<5°, ω<5%):")
    for c in cluster_summary_by_size:
        print(
            f"  rank{c['rank']:>3d}  size={c['size']:>5d}  "
            f"q_dist_truth_min={c['q_dist_to_truth_min_deg']:6.2f}°  "
            f"om_vec_min={c['om_vec_err_rel_min']*100:6.2f}%"
        )

    # ------------------------------------------------------------------
    # Stage 5: plots
    # ------------------------------------------------------------------
    print("[stage5] plotting")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    ax.hist(q_dist_truth_actual, bins=100, color="C0", alpha=0.85)
    ax.axvline(
        1.39,
        color="red",
        ls="--",
        label="truth_q_a (discrete) = 1.39°",
    )
    ax.set_xlabel("q distance to actual truth-q at t_0 (deg)")
    ax.set_ylabel("count")
    ax.set_title(
        f"q-distance to truth (n={n_surv})\n"
        f"min={q_dist_truth_actual.min():.2f}°  med={np.median(q_dist_truth_actual):.1f}°  max={q_dist_truth_actual.max():.1f}°"
    )
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    ax = axes[0, 1]
    ax.hist(
        om_vec_err_rel * 100, bins=np.linspace(0, 200, 101), color="C1", alpha=0.85
    )
    ax.axvline(16.82, color="red", ls="--", label="cascade ω-noise floor 16.82%")
    ax.axvline(26.71, color="orange", ls="--", label="cascade-truth ω-vec err 26.71%")
    ax.set_xlabel("|Δω| / |ω_truth|  (%)")
    ax.set_ylabel("count")
    ax.set_title("ω vector error (relative)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    ax = axes[0, 2]
    ax.hist(
        om_mag_err_rel * 100, bins=np.linspace(-100, 100, 101), color="C2", alpha=0.85
    )
    ax.axvline(0, color="red", ls="--")
    ax.set_xlabel("(|ω| - |ω_truth|) / |ω_truth|  (%)")
    ax.set_ylabel("count")
    ax.set_title("ω magnitude error (relative, signed)")
    ax.set_yscale("log")

    ax = axes[1, 0]
    h = ax.hexbin(
        q_dist_truth_actual,
        om_vec_err_rel * 100,
        gridsize=60,
        cmap="viridis",
        bins="log",
    )
    plt.colorbar(h, ax=ax, label="log10 count")
    ax.set_xlabel("q distance to truth (deg)")
    ax.set_ylabel("|Δω| / |ω_truth|  (%)")
    ax.set_title("joint distribution")
    ax.axvline(5, color="red", ls="--", lw=1, label="q=5°")
    ax.axhline(5, color="red", ls="--", lw=1, label="ω=5%")
    ax.axhline(16.82, color="orange", ls=":", lw=1, label="cascade ω-noise floor")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    h = ax.hexbin(
        q_dist_truth_actual,
        om_dir_err_deg,
        gridsize=60,
        cmap="viridis",
        bins="log",
    )
    plt.colorbar(h, ax=ax, label="log10 count")
    ax.set_xlabel("q distance to truth (deg)")
    ax.set_ylabel("ω direction error (deg)")
    ax.set_title("joint: q-dist vs ω-direction error")

    ax = axes[1, 2]
    h = ax.hexbin(
        om_dir_err_deg,
        om_mag_err_rel * 100,
        gridsize=60,
        cmap="viridis",
        bins="log",
    )
    plt.colorbar(h, ax=ax, label="log10 count")
    ax.set_xlabel("ω direction error (deg)")
    ax.set_ylabel("ω magnitude error (%)")
    ax.set_title("ω direction vs magnitude")

    plt.suptitle(
        f"s050a Phase 1a — seed 14 cascade survivors  "
        f"(tol=0.10/K=3, n={n_surv})",
        fontsize=14,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "phase1a_geometry.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[saved] {out_png}")

    # cluster-count plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    labels = [c["label"] for c in cluster_results]
    n_pre = [c["n_clusters_pre_canon"] for c in cluster_results]
    n_canon = [c["n_clusters_canon"] for c in cluster_results]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, n_pre, width=0.4, label="pre-canon (raw survivors)")
    ax.bar(x + 0.2, n_canon, width=0.4, label="canon (post-twin dedup)")
    for i, (a, b) in enumerate(zip(n_pre, n_canon)):
        ax.text(i - 0.2, a, str(a), ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, b, str(b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("# distinct clusters")
    ax.set_title(
        f"Greedy cluster count vs radius — seed 14 cascade survivors (n={n_surv})"
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_png_clusters = OUT_DIR / "phase1a_cluster_counts.png"
    plt.savefig(out_png_clusters, dpi=120)
    plt.close()
    print(f"[saved] {out_png_clusters}")

    # ------------------------------------------------------------------
    # Stage 6: save NPZ + JSON
    # ------------------------------------------------------------------
    cluster_npz = OUT_DIR / "cluster.npz"
    np.savez(
        cluster_npz,
        # survivor data
        survive_mask=survive_mask,
        qA_surv=qA_surv,
        om_surv_dps=om_surv_dps,
        qA_canon=qA_canon,
        om_canon_dps=om_canon_dps,
        # error metrics
        q_dist_truth_actual=q_dist_truth_actual,
        q_dist_truth_qa_disc=q_dist_truth_qa_disc,
        om_dir_err_deg=om_dir_err_deg,
        om_mag_err_rel=om_mag_err_rel,
        om_vec_err_rel=om_vec_err_rel,
        sum_delta_mag=sum_delta,
        # cluster assignment (chosen = 5°/5%, post canon)
        assign_canon_default=assign_canon,
        # truth references
        truth_q_actual_at_t0=truth_q_actual_at_t0,
        truth_q_a_discrete=truth_q_a_discrete,
        om_truth_dps=om_truth_dps,
        om_truth_mag_dps=om_truth_mag_dps,
    )
    print(f"[saved] {cluster_npz}")

    summary = {
        "seed": 14,
        "tol_mag": TOL_MAG,
        "K_required": K_REQUIRED,
        "n_survivors": n_surv,
        "n_total_hypotheses": int(len(qA_kept)),
        "om_truth_mag_dps_omega0": om_truth_mag_dps,
        "om_truth_at_t0_mag_dps": om_truth_at_t0_mag_dps,
        "om_truth_dps_omega0": om_truth_dps.tolist(),
        "om_truth_at_t0_dps": om_truth_at_t0_dps.tolist(),
        "truth_qa_enrichment": {
            "n_truth_qa_total_hypotheses": n_truth_qa_total,
            "n_truth_qa_survivors": n_truth_qa_surv,
            "random_throughput_pct": float(n_surv / 141706 * 100),
            "expected_truth_qa_if_random": float(expected_random),
            "enrichment_factor": float(
                n_truth_qa_surv / max(expected_random, 1e-9)
            ),
        },
        "stats": {
            "q_dist_truth_actual_deg": {
                "min": float(q_dist_truth_actual.min()),
                "p10": float(np.percentile(q_dist_truth_actual, 10)),
                "p50": float(np.percentile(q_dist_truth_actual, 50)),
                "p90": float(np.percentile(q_dist_truth_actual, 90)),
                "max": float(q_dist_truth_actual.max()),
            },
            "q_dist_truth_qa_disc_deg": {
                "min": float(q_dist_truth_qa_disc.min()),
                "p10": float(np.percentile(q_dist_truth_qa_disc, 10)),
                "p50": float(np.percentile(q_dist_truth_qa_disc, 50)),
                "p90": float(np.percentile(q_dist_truth_qa_disc, 90)),
            },
            "om_dir_err_deg": {
                "min": float(om_dir_err_deg.min()),
                "p10": float(np.percentile(om_dir_err_deg, 10)),
                "p50": float(np.percentile(om_dir_err_deg, 50)),
                "p90": float(np.percentile(om_dir_err_deg, 90)),
                "max": float(om_dir_err_deg.max()),
            },
            "om_mag_err_rel": {
                "p10": float(np.percentile(om_mag_err_rel, 10)),
                "p50": float(np.percentile(om_mag_err_rel, 50)),
                "p90": float(np.percentile(om_mag_err_rel, 90)),
            },
            "om_vec_err_rel": {
                "min": float(om_vec_err_rel.min()),
                "p10": float(np.percentile(om_vec_err_rel, 10)),
                "p50": float(np.percentile(om_vec_err_rel, 50)),
                "p90": float(np.percentile(om_vec_err_rel, 90)),
                "max": float(om_vec_err_rel.max()),
            },
        },
        "near_truth_buckets": {
            "q_lt_5deg": int(near_truth_q.sum()),
            "om_vec_lt_30pct": int(near_truth_om_vec.sum()),
            "q_lt_5deg_AND_om_vec_lt_30pct": int(
                (near_truth_q & near_truth_om_vec).sum()
            ),
            "q_lt_3deg": int((q_dist_truth_actual < 3.0).sum()),
            "om_vec_lt_20pct": int((om_vec_err_rel < 0.20).sum()),
        },
        "cluster_results": cluster_results,
        "cluster_breakdown_default_5deg_5pct": cluster_summary_by_size,
        "n_unique_qa_pre_canon": int(len(np.unique(qA_surv, axis=0))),
        "n_unique_canon_coarse_bins": n_unique_canon,
        "wall_total_s": float(time.time() - t_start),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[saved] {OUT_DIR / 'summary.json'}")
    print(f"[done] wall = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
