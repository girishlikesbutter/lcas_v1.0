"""s050b — validation-step stress test on s049 cascade hypotheses.

Tests whether smarter validation (tighter tol, score-by-continuous-rank,
bright-epoch validation) can concentrate the 141k cascade hypotheses on
truth WITHOUT touching the upstream cascade.

Three stages:
  1. Cached re-score: vary tol × K-subset over the 5 cached val_eps.
  2. Continuous-score rankings: sum/max/inv-Δk weighting.
  3. Bright-epoch addition: predict mag at Δk=-15..-13 and +13..+15
     via constant-ω propagation + v1 surrogate, then combined filtering.

Output: results/s050b_validation_stress/{summary.json, plots, npz}.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
SURROGATE_PATH = Path("/home/girish/surrogate_model")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(SURROGATE_PATH))

from surrogate_model.surrogate_v1 import SurrogateModel as SurrogateV1  # noqa: E402

CASCADE_NPZ = SURVEY_DIR / "results" / "s049_cascade_seed14" / "cascade.npz"
SUMMARY_JSON = SURVEY_DIR / "results" / "s049_cascade_seed14" / "summary.json"
TRAJ_NPZ = (
    PROJECT_ROOT
    / "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed014.npz"
)

OUT_DIR = SURVEY_DIR / "results" / "s050b_validation_stress"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
SEED = 14
N_TOTAL_HYP = 141706
N_TRUTH_QA_TOTAL = 153  # from s049 summary.json

BRIGHT_EPOCHS_TO_TEST = [260, 261, 288, 289, 290]  # Δk = -15, -14, +13, +14, +15

# Re-score grid
RESCORE_TOLS = [0.03, 0.05, 0.07, 0.10, 0.15]
TOP_N_LIST = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 35835]
CLUSTER_RADIUS_Q_DEG = 5.0
CLUSTER_RADIUS_OM_REL = 0.05


# ---------------------------------------------------------------------------
# Quaternion utilities (lifted from s049)
# ---------------------------------------------------------------------------


def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if q1.ndim == 1:
        q1 = q1[None, :]
    if q2.ndim == 1:
        q2 = q2[None, :]
    if q1.shape[0] == 1 and q2.shape[0] > 1:
        q1 = np.broadcast_to(q1, q2.shape).copy()
    if q2.shape[0] == 1 and q1.shape[0] > 1:
        q2 = np.broadcast_to(q2, q1.shape).copy()
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_exp_batch(omega: np.ndarray, dt: float) -> np.ndarray:
    omega = np.asarray(omega).reshape(-1, 3)
    om_mag = np.linalg.norm(omega, axis=1)
    theta = om_mag * dt
    half = 0.5 * theta
    safe = om_mag > 1e-12
    axis = np.zeros_like(omega)
    axis[safe] = omega[safe] / om_mag[safe, None]
    sin_h = np.sin(half)
    return np.column_stack([
        np.cos(half),
        sin_h * axis[:, 0],
        sin_h * axis[:, 1],
        sin_h * axis[:, 2],
    ])


def constant_omega_propagate(
    q0_wxyz: np.ndarray, omega: np.ndarray, dt: float
) -> np.ndarray:
    q_rot = quat_exp_batch(omega, dt)
    if q0_wxyz.ndim == 1:
        q0_wxyz = q0_wxyz[None, :]
    if q0_wxyz.shape[0] == 1 and q_rot.shape[0] > 1:
        q0_wxyz = np.broadcast_to(q0_wxyz, q_rot.shape).copy()
    return quat_mul_batch(q_rot, q0_wxyz)


def quat_to_R_i2b_batch(q_wxyz: np.ndarray) -> np.ndarray:
    qxyzw = q_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def quat_angular_dist_deg(q_pool: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    cos_half = np.abs(q_pool @ q_ref)
    cos_half = np.clip(cos_half, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cos_half))


# ---------------------------------------------------------------------------
# Greedy cluster (capped) — same as s050a
# ---------------------------------------------------------------------------


def greedy_cluster(
    qA: np.ndarray,
    om_dps: np.ndarray,
    om_truth_mag_dps: float,
    priority: np.ndarray,
    q_radius_deg: float,
    om_radius_rel: float,
    max_centers: int = 100000,
) -> int:
    n = len(qA)
    if n == 0:
        return 0
    assignment = -np.ones(n, dtype=np.int32)
    sort_idx = np.argsort(priority, kind="stable")
    n_clusters = 0
    for next_seed in sort_idx:
        if assignment[next_seed] >= 0:
            continue
        if n_clusters >= max_centers:
            return -1  # cap hit; caller can handle
        unassigned_mask = assignment < 0
        idx_un = np.where(unassigned_mask)[0]
        cos_half = np.abs(qA[idx_un] @ qA[next_seed])
        cos_half = np.clip(cos_half, 0.0, 1.0)
        q_dist = np.degrees(2.0 * np.arccos(cos_half))
        om_dist = np.linalg.norm(
            om_dps[idx_un] - om_dps[next_seed], axis=1
        ) / om_truth_mag_dps
        within = (q_dist < q_radius_deg) & (om_dist < om_radius_rel)
        member_idx = idx_un[within] if within.any() else np.array([next_seed])
        assignment[member_idx] = n_clusters
        n_clusters += 1
    return n_clusters


def metrics_for_subset(
    keep_idx: np.ndarray,
    truth_qa_mask: np.ndarray,
    qA_kept: np.ndarray,
    om_kept_dps: np.ndarray,
    om_truth_mag_dps: float,
    priority: np.ndarray,
    do_cluster: bool = True,
) -> dict:
    """Compute n_survivors, n_truth_qa_survivors, enrichment, optional cluster count."""
    n_set = int(len(keep_idx))
    if n_set == 0:
        return {
            "n_survivors": 0,
            "n_truth_qa": 0,
            "enrichment": 0.0,
            "cluster_count_5_5": 0,
        }
    n_truth_qa = int(truth_qa_mask[keep_idx].sum())
    expected_random = n_set / N_TOTAL_HYP * N_TRUTH_QA_TOTAL
    enrichment = n_truth_qa / max(expected_random, 1e-9)
    cluster_count = -1
    if do_cluster and n_set <= 50000:
        cluster_count = greedy_cluster(
            qA_kept[keep_idx],
            om_kept_dps[keep_idx],
            om_truth_mag_dps,
            priority[keep_idx],
            CLUSTER_RADIUS_Q_DEG,
            CLUSTER_RADIUS_OM_REL,
        )
    return {
        "n_survivors": n_set,
        "n_truth_qa": n_truth_qa,
        "enrichment": float(enrichment),
        "cluster_count_5_5": cluster_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"[load] cascade: {CASCADE_NPZ}")
    casc = np.load(CASCADE_NPZ)
    qA_kept = casc["qA_kept"]
    om_kept = casc["om_kept"]  # rad/s
    om_kept_dps = np.degrees(om_kept)
    delta_mag_cached = casc["delta_mag"]  # (N, 5)
    val_eps_cached = casc["val_eps"]
    t0_ep = int(casc["t0_ep"])
    delta_t_anchor = float(casc["delta_t"])
    truth_q_a_disc = casc["truth_q_a"]
    om_truth_cascade = casc["om_truth_cascade"]

    # Δk for cached val_eps
    delta_k_cached = (val_eps_cached - t0_ep).astype(int)
    print(
        f"[load] cached val_eps={val_eps_cached.tolist()}  "
        f"Δk={delta_k_cached.tolist()}"
    )

    print(f"[load] traj: {TRAJ_NPZ}")
    traj = np.load(TRAJ_NPZ)
    obs_times = np.asarray(traj["observation_times"], float)
    sun_pos = np.asarray(traj["sun_pos"], float)
    obs_pos = np.asarray(traj["obs_pos"], float)
    sat_pos = np.asarray(traj["sat_pos"], float)
    obs_dist_all = np.asarray(traj["obs_dist"], float)
    mag_hifi = np.asarray(traj["mag_hifi"], float)
    quats_truth = np.asarray(traj["quaternions"], float)

    sun_vec_j2000 = sun_pos - sat_pos
    obs_vec_j2000 = obs_pos - sat_pos
    sun_unit_all = sun_vec_j2000 / np.linalg.norm(
        sun_vec_j2000, axis=1, keepdims=True
    )
    obs_unit_all = obs_vec_j2000 / np.linalg.norm(
        obs_vec_j2000, axis=1, keepdims=True
    )

    # Truth ω at t_0 — taken from cascade.npz (rigid-body propagated)
    om_truth_at_t0 = casc["om_truth_at_t0"]  # rad/s
    om_truth_at_t0_dps = np.degrees(om_truth_at_t0)
    om_truth_at_t0_mag_dps = float(np.linalg.norm(om_truth_at_t0_dps))

    # truth-q_a mask over the 141k pool
    q_dist_to_truth_qa = quat_angular_dist_deg(qA_kept, truth_q_a_disc)
    truth_qa_mask = q_dist_to_truth_qa < 1e-6
    n_truth_qa_in_pool = int(truth_qa_mask.sum())
    assert (
        n_truth_qa_in_pool == N_TRUTH_QA_TOTAL
    ), f"unexpected truth_qa count: {n_truth_qa_in_pool}"
    print(
        f"[truth] truth-q_a hypotheses in pool: {n_truth_qa_in_pool}; "
        f"|ω_truth at t0| = {om_truth_at_t0_mag_dps:.4f} dps"
    )

    # Δmag stats per cached epoch
    print("\n[stage1] cached delta_mag stats per validation epoch:")
    for kk, ep in enumerate(val_eps_cached.tolist()):
        d = delta_mag_cached[:, kk]
        d_truth = delta_mag_cached[truth_qa_mask, kk]
        print(
            f"  ep{ep:3d} Δk={int(ep-t0_ep):+3d} mag={mag_hifi[ep]:.3f}  "
            f"|Δmag| pool p10/p50/p90 = {np.percentile(d, 10):.3f}/"
            f"{np.percentile(d, 50):.3f}/{np.percentile(d, 90):.3f}  "
            f"truth-qa p10/p50/p90 = {np.percentile(d_truth, 10):.3f}/"
            f"{np.percentile(d_truth, 50):.3f}/{np.percentile(d_truth, 90):.3f}"
        )

    # ------------------------------------------------------------------
    # Stage 1 — cached re-scoring grid (no compute)
    # ------------------------------------------------------------------
    print("\n=== Stage 1: cached re-scoring grid ===")

    # Use sum of cached |Δmag| as a stable priority for greedy cluster.
    priority_all = delta_mag_cached.sum(axis=1)

    rescore_results = []

    # K_subset variants from cached 5 epochs:
    #   subset name           subset (cols)         Δk values
    K_SUBSETS = [
        ("K=2 closest (Δk=+1,-2)", [0, 1]),
        ("K=3 local (Δk=+1,-2,+2)", [0, 1, 2]),
        ("K=2 distant (Δk=-7,-8)", [3, 4]),
        ("K=3 mixed (Δk=+1,-2,-7)", [0, 1, 3]),
        ("K=4 mixed (Δk=+1,-2,+2,-7)", [0, 1, 2, 3]),
        ("K=5 all", [0, 1, 2, 3, 4]),
    ]

    for tol in RESCORE_TOLS:
        for label, cols in K_SUBSETS:
            cols_arr = np.array(cols, dtype=int)
            mask = (delta_mag_cached[:, cols_arr] < tol).all(axis=1)
            keep_idx = np.where(mask)[0]
            m = metrics_for_subset(
                keep_idx,
                truth_qa_mask,
                qA_kept,
                om_kept_dps,
                om_truth_at_t0_mag_dps,
                priority_all,
                do_cluster=True,
            )
            rescore_results.append(
                {
                    "tol_mag": tol,
                    "K_subset": label,
                    **m,
                }
            )

    print(
        f"{'tol':>6} {'K_subset':<32} {'n_surv':>8} {'truth_qa':>10} "
        f"{'enrich':>8} {'clusters@5/5':>14}"
    )
    for r in rescore_results:
        print(
            f"{r['tol_mag']:>6.2f} {r['K_subset']:<32} {r['n_survivors']:>8} "
            f"{r['n_truth_qa']:>10} {r['enrichment']:>8.2f} "
            f"{r['cluster_count_5_5']:>14}"
        )

    # ------------------------------------------------------------------
    # Stage 2 — continuous-score rankings
    # ------------------------------------------------------------------
    print("\n=== Stage 2: continuous-score rankings ===")

    # Score variants on the 5 cached epochs
    score_sum_all = delta_mag_cached.sum(axis=1)
    score_max_all = delta_mag_cached.max(axis=1)
    score_sum_local = delta_mag_cached[:, :3].sum(axis=1)
    score_max_local = delta_mag_cached[:, :3].max(axis=1)

    # Inverse-Δk weighting: less weight to far-away epochs (where drift dominates)
    abs_dk = np.abs(delta_k_cached)
    weights_inv = 1.0 / abs_dk
    weights_inv = weights_inv / weights_inv.sum()
    score_invdk_all = (delta_mag_cached * weights_inv).sum(axis=1)

    # Squared-mean
    score_mse_all = (delta_mag_cached ** 2).mean(axis=1)

    score_variants = [
        ("sum_all_5", score_sum_all),
        ("max_all_5", score_max_all),
        ("sum_local_3", score_sum_local),
        ("max_local_3", score_max_local),
        ("invdk_all_5", score_invdk_all),
        ("mse_all_5", score_mse_all),
    ]

    rank_results = []
    for label, scores in score_variants:
        sort_idx = np.argsort(scores, kind="stable")
        for top_n in TOP_N_LIST:
            keep_idx = sort_idx[:top_n]
            m = metrics_for_subset(
                keep_idx,
                truth_qa_mask,
                qA_kept,
                om_kept_dps,
                om_truth_at_t0_mag_dps,
                scores,
                do_cluster=True,
            )
            rank_results.append(
                {"score": label, "top_n": top_n, **m}
            )

    # Print enrichment table for one representative score
    print(
        f"{'score':<14} {'top_n':>8} {'truth_qa':>10} {'enrich':>8} "
        f"{'clusters@5/5':>14}"
    )
    for r in rank_results[: len(TOP_N_LIST) * 2]:
        print(
            f"{r['score']:<14} {r['top_n']:>8} {r['n_truth_qa']:>10} "
            f"{r['enrichment']:>8.2f} {r['cluster_count_5_5']:>14}"
        )

    # ------------------------------------------------------------------
    # Stage 3 — bright-epoch validation (compute)
    # ------------------------------------------------------------------
    print("\n=== Stage 3: bright-epoch validation ===")
    print("[load] surrogate v1 ...")
    v1_weights = SURROGATE_PATH / "surrogate_model" / "s10_5M_weights.npz"
    v1_norm = SURROGATE_PATH / "surrogate_model" / "s10_5M_normalization.npz"
    model = SurrogateV1(str(v1_weights), str(v1_norm))

    # Compute new delta_mag at BRIGHT_EPOCHS_TO_TEST
    n_hyp = len(qA_kept)
    bright_delta_mag = np.zeros((n_hyp, len(BRIGHT_EPOCHS_TO_TEST)), dtype=np.float32)
    bright_meta = []
    for kk, ep in enumerate(BRIGHT_EPOCHS_TO_TEST):
        t_ep = time.time()
        dt_k = obs_times[ep] - obs_times[t0_ep]
        q_pred = constant_omega_propagate(qA_kept, om_kept, dt_k)
        R_pred = quat_to_R_i2b_batch(q_pred)
        k1 = R_pred @ sun_unit_all[ep]
        k2 = R_pred @ obs_unit_all[ep]
        obs_dist_arr = np.full(n_hyp, obs_dist_all[ep])
        pred_mag = model.predict_magnitude(
            k1, k2, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist_arr
        )
        bright_delta_mag[:, kk] = np.abs(pred_mag - mag_hifi[ep]).astype(np.float32)
        d_truth = bright_delta_mag[truth_qa_mask, kk]
        bright_meta.append(
            {
                "ep": int(ep),
                "delta_k": int(ep - t0_ep),
                "mag_measured": float(mag_hifi[ep]),
                "wall_s": float(time.time() - t_ep),
                "pool_p10": float(np.percentile(bright_delta_mag[:, kk], 10)),
                "pool_p50": float(np.percentile(bright_delta_mag[:, kk], 50)),
                "pool_p90": float(np.percentile(bright_delta_mag[:, kk], 90)),
                "truth_qa_p10": float(np.percentile(d_truth, 10)),
                "truth_qa_p50": float(np.percentile(d_truth, 50)),
                "truth_qa_p90": float(np.percentile(d_truth, 90)),
            }
        )
        print(
            f"  ep{ep:3d} Δk={int(ep-t0_ep):+3d} mag={mag_hifi[ep]:.3f}  "
            f"pool |Δmag| p10/p50/p90 = {np.percentile(bright_delta_mag[:, kk], 10):.3f}/"
            f"{np.percentile(bright_delta_mag[:, kk], 50):.3f}/"
            f"{np.percentile(bright_delta_mag[:, kk], 90):.3f}  "
            f"truth-qa p10/p50/p90 = {np.percentile(d_truth, 10):.3f}/"
            f"{np.percentile(d_truth, 50):.3f}/{np.percentile(d_truth, 90):.3f}  "
            f"wall={time.time()-t_ep:.2f}s"
        )

    # Combined (cached dim + bright)
    delta_mag_combined = np.column_stack([delta_mag_cached, bright_delta_mag])
    n_combined_eps = delta_mag_combined.shape[1]
    print(
        f"\n[stage3] combined |Δmag| shape: {delta_mag_combined.shape} "
        f"(5 cached + {len(BRIGHT_EPOCHS_TO_TEST)} bright = {n_combined_eps})"
    )

    # Score-by-sum on combined; also a tol-filter on bright epochs alone.
    score_sum_combined = delta_mag_combined.sum(axis=1)
    score_sum_bright = bright_delta_mag.sum(axis=1)
    score_max_bright = bright_delta_mag.max(axis=1)

    bright_results = []
    # score-by-sum: combined / bright-only
    for label, scores in [
        ("sum_combined", score_sum_combined),
        ("sum_bright_only", score_sum_bright),
        ("max_bright_only", score_max_bright),
    ]:
        sort_idx = np.argsort(scores, kind="stable")
        for top_n in TOP_N_LIST:
            keep_idx = sort_idx[:top_n]
            m = metrics_for_subset(
                keep_idx,
                truth_qa_mask,
                qA_kept,
                om_kept_dps,
                om_truth_at_t0_mag_dps,
                scores,
                do_cluster=True,
            )
            bright_results.append({"score": label, "top_n": top_n, **m})

    # Tol-filter on bright epochs (any K of len(BRIGHT_EPOCHS_TO_TEST))
    for tol in [0.10, 0.20, 0.50, 1.00]:
        for K_req in [1, 2, 3, len(BRIGHT_EPOCHS_TO_TEST)]:
            if K_req > len(BRIGHT_EPOCHS_TO_TEST):
                continue
            pass_per_ep = bright_delta_mag < tol
            mask = pass_per_ep[:, :K_req].all(axis=1)
            keep_idx = np.where(mask)[0]
            m = metrics_for_subset(
                keep_idx,
                truth_qa_mask,
                qA_kept,
                om_kept_dps,
                om_truth_at_t0_mag_dps,
                priority_all,
                do_cluster=True,
            )
            bright_results.append(
                {"score": f"bright_tol{tol:.2f}_K{K_req}", "top_n": -1, **m}
            )

    # Combined dim+bright threshold filter:
    # require dim-K=3 (the s049 default) AND bright-K=N_bright at varying tol.
    dim_pass = (delta_mag_cached[:, :3] < 0.10).all(axis=1)
    for bright_tol in [0.10, 0.20, 0.50, 1.00]:
        for bright_K in [1, 2, 3, len(BRIGHT_EPOCHS_TO_TEST)]:
            if bright_K > len(BRIGHT_EPOCHS_TO_TEST):
                continue
            bright_pass = (bright_delta_mag[:, :bright_K] < bright_tol).all(axis=1)
            mask = dim_pass & bright_pass
            keep_idx = np.where(mask)[0]
            m = metrics_for_subset(
                keep_idx,
                truth_qa_mask,
                qA_kept,
                om_kept_dps,
                om_truth_at_t0_mag_dps,
                priority_all,
                do_cluster=True,
            )
            bright_results.append(
                {
                    "score": f"dim010_K3_AND_bright{bright_tol:.2f}_K{bright_K}",
                    "top_n": -1,
                    **m,
                }
            )

    print("\n[stage3] combined-score rankings (top results by score):")
    print(
        f"{'score':<35} {'top_n':>8} {'n_surv':>8} {'truth_qa':>10} "
        f"{'enrich':>8} {'clusters@5/5':>14}"
    )
    for r in bright_results:
        print(
            f"{r['score']:<35} {r['top_n']:>8} {r['n_survivors']:>8} "
            f"{r['n_truth_qa']:>10} {r['enrichment']:>8.2f} "
            f"{r['cluster_count_5_5']:>14}"
        )

    # ------------------------------------------------------------------
    # Stage 4 — Pareto plot
    # ------------------------------------------------------------------
    print("\n=== Stage 4: Pareto plot ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Pareto: Stage 1 (cached re-score) — n_survivors vs enrichment
    ax = axes[0]
    for label, _ in K_SUBSETS:
        xs = [
            r["n_survivors"] for r in rescore_results if r["K_subset"] == label
        ]
        ys = [r["enrichment"] for r in rescore_results if r["K_subset"] == label]
        ax.plot(xs, ys, "o-", label=label, alpha=0.7)
    ax.axhline(1, color="grey", ls="--", lw=1, label="random baseline")
    ax.set_xscale("log")
    ax.set_xlabel("# survivors")
    ax.set_ylabel("truth-q_a enrichment factor")
    ax.set_title("Stage 1: cached re-score (varying tol per K-subset)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    # Pareto: Stage 2 (continuous score top-N) — n_survivors vs enrichment
    ax = axes[1]
    for label, _ in score_variants:
        xs = [r["n_survivors"] for r in rank_results if r["score"] == label]
        ys = [r["enrichment"] for r in rank_results if r["score"] == label]
        ax.plot(xs, ys, "o-", label=label, alpha=0.7)
    ax.axhline(1, color="grey", ls="--", lw=1, label="random baseline")
    ax.set_xscale("log")
    ax.set_xlabel("# survivors (top-N)")
    ax.set_ylabel("truth-q_a enrichment factor")
    ax.set_title("Stage 2: continuous score top-N rankings")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    # Pareto: Stage 3 — bright-epoch results
    ax = axes[2]
    bright_score_labels = ["sum_combined", "sum_bright_only", "max_bright_only"]
    for label in bright_score_labels:
        xs = [
            r["n_survivors"] for r in bright_results if r["score"] == label
        ]
        ys = [r["enrichment"] for r in bright_results if r["score"] == label]
        ax.plot(xs, ys, "o-", label=label, alpha=0.7)
    # Add the tol-filter points as scatter
    for r in bright_results:
        if r["score"].startswith("bright_tol") or r["score"].startswith("dim010"):
            ax.scatter(
                [r["n_survivors"]],
                [r["enrichment"]],
                marker="x" if "dim010" in r["score"] else "+",
                s=60,
                alpha=0.6,
            )
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("# survivors")
    ax.set_ylabel("truth-q_a enrichment factor")
    ax.set_title("Stage 3: bright-epoch validation (Δk = -15..-13, +13..+15)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"s050b — validation stress test, seed 14 cascade pool (n=141706)",
        fontsize=14,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "phase2_pareto.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[saved] {out_png}")

    # Δmag distribution for truth-qa vs pool — diagnostic
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for kk, ep in enumerate(val_eps_cached.tolist()):
        ax = axes[0, kk] if kk < 5 else None
        if ax is None:
            continue
        d_pool = delta_mag_cached[:, kk]
        d_truth = delta_mag_cached[truth_qa_mask, kk]
        ax.hist(
            d_pool,
            bins=50,
            range=(0, 1.0),
            alpha=0.5,
            color="C0",
            label="pool (141k)",
            density=True,
        )
        ax.hist(
            d_truth,
            bins=50,
            range=(0, 1.0),
            alpha=0.7,
            color="C3",
            label="truth-qa (153)",
            density=True,
        )
        ax.set_title(f"cached ep{ep} Δk={int(ep-t0_ep):+d} mag={mag_hifi[ep]:.2f}")
        ax.set_xlabel("|Δmag|")
        ax.legend(fontsize=8)
    for kk, ep in enumerate(BRIGHT_EPOCHS_TO_TEST):
        ax = axes[1, kk] if kk < 5 else None
        if ax is None:
            continue
        d_pool = bright_delta_mag[:, kk]
        d_truth = bright_delta_mag[truth_qa_mask, kk]
        max_x = max(np.percentile(d_pool, 95), 2.0)
        ax.hist(
            d_pool,
            bins=50,
            range=(0, max_x),
            alpha=0.5,
            color="C0",
            label="pool (141k)",
            density=True,
        )
        ax.hist(
            d_truth,
            bins=50,
            range=(0, max_x),
            alpha=0.7,
            color="C3",
            label="truth-qa (153)",
            density=True,
        )
        ax.set_title(
            f"bright ep{ep} Δk={int(ep-t0_ep):+d} mag={mag_hifi[ep]:.2f}"
        )
        ax.set_xlabel("|Δmag|")
        ax.legend(fontsize=8)
    plt.suptitle(
        "Δmag distribution per validation epoch — pool (blue) vs truth-q_a (red)",
        fontsize=14,
    )
    plt.tight_layout()
    out_png_dist = OUT_DIR / "phase2_dmag_distributions.png"
    plt.savefig(out_png_dist, dpi=110)
    plt.close()
    print(f"[saved] {out_png_dist}")

    # ------------------------------------------------------------------
    # Stage 5 — save artefacts
    # ------------------------------------------------------------------
    np.savez(
        OUT_DIR / "stress.npz",
        bright_epochs=np.array(BRIGHT_EPOCHS_TO_TEST),
        bright_delta_mag=bright_delta_mag,
        truth_qa_mask=truth_qa_mask,
        score_sum_combined=score_sum_combined,
        score_sum_local=score_sum_local,
    )
    print(f"[saved] {OUT_DIR / 'stress.npz'}")

    summary = {
        "seed": SEED,
        "n_total_hyp": N_TOTAL_HYP,
        "n_truth_qa_total": N_TRUTH_QA_TOTAL,
        "om_truth_at_t0_mag_dps": om_truth_at_t0_mag_dps,
        "cached_val_eps": val_eps_cached.tolist(),
        "cached_delta_k": delta_k_cached.tolist(),
        "bright_epochs": BRIGHT_EPOCHS_TO_TEST,
        "bright_meta": bright_meta,
        "rescore_results": rescore_results,
        "rank_results": rank_results,
        "bright_results": bright_results,
        "wall_total_s": float(time.time() - t_start),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[saved] {OUT_DIR / 'summary.json'}")
    print(f"[done] wall = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
