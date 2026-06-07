"""s050e — q-space agreement score on cascade hypotheses.

For each of the 141k cascade hypotheses (q_a, ω): propagate to K validation
t_k, find the nearest q_b in Q_k (3D quaternion-space distance), score by
mean angular distance over K. Q-space agreement uses the full 3D constraint
per epoch instead of mag-agreement's 1D projection — 2 extra DOF of
discrimination per validation epoch.

If this discriminates truth from coincidence, it's the right validation
metric for the cascade architecture.
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

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"

CASCADE_NPZ = SURVEY_DIR / "results" / "s049_cascade_seed14" / "cascade.npz"
SCAN_NPZ = SURVEY_DIR / "results" / "s049_cascade_seed14" / "scan.npz"
TRAJ_NPZ = (
    PROJECT_ROOT
    / "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed014.npz"
)
OUT_DIR = SURVEY_DIR / "results" / "s050e_qspace_score"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_TIGHT_TK = 5


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


def quat_angular_dist_deg(q_pool: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    cos_half = np.abs(q_pool @ q_ref)
    cos_half = np.clip(cos_half, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cos_half))


def nearest_q_dist_to_set_batch(
    q_pred_batch: np.ndarray, q_target_set: np.ndarray
) -> np.ndarray:
    """Vectorized nearest-neighbor angular distance.

    q_pred_batch: (N, 4)
    q_target_set: (M, 4)
    Returns: (N,) array of min angular distance in degrees.
    """
    # |dot(q_pred, q_target)| with antipodal handling
    # Shape (N, M)
    dots = np.abs(q_pred_batch @ q_target_set.T)
    dots = np.clip(dots, 0.0, 1.0)
    max_dot_per_pred = dots.max(axis=1)
    return np.degrees(2.0 * np.arccos(max_dot_per_pred))


def main() -> None:
    t_start = time.time()

    # Load
    print(f"[load] cascade: {CASCADE_NPZ}")
    casc = np.load(CASCADE_NPZ)
    qA_kept = casc["qA_kept"]  # (141706, 4)
    om_kept = casc["om_kept"]  # rad/s
    delta_mag_cached = casc["delta_mag"]  # (141706, 5) — for comparison
    truth_q_a = casc["truth_q_a"]
    om_truth_at_t0 = casc["om_truth_at_t0"]
    om_truth_dps = np.degrees(om_truth_at_t0)
    om_truth_mag_dps = float(np.linalg.norm(om_truth_dps))
    t0_ep = int(casc["t0_ep"])

    print(f"[load] scan: {SCAN_NPZ}")
    scan = np.load(SCAN_NPZ)
    q_pool = scan["q_pool_wxyz"]
    scan_eps = scan["scan_epochs"]
    survive_mask = scan["survive_mask"]
    n_surv_per_ep = scan["n_survivors"]
    obs_times_at_scan = scan["obs_times_at_scan"]

    # Truth-q_a mask in pool
    truth_qa_mask = np.all(qA_kept == truth_q_a, axis=1)
    n_truth_qa = int(truth_qa_mask.sum())
    n_total = len(qA_kept)
    print(
        f"[truth] truth-q_a hypotheses in pool: {n_truth_qa}/{n_total} ({100*n_truth_qa/n_total:.2f}%)"
    )

    # Pick K tightest non-anchor scan epochs as t_k
    idx_t0 = int(np.where(scan_eps == t0_ep)[0][0])
    sort_idx = np.argsort(n_surv_per_ep)
    chosen_meta = []
    for i in sort_idx:
        i = int(i)
        if i == idx_t0 or n_surv_per_ep[i] == 0:
            continue
        ep = int(scan_eps[i])
        Qk = q_pool[survive_mask[i]]
        dt_k = float(obs_times_at_scan[i] - obs_times_at_scan[idx_t0])
        chosen_meta.append((i, ep, ep - t0_ep, Qk, dt_k))
        print(
            f"  t_k ep={ep} Δk={ep-t0_ep:+d} |Q_k|={len(Qk)} Δt={dt_k:+.2f}s"
        )
        if len(chosen_meta) == K_TIGHT_TK:
            break

    # ------------------------------------------------------------------
    # For each (q_a, ω) in cascade pool, compute q-space agreement at each t_k.
    # Score by mean nearest-neighbor angular distance over K.
    # ------------------------------------------------------------------
    print(f"\n[qspace] computing per-hypothesis q-space score over {K_TIGHT_TK} t_k's")
    qspace_dist = np.zeros((n_total, K_TIGHT_TK), dtype=np.float32)
    BATCH = 10000  # tune for memory vs speed
    for kk, (idx_k, ep, dk, Qk, dt_k) in enumerate(chosen_meta):
        t_k_start = time.time()
        # Propagate all hypotheses to t_k
        for i0 in range(0, n_total, BATCH):
            i1 = min(i0 + BATCH, n_total)
            q_pred = constant_omega_propagate(qA_kept[i0:i1], om_kept[i0:i1], dt_k)
            qspace_dist[i0:i1, kk] = nearest_q_dist_to_set_batch(q_pred, Qk).astype(
                np.float32
            )
        print(
            f"  t_k ep{ep} done  wall={time.time()-t_k_start:.1f}s  "
            f"dist p10/p50/p90 pool = "
            f"{np.percentile(qspace_dist[:, kk], 10):.2f}°/"
            f"{np.percentile(qspace_dist[:, kk], 50):.2f}°/"
            f"{np.percentile(qspace_dist[:, kk], 90):.2f}°  "
            f"truth-qa p10/p50/p90 = "
            f"{np.percentile(qspace_dist[truth_qa_mask, kk], 10):.2f}°/"
            f"{np.percentile(qspace_dist[truth_qa_mask, kk], 50):.2f}°/"
            f"{np.percentile(qspace_dist[truth_qa_mask, kk], 90):.2f}°"
        )

    # Scoring
    score_mean = qspace_dist.mean(axis=1)
    score_max = qspace_dist.max(axis=1)
    score_sum = qspace_dist.sum(axis=1)

    # ------------------------------------------------------------------
    # Discriminator analysis
    # ------------------------------------------------------------------
    print("\n=== q-space score discriminator analysis ===")
    truth_qa_idx_in_pool = np.where(truth_qa_mask)[0]
    print(
        f"Truth-q_a hypotheses (153 of them); ranking by score_mean (lower=better):"
    )
    sort_pool = np.argsort(score_mean)
    pool_p10 = float(np.percentile(score_mean, 10))
    pool_p50 = float(np.percentile(score_mean, 50))
    pool_p90 = float(np.percentile(score_mean, 90))
    truth_p10 = float(np.percentile(score_mean[truth_qa_mask], 10))
    truth_p50 = float(np.percentile(score_mean[truth_qa_mask], 50))
    truth_p90 = float(np.percentile(score_mean[truth_qa_mask], 90))
    print(
        f"  pool score_mean p10/p50/p90 = {pool_p10:.2f}°/{pool_p50:.2f}°/{pool_p90:.2f}°"
    )
    print(
        f"  truth-qa score_mean p10/p50/p90 = {truth_p10:.2f}°/{truth_p50:.2f}°/{truth_p90:.2f}°"
    )

    # Truth-qa enrichment in top-N
    top_n_results = []
    for top_n in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 35835]:
        keep_idx = sort_pool[:top_n]
        n_truth = int(truth_qa_mask[keep_idx].sum())
        expected = top_n / n_total * n_truth_qa
        enrich = n_truth / max(expected, 1e-9)
        top_n_results.append({
            "top_n": top_n,
            "n_truth_qa": n_truth,
            "enrichment": float(enrich),
        })
        print(
            f"  top-{top_n:>5d} by score_mean: truth-qa={n_truth:>3d}  "
            f"enrichment={enrich:.2f}× (expected at random {expected:.1f})"
        )

    # Best truth-qa hypothesis
    best_truth_in_pool = sort_pool[truth_qa_mask[sort_pool]][0] if truth_qa_mask[sort_pool].any() else None
    if best_truth_in_pool is not None:
        rank = int(np.where(sort_pool == best_truth_in_pool)[0][0]) + 1
        print(
            f"\n  best truth-qa hypothesis ranks {rank}/{n_total} "
            f"({100*rank/n_total:.3f}%) by score_mean"
        )
        print(f"  best truth-qa score_mean = {score_mean[best_truth_in_pool]:.2f}°")

    # Compare to mag-agreement baseline
    score_mag_sum = delta_mag_cached.sum(axis=1)
    sort_mag = np.argsort(score_mag_sum)
    print("\n  comparison to s050b mag-agreement baseline (sum of cached |Δmag|):")
    for top_n in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        keep_qspace = sort_pool[:top_n]
        keep_mag = sort_mag[:top_n]
        n_q = int(truth_qa_mask[keep_qspace].sum())
        n_m = int(truth_qa_mask[keep_mag].sum())
        e_q = n_q / max(top_n / n_total * n_truth_qa, 1e-9)
        e_m = n_m / max(top_n / n_total * n_truth_qa, 1e-9)
        print(
            f"    top-{top_n:>5d}  q-space: {n_q:>3d} truth-qa (enrich {e_q:.2f}×)   "
            f"mag-only: {n_m:>3d} truth-qa (enrich {e_m:.2f}×)"
        )

    # Plot: comparison of pool vs truth-qa score distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: score_mean histogram
    ax = axes[0]
    bins = np.linspace(0, 90, 60)
    ax.hist(score_mean, bins=bins, alpha=0.5, color="C0", label="pool (141k)", density=True)
    ax.hist(
        score_mean[truth_qa_mask],
        bins=bins,
        alpha=0.7,
        color="C3",
        label="truth-q_a (153)",
        density=True,
    )
    ax.set_xlabel("score_mean (mean q-dist over K=5 t_k, deg)")
    ax.set_ylabel("density")
    ax.set_title("q-space score: pool vs truth-qa")
    ax.legend()
    ax.set_yscale("log")

    # Panel 2: per-t_k q-dist for truth-qa vs pool
    ax = axes[1]
    bins = np.linspace(0, 90, 50)
    for kk in range(K_TIGHT_TK):
        ax.hist(
            qspace_dist[truth_qa_mask, kk],
            bins=bins,
            alpha=0.5,
            histtype="step",
            label=f"t_k Δk={chosen_meta[kk][2]:+d} (truth-qa)",
            density=True,
        )
    ax.set_xlabel("q-dist to nearest q_b in Q_k (deg)")
    ax.set_ylabel("density")
    ax.set_title("Per-t_k q-dist for truth-q_a hypotheses (153)")
    ax.legend(fontsize=8)

    # Panel 3: enrichment factor vs top-N
    ax = axes[2]
    top_ns = [r["top_n"] for r in top_n_results]
    enrichs_q = [r["enrichment"] for r in top_n_results]
    # mag-baseline
    enrichs_m = []
    for top_n in top_ns:
        keep = sort_mag[:top_n]
        n_truth = int(truth_qa_mask[keep].sum())
        expected = top_n / n_total * n_truth_qa
        enrichs_m.append(n_truth / max(expected, 1e-9))
    ax.plot(top_ns, enrichs_q, "o-", label="q-space (s050e)", lw=2)
    ax.plot(top_ns, enrichs_m, "s-", label="mag-only (s050b baseline)", lw=2)
    ax.axhline(1, color="grey", ls="--", lw=1, label="random baseline")
    ax.set_xscale("log")
    ax.set_xlabel("top-N hypotheses by score")
    ax.set_ylabel("truth-q_a enrichment factor")
    ax.set_title("q-space vs mag-agreement enrichment")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"s050e — q-space agreement score (seed 14, K={K_TIGHT_TK} t_k)",
        fontsize=14,
    )
    plt.tight_layout()
    out_png = OUT_DIR / "qspace_score.png"
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[saved] {out_png}")

    np.savez(
        OUT_DIR / "qspace_score.npz",
        qspace_dist=qspace_dist,
        score_mean=score_mean,
        score_max=score_max,
        score_sum=score_sum,
        truth_qa_mask=truth_qa_mask,
        chosen_t_k_eps=np.array([m[1] for m in chosen_meta]),
    )
    print(f"[saved] {OUT_DIR / 'qspace_score.npz'}")

    summary = {
        "seed": 14,
        "K_tight_tk": K_TIGHT_TK,
        "n_total": n_total,
        "n_truth_qa": n_truth_qa,
        "chosen_t_k_eps": [m[1] for m in chosen_meta],
        "chosen_t_k_dks": [m[2] for m in chosen_meta],
        "top_n_results_qspace": top_n_results,
        "score_mean_pool": {
            "p10": pool_p10, "p50": pool_p50, "p90": pool_p90,
        },
        "score_mean_truth_qa": {
            "p10": truth_p10, "p50": truth_p50, "p90": truth_p90,
        },
        "wall_total_s": float(time.time() - t_start),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[saved] {OUT_DIR / 'summary.json'}")
    print(f"[done] wall = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
