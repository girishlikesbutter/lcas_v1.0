"""s059f — diagnose why forward-prop score doesn't rank truth highly.

The cloud-data architecture's score = Σ_v weight_v · indicator(propagated q
hits some survivor in C_v within HIT_THRESHOLD). Truth ranks at top-0.8%
by raw score (rank 1502/184k for seed 28 T_A=312, 7602/932k for T_A=25)
and top-1.5% by cluster sum-score — too low to catch with a top-5 polish
without an oracle injection.

This diagnostic asks: what's ABOVE truth by score, and WHY?

Direct inspections (no LM, no hi-fi):
  D1. Score histogram for all candidates; mark truth's score percentile.
  D2. Scatter of (qa_dist_to_truth, score). Are top-scorers near-truth in q?
  D3. Scatter of (om_dist_to_truth, score). Are top-scorers near-truth in ω?
  D4. Joint (qa_dist + om_dist, score). Look for "near-truth dominates".
  D5. Body-twin canonicalised: same plot post-canonical_batch. Does dedup
      lift truth's rank? If yes → twins are inflating non-truth scores.
  D6. Top-20 candidates: print their (qa_d_to_truth, om_d_to_truth, |ω|_diff,
      score, n_validator_hits). Compare to truth's row.
  D7. Per-validator hit pattern: for top-5 vs truth, which validators do they
      hit? Are they hitting the same informative validators (signal) or
      different small-cloud validators (noise lottery)?
  D8. Score component decomposition: fraction of score from large-weight
      validators (small |C_v|) vs small-weight (large |C_v|). Is the score
      a lottery on a few small-cloud validators?

Usage:
    python experiments/s059f_score_diagnostic.py --seed 28 --t-a 312
    python experiments/s059f_score_diagnostic.py --seed 28 --t-a 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s059_pilot import (
    fd_omega_passive, propagate_const_omega, quat_ang_deg_batch, ang_to_axis,
    DELTA_GEN, PRIOR_BRACKET, HIT_THRESHOLD_DEG, CONST_OMEGA_RELIABLE_MAX_DEG,
    SMOKE_DELTAS,
)
from lib.twin import canonical_batch
from lib.traj_load import load_truth as _load_traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--t-a", type=int, required=True, help="anchor epoch")
    p.add_argument("--out-root", default=str(SURVEY / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_root) / f"s059f_seed{args.seed:03d}_T_A{args.t_a:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"=== s059f score diagnostic — seed {args.seed} T_A={args.t_a} ===")
    cloud_dir = SURVEY / "results" / f"s059_seed{args.seed:03d}"
    cache = cloud_dir / "cloud.npz"
    z = np.load(cache)
    q_pool = z["q_pool_wxyz"]
    survive_all = z["survive_all"]
    obs_times = z["obs_times"]
    n_epochs = len(obs_times)
    dt_epoch = float(np.median(np.diff(obs_times)))
    log(f"cloud: {q_pool.shape[0]} pool × {n_epochs} epochs (cv min={survive_all.sum(1).min()})")

    truth = _load_traj(args.seed)
    q_truth_t = truth["quaternions"]
    T_A = args.t_a

    # truth ω at T_A from finite-diff
    om_truth = fd_omega_passive(
        q_truth_t[T_A:T_A+1], q_truth_t[T_A+1:T_A+2], dt_epoch)[0]
    om_truth_mag = float(np.linalg.norm(om_truth))
    log(f"truth at T_A: q={q_truth_t[T_A]}, |ω|={om_truth_mag:.6f} rad/s")

    # validators
    valid_max_delta = 1
    for d in SMOKE_DELTAS:
        if T_A + d >= n_epochs: continue
        q_pred = propagate_const_omega(q_truth_t[T_A], om_truth, d * dt_epoch)
        from experiments.s059_pilot import quat_ang_deg as _q
        err = _q(q_pred, q_truth_t[T_A + d])
        if err <= CONST_OMEGA_RELIABLE_MAX_DEG:
            valid_max_delta = d
        else:
            break
    log(f"const-ω reliable to Δ = {valid_max_delta} epochs")
    validators = []
    lo_v = max(0, T_A - valid_max_delta)
    hi_v = min(n_epochs, T_A + valid_max_delta + 1)
    for t_v in range(lo_v, hi_v):
        if t_v == T_A: continue
        idx_v = np.where(survive_all[t_v])[0]
        if len(idx_v) == 0: continue
        validators.append({
            "delta": t_v - T_A, "C_v": q_pool[idx_v],
            "n_surv": int(len(idx_v)),
            "weight": float(np.log(100000.0 / len(idx_v))),
        })
    log(f"validators: {len(validators)}, |C_v| range "
        f"[{min(v['n_surv'] for v in validators)}, "
        f"{max(v['n_surv'] for v in validators)}]")

    # candidate generation
    Δgen_t_b = T_A + DELTA_GEN
    if Δgen_t_b >= n_epochs:
        Δgen_t_b = T_A - DELTA_GEN
        Δt_gen = -DELTA_GEN * dt_epoch
    else:
        Δt_gen = DELTA_GEN * dt_epoch
    idx_a = np.where(survive_all[T_A])[0]
    idx_b = np.where(survive_all[Δgen_t_b])[0]
    C_a = q_pool[idx_a]
    C_b = q_pool[idx_b]
    n_a, n_b = len(C_a), len(C_b)
    Q_A = np.repeat(C_a, n_b, axis=0)
    Q_B = np.tile(C_b, (n_a, 1))
    om_all = fd_omega_passive(Q_A, Q_B, Δt_gen)
    om_mag_all = np.linalg.norm(om_all, axis=1)
    mask = (om_mag_all >= om_truth_mag * PRIOR_BRACKET[0]) & \
           (om_mag_all <= om_truth_mag * PRIOR_BRACKET[1])
    Q_A_pass = Q_A[mask]
    om_pass = om_all[mask]
    n_cand = len(Q_A_pass)
    log(f"candidates: {n_cand}")

    # score (chunked)
    scores = np.zeros(n_cand)
    hits_per_cand = np.zeros((n_cand, len(validators)), dtype=bool)
    cos_thresh = float(np.cos(np.radians(HIT_THRESHOLD_DEG / 2.0)))
    CHUNK = 10_000
    log("scoring...")
    for vi, v in enumerate(validators):
        Δt = v["delta"] * dt_epoch
        C_vT = v["C_v"].T
        weight = v["weight"]
        for s in range(0, n_cand, CHUNK):
            e = min(s + CHUNK, n_cand)
            q_pred_chunk = propagate_const_omega(
                Q_A_pass[s:e], om_pass[s:e], Δt)
            max_dot = np.abs(q_pred_chunk @ C_vT).max(axis=1)
            hit_chunk = max_dot > cos_thresh
            hits_per_cand[s:e, vi] = hit_chunk
            scores[s:e] += weight * hit_chunk
    log(f"score range [{scores.min():.2f}, {scores.max():.2f}]")

    # truth distance per candidate
    qa_dist = quat_ang_deg_batch(Q_A_pass, q_truth_t[T_A])
    om_dist = ang_to_axis(om_pass, om_truth)
    om_mag_pct = (om_mag_all[mask] - om_truth_mag) / om_truth_mag * 100

    truth_idx = int(np.argmin(qa_dist + om_dist))
    truth_score = float(scores[truth_idx])
    truth_rank = int((scores > truth_score).sum()) + 1
    log(f"truth-candidate idx={truth_idx}: qa_d={qa_dist[truth_idx]:.2f}°, "
        f"ω_d={om_dist[truth_idx]:.2f}°, |ω|Δ={om_mag_pct[truth_idx]:+.2f}%, "
        f"score={truth_score:.2f}, rank {truth_rank}/{n_cand}")
    log(f"truth hits {hits_per_cand[truth_idx].sum()}/{len(validators)} validators")

    # Top-20 candidates by score
    top20_idx = np.argsort(-scores)[:20]
    log("\n=== TOP 20 by score ===")
    log(f"{'rank':<5}{'idx':<10}{'qa_d°':<10}{'om_d°':<10}{'|ω|Δ%':<10}{'score':<10}{'n_hits':<8}{'is_twin':<10}")
    twin_q_truth = np.array([q_truth_t[T_A][0], -q_truth_t[T_A][1],
                              q_truth_t[T_A][2], q_truth_t[T_A][3]])  # 180x flip
    # actually body-twin convention: q_180x · q. Apply via quaternion mult.
    # easier: canonicalise top-20 + truth via canonical_batch and see if collapsing
    Q_top_canon, om_top_canon = canonical_batch(
        Q_A_pass[top20_idx], om_pass[top20_idx])
    q_truth_canon, om_truth_canon = canonical_batch(
        q_truth_t[T_A:T_A+1], om_truth.reshape(1, 3))
    qa_dist_canon = quat_ang_deg_batch(Q_top_canon, q_truth_canon[0])
    om_dist_canon = ang_to_axis(om_top_canon, om_truth_canon[0])
    for r, idx in enumerate(top20_idx):
        is_twin = "TWIN" if abs(qa_dist[idx] - 180) < 10 and qa_dist_canon[r] < 15 else ""
        is_truth = " (TRUTH)" if idx == truth_idx else ""
        log(f"{r+1:<5}{idx:<10}{qa_dist[idx]:<10.2f}{om_dist[idx]:<10.2f}"
            f"{om_mag_pct[idx]:<10.2f}{scores[idx]:<10.2f}"
            f"{int(hits_per_cand[idx].sum()):<8}{is_twin:<10}{is_truth}")
    log(f"\ncanonical (twin-collapsed) qa_d for top-20:")
    for r in range(min(10, len(top20_idx))):
        log(f"  rank {r+1}: raw qa_d={qa_dist[top20_idx[r]]:.2f}°  "
            f"canonical qa_d={qa_dist_canon[r]:.2f}°  "
            f"canonical om_d={om_dist_canon[r]:.2f}°")

    # truth's hit pattern vs top-1's hit pattern
    log("\n=== Per-validator hit pattern: truth vs top-1 vs top-3 ===")
    log(f"{'val#':<6}{'Δ':<5}{'|C_v|':<7}{'wt':<7}{'truth':<8}{'top1':<8}{'top2':<8}{'top3':<8}")
    for vi, v in enumerate(validators):
        log(f"{vi:<6}{v['delta']:<5}{v['n_surv']:<7}{v['weight']:<7.3f}"
            f"{int(hits_per_cand[truth_idx, vi]):<8}"
            f"{int(hits_per_cand[top20_idx[0], vi]):<8}"
            f"{int(hits_per_cand[top20_idx[1], vi]):<8}"
            f"{int(hits_per_cand[top20_idx[2], vi]):<8}")

    # score component breakdown: split validators into tiers by weight
    # tier 1: smallest |C_v| (informative), tier 4: largest |C_v| (noise)
    weights_arr = np.array([v["weight"] for v in validators])
    weight_quartiles = np.quantile(weights_arr, [0.25, 0.50, 0.75])
    tier_masks = [
        weights_arr >= weight_quartiles[2],  # top 25% weight (smallest |C_v|)
        (weights_arr >= weight_quartiles[1]) & (weights_arr < weight_quartiles[2]),
        (weights_arr >= weight_quartiles[0]) & (weights_arr < weight_quartiles[1]),
        weights_arr < weight_quartiles[0],   # bottom 25% weight (largest |C_v|)
    ]
    log("\n=== Score breakdown by validator weight tier ===")
    log(f"{'tier':<8}{'wt_range':<22}{'n_val':<8}{'truth_score':<14}{'top1_score':<14}{'top1/truth':<12}")
    for ti, m in enumerate(tier_masks):
        wt_min = float(weights_arr[m].min()) if m.sum() else 0
        wt_max = float(weights_arr[m].max()) if m.sum() else 0
        truth_tier_score = float(np.sum(weights_arr[m] * hits_per_cand[truth_idx, m]))
        top1_tier_score = float(np.sum(weights_arr[m] * hits_per_cand[top20_idx[0], m]))
        log(f"{ti+1:<8}[{wt_min:5.2f}, {wt_max:5.2f}]      {int(m.sum()):<8}"
            f"{truth_tier_score:<14.3f}{top1_tier_score:<14.3f}"
            f"{top1_tier_score/max(truth_tier_score, 1e-6):<12.2f}")

    # Plots
    log("\nplotting...")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    # qa_dist vs score
    ax[0, 0].scatter(qa_dist, scores, s=1, alpha=0.2, c="gray")
    ax[0, 0].scatter([qa_dist[truth_idx]], [truth_score], s=80, c="red",
                      marker="*", label="truth", zorder=5)
    ax[0, 0].scatter(qa_dist[top20_idx[:5]], scores[top20_idx[:5]],
                      s=40, c="blue", marker="o", label="top-5", zorder=4)
    ax[0, 0].set_xlabel("qa_dist to truth (°)")
    ax[0, 0].set_ylabel("score")
    ax[0, 0].legend(); ax[0, 0].set_title(f"score vs qa_d (seed {args.seed} T_A={T_A})")

    ax[0, 1].scatter(om_dist, scores, s=1, alpha=0.2, c="gray")
    ax[0, 1].scatter([om_dist[truth_idx]], [truth_score], s=80, c="red", marker="*", label="truth", zorder=5)
    ax[0, 1].scatter(om_dist[top20_idx[:5]], scores[top20_idx[:5]], s=40, c="blue", marker="o", label="top-5", zorder=4)
    ax[0, 1].set_xlabel("ω_dir distance to truth (°)")
    ax[0, 1].set_ylabel("score")
    ax[0, 1].legend(); ax[0, 1].set_title("score vs ω_dir distance")

    ax[1, 0].scatter(om_mag_pct, scores, s=1, alpha=0.2, c="gray")
    ax[1, 0].scatter([om_mag_pct[truth_idx]], [truth_score], s=80, c="red", marker="*", label="truth", zorder=5)
    ax[1, 0].scatter(om_mag_pct[top20_idx[:5]], scores[top20_idx[:5]], s=40, c="blue", marker="o", label="top-5", zorder=4)
    ax[1, 0].set_xlabel("|ω| relative error (%)")
    ax[1, 0].set_ylabel("score")
    ax[1, 0].legend(); ax[1, 0].set_title("score vs |ω| error")

    ax[1, 1].hist(scores, bins=50, log=True, color="gray", alpha=0.6)
    ax[1, 1].axvline(truth_score, c="red", lw=2, label=f"truth ({truth_score:.1f})")
    for s in scores[top20_idx[:5]]:
        ax[1, 1].axvline(s, c="blue", lw=0.5, alpha=0.5)
    ax[1, 1].set_xlabel("score"); ax[1, 1].set_ylabel("count (log)")
    ax[1, 1].legend(); ax[1, 1].set_title("score histogram")

    plt.tight_layout()
    fig_path = out_dir / "score_diagnostic.png"
    fig.savefig(fig_path, dpi=150)
    log(f"Saved: {fig_path}")

    summary = {
        "seed": args.seed, "T_A": T_A,
        "n_candidates": int(n_cand),
        "n_validators": len(validators),
        "truth_score": truth_score,
        "truth_rank": truth_rank,
        "truth_qa_d_t_a": float(qa_dist[truth_idx]),
        "truth_om_d_t_a": float(om_dist[truth_idx]),
        "truth_om_mag_err_pct": float(om_mag_pct[truth_idx]),
        "truth_n_validators_hit": int(hits_per_cand[truth_idx].sum()),
        "top20_idx": top20_idx.tolist(),
        "top20_qa_d": qa_dist[top20_idx].tolist(),
        "top20_om_d": om_dist[top20_idx].tolist(),
        "top20_score": scores[top20_idx].tolist(),
        "top20_canonical_qa_d": qa_dist_canon.tolist(),
        "top20_canonical_om_d": om_dist_canon.tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    log(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
