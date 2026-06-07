"""Quick interpretive summary of the m138 harmdiv surrogate-MSE re-rank.

Reads:
  h1_harmdiv/isoshell_ckpt.npz       (omega, q0, H1 cost — 80k)
  h1_harmdiv/rerank_surr_ckpt.npz    (surr_mse, surr_bright_mse — 80k)
  h1_harmdiv/result.json             (H1 verdict + truth diagnostics)
  h1_harmdiv/rerank_surr_result.json (Round A verdict)

Prints:
  - Verdict: did Round A surface the 12 joint truth-near candidates into top-K?
  - MSE distribution: median, IQR, min, max.
  - Joint candidates' position under each cost: H1, surrogate full-LC MSE, bright MSE.
  - Anti-correlation diagnostic: H1 cost vs surrogate MSE Spearman correlation.
"""
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
H1_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / "seed_047" / "h1_harmdiv"


def main():
    ck = np.load(H1_DIR / "isoshell_ckpt.npz", allow_pickle=True)
    rk = np.load(H1_DIR / "rerank_surr_ckpt.npz", allow_pickle=True)
    h1_result = json.loads((H1_DIR / "result.json").read_text())
    surr_result = json.loads((H1_DIR / "rerank_surr_result.json").read_text())

    omega = ck["omega_batch"]
    h1_cost = ck["cost"]
    surr_mse = rk["surr_mse"]
    surr_bright_mse = rk["surr_bright_mse"]

    # Recompute joint mask (from result.json metadata)
    truth_mag = surr_result["truth_omega_mag"]
    # Truth direction can't be loaded here without master, but joint indices are equivalent
    # to those in surr_result. Use them directly:
    joint_ranks = surr_result["joint_candidate_ranks_under_surr"]
    joint_mses = surr_result["joint_candidate_surr_mses"]
    joint_dirs = surr_result["joint_candidate_dir_errs_deg"]
    joint_mags = surr_result["joint_candidate_mag_errs_pct"]
    joint_q0_truth = surr_result["joint_candidate_q0_to_truth_deg"]
    n_joint = surr_result["n_pool_joint_5deg_5pct"]

    print("=" * 70)
    print(f"  m138 seed 47 — surrogate-MSE re-rank verdict")
    print("=" * 70)

    print(f"\n  truth |ω| = {truth_mag:.5f} rad/s")
    print(f"  pool size: {len(omega)}")
    print(f"  n_pool joint (5° dir + 5% mag): {n_joint}")

    print(f"\n--- H1 (densest-spot) cost ---")
    print(f"  rank-1 ω-dir:       {h1_result['rank1_omega_dir_err_deg']:.2f}°")
    print(f"  rank-1 ω-mag:       {h1_result['rank1_omega_mag_err_pct']:+.1f}%")
    print(f"  best joint cost rank: {h1_result['best_joint_in_grid_cost_rank']} / {len(omega)}")
    print(f"  n_top30 joint:      {h1_result['n_top30_joint_5deg_5pct']}")

    print(f"\n--- Round A: surrogate full-LC MSE (no q0 polish) ---")
    print(f"  rank-1 surr_MSE:    {surr_result['rank1_surr_mse']:.6f}")
    print(f"  rank-1 ω-dir:       {surr_result['rank1_omega_dir_err_deg']:.2f}°")
    print(f"  rank-1 ω-mag:       {surr_result['rank1_omega_mag_err_pct']:+.2f}%")
    print(f"  rank-1 q0_to_truth: {surr_result['rank1_q0_to_truth_deg']:.2f}°")
    print(f"  rank-1 q0_to_twin:  {surr_result['rank1_q0_to_twin_deg']:.2f}°")
    print(f"  rank-1 H1-rank:     {surr_result['rank1_h1_cost_rank']}")
    print(f"  median pool MSE:    {surr_result['median_pool_surr_mse']:.4f}")
    print(f"  min pool MSE:       {surr_result['min_pool_surr_mse']:.6f}")
    print(f"  n_top30 within 5° dir:    {surr_result['n_top30_within_5deg']}")
    print(f"  n_top30 joint:            {surr_result['n_top30_joint_5deg_5pct']}")
    print(f"  near_dir min rank:        {surr_result['near_dir_min_rank']}")

    print(f"\n--- Where the {n_joint} joint truth-near candidates landed ---")
    if n_joint > 0:
        for i, (r, m, d, mp, q) in enumerate(zip(joint_ranks, joint_mses,
                                                  joint_dirs, joint_mags,
                                                  joint_q0_truth)):
            print(f"  joint #{i+1:2d}: rank={r:6d}/{len(omega)}  "
                  f"MSE={m:.6f}  dir={d:.2f}°  mag={mp:+.2f}%  "
                  f"q0_to_truth={q:.1f}°")

    print(f"\n--- Surrogate MSE distribution (full pool) ---")
    finite = np.isfinite(surr_mse)
    s = surr_mse[finite]
    print(f"  n_finite: {finite.sum()}")
    print(f"  min:      {s.min():.6f}")
    print(f"  p1:       {np.percentile(s, 1):.4f}")
    print(f"  p5:       {np.percentile(s, 5):.4f}")
    print(f"  p25:      {np.percentile(s, 25):.4f}")
    print(f"  median:   {np.median(s):.4f}")
    print(f"  p75:      {np.percentile(s, 75):.4f}")
    print(f"  max:      {s.max():.4f}")

    # Spearman correlation between H1 cost and surrogate MSE
    common_finite = finite & np.isfinite(h1_cost)
    if common_finite.sum() > 100:
        from scipy.stats import spearmanr
        rho, p = spearmanr(h1_cost[common_finite], surr_mse[common_finite])
        print(f"\n  Spearman ρ(H1 cost, surr_MSE): {rho:+.4f}  (p={p:.2e})")
        print(f"    (H1 cost is negated — high cost = low rank in H1)")
        print(f"    Negative ρ means high H1-cost candidates have low surr_MSE,")
        print(f"    i.e. H1 ranking is anti-correlated with surrogate quality.")

    print(f"\n--- Decision ---")
    if surr_result['n_top30_joint_5deg_5pct'] >= 1:
        print(f"  Round A SUFFICES: {surr_result['n_top30_joint_5deg_5pct']} "
              f"joint truth-near candidate(s) in top-30.")
        print(f"  Next: feed top-30 to m115 hi-fi DE polish.")
    elif surr_result['near_dir_min_rank'] < 100:
        print(f"  Round A PARTIAL: nearest 5°-dir candidate at rank "
              f"{surr_result['near_dir_min_rank']}.")
        print(f"  Either: (a) widen m115 hand-off to top-100, or (b) Round B "
              f"(q0 polish, top-500).")
    else:
        print(f"  Round A INSUFFICIENT: nearest 5°-dir candidate at rank "
              f"{surr_result['near_dir_min_rank']}.")
        print(f"  Likely needs Round B (q0 polish on top-K) — H1's centroid q0 "
              f"hypothesis seems wrong.")


if __name__ == "__main__":
    main()
