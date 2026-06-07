"""s059k_diagnostic — Phase 1 polish-bridge probes on cached s059j seed-28 data.

NO new score-grid compute. Uses:
    results/s059j_cloud_data_omega_grid/seed028/{clusters.npz, score_grid.npz}

Three actions:

  (a) DIAGNOSTIC ONLY: polish the truth-cluster best-member with both
      (a-local) lm_polish_local — architecture-faithful (matches s059j gate)
      (a-full)  lm_polish        — full-LC, wider cost surface
      Hi-fi-render both, classify ρ-band. Tests whether the polish bridge
      from a TRUTH-GRADE ω-grid seed (om_d=6.92°, qa_d=2.43°) reaches
      Band A∪B. Forbidden for headline yield per
      `feedback_oracle_injection_taints_yield.md`.

  (b) FOR YIELD: re-polish rank-1 cluster (cluster 59) with full-LC
      residual. The s059j run hit local-window ρ=3.97 → hi-fi ρ=66.
      Wider cost surface might reach a Band A∪B multi-solution alternate.
      INDEPENDENT of densification.

  (c) PREDICT: from cached score_grid at the closest-q_a row, fit
      ρ_local(θ_ω) and predict at hypothetical N_DIRS={400, 800, 1600}
      nearest-cell distances. Validates the previous agent's "ρ in 1–3
      at ~1° noise" claim.

Usage:
    python experiments/s059k_diagnostic.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s059_pilot import back_propagate  # noqa: E402
from experiments.s059e_local_window import lm_polish_local  # noqa: E402
from experiments.s058_lm_polish_clusters import lm_polish  # noqa: E402
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402

SEED = 28
T_A = 25
W = 10
S059J = SURVEY / "results" / "s059j_cloud_data_omega_grid" / "seed028"
OUT = SURVEY / "results" / "s059k_diagnostic" / f"seed{SEED:03d}"
OUT.mkdir(parents=True, exist_ok=True)


def _to_jsonable(d):
    """Convert NumPy values in a dict to JSON-serialisable native types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = _to_jsonable(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [
                x.tolist() if isinstance(x, np.ndarray)
                else x.item() if isinstance(x, (np.floating, np.integer))
                else x for x in v
            ]
        else:
            out[k] = v
    return out


def hifi_classify(q0_pol_wxyz, om0_pol_rad, ctx, target):
    try:
        pred = render_hifi(np.asarray(q0_pol_wxyz), np.asarray(om0_pol_rad), ctx)
        rho_h = float(rho_from_hifi(pred, target))
        band = rho_band(rho_h)
        return rho_h, band, pred
    except Exception as e:
        return float("nan"), "ERR", None


def predict_density_curve(scores_row, om_grid, om_truth):
    """For a fixed q_a row of scores (1200 ω cells), return ρ_local(θ_ω) data
    suitable for predicting densified-grid behaviour.

    Returns dict with theta_deg (1200,) and rho_local (1200,) arrays + summary.
    """
    rho_row = np.sqrt(scores_row) / 0.05
    om_grid_mags = np.linalg.norm(om_grid, axis=1)
    om_truth_mag = float(np.linalg.norm(om_truth))
    om_dirs = om_grid / np.maximum(om_grid_mags[:, None], 1e-12)
    om_truth_dir = om_truth / max(1e-12, om_truth_mag)
    cos_t = np.clip(om_dirs @ om_truth_dir, -1, 1)
    theta_deg = np.degrees(np.arccos(cos_t))
    mag_pct = (om_grid_mags - om_truth_mag) / om_truth_mag * 100

    # Pick the single magnitude factor closest to truth |ω|; that gives a
    # 200-cell direction-only curve at fixed (close-to-truth) magnitude.
    unique_mags = np.unique(np.round(mag_pct, 4))
    closest_mag_pct = float(unique_mags[np.argmin(np.abs(unique_mags))])
    closest_mag_mask = np.isclose(mag_pct, closest_mag_pct, atol=1e-3)
    return {
        "theta_deg": theta_deg.tolist(),
        "rho_local": rho_row.tolist(),
        "mag_pct": mag_pct.tolist(),
        "closest_mag_pct": closest_mag_pct,
        "closest_mag_mask": closest_mag_mask.tolist(),
        "rho_at_closest_mag_dir_curve": rho_row[closest_mag_mask].tolist(),
        "theta_at_closest_mag_dir_curve": theta_deg[closest_mag_mask].tolist(),
    }


def predict_at_density(theta_curve, rho_curve, target_thetas, anchor_truth_rho=None):
    """Predict ρ_local at hypothetical θ values via quadratic fit.

    Strategy: take the bottom decile of the curve (smallest ρ, near the basin),
    fit ρ ≈ a + b·θ + c·θ² in MSE space (more honest than ρ-space because
    the cost is naturally quadratic in MSE near a minimum).

    `anchor_truth_rho` (optional): an explicit (θ=0, ρ) anchor for extrapolation
    near truth — e.g. the cached truth-EXACT cost ρ=0.39.
    """
    theta_arr = np.asarray(theta_curve)
    rho_arr = np.asarray(rho_curve)
    order = np.argsort(theta_arr)
    theta_arr = theta_arr[order]
    rho_arr = rho_arr[order]

    # Convert ρ → MSE (ρ = √MSE / 0.05 → MSE = (ρ·0.05)²)
    mse_arr = (rho_arr * 0.05) ** 2

    # Use bottom 25% of curve by ρ for fit (the "basin" portion)
    n_keep = max(5, len(rho_arr) // 4)
    keep_idx = np.argsort(rho_arr)[:n_keep]
    th_fit = theta_arr[keep_idx]
    mse_fit = mse_arr[keep_idx]
    if anchor_truth_rho is not None:
        th_fit = np.concatenate([[0.0], th_fit])
        mse_fit = np.concatenate([[(anchor_truth_rho * 0.05) ** 2], mse_fit])

    # Fit MSE ≈ a + b·θ + c·θ²
    A = np.column_stack([np.ones_like(th_fit), th_fit, th_fit ** 2])
    coef, *_ = np.linalg.lstsq(A, mse_fit, rcond=None)
    a_c, b_c, c_c = float(coef[0]), float(coef[1]), float(coef[2])

    out = {"_quadfit_coeffs": {"a": a_c, "b": b_c, "c": c_c}}
    out["_curve_min_rho"] = float(rho_arr.min())
    out["_curve_min_theta_deg"] = float(theta_arr[np.argmin(rho_arr)])
    for t in target_thetas:
        mse_t = max(1e-12, a_c + b_c * t + c_c * t * t)
        rho_t = float(np.sqrt(mse_t) / 0.05)
        out[f"rho_at_{t:.2f}deg"] = rho_t
    return out


def main():
    print(f"=== s059k_diagnostic — seed {SEED}, T_A={T_A}, W={W} ===\n")

    t_overall = time.time()

    print(f"loading cached: {S059J}/{{clusters.npz,score_grid.npz}}")
    cl = np.load(S059J / "clusters.npz")
    sg = np.load(S059J / "score_grid.npz")

    Q_top = cl["Q_top"]                 # (5000, 4)
    Om_top = cl["Om_top"]               # (5000, 3)
    mse_top = cl["mse_top"]             # (5000,)
    truth_idx = int(cl["truth_idx_diagnostic"])
    truth_cluster_rank = int(cl["truth_cluster_rank_diagnostic"])
    truth_score_rank = int(cl["truth_score_rank_in_topK"])
    print(f"  truth_idx_diagnostic    = {truth_idx} (rank {truth_idx+1}/5000 in top-K)")
    print(f"  truth_cluster_rank      = {truth_cluster_rank}")
    print(f"  truth_score_rank_in_topK= {truth_score_rank}")

    C_a = sg["C_a"]                     # (1534, 4)
    omega_grid = sg["omega_grid"]       # (1200, 3)
    scores_full = sg["scores"]          # (1534, 1200)
    q_a_truth = sg["q_a_truth"]         # (4,)
    om_a_truth = sg["om_a_truth"]       # (3,)

    print(f"\nbuilding hifi context for seed {SEED}...")
    t0 = time.time()
    ctx = build_context(seed=SEED)
    target = ctx["mag_hifi_truth"]
    print(f"  built in {time.time()-t0:.1f}s; target len={len(target)}")

    summary = {
        "experiment": "s059k_diagnostic",
        "seed": SEED, "T_A": T_A, "W": W,
        "truth_idx_diagnostic": truth_idx,
        "truth_cluster_rank_diagnostic": truth_cluster_rank,
        "truth_score_rank_in_topK": truth_score_rank,
    }

    # ============================================================
    # (a) Truth-cluster polish — DIAGNOSTIC ONLY
    # ============================================================
    print(f"\n=== (a) Truth-cluster polish — DIAGNOSTIC, NEVER counts toward yield ===")
    qa_seed = Q_top[truth_idx]
    om_seed = Om_top[truth_idx]
    om_seed_mag = float(np.linalg.norm(om_seed))
    om_truth_mag = float(np.linalg.norm(om_a_truth))
    print(f"  seed q_a = {qa_seed}")
    print(f"  seed ω_a = {om_seed} (|ω|={np.degrees(om_seed_mag):.3f} dps, "
          f"truth |ω|={np.degrees(om_truth_mag):.3f} dps)")
    print(f"  seed mse_local={mse_top[truth_idx]:.4e}, "
          f"ρ_local_seed={float(np.sqrt(mse_top[truth_idx])/0.05):.3f}")

    # (a-local)
    print(f"\n  (a-local) lm_polish_local (matches s059j architecture)")
    t0 = time.time()
    a_local = lm_polish_local(qa_seed, om_seed, T_A, W, ctx, target)
    rho_h, band, _ = hifi_classify(a_local["q0_pol_wxyz"], a_local["om0_pol_rad"], ctx, target)
    a_local["rho_polished_hifi"] = rho_h
    a_local["band_polished_hifi"] = band
    a_local_wall = time.time() - t0

    # Errors vs truth (post-polish, at t=0)
    q0_err = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(a_local["q0_pol_wxyz"], ctx["q0_truth"]))))))
    om_truth_t0 = ctx["omega0_truth_rad"]
    om_pol_t0 = a_local["om0_pol_rad"]
    om_truth_t0_mag = float(np.linalg.norm(om_truth_t0))
    om_mag_err_pct = float((np.linalg.norm(om_pol_t0) - om_truth_t0_mag) / om_truth_t0_mag * 100)
    om_dir_err_deg = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om_pol_t0 / max(1e-12, np.linalg.norm(om_pol_t0)),
                   om_truth_t0 / om_truth_t0_mag)), 0, 1))))
    print(f"    ρ_local_seed={a_local['surrogate_rho_local_seed']:.3f} → "
          f"ρ_local_polished={a_local['surrogate_rho_local_polished']:.3f}")
    print(f"    hi-fi ρ={rho_h:.3f} band={band}")
    print(f"    q0_err={q0_err:.2f}° |ω|err={om_mag_err_pct:+.2f}% ω_dir={om_dir_err_deg:.2f}°")
    print(f"    wall total={a_local_wall:.1f}s")

    # (a-full)
    print(f"\n  (a-full) lm_polish (full-LC residual)")
    t0 = time.time()
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])
    q0_seed_full, om0_seed_full = back_propagate(qa_seed, om_seed, t_a_seconds, ctx["inertia_tensor"])
    a_full = lm_polish(q0_seed_full, om0_seed_full, ctx, target, label="truth_cluster_full")
    rho_h_f, band_f, _ = hifi_classify(np.array(a_full["q0_pol_wxyz"]),
                                       np.array(a_full["om0_pol_rad"]), ctx, target)
    a_full["rho_polished_hifi"] = rho_h_f
    a_full["band_polished_hifi"] = band_f
    a_full_wall = time.time() - t0
    q0_err_f = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(np.array(a_full["q0_pol_wxyz"]),
                                                                 ctx["q0_truth"]))))))
    om_pol_f = np.array(a_full["om0_pol_rad"])
    om_mag_err_f = float((np.linalg.norm(om_pol_f) - om_truth_t0_mag) / om_truth_t0_mag * 100)
    om_dir_err_f = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om_pol_f / max(1e-12, np.linalg.norm(om_pol_f)),
                   om_truth_t0 / om_truth_t0_mag)), 0, 1))))
    a_full["q0_err_polished_deg"] = q0_err_f
    a_full["om_mag_err_pct"] = om_mag_err_f
    a_full["om_dir_err_deg"] = om_dir_err_f
    print(f"    ρ_seed={a_full['surrogate_rho_seed']:.3f} → ρ_polished={a_full['surrogate_rho_polished']:.3f}")
    print(f"    hi-fi ρ={rho_h_f:.3f} band={band_f}")
    print(f"    q0_err={q0_err_f:.2f}° |ω|err={om_mag_err_f:+.2f}% ω_dir={om_dir_err_f:.2f}°")
    print(f"    wall total={a_full_wall:.1f}s")

    summary["truth_cluster_polish"] = {
        "seed": {
            "qa_seed_wxyz": qa_seed.tolist(),
            "om_seed_rad": om_seed.tolist(),
            "rho_local_seed": float(np.sqrt(mse_top[truth_idx])/0.05),
            "om_seed_mag_dps": float(np.degrees(om_seed_mag)),
            "om_truth_mag_dps": float(np.degrees(om_truth_mag)),
        },
        "a_local": {
            "rho_local_polished": float(a_local["surrogate_rho_local_polished"]),
            "rho_polished_hifi": rho_h,
            "band_polished_hifi": band,
            "q0_err_deg": q0_err,
            "om_mag_err_pct": om_mag_err_pct,
            "om_dir_err_deg": om_dir_err_deg,
            "q0_pol_wxyz": list(a_local["q0_pol_wxyz"]),
            "om0_pol_rad": list(a_local["om0_pol_rad"]),
            "n_eval": int(a_local["n_eval"]),
            "wall_s_total": a_local_wall,
        },
        "a_full": {
            "rho_polished": float(a_full["surrogate_rho_polished"]),
            "rho_polished_hifi": rho_h_f,
            "band_polished_hifi": band_f,
            "q0_err_deg": q0_err_f,
            "om_mag_err_pct": om_mag_err_f,
            "om_dir_err_deg": om_dir_err_f,
            "q0_pol_wxyz": a_full["q0_pol_wxyz"],
            "om0_pol_rad": a_full["om0_pol_rad"],
            "n_eval": int(a_full["n_eval"]),
            "wall_s_total": a_full_wall,
        },
    }

    # ============================================================
    # (b) Rank-1 multi-solution polish — full-LC, FOR YIELD
    # ============================================================
    print(f"\n=== (b) Rank-1 multi-solution polish — full-LC, FOR YIELD ===")
    qa_r1 = Q_top[0]
    om_r1 = Om_top[0]
    print(f"  seed q_a = {qa_r1}")
    print(f"  seed ω_a = {om_r1} (|ω|={np.degrees(np.linalg.norm(om_r1)):.3f} dps)")
    print(f"  seed mse_local={mse_top[0]:.4e}, ρ_local_seed={float(np.sqrt(mse_top[0])/0.05):.3f}")

    t0 = time.time()
    q0_seed_b, om0_seed_b = back_propagate(qa_r1, om_r1, t_a_seconds, ctx["inertia_tensor"])
    b_full = lm_polish(q0_seed_b, om0_seed_b, ctx, target, label="rank1_multisol_full")
    rho_h_b, band_b, _ = hifi_classify(np.array(b_full["q0_pol_wxyz"]),
                                       np.array(b_full["om0_pol_rad"]), ctx, target)
    b_full["rho_polished_hifi"] = rho_h_b
    b_full["band_polished_hifi"] = band_b
    b_full_wall = time.time() - t0
    q0_err_b = float(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(np.array(b_full["q0_pol_wxyz"]),
                                                                 ctx["q0_truth"]))))))
    om_pol_b = np.array(b_full["om0_pol_rad"])
    om_mag_err_b = float((np.linalg.norm(om_pol_b) - om_truth_t0_mag) / om_truth_t0_mag * 100)
    om_dir_err_b = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om_pol_b / max(1e-12, np.linalg.norm(om_pol_b)),
                   om_truth_t0 / om_truth_t0_mag)), 0, 1))))
    print(f"    ρ_seed={b_full['surrogate_rho_seed']:.3f} → ρ_polished={b_full['surrogate_rho_polished']:.3f}")
    print(f"    hi-fi ρ={rho_h_b:.3f} band={band_b}")
    print(f"    q0_err={q0_err_b:.2f}° |ω|err={om_mag_err_b:+.2f}% ω_dir={om_dir_err_b:.2f}°")
    print(f"    wall total={b_full_wall:.1f}s")

    summary["rank1_multisol_polish"] = {
        "seed": {
            "qa_seed_wxyz": qa_r1.tolist(),
            "om_seed_rad": om_r1.tolist(),
            "rho_local_seed": float(np.sqrt(mse_top[0])/0.05),
        },
        "b_full": {
            "rho_polished": float(b_full["surrogate_rho_polished"]),
            "rho_polished_hifi": rho_h_b,
            "band_polished_hifi": band_b,
            "q0_err_deg": q0_err_b,
            "om_mag_err_pct": om_mag_err_b,
            "om_dir_err_deg": om_dir_err_b,
            "q0_pol_wxyz": b_full["q0_pol_wxyz"],
            "om0_pol_rad": b_full["om0_pol_rad"],
            "n_eval": int(b_full["n_eval"]),
            "wall_s_total": b_full_wall,
        },
    }

    # ============================================================
    # (c) Density prediction: ρ_local(θ_ω) at the closest q_a row
    # ============================================================
    print(f"\n=== (c) Density prediction from cached score grid ===")
    # Closest q_a in C_a to truth
    dots = np.abs(C_a @ q_a_truth)
    qa_dist = 2 * np.degrees(np.arccos(np.clip(dots, 0, 1)))
    closest_qa_idx = int(np.argmin(qa_dist))
    closest_qa_dist = float(qa_dist[closest_qa_idx])
    print(f"  closest q_a in C_a to truth: idx={closest_qa_idx}, dist={closest_qa_dist:.3f}°")
    print(f"  scoring this q_a's full ω-grid row...")
    scores_row = scores_full[closest_qa_idx]  # (1200,)

    curve = predict_density_curve(scores_row, omega_grid, om_a_truth)
    print(f"  closest-mag dir-only curve: {len(curve['theta_at_closest_mag_dir_curve'])} cells "
          f"at mag_pct={curve['closest_mag_pct']:+.2f}% from truth |ω|")

    # Empirical density expectations (worst-case nearest-neighbor distance from random pt
    # to Fibonacci grid, ~half the typical inter-point spacing).
    # spacing(N) ≈ √(4π/N) rad; expected nearest distance from random pt ≈ 0.5 * that.
    target_thetas = {
        "N=200_current_obs": 6.92,        # measured closest in current N=200 grid
        "N=400_pred":        4.0,
        "N=800_pred":        1.75,
        "N=1600_pred":       0.85,
    }
    pred = predict_at_density(
        curve["theta_at_closest_mag_dir_curve"],
        curve["rho_at_closest_mag_dir_curve"],
        list(target_thetas.values()),
    )
    print(f"\n  predicted ρ_local at truth-mag, varying θ_ω:")
    for label, theta in target_thetas.items():
        key = f"rho_at_{theta:.2f}deg"
        rho = pred.get(key, float("nan"))
        in_band = ("A" if rho < 2 else "B" if rho < 4 else "C" if rho < 8 else "D")
        print(f"    {label:24s} θ={theta:5.2f}° → ρ_local≈{rho:7.2f} ({in_band})")

    summary["density_prediction"] = {
        "closest_qa_idx": closest_qa_idx,
        "closest_qa_dist_deg": closest_qa_dist,
        "closest_mag_pct": curve["closest_mag_pct"],
        "target_thetas_deg": target_thetas,
        "predicted_rho_local": pred,
        "curve_theta_deg": curve["theta_at_closest_mag_dir_curve"],
        "curve_rho_local": curve["rho_at_closest_mag_dir_curve"],
    }

    summary["wall_total_s"] = time.time() - t_overall

    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2))
    print(f"\nSaved: {summary_path}")
    print(f"\nTotal wall: {summary['wall_total_s']:.1f}s ({summary['wall_total_s']/60:.2f} min)")

    # Final headline
    print(f"\n{'='*60}")
    print("PHASE 1 HEADLINE")
    print(f"{'='*60}")
    print(f"(a-local) truth-cluster polish: hi-fi ρ={rho_h:.3f} band={band}")
    print(f"(a-full)  truth-cluster polish: hi-fi ρ={rho_h_f:.3f} band={band_f}")
    print(f"(b-full)  rank-1 multi-sol:     hi-fi ρ={rho_h_b:.3f} band={band_b}")
    print()
    print(f"Decision gate:")
    print(f"  (a) Band A∪B → GO Phase 2 (densification will fix it)")
    print(f"  (a) Band C/D → ABORT Phase 2; pivot to architecture C")
    print(f"  (b) Band A∪B → BONUS yield (multi-sol on seed 28)")
    a_pass = band in {"A", "B"} or band_f in {"A", "B"}
    b_pass = band_b in {"A", "B"}
    print()
    print(f"  (a) any Band A∪B: {a_pass}")
    print(f"  (b) Band A∪B:     {b_pass}")
    return summary


if __name__ == "__main__":
    main()
