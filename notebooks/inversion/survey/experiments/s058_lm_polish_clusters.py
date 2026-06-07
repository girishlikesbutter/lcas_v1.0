"""s058 — LM polish on s057h cluster representatives.

Architecture: s057g/h identified 325 candidate clusters at seed 89 with
4400× discrimination over null. s057i hi-fi-rendered the top-15 + truth
cluster representatives and got ALL Band D (truth ρ=44.07) — back-prop
amplifies the +15-25% |ω| error in seeds over the 50-min window. The
operational reframe is: forward-prop is a SEED GENERATOR; each candidate
needs LM polish to refine (q_0, ω_0) before ρ-band classification.

This script runs LM polish on the top-5 cluster representatives + truth
cluster (6 polishes total), seeded from each cluster's already-back-
propagated (q0_back_wxyz, om0_back_rad) cached in s057i summary.json.

Cost: surrogate-v2 full-LC MSE residual (NOT hi-fi — `feedback_lm_cost_use
_surrogate.md`). 6 free params: rotvec (3) for q0 perturbation + ω_0 (3).
scipy least_squares(method='lm'), max_nfev=200, finite-diff Jacobian.

Hi-fi validation per polished candidate. Headline question: does ≥1 LM
polish from a top-cluster seed land in Band A∪B (ρ < 4)? If so, the
forward-propagation architecture is operational as a seed generator.

Smoke test FIRST: polish from truth (q0_truth, ω_truth) — should converge
in ~0 iters at ρ=0. Then polish from back-propagated truth (q0_truth_back,
which has q0_err=0.11°, ω0_err=0.24% per s057i smoke) — must recover
ρ<2 since smoke s057i already shows ρ=0.65 from this state.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))

from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band
from lib.forward import propagate_to_body_frame
from lib.surrogate_eval import predict as surrogate_predict, full_lc_mse

S057I_SUMMARY = SURVEY / "results" / "s057i_hifi_validate" / "summary.json"
OUT = SURVEY / "results" / "s058_lm_polish_clusters"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 89
TOP_K = 5  # polish top-K clusters + truth cluster
LM_MAX_NFEV = 200
LM_FTOL = 1e-8
LM_XTOL = 1e-8
RESIDUAL_CAP = 5.0  # mag — cap residual magnitude to handle inf surrogate predictions


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def quat_geodesic_deg(q1_wxyz, q2_wxyz):
    dot = float(np.abs(np.dot(q1_wxyz, q2_wxyz)))
    return float(2 * np.degrees(np.arccos(min(1.0, max(-1.0, dot)))))


def make_residual_fn(q0_seed_wxyz, ctx, target):
    """Closure: params=(rotvec_3, om_3) → residual vector vs target.

    rotvec applies a left-multiplication perturbation to q0_seed:
        q0 = exp(rotvec) ⊗ q0_seed
    so params=(0,0,0, om_seed) returns the cost at the seed.
    """
    q0_seed_xyzw = quat_wxyz_to_xyzw(q0_seed_wxyz)
    R_seed = Rotation.from_quat(q0_seed_xyzw)

    def residual(params):
        rotvec = params[:3]
        omega0 = params[3:]
        q_xyzw = (Rotation.from_rotvec(rotvec) * R_seed).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        try:
            k1, k2, _ = propagate_to_body_frame(
                q0_wxyz=q_wxyz, omega0_rad=omega0,
                observation_times=ctx["observation_times"],
                sun_pos=ctx["sun_pos"], obs_pos=ctx["obs_pos"],
                sat_pos=ctx["sat_pos"], inertia_tensor=ctx["inertia_tensor"],
                mode="tumbling",
            )
            pred = surrogate_predict(k1, k2, ctx["obs_dist"])
            r = pred - target
            r = np.where(np.isfinite(r), r, RESIDUAL_CAP)
            return np.clip(r, -RESIDUAL_CAP, RESIDUAL_CAP)
        except Exception:
            return np.full_like(target, RESIDUAL_CAP)

    return residual


def lm_polish(q0_seed_wxyz, omega0_seed_rad, ctx, target, label=""):
    """Run scipy least_squares from the given seed.

    Returns dict with q0_polished, omega0_polished, surrogate_mse_seed,
    surrogate_mse_polished, n_eval, converged, wall_s.
    """
    residual = make_residual_fn(q0_seed_wxyz, ctx, target)
    x0 = np.concatenate([np.zeros(3), np.asarray(omega0_seed_rad)])

    r_seed = residual(x0)
    mse_seed = float(np.mean(r_seed ** 2))

    t0 = time.time()
    result = least_squares(
        residual, x0,
        method="lm",
        max_nfev=LM_MAX_NFEV,
        ftol=LM_FTOL, xtol=LM_XTOL,
    )
    wall_s = time.time() - t0

    rotvec_pol = result.x[:3]
    omega0_pol = result.x[3:]
    q_xyzw_seed = quat_wxyz_to_xyzw(q0_seed_wxyz)
    q_pol_xyzw = (Rotation.from_rotvec(rotvec_pol) *
                  Rotation.from_quat(q_xyzw_seed)).as_quat()
    q0_pol_wxyz = quat_xyzw_to_wxyz(q_pol_xyzw)

    r_pol = result.fun
    mse_pol = float(np.mean(r_pol ** 2))

    return {
        "label": label,
        "q0_seed_wxyz": q0_seed_wxyz.tolist() if hasattr(q0_seed_wxyz, "tolist") else list(q0_seed_wxyz),
        "om0_seed_rad": list(omega0_seed_rad) if hasattr(omega0_seed_rad, "tolist") else list(omega0_seed_rad),
        "q0_pol_wxyz": q0_pol_wxyz.tolist(),
        "om0_pol_rad": omega0_pol.tolist(),
        "rotvec_pol": rotvec_pol.tolist(),
        "rotvec_pol_mag_deg": float(np.degrees(np.linalg.norm(rotvec_pol))),
        "om_change_pct": float(np.linalg.norm(omega0_pol - np.asarray(omega0_seed_rad)) /
                               max(1e-12, np.linalg.norm(omega0_seed_rad)) * 100),
        "surrogate_mse_seed": mse_seed,
        "surrogate_mse_polished": mse_pol,
        "surrogate_rho_seed": float(np.sqrt(mse_seed) / 0.05),
        "surrogate_rho_polished": float(np.sqrt(mse_pol) / 0.05),
        "n_eval": int(result.nfev),
        "status": int(result.status),
        "converged": bool(result.status > 0),
        "wall_s": wall_s,
    }


def main() -> dict:
    t_overall = time.time()

    print("=== s058: LM polish on s057h cluster representatives ===\n")
    print(f"Loading s057i back-propagated state from:\n  {S057I_SUMMARY}")
    with open(S057I_SUMMARY) as f:
        s057i = json.load(f)

    rendered = s057i["rendered"]
    rendered_sorted = sorted(rendered, key=lambda r: r["cluster_rank"])

    top_k = [r for r in rendered_sorted if r["cluster_rank"] <= TOP_K]
    truth_render = next(r for r in rendered_sorted if r["is_truth_cluster"])
    if truth_render not in top_k:
        seeds_to_polish = top_k + [truth_render]
    else:
        seeds_to_polish = top_k
    print(f"  selected {len(seeds_to_polish)} cluster reps to polish "
          f"(top {TOP_K} + truth)\n")

    print("=== building hifi context ===")
    t0 = time.time()
    ctx = build_context(seed=SEED)
    target = ctx["mag_hifi_truth"]
    print(f"  built in {time.time()-t0:.1f}s")

    print("\n=== smoke test 1: polish from truth itself (ρ should ≈ 0) ===")
    smoke1 = lm_polish(
        ctx["q0_truth"], ctx["omega0_truth_rad"], ctx, target,
        label="smoke_truth",
    )
    print(f"  surrogate ρ_seed={smoke1['surrogate_rho_seed']:.4e}, "
          f"ρ_polished={smoke1['surrogate_rho_polished']:.4e}, "
          f"n_eval={smoke1['n_eval']}, wall={smoke1['wall_s']:.1f}s")

    print("\n=== smoke test 2: polish from back-propagated truth (s057i smoke) ===")
    smoke2 = lm_polish(
        np.array(truth_render["q0_back_wxyz"]),
        np.array(truth_render["om0_back_rad"]),
        ctx, target, label="smoke_truth_back",
    )
    print(f"  surrogate ρ_seed={smoke2['surrogate_rho_seed']:.3f}, "
          f"ρ_polished={smoke2['surrogate_rho_polished']:.3f}, "
          f"rotvec_pol={smoke2['rotvec_pol_mag_deg']:.3f}°, "
          f"om_change={smoke2['om_change_pct']:.2f}%, "
          f"n_eval={smoke2['n_eval']}, wall={smoke2['wall_s']:.1f}s")

    polished = []
    print(f"\n=== polishing {len(seeds_to_polish)} cluster reps ===")
    for r in seeds_to_polish:
        is_truth = r["is_truth_cluster"]
        label = f"cluster_id={r['cluster_id']}" + (" (TRUTH)" if is_truth else "")
        print(f"\n  → cluster_rank={r['cluster_rank']:3d}  {label}")
        print(f"     seed: qa_d_t_a={r['qa_dist_t_a_to_truth_deg']:.2f}°, "
              f"om_d_t_a={r['om_dist_at_t_a_to_truth_deg']:.2f}°, "
              f"|ω|_t_a={r['om_mag_dps_at_t_a']:.3f}dps, "
              f"hi-fi ρ_seed (s057i)={r['rho']:.2f}")
        result = lm_polish(
            np.array(r["q0_back_wxyz"]),
            np.array(r["om0_back_rad"]),
            ctx, target, label=label,
        )
        result["cluster_rank"] = r["cluster_rank"]
        result["cluster_id"] = r["cluster_id"]
        result["is_truth_cluster"] = is_truth
        result["score_sum"] = r["score_sum"]
        result["score_max"] = r["score_max"]
        result["qa_dist_t_a_to_truth_deg"] = r["qa_dist_t_a_to_truth_deg"]
        result["om_dist_at_t_a_to_truth_deg"] = r["om_dist_at_t_a_to_truth_deg"]
        result["om_mag_dps_at_t_a"] = r["om_mag_dps_at_t_a"]
        result["rho_seed_hifi_s057i"] = r["rho"]
        result["band_seed_s057i"] = r["band"]

        print(f"     surrogate ρ_seed={result['surrogate_rho_seed']:.3f} → "
              f"ρ_polished={result['surrogate_rho_polished']:.3f}  "
              f"(rotvec_pol={result['rotvec_pol_mag_deg']:.2f}°, "
              f"om_change={result['om_change_pct']:.2f}%, "
              f"n_eval={result['n_eval']}, wall={result['wall_s']:.1f}s)")

        # === Hi-fi validation ===
        t0 = time.time()
        try:
            pred_pol = render_hifi(
                np.array(result["q0_pol_wxyz"]),
                np.array(result["om0_pol_rad"]),
                ctx,
            )
            rho_pol = rho_from_hifi(pred_pol, target)
            band_pol = rho_band(rho_pol)
        except Exception as e:
            print(f"     hi-fi render FAILED: {e}")
            rho_pol = float("nan")
            band_pol = "ERR"
            pred_pol = np.full_like(target, np.nan)
        result["rho_polished_hifi"] = float(rho_pol)
        result["band_polished_hifi"] = band_pol
        result["hifi_render_s"] = time.time() - t0
        result["pred_pol_hifi"] = pred_pol.tolist()
        # Errors vs truth
        result["q0_err_polished_deg"] = quat_geodesic_deg(
            np.array(result["q0_pol_wxyz"]), ctx["q0_truth"])
        om_truth = ctx["omega0_truth_rad"]
        om_pol = np.array(result["om0_pol_rad"])
        om_truth_mag = float(np.linalg.norm(om_truth))
        result["om_mag_err_polished_pct"] = float(
            (np.linalg.norm(om_pol) - om_truth_mag) / om_truth_mag * 100)
        result["om_dir_err_polished_deg"] = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om_pol / np.linalg.norm(om_pol),
                       om_truth / om_truth_mag)), 0, 1))))
        print(f"     HI-FI ρ_polished={rho_pol:.3f}  band={band_pol}   "
              f"(seed s057i ρ={r['rho']:.2f}, band={r['band']})")
        print(f"     errors vs truth: q0={result['q0_err_polished_deg']:.2f}°, "
              f"|ω|={result['om_mag_err_polished_pct']:+.2f}%, "
              f"ω_dir={result['om_dir_err_polished_deg']:.2f}°")
        polished.append(result)

    # === Aggregate ===
    bands = [p["band_polished_hifi"] for p in polished]
    band_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "ERR": 0}
    for b in bands:
        band_counts[b] = band_counts.get(b, 0) + 1
    n_AB = band_counts["A"] + band_counts["B"]
    print(f"\n=== POLISH RESULT SUMMARY ===")
    print(f"  Polished {len(polished)} cluster reps")
    print(f"  Band A: {band_counts['A']}, B: {band_counts['B']}, "
          f"C: {band_counts['C']}, D: {band_counts['D']}, ERR: {band_counts['ERR']}")
    print(f"  Band A∪B: {n_AB}/{len(polished)}  ← architecture {'PASS' if n_AB > 0 else 'FAIL'}")

    # === Figure ===
    print("\n=== generating figure ===")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) ρ_seed vs ρ_polished, log scale, per cluster
    ax = axes[0, 0]
    ranks = [p["cluster_rank"] for p in polished]
    rho_seed = [p["rho_seed_hifi_s057i"] for p in polished]
    rho_pol = [p["rho_polished_hifi"] for p in polished]
    is_truth = [p["is_truth_cluster"] for p in polished]
    color_by_band = {"A": "green", "B": "lime", "C": "orange", "D": "red", "ERR": "grey"}
    for r, ρs, ρp, t, b in zip(ranks, rho_seed, rho_pol, is_truth, bands):
        ax.plot([ρs, ρp], [r, r], "k-", lw=0.5, alpha=0.5)
        ax.scatter([ρs], [r], color="grey", s=80, marker="s",
                   edgecolor="black", zorder=5,
                   label="seed (s057i)" if r == ranks[0] else None)
        ax.scatter([ρp], [r], color=color_by_band.get(b, "k"),
                   s=300 if t else 150,
                   marker="*" if t else "o",
                   edgecolor="black", linewidth=1.5 if t else 0.8,
                   zorder=10 if t else 6,
                   label=f"polished (truth, ρ={ρp:.2f})" if t else None)
    ax.axvline(2, ls="--", color="green", lw=0.8, label="A/B (ρ=2)")
    ax.axvline(4, ls="--", color="orange", lw=0.8, label="B/C (ρ=4)")
    ax.axvline(8, ls="--", color="red", lw=0.8, label="C/D (ρ=8)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("ρ (hi-fi)")
    ax.set_ylabel("cluster rank by sum-score")
    ax.invert_yaxis()
    ax.set_title(f"hi-fi ρ before/after LM polish ({len(polished)} cluster reps)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) bar chart band counts before/after
    ax = axes[0, 1]
    band_counts_seed = {"A": 0, "B": 0, "C": 0, "D": 0, "ERR": 0}
    for p in polished:
        b = p["band_seed_s057i"]
        band_counts_seed[b] = band_counts_seed.get(b, 0) + 1
    bs = ["A", "B", "C", "D"]
    seed_counts = [band_counts_seed[b] for b in bs]
    pol_counts = [band_counts[b] for b in bs]
    x = np.arange(len(bs))
    w = 0.4
    ax.bar(x - w/2, seed_counts, w, label="seed (s057i)", color="grey",
           edgecolor="black")
    ax.bar(x + w/2, pol_counts, w, label="polished",
           color=[color_by_band[b] for b in bs], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(bs)
    ax.set_xlabel("ρ-band")
    ax.set_ylabel("# cluster reps")
    ax.set_title(f"band distribution: seed vs polished\n"
                 f"Band A∪B: {sum(seed_counts[:2])}→{sum(pol_counts[:2])}/{len(polished)}")
    ax.legend(loc="best", fontsize=9)
    for xi, v in enumerate(seed_counts):
        if v > 0: ax.text(xi - w/2, v, str(v), ha="center", va="bottom", fontsize=9)
    for xi, v in enumerate(pol_counts):
        if v > 0: ax.text(xi + w/2, v, str(v), ha="center", va="bottom", fontsize=9)

    # (c) q0_err vs ρ_polished
    ax = axes[1, 0]
    q_errs = [p["q0_err_polished_deg"] for p in polished]
    for q, ρ, t, b in zip(q_errs, rho_pol, is_truth, bands):
        ax.scatter([q], [ρ], color=color_by_band.get(b, "k"),
                   s=300 if t else 150,
                   marker="*" if t else "o",
                   edgecolor="black", linewidth=1.5 if t else 0.8,
                   zorder=10 if t else 6)
    ax.axhline(2, ls="--", color="green", lw=0.8)
    ax.axhline(4, ls="--", color="orange", lw=0.8)
    ax.axhline(8, ls="--", color="red", lw=0.8)
    ax.set_xlabel("q0 error vs truth (deg, post-polish)")
    ax.set_ylabel("ρ (hi-fi, post-polish)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("post-polish geometry: q0 error vs ρ")
    ax.grid(alpha=0.3)

    # (d) one example LC overlay — truth cluster polished
    ax = axes[1, 1]
    truth_pol = next((p for p in polished if p["is_truth_cluster"]), None)
    if truth_pol is not None and not np.any(np.isnan(truth_pol["pred_pol_hifi"])):
        ep = np.arange(len(target))
        ax.plot(ep, target, "k-", lw=1, label="truth", alpha=0.8)
        ax.plot(ep, truth_pol["pred_pol_hifi"], "r--", lw=1,
                label=f"polished truth-cluster (ρ={truth_pol['rho_polished_hifi']:.2f})",
                alpha=0.8)
        ax.set_xlabel("epoch index")
        ax.set_ylabel("magnitude")
        ax.invert_yaxis()
        ax.set_title("truth cluster: hi-fi LC after polish")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "truth cluster polish failed",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    fig_p = OUT / "lm_polish_clusters.png"
    plt.savefig(fig_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_p}")

    # === Save NPZ checkpoint (load-bearing per feedback_save_results) ===
    npz_p = OUT / "polished_states.npz"
    np.savez(
        npz_p,
        q0_pol_wxyz=np.array([p["q0_pol_wxyz"] for p in polished]),
        om0_pol_rad=np.array([p["om0_pol_rad"] for p in polished]),
        rho_seed=np.array([p["rho_seed_hifi_s057i"] for p in polished]),
        rho_polished=np.array([p["rho_polished_hifi"] for p in polished]),
        cluster_rank=np.array([p["cluster_rank"] for p in polished]),
        cluster_id=np.array([p["cluster_id"] for p in polished]),
        is_truth=np.array([p["is_truth_cluster"] for p in polished]),
        q0_err_deg=np.array([p["q0_err_polished_deg"] for p in polished]),
        om_mag_err_pct=np.array([p["om_mag_err_polished_pct"] for p in polished]),
        om_dir_err_deg=np.array([p["om_dir_err_polished_deg"] for p in polished]),
        pred_hifi=np.array([p["pred_pol_hifi"] for p in polished]),
    )
    print(f"Saved: {npz_p}")

    # === Save JSON summary ===
    summary = {
        "seed": SEED,
        "n_polished": len(polished),
        "top_k_clusters": TOP_K,
        "lm_config": {
            "method": "lm",
            "max_nfev": LM_MAX_NFEV,
            "ftol": LM_FTOL,
            "xtol": LM_XTOL,
            "residual_cap": RESIDUAL_CAP,
            "cost": "surrogate_v2_full_lc_residual",
            "free_params": "rotvec(3) + omega(3)",
        },
        "smoke_truth_self": {
            "rho_seed": smoke1["surrogate_rho_seed"],
            "rho_polished": smoke1["surrogate_rho_polished"],
            "n_eval": smoke1["n_eval"],
            "wall_s": smoke1["wall_s"],
        },
        "smoke_truth_back": {
            "rho_seed_surrogate": smoke2["surrogate_rho_seed"],
            "rho_polished_surrogate": smoke2["surrogate_rho_polished"],
            "rotvec_pol_mag_deg": smoke2["rotvec_pol_mag_deg"],
            "om_change_pct": smoke2["om_change_pct"],
            "n_eval": smoke2["n_eval"],
            "wall_s": smoke2["wall_s"],
        },
        "band_counts_seed_s057i": band_counts_seed,
        "band_counts_polished": band_counts,
        "n_band_AB_seed": sum(seed_counts[:2]),
        "n_band_AB_polished": n_AB,
        "polished": [{k: v for k, v in p.items() if k != "pred_pol_hifi"}
                     for p in polished],
        "wall_total_s": time.time() - t_overall,
    }
    json_p = OUT / "summary.json"
    with open(json_p, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"Saved: {json_p}")

    print(f"\nTotal wall: {time.time()-t_overall:.1f}s")
    return summary


if __name__ == "__main__":
    main()
