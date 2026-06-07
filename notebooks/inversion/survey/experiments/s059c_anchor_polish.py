"""s059c — anchor-frame LM polish (no pre-polish back-propagation).

s059b diagnosed the seed 28 failure: back-propagating a cluster rep's
(q_a, ω_a) over 2250s amplified the −16% |ω| error into a 113° q_0 error,
putting LM 113° outside any local basin. LM moves ~30° per polish; it
couldn't bridge.

Reframe (user-suggested 2026-05-08): keep LM in **anchor frame**. Free
params are (rotvec_a, ω_a) at t_a, initialised at the cluster rep
(rotvec_a = 0). Per-iteration: back-prop the CURRENT iterate to
(q_0, ω_0), forward-propagate, surrogate render, residual. The minimum
in the residual landscape is identical — only the starting point in
parameter space changes (8.5° / 0.7° from truth instead of 113° / 89°).
After LM converges, do ONE back-prop of the polished state to report
(q_0*, ω_0*).

Reuses the cached cloud + re-runs forward-prop + clustering (cheap; ~70s)
then swaps the polish stage. Hi-fi gate + classify identical to s059.

Usage:
    python experiments/s059c_anchor_polish.py --seed 28
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from experiments.s059_pilot import (  # reuse all cached stages
    stage_cloud_generation, stage_pick_anchor, stage_forward_prop, stage_cluster,
    back_propagate, wxyz_to_xyzw, xyzw_to_wxyz, quat_ang_deg,
    LM_MAX_NFEV, LM_FTOL, LM_XTOL, RESIDUAL_CAP,
    SURROGATE_RHO_HIFI_GATE, TOP_K_CLUSTERS,
)
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band
from lib.forward import propagate_to_body_frame
from lib.surrogate_eval import predict as surrogate_predict


def make_residual_anchor(q_a_rep_wxyz, om_a_rep_rad, t_a_seconds, ctx, target):
    """Residual fn parameterised at the anchor.

    params = [rotvec_a (3,), om_a (3,)] — rotvec_a perturbs q_a around q_a_rep.
    Per call: build (q_a, om_a) at t_a, back-prop to (q_0, om_0), forward-
    propagate, surrogate render, return residual vs target LC.
    """
    q_a_rep_xyzw = wxyz_to_xyzw(q_a_rep_wxyz)
    R_seed = Rotation.from_quat(q_a_rep_xyzw)
    inertia = ctx["inertia_tensor"]

    def residual(params):
        rotvec_a = params[:3]
        om_a = params[3:]
        try:
            q_a_xyzw = (Rotation.from_rotvec(rotvec_a) * R_seed).as_quat()
            q_a_wxyz = xyzw_to_wxyz(q_a_xyzw)
            q0_wxyz, om0_rad = back_propagate(q_a_wxyz, om_a, t_a_seconds, inertia)
            k1, k2, _ = propagate_to_body_frame(
                q0_wxyz=q0_wxyz, omega0_rad=om0_rad,
                observation_times=ctx["observation_times"],
                sun_pos=ctx["sun_pos"], obs_pos=ctx["obs_pos"],
                sat_pos=ctx["sat_pos"], inertia_tensor=inertia,
                mode="tumbling",
            )
            pred = surrogate_predict(k1, k2, ctx["obs_dist"])
            r = pred - target
            r = np.where(np.isfinite(r), r, RESIDUAL_CAP)
            return np.clip(r, -RESIDUAL_CAP, RESIDUAL_CAP)
        except Exception:
            return np.full_like(target, RESIDUAL_CAP)

    return residual


def lm_polish_anchor(q_a_rep_wxyz, om_a_rep_rad, t_a_seconds, ctx, target):
    residual = make_residual_anchor(q_a_rep_wxyz, om_a_rep_rad, t_a_seconds,
                                    ctx, target)
    x0 = np.concatenate([np.zeros(3), np.asarray(om_a_rep_rad)])
    r_seed = residual(x0)
    mse_seed = float(np.mean(r_seed ** 2))
    t0 = time.time()
    result = least_squares(residual, x0, method="lm",
                           max_nfev=LM_MAX_NFEV, ftol=LM_FTOL, xtol=LM_XTOL)
    wall = time.time() - t0
    rotvec_pol = result.x[:3]
    om_a_pol = result.x[3:]
    q_a_xyzw = (Rotation.from_rotvec(rotvec_pol) *
                Rotation.from_quat(wxyz_to_xyzw(q_a_rep_wxyz))).as_quat()
    q_a_pol_wxyz = xyzw_to_wxyz(q_a_xyzw)
    q0_pol, om0_pol = back_propagate(q_a_pol_wxyz, om_a_pol,
                                     t_a_seconds, ctx["inertia_tensor"])
    mse_pol = float(np.mean(result.fun ** 2))
    return {
        "q_a_pol_wxyz": q_a_pol_wxyz,
        "om_a_pol_rad": om_a_pol,
        "q0_pol_wxyz": q0_pol,
        "om0_pol_rad": om0_pol,
        "rotvec_a_pol_mag_deg": float(np.degrees(np.linalg.norm(rotvec_pol))),
        "om_a_change_pct": float(
            np.linalg.norm(om_a_pol - np.asarray(om_a_rep_rad)) /
            max(1e-12, np.linalg.norm(om_a_rep_rad)) * 100),
        "surrogate_mse_seed": mse_seed,
        "surrogate_mse_polished": mse_pol,
        "surrogate_rho_seed": float(np.sqrt(mse_seed) / 0.05),
        "surrogate_rho_polished": float(np.sqrt(mse_pol) / 0.05),
        "n_eval": int(result.nfev), "wall_s": wall,
    }


def stage_polish_anchor(seed, fp, cl, ctx, log):
    target = ctx["mag_hifi_truth"]
    obs_times = ctx["observation_times"]
    t_a_seconds = float(obs_times[fp["T_A"]] - obs_times[0])
    log(f"T_A = {fp['T_A']}, t_a_seconds = {t_a_seconds:.1f}")

    to_polish = list(cl["clusters_sorted"][:TOP_K_CLUSTERS])
    truth_cl = next(c for c in cl["clusters"]
                    if c["cluster_id"] == cl["truth_cluster_id"])
    if truth_cl not in to_polish:
        to_polish.append(truth_cl)
    log(f"polishing {len(to_polish)} clusters in ANCHOR FRAME (top {TOP_K_CLUSTERS} + truth)")

    polished = []
    for c in to_polish:
        rank = next(i for i, cc in enumerate(cl["clusters_sorted"])
                    if cc["cluster_id"] == c["cluster_id"]) + 1
        is_truth = c["cluster_id"] == cl["truth_cluster_id"]
        bm = c["best_member_idx"]
        q_a = fp["Q_A_pass"][bm]
        om_a = fp["om_pass"][bm]

        result = lm_polish_anchor(q_a, om_a, t_a_seconds, ctx, target)
        result["cluster_rank"] = rank
        result["cluster_id"] = c["cluster_id"]
        result["is_truth_cluster"] = is_truth
        result["score_sum"] = c["score_sum"]
        result["qa_dist_t_a_init"] = c["min_qa_dist_to_truth"]
        result["om_dist_t_a_init"] = c["min_om_dist_to_truth"]

        # errors vs truth (at t=0, after final back-prop)
        result["q0_err_polished_deg"] = quat_ang_deg(
            result["q0_pol_wxyz"], ctx["q0_truth"])
        om_truth = ctx["omega0_truth_rad"]
        om_pol = result["om0_pol_rad"]
        om_truth_mag = float(np.linalg.norm(om_truth))
        result["om_mag_err_pct"] = float(
            (np.linalg.norm(om_pol) - om_truth_mag) / om_truth_mag * 100)
        result["om_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om_pol / np.linalg.norm(om_pol),
                       om_truth / om_truth_mag)), 0, 1))))

        marker = " (TRUTH)" if is_truth else ""
        log(f"  rank {rank:3d}/{len(cl['clusters'])}  cluster_id={c['cluster_id']:3d}{marker}  "
            f"surrogate ρ_seed={result['surrogate_rho_seed']:6.2f} → "
            f"ρ_polished={result['surrogate_rho_polished']:6.3f}  "
            f"(rotvec_a={result['rotvec_a_pol_mag_deg']:6.2f}°, "
            f"|ω_a|Δ={result['om_a_change_pct']:+.2f}%, "
            f"q0_err={result['q0_err_polished_deg']:.2f}°, "
            f"|ω|_err={result['om_mag_err_pct']:+.2f}%, "
            f"ω_dir_err={result['om_dir_err_deg']:.2f}°, "
            f"n_eval={result['n_eval']:3d}, wall={result['wall_s']:.1f}s)")
        polished.append(result)

    # surrogate-ρ gate before hi-fi
    log(f"\napplying surrogate-ρ < {SURROGATE_RHO_HIFI_GATE} gate before hi-fi...")
    n_pass = sum(1 for p in polished
                 if p["surrogate_rho_polished"] < SURROGATE_RHO_HIFI_GATE)
    log(f"  {n_pass}/{len(polished)} candidates pass; hi-fi-rendering only those")

    for p in polished:
        if p["surrogate_rho_polished"] < SURROGATE_RHO_HIFI_GATE:
            t0 = time.time()
            try:
                pred = render_hifi(p["q0_pol_wxyz"], p["om0_pol_rad"], ctx)
                rho_h = rho_from_hifi(pred, target)
                band = rho_band(rho_h)
            except Exception as e:
                log(f"  cluster_id={p['cluster_id']}: hi-fi FAILED: {e}")
                rho_h, band = float("nan"), "ERR"
                pred = np.full_like(target, np.nan)
            p["pred_hifi"] = pred
            p["rho_polished_hifi"] = float(rho_h)
            p["band_polished_hifi"] = band
            p["hifi_render_s"] = time.time() - t0
            marker = " (TRUTH)" if p["is_truth_cluster"] else ""
            log(f"  HI-FI cluster_id={p['cluster_id']:3d}{marker}  "
                f"surrogate ρ={p['surrogate_rho_polished']:.3f} → "
                f"hi-fi ρ={rho_h:.3f}  band={band}")
        else:
            p["pred_hifi"] = None
            p["rho_polished_hifi"] = float("nan")
            p["band_polished_hifi"] = "GATED"
            p["hifi_render_s"] = 0.0

    return polished


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", default=str(SURVEY / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_root) / f"s059c_seed{args.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir = SURVEY / "results" / f"s059_seed{args.seed:03d}"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_buf = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_buf.append(line)
        log_path.write_text("\n".join(log_buf) + "\n")

    t_overall = time.time()
    log(f"=== s059c anchor-frame polish — seed {args.seed} ===")

    log("\n[1/4] cloud generation (reusing s059's cache)")
    cloud = stage_cloud_generation(args.seed, cloud_dir, log)

    # release heavy state
    import experiments.s059_pilot as s059
    s059._R_CACHE = s059._SUN_UNIT = s059._OBS_UNIT = None
    s059._OBS_DIST = s059._MAG_TARGET = s059._SURROGATE = None
    import gc; gc.collect()

    log("\n[2/4] anchor + forward-prop scoring")
    ctx = build_context(seed=args.seed)
    T_A = stage_pick_anchor(cloud["survive_all"], log)
    fp = stage_forward_prop(args.seed, cloud, T_A, ctx, log)

    log("\n[3/4] canonicalise + cluster")
    cl = stage_cluster(fp, log)

    log("\n[4/4] ANCHOR-FRAME LM polish + surrogate-ρ gate + hi-fi classify")
    polished = stage_polish_anchor(args.seed, fp, cl, ctx, log)

    bands = [p["band_polished_hifi"] for p in polished]
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for b in bands:
        counts[b] = counts.get(b, 0) + 1
    n_AB = counts["A"] + counts["B"]
    log(f"\n=== HEADLINE: seed {args.seed} (anchor-frame polish) ===")
    log(f"  Bands: A={counts['A']} B={counts['B']} C={counts['C']} D={counts['D']} "
        f"GATED={counts['GATED']} ERR={counts['ERR']}")
    log(f"  Band A∪B yield: {n_AB}/{len(polished)} polished candidates")
    log(f"  Wall total: {(time.time()-t_overall)/60:.1f} min "
        f"({time.time()-t_overall:.0f}s)")

    summary = {
        "seed": args.seed, "T_A": T_A,
        "n_candidates": int(len(fp["Q_A_pass"])),
        "n_clusters": int(len(cl["clusters"])),
        "truth_cluster_rank": cl["truth_cluster_rank"],
        "discrimination_ratio": float(
            fp["scores"].max() / max(fp["null_score"], 1e-6)),
        "band_counts": counts, "n_band_AB": n_AB, "n_polished": len(polished),
        "n_passed_surrogate_gate": sum(
            1 for p in polished if p["surrogate_rho_polished"] < SURROGATE_RHO_HIFI_GATE),
        "wall_total_s": time.time() - t_overall,
        "polished": [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in p.items() if k != "pred_hifi"}
            for p in polished
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
