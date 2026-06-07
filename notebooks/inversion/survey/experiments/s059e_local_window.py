"""s059e — local-window LM polish (the principled fix).

s059d's early-anchor variant only addressed back-prop sensitivity. The
forward-prop from t=0 to t=3600s ALSO accumulates |ω|·t error, so the
residual at late epochs is pathologically sensitive to initial conditions.

Local-window cost: compute residual only over [t_a - W, t_a + W] epochs
around the anchor. Inside each LM eval, propagate (q_a, ω_a) outward
(short forward + short backward integration) to fill the local window.
Jacobian sensitivity is bounded by |ω|·W instead of |ω|·T.

W is chosen below the const-ω reliability threshold (s057g smoke). For
seed 28, const-ω was reliable to Δ=15 at T_A=312. With full dynamics
inside the integrator, accuracy at ±10 epochs is high.

After LM converges to a precise (q_a*, ω_a*) on the local window, do
ONE back-prop to (q_0*, ω_0*) for reporting + render the full-trajectory
hi-fi LC for ρ-band classification.

Usage:
    python experiments/s059e_local_window.py --seed 28
    python experiments/s059e_local_window.py --seed 28 --window 10 --t-a-window 30
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

from experiments.s059_pilot import (
    stage_cloud_generation, stage_forward_prop, stage_cluster,
    back_propagate, wxyz_to_xyzw, xyzw_to_wxyz, quat_ang_deg,
    LM_MAX_NFEV, LM_FTOL, LM_XTOL, RESIDUAL_CAP,
    SURROGATE_RHO_HIFI_GATE, TOP_K_CLUSTERS,
)
from experiments.s059d_early_anchor import stage_pick_early_anchor
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band
from lib.forward import propagate_to_body_frame
from lib.surrogate_eval import predict as surrogate_predict
from src.dynamics.attitude_propagator import propagate_attitude


WINDOW_EPOCHS_DEFAULT = 10  # ±W epochs around anchor → 21-epoch local cost


def propagate_local_window(q_a_wxyz, om_a_rad, t_a_idx, W, ctx):
    """Propagate (q_a, om_a) at observation_times[t_a_idx] outward to fill
    [t_a_idx - W, t_a_idx + W] inclusive.

    Strategy: integrate backward from anchor to lo, then forward from lo
    to hi as a single propagate_to_body_frame call (which expects monotone
    times starting from times[0]).
    """
    obs_times = ctx["observation_times"]
    n = len(obs_times)
    lo = max(0, t_a_idx - W)
    hi = min(n, t_a_idx + W + 1)
    times_local = obs_times[lo:hi]

    # back-prop from anchor to lo
    dt_back = float(obs_times[t_a_idx] - obs_times[lo])
    if dt_back > 0:
        q_lo, om_lo = back_propagate(q_a_wxyz, om_a_rad, dt_back,
                                      ctx["inertia_tensor"])
    else:
        q_lo, om_lo = q_a_wxyz, om_a_rad

    # forward-prop from lo across the window
    k1, k2, _ = propagate_to_body_frame(
        q0_wxyz=q_lo, omega0_rad=om_lo,
        observation_times=times_local,
        sun_pos=ctx["sun_pos"][lo:hi],
        obs_pos=ctx["obs_pos"][lo:hi],
        sat_pos=ctx["sat_pos"][lo:hi],
        inertia_tensor=ctx["inertia_tensor"],
        mode="tumbling",
    )
    return k1, k2, lo, hi


def make_residual_local(q_a_rep_wxyz, om_a_rep_rad, t_a_idx, W, ctx, target):
    q_a_rep_xyzw = wxyz_to_xyzw(q_a_rep_wxyz)
    R_seed = Rotation.from_quat(q_a_rep_xyzw)

    def residual(params):
        rotvec_a = params[:3]
        om_a = params[3:]
        try:
            q_a_xyzw = (Rotation.from_rotvec(rotvec_a) * R_seed).as_quat()
            q_a_wxyz = xyzw_to_wxyz(q_a_xyzw)
            k1, k2, lo, hi = propagate_local_window(
                q_a_wxyz, om_a, t_a_idx, W, ctx)
            pred = surrogate_predict(k1, k2, ctx["obs_dist"][lo:hi])
            r = pred - target[lo:hi]
            r = np.where(np.isfinite(r), r, RESIDUAL_CAP)
            return np.clip(r, -RESIDUAL_CAP, RESIDUAL_CAP)
        except Exception:
            return np.full(2 * W + 1, RESIDUAL_CAP)

    return residual


def lm_polish_local(q_a_rep_wxyz, om_a_rep_rad, t_a_idx, W, ctx, target):
    residual = make_residual_local(q_a_rep_wxyz, om_a_rep_rad, t_a_idx, W,
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

    # ONE back-prop of polished anchor state to t=0
    t_a_seconds = float(ctx["observation_times"][t_a_idx] -
                         ctx["observation_times"][0])
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
        "surrogate_mse_local_seed": mse_seed,
        "surrogate_mse_local_polished": mse_pol,
        "surrogate_rho_local_seed": float(np.sqrt(mse_seed) / 0.05),
        "surrogate_rho_local_polished": float(np.sqrt(mse_pol) / 0.05),
        "n_eval": int(result.nfev), "wall_s": wall,
    }


def stage_polish_local(seed, fp, cl, ctx, W, log):
    target = ctx["mag_hifi_truth"]
    t_a_idx = fp["T_A"]
    log(f"local cost window: ±{W} epochs (= {2*W+1} epochs total) around T_A={t_a_idx}")

    to_polish = list(cl["clusters_sorted"][:TOP_K_CLUSTERS])
    truth_cl = next(c for c in cl["clusters"]
                    if c["cluster_id"] == cl["truth_cluster_id"])
    if truth_cl not in to_polish:
        to_polish.append(truth_cl)
    log(f"polishing {len(to_polish)} clusters with LOCAL-WINDOW cost (top {TOP_K_CLUSTERS} + truth)")

    polished = []
    for c in to_polish:
        rank = next(i for i, cc in enumerate(cl["clusters_sorted"])
                    if cc["cluster_id"] == c["cluster_id"]) + 1
        is_truth = c["cluster_id"] == cl["truth_cluster_id"]
        bm = c["best_member_idx"]
        q_a = fp["Q_A_pass"][bm]
        om_a = fp["om_pass"][bm]

        result = lm_polish_local(q_a, om_a, t_a_idx, W, ctx, target)
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
            f"local ρ_seed={result['surrogate_rho_local_seed']:6.2f} → "
            f"ρ_polished={result['surrogate_rho_local_polished']:6.3f}  "
            f"(rotvec_a={result['rotvec_a_pol_mag_deg']:6.2f}°, "
            f"|ω_a|Δ={result['om_a_change_pct']:+.2f}%, "
            f"q0_err={result['q0_err_polished_deg']:.2f}°, "
            f"|ω|_err={result['om_mag_err_pct']:+.2f}%, "
            f"ω_dir_err={result['om_dir_err_deg']:.2f}°, "
            f"n_eval={result['n_eval']:3d}, wall={result['wall_s']:.1f}s)")
        polished.append(result)

    # Hi-fi the FULL trajectory regardless of local ρ — the local window
    # might be Band A while full is Band D (we want to measure that gap).
    log(f"\nhi-fi rendering all {len(polished)} candidates over full trajectory...")
    for p in polished:
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
            f"local ρ={p['surrogate_rho_local_polished']:.3f} → "
            f"full-traj hi-fi ρ={rho_h:.3f}  band={band}")

    return polished


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--window", type=int, default=WINDOW_EPOCHS_DEFAULT,
                   help="local cost window radius in epochs")
    p.add_argument("--t-a-window", type=int, default=30,
                   help="search anchor T_A in early epochs [3, 3+t_a_window)")
    p.add_argument("--out-root", default=str(SURVEY / "results"))
    args = p.parse_args()

    out_dir = Path(args.out_root) / f"s059e_seed{args.seed:03d}_W{args.window}"
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
    log(f"=== s059e local-window — seed {args.seed} (W={args.window} epochs, t_a window={args.t_a_window}) ===")

    log("\n[1/4] cloud generation (reuse s059's cache)")
    cloud = stage_cloud_generation(args.seed, cloud_dir, log)

    import experiments.s059_pilot as s059
    s059._R_CACHE = s059._SUN_UNIT = s059._OBS_UNIT = None
    s059._OBS_DIST = s059._MAG_TARGET = s059._SURROGATE = None
    import gc; gc.collect()

    log("\n[2/4] anchor + forward-prop scoring")
    ctx = build_context(seed=args.seed)
    T_A = stage_pick_early_anchor(cloud["survive_all"], log,
                                   window=args.t_a_window)
    fp = stage_forward_prop(args.seed, cloud, T_A, ctx, log)

    log("\n[3/4] canonicalise + cluster")
    cl = stage_cluster(fp, log)

    log(f"\n[4/4] LOCAL-WINDOW LM polish (W={args.window}) + full-trajectory hi-fi classify")
    polished = stage_polish_local(args.seed, fp, cl, ctx, args.window, log)

    bands = [p["band_polished_hifi"] for p in polished]
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for b in bands:
        counts[b] = counts.get(b, 0) + 1
    n_AB = counts["A"] + counts["B"]
    log(f"\n=== HEADLINE: seed {args.seed} (local-window W={args.window}, T_A={T_A}) ===")
    log(f"  Bands: A={counts['A']} B={counts['B']} C={counts['C']} D={counts['D']} "
        f"GATED={counts['GATED']} ERR={counts['ERR']}")
    log(f"  Band A∪B yield: {n_AB}/{len(polished)} polished candidates")
    log(f"  Wall total: {(time.time()-t_overall)/60:.1f} min "
        f"({time.time()-t_overall:.0f}s)")

    summary = {
        "seed": args.seed, "T_A": T_A, "window": args.window,
        "n_candidates": int(len(fp["Q_A_pass"])),
        "n_clusters": int(len(cl["clusters"])),
        "truth_cluster_rank": cl["truth_cluster_rank"],
        "discrimination_ratio": float(
            fp["scores"].max() / max(fp["null_score"], 1e-6)),
        "band_counts": counts, "n_band_AB": n_AB, "n_polished": len(polished),
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
