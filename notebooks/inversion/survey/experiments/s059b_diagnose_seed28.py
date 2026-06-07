"""s059b — diagnose why LM polish on seed 28 fails to bridge to Band A.

Hypothesis tree (s059i-style decomposition):
  H1: Surrogate is dishonest at this geometry — surrogate-MSE landscape is not
      tracking hi-fi-MSE around truth, so LM minimises a misleading cost.
  H2: Surrogate is honest, but back-prop seed is too far from truth-basin so LM
      converges to a local minimum (LC under-determines (q0,ω) here).
  H3: Surrogate is honest at exact truth but degrades sharply off-truth; LM's
      finite-diff Jacobian is mis-pointed.

Renders performed:
  R1. Hi-fi at exact truth     (must be ρ=0 ± machine precision)
  R2. Surrogate at exact truth (must be ~surrogate noise floor)
  R3. Hi-fi at truth-cluster's BACK-PROP seed (q0_back, om0_back)
      → quantifies how degraded the LM seed already was
  R4. Hi-fi at truth-cluster's POLISHED state
      → does hi-fi confirm surrogate's ρ=53 verdict?
  R5. Hi-fi at truth-cluster's q_a / om_a at t_a, FORWARD-propagated to t=0
      via the same back-prop chain but with NO polish. (= R3 essentially)
  R6. Hi-fi at top-1 cluster polished state (cluster_id=523, ρ_surrogate=52.6)
      → cross-check that all polishes are uniformly Band D, not just truth.

Plus a small ω-magnitude / direction sweep at fixed q0_truth to show whether
the surrogate-MSE landscape is tracking hi-fi-MSE around truth.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band
from lib.surrogate_eval import predict as surrogate_predict
from lib.forward import propagate_to_body_frame
from lib.traj_load import load_truth

SEED = 28
OUT_DIR = SURVEY / "results" / "s059b_diagnose_seed028"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def quat_ang_deg(q1, q2):
    return float(2 * np.degrees(np.arccos(
        np.clip(abs(np.dot(q1, q2)), 0, 1))))


def surrogate_rho(q0_wxyz, omega0_rad, ctx):
    k1, k2, _ = propagate_to_body_frame(
        q0_wxyz=np.asarray(q0_wxyz, dtype=np.float64),
        omega0_rad=np.asarray(omega0_rad, dtype=np.float64),
        observation_times=ctx["observation_times"],
        sun_pos=ctx["sun_pos"], obs_pos=ctx["obs_pos"],
        sat_pos=ctx["sat_pos"], inertia_tensor=ctx["inertia_tensor"],
        mode="tumbling",
    )
    pred = surrogate_predict(k1, k2, ctx["obs_dist"])
    diff = pred - ctx["mag_hifi_truth"]
    return float(np.sqrt(np.mean(diff ** 2)) / 0.05), pred


def main():
    log(f"=== s059b diagnose seed {SEED} ===")
    ctx = build_context(seed=SEED)
    truth = load_truth(SEED)
    q0_truth = ctx["q0_truth"]
    om0_truth = ctx["omega0_truth_rad"]
    log(f"truth |ω|0 = {np.linalg.norm(om0_truth):.6f} rad/s "
        f"({np.degrees(np.linalg.norm(om0_truth)):.4f} dps)")

    # Load polished states
    summary = json.load(open(SURVEY / "results/s059_seed028/summary.json"))
    truth_cl = next(p for p in summary["polished"] if p["is_truth_cluster"])
    top1_cl = next(p for p in summary["polished"] if p["cluster_rank"] == 1)

    # ---- R1/R2: smoke at exact truth ---------------------------------------
    log("\n[R1/R2] sanity at exact truth")
    t0 = time.time()
    pred_truth = render_hifi(q0_truth, om0_truth, ctx)
    rho_h_truth = rho_from_hifi(pred_truth, ctx["mag_hifi_truth"])
    log(f"  hi-fi ρ at truth = {rho_h_truth:.3e} (wall {time.time()-t0:.1f}s)  EXPECT ≈ 0")
    rho_s_truth, _ = surrogate_rho(q0_truth, om0_truth, ctx)
    log(f"  surrogate ρ at truth = {rho_s_truth:.3f}  EXPECT noise floor ~0.4")

    # ---- R3: hi-fi at truth-cluster BACK-PROP seed ------------------------
    log("\n[R3] hi-fi at truth cluster's back-prop seed (LM start)")
    q0_back = np.array(truth_cl["q0_back_wxyz"])
    om0_back = np.array(truth_cl["om0_back_rad"])
    log(f"  q0_back vs truth = {quat_ang_deg(q0_back, q0_truth):.2f}°")
    log(f"  |ω|_back = {np.linalg.norm(om0_back):.6f}  vs truth {np.linalg.norm(om0_truth):.6f}  "
        f"(Δ={(np.linalg.norm(om0_back)/np.linalg.norm(om0_truth)-1)*100:+.2f}%)")
    log(f"  ω_dir_back vs truth = {np.degrees(np.arccos(np.clip(abs(np.dot(om0_back/np.linalg.norm(om0_back), om0_truth/np.linalg.norm(om0_truth))), 0, 1))):.2f}°")
    rho_s_back, _ = surrogate_rho(q0_back, om0_back, ctx)
    t0 = time.time()
    pred_back = render_hifi(q0_back, om0_back, ctx)
    rho_h_back = rho_from_hifi(pred_back, ctx["mag_hifi_truth"])
    log(f"  surrogate ρ = {rho_s_back:.3f}  hi-fi ρ = {rho_h_back:.3f}  band={rho_band(rho_h_back)}  (wall {time.time()-t0:.1f}s)")

    # ---- R4: hi-fi at truth-cluster POLISHED -------------------------------
    log("\n[R4] hi-fi at truth cluster polished state")
    q0_pol = np.array(truth_cl["q0_pol_wxyz"])
    om0_pol = np.array(truth_cl["om0_pol_rad"])
    rho_s_pol = truth_cl["surrogate_rho_polished"]
    t0 = time.time()
    pred_pol = render_hifi(q0_pol, om0_pol, ctx)
    rho_h_pol = rho_from_hifi(pred_pol, ctx["mag_hifi_truth"])
    log(f"  surrogate ρ = {rho_s_pol:.3f}  hi-fi ρ = {rho_h_pol:.3f}  band={rho_band(rho_h_pol)}  (wall {time.time()-t0:.1f}s)")
    log(f"  q0_err = {truth_cl['q0_err_polished_deg']:.2f}°  |ω|_err = {truth_cl['om_mag_err_pct']:+.2f}%  ω_dir = {truth_cl['om_dir_err_deg']:.2f}°")

    # ---- R6: hi-fi at top-1 cluster polished ------------------------------
    log("\n[R6] hi-fi at top-1 (rank 1) cluster polished state")
    q0_p1 = np.array(top1_cl["q0_pol_wxyz"])
    om0_p1 = np.array(top1_cl["om0_pol_rad"])
    rho_s_p1 = top1_cl["surrogate_rho_polished"]
    t0 = time.time()
    pred_p1 = render_hifi(q0_p1, om0_p1, ctx)
    rho_h_p1 = rho_from_hifi(pred_p1, ctx["mag_hifi_truth"])
    log(f"  surrogate ρ = {rho_s_p1:.3f}  hi-fi ρ = {rho_h_p1:.3f}  band={rho_band(rho_h_p1)}  (wall {time.time()-t0:.1f}s)")

    # ---- ω-perturbation sweep at q0_truth ---------------------------------
    log("\n[sweep] surrogate vs hi-fi MSE around truth (q0=truth, vary ω)")
    log("  scale     |  ωΔ%   surrogate ρ    hi-fi ρ    Δρ")
    sweep_records = []
    for scale in [1.000, 1.001, 1.005, 1.01, 1.02, 1.05, 1.10, 0.999, 0.995, 0.99, 0.98, 0.95, 0.90]:
        om_pert = om0_truth * scale
        rho_s, _ = surrogate_rho(q0_truth, om_pert, ctx)
        pred = render_hifi(q0_truth, om_pert, ctx)
        rho_h = rho_from_hifi(pred, ctx["mag_hifi_truth"])
        log(f"  {scale:.3f}    | {(scale-1)*100:+.2f}%   {rho_s:8.3f}   {rho_h:8.3f}   {rho_s-rho_h:+8.3f}")
        sweep_records.append({"scale": scale, "rho_s": rho_s, "rho_h": rho_h})

    # ---- Summary -----------------------------------------------------------
    log("\n=== HEADLINE ===")
    log(f"  R1 truth-self hi-fi:        ρ = {rho_h_truth:.3e}   (sanity)")
    log(f"  R2 truth-self surrogate:    ρ = {rho_s_truth:.3f}")
    log(f"  R3 back-prop seed hi-fi:    ρ = {rho_h_back:.3f}   surrogate {rho_s_back:.3f}")
    log(f"  R4 truth-cluster polished:  ρ = {rho_h_pol:.3f}   surrogate {rho_s_pol:.3f}")
    log(f"  R6 top-1 polished:          ρ = {rho_h_p1:.3f}   surrogate {rho_s_p1:.3f}")

    # save
    out = {
        "seed": SEED,
        "rho_hifi_at_truth": rho_h_truth,
        "rho_surrogate_at_truth": rho_s_truth,
        "rho_hifi_back_prop_seed": rho_h_back,
        "rho_surrogate_back_prop_seed": rho_s_back,
        "rho_hifi_truth_cluster_polished": rho_h_pol,
        "rho_surrogate_truth_cluster_polished": rho_s_pol,
        "q0_err_back_prop_deg": quat_ang_deg(q0_back, q0_truth),
        "rho_hifi_top1_polished": rho_h_p1,
        "rho_surrogate_top1_polished": rho_s_p1,
        "sweep": sweep_records,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "run.log").write_text("\n".join(LOG_LINES) + "\n")
    log(f"Saved: {OUT_DIR / 'summary.json'}")
    log(f"Saved: {OUT_DIR / 'run.log'}")


if __name__ == "__main__":
    main()
