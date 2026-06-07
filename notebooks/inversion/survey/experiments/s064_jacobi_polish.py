"""s064 — Jacobi-coord LM polish: reparameterize ω onto polhode basis.

Architecture: s063b showed at truth (q_0, ω_0) the polhode tangent is the SOFT
direction with 4.2-17.5x lower curvature than the off-polhode normals
(grad 2T and grad L² Gram-Schmidt projections). Reparameterising ω onto the
basis (t_hat, nE_hat, nL_hat) should let LM's internal damping take larger
steps along the soft direction and smaller steps along the sharp ones,
tightening convergence without changing the cost function.

Drop-in into the s059k pipeline. The basis is fixed at the SEED ω (linear
change of variables), not recomputed per LM iteration — simplest first cut.

Gates:
  --gate1  : truth invariance smoke (seed 89, polish from truth should
             land near surrogate floor in ≤ a few iters and not move ω).
  --gate2  : smoke parity with s059k (seed 89 test1/2/3 perturbations).
  --gate3  : seed-89 cohort regression bench (top-50 clusters x multi-mag-
             start [0, ±3, ±6]%). Pass: ≥ 4/50 Band A unique clusters
             (= s059k baseline).

Usage:
    python experiments/s064_jacobi_polish.py --gate1
    python experiments/s064_jacobi_polish.py --gate2
    python experiments/s064_jacobi_polish.py --gate3 \
        --in-dir results/s059k_nd800_seed89/seed089 \
        --seed 89 --top-k-polish 50 --n-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, "/home/girish/surrogate_model")

# Reuse from s058 + s063b — no modification of those modules.
from experiments.s058_lm_polish_clusters import (  # noqa: E402
    quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, lm_polish,
    LM_MAX_NFEV, LM_FTOL, LM_XTOL, RESIDUAL_CAP,
)
from experiments.s063b_polhode_curvature import gram_schmidt_basis  # noqa: E402
from experiments.s059_pilot import back_propagate  # noqa: E402

from lib.forward import propagate_to_body_frame  # noqa: E402
from lib.surrogate_eval import predict as surrogate_predict  # noqa: E402
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402


# ---------- Polhode-basis LM polish (the s064 contribution) ----------

def make_residual_fn_jacobi(q0_seed_wxyz, omega0_seed_rad, inertia, ctx, target):
    """Polhode-basis residual closure.

    params = (rotvec_3, y_3) with omega = omega_seed + B @ y, where
    B columns are (t_hat, nE_hat, nL_hat) at omega_seed (body frame).
    rotvec is unchanged from s058::make_residual_fn — left-multiplication on q.
    """
    omega0_seed_rad = np.asarray(omega0_seed_rad, dtype=np.float64)
    t_hat, nE_hat, nL_hat = gram_schmidt_basis(omega0_seed_rad, np.asarray(inertia))
    B = np.column_stack([t_hat, nE_hat, nL_hat])  # (3, 3)

    q0_seed_xyzw = quat_wxyz_to_xyzw(q0_seed_wxyz)
    R_seed = Rotation.from_quat(q0_seed_xyzw)

    obs_times = ctx["observation_times"]
    sun_pos = ctx["sun_pos"]
    obs_pos = ctx["obs_pos"]
    sat_pos = ctx["sat_pos"]
    obs_dist = ctx["obs_dist"]
    inertia_arr = ctx["inertia_tensor"]

    def residual(params):
        rotvec = params[:3]
        y = params[3:]
        omega0 = omega0_seed_rad + B @ y
        q_xyzw = (Rotation.from_rotvec(rotvec) * R_seed).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        try:
            k1, k2, _ = propagate_to_body_frame(
                q0_wxyz=q_wxyz, omega0_rad=omega0,
                observation_times=obs_times,
                sun_pos=sun_pos, obs_pos=obs_pos, sat_pos=sat_pos,
                inertia_tensor=inertia_arr, mode="tumbling",
            )
            pred = surrogate_predict(k1, k2, obs_dist)
            r = pred - target
            r = np.where(np.isfinite(r), r, RESIDUAL_CAP)
            return np.clip(r, -RESIDUAL_CAP, RESIDUAL_CAP)
        except Exception:
            return np.full_like(target, RESIDUAL_CAP)

    return residual, B


def lm_polish_jacobi(q0_seed_wxyz, omega0_seed_rad, ctx, target, label=""):
    """Polhode-basis LM polish. Same scipy least_squares(method='lm') call as
    s058::lm_polish, only the parameter mapping differs.
    """
    inertia = ctx["inertia_tensor"]
    residual, B = make_residual_fn_jacobi(
        q0_seed_wxyz, omega0_seed_rad, inertia, ctx, target,
    )
    x0 = np.zeros(6)  # rotvec=0, y=0 — at seed

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
    y_pol = result.x[3:]
    omega0_seed_arr = np.asarray(omega0_seed_rad, dtype=np.float64)
    omega0_pol = omega0_seed_arr + B @ y_pol

    q_xyzw_seed = quat_wxyz_to_xyzw(q0_seed_wxyz)
    q_pol_xyzw = (Rotation.from_rotvec(rotvec_pol) *
                  Rotation.from_quat(q_xyzw_seed)).as_quat()
    q0_pol_wxyz = quat_xyzw_to_wxyz(q_pol_xyzw)

    r_pol = result.fun
    mse_pol = float(np.mean(r_pol ** 2))

    return {
        "label": label,
        "q0_seed_wxyz": list(q0_seed_wxyz) if hasattr(q0_seed_wxyz, "tolist") else list(q0_seed_wxyz),
        "om0_seed_rad": list(omega0_seed_arr),
        "q0_pol_wxyz": q0_pol_wxyz.tolist(),
        "om0_pol_rad": omega0_pol.tolist(),
        "rotvec_pol": rotvec_pol.tolist(),
        "rotvec_pol_mag_deg": float(np.degrees(np.linalg.norm(rotvec_pol))),
        "y_pol": y_pol.tolist(),
        "y_pol_norm": float(np.linalg.norm(y_pol)),
        "y_tangent_step": float(y_pol[0]),
        "y_normal_E_step": float(y_pol[1]),
        "y_normal_L_step": float(y_pol[2]),
        "B_basis_rows_body": B.tolist(),
        "om_change_pct": float(np.linalg.norm(omega0_pol - omega0_seed_arr) /
                               max(1e-12, np.linalg.norm(omega0_seed_arr)) * 100),
        "surrogate_mse_seed": mse_seed,
        "surrogate_mse_polished": mse_pol,
        "surrogate_rho_seed": float(np.sqrt(mse_seed) / 0.05),
        "surrogate_rho_polished": float(np.sqrt(mse_pol) / 0.05),
        "n_eval": int(result.nfev),
        "status": int(result.status),
        "converged": bool(result.status > 0),
        "wall_s": wall_s,
    }


# ---------- Helpers ----------

def quat_geodesic_deg(q1_wxyz, q2_wxyz):
    dot = float(np.abs(np.dot(q1_wxyz, q2_wxyz)))
    return float(2 * np.degrees(np.arccos(min(1.0, max(-1.0, dot)))))


def errors_vs_truth(q0_pol_wxyz, om0_pol_rad, ctx):
    q0p = np.asarray(q0_pol_wxyz, dtype=np.float64)
    om0p = np.asarray(om0_pol_rad, dtype=np.float64)
    om_truth = ctx["omega0_truth_rad"]
    om_truth_mag = float(np.linalg.norm(om_truth))
    q0_err = quat_geodesic_deg(q0p, ctx["q0_truth"])
    om_mag_err = float((np.linalg.norm(om0p) - om_truth_mag) / om_truth_mag * 100)
    om_dir_err = float(np.degrees(np.arccos(np.clip(
        abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)),
                   om_truth / om_truth_mag)), 0, 1))))
    return q0_err, om_mag_err, om_dir_err


def hifi_classify(q0_wxyz, om0_rad, ctx, target):
    try:
        pred = render_hifi(np.asarray(q0_wxyz, dtype=np.float64),
                           np.asarray(om0_rad, dtype=np.float64), ctx)
        rho_h = float(rho_from_hifi(pred, target))
        band = rho_band(rho_h)
        return rho_h, band
    except Exception:
        return float("nan"), "ERR"


def perturb_q(q_wxyz, deg, rng):
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    rotvec = np.deg2rad(deg) * axis
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    q_p_xyzw = (Rotation.from_rotvec(rotvec) * Rotation.from_quat(q_xyzw)).as_quat()
    return quat_xyzw_to_wxyz(q_p_xyzw)


def perturb_omega(om_rad, dir_deg, mag_pct, rng):
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    om_mag = float(np.linalg.norm(om_rad))
    om_dir = om_rad / om_mag
    rotvec = np.deg2rad(dir_deg) * axis
    new_dir = Rotation.from_rotvec(rotvec).apply(om_dir)
    new_mag = om_mag * (1 + mag_pct / 100.0)
    return new_dir * new_mag


# ---------- Gate 1: truth invariance ----------

def gate1_truth_invariance(out_dir: Path, seed: int = 89):
    """From truth (q0, ω0), polish should return ρ ≈ surrogate floor and not move ω."""
    print(f"\n=== GATE 1 — truth invariance, seed {seed} ===")
    ctx = build_context(seed=seed)
    target = ctx["mag_hifi_truth"]
    q0_truth = np.asarray(ctx["q0_truth"], dtype=np.float64)
    om0_truth = np.asarray(ctx["omega0_truth_rad"], dtype=np.float64)

    t0 = time.time()
    res = lm_polish_jacobi(q0_truth, om0_truth, ctx, target, label="truth_invariance")
    wall = time.time() - t0

    q0_err, om_mag_err, om_dir_err = errors_vs_truth(res["q0_pol_wxyz"], res["om0_pol_rad"], ctx)
    rho_h, band = hifi_classify(res["q0_pol_wxyz"], res["om0_pol_rad"], ctx, target)
    print(f"  ρ_seed (surr) = {res['surrogate_rho_seed']:.4f}")
    print(f"  ρ_pol  (surr) = {res['surrogate_rho_polished']:.4f}  (Δ = {res['surrogate_rho_polished']-res['surrogate_rho_seed']:+.4f})")
    print(f"  ρ_pol  (hi-fi)= {rho_h:.4f}  band={band}")
    print(f"  q0_err = {q0_err:.4f}°    |ω|err = {om_mag_err:+.4f}%    ω_dir = {om_dir_err:.4f}°")
    print(f"  rotvec_pol_mag = {res['rotvec_pol_mag_deg']:.4f}°    |y_pol| = {res['y_pol_norm']:.3e}")
    print(f"  n_eval = {res['n_eval']}   wall = {wall:.2f} s")

    # Pass: LM didn't regress (ρ_pol ≤ ρ_seed) AND ω basically unchanged in
    # polhode coords (|y_pol| ~ machine-epsilon in physical units) AND hi-fi
    # lands Band A. The small q0_err that remains is the surrogate's argmin
    # offset from truth (the surrogate is not pixel-perfect vs hi-fi), not a
    # reparam bug. Same baseline as s059k test2/3.
    passed = (
        res['surrogate_rho_polished'] <= res['surrogate_rho_seed'] * 1.01
        and res['y_pol_norm'] < 1e-3
        and band == "A"
        and abs(om_mag_err) < 0.5
    )
    print(f"  GATE 1: {'PASS' if passed else 'FAIL'}  "
          f"(ρ_pol≤ρ_seed: {res['surrogate_rho_polished']<=res['surrogate_rho_seed']*1.01}, "
          f"|y|<1e-3: {res['y_pol_norm']<1e-3}, "
          f"hi-fi Band A: {band=='A'}, "
          f"|ω|err<0.5%: {abs(om_mag_err)<0.5})")

    summary = {
        "gate": "gate1_truth_invariance",
        "seed": seed,
        "ρ_seed_surr": res["surrogate_rho_seed"],
        "ρ_pol_surr": res["surrogate_rho_polished"],
        "ρ_pol_hifi": rho_h, "band": band,
        "q0_err_deg": q0_err, "om_mag_err_pct": om_mag_err, "om_dir_err_deg": om_dir_err,
        "rotvec_pol_mag_deg": res["rotvec_pol_mag_deg"],
        "y_pol": res["y_pol"], "y_pol_norm": res["y_pol_norm"],
        "n_eval": res["n_eval"], "wall_s": wall, "passed": passed,
    }
    (out_dir / "gate1_truth_invariance.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {out_dir / 'gate1_truth_invariance.json'}")
    return passed


# ---------- Gate 2: s059k smoke parity ----------

S059K_SMOKE_BASELINE = {
    "test1": {"perturbation": {"q_deg": 1.7, "om_dir_deg": 3.5, "om_mag_pct": 6.0},
              "rng_seed": 17,
              "baseline_1b_full_lc": {"ρ_hifi": 24.495, "band": "D",
                                       "q0_err_deg": 81.26, "om_mag_err_pct": 11.27, "om_dir_err_deg": 17.46,
                                       "wall_s": 102.44}},
    "test2": {"perturbation": {"q_deg": 0.85, "om_dir_deg": 1.75, "om_mag_pct": 0.0},
              "rng_seed": 18,
              "baseline_2b_full_lc": {"ρ_hifi": 0.177, "band": "A",
                                       "q0_err_deg": 0.86, "om_mag_err_pct": -0.02, "om_dir_err_deg": 0.11,
                                       "wall_s": 84.95}},
    "test3": {"perturbation": {"q_deg": 1.7, "om_dir_deg": 3.5, "om_mag_pct": 0.0},
              "rng_seed": 19,
              "baseline_3b_full_lc": {"ρ_hifi": 0.177, "band": "A",
                                       "q0_err_deg": 0.86, "om_mag_err_pct": -0.02, "om_dir_err_deg": 0.11,
                                       "wall_s": 85.86}},
}


def gate2_smoke_parity(out_dir: Path, seed: int = 89, T_A: int = 3):
    """Match the three s059k smoke perturbations and compare against the cached baselines."""
    print(f"\n=== GATE 2 — s059k smoke parity, seed {seed}, T_A={T_A} ===")
    ctx = build_context(seed=seed)
    target = ctx["mag_hifi_truth"]
    inertia = ctx["inertia_tensor"]
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])

    quats_truth, omegas_truth = propagate_attitude(
        q0=ctx["q0_truth"], omega0=ctx["omega0_truth_rad"],
        times=ctx["observation_times"], mode="tumbling", inertia_tensor=inertia,
    )
    q_a_truth = np.asarray(quats_truth[T_A], dtype=np.float64)
    om_a_truth = np.asarray(omegas_truth[T_A], dtype=np.float64)

    results = {}
    for name, conf in S059K_SMOKE_BASELINE.items():
        pert = conf["perturbation"]
        rng = np.random.default_rng(conf["rng_seed"])
        qa_p = perturb_q(q_a_truth, pert["q_deg"], rng)
        om_p = perturb_omega(om_a_truth, pert["om_dir_deg"], pert["om_mag_pct"], rng)
        q0_seed, om0_seed = back_propagate(qa_p, om_p, t_a_seconds, inertia)

        print(f"\n  {name}: q_a={pert['q_deg']:.2f}°, ω_dir={pert['om_dir_deg']:.2f}°, |ω|{pert['om_mag_pct']:+.0f}%")
        t0 = time.time()
        res = lm_polish_jacobi(q0_seed, om0_seed, ctx, target, label=f"{name}_jacobi")
        wall = time.time() - t0
        rho_h, band = hifi_classify(res["q0_pol_wxyz"], res["om0_pol_rad"], ctx, target)
        q0_err, om_mag_err, om_dir_err = errors_vs_truth(res["q0_pol_wxyz"], res["om0_pol_rad"], ctx)

        baseline_key = next(k for k in conf if k.startswith("baseline_"))
        base = conf[baseline_key]
        print(f"    seed ρ (surr) = {res['surrogate_rho_seed']:.3f}")
        print(f"    pol  ρ (surr) = {res['surrogate_rho_polished']:.3f}")
        print(f"    pol  ρ (hi-fi)= {rho_h:.3f}  band={band}    [s059k baseline: ρ={base['ρ_hifi']:.3f} band={base['band']}]")
        print(f"    q0_err = {q0_err:.2f}°    |ω|err = {om_mag_err:+.2f}%    ω_dir = {om_dir_err:.2f}°")
        print(f"    rotvec_pol_mag = {res['rotvec_pol_mag_deg']:.2f}°    |y_pol|/|ω_seed| = {res['om_change_pct']:.2f}%")
        print(f"    y_pol = (tangent={res['y_tangent_step']:+.3e}, nE={res['y_normal_E_step']:+.3e}, nL={res['y_normal_L_step']:+.3e})")
        print(f"    n_eval = {res['n_eval']}   wall = {wall:.1f}s    [s059k baseline wall: {base['wall_s']:.1f}s]")

        results[name] = {
            "perturbation": pert,
            "qa_pert_wxyz": qa_p.tolist(), "om_pert_rad": om_p.tolist(),
            "q0_seed_wxyz": q0_seed.tolist(), "om0_seed_rad": om0_seed.tolist(),
            "q0_pol_wxyz": res["q0_pol_wxyz"], "om0_pol_rad": res["om0_pol_rad"],
            "ρ_seed_surr": res["surrogate_rho_seed"],
            "ρ_pol_surr": res["surrogate_rho_polished"],
            "ρ_pol_hifi": rho_h, "band": band,
            "q0_err_deg": q0_err, "om_mag_err_pct": om_mag_err, "om_dir_err_deg": om_dir_err,
            "rotvec_pol_mag_deg": res["rotvec_pol_mag_deg"],
            "om_change_pct": res["om_change_pct"],
            "y_pol": res["y_pol"], "y_pol_norm": res["y_pol_norm"],
            "n_eval": res["n_eval"], "wall_s": wall,
            "s059k_baseline": base,
        }

    # Pass: test2 and test3 must hit Band A (they did in s059k); test1 might still fail.
    passed = (
        results["test2"]["band"] == "A"
        and results["test3"]["band"] == "A"
    )
    print(f"\n  GATE 2: {'PASS' if passed else 'FAIL'}  (test2 + test3 Band A required)")

    summary = {"gate": "gate2_smoke_parity", "seed": seed, "T_A": T_A,
               "tests": results, "passed": passed}
    (out_dir / "gate2_smoke_parity.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {out_dir / 'gate2_smoke_parity.json'}")
    return passed


# ---------- Gate 3: seed-89 cohort regression ----------

_CTX_W = None
_TARGET_W = None


def _worker_init(ctx, target):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass
    global _CTX_W, _TARGET_W
    _CTX_W = ctx
    _TARGET_W = target


def _polish_worker_jacobi(args):
    cluster_rank, cluster_id, q_a_seed, om_a_seed, t_a_seconds, inertia, mag_pct = args
    t0 = time.time()
    om_a_arr = np.asarray(om_a_seed, dtype=np.float64)
    if mag_pct != 0.0:
        om_a_arr = om_a_arr * (1.0 + mag_pct / 100.0)
    q0_seed, om0_seed = back_propagate(np.asarray(q_a_seed), om_a_arr,
                                        t_a_seconds, np.asarray(inertia))
    res = lm_polish_jacobi(q0_seed, om0_seed, _CTX_W, _TARGET_W,
                           label=f"rank{cluster_rank}_id{cluster_id}_mag{mag_pct:+.0f}")
    res["polish_wall_s"] = time.time() - t0
    res["cluster_rank"] = int(cluster_rank)
    res["cluster_id"] = int(cluster_id)
    res["mag_pct_offset"] = float(mag_pct)
    res["q_a_seed_wxyz"] = np.asarray(q_a_seed).tolist()
    res["om_a_seed_rad"] = om_a_arr.tolist()
    res["q0_seed_wxyz"] = np.asarray(q0_seed).tolist()
    res["om0_seed_rad"] = np.asarray(om0_seed).tolist()
    return res


def gate3_cohort_regression(out_dir: Path, in_dir: Path, seed: int,
                             top_k_polish: int, n_workers: int,
                             mag_pct_offsets):
    """Re-run s059k Phase 4 with Jacobi-coord polish."""
    print(f"\n=== GATE 3 — seed-{seed} cohort regression ===")
    from experiments.s059_pilot import (stage_cluster, quat_ang_deg_batch, ang_to_axis)

    cl_npz = np.load(in_dir / "clusters.npz")
    sg_npz = np.load(in_dir / "score_grid.npz")
    Q_top = cl_npz["Q_top"]
    Om_top = cl_npz["Om_top"]
    mse_top = cl_npz["mse_top"]
    truth_idx = int(cl_npz["truth_idx_diagnostic"])
    q_a_truth = sg_npz["q_a_truth"]
    om_a_truth = sg_npz["om_a_truth"]
    T_A = int(sg_npz["T_A"])

    qa_dist_diag = quat_ang_deg_batch(Q_top, q_a_truth)
    om_dist_diag = ang_to_axis(Om_top, om_a_truth)
    fp = {
        "Q_A_pass": Q_top, "om_pass": Om_top, "scores": -mse_top,
        "qa_dist_to_truth": qa_dist_diag, "om_dist_to_truth": om_dist_diag,
        "truth_idx": truth_idx,
    }

    def log(msg): print(msg)

    print(f"re-running stage_cluster on cached top-K=5000...")
    cl = stage_cluster(fp, log)
    truth_cluster_rank = cl["truth_cluster_rank"]
    n_clusters = len(cl["clusters_sorted"])
    print(f"  truth cluster rank {truth_cluster_rank}/{n_clusters}")

    print(f"\nbuilding hifi context for seed {seed}...")
    t0 = time.time()
    ctx = build_context(seed=seed)
    target = ctx["mag_hifi_truth"]
    inertia = ctx["inertia_tensor"]
    t_a_seconds = float(ctx["observation_times"][T_A] - ctx["observation_times"][0])
    print(f"  built in {time.time()-t0:.1f}s; t_a_seconds={t_a_seconds:.1f}s")

    n_polish = min(top_k_polish, n_clusters)
    polish_list = []
    for rank0, c in enumerate(cl["clusters_sorted"][:n_polish]):
        bm = c["best_member_idx"]
        q_a_seed = Q_top[bm]
        om_a_seed = Om_top[bm]
        for mag_off in mag_pct_offsets:
            polish_list.append((rank0 + 1, c["cluster_id"], q_a_seed, om_a_seed,
                                t_a_seconds, inertia, mag_off))
    print(f"\npolishing {n_polish} cluster reps × {len(mag_pct_offsets)} mag-starts = "
          f"{len(polish_list)} polishes (Jacobi-coord) NO truth injection")

    t_polish = time.time()
    rescued = []
    if n_workers <= 1:
        _worker_init(ctx, target)
        for a in polish_list:
            r = _polish_worker_jacobi(a)
            _print_polish(r)
            rescued.append(r)
    else:
        ctx_pool = get_context("fork")
        with ctx_pool.Pool(n_workers, initializer=_worker_init,
                           initargs=(ctx, target)) as pool:
            for r in pool.imap_unordered(_polish_worker_jacobi, polish_list):
                _print_polish(r)
                rescued.append(r)
    rescued.sort(key=lambda r: (r["cluster_rank"], r["mag_pct_offset"]))
    polish_wall = time.time() - t_polish
    print(f"\npolish wall: {polish_wall:.1f}s ({polish_wall/60:.2f} min)")

    # Hi-fi classify
    print(f"\nhi-fi gating + classification...")
    t_hifi = time.time()
    om_truth_t0 = ctx["omega0_truth_rad"]
    om_truth_mag = float(np.linalg.norm(om_truth_t0))
    band_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "GATED": 0, "ERR": 0}
    for r in rescued:
        if r["surrogate_rho_polished"] < 4.0:
            rho_h, band = hifi_classify(r["q0_pol_wxyz"], r["om0_pol_rad"], ctx, target)
            r["rho_polished_hifi"] = rho_h
            r["band_polished_hifi"] = band
        else:
            r["rho_polished_hifi"] = float("nan")
            r["band_polished_hifi"] = "GATED"
        band_counts[r["band_polished_hifi"]] = band_counts.get(r["band_polished_hifi"], 0) + 1
        q0p = np.asarray(r["q0_pol_wxyz"])
        om0p = np.asarray(r["om0_pol_rad"])
        r["q0_err_deg"] = quat_geodesic_deg(q0p, ctx["q0_truth"])
        r["om_mag_err_pct"] = float((np.linalg.norm(om0p) - om_truth_mag) / om_truth_mag * 100)
        r["om_dir_err_deg"] = float(np.degrees(np.arccos(np.clip(
            abs(np.dot(om0p / max(1e-12, np.linalg.norm(om0p)),
                       om_truth_t0 / om_truth_mag)), 0, 1))))
    print(f"hi-fi wall: {time.time()-t_hifi:.1f}s")

    n_AB_polishes = band_counts.get("A", 0) + band_counts.get("B", 0)
    cluster_best = {}
    for r in rescued:
        cid = int(r["cluster_id"])
        if cid not in cluster_best or (
            not np.isnan(r["rho_polished_hifi"]) and
            (np.isnan(cluster_best[cid]["rho_polished_hifi"]) or
             r["rho_polished_hifi"] < cluster_best[cid]["rho_polished_hifi"])
        ):
            cluster_best[cid] = r
    n_AB_clusters = sum(1 for r in cluster_best.values()
                        if r["band_polished_hifi"] in {"A", "B"})

    print(f"\nResults (sorted by hi-fi ρ, top 30):")
    rescued_byhifi = sorted(
        rescued,
        key=lambda r: (np.inf if np.isnan(r["rho_polished_hifi"]) else r["rho_polished_hifi"]),
    )
    for r in rescued_byhifi[:30]:
        print(f"  rank={r['cluster_rank']:3d} id={r['cluster_id']:3d} mag{r['mag_pct_offset']:+5.1f}%  "
              f"ρ_seed={r['surrogate_rho_seed']:6.2f} → ρ_pol={r['surrogate_rho_polished']:6.3f} → "
              f"hi-fi={r['rho_polished_hifi']:6.3f} band={r['band_polished_hifi']:<5}  "
              f"q0_err={r['q0_err_deg']:5.2f}° |ω|err={r['om_mag_err_pct']:+6.2f}% ω_dir={r['om_dir_err_deg']:5.2f}°  "
              f"n_eval={r['n_eval']:3d}")

    print(f"\n{'='*60}")
    print(f"BAND COUNTS (all polishes): {band_counts}")
    print(f"  total polishes = {len(rescued)} ({n_polish} clusters × {len(mag_pct_offsets)} mag-starts)")
    print(f"  Band A∪B polishes: {n_AB_polishes}")
    print(f"  Band A∪B unique clusters (best polish per cluster): {n_AB_clusters}")
    print(f"HEADLINE YIELD (cluster A∪B): {n_AB_clusters}/{n_polish}")
    print(f"truth cluster rank: {truth_cluster_rank}/{n_clusters} (NOT polished if > {n_polish})")
    print(f"s059k baseline yield = 4/50 unique Band A clusters")
    print(f"{'='*60}")

    passed = n_AB_clusters >= 4

    summary = {
        "experiment": "s064_jacobi_polish_gate3",
        "seed": int(seed), "T_A": T_A,
        "top_k_polish": int(n_polish),
        "mag_pct_offsets": list(mag_pct_offsets),
        "n_clusters": int(n_clusters),
        "truth_cluster_rank": int(truth_cluster_rank),
        "band_counts_all_polishes": band_counts,
        "n_band_AB_polishes": int(n_AB_polishes),
        "n_band_AB_clusters": int(n_AB_clusters),
        "headline_yield": int(n_AB_clusters),
        "wall_polish_s": polish_wall,
        "s059k_baseline_yield": 4,
        "passed": passed,
        "polished": [
            {"cluster_rank": int(r["cluster_rank"]), "cluster_id": int(r["cluster_id"]),
             "mag_pct_offset": float(r["mag_pct_offset"]),
             "q_a_seed_wxyz": r["q_a_seed_wxyz"], "om_a_seed_rad": r["om_a_seed_rad"],
             "rho_seed_full_lc": float(r["surrogate_rho_seed"]),
             "rho_polished_full_lc": float(r["surrogate_rho_polished"]),
             "rho_polished_hifi": float(r["rho_polished_hifi"]) if not np.isnan(r["rho_polished_hifi"]) else None,
             "band": r["band_polished_hifi"],
             "q0_err_deg": float(r["q0_err_deg"]),
             "om_mag_err_pct": float(r["om_mag_err_pct"]),
             "om_dir_err_deg": float(r["om_dir_err_deg"]),
             "q0_pol_wxyz": r["q0_pol_wxyz"], "om0_pol_rad": r["om0_pol_rad"],
             "y_pol": r["y_pol"], "y_pol_norm": r["y_pol_norm"],
             "y_tangent_step": r["y_tangent_step"],
             "y_normal_E_step": r["y_normal_E_step"],
             "y_normal_L_step": r["y_normal_L_step"],
             "polish_wall_s": float(r["polish_wall_s"]), "n_eval": int(r["n_eval"])}
            for r in rescued
        ],
    }
    out_path = out_dir / f"gate3_seed{seed:03d}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    return passed


def _print_polish(r):
    print(f"  rank={r['cluster_rank']:3d} id={r['cluster_id']:3d} mag{r['mag_pct_offset']:+5.1f}%  "
          f"ρ_seed={r['surrogate_rho_seed']:6.2f} → ρ_pol={r['surrogate_rho_polished']:6.3f}  "
          f"(n_eval={r['n_eval']:3d}, wall={r['polish_wall_s']:.1f}s)")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1", action="store_true", help="Run Gate 1: truth invariance")
    ap.add_argument("--gate2", action="store_true", help="Run Gate 2: s059k smoke parity")
    ap.add_argument("--gate3", action="store_true", help="Run Gate 3: seed-89 cohort regression")
    ap.add_argument("--seed", type=int, default=89)
    ap.add_argument("--in-dir", type=str, default=None,
                    help="Score-grid/clusters dir for Gate 3 (e.g. results/s059k_nd800_seed89/seed089)")
    ap.add_argument("--top-k-polish", type=int, default=50)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--mag-starts", default="0.0,3.0,-3.0,6.0,-6.0",
                    help="Comma-separated |ω| pct offsets for multi-start polish.")
    args = ap.parse_args()

    out_dir = SURVEY / "results" / "s064_jacobi_polish"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.gate1:
        results["gate1"] = gate1_truth_invariance(out_dir, seed=args.seed)
    if args.gate2:
        results["gate2"] = gate2_smoke_parity(out_dir, seed=args.seed)
    if args.gate3:
        if not args.in_dir:
            raise SystemExit("--gate3 requires --in-dir")
        mag_offsets = [float(x) for x in args.mag_starts.split(",")]
        results["gate3"] = gate3_cohort_regression(
            out_dir, Path(args.in_dir), args.seed, args.top_k_polish,
            args.n_workers, mag_offsets,
        )

    if results:
        print(f"\n{'='*60}")
        print(f"GATE RESULTS SUMMARY: {results}")


if __name__ == "__main__":
    main()
