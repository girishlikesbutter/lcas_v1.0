"""s036: Multi-seed generalisation pilot — density+polish+hifi on seeds 23/57/28.

Tests whether the s033→s034→s035 chain (bracket-padded density + LM polish +
hi-fi confirm) generalises beyond seed 89. Three cohort-representative seeds:
  - Seed 23: good-bracket (nearest_cell=5.6%), zero-survivor at default density
  - Seed 57: highest-survivor (19,847), nearest_cell=49% (far bracket — hard)
  - Seed 28: mid-bracket (nearest_cell=7.9%), known sub-Sobol-narrow basin (s006)

Pipeline per seed:
  1. s020 at N_dir=2000, N_mag=20, N_phi=12 (subprocess)
  2. LM polish top-50 survivors by surrogate-MSE
  3. Hi-fi confirm any surrogate Band A candidates

Usage:
    cd notebooks/inversion/survey
    python experiments/s036_multi_seed_pilot.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "false"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import subprocess
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.filter_costs import load_static_geometry
from lib.surrogate_eval import get_model
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

PILOT_SEEDS = [23, 57, 28]
N_DIR = 2000
N_MAG = 20
N_PHI = 12
N_TOP = 50
LM_MAX_NFEV = 500
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
N_HIFI_WORKERS = 8

RESULTS_ROOT = SURVEY_DIR / "results" / "s036_multi_seed_pilot"


# ─── Helpers (from s034) ────────────────────────────────────────────────────────

def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def quat_to_R_i2b_batch(q_arr):
    qxyzw = q_arr[:, [1, 2, 3, 0]]
    return Rotation.from_quat(qxyzw).as_matrix()


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def build_residual(q0_init, omega_init, obs_times, inertia,
                   sun_unit, obs_unit, obs_dist, mag_truth, valid_mask,
                   surrogate):
    mag_truth_valid = mag_truth[valid_mask]

    def residual(x):
        rotvec = x[:3]
        omega_delta = x[3:]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
        q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0_new = quat_mul(q_pert_wxyz, q0_init)
        omega_new = omega_init + omega_delta

        q_traj, _ = propagate_attitude(
            q0_new, omega_new, obs_times,
            mode="tumbling", inertia_tensor=inertia,
        )
        R_full = quat_to_R_i2b_batch(q_traj)
        k1_body = np.einsum('eij,ej->ei', R_full, sun_unit)
        k2_body = np.einsum('eij,ej->ei', R_full, obs_unit)

        mag_pred = surrogate.predict_magnitude(
            k1_body, k2_body, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist,
        )
        return (mag_pred[valid_mask] - mag_truth_valid).astype(np.float64)

    return residual


# ─── Hi-fi Pool worker ──────────────────���───────────────────────��───────────────

_WORKER_CTX = None

def _init_hifi_worker(ctx):
    global _WORKER_CTX
    _WORKER_CTX = ctx

def _render_one(args):
    idx, q0_wxyz, omega_rad = args
    ctx = _WORKER_CTX
    mag_pred = render_hifi(np.array(q0_wxyz), np.array(omega_rad), ctx)
    rho = rho_from_hifi(mag_pred, ctx["mag_hifi_truth"])
    return idx, rho, mag_pred


# ─── Per-seed pipeline ──────────────��───────────────────────────────────────────

def run_density_pass(seed: int, out_dir: Path) -> Path:
    """Run s020 pipeline with enhanced density via subprocess."""
    seed_dir = out_dir / f"seed{seed:03d}"
    if seed_dir.exists() and (seed_dir / "summary.json").exists():
        print(f"  [density] seed {seed}: already exists, skipping")
        return seed_dir

    cmd = [
        sys.executable, str(SURVEY_DIR / "experiments" / "s020_seed_pipeline.py"),
        str(seed),
        "--out-root", str(out_dir),
        "--n-dir", str(N_DIR),
        "--n-mag", str(N_MAG),
        "--n-phi", str(N_PHI),
    ]
    print(f"  [density] seed {seed}: running s020 at N_dir={N_DIR}, N_mag={N_MAG}, N_phi={N_PHI}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0
    if result.returncode != 0:
        print(f"  [density] seed {seed} FAILED (wall={wall:.0f}s):")
        print(result.stderr[-2000:] if result.stderr else "no stderr")
        return None
    print(f"  [density] seed {seed}: done in {wall:.0f}s")
    return seed_dir


def run_lm_polish(seed: int, seed_dir: Path, surrogate, inertia) -> list:
    """LM polish top-50 survivors by surrogate-MSE. Returns polished list."""
    surv = np.load(seed_dir / "survivor_lcs.npz")
    cands = np.load(seed_dir / "candidates_meta.npz")
    omega_grid = np.load(seed_dir / "omega_grid.npz")
    traj = np.load(SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz")

    surv_idx = surv["survivor_cand_idx"]
    surv_mag_pred = surv["survivor_mag_pred"]
    truth_mag_hifi = surv["truth_mag_hifi"]
    obs_times = surv["observation_times"]
    valid_mask = np.isfinite(truth_mag_hifi)

    if surv_idx.size == 0:
        print(f"  [polish] seed {seed}: 0 survivors — skipping")
        return []

    # Surrogate MSE per survivor
    mse_all = np.mean((surv_mag_pred[:, valid_mask]
                       - truth_mag_hifi[None, valid_mask])**2, axis=1)
    n_take = min(N_TOP, len(mse_all))
    order = np.argsort(mse_all)[:n_take]
    print(f"  [polish] seed {seed}: {surv_idx.size} survivors, polishing top-{n_take}")
    print(f"           surrogate-MSE range: {mse_all[order[0]]:.4f} - {mse_all[order[-1]]:.4f}"
          f" (ρ {np.sqrt(mse_all[order[0]])/0.05:.2f} - {np.sqrt(mse_all[order[-1]])/0.05:.2f})")

    cand_q0_all = cands["q0"]
    cand_omega_idx = cands["omega_cell_idx"]
    omega_vectors = omega_grid["omega_vectors"]

    top_global_idx = surv_idx[order]
    top_q0_init = cand_q0_all[top_global_idx].astype(np.float64)
    top_omega_init = omega_vectors[cand_omega_idx[top_global_idx]].astype(np.float64)
    top_mse_init = mse_all[order]

    # Truth references
    q0_truth = traj["q0_wxyz"].astype(np.float64)
    omega_truth = traj["omega0_rad"].astype(np.float64)
    q_180x = np.array([0.0, 1.0, 0.0, 0.0])
    twin_q0 = quat_mul(q_180x, q0_truth)

    # Forward-model inputs
    sun_pos = traj["sun_pos"]
    obs_pos = traj["obs_pos"]
    sat_pos = traj["sat_pos"]
    obs_dist = traj["obs_dist"].astype(np.float64)
    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = (sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)).astype(np.float64)
    obs_unit = (obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)).astype(np.float64)

    polished = []
    t0 = time.time()
    for i in range(n_take):
        q0_init = top_q0_init[i] / np.linalg.norm(top_q0_init[i])
        omega_init = top_omega_init[i]
        mse_init = top_mse_init[i]

        residual = build_residual(
            q0_init, omega_init, obs_times, inertia,
            sun_unit, obs_unit, obs_dist, truth_mag_hifi, valid_mask, surrogate,
        )

        x0 = np.zeros(6)
        try:
            result = least_squares(
                residual, x0, method='lm',
                max_nfev=LM_MAX_NFEV,
                xtol=1e-8, ftol=1e-8,
            )
            converged = True
            n_iter = result.nfev
            mse_final = float(np.mean(result.fun ** 2))
            rotvec = result.x[:3]
            omega_delta = result.x[3:]
            q_pert = Rotation.from_rotvec(rotvec).as_quat()
            q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
            q0_final = quat_mul(q_pert_wxyz, q0_init)
            omega_final = omega_init + omega_delta
        except Exception as e:
            converged = False
            n_iter = -1
            mse_final = mse_init
            q0_final = q0_init
            omega_final = omega_init

        q0_to_truth = angular_dist_deg(q0_final, q0_truth)
        q0_to_twin = angular_dist_deg(q0_final, twin_q0)
        omega_dir_truth = np.degrees(np.arccos(np.clip(
            np.dot(omega_final, omega_truth) /
            (np.linalg.norm(omega_final) * np.linalg.norm(omega_truth) + 1e-30),
            -1, 1)))
        omega_mag_pct = (np.linalg.norm(omega_final) - np.linalg.norm(omega_truth)) \
                        / (np.linalg.norm(omega_truth) + 1e-30) * 100

        polished.append({
            "rank": i,
            "global_idx": int(top_global_idx[i]),
            "mse_init": float(mse_init),
            "mse_final": float(mse_final),
            "rho_pred_init": float(np.sqrt(mse_init) / 0.05),
            "rho_pred_final": float(np.sqrt(mse_final) / 0.05),
            "n_iter": int(n_iter),
            "q0_final": q0_final.tolist(),
            "omega_final": omega_final.tolist(),
            "q0_to_truth_deg": float(q0_to_truth),
            "q0_to_twin_deg": float(q0_to_twin),
            "omega_dir_to_truth_deg": float(omega_dir_truth),
            "omega_mag_pct": float(omega_mag_pct),
            "converged": converged,
        })

    wall_polish = time.time() - t0
    n_band_a = sum(1 for p in polished if p["rho_pred_final"] < 2.0)
    best = min(polished, key=lambda p: p["rho_pred_final"])
    print(f"  [polish] seed {seed}: done in {wall_polish:.0f}s. "
          f"Band A (surr): {n_band_a}/{n_take}. "
          f"Best ρ={best['rho_pred_final']:.3f}, q0→truth={best['q0_to_truth_deg']:.2f}°, "
          f"ω-dir={best['omega_dir_to_truth_deg']:.2f}°, ω-mag={best['omega_mag_pct']:.2f}%")

    return polished, wall_polish


def run_hifi_confirm(seed: int, polished: list) -> dict:
    """Hi-fi confirm surrogate Band A candidates. Returns summary dict."""
    band_a = [c for c in polished if c["rho_pred_final"] < 2.0]
    if not band_a:
        print(f"  [hifi] seed {seed}: 0 surrogate-Band-A candidates — skipping")
        return {"n_candidates": 0, "n_band_a": 0, "n_band_b": 0}

    band_a.sort(key=lambda x: x["rho_pred_final"])
    print(f"  [hifi] seed {seed}: rendering {len(band_a)} surrogate-Band-A candidates in Pool({N_HIFI_WORKERS})")

    ctx = build_context(seed)
    work = [(i, c["q0_final"], c["omega_final"]) for i, c in enumerate(band_a)]

    t0 = time.time()
    fork_ctx = mp.get_context("fork")
    with fork_ctx.Pool(N_HIFI_WORKERS, initializer=_init_hifi_worker, initargs=(ctx,)) as pool:
        results_raw = pool.map(_render_one, work)
    wall_hifi = time.time() - t0
    print(f"  [hifi] seed {seed}: done in {wall_hifi:.1f}s ({wall_hifi/len(band_a):.1f}s/cand)")

    results_raw.sort(key=lambda x: x[0])
    hifi_rhos = np.array([r[1] for r in results_raw])

    n_a = sum(1 for r in hifi_rhos if r < 2.0)
    n_b = sum(1 for r in hifi_rhos if 2.0 <= r < 4.0)
    n_c = sum(1 for r in hifi_rhos if 4.0 <= r < 8.0)
    n_d = sum(1 for r in hifi_rhos if r >= 8.0)

    print(f"  [hifi] seed {seed}: Band A={n_a}, B={n_b}, C={n_c}, D={n_d} | "
          f"ρ min={hifi_rhos.min():.3f}, med={np.median(hifi_rhos):.3f}, max={hifi_rhos.max():.3f}")

    # Annotate band_a records with hi-fi results
    for i, c in enumerate(band_a):
        c["rho_hifi"] = float(hifi_rhos[i])
        c["band_hifi"] = rho_band(hifi_rhos[i])

    return {
        "n_candidates": len(band_a),
        "n_band_a": n_a,
        "n_band_b": n_b,
        "n_band_c": n_c,
        "n_band_d": n_d,
        "rho_hifi_min": float(hifi_rhos.min()),
        "rho_hifi_median": float(np.median(hifi_rhos)),
        "rho_hifi_max": float(hifi_rhos.max()),
        "wall_hifi_s": wall_hifi,
        "candidates": band_a,
    }


# ─── Main ─────────────��────────────────────────────────────────────────────────

def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Pre-load shared resources
    geo = load_static_geometry()
    inertia = geo["inertia_tensor"].astype(np.float64)
    surrogate = get_model()
    print(f"s036: multi-seed pilot on seeds {PILOT_SEEDS}")
    print(f"      N_dir={N_DIR}, N_mag={N_MAG}, N_phi={N_PHI}, LM top-{N_TOP}")
    print(f"      inertia loaded, surrogate loaded\n")

    all_results = {}
    t_total = time.time()

    for seed in PILOT_SEEDS:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")
        t_seed = time.time()

        # Stage 1: density pass
        seed_dir = run_density_pass(seed, RESULTS_ROOT)
        if seed_dir is None:
            all_results[seed] = {"status": "density_failed"}
            continue

        # Stage 2: LM polish
        polish_result = run_lm_polish(seed, seed_dir, surrogate, inertia)
        if not polish_result or not polish_result[0]:
            all_results[seed] = {"status": "no_survivors", "wall_total_s": time.time() - t_seed}
            continue
        polished, wall_polish = polish_result

        # Stage 3: hi-fi confirm
        hifi_result = run_hifi_confirm(seed, polished)

        wall_seed = time.time() - t_seed
        all_results[seed] = {
            "status": "complete",
            "wall_total_s": wall_seed,
            "wall_polish_s": wall_polish,
            "n_survivors": sum(1 for _ in open(seed_dir / "summary.json")),  # placeholder
            "polish_summary": {
                "n_polished": len(polished),
                "n_surr_band_a": sum(1 for p in polished if p["rho_pred_final"] < 2.0),
                "best_rho_surr": min(p["rho_pred_final"] for p in polished),
                "best_q0_truth": min(p["q0_to_truth_deg"] for p in polished),
            },
            "hifi_summary": hifi_result,
        }

        # Save per-seed JSON
        seed_out = RESULTS_ROOT / f"seed{seed:03d}_result.json"
        with open(seed_out, "w") as f:
            json.dump({
                "seed": seed,
                "config": {"N_DIR": N_DIR, "N_MAG": N_MAG, "N_PHI": N_PHI, "N_TOP": N_TOP},
                "polished": polished,
                "hifi": hifi_result,
                "wall_total_s": wall_seed,
            }, f, indent=2)
        print(f"  Saved: {seed_out}")

    # Final summary
    wall_total = time.time() - t_total
    print(f"\n\n{'='*70}")
    print(f"  s036 SUMMARY — multi-seed pilot ({wall_total:.0f}s total)")
    print(f"{'='*70}")
    print(f"  {'Seed':>4} {'Status':<16} {'Surr A':>6} {'Hifi A':>6} {'Hifi B':>6}"
          f" {'Best ρ':>7} {'q0→t°':>6} {'ω-dir°':>7} {'Wall':>6}")
    print(f"  {'-'*4} {'-'*16} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")

    for seed in PILOT_SEEDS:
        r = all_results[seed]
        if r["status"] != "complete":
            print(f"  {seed:>4} {r['status']:<16}")
            continue
        ps = r["polish_summary"]
        hs = r["hifi_summary"]
        print(f"  {seed:>4} {'complete':<16} {ps['n_surr_band_a']:>6} "
              f"{hs['n_band_a']:>6} {hs['n_band_b']:>6} "
              f"{hs.get('rho_hifi_min', 999):>7.3f} "
              f"{ps['best_q0_truth']:>6.2f} "
              f"{'—':>7} {r['wall_total_s']:>5.0f}s")

    # Save overall summary
    summary_path = RESULTS_ROOT / "pilot_summary.json"
    summary_out = {
        "seeds": PILOT_SEEDS,
        "config": {"N_DIR": N_DIR, "N_MAG": N_MAG, "N_PHI": N_PHI, "N_TOP": N_TOP},
        "wall_total_s": wall_total,
        "per_seed": {str(s): all_results[s] for s in PILOT_SEEDS},
    }
    with open(summary_path, "w") as f:
        json.dump(summary_path_serializable(summary_out), f, indent=2)
    print(f"\nSaved: {summary_path}")


def summary_path_serializable(obj):
    """Make JSON-serializable (strip numpy, Path objects)."""
    if isinstance(obj, dict):
        return {k: summary_path_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [summary_path_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


if __name__ == "__main__":
    main()
