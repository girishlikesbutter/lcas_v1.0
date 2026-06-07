"""s037: Sobol-Shoemake q0 ICs at bracket ω-cells — replaces phi-sweep.

Tests whether Sobol N=64 on SO(3) at near-truth bracket cells delivers
LM-bridgeable ρ (<15) on seeds where phi-sweep failed (23, 28) plus the
reference seed (89).

Three diagnostic levels per seed:
  L0: Sobol N=64 at exact truth-ω (replicates s011 baseline)
  L1: Sobol N=64 at nearest bracket cell
  L2: Sobol N=64 at ALL 20 bracket cells (full scan)

If L1/L2 delivers ρ < 15 on seeds 23/28: LM polish + hi-fi confirm.

Usage:
    cd notebooks/inversion/survey
    python experiments/s037_sobol_bracket_pilot.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "false"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

SURVEY_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR))

from src.dynamics.attitude_propagator import propagate_attitude
from lib.forward import propagate_to_body_frame
from lib.surrogate_eval import get_model, predict
from lib.traj_load import load_truth
from lib.filter_costs import load_static_geometry
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

PILOT_SEEDS = [23, 28, 89]
N_SOBOL = 64
SOBOL_SEED = 42
N_BRACKET_CELLS = 20
LM_MAX_NFEV = 500
LM_TOP_K = 20
N_HIFI_WORKERS = 8
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0

RESULTS_DIR = SURVEY_DIR / "results" / "s037_sobol_bracket_pilot"


# ─── Sobol-Shoemake (from s011) ────────────────────────────────────────────────

def shoemake_to_quat(u: np.ndarray) -> np.ndarray:
    """Shoemake's uniform-on-S^3 mapping. Returns (N, 4) wxyz."""
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    a2 = 2.0 * np.pi * u2
    a3 = 2.0 * np.pi * u3
    x = s1 * np.sin(a2)
    y = s1 * np.cos(a2)
    z = s2 * np.sin(a3)
    w = s2 * np.cos(a3)
    return np.column_stack([w, x, y, z])


def build_sobol_q0(n: int, sobol_seed: int) -> np.ndarray:
    """N uniform Sobol-Shoemake quaternions, (N, 4) wxyz."""
    sobol = qmc.Sobol(d=3, scramble=True, seed=sobol_seed)
    u = sobol.random(n)
    return shoemake_to_quat(u)


# ─── Helpers ────────────────────────────────────────────────────────────────────

def angular_dist_deg(q1, q2):
    d = float(abs(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def evaluate_candidates(q0_batch, omega, truth, inertia, surrogate):
    """Evaluate surrogate-MSE for a batch of q0 ICs at fixed omega.

    Args:
        q0_batch: (N, 4) wxyz quaternions
        omega: (3,) angular velocity in rad/s
        truth: dict from load_truth()
        inertia: (3, 3) inertia tensor
        surrogate: surrogate model

    Returns:
        mse_arr: (N,) surrogate MSE per candidate
        rho_arr: (N,) predicted ρ per candidate
    """
    times = truth["observation_times"]
    sun_pos = truth["sun_pos"]
    obs_pos = truth["obs_pos"]
    sat_pos = truth["sat_pos"]
    obs_dist = truth["obs_dist"]
    mag_truth = truth["mag_hifi"]
    valid = np.isfinite(mag_truth)

    n_cands = q0_batch.shape[0]
    mse_arr = np.full(n_cands, np.inf)

    for i in range(n_cands):
        q0 = q0_batch[i]
        try:
            k1, k2, _ = propagate_to_body_frame(
                q0, omega, times, sun_pos, obs_pos, sat_pos, inertia,
            )
            mag_pred = surrogate.predict_magnitude(
                k1, k2, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist,
            )
            residual = mag_pred[valid] - mag_truth[valid]
            mse_arr[i] = float(np.mean(residual**2))
        except Exception:
            pass

    rho_arr = np.sqrt(mse_arr) / 0.05
    return mse_arr, rho_arr


def get_bracket_cells(seed: int) -> np.ndarray:
    """Load bracket cells for this seed from the s036/s032 density pass."""
    # Try s036 first, fall back to s032
    for root in [RESULTS_DIR.parent / "s036_multi_seed_pilot",
                 RESULTS_DIR.parent / "s032_cohort_fast"]:
        bracket_path = root / f"seed{seed:03d}" / "bracket.npz"
        if bracket_path.exists():
            b = np.load(bracket_path)
            if "bracket_cells" in b:
                return b["bracket_cells"]
            if "selected_cells_rad_s" in b:
                return np.array(b["selected_cells_rad_s"])
    # Fallback: generate bracket from trajectory LS peaks
    raise FileNotFoundError(f"No bracket data found for seed {seed}")


# ─── LM polish (from s034/s036) ────────────────────────────────────────────────

def lm_polish_one(q0_init, omega_init, truth, inertia, surrogate):
    """Run LM polish on one candidate. Returns (q0_final, omega_final, mse_final, n_iter)."""
    times = truth["observation_times"]
    sun_pos = truth["sun_pos"]
    obs_pos = truth["obs_pos"]
    sat_pos = truth["sat_pos"]
    obs_dist = truth["obs_dist"]
    mag_truth = truth["mag_hifi"]
    valid = np.isfinite(mag_truth)
    mag_truth_valid = mag_truth[valid]

    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos
    sun_unit = (sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True))
    obs_unit = (obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True))

    def residual(x):
        rotvec = x[:3]
        omega_delta = x[3:]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
        q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0_new = quat_mul(q_pert_wxyz, q0_init)
        omega_new = omega_init + omega_delta

        q_traj, _ = propagate_attitude(
            q0_new, omega_new, times,
            mode="tumbling", inertia_tensor=inertia,
        )
        qxyzw = q_traj[:, [1, 2, 3, 0]]
        R_full = Rotation.from_quat(qxyzw).as_matrix()
        k1_body = np.einsum('eij,ej->ei', R_full, sun_unit)
        k2_body = np.einsum('eij,ej->ei', R_full, obs_unit)

        mag_pred = surrogate.predict_magnitude(
            k1_body, k2_body, SP_ANGLE_DEG, AD_ANGLE_DEG, obs_dist,
        )
        return (mag_pred[valid] - mag_truth_valid).astype(np.float64)

    x0 = np.zeros(6)
    try:
        result = least_squares(
            residual, x0, method='lm',
            max_nfev=LM_MAX_NFEV, xtol=1e-8, ftol=1e-8,
        )
        mse_final = float(np.mean(result.fun ** 2))
        rotvec = result.x[:3]
        omega_delta = result.x[3:]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()
        q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0_final = quat_mul(q_pert_wxyz, q0_init)
        omega_final = omega_init + omega_delta
        return q0_final, omega_final, mse_final, result.nfev
    except Exception:
        return q0_init, omega_init, np.inf, -1


# ─── Hi-fi Pool ────────────────────────────────────────────────────────────────

_WORKER_CTX = None

def _init_hifi_worker(ctx):
    global _WORKER_CTX
    _WORKER_CTX = ctx

def _render_one(args):
    idx, q0_wxyz, omega_rad = args
    ctx = _WORKER_CTX
    mag_pred = render_hifi(np.array(q0_wxyz), np.array(omega_rad), ctx)
    rho = rho_from_hifi(mag_pred, ctx["mag_hifi_truth"])
    return idx, rho


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total_start = time.time()

    # Pre-load shared resources
    geo = load_static_geometry()
    inertia = geo["inertia_tensor"].astype(np.float64)
    surrogate = get_model()

    # Generate Sobol q0 ICs (shared across all seeds/cells)
    q0_sobol = build_sobol_q0(N_SOBOL, SOBOL_SEED)
    print(f"s037: Sobol-Shoemake IC pilot on seeds {PILOT_SEEDS}")
    print(f"      N_sobol={N_SOBOL}, N_bracket_cells={N_BRACKET_CELLS}")
    print(f"      {N_SOBOL} q0 ICs generated (sobol_seed={SOBOL_SEED})")

    all_results = {}

    for seed in PILOT_SEEDS:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")
        t_seed = time.time()

        truth = load_truth(seed)
        q0_truth = truth["q0_wxyz"]
        omega_truth = truth["omega0_rad"]
        omega_truth_mag = np.linalg.norm(omega_truth)
        omega_truth_dir = omega_truth / omega_truth_mag

        # ─── L0: Sobol at exact truth-ω ───────────────────────────────────────
        print(f"\n  [L0] Sobol N={N_SOBOL} at truth-ω ({np.degrees(omega_truth_mag):.3f} dps)")
        t0 = time.time()
        mse_l0, rho_l0 = evaluate_candidates(q0_sobol, omega_truth, truth, inertia, surrogate)
        wall_l0 = time.time() - t0
        best_idx_l0 = np.argmin(rho_l0)
        q0_best_err_l0 = angular_dist_deg(q0_sobol[best_idx_l0], q0_truth)
        print(f"       wall={wall_l0:.1f}s | min ρ={rho_l0.min():.2f} | "
              f"best q0→truth={q0_best_err_l0:.2f}° | "
              f"n(ρ<2)={np.sum(rho_l0<2)} n(ρ<4)={np.sum(rho_l0<4)} n(ρ<15)={np.sum(rho_l0<15)}", flush=True)

        # ─── L1: Sobol at nearest bracket cell ────────────────────────────────
        try:
            bracket_cells = get_bracket_cells(seed)
        except FileNotFoundError as e:
            print(f"  [L1/L2] SKIPPED — {e}")
            all_results[seed] = {"L0": {"min_rho": float(rho_l0.min()), "wall_s": wall_l0}}
            continue

        nearest_idx = np.argmin(np.abs(bracket_cells - omega_truth_mag))
        nearest_cell = bracket_cells[nearest_idx]
        nearest_pct = abs(nearest_cell - omega_truth_mag) / omega_truth_mag * 100
        omega_l1 = omega_truth_dir * nearest_cell  # same direction, bracket magnitude

        print(f"\n  [L1] Sobol N={N_SOBOL} at nearest bracket cell "
              f"({np.degrees(nearest_cell):.4f} dps, {nearest_pct:.2f}% from truth)")
        t0 = time.time()
        mse_l1, rho_l1 = evaluate_candidates(q0_sobol, omega_l1, truth, inertia, surrogate)
        wall_l1 = time.time() - t0
        best_idx_l1 = np.argmin(rho_l1)
        q0_best_err_l1 = angular_dist_deg(q0_sobol[best_idx_l1], q0_truth)
        print(f"       wall={wall_l1:.1f}s | min ρ={rho_l1.min():.2f} | "
              f"best q0→truth={q0_best_err_l1:.2f}° | "
              f"n(ρ<2)={np.sum(rho_l1<2)} n(ρ<4)={np.sum(rho_l1<4)} n(ρ<15)={np.sum(rho_l1<15)}", flush=True)

        # ─── L2: Sobol at ALL bracket cells ───────────────────────────────────
        print(f"\n  [L2] Sobol N={N_SOBOL} at all {len(bracket_cells)} bracket cells")
        t0 = time.time()
        best_rho_per_cell = np.full(len(bracket_cells), np.inf)
        best_idx_per_cell = np.zeros(len(bracket_cells), dtype=int)
        all_mse_l2 = []
        all_omega_l2 = []

        for ci, cell_mag in enumerate(bracket_cells):
            omega_cell = omega_truth_dir * cell_mag
            mse_cell, rho_cell = evaluate_candidates(q0_sobol, omega_cell, truth, inertia, surrogate)
            best_rho_per_cell[ci] = rho_cell.min()
            best_idx_per_cell[ci] = np.argmin(rho_cell)
            all_mse_l2.append(mse_cell)
            all_omega_l2.append(omega_cell)

        wall_l2 = time.time() - t0
        all_mse_l2 = np.array(all_mse_l2)  # (N_cells, N_sobol)
        all_rho_l2 = np.sqrt(all_mse_l2) / 0.05

        n_cells_sub15 = np.sum(best_rho_per_cell < 15)
        n_cells_sub4 = np.sum(best_rho_per_cell < 4)
        print(f"       wall={wall_l2:.1f}s | cells with best ρ<15: {n_cells_sub15}/{len(bracket_cells)} | "
              f"ρ<4: {n_cells_sub4}/{len(bracket_cells)}", flush=True)
        print(f"       global min ρ={best_rho_per_cell.min():.2f} at cell "
              f"{np.argmin(best_rho_per_cell)} ({np.degrees(bracket_cells[np.argmin(best_rho_per_cell)]):.4f} dps, "
              f"{abs(bracket_cells[np.argmin(best_rho_per_cell)] - omega_truth_mag)/omega_truth_mag*100:.2f}% from truth)")

        # ─── Summary for this seed ────────────────────────────────────────────
        seed_result = {
            "L0": {
                "min_rho": float(rho_l0.min()),
                "n_sub2": int(np.sum(rho_l0 < 2)),
                "n_sub4": int(np.sum(rho_l0 < 4)),
                "n_sub15": int(np.sum(rho_l0 < 15)),
                "best_q0_err_deg": float(q0_best_err_l0),
                "wall_s": wall_l0,
            },
            "L1": {
                "nearest_cell_pct": float(nearest_pct),
                "min_rho": float(rho_l1.min()),
                "n_sub2": int(np.sum(rho_l1 < 2)),
                "n_sub4": int(np.sum(rho_l1 < 4)),
                "n_sub15": int(np.sum(rho_l1 < 15)),
                "best_q0_err_deg": float(q0_best_err_l1),
                "wall_s": wall_l1,
            },
            "L2": {
                "n_cells": len(bracket_cells),
                "n_cells_sub15": int(n_cells_sub15),
                "n_cells_sub4": int(n_cells_sub4),
                "global_min_rho": float(best_rho_per_cell.min()),
                "wall_s": wall_l2,
            },
        }

        # ─── LM polish if any level delivers ρ < 15 ──────────────────────────
        # Collect top-K candidates across L0 + L1 + L2
        candidates_for_polish = []

        # From L2 (all cells): flatten and pick top-K by surrogate-MSE
        flat_mse = all_mse_l2.ravel()  # (N_cells * N_sobol,)
        flat_order = np.argsort(flat_mse)[:LM_TOP_K]
        for flat_idx in flat_order:
            ci = flat_idx // N_SOBOL
            qi = flat_idx % N_SOBOL
            if flat_mse[flat_idx] < np.inf:
                candidates_for_polish.append({
                    "q0": q0_sobol[qi].copy(),
                    "omega": all_omega_l2[ci].copy(),
                    "mse_init": float(flat_mse[flat_idx]),
                    "source": f"L2_cell{ci}",
                })

        # Also include L0 best if not already covered
        if rho_l0.min() < 15:
            candidates_for_polish.insert(0, {
                "q0": q0_sobol[best_idx_l0].copy(),
                "omega": omega_truth.copy(),
                "mse_init": float(mse_l0[best_idx_l0]),
                "source": "L0_truth_omega",
            })

        min_rho_any = min(rho_l0.min(), rho_l1.min(), best_rho_per_cell.min())
        if min_rho_any >= 15:
            print(f"\n  [polish] SKIPPED — no candidate below ρ=15 (min={min_rho_any:.2f})")
            seed_result["polish"] = {"status": "skipped", "min_rho_all": float(min_rho_any)}
            all_results[seed] = seed_result
            continue

        # Deduplicate and limit
        candidates_for_polish = candidates_for_polish[:LM_TOP_K]
        print(f"\n  [polish] Running LM on top-{len(candidates_for_polish)} candidates "
              f"(min init ρ={np.sqrt(candidates_for_polish[0]['mse_init'])/0.05:.2f})", flush=True)

        polished = []
        t0 = time.time()
        for i, cand in enumerate(candidates_for_polish):
            q0_init = cand["q0"] / np.linalg.norm(cand["q0"])
            omega_init = cand["omega"]
            q0_f, omega_f, mse_f, nfev = lm_polish_one(
                q0_init, omega_init, truth, inertia, surrogate)
            rho_f = float(np.sqrt(mse_f) / 0.05)
            q0_err = angular_dist_deg(q0_f, q0_truth)
            omega_dir_err = np.degrees(np.arccos(np.clip(
                np.dot(omega_f, omega_truth) /
                (np.linalg.norm(omega_f) * np.linalg.norm(omega_truth) + 1e-30),
                -1, 1)))
            omega_mag_pct = (np.linalg.norm(omega_f) - omega_truth_mag) / omega_truth_mag * 100

            polished.append({
                "rank": i,
                "source": cand["source"],
                "rho_init": float(np.sqrt(cand["mse_init"]) / 0.05),
                "rho_final": rho_f,
                "q0_to_truth_deg": q0_err,
                "omega_dir_to_truth_deg": float(omega_dir_err),
                "omega_mag_pct": float(omega_mag_pct),
                "nfev": nfev,
                "q0_final": q0_f.tolist(),
                "omega_final": omega_f.tolist(),
            })

            if i < 5 or rho_f < 2:
                print(f"    #{i}: ρ {np.sqrt(cand['mse_init'])/0.05:.1f}→{rho_f:.3f} "
                      f"q0→truth={q0_err:.2f}° ω-dir={omega_dir_err:.2f}° "
                      f"ω-mag={omega_mag_pct:+.2f}% nfev={nfev} [{cand['source']}]", flush=True)

        wall_polish = time.time() - t0
        n_band_a_surr = sum(1 for p in polished if p["rho_final"] < 2.0)
        print(f"  [polish] done in {wall_polish:.0f}s. Surrogate Band A: {n_band_a_surr}/{len(polished)}")

        seed_result["polish"] = {
            "status": "complete",
            "n_polished": len(polished),
            "n_surr_band_a": n_band_a_surr,
            "wall_s": wall_polish,
            "best_rho": min(p["rho_final"] for p in polished),
            "best_q0_truth": min(p["q0_to_truth_deg"] for p in polished),
            "candidates": polished,
        }

        # ─── Hi-fi confirm if any surrogate Band A ───────────────────────────
        band_a_cands = [p for p in polished if p["rho_final"] < 2.0]
        if not band_a_cands:
            print(f"  [hifi] SKIPPED — 0 surrogate Band A")
            seed_result["hifi"] = {"status": "skipped"}
        else:
            print(f"\n  [hifi] Rendering {len(band_a_cands)} surrogate-Band-A candidates", flush=True)
            ctx = build_context(seed)
            work = [(i, c["q0_final"], c["omega_final"]) for i, c in enumerate(band_a_cands)]

            t0 = time.time()
            fork_ctx = mp.get_context("fork")
            with fork_ctx.Pool(N_HIFI_WORKERS, initializer=_init_hifi_worker, initargs=(ctx,)) as pool:
                hifi_raw = pool.map(_render_one, work)
            wall_hifi = time.time() - t0

            hifi_raw.sort(key=lambda x: x[0])
            hifi_rhos = [r[1] for r in hifi_raw]
            for i, c in enumerate(band_a_cands):
                c["rho_hifi"] = hifi_rhos[i]
                c["band_hifi"] = rho_band(hifi_rhos[i])

            n_hifi_a = sum(1 for r in hifi_rhos if r < 2.0)
            n_hifi_b = sum(1 for r in hifi_rhos if 2.0 <= r < 4.0)
            print(f"  [hifi] done in {wall_hifi:.1f}s. "
                  f"Band A={n_hifi_a}, B={n_hifi_b}, "
                  f"ρ min={min(hifi_rhos):.3f}", flush=True)

            seed_result["hifi"] = {
                "status": "complete",
                "n_band_a": n_hifi_a,
                "n_band_b": n_hifi_b,
                "rho_min": float(min(hifi_rhos)),
                "wall_s": wall_hifi,
            }

        all_results[seed] = seed_result
        wall_seed = time.time() - t_seed
        print(f"\n  Seed {seed} total wall: {wall_seed:.0f}s")

        # Save per-seed
        seed_path = RESULTS_DIR / f"seed{seed:03d}_result.json"
        with open(seed_path, "w") as f:
            json.dump({"seed": seed, **seed_result}, f, indent=2, default=float)
        print(f"  Saved: {seed_path}")

    # ─── Final summary ────────────────────────────────────────────────────────
    wall_total = time.time() - t_total_start
    print(f"\n\n{'='*70}")
    print(f"  s037 SUMMARY ({wall_total:.0f}s total)")
    print(f"{'='*70}")
    print(f"  {'Seed':>4} {'L0 min ρ':>9} {'L1 min ρ':>9} {'L2 min ρ':>9} "
          f"{'Polish':>8} {'Hifi A':>6}")
    print(f"  {'-'*4} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*6}")
    for seed in PILOT_SEEDS:
        r = all_results[seed]
        l0 = r.get("L0", {}).get("min_rho", "—")
        l1 = r.get("L1", {}).get("min_rho", "—")
        l2 = r.get("L2", {}).get("global_min_rho", "—")
        pol = r.get("polish", {}).get("n_surr_band_a", "—")
        hifi = r.get("hifi", {}).get("n_band_a", "—")
        l0_s = f"{l0:.2f}" if isinstance(l0, float) else l0
        l1_s = f"{l1:.2f}" if isinstance(l1, float) else l1
        l2_s = f"{l2:.2f}" if isinstance(l2, float) else l2
        print(f"  {seed:>4} {l0_s:>9} {l1_s:>9} {l2_s:>9} {str(pol):>8} {str(hifi):>6}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"seeds": PILOT_SEEDS, "wall_total_s": wall_total,
                   "per_seed": {str(s): all_results[s] for s in PILOT_SEEDS}},
                  f, indent=2, default=float)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
