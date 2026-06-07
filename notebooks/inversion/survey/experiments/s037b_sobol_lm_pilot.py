"""s037b: Sobol q0 + LM polish at bracket ω-cells — correct s011 architecture.

s037 evaluated raw Sobol ICs without polish (wrong — N=64 ICs are ~90° apart,
raw evaluation always gives ρ>>15). The s011 success was Sobol ICs + LM polish.
This script replicates the correct architecture:

  For each (Sobol q0, ω-cell): run joint LM with x0=[zeros(3), omega_cell]
  → the LM polishes q0 from the Sobol IC while starting omega at the cell value.

Two levels:
  L0: Sobol N=64 + LM, omega starts at truth-ω (replicates s011 on these seeds)
  L1: Sobol N=64 + LM, omega starts at (truth-dir × nearest-bracket-mag)

If L0 works on seeds 23/28 (as s011 showed for 9/10): validates that these
seeds are Sobol-recoverable at truth-ω. If L1 also works: validates that the
bracket offset (5-8%) doesn't break LM convergence.

Usage:
    cd notebooks/inversion/survey
    python experiments/s037b_sobol_lm_pilot.py
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
from lib.traj_load import load_truth
from lib.filter_costs import load_static_geometry
from lib.surrogate_eval import get_model
from lib.hifi_render import build_context, render_hifi, rho_from_hifi, rho_band

PILOT_SEEDS = [23, 28, 89]
N_SOBOL = 64
SOBOL_SEED = 42
LM_MAX_NFEV = 200
SP_ANGLE_DEG = 0.0
AD_ANGLE_DEG = 15.0
N_HIFI_WORKERS = 8

RESULTS_DIR = SURVEY_DIR / "results" / "s037b_sobol_lm_pilot"


# ─── Sobol-Shoemake ────────────────────────────────────────────────────────────

def shoemake_to_quat(u: np.ndarray) -> np.ndarray:
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


def get_bracket_cells(seed: int) -> np.ndarray:
    for root in [RESULTS_DIR.parent / "s036_multi_seed_pilot",
                 RESULTS_DIR.parent / "s032_cohort_fast"]:
        bracket_path = root / f"seed{seed:03d}" / "bracket.npz"
        if bracket_path.exists():
            b = np.load(bracket_path)
            if "bracket_cells" in b:
                return b["bracket_cells"]
    raise FileNotFoundError(f"No bracket data for seed {seed}")


# ─── LM worker ─────────────────────────────────────────────────────────────────

def run_single_lm(q0_ic, omega_start, truth, inertia, surrogate):
    """Run joint (q0, ω) LM from a Sobol IC with omega initialized at omega_start.

    Parameterization: x = [rotvec(3), omega(3)]
    - q0 = Rot(rotvec) ⊗ q0_ic
    - omega = x[3:6]  (absolute, not delta — matches s011)
    """
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
        omega = x[3:6]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
        q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0 = quat_mul(q_pert_wxyz, q0_ic)

        try:
            q_traj, _ = propagate_attitude(
                q0, omega, times,
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
        except Exception:
            return np.full(valid.sum(), 1e3, dtype=np.float64)

    x0 = np.concatenate([np.zeros(3), omega_start])

    try:
        result = least_squares(
            residual, x0, method='lm',
            max_nfev=LM_MAX_NFEV, xtol=1e-8, ftol=1e-8,
        )
        mse_final = float(np.mean(result.fun ** 2))
        rotvec = result.x[:3]
        omega_final = result.x[3:6]
        q_pert = Rotation.from_rotvec(rotvec).as_quat()
        q_pert_wxyz = np.array([q_pert[3], q_pert[0], q_pert[1], q_pert[2]])
        q0_final = quat_mul(q_pert_wxyz, q0_ic)
        return q0_final, omega_final, mse_final, int(result.nfev), True
    except Exception:
        return q0_ic, omega_start, np.inf, -1, False


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


# ─── Per-level run ──────────────────────────────────────────────────────────────

def run_level(level_name, q0_batch, omega_start, truth, inertia, surrogate, seed):
    """Run LM on all Sobol ICs at a single omega starting point."""
    q0_truth = truth["q0_wxyz"]
    omega_truth = truth["omega0_rad"]
    omega_truth_mag = np.linalg.norm(omega_truth)

    results = []
    t0 = time.time()
    for i in range(len(q0_batch)):
        q0_ic = q0_batch[i] / np.linalg.norm(q0_batch[i])
        q0_f, omega_f, mse_f, nfev, conv = run_single_lm(
            q0_ic, omega_start, truth, inertia, surrogate)

        rho_f = float(np.sqrt(mse_f) / 0.05) if mse_f < np.inf else np.inf
        q0_err = angular_dist_deg(q0_f, q0_truth)
        omega_dir_err = np.degrees(np.arccos(np.clip(
            np.dot(omega_f, omega_truth) /
            (np.linalg.norm(omega_f) * np.linalg.norm(omega_truth) + 1e-30),
            -1, 1)))
        omega_mag_pct = (np.linalg.norm(omega_f) - omega_truth_mag) / omega_truth_mag * 100

        results.append({
            "ic_idx": i,
            "rho_final": rho_f,
            "q0_to_truth_deg": q0_err,
            "omega_dir_to_truth_deg": float(omega_dir_err),
            "omega_mag_pct": float(omega_mag_pct),
            "nfev": nfev,
            "converged": conv,
            "q0_final": q0_f.tolist(),
            "omega_final": omega_f.tolist(),
        })

    wall = time.time() - t0
    rhos = np.array([r["rho_final"] for r in results])
    n_a = int(np.sum(rhos < 2))
    n_b = int(np.sum((rhos >= 2) & (rhos < 4)))
    n_sub15 = int(np.sum(rhos < 15))
    best = results[int(np.argmin(rhos))]

    print(f"  [{level_name}] wall={wall:.0f}s | "
          f"min ρ={rhos.min():.3f} | "
          f"n(ρ<2)={n_a} n(ρ<4)={n_a+n_b} n(ρ<15)={n_sub15} | "
          f"best: q0→truth={best['q0_to_truth_deg']:.2f}° "
          f"ω-dir={best['omega_dir_to_truth_deg']:.2f}° "
          f"ω-mag={best['omega_mag_pct']:+.2f}%", flush=True)

    return {
        "wall_s": wall,
        "min_rho": float(rhos.min()),
        "n_band_a": n_a,
        "n_band_ab": n_a + n_b,
        "n_sub15": n_sub15,
        "best_q0_err": best["q0_to_truth_deg"],
        "best_omega_dir_err": best["omega_dir_to_truth_deg"],
        "best_omega_mag_pct": best["omega_mag_pct"],
        "results": results,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    geo = load_static_geometry()
    inertia = geo["inertia_tensor"].astype(np.float64)
    surrogate = get_model()
    q0_sobol = build_sobol_q0(N_SOBOL, SOBOL_SEED)

    print(f"s037b: Sobol q0 + LM polish pilot on seeds {PILOT_SEEDS}")
    print(f"       N_sobol={N_SOBOL}, LM_max_nfev={LM_MAX_NFEV}")

    all_results = {}

    for seed in PILOT_SEEDS:
        print(f"\n{'='*70}")
        print(f"  SEED {seed}")
        print(f"{'='*70}")
        t_seed = time.time()

        truth = load_truth(seed)
        omega_truth = truth["omega0_rad"]
        omega_truth_mag = np.linalg.norm(omega_truth)
        omega_truth_dir = omega_truth / omega_truth_mag

        # ─── L0: Sobol + LM at truth-ω ───────────────────────────────────────
        print(f"\n  L0: Sobol N={N_SOBOL} + LM, omega starts at truth-ω "
              f"({np.degrees(omega_truth_mag):.3f} dps)", flush=True)
        l0 = run_level("L0", q0_sobol, omega_truth, truth, inertia, surrogate, seed)

        # ─── L1: Sobol + LM at nearest bracket cell ──────────────────────────
        try:
            bracket_cells = get_bracket_cells(seed)
            nearest_idx = np.argmin(np.abs(bracket_cells - omega_truth_mag))
            nearest_cell = bracket_cells[nearest_idx]
            nearest_pct = abs(nearest_cell - omega_truth_mag) / omega_truth_mag * 100
            omega_l1 = omega_truth_dir * nearest_cell

            print(f"\n  L1: Sobol N={N_SOBOL} + LM, omega starts at nearest bracket "
                  f"({np.degrees(nearest_cell):.4f} dps, {nearest_pct:.2f}% from truth)", flush=True)
            l1 = run_level("L1", q0_sobol, omega_l1, truth, inertia, surrogate, seed)
        except FileNotFoundError as e:
            print(f"\n  L1: SKIPPED — {e}")
            l1 = None

        # ─── Hi-fi confirm best candidates ────────────────────────────────────
        # Collect surrogate Band A candidates from both levels
        band_a_cands = []
        for lvl_name, lvl in [("L0", l0), ("L1", l1)]:
            if lvl is None:
                continue
            for r in lvl["results"]:
                if r["rho_final"] < 2.0:
                    band_a_cands.append({**r, "level": lvl_name})

        hifi_result = None
        if band_a_cands:
            # Deduplicate by q0 proximity (< 1° = same basin)
            unique = [band_a_cands[0]]
            for c in band_a_cands[1:]:
                is_dup = any(angular_dist_deg(
                    np.array(c["q0_final"]), np.array(u["q0_final"])) < 1.0
                    for u in unique)
                if not is_dup:
                    unique.append(c)

            print(f"\n  [hifi] {len(unique)} unique surrogate-Band-A candidates "
                  f"(from {len(band_a_cands)} raw)", flush=True)

            ctx = build_context(seed)
            work = [(i, c["q0_final"], c["omega_final"]) for i, c in enumerate(unique)]

            t0 = time.time()
            fork_ctx = mp.get_context("fork")
            with fork_ctx.Pool(N_HIFI_WORKERS, initializer=_init_hifi_worker, initargs=(ctx,)) as pool:
                hifi_raw = pool.map(_render_one, work)
            wall_hifi = time.time() - t0

            hifi_raw.sort(key=lambda x: x[0])
            for i, c in enumerate(unique):
                c["rho_hifi"] = hifi_raw[i][1]
                c["band_hifi"] = rho_band(hifi_raw[i][1])

            n_hifi_a = sum(1 for c in unique if c["rho_hifi"] < 2.0)
            n_hifi_b = sum(1 for c in unique if 2.0 <= c["rho_hifi"] < 4.0)
            print(f"  [hifi] done in {wall_hifi:.1f}s | "
                  f"Band A={n_hifi_a}, B={n_hifi_b} | "
                  f"ρ min={min(c['rho_hifi'] for c in unique):.3f}", flush=True)

            for c in unique:
                print(f"    [{c['level']}] q0→truth={c['q0_to_truth_deg']:.2f}° "
                      f"ω-dir={c['omega_dir_to_truth_deg']:.2f}° "
                      f"ω-mag={c['omega_mag_pct']:+.2f}% "
                      f"surr ρ={c['rho_final']:.3f} → hifi ρ={c['rho_hifi']:.3f} "
                      f"({c['band_hifi']})", flush=True)

            hifi_result = {
                "n_unique": len(unique),
                "n_hifi_a": n_hifi_a,
                "n_hifi_b": n_hifi_b,
                "rho_min": float(min(c["rho_hifi"] for c in unique)),
                "wall_s": wall_hifi,
                "candidates": unique,
            }

        wall_seed = time.time() - t_seed
        all_results[seed] = {
            "L0": {k: v for k, v in l0.items() if k != "results"},
            "L0_results": l0["results"],
            "L1": {k: v for k, v in l1.items() if k != "results"} if l1 else None,
            "L1_results": l1["results"] if l1 else None,
            "hifi": hifi_result,
            "wall_total_s": wall_seed,
        }

        # Save per-seed
        seed_path = RESULTS_DIR / f"seed{seed:03d}_result.json"
        with open(seed_path, "w") as f:
            json.dump({"seed": seed, **all_results[seed]}, f, indent=2, default=float)
        print(f"\n  Saved: {seed_path} (wall {wall_seed:.0f}s)")

    # ─── Summary ──────────────────────────────────────────────────────────────
    wall_total = time.time() - t_total
    print(f"\n\n{'='*70}")
    print(f"  s037b SUMMARY ({wall_total:.0f}s total)")
    print(f"{'='*70}")
    print(f"  {'Seed':>4} {'L0 ρ':>6} {'L0 A':>4} {'L1 ρ':>6} {'L1 A':>4} {'Hifi A':>6}")
    print(f"  {'-'*4} {'-'*6} {'-'*4} {'-'*6} {'-'*4} {'-'*6}")
    for seed in PILOT_SEEDS:
        r = all_results[seed]
        l0r = r["L0"]["min_rho"]
        l0a = r["L0"]["n_band_a"]
        l1r = r["L1"]["min_rho"] if r["L1"] else "—"
        l1a = r["L1"]["n_band_a"] if r["L1"] else "—"
        ha = r["hifi"]["n_hifi_a"] if r["hifi"] else 0
        l1_s = f"{l1r:.2f}" if isinstance(l1r, float) else l1r
        print(f"  {seed:>4} {l0r:>6.2f} {l0a:>4} {l1_s:>6} {str(l1a):>4} {ha:>6}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"seeds": PILOT_SEEDS, "wall_total_s": wall_total,
                   "per_seed": {str(s): {k: v for k, v in all_results[s].items()
                                         if k not in ("L0_results", "L1_results")}
                                for s in PILOT_SEEDS}},
                  f, indent=2, default=float)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
