#!/usr/bin/env python3
"""
retry_geo_serial.py — one-off rescue for m103 geo hang (Phase 2 pilot, 2026-04-17).

CONTEXT
=======
The Phase 2 pilot of the m048 migration ran `invert.py --seed N --traj-source m048`
on 3 seeds (24 low-phase, 91 mid, 28 high). Seeds 24 and 91 completed cleanly:

    seed=024 source=m048: q0=5.03° w_dir=0.99° w_mag=-0.16% hifi=0.03375 [PARTIAL]
    seed=091 source=m048: q0=0.05° w_dir=0.01° w_mag=-0.00% hifi=0.00247 [OK]

Seed 28 (high phase, pa_med=87.3°) HUNG twice in `m103_hybrid.py` Step 4 (Geo
refinement). After ~5 hours: all 24 Pool workers in `futex_do_wait` with 0% CPU,
parent stuck in pool.map(). Progression observed by user:

    T=0-5 min:   80% CPU across all 24 workers (compute-bound in L-BFGS-B)
    T=5-15 min:  fast-converging candidates finish, CPU drops
    T=15+ min:   1-3 straggler candidates still iterating
    T=hours:     all workers idle, parent never returns from pool.map()

Likely mechanism: L-BFGS-B's Fortran backend on the flat/discontinuous high-phase
surrogate cost drives a worker to a pathological state (NaN propagation, failed
pickle of result, or silent segfault). Pool.map waits forever for the missing
result. maxfun=1500 alone (already applied in m103) caps compute per call but
doesn't protect against worker death / lost result.

RESCUE PLAN
===========
`multi_phi_ckpt.npz` for seed 28 was saved at 04:05 BEFORE Step 4 started — intact
on disk. It contains q0s, w0s, omega_ranks, phi_ranks — the inputs Step 4 needs.
This script runs Step 4 *serially* (no Pool) with try/except per candidate, so a
bad candidate raises an exception we can catch instead of silently hanging the
harvest. Falls back to the pre-geo (q0, w0) with geo_cost=inf for any candidate
whose L-BFGS-B blows up.

RUNNING THE SCRIPT
==================
    python3 notebooks/inversion/11_casadi_formulation/retry_geo_serial.py \\
        --seed 28 --traj-source m048

Writes (matching the schemas `m115.load_omega_candidates` and `invert.py` expect):
    data/results/inversion_diagnostics/m103_hybrid_m048/seed_028/geo_ckpt.npz
    data/results/inversion_diagnostics/m103_hybrid_m048/seed_028/result.json

After this lands (~5 min), relaunch the normal pilot path:
    python3 notebooks/inversion/invert.py --seed 28 --traj-source m048

`invert.py` will detect geo_ckpt.npz, skip m103, and proceed through m115 → m126
→ lc_compare. Expected wall ~11 min.

WHY NOT JUST EDIT M103?
=======================
Intentional. m103 runs fine for seeds 24 and 91 on m048 and for all m046 seeds.
This rescue is scoped to the one known failure. If we see the same hang on more
m048 seeds during Phase 3, properly retrofit SERIAL_GEO + RESUME_FROM_MULTI_PHI
env flags into m103. For the pilot: isolated, reviewable, low-risk rescue.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import attitude_error_deg, save_results
from lib.traj_source import VALID_SOURCES
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.quaternion_utils import (
    axis_angle_to_quaternion, quaternion_to_axis_angle)

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# Constants vendored from m103_hybrid.py (must match exactly so retry geometry
# matches what Step 4 would have seen)
CONSTRAINT_WEIGHT = 10.0
NOISE_SEED = 42
NOISE_SIGMA = 0.05


def get_allowed_normals(mag):
    """Magnitude-to-allowed-normals mapping — vendored from m103_hybrid.py."""
    if mag < 5.9: return [0, 1]
    elif mag < 6.3: return [0, 1, 4, 5]
    elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
    else: return list(range(10))


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True,
                    help="Trajectory seed (0..99)")
    ap.add_argument("--traj-source", default="m048", choices=VALID_SOURCES,
                    help="'m046' or 'm048'")
    ap.add_argument("--maxfun", type=int, default=500,
                    help="L-BFGS-B maxfun cap per candidate (default 500; was "
                         "1500 in m103 post-fix, lowered here for safety)")
    ap.add_argument("--maxiter", type=int, default=50,
                    help="L-BFGS-B maxiter cap per candidate (default 50)")
    args = ap.parse_args()

    TRAJ_SEED = args.seed
    TRAJ_SOURCE = args.traj_source

    # Output dir matches m103's source-tagged convention
    if TRAJ_SOURCE == 'm046':
        out_base = RESULTS_DIR / "m103_hybrid"
    else:
        out_base = RESULTS_DIR / f"m103_hybrid_{TRAJ_SOURCE}"
    ckpt_dir = out_base / f"seed_{TRAJ_SEED:03d}"

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Seed dir not found: {ckpt_dir}. Run m103_hybrid.py first to "
            f"generate multi_phi_ckpt.npz.")

    multi_phi_path = ckpt_dir / "multi_phi_ckpt.npz"
    if not multi_phi_path.exists():
        raise FileNotFoundError(
            f"multi_phi_ckpt.npz missing: {multi_phi_path}. Cannot resume geo "
            f"without Step 3.5 output.")

    print(f"{'=' * 60}")
    print(f"retry_geo_serial (seed {TRAJ_SEED}, source {TRAJ_SOURCE})")
    print(f"  Resuming from: {multi_phi_path}")
    print(f"  L-BFGS-B caps: maxfun={args.maxfun}, maxiter={args.maxiter}")
    print(f"{'=' * 60}")
    t_global = time.time()

    # ── Load master + reconstruct cost environment ──────────────────────
    # Must match m103's per-seed setup EXACTLY so the cost surface is
    # identical to what Step 4 would have seen.
    if TRAJ_SOURCE == 'm046':
        master = np.load(
            str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
            allow_pickle=True)
        obs_times = master['observation_times']
        pab_j2000 = master['pab_j2000']
    else:
        master = np.load(
            str(RESULTS_DIR / "m048_trajectories" / "m048_trajectories.npz"),
            allow_pickle=True)
        obs_times = master['observation_times'][TRAJ_SEED]
        pab_j2000 = master['pab_j2000'][TRAJ_SEED]

    unique_normals = master['unique_normals']
    I_tensor = master['inertia_tensor']
    true_q0 = master['q0s'][TRAJ_SEED]
    true_omega0 = master['omega0s'][TRAJ_SEED]
    true_omega_mag_dps = float(master['omega_mags'][TRAJ_SEED])
    true_lc = master['mag_hifi'][TRAJ_SEED]

    # Reproduce m103's observed_lc (same noise seed) and peak selection
    rng = np.random.default_rng(NOISE_SEED)
    observed_lc = true_lc + rng.normal(0, NOISE_SIGMA, len(true_lc))
    peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
    spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]
    all_spec_epochs = spec_peaks
    all_spec_mags = observed_lc[all_spec_epochs]
    all_spec_allowed = [get_allowed_normals(m) for m in all_spec_mags]

    print(f"  Spec peaks: {len(spec_peaks)} (constraint epochs for geo)")

    # ── Load the 26 candidates from multi_phi_ckpt ──────────────────────
    mp = np.load(str(multi_phi_path), allow_pickle=True)
    n_cand = int(mp['n_candidates'])
    q0s_in = mp['q0s']
    w0s_in = mp['w0s']
    omega_ranks = mp['omega_ranks']
    phi_ranks = mp['phi_ranks']
    print(f"  Loaded {n_cand} candidates from multi_phi_ckpt.npz\n")

    # ── Geometric cost (vendored from m103) ─────────────────────────────
    def geometric_cost(params):
        q0 = axis_angle_to_quaternion(params[:3])
        quats, _ = propagate_attitude(
            q0, params[3:6], obs_times, "tumbling", I_tensor)
        cost = 0.0
        for i, ep in enumerate(all_spec_epochs):
            R = Rotation.from_quat(
                [quats[ep][1], quats[ep][2], quats[ep][3], quats[ep][0]]
            ).as_matrix()
            pb = R @ pab_j2000[ep]
            cost += CONSTRAINT_WEIGHT * (1.0 - max(
                np.dot(unique_normals[ni], pb)
                for ni in all_spec_allowed[i])) ** 2
        return cost

    # ── Serial geo refinement with try/except per candidate ─────────────
    print(f"--- Step 4 (SERIAL): Geo refinement ({n_cand} candidates) ---")
    t_step4 = time.time()

    q0_refs = np.zeros((n_cand, 4))
    w0_refs = np.zeros((n_cand, 3))
    geo_costs = np.zeros(n_cand)
    q0_ref_errs = np.zeros(n_cand)
    w0_ref_errs = np.zeros(n_cand)
    per_cand_times = np.zeros(n_cand)
    per_cand_status = [''] * n_cand

    for i in range(n_cand):
        t_c = time.time()
        q0_in = q0s_in[i]
        w0_in = w0s_in[i]
        x0 = np.concatenate([quaternion_to_axis_angle(q0_in), w0_in])
        try:
            res = minimize(
                geometric_cost, x0, method='L-BFGS-B',
                options={'maxiter': args.maxiter, 'maxfun': args.maxfun,
                         'ftol': 1e-8, 'gtol': 1e-6})
            q0_refs[i] = axis_angle_to_quaternion(res.x[:3])
            w0_refs[i] = res.x[3:6]
            geo_costs[i] = float(res.fun)
            per_cand_status[i] = 'ok'
        except Exception as e:
            # Fall back to unrefined state so downstream m115 still sees this
            # candidate. geo_cost=inf marks it as un-refined in any ranking.
            q0_refs[i] = q0_in.copy()
            w0_refs[i] = w0_in.copy()
            geo_costs[i] = float('inf')
            per_cand_status[i] = f'FAILED: {type(e).__name__}: {e}'
            print(f"  cand {i:3d}: L-BFGS-B raised {type(e).__name__}; "
                  f"falling back to unrefined (q0, w0)")
        q0_ref_errs[i] = attitude_error_deg(q0_refs[i], true_q0)
        w0_ref_errs[i] = omega_dir_err(w0_refs[i], true_omega0)
        per_cand_times[i] = time.time() - t_c
        tag = " <--" if w0_ref_errs[i] < 10 else ""
        print(f"  cand {i:3d}: {per_cand_times[i]:5.1f}s  "
              f"geo={geo_costs[i]:.6f}  q0={q0_ref_errs[i]:6.2f}  "
              f"w={w0_ref_errs[i]:6.2f}{tag}  [{per_cand_status[i]}]")

    step4_time = time.time() - t_step4
    n_failed = sum(1 for s in per_cand_status if s.startswith('FAILED'))
    print(f"\nSerial geo done in {step4_time:.1f}s "
          f"({n_failed} candidates failed, fell back to unrefined)")

    # ── Save geo_ckpt.npz matching m103's schema exactly ────────────────
    # (m115.load_omega_candidates reads: w0_refs, w0_ref_errs, q0_refs)
    geo_ckpt_path = ckpt_dir / "geo_ckpt.npz"
    np.savez(str(geo_ckpt_path),
             n_candidates=n_cand,
             omega_ranks=omega_ranks,
             phi_ranks=phi_ranks,
             q0_refs=q0_refs,
             w0_refs=w0_refs,
             geo_costs=geo_costs,
             q0_ref_errs=q0_ref_errs,
             w0_ref_errs=w0_ref_errs)
    print(f"Saved: {geo_ckpt_path}")

    # ── Pick geo winner, write minimal result.json + result.npz ─────────
    # Matches m103's SKIP_HIFI short-circuit format so invert.py reads it fine.
    winner_idx = int(np.argmin(geo_costs))
    q0_err = float(q0_ref_errs[winner_idx])
    w0_err = float(w0_ref_errs[winner_idx])
    w_mag = np.rad2deg(np.linalg.norm(w0_refs[winner_idx]))
    w_mag_err = float((w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100)
    total_time = time.time() - t_global

    print(f"\n{'=' * 60}")
    print(f"RESULT (seed {TRAJ_SEED}) — SKIP_HIFI + SERIAL-GEO rescue")
    print(f"{'=' * 60}")
    print(f"  q0 error:     {q0_err:.2f} deg")
    print(f"  w dir error:  {w0_err:.2f} deg")
    print(f"  w mag error:  {w_mag_err:+.2f}%")
    print(f"  w estimated:  {np.rad2deg(w0_refs[winner_idx])} deg/s")
    print(f"  w true:       {np.rad2deg(true_omega0)} deg/s")
    print(f"  Serial geo wall: {step4_time:.1f}s  (vs m103 Pool(24) which hung)")
    print(f"  Total wall:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Failed candidates: {n_failed} / {n_cand}")

    np.savez(str(ckpt_dir / "result.npz"),
             q0_refined=q0_refs[winner_idx],
             w0_refined=w0_refs[winner_idx],
             true_q0=true_q0, true_omega0=true_omega0)

    result_json = {
        'traj_seed': TRAJ_SEED,
        'traj_source': TRAJ_SOURCE,
        'experiment': 'm103_hybrid (SKIP_HIFI, serial-geo rescue)',
        'skip_hifi': True,
        'serial_geo_rescue': True,
        'params': {
            'maxfun': args.maxfun, 'maxiter': args.maxiter,
            'n_candidates': n_cand,
            'n_failed_candidates': n_failed,
        },
        'winner': {
            'q0_err': q0_err,
            'w0_err': w0_err,
            'w_mag_err_pct': w_mag_err,
            'omega_rank': int(omega_ranks[winner_idx]),
            'phi_rank': int(phi_ranks[winner_idx]),
            'selection_method': 'geo_winner_only',
            'q0_wxyz': q0_refs[winner_idx].tolist(),
            'w0_rad': w0_refs[winner_idx].tolist(),
        },
        'timing': {
            'step4_s': float(step4_time),
            'total_s': float(total_time),
        },
        'per_candidate_status': per_cand_status,
    }
    save_results(str(ckpt_dir / "result.json"), result_json)
    print(f"\nSaved: {ckpt_dir}/result.json")
    print(f"Saved: {ckpt_dir}/result.npz")
    print(f"\nNext step:  python3 notebooks/inversion/invert.py "
          f"--seed {TRAJ_SEED} --traj-source {TRAJ_SOURCE}")


if __name__ == '__main__':
    main()
