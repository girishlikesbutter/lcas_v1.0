#!/usr/bin/env python3
"""
Two-Epoch Pair Matching Experiment.

Pipeline: iso-brightness candidates at two epochs → analytic omega derivation
for all pairs → progressive filtering → lo-fi lightcurve ranking → hi-fi
refinement.

Phases:
  0. Setup & benchmarks
  1. Iso-brightness candidates at epoch 0
  2. Iso-brightness candidates at epoch T
  3. Pair matching + omega derivation
  4. Validation epoch checks
  5. Lo-fi lightcurve ranking
  6. Hi-fi refinement
"""

import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import (
    ExperimentContext, setup_experiment, brightness_single_epoch,
    attitude_error_deg, save_results,
)
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

# ─── Logging ────────────────────────────────────────────────────────────────

RESULTS_DIR = PROJECT_ROOT / 'data' / 'results' / 'inversion_diagnostics'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / 'exp_two_epoch_matching.log'
JSON_PATH = RESULTS_DIR / 'exp_two_epoch_matching.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__).info

# ─── Configuration ──────────────────────────────────────────────────────────

N_OBSERVATIONS = 500
NOISE_SIGMA = 0.05
N_SEEDS = 10000
N_MAX_CANDIDATES = 2000  # cap per epoch (truth has low residual, so safe)
N_WORKERS = 8
TOP_K_REFINE = 5
RANDOM_SEED = 42
END_TIME_UTC = '2020-02-05T11:00:00'  # 1-hour window
EPOCH_T_OFFSET = 1  # 1 index apart (~7.2s gap for meaningful culling)

# ─── Module-level globals for multiprocessing workers ───────────────────────

_worker_ctx_data = None   # dict of arrays needed for brightness eval
_worker_target = None     # target lo-fi magnitude
_worker_epoch_idx = None  # epoch index
_worker_true_q = None     # true quaternion at epoch (for error calc)


def _init_worker(ctx_data, target, epoch_idx, true_q):
    global _worker_ctx_data, _worker_target, _worker_epoch_idx, _worker_true_q
    _worker_ctx_data = ctx_data
    _worker_target = target
    _worker_epoch_idx = epoch_idx
    _worker_true_q = true_q


def _brightness_lofi_worker(q_wxyz):
    """Evaluate lo-fi brightness using worker globals (avoids passing ctx)."""
    d = _worker_ctx_data
    idx = _worker_epoch_idx
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    s = R @ (d['sun_pos'][idx] - d['sat_pos'][idx])
    s = s / np.linalg.norm(s)
    o = R @ (d['obs_pos'][idx] - d['sat_pos'][idx])
    o = o / np.linalg.norm(o)

    art_slice = {c: m[idx:idx + 1] for c, m in d['art_matrices'].items()}
    lit = create_no_shadow_lit_status(d['satellite'], 1)
    mag, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit,
        k1_vectors_array=s.reshape(1, 3),
        k2_vectors_array=o.reshape(1, 3),
        observer_distances=np.array([d['obs_dist'][idx]]),
        satellite=d['satellite'],
        epochs=np.array([0.0]),
        pre_computed_matrices=art_slice,
        generate_no_shadow=False, animate=False, show_progress=False)
    return float(mag[0])


def _aa2q(aa):
    """Axis-angle (3-vector) → quaternion (w,x,y,z)."""
    a = np.linalg.norm(aa)
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = aa / a
    return np.array([np.cos(a / 2), *(np.sin(a / 2) * ax)])


def _iso_brightness_worker(aa_init):
    """L-BFGS-B from one random seed to find iso-brightness attitude."""
    try:
        def obj(aa):
            q = _aa2q(aa)
            return (_brightness_lofi_worker(q) - _worker_target) ** 2

        res = minimize(obj, aa_init, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-12})
        q = _aa2q(res.x)
        mag = _brightness_lofi_worker(q)
        resid = abs(mag - _worker_target)

        R_true = Rotation.from_quat([_worker_true_q[1], _worker_true_q[2],
                                     _worker_true_q[3], _worker_true_q[0]])
        R_found = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        err = np.rad2deg((R_found.inv() * R_true).magnitude())

        return {
            'rotvec': res.x.tolist(),
            'quat': q.tolist(),
            'resid': float(resid),
            'att_err': float(err),
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    t_total = time.time()
    results = {
        'experiment': 'two_epoch_matching',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config': {},
        'benchmarks': {},
    }

    # ─── Phase 0: Setup & Benchmarks ───────────────────────────────────

    log("=" * 70)
    log("PHASE 0: Setup & Benchmarks")
    log("=" * 70)
    t0 = time.time()

    ctx = setup_experiment(
        n_observations=N_OBSERVATIONS,
        noise_sigma=NOISE_SIGMA,
        random_seed=RANDOM_SEED,
        end_time_utc=END_TIME_UTC)

    log(f"  Satellite loaded, {ctx.n_observations} epochs, "
        f"dt_sampling={ctx.dt_sampling:.1f}s")
    log(f"  Lightcurve range: [{ctx.observed_lc.min():.2f}, "
        f"{ctx.observed_lc.max():.2f}] mag")
    log(f"  True q0: {ctx.true_q0}")
    log(f"  True ω0 (°/s): {np.rad2deg(ctx.true_omega0)}")

    # FFT omega bound
    dt_s = ctx.dt_sampling
    fft_vals = np.abs(np.fft.rfft(ctx.observed_lc - np.mean(ctx.observed_lc)))
    freqs = np.fft.rfftfreq(ctx.n_observations, d=dt_s)
    peak_freq = freqs[1 + np.argmax(fft_vals[1:])]  # skip DC
    fft_omega_bound_deg = float(peak_freq * 360.0)  # full rotation = 360°
    # Use generous safety factor
    fft_omega_bound_deg = max(fft_omega_bound_deg, 0.2)
    log(f"  FFT peak freq: {peak_freq:.6f} Hz → ω bound: {fft_omega_bound_deg:.3f} °/s")

    # Benchmark: lo-fi single-epoch eval
    t_bench = time.time()
    for _ in range(100):
        brightness_single_epoch(ctx.true_q0, 0, ctx, use_shadows=False)
    lofi_single_ms = (time.time() - t_bench) / 100 * 1000
    log(f"  Benchmark lo-fi single epoch: {lofi_single_ms:.1f} ms")

    # Benchmark: hi-fi single-epoch eval
    t_bench = time.time()
    for _ in range(10):
        brightness_single_epoch(ctx.true_q0, 0, ctx, use_shadows=True)
    hifi_single_ms = (time.time() - t_bench) / 10 * 1000
    log(f"  Benchmark hi-fi single epoch: {hifi_single_ms:.1f} ms")

    # Choose epoch T: small gap for rotation angle culling (~7s at 500 obs/1hr)
    epoch_0_idx = 0
    epoch_T_idx = EPOCH_T_OFFSET

    # Validation epochs: spread across the window for progressive culling
    # Pick ~2%, ~5%, ~10%, ~25% through window
    check_epoch_indices = []
    for frac in [0.02, 0.05, 0.10, 0.25]:
        ci = max(EPOCH_T_OFFSET + 1, int(frac * ctx.n_observations))
        ci = min(ci, ctx.n_observations - 1)
        if ci not in check_epoch_indices:
            check_epoch_indices.append(ci)

    dt_epoch = ctx.observation_times[epoch_T_idx] - ctx.observation_times[epoch_0_idx]
    log(f"  Epoch 0: idx={epoch_0_idx}, mag={ctx.observed_lc[epoch_0_idx]:.3f}")
    log(f"  Epoch T: idx={epoch_T_idx}, mag={ctx.observed_lc[epoch_T_idx]:.3f}, "
        f"Δt={dt_epoch:.0f}s")
    log(f"  Validation epochs: {check_epoch_indices}")

    setup_time = time.time() - t0
    log(f"  Phase 0 time: {setup_time:.1f}s")

    results['config'] = {
        'n_observations': N_OBSERVATIONS,
        'noise_sigma': NOISE_SIGMA,
        'n_seeds': N_SEEDS,
        'true_q0': ctx.true_q0.tolist(),
        'true_omega0_deg': np.rad2deg(ctx.true_omega0).tolist(),
        'fft_omega_bound_deg': fft_omega_bound_deg,
        'epoch_0_idx': epoch_0_idx,
        'epoch_T_idx': epoch_T_idx,
        'validation_epoch_indices': check_epoch_indices,
        'dt_epoch_s': float(dt_epoch),
    }
    results['benchmarks'] = {
        'lofi_single_epoch_ms': lofi_single_ms,
        'hifi_single_epoch_ms': hifi_single_ms,
        'setup_time_s': setup_time,
    }
    save_results(JSON_PATH, results)

    # ─── Prepare worker data (picklable dict) ──────────────────────────

    worker_ctx_data = {
        'sun_pos': ctx.sun_pos,
        'obs_pos': ctx.obs_pos,
        'sat_pos': ctx.sat_pos,
        'obs_dist': ctx.obs_dist,
        'art_matrices': ctx.art_matrices,
        'satellite': ctx.satellite,
    }

    # ─── Phase 1: Iso-brightness candidates at epoch 0 ────────────────

    log("")
    log("=" * 70)
    log("PHASE 1: Iso-brightness candidates at epoch 0")
    log("=" * 70)
    t1 = time.time()

    # Lo-fi bias correction (uses truth — acceptable for controlled experiment)
    true_q_ep0 = ctx.true_quaternions[epoch_0_idx]
    lofi_mag_0 = brightness_single_epoch(true_q_ep0, epoch_0_idx, ctx, use_shadows=False)
    hifi_mag_0 = brightness_single_epoch(true_q_ep0, epoch_0_idx, ctx, use_shadows=True)
    lofi_bias_0 = lofi_mag_0 - hifi_mag_0
    target_lofi_0 = ctx.observed_lc[epoch_0_idx] + lofi_bias_0

    log(f"  Observed mag: {ctx.observed_lc[epoch_0_idx]:.4f}")
    log(f"  Lo-fi bias: {lofi_bias_0:.4f}, target lo-fi: {target_lofi_0:.4f}")

    seeds_0 = Rotation.random(N_SEEDS, random_state=2000)
    args_0 = [seeds_0[i].as_rotvec() for i in range(N_SEEDS)]

    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(worker_ctx_data, target_lofi_0, epoch_0_idx, true_q_ep0)) as pool:
        raw_1 = pool.map(_iso_brightness_worker, args_0)

    candidates_0_all = [r for r in raw_1 if r is not None and r['resid'] < NOISE_SIGMA]
    # Cap to N_MAX_CANDIDATES (take best by residual; truth has low residual)
    candidates_0_all.sort(key=lambda c: c['resid'])
    candidates_0 = candidates_0_all[:N_MAX_CANDIDATES]
    errs_0 = sorted([c['att_err'] for c in candidates_0])

    phase1_time = time.time() - t1
    log(f"  Converged: {len(candidates_0_all)}/{N_SEEDS} in {phase1_time:.0f}s "
        f"({phase1_time/N_SEEDS*1000:.1f} ms/seed)")
    if len(candidates_0_all) > N_MAX_CANDIDATES:
        log(f"  Capped: {len(candidates_0_all)} → {N_MAX_CANDIDATES}")
    if errs_0:
        log(f"  Min att error: {errs_0[0]:.2f}°")
        log(f"  Within 5°: {sum(1 for e in errs_0 if e < 5)}")
    else:
        log("  WARNING: no candidates found!")

    results['phase_1'] = {
        'n_seeds': N_SEEDS,
        'n_converged': len(candidates_0),
        'n_within_5deg': sum(1 for e in errs_0 if e < 5) if errs_0 else 0,
        'min_att_err_deg': errs_0[0] if errs_0 else None,
        'time_s': phase1_time,
        'per_seed_ms': phase1_time / N_SEEDS * 1000,
        'candidates': candidates_0,
    }
    save_results(JSON_PATH, results)

    # ─── Phase 2: Iso-brightness candidates at epoch T ─────────────────

    log("")
    log("=" * 70)
    log("PHASE 2: Iso-brightness candidates at epoch T")
    log("=" * 70)
    t2 = time.time()

    true_q_epT = ctx.true_quaternions[epoch_T_idx]
    lofi_mag_T = brightness_single_epoch(true_q_epT, epoch_T_idx, ctx, use_shadows=False)
    hifi_mag_T = brightness_single_epoch(true_q_epT, epoch_T_idx, ctx, use_shadows=True)
    lofi_bias_T = lofi_mag_T - hifi_mag_T
    target_lofi_T = ctx.observed_lc[epoch_T_idx] + lofi_bias_T

    log(f"  Observed mag: {ctx.observed_lc[epoch_T_idx]:.4f}")
    log(f"  Lo-fi bias: {lofi_bias_T:.4f}, target lo-fi: {target_lofi_T:.4f}")

    seeds_T = Rotation.random(N_SEEDS, random_state=3000)
    args_T = [seeds_T[i].as_rotvec() for i in range(N_SEEDS)]

    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(worker_ctx_data, target_lofi_T, epoch_T_idx, true_q_epT)) as pool:
        raw_2 = pool.map(_iso_brightness_worker, args_T)

    candidates_T_all = [r for r in raw_2 if r is not None and r['resid'] < NOISE_SIGMA]
    candidates_T_all.sort(key=lambda c: c['resid'])
    candidates_T = candidates_T_all[:N_MAX_CANDIDATES]
    errs_T = sorted([c['att_err'] for c in candidates_T])

    phase2_time = time.time() - t2
    log(f"  Converged: {len(candidates_T_all)}/{N_SEEDS} in {phase2_time:.0f}s "
        f"({phase2_time/N_SEEDS*1000:.1f} ms/seed)")
    if len(candidates_T_all) > N_MAX_CANDIDATES:
        log(f"  Capped: {len(candidates_T_all)} → {N_MAX_CANDIDATES}")
    if errs_T:
        log(f"  Min att error: {errs_T[0]:.2f}°")
        log(f"  Within 5°: {sum(1 for e in errs_T if e < 5)}")
    else:
        log("  WARNING: no candidates found!")

    results['phase_2'] = {
        'n_seeds': N_SEEDS,
        'n_converged': len(candidates_T),
        'n_within_5deg': sum(1 for e in errs_T if e < 5) if errs_T else 0,
        'min_att_err_deg': errs_T[0] if errs_T else None,
        'time_s': phase2_time,
        'per_seed_ms': phase2_time / N_SEEDS * 1000,
        'candidates': candidates_T,
    }
    save_results(JSON_PATH, results)

    # ─── Phase 3: Pair matching + omega derivation ─────────────────────

    log("")
    log("=" * 70)
    log("PHASE 3: Pair matching + omega derivation")
    log("=" * 70)
    t3 = time.time()

    n0 = len(candidates_0)
    nT = len(candidates_T)
    log(f"  Candidates: {n0} × {nT} = {n0 * nT:,} total pairs")

    if n0 == 0 or nT == 0:
        log("  ERROR: Not enough candidates for pair matching!")
        results['phase_3'] = {
            'n_total_pairs': 0, 'n_after_omega_filter': 0, 'time_s': 0, 'pairs': [],
        }
        save_results(JSON_PATH, results)
        sys.exit(1)

    # Convert to wxyz arrays for vectorized quaternion math
    quats_0_wxyz = np.array([c['quat'] for c in candidates_0])  # (n0, 4) w,x,y,z
    quats_T_wxyz = np.array([c['quat'] for c in candidates_T])  # (nT, 4) w,x,y,z

    safety_factor = 1.5
    max_omega_rad = np.deg2rad(fft_omega_bound_deg * safety_factor)
    log(f"  ω bound: {fft_omega_bound_deg:.3f} °/s × {safety_factor} safety "
        f"→ max |ω|: {np.rad2deg(max_omega_rad):.3f} °/s")

    # Fully vectorized pair matching + omega derivation (chunked for memory)
    # Body-frame relative rotation: q_rel = conj(q_0) * q_T
    # Then omega_body = rotvec(q_rel) / dt
    true_omega_deg_arr = np.rad2deg(ctx.true_omega0)
    surviving_pairs = []
    CHUNK = 500  # 500 × nT pairs per chunk

    def _quat_conj_wxyz(q):
        """Conjugate of quaternion(s) in w,x,y,z format."""
        c = q.copy()
        c[..., 1:] *= -1
        return c

    def _quat_mult_wxyz(q1, q2):
        """Hamilton product of quaternion arrays, w,x,y,z convention.
        q1: (M, 4), q2: (N, 4) → broadcast to (M, N, 4)"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return np.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], axis=-1)

    def _quat_to_rotvec_wxyz(q):
        """Convert quaternion(s) w,x,y,z → rotation vector(s).
        q: (..., 4) → (..., 3)"""
        # Ensure w > 0 (shorter path)
        sign = np.sign(q[..., 0:1])
        sign[sign == 0] = 1
        q = q * sign
        w = np.clip(q[..., 0], -1.0, 1.0)
        half_angle = np.arccos(w)
        angle = 2.0 * half_angle
        sin_half = np.sin(half_angle)
        # Avoid division by zero for near-identity rotations
        safe = sin_half > 1e-12
        xyz = q[..., 1:4]
        axis = np.where(safe[..., None], xyz / np.where(safe, sin_half, 1.0)[..., None],
                        np.zeros_like(xyz))
        return angle[..., None] * axis

    for i_start in range(0, n0, CHUNK):
        i_end = min(i_start + CHUNK, n0)
        chunk_0 = quats_0_wxyz[i_start:i_end]  # (chunk, 4)

        # conj(q_0): (chunk, 4)
        q0_conj = _quat_conj_wxyz(chunk_0)

        # Broadcast multiply: (chunk, 1, 4) * (1, nT, 4) → (chunk, nT, 4)
        q_rel = _quat_mult_wxyz(q0_conj[:, None, :], quats_T_wxyz[None, :, :])

        # Rotation vectors: (chunk, nT, 3)
        rotvecs = _quat_to_rotvec_wxyz(q_rel)

        # Omega = rotvec / dt: (chunk, nT, 3)
        omega_rad = rotvecs / dt_epoch

        # Filter by omega magnitude
        omega_mag = np.linalg.norm(omega_rad, axis=-1)  # (chunk, nT)
        valid_i, valid_j = np.where(omega_mag < max_omega_rad)

        for ii, jj in zip(valid_i, valid_j):
            gi = i_start + int(ii)
            omega_est_rad = omega_rad[ii, jj]
            omega_est_deg = np.rad2deg(omega_est_rad)
            omega_err = float(np.linalg.norm(omega_est_deg - true_omega_deg_arr))
            surviving_pairs.append({
                'q0_idx': gi,
                'qT_idx': int(jj),
                'omega_deg': omega_est_deg.tolist(),
                'omega_rad': omega_est_rad.tolist(),
                'omega_err_deg': omega_err,
                'att_err_0': candidates_0[gi]['att_err'],
                'att_err_T': candidates_T[int(jj)]['att_err'],
            })

        if (i_start // CHUNK) % 2 == 0:
            log(f"    Chunk {i_start}-{i_end}/{n0}: "
                f"{len(surviving_pairs):,} survivors so far")

    phase3_time = time.time() - t3
    log(f"  After omega magnitude filter: {len(surviving_pairs):,} pairs")
    log(f"  Phase 3 time: {phase3_time:.1f}s")

    # Sort by omega error for logging
    surviving_pairs.sort(key=lambda p: p['omega_err_deg'])
    if surviving_pairs:
        log(f"  Best omega error: {surviving_pairs[0]['omega_err_deg']:.4f} °/s")
        log(f"  Top-5 pairs:")
        for i, p in enumerate(surviving_pairs[:5]):
            log(f"    {i+1}. att_err_0={p['att_err_0']:.1f}°, "
                f"att_err_T={p['att_err_T']:.1f}°, "
                f"ω_err={p['omega_err_deg']:.4f}°/s, "
                f"ω={[f'{x:.4f}' for x in p['omega_deg']]}°/s")

    results['phase_3'] = {
        'n_total_pairs': n0 * nT,
        'n_after_rotation_filter': len(surviving_indices),
        'n_after_omega_filter': len(surviving_pairs),
        'time_s': phase3_time,
        'pairs': surviving_pairs[:500],  # save top 500
    }
    save_results(JSON_PATH, results)

    # ─── Phase 4: Validation epoch checks ──────────────────────────────

    log("")
    log("=" * 70)
    log("PHASE 4: Validation epoch checks (progressive culling)")
    log("=" * 70)
    t4 = time.time()

    active_pairs = surviving_pairs[:]
    n_after_each = []

    for ci, check_idx in enumerate(check_epoch_indices):
        if not active_pairs:
            log(f"  Check epoch {ci+1}: 0 pairs remaining, skipping")
            n_after_each.append(0)
            continue

        t_check = time.time()
        check_dt = ctx.observation_times[check_idx] - ctx.observation_times[epoch_0_idx]

        # Evaluate lo-fi brightness at check epoch for each pair
        new_active = []
        for p in active_pairs:
            q0_wxyz = np.array(candidates_0[p['q0_idx']]['quat'])
            omega_rad = np.array(p['omega_rad'])

            # Propagate: R_check = R_0 * Rotation.from_rotvec(omega_body * dt)
            R_0 = Rotation.from_quat([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])
            R_check = R_0 * Rotation.from_rotvec(omega_rad * check_dt)
            q_check_scipy = R_check.as_quat()  # x,y,z,w
            q_check_wxyz = np.array([q_check_scipy[3], q_check_scipy[0],
                                     q_check_scipy[1], q_check_scipy[2]])

            mag_pred = brightness_single_epoch(q_check_wxyz, check_idx, ctx,
                                               use_shadows=False)
            # Compare with observed (with lo-fi bias — use average of ep0/epT biases)
            avg_bias = (lofi_bias_0 + lofi_bias_T) / 2
            resid = abs(mag_pred - (ctx.observed_lc[check_idx] + avg_bias))

            if resid < 3 * NOISE_SIGMA:
                p['check_resids'] = p.get('check_resids', []) + [float(resid)]
                new_active.append(p)

        active_pairs = new_active
        n_after_each.append(len(active_pairs))
        dt_check = time.time() - t_check
        log(f"  Check epoch {ci+1} (idx={check_idx}): "
            f"{n_after_each[-1]} survivors ({dt_check:.1f}s)")

    phase4_time = time.time() - t4
    log(f"  Phase 4 time: {phase4_time:.1f}s")
    log(f"  Survivors after all checks: {len(active_pairs)}")

    results['phase_4'] = {
        'check_epochs': check_epoch_indices,
        'n_after_each_check': n_after_each,
        'time_s': phase4_time,
        'n_final_survivors': len(active_pairs),
    }
    save_results(JSON_PATH, results)

    # ─── Phase 5: Lo-fi lightcurve ranking ─────────────────────────────

    log("")
    log("=" * 70)
    log("PHASE 5: Lo-fi lightcurve ranking")
    log("=" * 70)
    t5 = time.time()

    if not active_pairs:
        log("  No pairs to rank!")
        results['phase_5'] = {'n_evaluated': 0, 'time_s': 0, 'ranked': []}
        save_results(JSON_PATH, results)
    else:
        ranked = []
        n_eval = len(active_pairs)
        log(f"  Evaluating {n_eval} pairs over full lightcurve...")

        for pi, p in enumerate(active_pairs):
            t_pair = time.time()
            q0_wxyz = np.array(candidates_0[p['q0_idx']]['quat'])
            omega_rad = np.array(p['omega_rad'])

            # Propagate over full observation window (tumbling dynamics)
            quats_prop, _ = propagate_attitude(
                q0=q0_wxyz, omega0=omega_rad,
                times=ctx.observation_times,
                mode="tumbling",
                inertia_tensor=ctx.inertia_tensor)

            # Create lo-fi ObjectiveFunction for full LC evaluation
            obj_lofi = ObjectiveFunction(
                satellite=ctx.satellite,
                observation_times=ctx.observation_times,
                observed_lightcurve=ctx.observed_lc,
                sun_positions_j2000=ctx.sun_pos,
                observer_positions_j2000=ctx.obs_pos,
                satellite_positions_j2000=ctx.sat_pos,
                observer_distances=ctx.obs_dist,
                compute_shadows_flag=False,
                articulation_matrices=ctx.art_matrices,
                mode="tumbling",
                inertia_tensor=ctx.inertia_tensor,
                show_progress=False)

            # Compute body-frame vectors and predicted LC
            k1, k2 = obj_lofi._compute_body_frame_vectors(quats_prop)
            predicted = obj_lofi._generate_predicted_lightcurve(k1, k2)

            # RMS residual
            valid = np.isfinite(predicted) & np.isfinite(ctx.observed_lc)
            if np.sum(valid) > 0:
                rms = float(np.sqrt(np.mean(
                    (predicted[valid] - ctx.observed_lc[valid]) ** 2)))
            else:
                rms = 1e10

            # Attitude error at epoch 0
            att_err = attitude_error_deg(q0_wxyz, ctx.true_q0)
            omega_err = float(np.linalg.norm(
                np.rad2deg(omega_rad) - np.rad2deg(ctx.true_omega0)))

            ranked.append({
                'q0': q0_wxyz.tolist(),
                'omega_deg': np.rad2deg(omega_rad).tolist(),
                'omega_rad': omega_rad.tolist(),
                'rms': rms,
                'att_err': att_err,
                'omega_err': omega_err,
                'q0_idx': p['q0_idx'],
                'qT_idx': p['qT_idx'],
            })

            if (pi + 1) % 50 == 0:
                log(f"    Evaluated {pi+1}/{n_eval}...")

        # Sort by RMS
        ranked.sort(key=lambda r: r['rms'])
        phase5_time = time.time() - t5
        log(f"  Phase 5 time: {phase5_time:.1f}s")

        log(f"  Top-20 by RMS:")
        for i, r in enumerate(ranked[:20]):
            log(f"    {i+1}. RMS={r['rms']:.4f}, att_err={r['att_err']:.1f}°, "
                f"ω_err={r['omega_err']:.4f}°/s")

        results['phase_5'] = {
            'n_evaluated': n_eval,
            'time_s': phase5_time,
            'ranked': ranked[:100],  # save top 100
        }
        save_results(JSON_PATH, results)

    # ─── Phase 6: Hi-fi refinement ─────────────────────────────────────

    log("")
    log("=" * 70)
    log("PHASE 6: Hi-fi refinement")
    log("=" * 70)
    t6 = time.time()

    if 'ranked' not in results.get('phase_5', {}) or not results['phase_5']['ranked']:
        log("  No candidates to refine!")
        results['phase_6'] = {'n_refined': 0, 'time_s': 0, 'results': []}
    else:
        top_k = ranked[:TOP_K_REFINE]
        refined_results = []

        for ki, cand in enumerate(top_k):
            log(f"  Refining candidate {ki+1}/{TOP_K_REFINE}: "
                f"RMS={cand['rms']:.4f}, att_err={cand['att_err']:.1f}°")
            t_ref = time.time()

            # Convert q0 to axis-angle for optimization
            R_init = Rotation.from_quat([cand['q0'][1], cand['q0'][2],
                                         cand['q0'][3], cand['q0'][0]])
            aa_init = R_init.as_rotvec()
            omega_init = np.array(cand['omega_rad'])
            x0 = np.concatenate([aa_init, omega_init])

            # Create hi-fi objective
            obj_hifi = ObjectiveFunction(
                satellite=ctx.satellite,
                observation_times=ctx.observation_times,
                observed_lightcurve=ctx.observed_lc,
                sun_positions_j2000=ctx.sun_pos,
                observer_positions_j2000=ctx.obs_pos,
                satellite_positions_j2000=ctx.sat_pos,
                observer_distances=ctx.obs_dist,
                compute_shadows_flag=True,
                articulation_matrices=ctx.art_matrices,
                mode="tumbling",
                inertia_tensor=ctx.inertia_tensor,
                show_progress=True)

            # Bounds: axis-angle ±2π, omega ± FFT bound
            omega_bound_rad = np.deg2rad(fft_omega_bound_deg * safety_factor)
            bounds = [
                (-2 * np.pi, 2 * np.pi),
                (-2 * np.pi, 2 * np.pi),
                (-2 * np.pi, 2 * np.pi),
                (-omega_bound_rad, omega_bound_rad),
                (-omega_bound_rad, omega_bound_rad),
                (-omega_bound_rad, omega_bound_rad),
            ]

            res = minimize(
                obj_hifi.evaluate, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})

            # Extract results
            aa_final = res.x[:3]
            omega_final = res.x[3:6]
            q_final = axis_angle_to_quaternion(aa_final)
            att_err_final = attitude_error_deg(q_final, ctx.true_q0)
            omega_err_final = float(np.linalg.norm(
                np.rad2deg(omega_final) - np.rad2deg(ctx.true_omega0)))
            rms_final = float(np.sqrt(res.fun))  # objective is MSE

            ref_time = time.time() - t_ref
            log(f"    → att_err={att_err_final:.2f}°, ω_err={omega_err_final:.4f}°/s, "
                f"RMS={rms_final:.4f}, evals={obj_hifi.n_evaluations}, "
                f"time={ref_time:.1f}s")

            refined_results.append({
                'q0_final': q_final.tolist(),
                'omega_final_deg': np.rad2deg(omega_final).tolist(),
                'att_err': att_err_final,
                'omega_err': omega_err_final,
                'rms': rms_final,
                'n_evaluations': obj_hifi.n_evaluations,
                'time_s': ref_time,
                'success': bool(res.success),
                'initial_att_err': cand['att_err'],
                'initial_omega_err': cand['omega_err'],
            })

        phase6_time = time.time() - t6
        log(f"  Phase 6 time: {phase6_time:.1f}s")

        results['phase_6'] = {
            'n_refined': len(refined_results),
            'time_s': phase6_time,
            'results': refined_results,
        }
        save_results(JSON_PATH, results)

    # ─── Final Summary ─────────────────────────────────────────────────

    total_time = time.time() - t_total
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)

    phase_times = {
        'setup': setup_time,
        'phase_1': results.get('phase_1', {}).get('time_s', 0),
        'phase_2': results.get('phase_2', {}).get('time_s', 0),
        'phase_3': results.get('phase_3', {}).get('time_s', 0),
        'phase_4': results.get('phase_4', {}).get('time_s', 0),
        'phase_5': results.get('phase_5', {}).get('time_s', 0),
        'phase_6': results.get('phase_6', {}).get('time_s', 0),
    }

    log(f"  Total time: {total_time:.0f}s")
    for name, t in phase_times.items():
        log(f"    {name}: {t:.1f}s")

    # Best result
    best_att = None
    best_omega = None
    best_rms = None
    refined = results.get('phase_6', {}).get('results', [])
    if refined:
        best = min(refined, key=lambda r: r['att_err'])
        best_att = best['att_err']
        best_omega = best['omega_err']
        best_rms = best['rms']
        log(f"  Best refined: att_err={best_att:.2f}°, "
            f"ω_err={best_omega:.4f}°/s, RMS={best_rms:.4f}")
    else:
        log("  No refined results available")

    success = best_att is not None and best_att < 10.0

    results['summary'] = {
        'total_time_s': total_time,
        'phase_times_s': phase_times,
        'best_att_err_deg': best_att,
        'best_omega_err_deg': best_omega,
        'best_rms': best_rms,
        'success': success,
    }
    save_results(JSON_PATH, results)

    log(f"\n  Results: {JSON_PATH}")
    log(f"  Log:     {LOG_PATH}")
    log(f"  Success: {success}")
