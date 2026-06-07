#!/usr/bin/env python3
"""
Micro-05 — Hi-Fi Candidate Count at Multiple Peaks.

Tests whether hi-fi evaluation (shadows/ray tracing) reduces attitude
candidate count vs lo-fi.  At 3 brightness peaks, sample 10,000 random
SO(3) attitudes, evaluate in both lo-fi and hi-fi mode, then apply
the dL/dt ≈ 0 derivative filter on hi-fi matches.
"""
import sys, time, numpy as np, multiprocessing as mp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from lib.experiment_setup import (
    setup_experiment, brightness_single_epoch, save_results,
)
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
N_SAMPLES   = 10_000
SEED        = 42
TOL_PCT     = 1       # brightness match tolerance (%)
DPHI        = 1e-5    # finite-diff step for gradient (rad)
N_PEAKS     = 3
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Globals for fork workers ──
_ctx = None
_k1_work = None   # (N, 3) body-frame sun vectors for current batch
_k2_work = None   # (N, 3) body-frame observer vectors
_eidx    = None   # epoch index


def _hifi_one(i):
    """Worker: hi-fi brightness for work item i at epoch _eidx."""
    k1, k2 = _k1_work[i], _k2_work[i]
    art = {c: m[_eidx:_eidx+1] for c, m in _ctx.art_matrices.items()}
    lit = compute_shadows(
        satellite=_ctx.satellite, k1_vectors=k1.reshape(1, 3),
        explicit_component_matrices=art, show_progress=False)
    mag, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1.reshape(1, 3),
        k2_vectors_array=k2.reshape(1, 3),
        observer_distances=np.array([_ctx.obs_dist[_eidx]]),
        satellite=_ctx.satellite, epochs=np.array([0.0]),
        pre_computed_matrices=art,
        generate_no_shadow=False, animate=False, show_progress=False)
    return float(mag[0])


def lofi_batch(k1, k2, eidx, ctx):
    """Vectorised lo-fi brightness for N attitudes at one epoch."""
    N = len(k1)
    art = {c: np.tile(m[eidx:eidx+1], (N, 1, 1))
           for c, m in ctx.art_matrices.items()}
    lit = create_no_shadow_lit_status(ctx.satellite, N)
    mags, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.full(N, ctx.obs_dist[eidx]),
        satellite=ctx.satellite, epochs=np.arange(N, dtype=float),
        pre_computed_matrices=art,
        generate_no_shadow=False, animate=False, show_progress=False)
    return mags


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
# ══════════════════════════════════════════════════════════════

    t0 = time.time()
    ctx = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=SEED,
        true_omega_deg=(0.5, -0.3, 2.0),
        end_time_utc='2020-02-05T11:00:00',
    )
    _ctx = ctx
    _, omega_hist = propagate_attitude(
        ctx.true_q0, ctx.true_omega0, ctx.observation_times,
        mode="tumbling", inertia_tensor=ctx.inertia_tensor)
    dLdt_thr = ctx.noise_sigma / ctx.dt_sampling
    print(f"Setup: {time.time() - t0:.1f}s")

    # ── Random attitudes ──
    rng = np.random.default_rng(SEED)
    Rs = Rotation.random(N_SAMPLES, random_state=rng).as_matrix()

    # ── Find peaks (minima in mag = maxima in brightness) ──
    peaks, props = find_peaks(-ctx.true_lc, prominence=0.05, distance=5)
    order = np.argsort(-props['prominences'])
    selected = sorted(peaks[order[:N_PEAKS]])
    print(f"\nFound {len(peaks)} peaks, selected top {N_PEAKS} by prominence:")
    for pk in selected:
        print(f"  epoch {pk}, t={ctx.observation_times[pk]:.0f}s, "
              f"mag={ctx.true_lc[pk]:.4f}")

    # ── Main loop ──
    all_results = {}
    axes = np.eye(3)
    fork_mp = mp.get_context('fork')

    for pi, peak_idx in enumerate(selected):
        peak_idx = int(peak_idx)
        hifi_truth = ctx.true_lc[peak_idx]
        omega_pk = omega_hist[peak_idx]

        print(f"\n{'='*60}")
        print(f"Peak {pi+1}/{N_PEAKS}: epoch {peak_idx} "
              f"(t={ctx.observation_times[peak_idx]:.0f}s, "
              f"hi-fi={hifi_truth:.4f} mag)")
        print(f"{'='*60}")

        # Body-frame vectors for all random attitudes
        sv = ctx.sun_pos[peak_idx] - ctx.sat_pos[peak_idx]
        ov = ctx.obs_pos[peak_idx] - ctx.sat_pos[peak_idx]
        k1_all = np.einsum('nij,j->ni', Rs, sv)
        k1_all /= np.linalg.norm(k1_all, axis=1, keepdims=True)
        k2_all = np.einsum('nij,j->ni', Rs, ov)
        k2_all /= np.linalg.norm(k2_all, axis=1, keepdims=True)

        # ── Lo-fi (vectorised, fast) ──
        t1 = time.time()
        lofi_ref = brightness_single_epoch(
            ctx.true_quaternions[peak_idx], peak_idx, ctx, use_shadows=False)
        lofi_mags = lofi_batch(k1_all, k2_all, peak_idx, ctx)
        lofi_dt = time.time() - t1
        lofi_thr = abs(lofi_ref) * TOL_PCT / 100
        n_lofi = int(np.sum(np.abs(lofi_mags - lofi_ref) < lofi_thr))
        print(f"  Lo-fi: {n_lofi}/{N_SAMPLES} bright matches "
              f"(ref={lofi_ref:.4f}, ±{lofi_thr:.3f}) [{lofi_dt:.1f}s]")

        # ── Hi-fi (parallel, 8 workers) ──
        _k1_work, _k2_work, _eidx = k1_all, k2_all, peak_idx
        print(f"  Hi-fi: evaluating {N_SAMPLES} attitudes (8 cores)...")
        t2 = time.time()
        with fork_mp.Pool(8) as pool:
            hifi_list = pool.map(_hifi_one, range(N_SAMPLES))
        hifi_dt = time.time() - t2
        hifi_mags = np.array(hifi_list)
        hifi_thr = abs(hifi_truth) * TOL_PCT / 100
        hifi_mask = np.abs(hifi_mags - hifi_truth) < hifi_thr
        n_hifi = int(hifi_mask.sum())
        ms_per = 1000 * hifi_dt / N_SAMPLES
        print(f"  Hi-fi: {n_hifi}/{N_SAMPLES} bright matches "
              f"(ref={hifi_truth:.4f}, ±{hifi_thr:.3f}) "
              f"[{hifi_dt:.1f}s, {ms_per:.1f}ms/eval eff.]")

        # ── Derivative filter on hi-fi matches ──
        cidx = np.where(hifi_mask)[0]
        n_d1x = n_d5x = 0
        grad_dt = 0.0
        if len(cidx) > 0:
            gk1, gk2 = [], []
            for ci in cidx:
                for j in range(3):
                    k1p = k1_all[ci] + DPHI * np.cross(axes[j], k1_all[ci])
                    k1p /= np.linalg.norm(k1p)
                    k2p = k2_all[ci] + DPHI * np.cross(axes[j], k2_all[ci])
                    k2p /= np.linalg.norm(k2p)
                    gk1.append(k1p); gk2.append(k2p)
            _k1_work, _k2_work = np.array(gk1), np.array(gk2)
            t3 = time.time()
            with fork_mp.Pool(8) as pool:
                gm = pool.map(_hifi_one, range(len(gk1)))
            grad_dt = time.time() - t3
            gm = np.array(gm).reshape(len(cidx), 3)
            g_vecs = (gm - hifi_mags[cidx, None]) / DPHI
            gdotw = g_vecs @ omega_pk
            n_d1x = int(np.sum(np.abs(gdotw) < dLdt_thr))
            n_d5x = int(np.sum(np.abs(gdotw) < 5 * dLdt_thr))
            print(f"  Deriv filter: 1x={n_d1x}, 5x={n_d5x} "
                  f"(of {len(cidx)} hi-fi matches) [{grad_dt:.1f}s]")

        bright_red = f"{n_lofi/max(n_hifi,1):.1f}x" if n_hifi > 0 else "inf"
        print(f"  Reduction: lo-fi {n_lofi} -> hi-fi {n_hifi} ({bright_red})")

        all_results[f'peak_{pi}'] = {
            'epoch_idx': peak_idx,
            'time_s': float(ctx.observation_times[peak_idx]),
            'hifi_truth_mag': float(hifi_truth),
            'lofi_truth_mag': float(lofi_ref),
            'n_lofi_bright': n_lofi,
            'n_hifi_bright': n_hifi,
            'n_hifi_deriv_1x': n_d1x,
            'n_hifi_deriv_5x': n_d5x,
            'hifi_eval_time_s': round(hifi_dt, 1),
            'ms_per_hifi_eval': round(ms_per, 1),
            'grad_eval_time_s': round(grad_dt, 1),
        }

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY: Lo-fi vs Hi-fi Candidate Counts (10k random, 1% tol)")
    print(f"{'='*70}")
    print(f"{'Peak':>5} {'Epoch':>6} {'Lo-fi':>7} {'Hi-fi':>7} "
          f"{'D-1x':>6} {'D-5x':>6} {'BrightRed':>10} {'ms/eval':>8}")
    print(f"{'-'*70}")
    for i in range(N_PEAKS):
        r = all_results[f'peak_{i}']
        red = f"{r['n_lofi_bright']/max(r['n_hifi_bright'],1):.1f}x"
        print(f"{i+1:>5} {r['epoch_idx']:>6} {r['n_lofi_bright']:>7} "
              f"{r['n_hifi_bright']:>7} {r['n_hifi_deriv_1x']:>6} "
              f"{r['n_hifi_deriv_5x']:>6} {red:>10} "
              f"{r['ms_per_hifi_eval']:>8.1f}")

    print(f"\nReference (micro-01/02 lo-fi): ~800 bright -> ~50-280 deriv")
    total = time.time() - t0
    print(f"Total runtime: {total:.1f}s")

    all_results['config'] = {
        'n_samples': N_SAMPLES, 'seed': SEED,
        'brightness_tol_pct': TOL_PCT, 'dphi': DPHI,
        'n_peaks': N_PEAKS, 'dLdt_threshold': float(dLdt_thr),
    }
    save_results(RESULTS_DIR / 'm005_hifi_candidates.json', all_results)
    print(f"Saved to {RESULTS_DIR / 'm005_hifi_candidates.json'}")
