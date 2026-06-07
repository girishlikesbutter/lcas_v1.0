#!/usr/bin/env python3
"""
m096 Exp 4: Lo-fi Full-Curve Discrimination Power.

Question: How well can lo-fi LC MSE distinguish truth from random omegas
— without any peak-anchoring?

For each seed: generate lo-fi LC at truth + 200 random omega directions
(same |w|). Compute MSE. Report truth's percentile rank.
"""

import sys, os, time
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1 = RESULTS_DIR / "m096_stage1"
CKPT = RESULTS_DIR / "m096_exp4_lofi_discrim"
CKPT.mkdir(exist_ok=True)

master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                 allow_pickle=True)
I_tensor = master['inertia_tensor']

N_RANDOM = 200
LOFI_WORKERS = 24

print("=" * 70)
print(f"EXP 4: Lo-fi Discrimination ({N_RANDOM} random directions per seed)")
print("=" * 70)

# Load satellite model (once)
print("Loading satellite model...", flush=True)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)

_satellite = CTX.satellite
_sun = CTX.sun_pos
_obs = CTX.obs_pos
_sat = CTX.sat_pos
_dist = CTX.obs_dist
_art = CTX.art_matrices
_I = I_tensor

ALL_SEEDS = list(range(100))
rng = np.random.default_rng(456)
summary = []
t_total = time.time()


def eval_lofi_lc(args):
    """Generate lo-fi LC for a (q0, w0) state. Returns predicted magnitudes."""
    q0_wxyz, w0_rad, obs_times = args
    from src.computation.shadow_engine import create_no_shadow_lit_status as _no_shadow
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

    quats, _ = propagate_attitude(q0_wxyz, w0_rad, obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = (_sun[:n_ep] - _sat[:n_ep]); sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = (_obs[:n_ep] - _sat[:n_ep]); ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)
    lit = _no_shadow(_satellite, n_ep)
    pred, _, _, _, _, _ = _gen_lc(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=_dist, satellite=_satellite,
        epochs=np.arange(n_ep, dtype=float), pre_computed_matrices=_art, show_progress=False)
    return pred


for seed in ALL_SEEDS:
    d = np.load(str(STAGE1 / f"seed_{seed:03d}.npz"), allow_pickle=True)

    true_q0 = d['true_q0']
    true_omega0 = d['true_omega0']
    obs_times = d['obs_times']
    observed_lc = d['observed_lc']
    true_omega_mag = np.linalg.norm(true_omega0)

    # Generate random omega directions (same magnitude as truth)
    random_dirs = rng.standard_normal((N_RANDOM, 3))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
    random_omegas = random_dirs * true_omega_mag

    # Also generate random initial attitudes for random omegas
    random_q0s = Rotation.random(N_RANDOM, random_state=rng).as_quat()  # xyzw
    random_q0s_wxyz = random_q0s[:, [3, 0, 1, 2]]  # convert to wxyz

    # Build evaluation args: truth + N_RANDOM random states
    _obs_times = obs_times
    eval_args = [(true_q0, true_omega0, obs_times)]  # index 0 = truth
    for i in range(N_RANDOM):
        eval_args.append((random_q0s_wxyz[i], random_omegas[i], obs_times))

    t0 = time.time()
    with Pool(LOFI_WORKERS) as pool:
        all_lcs = pool.map(eval_lofi_lc, eval_args)
    eval_time = time.time() - t0

    # Compute MSEs
    truth_lc = all_lcs[0]
    mse_truth = float(np.mean((truth_lc - observed_lc) ** 2))
    mse_random = np.array([float(np.mean((lc - observed_lc) ** 2)) for lc in all_lcs[1:]])

    # Truth percentile (what fraction of random omegas have HIGHER MSE)
    truth_percentile = float(np.mean(mse_random > mse_truth) * 100)

    # Windowed MSE (360s around brightest peak)
    peaks_idx = d['peaks_idx']
    if len(peaks_idx) > 0:
        peak_mags = d['peak_mags']
        brightest_ep = peaks_idx[np.argmin(peak_mags)]
        brightest_time = obs_times[brightest_ep]
        win_mask = np.abs(obs_times - brightest_time) <= 180.0
        if win_mask.sum() > 5:
            mse_truth_win = float(np.mean((truth_lc[win_mask] - observed_lc[win_mask]) ** 2))
            mse_random_win = np.array([float(np.mean((lc[win_mask] - observed_lc[win_mask]) ** 2))
                                        for lc in all_lcs[1:]])
            truth_pct_win = float(np.mean(mse_random_win > mse_truth_win) * 100)
        else:
            mse_truth_win = mse_truth
            mse_random_win = mse_random
            truth_pct_win = truth_percentile
    else:
        mse_truth_win = mse_truth
        mse_random_win = mse_random
        truth_pct_win = truth_percentile

    np.savez(str(CKPT / f"seed_{seed:03d}.npz"),
             seed=seed,
             mse_truth=mse_truth,
             mse_random=mse_random,
             truth_percentile=truth_percentile,
             mse_truth_windowed=mse_truth_win,
             mse_random_windowed=mse_random_win,
             truth_percentile_windowed=truth_pct_win,
             random_omegas=random_omegas,
             random_q0s=random_q0s_wxyz,
             truth_lc_lofi=truth_lc,
             observed_lc=observed_lc,
             eval_time=eval_time,
    )

    print(f"  seed {seed:3d}: truth_pct={truth_percentile:5.1f}% "
          f"win_pct={truth_pct_win:5.1f}% "
          f"mse_truth={mse_truth:.3f} "
          f"mse_random_med={np.median(mse_random):.3f} "
          f"{eval_time:.1f}s")

    summary.append({
        'seed': seed,
        'mse_truth': mse_truth,
        'mse_random_median': float(np.median(mse_random)),
        'mse_random_min': float(mse_random.min()),
        'truth_percentile': truth_percentile,
        'truth_percentile_windowed': truth_pct_win,
        'eval_time': eval_time,
    })

save_results(str(CKPT / "summary.json"), summary)
total_time = time.time() - t_total

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"ANALYSIS ({total_time:.0f}s total)")
print(f"{'='*70}")

pcts = [r['truth_percentile'] for r in summary]
pcts_win = [r['truth_percentile_windowed'] for r in summary]

print(f"\n  Full-curve lo-fi MSE — truth percentile among 200 random:")
print(f"    median={np.median(pcts):.1f}%, mean={np.mean(pcts):.1f}%")
print(f"    > 99%: {sum(1 for p in pcts if p > 99)} seeds (truth clearly best)")
print(f"    > 95%: {sum(1 for p in pcts if p > 95)} seeds")
print(f"    > 90%: {sum(1 for p in pcts if p > 90)} seeds")
print(f"    > 50%: {sum(1 for p in pcts if p > 50)} seeds (truth better than median)")
print(f"    < 50%: {sum(1 for p in pcts if p <= 50)} seeds (truth WORSE than median)")

print(f"\n  Windowed (360s) lo-fi MSE — truth percentile:")
print(f"    median={np.median(pcts_win):.1f}%, mean={np.mean(pcts_win):.1f}%")
print(f"    > 99%: {sum(1 for p in pcts_win if p > 99)} seeds")
print(f"    > 95%: {sum(1 for p in pcts_win if p > 95)} seeds")
print(f"    > 50%: {sum(1 for p in pcts_win if p > 50)} seeds")

print(f"\nSaved to {CKPT}/")
