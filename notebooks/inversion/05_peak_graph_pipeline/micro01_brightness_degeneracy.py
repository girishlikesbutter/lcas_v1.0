#!/usr/bin/env python3
"""
Micro-01 — Brightness Degeneracy Test.

At selected epochs, sample 10,000 random attitudes uniformly from SO(3).
For each, evaluate the forward brightness model (lo-fi, no shadows).
Count how many match the truth brightness within tolerances of 1%, 5%, 10%.

This quantifies the many-to-one ambiguity that peak anchoring must overcome.
"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from lib.experiment_setup import (
    setup_experiment, brightness_single_epoch, save_results,
)
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

# ── Config ──
N_SAMPLES = 10_000
SEED = 42
EPOCH_INDICES = [0, 125, 250, 375, 499]
TOLERANCES_PCT = [1, 5, 10]
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(
    n_observations=500, noise_sigma=0.05, random_seed=SEED,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)
print(f"Setup: {time.time() - t0:.1f}s")

# ── Sample random attitudes uniformly from SO(3) ──
rng = np.random.default_rng(SEED)
rotations = Rotation.random(N_SAMPLES, random_state=rng)
rot_matrices = rotations.as_matrix()  # (N, 3, 3)


def evaluate_epoch_vectorized(epoch_idx: int) -> dict:
    """
    Evaluate brightness for all N_SAMPLES random attitudes at one epoch.

    Treats 10,000 attitudes as 10,000 "fake epochs" with different body-frame
    sun/observer vectors, then calls generate_lightcurves once (vectorized).
    """
    # Inertial-frame sun and observer vectors at this epoch
    sun_vec = ctx.sun_pos[epoch_idx] - ctx.sat_pos[epoch_idx]
    obs_vec = ctx.obs_pos[epoch_idx] - ctx.sat_pos[epoch_idx]

    # Transform to body frame for each random attitude
    k1 = np.einsum('nij,j->ni', rot_matrices, sun_vec)
    k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 = np.einsum('nij,j->ni', rot_matrices, obs_vec)
    k2 /= np.linalg.norm(k2, axis=1, keepdims=True)

    # Tile articulation matrices and observer distances
    art_tiled = {
        c: np.tile(m[epoch_idx:epoch_idx + 1], (N_SAMPLES, 1, 1))
        for c, m in ctx.art_matrices.items()
    }
    obs_dist_tiled = np.full(N_SAMPLES, ctx.obs_dist[epoch_idx])
    lit = create_no_shadow_lit_status(ctx.satellite, N_SAMPLES)

    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit,
        k1_vectors_array=k1,
        k2_vectors_array=k2,
        observer_distances=obs_dist_tiled,
        satellite=ctx.satellite,
        epochs=np.arange(N_SAMPLES, dtype=float),
        pre_computed_matrices=art_tiled,
        generate_no_shadow=False, animate=False, show_progress=False,
    )
    return mags


# ── Evaluate truth brightness (lo-fi) at each epoch ──
# Use the propagated true quaternion at each epoch
truth_mags = {}
for idx in EPOCH_INDICES:
    q_true_epoch = ctx.true_quaternions[idx]  # (w,x,y,z)
    truth_mags[idx] = brightness_single_epoch(
        q_true_epoch, idx, ctx, use_shadows=False
    )

# ── Main loop ──
results = {}
print(f"\nSampling {N_SAMPLES} random attitudes from SO(3)")
print(f"Evaluating lo-fi brightness at {len(EPOCH_INDICES)} epochs\n")

for epoch_idx in EPOCH_INDICES:
    t1 = time.time()
    mags = evaluate_epoch_vectorized(epoch_idx)
    dt = time.time() - t1

    truth_mag = truth_mags[epoch_idx]
    valid = np.isfinite(mags)
    n_valid = valid.sum()

    epoch_results = {
        'epoch_idx': int(epoch_idx),
        'time_s': float(ctx.observation_times[epoch_idx]),
        'truth_mag': float(truth_mag),
        'eval_time_s': round(dt, 2),
        'n_valid': int(n_valid),
        'mag_min': float(np.nanmin(mags)),
        'mag_max': float(np.nanmax(mags)),
        'mag_median': float(np.nanmedian(mags)),
    }

    print(f"Epoch {epoch_idx:3d} (t = {ctx.observation_times[epoch_idx]:7.1f}s):")
    print(f"  Truth brightness:  {truth_mag:.4f} mag")
    print(f"  Random mag range:  [{np.nanmin(mags):.4f}, {np.nanmax(mags):.4f}]")
    print(f"  Random mag median: {np.nanmedian(mags):.4f}")
    print(f"  Valid samples:     {n_valid}/{N_SAMPLES}")
    print(f"  Eval time:         {dt:.2f}s")

    for tol_pct in TOLERANCES_PCT:
        threshold = abs(truth_mag) * tol_pct / 100.0
        n_match = int(np.sum(np.abs(mags - truth_mag) < threshold))
        pct_match = 100.0 * n_match / N_SAMPLES
        epoch_results[f'tol_{tol_pct}pct_threshold_mag'] = round(threshold, 4)
        epoch_results[f'tol_{tol_pct}pct_n_match'] = n_match
        epoch_results[f'tol_{tol_pct}pct_pct_match'] = round(pct_match, 2)
        print(f"  Within {tol_pct:2d}% (±{threshold:.3f} mag): "
              f"{n_match:5d} / {N_SAMPLES} ({pct_match:.2f}%)")

    results[f'epoch_{epoch_idx}'] = epoch_results
    print()

# ── Summary table ──
print("=" * 72)
print("SUMMARY: Brightness Degeneracy (% of 10k random attitudes matching truth)")
print("=" * 72)
header = f"{'Epoch':>6} {'t(s)':>8} {'Truth(mag)':>11} "
header += "  ".join(f'{t}% (±mag)' for t in TOLERANCES_PCT)
print(header)
print("-" * 72)

for epoch_idx in EPOCH_INDICES:
    r = results[f'epoch_{epoch_idx}']
    line = f"{epoch_idx:>6d} {r['time_s']:>8.1f} {r['truth_mag']:>11.4f} "
    parts = []
    for t in TOLERANCES_PCT:
        n = r[f'tol_{t}pct_n_match']
        pct = r[f'tol_{t}pct_pct_match']
        thr = r[f'tol_{t}pct_threshold_mag']
        parts.append(f"{pct:5.1f}% (±{thr:.2f})")
    line += "  ".join(parts)
    print(line)

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.1f}s")

# ── Save ──
results['config'] = {
    'n_samples': N_SAMPLES,
    'seed': SEED,
    'epoch_indices': EPOCH_INDICES,
    'tolerances_pct': TOLERANCES_PCT,
    'n_observations': 500,
    'true_omega_deg': [0.5, -0.3, 2.0],
    'model': 'lo-fi (no shadows)',
}
save_results(RESULTS_DIR / 'micro01_brightness_degeneracy.json', results)
print(f"Results saved to {RESULTS_DIR / 'micro01_brightness_degeneracy.json'}")
