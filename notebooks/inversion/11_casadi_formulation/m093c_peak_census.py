#!/usr/bin/env python3
"""
m093b — Lo-fi vs hi-fi peak census.

For each of 100 seeds, generate lo-fi LC from true q0/omega and compare
peak counts with the hi-fi LC from the trajectory database.

Questions:
  1. Does lo-fi always have >= hi-fi peaks?
  2. How often does hi-fi have a peak that doesn't exist in lo-fi?
  3. What's the distribution of (n_lofi - n_hifi)?
"""

import sys
import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

# Load trajectory database
master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
I_tensor = master['inertia_tensor']
n_seeds = len(master['q0s'])

# Load satellite model for lo-fi
print("Loading satellite model...", flush=True)
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

satellite = ctx.satellite
sun_pos = ctx.sun_pos
obs_pos = ctx.obs_pos
sat_pos = ctx.sat_pos
obs_dist = ctx.obs_dist
art_matrices = ctx.art_matrices

# Peak detection parameters (same as pipeline)
PEAK_DISTANCE = 5
PEAK_PROMINENCE = 0.3
MATCH_WINDOW = 3  # ±3 epochs counts as a match

print(f"Analysing {n_seeds} seeds...\n")

results = []
for seed in range(n_seeds):
    true_q0 = master['q0s'][seed]
    true_w0 = master['omega0s'][seed]
    hifi_lc = master['mag_hifi'][seed]

    # Generate lo-fi LC from true state
    quats, _ = propagate_attitude(true_q0, true_w0, obs_times, "tumbling", I_tensor)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = sun_pos[:n_ep] - sat_pos[:n_ep]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = obs_pos[:n_ep] - sat_pos[:n_ep]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)

    lit = create_no_shadow_lit_status(satellite, n_ep)
    lofi_lc, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=obs_dist,
        satellite=satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=art_matrices, show_progress=False)

    # Detect peaks in both
    lofi_peaks, _ = find_peaks(-lofi_lc, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    hifi_peaks, _ = find_peaks(-hifi_lc, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)

    # Check: which hi-fi peaks have no lo-fi match?
    hifi_only = []
    for hp in hifi_peaks:
        matched = False
        for offset in range(-MATCH_WINDOW, MATCH_WINDOW + 1):
            if (hp + offset) in set(lofi_peaks):
                matched = True
                break
        if not matched:
            hifi_only.append(hp)

    # Check: which lo-fi peaks have no hi-fi match?
    lofi_only = []
    for lp in lofi_peaks:
        matched = False
        for offset in range(-MATCH_WINDOW, MATCH_WINDOW + 1):
            if (lp + offset) in set(hifi_peaks):
                matched = True
                break
        if not matched:
            lofi_only.append(lp)

    results.append({
        'seed': seed,
        'n_lofi': len(lofi_peaks),
        'n_hifi': len(hifi_peaks),
        'n_hifi_only': len(hifi_only),
        'n_lofi_only': len(lofi_only),
        'hifi_only_epochs': hifi_only,
    })

    if len(hifi_only) > 0:
        print(f"  seed {seed:3d}: lofi={len(lofi_peaks):2d} hifi={len(hifi_peaks):2d} "
              f"hifi_only={len(hifi_only)} at epochs {hifi_only}")

# Summary statistics
n_lofi_arr = np.array([r['n_lofi'] for r in results])
n_hifi_arr = np.array([r['n_hifi'] for r in results])
diff = n_lofi_arr - n_hifi_arr
n_hifi_only = np.array([r['n_hifi_only'] for r in results])

print(f"\n{'='*60}")
print(f"PEAK CENSUS — {n_seeds} seeds")
print(f"{'='*60}")
print(f"Lo-fi peaks:  mean={n_lofi_arr.mean():.1f}, median={np.median(n_lofi_arr):.0f}, "
      f"range=[{n_lofi_arr.min()}, {n_lofi_arr.max()}]")
print(f"Hi-fi peaks:  mean={n_hifi_arr.mean():.1f}, median={np.median(n_hifi_arr):.0f}, "
      f"range=[{n_hifi_arr.min()}, {n_hifi_arr.max()}]")
print(f"Difference (lofi - hifi): mean={diff.mean():.1f}, median={np.median(diff):.0f}, "
      f"range=[{diff.min()}, {diff.max()}]")
print(f"")
print(f"Seeds where lofi >= hifi: {np.sum(diff >= 0)}/{n_seeds} ({100*np.sum(diff >= 0)/n_seeds:.0f}%)")
print(f"Seeds where lofi < hifi:  {np.sum(diff < 0)}/{n_seeds} ({100*np.sum(diff < 0)/n_seeds:.0f}%)")
print(f"")
print(f"Hi-fi-only peaks (exist in hi-fi but NOT in lo-fi):")
print(f"  Seeds with any: {np.sum(n_hifi_only > 0)}/{n_seeds}")
print(f"  Total count:    {n_hifi_only.sum()}")
print(f"  Per-seed:       mean={n_hifi_only.mean():.2f}, max={n_hifi_only.max()}")

# Save results
import json
save_path = RESULTS_DIR / "m093b_peak_census.json"
save_data = {
    'n_seeds': n_seeds,
    'lofi_peaks_mean': float(n_lofi_arr.mean()),
    'hifi_peaks_mean': float(n_hifi_arr.mean()),
    'diff_mean': float(diff.mean()),
    'lofi_gte_hifi_pct': float(100 * np.sum(diff >= 0) / n_seeds),
    'seeds_with_hifi_only': int(np.sum(n_hifi_only > 0)),
    'total_hifi_only_peaks': int(n_hifi_only.sum()),
    'per_seed': [
        {'seed': r['seed'], 'n_lofi': r['n_lofi'], 'n_hifi': r['n_hifi'],
         'n_hifi_only': r['n_hifi_only'], 'hifi_only_epochs': r['hifi_only_epochs']}
        for r in results
    ],
}
with open(save_path, 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")
