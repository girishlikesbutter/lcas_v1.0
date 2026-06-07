#!/usr/bin/env python3
"""Micro-30 — Hi-fi pruning of lo-fi iso-brightness candidates at peaks.

Generate iso-brightness attitude candidates via lo-fi L-BFGS-B optimization
at each of 3 peaks. Then evaluate each candidate at hi-fi (single-epoch with
ray-traced shadows). Measure how many lo-fi candidates survive hi-fi pruning
at various tolerances, and whether the nearest-to-truth candidate is preserved.
"""

import sys
import os
import time
import json
import numpy as np
import multiprocessing as mp
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.experiment_setup import (setup_experiment, save_results,
                                  brightness_single_epoch, attitude_error_deg)
from src.dynamics.attitude_propagator import propagate_attitude

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS = 1000
N_WORKERS = 8
LOFI_CONVERGENCE_THRESHOLD = 0.01   # mag
CLUSTER_THRESHOLD_DEG = 1.0
THRESHOLDS = [0.2, 0.1, 0.05, 0.025, 0.01]

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("=" * 60)
print("m030 — Hi-fi pruning of lo-fi iso-brightness candidates")
print("=" * 60)

t_global = time.time()

CTX = setup_experiment(
    n_observations=500,
    noise_sigma=0.05,
    random_seed=42,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)
I = CTX.inertia_tensor
OBS_T = CTX.observation_times

# Load peak indices
PEAKS = [int(x) for x in np.load(RESULTS_DIR / "m013_stage1.npz")["peaks"]]
print(f"Peak epochs: {PEAKS}")

# Propagate true attitude to each peak
times_for_peaks = np.array([0.0] + [float(OBS_T[p]) for p in PEAKS])
q_traj, _ = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_for_peaks,
                                "tumbling", I)
truth_q_at_peak = {PEAKS[i]: q_traj[i + 1] for i in range(len(PEAKS))}

for pidx in PEAKS:
    m_true = brightness_single_epoch(truth_q_at_peak[pidx], pidx, CTX, use_shadows=True)
    print(f"  Peak {pidx}: truth q = {truth_q_at_peak[pidx]}, "
          f"hi-fi mag = {m_true:.4f}, observed = {CTX.observed_lc[pidx]:.4f}")


# ---------------------------------------------------------------------------
# Lo-fi iso-brightness optimization
# ---------------------------------------------------------------------------
def iso_objective(rotvec, target_mag, eidx):
    """Minimize squared brightness error at one epoch (lo-fi)."""
    R = Rotation.from_rotvec(rotvec)
    q_xyzw = R.as_quat()  # scipy convention: (x, y, z, w)
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    mag = brightness_single_epoch(q_wxyz, eidx, CTX, use_shadows=False)
    return (mag - target_mag) ** 2


def run_one_seed(args):
    """Run one L-BFGS-B from a random SO(3) seed."""
    seed_idx, rotvec0, target_mag, eidx = args
    try:
        res = minimize(iso_objective, rotvec0, args=(target_mag, eidx),
                       method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
        R = Rotation.from_rotvec(res.x)
        q_xyzw = R.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        final_mag = brightness_single_epoch(q_wxyz, eidx, CTX, use_shadows=False)
        return {
            'q_wxyz': q_wxyz.tolist(),
            'final_mag': float(final_mag),
            'residual': float(abs(final_mag - target_mag)),
            'nfev': res.nfev,
        }
    except Exception as e:
        return {'error': str(e)}


# ---------------------------------------------------------------------------
# Clustering (quaternion geodesic distance)
# ---------------------------------------------------------------------------
def cluster_quaternions(q_array, threshold_deg=1.0):
    """Cluster quaternions within threshold_deg geodesic distance.
    Returns cluster labels (1-indexed) from hierarchical clustering."""
    n = len(q_array)
    if n <= 1:
        return np.ones(n, dtype=int)

    # Compute pairwise geodesic distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = attitude_error_deg(q_array[i], q_array[j])
            dists[i, j] = d
            dists[j, i] = d

    condensed = squareform(dists)
    Z = linkage(condensed, method='complete')
    labels = fcluster(Z, t=threshold_deg, criterion='distance')
    return labels


# ---------------------------------------------------------------------------
# Hi-fi evaluation
# ---------------------------------------------------------------------------
def hifi_eval(args):
    """Evaluate hi-fi brightness at one epoch for one quaternion."""
    q_wxyz, eidx = args
    return float(brightness_single_epoch(np.array(q_wxyz), eidx, CTX,
                                         use_shadows=True))


# ---------------------------------------------------------------------------
# Main loop over peaks
# ---------------------------------------------------------------------------
all_peak_results = []

for peak_idx in PEAKS:
    print(f"\n{'='*60}")
    print(f"Peak epoch {peak_idx}")
    print(f"{'='*60}")

    m_target = float(CTX.observed_lc[peak_idx])
    truth_q = truth_q_at_peak[peak_idx]
    print(f"  Target magnitude (observed hi-fi): {m_target:.4f}")

    # --- Step 1: Generate random SO(3) seeds ---
    rng = np.random.default_rng(seed=42 + peak_idx)
    random_rotations = Rotation.random(N_SEEDS, random_state=rng)
    seed_rotvecs = random_rotations.as_rotvec()

    tasks = [(i, seed_rotvecs[i], m_target, peak_idx) for i in range(N_SEEDS)]

    # --- Step 2: Lo-fi iso-brightness optimization (parallel) ---
    t_lofi = time.time()
    pool = mp.get_context('fork').Pool(N_WORKERS)
    raw_results = pool.map(run_one_seed, tasks, chunksize=16)
    pool.close()
    pool.join()
    dt_lofi = time.time() - t_lofi
    print(f"  Lo-fi optimization: {dt_lofi:.1f}s for {N_SEEDS} seeds")

    # --- Step 3: Filter converged candidates ---
    converged = [r for r in raw_results
                 if 'error' not in r and r['residual'] < LOFI_CONVERGENCE_THRESHOLD]
    n_errors = sum(1 for r in raw_results if 'error' in r)
    print(f"  Converged (|resid| < {LOFI_CONVERGENCE_THRESHOLD}): "
          f"{len(converged)} / {N_SEEDS}  (errors: {n_errors})")

    if len(converged) == 0:
        print("  WARNING: No converged candidates — skipping this peak")
        all_peak_results.append({
            'peak_idx': peak_idx,
            'm_target': m_target,
            'n_seeds': N_SEEDS,
            'n_converged': 0,
            'n_unique': 0,
            'nearest_to_truth_deg': None,
            'nearest_to_truth_hifi_residual': None,
            'survival_counts': {str(t): 0 for t in THRESHOLDS},
            'nearest_survives': {str(t): False for t in THRESHOLDS},
        })
        continue

    # --- Step 4: Cluster and deduplicate ---
    q_converged = np.array([r['q_wxyz'] for r in converged])
    residuals = np.array([r['residual'] for r in converged])

    labels = cluster_quaternions(q_converged, threshold_deg=CLUSTER_THRESHOLD_DEG)
    n_clusters = labels.max()

    # Keep the candidate with the smallest residual in each cluster
    unique_indices = []
    for cl in range(1, n_clusters + 1):
        members = np.where(labels == cl)[0]
        best_in_cluster = members[np.argmin(residuals[members])]
        unique_indices.append(best_in_cluster)

    unique_qs = q_converged[unique_indices]
    unique_residuals = residuals[unique_indices]
    print(f"  Clusters ({CLUSTER_THRESHOLD_DEG} deg): {n_clusters} unique candidates")

    # --- Step 5: Attitude error to truth for each unique candidate ---
    att_errors = np.array([attitude_error_deg(unique_qs[i], truth_q)
                           for i in range(len(unique_qs))])
    nearest_idx = np.argmin(att_errors)
    nearest_deg = float(att_errors[nearest_idx])
    print(f"  Nearest-to-truth: {nearest_deg:.2f} deg "
          f"(lo-fi resid = {unique_residuals[nearest_idx]:.6f})")

    # --- Step 6: Hi-fi evaluation (parallel) ---
    hifi_tasks = [(unique_qs[i].tolist(), peak_idx)
                  for i in range(len(unique_qs))]

    t_hifi = time.time()
    pool = mp.get_context('fork').Pool(N_WORKERS)
    hifi_mags = pool.map(hifi_eval, hifi_tasks, chunksize=4)
    pool.close()
    pool.join()
    dt_hifi = time.time() - t_hifi
    hifi_mags = np.array(hifi_mags)
    print(f"  Hi-fi evaluation: {dt_hifi:.1f}s for {len(unique_qs)} candidates")

    # --- Step 7: Compute hi-fi residuals and survival ---
    hifi_residuals = np.abs(hifi_mags - m_target)

    nearest_hifi_resid = float(hifi_residuals[nearest_idx])
    print(f"  Nearest-to-truth hi-fi residual: {nearest_hifi_resid:.4f} mag")

    survival_counts = {}
    nearest_survives = {}
    for thr in THRESHOLDS:
        survivors = int(np.sum(hifi_residuals < thr))
        survival_counts[str(thr)] = survivors
        nearest_survives[str(thr)] = bool(hifi_residuals[nearest_idx] < thr)
        pct = 100.0 * survivors / len(unique_qs) if len(unique_qs) > 0 else 0
        flag = " <-- nearest survives" if nearest_survives[str(thr)] else ""
        print(f"    |Δm| < {thr:.3f}: {survivors:4d} / {len(unique_qs)} "
              f"({pct:5.1f}%){flag}")

    # Store per-peak results
    peak_result = {
        'peak_idx': peak_idx,
        'm_target': m_target,
        'n_seeds': N_SEEDS,
        'n_converged': len(converged),
        'n_unique': len(unique_qs),
        'nearest_to_truth_deg': nearest_deg,
        'nearest_to_truth_hifi_residual': nearest_hifi_resid,
        'survival_counts': survival_counts,
        'nearest_survives': nearest_survives,
        'lofi_time_s': round(dt_lofi, 1),
        'hifi_time_s': round(dt_hifi, 1),
        # Arrays for plotting (not saved to JSON)
        '_att_errors': att_errors,
        '_hifi_residuals': hifi_residuals,
        '_nearest_idx': nearest_idx,
    }
    all_peak_results.append(peak_result)


# ---------------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------------
runtime = time.time() - t_global

print(f"\n{'='*60}")
print(f"OVERALL SUMMARY")
print(f"{'='*60}")
print(f"Total runtime: {runtime:.1f}s")
for pr in all_peak_results:
    if pr['n_unique'] == 0:
        print(f"  Peak {pr['peak_idx']}: no candidates")
        continue
    print(f"  Peak {pr['peak_idx']}: {pr['n_unique']} unique, "
          f"nearest={pr['nearest_to_truth_deg']:.2f} deg, "
          f"hifi_resid={pr['nearest_to_truth_hifi_residual']:.4f}")
    for thr in THRESHOLDS:
        sc = pr['survival_counts'][str(thr)]
        ns = pr['nearest_survives'][str(thr)]
        print(f"    |Δm|<{thr:.3f}: {sc:4d} survive, nearest={'YES' if ns else 'NO'}")


# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------
json_results = {
    'peaks': [],
    'thresholds': THRESHOLDS,
    'runtime_s': round(runtime, 1),
}
for pr in all_peak_results:
    entry = {k: v for k, v in pr.items() if not k.startswith('_')}
    json_results['peaks'].append(entry)

out_json = RESULTS_DIR / "m030_hifi_pruning.json"
save_results(out_json, json_results)
print(f"\nResults saved to {out_json}")


# ---------------------------------------------------------------------------
# Plot: 3 panels, one per peak
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax_idx, pr in enumerate(all_peak_results):
    ax = axes[ax_idx]
    pidx = pr['peak_idx']

    if pr['n_unique'] == 0:
        ax.set_title(f"Peak {pidx}: no candidates")
        continue

    att_errors = pr['_att_errors']
    hifi_residuals = pr['_hifi_residuals']
    nearest_idx = pr['_nearest_idx']

    # Scatter all candidates
    ax.scatter(att_errors, hifi_residuals, s=15, alpha=0.5, color='steelblue',
               label='candidates')

    # Highlight nearest-to-truth
    ax.scatter(att_errors[nearest_idx], hifi_residuals[nearest_idx],
               s=200, marker='*', color='red', zorder=5,
               label=f'nearest ({att_errors[nearest_idx]:.1f}°)')

    # Threshold lines
    colors_thr = ['#cccccc', '#999999', '#666666', '#333333', '#000000']
    for i, thr in enumerate(THRESHOLDS):
        sc = pr['survival_counts'][str(thr)]
        ax.axhline(thr, color=colors_thr[i], linestyle='--', linewidth=0.8,
                    alpha=0.7)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 180, thr,
                f'  {thr} ({sc})', fontsize=7, va='bottom')

    # Labels
    n_survive_005 = pr['survival_counts']['0.05']
    ax.set_title(f"Peak {pidx}: {pr['n_unique']} unique, "
                 f"{n_survive_005} survive @0.05",
                 fontsize=10)
    ax.set_xlabel("Attitude error to truth (deg)")
    ax.set_ylabel("|m_hifi − m_target| (mag)")
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle("Micro-30: Hi-fi pruning of lo-fi iso-brightness candidates",
             fontsize=13, fontweight='bold')
plt.tight_layout()

out_png = RESULTS_DIR / "m030_hifi_pruning.png"
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved to {out_png}")

print("\nDone.")
