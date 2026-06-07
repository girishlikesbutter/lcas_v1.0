#!/usr/bin/env python3
"""
m096 Exp 3: Normal Identification from Dim Peaks.

Question: At truth, can we identify which body normal produced each peak —
including the 1277 currently-discarded dim peaks (mag > 9)?

For every peak, propagate truth to that epoch, compute alignment with
all 10 normals, cross-reference with calibration table.
"""

import sys, os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1 = RESULTS_DIR / "m096_stage1"
CKPT = RESULTS_DIR / "m096_exp3_normal_id"
CKPT.mkdir(exist_ok=True)

# Load shared data
master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                 allow_pickle=True)
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
pab_j2000 = master['pab_j2000']
group_names = list(master['group_names'])
n_normals = len(unique_normals)

# Load calibration table (trajectory-independent)
calib = np.load(str(RESULTS_DIR / "m093_seed093" / "calibration.npz"))
mag_table = calib['mag_table']  # (n_normals, n_angles)
calib_dots = np.cos(np.deg2rad(calib['calib_angles_deg']))

ALL_SEEDS = list(range(100))
summary = []

for seed in ALL_SEEDS:
    d = np.load(str(STAGE1 / f"seed_{seed:03d}.npz"), allow_pickle=True)

    if not bool(d['valid']):
        # Still process invalid seeds — they have peaks, just < 2 specular
        pass

    true_q0 = d['true_q0']
    true_omega0 = d['true_omega0']
    obs_times = d['obs_times']
    peaks_idx = d['peaks_idx']
    peak_mags = d['peak_mags']
    observed_lc = d['observed_lc']
    n_peaks = len(peaks_idx)

    if n_peaks == 0:
        np.savez(str(CKPT / f"seed_{seed:03d}.npz"),
                 seed=seed, n_peaks=0, valid=False)
        summary.append({'seed': seed, 'n_peaks': 0, 'error': 'no_peaks'})
        continue

    # Propagate truth to all peak epochs
    peak_times = obs_times[peaks_idx]
    all_times = np.concatenate([[0.0], np.sort(peak_times)])
    quats, _ = propagate_attitude(true_q0, true_omega0, all_times, "tumbling", I_tensor)

    # Map sorted times back to peak order
    sort_order = np.argsort(peak_times)
    unsort_order = np.argsort(sort_order)
    peak_quats = quats[1:][unsort_order]  # skip t=0

    # For each peak: compute body-frame PAB, dot with all normals
    alignments = np.zeros((n_peaks, n_normals))  # dot(normal, PAB_body)
    best_normal = np.zeros(n_peaks, dtype=int)
    best_alignment = np.zeros(n_peaks)
    second_best_normal = np.zeros(n_peaks, dtype=int)
    second_best_alignment = np.zeros(n_peaks)

    for pi in range(n_peaks):
        q = peak_quats[pi]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        pab_body = R @ pab_j2000[peaks_idx[pi]]  # J2000 PAB → body frame

        for ni in range(n_normals):
            alignments[pi, ni] = np.dot(unique_normals[ni], pab_body)

        # Rank normals by alignment
        ranked = np.argsort(-alignments[pi])  # descending
        best_normal[pi] = ranked[0]
        best_alignment[pi] = alignments[pi, ranked[0]]
        second_best_normal[pi] = ranked[1]
        second_best_alignment[pi] = alignments[pi, ranked[1]]

    # Expected magnitude from calibration table for best normal
    expected_mag_best = np.zeros(n_peaks)
    expected_mag_second = np.zeros(n_peaks)
    for pi in range(n_peaks):
        dot_val = np.clip(best_alignment[pi], 0, 1)
        expected_mag_best[pi] = np.interp(dot_val, calib_dots[::-1], mag_table[best_normal[pi]][::-1])
        dot_val2 = np.clip(second_best_alignment[pi], 0, 1)
        expected_mag_second[pi] = np.interp(dot_val2, calib_dots[::-1], mag_table[second_best_normal[pi]][::-1])

    # Brightness consistency: is observed mag within ±1.5 of expected?
    obs_mags = observed_lc[peaks_idx]
    consistent_best = np.abs(obs_mags - expected_mag_best) < 1.5
    consistent_second = np.abs(obs_mags - expected_mag_second) < 1.5

    # Gap: how much better is best vs second-best alignment?
    alignment_gap = best_alignment - second_best_alignment

    # Save everything
    np.savez(str(CKPT / f"seed_{seed:03d}.npz"),
             seed=seed,
             valid=True,
             n_peaks=n_peaks,
             peaks_idx=peaks_idx,
             peak_mags=peak_mags,
             obs_mags=obs_mags,
             alignments=alignments,
             best_normal=best_normal,
             best_alignment=best_alignment,
             second_best_normal=second_best_normal,
             second_best_alignment=second_best_alignment,
             alignment_gap=alignment_gap,
             expected_mag_best=expected_mag_best,
             expected_mag_second=expected_mag_second,
             consistent_best=consistent_best,
             consistent_second=consistent_second,
             peak_quats=peak_quats,
    )

    # Summary stats
    n_spec = int(np.sum(obs_mags < 9.0))
    n_dim = int(np.sum(obs_mags >= 9.0))
    n_dim_identifiable = int(np.sum((obs_mags >= 9.0) & (alignment_gap > 0.1)))
    n_dim_consistent = int(np.sum((obs_mags >= 9.0) & consistent_best))
    n_dim_usable = int(np.sum((obs_mags >= 9.0) & (alignment_gap > 0.1) & consistent_best))

    summary.append({
        'seed': seed,
        'n_peaks': n_peaks,
        'n_spec': n_spec,
        'n_dim': n_dim,
        'n_dim_identifiable': n_dim_identifiable,
        'n_dim_consistent': n_dim_consistent,
        'n_dim_usable': n_dim_usable,
        'median_gap_spec': float(np.median(alignment_gap[obs_mags < 9.0])) if n_spec > 0 else 0,
        'median_gap_dim': float(np.median(alignment_gap[obs_mags >= 9.0])) if n_dim > 0 else 0,
    })

save_results(str(CKPT / "summary.json"), summary)

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP 3: NORMAL IDENTIFICATION FROM DIM PEAKS — 100 seeds")
print("=" * 70)

valid_s = [r for r in summary if 'error' not in r]

total_dim = sum(r['n_dim'] for r in valid_s)
total_dim_id = sum(r['n_dim_identifiable'] for r in valid_s)
total_dim_con = sum(r['n_dim_consistent'] for r in valid_s)
total_dim_usable = sum(r['n_dim_usable'] for r in valid_s)
total_spec = sum(r['n_spec'] for r in valid_s)

print(f"\n  Specular peaks (mag < 9): {total_spec} total across {len(valid_s)} seeds")
print(f"  Dim peaks (mag ≥ 9):      {total_dim} total")
print(f"    Identifiable (gap > 0.1):  {total_dim_id} ({total_dim_id/total_dim*100:.1f}%)")
print(f"    Brightness-consistent:     {total_dim_con} ({total_dim_con/total_dim*100:.1f}%)")
print(f"    Usable (both):             {total_dim_usable} ({total_dim_usable/total_dim*100:.1f}%)")

# Per magnitude band
print(f"\n  By magnitude band (at truth):")
all_gaps = []
all_consist = []
all_mags = []
for seed in range(100):
    try:
        dd = np.load(str(CKPT / f"seed_{seed:03d}.npz"), allow_pickle=True)
        if not bool(dd['valid']): continue
        all_gaps.extend(dd['alignment_gap'].tolist())
        all_consist.extend(dd['consistent_best'].tolist())
        all_mags.extend(dd['obs_mags'].tolist())
    except:
        continue

all_gaps = np.array(all_gaps)
all_consist = np.array(all_consist)
all_mags = np.array(all_mags)

bands = [(4, 6, 'bright'), (6, 7.3, 'medium'), (7.3, 9, 'dim-spec'),
         (9, 10, '9-10'), (10, 11, '10-11'), (11, 12, '11-12'), (12, 14, '12-14')]

print(f"  {'Band':>10} {'Count':>6} {'Gap>0.1':>8} {'Consist':>8} {'Usable':>8} {'Med gap':>8}")
for lo, hi, label in bands:
    mask = (all_mags >= lo) & (all_mags < hi)
    n = mask.sum()
    if n == 0: continue
    n_id = (mask & (all_gaps > 0.1)).sum()
    n_con = (mask & all_consist).sum()
    n_use = (mask & (all_gaps > 0.1) & all_consist).sum()
    mg = np.median(all_gaps[mask])
    print(f"  {label:>10} {n:6d} {n_id:6d} ({n_id/n*100:4.0f}%) "
          f"{n_con:5d} ({n_con/n*100:4.0f}%) "
          f"{n_use:5d} ({n_use/n*100:4.0f}%) {mg:8.3f}")

# How many extra usable constraints would each seed gain?
print(f"\n  Extra usable constraints per seed (from dim peaks):")
gains = [r['n_dim_usable'] for r in valid_s]
print(f"    min={min(gains)}, median={int(np.median(gains))}, "
      f"mean={np.mean(gains):.1f}, max={max(gains)}")

print(f"\nSaved to {CKPT}/")
