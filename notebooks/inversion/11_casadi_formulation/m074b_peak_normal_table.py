#!/usr/bin/env python3
"""
m074b — Build exhaustive peak-normal alignment table.

For every peak in the 100-trajectory database, record the magnitude and the
angular distance between the PAB and each of the 10 normal groups.

Output: CSV table + NPZ checkpoint.  One row per peak, columns:
  seed, epoch, mag_hifi, prominence, best_group, min_ang_dist,
  ang_G0, ang_G1, ..., ang_G9
"""

import numpy as np
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m046_trajectories"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m074_seed27_diagnosis"
OUT_DIR.mkdir(exist_ok=True)

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)

peak_seeds = master['peak_seeds']
peak_epochs = master['peak_epochs']
peak_proms = master['peak_prominences']
mag_hifi = master['mag_hifi']
ang_dist = master['ang_dist']          # (100, 10, 500)
min_ang_dist = master['min_ang_dist']  # (100, 500)
best_group = master['best_group']      # (100, 500)
group_names = list(master['group_names'])

n_peaks = len(peak_seeds)
n_groups = len(group_names)

print(f"Building table: {n_peaks} peaks, {n_groups} normal groups")
print(f"Groups: {group_names}")

# Assemble arrays
mags = np.array([mag_hifi[s, e] for s, e in zip(peak_seeds, peak_epochs)])
proms = peak_proms
best_grps = np.array([best_group[s, e] for s, e in zip(peak_seeds, peak_epochs)])
min_angs = np.array([min_ang_dist[s, e] for s, e in zip(peak_seeds, peak_epochs)])
per_group_ang = np.array([ang_dist[s, :, e] for s, e in zip(peak_seeds, peak_epochs)])  # (N, 10)

# Save NPZ
npz_path = OUT_DIR / "peak_normal_table.npz"
np.savez_compressed(str(npz_path),
    seed=peak_seeds, epoch=peak_epochs, mag=mags, prominence=proms,
    best_group=best_grps, min_ang_dist=min_angs,
    per_group_ang=per_group_ang, group_names=group_names)
print(f"NPZ: {npz_path}")

# Save CSV
csv_path = OUT_DIR / "peak_normal_table.csv"
header = ['seed', 'epoch', 'mag_hifi', 'prominence', 'best_group', 'best_group_name',
          'min_ang_dist'] + [f'ang_{g}' for g in group_names]

with open(str(csv_path), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for i in range(n_peaks):
        row = [int(peak_seeds[i]), int(peak_epochs[i]),
               f'{mags[i]:.4f}', f'{proms[i]:.4f}',
               int(best_grps[i]), group_names[best_grps[i]],
               f'{min_angs[i]:.2f}']
        row += [f'{per_group_ang[i, g]:.2f}' for g in range(n_groups)]
        w.writerow(row)
print(f"CSV: {csv_path}")

# Quick validation: for peaks with mag < 6.0, what is the best group?
bright_mask = mags < 6.0
n_bright = bright_mask.sum()
bright_groups = best_grps[bright_mask]
bright_mags = mags[bright_mask]
bright_min_ang = min_angs[bright_mask]

print(f"\n{'='*60}")
print(f"Peaks with mag < 6.0: {n_bright}")
print(f"{'='*60}")
print(f"{'Seed':>4s} {'Ep':>4s} {'Mag':>7s} {'BestGrp':>8s} {'MinAng':>7s} "
      + ''.join(f' {g:>6s}' for g in group_names))
print('-' * (40 + 7 * n_groups))
for i in np.where(bright_mask)[0]:
    print(f"{peak_seeds[i]:4d} {peak_epochs[i]:4d} {mags[i]:7.3f} "
          f"{group_names[best_grps[i]]:>8s} {min_angs[i]:7.2f} "
          + ''.join(f' {per_group_ang[i, g]:6.2f}' for g in range(n_groups)))

# Summary: which groups appear at mag < 6?
print(f"\nGroup distribution for mag < 6.0:")
for g in range(n_groups):
    count = np.sum(bright_groups == g)
    if count > 0:
        print(f"  {group_names[g]:>5s}: {count:3d}/{n_bright} ({100*count/n_bright:.1f}%)")

# Same for mag < 5.5
very_bright_mask = mags < 5.5
n_vb = very_bright_mask.sum()
vb_groups = best_grps[very_bright_mask]
print(f"\nGroup distribution for mag < 5.5:")
for g in range(n_groups):
    count = np.sum(vb_groups == g)
    if count > 0:
        print(f"  {group_names[g]:>5s}: {count:3d}/{n_vb} ({100*count/n_vb:.1f}%)")

print(f"\nDone. {n_peaks} rows written.")
