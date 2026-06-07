#!/usr/bin/env python3
"""Micro-47 -- Glint classification statistics across 100 realistic trajectories.

Exhaustive analysis of PAB-normal alignment and its relationship to
observed lightcurve features. Key questions:

1. What angular distance threshold reliably predicts a brightness peak?
2. Do +X/-X alternate in driving the lightcurve? (user hypothesis)
3. Which normal groups dominate glint production?
4. How does omega magnitude affect glint statistics?
5. Can we distinguish glint sources from alignment data alone?
6. What fraction of LC peaks are specular vs diffuse?
7. Cross-correlation structure between alignment curves and LC.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import pearsonr

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m047_glint_stats"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m046_trajectories" / "m046_trajectories.npz"

# ===========================================================================
# Load data
# ===========================================================================
print("=" * 70)
print("m047 -- Glint classification statistics (100 trajectories)")
print("=" * 70)
t0 = time.time()

print("Loading m046 data...")
D = np.load(str(DATA_PATH), allow_pickle=True)

n_traj = int(D['n_trajectories'])
n_obs = int(D['n_obs'])
dt_sampling = float(D['dt_sampling'])
group_names = list(D['group_names'])
n_groups = len(group_names)
unique_normals = D['unique_normals']
group_areas = D['group_areas']

mag_hifi = D['mag_hifi']           # (T, N)
mag_lofi = D['mag_lofi']           # (T, N)
ang_dist = D['ang_dist']           # (T, G, N)
min_ang_dist = D['min_ang_dist']   # (T, N)
best_group = D['best_group']       # (T, N)
group_flux = D['group_flux']       # (T, G, N)
group_frac_flux = D['group_frac_flux']  # (T, G, N)
omega_mags = D['omega_mags']       # (T,)
observation_times = D['observation_times']
time_minutes = observation_times / 60.0

peak_seeds = D['peak_seeds']
peak_epochs = D['peak_epochs']
peak_proms = D['peak_prominences']

print(f"  {n_traj} trajectories, {n_obs} epochs, {n_groups} groups")
print(f"  omega range: [{omega_mags.min():.2f}, {omega_mags.max():.2f}] deg/s")
print(f"  {len(peak_seeds)} total peaks")


# ===========================================================================
# 1. Angular distance at LC peaks — what threshold predicts brightness?
# ===========================================================================
print("\n--- 1. Angular distance distribution at LC peaks ---")

# For each peak, get min angular distance and the responsible group
peak_min_ang = np.array([min_ang_dist[s, e] for s, e in zip(peak_seeds, peak_epochs)])
peak_best_grp = np.array([best_group[s, e] for s, e in zip(peak_seeds, peak_epochs)])

# Also get angular distance at random (non-peak) epochs for comparison
rng = np.random.RandomState(42)
rand_seeds = rng.randint(0, n_traj, size=len(peak_seeds))
rand_epochs = rng.randint(0, n_obs, size=len(peak_seeds))
rand_min_ang = np.array([min_ang_dist[s, e] for s, e in zip(rand_seeds, rand_epochs)])

print(f"  Peak min ang dist:   median={np.median(peak_min_ang):.1f}°  "
      f"mean={np.mean(peak_min_ang):.1f}°  <10°: {np.mean(peak_min_ang < 10)*100:.0f}%  "
      f"<5°: {np.mean(peak_min_ang < 5)*100:.0f}%")
print(f"  Random min ang dist: median={np.median(rand_min_ang):.1f}°  "
      f"mean={np.mean(rand_min_ang):.1f}°  <10°: {np.mean(rand_min_ang < 10)*100:.0f}%  "
      f"<5°: {np.mean(rand_min_ang < 5)*100:.0f}%")


# ===========================================================================
# 2. +X/-X alternation hypothesis
# ===========================================================================
print("\n--- 2. +X/-X alternation analysis ---")

# For each trajectory, compute sliding correlation of +X and -X alignment
# with the lightcurve, and check if they alternate
CORR_WINDOW = 21
half_w = CORR_WINDOW // 2
ix_plus_x = 0   # +X is first in group order
ix_minus_x = 1  # -X is second

alternation_scores = []
dominance_fracs = []  # fraction of time +X dominates vs -X

for t in range(n_traj):
    brightness = -mag_hifi[t]
    align_px = -ang_dist[t, ix_plus_x]
    align_mx = -ang_dist[t, ix_minus_x]

    if n_obs < CORR_WINDOW:
        continue

    # Sliding correlation for +X and -X
    from numpy.lib.stride_tricks import sliding_window_view
    b_win = sliding_window_view(brightness, CORR_WINDOW)
    px_win = sliding_window_view(align_px, CORR_WINDOW)
    mx_win = sliding_window_view(align_mx, CORR_WINDOW)

    def sliding_corr(a_win, b_win):
        n = a_win.shape[1]
        a_m = a_win.mean(axis=1)
        b_m = b_win.mean(axis=1)
        a_s = a_win.std(axis=1)
        b_s = b_win.std(axis=1)
        safe = (a_s > 1e-10) & (b_s > 1e-10)
        r = np.zeros(len(a_m))
        r[safe] = np.sum((a_win[safe] - a_m[safe, None]) *
                         (b_win[safe] - b_m[safe, None]), axis=1) / (n * a_s[safe] * b_s[safe])
        return r

    corr_px = sliding_corr(px_win, b_win)
    corr_mx = sliding_corr(mx_win, b_win)

    # Alternation: count sign changes in (corr_px - corr_mx)
    diff = corr_px - corr_mx
    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    alternation_scores.append(sign_changes)

    # Dominance: fraction of windows where +X has higher correlation
    px_dominant = np.mean(corr_px > corr_mx)
    dominance_fracs.append(px_dominant)

alternation_scores = np.array(alternation_scores)
dominance_fracs = np.array(dominance_fracs)

print(f"  +X/-X correlation sign changes: mean={alternation_scores.mean():.1f}  "
      f"std={alternation_scores.std():.1f}")
print(f"  +X dominance fraction: mean={dominance_fracs.mean():.2f}  "
      f"std={dominance_fracs.std():.2f}")
print(f"  (0.5 = perfectly alternating, 0 or 1 = one always dominates)")


# ===========================================================================
# 3. Which groups produce the most glints?
# ===========================================================================
print("\n--- 3. Glint production by normal group ---")

# For prominent peaks, which group has the smallest angular distance?
prominent_mask = peak_proms >= 0.3
prom_peak_grps = peak_best_grp[prominent_mask]
prom_peak_angs = peak_min_ang[prominent_mask]

# Also: which group has the largest fractional flux at each peak?
peak_flux_grps = np.array([
    np.argmax(group_frac_flux[s, :, e])
    for s, e in zip(peak_seeds[prominent_mask], peak_epochs[prominent_mask])
])

print(f"  {prominent_mask.sum()} prominent peaks (prominence >= 0.3 mag)")
print(f"\n  {'Group':>6s}  {'Name':>5s}  {'Area':>6s}  {'Align-based':>12s}  {'Flux-based':>11s}  {'Agreement':>10s}")
for g in range(n_groups):
    n_align = np.sum(prom_peak_grps == g)
    n_flux = np.sum(peak_flux_grps == g)
    n_agree = np.sum((prom_peak_grps == g) & (peak_flux_grps == g))
    pct_a = 100 * n_align / len(prom_peak_grps) if len(prom_peak_grps) > 0 else 0
    pct_f = 100 * n_flux / len(peak_flux_grps) if len(peak_flux_grps) > 0 else 0
    print(f"  {g:>6d}  {group_names[g]:>5s}  {group_areas[g]:>6.1f}  "
          f"{n_align:>5d} ({pct_a:>4.1f}%)  {n_flux:>5d} ({pct_f:>4.1f}%)  {n_agree:>5d}")


# ===========================================================================
# 4. Omega magnitude vs glint statistics
# ===========================================================================
print("\n--- 4. Omega magnitude effects ---")

# Bin trajectories by omega magnitude
omega_bins = [(0.1, 0.5), (0.5, 1.0), (1.0, 1.5)]
for lo, hi in omega_bins:
    mask = (omega_mags >= lo) & (omega_mags < hi)
    n_in_bin = mask.sum()
    if n_in_bin == 0:
        continue
    # Peaks in this bin
    traj_in_bin = set(np.where(mask)[0])
    bin_peak_mask = np.array([s in traj_in_bin for s in peak_seeds])
    n_peaks_bin = bin_peak_mask.sum()
    peaks_per_traj = n_peaks_bin / n_in_bin
    # Angular distances at peaks
    bin_angs = peak_min_ang[bin_peak_mask]
    pct_under_5 = np.mean(bin_angs < 5) * 100 if len(bin_angs) > 0 else 0
    pct_under_10 = np.mean(bin_angs < 10) * 100 if len(bin_angs) > 0 else 0
    print(f"  omega [{lo:.1f}, {hi:.1f}) dps: {n_in_bin:3d} traj, "
          f"{peaks_per_traj:.1f} peaks/traj, "
          f"<5°: {pct_under_5:.0f}%, <10°: {pct_under_10:.0f}%")


# ===========================================================================
# 5. Specular vs diffuse classification
# ===========================================================================
print("\n--- 5. Specular vs diffuse peak classification ---")

# Classify: if min_ang_dist < threshold at peak → specular, else diffuse
for thresh in [3, 5, 8, 10, 15]:
    n_spec = np.sum(peak_min_ang < thresh)
    n_diff = len(peak_min_ang) - n_spec
    pct = 100 * n_spec / len(peak_min_ang)
    # Prominent only
    n_spec_p = np.sum(peak_min_ang[prominent_mask] < thresh)
    pct_p = 100 * n_spec_p / prominent_mask.sum() if prominent_mask.sum() > 0 else 0
    print(f"  threshold {thresh:>2d}°: specular={n_spec:4d} ({pct:4.1f}%)  "
          f"prominent: {n_spec_p:4d} ({pct_p:4.1f}%)")


# ===========================================================================
# 6. Per-group alignment statistics at peak vs non-peak epochs
# ===========================================================================
print("\n--- 6. Per-group angular distance: peak vs non-peak ---")

# For each group, compare ang_dist at peak epochs vs random epochs
print(f"  {'Group':>5s}  {'At peaks (med)':>14s}  {'Random (med)':>12s}  {'Ratio':>6s}")
for g in range(n_groups):
    peak_ang_g = np.array([ang_dist[s, g, e] for s, e in zip(peak_seeds, peak_epochs)])
    rand_ang_g = np.array([ang_dist[s, g, e] for s, e in zip(rand_seeds, rand_epochs)])
    ratio = np.median(rand_ang_g) / np.median(peak_ang_g) if np.median(peak_ang_g) > 0 else 0
    print(f"  {group_names[g]:>5s}  {np.median(peak_ang_g):>12.1f}°  "
          f"{np.median(rand_ang_g):>10.1f}°  {ratio:>6.2f}x")


# ===========================================================================
# 7. Cross-correlation between +X and -X alignment curves
# ===========================================================================
print("\n--- 7. +X vs -X anti-correlation ---")

# Are +X and -X alignment curves anti-correlated?
px_mx_corrs = []
for t in range(n_traj):
    r, _ = pearsonr(ang_dist[t, ix_plus_x], ang_dist[t, ix_minus_x])
    px_mx_corrs.append(r)
px_mx_corrs = np.array(px_mx_corrs)
print(f"  +X vs -X Pearson r: mean={px_mx_corrs.mean():.3f}  "
      f"std={px_mx_corrs.std():.3f}  "
      f"[{px_mx_corrs.min():.3f}, {px_mx_corrs.max():.3f}]")

# Same for other opposite pairs
opposite_pairs = [(0, 1, '+X/-X'), (2, 3, '+Y/-Y'), (4, 5, '+Z/-Z'),
                  (6, 7, '+WD/-WD'), (8, 9, '+ED/-ED')]
print(f"\n  Opposite-pair correlations:")
for g1, g2, label in opposite_pairs:
    corrs = [pearsonr(ang_dist[t, g1], ang_dist[t, g2])[0] for t in range(n_traj)]
    corrs = np.array(corrs)
    print(f"    {label:>8s}: r = {corrs.mean():+.3f} ± {corrs.std():.3f}")


# ===========================================================================
# 8. Glint timing patterns — do glints cluster or spread evenly?
# ===========================================================================
print("\n--- 8. Glint timing patterns ---")

inter_glint_times = []
for t in range(n_traj):
    t_peaks = peak_epochs[peak_seeds == t]
    if len(t_peaks) > 1:
        t_peaks_sorted = np.sort(t_peaks)
        diffs = np.diff(t_peaks_sorted) * dt_sampling
        inter_glint_times.extend(diffs.tolist())

inter_glint_times = np.array(inter_glint_times)
print(f"  Inter-glint time: median={np.median(inter_glint_times):.1f}s  "
      f"mean={np.mean(inter_glint_times):.1f}s  "
      f"std={np.std(inter_glint_times):.1f}s")
print(f"  min={inter_glint_times.min():.1f}s  max={inter_glint_times.max():.1f}s")


# ===========================================================================
# 9. Dominant group switching rate (generalized alternation)
# ===========================================================================
print("\n--- 9. Dominant group switching ---")

# At each epoch, which group is closest to PAB? How often does it switch?
switch_rates = []
for t in range(n_traj):
    bg = best_group[t]
    switches = np.sum(np.diff(bg) != 0)
    switch_rates.append(switches / (n_obs - 1))

switch_rates = np.array(switch_rates)
print(f"  Group switch rate: mean={switch_rates.mean():.3f}  "
      f"std={switch_rates.std():.3f} (fraction of epochs)")
print(f"  Mean switches per trajectory: {(switch_rates * (n_obs-1)).mean():.0f}")

# Correlation between switch rate and omega magnitude
r_switch_omega, p_val = pearsonr(switch_rates, omega_mags)
print(f"  Switch rate vs |omega|: r={r_switch_omega:.3f} (p={p_val:.2e})")


# ===========================================================================
# PLOTS
# ===========================================================================
print("\n--- Generating plots ---")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# Panel 1: Angular distance distribution at peaks vs random
ax = axes[0, 0]
bins = np.linspace(0, 90, 60)
ax.hist(peak_min_ang, bins=bins, alpha=0.6, color='red', label='At LC peaks', density=True)
ax.hist(rand_min_ang, bins=bins, alpha=0.4, color='grey', label='Random epochs', density=True)
ax.set_xlabel('Min angular distance to PAB (°)')
ax.set_ylabel('Density')
ax.set_title('1. Alignment at peaks vs random')
ax.legend(fontsize=8)
ax.set_xlim(0, 90)

# Panel 2: +X/-X dominance fraction histogram
ax = axes[0, 1]
ax.hist(dominance_fracs, bins=20, color='#1f77b4', edgecolor='black', linewidth=0.5)
ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Perfect alternation')
ax.set_xlabel('+X dominance fraction')
ax.set_ylabel('Count (trajectories)')
ax.set_title('2. +X vs -X dominance')
ax.legend(fontsize=8)

# Panel 3: Glint production by group (bar chart)
ax = axes[0, 2]
align_counts = [np.sum(prom_peak_grps == g) for g in range(n_groups)]
flux_counts = [np.sum(peak_flux_grps == g) for g in range(n_groups)]
x = np.arange(n_groups)
w = 0.35
ax.bar(x - w/2, align_counts, w, label='Alignment-based', alpha=0.8)
ax.bar(x + w/2, flux_counts, w, label='Flux-based', alpha=0.8, color='orange')
ax.set_xticks(x)
ax.set_xticklabels(group_names, fontsize=8)
ax.set_ylabel('Peak count')
ax.set_title('3. Glint production by group')
ax.legend(fontsize=8)

# Panel 4: Omega magnitude vs peaks per trajectory
ax = axes[1, 0]
peaks_per_traj = np.array([np.sum(peak_seeds == t) for t in range(n_traj)])
ax.scatter(omega_mags, peaks_per_traj, s=20, alpha=0.6, edgecolors='black', linewidth=0.3)
r_om, p_om = pearsonr(omega_mags, peaks_per_traj)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Peaks per trajectory')
ax.set_title(f'4. Omega vs peak count (r={r_om:.2f})')
ax.grid(True, alpha=0.2)

# Panel 5: Specular fraction vs prominence threshold
ax = axes[1, 1]
thresholds = np.arange(1, 30)
spec_fracs_all = [np.mean(peak_min_ang < th) for th in thresholds]
spec_fracs_prom = [np.mean(peak_min_ang[prominent_mask] < th) for th in thresholds]
ax.plot(thresholds, spec_fracs_all, 'b-', label='All peaks')
ax.plot(thresholds, spec_fracs_prom, 'r-', label='Prominent (>0.3 mag)')
ax.set_xlabel('Alignment threshold (°)')
ax.set_ylabel('Fraction classified as specular')
ax.set_title('5. Specular fraction vs threshold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel 6: Opposite-pair correlations box plot
ax = axes[1, 2]
pair_data = []
pair_labels = []
for g1, g2, label in opposite_pairs:
    corrs = [pearsonr(ang_dist[t, g1], ang_dist[t, g2])[0] for t in range(n_traj)]
    pair_data.append(corrs)
    pair_labels.append(label)
bp = ax.boxplot(pair_data, labels=pair_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728']):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
ax.set_ylabel('Pearson r')
ax.set_title('6. Opposite-pair alignment correlation')

# Panel 7: Inter-glint time distribution
ax = axes[2, 0]
ax.hist(inter_glint_times, bins=50, color='#2ca02c', edgecolor='black',
        linewidth=0.3, alpha=0.7)
ax.set_xlabel('Inter-glint time (s)')
ax.set_ylabel('Count')
ax.set_title(f'7. Inter-glint timing (median={np.median(inter_glint_times):.0f}s)')

# Panel 8: Group switch rate vs omega
ax = axes[2, 1]
ax.scatter(omega_mags, switch_rates * (n_obs - 1), s=20, alpha=0.6,
           edgecolors='black', linewidth=0.3)
ax.set_xlabel('|omega| (deg/s)')
ax.set_ylabel('Group switches per trajectory')
ax.set_title(f'8. Switch rate vs omega (r={r_switch_omega:.2f})')
ax.grid(True, alpha=0.2)

# Panel 9: +X/-X alternation example (one trajectory)
ax = axes[2, 2]
# Pick a trajectory near median omega
med_idx = np.argsort(omega_mags)[n_traj // 2]
corr_px_ex = sliding_corr(
    sliding_window_view(-ang_dist[med_idx, ix_plus_x], CORR_WINDOW),
    sliding_window_view(-mag_hifi[med_idx], CORR_WINDOW))
corr_mx_ex = sliding_corr(
    sliding_window_view(-ang_dist[med_idx, ix_minus_x], CORR_WINDOW),
    sliding_window_view(-mag_hifi[med_idx], CORR_WINDOW))
t_corr = time_minutes[half_w:half_w + len(corr_px_ex)]
ax.plot(t_corr, corr_px_ex, color='#1f77b4', linewidth=1, label='+X', alpha=0.8)
ax.plot(t_corr, corr_mx_ex, color='#ff7f0e', linewidth=1, label='-X', alpha=0.8)
ax.fill_between(t_corr, corr_px_ex, corr_mx_ex, alpha=0.1, color='grey')
ax.axhline(0, color='grey', linewidth=0.5)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Correlation with LC')
ax.set_title(f'9. +X/-X alternation (seed {med_idx}, |ω|={omega_mags[med_idx]:.2f}°/s)')
ax.legend(fontsize=8)
ax.set_ylim(-1.05, 1.05)

fig.suptitle('Micro-47: Glint Classification Statistics — 100 Trajectories\n'
             f'omega range [{omega_mags.min():.2f}, {omega_mags.max():.2f}] deg/s  •  '
             f'{len(peak_seeds)} total peaks',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

plot_path = RESULTS_DIR / "m047_glint_statistics.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Plot saved: {plot_path}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")
