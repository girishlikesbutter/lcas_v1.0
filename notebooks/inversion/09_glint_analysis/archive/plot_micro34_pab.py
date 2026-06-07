#!/usr/bin/env python3
"""Standalone plot for m034 PAB alignment diagnostic.

Loads pre-computed arrays from NPZ + group metadata from JSON.
Run m034_pab_alignment.py first to generate the data files.
"""

import json
import shutil
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
ASSETS_DIR = PROJECT_ROOT / "docs" / "reports" / "assets"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
npz = np.load(str(RESULTS_DIR / "m034_pab_alignment.npz"))
true_lc = npz['true_lc']
peak_indices = npz['peak_indices']
alignment = npz['alignment']
frac_flux = npz['frac_flux']
top5_by_alignment = npz['top5_by_alignment']
top5_by_frac = npz['top5_by_frac']

with open(str(RESULTS_DIR / "m034_pab_alignment.json")) as f:
    meta = json.load(f)

group_info = meta['all_groups']
n_obs = len(true_lc)
epoch_arr = np.arange(n_obs)

# Derived
dominant_frac = np.max(frac_flux, axis=0)
dominant_group_idx = np.argmax(frac_flux, axis=0)

# Colors for Panel 2 alignment traces
trace_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Colors for dominant-group markers (Panels 3-4)
unique_dom_groups = sorted(set(dominant_group_idx[peak_indices]))
cmap_tab = plt.cm.tab10
group_cmap = {g: cmap_tab(i % 10) for i, g in enumerate(unique_dom_groups)}

# ---------------------------------------------------------------------------
# Figure layout: Panels 1-3 share x (hspace=0), Panel 4 separate
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 18))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.1], hspace=0.22)

gs_top = gs[0].subgridspec(3, 1, hspace=0)
ax1 = fig.add_subplot(gs_top[0])
ax2 = fig.add_subplot(gs_top[1], sharex=ax1)
ax3 = fig.add_subplot(gs_top[2], sharex=ax1)
ax4 = fig.add_subplot(gs[1])

ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)

# ---------------------------------------------------------------------------
# Red vertical lines at peak epochs — spanning all 3 top panels
# With hspace=0, per-axes vlines visually connect into spanning lines.
# ---------------------------------------------------------------------------
for pidx in peak_indices:
    is_bright = true_lc[pidx] < 9.0
    alpha = 0.4 if is_bright else 0.15
    lw = 0.7 if is_bright else 0.3
    for ax in [ax1, ax2, ax3]:
        ax.axvline(pidx, color='red', linewidth=lw, alpha=alpha, zorder=0)

# ---------------------------------------------------------------------------
# Panel 1: Lightcurve
# ---------------------------------------------------------------------------
ax1.plot(epoch_arr, true_lc, 'k-', linewidth=0.6, alpha=0.9)
ax1.scatter(peak_indices, true_lc[peak_indices], color='red', s=20, zorder=5)
ax1.invert_yaxis()
ax1.set_ylabel('Apparent magnitude')
ax1.text(0.01, 0.93, 'Hi-fi lightcurve (lower = brighter)',
         transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')
ax1.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Panel 2: PAB alignment traces
# ---------------------------------------------------------------------------
for rank, g in enumerate(top5_by_alignment):
    info = group_info[g]
    label = f"G{g} {','.join(info['components'])} (r_s={info['mean_r_s']:.2f})"
    ax2.plot(epoch_arr, alignment[g, :], color=trace_colors[rank],
             linewidth=0.8, alpha=0.85, label=label)
ax2.set_ylabel('n . PAB alignment')
ax2.text(0.01, 0.93, 'PAB alignment for top-5 normal groups',
         transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')
ax2.legend(fontsize=7, loc='upper right', ncol=2)
ax2.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Panel 3: Dominant-group fractional flux
# ---------------------------------------------------------------------------
ax3.plot(epoch_arr, dominant_frac, 'k-', linewidth=0.4, alpha=0.4)
ax3.axhline(0.95, color='red', linewidth=0.8, linestyle='--', alpha=0.4)

for pidx in peak_indices:
    g = dominant_group_idx[pidx]
    c = group_cmap.get(g, 'gray')
    ax3.scatter(pidx, dominant_frac[pidx], color=c, s=35, zorder=5,
                edgecolors='black', linewidth=0.4)

# Legend shared with Panel 4
legend_handles = []
for g in unique_dom_groups:
    info = group_info[g]
    legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=group_cmap[g],
                                  markeredgecolor='black', markersize=7,
                                  label=f"G{g} {','.join(info['components'])}"))
legend_handles.append(Line2D([0], [0], color='red', linestyle='--',
                              alpha=0.5, label='95% dominance'))

ax3.legend(handles=legend_handles, fontsize=6, loc='lower right', ncol=2)
ax3.set_xlabel('Epoch index')
ax3.set_ylabel('Max single-group flux fraction')
ax3.text(0.01, 0.93, 'Dominant group flux fraction at detected peaks',
         transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top')
ax3.set_ylim(0, 1.08)
ax3.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Panel 4: Two-regime scatter
# ---------------------------------------------------------------------------
peak_mags = true_lc[peak_indices]
peak_dom_fracs = dominant_frac[peak_indices]
peak_dom_groups = dominant_group_idx[peak_indices]

for pidx_i, pidx in enumerate(peak_indices):
    g = peak_dom_groups[pidx_i]
    c = group_cmap.get(g, 'gray')
    ax4.scatter(peak_mags[pidx_i], peak_dom_fracs[pidx_i],
                color=c, s=60, edgecolors='black', linewidth=0.5, zorder=5)

for pidx_i, pidx in enumerate(peak_indices):
    if peak_mags[pidx_i] < 9.0:
        ax4.annotate(str(pidx), (peak_mags[pidx_i], peak_dom_fracs[pidx_i]),
                     fontsize=6, xytext=(4, 4), textcoords='offset points')

ax4.axvline(9.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax4.axhline(0.95, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
ax4.text(7.8, 0.45, 'Specular\nglints', fontsize=11, color='#d62728',
         alpha=0.6, ha='center', fontstyle='italic')
ax4.text(12.5, 0.45, 'Diffuse\npeaks', fontsize=11, color='gray',
         alpha=0.6, ha='center', fontstyle='italic')

ax4.legend(handles=legend_handles[:-1], fontsize=6, loc='lower left', ncol=2)
ax4.set_xlabel('Peak magnitude (lower = brighter)')
ax4.set_ylabel('Dominant group fractional flux')
ax4.set_title('Specular glints vs diffuse peaks — two distinct regimes')
ax4.set_ylim(0, 1.08)
ax4.grid(True, alpha=0.3)

fig.suptitle('Micro-34: PAB Alignment Diagnostic at Lightcurve Peaks', fontsize=14)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plot_path = RESULTS_DIR / "m034_pab_alignment.png"
fig.savefig(str(plot_path), dpi=150)
plt.close(fig)

shutil.copy(str(plot_path), str(ASSETS_DIR / "13_pab_alignment_diagnostic.png"))
print(f"Plot saved: {plot_path}")
print(f"Report asset updated: {ASSETS_DIR / '13_pab_alignment_diagnostic.png'}")
