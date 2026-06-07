#!/usr/bin/env python3
"""
m130_replot -- Regenerate m130 plots at publication quality from cached arrays.

Does NOT re-run any hi-fi or surrogate computation. Reads:
  data/results/inversion_diagnostics/m130_v1v2_plots/
      m130_seed019_arrays.npz    (1000-epoch single-seed comparison)
      m130_seed019_meta.json
      m130b_population_arrays.npz (500-epoch × 100 seeds)
      m130b_population_summary.json

and rewrites:
  m130_seed019_v1.png / m130_seed019_v2.png    (300 dpi, larger fonts)
  m130_seed019_both.png                        (stacked v1+v2 with shared LC)
  m130b_population_plot.png                    (300 dpi, reworked 4-panel)
  m130b_residual_vs_brightness.png             (bonus: residual vs truth mag)
"""

import os, json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = Path('/home/girish/projects/lcas_v1.0/data/results/inversion_diagnostics/m130_v1v2_plots')

# Typography / styling.
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.axisbelow': True,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.linewidth': 0.9,
    'lines.linewidth': 1.2,
})

C_TRUTH = '#111111'
C_V1 = '#d62728'    # red
C_V2 = '#1f77b4'    # blue


# ── 1. Single chaotic seed (seed 19, 1000 epochs) ──────────────────────

arr = np.load(str(OUT_DIR / 'm130_seed019_arrays.npz'), allow_pickle=True)
with open(OUT_DIR / 'm130_seed019_meta.json') as f:
    meta = json.load(f)

SEED = int(arr['seed'])
N = int(arr['n_epochs'])
t = arr['obs_times']          # seconds since epoch start
mag_hifi = arr['mag_hifi']
mag_v1 = arr['mag_v1']
mag_v2 = arr['mag_v2']
w_mag_dps = float(np.linalg.norm(np.rad2deg(arr['w0_true'])))

res_v1 = mag_v1 - mag_hifi
res_v2 = mag_v2 - mag_hifi


# Shared residual y-limit — same scale for both v1 and v2 plots so tail
# differences are visually honest.
_SHARED_RES_YLIM = max(meta['v1']['max'], meta['v2']['max']) * 1.05


def _make_single_plot(pred, residual, version, color, outpath):
    """Single-version plot: 1000-epoch LC with residual subplot."""
    mets = meta[version]
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_lc = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_lc)

    ax_lc.plot(t, mag_hifi, color=C_TRUTH, lw=1.3, label='hi-fi truth (ray-traced shadows + BRDF, 1000 epochs)')
    ax_lc.plot(t, pred, color=color, lw=1.1, alpha=0.88, label=f'surrogate {version}')
    ax_lc.invert_yaxis()
    ax_lc.set_ylabel('apparent magnitude', labelpad=8)
    ax_lc.legend(loc='upper right', frameon=True, framealpha=0.92, fancybox=False,
                 edgecolor='#999999')
    ax_lc.tick_params(labelbottom=False)

    ax_lc.set_title(
        f'Seed {SEED}   |   |ω| = {w_mag_dps:.2f} deg/s   |   '
        f'{version} MAE = {mets["mae"]:.4f} mag   '
        f'(bright={mets["bright_mae"]:.4f},  p90={mets["p90"]:.4f},  '
        f'max={mets["max"]:.3f})',
        loc='left', pad=10)

    ax_res.axhline(0, color=C_TRUTH, lw=0.7, alpha=0.6)
    ax_res.plot(t, residual, color=color, lw=0.8)
    ax_res.fill_between(t, residual, 0, where=(residual > 0), color=color, alpha=0.18)
    ax_res.fill_between(t, residual, 0, where=(residual < 0), color=color, alpha=0.18)
    ax_res.set_ylabel(f'{version} − truth\n(mag)', labelpad=8)
    ax_res.set_xlabel('time since epoch start (s)', labelpad=6)
    ax_res.set_ylim(-_SHARED_RES_YLIM, _SHARED_RES_YLIM)

    for ax in (ax_lc, ax_res):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outpath}')

_make_single_plot(mag_v1, res_v1, 'v1', C_V1, OUT_DIR / 'm130_seed019_v1.png')
_make_single_plot(mag_v2, res_v2, 'v2', C_V2, OUT_DIR / 'm130_seed019_v2.png')


# ── 1b. Combined stacked panel (LC shared, both residuals) ───────────

fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)
ax_lc = fig.add_subplot(gs[0])
ax_r1 = fig.add_subplot(gs[1], sharex=ax_lc)
ax_r2 = fig.add_subplot(gs[2], sharex=ax_lc)

ax_lc.plot(t, mag_hifi, color=C_TRUTH, lw=1.3, label='hi-fi truth (1000 epochs)')
ax_lc.plot(t, mag_v1, color=C_V1, lw=1.0, alpha=0.82, label=f'v1  (MAE={meta["v1"]["mae"]:.4f})')
ax_lc.plot(t, mag_v2, color=C_V2, lw=1.0, alpha=0.82, label=f'v2  (MAE={meta["v2"]["mae"]:.4f})')
ax_lc.invert_yaxis()
ax_lc.set_ylabel('apparent magnitude')
ax_lc.legend(loc='upper right', frameon=True, framealpha=0.92, fancybox=False,
             edgecolor='#999999')
ax_lc.tick_params(labelbottom=False)
ax_lc.set_title(
    f'Seed {SEED} — chaotic m048 trajectory  |   |ω|={w_mag_dps:.2f} deg/s,   '
    f'{int(arr["n_epochs"])} epochs   |   '
    f'v2 improves overall MAE by {meta["v1"]["mae"]/meta["v2"]["mae"]:.1f}×, '
    f'bright MAE by {meta["v1"]["bright_mae"]/meta["v2"]["bright_mae"]:.1f}×, '
    f'max error by {meta["v1"]["max"]/meta["v2"]["max"]:.1f}×',
    loc='left', pad=10)

_ylim = max(meta['v1']['p99'] * 1.6, meta['v2']['p99'] * 1.6, 0.04)

for ax, res, color, version, mets in [
    (ax_r1, res_v1, C_V1, 'v1', meta['v1']),
    (ax_r2, res_v2, C_V2, 'v2', meta['v2']),
]:
    ax.axhline(0, color=C_TRUTH, lw=0.7, alpha=0.6)
    ax.plot(t, res, color=color, lw=0.7)
    ax.fill_between(t, res, 0, where=(res > 0), color=color, alpha=0.18)
    ax.fill_between(t, res, 0, where=(res < 0), color=color, alpha=0.18)
    ax.set_ylim(-_ylim, _ylim)
    ax.set_ylabel(f'{version} − truth\n(mag)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax_r1.tick_params(labelbottom=False)
ax_r2.set_xlabel('time since epoch start (s)')
ax_lc.spines['top'].set_visible(False); ax_lc.spines['right'].set_visible(False)

fig.savefig(OUT_DIR / 'm130_seed019_both.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_DIR / "m130_seed019_both.png"}')


# ── 2. Population 4-panel (from cached 500-epoch × 100 seeds) ─────────

pop = np.load(str(OUT_DIR / 'm130b_population_arrays.npz'), allow_pickle=True)
with open(OUT_DIR / 'm130b_population_summary.json') as f:
    popsum = json.load(f)
pooled = popsum['pooled']

seeds = pop['seeds']
w_dps = pop['w_dps']
v1_mae = pop['v1_mae']; v2_mae = pop['v2_mae']
v1_bright_mae = pop['v1_bright_mae']; v2_bright_mae = pop['v2_bright_mae']
v1_max = pop['v1_max']; v2_max = pop['v2_max']
pool_v1 = pop['pool_abs_err_v1'].astype(float)
pool_v2 = pop['pool_abs_err_v2'].astype(float)
pool_truth = pop['pool_truth_mag'].astype(float)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
ax1, ax2, ax3, ax4 = axes.flatten()

# (1) Per-seed MAE scatter: v1 vs v2 (log-log, shows that v2 beats v1 on EVERY seed)
mx = max(v1_mae.max(), v2_mae.max()) * 1.3
mn = min(v1_mae.min(), v2_mae.min()) * 0.7
diag = np.array([mn, mx])
ax1.plot(diag, diag, color=C_TRUTH, lw=0.9, ls='--', alpha=0.55, label='v1 = v2 (diagonal)')
ax1.plot(diag, diag / 3, color=C_TRUTH, lw=0.6, ls=':', alpha=0.35, label='3×, 10× contours')
ax1.plot(diag, diag / 10, color=C_TRUTH, lw=0.6, ls=':', alpha=0.35)
sc = ax1.scatter(v2_mae, v1_mae, c=w_dps, s=45, cmap='viridis', alpha=0.85,
                 edgecolors='white', linewidths=0.6)
cbar = plt.colorbar(sc, ax=ax1, pad=0.02)
cbar.set_label('|ω| (deg/s)', fontsize=11)
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlim(mn, mx); ax1.set_ylim(mn, mx)
ax1.set_xlabel('v2 per-seed MAE (mag)')
ax1.set_ylabel('v1 per-seed MAE (mag)')
ax1.set_title('Per-seed MAE  —  every one of 100 seeds sits above the diagonal',
              loc='left', pad=8)
ax1.legend(loc='lower right', frameon=True, framealpha=0.92, edgecolor='#999999')

# (2) Improvement ratio histogram
ratio = v1_mae / v2_mae
ax2.hist(ratio, bins=30, color='#62a055', edgecolor='#304527', alpha=0.85, lw=0.8)
ax2.axvline(1.0, color=C_TRUTH, lw=1, ls='--', alpha=0.6, label='v1 = v2')
ax2.axvline(float(ratio.mean()), color=C_V1, lw=2, label=f'mean {ratio.mean():.1f}×')
ax2.axvline(float(np.median(ratio)), color=C_V2, lw=2, label=f'median {np.median(ratio):.1f}×')
ax2.set_xlabel('v1 MAE  /  v2 MAE')
ax2.set_ylabel('# seeds')
ax2.set_title('Per-seed improvement ratio distribution', loc='left', pad=8)
ax2.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#999999')

# (3) Pooled |residual| histogram on log-x
bins = np.logspace(-4, 1, 80)
ax3.hist(pool_v1, bins=bins, alpha=0.55, color=C_V1, label=f'v1 (MAE {pooled["v1_mae"]:.4f})',
         density=True, edgecolor='#aa1020', lw=0.4)
ax3.hist(pool_v2, bins=bins, alpha=0.55, color=C_V2, label=f'v2 (MAE {pooled["v2_mae"]:.4f})',
         density=True, edgecolor='#0e4c72', lw=0.4)
ax3.axvline(pooled['v1_mae'], color=C_V1, lw=1.2, ls=':')
ax3.axvline(pooled['v2_mae'], color=C_V2, lw=1.2, ls=':')
ax3.set_xscale('log')
ax3.set_xlabel('|predicted − truth| (mag)')
ax3.set_ylabel('density')
ax3.set_title(f'Pooled per-epoch residual  '
              f'({pooled["n_epochs_total"]:,} points across 100 seeds)',
              loc='left', pad=8)
ax3.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#999999')

# (4) Per-seed max error vs chaotic-ness
ax4.scatter(w_dps, v1_max, c=C_V1, s=45, alpha=0.7, edgecolors='white', linewidths=0.6, label='v1')
ax4.scatter(w_dps, v2_max, c=C_V2, s=45, alpha=0.7, edgecolors='white', linewidths=0.6, label='v2')
ax4.set_yscale('log')
ax4.set_xlabel('|ω| (deg/s)')
ax4.set_ylabel('per-seed worst-epoch |error| (mag)')
ax4.set_title('Tail error vs trajectory chaos', loc='left', pad=8)
ax4.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#999999')

for ax in (ax1, ax2, ax3, ax4):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle(
    f'v1 vs v2 surrogate — 100 m048 seeds × 500 epochs = '
    f'{pooled["n_epochs_total"]:,} predictions\n'
    f'overall: v1 MAE {pooled["v1_mae"]:.4f} → v2 {pooled["v2_mae"]:.4f} '
    f'({pooled["ratio_mae"]:.1f}×)   |   '
    f'bright: v1 {pooled["v1_bright_mae"]:.4f} → v2 {pooled["v2_bright_mae"]:.4f} '
    f'({pooled["ratio_bright_mae"]:.1f}×)',
    fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT_DIR / 'm130b_population_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_DIR / "m130b_population_plot.png"}')


# ── 3. Bonus: residual magnitude vs brightness (answers "where does v2 win?") ─

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Hexbin of |residual| vs truth brightness
for ax, pool_err, version, color in [(ax_l, pool_v1, 'v1', C_V1), (ax_r, pool_v2, 'v2', C_V2)]:
    hb = ax.hexbin(pool_truth, np.clip(pool_err, 1e-5, None),
                   gridsize=(60, 50), yscale='log', cmap='Blues' if version == 'v2' else 'Reds',
                   mincnt=1, bins='log')
    med_by_bin_edges = np.linspace(pool_truth.min(), pool_truth.max(), 25)
    centers = 0.5 * (med_by_bin_edges[:-1] + med_by_bin_edges[1:])
    med_vals = []
    for i in range(len(med_by_bin_edges) - 1):
        sel = (pool_truth >= med_by_bin_edges[i]) & (pool_truth < med_by_bin_edges[i + 1])
        med_vals.append(np.median(pool_err[sel]) if sel.any() else np.nan)
    ax.plot(centers, med_vals, color=color, lw=2.2, marker='o', ms=5,
            mec='white', mew=0.8, label=f'{version} running median')

    pool_mae = float(pool_err.mean())
    ax.axhline(pool_mae, color=color, lw=1.0, ls=':', alpha=0.8,
               label=f'pooled MAE {pool_mae:.4f}')
    ax.set_xlabel('truth apparent magnitude')
    ax.set_title(f'{version}: |residual| vs brightness', loc='left', pad=8)
    ax.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#999999')
    cbar = plt.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label('log₁₀ density', fontsize=11)

ax_l.set_ylabel('|predicted − truth|  (mag, log scale)')
for ax in (ax_l, ax_r):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # bright regime shaded
    ax.axvspan(ax.get_xlim()[0], 10, color='#f7d060', alpha=0.14, zorder=0, label=None)
ax_l.text(ax_l.get_xlim()[0] + 0.3, 4,   'bright (mag<10)\n — where SNR matters', fontsize=10,
          color='#8a6110', alpha=0.85, ha='left')

fig.suptitle('Residual behaviour stratified by truth brightness — the bright regime is where v2 wins most',
             fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / 'm130b_residual_vs_brightness.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT_DIR / "m130b_residual_vs_brightness.png"}')

print('DONE')
