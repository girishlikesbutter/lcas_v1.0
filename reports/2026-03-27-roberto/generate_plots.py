#!/usr/bin/env python3
"""
Generate plots for the 2026-03-27 Roberto report.

Plots:
  1. Pipeline overview: seed 93 hi-fi cluster with candidate solutions annotated
  2. LC comparison: true vs degenerate twin (seed 93)
  3. Multi-seed results summary table (generated after all seeds complete)
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# PLOT 1: Seed 93 hi-fi cluster — candidate solutions
# ══════════════════════════════════════════════════════════════════════

def plot_hifi_cluster(seed=93):
    result_dir = RESULTS_DIR / f"micro73_pipeline_seed{seed:03d}"
    with open(str(result_dir / "result.json")) as f:
        result = json.load(f)

    hifi_cands = result['hifi_candidates']
    all_cands = result['all_geo_candidates']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: geo cost for all 10 candidates, with cluster boundary
    geo_costs = [c['geo_cost'] for c in all_cands]
    colors = ['#2ecc71' if c['w0_err'] < 5 else '#e74c3c' for c in all_cands]
    ax1.bar(range(1, len(geo_costs)+1), geo_costs, color=colors, edgecolor='k', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Candidate (sorted by geo cost)')
    ax1.set_ylabel('Geometric cost')
    ax1.set_title(f'Seed {seed}: Geometric refinement — cluster detection')

    # Mark cluster boundary
    cluster_size = result['hifi_cluster_size']
    ax1.axvline(cluster_size + 0.5, color='orange', linestyle='--', linewidth=2,
                label=f'Cluster boundary (n={cluster_size})')
    ax1.legend(fontsize=9)

    # Add legend for colors
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor='#2ecc71', edgecolor='k', label=r'$\omega$ err < 5$\degree$'),
        Patch(facecolor='#e74c3c', edgecolor='k', label=r'$\omega$ err > 5$\degree$'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label=f'Cluster cut (n={cluster_size})')
    ], fontsize=8, loc='upper left')

    # Right: hi-fi residual for cluster members
    hifi_res = [c['hifi'] for c in hifi_cands]
    w_errs = [c['w0_err'] for c in hifi_cands]
    labels = [f"w#{c['omega_rank']+1} {c['anchor']}" for c in hifi_cands]
    colors2 = ['#2ecc71' if w < 5 else '#e74c3c' for w in w_errs]

    bars = ax2.barh(range(len(hifi_res)), hifi_res, color=colors2, edgecolor='k', linewidth=0.5)
    ax2.set_yticks(range(len(hifi_res)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Hi-fi MSE residual')
    ax2.set_title(f'Seed {seed}: Hi-fi ranking (cluster only)')
    ax2.invert_yaxis()

    # Annotate with omega error
    for i, (res, werr) in enumerate(zip(hifi_res, w_errs)):
        ax2.text(res + 0.005, i, f'{werr:.1f}$\\degree$', va='center', fontsize=8)

    plt.tight_layout()
    out = ASSETS_DIR / "hifi_cluster_seed93.pdf"
    plt.savefig(str(out), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2: LC comparison — true vs degenerate twin
# ══════════════════════════════════════════════════════════════════════

def plot_lc_comparison(seed=93):
    """Use the twin comparison data from micro72b."""
    twin_dir = RESULTS_DIR / "micro72_twin_comparison"
    if not (twin_dir / "twin_lightcurves.npz").exists():
        print("Twin LC data not found — skipping LC comparison plot")
        return

    d = np.load(str(twin_dir / "twin_lightcurves.npz"))
    # These are 10-min LCs from the latest run
    true_mags = d['true_mags']
    alt_mags = d['alt_mags']
    obs_times = d['obs_times']
    time_min = obs_times / 60.0

    # Load the full 60-min versions from the trajectory DB
    master = np.load(str(RESULTS_DIR / "micro46_trajectories" / "micro46_trajectories.npz"),
                     allow_pickle=True)
    true_mags_full = master['mag_hifi'][seed]

    # We need the alt full LC too — check if micro73 result has it
    # Actually we computed it in micro72c. Let's use the 10-min version for the plot
    # since it shows the detail better.

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(time_min, true_mags, 'b-', alpha=0.8, linewidth=0.8, label='True solution')
    ax1.plot(time_min, alt_mags, 'r-', alpha=0.6, linewidth=0.8, label='Degenerate twin')
    ax1.set_ylabel('Apparent magnitude')
    ax1.legend(fontsize=10)
    ax1.invert_yaxis()
    ax1.set_title(f'Seed {seed}: True vs degenerate twin light curves (first 10 min)')
    ax1.grid(True, alpha=0.2)

    # Highlight specular glints
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(-true_mags, distance=5, prominence=0.3)
    spec = peaks[true_mags[peaks] < 6.0]
    if len(spec) > 0:
        ax1.scatter(time_min[spec], true_mags[spec], c='gold', s=60, zorder=5,
                    edgecolors='k', linewidths=0.5, label='Specular glints')
        ax1.legend(fontsize=9)

    # Residual
    residual = true_mags - alt_mags
    ax2.plot(time_min, residual, 'k-', linewidth=0.6)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('True $-$ Twin (mag)')
    rms = np.sqrt(np.mean(residual**2))
    ax2.set_title(f'Residual (RMS = {rms:.3f} mag)')
    ax2.grid(True, alpha=0.2)

    # Annotate non-glint peaks where residual is large
    big_diff = np.where(np.abs(residual) > 0.5)[0]
    if len(big_diff) > 0:
        ax2.fill_between(time_min, residual, where=np.abs(residual) > 0.5,
                         color='red', alpha=0.15, label='|Residual| > 0.5 mag')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    out = ASSETS_DIR / "lc_twin_comparison.pdf"
    plt.savefig(str(out), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 3: Multi-seed summary
# ══════════════════════════════════════════════════════════════════════

def plot_multiseed_summary(seeds=[93, 0, 74, 36, 27, 14]):
    """Bar chart of omega errors across seeds."""
    results = []
    for seed in seeds:
        result_dir = RESULTS_DIR / f"micro73_pipeline_seed{seed:03d}"
        json_path = result_dir / "result.json"
        if not json_path.exists():
            print(f"  Seed {seed}: no result yet")
            continue
        with open(str(json_path)) as f:
            r = json.load(f)
        results.append({
            'seed': seed,
            'w_err': r['winner']['w0_err'],
            'q0_err': r['winner']['q0_err'],
            'w_mag_err': r['winner']['w_mag_err_pct'],
            'n_spec': r['n_specular'],
            'time': r['timing']['total_s'],
            'cluster': r['hifi_cluster_size'],
        })

    if not results:
        print("No multi-seed results available yet")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    seeds_plot = [r['seed'] for r in results]
    w_errs = [r['w_err'] for r in results]
    q0_errs = [r['q0_err'] for r in results]
    times = [r['time'] / 60 for r in results]

    # Omega direction error
    colors = ['#2ecc71' if w < 5 else '#f39c12' if w < 20 else '#e74c3c' for w in w_errs]
    axes[0].bar(range(len(seeds_plot)), w_errs, color=colors, edgecolor='k', linewidth=0.5)
    axes[0].set_xticks(range(len(seeds_plot)))
    axes[0].set_xticklabels([f's{s}' for s in seeds_plot])
    axes[0].set_ylabel(r'$\omega$ direction error ($\degree$)')
    axes[0].set_title(r'$\omega$ recovery')
    axes[0].axhline(5, color='gray', linestyle='--', alpha=0.5, label=r'5$\degree$ threshold')
    axes[0].legend(fontsize=8)

    # Attitude error
    axes[1].bar(range(len(seeds_plot)), q0_errs, color='#3498db', edgecolor='k', linewidth=0.5)
    axes[1].set_xticks(range(len(seeds_plot)))
    axes[1].set_xticklabels([f's{s}' for s in seeds_plot])
    axes[1].set_ylabel(r'$q_0$ error ($\degree$)')
    axes[1].set_title('Attitude recovery')
    axes[1].axhline(180, color='orange', linestyle='--', alpha=0.5, label=r'180$\degree$ (phi degeneracy)')
    axes[1].legend(fontsize=8)

    # Runtime
    axes[2].bar(range(len(seeds_plot)), times, color='#9b59b6', edgecolor='k', linewidth=0.5)
    axes[2].set_xticks(range(len(seeds_plot)))
    axes[2].set_xticklabels([f's{s}' for s in seeds_plot])
    axes[2].set_ylabel('Runtime (min)')
    axes[2].set_title('Pipeline runtime')

    plt.suptitle('Blind inversion results across trajectory seeds', fontsize=13, y=1.02)
    plt.tight_layout()
    out = ASSETS_DIR / "multiseed_summary.pdf"
    plt.savefig(str(out), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating report plots...")
    plot_hifi_cluster()
    plot_lc_comparison()
    plot_multiseed_summary()
    print("Done.")
