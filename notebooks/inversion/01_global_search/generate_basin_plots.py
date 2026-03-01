#!/usr/bin/env python3
"""Generate basin characterization visualizations for Roberto meeting slides."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data/results/inversion_diagnostics/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "data/results/inversion_diagnostics/exp3_convergence_basin_full.json") as f:
    data = json.load(f)

results = data['results']

# ─── Plot 1: Phase 3 Statistical Heatmap ───
print("Plot 1: Basin width heatmap...")
att_perts = [3, 5, 7, 10]
omega_perts = [0.01, 0.05, 0.1]

# Parse phase3 data
phase3 = results['phase3_statistical']
grid = np.zeros((len(att_perts), len(omega_perts)))

for trial in phase3:
    label = trial['label']  # e.g. "stat_3d_0.01dps_t0"
    parts = label.split('_')
    att_val = float(parts[1].replace('d', ''))
    omega_val = float(parts[2].replace('dps', ''))
    success = trial['success'] == 'True' or trial['success'] is True
    
    ai = att_perts.index(int(att_val))
    oi = omega_perts.index(omega_val)
    if success:
        grid[ai, oi] += 1

# 20 trials per combo
grid = grid / 20 * 100

fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=50, aspect='auto', origin='upper')

ax.set_xticks(range(len(omega_perts)))
ax.set_xticklabels([f'{o}' for o in omega_perts], fontsize=12)
ax.set_yticks(range(len(att_perts)))
ax.set_yticklabels([f'{a}°' for a in att_perts], fontsize=12)
ax.set_xlabel('ω perturbation (°/s)', fontsize=13)
ax.set_ylabel('Attitude perturbation', fontsize=13)
ax.set_title('Joint 6D Convergence Rate (%)\n240 trials, 20 per cell', fontsize=14)

for i in range(len(att_perts)):
    for j in range(len(omega_perts)):
        val = grid[i, j]
        color = 'white' if val < 20 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=14, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax, label='Convergence rate (%)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'basin_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR / 'basin_heatmap.png'}")

# ─── Plot 2: Attitude-only convergence ───
print("Plot 2: Attitude-only convergence...")
phase1_att = results['phase1_attitude_only']

init_att = [r['initial_att_deg'] for r in phase1_att]
final_att = [r['final_att_deg'] for r in phase1_att]
success_att = [r['success'] == 'True' or r['success'] is True for r in phase1_att]

fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (x, y, s) in enumerate(zip(init_att, final_att, success_att)):
    color = '#2ecc71' if s else '#e74c3c'
    marker = 'o' if s else 'x'
    ms = 10 if s else 9
    ax.plot(x, y, marker, color=color, markersize=ms, markeredgewidth=2)

# Reference line: no improvement
ax.plot([0, 180], [0, 180], '--', color='gray', alpha=0.5, label='No improvement')
# Success threshold
ax.axhline(y=5, color='#2ecc71', linestyle=':', alpha=0.7, label='5° threshold')

ax.set_xlabel('Initial attitude perturbation (°)', fontsize=13)
ax.set_ylabel('Final attitude error (°)', fontsize=13)
ax.set_title('Attitude-Only Estimation (true ω given)\nGreen=converged, Red=failed', fontsize=14)
ax.set_xlim(-2, 185)
ax.set_ylim(-2, 185)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'attitude_only_basin.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR / 'attitude_only_basin.png'}")

# ─── Plot 3: Omega-only convergence ───
print("Plot 3: Omega-only convergence...")
phase1_omega = results['phase1_omega_only']

init_omega = [r['initial_omega_dps'] for r in phase1_omega]
final_omega = [r['final_omega_dps'] for r in phase1_omega]
success_omega = [r['success'] == 'True' or r['success'] is True for r in phase1_omega]

fig, ax = plt.subplots(figsize=(7, 4.5))
for x, y, s in zip(init_omega, final_omega, success_omega):
    color = '#2ecc71' if s else '#e74c3c'
    marker = 'o' if s else 'x'
    ms = 10 if s else 9
    ax.plot(x, y, marker, color=color, markersize=ms, markeredgewidth=2)

ax.plot([0, 1.1], [0, 1.1], '--', color='gray', alpha=0.5, label='No improvement')
# True omega magnitude
true_omega_mag = np.linalg.norm(np.deg2rad([0.005, -0.003, 0.05]))
true_omega_mag_dps = np.rad2deg(true_omega_mag)
ax.axhline(y=0.02, color='#2ecc71', linestyle=':', alpha=0.7, label='0.02°/s threshold')

ax.set_xlabel('Initial ω perturbation (°/s)', fontsize=13)
ax.set_ylabel('Final ω error (°/s)', fontsize=13)
ax.set_title('Omega-Only Estimation (true attitude given)\nGreen=converged, Red=failed', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(OUT_DIR / 'omega_only_basin.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR / 'omega_only_basin.png'}")

# ─── Plot 4: Mixed-fidelity correlation ───
print("Plot 4: Mixed-fidelity correlation...")
# Need to generate lo-fi and hi-fi for the true lightcurve
# Check if exp1 data exists
exp1_path = PROJECT_ROOT / "data/results/inversion_diagnostics/exp1_fidelity_benchmark.json"
if exp1_path.exists():
    with open(exp1_path) as f:
        exp1 = json.load(f)
    
    hifi_mags = np.array(exp1.get('hifi_magnitudes', []))
    lofi_mags = np.array(exp1.get('lofi_magnitudes', []))
    
    if len(hifi_mags) > 0 and len(lofi_mags) > 0:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(hifi_mags, lofi_mags, s=30, alpha=0.7, c='#3498db', edgecolors='#2980b9', linewidth=0.5)
        
        mn = min(hifi_mags.min(), lofi_mags.min()) - 0.2
        mx = max(hifi_mags.max(), lofi_mags.max()) + 0.2
        ax.plot([mn, mx], [mn, mx], '--', color='gray', alpha=0.5, label='Perfect correlation')
        
        from scipy.stats import spearmanr
        rho, _ = spearmanr(hifi_mags, lofi_mags)
        
        ax.set_xlabel('Hi-fi magnitude (with shadows)', fontsize=13)
        ax.set_ylabel('Lo-fi magnitude (no shadows)', fontsize=13)
        ax.set_title(f'Fidelity Correlation (Spearman ρ = {rho:.3f})\n167× speedup', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'fidelity_correlation.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUT_DIR / 'fidelity_correlation.png'}")
    else:
        print("  WARNING: exp1 data missing magnitude arrays, checking keys...")
        print(f"  Available keys: {list(exp1.keys())}")
else:
    print(f"  WARNING: {exp1_path} not found, skipping plot 4")

print("\nDone! All plots in:", OUT_DIR)
