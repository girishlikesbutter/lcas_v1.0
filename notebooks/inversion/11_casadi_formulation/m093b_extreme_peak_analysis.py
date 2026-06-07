#!/usr/bin/env python3
"""
m093b — Extreme lo-fi vs hi-fi peak comparison.

One PNG, two panels stacked vertically:
  Top:    Most extreme lofi > hifi seed (shadows killed the most peaks)
  Bottom: Most extreme hifi > lofi seed (shadows created the most peaks)

Each panel overlays lo-fi and hi-fi LCs with peaks marked and the
difference shaded.
"""

import sys
import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

PEAK_DISTANCE = 5
PEAK_PROMINENCE = 0.3
MATCH_WINDOW = 3

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
I_tensor = master['inertia_tensor']
n_seeds = len(master['q0s'])
hours = (obs_times - obs_times[0]) / 3600.0

print("Loading satellite model...", flush=True)
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

satellite = ctx.satellite
sun_pos, obs_pos, sat_pos = ctx.sun_pos, ctx.obs_pos, ctx.sat_pos
obs_dist, art_matrices = ctx.obs_dist, ctx.art_matrices


def generate_lofi(seed):
    q0 = master['q0s'][seed]
    w0 = master['omega0s'][seed]
    quats, _ = propagate_attitude(q0, w0, obs_times, "tumbling", I_tensor)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = sun_pos[:n_ep] - sat_pos[:n_ep]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = obs_pos[:n_ep] - sat_pos[:n_ep]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)
    lit = create_no_shadow_lit_status(satellite, n_ep)
    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=obs_dist,
        satellite=satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=art_matrices, show_progress=False)
    return mags


# Pass 1: find extremes
print(f"Scanning {n_seeds} seeds...", flush=True)
best_lofi_extra = (-1, -999)
best_hifi_extra = (-1, -999)

for seed in range(n_seeds):
    hifi_lc = master['mag_hifi'][seed]
    lofi_lc = generate_lofi(seed)
    lp, _ = find_peaks(-lofi_lc, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    hp, _ = find_peaks(-hifi_lc, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    diff = len(lp) - len(hp)
    if diff > best_lofi_extra[1]:
        best_lofi_extra = (seed, diff)
    if -diff > best_hifi_extra[1]:
        best_hifi_extra = (seed, -diff)

print(f"Most lofi>hifi: seed {best_lofi_extra[0]} (diff=+{best_lofi_extra[1]})")
print(f"Most hifi>lofi: seed {best_hifi_extra[0]} (diff=+{best_hifi_extra[1]})")

# Pass 2: generate combined plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, (seed, _), panel_label in [
    (ax1, best_lofi_extra, "Shadows killed peaks"),
    (ax2, best_hifi_extra, "Shadows created peaks"),
]:
    hifi_lc = master['mag_hifi'][seed]
    lofi_lc = generate_lofi(seed)
    lp, _ = find_peaks(-lofi_lc, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    hp, _ = find_peaks(-hifi_lc, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    omega_mag = float(master['omega_mags'][seed])

    ax.plot(hours, lofi_lc, '-', color='#1f77b4', linewidth=0.8, label=f'Lo-fi ({len(lp)} peaks)')
    ax.plot(hours, hifi_lc, '-', color='#d62728', linewidth=0.8, label=f'Hi-fi ({len(hp)} peaks)')

    # Shade the difference
    ax.fill_between(hours, lofi_lc, hifi_lc, alpha=0.15, color='#9467bd')

    # Mark peaks
    ax.plot(hours[lp], lofi_lc[lp], 'v', color='#1f77b4', markersize=5, zorder=5)
    ax.plot(hours[hp], hifi_lc[hp], 'v', color='#d62728', markersize=5, zorder=5)

    ax.invert_yaxis()
    ax.set_ylabel('Magnitude')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Seed {seed} — {panel_label} | '
                 f'|w|={omega_mag:.2f} dps | '
                 f'lofi={len(lp)}, hifi={len(hp)}, diff={len(lp)-len(hp):+d}',
                 fontsize=10)

ax2.set_xlabel('Time (hours)')
plt.tight_layout()
out_path = RESULTS_DIR / "m093b_extreme_peak_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
