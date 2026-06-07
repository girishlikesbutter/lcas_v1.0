#!/usr/bin/env python3
"""
m093c — Comprehensive LC feature table + correlation analysis.

Phase 1: Build a feature table (~55 features × 100 seeds) covering:
  - Omega state (direction vs body axes, vs observer, dynamics characterisation)
  - Peak features (counts by band, timing, magnitudes, shapes)
  - Overall LC shape (moments, derivative stats)
  - Spectral features (Lomb-Scargle)
  - Shadow features (lo-fi vs hi-fi differences)

Phase 2: Correlation analysis:
  - Spearman rank correlation matrix
  - Top correlations with omega direction features
  - Scatter plots for top hits
  - Partial correlations controlling for |omega|

Saves: feature_table.npz, correlation_matrix.npz, scatter plots PNG
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.stats import spearmanr, pearsonr, skew, kurtosis
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_DIR = RESULTS_DIR / "m093c_feature_analysis"
OUT_DIR.mkdir(exist_ok=True)

PEAK_DISTANCE = 5
PEAK_PROMINENCE = 0.3
MATCH_WINDOW = 3

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Build feature table
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Phase 1: Computing feature table")
print("=" * 60)
t0 = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
I_tensor = master['inertia_tensor']
n_seeds = len(master['q0s'])
dt = np.median(np.diff(obs_times))

# Principal axes
eigenvalues, eigenvectors = np.linalg.eigh(I_tensor)
order = np.argsort(eigenvalues)
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
# e0=minor(Z), e1=intermediate(X), e2=major(Y)
T_max = 0.5 * eigenvalues[2]  # max KE per unit |omega|^2 (wrong — need to compute properly)

print(f"Principal moments: {eigenvalues[0]:.0f}, {eigenvalues[1]:.0f}, {eigenvalues[2]:.0f}")
print(f"Principal axes: Z(minor), X(inter), Y(major)")

# Body-frame axes (for facet normals)
body_axes = np.eye(3)  # +X, +Y, +Z as rows 0, 1, 2

# Load satellite model for lo-fi
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

# Observer and sun directions at epoch 0 (J2000)
obs_dir_j2000 = obs_pos[0] - sat_pos[0]
obs_dir_j2000 /= np.linalg.norm(obs_dir_j2000)
sun_dir_j2000 = sun_pos[0] - sat_pos[0]
sun_dir_j2000 /= np.linalg.norm(sun_dir_j2000)
pab_dir_j2000 = pab_j2000[0]


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


# Feature names and storage
feature_names = []
feature_data = []

print(f"Computing features for {n_seeds} seeds...", flush=True)

all_features = []
for seed in range(n_seeds):
    f = {}
    q0 = master['q0s'][seed]
    w0 = master['omega0s'][seed]
    hifi_lc = master['mag_hifi'][seed]
    w_mag = np.linalg.norm(w0)
    w_dir = w0 / w_mag

    # ── Group 1: Omega state ──────────────────────────────────────
    # Direction vs body axes
    f['dot_w_Xpos'] = abs(np.dot(w_dir, body_axes[0]))
    f['dot_w_Ypos'] = abs(np.dot(w_dir, body_axes[1]))
    f['dot_w_Zpos'] = abs(np.dot(w_dir, body_axes[2]))

    # Angle to nearest body axis
    body_dots = [f['dot_w_Xpos'], f['dot_w_Ypos'], f['dot_w_Zpos']]
    f['nearest_body_axis'] = np.argmax(body_dots)  # 0=X, 1=Y, 2=Z
    f['angle_to_nearest_axis'] = np.rad2deg(np.arccos(np.clip(max(body_dots), 0, 1)))

    # Spherical coords of omega in body frame
    f['w_theta'] = np.rad2deg(np.arccos(np.clip(w_dir[2], -1, 1)))  # angle from Z
    f['w_phi'] = np.rad2deg(np.arctan2(w_dir[1], w_dir[0]))  # angle in XY plane

    # Direction vs observer/sun/PAB (need to transform omega to J2000)
    R_q0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
    w_j2000 = R_q0 @ w0  # omega in J2000
    w_j2000_dir = w_j2000 / np.linalg.norm(w_j2000)
    f['dot_w_observer'] = abs(np.dot(w_j2000_dir, obs_dir_j2000))
    f['dot_w_sun'] = abs(np.dot(w_j2000_dir, sun_dir_j2000))
    f['dot_w_pab'] = abs(np.dot(w_j2000_dir, pab_dir_j2000))

    # Omega magnitude
    f['w_mag_dps'] = np.rad2deg(w_mag)

    # Kinetic energy ratio
    T = 0.5 * w0 @ I_tensor @ w0
    T_min_at_this_w = 0.5 * eigenvalues[0] * w_mag**2  # rotation about minor axis
    T_max_at_this_w = 0.5 * eigenvalues[2] * w_mag**2  # rotation about major axis
    if T_max_at_this_w > T_min_at_this_w:
        f['energy_ratio'] = (T - T_min_at_this_w) / (T_max_at_this_w - T_min_at_this_w)
    else:
        f['energy_ratio'] = 0.5

    # L direction and L-omega misalignment
    L = I_tensor @ w0
    L_dir = L / np.linalg.norm(L)
    f['L_omega_angle'] = np.rad2deg(np.arccos(np.clip(abs(np.dot(L_dir, w_dir)), 0, 1)))

    # ── Group 2: Peak features ────────────────────────────────────
    peaks, props = find_peaks(-hifi_lc, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    peak_mags = hifi_lc[peaks]

    f['n_peaks'] = len(peaks)
    f['n_X'] = int(np.sum(peak_mags < 5.9))
    f['n_YZ'] = int(np.sum((peak_mags >= 5.9) & (peak_mags < 7.3)))
    f['n_dim'] = int(np.sum(peak_mags >= 7.3))
    f['frac_X'] = f['n_X'] / max(f['n_peaks'], 1)
    f['frac_YZ'] = f['n_YZ'] / max(f['n_peaks'], 1)
    f['frac_dim'] = f['n_dim'] / max(f['n_peaks'], 1)
    f['ratio_X_to_YZ'] = f['n_X'] / max(f['n_YZ'], 0.5)
    f['ratio_X_to_dim'] = f['n_X'] / max(f['n_dim'], 0.5)

    if len(peak_mags) > 0:
        f['peak_mag_brightest'] = float(np.min(peak_mags))
        f['peak_mag_dimmest'] = float(np.max(peak_mags))
        f['peak_mag_mean'] = float(np.mean(peak_mags))
        f['peak_mag_std'] = float(np.std(peak_mags))
        f['peak_mag_range'] = f['peak_mag_dimmest'] - f['peak_mag_brightest']
        f['peak_mag_skew'] = float(skew(peak_mags)) if len(peak_mags) > 2 else 0.0
    else:
        f['peak_mag_brightest'] = 15.0
        f['peak_mag_dimmest'] = 15.0
        f['peak_mag_mean'] = 15.0
        f['peak_mag_std'] = 0.0
        f['peak_mag_range'] = 0.0
        f['peak_mag_skew'] = 0.0

    # Peak timing
    if len(peaks) > 2:
        spacings = np.diff(peaks) * dt  # in seconds
        f['spacing_mean'] = float(np.mean(spacings))
        f['spacing_std'] = float(np.std(spacings))
        f['spacing_cv'] = f['spacing_std'] / max(f['spacing_mean'], 1e-6)
        f['spacing_max'] = float(np.max(spacings))
        f['spacing_min'] = float(np.min(spacings))
        # Autocorrelation of spacings
        sp_centered = spacings - np.mean(spacings)
        sp_var = np.var(spacings)
        if sp_var > 1e-10 and len(spacings) > 3:
            f['spacing_acf1'] = float(np.corrcoef(sp_centered[:-1], sp_centered[1:])[0, 1])
            f['spacing_acf2'] = float(np.corrcoef(sp_centered[:-2], sp_centered[2:])[0, 1]) if len(spacings) > 4 else 0.0
        else:
            f['spacing_acf1'] = 0.0
            f['spacing_acf2'] = 0.0
    else:
        f['spacing_mean'] = 0.0
        f['spacing_std'] = 0.0
        f['spacing_cv'] = 0.0
        f['spacing_max'] = 0.0
        f['spacing_min'] = 0.0
        f['spacing_acf1'] = 0.0
        f['spacing_acf2'] = 0.0

    # Peak width (half-prominence width)
    if len(peaks) > 0:
        widths = props.get('widths', None)
        if widths is not None and len(widths) > 0:
            f['peak_width_mean'] = float(np.mean(widths)) * dt
        else:
            # Estimate width manually: half-prominence crossing
            pw = []
            for pi, pk in enumerate(peaks):
                prom = props['prominences'][pi]
                half_level = hifi_lc[pk] + prom / 2  # magnitude threshold
                # Search left and right for crossing
                left = pk
                while left > 0 and hifi_lc[left] < half_level:
                    left -= 1
                right = pk
                while right < len(hifi_lc) - 1 and hifi_lc[right] < half_level:
                    right += 1
                pw.append((right - left) * dt)
            f['peak_width_mean'] = float(np.mean(pw)) if pw else 0.0

        # Mean |dMag/dt| at peaks
        dmag = np.gradient(hifi_lc, dt)
        f['peak_slope_mean'] = float(np.mean(np.abs(dmag[peaks])))
    else:
        f['peak_width_mean'] = 0.0
        f['peak_slope_mean'] = 0.0

    # ── Group 3: Overall LC shape ─────────────────────────────────
    f['lc_mean'] = float(np.mean(hifi_lc))
    f['lc_std'] = float(np.std(hifi_lc))
    f['lc_skew'] = float(skew(hifi_lc))
    f['lc_kurtosis'] = float(kurtosis(hifi_lc))
    f['lc_frac_below_8'] = float(np.mean(hifi_lc < 8.0))
    f['lc_frac_below_10'] = float(np.mean(hifi_lc < 10.0))
    f['lc_frac_below_12'] = float(np.mean(hifi_lc < 12.0))

    dmag_dt = np.gradient(hifi_lc, dt)
    f['lc_deriv_rms'] = float(np.sqrt(np.mean(dmag_dt**2)))

    # ── Group 4: Spectral features ────────────────────────────────
    from astropy.timeseries import LombScargle
    freq_grid = np.linspace(0.001, 0.1, 1000)  # Hz
    ls = LombScargle(obs_times, hifi_lc)
    power = ls.power(freq_grid)

    f['ls_dominant_freq'] = float(freq_grid[np.argmax(power)])
    f['ls_dominant_power'] = float(np.max(power))
    f['ls_total_power'] = float(np.sum(power))
    f['ls_power_ratio'] = f['ls_dominant_power'] / max(f['ls_total_power'], 1e-10)

    # Ratio of dominant freq to omega magnitude
    w_freq = np.rad2deg(w_mag) / 360.0  # rotations per second
    f['ls_freq_over_w'] = f['ls_dominant_freq'] / max(w_freq, 1e-10)

    # Number of significant spectral peaks
    ls_peaks, _ = find_peaks(power, prominence=0.05 * np.max(power))
    f['ls_n_significant'] = len(ls_peaks)

    # Second strongest frequency
    if len(ls_peaks) >= 2:
        sorted_ls = ls_peaks[np.argsort(power[ls_peaks])[::-1]]
        f['ls_freq2_over_freq1'] = float(freq_grid[sorted_ls[1]] / max(freq_grid[sorted_ls[0]], 1e-10))
    else:
        f['ls_freq2_over_freq1'] = 0.0

    # ── Group 5: Shadow features ──────────────────────────────────
    lofi_lc = generate_lofi(seed)

    lp, _ = find_peaks(-lofi_lc, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    hp = peaks  # hi-fi peaks already computed

    # Peaks killed/created by shadows
    lofi_set, hifi_set = set(lp), set(hp)
    n_killed = 0
    for p in lp:
        if not any((p + off) in hifi_set for off in range(-MATCH_WINDOW, MATCH_WINDOW + 1)):
            n_killed += 1
    n_created = 0
    for p in hp:
        if not any((p + off) in lofi_set for off in range(-MATCH_WINDOW, MATCH_WINDOW + 1)):
            n_created += 1

    f['shadow_peaks_killed'] = n_killed
    f['shadow_peaks_created'] = n_created
    f['shadow_net_effect'] = n_killed - n_created

    diff_lc = hifi_lc - lofi_lc
    f['shadow_rms'] = float(np.sqrt(np.mean(diff_lc**2)))
    f['shadow_frac_gt05'] = float(np.mean(np.abs(diff_lc) > 0.5))

    all_features.append(f)
    if seed % 20 == 0:
        print(f"  seed {seed}/100...", flush=True)

# Build feature matrix
feature_names = sorted(all_features[0].keys())
n_features = len(feature_names)
feature_matrix = np.zeros((n_seeds, n_features))
for i, f in enumerate(all_features):
    for j, name in enumerate(feature_names):
        feature_matrix[i, j] = f[name]

# Save feature table
np.savez(str(OUT_DIR / "feature_table.npz"),
         matrix=feature_matrix,
         names=np.array(feature_names),
         seeds=np.arange(n_seeds))
print(f"\nPhase 1 done in {time.time() - t0:.1f}s")
print(f"  {n_seeds} seeds × {n_features} features")
print(f"  Saved: {OUT_DIR / 'feature_table.npz'}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Correlation analysis
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Phase 2: Correlation analysis")
print(f"{'='*60}")

# Target features (omega direction — what we want to predict from LC)
omega_features = ['dot_w_Xpos', 'dot_w_Ypos', 'dot_w_Zpos',
                   'angle_to_nearest_axis', 'dot_w_observer', 'dot_w_pab',
                   'energy_ratio', 'L_omega_angle']

# Observable features (what we can measure from the LC)
observable_features = [n for n in feature_names if n not in
    ['dot_w_Xpos', 'dot_w_Ypos', 'dot_w_Zpos', 'nearest_body_axis',
     'angle_to_nearest_axis', 'w_theta', 'w_phi',
     'dot_w_observer', 'dot_w_sun', 'dot_w_pab',
     'energy_ratio', 'L_omega_angle']]

# Remove w_mag_dps from observables only for partial correlation later
# but keep it in the full analysis since peak count IS observable

def get_col(name):
    return feature_matrix[:, feature_names.index(name)]

# Spearman correlation matrix: observable × omega
print(f"\nSpearman correlations: {len(observable_features)} observables × {len(omega_features)} targets")
print(f"\nTop correlations with omega direction features:")
print(f"{'Observable':>30} | {'Target':>20} | {'Spearman':>8} {'Pearson':>8} | {'p_spear':>8}")
print("-" * 95)

all_corrs = []
for obs_name in observable_features:
    obs_col = get_col(obs_name)
    for tgt_name in omega_features:
        tgt_col = get_col(tgt_name)
        # Skip if either column is constant
        if np.std(obs_col) < 1e-10 or np.std(tgt_col) < 1e-10:
            continue
        rho_s, p_s = spearmanr(obs_col, tgt_col)
        rho_p, p_p = pearsonr(obs_col, tgt_col)
        all_corrs.append((obs_name, tgt_name, rho_s, rho_p, p_s))

# Sort by absolute Spearman
all_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

for obs_name, tgt_name, rho_s, rho_p, p_s in all_corrs[:40]:
    sig = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else ""
    print(f"{obs_name:>30} | {tgt_name:>20} | {rho_s:+8.3f} {rho_p:+8.3f} | {p_s:8.4f} {sig}")

# Save full correlation data
np.savez(str(OUT_DIR / "correlations.npz"),
         obs_names=np.array([c[0] for c in all_corrs]),
         tgt_names=np.array([c[1] for c in all_corrs]),
         spearman=np.array([c[2] for c in all_corrs]),
         pearson=np.array([c[3] for c in all_corrs]),
         p_values=np.array([c[4] for c in all_corrs]))

# ── Partial correlations controlling for |omega| ──────────────────
print(f"\n{'='*60}")
print("Partial correlations (controlling for |omega|)")
print(f"{'='*60}")

w_mag_col = get_col('w_mag_dps')

def partial_corr(x, y, z):
    """Partial Spearman correlation of x,y controlling for z."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    # Residualize x and y on z
    def residualize(a, b):
        slope = np.corrcoef(a, b)[0, 1] * np.std(a) / max(np.std(b), 1e-10)
        return a - slope * b
    rx_res = residualize(rx, rz)
    ry_res = residualize(ry, rz)
    if np.std(rx_res) < 1e-10 or np.std(ry_res) < 1e-10:
        return 0.0
    return float(np.corrcoef(rx_res, ry_res)[0, 1])

print(f"\n{'Observable':>30} | {'Target':>20} | {'Raw ρ':>8} {'Partial ρ':>9} | {'Δ':>6}")
print("-" * 90)

for obs_name, tgt_name, rho_s, rho_p, p_s in all_corrs[:30]:
    obs_col = get_col(obs_name)
    tgt_col = get_col(tgt_name)
    rho_partial = partial_corr(obs_col, tgt_col, w_mag_col)
    delta = rho_partial - rho_s
    print(f"{obs_name:>30} | {tgt_name:>20} | {rho_s:+8.3f} {rho_partial:+9.3f} | {delta:+6.3f}")


# ── Scatter plots for top 12 correlations ─────────────────────────
print(f"\nGenerating scatter plots...")

fig, axes = plt.subplots(4, 3, figsize=(24, 28))
axes = axes.flatten()

for idx, (obs_name, tgt_name, rho_s, rho_p, p_s) in enumerate(all_corrs[:12]):
    ax = axes[idx]
    obs_col = get_col(obs_name)
    tgt_col = get_col(tgt_name)
    w_mag = get_col('w_mag_dps')

    sc = ax.scatter(obs_col, tgt_col, c=w_mag, cmap='viridis', s=20, alpha=0.7)
    ax.set_xlabel(obs_name, fontsize=8)
    ax.set_ylabel(tgt_name, fontsize=8)
    ax.set_title(f'ρ_s={rho_s:+.3f} (p={p_s:.3f})', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # Mark tested seeds
    for seed, status in [(0,'OK'),(6,'P'),(12,'F'),(14,'OK'),(24,'OK'),
                          (27,'F'),(33,'F'),(36,'P'),(74,'P'),(93,'OK')]:
        marker = 'o' if status == 'OK' else 's' if status == 'P' else 'x'
        color = 'green' if status == 'OK' else 'orange' if status == 'P' else 'red'
        ax.plot(obs_col[seed], tgt_col[seed], marker, color=color,
                markersize=8, markeredgewidth=2, zorder=10)

plt.tight_layout()
out_path = OUT_DIR / "scatter_top12.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"\nStrongest predictors of omega direction (|Spearman| > 0.3):")
for obs_name, tgt_name, rho_s, rho_p, p_s in all_corrs:
    if abs(rho_s) >= 0.3:
        print(f"  {obs_name:>30} → {tgt_name:<20} ρ={rho_s:+.3f} (p={p_s:.4f})")

print(f"\nFeatures that SURVIVE partial correlation (|partial ρ| > 0.25):")
for obs_name, tgt_name, rho_s, rho_p, p_s in all_corrs[:30]:
    obs_col = get_col(obs_name)
    tgt_col = get_col(tgt_name)
    rho_partial = partial_corr(obs_col, tgt_col, w_mag_col)
    if abs(rho_partial) >= 0.25:
        print(f"  {obs_name:>30} → {tgt_name:<20} raw={rho_s:+.3f} partial={rho_partial:+.3f}")

print(f"\nTotal time: {time.time() - t0:.1f}s")
print(f"All outputs in: {OUT_DIR}/")
