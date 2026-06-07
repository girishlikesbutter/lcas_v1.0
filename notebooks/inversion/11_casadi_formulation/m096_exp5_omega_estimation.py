#!/usr/bin/env python3
"""
m096 Exp 5: Alternative |w| Estimators.

Question: Can we improve |w| estimation beyond peak-count formula?
Tests: peak count (current), prominence-weighted, Lomb-Scargle,
       mean inter-peak interval, magnitude-band peak count.
"""

import sys, os
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, lombscargle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import save_results

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1 = RESULTS_DIR / "m096_stage1"
CKPT = RESULTS_DIR / "m096_exp5_omega_est"
CKPT.mkdir(exist_ok=True)

ALL_SEEDS = list(range(100))
summary = []

for seed in ALL_SEEDS:
    d = np.load(str(STAGE1 / f"seed_{seed:03d}.npz"), allow_pickle=True)
    obs_lc = d['observed_lc']
    obs_times = d['obs_times']
    peaks = d['peaks_idx']
    peak_mags = d['peak_mags']
    peak_proms = d['peak_prominences']
    true_wmag = float(d['true_omega_mag_dps'])

    n_peaks = len(peaks)

    # Estimator 1: Current formula (peak count)
    est_count = 0.0397 * n_peaks + 0.0417

    # Estimator 2: Bright peak count only (mag < 11)
    n_bright_peaks = int(np.sum(peak_mags < 11.0))
    est_bright = 0.0397 * n_bright_peaks + 0.0417  # same formula, fewer peaks

    # Estimator 3: Prominence-weighted count
    if n_peaks > 0:
        prom_weight = np.sum(peak_proms > 0.5)  # count only prominent peaks
        est_prom = 0.0397 * prom_weight + 0.0417
    else:
        est_prom = 0.0417

    # Estimator 4: Mean inter-peak interval → frequency
    if n_peaks >= 2:
        dt_peaks = np.diff(obs_times[peaks])
        mean_interval = np.mean(dt_peaks)
        # One full rotation ≈ one complete brightness cycle
        # Rough: |w| ≈ 360° / mean_interval
        est_interval_dps = 360.0 / mean_interval if mean_interval > 0 else 0.0
        median_interval = np.median(dt_peaks)
        est_median_interval_dps = 360.0 / median_interval if median_interval > 0 else 0.0
    else:
        mean_interval = 0.0
        median_interval = 0.0
        est_interval_dps = 0.0
        est_median_interval_dps = 0.0

    # Estimator 5: Lomb-Scargle dominant frequency
    if n_peaks >= 2:
        # Use full LC, search for dominant frequency
        t = obs_times - obs_times[0]
        # Convert magnitudes to flux-like (brighter = higher)
        flux = 10 ** (-0.4 * obs_lc)
        flux -= np.mean(flux)

        # Frequency grid: 0.5 to 5 rotations per hour
        freqs_hz = np.linspace(0.5 / 3600, 5.0 / 3600, 500)
        angular_freqs = 2 * np.pi * freqs_hz
        power = lombscargle(t, flux, angular_freqs, normalize=True)
        best_freq = freqs_hz[np.argmax(power)]
        # Frequency in Hz → deg/s: one cycle = one peak pair ≈ 180° rotation
        # (each peak is a half-rotation for a biaxial object)
        est_ls_dps = best_freq * 360.0  # full rotation
        est_ls_half_dps = best_freq * 180.0  # half rotation
        ls_power_max = float(power.max())
        ls_freqs = freqs_hz
        ls_power = power
    else:
        est_ls_dps = 0.0
        est_ls_half_dps = 0.0
        ls_power_max = 0.0
        ls_freqs = np.array([])
        ls_power = np.array([])

    # Save everything
    np.savez(str(CKPT / f"seed_{seed:03d}.npz"),
             seed=seed,
             true_omega_mag_dps=true_wmag,
             n_peaks=n_peaks,
             n_bright_peaks=n_bright_peaks,
             peak_mags=peak_mags,
             peak_prominences=peak_proms,
             # Estimator values
             est_count=est_count,
             est_bright=est_bright,
             est_prom=est_prom,
             est_interval_dps=est_interval_dps,
             est_median_interval_dps=est_median_interval_dps,
             est_ls_dps=est_ls_dps,
             est_ls_half_dps=est_ls_half_dps,
             # Intermediate values
             mean_interval=mean_interval,
             median_interval=median_interval,
             ls_power_max=ls_power_max,
             ls_freqs=ls_freqs,
             ls_power=ls_power,
    )

    # Errors
    def err(est):
        return (est - true_wmag) / true_wmag * 100 if true_wmag > 0.01 else 0.0
    def in_range(est, pct):
        return est * (1 - pct/100) <= true_wmag <= est * (1 + pct/100)

    summary.append({
        'seed': seed,
        'true_wmag': true_wmag,
        'n_peaks': n_peaks,
        'est_count': float(est_count), 'err_count': err(est_count),
        'est_bright': float(est_bright), 'err_bright': err(est_bright),
        'est_prom': float(est_prom), 'err_prom': err(est_prom),
        'est_interval': float(est_interval_dps), 'err_interval': err(est_interval_dps),
        'est_ls_full': float(est_ls_dps), 'err_ls_full': err(est_ls_dps),
        'est_ls_half': float(est_ls_half_dps), 'err_ls_half': err(est_ls_half_dps),
    })

save_results(str(CKPT / "summary.json"), summary)

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXP 5: OMEGA MAGNITUDE ESTIMATORS — 100 seeds")
print("=" * 70)

estimators = [
    ('Peak count (current)', 'err_count', 'est_count'),
    ('Bright peaks (<11)', 'err_bright', 'est_bright'),
    ('Prominent peaks', 'err_prom', 'est_prom'),
    ('Mean interval', 'err_interval', 'est_interval'),
    ('Lomb-Scargle (full)', 'err_ls_full', 'est_ls_full'),
    ('Lomb-Scargle (half)', 'err_ls_half', 'est_ls_half'),
]

print(f"\n{'Estimator':25s} {'Med |err|':>9} {'Mean |err|':>10} {'Max |err|':>9} "
      f"{'In ±20%':>7} {'In ±30%':>7} {'R²':>6}")
print("-" * 85)

for name, err_key, est_key in estimators:
    errs = [abs(r[err_key]) for r in summary if r['true_wmag'] > 0.01]
    ests = [r[est_key] for r in summary if r['true_wmag'] > 0.01]
    trues = [r['true_wmag'] for r in summary if r['true_wmag'] > 0.01]

    in20 = sum(1 for e, t in zip(ests, trues) if e * 0.8 <= t <= e * 1.2)
    in30 = sum(1 for e, t in zip(ests, trues) if e * 0.7 <= t <= e * 1.3)

    corr = np.corrcoef(ests, trues)[0, 1] ** 2 if len(ests) > 1 else 0

    print(f"{name:25s} {np.median(errs):8.1f}% {np.mean(errs):9.1f}% "
          f"{max(errs):8.1f}% {in20:5d}/100 {in30:5d}/100 {corr:5.3f}")

print(f"\nSaved to {CKPT}/")
