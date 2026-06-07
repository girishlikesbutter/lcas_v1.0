#!/usr/bin/env python3
"""
m130b -- Population LC fidelity: v1 vs v2 on all 100 m048 seeds.

Loads each m048 per-seed trajectory NPZ (k1_body, k2_body, obs_dist_km,
mag_hifi already cached at 500 epochs), predicts the same LC under v1 and
v2 surrogates, and records per-seed + per-epoch residual stats. No new
hi-fi propagation needed — the m048 NPZs are the truth.

Saves:
  data/results/inversion_diagnostics/m130_v1v2_plots/
    m130b_population_arrays.npz   (per-seed, per-epoch preds & residuals)
    m130b_population_summary.json (per-seed summary metrics)
    m130b_population_plot.png     (improvement-ratio histogram + per-seed scatter)
"""

import sys, os, json, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)

# Load BOTH surrogate modules by path to avoid name collision.
import importlib.util
_spec_v1 = importlib.util.spec_from_file_location(
    'surrogate_v1_module', '/home/girish/surrogate_model/surrogate.py')
_surr_v1 = importlib.util.module_from_spec(_spec_v1); _spec_v1.loader.exec_module(_surr_v1)
_spec_v2 = importlib.util.spec_from_file_location(
    'surrogate_v2_module', '/home/girish/surrogate_model/surrogate_model/surrogate.py')
_surr_v2 = importlib.util.module_from_spec(_spec_v2); _spec_v2.loader.exec_module(_surr_v2)

M048_DIR = PROJECT_ROOT / 'data' / 'results' / 'inversion_diagnostics' / 'm048_trajectories' / 'per_trajectory'
OUT_DIR = PROJECT_ROOT / 'data' / 'results' / 'inversion_diagnostics' / 'm130_v1v2_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

seed_files = sorted(M048_DIR.glob('traj_seed*.npz'))
print(f'Found {len(seed_files)} m048 seeds')

m_v1 = _surr_v1.SurrogateModel(
    '/home/girish/surrogate_model/s10_5M_weights.npz',
    '/home/girish/surrogate_model/s10_5M_normalization.npz')
m_v2 = _surr_v2.SurrogateModel.load_default()
print('Loaded v1 and v2 surrogates')


# ── Per-seed evaluation ───────────────────────────────────────────────

def metrics(pred, truth):
    valid = np.isfinite(pred) & np.isfinite(truth)
    if not valid.any():
        return None
    err = pred[valid] - truth[valid]
    abs_err = np.abs(err)
    bright = truth[valid] < 10.0
    dim = ~bright
    return {
        'n_valid':     int(valid.sum()),
        'n_bright':    int(bright.sum()),
        'mae':         float(np.mean(abs_err)),
        'rmse':        float(np.sqrt(np.mean(err**2))),
        'p50':         float(np.quantile(abs_err, 0.5)),
        'p90':         float(np.quantile(abs_err, 0.9)),
        'p99':         float(np.quantile(abs_err, 0.99)),
        'max':         float(np.max(abs_err)),
        'bright_mae':  float(np.mean(abs_err[bright])) if bright.any() else None,
        'dim_mae':     float(np.mean(abs_err[dim]))    if dim.any()    else None,
        'bright_max':  float(np.max(abs_err[bright])) if bright.any() else None,
        'dim_max':     float(np.max(abs_err[dim]))    if dim.any()    else None,
    }


rows = []
all_abs_err_v1 = []  # per-epoch over all seeds, for pooled histogram
all_abs_err_v2 = []
all_truth_for_pool = []

t0 = time.time()
for fp in seed_files:
    d = np.load(str(fp), allow_pickle=True)
    seed = int(d['seed'])
    k1 = d['k1_body']; k2 = d['k2_body']
    obs_dist = d['obs_dist']
    mag_hifi = d['mag_hifi']
    N = len(k1)
    panel = np.zeros(N); dish = np.full(N, 15.0)

    mag_v1 = m_v1.predict_magnitude(k1, k2, panel, dish, obs_dist)
    mag_v2 = m_v2.predict_magnitude(k1, k2, panel, dish, obs_dist)

    mets_v1 = metrics(mag_v1, mag_hifi)
    mets_v2 = metrics(mag_v2, mag_hifi)

    w_dps = float(d['omega_mag_dps'])
    n_peaks = int(len(d['hifi_peak_epochs']))
    mag_range = float(np.nanmax(mag_hifi) - np.nanmin(mag_hifi))
    phase = float(np.nanmean(d['phase_angle_3d']))

    rows.append({
        'seed': seed,
        'w_dps': w_dps,
        'n_peaks': n_peaks,
        'mag_range': mag_range,
        'mean_phase_deg': phase,
        'v1': mets_v1,
        'v2': mets_v2,
        'improvement_factor_mae': mets_v1['mae'] / mets_v2['mae'] if mets_v2['mae'] > 0 else None,
        'improvement_factor_bright_mae': (
            mets_v1['bright_mae'] / mets_v2['bright_mae']
            if (mets_v2['bright_mae'] and mets_v2['bright_mae'] > 0) else None),
    })

    valid = np.isfinite(mag_v1) & np.isfinite(mag_v2) & np.isfinite(mag_hifi)
    all_abs_err_v1.append(np.abs(mag_v1[valid] - mag_hifi[valid]))
    all_abs_err_v2.append(np.abs(mag_v2[valid] - mag_hifi[valid]))
    all_truth_for_pool.append(mag_hifi[valid])

print(f'Processed {len(rows)} seeds in {time.time()-t0:.1f}s')

# Pool across all seeds
all_abs_err_v1 = np.concatenate(all_abs_err_v1)
all_abs_err_v2 = np.concatenate(all_abs_err_v2)
all_truth = np.concatenate(all_truth_for_pool)

pooled = {
    'n_epochs_total':       int(len(all_abs_err_v1)),
    'v1_mae':               float(np.mean(all_abs_err_v1)),
    'v1_p90':               float(np.quantile(all_abs_err_v1, 0.9)),
    'v1_p99':               float(np.quantile(all_abs_err_v1, 0.99)),
    'v1_max':               float(np.max(all_abs_err_v1)),
    'v2_mae':               float(np.mean(all_abs_err_v2)),
    'v2_p90':               float(np.quantile(all_abs_err_v2, 0.9)),
    'v2_p99':               float(np.quantile(all_abs_err_v2, 0.99)),
    'v2_max':               float(np.max(all_abs_err_v2)),
    'bright_fraction':      float(np.mean(all_truth < 10)),
    'v1_bright_mae':        float(np.mean(all_abs_err_v1[all_truth < 10])),
    'v2_bright_mae':        float(np.mean(all_abs_err_v2[all_truth < 10])),
    'ratio_mae':            float(np.mean(all_abs_err_v1) / np.mean(all_abs_err_v2)),
    'ratio_bright_mae':     float(np.mean(all_abs_err_v1[all_truth < 10]) /
                                  np.mean(all_abs_err_v2[all_truth < 10])),
}
print('\nPooled metrics:')
for k, v in pooled.items():
    print(f'  {k:>22s} = {v:.6g}')

# Checkpoint
np.savez_compressed(
    OUT_DIR / 'm130b_population_arrays.npz',
    seeds=np.array([r['seed'] for r in rows]),
    w_dps=np.array([r['w_dps'] for r in rows]),
    n_peaks=np.array([r['n_peaks'] for r in rows]),
    mag_range=np.array([r['mag_range'] for r in rows]),
    mean_phase_deg=np.array([r['mean_phase_deg'] for r in rows]),
    v1_mae=np.array([r['v1']['mae'] for r in rows]),
    v2_mae=np.array([r['v2']['mae'] for r in rows]),
    v1_bright_mae=np.array([r['v1']['bright_mae'] or np.nan for r in rows]),
    v2_bright_mae=np.array([r['v2']['bright_mae'] or np.nan for r in rows]),
    v1_p99=np.array([r['v1']['p99'] for r in rows]),
    v2_p99=np.array([r['v2']['p99'] for r in rows]),
    v1_max=np.array([r['v1']['max'] for r in rows]),
    v2_max=np.array([r['v2']['max'] for r in rows]),
    # pooled per-epoch abs err (for histograms)
    pool_abs_err_v1=all_abs_err_v1.astype(np.float32),
    pool_abs_err_v2=all_abs_err_v2.astype(np.float32),
    pool_truth_mag=all_truth.astype(np.float32),
)
with open(OUT_DIR / 'm130b_population_summary.json', 'w') as f:
    json.dump({'pooled': pooled, 'per_seed': rows}, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'm130b_population_arrays.npz'}")
print(f"Saved: {OUT_DIR / 'm130b_population_summary.json'}")


# ── Plots ─────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax1, ax2, ax3, ax4 = axes.flatten()

# (1) Per-seed MAE scatter: v1 vs v2
v1_mae = np.array([r['v1']['mae'] for r in rows])
v2_mae = np.array([r['v2']['mae'] for r in rows])
seeds = np.array([r['seed'] for r in rows])
mx = max(v1_mae.max(), v2_mae.max()) * 1.1
ax1.plot([0, mx], [0, mx], 'k--', lw=0.6, alpha=0.5, label='v1=v2')
ax1.scatter(v2_mae, v1_mae, c='C0', s=24, alpha=0.6)
ax1.set_xlabel('v2 per-seed MAE (mag)')
ax1.set_ylabel('v1 per-seed MAE (mag)')
ax1.set_title(f'100 m048 seeds — all above diagonal ⇒ v2 beats v1 on every seed')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, mx); ax1.set_ylim(0, mx)

# (2) Improvement ratio histogram (v1_mae / v2_mae)
ratio = v1_mae / v2_mae
ax2.hist(ratio, bins=30, color='C2', alpha=0.8, edgecolor='black')
ax2.axvline(1.0, color='black', lw=1, alpha=0.5, label='v1=v2')
ax2.axvline(ratio.mean(), color='C3', lw=2, label=f'mean = {ratio.mean():.1f}×')
ax2.axvline(np.median(ratio), color='C1', lw=2, label=f'median = {np.median(ratio):.1f}×')
ax2.set_xlabel('v1_MAE / v2_MAE (higher = v2 wins more)')
ax2.set_ylabel('# seeds')
ax2.set_title('Per-seed improvement ratio')
ax2.legend()
ax2.grid(True, alpha=0.3)

# (3) Pooled abs error histogram (log scale)
bins = np.logspace(-4, 1, 60)
ax3.hist(all_abs_err_v1, bins=bins, alpha=0.5, color='C3', label=f'v1 (pool MAE {pooled["v1_mae"]:.4f})', density=True)
ax3.hist(all_abs_err_v2, bins=bins, alpha=0.5, color='C0', label=f'v2 (pool MAE {pooled["v2_mae"]:.4f})', density=True)
ax3.set_xscale('log')
ax3.set_xlabel('|predicted − truth| (mag), log scale')
ax3.set_ylabel('density')
ax3.set_title(f'Pooled per-epoch residual ({pooled["n_epochs_total"]} points over 100 seeds)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# (4) MAE vs omega rate (is the improvement bigger for chaotic seeds?)
w_dps = np.array([r['w_dps'] for r in rows])
ax4.scatter(w_dps, v1_mae, c='C3', s=22, alpha=0.6, label='v1')
ax4.scatter(w_dps, v2_mae, c='C0', s=22, alpha=0.6, label='v2')
ax4.set_xlabel('|omega| (deg/s)')
ax4.set_ylabel('per-seed MAE (mag)')
ax4.set_title('Accuracy vs trajectory chaos')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3)

fig.suptitle(
    f'v1 vs v2 surrogate — m048 100-seed population\n'
    f'v1 MAE {pooled["v1_mae"]:.4f} → v2 MAE {pooled["v2_mae"]:.4f}  '
    f'({pooled["ratio_mae"]:.1f}× improvement)   |   '
    f'v1 bright {pooled["v1_bright_mae"]:.4f} → v2 bright {pooled["v2_bright_mae"]:.4f}  '
    f'({pooled["ratio_bright_mae"]:.1f}×)',
    fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / 'm130b_population_plot.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR / 'm130b_population_plot.png'}")
print('DONE')
