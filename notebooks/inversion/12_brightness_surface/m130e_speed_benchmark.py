#!/usr/bin/env python3
"""
m130e -- Speed benchmark of v1 / v2 / v2-fast-mode across realistic scales.

Configurations measured:
  - single-epoch predict (surrogate-optimisation overhead)
  - 500-epoch LC (typical inversion objective call)
  - 10000-direction icosphere (PAB manifold viewer per epoch)
  - 100 × 500-epoch LC batch concatenated (simulated DE generation)

Outputs:
  data/results/inversion_diagnostics/m130_v1v2_plots/
    m130e_speed_benchmark.json
    m130e_speed_benchmark.png
"""

import os, json, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path

import importlib.util
_spec_v1 = importlib.util.spec_from_file_location(
    'surrogate_v1_module', '/home/girish/surrogate_model/surrogate.py')
_surr_v1 = importlib.util.module_from_spec(_spec_v1); _spec_v1.loader.exec_module(_surr_v1)
_spec_v2 = importlib.util.spec_from_file_location(
    'surrogate_v2_module', '/home/girish/surrogate_model/surrogate_model/surrogate.py')
_surr_v2 = importlib.util.module_from_spec(_spec_v2); _spec_v2.loader.exec_module(_surr_v2)

OUT_DIR = Path('/home/girish/projects/lcas_v1.0/data/results/inversion_diagnostics/m130_v1v2_plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PACKAGE = Path('/home/girish/surrogate_model/surrogate_model')

# Models
m_v1 = _surr_v1.SurrogateModel(
    '/home/girish/surrogate_model/s10_5M_weights.npz',
    '/home/girish/surrogate_model/s10_5M_normalization.npz')
m_v2 = _surr_v2.SurrogateModel.load_default()
# v2 fast mode: single ensemble member (seed 777 is cited as best single)
m_v2_fast = _surr_v2.SurrogateModel(
    weights_paths=[PACKAGE / 's12_residual_5M_s777_weights.npz'],
    normalization_path=PACKAGE / 's12_residual_5M_normalization.npz',
    geometry_path=PACKAGE / 's11_geometry.npz',
)

print('v1:         ', m_v1)
print('v2:         ', m_v2)
print('v2 (1-mem): ', m_v2_fast)


def time_it(fn, n_warm=3, n_runs=10):
    for _ in range(n_warm):
        fn()
    dt = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        dt.append(time.perf_counter() - t0)
    return float(np.median(dt)), float(np.std(dt))


rng = np.random.default_rng(0)


def mk_sample(N):
    k1 = rng.standard_normal((N, 3)); k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 = rng.standard_normal((N, 3)); k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    panel = np.zeros(N); dish = np.full(N, 15.0)
    dist = np.full(N, 40000.0)
    return k1, k2, panel, dish, dist


configs = [
    ('single-epoch', 1),
    ('500-epoch LC', 500),
    ('1000-epoch LC', 1000),
    ('10000-dir manifold (per epoch)', 10000),
    ('50k batch (100 × 500)', 50000),
]

results = {}
for label, N in configs:
    k1, k2, panel, dish, dist = mk_sample(N)
    print(f'\n--- {label} (N={N}) ---')
    for vname, model in [('v1', m_v1), ('v2', m_v2), ('v2_fast', m_v2_fast)]:
        t_med, t_std = time_it(
            lambda: model.predict_magnitude(k1, k2, panel, dish, dist),
            n_warm=2, n_runs=(15 if N < 5000 else 7))
        results.setdefault(label, {})[vname] = {'ms_median': t_med * 1000,
                                                 'ms_std': t_std * 1000,
                                                 'N': N}
        print(f'  {vname:>8s}:  median {t_med*1000:9.3f} ms  (std {t_std*1000:6.3f})  '
              f'→ {t_med*1e6/N:8.2f} μs/sample')

# Relative speedups vs v1
summary = {}
for label in results:
    r = results[label]
    summary[label] = {
        'N': r['v1']['N'],
        'v1_ms':      r['v1']['ms_median'],
        'v2_ms':      r['v2']['ms_median'],
        'v2_fast_ms': r['v2_fast']['ms_median'],
        'v2_slowdown_vs_v1':      r['v2']['ms_median']      / r['v1']['ms_median'],
        'v2_fast_slowdown_vs_v1': r['v2_fast']['ms_median'] / r['v1']['ms_median'],
        'v2_fast_speedup_vs_v2':  r['v2']['ms_median']      / r['v2_fast']['ms_median'],
    }

with open(OUT_DIR / 'm130e_speed_benchmark.json', 'w') as f:
    json.dump({'results': results, 'summary': summary}, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'm130e_speed_benchmark.json'}")

# Compact table
print('\n\n=== Latency summary ===')
print(f'{"config":>35s} {"N":>6} {"v1 ms":>10} {"v2 ms":>10} {"v2fast ms":>12} '
      f'{"v2 vs v1":>10} {"vfast/v1":>10}')
for label, s in summary.items():
    print(f'{label:>35s} {s["N"]:>6d} {s["v1_ms"]:>10.3f} {s["v2_ms"]:>10.3f} '
          f'{s["v2_fast_ms"]:>12.3f} {s["v2_slowdown_vs_v1"]:>9.1f}× '
          f'{s["v2_fast_slowdown_vs_v1"]:>9.1f}×')


# ── Plot ────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'axes.grid': True, 'grid.alpha': 0.25, 'axes.axisbelow': True,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})

labels = list(results.keys())
ns = [results[k]['v1']['N'] for k in labels]
v1_ms = [results[k]['v1']['ms_median'] for k in labels]
v2_ms = [results[k]['v2']['ms_median'] for k in labels]
v2fast_ms = [results[k]['v2_fast']['ms_median'] for k in labels]

fig, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(15, 6))

x = np.arange(len(labels))
w = 0.26
ax_abs.bar(x - w, v1_ms,     w, label='v1 (direct MLP)',          color='#d62728', edgecolor='white', lw=0.8)
ax_abs.bar(x,     v2_ms,     w, label='v2 (residual ensemble×3)', color='#1f77b4', edgecolor='white', lw=0.8)
ax_abs.bar(x + w, v2fast_ms, w, label='v2 fast (single member)',  color='#7fbfec', edgecolor='white', lw=0.8)
ax_abs.set_xticks(x)
ax_abs.set_xticklabels([f'{l}\n(N={n})' for l, n in zip(labels, ns)], rotation=20, ha='right')
ax_abs.set_yscale('log')
ax_abs.set_ylabel('latency (ms, log scale)')
ax_abs.set_title('Absolute latency per call', loc='left', pad=8)
ax_abs.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#999')

ratios_v2 = [a/b for a, b in zip(v2_ms, v1_ms)]
ratios_fast = [a/b for a, b in zip(v2fast_ms, v1_ms)]
ax_ratio.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.6, label='v1 baseline')
ax_ratio.bar(x - w/2, ratios_v2,   w, label='v2 / v1',      color='#1f77b4', edgecolor='white', lw=0.8)
ax_ratio.bar(x + w/2, ratios_fast, w, label='v2_fast / v1', color='#7fbfec', edgecolor='white', lw=0.8)
ax_ratio.set_xticks(x)
ax_ratio.set_xticklabels([f'N={n}' for n in ns], rotation=0)
ax_ratio.set_ylabel('slowdown factor vs v1 (>1 = slower)')
ax_ratio.set_title('Relative slowdown vs v1 per config', loc='left', pad=8)
ax_ratio.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#999')

for ax in (ax_abs, ax_ratio):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle(
    'Latency benchmark — v1 vs v2 vs v2-fast-mode (single ensemble member)',
    fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / 'm130e_speed_benchmark.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR / 'm130e_speed_benchmark.png'}")
print('DONE')
