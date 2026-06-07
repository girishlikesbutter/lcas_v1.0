#!/usr/bin/env python3
"""
m130d -- Re-score m124 polished candidates with v1 and v2 surrogates.

m124 stored 25 candidates (5 seeds × (1 truth + 3 off-truth basin polishes
+ 1 polished-from-truth)), with hi-fi MSE computed. These are the exact
candidates that surfaced the "surrogate gradient anti-correlated with hi-fi
gradient on the dark-mag plateau" finding.

This script re-evaluates v1 and v2 surrogate MSE at each candidate's final
(q0, omega) and compares to the stored hifi_mse. If v2 disagrees with hi-fi
on fewer candidates than v1, or if v2's log-ratio spread is tighter, then
v2 partially or fully fixes the plateau catastrophe.

Outputs:
  data/results/inversion_diagnostics/m130_v1v2_plots/
    m130d_rescore_m124_arrays.npz
    m130d_rescore_m124_summary.json
    m130d_rescore_m124_plot.png
"""

import sys, os, json, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'notebooks' / 'inversion'))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from lib.traj_source import load_truth
from src.dynamics.attitude_propagator import propagate_attitude

import importlib.util
_spec_v1 = importlib.util.spec_from_file_location(
    'surrogate_v1_module', '/home/girish/surrogate_model/surrogate.py')
_surr_v1 = importlib.util.module_from_spec(_spec_v1); _spec_v1.loader.exec_module(_surr_v1)
_spec_v2 = importlib.util.spec_from_file_location(
    'surrogate_v2_module', '/home/girish/surrogate_model/surrogate_model/surrogate.py')
_surr_v2 = importlib.util.module_from_spec(_spec_v2); _spec_v2.loader.exec_module(_surr_v2)

RESULTS = PROJECT_ROOT / 'data' / 'results' / 'inversion_diagnostics'
M124 = RESULTS / 'm124'
OUT_DIR = RESULTS / 'm130_v1v2_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOISE_SEED = 42; NOISE_SIGMA = 0.05   # m124 used same as m115

# Load m124 stored results
d = np.load(str(M124 / 'hifi_results.npz'), allow_pickle=True)
m124_results = json.loads(str(d['results_json']))
print(f'Loaded {len(m124_results)} m124 candidates')


# Surrogates
m_v1 = _surr_v1.SurrogateModel(
    '/home/girish/surrogate_model/s10_5M_weights.npz',
    '/home/girish/surrogate_model/s10_5M_normalization.npz')
m_v2 = _surr_v2.SurrogateModel.load_default()


# Helper: compute surrogate MSE at (q0, omega)
def q_mul(q1, q2):
    """Hamilton product, wxyz convention. q1 scalar, q2 (N,4)."""
    w0, x0, y0, z0 = q1[0], q1[1], q1[2], q1[3]
    w2 = q2[:, 0]; x2 = q2[:, 1]; y2 = q2[:, 2]; z2 = q2[:, 3]
    return np.column_stack([
        w0*w2 - x0*x2 - y0*y2 - z0*z2,
        w0*x2 + x0*w2 + y0*z2 - z0*y2,
        w0*y2 - x0*z2 + y0*w2 + z0*x2,
        w0*z2 + x0*y2 - y0*x2 + z0*w2,
    ])


def surr_mse(model, q0, omega, inertia, obs_times, sun_j2k, obs_j2k, obs_dist, observed_lc):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    delta_qs, _ = propagate_attitude(q_id, omega, obs_times, 'tumbling', inertia)
    q_all = q_mul(q0, delta_qs)
    R = Rotation.from_quat(q_all[:, [1, 2, 3, 0]]).as_matrix()
    k1 = np.einsum('nij,nj->ni', R, sun_j2k)
    k2 = np.einsum('nij,nj->ni', R, obs_j2k)
    k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    N = len(obs_times)
    panel = np.zeros(N); dish = np.full(N, 15.0)
    pred = model.predict_magnitude(k1, k2, panel, dish, obs_dist)
    valid = np.isfinite(observed_lc) & np.isfinite(pred)
    return float(np.mean((pred[valid] - observed_lc[valid]) ** 2)) if valid.sum() > 10 else np.nan


# Cache per-seed ctx (avoid reloading SPICE)
ctx_cache = {}

def get_seed_ctx(seed):
    if seed in ctx_cache:
        return ctx_cache[seed]
    truth = load_truth(seed, source='m046')  # m124 used m046 traj source
    rng = np.random.default_rng(NOISE_SEED)
    observed_lc = truth['mag_hifi'] + rng.normal(0, NOISE_SIGMA, len(truth['mag_hifi']))
    ctx = setup_experiment(
        n_observations=len(truth['observation_times']),
        noise_sigma=0.0, skip_true_lc=True,
        end_time_utc=truth.get('end_time_utc'),
        duration_s=truth['duration_s'],
        true_q0_wxyz=truth['q0_wxyz'], true_omega0_rad=truth['omega0_rad'],
    )
    entry = {
        'truth':        truth,
        'observed_lc':  observed_lc,
        'obs_times':    truth['observation_times'],
        'inertia':      truth['inertia_tensor'],
        'sun_j2k':      ctx.sun_pos - ctx.sat_pos,
        'obs_j2k':      ctx.obs_pos - ctx.sat_pos,
        'obs_dist':     ctx.obs_dist,
    }
    ctx_cache[seed] = entry
    return entry


# Re-score each candidate
rows = []
t0 = time.time()
for cand in m124_results:
    seed = cand['seed']
    c = get_seed_ctx(seed)
    q0 = np.asarray(cand['q0_wxyz'], dtype=float)
    w  = np.asarray(cand['omega_rad'], dtype=float)
    hifi = float(cand['hifi_mse'])
    v1  = surr_mse(m_v1, q0, w, c['inertia'], c['obs_times'],
                   c['sun_j2k'], c['obs_j2k'], c['obs_dist'], c['observed_lc'])
    v2  = surr_mse(m_v2, q0, w, c['inertia'], c['obs_times'],
                   c['sun_j2k'], c['obs_j2k'], c['obs_dist'], c['observed_lc'])
    # q0 error vs truth
    from scipy.spatial.transform import Rotation as R_
    R_found = R_.from_quat([q0[1], q0[2], q0[3], q0[0]])
    R_true  = R_.from_quat([c['truth']['q0_wxyz'][1], c['truth']['q0_wxyz'][2],
                            c['truth']['q0_wxyz'][3], c['truth']['q0_wxyz'][0]])
    q0_err = float(np.rad2deg((R_found.inv() * R_true).magnitude()))
    rows.append({
        'seed':  seed,
        'tag':   cand['tag'],
        'kind':  cand['kind'],
        'q0_err_deg': q0_err,
        'hifi_mse':   hifi,
        'v1_mse':     v1,
        'v2_mse':     v2,
        'log_ratio_v1': float(np.log10(v1 / hifi)) if hifi > 0 and v1 > 0 else None,
        'log_ratio_v2': float(np.log10(v2 / hifi)) if hifi > 0 and v2 > 0 else None,
    })
print(f'\nRe-scored {len(rows)} candidates in {time.time()-t0:.1f}s\n')

# Sort for nicer table
rows.sort(key=lambda r: (r['seed'], r['tag']))
print(f'{"seed":>4} {"kind":>18} {"tag":>40} {"q0err":>7} {"hifi":>8} {"v1":>8} {"v2":>8} {"v1/hifi":>8} {"v2/hifi":>8}')
for r in rows:
    lr1 = r['log_ratio_v1']; lr2 = r['log_ratio_v2']
    print(f'{r["seed"]:>4} {r["kind"]:>18} {r["tag"]:>40} '
          f'{r["q0_err_deg"]:>7.2f} {r["hifi_mse"]:>8.4f} '
          f'{r["v1_mse"]:>8.4f} {r["v2_mse"]:>8.4f} '
          f'{10**lr1 if lr1 is not None else float("nan"):>8.3f} '
          f'{10**lr2 if lr2 is not None else float("nan"):>8.3f}')

# Metrics
hifi = np.array([r['hifi_mse'] for r in rows])
v1   = np.array([r['v1_mse'] for r in rows])
v2   = np.array([r['v2_mse'] for r in rows])
q0err = np.array([r['q0_err_deg'] for r in rows])
lr_v1 = np.log10(v1 / hifi)
lr_v2 = np.log10(v2 / hifi)

# Polished-only subset (exclude truth_reference which is trivially near 0)
polished_mask = np.array([r['kind'] == 'polished' for r in rows])

def _stats(ar, mask=None):
    if mask is not None:
        ar = ar[mask]
    return {
        'median': float(np.median(ar)),
        'p10':    float(np.quantile(ar, 0.1)),
        'p90':    float(np.quantile(ar, 0.9)),
        'within_0.3': float(np.mean(np.abs(ar) < 0.3)),
        'catastrophic_>1dex': float(np.mean(np.abs(ar) > 1.0)),
        'n': int(len(ar)),
    }

rho_v1_all, _ = spearmanr(v1, hifi)
rho_v2_all, _ = spearmanr(v2, hifi)
rho_v1_pol, _ = spearmanr(v1[polished_mask], hifi[polished_mask])
rho_v2_pol, _ = spearmanr(v2[polished_mask], hifi[polished_mask])

summary = {
    'n_total':     len(rows),
    'n_polished':  int(polished_mask.sum()),
    'spearman_v1_vs_hifi_all':       float(rho_v1_all),
    'spearman_v2_vs_hifi_all':       float(rho_v2_all),
    'spearman_v1_vs_hifi_polished':  float(rho_v1_pol),
    'spearman_v2_vs_hifi_polished':  float(rho_v2_pol),
    'log_ratio_v1_all':      _stats(lr_v1),
    'log_ratio_v2_all':      _stats(lr_v2),
    'log_ratio_v1_polished': _stats(lr_v1, polished_mask),
    'log_ratio_v2_polished': _stats(lr_v2, polished_mask),
}
print('\n=== SUMMARY (m124 re-scored) ===')
print(json.dumps(summary, indent=2))

with open(OUT_DIR / 'm130d_rescore_m124_summary.json', 'w') as f:
    json.dump({'summary': summary, 'per_candidate': rows}, f, indent=2)
np.savez_compressed(
    OUT_DIR / 'm130d_rescore_m124_arrays.npz',
    seeds=np.array([r['seed'] for r in rows]),
    tags=np.array([r['tag'] for r in rows]),
    kinds=np.array([r['kind'] for r in rows]),
    q0_err=q0err, hifi=hifi, v1=v1, v2=v2, lr_v1=lr_v1, lr_v2=lr_v2,
)
print(f"\nSaved: {OUT_DIR / 'm130d_rescore_m124_summary.json'}")
print(f"Saved: {OUT_DIR / 'm130d_rescore_m124_arrays.npz'}")


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

C_V1 = '#d62728'; C_V2 = '#1f77b4'
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
ax1, ax2, ax3 = axes

# Panel 1: surr MSE vs hi-fi MSE (log-log)
lo = min(v1.min(), v2.min(), hifi.min()) * 0.5
hi = max(v1.max(), v2.max(), hifi.max()) * 1.5
xx = np.array([lo, hi])
for ax_i in (ax1,):
    ax_i.plot(xx, xx, 'k--', lw=0.9, alpha=0.55, label='perfect agreement')
    ax_i.plot(xx, xx * 2, 'k:', lw=0.55, alpha=0.4);  ax_i.plot(xx, xx / 2, 'k:', lw=0.55, alpha=0.4)
    ax_i.plot(xx, xx * 10, 'k:', lw=0.4, alpha=0.3); ax_i.plot(xx, xx / 10, 'k:', lw=0.4, alpha=0.3)
    trmask = np.array([r['kind'] == 'truth_reference' for r in rows])
    ax_i.scatter(hifi[trmask], v1[trmask], c=C_V1, s=95, alpha=0.9, marker='*',
                 edgecolors='white', linewidths=0.9, zorder=4,
                 label='v1 — truth ref')
    ax_i.scatter(hifi[polished_mask], v1[polished_mask], c=C_V1, s=55, alpha=0.75,
                 edgecolors='white', linewidths=0.6, zorder=3, label='v1 — polished')
    ax_i.scatter(hifi[trmask], v2[trmask], c=C_V2, s=95, alpha=0.9, marker='*',
                 edgecolors='white', linewidths=0.9, zorder=4,
                 label='v2 — truth ref')
    ax_i.scatter(hifi[polished_mask], v2[polished_mask], c=C_V2, s=55, alpha=0.75,
                 edgecolors='white', linewidths=0.6, zorder=3, label='v2 — polished')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('hi-fi MSE'); ax1.set_ylabel('surrogate MSE')
ax1.set_title(
    f'Surrogate vs hi-fi  (Spearman polished: v1 {rho_v1_pol:.3f}, v2 {rho_v2_pol:.3f})',
    loc='left', pad=8)
ax1.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#999')
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)

# Panel 2: log-ratio histograms (polished only — the interesting regime)
bins = np.linspace(-2.5, 2.5, 40)
ax2.hist(lr_v1[polished_mask], bins=bins, alpha=0.6, color=C_V1,
         label=f'v1 polished (N={polished_mask.sum()}, |.|>1dex: '
               f'{summary["log_ratio_v1_polished"]["catastrophic_>1dex"]*100:.0f}%)')
ax2.hist(lr_v2[polished_mask], bins=bins, alpha=0.6, color=C_V2,
         label=f'v2 polished (N={polished_mask.sum()}, |.|>1dex: '
               f'{summary["log_ratio_v2_polished"]["catastrophic_>1dex"]*100:.0f}%)')
ax2.axvline(0, color='black', lw=0.9, alpha=0.6)
ax2.axvspan(-0.3, 0.3, color='grey', alpha=0.12)
ax2.set_xlabel('log₁₀(surrogate MSE / hi-fi MSE)')
ax2.set_ylabel('# candidates')
ax2.set_title('Off-truth bias — narrower around 0 is better', loc='left', pad=8)
ax2.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#999')

# Panel 3: q0 error vs log-ratio (does bias grow off-truth?)
ax3.axhline(0, color='black', lw=0.8, alpha=0.6)
ax3.axhspan(-0.3, 0.3, color='grey', alpha=0.12)
for ax_i in (ax3,):
    ax_i.scatter(q0err[polished_mask], lr_v1[polished_mask], c=C_V1, s=60, alpha=0.75,
                 edgecolors='white', linewidths=0.6, label='v1 polished')
    ax_i.scatter(q0err[polished_mask], lr_v2[polished_mask], c=C_V2, s=60, alpha=0.75,
                 edgecolors='white', linewidths=0.6, label='v2 polished')
    ax_i.scatter(q0err[~polished_mask], lr_v1[~polished_mask], c=C_V1, s=120, alpha=0.9,
                 marker='*', edgecolors='white', linewidths=0.8, label='v1 truth ref')
    ax_i.scatter(q0err[~polished_mask], lr_v2[~polished_mask], c=C_V2, s=120, alpha=0.9,
                 marker='*', edgecolors='white', linewidths=0.8, label='v2 truth ref')
ax3.set_xlabel('q0 error (deg)  —  distance from truth')
ax3.set_ylabel('log₁₀(surrogate / hi-fi)')
ax3.set_title('Bias vs distance from truth', loc='left', pad=8)
ax3.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='#999')
# annotate the seed 27 basin_0 catastrophe if present
for r, lr1, lr2 in zip(rows, lr_v1, lr_v2):
    if r['seed'] == 27 and 'basin_0' in r['tag']:
        ax3.annotate(f"seed 27 basin_0\n(v1 {lr1:+.2f}, v2 {lr2:+.2f})",
                     xy=(r['q0_err_deg'], lr1), xytext=(30, -60),
                     textcoords='offset points', fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='grey', alpha=0.6))

for ax in (ax1, ax2, ax3):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle('Re-scoring m124 polished candidates — does v2 fix the off-truth anti-correlation?',
             fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / 'm130d_rescore_m124_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR / 'm130d_rescore_m124_plot.png'}")
print('DONE')
