#!/usr/bin/env python3
"""
m130c -- Re-score existing m115 DE candidates with v2.

For each seed that has m115 checkpoints, we already have:
  - step1_de.npz :: solutions_json = list of DE outputs with stored v1 surr_mse
                    and (q0, omega) for each candidate.
  - step2_hifi.npz :: hifi_json = list of clustered basins with stored hifi_mse.

This script:
  1. Loads each m115 checkpoint.
  2. For each (q0, omega) candidate, propagates attitude (same as m115 did),
     derives k1_body/k2_body, runs v2 predict_magnitude, computes MSE vs the
     SAME observed_lc m115 used (noise seed 42, sigma 0.05).
  3. Saves per-candidate (v1_mse, v2_mse, hifi_mse where available, q0_err,
     w_dir_err, w_mag_err_pct, q0_wxyz, omega_rad).
  4. Computes ranking-fidelity metrics: Spearman(v1, hifi), Spearman(v2, hifi)
     pooled across the hi-fi validated subset; per-seed minimum-candidate
     agreement (does surrogate-best match hifi-best?).

Outputs:
  data/results/inversion_diagnostics/m130_v1v2_plots/
    m130c_rescore_arrays.npz
    m130c_rescore_summary.json
    m130c_rescore_plot.png
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

NOISE_SEED = 42    # must match m115
NOISE_SIGMA = 0.05

RESULTS = PROJECT_ROOT / 'data' / 'results' / 'inversion_diagnostics'
OUT_DIR = RESULTS / 'm130_v1v2_plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Two m115 source locations — m046 trajs (11 seeds) + m048 trajs (6 seeds)
SOURCES = [
    (RESULTS / 'm115_surrogate_pipeline', 'm046'),
    (RESULTS / 'm115_surrogate_pipeline_m048', 'm048'),
]


def q0_err_deg(q_found, q_true):
    R_found = Rotation.from_quat([q_found[1], q_found[2], q_found[3], q_found[0]])
    R_true  = Rotation.from_quat([q_true[1],  q_true[2],  q_true[3],  q_true[0]])
    return float(np.rad2deg((R_found.inv() * R_true).magnitude()))


def predict_mse_batch(model, q0_list, omega_list, delta_qs_cache, sun_j2k, obs_j2k,
                      obs_dist_km, observed_lc, inertia_tensor, obs_times):
    """
    Vectorized-ish: for each (q0, omega) build k1_body/k2_body and predict.
    delta_qs_cache: dict {hash(omega) -> delta_qs}
    Returns: array of len(q0_list) with surrogate MSE vs observed_lc.
    """
    obs_valid = np.isfinite(observed_lc)
    out = np.empty(len(q0_list))
    for i, (q0, w) in enumerate(zip(q0_list, omega_list)):
        key = w.tobytes()
        if key not in delta_qs_cache:
            q_id = np.array([1.0, 0.0, 0.0, 0.0])
            d_q, _ = propagate_attitude(q_id, w, obs_times, 'tumbling', inertia_tensor)
            delta_qs_cache[key] = d_q
        delta_qs = delta_qs_cache[key]

        # q0 @ delta_qs_i  (quaternion multiplication, scalar-first)
        w0, x0, y0, z0 = q0[0], q0[1], q0[2], q0[3]
        w2 = delta_qs[:, 0]; x2 = delta_qs[:, 1]; y2 = delta_qs[:, 2]; z2 = delta_qs[:, 3]
        q_all_w = w0*w2 - x0*x2 - y0*y2 - z0*z2
        q_all_x = w0*x2 + x0*w2 + y0*z2 - z0*y2
        q_all_y = w0*y2 - x0*z2 + y0*w2 + z0*x2
        q_all_z = w0*z2 + x0*y2 - y0*x2 + z0*w2
        q_xyzw = np.stack([q_all_x, q_all_y, q_all_z, q_all_w], axis=1)
        R = Rotation.from_quat(q_xyzw).as_matrix()
        k1 = np.einsum('nij,nj->ni', R, sun_j2k)
        k2 = np.einsum('nij,nj->ni', R, obs_j2k)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)

        N = len(obs_times)
        panel = np.zeros(N); dish = np.full(N, 15.0)
        pred = model.predict_magnitude(k1, k2, panel, dish, obs_dist_km)
        valid = obs_valid & np.isfinite(pred)
        out[i] = np.mean((pred[valid] - observed_lc[valid]) ** 2) if valid.sum() > 10 else 1e6
    return out


def process_seed(seed, traj_source, seed_dir, m_v1, m_v2):
    """Return dict with per-candidate v1/v2/hi-fi MSEs for this seed."""
    step1_path = seed_dir / 'step1_de.npz'
    step2_path = seed_dir / 'step2_hifi.npz'
    if not step1_path.exists():
        return None

    # Load truth + build same observed_lc m115 used
    truth = load_truth(seed, source=traj_source)
    true_q0 = truth['q0_wxyz']; true_w0 = truth['omega0_rad']
    true_lc = truth['mag_hifi']; obs_times = truth['observation_times']
    inertia = truth['inertia_tensor']
    rng = np.random.default_rng(NOISE_SEED)
    observed_lc = true_lc + rng.normal(0, NOISE_SIGMA, len(true_lc))

    # Setup experiment for sun_pos, obs_pos, sat_pos, obs_dist at these epochs
    start_et = truth.get('start_et'); end_time_utc = truth.get('end_time_utc')
    duration = truth['duration_s']
    ctx = setup_experiment(
        n_observations=len(obs_times), noise_sigma=0.0, skip_true_lc=True,
        start_et=start_et, end_time_utc=end_time_utc, duration_s=duration,
        true_q0_wxyz=true_q0, true_omega0_rad=true_w0,
    )
    sun_j2k = ctx.sun_pos - ctx.sat_pos        # (N, 3) J2000
    obs_j2k = ctx.obs_pos - ctx.sat_pos
    obs_dist = ctx.obs_dist

    # Load DE candidates
    s1 = np.load(str(step1_path), allow_pickle=True)
    solutions = json.loads(str(s1['solutions_json']))
    q0_list = [np.array(s['q0_wxyz']) for s in solutions]
    w_list  = [np.array(s['omega_rad']) for s in solutions]
    v1_mse_stored = np.array([s['surr_mse'] for s in solutions])

    # Re-compute v1 and v2 MSE over observed_lc (trusted re-derivation)
    delta_cache = {}
    t0 = time.time()
    v1_rescored = predict_mse_batch(m_v1, q0_list, w_list, delta_cache,
                                    sun_j2k, obs_j2k, obs_dist, observed_lc,
                                    inertia, obs_times)
    # delta_cache is re-usable across v1/v2 for the same omegas
    v2_rescored = predict_mse_batch(m_v2, q0_list, w_list, delta_cache,
                                    sun_j2k, obs_j2k, obs_dist, observed_lc,
                                    inertia, obs_times)
    t_rescore = time.time() - t0

    # Sanity: v1_rescored should match v1_mse_stored closely
    agree_rel = np.median(np.abs(v1_rescored - v1_mse_stored) /
                          np.maximum(v1_mse_stored, 1e-6))

    # Map stored hi-fi MSE by rounded q0_wxyz tuple (as m115 clustered per seed)
    hifi_lookup = {}  # key = tuple(q0_rounded), value = hifi_mse
    if step2_path.exists():
        s2 = np.load(str(step2_path), allow_pickle=True)
        hifi_list = json.loads(str(s2['hifi_json']))
        for h in hifi_list:
            q_key = tuple(np.round(h['q0_wxyz'], 6))
            hifi_lookup[q_key] = h['hifi_mse']

    q0_errs = np.array([q0_err_deg(q, true_q0) for q in q0_list])
    w_dir_errs = np.array([s['w_dir_err'] for s in solutions])
    w_mag_errs = np.array([s['w_mag_err_pct'] for s in solutions])
    is_twins = np.array([bool(s['is_twin']) for s in solutions])

    # For each candidate, look up stored hi-fi MSE by matching q0 key
    hifi_mse_per_cand = np.full(len(q0_list), np.nan)
    for i, q in enumerate(q0_list):
        qk = tuple(np.round(q, 6))
        if qk in hifi_lookup:
            hifi_mse_per_cand[i] = hifi_lookup[qk]

    return {
        'seed': seed,
        'traj_source': traj_source,
        'n_candidates': len(q0_list),
        'q0_err_deg': q0_errs,
        'w_dir_err_deg': w_dir_errs,
        'w_mag_err_pct': w_mag_errs,
        'is_twin': is_twins,
        'v1_mse_stored': v1_mse_stored,
        'v1_mse_rescored': v1_rescored,
        'v2_mse_rescored': v2_rescored,
        'hifi_mse': hifi_mse_per_cand,
        'v1_rescored_median_relerr_vs_stored': float(agree_rel),
        't_rescore_s': float(t_rescore),
    }


# ── Run across all seed folders ──────────────────────────────────────

print("Loading surrogate models...")
m_v1 = _surr_v1.SurrogateModel(
    '/home/girish/surrogate_model/s10_5M_weights.npz',
    '/home/girish/surrogate_model/s10_5M_normalization.npz')
m_v2 = _surr_v2.SurrogateModel.load_default()

all_rows = []
for src_dir, traj_source in SOURCES:
    if not src_dir.exists():
        continue
    seed_dirs = sorted([p for p in src_dir.iterdir() if p.is_dir() and p.name.startswith('seed_')])
    for sd in seed_dirs:
        seed = int(sd.name.split('_')[1])
        print(f"  [{traj_source}] seed {seed:03d}...")
        try:
            row = process_seed(seed, traj_source, sd, m_v1, m_v2)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        if row is None:
            continue
        print(f"    n_cand={row['n_candidates']}, "
              f"v1_rescore_relerr={row['v1_rescored_median_relerr_vs_stored']:.3g}, "
              f"t={row['t_rescore_s']:.1f}s")
        all_rows.append(row)

print(f"\nProcessed {len(all_rows)} seeds total")

# ── Aggregate metrics ──────────────────────────────────────────────

# (A) Per-seed ranking correlation on the hi-fi validated subset
#     (Not many candidates per seed -> pooled across all seeds is more useful.)
# (B) Pooled (v1, hifi) and (v2, hifi) Spearman across all hi-fi-validated candidates
v1_pool = []; v2_pool = []; hifi_pool = []; q0err_pool = []; seed_pool = []
for r in all_rows:
    has_hifi = np.isfinite(r['hifi_mse'])
    v1_pool.extend(r['v1_mse_rescored'][has_hifi].tolist())
    v2_pool.extend(r['v2_mse_rescored'][has_hifi].tolist())
    hifi_pool.extend(r['hifi_mse'][has_hifi].tolist())
    q0err_pool.extend(r['q0_err_deg'][has_hifi].tolist())
    seed_pool.extend([r['seed']] * int(has_hifi.sum()))

v1_pool = np.array(v1_pool); v2_pool = np.array(v2_pool); hifi_pool = np.array(hifi_pool)
q0err_pool = np.array(q0err_pool); seed_pool = np.array(seed_pool)

rho_v1_hifi, p_v1 = spearmanr(v1_pool, hifi_pool) if len(v1_pool) > 2 else (np.nan, np.nan)
rho_v2_hifi, p_v2 = spearmanr(v2_pool, hifi_pool) if len(v2_pool) > 2 else (np.nan, np.nan)

# (C) Catastrophic-ratio check (m124-style): for each hi-fi-validated candidate,
#     how far off is surrogate_MSE from hi-fi_MSE (in log-ratio)?
log_ratio_v1 = np.log10(v1_pool / hifi_pool)
log_ratio_v2 = np.log10(v2_pool / hifi_pool)

# (D) Per-seed: does surrogate-best match hi-fi-best on the validated subset?
per_seed_best_agree = []
for r in all_rows:
    has_hifi = np.isfinite(r['hifi_mse'])
    if has_hifi.sum() < 2:
        continue
    v1_sub = r['v1_mse_rescored'][has_hifi]
    v2_sub = r['v2_mse_rescored'][has_hifi]
    h_sub  = r['hifi_mse'][has_hifi]
    idx_v1 = int(np.argmin(v1_sub))
    idx_v2 = int(np.argmin(v2_sub))
    idx_h  = int(np.argmin(h_sub))
    per_seed_best_agree.append({
        'seed': r['seed'],
        'traj_source': r['traj_source'],
        'n_hifi': int(has_hifi.sum()),
        'v1_best_matches_hifi_best': bool(idx_v1 == idx_h),
        'v2_best_matches_hifi_best': bool(idx_v2 == idx_h),
    })

v1_agree_count = sum(1 for x in per_seed_best_agree if x['v1_best_matches_hifi_best'])
v2_agree_count = sum(1 for x in per_seed_best_agree if x['v2_best_matches_hifi_best'])

summary = {
    'n_seeds':          len(all_rows),
    'n_hifi_validated': int(len(v1_pool)),
    'spearman_v1_vs_hifi': float(rho_v1_hifi),
    'spearman_v2_vs_hifi': float(rho_v2_hifi),
    'log_ratio_v1': {
        'median': float(np.median(log_ratio_v1)),
        'p10':    float(np.quantile(log_ratio_v1, 0.1)),
        'p90':    float(np.quantile(log_ratio_v1, 0.9)),
        'fraction_within_0.3': float(np.mean(np.abs(log_ratio_v1) < 0.3)),
        'fraction_catastrophic_>1dex': float(np.mean(np.abs(log_ratio_v1) > 1.0)),
    },
    'log_ratio_v2': {
        'median': float(np.median(log_ratio_v2)),
        'p10':    float(np.quantile(log_ratio_v2, 0.1)),
        'p90':    float(np.quantile(log_ratio_v2, 0.9)),
        'fraction_within_0.3': float(np.mean(np.abs(log_ratio_v2) < 0.3)),
        'fraction_catastrophic_>1dex': float(np.mean(np.abs(log_ratio_v2) > 1.0)),
    },
    'best_candidate_agreement': {
        'n_seeds_with_2plus_hifi': len(per_seed_best_agree),
        'v1_picks_hifi_best':      v1_agree_count,
        'v2_picks_hifi_best':      v2_agree_count,
    },
    'v1_sanity_rescore_relerr_median':
        float(np.median([r['v1_rescored_median_relerr_vs_stored'] for r in all_rows])),
}
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))

with open(OUT_DIR / 'm130c_rescore_summary.json', 'w') as f:
    json.dump({
        'summary': summary,
        'per_seed': [{
            **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)},
            'per_candidate': {
                'q0_err_deg':       r['q0_err_deg'].tolist(),
                'w_dir_err_deg':    r['w_dir_err_deg'].tolist(),
                'w_mag_err_pct':    r['w_mag_err_pct'].tolist(),
                'is_twin':          r['is_twin'].tolist(),
                'v1_mse_stored':    r['v1_mse_stored'].tolist(),
                'v1_mse_rescored':  r['v1_mse_rescored'].tolist(),
                'v2_mse_rescored':  r['v2_mse_rescored'].tolist(),
                'hifi_mse':         [None if not np.isfinite(x) else float(x) for x in r['hifi_mse']],
            }
        } for r in all_rows],
        'per_seed_best_agreement': per_seed_best_agree,
    }, f, indent=2)

np.savez_compressed(
    OUT_DIR / 'm130c_rescore_arrays.npz',
    v1_pool=v1_pool, v2_pool=v2_pool, hifi_pool=hifi_pool,
    q0err_pool=q0err_pool, seed_pool=seed_pool,
)
print(f"\nSaved: {OUT_DIR / 'm130c_rescore_summary.json'}")
print(f"Saved: {OUT_DIR / 'm130c_rescore_arrays.npz'}")


# ── Plots ─────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

# (1) v1 MSE vs hi-fi MSE, log-log, per candidate
lo = min(v1_pool.min(), v2_pool.min(), hifi_pool.min()) * 0.5
hi = max(v1_pool.max(), v2_pool.max(), hifi_pool.max()) * 2
x = np.array([lo, hi])
ax1.plot(x, x, 'k--', lw=0.8, alpha=0.6, label='surrogate = hifi')
ax1.plot(x, x*2, 'k:', lw=0.5, alpha=0.4); ax1.plot(x, x/2, 'k:', lw=0.5, alpha=0.4)
ax1.scatter(hifi_pool, v1_pool, c='C3', s=28, alpha=0.7, label='v1')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('hi-fi MSE'); ax1.set_ylabel('v1 surrogate MSE (re-scored)')
ax1.set_title(f'v1 vs hi-fi  (Spearman rho={rho_v1_hifi:.3f}, N={len(v1_pool)})\n'
              f'{summary["log_ratio_v1"]["fraction_within_0.3"]*100:.0f}% within ±0.3dex, '
              f'{summary["log_ratio_v1"]["fraction_catastrophic_>1dex"]*100:.0f}% catastrophic')
ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)

# (2) v2 MSE vs hi-fi MSE
ax2.plot(x, x, 'k--', lw=0.8, alpha=0.6, label='surrogate = hifi')
ax2.plot(x, x*2, 'k:', lw=0.5, alpha=0.4); ax2.plot(x, x/2, 'k:', lw=0.5, alpha=0.4)
ax2.scatter(hifi_pool, v2_pool, c='C0', s=28, alpha=0.7, label='v2')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('hi-fi MSE'); ax2.set_ylabel('v2 surrogate MSE')
ax2.set_title(f'v2 vs hi-fi  (Spearman rho={rho_v2_hifi:.3f}, N={len(v2_pool)})\n'
              f'{summary["log_ratio_v2"]["fraction_within_0.3"]*100:.0f}% within ±0.3dex, '
              f'{summary["log_ratio_v2"]["fraction_catastrophic_>1dex"]*100:.0f}% catastrophic')
ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)

# (3) Log-ratio distributions
bins = np.linspace(-3, 3, 40)
ax3.hist(log_ratio_v1, bins=bins, alpha=0.5, color='C3', label=f'v1 (med={np.median(log_ratio_v1):+.2f})')
ax3.hist(log_ratio_v2, bins=bins, alpha=0.5, color='C0', label=f'v2 (med={np.median(log_ratio_v2):+.2f})')
ax3.axvline(0, color='black', lw=0.6, alpha=0.6)
ax3.axvline(-0.3, color='black', lw=0.4, ls=':', alpha=0.4)
ax3.axvline(+0.3, color='black', lw=0.4, ls=':', alpha=0.4)
ax3.set_xlabel('log10(surrogate MSE / hi-fi MSE)')
ax3.set_ylabel('# candidates')
ax3.set_title('Off-truth surrogate bias (closer to zero = better)')
ax3.legend(); ax3.grid(True, alpha=0.3)

# (4) q0_err vs log_ratio, split by surrogate
ax4.scatter(q0err_pool, log_ratio_v1, c='C3', s=28, alpha=0.6, label='v1')
ax4.scatter(q0err_pool, log_ratio_v2, c='C0', s=28, alpha=0.6, label='v2')
ax4.axhline(0, color='black', lw=0.6, alpha=0.6)
ax4.axhspan(-0.3, 0.3, color='grey', alpha=0.1)
ax4.set_xlabel('q0 error (deg) — distance from truth')
ax4.set_ylabel('log10(surrogate MSE / hi-fi MSE)')
ax4.set_title('Does bias grow with distance from truth?')
ax4.legend(); ax4.grid(True, alpha=0.3)

fig.suptitle(
    f'v1 vs v2 — re-scoring m115 DE candidates ({summary["n_seeds"]} seeds, '
    f'{summary["n_hifi_validated"]} hi-fi validated candidates)',
    fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / 'm130c_rescore_plot.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR / 'm130c_rescore_plot.png'}")
print('DONE')
