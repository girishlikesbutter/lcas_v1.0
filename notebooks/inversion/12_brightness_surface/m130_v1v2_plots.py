#!/usr/bin/env python3
"""
m130 -- Visual v1 vs v2 surrogate fidelity at a chaotic trajectory.

Picks a chaotic m048 seed (high omega, many peaks), re-runs hi-fi at 1000
epochs (vs the 500 in the m048 cache) for the truth LC, then predicts the
same LC with surrogate v1 and v2 and plots truth-vs-prediction with a
residual subplot underneath for each surrogate.

Outputs:
  data/results/inversion_diagnostics/m130_v1v2_plots/
    m130_seed{NNN}_v1.png
    m130_seed{NNN}_v2.png
    m130_seed{NNN}_arrays.npz   (truth, pred_v1, pred_v2, epochs, etc.)
    m130_seed{NNN}_meta.json    (seed, timing, summary stats)
"""

import sys, os, json, time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, '/home/girish/surrogate_model')                       # v1 top-level module
sys.path.insert(0, '/home/girish/surrogate_model/surrogate_model')        # v2 nested package
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from lib.traj_source import load_truth
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

# v1 and v2 surrogate modules — import both by path to avoid name collision.
import importlib.util
_spec_v1 = importlib.util.spec_from_file_location(
    'surrogate_v1_module', '/home/girish/surrogate_model/surrogate.py')
_surr_v1 = importlib.util.module_from_spec(_spec_v1); _spec_v1.loader.exec_module(_surr_v1)
_spec_v2 = importlib.util.spec_from_file_location(
    'surrogate_v2_module', '/home/girish/surrogate_model/surrogate_model/surrogate.py')
_surr_v2 = importlib.util.module_from_spec(_spec_v2); _spec_v2.loader.exec_module(_surr_v2)

SEED = int(os.environ.get('M130_SEED', 19))
N_EPOCHS = int(os.environ.get('M130_N_EPOCHS', 1000))
OUT_DIR = (PROJECT_ROOT / 'data' / 'results' / 'inversion_diagnostics'
           / 'm130_v1v2_plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== m130: seed {SEED}, N_EPOCHS={N_EPOCHS} ===")


# ── Load truth ────────────────────────────────────────────────────────

truth = load_truth(SEED, source='m048')
q0_true = truth['q0_wxyz']
w0_true = truth['omega0_rad']
start_et = truth['start_et']
duration_s = truth['duration_s']
print(f"  start_et = {start_et}, duration = {duration_s}s")
print(f"  q0_true = {q0_true}")
print(f"  w0_true = {w0_true} (mag = {np.linalg.norm(np.rad2deg(w0_true)):.3f} deg/s)")


# ── Setup 1000-epoch experiment ────────────────────────────────────────

t0 = time.time()
ctx = setup_experiment(
    n_observations=N_EPOCHS, noise_sigma=0.0, skip_true_lc=True,
    start_et=start_et, duration_s=duration_s,
    true_q0_wxyz=q0_true, true_omega0_rad=w0_true,
)
print(f"  setup_experiment: {time.time()-t0:.1f}s")


# ── Propagate attitude at 1000 epochs ──────────────────────────────────

t0 = time.time()
quats, _ = propagate_attitude(
    q0_true, w0_true, ctx.observation_times, 'tumbling', ctx.inertia_tensor)
print(f"  attitude propagation: {time.time()-t0:.1f}s ({quats.shape})")


# ── Build k1_body, k2_body at 1000 epochs (vectorised over N) ─────────

quats_xyzw = quats[:, [1, 2, 3, 0]]
R_all = Rotation.from_quat(quats_xyzw).as_matrix()   # (N, 3, 3) J2000→body

sun_vec_j2000 = ctx.sun_pos - ctx.sat_pos            # (N, 3)
obs_vec_j2000 = ctx.obs_pos - ctx.sat_pos
k1_body = np.einsum('nij,nj->ni', R_all, sun_vec_j2000)
k2_body = np.einsum('nij,nj->ni', R_all, obs_vec_j2000)
k1_body /= np.linalg.norm(k1_body, axis=1, keepdims=True)
k2_body /= np.linalg.norm(k2_body, axis=1, keepdims=True)

obs_dist_km = ctx.obs_dist   # already in km (SPICE default)
panel = np.zeros(N_EPOCHS); dish = np.full(N_EPOCHS, 15.0)


# ── Compute hi-fi truth LC (vectorised over all 1000 epochs) ──────────

t0 = time.time()
lit_status = compute_shadows(
    satellite=ctx.satellite,
    k1_vectors=k1_body,
    explicit_component_matrices=ctx.art_matrices,
    show_progress=True,
)
print(f"  compute_shadows: {time.time()-t0:.1f}s")

t0 = time.time()
mag_hifi, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status,
    k1_vectors_array=k1_body,
    k2_vectors_array=k2_body,
    observer_distances=obs_dist_km,
    satellite=ctx.satellite,
    epochs=ctx.epochs,
    pre_computed_matrices=ctx.art_matrices,
    generate_no_shadow=False,
    animate=False,
    show_progress=True,
)
print(f"  generate_lightcurves: {time.time()-t0:.1f}s  ({N_EPOCHS} epochs)")


# ── Predict with v1 and v2 ────────────────────────────────────────────

t0 = time.time()
m_v1 = _surr_v1.SurrogateModel(
    '/home/girish/surrogate_model/s10_5M_weights.npz',
    '/home/girish/surrogate_model/s10_5M_normalization.npz')
mag_v1 = m_v1.predict_magnitude(k1_body, k2_body, panel, dish, obs_dist_km)
print(f"  v1 predict: {(time.time()-t0)*1000:.1f}ms")

t0 = time.time()
m_v2 = _surr_v2.SurrogateModel.load_default()
mag_v2 = m_v2.predict_magnitude(k1_body, k2_body, panel, dish, obs_dist_km)
print(f"  v2 predict: {(time.time()-t0)*1000:.1f}ms")


# ── Metrics ───────────────────────────────────────────────────────────

def metrics(pred, truth):
    valid = np.isfinite(pred) & np.isfinite(truth)
    err = pred[valid] - truth[valid]
    abs_err = np.abs(err)
    bright = (truth[valid] < 10.0)
    dim = ~bright
    return {
        'mae': float(np.mean(abs_err)),
        'rmse': float(np.sqrt(np.mean(err**2))),
        'p90': float(np.quantile(abs_err, 0.9)),
        'p99': float(np.quantile(abs_err, 0.99)),
        'max': float(np.max(abs_err)),
        'bright_mae': float(np.mean(abs_err[bright])) if bright.any() else None,
        'dim_mae':    float(np.mean(abs_err[dim]))    if dim.any()    else None,
        'n_valid': int(valid.sum()),
        'n_bright': int(bright.sum()),
    }

mets_v1 = metrics(mag_v1, mag_hifi)
mets_v2 = metrics(mag_v2, mag_hifi)
print(f"\n  v1: MAE={mets_v1['mae']:.4f}  bright={mets_v1['bright_mae']:.4f}  max={mets_v1['max']:.3f}")
print(f"  v2: MAE={mets_v2['mae']:.4f}  bright={mets_v2['bright_mae']:.4f}  max={mets_v2['max']:.3f}")


# ── Checkpoint everything ────────────────────────────────────────────

np.savez_compressed(
    OUT_DIR / f'm130_seed{SEED:03d}_arrays.npz',
    seed=SEED, n_epochs=N_EPOCHS,
    start_et=start_et, duration_s=duration_s,
    obs_times=ctx.observation_times,
    q0_true=q0_true, w0_true=w0_true,
    quats=quats, k1_body=k1_body, k2_body=k2_body,
    obs_dist_km=obs_dist_km,
    mag_hifi=mag_hifi, mag_v1=mag_v1, mag_v2=mag_v2,
)
with open(OUT_DIR / f'm130_seed{SEED:03d}_meta.json', 'w') as f:
    json.dump({
        'seed': SEED, 'n_epochs': N_EPOCHS,
        'start_et': float(start_et), 'duration_s': float(duration_s),
        'v1': mets_v1, 'v2': mets_v2,
    }, f, indent=2)
print(f"\nSaved: {OUT_DIR / f'm130_seed{SEED:03d}_arrays.npz'}")
print(f"Saved: {OUT_DIR / f'm130_seed{SEED:03d}_meta.json'}")


# ── Plots ─────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_plot(pred, mets, version, outpath):
    valid = np.isfinite(pred) & np.isfinite(mag_hifi)
    residual = np.where(valid, pred - mag_hifi, np.nan)

    fig, (ax_lc, ax_res) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}, sharex=True)

    t = ctx.observation_times

    ax_lc.plot(t, mag_hifi,  color='black', lw=1.0, label='hi-fi truth (shadows+BRDF, 1000 ep)')
    ax_lc.plot(t, pred,      color='C3' if version == 'v1' else 'C0', lw=0.9,
               alpha=0.85, label=f'surrogate {version}')
    ax_lc.invert_yaxis()
    ax_lc.set_ylabel('apparent magnitude')
    ax_lc.set_title(
        f'Seed {SEED} — chaotic m048 traj  |  {version}: '
        f'MAE={mets["mae"]:.4f}  bright={mets["bright_mae"]:.4f}  '
        f'p90={mets["p90"]:.4f}  max={mets["max"]:.3f} mag')
    ax_lc.legend(loc='upper right')
    ax_lc.grid(True, alpha=0.3)

    ax_res.axhline(0, color='black', lw=0.6, alpha=0.5)
    ax_res.plot(t, residual, color='C3' if version == 'v1' else 'C0', lw=0.7)
    ax_res.fill_between(t, residual, 0,
                        where=(residual > 0), color='C3' if version == 'v1' else 'C0',
                        alpha=0.15)
    ax_res.fill_between(t, residual, 0,
                        where=(residual < 0), color='C3' if version == 'v1' else 'C0',
                        alpha=0.15)
    ax_res.set_ylabel(f'{version} − truth (mag)')
    ax_res.set_xlabel('time since epoch start (s)')
    ax_res.grid(True, alpha=0.3)

    # symmetric y-range, but zoomed based on 99th percentile
    ymax = max(mets['p99'] * 1.5, 0.05)
    ax_res.set_ylim(-ymax, ymax)

    fig.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")

make_plot(mag_v1, mets_v1, 'v1', OUT_DIR / f'm130_seed{SEED:03d}_v1.png')
make_plot(mag_v2, mets_v2, 'v2', OUT_DIR / f'm130_seed{SEED:03d}_v2.png')

print("\nDONE")
