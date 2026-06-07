#!/usr/bin/env python3
"""
m119 -- Attitude isoshell POC on seed 14 (kernel-factored surrogate).

Hypothesis
----------
Replacing the pab-contour brightness function B(n_body) with the surrogate
B(k1_body, k2_body, panel=0, dish=15, dist) eliminates the ~25 deg pab-contour
noise floor shown in m118. Per-epoch attitude isoshells
    { R : |surrogate(R @ k1_J2000, R @ k2_J2000, 0, 15, d) - observed_mag| < sigma }
should be 2-D submanifolds of SO(3) that the truth trajectory passes through
at nearly all 255 constraint epochs, while the m118 rank-1 competitor
fails most.

This script is KERNEL-FACTORISED. Stage C builds a single expensive residual
tensor residual[N_SO3, 255] (float32). Stage D re-scores multiple cost
variants and sigma thresholds against that tensor cheaply. No sigma or cost
form is baked in.

Pipeline
--------
A. Setup: load m118 kernel (epochs, truth, observed), regenerate J2000
   geometry via setup_experiment, load surrogate.
B. SO(3) grid: super-Fibonacci N_SO3 = 60000 rotations.
C. Residual kernel: parallel surrogate eval over epochs, producing
   residual[N_SO3, 255] (float32). Skip if cached unless MICRO119_FORCE=1.
D. Cost variants: mean_L2, mean_L1, max_abs, count_pass_* (sigma in
   {0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00}), soft_pass_*
   (sigma in {0.10, 0.20, 0.50}).
E. Target scoring: truth, m115 basins, m118 rank-1 (facet + IPL),
   using exact propagated attitudes (not grid-nearest).
F. Diagnostics + plots + POC PASS/FAIL print.

Usage
-----
    python3 m119_attitude_isoshell.py
    MICRO119_FORCE=1 python3 m119_attitude_isoshell.py
    MICRO119_N_SO3=5000 MICRO119_POOL=4 python3 m119_attitude_isoshell.py

Outputs
-------
    data/results/inversion_diagnostics/m119/seed_014/
        setup.npz
        so3_grid.npz
        residual_kernel.npz      (~60 MB)
        cost_variants.npz
        target_scores.npz
        summary.json
        run.log
        plots/
            residual_distribution.png
            target_pass_curves.png
            topK_overlap.png
            truth_vs_competitors_rank.png
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
import time
import json
from pathlib import Path
import multiprocessing as mp

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, '/home/girish/surrogate_model')
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude
from surrogate import SurrogateModel


# ---- Config --------------------------------------------------------------
SEED = int(os.environ.get('MICRO119_SEED', '14'))
POOL_SIZE = int(os.environ.get('MICRO119_POOL', '8'))
FORCE = os.environ.get('MICRO119_FORCE', '0') == '1'
N_SO3 = int(os.environ.get('MICRO119_N_SO3', '60000'))

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
MICRO46_NPZ = RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"
MICRO118_DIR = RESULTS_DIR / "m118" / f"seed_{SEED:03d}"
MICRO115_JSON = (RESULTS_DIR / "m115_surrogate_pipeline"
                 / f"seed_{SEED:03d}" / "result.json")

OUT_BASE = RESULTS_DIR / "m119"
OUT_DIR = OUT_BASE / f"seed_{SEED:03d}"
PLOT_DIR = OUT_DIR / "plots"

SURROGATE_WEIGHTS = Path('/home/girish/surrogate_model/s10_5M_weights.npz')
SURROGATE_NORM = Path('/home/girish/surrogate_model/s10_5M_normalization.npz')

PANEL_DEG = 0.0
DISH_DEG = 15.0

COST_SIGMA_COUNT = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
COST_SIGMA_SOFT = [0.10, 0.20, 0.50]
TOP_K = 1000

# POC thresholds
POC_MEDIAN_MAX = 0.1
POC_P90_MAX = 0.3


# ---- Logging -------------------------------------------------------------
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except (ValueError, OSError):
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except (ValueError, OSError):
                pass


def atomic_json_save(filepath, data):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(filepath)


# ---- Quaternion helpers (copied from m115, do NOT modify lib) --------
def attitude_error_deg(q1, q2):
    """Geodesic distance between two wxyz quaternions in degrees."""
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return float(np.rad2deg((R1.inv() * R2).magnitude()))


def omega_dir_err(w1, w2):
    d1 = w1 / np.linalg.norm(w1)
    d2 = w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def omega_mag_err_pct(w_est, w_true):
    return float(100.0 * (np.linalg.norm(w_est) - np.linalg.norm(w_true))
                 / np.linalg.norm(w_true))


# ---- Super-Fibonacci SO(3) grid -----------------------------------------
def super_fibonacci_quats(n):
    """
    Alexa 2022 super-Fibonacci spiral on SO(3).

    Returns quaternions in xyzw (scipy) convention, shape (n, 4), unit norm.
    Deterministic.
    """
    i = np.arange(n, dtype=np.float64)
    s = i + 0.5
    t = s / n
    d = 2.0 * np.pi * s
    r = np.sqrt(t)
    R = np.sqrt(1.0 - t)
    alpha = d * np.sqrt(2.0)
    beta = d * np.sqrt(3.0)
    x = r * np.sin(alpha)
    y = r * np.cos(alpha)
    z = R * np.sin(beta)
    w = R * np.cos(beta)
    q_xyzw = np.stack([x, y, z, w], axis=1)
    q_xyzw /= np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    return q_xyzw


def xyzw_to_wxyz(q):
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]])
    return q[:, [3, 0, 1, 2]]


def wxyz_to_xyzw(q):
    if q.ndim == 1:
        return np.array([q[1], q[2], q[3], q[0]])
    return q[:, [1, 2, 3, 0]]


# ---- Stage C worker -----------------------------------------------------
_WORKER = {}


def _init_worker(R_matrices, k1_j2000_ce, k2_j2000_ce, obs_dist_ce,
                 weights_path, norm_path):
    _WORKER['R'] = R_matrices
    _WORKER['k1'] = k1_j2000_ce
    _WORKER['k2'] = k2_j2000_ce
    _WORKER['dist'] = obs_dist_ce
    _WORKER['model'] = SurrogateModel(weights_path, norm_path)


def _compute_epoch_slice(ep_range):
    """Compute mag_pred[N_SO3, len(ep_range)] for a block of epochs."""
    R = _WORKER['R']
    k1_all = _WORKER['k1']
    k2_all = _WORKER['k2']
    dist_all = _WORKER['dist']
    model = _WORKER['model']

    N = R.shape[0]
    out = np.zeros((N, len(ep_range)), dtype=np.float32)
    zeros_N = np.zeros(N, dtype=np.float64)
    dish_N = np.full(N, DISH_DEG, dtype=np.float64)

    for k, ep in enumerate(ep_range):
        k1_body = np.einsum('nij,j->ni', R, k1_all[ep])
        k2_body = np.einsum('nij,j->ni', R, k2_all[ep])
        k1_body /= np.linalg.norm(k1_body, axis=1, keepdims=True)
        k2_body /= np.linalg.norm(k2_body, axis=1, keepdims=True)
        dist_N = np.full(N, dist_all[ep], dtype=np.float64)
        mag = model.predict_magnitude(k1_body, k2_body, zeros_N, dish_N, dist_N)
        out[:, k] = mag.astype(np.float32)
    return (ep_range, out)


# ---- Target scoring helpers ---------------------------------------------
def evaluate_target(q0_wxyz, omega0, obs_times, inertia_tensor,
                    constraint_epochs, k1_j2000_ce, k2_j2000_ce,
                    obs_dist_ce, observed_mag_ce, model):
    """
    Propagate (q0, omega0) over obs_times; subset to constraint epochs;
    evaluate surrogate; return per-epoch residual + the constraint-epoch
    quaternions.
    """
    quats, _ = propagate_attitude(
        q0_wxyz, omega0, obs_times, "tumbling", inertia_tensor)
    q_ce = quats[constraint_epochs]  # [255, 4] wxyz
    R_ce = Rotation.from_quat(wxyz_to_xyzw(q_ce)).as_matrix()  # [255,3,3]
    k1_body = np.einsum('nij,nj->ni', R_ce, k1_j2000_ce)
    k2_body = np.einsum('nij,nj->ni', R_ce, k2_j2000_ce)
    k1_body /= np.linalg.norm(k1_body, axis=1, keepdims=True)
    k2_body /= np.linalg.norm(k2_body, axis=1, keepdims=True)
    n = len(constraint_epochs)
    mag_pred = model.predict_magnitude(
        k1_body, k2_body,
        np.zeros(n, dtype=np.float64),
        np.full(n, DISH_DEG, dtype=np.float64),
        obs_dist_ce.astype(np.float64),
    )
    residual = mag_pred.astype(np.float32) - observed_mag_ce.astype(np.float32)
    return q_ce, residual


def variant_keys():
    keys = ['mean_L2', 'mean_L1', 'max_abs']
    for s in COST_SIGMA_COUNT:
        keys.append(f'count_pass_{int(round(s*100)):03d}')
    for s in COST_SIGMA_SOFT:
        keys.append(f'soft_pass_{int(round(s*100)):03d}')
    return keys


def score_variant(name, residual):
    """
    Apply named cost variant to a residual array.
    residual shape: (..., N_epochs). Returns scores of shape (...,) such that
    lower (for loss-type) or higher (for pass-type) is better. We always
    return "raw" scores and let the caller interpret; we also return a
    `descending` flag.
    """
    abs_r = np.abs(residual)
    if name == 'mean_L2':
        return np.mean(residual ** 2, axis=-1), False  # lower better
    if name == 'mean_L1':
        return np.mean(abs_r, axis=-1), False
    if name == 'max_abs':
        return np.max(abs_r, axis=-1), False
    if name.startswith('count_pass_'):
        sigma = int(name.split('_')[-1]) / 100.0
        return np.sum(abs_r < sigma, axis=-1).astype(np.float32), True
    if name.startswith('soft_pass_'):
        sigma = int(name.split('_')[-1]) / 100.0
        return np.sum(np.exp(-(residual ** 2) / (2.0 * sigma * sigma)),
                      axis=-1).astype(np.float32), True
    raise ValueError(f"Unknown variant: {name}")


def top_k_indices(scores, descending, k):
    if descending:
        # largest first
        order = np.argsort(-scores, kind='stable')
    else:
        order = np.argsort(scores, kind='stable')
    return order[:k].astype(np.int64)


# ---- Stage loaders ------------------------------------------------------
def stage_A_setup():
    print("\n[Stage A] Setup")
    t0 = time.time()
    out = OUT_DIR / "setup.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=False))
        print(f"  Stage A (cached) done in {time.time()-t0:.1f}s")
        return d

    # Kernel from m118
    kernel = np.load(MICRO118_DIR / "kernel.npz", allow_pickle=True)
    constraint_epochs = kernel['constraint_epochs'].astype(np.int64)
    truth_q0 = kernel['truth_q0'].astype(np.float64)
    truth_omega0 = kernel['truth_omega0'].astype(np.float64)
    observed_lc = kernel['observed_lc'].astype(np.float64)
    obs_times = kernel['obs_times'].astype(np.float64)

    # Cross-check truth_q0 against m046
    m046 = np.load(MICRO46_NPZ)
    q0_micro46 = m046['q0s'][SEED].astype(np.float64)
    q0_diff = attitude_error_deg(truth_q0, q0_micro46)
    print(f"  truth_q0 vs m046 q0s[{SEED}]: {q0_diff:.3f} deg "
          f"({'match' if q0_diff < 1e-3 else 'DIFF'})")

    obs_dist_full = m046['obs_dist'].astype(np.float64)  # [500]

    # Regenerate J2000 geometry (skip true LC)
    print("  calling setup_experiment(500, skip_true_lc=True) ...")
    ctx = setup_experiment(n_observations=500, skip_true_lc=True)
    inertia_tensor = ctx.inertia_tensor.astype(np.float64)

    k1_j2000 = ctx.sun_pos - ctx.sat_pos
    k1_j2000 /= np.linalg.norm(k1_j2000, axis=1, keepdims=True)
    k2_j2000 = ctx.obs_pos - ctx.sat_pos
    k2_j2000 /= np.linalg.norm(k2_j2000, axis=1, keepdims=True)

    # Subset
    ce = constraint_epochs
    k1_j2000_ce = k1_j2000[ce]
    k2_j2000_ce = k2_j2000[ce]
    obs_dist_ce = obs_dist_full[ce]
    observed_mag_ce = observed_lc[ce]

    # Also sanity: ctx.obs_dist should match m046 obs_dist
    diff_dist = float(np.max(np.abs(ctx.obs_dist - obs_dist_full)))
    print(f"  ctx.obs_dist vs m046 obs_dist max diff: {diff_dist:.3e} km")

    np.savez(
        out,
        constraint_epochs=constraint_epochs,
        truth_q0=truth_q0,
        truth_omega0=truth_omega0,
        observed_mag_ce=observed_mag_ce,
        k1_j2000_ce=k1_j2000_ce,
        k2_j2000_ce=k2_j2000_ce,
        obs_dist_ce=obs_dist_ce,
        inertia_tensor=inertia_tensor,
        obs_times=obs_times,
        observed_lc=observed_lc,
        q0_micro46=q0_micro46,
    )
    print(f"  saved {out}")
    print(f"  Stage A done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=False))


def stage_B_grid():
    print(f"\n[Stage B] SO(3) grid (N_SO3={N_SO3}, super-fibonacci)")
    t0 = time.time()
    out = OUT_DIR / "so3_grid.npz"
    if out.exists() and not FORCE:
        d = dict(np.load(out, allow_pickle=True))
        if int(d.get('N_SO3', 0)) == N_SO3:
            print(f"  loading cached {out.name}")
            print(f"  Stage B (cached) done in {time.time()-t0:.1f}s")
            return d
        else:
            print(f"  cache has N_SO3={d.get('N_SO3')} != {N_SO3}; rebuilding")

    q_xyzw = super_fibonacci_quats(N_SO3)
    q_wxyz = xyzw_to_wxyz(q_xyzw)
    R_matrices = Rotation.from_quat(q_xyzw).as_matrix().astype(np.float64)
    # Sanity
    print(f"  R_matrices shape={R_matrices.shape}, det min/max="
          f"{np.linalg.det(R_matrices).min():.6f}/"
          f"{np.linalg.det(R_matrices).max():.6f}")

    np.savez(
        out,
        quats_wxyz=q_wxyz.astype(np.float64),
        quats_xyzw=q_xyzw.astype(np.float64),
        R_matrices=R_matrices,
        grid_type=np.array("super-fibonacci"),
        N_SO3=np.array(N_SO3, dtype=np.int64),
    )
    print(f"  saved {out}")
    print(f"  Stage B done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=True))


def stage_C_residual(setup, grid):
    print(f"\n[Stage C] Residual kernel (POOL_SIZE={POOL_SIZE})")
    t0 = time.time()
    out = OUT_DIR / "residual_kernel.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=False))
        print(f"  residual shape={d['residual'].shape}, dtype={d['residual'].dtype}")
        print(f"  Stage C (cached) done in {time.time()-t0:.1f}s")
        return d

    R_matrices = grid['R_matrices']
    k1_ce = setup['k1_j2000_ce']
    k2_ce = setup['k2_j2000_ce']
    dist_ce = setup['obs_dist_ce']
    observed_mag_ce = setup['observed_mag_ce']
    constraint_epochs = setup['constraint_epochs']

    N = R_matrices.shape[0]
    E = len(constraint_epochs)
    print(f"  N_SO3={N}, N_epochs={E}, allocating mag_pred[{N},{E}] float32 "
          f"(~{N*E*4/1e6:.1f} MB)")

    # Chunk epochs among workers
    chunk = max(1, E // (POOL_SIZE * 2))
    ep_ranges = [list(range(i, min(i + chunk, E))) for i in range(0, E, chunk)]
    print(f"  {len(ep_ranges)} work chunks of up to {chunk} epochs each")

    mag_pred = np.zeros((N, E), dtype=np.float32)

    init_args = (R_matrices, k1_ce, k2_ce, dist_ce.astype(np.float64),
                 str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))

    t_pool = time.time()
    with mp.Pool(POOL_SIZE, initializer=_init_worker, initargs=init_args) as pool:
        for ep_range, block in pool.imap_unordered(_compute_epoch_slice, ep_ranges):
            mag_pred[:, ep_range] = block
    print(f"  pool finished in {time.time()-t_pool:.1f}s")

    residual = mag_pred - observed_mag_ce.astype(np.float32)[None, :]
    print(f"  residual stats: median|r|={np.median(np.abs(residual)):.4f}, "
          f"p90|r|={np.percentile(np.abs(residual),90):.4f}, "
          f"max|r|={np.max(np.abs(residual)):.4f}")

    np.savez(
        out,
        residual=residual,
        mag_pred=mag_pred,
        observed_mag_ce=observed_mag_ce.astype(np.float32),
        constraint_epochs=constraint_epochs,
    )
    print(f"  saved {out} (~{out.stat().st_size/1e6:.1f} MB)")
    print(f"  Stage C done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=False))


def stage_D_variants(residuals):
    print("\n[Stage D] Cost variants")
    t0 = time.time()
    out = OUT_DIR / "cost_variants.npz"
    residual = residuals['residual']  # [N, E] float32

    variants = variant_keys()
    save_dict = {}
    descending_map = {}
    print(f"  computing {len(variants)} variants on residual {residual.shape}")
    for name in variants:
        scores, descending = score_variant(name, residual)
        topK = top_k_indices(scores, descending, TOP_K)
        save_dict[f'score_{name}'] = scores.astype(np.float32)
        save_dict[f'topK_{name}'] = topK
        descending_map[name] = descending
        if descending:
            best = scores[topK[0]]
            worst_sample = scores[topK[-1]]
            print(f"  {name:25s} (higher=better) top1={best:.4f} "
                  f"top{TOP_K}={worst_sample:.4f}")
        else:
            best = scores[topK[0]]
            worst_sample = scores[topK[-1]]
            print(f"  {name:25s} (lower=better)  top1={best:.6f} "
                  f"top{TOP_K}={worst_sample:.6f}")

    save_dict['variant_names'] = np.array(variants)
    save_dict['descending'] = np.array(
        [descending_map[v] for v in variants], dtype=bool)
    np.savez(out, **save_dict)
    print(f"  saved {out}")
    print(f"  Stage D done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=True)), descending_map, variants


# ---- Stage E: targets ---------------------------------------------------
def parse_micro115_basins():
    """Return list of (name, q0, omega) tuples for m115 seed basins."""
    if not MICRO115_JSON.exists():
        print(f"  [warn] {MICRO115_JSON} missing; skipping m115 basins")
        return []
    try:
        with open(MICRO115_JSON) as f:
            r = json.load(f)
        basins = r.get('de_results', {}).get('basins', [])
        out = []
        for i, b in enumerate(basins):
            q0 = np.array(b['q0_wxyz'], dtype=np.float64)
            w = np.array(b['omega_rad'], dtype=np.float64)
            out.append((f'm115_basin{i}', q0, w))
        print(f"  m115: parsed {len(out)} basins")
        return out
    except Exception as e:
        print(f"  [warn] failed to parse m115 result.json: {e}")
        return []


def parse_micro118_topK(filename, tag):
    path = MICRO118_DIR / filename
    if not path.exists():
        print(f"  [warn] {path} missing; skipping {tag}")
        return []
    try:
        d = np.load(path, allow_pickle=True)
        q0 = np.array(d['q0'][0], dtype=np.float64)
        w = np.array(d['omega'][0], dtype=np.float64)
        print(f"  {tag}: rank1 q0_err={float(d['q0_err'][0]):.2f} "
              f"w_dir_err={float(d['w_dir_err'][0]):.2f}")
        return [(tag, q0, w)]
    except Exception as e:
        print(f"  [warn] failed to parse {path}: {e}")
        return []


def nearest_grid_R(q_target_wxyz, grid_quats_wxyz):
    """Per-epoch argmax |dot|. Returns idx and geodesic deg."""
    # Flip signs for hemisphere consistency (|dot|)
    dots = np.einsum('ni,mi->nm', q_target_wxyz, grid_quats_wxyz)
    abs_dots = np.abs(dots)
    idx = np.argmax(abs_dots, axis=1)
    best = abs_dots[np.arange(len(q_target_wxyz)), idx]
    deg = 2.0 * np.rad2deg(np.arccos(np.clip(best, 0.0, 1.0)))
    return idx.astype(np.int64), deg.astype(np.float64)


def stage_E_targets(setup, grid, variants_data, descending_map, variant_list):
    print("\n[Stage E] Target scoring")
    t0 = time.time()

    # Surrogate (single-threaded in main process)
    model = SurrogateModel(str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))

    truth_q0 = setup['truth_q0']
    truth_omega0 = setup['truth_omega0']
    obs_times = setup['obs_times']
    constraint_epochs = setup['constraint_epochs']
    k1_ce = setup['k1_j2000_ce']
    k2_ce = setup['k2_j2000_ce']
    dist_ce = setup['obs_dist_ce']
    observed_mag_ce = setup['observed_mag_ce']
    inertia_tensor = setup['inertia_tensor']

    grid_quats_wxyz = grid['quats_wxyz']

    targets = [('truth', truth_q0, truth_omega0)]
    targets += parse_micro118_topK('topK_facet_normal.npz', 'm118_facet_rank1')
    targets += parse_micro118_topK('topK_ipl_centroid_weighted_ext.npz',
                                   'm118_ipl_weighted_ext_rank1')
    targets += parse_micro115_basins()

    save_dict = {}
    target_summary = {}

    # Pre-extract grid scores
    grid_scores = {v: variants_data[f'score_{v}'] for v in variant_list}

    # Grid top-1 q0 err under each variant (treat epoch-0 grid R quaternion)
    # We'll compute per variant: top1 grid idx; use its quaternion as "q0 proposal"
    print("  per-variant grid top-1 vs truth:")
    grid_top1_errs = {}
    for v in variant_list:
        top_idx = variants_data[f'topK_{v}'][0]
        q_top1 = grid_quats_wxyz[int(top_idx)]
        err = attitude_error_deg(q_top1, truth_q0)
        grid_top1_errs[v] = err
        print(f"    {v:25s} top1_idx={int(top_idx)} q0_err_to_truth={err:.2f} deg")

    for name, q0, w in targets:
        print(f"\n  -- target: {name}")
        q_ce, residual = evaluate_target(
            q0, w, obs_times, inertia_tensor, constraint_epochs,
            k1_ce, k2_ce, dist_ce, observed_mag_ce, model)
        abs_r = np.abs(residual)
        med = float(np.median(abs_r))
        p90 = float(np.percentile(abs_r, 90))
        mx = float(np.max(abs_r))
        print(f"    residual: median={med:.4f} p90={p90:.4f} max={mx:.4f}")

        # Attitude error to truth
        q0_err = attitude_error_deg(q0, truth_q0)
        w_derr = omega_dir_err(w, truth_omega0)
        w_merr = omega_mag_err_pct(w, truth_omega0)
        print(f"    q0_err={q0_err:.2f} deg, w_dir_err={w_derr:.2f} deg, "
              f"w_mag_err={w_merr:+.2f}%")

        # Grid quantisation
        g_idx, g_deg = nearest_grid_R(q_ce, grid_quats_wxyz)
        print(f"    grid quant deg: median={np.median(g_deg):.2f}, "
              f"p90={np.percentile(g_deg,90):.2f}")

        # Per-variant scores for this target and rank vs grid
        scores_per_variant = {}
        ranks = {}
        for v in variant_list:
            tgt_score, descending = score_variant(v, residual)
            tgt_score = float(tgt_score)
            grid_sc = grid_scores[v]
            if descending:
                # better = higher; rank = # with strictly better score
                rank = int(np.sum(grid_sc > tgt_score))
            else:
                rank = int(np.sum(grid_sc < tgt_score))
            scores_per_variant[v] = tgt_score
            ranks[v] = rank

        # Save per-target arrays
        save_dict[f'{name}_q0'] = q0
        save_dict[f'{name}_omega'] = w
        save_dict[f'{name}_q_ce'] = q_ce
        save_dict[f'{name}_residual'] = residual
        save_dict[f'{name}_nearest_grid_R'] = g_idx
        save_dict[f'{name}_grid_quant_deg'] = g_deg
        save_dict[f'{name}_scores_values'] = np.array(
            [scores_per_variant[v] for v in variant_list], dtype=np.float64)
        save_dict[f'{name}_ranks'] = np.array(
            [ranks[v] for v in variant_list], dtype=np.int64)
        save_dict[f'{name}_q0_err_to_truth_deg'] = np.array(q0_err)
        save_dict[f'{name}_w_dir_err_deg'] = np.array(w_derr)
        save_dict[f'{name}_w_mag_err_pct'] = np.array(w_merr)

        target_summary[name] = {
            'residual_median': med,
            'residual_p90': p90,
            'residual_max': mx,
            'q0_err_deg': q0_err,
            'w_dir_err_deg': w_derr,
            'w_mag_err_pct': w_merr,
            'grid_quant_median_deg': float(np.median(g_deg)),
            'grid_quant_p90_deg': float(np.percentile(g_deg, 90)),
            'scores': scores_per_variant,
            'ranks': ranks,
        }

        print(f"    rank under variants (out of {N_SO3}):")
        for v in variant_list:
            print(f"      {v:25s} rank={ranks[v]:6d}  score={scores_per_variant[v]:.6f}")

    save_dict['target_names'] = np.array([t[0] for t in targets])
    save_dict['variant_names'] = np.array(variant_list)
    save_dict['grid_top1_q0_err_deg'] = np.array(
        [grid_top1_errs[v] for v in variant_list], dtype=np.float64)

    out = OUT_DIR / "target_scores.npz"
    np.savez(out, **save_dict)
    print(f"\n  saved {out}")
    print(f"  Stage E done in {time.time()-t0:.1f}s")
    return target_summary, grid_top1_errs


# ---- Stage F: plots ------------------------------------------------------
def plot_residual_distribution(residuals):
    print("\n[Stage F.1] residual_distribution.png")
    residual = residuals['residual']
    abs_r = np.abs(residual)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(abs_r.ravel(), bins=200, log=True, color='steelblue', alpha=0.8)
    for s in COST_SIGMA_COUNT:
        ax.axvline(s, color='red', linestyle='--', alpha=0.5)
        ax.text(s, ax.get_ylim()[1] * 0.6, f'{s}', rotation=90,
                fontsize=7, color='red', va='top')
    ax.set_xlabel('|residual| [mag]')
    ax.set_ylabel('count (log)')
    ax.set_title(f'|residual| over {residual.shape[0]}x{residual.shape[1]} grid entries')
    ax.set_xlim(0, min(3.0, abs_r.max() * 1.05))

    ax = axes[1]
    med = np.median(abs_r, axis=0)
    p90 = np.percentile(abs_r, 90, axis=0)
    mx = np.max(abs_r, axis=0)
    ep = np.arange(residual.shape[1])
    ax.plot(ep, med, label='median', color='steelblue')
    ax.plot(ep, p90, label='p90', color='orange')
    ax.plot(ep, mx, label='max', color='red', alpha=0.5)
    ax.set_xlabel('constraint epoch index')
    ax.set_ylabel('|residual| [mag]')
    ax.set_yscale('log')
    ax.set_title('per-epoch |residual| statistics over grid')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = PLOT_DIR / 'residual_distribution.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_target_pass_curves(target_residuals):
    print("\n[Stage F.2] target_pass_curves.png")
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    sigmas = np.logspace(np.log10(0.01), np.log10(2.0), 80)
    cmap = plt.get_cmap('tab10')
    for i, (name, residual) in enumerate(target_residuals.items()):
        abs_r = np.abs(residual)
        frac = np.array([np.mean(abs_r < s) for s in sigmas])
        style = '-' if name == 'truth' else ('--' if 'm118' in name else ':')
        lw = 2.5 if name == 'truth' else 1.5
        ax.plot(sigmas, frac, style, lw=lw, label=name, color=cmap(i % 10))
    ax.set_xscale('log')
    ax.set_xlabel('sigma [mag]')
    ax.set_ylabel('fraction of 255 epochs passing |r|<sigma')
    ax.set_title('target pass curves')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out = PLOT_DIR / 'target_pass_curves.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_topK_overlap(variants_data, variant_list):
    print("\n[Stage F.3] topK_overlap.png")
    V = len(variant_list)
    M = np.zeros((V, V), dtype=np.float32)
    sets = [set(variants_data[f'topK_{v}'].tolist()) for v in variant_list]
    for i in range(V):
        for j in range(V):
            M[i, j] = len(sets[i] & sets[j]) / TOP_K
    fig, ax = plt.subplots(1, 1, figsize=(max(8, V * 0.5), max(7, V * 0.45)))
    im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(V))
    ax.set_yticks(range(V))
    ax.set_xticklabels(variant_list, rotation=75, fontsize=8, ha='right')
    ax.set_yticklabels(variant_list, fontsize=8)
    for i in range(V):
        for j in range(V):
            ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center',
                    color='white' if M[i, j] < 0.5 else 'black', fontsize=6)
    fig.colorbar(im, ax=ax, label=f'|topK∩|/{TOP_K}')
    ax.set_title(f'pairwise top-{TOP_K} overlap across cost variants')
    fig.tight_layout()
    out = PLOT_DIR / 'topK_overlap.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_rank_comparison(target_summary, variant_list):
    print("\n[Stage F.4] truth_vs_competitors_rank.png")
    competitor_keys = [k for k in target_summary
                       if k != 'truth' and 'm118' in k]
    if not competitor_keys:
        competitor_keys = [k for k in target_summary if k != 'truth'][:1]

    truth_ranks = [target_summary['truth']['ranks'][v] for v in variant_list]
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(variant_list) * 0.5), 6))
    x = np.arange(len(variant_list))
    width = 0.35
    ax.bar(x - width/2, truth_ranks, width, label='truth', color='steelblue')
    for ci, ck in enumerate(competitor_keys[:1]):
        comp_ranks = [target_summary[ck]['ranks'][v] for v in variant_list]
        ax.bar(x + width/2, comp_ranks, width, label=ck, color='orange')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_list, rotation=75, fontsize=8, ha='right')
    ax.set_ylabel(f'rank out of {N_SO3} (log)')
    ax.set_title('truth rank vs m118 rank-1 competitor (lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = PLOT_DIR / 'truth_vs_competitors_rank.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---- POC classification --------------------------------------------------
def classify_poc(target_summary, variant_list):
    truth = target_summary['truth']
    competitor_keys = [k for k in target_summary
                       if k != 'truth' and 'm118' in k]
    # m118 rank-1 facet_normal preferred
    comp_key = None
    for k in ['m118_facet_rank1', 'm118_ipl_weighted_ext_rank1']:
        if k in target_summary:
            comp_key = k
            break
    if comp_key is None and competitor_keys:
        comp_key = competitor_keys[0]

    truth_in_top1000_any = any(
        truth['ranks'][v] < TOP_K for v in variant_list)

    truth_beats_comp = False
    if comp_key is not None:
        truth_beats_comp = any(
            truth['ranks'][v] < target_summary[comp_key]['ranks'][v]
            for v in variant_list)

    checks = {
        'median_residual': truth['residual_median'] < POC_MEDIAN_MAX,
        'p90_residual': truth['residual_p90'] < POC_P90_MAX,
        'truth_in_top1000_any_variant': truth_in_top1000_any,
        'truth_beats_competitor_any_variant': truth_beats_comp,
    }
    verdict = 'POC PASS' if all(checks.values()) else 'POC FAIL'
    return verdict, checks, comp_key


# ---- Main ----------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "run.log"
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"m119 -- attitude isoshell POC on seed {SEED}")
    print(f"  N_SO3={N_SO3}, POOL_SIZE={POOL_SIZE}, FORCE={FORCE}")
    print(f"  OUT_DIR={OUT_DIR}")
    t_all = time.time()
    timings = {}

    ts = time.time()
    setup = stage_A_setup()
    timings['stage_A_setup'] = time.time() - ts

    ts = time.time()
    grid = stage_B_grid()
    timings['stage_B_grid'] = time.time() - ts

    ts = time.time()
    residuals = stage_C_residual(setup, grid)
    timings['stage_C_residual'] = time.time() - ts

    ts = time.time()
    variants_data, descending_map, variant_list = stage_D_variants(residuals)
    timings['stage_D_variants'] = time.time() - ts

    ts = time.time()
    target_summary, grid_top1_errs = stage_E_targets(
        setup, grid, variants_data, descending_map, variant_list)
    timings['stage_E_targets'] = time.time() - ts

    # Reload per-target residuals for plotting
    target_scores = np.load(OUT_DIR / "target_scores.npz", allow_pickle=True)
    target_residuals = {}
    for name in target_scores['target_names']:
        key = f'{str(name)}_residual'
        if key in target_scores.files:
            target_residuals[str(name)] = target_scores[key]

    ts = time.time()
    p1 = plot_residual_distribution(residuals)
    p2 = plot_target_pass_curves(target_residuals)
    p3 = plot_topK_overlap(variants_data, variant_list)
    p4 = plot_rank_comparison(target_summary, variant_list)
    timings['stage_F_plots'] = time.time() - ts

    verdict, checks, comp_key = classify_poc(target_summary, variant_list)

    print("\n" + "=" * 72)
    print(f"POC classification: {verdict}")
    for ck, passed in checks.items():
        mark = 'PASS' if passed else 'FAIL'
        print(f"  [{mark}] {ck}")
    print("=" * 72)

    # Summary JSON
    summary = {
        'seed': SEED,
        'N_SO3': N_SO3,
        'pool_size': POOL_SIZE,
        'timings_s': {k: round(v, 2) for k, v in timings.items()},
        'total_runtime_s': round(time.time() - t_all, 2),
        'truth_residual_median': target_summary['truth']['residual_median'],
        'truth_residual_p90': target_summary['truth']['residual_p90'],
        'truth_residual_max': target_summary['truth']['residual_max'],
        'truth_ranks': target_summary['truth']['ranks'],
        'competitor_key': comp_key,
        'competitor_ranks': target_summary[comp_key]['ranks'] if comp_key else None,
        'grid_top1_q0_err_deg': {v: grid_top1_errs[v] for v in variant_list},
        'poc_verdict': verdict,
        'poc_checks': checks,
        'targets': {n: {k: (v if not isinstance(v, dict) else v)
                        for k, v in s.items()}
                    for n, s in target_summary.items()},
        'files': {
            'setup': str(OUT_DIR / "setup.npz"),
            'so3_grid': str(OUT_DIR / "so3_grid.npz"),
            'residual_kernel': str(OUT_DIR / "residual_kernel.npz"),
            'cost_variants': str(OUT_DIR / "cost_variants.npz"),
            'target_scores': str(OUT_DIR / "target_scores.npz"),
            'plots': [str(p1), str(p2), str(p3), str(p4)],
            'log': str(log_path),
        },
    }
    atomic_json_save(OUT_DIR / "summary.json", summary)
    print(f"\nSaved: {OUT_DIR / 'summary.json'}")
    print(f"\nTotal runtime: {time.time() - t_all:.1f}s")

    log_file.close()


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
