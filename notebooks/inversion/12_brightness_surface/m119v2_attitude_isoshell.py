#!/usr/bin/env python3
"""
m119v2 -- Attitude isoshell POC on seed 14, TIME-MISMATCH BUG FIX.

THIS IS THE FIX for the retracted m119_attitude_isoshell.py. See
m119_BUG.md for the full diagnosis. Summary: the original v1 script
mixed (a) the m046 pre-computed dataset sampled at 1-hour window /
7.21 s per epoch with (b) setup_experiment's current 6-hour window /
43.29 s per epoch, so epoch index i meant two different wall times. k2
was off by up to 74 deg at epoch 499, breaking all truth-rank claims.

This v2 rebuilds Stage A from scratch. Stage A takes truth q0 and omega0
scalars out of m046 (safe: not time-dependent), hands them to the lib
via the NEW true_q0_wxyz / true_omega0_rad kwargs of setup_experiment,
and derives EVERY per-epoch array (k1, k2, obs_dist, obs_times,
observed_lc) from the same setup_experiment call. All 500 epochs are used
as constraint epochs (no m046-derived subset). Stages B/C/D/F are
structurally unchanged - only the kernel width grows from 255 to 500.

Stage E targets are reduced to 'truth' only: m115 basins and m118
competitors would be propagated on inconsistent obs_times and would give
misleading attitudes - they are not our comparison baseline here. The
experiment's only question is: with consistent geometry, what is truth's
honest rank in the 60000-candidate super-Fibonacci SO(3) grid under each
of 13 cost variants?

Output dir: data/results/inversion_diagnostics/m119v2/seed_{SEED:03d}/
(NOT m119 - the v1 tree stays as provenance of the retracted claim.)

Usage
-----
    python3 m119v2_attitude_isoshell.py
    MICRO119_FORCE=1 python3 m119v2_attitude_isoshell.py
    MICRO119_N_SO3=5000 MICRO119_POOL=4 python3 m119v2_attitude_isoshell.py

Env vars (same as v1): MICRO119_SEED (14), MICRO119_POOL (8),
MICRO119_FORCE (0), MICRO119_N_SO3 (60000).
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

OUT_BASE = RESULTS_DIR / "m119v2"
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
POC_TOP_PERCENTILE = 0.001  # 0.1%


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


# ---- Quaternion helpers (same as v1) -------------------------------------
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


# ---- Super-Fibonacci SO(3) grid (same as v1) ----------------------------
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


# ---- Stage C worker (same as v1) ----------------------------------------
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


# ---- Target scoring helpers (same as v1) --------------------------------
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
    q_ce = quats[constraint_epochs]  # [E, 4] wxyz
    R_ce = Rotation.from_quat(wxyz_to_xyzw(q_ce)).as_matrix()  # [E,3,3]
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
        order = np.argsort(-scores, kind='stable')
    else:
        order = np.argsort(scores, kind='stable')
    return order[:k].astype(np.int64)


# ---- Stage A: REWRITTEN for consistent geometry -------------------------
def stage_A_setup():
    print("\n[Stage A] Setup (v2: consistent geometry via setup_experiment)")
    t0 = time.time()
    out = OUT_DIR / "setup.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=False))
        print(f"  Stage A (cached) done in {time.time()-t0:.1f}s")
        return d

    # 1. Load seed-parameter scalars from m046 (safe: NOT time-dependent)
    print(f"  loading truth q0/omega0 for seed {SEED} from m046")
    m046 = np.load(MICRO46_NPZ)
    truth_q0 = m046['q0s'][SEED].astype(np.float64)       # (4,) wxyz
    truth_omega0 = m046['omega0s'][SEED].astype(np.float64)  # (3,) rad/s
    print(f"  truth_q0={truth_q0}")
    print(f"  truth_omega0={truth_omega0} rad/s (|w|={np.linalg.norm(truth_omega0):.5f})")

    # 2. Call setup_experiment WITH these as overrides. Everything else
    #    lib-native. skip_true_lc=False => ctx.true_lc and ctx.observed_lc
    #    populated via hi-fi pipeline on today's 6-hr window.
    print("  calling setup_experiment(500, skip_true_lc=False, "
          "true_q0_wxyz=..., true_omega0_rad=...)")
    ctx = setup_experiment(
        n_observations=500,
        noise_sigma=0.05,
        random_seed=42,
        skip_true_lc=False,
        true_q0_wxyz=truth_q0,
        true_omega0_rad=truth_omega0,
    )

    # 3. Derive consistent J2000 unit vectors
    k1_j2000 = ctx.sun_pos - ctx.sat_pos
    k1_j2000 /= np.linalg.norm(k1_j2000, axis=1, keepdims=True)
    k2_j2000 = ctx.obs_pos - ctx.sat_pos
    k2_j2000 /= np.linalg.norm(k2_j2000, axis=1, keepdims=True)

    # 4. Constraint epochs = ALL 500 (no more pab_j2000-based selection)
    constraint_epochs = np.arange(500, dtype=np.int64)

    # 5. Slice (trivial for all-500; keeps parity with v1 kernel code)
    k1_j2000_ce = k1_j2000[constraint_epochs]
    k2_j2000_ce = k2_j2000[constraint_epochs]
    obs_dist_ce = ctx.obs_dist[constraint_epochs]
    observed_mag_ce = ctx.observed_lc[constraint_epochs]
    obs_times = ctx.observation_times
    inertia_tensor = ctx.inertia_tensor.astype(np.float64)

    # 6. Sanity assert: dt is consistent (no mixing!)
    assert obs_times[-1] > 20000.0, (
        f"obs_times span {obs_times[-1]} s -- expected ~21600 s; "
        f"config may have changed")
    assert np.allclose(np.diff(obs_times), obs_times[1] - obs_times[0], rtol=1e-3)
    print(f"  obs_times: dt={obs_times[1]-obs_times[0]:.2f} s, "
          f"span={obs_times[-1]:.1f} s")
    print(f"  constraint_epochs: {len(constraint_epochs)} (all 500)")
    print(f"  observed_mag_ce: min={observed_mag_ce.min():.2f} "
          f"max={observed_mag_ce.max():.2f}")

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
    )
    print(f"  saved {out}")
    print(f"  Stage A done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=False))


# ---- Stage B: SO(3) grid (same as v1) -----------------------------------
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


# ---- Stage C: residual kernel (same logic, shape now [N_SO3, 500]) -----
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


# ---- Stage D: cost variants (same as v1) --------------------------------
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


# ---- Stage E: targets (simplified: truth only) --------------------------
def nearest_grid_R(q_target_wxyz, grid_quats_wxyz):
    """Per-epoch argmax |dot|. Returns idx and geodesic deg."""
    dots = np.einsum('ni,mi->nm', q_target_wxyz, grid_quats_wxyz)
    abs_dots = np.abs(dots)
    idx = np.argmax(abs_dots, axis=1)
    best = abs_dots[np.arange(len(q_target_wxyz)), idx]
    deg = 2.0 * np.rad2deg(np.arccos(np.clip(best, 0.0, 1.0)))
    return idx.astype(np.int64), deg.astype(np.float64)


def stage_E_targets(setup, grid, variants_data, descending_map, variant_list):
    print("\n[Stage E] Target scoring (truth-only)")
    t0 = time.time()

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

    # Truth is the sole target now. m115 basins + m118 rank-1 removed:
    # they're not propagated on today's obs_times and would give the wrong
    # attitudes; they're not the comparison baseline for this honest POC.
    targets = [('truth', truth_q0, truth_omega0)]

    save_dict = {}
    target_summary = {}

    grid_scores = {v: variants_data[f'score_{v}'] for v in variant_list}

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

        q0_err = attitude_error_deg(q0, truth_q0)
        w_derr = omega_dir_err(w, truth_omega0)
        w_merr = omega_mag_err_pct(w, truth_omega0)
        print(f"    q0_err={q0_err:.2f} deg, w_dir_err={w_derr:.2f} deg, "
              f"w_mag_err={w_merr:+.2f}%")

        g_idx, g_deg = nearest_grid_R(q_ce, grid_quats_wxyz)
        print(f"    grid quant deg: median={np.median(g_deg):.2f}, "
              f"p90={np.percentile(g_deg,90):.2f}")

        scores_per_variant = {}
        ranks = {}
        for v in variant_list:
            tgt_score, descending = score_variant(v, residual)
            tgt_score = float(tgt_score)
            grid_sc = grid_scores[v]
            if descending:
                rank = int(np.sum(grid_sc > tgt_score))
            else:
                rank = int(np.sum(grid_sc < tgt_score))
            scores_per_variant[v] = tgt_score
            ranks[v] = rank

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


# ---- Stage F: plots (same structure as v1) ------------------------------
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
        style = '-' if name == 'truth' else '--'
        lw = 2.5 if name == 'truth' else 1.5
        ax.plot(sigmas, frac, style, lw=lw, label=name, color=cmap(i % 10))
    ax.set_xscale('log')
    ax.set_xlabel('sigma [mag]')
    ax.set_ylabel('fraction of epochs passing |r|<sigma')
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
    fig.colorbar(im, ax=ax, label=f'|topK cap|/{TOP_K}')
    ax.set_title(f'pairwise top-{TOP_K} overlap across cost variants')
    fig.tight_layout()
    out = PLOT_DIR / 'topK_overlap.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_truth_rank(target_summary, variant_list):
    print("\n[Stage F.4] truth_rank.png")
    truth_ranks = [target_summary['truth']['ranks'][v] for v in variant_list]
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(variant_list) * 0.5), 6))
    x = np.arange(len(variant_list))
    ax.bar(x, truth_ranks, color='steelblue', label='truth rank')
    ax.axhline(N_SO3 * POC_TOP_PERCENTILE, color='red', linestyle='--',
               alpha=0.7, label=f'top {POC_TOP_PERCENTILE*100:.1f}% '
               f'(rank {int(N_SO3*POC_TOP_PERCENTILE)})')
    ax.axhline(N_SO3 // 2, color='grey', linestyle=':', alpha=0.5,
               label='mid-pack')
    ax.set_yscale('symlog')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_list, rotation=75, fontsize=8, ha='right')
    ax.set_ylabel(f'rank out of {N_SO3} (symlog)')
    ax.set_title('truth rank per cost variant (lower = better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    out = PLOT_DIR / 'truth_rank.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---- POC classification --------------------------------------------------
def classify_poc(target_summary, variant_list):
    truth = target_summary['truth']

    truth_in_top1000_any = any(
        truth['ranks'][v] < TOP_K for v in variant_list)

    min_rank = min(truth['ranks'][v] for v in variant_list)
    top_percentile_rank = min_rank / float(N_SO3)
    truth_top_percentile_pass = top_percentile_rank < POC_TOP_PERCENTILE

    checks = {
        'median_residual': truth['residual_median'] < POC_MEDIAN_MAX,
        'p90_residual': truth['residual_p90'] < POC_P90_MAX,
        'truth_in_top1000_any_variant': truth_in_top1000_any,
        'truth_rank_top_percentile': truth_top_percentile_pass,
    }
    verdict = 'POC PASS' if all(checks.values()) else 'POC FAIL'
    return verdict, checks, min_rank, top_percentile_rank


# ---- Main ----------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "run.log"
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"m119v2 -- attitude isoshell POC (TIME-MISMATCH FIX) on seed {SEED}")
    print(f"  see m119_BUG.md for background")
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
    p4 = plot_truth_rank(target_summary, variant_list)
    timings['stage_F_plots'] = time.time() - ts

    verdict, checks, min_rank, top_percentile_rank = classify_poc(
        target_summary, variant_list)

    print("\n" + "=" * 72)
    print(f"POC classification: {verdict}")
    for ck, passed in checks.items():
        mark = 'PASS' if passed else 'FAIL'
        print(f"  [{mark}] {ck}")
    print(f"  truth min rank = {min_rank} / {N_SO3} "
          f"({top_percentile_rank*100:.4f}%)")
    print("=" * 72)

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
        'truth_min_rank': int(min_rank),
        'truth_min_rank_percentile': float(top_percentile_rank),
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
