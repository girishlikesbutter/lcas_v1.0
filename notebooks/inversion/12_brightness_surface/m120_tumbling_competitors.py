#!/usr/bin/env python3
"""
m120 -- Tumbling-competitor test for surrogate-attitude-isoshell (seed 14).

Successor to m119v2. The v2 POC showed truth ranks 0/60000 in a SO(3) grid
of STATIC rotations (ω=0). That test is near-tautological: a tumbling satellite
is compared against candidates that don't tumble, so truth beats them trivially
whenever any non-trivial residual exists.

This script replaces the competitor pool with TUMBLING candidates: each is a
(q0, ω) pair that gets propagated through all 500 obs_times, scored by the same
13 surrogate-based cost variants as v2. This is the first honest test of the
surrogate-attitude-isoshell framing.

Pool composition (~10000 candidates):
  - known:           truth + m115 seed-14 basins [b0, b1, b2]
  - uniform:         q0 ~ super-Fibonacci SO(3); ω ~ uniform S^2 x U[0.5|ω|, 2|ω|]
  - close-ω:         ω = ω_true + dir perturb 2° + mag perturb 5%; q0 uniform
  - close-q0:        q0 = q0_true rotated by up to 10° random axis; ω uniform
  - near-truth:      both perturbed at {2°, 5°, 10°, 20°} scales

For each candidate: propagate via src.dynamics.attitude_propagator, compute
surrogate magnitude at constraint epochs, residual = pred - observed_mag_ce.
Then score 13 cost variants (same as v119v2).

Pass criteria
-------------
- truth ranks in top 0.1% under mean_L1 OR soft_pass_020
- truth beats known wrong-basin competitors (b1, b2) under at least one variant

Env
---
MICRO120_SEED    (14)      seed to run (seed_014 has all infra; other seeds would
                           need their own m119v2 setup.npz)
MICRO120_POOL    (8)       multiprocessing pool size
MICRO120_FORCE   (0)       1 = ignore cache and re-run
MICRO120_N_POOL  (10000)   total competitor pool size (excluding known)

Outputs (data/results/inversion_diagnostics/m120/seed_014/):
  candidates.npz         q0s [N,4], omegas [N,3], labels [N], bucket_names
  residuals.npz          residual[N, 500] float32 + observed_mag
  cost_variants.npz      score_<variant>[N] x13 + descending map
  summary.json           ranks, score tables, pass/fail
  run.log                combined stdout/stderr
  plots/rank_histogram.png
  plots/scores_truth_vs_competitors.png
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

from src.dynamics.attitude_propagator import propagate_attitude
from surrogate import SurrogateModel


# ---- Config --------------------------------------------------------------
SEED = int(os.environ.get('MICRO120_SEED', '14'))
POOL_SIZE = int(os.environ.get('MICRO120_POOL', '8'))
FORCE = os.environ.get('MICRO120_FORCE', '0') == '1'
N_POOL = int(os.environ.get('MICRO120_N_POOL', '10000'))

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
SETUP_NPZ = RESULTS_DIR / "m119v2" / f"seed_{SEED:03d}" / "setup.npz"
M115_NPZ = (RESULTS_DIR / "m115_surrogate_pipeline"
            / f"seed_{SEED:03d}" / "step2_hifi.npz")

OUT_BASE = RESULTS_DIR / "m120"
OUT_DIR = OUT_BASE / f"seed_{SEED:03d}"
PLOT_DIR = OUT_DIR / "plots"

SURROGATE_WEIGHTS = Path('/home/girish/surrogate_model/s10_5M_weights.npz')
SURROGATE_NORM = Path('/home/girish/surrogate_model/s10_5M_normalization.npz')

PANEL_DEG = 0.0
DISH_DEG = 15.0

RNG_SEED = 12345

# Pool composition fractions (will be scaled to N_POOL)
FRAC_UNIFORM = 0.30
FRAC_CLOSE_OMEGA = 0.20
FRAC_CLOSE_Q0 = 0.20
FRAC_NEAR_TRUTH = 0.30

# Cost variant params (same as m119v2)
COST_SIGMA_COUNT = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
COST_SIGMA_SOFT = [0.10, 0.20, 0.50]


# ---- Logging -------------------------------------------------------------
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data); f.flush()
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


# ---- Quaternion / geometry helpers --------------------------------------
def wxyz_to_xyzw(q):
    if q.ndim == 1:
        return np.array([q[1], q[2], q[3], q[0]])
    return q[:, [1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    if q.ndim == 1:
        return np.array([q[3], q[0], q[1], q[2]])
    return q[:, [3, 0, 1, 2]]


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def axis_angle_to_quat_wxyz(axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    ha = angle_rad / 2.0
    return np.array([np.cos(ha), *(np.sin(ha) * axis)])


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


def super_fibonacci_quats(n, rng_offset=0):
    """Alexa 2022 super-Fibonacci. Returns xyzw, shape (n, 4)."""
    i = np.arange(n, dtype=np.float64) + rng_offset
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


def uniform_s2(n, rng):
    """n uniform unit vectors on S^2."""
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def random_small_rotation_quats(n, max_deg, rng):
    """n wxyz quaternions with rotation angle drawn uniformly in [0, max_deg]."""
    axes = uniform_s2(n, rng)
    angles_rad = np.deg2rad(rng.uniform(0.0, max_deg, n))
    ha = angles_rad / 2.0
    out = np.zeros((n, 4))
    out[:, 0] = np.cos(ha)
    out[:, 1:] = np.sin(ha)[:, None] * axes
    return out


# ---- Cost variants (same as m119v2) ---------------------------------
def variant_keys():
    keys = ['mean_L2', 'mean_L1', 'max_abs']
    for s in COST_SIGMA_COUNT:
        keys.append(f'count_pass_{int(round(s*100)):03d}')
    for s in COST_SIGMA_SOFT:
        keys.append(f'soft_pass_{int(round(s*100)):03d}')
    return keys


def score_variant(name, residual):
    abs_r = np.abs(residual)
    if name == 'mean_L2':
        return np.mean(residual ** 2, axis=-1), False
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


# ---- Pool building -------------------------------------------------------
def build_pool(truth_q0, truth_omega0, m115_basins, rng):
    """
    Returns (q0s [N,4], omegas [N,3], labels [N], bucket_names, known_indices)

    labels: 0=known, 1=uniform, 2=close_omega, 3=close_q0, 4=near_truth
    """
    bucket_names = ['known', 'uniform', 'close_omega', 'close_q0', 'near_truth']

    # Scaled sizes
    n_uniform = int(N_POOL * FRAC_UNIFORM)
    n_close_w = int(N_POOL * FRAC_CLOSE_OMEGA)
    n_close_q = int(N_POOL * FRAC_CLOSE_Q0)
    n_near = N_POOL - n_uniform - n_close_w - n_close_q
    print(f"  pool buckets: uniform={n_uniform}, close_omega={n_close_w}, "
          f"close_q0={n_close_q}, near_truth={n_near}")

    w_mag_true = np.linalg.norm(truth_omega0)

    # --- Known (truth + m115 basins) ---
    known_q0s = [truth_q0]
    known_ws = [truth_omega0]
    known_labels_sub = ['truth']
    for i, b in enumerate(m115_basins):
        known_q0s.append(np.array(b['q0_wxyz']))
        known_ws.append(np.array(b['omega_rad']))
        known_labels_sub.append(f"m115_b{i}")
    known_q0s = np.array(known_q0s)
    known_ws = np.array(known_ws)
    n_known = len(known_q0s)
    print(f"  known candidates: {n_known} -- {known_labels_sub}")

    # --- Uniform 6-DOF ---
    uq_xyzw = super_fibonacci_quats(n_uniform)
    # randomize phase to avoid overlap with m119v2 grid
    perm = rng.permutation(n_uniform)
    uq_xyzw = uq_xyzw[perm]
    uq = xyzw_to_wxyz(uq_xyzw)
    uw_dirs = uniform_s2(n_uniform, rng)
    uw_mags = rng.uniform(0.5 * w_mag_true, 2.0 * w_mag_true, n_uniform)
    uw = uw_dirs * uw_mags[:, None]

    # --- Close-omega wrong-q0 ---
    cwq_xyzw = super_fibonacci_quats(n_close_w, rng_offset=n_uniform)
    perm = rng.permutation(n_close_w)
    cwq_xyzw = cwq_xyzw[perm]
    cwq = xyzw_to_wxyz(cwq_xyzw)
    # ω = ω_true + dir perturb ~2°, mag perturb ~5%
    dir_perturb_axes = uniform_s2(n_close_w, rng)
    dir_perturb_angs = np.deg2rad(rng.normal(0.0, 2.0, n_close_w))
    # rotate w_true direction by small rotation
    w_true_dir = truth_omega0 / w_mag_true
    # create small rotations
    rotvecs = dir_perturb_axes * dir_perturb_angs[:, None]
    Rs = Rotation.from_rotvec(rotvecs).as_matrix()
    cw_dirs = np.einsum('nij,j->ni', Rs, w_true_dir)
    cw_mags = w_mag_true * (1.0 + rng.normal(0.0, 0.05, n_close_w))
    cw = cw_dirs * cw_mags[:, None]

    # --- Close-q0 wrong-ω ---
    perturb_q = random_small_rotation_quats(n_close_q, 10.0, rng)  # up to 10°
    cqq = np.zeros((n_close_q, 4))
    for i in range(n_close_q):
        # truth_q0 LEFT-multiplied by perturbation in body-frame style
        cqq[i] = quat_mul_wxyz(perturb_q[i], truth_q0)
    cqw_dirs = uniform_s2(n_close_q, rng)
    cqw_mags = rng.uniform(0.5 * w_mag_true, 2.0 * w_mag_true, n_close_q)
    cqw = cqw_dirs * cqw_mags[:, None]

    # --- Near-truth perturbations at varying scales ---
    scales_deg = np.array([2.0, 5.0, 10.0, 20.0])
    per_scale = n_near // len(scales_deg)
    rest = n_near - per_scale * len(scales_deg)
    ntq = np.zeros((n_near, 4))
    ntw = np.zeros((n_near, 3))
    idx = 0
    for si, s_deg in enumerate(scales_deg):
        m = per_scale + (rest if si == len(scales_deg) - 1 else 0)
        # q perturbations at this scale
        pert = random_small_rotation_quats(m, s_deg, rng)
        for i in range(m):
            ntq[idx + i] = quat_mul_wxyz(pert[i], truth_q0)
        # ω perturbations at same scale (direction + mag)
        w_pert_axes = uniform_s2(m, rng)
        w_pert_angs = np.deg2rad(rng.uniform(0.0, s_deg, m))
        w_rotvecs = w_pert_axes * w_pert_angs[:, None]
        R_w = Rotation.from_rotvec(w_rotvecs).as_matrix()
        w_dirs = np.einsum('nij,j->ni', R_w, w_true_dir)
        # magnitude perturbation proportional to scale (scale_deg % as rel mag)
        rel_mag = np.deg2rad(s_deg) * 0.5  # small mag change
        w_mags = w_mag_true * (1.0 + rng.normal(0.0, rel_mag, m))
        ntw[idx:idx + m] = w_dirs * w_mags[:, None]
        idx += m

    # --- Assemble ---
    q0s = np.concatenate([known_q0s, uq, cwq, cqq, ntq], axis=0)
    omegas = np.concatenate([known_ws, uw, cw, cqw, ntw], axis=0)
    labels = np.concatenate([
        np.zeros(n_known, dtype=np.int32),
        np.full(n_uniform, 1, dtype=np.int32),
        np.full(n_close_w, 2, dtype=np.int32),
        np.full(n_close_q, 3, dtype=np.int32),
        np.full(n_near, 4, dtype=np.int32),
    ])

    # Normalize quaternions
    q0s = q0s / np.linalg.norm(q0s, axis=1, keepdims=True)

    known_indices = list(range(n_known))
    return q0s, omegas, labels, bucket_names, known_indices, known_labels_sub


# ---- Worker for residual computation ------------------------------------
_WORKER = {}


def _init_worker(k1_j2000_ce, k2_j2000_ce, obs_dist_ce, observed_mag_ce,
                 obs_times, constraint_epochs, inertia_tensor,
                 weights_path, norm_path):
    _WORKER['k1'] = k1_j2000_ce
    _WORKER['k2'] = k2_j2000_ce
    _WORKER['dist'] = obs_dist_ce
    _WORKER['obs'] = observed_mag_ce.astype(np.float32)
    _WORKER['times'] = obs_times
    _WORKER['ce'] = constraint_epochs
    _WORKER['I'] = inertia_tensor
    _WORKER['model'] = SurrogateModel(weights_path, norm_path)


def _propagate_and_score(args):
    """
    args = (chunk_start, q0s_chunk, omegas_chunk)
    Returns (chunk_start, residual_chunk [M, E] float32)
    """
    chunk_start, q0s_chunk, omegas_chunk = args
    k1 = _WORKER['k1']
    k2 = _WORKER['k2']
    dist = _WORKER['dist']
    obs = _WORKER['obs']
    times = _WORKER['times']
    ce = _WORKER['ce']
    I_t = _WORKER['I']
    model = _WORKER['model']

    M = len(q0s_chunk)
    E = len(ce)
    out = np.zeros((M, E), dtype=np.float32)
    zeros_E = np.zeros(E, dtype=np.float64)
    dish_E = np.full(E, DISH_DEG, dtype=np.float64)
    dist_E = dist.astype(np.float64)

    for i in range(M):
        q0 = q0s_chunk[i]
        w0 = omegas_chunk[i]
        try:
            quats, _ = propagate_attitude(q0, w0, times, "tumbling", I_t)
        except Exception as e:
            # Extremely rare; log and fill with large residual
            out[i, :] = 10.0
            continue
        q_ce = quats[ce]
        q_ce_xyzw = q_ce[:, [1, 2, 3, 0]]
        R_ce = Rotation.from_quat(q_ce_xyzw).as_matrix()
        k1_body = np.einsum('nij,nj->ni', R_ce, k1)
        k2_body = np.einsum('nij,nj->ni', R_ce, k2)
        k1_body /= np.linalg.norm(k1_body, axis=1, keepdims=True)
        k2_body /= np.linalg.norm(k2_body, axis=1, keepdims=True)
        pred = model.predict_magnitude(k1_body, k2_body, zeros_E, dish_E, dist_E)
        out[i, :] = (pred.astype(np.float32) - obs)

    return (chunk_start, out)


# ---- Stage A: load setup + m115 basins ---------------------------------
def stage_A_load():
    print("\n[Stage A] Loading m119v2 setup + m115 basins")
    t0 = time.time()
    if not SETUP_NPZ.exists():
        raise FileNotFoundError(
            f"m119v2 setup not found at {SETUP_NPZ}. "
            f"Run m119v2_attitude_isoshell.py first (seed {SEED}).")
    setup = dict(np.load(SETUP_NPZ, allow_pickle=False))
    print(f"  loaded {SETUP_NPZ}")
    print(f"  constraint_epochs: {len(setup['constraint_epochs'])} "
          f"(range {setup['constraint_epochs'][0]}..{setup['constraint_epochs'][-1]})")
    print(f"  obs_times: dt={setup['obs_times'][1]-setup['obs_times'][0]:.2f} s, "
          f"span={setup['obs_times'][-1]:.1f} s")
    print(f"  truth_q0={setup['truth_q0']}")
    print(f"  truth_omega0={setup['truth_omega0']} "
          f"(|w|={np.linalg.norm(setup['truth_omega0']):.5f})")

    m115_basins = []
    if M115_NPZ.exists():
        d = dict(np.load(M115_NPZ, allow_pickle=True))
        m115_basins = json.loads(str(d['hifi_json']))
        print(f"  loaded {len(m115_basins)} m115 basins from {M115_NPZ}")
        for i, b in enumerate(m115_basins):
            print(f"    b{i}: q0_err={b['q0_err']:.2f}° "
                  f"w_dir={b['w_dir_err']:.2f}° "
                  f"hifi_mse={b['hifi_mse']:.4f} twin={b['is_twin']}")
    else:
        print(f"  WARN: {M115_NPZ} not found; skipping m115 basins")

    print(f"  Stage A done in {time.time()-t0:.1f}s")
    return setup, m115_basins


# ---- Stage B: build candidate pool --------------------------------------
def stage_B_pool(setup, m115_basins):
    print("\n[Stage B] Building candidate pool")
    t0 = time.time()
    out = OUT_DIR / "candidates.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=True))
        bucket_names = [str(x) for x in d['bucket_names']]
        known_labels_sub = [str(x) for x in d['known_labels_sub']]
        print(f"  N_cand={len(d['q0s'])}, known={len(d['known_indices'])}")
        print(f"  Stage B (cached) done in {time.time()-t0:.1f}s")
        return d, bucket_names, known_labels_sub

    rng = np.random.default_rng(RNG_SEED)
    q0s, omegas, labels, bucket_names, known_indices, known_labels_sub = build_pool(
        truth_q0=setup['truth_q0'],
        truth_omega0=setup['truth_omega0'],
        m115_basins=m115_basins,
        rng=rng,
    )
    print(f"  total candidates: {len(q0s)}")
    print(f"  per-bucket counts:")
    for bi, bn in enumerate(bucket_names):
        cnt = int(np.sum(labels == bi))
        print(f"    {bi} {bn:15s}: {cnt}")

    np.savez(
        out,
        q0s=q0s.astype(np.float64),
        omegas=omegas.astype(np.float64),
        labels=labels.astype(np.int32),
        bucket_names=np.array(bucket_names),
        known_indices=np.array(known_indices, dtype=np.int32),
        known_labels_sub=np.array(known_labels_sub),
    )
    print(f"  saved {out}")
    print(f"  Stage B done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=True)), bucket_names, known_labels_sub


# ---- Stage C: propagate + surrogate + residual --------------------------
def stage_C_residuals(setup, pool):
    print(f"\n[Stage C] Propagating + scoring residuals (POOL_SIZE={POOL_SIZE})")
    t0 = time.time()
    out = OUT_DIR / "residuals.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=False))
        print(f"  residual shape={d['residual'].shape} dtype={d['residual'].dtype}")
        print(f"  Stage C (cached) done in {time.time()-t0:.1f}s")
        return d

    q0s = pool['q0s']
    omegas = pool['omegas']
    N = len(q0s)
    E = len(setup['constraint_epochs'])
    print(f"  N_cand={N}, N_epochs={E}, allocating residual[{N},{E}] float32 "
          f"(~{N*E*4/1e6:.1f} MB)")

    # chunk
    chunk_size = max(32, N // (POOL_SIZE * 8))
    chunks = []
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        chunks.append((i, q0s[i:j], omegas[i:j]))
    print(f"  {len(chunks)} work chunks of up to {chunk_size} candidates")

    residual = np.zeros((N, E), dtype=np.float32)

    init_args = (
        setup['k1_j2000_ce'].astype(np.float64),
        setup['k2_j2000_ce'].astype(np.float64),
        setup['obs_dist_ce'].astype(np.float64),
        setup['observed_mag_ce'].astype(np.float64),
        setup['obs_times'].astype(np.float64),
        setup['constraint_epochs'].astype(np.int64),
        setup['inertia_tensor'].astype(np.float64),
        str(SURROGATE_WEIGHTS), str(SURROGATE_NORM),
    )

    t_pool = time.time()
    with mp.Pool(POOL_SIZE, initializer=_init_worker, initargs=init_args) as pool_:
        done = 0
        for chunk_start, block in pool_.imap_unordered(_propagate_and_score, chunks):
            residual[chunk_start:chunk_start + len(block)] = block
            done += len(block)
            if done % (5 * chunk_size) == 0 or done == N:
                elapsed = time.time() - t_pool
                rate = done / max(elapsed, 1e-3)
                remaining = (N - done) / max(rate, 1e-3)
                print(f"    {done}/{N} done, {rate:.1f} cand/s, "
                      f"~{remaining:.0f} s remaining")

    print(f"  pool finished in {time.time()-t_pool:.1f}s")
    abs_r = np.abs(residual)
    print(f"  residual stats: median|r|={np.median(abs_r):.4f}, "
          f"p90|r|={np.percentile(abs_r,90):.4f}, "
          f"max|r|={np.max(abs_r):.4f}")

    np.savez(
        out,
        residual=residual,
        observed_mag=setup['observed_mag_ce'].astype(np.float32),
    )
    print(f"  saved {out} (~{out.stat().st_size/1e6:.1f} MB)")
    print(f"  Stage C done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=False))


# ---- Stage D: cost variants + ranking ----------------------------------
def stage_D_variants(residuals):
    print("\n[Stage D] Cost variants")
    t0 = time.time()
    residual = residuals['residual']
    variants = variant_keys()

    save_dict = {}
    desc_map = {}
    for name in variants:
        scores, descending = score_variant(name, residual)
        save_dict[f'score_{name}'] = scores.astype(np.float32)
        desc_map[name] = descending
        if descending:
            print(f"  {name:25s} (higher=better) "
                  f"max={scores.max():.4f}, min={scores.min():.4f}")
        else:
            print(f"  {name:25s} (lower=better)  "
                  f"min={scores.min():.6f}, max={scores.max():.6f}")

    save_dict['variant_names'] = np.array(variants)
    save_dict['descending'] = np.array(
        [desc_map[v] for v in variants], dtype=bool)

    out = OUT_DIR / "cost_variants.npz"
    np.savez(out, **save_dict)
    print(f"  saved {out}")
    print(f"  Stage D done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=True)), desc_map, variants


# ---- Stage E: analyze ranks (truth + known competitors) ----------------
def stage_E_analyze(pool, residuals, variants_data, desc_map, variant_list,
                    known_labels_sub, bucket_names):
    print("\n[Stage E] Analysis")
    t0 = time.time()
    q0s = pool['q0s']
    omegas = pool['omegas']
    labels = pool['labels']
    known_indices = pool['known_indices']
    truth_q0 = q0s[0]
    truth_omega0 = omegas[0]
    N = len(q0s)

    # For every known candidate, report rank under every variant
    print(f"\n  Truth is candidate index 0 of {N}")
    print(f"  Known candidates: {list(known_labels_sub)}")

    rank_table = {}  # rank_table[label][variant] = rank
    score_table = {}  # score_table[label][variant] = score
    for k_idx, k_label in zip(known_indices, known_labels_sub):
        rank_table[k_label] = {}
        score_table[k_label] = {}
        for v in variant_list:
            scores = variants_data[f'score_{v}']
            descending = desc_map[v]
            target = float(scores[int(k_idx)])
            if descending:
                # higher=better, rank = # with strictly higher score + 1
                rank = int(np.sum(scores > target))
            else:
                rank = int(np.sum(scores < target))
            rank_table[k_label][v] = rank
            score_table[k_label][v] = target

    # print rank table
    print(f"\n  Rank of known candidates under each variant (0 = best):")
    header = f"  {'variant':25s}" + "".join(
        f" {lbl:>10s}" for lbl in known_labels_sub)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for v in variant_list:
        row = f"  {v:25s}"
        for lbl in known_labels_sub:
            row += f" {rank_table[lbl][v]:>10d}"
        print(row)

    # Per-candidate summary for known
    known_summary = {}
    for k_idx, k_label in zip(known_indices, known_labels_sub):
        q = q0s[int(k_idx)]
        w = omegas[int(k_idx)]
        known_summary[k_label] = {
            'q0_err_to_truth_deg': attitude_error_deg(q, truth_q0),
            'w_dir_err_deg': omega_dir_err(w, truth_omega0),
            'w_mag_err_pct': omega_mag_err_pct(w, truth_omega0),
            'ranks': rank_table[k_label],
            'scores': {v: float(score_table[k_label][v]) for v in variant_list},
        }

    # Pass criteria
    truth_ranks = rank_table['truth']
    truth_top_pct = {v: 100.0 * (truth_ranks[v] / N) for v in variant_list}
    print(f"\n  Truth percentile (lower = better, 0 = best):")
    for v in variant_list:
        pct = truth_top_pct[v]
        print(f"    {v:25s} rank={truth_ranks[v]:>6d}/{N} = {pct:.3f}%")

    # Best variant for truth
    best_v_for_truth = min(variant_list, key=lambda v: truth_ranks[v])
    print(f"\n  Truth's best variant: {best_v_for_truth} "
          f"(rank {truth_ranks[best_v_for_truth]}/{N})")

    pass_top01 = any(truth_ranks[v] < N * 0.001 for v in variant_list)
    pass_top1 = any(truth_ranks[v] < N * 0.01 for v in variant_list)

    # Truth beats known wrong basins check
    wrong_labels = [lbl for lbl in known_labels_sub
                    if lbl != 'truth' and
                    known_summary[lbl]['q0_err_to_truth_deg'] > 90.0]
    truth_beats_all_wrong = {}
    for v in variant_list:
        t_r = truth_ranks[v]
        beats_all = all(t_r < rank_table[lbl][v] for lbl in wrong_labels)
        truth_beats_all_wrong[v] = beats_all
    pass_beats_wrong = any(truth_beats_all_wrong.values())

    print(f"\n  Wrong-basin known competitors (q0_err > 90°): {wrong_labels}")
    print(f"  Truth beats all wrong under variant (any): "
          f"{pass_beats_wrong}")
    if pass_beats_wrong:
        variants_ok = [v for v in variant_list if truth_beats_all_wrong[v]]
        print(f"    variants where truth wins: {variants_ok[:5]}"
              f"{'...' if len(variants_ok) > 5 else ''}")

    # Per-bucket best rank
    print(f"\n  Per-bucket: best (lowest) rank of any candidate in bucket under each variant:")
    bucket_best = {}
    for bi, bn in enumerate(bucket_names):
        bmask = labels == bi
        if not bmask.any():
            continue
        bucket_best[bn] = {}
        for v in variant_list:
            scores = variants_data[f'score_{v}']
            desc = desc_map[v]
            sub = scores[bmask]
            if desc:
                # higher = better, we want to know the RANK of the bucket's best
                bst = sub.max()
                rank_best = int(np.sum(scores > bst))
            else:
                bst = sub.min()
                rank_best = int(np.sum(scores < bst))
            bucket_best[bn][v] = rank_best
    # print a compact summary for 3 headline variants
    headline = ['mean_L1', 'mean_L2', 'soft_pass_020']
    print(f"  bucket         | " + " | ".join(f"{v:>15s}" for v in headline))
    for bn in bucket_names:
        if bn not in bucket_best:
            continue
        row = f"  {bn:14s}"
        for v in headline:
            row += f" | {bucket_best[bn].get(v, -1):>15d}"
        print(row)

    # Cast-safe summary
    summary = {
        'seed': SEED,
        'N_cand': int(N),
        'truth_idx': 0,
        'known_labels': list(known_labels_sub),
        'bucket_names': list(bucket_names),
        'variant_list': list(variant_list),
        'truth_ranks': {v: int(truth_ranks[v]) for v in variant_list},
        'truth_percentile': truth_top_pct,
        'truth_best_variant': best_v_for_truth,
        'truth_best_rank': int(truth_ranks[best_v_for_truth]),
        'known_summary': {
            lbl: {
                'q0_err_to_truth_deg': float(info['q0_err_to_truth_deg']),
                'w_dir_err_deg': float(info['w_dir_err_deg']),
                'w_mag_err_pct': float(info['w_mag_err_pct']),
                'ranks': {k: int(r) for k, r in info['ranks'].items()},
                'scores': info['scores'],
            }
            for lbl, info in known_summary.items()
        },
        'truth_beats_all_wrong': {v: bool(b)
                                  for v, b in truth_beats_all_wrong.items()},
        'pass_top_0_1_pct': bool(pass_top01),
        'pass_top_1_pct': bool(pass_top1),
        'pass_beats_wrong_any_variant': bool(pass_beats_wrong),
        'wrong_basin_labels': wrong_labels,
        'bucket_best_rank': {bn: {v: int(r) for v, r in bb.items()}
                             for bn, bb in bucket_best.items()},
    }

    atomic_json_save(OUT_DIR / "summary.json", summary)
    print(f"\n  saved {OUT_DIR / 'summary.json'}")

    print(f"  Stage E done in {time.time()-t0:.1f}s")
    return summary


# ---- Stage F: plots -----------------------------------------------------
def plot_rank_histogram(variants_data, desc_map, variant_list, truth_idx):
    print("\n[Stage F.1] rank_histogram.png")
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()
    for i, v in enumerate(variant_list):
        if i >= len(axes):
            break
        ax = axes[i]
        scores = variants_data[f'score_{v}']
        ax.hist(scores, bins=80, color='steelblue', alpha=0.7)
        truth_score = scores[truth_idx]
        ax.axvline(truth_score, color='red', lw=2, label=f'truth={truth_score:.4f}')
        ax.set_title(v, fontsize=9)
        ax.set_yscale('log')
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(len(variant_list), len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    out = PLOT_DIR / 'rank_histogram.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_scores_truth_vs_competitors(variants_data, pool, summary, variant_list,
                                     known_labels_sub):
    print("\n[Stage F.2] scores_truth_vs_competitors.png")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    labels = pool['labels']
    bucket_names = [str(x) for x in pool['bucket_names']]
    cmap = plt.get_cmap('tab10')

    # Focus on one headline variant: mean_L1
    v = 'mean_L1'
    scores = variants_data[f'score_{v}']
    # scatter by bucket
    for bi, bn in enumerate(bucket_names):
        m = labels == bi
        if not m.any():
            continue
        ax.scatter(np.arange(len(scores))[m], scores[m],
                   c=[cmap(bi)], s=5, alpha=0.5, label=bn)
    # annotate known
    for k_idx, k_label in zip(pool['known_indices'], known_labels_sub):
        ax.scatter([int(k_idx)], [scores[int(k_idx)]],
                   c='red' if k_label == 'truth' else 'black',
                   s=60, zorder=10, edgecolor='white', lw=1)
        ax.annotate(k_label, (int(k_idx), scores[int(k_idx)]),
                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('candidate index')
    ax.set_ylabel(f'{v} (lower = better)')
    ax.set_title(f'Seed {SEED} — {v} score per candidate, known annotated')
    ax.legend(loc='best', fontsize=8, markerscale=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = PLOT_DIR / 'scores_truth_vs_competitors.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---- Main ---------------------------------------------------------------
def main():
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "run.log"
    log_f = open(log_path, 'w')
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    try:
        t_global = time.time()
        print(f"=" * 70)
        print(f"m120 -- tumbling competitors (seed {SEED})")
        print(f"=" * 70)
        print(f"  OUT_DIR={OUT_DIR}")
        print(f"  N_POOL={N_POOL}, POOL_SIZE={POOL_SIZE}, FORCE={FORCE}")

        setup, m115_basins = stage_A_load()
        pool, bucket_names, known_labels_sub = stage_B_pool(setup, m115_basins)
        residuals = stage_C_residuals(setup, pool)
        variants_data, desc_map, variant_list = stage_D_variants(residuals)
        summary = stage_E_analyze(
            pool, residuals, variants_data, desc_map, variant_list,
            known_labels_sub, bucket_names)
        plot_rank_histogram(variants_data, desc_map, variant_list, truth_idx=0)
        plot_scores_truth_vs_competitors(
            variants_data, pool, summary, variant_list, known_labels_sub)

        dt = time.time() - t_global
        print(f"\n{'='*70}")
        print(f"TOTAL TIME: {dt:.1f}s")
        print(f"PASS top 0.1%:           {summary['pass_top_0_1_pct']}")
        print(f"PASS top 1%:             {summary['pass_top_1_pct']}")
        print(f"PASS beats-wrong (any):  {summary['pass_beats_wrong_any_variant']}")
        print(f"Truth best variant:      {summary['truth_best_variant']} "
              f"(rank {summary['truth_best_rank']}/{summary['N_cand']})")
        print(f"{'='*70}")
    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == '__main__':
    main()
