#!/usr/bin/env python3
"""
m121 -- Basin-width characterization of the surrogate-residual cost.

Hypothesis
----------
The surrogate-residual cost basin-of-attraction is much tighter than the 20°
upper bound suggested by m120. Expected:
  - at <1° perturbation, cost ~ surrogate noise floor (~0.055 mag mean_L1)
  - at 5-10° it discriminates clearly
  - cost is likely ELONGATED: wider along ω-magnitude than along ω-direction
    or q0-rotation axes (a 1% magnitude error accumulates less trajectory drift
    over 500 epochs than a 1° direction error).

Falsifiable: if cost at 20° is indistinguishable from cost at 0.25° → no local
basin structure → search integration infeasible with this cost.

Relationship to m120
------------------------
Direct cousin of m120_tumbling_competitors.py. Keeps the surrogate-eval
code path IDENTICAL so cost values are directly comparable. Differences:
  1. Pool composition replaces m120's {uniform, close-ω, close-q0,
     near-truth} buckets with axis-separated perturbations (q0-only,
     ω-direction-only, ω-magnitude-only, joint). This isolates the basin
     along each DOF independently.
  2. Cost variants are RESTRICTED to {mean_L1, mean_L2}. The m120 13-variant
     sweep is not required here: basin shape is about the rank statistic, not
     the soft/count-pass thresholding, so the extra 11 variants would cost
     ~1.5× compute for diminishing information. This scope-narrowing is a
     deviation from m120; see REPORT.md.

Pool composition (~6751 candidates)
-----------------------------------
  1     × truth (q0_true, ω_true)
  2000  × q0-only:        8 scales {0.1, 0.25, 0.5, 1, 2, 5, 10, 20}°   × 250
  2000  × ω-direction:    8 scales {0.1, 0.25, 0.5, 1, 2, 5, 10, 20}°   × 250
  1750  × ω-magnitude:    7 scales {0.25, 0.5, 1, 2, 5, 10, 20}%        × 250
  1000  × joint:          4 scales {0.25, 0.5, 1, 2}°                   × 250
                          (q0 rot + ω-dir rot + |ω| scale all at `scale`)

Axis enum
---------
  0 = 'truth'
  1 = 'q0'          scale in degrees
  2 = 'omega_dir'   scale in degrees
  3 = 'omega_mag'   scale in percent
  4 = 'joint'       scale in degrees (magnitude applied as percent of same value)

Env
---
MICRO121_SEED    (14)  must be one of {14, 27, 46} (existing setup.npz)
MICRO121_POOL    (8)   multiprocessing pool size
MICRO121_FORCE   (0)   1 = ignore cache and re-run

Outputs (data/results/inversion_diagnostics/m121/seed_NNN/):
  candidates.npz         q0s, omegas, bucket, axis (S-bytes), scale, labels
  residuals.npz          residual[N,500] float32 + observed_mag
  cost_variants.npz      score_mean_L1, score_mean_L2 + argsort maps
  summary.json           per-axis medians/p10/p90, truth_rank_mean_L1, etc.
  run.log
  plots/cost_vs_scale_per_axis.png
  plots/basin_boundary.png

Kill criteria: any single seed >15 min wall-clock.
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
SEED = int(os.environ.get('MICRO121_SEED', '14'))
POOL_SIZE = int(os.environ.get('MICRO121_POOL', '8'))
FORCE = os.environ.get('MICRO121_FORCE', '0') == '1'

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
SETUP_NPZ = RESULTS_DIR / "m119v2" / f"seed_{SEED:03d}" / "setup.npz"

OUT_BASE = RESULTS_DIR / "m121"
OUT_DIR = OUT_BASE / f"seed_{SEED:03d}"
PLOT_DIR = OUT_DIR / "plots"

SURROGATE_WEIGHTS = Path('/home/girish/surrogate_model/s10_5M_weights.npz')
SURROGATE_NORM = Path('/home/girish/surrogate_model/s10_5M_normalization.npz')

PANEL_DEG = 0.0
DISH_DEG = 15.0

RNG_SEED = 121_000 + SEED  # per-seed reproducibility

# Axis enum
AXIS_TRUTH = 0
AXIS_Q0 = 1
AXIS_OMEGA_DIR = 2
AXIS_OMEGA_MAG = 3
AXIS_JOINT = 4
AXIS_NAMES = {
    AXIS_TRUTH: 'truth',
    AXIS_Q0: 'q0',
    AXIS_OMEGA_DIR: 'omega_dir',
    AXIS_OMEGA_MAG: 'omega_mag',
    AXIS_JOINT: 'joint',
}

# Scale schedules
SCALES_DEG_Q0 = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SCALES_DEG_WDIR = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SCALES_PCT_WMAG = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SCALES_DEG_JOINT = [0.25, 0.5, 1.0, 2.0]
N_PER_SCALE = 250

# Variants (scope-narrowed vs m120)
VARIANT_LIST = ['mean_L1', 'mean_L2']


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


# ---- Quaternion / geometry helpers (match m120 conventions) ---------
def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def uniform_s2(n, rng):
    """n uniform unit vectors on S^2."""
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def axis_angle_to_quat_wxyz_batch(axes, angles_rad):
    ha = angles_rad / 2.0
    out = np.zeros((len(axes), 4))
    out[:, 0] = np.cos(ha)
    out[:, 1:] = np.sin(ha)[:, None] * (
        axes / np.linalg.norm(axes, axis=1, keepdims=True))
    return out


def perp_axes_to(vec, n, rng):
    """Return n unit vectors perpendicular to `vec`."""
    vec = vec / np.linalg.norm(vec)
    # Sample candidate axes, project out parallel component, renormalize.
    # Resample any degenerate near-zero cases.
    out = np.zeros((n, 3))
    filled = 0
    while filled < n:
        need = n - filled
        cand = uniform_s2(need * 2, rng)
        proj = cand @ vec
        perp = cand - proj[:, None] * vec[None, :]
        nrm = np.linalg.norm(perp, axis=1)
        good = nrm > 1e-6
        take = perp[good] / nrm[good, None]
        take = take[:need]
        out[filled:filled + len(take)] = take
        filled += len(take)
    return out


# ---- Cost variants -------------------------------------------------------
def score_variant(name, residual):
    abs_r = np.abs(residual)
    if name == 'mean_L2':
        return np.mean(residual ** 2, axis=-1), False
    if name == 'mean_L1':
        return np.mean(abs_r, axis=-1), False
    raise ValueError(f"Unknown variant: {name}")


# ---- Pool building -------------------------------------------------------
def build_pool(truth_q0, truth_omega0, rng):
    """
    Returns dict with q0s [N,4], omegas [N,3], bucket [N], axis (bytes) [N],
    scale [N], labels [N] (str), and N_total.
    """
    truth_q0 = np.asarray(truth_q0, dtype=np.float64)
    truth_omega0 = np.asarray(truth_omega0, dtype=np.float64)
    truth_q0 = truth_q0 / np.linalg.norm(truth_q0)
    w_mag_true = np.linalg.norm(truth_omega0)
    w_dir_true = truth_omega0 / w_mag_true

    all_q0 = []
    all_w = []
    all_bucket = []
    all_axis = []
    all_scale = []
    all_labels = []

    # --- 0. truth ---
    all_q0.append(truth_q0[None, :])
    all_w.append(truth_omega0[None, :])
    all_bucket.append(np.array([AXIS_TRUTH], dtype=np.int32))
    all_axis.append(np.array(['truth'], dtype='S12'))
    all_scale.append(np.array([0.0], dtype=np.float64))
    all_labels.append(np.array(['truth'], dtype='S32'))

    # --- 1. q0-only perturbations ---
    for s_deg in SCALES_DEG_Q0:
        axes = uniform_s2(N_PER_SCALE, rng)
        angles = np.full(N_PER_SCALE, np.deg2rad(s_deg))  # exact scale
        pert_q = axis_angle_to_quat_wxyz_batch(axes, angles)
        q_batch = np.zeros((N_PER_SCALE, 4))
        for i in range(N_PER_SCALE):
            q_batch[i] = quat_mul_wxyz(pert_q[i], truth_q0)
        q_batch /= np.linalg.norm(q_batch, axis=1, keepdims=True)
        w_batch = np.broadcast_to(truth_omega0, (N_PER_SCALE, 3)).copy()

        all_q0.append(q_batch)
        all_w.append(w_batch)
        all_bucket.append(np.full(N_PER_SCALE, AXIS_Q0, dtype=np.int32))
        all_axis.append(np.full(N_PER_SCALE, 'q0', dtype='S12'))
        all_scale.append(np.full(N_PER_SCALE, s_deg, dtype=np.float64))
        all_labels.append(np.array(
            [f'q0_{s_deg:g}deg' for _ in range(N_PER_SCALE)], dtype='S32'))

    # --- 2. ω-direction-only perturbations (perp axis, preserve |ω|) ---
    for s_deg in SCALES_DEG_WDIR:
        axes = perp_axes_to(w_dir_true, N_PER_SCALE, rng)
        angles = np.full(N_PER_SCALE, np.deg2rad(s_deg))
        rotvecs = axes * angles[:, None]
        Rs = Rotation.from_rotvec(rotvecs).as_matrix()
        new_dirs = np.einsum('nij,j->ni', Rs, w_dir_true)
        new_dirs /= np.linalg.norm(new_dirs, axis=1, keepdims=True)
        w_batch = new_dirs * w_mag_true

        q_batch = np.broadcast_to(truth_q0, (N_PER_SCALE, 4)).copy()
        all_q0.append(q_batch)
        all_w.append(w_batch)
        all_bucket.append(np.full(N_PER_SCALE, AXIS_OMEGA_DIR, dtype=np.int32))
        all_axis.append(np.full(N_PER_SCALE, 'omega_dir', dtype='S12'))
        all_scale.append(np.full(N_PER_SCALE, s_deg, dtype=np.float64))
        all_labels.append(np.array(
            [f'wdir_{s_deg:g}deg' for _ in range(N_PER_SCALE)], dtype='S32'))

    # --- 3. ω-magnitude-only perturbations (preserve direction) ---
    for s_pct in SCALES_PCT_WMAG:
        signs = rng.choice([-1.0, 1.0], size=N_PER_SCALE)
        factors = 1.0 + (s_pct / 100.0) * signs
        new_mags = w_mag_true * factors
        w_batch = w_dir_true[None, :] * new_mags[:, None]

        q_batch = np.broadcast_to(truth_q0, (N_PER_SCALE, 4)).copy()
        all_q0.append(q_batch)
        all_w.append(w_batch)
        all_bucket.append(np.full(N_PER_SCALE, AXIS_OMEGA_MAG, dtype=np.int32))
        all_axis.append(np.full(N_PER_SCALE, 'omega_mag', dtype='S12'))
        all_scale.append(np.full(N_PER_SCALE, s_pct, dtype=np.float64))
        all_labels.append(np.array(
            [f'wmag_{s_pct:g}pct' for _ in range(N_PER_SCALE)], dtype='S32'))

    # --- 4. joint: q0 rot + ω-dir rot + |ω| scale at same nominal scale ---
    for s_deg in SCALES_DEG_JOINT:
        # q0 rotation
        q_axes = uniform_s2(N_PER_SCALE, rng)
        q_angles = np.full(N_PER_SCALE, np.deg2rad(s_deg))
        pert_q = axis_angle_to_quat_wxyz_batch(q_axes, q_angles)
        q_batch = np.zeros((N_PER_SCALE, 4))
        for i in range(N_PER_SCALE):
            q_batch[i] = quat_mul_wxyz(pert_q[i], truth_q0)
        q_batch /= np.linalg.norm(q_batch, axis=1, keepdims=True)

        # ω-direction rotation about perp axis
        w_axes = perp_axes_to(w_dir_true, N_PER_SCALE, rng)
        w_angles = np.full(N_PER_SCALE, np.deg2rad(s_deg))
        rotvecs = w_axes * w_angles[:, None]
        Rs = Rotation.from_rotvec(rotvecs).as_matrix()
        new_dirs = np.einsum('nij,j->ni', Rs, w_dir_true)
        new_dirs /= np.linalg.norm(new_dirs, axis=1, keepdims=True)

        # ω-magnitude scale by (1 + s_deg/100 * sign)
        signs = rng.choice([-1.0, 1.0], size=N_PER_SCALE)
        factors = 1.0 + (s_deg / 100.0) * signs
        new_mags = w_mag_true * factors
        w_batch = new_dirs * new_mags[:, None]

        all_q0.append(q_batch)
        all_w.append(w_batch)
        all_bucket.append(np.full(N_PER_SCALE, AXIS_JOINT, dtype=np.int32))
        all_axis.append(np.full(N_PER_SCALE, 'joint', dtype='S12'))
        all_scale.append(np.full(N_PER_SCALE, s_deg, dtype=np.float64))
        all_labels.append(np.array(
            [f'joint_{s_deg:g}deg' for _ in range(N_PER_SCALE)], dtype='S32'))

    q0s = np.concatenate(all_q0, axis=0)
    omegas = np.concatenate(all_w, axis=0)
    bucket = np.concatenate(all_bucket, axis=0)
    axis_arr = np.concatenate(all_axis, axis=0)
    scale_arr = np.concatenate(all_scale, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Normalize quaternions once more defensively
    q0s = q0s / np.linalg.norm(q0s, axis=1, keepdims=True)

    return {
        'q0s': q0s,
        'omegas': omegas,
        'bucket': bucket,
        'axis': axis_arr,
        'scale': scale_arr,
        'labels': labels,
    }


# ---- Worker (IDENTICAL propagation+surrogate path as m120) ----------
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
        except Exception:
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


# ---- Stage A: load setup -------------------------------------------------
def stage_A_load():
    print("\n[Stage A] Loading m119v2 setup.npz")
    t0 = time.time()
    if not SETUP_NPZ.exists():
        raise FileNotFoundError(
            f"m119v2 setup not found at {SETUP_NPZ}. "
            f"Seed {SEED} must have an existing setup.npz.")
    setup = dict(np.load(SETUP_NPZ, allow_pickle=False))
    print(f"  loaded {SETUP_NPZ}")
    print(f"  constraint_epochs: {len(setup['constraint_epochs'])} "
          f"(range {setup['constraint_epochs'][0]}..{setup['constraint_epochs'][-1]})")
    print(f"  obs_times: dt={setup['obs_times'][1]-setup['obs_times'][0]:.2f} s, "
          f"span={setup['obs_times'][-1]:.1f} s")
    print(f"  truth_q0={setup['truth_q0']}")
    print(f"  truth_omega0={setup['truth_omega0']} "
          f"(|w|={np.linalg.norm(setup['truth_omega0']):.5f})")
    print(f"  Stage A done in {time.time()-t0:.1f}s")
    return setup


# ---- Stage B: build candidate pool (checkpoint FIRST) -------------------
def stage_B_pool(setup):
    print("\n[Stage B] Building candidate pool")
    t0 = time.time()
    out = OUT_DIR / "candidates.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=False))
        print(f"  N_cand={len(d['q0s'])}")
        print(f"  Stage B (cached) done in {time.time()-t0:.1f}s")
        return d

    rng = np.random.default_rng(RNG_SEED)
    pool = build_pool(
        truth_q0=setup['truth_q0'],
        truth_omega0=setup['truth_omega0'],
        rng=rng,
    )
    N = len(pool['q0s'])
    print(f"  total candidates: {N}")
    for ax_id, ax_name in AXIS_NAMES.items():
        cnt = int(np.sum(pool['bucket'] == ax_id))
        print(f"    axis {ax_id} {ax_name:12s}: {cnt}")

    # Expected pool size sanity check
    expected = (1
                + len(SCALES_DEG_Q0) * N_PER_SCALE
                + len(SCALES_DEG_WDIR) * N_PER_SCALE
                + len(SCALES_PCT_WMAG) * N_PER_SCALE
                + len(SCALES_DEG_JOINT) * N_PER_SCALE)
    assert N == expected, f"pool size mismatch: {N} != {expected}"

    # --- Skeleton save block (design-first) ---
    np.savez_compressed(
        out,
        q0s=pool['q0s'].astype(np.float64),
        omegas=pool['omegas'].astype(np.float64),
        bucket=pool['bucket'].astype(np.int32),
        axis=pool['axis'],         # bytes array
        scale=pool['scale'].astype(np.float64),
        labels=pool['labels'],     # bytes array
    )
    print(f"  saved {out} (~{out.stat().st_size/1e6:.2f} MB)")
    print(f"  Stage B done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=False))


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

    # --- Skeleton save block (design-first) ---
    np.savez_compressed(
        out,
        residual=residual,
        observed_mag=setup['observed_mag_ce'].astype(np.float32),
    )
    print(f"  saved {out} (~{out.stat().st_size/1e6:.1f} MB)")
    print(f"  Stage C done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=False))


# ---- Stage D: cost variants + ranking ----------------------------------
def stage_D_variants(residuals):
    print("\n[Stage D] Cost variants (mean_L1, mean_L2 only; scope-narrowed vs m120)")
    t0 = time.time()
    out = OUT_DIR / "cost_variants.npz"
    if out.exists() and not FORCE:
        print(f"  loading cached {out.name}")
        d = dict(np.load(out, allow_pickle=True))
        print(f"  Stage D (cached) done in {time.time()-t0:.1f}s")
        return d

    residual = residuals['residual']
    save_dict = {}
    for name in VARIANT_LIST:
        scores, descending = score_variant(name, residual)
        save_dict[f'score_{name}'] = scores.astype(np.float32)
        # argsort (ascending) so argsort[0] is best for both mean_L1/L2
        order = np.argsort(scores)
        save_dict[f'argsort_{name}'] = order.astype(np.int32)
        print(f"  {name:10s} (lower=better) min={scores.min():.6f}, "
              f"max={scores.max():.6f}, median={np.median(scores):.6f}")

    save_dict['variant_names'] = np.array(VARIANT_LIST)

    # --- Skeleton save block (design-first) ---
    np.savez_compressed(out, **save_dict)
    print(f"  saved {out}")
    print(f"  Stage D done in {time.time()-t0:.1f}s")
    return dict(np.load(out, allow_pickle=True))


# ---- Stage E: per-axis analysis ----------------------------------------
def stage_E_analyze(pool, variants_data):
    print("\n[Stage E] Per-axis basin analysis")
    t0 = time.time()
    bucket = pool['bucket']
    scale = pool['scale']
    N = len(bucket)

    truth_idx = int(np.where(bucket == AXIS_TRUTH)[0][0])

    per_axis_summary = {}
    headline = {}

    for v in VARIANT_LIST:
        scores = variants_data[f'score_{v}']
        truth_cost = float(scores[truth_idx])
        truth_rank = int(np.sum(scores < truth_cost))
        headline[v] = {
            'truth_cost': truth_cost,
            'truth_rank': truth_rank,
            'N': int(N),
        }
        print(f"\n  variant={v}")
        print(f"    truth idx={truth_idx}, truth_cost={truth_cost:.6f}, "
              f"truth_rank={truth_rank}/{N}")

        per_axis_summary[v] = {}
        for ax_id, ax_name in AXIS_NAMES.items():
            if ax_id == AXIS_TRUTH:
                continue
            m_ax = bucket == ax_id
            if not m_ax.any():
                continue
            ax_scales = np.unique(scale[m_ax])
            per_axis_summary[v][ax_name] = []
            print(f"    axis {ax_name}:")
            for s in ax_scales:
                m = m_ax & (scale == s)
                sub = scores[m]
                rec = {
                    'axis': ax_name,
                    'scale': float(s),
                    'n_samples': int(m.sum()),
                    'median_cost': float(np.median(sub)),
                    'p10_cost': float(np.percentile(sub, 10)),
                    'p90_cost': float(np.percentile(sub, 90)),
                    'min_cost': float(np.min(sub)),
                    'max_cost': float(np.max(sub)),
                    'truth_beats_all': bool(truth_cost < np.min(sub)),
                }
                per_axis_summary[v][ax_name].append(rec)
                unit = 'pct' if ax_name == 'omega_mag' else 'deg'
                print(f"      scale={s:>7.3g} {unit}: "
                      f"median={rec['median_cost']:.5f} "
                      f"(p10={rec['p10_cost']:.5f}, p90={rec['p90_cost']:.5f}), "
                      f"min={rec['min_cost']:.5f}, "
                      f"truth_beats_all={rec['truth_beats_all']}")

    # Basin boundary per axis: smallest scale at which median_cost > truth_cost + k*σ
    # σ is the std of truth neighborhood: use std of costs at smallest q0 scale
    # as a proxy for "surrogate noise floor at truth".
    noise_ref_variant = 'mean_L1'
    scores_ref = variants_data[f'score_{noise_ref_variant}']
    # Use the smallest-scale q0 bucket as noise proxy
    smallest_q0 = min(SCALES_DEG_Q0)
    m_noise = (bucket == AXIS_Q0) & (scale == smallest_q0)
    sigma_truth = float(np.std(scores_ref[m_noise])) if m_noise.any() else 0.0
    truth_cost_ref = float(scores_ref[truth_idx])
    print(f"\n  Noise floor proxy from {noise_ref_variant} at q0={smallest_q0}°: "
          f"sigma={sigma_truth:.6f}")

    basin_boundary = {}
    for v in VARIANT_LIST:
        basin_boundary[v] = {}
        scores = variants_data[f'score_{v}']
        truth_cost = float(scores[truth_idx])
        # Compute a per-variant sigma using the same m_noise
        sigma_v = float(np.std(scores[m_noise])) if m_noise.any() else 0.0
        for ax_name, records in per_axis_summary[v].items():
            boundary = {}
            for k in (1, 2, 3):
                thresh = truth_cost + k * sigma_v
                found = None
                for rec in records:
                    if rec['median_cost'] > thresh:
                        found = rec['scale']
                        break
                boundary[f'k{k}'] = found
            boundary['truth_cost'] = truth_cost
            boundary['sigma'] = sigma_v
            basin_boundary[v][ax_name] = boundary

    summary = {
        'seed': SEED,
        'N_cand': int(N),
        'truth_idx': int(truth_idx),
        'variant_list': list(VARIANT_LIST),
        'scales': {
            'q0_deg': list(SCALES_DEG_Q0),
            'omega_dir_deg': list(SCALES_DEG_WDIR),
            'omega_mag_pct': list(SCALES_PCT_WMAG),
            'joint_deg': list(SCALES_DEG_JOINT),
        },
        'n_per_scale': int(N_PER_SCALE),
        'headline': headline,
        'per_axis': per_axis_summary,
        'basin_boundary': basin_boundary,
        'noise_floor_ref': {
            'variant': noise_ref_variant,
            'bucket': 'q0',
            'scale_deg': smallest_q0,
            'sigma': sigma_truth,
        },
        'plots': {},  # filled after plotting
    }

    # atomic save (will re-save in main after plots added)
    atomic_json_save(OUT_DIR / "summary.json", summary)
    print(f"\n  saved {OUT_DIR / 'summary.json'} (preliminary)")
    print(f"  Stage E done in {time.time()-t0:.1f}s")
    return summary


# ---- Stage F: plots -----------------------------------------------------
def plot_cost_vs_scale_per_axis(summary, variants_data, pool):
    print("\n[Stage F.1] cost_vs_scale_per_axis.png")
    # Use mean_L1 as the primary variant for this plot
    v = 'mean_L1'
    truth_cost = summary['headline'][v]['truth_cost']

    axis_order = ['q0', 'omega_dir', 'omega_mag', 'joint']
    axis_units = {'q0': 'deg', 'omega_dir': 'deg', 'omega_mag': 'pct', 'joint': 'deg'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i, ax_name in enumerate(axis_order):
        ax = axes[i]
        records = summary['per_axis'][v].get(ax_name, [])
        if not records:
            ax.axis('off')
            continue
        scales = np.array([r['scale'] for r in records])
        med = np.array([r['median_cost'] for r in records])
        p10 = np.array([r['p10_cost'] for r in records])
        p90 = np.array([r['p90_cost'] for r in records])
        mn = np.array([r['min_cost'] for r in records])

        order = np.argsort(scales)
        scales = scales[order]; med = med[order]
        p10 = p10[order]; p90 = p90[order]; mn = mn[order]

        ax.fill_between(scales, p10, p90, alpha=0.25, color='steelblue',
                        label='p10-p90')
        ax.plot(scales, med, 'o-', color='steelblue', label='median')
        ax.plot(scales, mn, 's--', color='navy', alpha=0.6, label='min',
                markersize=4)
        ax.axhline(truth_cost, color='red', ls='--', lw=1.5,
                   label=f'truth={truth_cost:.4f}')
        ax.set_xscale('log')
        ax.set_xlabel(f"scale ({axis_units[ax_name]})")
        ax.set_ylabel(f'{v}')
        ax.set_title(f'axis = {ax_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f'm121 seed {SEED} — cost vs perturbation scale per axis ({v})',
                 fontsize=12)
    fig.tight_layout()
    out = PLOT_DIR / 'cost_vs_scale_per_axis.png'
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_basin_boundary(summary):
    print("\n[Stage F.2] basin_boundary.png")
    v = 'mean_L1'
    bb = summary['basin_boundary'][v]
    axes_names = list(bb.keys())

    ks = ('k1', 'k2', 'k3')
    width = 0.25
    x = np.arange(len(axes_names))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, k in enumerate(ks):
        vals = []
        for ax_name in axes_names:
            val = bb[ax_name].get(k)
            # None means never crossed within sampled scales
            vals.append(val if val is not None else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=k)
        for xi, val in zip(x + (i - 1) * width, vals):
            if np.isnan(val):
                ax.text(xi, 0.01, 'n/a', ha='center', va='bottom',
                        fontsize=8, color='grey')

    ax.set_xticks(x)
    ax.set_xticklabels(axes_names)
    ax.set_ylabel('smallest scale where median > truth + k·σ')
    ax.set_yscale('log')
    ax.set_title(f'm121 seed {SEED} — basin boundary per axis ({v}); '
                 f'σ={summary["basin_boundary"][v][axes_names[0]]["sigma"]:.5f}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    fig.tight_layout()
    out = PLOT_DIR / 'basin_boundary.png'
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
        print("=" * 70)
        print(f"m121 -- basin-width characterization (seed {SEED})")
        print("=" * 70)
        print(f"  OUT_DIR={OUT_DIR}")
        print(f"  POOL_SIZE={POOL_SIZE}, FORCE={FORCE}")
        print(f"  variants: {VARIANT_LIST} (scope-narrowed vs m120's 13)")

        setup = stage_A_load()
        pool = stage_B_pool(setup)
        residuals = stage_C_residuals(setup, pool)
        variants_data = stage_D_variants(residuals)
        summary = stage_E_analyze(pool, variants_data)

        plot1 = plot_cost_vs_scale_per_axis(summary, variants_data, pool)
        plot2 = plot_basin_boundary(summary)
        summary['plots'] = {
            'cost_vs_scale_per_axis': str(plot1),
            'basin_boundary': str(plot2),
        }
        atomic_json_save(OUT_DIR / "summary.json", summary)
        print(f"\n  re-saved {OUT_DIR / 'summary.json'} with plot paths")

        dt = time.time() - t_global

        # Headline print matching m120 style
        v = 'mean_L1'
        head = summary['headline'][v]
        # Find smallest competitor cost across ALL non-truth candidates and its scale
        scores = variants_data[f'score_{v}']
        bucket = pool['bucket']
        scale_arr = pool['scale']
        non_truth_mask = bucket != AXIS_TRUTH
        sub = scores[non_truth_mask]
        sub_scales = scale_arr[non_truth_mask]
        sub_axes = pool['axis'][non_truth_mask]
        i_min = int(np.argmin(sub))
        min_competitor_cost = float(sub[i_min])
        min_competitor_scale = float(sub_scales[i_min])
        min_competitor_axis = sub_axes[i_min].decode() if isinstance(
            sub_axes[i_min], (bytes, bytearray)) else str(sub_axes[i_min])

        print(f"\n{'='*70}")
        print(f"TOTAL TIME: {dt:.1f}s")
        print(f"HEADLINE: truth rank {head['truth_rank']}/{head['N']} under "
              f"{v} | min competitor cost {min_competitor_cost:.5f} at "
              f"scale {min_competitor_scale:g} ({min_competitor_axis})")
        print(f"Truth cost ({v}): {head['truth_cost']:.6f}")
        print(f"Noise floor σ ({v} @ q0={summary['noise_floor_ref']['scale_deg']}°): "
              f"{summary['basin_boundary'][v]['q0']['sigma']:.6f}")
        print(f"{'='*70}")
    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == '__main__':
    main()
