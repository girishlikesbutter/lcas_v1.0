#!/usr/bin/env python3
"""
m128 -- Warm-start flipped-omega polish from m115 DE basins.

Hypothesis (falsifiable)
------------------------
At least 2 of seeds {0, 6, 14, 24, 27, 36, 46, 74, 93} yield polished
(q0', -omega_basin) with hi-fi MSE < 0.5.

  CONFIRMED  iff >= 2 non-control seeds hit hifi < 0.5
  REFUTED    iff  0 non-control seeds hit hifi < 0.5
  MIXED      iff exactly 1 non-control seed hits hifi < 0.5
  INVALID    iff seed 33 control fails (best hifi >= 0.15) -- polish mechanics bug

Positive controls:
  - seed 33 MUST reproduce hifi ~0.082 (m126 flipped-omega basin),
    threshold best_hifi < 0.15, else INVALID.
  - seed 12 sanity, expects hifi ~0.17 from the known wide flipped-omega
    attractor identified in m127.

Method
------
For each seed in BASELINE_SEEDS (11 total incl. controls 33, 12):
  A. Load m115 step2_hifi.npz and parse hifi_json -> 3 basin dicts.
  B. For each basin (all 3):
       - Build warm-start: q0_init = basin['q0_wxyz'],
                           omega_flipped = -np.array(basin['omega_rad']).
       - L-BFGS-B polish (3-DOF rotvec over q0; omega fixed at omega_flipped).
       - Record q0_polished, surr_cost_start/end, timings, nit, success.
  C. Hi-fi validate (Pool) both (q0_init, omega_flipped) and
     (q0_polished, omega_flipped) for all 3 basins -> 6 evals/seed.
  D. Per seed: best_hifi_mse = min over (basin x before/after); classify.

Key distinction from m127
-----------------------------
  - NO SO(3) grid. We warm-start at the per-basin q0 (NOT truth q0).
  - We negate the BASIN omega, not truth omega. Per-basin omega can differ
    from truth omega by up to ~5 deg (e.g. seed 74). So
    omega_flipped = -basin['omega_rad'] -- recorded per-basin. See
    `omega_flipped_is_negated_per_basin_not_truth: true` in result.json.

Checkpoints
-----------
Per seed under data/results/inversion_diagnostics/m128_warmstart_polish/
  seed_NNN/
    stage_a_polish.npz : per-basin q0_init, omega_flipped, q0_polished,
                         surr costs, timings, status. hi-fi fields init NaN,
                         rewritten after Stage B.
    stage_b_hifi.npz   : hifi_before/after_mse, hifi_before/after_mags,
                         observed_lc, hifi_wall_s.
    result.json        : per-seed summary with all basins' best hifi etc.
    run.log            : teed stdout.

Batch:
  batch_summary.json : per-seed rows, counts, controls, verdict.

Classification
--------------
  FLIPPED_VALID    best_hifi_mse < 0.1
  FLIPPED_PARTIAL  best_hifi_mse < 0.5
  FLIPPED_FAIL     otherwise

Env
---
  MICRO128_SEEDS : CSV override for SEEDS (default full 11-seed list).
  MICRO128_POOL  : Pool size for hi-fi stage (default 4).
  MICRO128_FORCE : '1' to re-run even if checkpoints exist (default '0').

Kill criteria
-------------
  - Per-seed hard cap: 5 min wall -> investigate + abort.
  - Batch hard cap: 30 min wall.
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
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion" / "12_brightness_surface"))
sys.path.insert(0, '/home/girish/surrogate_model')
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, brightness_single_epoch  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude              # noqa: E402
from surrogate import SurrogateModel                                         # noqa: E402

# ---- Config --------------------------------------------------------------
BASELINE_SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93]
POSITIVE_CONTROL_SEED = 33          # narrow-basin flipped-omega control (~0.082)
SANITY_SEED = 12                    # wide-basin flipped-omega sanity (~0.17)

_env = os.environ.get('MICRO128_SEEDS', '').strip()
SEEDS = ([int(s) for s in _env.split(',') if s.strip()]
         if _env else list(BASELINE_SEEDS))
HIFI_POOL = int(os.environ.get('MICRO128_POOL', '4'))
FORCE = os.environ.get('MICRO128_FORCE', '0') == '1'

N_OBS = 500
NOISE_SEED_SHARED = 42
NOISE_SIGMA = 0.05
END_TIME_UTC = '2020-02-05T11:00:00'
PANEL_DEG = 0.0
DISH_DEG = 15.0

# Stage A: L-BFGS-B polish options (identical to m127)
LBFGS_OPTIONS = {'ftol': 1e-7, 'gtol': 1e-4, 'maxiter': 200,
                 'maxfun': 1000, 'disp': False}

# Classification thresholds
CLS_VALID_MAX = 0.1
CLS_PARTIAL_MAX = 0.5
# Verdict thresholds
SEED33_CONTROL_MAX = 0.15
SEED12_SANITY_MAX = 0.3
N_NONCONTROL_SEEDS_CONFIRM = 2
N_NONCONTROL_SEEDS_MIXED = 1

SURROGATE_WEIGHTS = Path('/home/girish/surrogate_model/s10_5M_weights.npz')
SURROGATE_NORM = Path('/home/girish/surrogate_model/s10_5M_normalization.npz')

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
MICRO46_NPZ = RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"
MICRO115_BASE = RESULTS_DIR / "m115_surrogate_pipeline"
OUT_BASE = RESULTS_DIR / "m128_warmstart_polish"


# ---- Logging / IO (vendored from m127 lines 91-115) -----------------
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


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode('utf-8', errors='replace')
    return str(o)


def atomic_json_save(filepath, data):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, filepath)


# ---- Quaternion helpers (vendored from m127) -------------------------
def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def rotvec_to_wxyz(r):
    theta = float(np.linalg.norm(r))
    if theta < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = r / theta
    ha = theta / 2.0
    s = np.sin(ha)
    return np.array([np.cos(ha), s*axis[0], s*axis[1], s*axis[2]])


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]]) if q.ndim == 1 else q[:, [1, 2, 3, 0]]


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]]) if q.ndim == 1 else q[:, [3, 0, 1, 2]]


def quat_mul_batch(q1, q2):
    """q1:(4,) wxyz scalar, q2:(N,4) wxyz batch -> (N,4) wxyz."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.column_stack([w, x, y, z])


def quat_geodesic_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * np.degrees(np.arccos(
        min(1.0, max(-1.0, float(abs(np.dot(a, b))))))))


# ---- Vendored hi-fi helpers (from m127) ------------------------------
def precompute_delta_qs(omega_vec, obs_times, I_tensor):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    quats, _ = propagate_attitude(q_id, omega_vec, obs_times, "tumbling", I_tensor)
    return quats


def hifi_validate(q0_wxyz, omega_rad, obs_times, I_tensor, observed_lc, ctx):
    delta_qs = precompute_delta_qs(omega_rad, obs_times, I_tensor)
    quats = quat_mul_batch(q0_wxyz, delta_qs)
    hifi_mags = np.full(len(obs_times), np.nan)
    for i in range(len(obs_times)):
        hifi_mags[i] = brightness_single_epoch(quats[i], i, ctx, use_shadows=True)
    valid = np.isfinite(hifi_mags) & np.isfinite(observed_lc)
    if np.sum(valid) < 10:
        return 1e6, hifi_mags
    return float(np.mean((hifi_mags[valid] - observed_lc[valid])**2)), hifi_mags


# ---- Vendored make_polish_cost (from m127 lines 247-273) ------------
def make_polish_cost(q0_init, omega_fixed, obs_times, I_tensor,
                     sun_dirs, obs_dirs, obs_dist, observed_lc, model):
    delta_qs = precompute_delta_qs(omega_fixed, obs_times, I_tensor)
    E = len(obs_times)
    zeros_E = np.zeros(E)
    dish_E = np.full(E, DISH_DEG)
    dist_E = obs_dist.astype(np.float64)
    obs_lc = observed_lc.astype(np.float64)
    obs_valid = np.isfinite(obs_lc)

    def cost(r):
        q0 = quat_mul_wxyz(rotvec_to_wxyz(np.asarray(r, dtype=np.float64)), q0_init)
        q0 /= np.linalg.norm(q0)
        quats = quat_mul_batch(q0, delta_qs)
        R_all = Rotation.from_quat(wxyz_to_xyzw(quats)).as_matrix()
        k1 = np.einsum('tij,tj->ti', R_all, sun_dirs)
        k2 = np.einsum('tij,tj->ti', R_all, obs_dirs)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
        pred = model.predict_magnitude(k1, k2, zeros_E, dish_E, dist_E)
        valid = obs_valid & np.isfinite(pred)
        if np.sum(valid) < 10:
            return 10.0
        return float(np.mean(np.abs(pred[valid] - obs_lc[valid])))

    def final_q0(r):
        q0 = quat_mul_wxyz(rotvec_to_wxyz(np.asarray(r, dtype=np.float64)), q0_init)
        return q0 / np.linalg.norm(q0)

    return cost, final_q0


# ---- Load m115 basins -----------------------------------------------
def load_m115_basins(seed):
    """Parse step2_hifi.npz -> hifi_json -> list of 3 basin dicts."""
    npz_path = MICRO115_BASE / f"seed_{seed:03d}" / "step2_hifi.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing m115 step2_hifi.npz: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    blob = str(d['hifi_json'])
    basins = json.loads(blob)
    if len(basins) != 3:
        print(f"    WARN: seed {seed} has {len(basins)} basins (expected 3)")
    return basins


# ---- Stage A: L-BFGS polish (serial, 3 basins per seed) -----------------
def save_stage_a(out_path, records, truth_q0, truth_omega):
    """Write stage_a_polish.npz FIRST, with hi-fi fields init NaN."""
    n = len(records)
    q0_init = np.array([r['q0_init_wxyz'] for r in records], dtype=np.float64)
    omega_flipped = np.array([r['omega_flipped'] for r in records], dtype=np.float64)
    q0_polished = np.array([r['q0_polished_wxyz'] for r in records], dtype=np.float64)
    surr_cost_start = np.array([r['surr_cost_start'] for r in records], dtype=np.float64)
    surr_cost_end = np.array([r['surr_cost_end'] for r in records], dtype=np.float64)
    polish_wall_s = np.array([r['polish_wall_s'] for r in records], dtype=np.float64)
    nit = np.array([r['nit'] for r in records], dtype=np.int32)
    success = np.array([r['success'] for r in records], dtype=bool)
    messages = np.array([r['message'] for r in records], dtype=object)
    basin_idx = np.arange(n, dtype=np.int32)

    hifi_before_mse = np.full(n, np.nan, dtype=np.float64)
    hifi_after_mse = np.full(n, np.nan, dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        basin_idx=basin_idx,
        q0_init_wxyz=q0_init,
        omega_flipped=omega_flipped,
        q0_polished_wxyz=q0_polished,
        surr_cost_start=surr_cost_start,
        surr_cost_end=surr_cost_end,
        polish_wall_s=polish_wall_s,
        nit=nit,
        success=success,
        messages=messages,
        hifi_before_mse=hifi_before_mse,
        hifi_after_mse=hifi_after_mse,
        truth_q0_wxyz=truth_q0.astype(np.float64),
        truth_omega=truth_omega.astype(np.float64),
    )


def stage_a_polish(seed, seed_dir, m115_basins, sun_dirs, obs_dirs,
                   obs_dist, obs_times, I_tensor, observed_lc,
                   truth_q0, truth_omega, model):
    """Serial L-BFGS polish across 3 basins. Writes stage_a_polish.npz."""
    print(f"\n[Stage A] seed {seed} -- warm-start L-BFGS polish on "
          f"{len(m115_basins)} basins")
    out = seed_dir / "stage_a_polish.npz"
    if out.exists() and not FORCE:
        print(f"  cached {out.name}")
        d = dict(np.load(out, allow_pickle=True))
        records = []
        for i in range(int(len(d['basin_idx']))):
            records.append({
                'basin_idx': int(d['basin_idx'][i]),
                'q0_init_wxyz': d['q0_init_wxyz'][i].tolist(),
                'omega_flipped': d['omega_flipped'][i].tolist(),
                'q0_polished_wxyz': d['q0_polished_wxyz'][i].tolist(),
                'surr_cost_start': float(d['surr_cost_start'][i]),
                'surr_cost_end': float(d['surr_cost_end'][i]),
                'polish_wall_s': float(d['polish_wall_s'][i]),
                'nit': int(d['nit'][i]),
                'success': bool(d['success'][i]),
                'message': str(d['messages'][i]),
            })
        return records

    t0 = time.time()
    records = []
    for bi, basin in enumerate(m115_basins):
        q0_init = np.asarray(basin['q0_wxyz'], dtype=np.float64)
        q0_init /= np.linalg.norm(q0_init)
        omega_basin = np.asarray(basin['omega_rad'], dtype=np.float64)
        omega_flipped = -omega_basin.copy()

        cost_fn, final_q0_fn = make_polish_cost(
            q0_init, omega_flipped, obs_times, I_tensor,
            sun_dirs, obs_dirs, obs_dist, observed_lc, model)

        ts = time.time()
        x0 = np.zeros(3)
        start_cost = cost_fn(x0)
        res = minimize(cost_fn, x0=x0, jac=None, method='L-BFGS-B',
                       options=LBFGS_OPTIONS)
        wall = time.time() - ts
        q0_polished = final_q0_fn(res.x)

        rec = {
            'basin_idx': bi,
            'q0_init_wxyz': q0_init.tolist(),
            'omega_flipped': omega_flipped.tolist(),
            'q0_polished_wxyz': q0_polished.tolist(),
            'surr_cost_start': float(start_cost),
            'surr_cost_end': float(res.fun),
            'polish_wall_s': float(wall),
            'nit': int(getattr(res, 'nit', -1)),
            'success': bool(res.success),
            'message': str(res.message),
        }
        records.append(rec)
        print(f"  basin {bi}: surr {start_cost:.6f} -> {res.fun:.6f}  "
              f"iter={rec['nit']} success={rec['success']} "
              f"wall={wall:.1f}s")

    save_stage_a(out, records, truth_q0, truth_omega)
    print(f"  saved {out}  (Stage A total: {time.time()-t0:.1f}s)")
    return records


# ---- Stage B: hi-fi validate (Pool) -------------------------------------
# Module-global state inherited by forked workers.
_HIFI_STATE = {}


def _hifi_init_worker(ctx, obs_times, I_tensor, observed_lc):
    _HIFI_STATE.update(dict(
        ctx=ctx, obs_times=obs_times,
        I_tensor=I_tensor, observed_lc=observed_lc))


def _hifi_worker(job):
    """job = (basin_idx, q0_wxyz, omega_flipped, tag)."""
    basin_idx, q0_wxyz, omega_flipped, tag = job
    t0 = time.time()
    mse, mags = hifi_validate(
        np.asarray(q0_wxyz, dtype=np.float64),
        np.asarray(omega_flipped, dtype=np.float64),
        _HIFI_STATE['obs_times'], _HIFI_STATE['I_tensor'],
        _HIFI_STATE['observed_lc'], _HIFI_STATE['ctx'])
    return {
        'basin_idx': int(basin_idx),
        'tag': tag,
        'hifi_mse': float(mse),
        'hifi_mags': np.asarray(mags, dtype=np.float64),
        'wall_s': float(time.time() - t0),
    }


def save_stage_b(out_path, hifi_before, hifi_after, mags_before, mags_after,
                 observed_lc, hifi_wall_s):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        hifi_before_mse=hifi_before.astype(np.float64),
        hifi_after_mse=hifi_after.astype(np.float64),
        hifi_before_mags=mags_before.astype(np.float64),
        hifi_after_mags=mags_after.astype(np.float64),
        observed_lc=observed_lc.astype(np.float64),
        hifi_wall_s=np.array(hifi_wall_s, dtype=np.float64),
    )


def backfill_stage_a(stage_a_path, hifi_before, hifi_after):
    """Rewrite stage_a_polish.npz with hi-fi fields filled in."""
    d = dict(np.load(stage_a_path, allow_pickle=True))
    d['hifi_before_mse'] = hifi_before.astype(np.float64)
    d['hifi_after_mse'] = hifi_after.astype(np.float64)
    # Tmp path must end in '.npz' because np.savez auto-appends '.npz' otherwise.
    tmp = stage_a_path.parent / (stage_a_path.stem + '_tmp.npz')
    np.savez(tmp, **d)
    os.replace(tmp, stage_a_path)


def stage_b_hifi(seed, seed_dir, polish_records, ctx, observed_lc,
                 obs_times, I_tensor):
    """Pool-parallel hi-fi validate for (basin x before/after). 6 evals."""
    print(f"\n[Stage B] seed {seed} -- hi-fi validate (Pool({HIFI_POOL}))")
    out = seed_dir / "stage_b_hifi.npz"
    if out.exists() and not FORCE:
        print(f"  cached {out.name}")
        d = dict(np.load(out, allow_pickle=True))
        return (d['hifi_before_mse'], d['hifi_after_mse'],
                d['hifi_before_mags'], d['hifi_after_mags'],
                float(d['hifi_wall_s']))

    t0 = time.time()
    n = len(polish_records)
    jobs = []
    for r in polish_records:
        jobs.append((r['basin_idx'], r['q0_init_wxyz'],
                     r['omega_flipped'], 'before'))
        jobs.append((r['basin_idx'], r['q0_polished_wxyz'],
                     r['omega_flipped'], 'after'))
    pool_size = max(1, min(len(jobs), HIFI_POOL))
    print(f"  {len(jobs)} jobs ({n} basins x 2 tags), Pool({pool_size})")

    hifi_before = np.full(n, np.nan, dtype=np.float64)
    hifi_after = np.full(n, np.nan, dtype=np.float64)
    mags_before = np.full((n, N_OBS), np.nan, dtype=np.float64)
    mags_after = np.full((n, N_OBS), np.nan, dtype=np.float64)

    init_args = (ctx, obs_times, I_tensor, observed_lc)
    with mp.Pool(pool_size, initializer=_hifi_init_worker,
                 initargs=init_args) as pool:
        for k, r in enumerate(pool.imap_unordered(_hifi_worker, jobs)):
            bi = r['basin_idx']
            if r['tag'] == 'before':
                hifi_before[bi] = r['hifi_mse']
                mags_before[bi, :] = r['hifi_mags']
            else:
                hifi_after[bi] = r['hifi_mse']
                mags_after[bi, :] = r['hifi_mags']
            print(f"  [{k+1}/{len(jobs)}] basin {bi} ({r['tag']}): "
                  f"hifi={r['hifi_mse']:.6f} wall={r['wall_s']:.1f}s")

    total_wall = time.time() - t0
    save_stage_b(out, hifi_before, hifi_after, mags_before, mags_after,
                 observed_lc, total_wall)
    print(f"  saved {out}  (Stage B total: {total_wall:.1f}s)")

    # Backfill stage_a with the now-known hi-fi values.
    stage_a_path = seed_dir / "stage_a_polish.npz"
    if stage_a_path.exists():
        backfill_stage_a(stage_a_path, hifi_before, hifi_after)
        print(f"  backfilled hi-fi fields into {stage_a_path.name}")

    return hifi_before, hifi_after, mags_before, mags_after, total_wall


# ---- Classification ------------------------------------------------------
def classify_flipped(min_hifi):
    if min_hifi < CLS_VALID_MAX:
        return 'FLIPPED_VALID'
    if min_hifi < CLS_PARTIAL_MAX:
        return 'FLIPPED_PARTIAL'
    return 'FLIPPED_FAIL'


# ---- Per-seed driver -----------------------------------------------------
def run_seed(seed, ctx, sun_dirs, obs_dirs, obs_dist, obs_times, I_tensor,
             truth_q0, truth_omega, model, m115_basin_errors):
    """
    m115_basin_errors: list of (q0_err, w_dir_err, w_mag_err_pct, hifi_mse) per basin
    for provenance in result.json.
    """
    seed_dir = OUT_BASE / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(seed_dir / "run.log", "w")
    old_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_file)

    t_seed = time.time()
    timings = {}
    try:
        # Load basins and build the observed LC (shared with ctx.observed_lc).
        m115_basins = load_m115_basins(seed)
        observed_lc = ctx.observed_lc.astype(np.float64)

        print(f"\n===== Seed {seed} =====")
        print(f"  truth_q0={truth_q0}")
        print(f"  truth_omega={truth_omega} (|w|={np.linalg.norm(truth_omega):.5f})")
        print(f"  n_basins (from m115)={len(m115_basins)}")
        for i, b in enumerate(m115_basins):
            print(f"    basin {i}: m115_hifi={b['hifi_mse']:.6f}  "
                  f"q0_err={b['q0_err']:.2f}  w_dir_err={b['w_dir_err']:.2f}  "
                  f"w_mag_err_pct={b['w_mag_err_pct']:.3f}  twin={b['is_twin']}")

        # -- Stage A: polish --
        ts = time.time()
        polish_records = stage_a_polish(
            seed, seed_dir, m115_basins, sun_dirs, obs_dirs, obs_dist,
            obs_times, I_tensor, observed_lc, truth_q0, truth_omega, model)
        timings['stage_a_s'] = time.time() - ts

        # -- Stage B: hi-fi --
        ts = time.time()
        hifi_before, hifi_after, mags_before, mags_after, hifi_wall_s = stage_b_hifi(
            seed, seed_dir, polish_records, ctx, observed_lc, obs_times, I_tensor)
        timings['stage_b_s'] = time.time() - ts

        # -- Assemble per-basin records --
        n_basins = len(polish_records)
        basin_out = []
        for bi in range(n_basins):
            r = polish_records[bi]
            hb = float(hifi_before[bi]) if np.isfinite(hifi_before[bi]) else float('inf')
            ha = float(hifi_after[bi]) if np.isfinite(hifi_after[bi]) else float('inf')
            best = min(hb, ha)
            best_is_polish = ha <= hb
            m115 = m115_basins[bi]
            basin_out.append({
                'basin_idx': bi,
                'q0_init_wxyz': r['q0_init_wxyz'],
                'q0_polished_wxyz': r['q0_polished_wxyz'],
                'omega_flipped': r['omega_flipped'],
                'm115_q0_err': float(m115.get('q0_err', float('nan'))),
                'm115_w_dir_err': float(m115.get('w_dir_err', float('nan'))),
                'm115_w_mag_err_pct': float(m115.get('w_mag_err_pct', float('nan'))),
                'm115_hifi_mse': float(m115.get('hifi_mse', float('nan'))),
                'm115_is_twin': bool(m115.get('is_twin', False)),
                'surr_cost_start': r['surr_cost_start'],
                'surr_cost_end': r['surr_cost_end'],
                'hifi_before_mse': hb,
                'hifi_after_mse': ha,
                'best_hifi_mse': best,
                'best_is_polish': bool(best_is_polish),
                'polish_wall_s': r['polish_wall_s'],
                'nit': r['nit'],
                'success': r['success'],
            })

        # Seed-level best.
        best_per_basin = np.array([b['best_hifi_mse'] for b in basin_out])
        best_idx = int(np.argmin(best_per_basin))
        best_hifi_seed = float(best_per_basin[best_idx])
        classification = classify_flipped(best_hifi_seed)
        timings['total_s'] = time.time() - t_seed

        result = {
            'traj_seed': int(seed),
            'experiment': 'm128_warmstart_polish',
            'basins': basin_out,
            'best_hifi_mse_seed': best_hifi_seed,
            'best_basin_idx': best_idx,
            'classification': classification,
            'timing_total_s': round(timings['total_s'], 2),
            'timing_stages': {k: round(v, 2) for k, v in timings.items()},
            'omega_flipped_is_negated_per_basin_not_truth': True,
            'truth_q0_wxyz': truth_q0.tolist(),
            'truth_omega': truth_omega.tolist(),
        }
        atomic_json_save(seed_dir / "result.json", result)

        print(f"\n  classification: {classification}")
        print(f"  best basin {best_idx}: best_hifi={best_hifi_seed:.6f}  "
              f"(before={basin_out[best_idx]['hifi_before_mse']:.6f}, "
              f"after={basin_out[best_idx]['hifi_after_mse']:.6f})")
        print(f"  seed total: {timings['total_s']:.1f}s")
        return result
    except Exception as e:
        import traceback
        print(f"  ERROR seed {seed}: {e}")
        traceback.print_exc()
        result = {
            'traj_seed': int(seed),
            'experiment': 'm128_warmstart_polish',
            'error': str(e),
            'classification': 'ERROR',
            'timing_total_s': round(time.time() - t_seed, 2),
            'omega_flipped_is_negated_per_basin_not_truth': True,
        }
        atomic_json_save(seed_dir / "result.json", result)
        return result
    finally:
        sys.stdout = old_stdout
        log_file.close()


# ---- Main ----------------------------------------------------------------
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    log_path = OUT_BASE / "run.log"
    log_f = open(log_path, 'w')
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    try:
        print("=" * 72)
        print("m128 -- warm-start flipped-omega polish from m115 DE basins")
        print("=" * 72)
        print(f"  SEEDS            = {SEEDS}")
        print(f"  HIFI_POOL        = {HIFI_POOL}")
        print(f"  FORCE            = {FORCE}")
        print(f"  LBFGS_OPTIONS    = {LBFGS_OPTIONS}")
        print(f"  N_OBS            = {N_OBS}")
        print(f"  END_TIME_UTC     = {END_TIME_UTC}")
        print(f"  NOISE_SIGMA      = {NOISE_SIGMA}")
        print(f"  positive control = seed {POSITIVE_CONTROL_SEED} (must < {SEED33_CONTROL_MAX})")
        print(f"  sanity control   = seed {SANITY_SEED} (expect < {SEED12_SANITY_MAX})")
        print(f"  out_dir          = {OUT_BASE}")

        t_all = time.time()

        # ── Setup: load truth + build shared ExperimentContext (seed-agnostic
        # geometry, truth only used for error metrics). Hi-fi needs ctx per seed
        # to have correct observed_lc. We build ctx per seed with skip_true_lc
        # = False so ctx.observed_lc is generated identically to m126/115.
        print("\n[Setup] loading m046 truth NPZ")
        m46 = np.load(MICRO46_NPZ)
        m46_q0s = m46['q0s'].astype(np.float64)
        m46_omega0s = m46['omega0s'].astype(np.float64)

        # Surrogate model loaded ONCE in the parent process (polish is serial).
        print("[Setup] loading surrogate model...")
        t_mdl = time.time()
        model = SurrogateModel(str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))
        print(f"  model loaded in {time.time()-t_mdl:.1f}s")

        all_results = []
        for seed in SEEDS:
            print("\n" + "=" * 72)
            print(f"[SEED {seed}]")
            print("=" * 72)
            ts = time.time()

            # Per-seed context: observed_lc is generated inside setup_experiment
            # using truth_q0/omega for seed (not m046 mag_hifi -- identical to
            # m126 which uses skip_true_lc=False).
            truth_q0 = m46_q0s[seed].astype(np.float64)
            truth_q0 /= np.linalg.norm(truth_q0)
            truth_omega = m46_omega0s[seed].astype(np.float64)

            print(f"  building ExperimentContext for seed {seed}...")
            t_ctx = time.time()
            ctx = setup_experiment(
                n_observations=N_OBS,
                noise_sigma=NOISE_SIGMA,
                random_seed=NOISE_SEED_SHARED,
                end_time_utc=END_TIME_UTC,
                skip_true_lc=False,
                true_q0_wxyz=truth_q0,
                true_omega0_rad=truth_omega,
            )
            print(f"    ctx built in {time.time()-t_ctx:.1f}s")

            obs_times = ctx.observation_times.astype(np.float64)
            I_tensor = ctx.inertia_tensor.astype(np.float64)
            sun_dirs = ctx.sun_pos - ctx.sat_pos
            sun_dirs /= np.linalg.norm(sun_dirs, axis=1, keepdims=True)
            obs_dirs = ctx.obs_pos - ctx.sat_pos
            obs_dirs /= np.linalg.norm(obs_dirs, axis=1, keepdims=True)
            obs_dist = ctx.obs_dist.astype(np.float64)

            result = run_seed(
                seed, ctx, sun_dirs, obs_dirs, obs_dist,
                obs_times, I_tensor, truth_q0, truth_omega, model,
                m115_basin_errors=None)
            print(f"  [seed {seed}] {result.get('classification','?')} "
                  f"in {time.time()-ts:.1f}s")
            all_results.append(result)

        # ── Batch assembly ─────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("[Batch summary]")
        print("=" * 72)

        per_seed = {}
        counts = {'FLIPPED_VALID': 0, 'FLIPPED_PARTIAL': 0,
                  'FLIPPED_FAIL': 0, 'ERROR': 0}
        for r in all_results:
            seed = r['traj_seed']
            cls = r.get('classification', 'ERROR')
            counts[cls] = counts.get(cls, 0) + 1
            per_seed[str(seed)] = {
                'best_hifi_mse': r.get('best_hifi_mse_seed'),
                'classification': cls,
                'wall_s': r.get('timing_total_s'),
            }

        # Control checks
        seed33_best = per_seed.get(str(POSITIVE_CONTROL_SEED), {}).get('best_hifi_mse')
        seed12_best = per_seed.get(str(SANITY_SEED), {}).get('best_hifi_mse')
        seed_33_control_ok = (seed33_best is not None
                              and np.isfinite(seed33_best)
                              and seed33_best < SEED33_CONTROL_MAX)
        seed_12_sanity_ok = (seed12_best is not None
                             and np.isfinite(seed12_best)
                             and seed12_best < SEED12_SANITY_MAX)

        # Count non-control seeds with best_hifi < 0.5 (excluding 33, 12).
        excluded = {POSITIVE_CONTROL_SEED, SANITY_SEED}
        noncontrol_hits = []
        for r in all_results:
            if r['traj_seed'] in excluded:
                continue
            b = r.get('best_hifi_mse_seed')
            if b is not None and np.isfinite(b) and b < CLS_PARTIAL_MAX:
                noncontrol_hits.append(r['traj_seed'])
        n_noncontrol_hits = len(noncontrol_hits)

        # Verdict rules.
        if not seed_33_control_ok:
            verdict = 'INVALID'
            verdict_reason = (f"seed 33 control best_hifi={seed33_best} >= "
                              f"{SEED33_CONTROL_MAX} -- polish mechanics bug")
        elif n_noncontrol_hits >= N_NONCONTROL_SEEDS_CONFIRM:
            verdict = 'CONFIRMED'
            verdict_reason = (f"{n_noncontrol_hits} non-control seeds hit "
                              f"hifi<{CLS_PARTIAL_MAX}: {noncontrol_hits}")
        elif n_noncontrol_hits == N_NONCONTROL_SEEDS_MIXED:
            verdict = 'MIXED'
            verdict_reason = (f"only 1 non-control seed hit hifi<{CLS_PARTIAL_MAX}: "
                              f"{noncontrol_hits}")
        else:
            verdict = 'REFUTED'
            verdict_reason = (f"0 non-control seeds hit hifi<{CLS_PARTIAL_MAX}")

        summary = {
            'seeds': SEEDS,
            'per_seed': per_seed,
            'n_valid': counts['FLIPPED_VALID'],
            'n_partial': counts['FLIPPED_PARTIAL'],
            'n_fail': counts['FLIPPED_FAIL'],
            'n_error': counts['ERROR'],
            'seed_33_control_ok': bool(seed_33_control_ok),
            'seed_33_best_hifi': seed33_best,
            'seed_12_sanity_ok': bool(seed_12_sanity_ok),
            'seed_12_best_hifi': seed12_best,
            'noncontrol_hits_seeds': noncontrol_hits,
            'n_noncontrol_hits': n_noncontrol_hits,
            'verdict': verdict,
            'verdict_reason': verdict_reason,
            'total_wall_s': round(time.time() - t_all, 2),
            'config': {
                'BASELINE_SEEDS': BASELINE_SEEDS,
                'POSITIVE_CONTROL_SEED': POSITIVE_CONTROL_SEED,
                'SANITY_SEED': SANITY_SEED,
                'N_OBS': N_OBS,
                'NOISE_SIGMA': NOISE_SIGMA,
                'NOISE_SEED_SHARED': NOISE_SEED_SHARED,
                'END_TIME_UTC': END_TIME_UTC,
                'HIFI_POOL': HIFI_POOL,
                'LBFGS_OPTIONS': LBFGS_OPTIONS,
                'CLS_VALID_MAX': CLS_VALID_MAX,
                'CLS_PARTIAL_MAX': CLS_PARTIAL_MAX,
                'SEED33_CONTROL_MAX': SEED33_CONTROL_MAX,
                'SEED12_SANITY_MAX': SEED12_SANITY_MAX,
                'DISH_DEG': DISH_DEG,
            },
        }
        atomic_json_save(OUT_BASE / "batch_summary.json", summary)

        print(f"\n  counts: VALID={counts['FLIPPED_VALID']} "
              f"PARTIAL={counts['FLIPPED_PARTIAL']} "
              f"FAIL={counts['FLIPPED_FAIL']} ERROR={counts['ERROR']}")
        print(f"  seed 33 control (< {SEED33_CONTROL_MAX}): "
              f"{seed33_best} -> {'OK' if seed_33_control_ok else 'FAIL'}")
        print(f"  seed 12 sanity  (< {SEED12_SANITY_MAX}): "
              f"{seed12_best} -> {'OK' if seed_12_sanity_ok else 'FAIL'}")
        print(f"  non-control hits ({n_noncontrol_hits}): {noncontrol_hits}")
        print(f"  verdict: {verdict}")
        print(f"  reason:  {verdict_reason}")
        print(f"  saved {OUT_BASE / 'batch_summary.json'}")
        print(f"  total wall: {time.time()-t_all:.1f}s")
        print("=" * 72)
    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
