#!/usr/bin/env python3
"""
m126 -- Wrapped pipeline (DE -> L-BFGS polish -> hi-fi(before, after) -> keep-min)
on the 6 m115 baseline seeds that were never tested with the keep_better
wrapper: [0, 6, 12, 24, 33, 36].

Hypothesis (falsifiable)
------------------------
On these 6 untested seeds, the wrapped pipeline improves seed-level best hi-fi
MSE by >= 10% vs plain m115 on >= 3 of 6 seeds.

    CONFIRMED  iff  n_seeds_improved_ge10pct >= 3
    REFUTED    iff  n_seeds_improved_ge10pct <  3

A CONFIRMED verdict promotes the gradient-based-inversion branch to #validated.

Method
------
This experiment does NOT re-run the DE stage. For each seed it:
  (1) Loads existing m115 DE basins from
      data/results/inversion_diagnostics/m115_surrogate_pipeline/seed_NNN/
      result.json -> de_results.basins (each basin has hifi_mse which we reuse
      verbatim as hifi_before; no recompute).
  (2) L-BFGS polishes every basin using the same tangent-at-start 6-DOF
      parameterisation + mean-|residual| surrogate cost as m123.
  (3) Hi-fi validates the POLISHED state using the same propagate_attitude +
      brightness_single_epoch flow as m115.hifi_validate (imported, not
      reimplemented), parallelised across (seed x basin) via Pool(fork).
  (4) Applies the keep_better wrapper: hifi_wrapped = min(hifi_before, hifi_after).
  (5) Reports seed-level best_hifi_wrapped = min(hifi_wrapped over basins) and
      compares to m115's best_hifi_mse.

Provenance
----------
- Polish logic copied (not imported) from m123_lbfgs_polish.py to avoid
  cross-file coupling per spec.
- Hi-fi-eval worker pattern copied (not imported) from m124_hifi_validate.py;
  uses fork-based multiprocessing with module-global _WORKER_STATE inherited
  from parent. hifi_validate itself is imported from m115 (same as
  m124).
- ExperimentContext built via setup_experiment (same call pattern as
  m119v2), reusing truth_q0/truth_omega0 from m046_trajectories for
  reproducibility. Constraint epochs = all 500 observations.

Script does NOT modify any library code. All helper functions are duplicated.

Checkpoints (design-first per project discipline)
-------------------------------------------------
Per seed (data/results/inversion_diagnostics/m126_wrapped/seed_NNN/):
  polish_ckpt.npz : per-basin q0_before/after, omega_before/after,
                    surr_mse_before/after, q0_err_before/after,
                    w_dir_err_before/after, w_mag_err_pct_before/after.
                    ALL basins, winners and not.
  hifi_ckpt.npz   : per-basin hifi_before, hifi_after, hifi_wrapped (= min),
                    polish_helped (bool).
  result.json     : per-seed summary with basin-level detail +
                    best_hifi_wrapped, best_hifi_m115, improvement_pct +
                    full timing breakdown.

Top-level:
  batch_summary.json : per-seed rows + population aggregates.
  run.log            : tee'd stdout.

Classification (per m124/m125 convention)
-------------------------------------------------
  OK       hifi_wrapped <  0.01
  PARTIAL  0.01 <= hifi_wrapped < 0.1
  FAIL     hifi_wrapped >= 0.1

Env
---
  (none; process all 6 seeds internally)
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

from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402
from surrogate import SurrogateModel                              # noqa: E402
from lib.experiment_setup import setup_experiment                 # noqa: E402
from lib.traj_source import load_truth, VALID_SOURCES             # noqa: E402
# Reuse hifi_validate exactly as m124 did; do NOT reimplement.
from m115_surrogate_pipeline import hifi_validate             # noqa: E402


# ---- Config --------------------------------------------------------------
# Trajectory source ('m046' legacy single-window, or 'm048' per-seed).
TRAJ_SOURCE = os.environ.get('TRAJ_SOURCE', 'm046').strip() or 'm046'
if TRAJ_SOURCE not in VALID_SOURCES:
    raise ValueError(f"TRAJ_SOURCE must be in {VALID_SOURCES}; got {TRAJ_SOURCE!r}")

DEFAULT_SEEDS = [0, 6, 12, 24, 33, 36]
# Env override for single-seed invert.py driver (comma-separated).
_env_seeds = os.environ.get('MICRO126_SEEDS', '')
if _env_seeds.strip():
    SEEDS = [int(s) for s in _env_seeds.split(',')]
else:
    SEEDS = DEFAULT_SEEDS

POOL_SIZE = 8

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

# m115's output is source-tagged; mirror that here.
if TRAJ_SOURCE == 'm046':
    MICRO115_BASE = RESULTS_DIR / "m115_surrogate_pipeline"
    OUT_BASE = RESULTS_DIR / "m126_wrapped"
else:
    MICRO115_BASE = RESULTS_DIR / f"m115_surrogate_pipeline_{TRAJ_SOURCE}"
    OUT_BASE = RESULTS_DIR / f"m126_wrapped_{TRAJ_SOURCE}"

SURROGATE_WEIGHTS = Path('/home/girish/surrogate_model/s10_5M_weights.npz')
SURROGATE_NORM = Path('/home/girish/surrogate_model/s10_5M_normalization.npz')

PANEL_DEG = 0.0
DISH_DEG = 15.0

N_OBS = 500
NOISE_SEED = 42
NOISE_SIGMA = 0.05

# L-BFGS-B options (identical to m123)
LBFGS_OPTIONS = {
    'ftol': 1e-6,
    'gtol': 1e-3,
    'maxiter': 100,
    'maxfun': 500,
    'disp': False,
}

# Classification thresholds (hi-fi MSE)
CLS_OK_MAX = 0.01
CLS_PARTIAL_MAX = 0.1

# Hypothesis thresholds
IMPROVEMENT_THRESHOLD_PCT = 10.0
N_SEEDS_NEEDED = 3


# ---- Logging -------------------------------------------------------------
class Tee:
    """Dual-stream writer (stdout + log file). Copied from m123/124."""
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
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
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


# ---- Quaternion helpers (copied from m123) ---------------------------
def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def rotvec_to_wxyz(r):
    theta = np.linalg.norm(r)
    if theta < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = r / theta
    ha = theta / 2.0
    s = np.sin(ha)
    return np.array([np.cos(ha), s * axis[0], s * axis[1], s * axis[2]])


def wxyz_to_xyzw(q):
    if q.ndim == 1:
        return np.array([q[1], q[2], q[3], q[0]])
    return q[:, [1, 2, 3, 0]]


def quat_geodesic_deg(q_a_wxyz, q_b_wxyz):
    """Geodesic distance in degrees between two unit quaternions (wxyz)."""
    q_a = q_a_wxyz / np.linalg.norm(q_a_wxyz)
    q_b = q_b_wxyz / np.linalg.norm(q_b_wxyz)
    dot = float(abs(np.dot(q_a, q_b)))
    dot = min(1.0, max(-1.0, dot))
    return float(2.0 * np.degrees(np.arccos(dot)))


def angle_between_unit_vecs_deg(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def tangent_basis_perp(w_hat):
    """Deterministic Gram-Schmidt basis (e1, e2) perp to w_hat, unit norm."""
    w_hat = w_hat / np.linalg.norm(w_hat)
    abs_w = np.abs(w_hat)
    idx = int(np.argmin(abs_w))
    ref = np.zeros(3); ref[idx] = 1.0
    e1 = ref - np.dot(ref, w_hat) * w_hat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(w_hat, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


# ---- 6-DOF tangent-at-start parameterisation (copied from m123) ------
def unpack_param(x, start):
    """
    Apply 6-DOF tangent perturbation at the start point.
    start: dict with keys q0_start (wxyz), w_mag_start, w_hat_start,
           e1_perp, e2_perp.
    Returns (q0_wxyz unit, omega0 rad/s).
    """
    r = x[0:3]
    dq = rotvec_to_wxyz(r)
    q0 = quat_mul_wxyz(dq, start['q0_start'])
    q0 = q0 / np.linalg.norm(q0)

    xi = x[3:5]
    new_dir = start['w_hat_start'] + xi[0] * start['e1_perp'] + xi[1] * start['e2_perp']
    new_dir = new_dir / np.linalg.norm(new_dir)

    delta = x[5]
    new_mag = start['w_mag_start'] * (1.0 + delta)
    omega0 = new_dir * new_mag
    return q0, omega0


# ---- Surrogate mean_L1 cost (copied from m123 eval_cost) -------------
def eval_cost(x, start, polish_ctx, model):
    q0, w = unpack_param(x, start)
    try:
        quats, _ = propagate_attitude(q0, w, polish_ctx['obs_times'],
                                      "tumbling", polish_ctx['inertia_tensor'])
    except Exception:
        return 10.0

    q_ce = quats[polish_ctx['constraint_epochs']]
    R_ce = Rotation.from_quat(wxyz_to_xyzw(q_ce)).as_matrix()
    k1_body = np.einsum('nij,nj->ni', R_ce, polish_ctx['k1_j2000_ce'])
    k2_body = np.einsum('nij,nj->ni', R_ce, polish_ctx['k2_j2000_ce'])
    k1_body /= np.linalg.norm(k1_body, axis=1, keepdims=True)
    k2_body /= np.linalg.norm(k2_body, axis=1, keepdims=True)

    E = len(polish_ctx['constraint_epochs'])
    zeros_E = np.zeros(E, dtype=np.float64)
    dish_E = np.full(E, DISH_DEG, dtype=np.float64)
    dist_E = polish_ctx['obs_dist_ce'].astype(np.float64)

    pred = model.predict_magnitude(k1_body, k2_body, zeros_E, dish_E, dist_E)
    residual = pred.astype(np.float64) - polish_ctx['observed_mag_ce'].astype(np.float64)
    return float(np.mean(np.abs(residual)))


# ---- Basin loading -------------------------------------------------------
def load_m115_basins(seed):
    """Load m115 DE basins; drop NaNs (same rule as m123).

    Also reads the optional `hifi_mags` array from step2_hifi.npz (added with
    the audit-gap-#7 patch) and stashes it on each surviving basin dict as
    `_hifi_mags_before`. Older runs without this field still work — the
    `_hifi_mags_before` key just stays absent.
    """
    path = MICRO115_BASE / f"seed_{seed:03d}" / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing m115 result.json for seed {seed}: {path}")
    with open(path) as f:
        d = json.load(f)
    best_hifi = d.get('best_hifi_mse', None)
    basins = d.get('de_results', {}).get('basins', [])

    # Try to load hifi_mags from step2_hifi.npz (1:1 with `basins` ordering).
    s2_path = MICRO115_BASE / f"seed_{seed:03d}" / "step2_hifi.npz"
    hifi_mags_arr = None
    if s2_path.exists():
        try:
            s2 = np.load(str(s2_path), allow_pickle=True)
            if 'hifi_mags' in s2.files:
                hifi_mags_arr = np.asarray(s2['hifi_mags'])
                if hifi_mags_arr.ndim != 2 or hifi_mags_arr.shape[0] != len(basins):
                    print(f"    [load] step2_hifi hifi_mags shape mismatch "
                          f"({hifi_mags_arr.shape} vs {len(basins)} basins); ignoring")
                    hifi_mags_arr = None
        except Exception as e:
            print(f"    [load] step2_hifi.npz read failed: {e}; ignoring")

    clean = []
    for i, b in enumerate(basins):
        q = np.asarray(b['q0_wxyz'], dtype=np.float64)
        w = np.asarray(b['omega_rad'], dtype=np.float64)
        if np.any(np.isnan(q)) or np.any(np.isnan(w)):
            print(f"    [load] dropping basin {i} (NaN)")
            continue
        if hifi_mags_arr is not None:
            b = {**b, '_hifi_mags_before': hifi_mags_arr[i]}
        clean.append(b)
    return clean, best_hifi


def make_start(q0_wxyz, omega_rad, label, is_twin):
    """Pack a start-point dict used by unpack_param."""
    q0 = np.asarray(q0_wxyz, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    w = np.asarray(omega_rad, dtype=np.float64)
    w_mag = float(np.linalg.norm(w))
    w_hat = w / w_mag
    e1, e2 = tangent_basis_perp(w_hat)
    return {
        'label': label,
        'q0_start': q0,
        'omega0_start': w,
        'w_mag_start': w_mag,
        'w_hat_start': w_hat,
        'e1_perp': e1,
        'e2_perp': e2,
        'is_twin_start': bool(is_twin),
    }


# ---- Seed-level context build --------------------------------------------
def build_seed_context(seed):
    """
    Build BOTH the polish-cost ctx (J2000 sun/obs unit vectors, constraint
    epoch indices, observed mag, etc.) AND the hi-fi ctx (ExperimentContext)
    from a single setup_experiment call. Also returns truth for error metrics.
    """
    print(f"    [ctx] loading truth from {TRAJ_SOURCE} for seed {seed}")
    truth = load_truth(seed, TRAJ_SOURCE)
    truth_q0 = truth['q0_wxyz']
    truth_q0 = truth_q0 / np.linalg.norm(truth_q0)
    truth_omega0 = truth['omega0_rad']

    # skip_true_lc=True because mag_hifi is pre-computed in the trajectory
    # dataset and observed_lc is canonicalised by lib.traj_source.load_truth
    # (same noise realisation as m103 + m115).
    print(f"    [ctx] setup_experiment(n_obs={N_OBS}, skip_true_lc=True, "
          f"start_et={truth['start_et']}, end_time_utc={truth['end_time_utc']})")
    t0 = time.time()
    ctx = setup_experiment(
        n_observations=N_OBS,
        # m046: end_time_utc='2020-02-05T11:00:00' (1-hour window);
        # m048: start_et=<per-seed> + duration_s=3600.0.
        end_time_utc=truth['end_time_utc'],
        start_et=truth['start_et'],
        duration_s=truth['duration_s'],
        skip_true_lc=True,
    )
    # Attach canonical truth + observed LC post-hoc (setup_experiment skipped
    # the ~50s LC regeneration; we're using the pre-computed mag_hifi + the
    # canonical default_rng(42) noise from traj_source).
    ctx.true_q0 = truth_q0
    ctx.true_omega0 = truth_omega0
    ctx.true_lc = truth['mag_hifi'].astype(np.float64)
    ctx.observed_lc = truth['observed_lc'].astype(np.float64)
    print(f"    [ctx] setup_experiment done in {time.time()-t0:.1f}s")

    # J2000 unit vectors (sun, obs) over all observations; constraint = all 500.
    k1_j2000 = ctx.sun_pos - ctx.sat_pos
    k1_j2000 /= np.linalg.norm(k1_j2000, axis=1, keepdims=True)
    k2_j2000 = ctx.obs_pos - ctx.sat_pos
    k2_j2000 /= np.linalg.norm(k2_j2000, axis=1, keepdims=True)
    constraint_epochs = np.arange(N_OBS, dtype=np.int64)

    polish_ctx = {
        'obs_times': ctx.observation_times.astype(np.float64),
        'constraint_epochs': constraint_epochs,
        'k1_j2000_ce': k1_j2000[constraint_epochs].astype(np.float64),
        'k2_j2000_ce': k2_j2000[constraint_epochs].astype(np.float64),
        'obs_dist_ce': ctx.obs_dist[constraint_epochs].astype(np.float64),
        'observed_mag_ce': ctx.observed_lc[constraint_epochs].astype(np.float64),
        'inertia_tensor': ctx.inertia_tensor.astype(np.float64),
    }
    hifi_state = {
        'ctx': ctx,
        'obs_times': ctx.observation_times.astype(np.float64),
        'I_tensor': ctx.inertia_tensor.astype(np.float64),
        'observed_lc': ctx.observed_lc.astype(np.float64),
    }
    return polish_ctx, hifi_state, truth_q0, truth_omega0


# ---- Stage B: L-BFGS polish (serial per basin; spec says not to fuse) ----
def polish_one_basin(basin_idx, basin, polish_ctx, model, truth_q0, truth_w_hat,
                    truth_w_mag):
    """
    L-BFGS polish one DE basin. Returns dict with before/after q0, omega,
    surr_mse, q0_err, w_dir_err, w_mag_err_pct, timings.
    """
    q0_before = np.asarray(basin['q0_wxyz'], dtype=np.float64)
    q0_before /= np.linalg.norm(q0_before)
    omega_before = np.asarray(basin['omega_rad'], dtype=np.float64)
    start = make_start(q0_before, omega_before, f'basin_{basin_idx}',
                       is_twin=basin.get('is_twin', False))

    t0 = time.time()
    x0 = np.zeros(6, dtype=np.float64)
    cost_fn = lambda x: eval_cost(x, start, polish_ctx, model)
    surr_mse_before = float(cost_fn(x0))

    # Record per-iteration x and cost for stage-by-stage viz (audit gap #5).
    _traj_x = [x0.copy()]
    _traj_cost = [surr_mse_before]
    def _cb(xk, *args, **kwargs):
        _traj_x.append(np.asarray(xk, dtype=np.float64).copy())
        _traj_cost.append(float(cost_fn(np.asarray(xk, dtype=np.float64))))

    res = minimize(
        cost_fn,
        x0=x0,
        jac=None,
        method='L-BFGS-B',
        options=LBFGS_OPTIONS,
        callback=_cb,
    )
    surr_mse_after = float(res.fun)
    wall = float(time.time() - t0)

    q0_after, omega_after = unpack_param(np.asarray(res.x, dtype=np.float64), start)

    # Before/after error metrics vs truth
    q0_err_before = quat_geodesic_deg(q0_before, truth_q0)
    q0_err_after = quat_geodesic_deg(q0_after, truth_q0)

    w_hat_before = omega_before / np.linalg.norm(omega_before)
    w_hat_after = omega_after / np.linalg.norm(omega_after)
    w_dir_err_before = angle_between_unit_vecs_deg(w_hat_before, truth_w_hat)
    w_dir_err_after = angle_between_unit_vecs_deg(w_hat_after, truth_w_hat)

    w_mag_before = float(np.linalg.norm(omega_before))
    w_mag_after = float(np.linalg.norm(omega_after))
    w_mag_err_pct_before = (w_mag_before / truth_w_mag - 1.0) * 100.0
    w_mag_err_pct_after = (w_mag_after / truth_w_mag - 1.0) * 100.0

    return {
        'basin_idx': basin_idx,
        'is_twin': bool(basin.get('is_twin', False)),
        # before
        'q0_before': q0_before.tolist(),
        'omega_before': omega_before.tolist(),
        'surr_mse_before': surr_mse_before,
        'q0_err_before': q0_err_before,
        'w_dir_err_before': w_dir_err_before,
        'w_mag_err_pct_before': w_mag_err_pct_before,
        # after
        'q0_after': q0_after.tolist(),
        'omega_after': omega_after.tolist(),
        'surr_mse_after': surr_mse_after,
        'q0_err_after': q0_err_after,
        'w_dir_err_after': w_dir_err_after,
        'w_mag_err_pct_after': w_mag_err_pct_after,
        # diagnostics
        'n_iter': int(getattr(res, 'nit', -1)),
        'n_fev': int(getattr(res, 'nfev', -1)),
        'success': bool(res.success),
        'message': str(res.message),
        'polish_wall_s': wall,
        # trajectory (NPZ-only; stripped before JSON dump in save_polish_ckpt)
        '_traj_x': np.asarray(_traj_x, dtype=np.float64),
        '_traj_cost': np.asarray(_traj_cost, dtype=np.float64),
    }


def save_polish_ckpt(out_path, polish_records):
    """Atomic-ish NPZ save of all per-basin polish arrays. Includes JSON blob
    for the full record list (consumed by hi-fi stage + result.json writer)."""
    n = len(polish_records)
    # Stack arrays from the records
    q0_before = np.array([r['q0_before'] for r in polish_records], dtype=np.float64)
    q0_after = np.array([r['q0_after'] for r in polish_records], dtype=np.float64)
    omega_before = np.array([r['omega_before'] for r in polish_records], dtype=np.float64)
    omega_after = np.array([r['omega_after'] for r in polish_records], dtype=np.float64)
    surr_before = np.array([r['surr_mse_before'] for r in polish_records], dtype=np.float64)
    surr_after = np.array([r['surr_mse_after'] for r in polish_records], dtype=np.float64)
    q0_err_before = np.array([r['q0_err_before'] for r in polish_records], dtype=np.float64)
    q0_err_after = np.array([r['q0_err_after'] for r in polish_records], dtype=np.float64)
    w_dir_err_before = np.array([r['w_dir_err_before'] for r in polish_records], dtype=np.float64)
    w_dir_err_after = np.array([r['w_dir_err_after'] for r in polish_records], dtype=np.float64)
    w_mag_err_pct_before = np.array([r['w_mag_err_pct_before'] for r in polish_records], dtype=np.float64)
    w_mag_err_pct_after = np.array([r['w_mag_err_pct_after'] for r in polish_records], dtype=np.float64)
    basin_idx = np.array([r['basin_idx'] for r in polish_records], dtype=np.int32)
    is_twin = np.array([r['is_twin'] for r in polish_records], dtype=bool)
    n_iter = np.array([r['n_iter'] for r in polish_records], dtype=np.int32)
    n_fev = np.array([r['n_fev'] for r in polish_records], dtype=np.int32)
    polish_wall_s = np.array([r['polish_wall_s'] for r in polish_records], dtype=np.float64)

    # Strip per-iteration trajectories before JSON-dumping (NPZ-only).
    traj_x = np.asarray([r.pop('_traj_x', np.zeros((0, 6))) for r in polish_records], dtype=object)
    traj_cost = np.asarray([r.pop('_traj_cost', np.zeros((0,))) for r in polish_records], dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        q0_before=q0_before, q0_after=q0_after,
        omega_before=omega_before, omega_after=omega_after,
        surr_mse_before=surr_before, surr_mse_after=surr_after,
        q0_err_before=q0_err_before, q0_err_after=q0_err_after,
        w_dir_err_before=w_dir_err_before, w_dir_err_after=w_dir_err_after,
        w_mag_err_pct_before=w_mag_err_pct_before,
        w_mag_err_pct_after=w_mag_err_pct_after,
        basin_idx=basin_idx, is_twin=is_twin,
        n_iter=n_iter, n_fev=n_fev,
        polish_wall_s=polish_wall_s,
        n_basins=np.array(n),
        records_json=np.array(json.dumps(polish_records, default=_json_default)),
        traj_x=traj_x,
        traj_cost=traj_cost,
    )


# ---- Stage C: hi-fi validation (parallel across seed x basin) -----------
# Module-global state inherited by forked workers.
_WORKER_HIFI_STATE = {}  # seed -> {ctx, obs_times, I_tensor, observed_lc}


def hifi_worker(job):
    """
    Worker: compute hi-fi MSE for a polished (q0_after, omega_after) of one
    (seed, basin_idx). Uses module-global _WORKER_HIFI_STATE inherited via fork.
    """
    seed = job['seed']
    basin_idx = job['basin_idx']
    q0 = np.asarray(job['q0_wxyz'], dtype=np.float64)
    w = np.asarray(job['omega_rad'], dtype=np.float64)

    state = _WORKER_HIFI_STATE[seed]
    t0 = time.time()
    hifi_mse, hifi_mags = hifi_validate(
        q0, w,
        state['obs_times'], state['I_tensor'],
        state['observed_lc'], state['ctx'])
    wall = time.time() - t0

    return {
        'seed': seed,
        'basin_idx': basin_idx,
        'hifi_after': float(hifi_mse),
        'hifi_mags_after': np.asarray(hifi_mags, dtype=np.float64),
        'hifi_wall_s': float(wall),
    }


def save_hifi_ckpt(out_path, hifi_rows):
    """Save per-basin hifi_before/after/wrapped + polish_helped flag."""
    n = len(hifi_rows)
    basin_idx = np.array([r['basin_idx'] for r in hifi_rows], dtype=np.int32)
    hifi_before = np.array([r['hifi_before'] for r in hifi_rows], dtype=np.float64)
    hifi_after = np.array([r['hifi_after'] for r in hifi_rows], dtype=np.float64)
    hifi_wrapped = np.array([r['hifi_wrapped'] for r in hifi_rows], dtype=np.float64)
    polish_helped = np.array([r['polish_helped'] for r in hifi_rows], dtype=bool)
    hifi_mags_after = np.stack(
        [np.asarray(r['hifi_mags_after']) for r in hifi_rows])  # (n, N_OBS)
    hifi_wall_s = np.array([r['hifi_wall_s'] for r in hifi_rows], dtype=np.float64)

    # Optional before-LCs (audit gap #8) — present when m115 saved hifi_mags
    # in step2_hifi.npz; otherwise field omitted.
    has_before = all(r.get('hifi_mags_before') is not None for r in hifi_rows)
    extras = {}
    if has_before:
        extras['hifi_mags_before'] = np.stack(
            [np.asarray(r['hifi_mags_before']) for r in hifi_rows])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        basin_idx=basin_idx,
        hifi_before=hifi_before,
        hifi_after=hifi_after,
        hifi_wrapped=hifi_wrapped,
        polish_helped=polish_helped,
        hifi_mags_after=hifi_mags_after,
        hifi_wall_s=hifi_wall_s,
        n_basins=np.array(n),
        **extras,
    )


# ---- Classification ------------------------------------------------------
def classify(hifi_wrapped):
    if hifi_wrapped < CLS_OK_MAX:
        return 'OK'
    elif hifi_wrapped < CLS_PARTIAL_MAX:
        return 'PARTIAL'
    else:
        return 'FAIL'


# ---- Main ----------------------------------------------------------------
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    log_path = OUT_BASE / "run.log"
    log_f = open(log_path, 'w')
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    timing = {
        'wall_total_s': 0.0,
        'wall_ctx_build_s': 0.0,
        'wall_polish_total_s': 0.0,
        'wall_hifi_total_s': 0.0,
        'per_seed_ctx_build_s': {},
        'per_seed_polish_s': {},
        'per_seed_hifi_s': {},  # wall clock the seed's jobs sat in the pool
    }
    t_all = time.time()

    try:
        print("=" * 72)
        print("m126 -- wrapped pipeline (DE -> polish -> hi-fi(before,after) -> keep-min)")
        print("=" * 72)
        print(f"  traj_source        = {TRAJ_SOURCE}")
        print(f"  seeds              = {SEEDS}")
        print(f"  m115 input         = {MICRO115_BASE}")
        print(f"  POOL_SIZE          = {POOL_SIZE}")
        print(f"  LBFGS_OPTIONS      = {LBFGS_OPTIONS}")
        print(f"  classification     = OK<{CLS_OK_MAX}, PARTIAL<{CLS_PARTIAL_MAX}, FAIL>=")
        print(f"  improvement thresh = {IMPROVEMENT_THRESHOLD_PCT}%")
        print(f"  n_seeds needed     = {N_SEEDS_NEEDED}/{len(SEEDS)}")
        print(f"  out_dir            = {OUT_BASE}")

        # ── Stage A: build contexts + load basins ──────────────────────
        print("\n" + "=" * 72)
        print("[Stage A] Build ExperimentContext + load DE basins per seed (serial)")
        print("=" * 72)
        t_ctx_all = time.time()
        seed_state = {}  # seed -> dict of everything for this seed
        global _WORKER_HIFI_STATE
        for seed in SEEDS:
            t_s = time.time()
            print(f"\n  seed {seed}")
            basins, m115_best_hifi = load_m115_basins(seed)
            print(f"    [load] {len(basins)} non-NaN basins  "
                  f"m115_best_hifi={m115_best_hifi}")
            polish_ctx, hifi_state, truth_q0, truth_omega0 = build_seed_context(seed)
            truth_w_mag = float(np.linalg.norm(truth_omega0))
            truth_w_hat = truth_omega0 / truth_w_mag

            seed_state[seed] = {
                'basins': basins,
                'm115_best_hifi': m115_best_hifi,
                'polish_ctx': polish_ctx,
                'truth_q0': truth_q0,
                'truth_omega0': truth_omega0,
                'truth_w_mag': truth_w_mag,
                'truth_w_hat': truth_w_hat,
            }
            # hifi_state goes in the worker-global dict (forked into workers)
            _WORKER_HIFI_STATE[seed] = hifi_state

            dt = time.time() - t_s
            timing['per_seed_ctx_build_s'][str(seed)] = round(dt, 2)
            print(f"    seed {seed} ctx+basins ready in {dt:.1f}s")

        timing['wall_ctx_build_s'] = round(time.time() - t_ctx_all, 2)
        print(f"\n  Stage A total: {timing['wall_ctx_build_s']:.1f}s")

        # ── Stage B: L-BFGS polish (serial per basin, sequential over seeds) ──
        print("\n" + "=" * 72)
        print("[Stage B] L-BFGS polish per basin (serial; fast ~30s/basin)")
        print("=" * 72)
        t_polish_all = time.time()

        # Surrogate model loaded ONCE in parent (shared across all polish calls
        # because polish runs serial in this process).
        print("  loading surrogate model...")
        t_mdl = time.time()
        model = SurrogateModel(str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))
        print(f"    model loaded in {time.time()-t_mdl:.1f}s")

        for seed in SEEDS:
            t_s = time.time()
            state = seed_state[seed]
            print(f"\n  seed {seed}: polishing {len(state['basins'])} basins...")
            polish_records = []
            for bi, basin in enumerate(state['basins']):
                t_b = time.time()
                rec = polish_one_basin(
                    bi, basin, state['polish_ctx'], model,
                    state['truth_q0'], state['truth_w_hat'], state['truth_w_mag'])
                polish_records.append(rec)
                print(f"    basin {bi}: surr {rec['surr_mse_before']:.6f} -> "
                      f"{rec['surr_mse_after']:.6f}  "
                      f"q0_err {rec['q0_err_before']:.2f}° -> {rec['q0_err_after']:.2f}°  "
                      f"w_dir {rec['w_dir_err_before']:.3f}° -> {rec['w_dir_err_after']:.3f}°  "
                      f"iter={rec['n_iter']} fev={rec['n_fev']} "
                      f"({time.time()-t_b:.1f}s)")
            state['polish_records'] = polish_records

            # Stage B checkpoint per seed.
            ckpt_path = OUT_BASE / f"seed_{seed:03d}" / "polish_ckpt.npz"
            save_polish_ckpt(ckpt_path, polish_records)
            print(f"    saved {ckpt_path}")

            dt = time.time() - t_s
            timing['per_seed_polish_s'][str(seed)] = round(dt, 2)
            print(f"    seed {seed} polish total: {dt:.1f}s")

        timing['wall_polish_total_s'] = round(time.time() - t_polish_all, 2)
        print(f"\n  Stage B total: {timing['wall_polish_total_s']:.1f}s")

        # Free surrogate model (no longer needed; hi-fi uses full pipeline).
        del model

        # ── Stage C: hi-fi validate polished states in parallel ────────
        print("\n" + "=" * 72)
        print(f"[Stage C] Hi-fi validate polished states on Pool({POOL_SIZE}) "
              "(across seed x basin)")
        print("=" * 72)
        t_hifi_all = time.time()

        jobs = []
        for seed in SEEDS:
            for rec in seed_state[seed]['polish_records']:
                jobs.append({
                    'seed': seed,
                    'basin_idx': rec['basin_idx'],
                    'q0_wxyz': rec['q0_after'],
                    'omega_rad': rec['omega_after'],
                })
        n_jobs = len(jobs)
        print(f"  dispatching {n_jobs} hi-fi evals "
              f"({sum(len(s['polish_records']) for s in seed_state.values())} "
              f"polished basins across {len(SEEDS)} seeds)")

        results_by_key = {}
        t_pool = time.time()
        with mp.Pool(POOL_SIZE) as pool:
            for k, res in enumerate(pool.imap_unordered(hifi_worker, jobs)):
                key = (res['seed'], res['basin_idx'])
                results_by_key[key] = res
                print(f"    [{k+1}/{n_jobs}] seed {res['seed']} basin "
                      f"{res['basin_idx']}: hifi_after={res['hifi_after']:.6f}  "
                      f"({res['hifi_wall_s']:.1f}s)")
        print(f"  pool done in {time.time()-t_pool:.1f}s")

        timing['wall_hifi_total_s'] = round(time.time() - t_hifi_all, 2)

        # ── Stage D: assemble per-seed + batch summaries ───────────────
        print("\n" + "=" * 72)
        print("[Stage D] Wrapper + seed-level assembly")
        print("=" * 72)

        per_seed_rows = []
        total_basins = 0
        total_helped = 0
        total_hurt = 0

        for seed in SEEDS:
            state = seed_state[seed]
            hifi_rows = []
            for rec in state['polish_records']:
                bi = rec['basin_idx']
                source_basin = state['basins'][bi]
                hifi_before = float(source_basin['hifi_mse'])
                res = results_by_key[(seed, bi)]
                hifi_after = float(res['hifi_after'])
                hifi_wrapped = min(hifi_before, hifi_after)
                polish_helped = hifi_after < hifi_before
                hifi_rows.append({
                    'basin_idx': bi,
                    'is_twin': rec['is_twin'],
                    # polish summary (for convenience in result.json)
                    'q0_before': rec['q0_before'],
                    'q0_after': rec['q0_after'],
                    'omega_before': rec['omega_before'],
                    'omega_after': rec['omega_after'],
                    'surr_mse_before': rec['surr_mse_before'],
                    'surr_mse_after': rec['surr_mse_after'],
                    'q0_err_before': rec['q0_err_before'],
                    'q0_err_after': rec['q0_err_after'],
                    'w_dir_err_before': rec['w_dir_err_before'],
                    'w_dir_err_after': rec['w_dir_err_after'],
                    'w_mag_err_pct_before': rec['w_mag_err_pct_before'],
                    'w_mag_err_pct_after': rec['w_mag_err_pct_after'],
                    'n_iter': rec['n_iter'],
                    'n_fev': rec['n_fev'],
                    # hi-fi
                    'hifi_before': hifi_before,
                    'hifi_after': hifi_after,
                    'hifi_wrapped': hifi_wrapped,
                    'polish_helped': polish_helped,
                    'hifi_mags_after': res['hifi_mags_after'],
                    'hifi_mags_before': source_basin.get('_hifi_mags_before'),
                    'polish_wall_s': rec['polish_wall_s'],
                    'hifi_wall_s': res['hifi_wall_s'],
                })
                total_basins += 1
                if polish_helped:
                    total_helped += 1
                else:
                    total_hurt += 1

            # Persist hifi_ckpt.npz per seed.
            hifi_ckpt_path = OUT_BASE / f"seed_{seed:03d}" / "hifi_ckpt.npz"
            save_hifi_ckpt(hifi_ckpt_path, hifi_rows)
            print(f"  seed {seed}: saved {hifi_ckpt_path}")

            # Seed-level best + improvement
            best_hifi_wrapped = min(r['hifi_wrapped'] for r in hifi_rows)
            best_hifi_after = min(r['hifi_after'] for r in hifi_rows)
            best_hifi_before = min(r['hifi_before'] for r in hifi_rows)
            m115_best = float(state['m115_best_hifi']) if state['m115_best_hifi'] \
                is not None else float('nan')
            if np.isfinite(m115_best) and m115_best > 0:
                improvement_pct = (m115_best - best_hifi_wrapped) / m115_best * 100.0
            else:
                improvement_pct = None
            cls = classify(best_hifi_wrapped)

            # Strip the large hifi_mags_{before,after} arrays before JSON dump.
            basin_rows_for_json = []
            for r in hifi_rows:
                r_json = {k: v for k, v in r.items() if k not in ('hifi_mags_after', 'hifi_mags_before')}
                basin_rows_for_json.append(r_json)

            timing_seed = {
                'ctx_build_s': timing['per_seed_ctx_build_s'].get(str(seed)),
                'polish_total_s': timing['per_seed_polish_s'].get(str(seed)),
                # seed-level hi-fi wall is sum of its basin hi-fi walls (CPU sum,
                # NOT wall; pool is shared across seeds). Informational only.
                'hifi_cpu_sum_s': round(sum(r['hifi_wall_s'] for r in hifi_rows), 2),
                'n_basins': len(hifi_rows),
            }

            per_seed_summary = {
                'seed': int(seed),
                'n_basins': len(hifi_rows),
                'best_hifi_m115': m115_best,
                'best_hifi_wrapped': float(best_hifi_wrapped),
                'best_hifi_naive_polish': float(best_hifi_after),
                'best_hifi_before_polish': float(best_hifi_before),
                'improvement_pct': (float(improvement_pct)
                                    if improvement_pct is not None else None),
                'classification': cls,
                'n_basins_helped': sum(1 for r in hifi_rows if r['polish_helped']),
                'n_basins_hurt': sum(1 for r in hifi_rows if not r['polish_helped']),
                'basins': basin_rows_for_json,
                'timing': timing_seed,
            }
            atomic_json_save(
                OUT_BASE / f"seed_{seed:03d}" / "result.json", per_seed_summary)
            print(f"    best_m115={m115_best:.6f}  "
                  f"best_wrapped={best_hifi_wrapped:.6f}  "
                  f"improvement={improvement_pct:.1f}% "
                  f"[{cls}]  "
                  f"helped={per_seed_summary['n_basins_helped']}/"
                  f"{per_seed_summary['n_basins']}")

            per_seed_rows.append(per_seed_summary)

        # ── Stage E: batch_summary.json ─────────────────────────────────
        timing['wall_total_s'] = round(time.time() - t_all, 2)

        n_improved_10 = sum(
            1 for r in per_seed_rows
            if r['improvement_pct'] is not None
            and r['improvement_pct'] >= IMPROVEMENT_THRESHOLD_PCT)
        n_improved_any = sum(
            1 for r in per_seed_rows
            if r['improvement_pct'] is not None
            and r['improvement_pct'] > 0)

        if n_improved_10 >= N_SEEDS_NEEDED:
            verdict = 'CONFIRMED'
            reason = (f'{n_improved_10}/{len(SEEDS)} seeds improved >='
                      f'{IMPROVEMENT_THRESHOLD_PCT}%; >= {N_SEEDS_NEEDED} needed')
        else:
            verdict = 'REFUTED'
            reason = (f'only {n_improved_10}/{len(SEEDS)} seeds improved '
                      f'>={IMPROVEMENT_THRESHOLD_PCT}%; needed {N_SEEDS_NEEDED}')

        # Tally classifications
        cls_counts = {'OK': 0, 'PARTIAL': 0, 'FAIL': 0}
        for r in per_seed_rows:
            cls_counts[r['classification']] += 1

        # Strip basins[] from the batch-level rows to keep the file lean (full
        # basin detail is in each seed's result.json).
        batch_rows = []
        for r in per_seed_rows:
            batch_rows.append({k: v for k, v in r.items() if k != 'basins'})

        batch_summary = {
            'experiment': 'm126_wrapped_pipeline',
            'traj_source': TRAJ_SOURCE,
            'hypothesis': (
                f'>= {N_SEEDS_NEEDED} of {len(SEEDS)} seeds improved by >= '
                f'{IMPROVEMENT_THRESHOLD_PCT}% via keep_better wrapper'),
            'seeds': SEEDS,
            'n_seeds': len(SEEDS),
            'classification_thresholds': {
                'OK_max': CLS_OK_MAX, 'PARTIAL_max': CLS_PARTIAL_MAX},
            'improvement_threshold_pct': IMPROVEMENT_THRESHOLD_PCT,
            'n_seeds_needed': N_SEEDS_NEEDED,
            'per_seed': batch_rows,
            'aggregate': {
                'n_seeds_improved_ge10pct': n_improved_10,
                'n_seeds_improved_any': n_improved_any,
                'n_basins_total': total_basins,
                'n_basins_helped': total_helped,
                'n_basins_hurt': total_hurt,
                'classification_counts': cls_counts,
            },
            'verdict': verdict,
            'reason': reason,
            'lbfgs_options': LBFGS_OPTIONS,
            'pool_size': POOL_SIZE,
            'timing': timing,
        }
        atomic_json_save(OUT_BASE / "batch_summary.json", batch_summary)

        # Console report
        print("\n" + "=" * 72)
        print("SUMMARY")
        print("=" * 72)
        print(f"{'seed':>5}  {'m115_best':>10}  {'wrapped':>10}  "
              f"{'improv%':>8}  {'cls':>8}  helped/total")
        for r in per_seed_rows:
            ip = r['improvement_pct']
            ip_str = f"{ip:>7.1f}%" if ip is not None else f"{'--':>8}"
            print(f"{r['seed']:>5}  {r['best_hifi_m115']:>10.6f}  "
                  f"{r['best_hifi_wrapped']:>10.6f}  "
                  f"{ip_str}  {r['classification']:>8}  "
                  f"{r['n_basins_helped']}/{r['n_basins']}")
        print()
        print(f"  n_seeds_improved_ge10pct = {n_improved_10}/{len(SEEDS)} "
              f"(threshold {N_SEEDS_NEEDED})")
        print(f"  n_seeds_improved_any      = {n_improved_any}/{len(SEEDS)}")
        print(f"  n_basins_helped           = {total_helped}/{total_basins}")
        print(f"  n_basins_hurt             = {total_hurt}/{total_basins}")
        print(f"  classification counts     = {cls_counts}")
        print()
        print(f"  VERDICT: {verdict}")
        print(f"  reason:  {reason}")
        print(f"\n  total wall: {timing['wall_total_s']:.1f}s "
              f"(ctx={timing['wall_ctx_build_s']:.1f}s, "
              f"polish={timing['wall_polish_total_s']:.1f}s, "
              f"hifi={timing['wall_hifi_total_s']:.1f}s)")
        print(f"\n  saved {OUT_BASE / 'batch_summary.json'}")

    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == '__main__':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
