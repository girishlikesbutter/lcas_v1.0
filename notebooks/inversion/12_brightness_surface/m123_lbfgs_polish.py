#!/usr/bin/env python3
"""
m123 -- L-BFGS polish from truth + m115 DE basins.

Hypotheses (3 falsifiable claims)
---------------------------------
1. Surrogate-truth offset is small. L-BFGS started from truth converges to a
   point offset from truth by < 1° q0 AND < 0.01° ω-direction AND < 0.01%
   ω-magnitude. REFUTED ⇔ surrogate's local minimum is displaced from truth
   beyond inversion-relevant precision => gradient-based inversion returns wrong
   answers even with a perfect initialiser.

2. DE basins do not benefit from GD polish. L-BFGS started from each m115
   DE basin (starting q0 within ~2-180° of truth, ω within ~3° / ~10%)
   converges back to a point within 0.1° q0 AND 0.005° ω-dir of its start.
   CONFIRMED ⇔ DE basins are outside the gradient-bearing region (saturated per
   dark-mag-saturation); REFUTED ⇔ GD-polish-post-DE genuinely improves attitude
   and is the real gradient-based pipeline.

3. L-BFGS attractor count ≤ m115 basin count per seed. Upper-bounded
   because GD can only converge to minima that already exist near the start
   points. Equality => DE found all attractors; inequality (fewer) => some DE
   basins collapse to the same GD minimum.

Method -- 6-DOF tangent-at-START parameterisation
-------------------------------------------------
CRITICAL: differs from m122 which parameterised tangent at TRUTH for ALL
candidates. Here each start has its own tangent basis because L-BFGS moves
incrementally from its START, not from truth.

For start k with (q0_start, ω_start) and ω̂_start = ω_start / |ω_start|:
  param[0:3] = r ∈ ℝ³ Rodrigues (radians). q0_pert = quat(r) ∘_LEFT q0_start.
  param[3:5] = ξ ∈ ℝ² tangent plane ⟂ ω̂_start via Gram-Schmidt basis
               (e1_perp, e2_perp). ω̂_pert = normalize(ω̂_start + ξ[0]·e1 +
               ξ[1]·e2).
  param[5]   = δ. |ω|_pert = |ω_start| · (1 + δ).
At param=0 the parameterisation reproduces the start point exactly.

Cost (identical to m121/122): mean |predicted_mag - observed_mag| over
500 constraint epochs, surrogate panel=0°, dish=15°.

Optimiser
---------
scipy.optimize.minimize(cost, x0=zeros(6), method='L-BFGS-B', jac=None,
  options={'ftol':1e-6,'gtol':1e-3,'maxiter':100,'maxfun':500,'disp':False})
jac=None => scipy uses 2-sided FD. No custom Jacobian.

Stages (design-first checkpoints)
---------------------------------
  A: reuse m122 setup.npz (exists for seeds 14/27/46/74/93).
  B: per-seed polish.npz -- per-start L-BFGS trajectories (start/final params,
     costs, grads, n_iter/n_fev, clustering data).
  C: per-seed summary.json -- record list + attractor clustering + per-seed
     verdicts.
  D: run-level summary.json -- aggregate verdicts.

Each stage skip-if-exists unless MICRO123_FORCE=1.

Env
---
MICRO123_SEEDS   comma-separated list, default "14,27,46,74,93"
MICRO123_FORCE   "1" = ignore cache

Outputs per seed (data/results/inversion_diagnostics/m123/seed_NNN/):
  polish.npz       per-start arrays (start_params, final_params, costs, grads,
                   n_iter/n_fev, clustering data)
  summary.json     per-seed record list + clustering + verdicts
  run.log

Run-level: data/results/inversion_diagnostics/m123/summary.json

TODOs (deferred to v2 -- not implemented)
-----------------------------------------
- Richardson extrapolation / alternate FD step sizes for L-BFGS FD jac.
- Alternative cost variants (mean_L2).
- Plotting: per-seed attractor scatter, convergence traces.
- Save predicted hi-fi light curves at each final point.
- Sensitivity to options (ftol/gtol/maxiter) — only one setting probed.
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
sys.path.insert(0, '/home/girish/surrogate_model')
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_attitude
from surrogate import SurrogateModel
from lib.experiment_setup import setup_experiment  # noqa: F401 (used for fallback setup)


# ---- Config --------------------------------------------------------------
SEEDS = [int(s) for s in os.environ.get(
    'MICRO123_SEEDS', '14,27,46,74,93').split(',') if s.strip()]
FORCE = os.environ.get('MICRO123_FORCE', '0') == '1'

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
MICRO46_NPZ = RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"
MICRO122_BASE = RESULTS_DIR / "m122"
MICRO115_BASE = RESULTS_DIR / "m115_surrogate_pipeline"

OUT_BASE = RESULTS_DIR / "m123"

SURROGATE_WEIGHTS = Path('/home/girish/surrogate_model/s10_5M_weights.npz')
SURROGATE_NORM = Path('/home/girish/surrogate_model/s10_5M_normalization.npz')

PANEL_DEG = 0.0
DISH_DEG = 15.0

# L-BFGS-B options
LBFGS_OPTIONS = {
    'ftol': 1e-6,
    'gtol': 1e-3,
    'maxiter': 100,
    'maxfun': 500,
    'disp': False,
}

# Attractor clustering thresholds
CLUSTER_Q0_DEG = 1.0      # geodesic q0 distance
CLUSTER_WDIR_DEG = 0.01   # ω̂ angular distance
CLUSTER_WMAG_PCT = 0.05   # |ω| percent-difference

# Hypothesis-2 per-start thresholds
HYP2_Q0_MOVE_DEG = 0.1
HYP2_WDIR_MOVE_DEG = 0.005


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


# ---- Quaternion helpers --------------------------------------------------
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


# ---- Parameterisation: param (6D) -> (q0, omega0) -----------------------
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


# ---- Cost evaluation (identical to m121/122 mean_L1 path) -----------
def eval_cost(x, start, ctx, model):
    q0, w = unpack_param(x, start)
    try:
        quats, _ = propagate_attitude(q0, w, ctx['obs_times'],
                                      "tumbling", ctx['inertia_tensor'])
    except Exception:
        return 10.0

    q_ce = quats[ctx['constraint_epochs']]
    R_ce = Rotation.from_quat(wxyz_to_xyzw(q_ce)).as_matrix()
    k1_body = np.einsum('nij,nj->ni', R_ce, ctx['k1_j2000_ce'])
    k2_body = np.einsum('nij,nj->ni', R_ce, ctx['k2_j2000_ce'])
    k1_body /= np.linalg.norm(k1_body, axis=1, keepdims=True)
    k2_body /= np.linalg.norm(k2_body, axis=1, keepdims=True)

    E = len(ctx['constraint_epochs'])
    zeros_E = np.zeros(E, dtype=np.float64)
    dish_E = np.full(E, DISH_DEG, dtype=np.float64)
    dist_E = ctx['obs_dist_ce'].astype(np.float64)

    pred = model.predict_magnitude(k1_body, k2_body, zeros_E, dish_E, dist_E)
    residual = pred.astype(np.float64) - ctx['observed_mag_ce'].astype(np.float64)
    return float(np.mean(np.abs(residual)))


# ---- Stage A: load setup from m122 ----------------------------------
def stage_A_setup(seed, out_dir):
    print(f"\n[Stage A] Setup for seed {seed}")
    t0 = time.time()

    m122_setup = MICRO122_BASE / f"seed_{seed:03d}" / "setup.npz"
    if m122_setup.exists():
        print(f"  reusing m122 setup: {m122_setup}")
        setup = dict(np.load(m122_setup, allow_pickle=False))
        print(f"  Stage A done in {time.time()-t0:.1f}s")
        return setup

    # Fallback: m119v2 setup (some seeds may only have this)
    v2_setup = RESULTS_DIR / "m119v2" / f"seed_{seed:03d}" / "setup.npz"
    if v2_setup.exists():
        print(f"  reusing m119v2 setup: {v2_setup}")
        setup = dict(np.load(v2_setup, allow_pickle=False))
        print(f"  Stage A done in {time.time()-t0:.1f}s")
        return setup

    raise FileNotFoundError(
        f"No setup.npz found for seed {seed} (checked m122 and m119v2). "
        f"Run m122 first to materialise setup.")


# ---- Stage B helpers: assemble starts ------------------------------------
def load_truth(seed):
    m46 = np.load(MICRO46_NPZ)
    q0 = m46['q0s'][seed].astype(np.float64)
    q0 = q0 / np.linalg.norm(q0)
    omega0 = m46['omega0s'][seed].astype(np.float64)
    return q0, omega0


def load_m115_basins(seed):
    path = MICRO115_BASE / f"seed_{seed:03d}" / "result.json"
    if not path.exists():
        print(f"  WARNING: no m115 result.json at {path}; only truth start will run")
        return []
    with open(path) as f:
        d = json.load(f)
    basins = d.get('de_results', {}).get('basins', [])
    # Skip basins with NaN q0_wxyz or omega_rad
    clean = []
    for b in basins:
        q = np.asarray(b['q0_wxyz'], dtype=np.float64)
        w = np.asarray(b['omega_rad'], dtype=np.float64)
        if np.any(np.isnan(q)) or np.any(np.isnan(w)):
            continue
        clean.append(b)
    return clean


def make_start(q0_wxyz, omega_rad, label, is_twin):
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


# ---- Stage B: per-start L-BFGS polish ------------------------------------
def run_one_start(start, ctx, model, truth_q0, truth_w_hat, truth_w_mag):
    t0 = time.time()
    x0 = np.zeros(6, dtype=np.float64)

    cost_fn = lambda x: eval_cost(x, start, ctx, model)

    # Pre-call cost at x0 for diagnostics (cheap; one extra eval)
    start_cost = cost_fn(x0)

    res = minimize(
        cost_fn,
        x0=x0,
        jac=None,              # 2-sided FD inside L-BFGS-B
        method='L-BFGS-B',
        options=LBFGS_OPTIONS,
    )

    final_param = np.asarray(res.x, dtype=np.float64)
    final_cost = float(res.fun)
    final_grad = np.asarray(res.jac, dtype=np.float64) if res.jac is not None else np.full(6, np.nan)
    n_iter = int(getattr(res, 'nit', -1))
    n_fev = int(getattr(res, 'nfev', -1))
    success = bool(res.success)
    message = str(res.message)

    q0_final, w_final = unpack_param(final_param, start)

    # distances to truth
    start_q0_err = quat_geodesic_deg(start['q0_start'], truth_q0)
    final_q0_err = quat_geodesic_deg(q0_final, truth_q0)
    q0_move = quat_geodesic_deg(q0_final, start['q0_start'])

    start_w_dir_err = angle_between_unit_vecs_deg(start['w_hat_start'], truth_w_hat)
    final_w_hat = w_final / np.linalg.norm(w_final)
    final_w_dir_err = angle_between_unit_vecs_deg(final_w_hat, truth_w_hat)
    w_dir_move = angle_between_unit_vecs_deg(final_w_hat, start['w_hat_start'])

    start_w_mag = start['w_mag_start']
    final_w_mag = float(np.linalg.norm(w_final))
    start_w_mag_err_pct = (start_w_mag / truth_w_mag - 1.0) * 100.0
    final_w_mag_err_pct = (final_w_mag / truth_w_mag - 1.0) * 100.0
    w_mag_move_pct = (final_w_mag / start_w_mag - 1.0) * 100.0

    wall = float(time.time() - t0)

    return {
        'start_label': start['label'],
        'is_twin_start': start['is_twin_start'],
        'start_q0_wxyz': start['q0_start'].tolist(),
        'start_omega_rad': start['omega0_start'].tolist(),
        'start_cost': float(start_cost),
        'final_param': final_param.tolist(),
        'final_q0_wxyz': q0_final.tolist(),
        'final_omega_rad': w_final.tolist(),
        'final_cost': final_cost,
        'final_grad': final_grad.tolist(),
        'final_grad_norm': float(np.linalg.norm(final_grad)) if np.all(np.isfinite(final_grad)) else None,
        'n_iter': n_iter,
        'n_fev': n_fev,
        'success': success,
        'message': message,
        'start_q0_err_deg': start_q0_err,
        'final_q0_err_deg': final_q0_err,
        'q0_move_deg': q0_move,
        'start_w_dir_err_deg': start_w_dir_err,
        'final_w_dir_err_deg': final_w_dir_err,
        'w_dir_move_deg': w_dir_move,
        'start_w_mag_err_pct': start_w_mag_err_pct,
        'final_w_mag_err_pct': final_w_mag_err_pct,
        'w_mag_move_pct': w_mag_move_pct,
        'wall_seconds': wall,
    }


def stage_B_polish(seed, setup, out_dir):
    print(f"\n[Stage B] L-BFGS polish for seed {seed}")
    t0 = time.time()
    out_npz = out_dir / "polish.npz"

    # Build ctx
    ctx = {
        'obs_times': setup['obs_times'].astype(np.float64),
        'constraint_epochs': setup['constraint_epochs'].astype(np.int64),
        'k1_j2000_ce': setup['k1_j2000_ce'].astype(np.float64),
        'k2_j2000_ce': setup['k2_j2000_ce'].astype(np.float64),
        'obs_dist_ce': setup['obs_dist_ce'].astype(np.float64),
        'observed_mag_ce': setup['observed_mag_ce'].astype(np.float64),
        'inertia_tensor': setup['inertia_tensor'].astype(np.float64),
    }

    # Truth reference (for error metrics)
    truth_q0 = setup['truth_q0'].astype(np.float64)
    truth_q0 = truth_q0 / np.linalg.norm(truth_q0)
    truth_omega0 = setup['truth_omega0'].astype(np.float64)
    truth_w_mag = float(np.linalg.norm(truth_omega0))
    truth_w_hat = truth_omega0 / truth_w_mag

    # Assemble starts
    starts = []
    # truth start
    starts.append(make_start(truth_q0, truth_omega0, 'truth', is_twin=False))
    # m115 DE basins
    basins = load_m115_basins(seed)
    print(f"  m115 basins: {len(basins)}")
    for i, b in enumerate(basins):
        starts.append(make_start(
            q0_wxyz=b['q0_wxyz'],
            omega_rad=b['omega_rad'],
            label=f'basin_{i}',
            is_twin=b.get('is_twin', False),
        ))
    n_starts = len(starts)
    n_basins = len(basins)
    print(f"  total starts: {n_starts} (1 truth + {n_basins} basins)")

    if out_npz.exists() and not FORCE:
        print(f"  loading cached {out_npz.name}")
        d = dict(np.load(out_npz, allow_pickle=True))
        print(f"  cached n_starts={int(d['n_starts'])}")
        print(f"  Stage B (cached) done in {time.time()-t0:.1f}s")
        # Also need per-record dicts for summary; reconstruct from npz arrays
        records = json.loads(str(d['records_json']))
        return {'records': records, 'n_basins': int(d['n_basins']),
                'n_starts': int(d['n_starts'])}

    model = SurrogateModel(str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))
    print(f"  loaded surrogate model")

    records = []
    for k, start in enumerate(starts):
        print(f"  [{k+1}/{n_starts}] start={start['label']} "
              f"(is_twin={start['is_twin_start']})")
        rec = run_one_start(start, ctx, model,
                            truth_q0, truth_w_hat, truth_w_mag)
        print(f"    start_cost={rec['start_cost']:.5f}  "
              f"final_cost={rec['final_cost']:.5f}  "
              f"iter={rec['n_iter']}  fev={rec['n_fev']}  "
              f"wall={rec['wall_seconds']:.1f}s")
        print(f"    start→final: q0_move={rec['q0_move_deg']:.4f}°  "
              f"w_dir_move={rec['w_dir_move_deg']:.5f}°  "
              f"w_mag_move={rec['w_mag_move_pct']:.4f}%")
        print(f"    final vs truth: q0={rec['final_q0_err_deg']:.4f}°  "
              f"w_dir={rec['final_w_dir_err_deg']:.5f}°  "
              f"w_mag={rec['final_w_mag_err_pct']:.4f}%")
        records.append(rec)

    # --- checkpoint: polish.npz (arrays for convenience + raw JSON blob) ---
    start_params = np.array([r['start_q0_wxyz'] + r['start_omega_rad']
                             for r in records], dtype=np.float64)
    final_params = np.array([r['final_q0_wxyz'] + r['final_omega_rad']
                             for r in records], dtype=np.float64)
    final_costs = np.array([r['final_cost'] for r in records], dtype=np.float64)
    final_grads = np.array([r['final_grad'] for r in records], dtype=np.float64)
    n_iters = np.array([r['n_iter'] for r in records], dtype=np.int32)
    n_fevs = np.array([r['n_fev'] for r in records], dtype=np.int32)
    successes = np.array([r['success'] for r in records], dtype=bool)
    start_labels = np.array([r['start_label'] for r in records], dtype='S32')
    is_twin_arr = np.array([r['is_twin_start'] for r in records], dtype=bool)

    np.savez_compressed(
        out_npz,
        start_params=start_params,
        final_params=final_params,
        final_costs=final_costs,
        final_grads=final_grads,
        n_iters=n_iters,
        n_fevs=n_fevs,
        successes=successes,
        start_labels=start_labels,
        is_twin=is_twin_arr,
        n_starts=np.array(n_starts),
        n_basins=np.array(n_basins),
        records_json=np.array(json.dumps(records, default=_json_default)),
    )
    print(f"  saved {out_npz}")
    print(f"  Stage B done in {time.time()-t0:.1f}s")
    return {'records': records, 'n_basins': n_basins, 'n_starts': n_starts}


# ---- Stage C: clustering + per-seed summary ------------------------------
def cluster_attractors(records):
    """
    Cluster finals by mutual (q0_geodesic, w_dir_deg, |w_mag_pct diff|).
    Two finals are same attractor iff ALL three mutual distances are under
    their thresholds.
    Returns: list of clusters; each cluster is a dict with 'members' (list of
    record indices), 'representative' (first index).
    """
    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    finals_q0 = [np.asarray(r['final_q0_wxyz'], dtype=np.float64) for r in records]
    finals_wmag = [float(np.linalg.norm(np.asarray(r['final_omega_rad']))) for r in records]
    finals_whut = []
    for r in records:
        w = np.asarray(r['final_omega_rad'], dtype=np.float64)
        finals_whut.append(w / np.linalg.norm(w))

    for i in range(n):
        for j in range(i + 1, n):
            q_d = quat_geodesic_deg(finals_q0[i], finals_q0[j])
            if q_d > CLUSTER_Q0_DEG:
                continue
            wd = angle_between_unit_vecs_deg(finals_whut[i], finals_whut[j])
            if wd > CLUSTER_WDIR_DEG:
                continue
            wmag_diff_pct = abs((finals_wmag[i] / finals_wmag[j] - 1.0) * 100.0)
            if wmag_diff_pct > CLUSTER_WMAG_PCT:
                continue
            union(i, j)

    clusters_map = {}
    for i in range(n):
        root = find(i)
        clusters_map.setdefault(root, []).append(i)

    clusters = []
    for root in sorted(clusters_map.keys()):
        members = sorted(clusters_map[root])
        clusters.append({
            'representative_idx': members[0],
            'members_idx': members,
            'members_labels': [records[k]['start_label'] for k in members],
            'n_members': len(members),
            'rep_final_cost': records[members[0]]['final_cost'],
            'rep_final_q0_err_deg': records[members[0]]['final_q0_err_deg'],
            'rep_final_w_dir_err_deg': records[members[0]]['final_w_dir_err_deg'],
            'rep_final_w_mag_err_pct': records[members[0]]['final_w_mag_err_pct'],
        })
    return clusters


def stage_C_summary(seed, records, n_basins, out_dir, wall_start):
    print(f"\n[Stage C] Summary for seed {seed}")
    t0 = time.time()

    n_starts = len(records)
    clusters = cluster_attractors(records)
    n_attractors = len(clusters)

    # Hypothesis 1: truth start
    truth_rec = next((r for r in records if r['start_label'] == 'truth'), None)
    if truth_rec is None:
        hyp1 = 'INCONCLUSIVE'
    else:
        q0_ok = truth_rec['final_q0_err_deg'] < 1.0
        wdir_ok = truth_rec['final_w_dir_err_deg'] < 0.01
        wmag_ok = abs(truth_rec['final_w_mag_err_pct']) < 0.01
        hyp1 = 'CONFIRMED' if (q0_ok and wdir_ok and wmag_ok) else 'REFUTED'

    # Hypothesis 2: per-basin-start (non-truth) — did GD stay put?
    non_truth = [r for r in records if r['start_label'] != 'truth']
    hyp2_flags = []
    for r in non_truth:
        stayed = (r['q0_move_deg'] < HYP2_Q0_MOVE_DEG and
                  r['w_dir_move_deg'] < HYP2_WDIR_MOVE_DEG)
        hyp2_flags.append(stayed)
    hyp2_confirmed = int(sum(hyp2_flags))
    hyp2_total = len(hyp2_flags)
    hyp2_frac = (hyp2_confirmed / hyp2_total) if hyp2_total > 0 else None
    if hyp2_total == 0:
        hyp2 = 'INCONCLUSIVE'
    elif hyp2_frac > 0.8:
        hyp2 = 'CONFIRMED'
    elif hyp2_frac < 0.2:
        hyp2 = 'REFUTED'
    else:
        hyp2 = 'MIXED'

    # Hypothesis 3: n_attractors <= n_basins (upper-bound)
    if n_basins == 0:
        hyp3 = 'INCONCLUSIVE'
    elif n_attractors <= n_basins:
        hyp3 = 'CONFIRMED'
    else:
        hyp3 = 'REFUTED'

    # Console headline per spec
    print(f"\n  SEED {seed}  n_starts={n_starts}  n_attractors={n_attractors}")
    if truth_rec is not None:
        print(f"    truth_start: final_q0_err={truth_rec['final_q0_err_deg']:.3f}°  "
              f"final_w_dir_err={truth_rec['final_w_dir_err_deg']:.4f}°  "
              f"final_w_mag_err={truth_rec['final_w_mag_err_pct']:.3f}%  "
              f"(iter={truth_rec['n_iter']}, fev={truth_rec['n_fev']})")
    for r, stayed in zip(non_truth, hyp2_flags):
        tag = 'SAME_ATTRACTOR' if stayed else 'MOVED'
        print(f"    {r['start_label']} start: move q0={r['q0_move_deg']:.3f}°  "
              f"w_dir={r['w_dir_move_deg']:.5f}°    hyp2: {tag}")
    print(f"    hyp1: {hyp1}   "
          f"hyp2: {hyp2_confirmed}/{hyp2_total} confirmed   "
          f"hyp3: n_attractors={n_attractors} vs n_basins={n_basins} -> {hyp3}")

    wall_seconds = float(time.time() - wall_start)

    summary = {
        'seed': int(seed),
        'n_starts': int(n_starts),
        'n_basins_micro115': int(n_basins),
        'n_attractors': int(n_attractors),
        'records': records,
        'clusters': clusters,
        'hyp1_verdict': hyp1,
        'hyp2_verdict': hyp2,
        'hyp2_confirmed': hyp2_confirmed,
        'hyp2_total': hyp2_total,
        'hyp2_fraction': hyp2_frac,
        'hyp3_verdict': hyp3,
        'cluster_thresholds': {
            'q0_deg': CLUSTER_Q0_DEG,
            'w_dir_deg': CLUSTER_WDIR_DEG,
            'w_mag_pct': CLUSTER_WMAG_PCT,
        },
        'hyp2_thresholds': {
            'q0_move_deg': HYP2_Q0_MOVE_DEG,
            'w_dir_move_deg': HYP2_WDIR_MOVE_DEG,
        },
        'lbfgs_options': LBFGS_OPTIONS,
        'wall_seconds': wall_seconds,
    }
    atomic_json_save(out_dir / "summary.json", summary)
    print(f"  saved {out_dir / 'summary.json'}")
    print(f"  Stage C done in {time.time()-t0:.1f}s")
    return summary


# ---- Per-seed driver -----------------------------------------------------
def run_seed(seed):
    out_dir = OUT_BASE / f"seed_{seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    log_f = open(log_path, 'w')
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    t_seed = time.time()
    try:
        print("=" * 70)
        print(f"m123 -- L-BFGS polish from truth + m115 DE basins, seed {seed}")
        print("=" * 70)
        print(f"  out_dir={out_dir}  FORCE={FORCE}")
        print(f"  lbfgs options={LBFGS_OPTIONS}")

        setup = stage_A_setup(seed, out_dir)
        polish = stage_B_polish(seed, setup, out_dir)
        summary = stage_C_summary(
            seed, polish['records'], polish['n_basins'], out_dir, t_seed)

        dt = time.time() - t_seed
        print(f"\n  Total for seed {seed}: {dt:.1f}s")
        return summary
    finally:
        sys.stdout = orig_stdout
        log_f.close()


# ---- Stage D: run-level aggregation --------------------------------------
def aggregate_runlevel(per_seed_summaries):
    print(f"\n{'=' * 72}")
    print(f"Run-level aggregation")
    print(f"{'=' * 72}")

    # Hyp1: ALL seeds must confirm
    hyp1_per_seed = {s['seed']: s['hyp1_verdict'] for s in per_seed_summaries}
    if all(v == 'CONFIRMED' for v in hyp1_per_seed.values()):
        hyp1_overall = 'CONFIRMED'
        hyp1_reason = 'all seeds confirmed truth-start stays within thresholds'
    elif any(v == 'REFUTED' for v in hyp1_per_seed.values()):
        hyp1_overall = 'REFUTED'
        violators = [k for k, v in hyp1_per_seed.items() if v == 'REFUTED']
        hyp1_reason = f'violators: {violators}'
    else:
        hyp1_overall = 'INCONCLUSIVE'
        hyp1_reason = f'per-seed: {hyp1_per_seed}'

    # Hyp2: aggregate over all non-truth starts
    tot_conf = sum(s['hyp2_confirmed'] for s in per_seed_summaries)
    tot_n = sum(s['hyp2_total'] for s in per_seed_summaries)
    agg_frac = (tot_conf / tot_n) if tot_n > 0 else None
    if tot_n == 0:
        hyp2_overall = 'INCONCLUSIVE'
        hyp2_reason = 'no non-truth starts'
    elif agg_frac > 0.8:
        hyp2_overall = 'CONFIRMED'
        hyp2_reason = f'aggregate {tot_conf}/{tot_n} = {agg_frac:.2f} > 0.8'
    elif agg_frac < 0.2:
        hyp2_overall = 'REFUTED'
        hyp2_reason = f'aggregate {tot_conf}/{tot_n} = {agg_frac:.2f} < 0.2'
    else:
        hyp2_overall = 'MIXED'
        hyp2_reason = f'aggregate {tot_conf}/{tot_n} = {agg_frac:.2f} in [0.2,0.8]'

    # Hyp3: ALL seeds must have n_attractors <= n_basins
    hyp3_per_seed = {s['seed']: (s['n_attractors'], s['n_basins_micro115'])
                     for s in per_seed_summaries}
    if all(s['hyp3_verdict'] == 'CONFIRMED' for s in per_seed_summaries):
        hyp3_overall = 'CONFIRMED'
        hyp3_reason = f'all seeds: {hyp3_per_seed}'
    elif any(s['hyp3_verdict'] == 'REFUTED' for s in per_seed_summaries):
        hyp3_overall = 'REFUTED'
        violators = {k: v for k, v in hyp3_per_seed.items()
                     if v[0] > v[1]}
        hyp3_reason = f'violators (n_attractors>n_basins): {violators}'
    else:
        hyp3_overall = 'INCONCLUSIVE'
        hyp3_reason = f'per-seed: {hyp3_per_seed}'

    runlevel = {
        'seeds_run': [s['seed'] for s in per_seed_summaries],
        'per_seed_hyp1': hyp1_per_seed,
        'per_seed_hyp2_fraction': {
            s['seed']: s['hyp2_fraction'] for s in per_seed_summaries},
        'per_seed_n_attractors_vs_n_basins': {
            s['seed']: {'n_attractors': s['n_attractors'],
                        'n_basins_micro115': s['n_basins_micro115']}
            for s in per_seed_summaries},
        'per_seed_truth_final_errors': {
            s['seed']: {
                'q0_err_deg': next((r['final_q0_err_deg']
                                    for r in s['records']
                                    if r['start_label'] == 'truth'), None),
                'w_dir_err_deg': next((r['final_w_dir_err_deg']
                                       for r in s['records']
                                       if r['start_label'] == 'truth'), None),
                'w_mag_err_pct': next((r['final_w_mag_err_pct']
                                       for r in s['records']
                                       if r['start_label'] == 'truth'), None),
            } for s in per_seed_summaries},
        'hyp2_aggregate_confirmed': int(tot_conf),
        'hyp2_aggregate_total': int(tot_n),
        'hyp2_aggregate_fraction': agg_frac,
        'verdicts': {
            'hyp1': {'verdict': hyp1_overall, 'reason': hyp1_reason},
            'hyp2': {'verdict': hyp2_overall, 'reason': hyp2_reason},
            'hyp3': {'verdict': hyp3_overall, 'reason': hyp3_reason},
        },
        'cluster_thresholds': {
            'q0_deg': CLUSTER_Q0_DEG,
            'w_dir_deg': CLUSTER_WDIR_DEG,
            'w_mag_pct': CLUSTER_WMAG_PCT,
        },
        'hyp2_thresholds': {
            'q0_move_deg': HYP2_Q0_MOVE_DEG,
            'w_dir_move_deg': HYP2_WDIR_MOVE_DEG,
        },
        'lbfgs_options': LBFGS_OPTIONS,
    }
    atomic_json_save(OUT_BASE / "summary.json", runlevel)
    print(f"\nRun-level summary: {OUT_BASE / 'summary.json'}")
    print(f"  hyp1: {hyp1_overall} -- {hyp1_reason}")
    print(f"  hyp2: {hyp2_overall} -- {hyp2_reason}")
    print(f"  hyp3: {hyp3_overall} -- {hyp3_reason}")
    return runlevel


# ---- Main ----------------------------------------------------------------
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"m123 -- L-BFGS polish; seeds={SEEDS}  FORCE={FORCE}")

    per_seed = []
    t_all = time.time()
    for seed in SEEDS:
        s = run_seed(seed)
        per_seed.append(s)

    print(f"\n{'=' * 72}")
    print(f"All seeds done in {time.time() - t_all:.1f}s")
    print(f"{'=' * 72}")

    aggregate_runlevel(per_seed)


if __name__ == '__main__':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
