#!/usr/bin/env python3
"""
m127 -- Flipped-omega compensating-q0 search.

Hypothesis: for >=3 of the 11 baseline seeds other than seed 33, a 3-DOF q0
search with omega fixed at -omega_truth finds a (q0', -omega_truth) state
whose hi-fi MSE is < 0.5.
  CONFIRMED  iff >= 3 seeds hit min hi-fi MSE < 0.5 (excluding seed 33).
  REFUTED    iff <= 1 seed hits it.
  MIXED      otherwise.
Seed 33 is positive control.

Pipeline per seed:
  A. SO(3) super-Fibonacci grid scan (N=60000), omega_search=-omega_true,
     mean_L1 residual vs observed_lc.
  B. L-BFGS-B polish of top-20 q0s (3-DOF rotvec), omega fixed.
  C. Cluster polished q0s by geodesic < 5 deg (cap 5 basins).
  D. Hi-fi validate each basin (vendored hifi_validate).
  E. Classify FLIPPED_VALID (<0.1), FLIPPED_PARTIAL (<0.5), FLIPPED_FAIL.

Env: MICRO127_SEEDS, MICRO127_POOL, MICRO127_FORCE.
Out: data/results/inversion_diagnostics/m127_flipped_omega/seed_NNN/
     stage_a_grid.npz, stage_b_polish.npz, stage_c_hifi.npz, result.json.
Batch: batch_summary.json.

Kill criteria (see m127_REPORT.md):
  - Per-seed hard cap: 10 min wall (investigate and abort if hit).
  - Batch hard cap: 60 min total.
  - Stage A pool watch: >3 min on any seed likely indicates worker crash.

Note on omitted errors: w_dir_err and w_mag_err_pct are NOT reported per-basin
because omega_search = -omega_truth exactly by construction, so w_dir_err is
guaranteed 180 deg (signed) / 0 deg (axis-angle) and w_mag_err_pct is
guaranteed 0% (magnitude preserved under negation). See omega_search_is_negated_truth
flag in result.json.
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys, time, json
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

from lib.experiment_setup import setup_experiment, brightness_single_epoch
from src.dynamics.attitude_propagator import propagate_attitude
from surrogate import SurrogateModel

# ---- Config --------------------------------------------------------------
BASELINE_SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93]
POSITIVE_CONTROL_SEED = 33

_env = os.environ.get('MICRO127_SEEDS', '').strip()
SEEDS = [int(s) for s in _env.split(',') if s.strip()] if _env else list(BASELINE_SEEDS)
POOL_SIZE = int(os.environ.get('MICRO127_POOL', '8'))
FORCE = os.environ.get('MICRO127_FORCE', '0') == '1'

N_SO3 = 60000
TOP_K = 20
CLUSTER_Q0_DEG = 5.0
MAX_BASINS = 5
HIFI_POOL_CAP = 4
NOISE_SEED_SHARED = 42
NOISE_SIGMA = 0.05
N_OBS = 500
END_TIME_UTC = '2020-02-05T11:00:00'
PANEL_DEG = 0.0
DISH_DEG = 15.0

SURROGATE_WEIGHTS = Path('/home/girish/surrogate_model/s10_5M_weights.npz')
SURROGATE_NORM = Path('/home/girish/surrogate_model/s10_5M_normalization.npz')
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
MICRO46_NPZ = RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"
OUT_BASE = RESULTS_DIR / "m127_flipped_omega"

LBFGS_OPTIONS = {'ftol': 1e-7, 'gtol': 1e-4, 'maxiter': 200,
                 'maxfun': 1000, 'disp': False}

# ---- Logging / IO --------------------------------------------------------
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files:
            try: f.write(data); f.flush()
            except (ValueError, OSError): pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except (ValueError, OSError): pass

def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (bytes, bytearray)): return o.decode('utf-8', errors='replace')
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
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def rotvec_to_wxyz(r):
    theta = float(np.linalg.norm(r))
    if theta < 1e-14: return np.array([1.0, 0.0, 0.0, 0.0])
    axis = r / theta; ha = theta / 2.0; s = np.sin(ha)
    return np.array([np.cos(ha), s*axis[0], s*axis[1], s*axis[2]])

def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]]) if q.ndim == 1 else q[:, [1,2,3,0]]

def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]]) if q.ndim == 1 else q[:, [3,0,1,2]]

def quat_geodesic_deg(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(2.0 * np.degrees(np.arccos(min(1.0, max(-1.0, float(abs(np.dot(a, b))))))))

def quat_mul_batch(q1, q2):
    """q1:(4,) wxyz scalar, q2:(N,4) wxyz batch -> (N,4) wxyz."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.column_stack([w, x, y, z])

def super_fibonacci_quats(n):
    """Alexa 2022 super-Fibonacci (xyzw, unit, deterministic)."""
    i = np.arange(n, dtype=np.float64); s = i + 0.5; t = s / n
    d = 2.0*np.pi*s; r = np.sqrt(t); R = np.sqrt(1.0 - t)
    alpha = d * np.sqrt(2.0); beta = d * np.sqrt(3.0)
    q = np.stack([r*np.sin(alpha), r*np.cos(alpha),
                  R*np.sin(beta),  R*np.cos(beta)], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q

# ---- Vendored hifi_validate (from m115) ------------------------------
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

# ---- Stage A: SO(3) grid scan --------------------------------------------
_WORKER = {}

def _init_grid_worker(R_matrices, sun_dirs, obs_dirs, obs_dist, observed_lc,
                      weights_path, norm_path):
    _WORKER['R'] = R_matrices
    _WORKER['sun'] = sun_dirs
    _WORKER['obs'] = obs_dirs
    _WORKER['dist'] = obs_dist
    _WORKER['obs_lc'] = observed_lc
    _WORKER['model'] = SurrogateModel(weights_path, norm_path)

def _grid_score_chunk(args):
    idxs, q_delta_wxyz = args
    R_all = _WORKER['R']; sun = _WORKER['sun']; obs = _WORKER['obs']
    dist = _WORKER['dist']; obs_lc = _WORKER['obs_lc']; model = _WORKER['model']
    E = sun.shape[0]
    zeros_E = np.zeros(E); dish_E = np.full(E, DISH_DEG); dist_E = dist.astype(np.float64)
    R_delta = Rotation.from_quat(wxyz_to_xyzw(q_delta_wxyz)).as_matrix()
    out = np.zeros(len(idxs), dtype=np.float32)
    for k, i in enumerate(idxs):
        R_t = np.einsum('tij,jk->tik', R_delta, R_all[i])
        k1 = np.einsum('tij,tj->ti', R_t, sun)
        k2 = np.einsum('tij,tj->ti', R_t, obs)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
        pred = model.predict_magnitude(k1, k2, zeros_E, dish_E, dist_E)
        out[k] = float(np.mean(np.abs(pred.astype(np.float64) - obs_lc)))
    return idxs, out

def stage_a_grid(seed, seed_dir, truth_q0, truth_omega, observed_lc,
                 sun_dirs, obs_dirs, obs_dist, obs_times, I_tensor):
    print(f"\n[Stage A] seed {seed} - SO(3) grid (N={N_SO3})")
    t0 = time.time(); out = seed_dir / "stage_a_grid.npz"
    omega_search = -truth_omega
    if out.exists() and not FORCE:
        print(f"  cached {out.name}")
        return dict(np.load(out, allow_pickle=False)), omega_search
    q_xyzw = super_fibonacci_quats(N_SO3)
    q_wxyz = xyzw_to_wxyz(q_xyzw)
    R_matrices = Rotation.from_quat(q_xyzw).as_matrix().astype(np.float64)
    q_delta_wxyz = precompute_delta_qs(omega_search, obs_times, I_tensor)
    cs = max(1, N_SO3 // (POOL_SIZE * 4))
    chunks = [(np.arange(i, min(i+cs, N_SO3), dtype=np.int64), q_delta_wxyz)
              for i in range(0, N_SO3, cs)]
    print(f"  {len(chunks)} chunks of ~{cs}, POOL={POOL_SIZE}")
    scores = np.zeros(N_SO3, dtype=np.float32)
    init_args = (R_matrices, sun_dirs.astype(np.float64),
                 obs_dirs.astype(np.float64), obs_dist.astype(np.float64),
                 observed_lc.astype(np.float64),
                 str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))
    tp = time.time()
    with mp.Pool(POOL_SIZE, initializer=_init_grid_worker, initargs=init_args) as pool:
        for idxs, s in pool.imap_unordered(_grid_score_chunk, chunks):
            scores[idxs] = s
    print(f"  pool done in {time.time()-tp:.1f}s")
    order = np.argsort(scores, kind='stable')
    topK_idx = order[:TOP_K].astype(np.int64)
    topK_q0 = q_wxyz[topK_idx].astype(np.float64)
    topK_scores = scores[topK_idx].astype(np.float32)
    print(f"  best={scores[order[0]]:.6f}, top{TOP_K}_worst={topK_scores[-1]:.6f}")
    np.savez(out, scores=scores, topK_idx=topK_idx, topK_q0_wxyz=topK_q0,
             topK_scores=topK_scores, omega_search=omega_search.astype(np.float64),
             truth_omega=truth_omega.astype(np.float64), truth_q0=truth_q0.astype(np.float64))
    print(f"  saved {out}  ({time.time()-t0:.1f}s)")
    return dict(np.load(out, allow_pickle=False)), omega_search

# ---- Stage B: L-BFGS-B polish (3-DOF, omega fixed) ----------------------
def make_polish_cost(q0_init, omega_fixed, obs_times, I_tensor,
                     sun_dirs, obs_dirs, obs_dist, observed_lc, model):
    delta_qs = precompute_delta_qs(omega_fixed, obs_times, I_tensor)
    E = len(obs_times)
    zeros_E = np.zeros(E); dish_E = np.full(E, DISH_DEG)
    dist_E = obs_dist.astype(np.float64); obs_lc = observed_lc.astype(np.float64)
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
        if np.sum(valid) < 10: return 10.0
        return float(np.mean(np.abs(pred[valid] - obs_lc[valid])))

    def final_q0(r):
        q0 = quat_mul_wxyz(rotvec_to_wxyz(np.asarray(r, dtype=np.float64)), q0_init)
        return q0 / np.linalg.norm(q0)

    return cost, final_q0

def stage_b_polish(seed, seed_dir, grid_data, omega_search, truth_q0,
                   observed_lc, sun_dirs, obs_dirs, obs_dist, obs_times, I_tensor):
    print(f"\n[Stage B] seed {seed} - L-BFGS-B polish (top-{TOP_K})")
    t0 = time.time(); out = seed_dir / "stage_b_polish.npz"
    if out.exists() and not FORCE:
        print(f"  cached {out.name}")
        return dict(np.load(out, allow_pickle=True))
    topK_q0 = grid_data['topK_q0_wxyz']; topK_scores = grid_data['topK_scores']
    model = SurrogateModel(str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))
    polished = []
    for k in range(topK_q0.shape[0]):
        q0_init = topK_q0[k].astype(np.float64)
        q0_init /= np.linalg.norm(q0_init)
        cost_fn, final_q0_fn = make_polish_cost(
            q0_init, omega_search, obs_times, I_tensor,
            sun_dirs, obs_dirs, obs_dist, observed_lc, model)
        ts = time.time(); x0 = np.zeros(3); start_cost = cost_fn(x0)
        res = minimize(cost_fn, x0=x0, jac=None, method='L-BFGS-B',
                       options=LBFGS_OPTIONS)
        wall = time.time() - ts
        q0_final = final_q0_fn(res.x)
        polished.append({
            'start_idx': int(k), 'grid_score': float(topK_scores[k]),
            'q0_init_wxyz': q0_init.tolist(),
            'q0_polished_wxyz': q0_final.tolist(),
            'start_cost': float(start_cost), 'final_cost': float(res.fun),
            'n_iter': int(getattr(res, 'nit', -1)),
            'n_fev': int(getattr(res, 'nfev', -1)),
            'success': bool(res.success), 'wall_s': float(wall),
        })
        print(f"  [{k+1}/{topK_q0.shape[0]}] start={start_cost:.5f} "
              f"final={res.fun:.5f} iter={polished[-1]['n_iter']} "
              f"fev={polished[-1]['n_fev']} wall={wall:.1f}s")
    polished_sorted = sorted(polished, key=lambda r: r['final_cost'])
    basins = []
    for rec in polished_sorted:
        q = np.asarray(rec['q0_polished_wxyz'], dtype=np.float64)
        merged = False
        for b in basins:
            if quat_geodesic_deg(q, np.asarray(b['q0_polished_wxyz'])) < CLUSTER_Q0_DEG:
                b['n_members'] = b.get('n_members', 1) + 1
                merged = True; break
        if not merged:
            new_b = dict(rec); new_b['n_members'] = 1
            basins.append(new_b)
        if len(basins) >= MAX_BASINS: break
    print(f"  -> {len(basins)} basins (cap={MAX_BASINS})")
    for i, b in enumerate(basins):
        q0_err = quat_geodesic_deg(np.asarray(b['q0_polished_wxyz']), truth_q0)
        print(f"    basin {i}: surr={b['final_cost']:.5f} "
              f"q0_err={q0_err:.2f}deg (n={b['n_members']})")
    basins_q0 = np.array([b['q0_polished_wxyz'] for b in basins], dtype=np.float64)
    np.savez(out,
             basins_q0_wxyz=basins_q0,
             basins_surr_mse=np.array([b['final_cost'] for b in basins], dtype=np.float64),
             basins_wall_s=np.array([b['wall_s'] for b in basins], dtype=np.float64),
             basins_n_iter=np.array([b['n_iter'] for b in basins], dtype=np.int32),
             basins_n_fev=np.array([b['n_fev'] for b in basins], dtype=np.int32),
             polished_json=np.array(json.dumps(polished, default=_json_default)),
             basins_json=np.array(json.dumps(basins, default=_json_default)),
             n_basins=np.array(len(basins), dtype=np.int64))
    print(f"  saved {out}  ({time.time()-t0:.1f}s)")
    return dict(np.load(out, allow_pickle=True))

# ---- Stage C: hi-fi validation -------------------------------------------
_HIFI_STATE = {}

def _hifi_init_worker(ctx, obs_times, I_tensor, observed_lc, omega_search):
    _HIFI_STATE.update(dict(ctx=ctx, obs_times=obs_times, I_tensor=I_tensor,
                            observed_lc=observed_lc, omega_search=omega_search))

def _hifi_worker(job):
    idx, q0 = job
    t0 = time.time()
    mse, mags = hifi_validate(
        np.asarray(q0, dtype=np.float64),
        _HIFI_STATE['omega_search'], _HIFI_STATE['obs_times'],
        _HIFI_STATE['I_tensor'], _HIFI_STATE['observed_lc'], _HIFI_STATE['ctx'])
    return {'basin_idx': int(idx), 'hifi_mse': float(mse),
            'hifi_mags': np.asarray(mags, dtype=np.float64),
            'wall_s': float(time.time() - t0)}

def stage_c_hifi(seed, seed_dir, polish_data, omega_search, ctx,
                 observed_lc, obs_times, I_tensor):
    print(f"\n[Stage C] seed {seed} - hi-fi validation")
    t0 = time.time(); out = seed_dir / "stage_c_hifi.npz"
    if out.exists() and not FORCE:
        print(f"  cached {out.name}")
        return dict(np.load(out, allow_pickle=True))
    basins_q0 = polish_data['basins_q0_wxyz']
    n_basins = basins_q0.shape[0]
    jobs = [(i, basins_q0[i]) for i in range(n_basins)]
    pool_size = max(1, min(n_basins, HIFI_POOL_CAP))
    print(f"  {n_basins} basins, Pool({pool_size})")
    init_args = (ctx, obs_times, I_tensor, observed_lc, omega_search)
    results = {}
    with mp.Pool(pool_size, initializer=_hifi_init_worker, initargs=init_args) as pool:
        for r in pool.imap_unordered(_hifi_worker, jobs):
            results[r['basin_idx']] = r
            print(f"  basin {r['basin_idx']}: hifi={r['hifi_mse']:.6f} "
                  f"wall={r['wall_s']:.1f}s")
    hifi_mse = np.array([results[i]['hifi_mse'] for i in range(n_basins)])
    hifi_mags = np.stack([results[i]['hifi_mags'] for i in range(n_basins)])
    hifi_wall = np.array([results[i]['wall_s'] for i in range(n_basins)])
    np.savez(out, hifi_mse=hifi_mse, hifi_mags=hifi_mags, hifi_wall_s=hifi_wall,
             basins_q0_wxyz=basins_q0.astype(np.float64))
    print(f"  saved {out}  ({time.time()-t0:.1f}s)")
    return dict(np.load(out, allow_pickle=True))

# ---- Classification ------------------------------------------------------
def classify_flipped(min_hifi_mse):
    if min_hifi_mse < 0.1: return 'FLIPPED_VALID'
    if min_hifi_mse < 0.5: return 'FLIPPED_PARTIAL'
    return 'FLIPPED_FAIL'

# ---- Per-seed driver -----------------------------------------------------
def run_seed(seed, ctx, sun_dirs, obs_dirs, obs_dist, obs_times, I_tensor,
             m46_q0s, m46_omega0s, m46_mag_hifi):
    seed_dir = OUT_BASE / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(seed_dir / "run.log", "w")
    old_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_file)
    t_seed = time.time(); timings = {}
    try:
        truth_q0 = m46_q0s[seed].astype(np.float64)
        truth_q0 /= np.linalg.norm(truth_q0)
        truth_omega = m46_omega0s[seed].astype(np.float64)
        rng = np.random.default_rng(int(seed))
        observed_lc = (m46_mag_hifi[seed].astype(np.float64)
                       + rng.normal(0, NOISE_SIGMA, N_OBS))
        print(f"\n===== Seed {seed} =====")
        print(f"  truth_q0={truth_q0}  truth_omega={truth_omega} "
              f"(|w|={np.linalg.norm(truth_omega):.5f})")
        print(f"  omega_search = -truth (flipped)")

        ts = time.time()
        grid_data, omega_search = stage_a_grid(
            seed, seed_dir, truth_q0, truth_omega, observed_lc,
            sun_dirs, obs_dirs, obs_dist, obs_times, I_tensor)
        timings['stage_a_s'] = time.time() - ts
        ts = time.time()
        polish_data = stage_b_polish(
            seed, seed_dir, grid_data, omega_search, truth_q0, observed_lc,
            sun_dirs, obs_dirs, obs_dist, obs_times, I_tensor)
        timings['stage_b_s'] = time.time() - ts
        ts = time.time()
        hifi_data = stage_c_hifi(seed, seed_dir, polish_data, omega_search, ctx,
                                 observed_lc, obs_times, I_tensor)
        timings['stage_c_s'] = time.time() - ts

        basins_q0 = polish_data['basins_q0_wxyz']
        n_basins = int(polish_data['n_basins'])
        hifi_mse = hifi_data['hifi_mse']
        basin_records = [{
            'basin_idx': i, 'q0_wxyz': basins_q0[i].tolist(),
            'surr_mse': float(polish_data['basins_surr_mse'][i]),
            'hifi_mse': float(hifi_mse[i]),
            'q0_err_deg': quat_geodesic_deg(basins_q0[i], truth_q0),
            'polish_n_iter': int(polish_data['basins_n_iter'][i]),
            'polish_n_fev': int(polish_data['basins_n_fev'][i]),
            'polish_wall_s': float(polish_data['basins_wall_s'][i]),
            'hifi_wall_s': float(hifi_data['hifi_wall_s'][i]),
        } for i in range(n_basins)]
        winner = basin_records[int(np.argmin(hifi_mse))]
        classification = classify_flipped(winner['hifi_mse'])
        timings['total_s'] = time.time() - t_seed

        result = {
            'traj_seed': int(seed), 'omega_search_is_negated_truth': True,
            'omega_search': (-truth_omega).tolist(),
            'omega_true': truth_omega.tolist(),
            'truth_q0_wxyz': truth_q0.tolist(),
            'stage_a_best_surr': float(grid_data['topK_scores'][0]),
            'stage_a_best_q0_idx': int(grid_data['topK_idx'][0]),
            'stage_b_n_basins': n_basins, 'n_basins_found': n_basins,
            'basins': basin_records,
            'winner': {k: winner[k] for k in
                       ('basin_idx', 'hifi_mse', 'surr_mse', 'q0_err_deg')},
            'classification': classification,
            'timing': {k: round(v, 2) for k, v in timings.items()},
        }
        atomic_json_save(seed_dir / "result.json", result)
        print(f"\n  classification: {classification}")
        print(f"  winner basin {winner['basin_idx']}: hifi={winner['hifi_mse']:.5f} "
              f"q0_err={winner['q0_err_deg']:.2f}deg  total={timings['total_s']:.1f}s")
        return result
    except Exception as e:
        import traceback
        print(f"  ERROR seed {seed}: {e}"); traceback.print_exc()
        result = {'traj_seed': int(seed), 'error': str(e),
                  'classification': 'ERROR',
                  'timing': {'total_s': time.time() - t_seed}}
        atomic_json_save(seed_dir / "result.json", result)
        return result
    finally:
        sys.stdout = old_stdout
        log_file.close()

# ---- Main ----------------------------------------------------------------
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"m127 -- flipped-omega compensating-q0 search")
    print(f"  SEEDS={SEEDS}  POOL={POOL_SIZE}  FORCE={FORCE}")
    print(f"  N_SO3={N_SO3} TOP_K={TOP_K} CLUSTER_Q0={CLUSTER_Q0_DEG} "
          f"MAX_BASINS={MAX_BASINS}")

    t_all = time.time()
    print("\n[Setup] m046 + setup_experiment(skip_true_lc=True)")
    m46 = np.load(MICRO46_NPZ)
    m46_q0s = m46['q0s'].astype(np.float64)
    m46_omega0s = m46['omega0s'].astype(np.float64)
    m46_mag_hifi = m46['mag_hifi'].astype(np.float64)

    # Shared geometry ctx (skip_true_lc=True -> true_q0/omega unused).
    ctx = setup_experiment(
        n_observations=N_OBS, noise_sigma=NOISE_SIGMA,
        random_seed=NOISE_SEED_SHARED, end_time_utc=END_TIME_UTC,
        skip_true_lc=True,
        true_q0_wxyz=m46_q0s[SEEDS[0]].copy(),
        true_omega0_rad=m46_omega0s[SEEDS[0]].copy())
    obs_times = ctx.observation_times.astype(np.float64)
    I_tensor = ctx.inertia_tensor.astype(np.float64)
    sun_dirs = ctx.sun_pos - ctx.sat_pos
    sun_dirs /= np.linalg.norm(sun_dirs, axis=1, keepdims=True)
    obs_dirs = ctx.obs_pos - ctx.sat_pos
    obs_dirs /= np.linalg.norm(obs_dirs, axis=1, keepdims=True)
    obs_dist = ctx.obs_dist.astype(np.float64)
    print(f"  dt={obs_times[1]-obs_times[0]:.3f}s span={obs_times[-1]:.1f}s "
          f"N_obs={len(obs_times)}")

    all_results = []
    for seed in SEEDS:
        ts = time.time()
        r = run_seed(seed, ctx, sun_dirs, obs_dirs, obs_dist, obs_times,
                     I_tensor, m46_q0s, m46_omega0s, m46_mag_hifi)
        print(f"  [seed {seed}] {r.get('classification','?')} in {time.time()-ts:.1f}s")
        all_results.append(r)

    counts = {'FLIPPED_VALID': 0, 'FLIPPED_PARTIAL': 0,
              'FLIPPED_FAIL': 0, 'ERROR': 0}
    rows = []; control = None
    for r in all_results:
        cls = r.get('classification', 'ERROR')
        counts[cls] = counts.get(cls, 0) + 1
        if r['traj_seed'] == POSITIVE_CONTROL_SEED: control = r
        rows.append({
            'seed': r['traj_seed'],
            'stage_a_best_surr': r.get('stage_a_best_surr'),
            'stage_b_best_surr': (min(b['surr_mse'] for b in r['basins'])
                                  if r.get('basins') else None),
            'stage_c_best_hifi': (r.get('winner') or {}).get('hifi_mse'),
            'stage_c_winner_q0_err_deg': (r.get('winner') or {}).get('q0_err_deg'),
            'classification': cls,
            'wall_s': r.get('timing', {}).get('total_s'),
        })
    excl_valid = sum(1 for r in all_results
                     if r['traj_seed'] != POSITIVE_CONTROL_SEED
                     and r.get('classification') in ('FLIPPED_VALID', 'FLIPPED_PARTIAL'))
    verdict = ('CONFIRMED' if excl_valid >= 3
               else 'REFUTED' if excl_valid <= 1 else 'MIXED')

    summary = {
        'seeds': SEEDS, 'rows': rows,
        'counts': {'n_flipped_valid': counts['FLIPPED_VALID'],
                   'n_flipped_partial': counts['FLIPPED_PARTIAL'],
                   'n_flipped_fail': counts['FLIPPED_FAIL'],
                   'n_error': counts['ERROR']},
        'control_seed': POSITIVE_CONTROL_SEED,
        'control_classification': (control or {}).get('classification'),
        'control_winner': (control or {}).get('winner'),
        'excluding_control_valid_or_partial': excl_valid,
        'hypothesis_verdict': verdict,
        'total_wall_s': round(time.time() - t_all, 2),
        'config': {'N_SO3': N_SO3, 'TOP_K': TOP_K, 'MAX_BASINS': MAX_BASINS,
                   'CLUSTER_Q0_DEG': CLUSTER_Q0_DEG, 'POOL_SIZE': POOL_SIZE,
                   'HIFI_POOL_CAP': HIFI_POOL_CAP, 'NOISE_SIGMA': NOISE_SIGMA,
                   'N_OBS': N_OBS, 'END_TIME_UTC': END_TIME_UTC,
                   'LBFGS_OPTIONS': LBFGS_OPTIONS},
    }
    atomic_json_save(OUT_BASE / "batch_summary.json", summary)
    print("\n" + "=" * 72)
    print(f"m127 batch summary: "
          f"VALID={counts['FLIPPED_VALID']} PARTIAL={counts['FLIPPED_PARTIAL']} "
          f"FAIL={counts['FLIPPED_FAIL']} ERROR={counts['ERROR']}")
    print(f"  control (seed {POSITIVE_CONTROL_SEED}): "
          f"{summary['control_classification']}")
    print(f"  excl-control (VALID+PARTIAL): {excl_valid}")
    print(f"  hypothesis verdict: {verdict}  "
          f"total wall: {time.time()-t_all:.1f}s")
    print(f"  saved {OUT_BASE / 'batch_summary.json'}")
    print("=" * 72)


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
