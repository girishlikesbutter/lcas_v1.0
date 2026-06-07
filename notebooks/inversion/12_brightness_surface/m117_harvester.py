#!/usr/bin/env python3
"""
m117 — Grid + NM harvester (skip geo / phi-sweep / hi-fi).

Produces a geo_ckpt-schema NPZ pool of 26 omega candidates for one seed,
matching the schema downstream `inline_omega_selection_test.py` consumes
(`w0_refs`, `geo_costs`). Geo refinement is skipped on purpose — `geo_costs`
is repurposed to hold NM cost (used purely as a ranking proxy downstream).

Reuses the grid + NM helpers from m103_hybrid.py exactly:
    propagate_delta_qs, vectorized_phi_cost_excl, anchor_q_from_phi,
    fibonacci_sphere, get_allowed_normals, omega_dir_err
The same step-1 anchor selection logic from m103 is replicated inline
because m103 runs it at module top-level (no helper to import).

Usage:
  MICRO117_SEED=14 python3 m117_harvester.py
  MICRO117_POOL_SIZE=3 MICRO117_SEED=14 python3 m117_harvester.py
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys, time, json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment  # noqa: F401  (parity w/ inline test)
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

# ---------- Parameters (env-overridable) -----------------------------------
TRAJ_SEED = int(os.environ['MICRO117_SEED'])  # required
POOL_SIZE = int(os.environ.get('MICRO117_POOL_SIZE', '24'))
OUTPUT_DIR = Path(os.environ.get(
    'MICRO117_OUTPUT_DIR',
    str(RESULTS_DIR / "harvester")))

# Grid + NM parameters — match m103_hybrid exactly
N_DIRS = 2000
N_MAGS = 20  # >= 20 enforced by project rule
N_PHI_COARSE = 36
N_PHI_FINE = 360
NM_TOP = 300
GEO_TOP = 26  # final pool size (was 20 in m103; 26 to match geo_ckpt shape post multi-phi)
PEAK_WINDOW = 3
CONSTRAINT_WEIGHT = 10.0
Z_NORMALS = {4, 5}
SCRIPT_PATH = str(Path(__file__).resolve())


# ---------- Helpers (copied identically from m103_hybrid) --------------
def get_allowed_normals(mag):
    if mag < 5.9: return [0, 1]
    elif mag < 6.3: return [0, 1, 4, 5]
    elif mag < 7.3: return [0, 1, 2, 3, 4, 5]
    else: return list(range(10))


def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(phi)])


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def omega_dir_err(w1, w2):
    d1 = w1 / np.linalg.norm(w1); d2 = w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


class Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files: f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def atomic_json_dump(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


# ---------- Module-level globals filled in main() (Pool fork inherits) -----
_I_TENSOR = None
_DT_CONSTRAINTS = None
_PAB_AT_CONSTR = None
_CONSTRAINT_ALLOWED = None
_UNIQUE_NORMALS = None
_OMEGA_DIRS = None
_OMEGA_MAGS = None
_QA_ANCHOR_SETS = None
_FINE_PHI_CACHE = None


def propagate_delta_qs(omega_vec, dt_arr):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6; bwd = dt_arr < -1e-6; zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec,
            np.concatenate([[0.0], fwd_dt]), "tumbling", _I_TENSOR)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec,
            np.concatenate([[0.0], bwd_dt]), "tumbling", _I_TENSOR)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


def vectorized_phi_cost_excl(q_anchors_xyzw, delta_qs, pab_arr,
                              allowed_per_constraint, normals, w):
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        allowed = allowed_per_constraint[ci]
        bds = (pbs @ normals[allowed].T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


def eval_one_direction(wi):
    wd = _OMEGA_DIRS[wi]
    best_cost, best_omega, best_ni, best_phi_idx = np.inf, None, -1, -1
    for mag in _OMEGA_MAGS:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, _DT_CONSTRAINTS)
        for ni, qa_xyzw in _QA_ANCHOR_SETS:
            c = vectorized_phi_cost_excl(qa_xyzw, dqs, _PAB_AT_CONSTR,
                _CONSTRAINT_ALLOWED, _UNIQUE_NORMALS, CONSTRAINT_WEIGHT)
            bi = int(np.argmin(c))
            if c[bi] < best_cost:
                best_cost = c[bi]; best_omega = omega_test.copy()
                best_ni = ni; best_phi_idx = bi
    return best_cost, best_omega, best_ni, best_phi_idx


def refine_one_nm(args):
    idx, omega_start, fixed_ni = args
    qa_wxyz, qa_xyzw, phi_arr = _FINE_PHI_CACHE[fixed_ni]

    def glint_cost(omega_vec):
        dqs = propagate_delta_qs(omega_vec, _DT_CONSTRAINTS)
        return vectorized_phi_cost_excl(qa_xyzw, dqs, _PAB_AT_CONSTR,
            _CONSTRAINT_ALLOWED, _UNIQUE_NORMALS, CONSTRAINT_WEIGHT).min()

    res = minimize(glint_cost, omega_start, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
    dqs = propagate_delta_qs(res.x, _DT_CONSTRAINTS)
    c = vectorized_phi_cost_excl(qa_xyzw, dqs, _PAB_AT_CONSTR,
        _CONSTRAINT_ALLOWED, _UNIQUE_NORMALS, CONSTRAINT_WEIGHT)
    return idx, float(res.fun), res.x, int(np.argmin(c)), fixed_ni


def main():
    global _I_TENSOR, _DT_CONSTRAINTS, _PAB_AT_CONSTR, _CONSTRAINT_ALLOWED
    global _UNIQUE_NORMALS, _OMEGA_DIRS, _OMEGA_MAGS, _QA_ANCHOR_SETS
    global _FINE_PHI_CACHE

    seed_dir = OUTPUT_DIR / f"seed_{TRAJ_SEED:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(str(seed_dir / "run.log"), "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("=" * 60, flush=True)
    print(f"m117 — grid+NM harvester (seed {TRAJ_SEED})")
    print(f"  N_DIRS={N_DIRS}, N_MAGS={N_MAGS}, NM_TOP={NM_TOP}, GEO_TOP={GEO_TOP}")
    print(f"  POOL_SIZE={POOL_SIZE}, output={seed_dir}")
    print("=" * 60)
    t_global = time.time()

    # ---- Skeleton checkpoint paths defined up-front ----------------------
    grid_ckpt_path = seed_dir / "grid_ckpt.npz"
    nm_ckpt_path = seed_dir / "geo_ckpt.npz"  # geo-schema, NM-cost contents
    result_json_path = seed_dir / "result.json"

    # ---- Load master trajectory data ------------------------------------
    master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
    obs_times = master['observation_times']
    pab_j2000 = master['pab_j2000']
    unique_normals = master['unique_normals']
    I_tensor = master['inertia_tensor']
    true_lc = master['mag_hifi'][TRAJ_SEED]
    _I_TENSOR = I_tensor
    _UNIQUE_NORMALS = unique_normals

    rng = np.random.default_rng(42)
    observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

    # ---- Step 1: anchor + constraints (replicated from m103) --------
    peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
    omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
    omega_est_rad = np.deg2rad(omega_est_dps)
    spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

    smoothed_lc = savgol_filter(observed_lc, window_length=7, polyorder=3)
    smooth_mags = smoothed_lc[spec_peaks]
    sr = np.argsort(smooth_mags)
    if len(sr) >= 2 and abs(smooth_mags[sr[0]] - smooth_mags[sr[1]]) < 0.05:
        anchor_rank = sr[:2][np.argmin(spec_peaks[sr[:2]])]
    else:
        anchor_rank = sr[0]
    anchor_idx = int(spec_peaks[anchor_rank])
    anchor_time = obs_times[anchor_idx]
    anchor_mag = observed_lc[anchor_idx]
    anchor_allowed = get_allowed_normals(anchor_mag)

    non_anchor = spec_peaks[spec_peaks != anchor_idx]
    constraint_epochs = non_anchor
    constraint_mags = observed_lc[constraint_epochs]
    _CONSTRAINT_ALLOWED = [get_allowed_normals(m) for m in constraint_mags]
    _DT_CONSTRAINTS = obs_times[constraint_epochs] - anchor_time
    _PAB_AT_CONSTR = pab_j2000[constraint_epochs]

    print(f"\nPeaks: {len(peaks_idx)} total, {len(spec_peaks)} spec | "
          f"|omega| est = {omega_est_dps:.3f} dps")
    print(f"Anchor: ep {anchor_idx}, mag={anchor_mag:.2f}, "
          f"allowed normals={anchor_allowed}")

    # ---- Step 2: Grid ----------------------------------------------------
    t_grid = time.time()
    _OMEGA_DIRS = fibonacci_sphere(N_DIRS)
    _OMEGA_MAGS = omega_est_rad * np.linspace(0.70, 1.30, N_MAGS)
    phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
    phi_coarse_z = np.linspace(0, 2*np.pi, 2*N_PHI_COARSE, endpoint=False)
    qa_anchor_sets = []
    for ni in anchor_allowed:
        phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
        qa = np.array([anchor_q_from_phi(p, unique_normals[ni],
                                          pab_j2000[anchor_idx]) for p in phi_arr])
        qa_anchor_sets.append((ni, qa[:, [1, 2, 3, 0]]))
    _QA_ANCHOR_SETS = qa_anchor_sets

    print(f"\n--- Step 2: Grid ({N_DIRS}x{N_MAGS}) on {POOL_SIZE} workers ---",
          flush=True)
    from multiprocessing import Pool  # imported here so fork sees globals
    with Pool(POOL_SIZE) as pool:
        results = pool.map(eval_one_direction, range(N_DIRS))
    grid_costs = np.array([r[0] for r in results])
    grid_omegas = np.array([r[1] for r in results])
    grid_best_ni = np.array([r[2] for r in results], dtype=int)
    grid_best_phi = np.array([r[3] for r in results], dtype=int)
    grid_time = time.time() - t_grid
    print(f"Grid done in {grid_time:.1f}s")

    # Save grid checkpoint (cheap insurance — top NM_TOP)
    sorted_grid = np.argsort(grid_costs)
    top_grid_idx = sorted_grid[:NM_TOP]
    np.savez(str(grid_ckpt_path),
             grid_idx=top_grid_idx,
             grid_costs=grid_costs[top_grid_idx],
             grid_omegas=grid_omegas[top_grid_idx],
             grid_best_ni=grid_best_ni[top_grid_idx],
             grid_best_phi=grid_best_phi[top_grid_idx],
             omega_dirs=_OMEGA_DIRS,
             omega_mags=_OMEGA_MAGS,
             anchor_idx=anchor_idx,
             anchor_allowed=np.array(anchor_allowed))
    print(f"  Saved: {grid_ckpt_path}")

    # ---- Step 3: NM refinement ------------------------------------------
    t_nm = time.time()
    phi_fine_xy = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
    phi_fine_z = np.linspace(0, 2*np.pi, 2*N_PHI_FINE, endpoint=False)
    nm_pool_ni = set(int(grid_best_ni[i]) for i in top_grid_idx)
    fine_phi_cache = {}
    for ni in nm_pool_ni:
        phi_arr = phi_fine_z if ni in Z_NORMALS else phi_fine_xy
        qa = np.array([anchor_q_from_phi(p, unique_normals[ni],
                                          pab_j2000[anchor_idx]) for p in phi_arr])
        fine_phi_cache[ni] = (qa, qa[:, [1, 2, 3, 0]], phi_arr)
    _FINE_PHI_CACHE = fine_phi_cache

    print(f"\n--- Step 3: NM ({len(top_grid_idx)} starts) on {POOL_SIZE} workers ---",
          flush=True)
    nm_args = [(i, grid_omegas[gi].copy(), int(grid_best_ni[gi]))
               for i, gi in enumerate(top_grid_idx)]
    with Pool(POOL_SIZE) as pool:
        nm_results = pool.map(refine_one_nm, nm_args)

    refined_costs = np.zeros(len(top_grid_idx))
    refined_omegas = np.zeros((len(top_grid_idx), 3))
    for idx, cost, omega, _bpi, _bni in nm_results:
        refined_costs[idx] = cost
        refined_omegas[idx] = omega
    nm_time = time.time() - t_nm
    print(f"NM done in {nm_time:.1f}s")

    # ---- Select top-26 by NM cost (no dedup — pool diversity preserved) -
    nm_sorted = np.argsort(refined_costs)
    n_keep = min(GEO_TOP, len(nm_sorted))
    selected = nm_sorted[:n_keep]

    # ---- Save geo_ckpt (NM-cost contents, geo-schema layout) ------------
    n_candidates = int(n_keep)
    omega_ranks = np.array([int(top_grid_idx[i]) for i in selected], dtype=np.int64)
    phi_ranks = np.zeros(n_candidates, dtype=np.int64)
    q0_refs = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_candidates, 1))
    w0_refs = np.array([refined_omegas[i] for i in selected], dtype=np.float64)
    geo_costs = np.array([refined_costs[i] for i in selected], dtype=np.float64)
    q0_ref_errs = np.zeros(n_candidates, dtype=np.float64)
    w0_ref_errs = np.zeros(n_candidates, dtype=np.float64)

    np.savez(str(nm_ckpt_path),
             n_candidates=np.int64(n_candidates),
             omega_ranks=omega_ranks,
             phi_ranks=phi_ranks,
             q0_refs=q0_refs,
             w0_refs=w0_refs,
             geo_costs=geo_costs,
             q0_ref_errs=q0_ref_errs,
             w0_ref_errs=w0_ref_errs)
    print(f"\nSaved geo_ckpt (NM-cost): {nm_ckpt_path}  ({n_candidates} candidates)")

    # ---- Result JSON ----------------------------------------------------
    total_time = time.time() - t_global
    result = {
        "seed": TRAJ_SEED,
        "pool_size": POOL_SIZE,
        "n_candidates": n_candidates,
        "nm_cost_stats": {
            "min": float(np.min(geo_costs)),
            "median": float(np.median(geo_costs)),
            "max": float(np.max(geo_costs)),
        },
        "timing": {
            "grid": float(grid_time),
            "nm": float(nm_time),
            "total": float(total_time),
        },
        "script_path": SCRIPT_PATH,
    }
    atomic_json_dump(result, result_json_path)
    print(f"Saved result.json: {result_json_path}")

    print(f"\n=== Total: {total_time:.1f}s "
          f"(grid {grid_time:.1f}s, NM {nm_time:.1f}s) ===")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()
