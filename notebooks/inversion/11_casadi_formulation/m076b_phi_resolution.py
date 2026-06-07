#!/usr/bin/env python3
"""
m076b — Grid search with increased phi resolution.

Same as m074c Mode B but with:
  - 100 phi bins in [0, 180) for normals in the x-y plane (±X, ±Y, ±WD, ±ED)
  - 200 phi bins in [0, 360) for ±Z normals
  - 10K dirs, 20 mags

Seed 27 only. Reports truth rank, q0 error, and per-normal breakdown.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_DIR = RESULTS_DIR / "m076b_phi_resolution"
OUT_DIR.mkdir(exist_ok=True)

SEED = 27
N_DIRS = 10000
N_MAGS = 20
N_PHI_XY = 100       # [0, 180) for normals in x-y plane
N_PHI_Z = 200        # [0, 360) for ±Z normals
WEIGHT = 10.0
N_WORKERS = 24


# ── Helpers ────────────────────────────────────────────────────────────

def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def attitude_error_deg(q1, q2):
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return float(np.rad2deg((R1.inv() * R2).magnitude()))


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    q_xyzw = (R_twist * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def propagate_delta_qs(omega_vec, dt_arr, I_tensor):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6
    bwd = dt_arr < -1e-6
    zero = np.abs(dt_arr) < 1e-6
    delta_qs[zero] = q_id
    if np.any(fwd):
        fwd_dt = np.sort(dt_arr[fwd])
        dq, _ = propagate_attitude(q_id, omega_vec,
            np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd] = dq[1:][np.argsort(np.argsort(dt_arr[fwd]))]
    if np.any(bwd):
        bwd_dt = np.sort(-dt_arr[bwd])
        dq, _ = propagate_attitude(q_id, -omega_vec,
            np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd] = dq_c[np.argsort(np.argsort(-dt_arr[bwd]))]
    return delta_qs


def phi_cost_open(q_anchors_xyzw, delta_qs, pab_arr, normals, w):
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)
    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_arr[ci])
        bds = (pbs @ normals.T).max(axis=1)
        costs += w * (1.0 - bds) ** 2
    return costs


# ── Load data ──────────────────────────────────────────────────────────

print("=" * 60)
print(f"m076b — Phi resolution test (seed {SEED})")
print("=" * 60)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][SEED]
true_omega0 = master['omega0s'][SEED]
true_omega_mag_dps = float(master['omega_mags'][SEED])
true_lc = master['mag_hifi'][SEED]
n_normals = len(unique_normals)
group_names = list(master['group_names'])

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)
omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, N_MAGS)

all_spec = peaks_idx[observed_lc[peaks_idx] < 9.0]
anchor_idx = int(all_spec[np.argmin(observed_lc[all_spec])])
anchor_time = obs_times[anchor_idx]
anchor_pab = pab_j2000[anchor_idx]
non_anchor = all_spec[all_spec != anchor_idx]
dt_constraints = obs_times[non_anchor] - anchor_time
pab_at_constraints = pab_j2000[non_anchor]

_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"Anchor: ep {anchor_idx}, {len(non_anchor)} constraints")
print(f"|omega| est: {omega_est_dps:.3f} (true: {true_omega_mag_dps:.3f})")

# ── Pre-compute anchor quaternions ────────────────────────────────────
# ±Z (indices 4, 5): 200 phis in [0, 2*pi)
# All others: 100 phis in [0, pi)

z_normals = {4, 5}  # +Z, -Z

phi_xy = np.linspace(0, np.pi, N_PHI_XY, endpoint=False)
phi_z = np.linspace(0, 2 * np.pi, N_PHI_Z, endpoint=False)

# Build per-normal anchor quaternion arrays and phi arrays
qa_per_normal = []  # list of (qa_xyzw_array, qa_wxyz_array, phi_array, label)
for ni in range(n_normals):
    if ni in z_normals:
        pv = phi_z
    else:
        pv = phi_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], anchor_pab) for p in pv])
    qa_per_normal.append((qa[:, [1, 2, 3, 0]], qa, pv, group_names[ni]))

total_phi_per_dir = sum(len(x[2]) for x in qa_per_normal)
print(f"Phi bins: {N_PHI_XY} (x-y normals) + {N_PHI_Z} (±Z) = {total_phi_per_dir} per normal set")
print(f"Grid: {N_DIRS} dirs x {N_MAGS} mags x {total_phi_per_dir} phi-normal combos")

# ── Grid search ───────────────────────────────────────────────────────

ckpt = OUT_DIR / f"grid_seed{SEED:03d}.npz"
omega_dirs = fibonacci_sphere(N_DIRS)

_omega_dirs = omega_dirs
_omega_mags_s = omega_mags_search
_dt_c = dt_constraints
_pab_c = pab_at_constraints
_qa_per_normal = qa_per_normal
_normals = unique_normals


def _eval_direction(wi):
    wd = _omega_dirs[wi]
    best_cost = np.inf
    best_omega = None
    best_normal = -1
    best_phi_deg = 0.0
    best_qa = None

    for mag in _omega_mags_s:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, _dt_c, I_tensor)

        for ni, (qa_xyzw, qa_wxyz, pv, label) in enumerate(_qa_per_normal):
            c = phi_cost_open(qa_xyzw, dqs, _pab_c, _normals, WEIGHT)
            bi = int(np.argmin(c))
            if c[bi] < best_cost:
                best_cost = c[bi]
                best_omega = omega_test.copy()
                best_normal = ni
                best_phi_deg = float(np.rad2deg(pv[bi]))
                best_qa = qa_wxyz[bi].copy()

    return best_cost, best_omega, best_normal, best_phi_deg, best_qa


if ckpt.exists():
    print(f"\nLoading checkpoint...")
    ck = np.load(str(ckpt), allow_pickle=True)
    grid_costs = ck['costs']
    grid_omegas = ck['omegas']
    grid_normals = ck['normals']
    grid_phis = ck['phis']
    grid_qas = ck['qas']
else:
    print(f"\nRunning grid search ({N_DIRS} dirs, {N_WORKERS} cores)...")
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        results = pool.map(_eval_direction, range(N_DIRS))
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")

    grid_costs = np.array([r[0] for r in results])
    grid_omegas = np.array([r[1] for r in results])
    grid_normals = np.array([r[2] for r in results])
    grid_phis = np.array([r[3] for r in results])
    grid_qas = np.array([r[4] for r in results])

    np.savez(str(ckpt), costs=grid_costs, omegas=grid_omegas,
             normals=grid_normals, phis=grid_phis, qas=grid_qas)
    print(f"Saved: {ckpt}")

# ── Analysis ──────────────────────────────────────────────────────────

sorted_idx = np.argsort(grid_costs)
dir_errors = np.array([omega_dir_err(grid_omegas[i], true_omega_anchor)
                        for i in range(len(grid_costs))])

# Back-propagate top candidates and closest-to-truth to get q0
def get_q0_w0(qa_wxyz, omega_vec):
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(qa_wxyz, -omega_vec, bt, "tumbling", I_tensor)
    return qb[-1], -ob[-1]

closest_idx = int(np.argmin(dir_errors))
truth_rank = int(np.where(sorted_idx == closest_idx)[0][0]) + 1

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Truth rank: #{truth_rank}/{N_DIRS}")
print(f"Closest dir err: {dir_errors.min():.1f}°")

print(f"\nTop 15 by cost:")
print(f"{'Rank':>5s} {'Cost':>10s} {'wErr':>6s} {'Normal':>6s} {'Phi':>6s} {'q0Err':>7s} {'wMag':>6s}")
print("-" * 55)

check_list = list(sorted_idx[:15])
if closest_idx not in check_list:
    check_list.append(closest_idx)

for ri in check_list:
    rank = int(np.where(sorted_idx == ri)[0][0]) + 1
    q0, w0 = get_q0_w0(grid_qas[ri], grid_omegas[ri])
    q0_err = attitude_error_deg(q0, true_q0)
    w_err = dir_errors[ri]
    w_mag = np.rad2deg(np.linalg.norm(grid_omegas[ri]))
    is_truth = " <-- TRUTH" if ri == closest_idx else ""
    is_good = " *" if q0_err < 20 and w_err < 5 else ""
    print(f"  #{rank:4d} {grid_costs[ri]:10.6f} {w_err:5.1f}° "
          f"{group_names[grid_normals[ri]]:>6s} {grid_phis[ri]:5.1f}° "
          f"{q0_err:6.1f}° {w_mag:5.3f}{is_truth}{is_good}")

# Top-N best omega direction error
for topn in [5, 10, 20, 50]:
    best_in_top = dir_errors[sorted_idx[:topn]].min()
    print(f"  Best w_dir in top-{topn:2d}: {best_in_top:.1f}°")

print(f"\nTotal time: {time.time() - t_global:.0f}s")
