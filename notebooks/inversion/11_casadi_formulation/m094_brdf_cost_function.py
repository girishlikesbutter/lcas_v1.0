#!/usr/bin/env python3
"""
m094 — BRDF-Based Cost Function.

Replaces alignment/expected-dot cost in Steps 2 (grid) and 3 (NM) with direct
Ashikhmin-Shirley BRDF brightness prediction. Uses all 3 independent dot products
(n·k1, n·k2, n·h) per epoch instead of just n·h. Eliminates calibration table.

Key differences from m093:
  - NEW: vectorized_phi_cost_brdf replaces alignment cost in grid AND NM
  - REMOVED: calibration step (~86s saved per seed)
  - NEW: k1_j2000, k2_j2000 pre-computed at constraint epochs
  - Steps 4, 4b, 5 unchanged

See MICRO94_BRDF_COST_PLAN.md for full rationale and critical evaluation.

Usage:
  MICRO94_SEED=93 python3 m094_brdf_cost.py
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

TRAJ_SEED = int(os.environ.get('MICRO94_SEED', '93'))

# Pipeline parameters
N_DIRS = 8000         # Fibonacci sphere directions (~1° spacing)
N_MAGS = 20           # magnitude grid points (±20% around estimate)
N_PHI_COARSE = 36     # phi bins for grid search
N_PHI_FINE = 360      # phi bins for NM refinement
NM_TOP = 20           # NM refinement pool size
LOFI_TOP = 200        # candidates to evaluate with lo-fi peak matching
PEAK_WINDOW = 3       # lo-fi peak matches observed if within ±3 epochs
CONSTRAINT_WEIGHT = 10.0  # uniform weight for all constraints
GRID_WORKERS = 24
LOFI_WORKERS = 24
NM_WORKERS = 24
GEO_WORKERS = 24
HIFI_WORKERS = 8

# Magnitude-based normal exclusion bands
Z_NORMALS = {4, 5}

# Cost function strategy thresholds
BRDF_MAG_CUTOFF = 7.0   # bright vs dim boundary for hybrid cost
MIN_BRIGHT_FOR_BRDF = 6  # need ≥6 bright non-anchor constraints for BRDF strategy

# BRDF parameters per normal group.
# ±X dominated by solar panels (87.5 of 97.3 m²), ±Y/±Z are bus faces,
# ±WD/±ED are antenna dishes.
GROUP_RD = np.array([0.026, 0.026, 0.02, 0.02, 0.02, 0.02,
                     0.01, 0.01, 0.01, 0.01])
GROUP_RS = np.array([0.3, 0.3, 0.5, 0.5, 0.5, 0.5,
                     0.4, 0.4, 0.4, 0.4])
GROUP_NPHONG = np.array([250.0, 250.0, 300.0, 300.0, 300.0, 300.0,
                         200.0, 200.0, 200.0, 200.0])
SUN_MAG = -26.74  # solar apparent magnitude constant


def get_allowed_normals(mag):
    """Return list of normal indices allowed for a peak at this magnitude."""
    if mag < 5.9:
        return [0, 1]                          # ±X only
    elif mag < 6.3:
        return [0, 1, 4, 5]                    # ±X, ±Z
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]             # ±X, ±Y, ±Z (bus faces)
    else:
        return list(range(10))                  # all normals


# ── Helpers ─────────────────────────────────────────────────────────────

def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


# ── Logging ────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


CKPT_DIR = RESULTS_DIR / f"m094_seed{TRAJ_SEED:03d}"
CKPT_DIR.mkdir(exist_ok=True)
_log_file = open(str(CKPT_DIR / "pipeline.log"), "w")
sys.stdout = Tee(sys.__stdout__, _log_file)


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print(f"m094 — BRDF cost pipeline (seed {TRAJ_SEED})")
print("=" * 60)
t_global = time.time()

# Load trajectory database
master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][TRAJ_SEED]
true_omega0 = master['omega0s'][TRAJ_SEED]
true_omega_mag_dps = float(master['omega_mags'][TRAJ_SEED])
true_lc = master['mag_hifi'][TRAJ_SEED]
n_normals = len(unique_normals)
group_names = list(master['group_names'])
group_areas = master['group_areas']  # (10,) total area per normal group

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Load satellite model (needed for hi-fi in Step 5, also gives us J2000 positions)
print("  Loading satellite model...", flush=True)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)
_satellite = CTX.satellite

# Pre-compute k1_j2000 and k2_j2000 at ALL epochs (needed for constraint selection)
_sun_j2000 = CTX.sun_pos - CTX.sat_pos  # (500, 3)
_sun_j2000 /= np.linalg.norm(_sun_j2000, axis=1, keepdims=True)
_obs_j2000 = CTX.obs_pos - CTX.sat_pos  # (500, 3)
_obs_j2000 /= np.linalg.norm(_obs_j2000, axis=1, keepdims=True)

print(f"Setup done in {time.time() - t_global:.1f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Peak count → |omega| estimate + constraint selection
# ══════════════════════════════════════════════════════════════════════
t_step1 = time.time()
print(f"\n--- Step 1: Peak count + anchor + constraints ---", flush=True)

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]

if len(spec_peaks) < 2:
    print(f"ERROR: Need >=2 specular peaks, found {len(spec_peaks)}")
    sys.exit(1)

# Anchor = brightest peak using SMOOTHED magnitudes for noise-robust ranking
smoothed_lc = savgol_filter(observed_lc, window_length=7, polyorder=3)
smooth_mags_at_spec = smoothed_lc[spec_peaks]
smooth_ranking = np.argsort(smooth_mags_at_spec)
if (len(smooth_ranking) >= 2 and
    abs(smooth_mags_at_spec[smooth_ranking[0]] - smooth_mags_at_spec[smooth_ranking[1]]) < 0.05):
    tied = smooth_ranking[:2]
    anchor_rank = tied[np.argmin(spec_peaks[tied])]
else:
    anchor_rank = smooth_ranking[0]
anchor_idx = int(spec_peaks[anchor_rank])
anchor_time = obs_times[anchor_idx]
anchor_mag = observed_lc[anchor_idx]

anchor_allowed = get_allowed_normals(anchor_mag)

non_anchor = spec_peaks[spec_peaks != anchor_idx]
constraint_epochs = non_anchor
constraint_mags = observed_lc[constraint_epochs]
constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
dt_constraints = obs_times[constraint_epochs] - anchor_time
pab_at_constraints = pab_j2000[constraint_epochs]
n_constraints = len(constraint_epochs)

# k1/k2 in J2000 at constraint epochs (for BRDF cost)
k1_j2000_at_constraints = _sun_j2000[constraint_epochs]  # (N_c, 3)
k2_j2000_at_constraints = _obs_j2000[constraint_epochs]  # (N_c, 3)
obs_dist_at_constraints = CTX.obs_dist[constraint_epochs]  # (N_c,) in km

# ── Strategy selection: count bright non-anchor constraints ──
n_bright_constraints = int(np.sum(constraint_mags < BRDF_MAG_CUTOFF))
USE_BRDF = n_bright_constraints >= MIN_BRIGHT_FOR_BRDF

# Oracle reporting
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"Peaks: {len(peaks_idx)} total, {len(spec_peaks)} spec (< 9)")
print(f"|omega| est: {omega_est_dps:.3f} deg/s (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, t={anchor_time:.1f}s, mag={anchor_mag:.2f}")
print(f"  Anchor allowed normals: {[group_names[i] for i in anchor_allowed]}")
print(f"  Bright constraints (mag<{BRDF_MAG_CUTOFF}): {n_bright_constraints}/{n_constraints}")
print(f"  >>> STRATEGY: {'hybrid BRDF+align' if USE_BRDF else 'pure alignment'} <<<")
print(f"Constraints: {n_constraints}")
for ci in range(n_constraints):
    ep = constraint_epochs[ci]
    allowed = [group_names[i] for i in constraint_allowed[ci]]
    print(f"  ep {ep}: mag={constraint_mags[ci]:.2f} -> {allowed}")
step1_time = time.time() - t_step1
print(f"Step 1 done in {step1_time:.1f}s")


# ── Shared: delta-q propagation helper ─────────────────────────────────

def propagate_delta_qs(omega_vec, dt_arr):
    """Propagate identity quaternion to get delta-qs at constraint times."""
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


# ── BRDF cost function (replaces alignment and expected-dot costs) ────

def _brdf_flux_for_normal(ni, n_dot_k1, n_dot_k2, n_dot_h, h_dot_k1):
    """
    Compute flux from normal group ni for vectorised phi candidates.

    Parameters
    ----------
    ni : int — normal group index
    n_dot_k1 : (N_phi,) — normal · sun in body frame
    n_dot_k2 : (N_phi,) — normal · observer in body frame
    n_dot_h  : (N_phi,) — normal · halfway in body frame
    h_dot_k1 : (N_phi,) — halfway · sun (≈ cos(phase_angle/2))

    Returns
    -------
    flux : (N_phi,) — total flux contribution from this normal group
    """
    rd, rs, nph = GROUP_RD[ni], GROUP_RS[ni], GROUP_NPHONG[ni]
    area = group_areas[ni]

    # Diffuse component (Ashikhmin-Shirley)
    alpha = 1.0 - n_dot_k1 / 2.0
    beta = 1.0 - n_dot_k2 / 2.0
    rho_diff = (28.0 * rd / (23.0 * np.pi)) * (1.0 - rs) * \
               (1.0 - alpha**5) * (1.0 - beta**5)

    # Specular component with Fresnel
    fresnel = rs + (1.0 - rs) * (1.0 - h_dot_k1)**5
    max_n_dot = np.maximum(n_dot_k1, n_dot_k2)
    denom = h_dot_k1 * max_n_dot
    safe_denom = np.where(denom > 1e-10, denom, 1.0)
    rho_spec = np.where(
        denom > 1e-10,
        ((nph + 1.0) / (8.0 * np.pi)) * (n_dot_h**nph / safe_denom) * fresnel,
        0.0
    )

    rho = rho_diff + rho_spec
    return rho * area * n_dot_k1 * n_dot_k2


def vectorized_phi_cost_hybrid(q_anchors_xyzw, delta_qs,
                                k1_j2000_arr, k2_j2000_arr,
                                pab_j2000_arr,
                                obs_mags_arr, obs_dist_arr,
                                allowed_per_constraint, normals, w):
    """
    Hybrid cost: BRDF for bright constraints (mag < 7), alignment for dim.

    Bright peaks (mag < 7): single facet dominates → BRDF prediction is accurate.
      Cost = min_over_normals( (predicted_mag - observed_mag)^2 )
    Dim peaks (mag >= 7): multiple facets contribute → single-facet BRDF is biased.
      Cost = (1 - max_over_normals(n · PAB_body))^2  (standard alignment cost)

    Parameters
    ----------
    q_anchors_xyzw : (N_phi, 4) — anchor quaternions in xyzw format
    delta_qs : (N_constraints, 4) — delta quats at constraint times (wxyz)
    k1_j2000_arr : (N_constraints, 3) — sun direction in J2000
    k2_j2000_arr : (N_constraints, 3) — observer direction in J2000
    pab_j2000_arr : (N_constraints, 3) — PAB in J2000 (for alignment fallback)
    obs_mags_arr : (N_constraints,) — observed magnitudes
    obs_dist_arr : (N_constraints,) — observer distance in km
    allowed_per_constraint : list of lists — allowed normal indices
    normals : (10, 3) — unique normal vectors (body frame)
    w : float — weight per constraint
    """
    n_phi = len(q_anchors_xyzw)
    R_anchors = Rotation.from_quat(q_anchors_xyzw)
    costs = np.zeros(n_phi)

    for ci in range(len(delta_qs)):
        dq = delta_qs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta  # J2000→body rotation for all phi candidates

        obs_mag = obs_mags_arr[ci]
        allowed = allowed_per_constraint[ci]

        if obs_mag < BRDF_MAG_CUTOFF:
            # ── BRDF cost for bright peaks ──
            k1_body = R_all.apply(k1_j2000_arr[ci])  # (N_phi, 3)
            k2_body = R_all.apply(k2_j2000_arr[ci])  # (N_phi, 3)

            h_body = k1_body + k2_body
            h_norm = np.linalg.norm(h_body, axis=1, keepdims=True)
            h_body = h_body / np.maximum(h_norm, 1e-10)
            h_dot_k1 = np.sum(h_body * k1_body, axis=1)

            dist_m = obs_dist_arr[ci] * 1000.0
            five_log_d = 5.0 * np.log10(dist_m)

            best_cost = np.full(n_phi, np.inf)
            for ni in allowed:
                n_vec = normals[ni]
                n_dot_k1 = k1_body @ n_vec
                n_dot_k2 = k2_body @ n_vec
                n_dot_h = h_body @ n_vec

                visible = (n_dot_k1 > 0.01) & (n_dot_k2 > 0.01)
                flux = np.where(visible,
                                _brdf_flux_for_normal(ni, np.maximum(n_dot_k1, 0.01),
                                                      np.maximum(n_dot_k2, 0.01),
                                                      np.maximum(n_dot_h, 0.0), h_dot_k1),
                                0.0)
                pred_mag = np.where(
                    flux > 1e-20,
                    SUN_MAG + five_log_d - 2.5 * np.log10(flux),
                    20.0)
                cost_ni = np.where(visible, (pred_mag - obs_mag)**2, np.inf)
                best_cost = np.minimum(best_cost, cost_ni)

            costs += w * best_cost
        else:
            # ── Alignment cost for dim peaks ──
            pbs = R_all.apply(pab_j2000_arr[ci])  # (N_phi, 3)
            bds = (pbs @ normals[allowed].T).max(axis=1)
            costs += w * (1.0 - bds) ** 2

    return costs


# ── Alignment cost (kept for Step 4 geometric refinement) ─────────────

def vectorized_phi_cost_excl(q_anchors_xyzw, delta_qs, pab_arr,
                              allowed_per_constraint, normals, w):
    """Alignment cost: (1 - max_dot)^2 over allowed normals."""
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


def grid_cost(q_anchors_xyzw, delta_qs):
    """Adaptive cost: hybrid BRDF or pure alignment based on bright constraint count."""
    if USE_BRDF:
        return vectorized_phi_cost_hybrid(
            q_anchors_xyzw, delta_qs,
            _k1_j2000_c, _k2_j2000_c, _pab_j2000_c,
            _obs_mags_c, _obs_dist_c,
            _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)
    else:
        return vectorized_phi_cost_excl(
            q_anchors_xyzw, delta_qs, pab_at_constraints,
            _constraint_allowed, unique_normals, CONSTRAINT_WEIGHT)


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Omega grid search (BRDF COST — key change from m093)
# ══════════════════════════════════════════════════════════════════════
t_step2 = time.time()

omega_dirs = fibonacci_sphere(N_DIRS)
omega_mags_search = omega_est_rad * np.linspace(0.70, 1.30, N_MAGS)

# Pre-compute coarse phi anchor quaternions for each allowed anchor normal
phi_coarse_xy = np.linspace(0, np.pi, N_PHI_COARSE, endpoint=False)
phi_coarse_z = np.linspace(0, 2 * np.pi, 2 * N_PHI_COARSE, endpoint=False)
qa_anchor_sets = []
for ni in anchor_allowed:
    phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                    for p in phi_arr])
    qa_anchor_sets.append((ni, qa[:, [1, 2, 3, 0]]))

print(f"\n--- Step 2: Grid search ({N_DIRS} dirs x {N_MAGS} mags, "
      f"{N_PHI_COARSE} phis, {len(qa_anchor_sets)} anchor normals) "
      f"[{'hybrid BRDF+align' if USE_BRDF else 'alignment'}] ---",
      flush=True)

_omega_dirs = omega_dirs
_omega_mags_s = omega_mags_search
_qa_anchor_sets = qa_anchor_sets
_constraint_allowed = constraint_allowed
_k1_j2000_c = k1_j2000_at_constraints
_k2_j2000_c = k2_j2000_at_constraints
_obs_mags_c = constraint_mags
_obs_dist_c = obs_dist_at_constraints


_pab_j2000_c = pab_at_constraints


def eval_one_direction(wi):
    wd = _omega_dirs[wi]
    best_cost = np.inf
    best_omega = None
    best_ni = -1
    best_phi_idx = -1
    for mag in _omega_mags_s:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints)
        for ni, qa_xyzw in _qa_anchor_sets:
            c = grid_cost(qa_xyzw, dqs)
            bi = int(np.argmin(c))
            if c[bi] < best_cost:
                best_cost = c[bi]
                best_omega = omega_test.copy()
                best_ni = ni
                best_phi_idx = bi
    return best_cost, best_omega, best_ni, best_phi_idx


t_grid = time.time()
with Pool(GRID_WORKERS) as pool:
    results = pool.map(eval_one_direction, range(N_DIRS))
grid_time = time.time() - t_grid

grid_costs = np.array([r[0] for r in results])
grid_omegas = np.array([r[1] for r in results])
grid_best_ni = np.array([r[2] for r in results], dtype=int)
grid_best_phi = np.array([r[3] for r in results], dtype=int)

# Oracle check
sorted_idx = np.argsort(grid_costs)
for i in range(min(5, len(sorted_idx))):
    ri = sorted_idx[i]
    w_err = omega_dir_err(grid_omegas[ri], true_omega_anchor)
    print(f"  Grid#{i+1}: cost={grid_costs[ri]:.6f} | w_err={w_err:.1f}deg")

step2_time = time.time() - t_step2
print(f"Step 2 done in {step2_time:.1f}s")


# ══════════════════════════════════════════════════════════════════════
# STEP 2b: Lo-fi peak matching filter (unchanged from m090)
# ══════════════════════════════════════════════════════════════════════
t_step2b = time.time()

obs_peaks = peaks_idx

sorted_grid = np.argsort(grid_costs)
lofi_pool_size = min(LOFI_TOP, len(sorted_grid))

print(f"\n--- Step 2b: Lo-fi peak matching ({lofi_pool_size} candidates, "
      f"{LOFI_WORKERS} cores) ---", flush=True)

lofi_candidates = []
for rank in range(lofi_pool_size):
    gi = sorted_grid[rank]
    omega_cand = grid_omegas[gi]
    best_ni = int(grid_best_ni[gi])
    best_phi_idx = int(grid_best_phi[gi])

    phi_arr = phi_coarse_z if best_ni in Z_NORMALS else phi_coarse_xy
    best_qa = anchor_q_from_phi(phi_arr[best_phi_idx], unique_normals[best_ni],
                                 pab_j2000[anchor_idx])

    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(best_qa, -omega_cand, bt, "tumbling", I_tensor)
    q0_cand = qb[-1]
    w0_cand = -ob[-1]

    lofi_candidates.append({
        'grid_rank': rank, 'grid_idx': gi,
        'align_cost': float(grid_costs[gi]),
        'anchor_ni': best_ni,
        'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
        'omega_grid': omega_cand.copy(),
    })

_obs_times = obs_times
_obs_lc = observed_lc
_sun = CTX.sun_pos
_obs = CTX.obs_pos
_sat = CTX.sat_pos
_dist = CTX.obs_dist
_art = CTX.art_matrices
_I = I_tensor
_obs_peaks = obs_peaks


def eval_lofi_peaks(args):
    idx, q0_wxyz, w0_rad = args
    from src.computation.shadow_engine import create_no_shadow_lit_status as _no_shadow
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = _sun[:n_ep] - _sat[:n_ep]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = _obs[:n_ep] - _sat[:n_ep]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)

    lit = _no_shadow(_satellite, n_ep)
    pred_mags, _, _, _, _, _ = _gen_lc(
        facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=_dist,
        satellite=_satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=_art, show_progress=False)

    cand_peaks, _ = find_peaks(-pred_mags, distance=3, prominence=0.2)
    cand_peak_set = set(cand_peaks)
    n_matched = 0
    for op in _obs_peaks:
        for offset in range(-PEAK_WINDOW, PEAK_WINDOW + 1):
            if (op + offset) in cand_peak_set:
                n_matched += 1
                break

    mse = float(np.mean((pred_mags - _obs_lc) ** 2))
    return idx, n_matched, mse


lofi_args = [(i, lc['q0'], lc['w0']) for i, lc in enumerate(lofi_candidates)]
with Pool(LOFI_WORKERS) as pool:
    lofi_results = pool.map(eval_lofi_peaks, lofi_args)

for idx, n_matched, mse in lofi_results:
    lofi_candidates[idx]['n_matched'] = n_matched
    lofi_candidates[idx]['match_frac'] = n_matched / len(obs_peaks)
    lofi_candidates[idx]['lofi_mse'] = mse

lofi_candidates.sort(key=lambda c: (-c['n_matched'], c['lofi_mse']))

step2b_time = time.time() - t_step2b
print(f"Observed peaks: {len(obs_peaks)}")
print(f"Step 2b done in {step2b_time:.1f}s")
print(f"\nTop 10 by peak match:")
for i, lc in enumerate(lofi_candidates[:10]):
    w_err = omega_dir_err(lc['w0'], true_omega0)
    q0_err = attitude_error_deg(lc['q0'], true_q0)
    tag = " <--" if w_err < 10 and q0_err < 25 else ""
    print(f"  #{i+1}: matched={lc['n_matched']}/{len(obs_peaks)} "
          f"mse={lc['lofi_mse']:.3f} | q0={q0_err:.1f} w={w_err:.1f}{tag}")

nm_pool = lofi_candidates[:NM_TOP]


# ══════════════════════════════════════════════════════════════════════
# STEP 3: NM refinement (BRDF COST — key change from m093)
# ══════════════════════════════════════════════════════════════════════
t_step3 = time.time()

phi_fine_xy = np.linspace(0, np.pi, N_PHI_FINE, endpoint=False)
phi_fine_z = np.linspace(0, 2 * np.pi, 2 * N_PHI_FINE, endpoint=False)

_fine_phi_cache = {}
for ni in set(c['anchor_ni'] for c in nm_pool):
    phi_arr = phi_fine_z if ni in Z_NORMALS else phi_fine_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                    for p in phi_arr])
    _fine_phi_cache[ni] = (qa, qa[:, [1, 2, 3, 0]], phi_arr)

n_nm = min(NM_TOP, len(nm_pool))
print(f"\n--- Step 3: NM refinement of top {n_nm} "
      f"({'hybrid BRDF+align' if USE_BRDF else 'alignment'}, {N_PHI_FINE} phis) ---",
      flush=True)

_constraint_mags = constraint_mags


def refine_one_nm(args):
    idx, omega_start, fixed_ni = args
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[fixed_ni]

    def glint_cost(omega_vec):
        dqs = propagate_delta_qs(omega_vec, dt_constraints)
        return grid_cost(qa_xyzw, dqs).min()

    res = minimize(glint_cost, omega_start, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})

    # Final eval: best phi for this fixed normal
    dqs = propagate_delta_qs(res.x, dt_constraints)
    c = grid_cost(qa_xyzw, dqs)
    best_phi_idx = int(np.argmin(c))

    return idx, res.fun, res.x, best_phi_idx, fixed_ni, res.nfev


t_nm = time.time()
nm_args = [(i, nm_pool[i]['omega_grid'].copy(), nm_pool[i]['anchor_ni'])
           for i in range(n_nm)]
print(f"  Launching {n_nm} NM jobs on {NM_WORKERS} cores...", flush=True)
with Pool(NM_WORKERS) as pool:
    nm_results = pool.map(refine_one_nm, nm_args)

refined_costs = np.zeros(n_nm)
refined_omegas = np.zeros((n_nm, 3))
refined_best_phi_idx = np.zeros(n_nm, dtype=int)
refined_best_normal = np.zeros(n_nm, dtype=int)
for idx, cost, omega, bpi, bni, nfev in nm_results:
    refined_costs[idx] = cost
    refined_omegas[idx] = omega
    refined_best_phi_idx[idx] = bpi
    refined_best_normal[idx] = bni
print(f"NM done in {time.time() - t_nm:.1f}s")

step3_time = time.time() - t_step3

# Build candidates: cost-cluster then angular-deduplicate
ref_sorted = np.argsort(refined_costs)
sorted_ref_costs = refined_costs[ref_sorted]

COST_CLUSTER_GAP = 3.0
cluster_end = n_nm
for i in range(1, n_nm):
    if sorted_ref_costs[i] / max(sorted_ref_costs[i-1], 1e-15) > COST_CLUSTER_GAP:
        cluster_end = i
        break
MIN_CANDIDATES = 6  # BRDF cost is more discriminating — need a real pool for Steps 4-5
cluster_end = max(cluster_end, MIN_CANDIDATES)

ANGULAR_DEDUP_DEG = 10.0
cluster_indices = ref_sorted[:cluster_end]
keep = [0]
for i in range(1, len(cluster_indices)):
    ri = cluster_indices[i]
    is_dup = False
    for k in keep:
        rk = cluster_indices[k]
        if omega_dir_err(refined_omegas[ri], refined_omegas[rk]) < ANGULAR_DEDUP_DEG:
            is_dup = True
            break
    if not is_dup:
        keep.append(i)

deduped_indices = cluster_indices[keep]
print(f"  Cost cluster: {cluster_end}/{n_nm} (gap {COST_CLUSTER_GAP}x), "
      f"after omega dedup: {len(deduped_indices)}")

# ── Multi-phi: for each deduped omega, keep top-K separated phis ──
TOP_K_PHI = 4
PHI_SEP_DEG = 20.0  # minimum angular separation between phi candidates

candidates = []

for omega_rank, ri in enumerate(deduped_indices):
    ri = int(ri)
    omega_cand = refined_omegas[ri]
    ni = refined_best_normal[ri]
    qa_wxyz, qa_xyzw, phi_arr = _fine_phi_cache[ni]
    label = group_names[ni]
    gcost = refined_costs[ri]

    # Re-evaluate ALL phis with the refined omega
    dqs = propagate_delta_qs(omega_cand, dt_constraints)
    phi_costs = grid_cost(qa_xyzw, dqs)

    # Select top-K phis with angular separation
    phi_sorted = np.argsort(phi_costs)
    phi_step = np.deg2rad(PHI_SEP_DEG)
    selected_phis = [phi_sorted[0]]
    for pi in phi_sorted[1:]:
        if len(selected_phis) >= TOP_K_PHI:
            break
        # Check angular distance to all already-selected phis
        sep_ok = True
        for sp in selected_phis:
            # Angular distance on the phi circle
            diff = abs(phi_arr[pi] - phi_arr[sp])
            diff = min(diff, 2 * np.pi - diff)  # wrap around
            if diff < phi_step:
                sep_ok = False
                break
        if sep_ok:
            selected_phis.append(pi)

    for phi_rank, bpi in enumerate(selected_phis):
        qa = qa_wxyz[bpi]
        phi_deg = float(np.rad2deg(phi_arr[bpi]))
        pcost = float(phi_costs[bpi])

        # Back-propagate to t=0
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
        q0_cand = qb[-1]; w0_cand = -ob[-1]

        candidates.append({
            'omega_rank': omega_rank, 'phi_rank': phi_rank,
            'anchor': label, 'anchor_ni': ni,
            'phi_deg': phi_deg, 'glint_cost': pcost,
            'q_anchor': qa.copy(), 'omega_anchor': omega_cand.copy(),
            'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
            'q0_err': attitude_error_deg(q0_cand, true_q0),
            'w0_err': omega_dir_err(w0_cand, true_omega0),
            'w_mag_err': (np.rad2deg(np.linalg.norm(w0_cand)) - true_omega_mag_dps)
                         / true_omega_mag_dps * 100,
        })
        tag = " <--" if candidates[-1]['w0_err'] < 10 else ""
        marker = f"w#{omega_rank+1}.{phi_rank+1}"
        print(f"  {marker} {label} phi={phi_deg:5.1f} "
              f"cost={pcost:.2e} | q0={candidates[-1]['q0_err']:.1f} "
              f"w={candidates[-1]['w0_err']:.1f}{tag}")

print(f"Step 3 done in {step3_time:.1f}s, {len(candidates)} candidates "
      f"({len(deduped_indices)} omegas x up to {TOP_K_PHI} phis)")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Geometric refinement (alignment cost — unchanged from m090)
# ══════════════════════════════════════════════════════════════════════
t_step4 = time.time()

all_spec_epochs = spec_peaks
all_spec_mags = observed_lc[all_spec_epochs]
all_spec_allowed = [get_allowed_normals(m) for m in all_spec_mags]


def geometric_cost(params):
    """Alignment cost at all spec peaks — sharp gradients for L-BFGS-B refinement."""
    q0 = axis_angle_to_quaternion(params[:3])
    omega0 = params[3:6]
    quats, _ = propagate_attitude(q0, omega0, obs_times, "tumbling", I_tensor)
    cost = 0.0
    for i, ep in enumerate(all_spec_epochs):
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        allowed = all_spec_allowed[i]
        bd = max(np.dot(unique_normals[ni], pb) for ni in allowed)
        cost += CONSTRAINT_WEIGHT * (1.0 - bd) ** 2
    return cost


def refine_one_geo(args):
    idx, q0_wxyz, w0_rad = args
    aa = quaternion_to_axis_angle(q0_wxyz)
    x0 = np.concatenate([aa, w0_rad])
    res = minimize(geometric_cost, x0, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    q0_ref = axis_angle_to_quaternion(res.x[:3])
    w0_ref = res.x[3:6]
    return idx, res.fun, q0_ref, w0_ref, res.nfev


n_cand = len(candidates)
print(f"\n--- Step 4: Geometric refinement ({n_cand} cands, {GEO_WORKERS} cores) ---",
      flush=True)

geo_args = [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)]
with Pool(GEO_WORKERS) as pool:
    geo_results = pool.map(refine_one_geo, geo_args)

for idx, cost, q0_ref, w0_ref, nfev in geo_results:
    candidates[idx]['geo_cost'] = float(cost)
    candidates[idx]['q0_ref'] = q0_ref
    candidates[idx]['w0_ref'] = w0_ref
    candidates[idx]['q0_ref_err'] = attitude_error_deg(q0_ref, true_q0)
    candidates[idx]['w0_ref_err'] = omega_dir_err(w0_ref, true_omega0)

candidates.sort(key=lambda x: x['geo_cost'])
step4_time = time.time() - t_step4
print(f"Step 4 done in {step4_time:.1f}s")
print(f"\nGeo-refined ranking:")
for rank, c in enumerate(candidates):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor']} | "
          f"geo={c['geo_cost']:.6f} | q0={c['q0_ref_err']:.1f} "
          f"w={c['w0_ref_err']:.1f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4b: Lo-fi LC re-ranking of geo-refined candidates
# ══════════════════════════════════════════════════════════════════════
t_step4b = time.time()
n_rerank = len(candidates)
print(f"\n--- Step 4b: Lo-fi LC re-ranking ({n_rerank} geo-refined candidates) ---",
      flush=True)

lofi_rerank_args = [(i, c['q0_ref'].copy(), c['w0_ref'].copy())
                    for i, c in enumerate(candidates)]
if n_rerank > 4:
    with Pool(min(LOFI_WORKERS, n_rerank)) as pool:
        lofi_rerank = pool.map(eval_lofi_peaks, lofi_rerank_args)
else:
    lofi_rerank = [eval_lofi_peaks(a) for a in lofi_rerank_args]

for idx, n_matched, mse in lofi_rerank:
    candidates[idx]['lofi_rerank_matched'] = n_matched
    candidates[idx]['lofi_rerank_mse'] = mse

candidates.sort(key=lambda c: (-c['lofi_rerank_matched'], c['lofi_rerank_mse']))

step4b_time = time.time() - t_step4b
print(f"Step 4b done in {step4b_time:.1f}s")
for rank, c in enumerate(candidates):
    tag = " <--" if c['w0_ref_err'] < 10 else ""
    print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor']} | "
          f"matched={c['lofi_rerank_matched']}/{len(obs_peaks)} "
          f"mse={c['lofi_rerank_mse']:.4f} | "
          f"q0={c['q0_ref_err']:.1f} w={c['w0_ref_err']:.1f}{tag}")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Hi-fi LC evaluation (skippable with SKIP_HIFI=1)
# ══════════════════════════════════════════════════════════════════════
SKIP_HIFI = int(os.environ.get('SKIP_HIFI', '0'))

if SKIP_HIFI:
    step5_time = 0.0
    print(f"\n--- Step 5: SKIPPED (SKIP_HIFI=1) — using geo ranking ---", flush=True)
    # Winner is geo rank #1 (candidates already sorted by geo_cost)
    hifi_candidates = list(candidates)
else:
    t_step5 = time.time()

    hifi_candidates = list(candidates)
    print(f"\n--- Step 5: Hi-fi LC ({len(hifi_candidates)} candidates, "
          f"{HIFI_WORKERS} cores) ---", flush=True)

    _obs_times = obs_times
    _obs_lc = observed_lc


    def eval_full_hifi(args):
        idx, q0_wxyz, w0_rad = args
        from src.computation.shadow_engine import compute_shadows as _compute_shadows
        from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

        quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)

        n_ep = len(quats)
        R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        sv = _sun[:n_ep] - _sat[:n_ep]
        sv /= np.linalg.norm(sv, axis=1, keepdims=True)
        ov = _obs[:n_ep] - _sat[:n_ep]
        ov /= np.linalg.norm(ov, axis=1, keepdims=True)
        k1 = np.einsum('nij,nj->ni', R_all, sv)
        k2 = np.einsum('nij,nj->ni', R_all, ov)

        lit = _compute_shadows(satellite=_satellite, k1_vectors=k1,
                               explicit_component_matrices=_art, show_progress=False)
        pred_mags, _, _, _, _, _ = _gen_lc(
            facet_lit_status_dict=lit, k1_vectors_array=k1,
            k2_vectors_array=k2, observer_distances=_dist,
            satellite=_satellite, epochs=np.arange(n_ep, dtype=float),
            pre_computed_matrices=_art, show_progress=False)

        residual = float(np.mean((pred_mags - _obs_lc) ** 2))
        return idx, residual, pred_mags


    eval_args = [(i, c['q0_ref'].copy(), c['w0_ref'].copy())
                 for i, c in enumerate(hifi_candidates)]
    with Pool(HIFI_WORKERS) as pool:
        hifi_results = pool.map(eval_full_hifi, eval_args)

    for idx, hifi_res, pred_mags in hifi_results:
        hifi_candidates[idx]['hifi'] = hifi_res

    hifi_candidates.sort(key=lambda x: x['hifi'])
    step5_time = time.time() - t_step5

    print(f"Step 5 done in {step5_time:.1f}s")
    print(f"\nHi-fi ranking:")
    for rank, c in enumerate(hifi_candidates):
        tag = " <--" if c['w0_ref_err'] < 10 else ""
        print(f"  #{rank+1}: w#{c['omega_rank']+1} {c['anchor']} | "
              f"hifi={c['hifi']:.4f} | q0={c['q0_ref_err']:.1f} "
              f"w={c['w0_ref_err']:.1f}{tag}")

winner = hifi_candidates[0]
q0_refined = winner['q0_ref']
w0_refined = winner['w0_ref']
q0_err_refined = winner['q0_ref_err']
w0_err_refined = winner['w0_ref_err']
w_mag_refined = np.rad2deg(np.linalg.norm(w0_refined))
w_mag_err_refined = (w_mag_refined - true_omega_mag_dps) / true_omega_mag_dps * 100

total_time = time.time() - t_global


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RESULT (seed {TRAJ_SEED})")
print(f"{'='*60}")
print(f"  q0 error:     {q0_err_refined:.2f} deg")
print(f"  w dir error:  {w0_err_refined:.2f} deg")
print(f"  w mag error:  {w_mag_err_refined:+.2f}%")
print(f"  w estimated:  {np.rad2deg(w0_refined)} deg/s")
print(f"  w true:       {np.rad2deg(true_omega0)} deg/s")
print(f"")
print(f"Timing:")
print(f"  Step 1  (peaks+constraints):  {step1_time:6.1f}s")
print(f"  Step 2  (grid search):  {step2_time:6.1f}s")
print(f"  Step 2b (lo-fi peak match):  {step2b_time:6.1f}s")
print(f"  Step 3  (NM refinement):           {step3_time:6.1f}s")
print(f"  Step 4  (geo refinement):    {step4_time:6.1f}s")
print(f"  Step 4b (lo-fi re-rank):     {step4b_time:6.1f}s")
print(f"  Step 5  (hi-fi):             {step5_time:6.1f}s")
print(f"  Total:                      {total_time:6.1f}s ({total_time/60:.1f} min)")


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
np.savez(str(CKPT_DIR / "result.npz"),
         q0_refined=q0_refined, w0_refined=w0_refined,
         q0_pre_refine=winner['q0'], w0_pre_refine=winner['w0'],
         true_q0=true_q0, true_omega0=true_omega0)

result_json = {
    'traj_seed': TRAJ_SEED,
    'experiment': 'm094',
    'description': 'Hybrid BRDF (bright) + alignment (dim) cost in grid + NM',
    'n_peaks': int(len(peaks_idx)),
    'n_spec': int(len(spec_peaks)),
    'omega_est_dps': float(omega_est_dps),
    'n_hifi_candidates': len(hifi_candidates),
    'winner': {
        'omega_rank': winner['omega_rank'], 'phi_rank': winner.get('phi_rank', 0),
        'anchor': winner['anchor'],
        'phi_deg': winner['phi_deg'], 'glint_cost': winner['glint_cost'],
        'geo_cost': winner['geo_cost'], 'hifi': winner['hifi'],
        'q0_err': float(q0_err_refined),
        'w0_err': float(w0_err_refined),
        'w_mag_err_pct': float(w_mag_err_refined),
        'q0_wxyz': q0_refined.tolist(),
        'w0_rad': w0_refined.tolist(),
        'w0_dps': np.rad2deg(w0_refined).tolist(),
    },
    'hifi_candidates': [
        {
            'omega_rank': c['omega_rank'], 'anchor': c['anchor'],
            'phi_deg': c['phi_deg'], 'glint_cost': c['glint_cost'],
            'geo_cost': c['geo_cost'], 'hifi': c['hifi'],
            'q0_err': c['q0_ref_err'], 'w0_err': c['w0_ref_err'],
            'q0_wxyz': c['q0_ref'].tolist(),
            'w0_rad': c['w0_ref'].tolist(),
            'w0_dps': np.rad2deg(c['w0_ref']).tolist(),
            'w_mag_err_pct': (np.rad2deg(np.linalg.norm(c['w0_ref'])) - true_omega_mag_dps)
                             / true_omega_mag_dps * 100,
        }
        for c in hifi_candidates
    ],
    'all_geo_candidates': [
        {
            'omega_rank': c['omega_rank'], 'anchor': c['anchor'],
            'phi_deg': c['phi_deg'], 'glint_cost': c['glint_cost'],
            'geo_cost': c['geo_cost'],
            'q0_err': c['q0_ref_err'], 'w0_err': c['w0_ref_err'],
        }
        for c in candidates
    ],
    'timing': {
        'step1_s': float(step1_time),
        'step2_grid_s': float(step2_time),
        'step2b_lofi_s': float(step2b_time),
        'step3_nm_s': float(step3_time),
        'step4_geo_s': float(step4_time),
        'step4b_lofi_rerank_s': float(step4b_time),
        'step5_hifi_s': float(step5_time),
        'total_s': float(total_time),
    },
}

save_results(str(CKPT_DIR / "result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")

if w0_err_refined < 2:
    print(f"\n*** SUCCESS (omega) ***")
    if q0_err_refined < 5:
        print(f"*** SUCCESS (attitude) ***")
    else:
        print(f"    Attitude: {q0_err_refined:.1f} deg (phi degeneracy expected ~180 deg)")
elif w0_err_refined < 5:
    print(f"\n*** PARTIAL SUCCESS ***")
else:
    print(f"\n*** FAILED ***")

sys.stdout = sys.__stdout__
_log_file.close()
