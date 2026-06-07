#!/usr/bin/env python3
"""
m069 — Full inversion pipeline (single seed, single anchor hypothesis).

End-to-end pipeline for recovering (q0, omega0) from an observed hi-fi
light curve. No oracles at any stage.

Pipeline:
  Step 1: Peak count → |omega| estimate
  Step 2: Anchor selection (brightest specular glint, mag < 6.0)
  Step 3: Omega grid search (2000 dirs × 20 mags, ±X glint alignment,
          parallelized on 8 cores) → NM refinement of top 50 → top 5
          [CHECKPOINTED]
  Step 4: Phi sweep (360 bins, 1° spacing, ±X-only scoring)
          → top 2 phi per omega → 10 candidates
  Step 5: Full-curve hi-fi (500 ep) for ALL 10 candidates (parallelized)
          → winner by hi-fi residual
  Step 6: Geometric refinement of winner (L-BFGS-B on specular + bright
          alignment constraints, no BRDF/shadows)

Usage:
  MICRO69_SEED=93 python3 m069_full_pipeline.py
  MICRO69_SEED=93 MICRO69_ANCHOR=0 python3 m069_full_pipeline.py
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
from scipy.signal import find_peaks
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

TRAJ_SEED = int(os.environ.get('MICRO69_SEED', '93'))
ANCHOR_GROUP = int(os.environ.get('MICRO69_ANCHOR', '0'))  # 0=+X, 1=-X

# Pipeline parameters
N_DIRS = 2000         # Fibonacci sphere directions
N_MAGS = 20           # magnitude grid points (±20% around estimate)
N_PHI = 360           # phi bins (1° spacing)
TOP_PHI = 2           # phi candidates per omega
TOP_N_OMEGA = 5       # omega candidates from NM
NM_TOP = 50           # NM refinement pool size


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


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print(f"m069 — Full pipeline (seed {TRAJ_SEED}, G{ANCHOR_GROUP} anchor)")
print("=" * 60)
t_global = time.time()

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][TRAJ_SEED]
true_omega0 = master['omega0s'][TRAJ_SEED]
true_omega_mag_dps = float(master['omega_mags'][TRAJ_SEED])
true_quats = master['quaternions'][TRAJ_SEED]
true_lc = master['mag_hifi'][TRAJ_SEED]
n_normals = len(unique_normals)
n_pX = unique_normals[0]   # +X
n_mX = unique_normals[1]   # -X

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

print(f"Setup done in {time.time() - t_global:.1f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Peak count → |omega| estimate
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 1: Peak count + anchor selection ---", flush=True)

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

specular = peaks_idx[observed_lc[peaks_idx] < 6.0]
bright = peaks_idx[(observed_lc[peaks_idx] >= 6.0) & (observed_lc[peaks_idx] < 9.0)]

if len(specular) < 2:
    print(f"ERROR: Need ≥2 specular glints, found {len(specular)}")
    sys.exit(1)

anchor_idx = int(specular[np.argmin(observed_lc[specular])])
anchor_time = obs_times[anchor_idx]
non_anchor_spec = specular[specular != anchor_idx]
dt_constraints = obs_times[non_anchor_spec] - anchor_time
pab_at_constraints = pab_j2000[non_anchor_spec]
n_body_anchor = unique_normals[ANCHOR_GROUP]

# Oracle reporting
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"Peaks: {len(peaks_idx)} total, {len(specular)} specular, {len(bright)} bright")
print(f"|omega| est: {omega_est_dps:.3f} deg/s (true: {true_omega_mag_dps:.3f})")
print(f"Anchor: ep {anchor_idx}, t={anchor_time:.1f}s, mag={observed_lc[anchor_idx]:.2f}")
print(f"Constraints: {len(non_anchor_spec)} specular glints")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Omega grid search (PARALLELIZED, CHECKPOINTED)
# ══════════════════════════════════════════════════════════════════════
CKPT_DIR = RESULTS_DIR / f"m069_pipeline_seed{TRAJ_SEED:03d}"
CKPT_DIR.mkdir(exist_ok=True)
ckpt_grid = CKPT_DIR / f"G{ANCHOR_GROUP}_grid.npz"

if ckpt_grid.exists():
    print(f"\n--- Step 2: Loading grid checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_grid))
    grid_costs = ckpt['costs']
    grid_omegas = ckpt['omegas']
    print(f"Loaded {len(grid_costs)} grid results")
else:
    print(f"\n--- Step 2: Omega grid search ({N_DIRS} dirs × {N_MAGS} mags) ---",
          flush=True)

    omega_dirs = fibonacci_sphere(N_DIRS)
    omega_mags_search = omega_est_rad * np.linspace(0.80, 1.20, N_MAGS)

    phi_vals = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    q_anchors = np.array([anchor_q_from_phi(p, n_body_anchor, pab_j2000[anchor_idx])
                          for p in phi_vals])

    q_identity = np.array([1.0, 0.0, 0.0, 0.0])

    def eval_one_direction(wi):
        wd = omega_dirs[wi]
        best_cost = np.inf
        best_omega = None
        for mag in omega_mags_search:
            omega_test = wd * mag
            fwd_mask = dt_constraints > 1e-6
            bwd_mask = dt_constraints < -1e-6
            delta_qs = np.zeros((len(dt_constraints), 4))
            delta_qs[np.abs(dt_constraints) < 1e-6] = q_identity
            if np.any(fwd_mask):
                fwd_dt = np.sort(dt_constraints[fwd_mask])
                dq, _ = propagate_attitude(q_identity, omega_test,
                    np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
                delta_qs[fwd_mask] = dq[1:][np.argsort(np.argsort(dt_constraints[fwd_mask]))]
            if np.any(bwd_mask):
                bwd_dt = np.sort(-dt_constraints[bwd_mask])
                dq, _ = propagate_attitude(q_identity, -omega_test,
                    np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
                dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
                delta_qs[bwd_mask] = dq_c[np.argsort(np.argsort(-dt_constraints[bwd_mask]))]
            for qa in q_anchors:
                cost = 0.0
                for ci in range(len(dt_constraints)):
                    qg = quat_multiply(qa, delta_qs[ci])
                    qg = qg / np.linalg.norm(qg)
                    R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
                    pb = R @ pab_at_constraints[ci]
                    bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
                    cost += (1.0 - bd) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best_omega = omega_test.copy()
        return best_cost, best_omega

    t_grid = time.time()
    with Pool(8) as pool:
        results = pool.map(eval_one_direction, range(N_DIRS))
    grid_time = time.time() - t_grid

    grid_costs = np.array([r[0] for r in results])
    grid_omegas = np.array([r[1] for r in results])
    np.savez(str(ckpt_grid), costs=grid_costs, omegas=grid_omegas)
    print(f"Grid done in {grid_time:.1f}s, saved checkpoint")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: NM refinement of top 50 (CHECKPOINTED)
# ══════════════════════════════════════════════════════════════════════
ckpt_nm = CKPT_DIR / f"G{ANCHOR_GROUP}_refined.npz"

if ckpt_nm.exists():
    print(f"\n--- Step 3: Loading NM checkpoint ---", flush=True)
    ckpt = np.load(str(ckpt_nm))
    refined_costs = ckpt['costs']
    refined_omegas = ckpt['omegas']
else:
    print(f"\n--- Step 3: NM refinement of top {NM_TOP} ---", flush=True)

    sorted_idx = np.argsort(grid_costs)
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    phi_vals = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    q_anchors = np.array([anchor_q_from_phi(p, n_body_anchor, pab_j2000[anchor_idx])
                          for p in phi_vals])

    def glint_cost(omega_vec):
        fwd_mask = dt_constraints > 1e-6
        bwd_mask = dt_constraints < -1e-6
        delta_qs = np.zeros((len(dt_constraints), 4))
        delta_qs[np.abs(dt_constraints) < 1e-6] = q_identity
        if np.any(fwd_mask):
            fwd_dt = np.sort(dt_constraints[fwd_mask])
            dq, _ = propagate_attitude(q_identity, omega_vec,
                np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
            delta_qs[fwd_mask] = dq[1:][np.argsort(np.argsort(dt_constraints[fwd_mask]))]
        if np.any(bwd_mask):
            bwd_dt = np.sort(-dt_constraints[bwd_mask])
            dq, _ = propagate_attitude(q_identity, -omega_vec,
                np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
            dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
            delta_qs[bwd_mask] = dq_c[np.argsort(np.argsort(-dt_constraints[bwd_mask]))]
        best_cost = np.inf
        for qa in q_anchors:
            cost = 0.0
            for ci in range(len(dt_constraints)):
                qg = quat_multiply(qa, delta_qs[ci])
                qg /= np.linalg.norm(qg)
                R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
                pb = R @ pab_at_constraints[ci]
                bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
                cost += (1.0 - bd) ** 2
            best_cost = min(best_cost, cost)
        return best_cost

    t_nm = time.time()
    refined_costs = np.zeros(NM_TOP)
    refined_omegas = np.zeros((NM_TOP, 3))
    for i in range(NM_TOP):
        wi = sorted_idx[i]
        res = minimize(glint_cost, grid_omegas[wi], method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-10})
        refined_costs[i] = res.fun
        refined_omegas[i] = res.x
        if i < 5:
            w_err = omega_dir_err(res.x, true_omega_anchor)
            print(f"  #{i+1}: {grid_costs[wi]:.6f} → {res.fun:.6f} | "
                  f"ω_err={w_err:.1f}° nfev={res.nfev}", flush=True)

    np.savez(str(ckpt_nm), costs=refined_costs, omegas=refined_omegas)
    print(f"NM done in {time.time() - t_nm:.1f}s, saved checkpoint")

ref_sorted = np.argsort(refined_costs)
print(f"\nTop {TOP_N_OMEGA} by NM glint cost:")
for i in range(TOP_N_OMEGA):
    ri = ref_sorted[i]
    w_err = omega_dir_err(refined_omegas[ri], true_omega_anchor)
    print(f"  NM#{i+1}: cost={refined_costs[ri]:.8f} | ω_err={w_err:.1f}°")


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Phi sweep (360 bins, ±X-only scoring)
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 4: Phi sweep ({N_PHI} bins, ±X only) ---", flush=True)
t_phi = time.time()

q_identity = np.array([1.0, 0.0, 0.0, 0.0])
phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
candidates = []

for omega_rank in range(TOP_N_OMEGA):
    ri = ref_sorted[omega_rank]
    omega_cand = refined_omegas[ri]

    # Delta-q propagation
    fwd_mask = dt_constraints > 1e-6
    bwd_mask = dt_constraints < -1e-6
    delta_qs = np.zeros((len(dt_constraints), 4))
    delta_qs[np.abs(dt_constraints) < 1e-6] = q_identity
    if np.any(fwd_mask):
        fwd_dt = np.sort(dt_constraints[fwd_mask])
        dq, _ = propagate_attitude(q_identity, omega_cand,
            np.concatenate([[0.0], fwd_dt]), "tumbling", I_tensor)
        delta_qs[fwd_mask] = dq[1:][np.argsort(np.argsort(dt_constraints[fwd_mask]))]
    if np.any(bwd_mask):
        bwd_dt = np.sort(-dt_constraints[bwd_mask])
        dq, _ = propagate_attitude(q_identity, -omega_cand,
            np.concatenate([[0.0], bwd_dt]), "tumbling", I_tensor)
        dq_c = dq[1:].copy(); dq_c[:, 1:] *= -1
        delta_qs[bwd_mask] = dq_c[np.argsort(np.argsort(-dt_constraints[bwd_mask]))]

    # Sweep phi, score ±X only
    phi_results = []
    for phi in phi_values:
        qa = anchor_q_from_phi(phi, n_body_anchor, pab_j2000[anchor_idx])
        cost = 0.0
        for ci in range(len(dt_constraints)):
            qg = quat_multiply(qa, delta_qs[ci])
            qg = qg / np.linalg.norm(qg)
            R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
            pb = R @ pab_at_constraints[ci]
            bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
            cost += (1.0 - bd) ** 2
        phi_results.append((phi, cost, qa))
    phi_results.sort(key=lambda x: x[1])

    for phi_rank in range(TOP_PHI):
        phi, gcost, qa = phi_results[phi_rank]
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
        q0_cand = qb[-1]; w0_cand = -ob[-1]
        candidates.append({
            'omega_rank': omega_rank, 'phi_rank': phi_rank,
            'phi_deg': float(np.rad2deg(phi)), 'glint_cost': gcost,
            'q_anchor': qa.copy(), 'omega_anchor': omega_cand.copy(),
            'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
            'q0_err': attitude_error_deg(q0_cand, true_q0),
            'w0_err': omega_dir_err(w0_cand, true_omega0),
            'w_mag_err': (np.rad2deg(np.linalg.norm(w0_cand)) - true_omega_mag_dps)
                         / true_omega_mag_dps * 100,
        })
        tag = " <--" if candidates[-1]['w0_err'] < 10 else ""
        print(f"  ω#{omega_rank+1} φ#{phi_rank+1}: phi={np.rad2deg(phi):.1f}° "
              f"glint={gcost:.8f} | q0={candidates[-1]['q0_err']:.1f}° "
              f"ω={candidates[-1]['w0_err']:.1f}°{tag}")

print(f"Phi sweep done in {time.time() - t_phi:.1f}s, {len(candidates)} candidates")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Full-curve hi-fi for ALL candidates (PARALLELIZED)
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 5: Full hi-fi ({len(candidates)} cands, 8 cores) ---", flush=True)
t_hifi = time.time()

_satellite = CTX.satellite
_obs_times = obs_times
_obs_lc = observed_lc
_sun = CTX.sun_pos
_obs = CTX.obs_pos
_sat = CTX.sat_pos
_dist = CTX.obs_dist
_art = CTX.art_matrices
_I = I_tensor


def eval_full_hifi(args):
    idx, q0_wxyz, w0_rad = args
    obj = ObjectiveFunction(
        satellite=_satellite, observation_times=_obs_times,
        observed_lightcurve=_obs_lc,
        sun_positions_j2000=_sun, observer_positions_j2000=_obs,
        satellite_positions_j2000=_sat, observer_distances=_dist,
        compute_shadows_flag=True, articulation_matrices=_art,
        mode="tumbling", inertia_tensor=_I, show_progress=False)
    rv = Rotation.from_quat([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3],
                              q0_wxyz[0]]).as_rotvec()
    return idx, obj.evaluate(np.concatenate([rv, w0_rad]))


eval_args = [(i, c['q0'].copy(), c['w0'].copy()) for i, c in enumerate(candidates)]
with Pool(8) as pool:
    results = pool.map(eval_full_hifi, eval_args)

for idx, hifi_res in results:
    candidates[idx]['full_hifi'] = hifi_res

candidates.sort(key=lambda x: x['full_hifi'])
hifi_time = time.time() - t_hifi

print(f"Full hi-fi done in {hifi_time:.1f}s")
print(f"\nRanking:")
for rank, c in enumerate(candidates):
    tag = " <--" if c['w0_err'] < 10 else ""
    print(f"  #{rank+1}: ω#{c['omega_rank']+1} φ#{c['phi_rank']+1} | "
          f"hifi={c['full_hifi']:.4f} | q0={c['q0_err']:.1f}° "
          f"ω={c['w0_err']:.1f}°{tag}")

winner = candidates[0]
print(f"\nHi-fi winner: q0={winner['q0_err']:.1f}° ω={winner['w0_err']:.1f}°")


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Geometric refinement of winner
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 6: Geometric refinement ---", flush=True)
t_refine = time.time()

specular_epochs = specular
bright_epochs = bright


def geometric_cost(params):
    q0 = axis_angle_to_quaternion(params[:3])
    omega0 = params[3:6]
    quats, _ = propagate_attitude(q0, omega0, obs_times, "tumbling", I_tensor)
    cost = 0.0
    for ep in specular_epochs:
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        bd = max(np.dot(n_pX, pb), np.dot(n_mX, pb))
        cost += 10.0 * (1.0 - bd) ** 2
    for ep in bright_epochs:
        R = Rotation.from_quat([quats[ep][1], quats[ep][2],
                                 quats[ep][3], quats[ep][0]]).as_matrix()
        pb = R @ pab_j2000[ep]
        bd = max(np.dot(unique_normals[ni], pb) for ni in range(n_normals))
        cost += 5.0 * (1.0 - bd) ** 2
    return cost


aa_start = quaternion_to_axis_angle(winner['q0'])
x0 = np.concatenate([aa_start, winner['w0']])

res = minimize(geometric_cost, x0, method='L-BFGS-B',
               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})

q0_refined = axis_angle_to_quaternion(res.x[:3])
w0_refined = res.x[3:6]

q0_err_refined = attitude_error_deg(q0_refined, true_q0)
w0_err_refined = omega_dir_err(w0_refined, true_omega0)
w_mag_refined = np.rad2deg(np.linalg.norm(w0_refined))
w_mag_err_refined = (w_mag_refined - true_omega_mag_dps) / true_omega_mag_dps * 100

refine_time = time.time() - t_refine
print(f"Refined in {refine_time:.1f}s ({res.nfev} evals)")
print(f"  q0: {winner['q0_err']:.1f}° → {q0_err_refined:.2f}°")
print(f"  ω:  {winner['w0_err']:.1f}° → {w0_err_refined:.2f}°")
print(f"  |ω|: {w_mag_err_refined:+.2f}%")

total_time = time.time() - t_global


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RESULT (seed {TRAJ_SEED}, G{ANCHOR_GROUP} anchor)")
print(f"{'='*60}")
print(f"  q0 error:     {q0_err_refined:.2f}°")
print(f"  ω dir error:  {w0_err_refined:.2f}°")
print(f"  ω mag error:  {w_mag_err_refined:+.2f}%")
print(f"  ω estimated:  {np.rad2deg(w0_refined)} deg/s")
print(f"  ω true:       {np.rad2deg(true_omega0)} deg/s")
print(f"")
print(f"Timing:")
print(f"  Grid search:   checkpointed")
print(f"  Phi sweep:     {time.time() - t_phi - hifi_time - refine_time:.1f}s")
print(f"  Full hi-fi:    {hifi_time:.1f}s")
print(f"  Refinement:    {refine_time:.1f}s")
print(f"  Total:         {total_time:.1f}s ({total_time/60:.1f} min)")


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
np.savez(str(CKPT_DIR / f"G{ANCHOR_GROUP}_result.npz"),
         q0_refined=q0_refined, w0_refined=w0_refined,
         q0_pre_refine=winner['q0'], w0_pre_refine=winner['w0'],
         true_q0=true_q0, true_omega0=true_omega0)

result_json = {
    'traj_seed': TRAJ_SEED,
    'anchor_group': ANCHOR_GROUP,
    'n_peaks': int(len(peaks_idx)),
    'n_specular': int(len(specular)),
    'n_bright': int(len(bright)),
    'omega_est_dps': float(omega_est_dps),
    'pre_refine': {
        'q0_err': float(winner['q0_err']),
        'w0_err': float(winner['w0_err']),
        'w_mag_err': float(winner['w_mag_err']),
        'hifi_residual': float(winner['full_hifi']),
        'omega_rank': winner['omega_rank'],
        'phi_deg': winner['phi_deg'],
    },
    'refined': {
        'q0_err': float(q0_err_refined),
        'w0_err': float(w0_err_refined),
        'w_mag_err_pct': float(w_mag_err_refined),
        'q0_wxyz': q0_refined.tolist(),
        'w0_rad': w0_refined.tolist(),
        'w0_dps': np.rad2deg(w0_refined).tolist(),
    },
    'all_candidates': [
        {
            'omega_rank': c['omega_rank'], 'phi_rank': c['phi_rank'],
            'phi_deg': c['phi_deg'], 'glint_cost': c['glint_cost'],
            'full_hifi': c['full_hifi'],
            'q0_err': c['q0_err'], 'w0_err': c['w0_err'],
        }
        for c in candidates
    ],
    'timing': {
        'hifi_s': float(hifi_time),
        'refine_s': float(refine_time),
        'total_s': float(total_time),
    },
}

save_results(str(CKPT_DIR / f"G{ANCHOR_GROUP}_result.json"), result_json)
print(f"\nSaved to {CKPT_DIR}/")

if q0_err_refined < 5 and w0_err_refined < 2:
    print(f"\n*** SUCCESS ***")
elif q0_err_refined < 10 and w0_err_refined < 5:
    print(f"\n*** PARTIAL SUCCESS ***")
else:
    print(f"\n*** FAILED ***")
