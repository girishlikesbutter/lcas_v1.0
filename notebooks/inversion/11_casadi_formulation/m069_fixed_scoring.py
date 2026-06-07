#!/usr/bin/env python3
"""
m069 — Fixed scoring pipeline for seed 93.

Fixes applied vs m068:
  1. Phi sweep uses ±X-only alignment (not all 10 normals)
  2. Phi sweep uses 360 bins (1° spacing), not 36 (10°)
  3. Full-curve hi-fi scoring on ALL candidates (no lo-fi, no elimination)

Architecture:
  Step 1: Load NM-refined top-5 omega from m068 checkpoint
  Step 2: Phi sweep (360 bins, ±X scoring) → top 2 phi per omega → 10 candidates
  Step 3: Back-propagate all 10 to t=0
  Step 4: Full-curve hi-fi (500 ep) for ALL 10, parallelized on 8 cores
  Step 5: Winner by full hi-fi residual
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
CHECKPOINT_DIR = RESULTS_DIR / "m068_checkpoints"

TRAJ_SEED = 93
ANCHOR_GROUP = 0  # +X
N_PHI = 360       # 1-degree spacing
TOP_PHI = 2       # top 2 phi per omega (correct + antiparallel)
TOP_N_OMEGA = 5   # top 5 from NM glint cost


# ── Helpers ─────────────────────────────────────────────────────────────

def anchor_q_from_phi(phi, n_body, pab):
    """Quaternion (wxyz) aligning body normal with PAB, twist angle phi."""
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
print(f"m069 — Fixed scoring (seed {TRAJ_SEED}, G{ANCHOR_GROUP} anchor)")
print(f"  Fix 1: ±X-only phi sweep")
print(f"  Fix 2: {N_PHI} phi bins (1° spacing)")
print(f"  Fix 3: full-curve hi-fi on ALL candidates")
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

rng = np.random.default_rng(42)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Peak detection and anchor selection
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
specular = peaks_idx[observed_lc[peaks_idx] < 6.0]
anchor_idx = int(specular[np.argmin(observed_lc[specular])])
anchor_time = obs_times[anchor_idx]
non_anchor = specular[specular != anchor_idx]
dt_constraints = obs_times[non_anchor] - anchor_time
pab_at_constraints = pab_j2000[non_anchor]

# True omega at anchor (oracle reporting only)
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]
true_q_anchor = true_quats[anchor_idx]

print(f"Setup done in {time.time() - t_global:.1f}s")
print(f"Anchor: epoch {anchor_idx}, t={anchor_time:.1f}s, mag={observed_lc[anchor_idx]:.2f}")
print(f"Non-anchor specular glints: {len(non_anchor)} at epochs {non_anchor}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Load NM-refined omega candidates
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 1: Load NM-refined omega candidates ---", flush=True)

ckpt = np.load(str(CHECKPOINT_DIR / f"seed{TRAJ_SEED:03d}_G{ANCHOR_GROUP}_refined.npz"))
refined_costs = ckpt['costs']
refined_omegas = ckpt['omegas']
ref_sorted = np.argsort(refined_costs)

print(f"Top {TOP_N_OMEGA} by NM glint cost:")
for i in range(TOP_N_OMEGA):
    ri = ref_sorted[i]
    w_err = omega_dir_err(refined_omegas[ri], true_omega_anchor)
    print(f"  NM#{i+1}: cost={refined_costs[ri]:.8f} | ω_err_anchor={w_err:.1f}°")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Phi sweep (360 bins, ±X-only scoring)
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 2: Phi sweep ({N_PHI} bins, ±X only) ---", flush=True)
t_phi = time.time()

n_body_anchor = unique_normals[ANCHOR_GROUP]  # +X
n_plus_x = unique_normals[0]
n_minus_x = unique_normals[1]
q_identity = np.array([1.0, 0.0, 0.0, 0.0])
phi_values = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

candidates = []

for omega_rank in range(TOP_N_OMEGA):
    ri = ref_sorted[omega_rank]
    omega_cand = refined_omegas[ri]

    # Propagate delta_q for all constraint epochs
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
        dq_c = dq[1:].copy()
        dq_c[:, 1:] *= -1
        delta_qs[bwd_mask] = dq_c[np.argsort(np.argsort(-dt_constraints[bwd_mask]))]

    # Sweep 360 phi, score by ±X ONLY
    phi_results = []
    for phi in phi_values:
        qa = anchor_q_from_phi(phi, n_body_anchor, pab_j2000[anchor_idx])
        cost = 0.0
        for ci in range(len(dt_constraints)):
            qg = quat_multiply(qa, delta_qs[ci])
            qg = qg / np.linalg.norm(qg)
            R = Rotation.from_quat([qg[1], qg[2], qg[3], qg[0]]).as_matrix()
            pb = R @ pab_at_constraints[ci]
            bd = max(np.dot(n_plus_x, pb), np.dot(n_minus_x, pb))
            cost += (1.0 - bd) ** 2
        phi_results.append((phi, cost, qa))

    phi_results.sort(key=lambda x: x[1])

    # Top 2 phi per omega
    for phi_rank in range(TOP_PHI):
        phi, gcost, qa = phi_results[phi_rank]

        # Back-propagate to t=0
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega_cand, bt, "tumbling", I_tensor)
        q0_cand = qb[-1]
        w0_cand = -ob[-1]

        q0_err = attitude_error_deg(q0_cand, true_q0)
        w0_err = omega_dir_err(w0_cand, true_omega0)
        w_mag_dps = np.rad2deg(np.linalg.norm(w0_cand))
        w_mag_err = (w_mag_dps - true_omega_mag_dps) / true_omega_mag_dps * 100

        candidates.append({
            'omega_rank': omega_rank, 'phi_rank': phi_rank,
            'phi_deg': float(np.rad2deg(phi)), 'glint_cost': gcost,
            'q0': q0_cand.copy(), 'w0': w0_cand.copy(),
            'q0_err': q0_err, 'w0_err': w0_err, 'w_mag_err': w_mag_err,
        })

        tag = " <--" if w0_err < 10 else ""
        print(f"  ω#{omega_rank+1} φ#{phi_rank+1}: phi={np.rad2deg(phi):.1f}° "
              f"glint={gcost:.8f} | q0={q0_err:.1f}° ω={w0_err:.1f}° "
              f"|ω|={w_mag_err:+.1f}%{tag}")

phi_time = time.time() - t_phi
print(f"\nPhi sweep done in {phi_time:.1f}s, {len(candidates)} candidates")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Full-curve hi-fi for ALL candidates (parallelized)
# ══════════════════════════════════════════════════════════════════════
print(f"\n--- Step 3: Full hi-fi (500 ep) for all {len(candidates)} (8 cores) ---",
      flush=True)
t_hifi = time.time()

# Module-level refs for forked workers
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
    """Full-curve hi-fi evaluation of one candidate."""
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

print(f"\nFinal ranking by full-curve hi-fi:")
for rank, c in enumerate(candidates):
    tag = " <--" if c['w0_err'] < 10 else ""
    print(f"  #{rank+1}: ω#{c['omega_rank']+1} φ#{c['phi_rank']+1} "
          f"phi={c['phi_deg']:.1f}° | hifi={c['full_hifi']:.4f} | "
          f"q0={c['q0_err']:.1f}° ω={c['w0_err']:.1f}° "
          f"|ω|={c['w_mag_err']:+.1f}%{tag}")

winner = candidates[0]
total_time = time.time() - t_global


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RESULT")
print(f"{'='*60}")
print(f"q0 error:     {winner['q0_err']:.2f}°")
print(f"ω dir error:  {winner['w0_err']:.2f}°")
print(f"ω mag error:  {winner['w_mag_err']:+.2f}%")
print(f"ω final: {np.rad2deg(winner['w0'])} deg/s")
print(f"ω true:  {np.rad2deg(true_omega0)} deg/s")
print(f"")
print(f"Timing:")
print(f"  Phi sweep ({N_PHI} bins): {phi_time:.1f}s")
print(f"  Full hi-fi ({len(candidates)} cands): {hifi_time:.1f}s")
print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} min)")

# Oracle tracking
print(f"\nOracle tracking:")
for rank, c in enumerate(candidates):
    if c['w0_err'] < 5:
        print(f"  Truth at full hi-fi rank #{rank+1}/{len(candidates)}")
        print(f"    q0={c['q0_err']:.1f}° ω={c['w0_err']:.1f}° hifi={c['full_hifi']:.4f}")
        break
else:
    print(f"  No candidate within 5° of true omega direction")

# ── Save ──────────────────────────────────────────────────────────────
results = {
    'traj_seed': TRAJ_SEED,
    'anchor_group': ANCHOR_GROUP,
    'n_phi_bins': N_PHI,
    'top_phi_per_omega': TOP_PHI,
    'top_n_omega': TOP_N_OMEGA,
    'n_candidates': len(candidates),
    'candidates': [
        {
            'omega_rank': c['omega_rank'], 'phi_rank': c['phi_rank'],
            'phi_deg': c['phi_deg'], 'glint_cost': c['glint_cost'],
            'q0_err': c['q0_err'], 'w0_err': c['w0_err'],
            'w_mag_err': c['w_mag_err'], 'full_hifi': c['full_hifi'],
        }
        for c in candidates
    ],
    'final_q0_err': float(winner['q0_err']),
    'final_w_dir_err': float(winner['w0_err']),
    'final_w_mag_err_pct': float(winner['w_mag_err']),
    'final_w_vec_dps': np.rad2deg(winner['w0']).tolist(),
    'timing': {
        'phi_sweep_s': phi_time,
        'full_hifi_s': hifi_time,
        'total_s': total_time,
    },
}

out_path = RESULTS_DIR / 'm069_fixed_scoring.json'
save_results(str(out_path), results)
print(f"\nSaved: {out_path}")

if winner['q0_err'] < 5 and winner['w0_err'] < 5:
    print(f"\n*** SUCCESS ***")
elif winner['q0_err'] < 10 and winner['w0_err'] < 10:
    print(f"\n*** PARTIAL SUCCESS ***")
else:
    print(f"\n*** FAILED ***")
