#!/usr/bin/env python3
"""
m079 — Normal sequence hypothesis test on seed 018.

Seed 018 is a hard failure case: only 1 tight constraint (the anchor, ±X),
5 non-anchor constraints all wide open (6-10 normals). The max-over-allowed
grid can't discriminate — best candidate is ~20° omega error.

Phase A (oracle): Fix the TRUE normals at all constraint epochs.
  → Does truth rank #1 in the grid? If yes, normal ambiguity is the blocker.

Phase B (blind enumeration): Hypothesize normals at the 2 medium constraints
  (ep 423 mag 6.45, ep 446 mag 6.79 — 6 options each = 36 hypotheses).
  Leave the 3 fully-open constraints as max-over-allowed.
  Score each hypothesis's grid winner with lo-fi peak matching.
  → Does the correct hypothesis separate cleanly?

Phase C (comparison): Also score the max-over-all baseline for reference.
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
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
CKPT_DIR = RESULTS_DIR / "m079_hypothesis_seed018"
CKPT_DIR.mkdir(exist_ok=True)

SEED = 18
N_DIRS = 2000
N_MAGS = 20
N_PHI = 36
N_WORKERS = 24
CONSTRAINT_WEIGHT = 10.0
Z_NORMALS = {4, 5}
LOFI_TOP = 10          # lo-fi candidates per hypothesis
PEAK_WINDOW = 3


# ── Helpers ──────────────────────────────────────────────────────────

def fibonacci_sphere(n):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([np.sin(phi)*np.cos(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(phi)])

def omega_dir_err(w1, w2):
    d1, d2 = w1/np.linalg.norm(w1), w2/np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    q_xyzw = (R_twist * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

def get_allowed_normals(mag):
    if mag < 5.9:
        return [0, 1]
    elif mag < 6.3:
        return [0, 1, 4, 5]
    elif mag < 7.3:
        return [0, 1, 2, 3, 4, 5]
    else:
        return list(range(10))

def propagate_delta_qs(omega_vec, dt_arr, I_tensor):
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    n = len(dt_arr)
    delta_qs = np.zeros((n, 4))
    fwd = dt_arr > 1e-6; bwd = dt_arr < -1e-6; zero = np.abs(dt_arr) < 1e-6
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


# ── Load data ────────────────────────────────────────────────────────

print("=" * 60)
print(f"m079 — Hypothesis test (seed {SEED})")
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

rng = np.random.default_rng(42 + SEED)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
omega_est_dps = 0.0397 * len(peaks_idx) + 0.0417
omega_est_rad = np.deg2rad(omega_est_dps)

spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]
anchor_idx = int(spec_peaks[np.argmin(observed_lc[spec_peaks])])
anchor_time = obs_times[anchor_idx]
anchor_mag = observed_lc[anchor_idx]
anchor_allowed = get_allowed_normals(anchor_mag)

non_anchor = spec_peaks[spec_peaks != anchor_idx]
dt_constraints = obs_times[non_anchor] - anchor_time
pab_at_constraints = pab_j2000[non_anchor]
constraint_mags = observed_lc[non_anchor]
constraint_allowed = [get_allowed_normals(m) for m in constraint_mags]
n_constraints = len(non_anchor)

# Oracle: propagate truth to anchor, get true omega at anchor
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

# Oracle: find true normals at each constraint epoch
quats_true, _ = propagate_attitude(true_q0, true_omega0, obs_times, 'tumbling', I_tensor)
true_normals_at_constraints = []
for ci in range(n_constraints):
    ep = non_anchor[ci]
    q = quats_true[ep]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    pb = R @ pab_j2000[ep]
    # Best-aligned normal from the full set
    best_ni = int(np.argmax(unique_normals @ pb))
    true_normals_at_constraints.append(best_ni)

print(f"Anchor: ep {anchor_idx}, mag {anchor_mag:.2f}, allowed: {[group_names[i] for i in anchor_allowed]}")
print(f"Constraints: {n_constraints}")
for ci in range(n_constraints):
    ep = non_anchor[ci]
    allowed = constraint_allowed[ci]
    true_ni = true_normals_at_constraints[ci]
    in_allowed = "YES" if true_ni in allowed else "NO"
    print(f"  ep {ep}: mag={constraint_mags[ci]:.2f}, "
          f"allowed={[group_names[i] for i in allowed]}, "
          f"true={group_names[true_ni]} (in allowed: {in_allowed})")
print(f"|omega| est: {omega_est_dps:.3f} deg/s (true: {true_omega_mag_dps:.3f})")


# ── Grid setup ───────────────────────────────────────────────────────

omega_dirs = fibonacci_sphere(N_DIRS)
omega_mags_search = omega_est_rad * np.linspace(0.70, 1.30, N_MAGS)

phi_coarse_xy = np.linspace(0, np.pi, N_PHI, endpoint=False)
phi_coarse_z = np.linspace(0, 2 * np.pi, 2 * N_PHI, endpoint=False)

qa_anchor_sets = []
for ni in anchor_allowed:
    phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
    qa = np.array([anchor_q_from_phi(p, unique_normals[ni], pab_j2000[anchor_idx])
                    for p in phi_arr])
    qa_anchor_sets.append((ni, qa[:, [1, 2, 3, 0]]))


# ══════════════════════════════════════════════════════════════════════
# PHASE A: Oracle grid — fix true normals at ALL constraints
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE A: Oracle — fix true normals at all constraints")
print(f"{'='*60}")

_oracle_normals = true_normals_at_constraints

def eval_direction_oracle(wi):
    """Grid search with oracle-fixed normals at every constraint."""
    wd = omega_dirs[wi]
    best_cost = np.inf
    best_omega = None
    for mag in omega_mags_search:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints, I_tensor)
        for ni, qa_xyzw in qa_anchor_sets:
            n_phi = len(qa_xyzw)
            R_anchors = Rotation.from_quat(qa_xyzw)
            cost = np.zeros(n_phi)
            for ci in range(n_constraints):
                dq = dqs[ci]
                R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
                R_all = R_anchors * R_delta
                pbs = R_all.apply(pab_at_constraints[ci])
                # Fixed normal — single dot product
                bds = pbs @ unique_normals[_oracle_normals[ci]]
                cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2
            mc = cost.min()
            if mc < best_cost:
                best_cost = mc
                best_omega = omega_test.copy()
    return best_cost, best_omega

t0 = time.time()
with Pool(N_WORKERS) as pool:
    oracle_results = pool.map(eval_direction_oracle, range(N_DIRS))
print(f"Oracle grid done in {time.time() - t0:.0f}s")

oracle_costs = np.array([r[0] for r in oracle_results])
oracle_omegas = np.array([r[1] for r in oracle_results])
sorted_oracle = np.argsort(oracle_costs)

print(f"\nOracle grid top-10:")
for i in range(min(10, len(sorted_oracle))):
    ri = sorted_oracle[i]
    w_err = omega_dir_err(oracle_omegas[ri], true_omega_anchor)
    w_mag = np.rad2deg(np.linalg.norm(oracle_omegas[ri]))
    w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100
    tag = " <--" if w_err < 10 else ""
    print(f"  #{i+1}: cost={oracle_costs[ri]:.6f} | "
          f"w_dir={w_err:.1f}° w_mag={w_mag_err:+.1f}%{tag}")

# Where does truth rank?
dir_errors = np.array([omega_dir_err(oracle_omegas[i], true_omega_anchor)
                        for i in range(N_DIRS)])
closest = int(np.argmin(dir_errors))
truth_rank = int(np.where(sorted_oracle == closest)[0][0]) + 1
print(f"\nClosest to truth: dir #{closest}, err={dir_errors[closest]:.1f}°, "
      f"rank=#{truth_rank}/{N_DIRS}")


# ══════════════════════════════════════════════════════════════════════
# PHASE B: Blind hypothesis enumeration
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE B: Blind hypothesis enumeration")
print(f"{'='*60}")

# Identify medium constraints (< 7.3 mag, 6 normals) for hypothesis fixing.
# Leave fully-open constraints (>= 7.3, 10 normals) as max-over-allowed.
HYP_MAG_THRESHOLD = 7.3
hyp_mask = constraint_mags < HYP_MAG_THRESHOLD
hyp_indices = np.where(hyp_mask)[0]
open_indices = np.where(~hyp_mask)[0]

hyp_options = [constraint_allowed[i] for i in hyp_indices]
open_allowed = [constraint_allowed[i] for i in open_indices]
hypotheses = list(product(*hyp_options))
n_hyp = len(hypotheses)

print(f"Hypothesis constraints (mag < {HYP_MAG_THRESHOLD}):")
for i, hi in enumerate(hyp_indices):
    ep = non_anchor[hi]
    opts = [group_names[ni] for ni in constraint_allowed[hi]]
    true_ni = true_normals_at_constraints[hi]
    print(f"  ep {ep}: mag={constraint_mags[hi]:.2f} -> {opts} (true: {group_names[true_ni]})")
print(f"Open constraints (max-over-allowed): {len(open_indices)}")
for oi in open_indices:
    ep = non_anchor[oi]
    print(f"  ep {ep}: mag={constraint_mags[oi]:.2f} -> {len(constraint_allowed[oi])} normals")
print(f"Total hypotheses: {n_hyp}")

# Grid search scoring all hypotheses simultaneously.
# For each omega direction: propagate once, score under all hypotheses.
_hyp_indices = hyp_indices
_open_indices = open_indices
_open_allowed = open_allowed
_hypotheses = hypotheses

def eval_direction_all_hyps(wi):
    """For one omega direction, evaluate all hypotheses."""
    wd = omega_dirs[wi]
    best_costs = [np.inf] * n_hyp
    best_omegas = [None] * n_hyp

    for mag in omega_mags_search:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints, I_tensor)

        for anchor_ni, qa_xyzw in qa_anchor_sets:
            n_phi = len(qa_xyzw)
            R_anchors = Rotation.from_quat(qa_xyzw)

            # Pre-compute PAB in body frame at each constraint for all phis
            pbs_all = []
            for ci in range(n_constraints):
                dq = dqs[ci]
                R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
                R_all = R_anchors * R_delta
                pbs_all.append(R_all.apply(pab_at_constraints[ci]))

            # Open-constraint cost (shared across all hypotheses)
            open_cost = np.zeros(n_phi)
            for k, ci in enumerate(_open_indices):
                allowed = _open_allowed[k]
                bds = (pbs_all[ci] @ unique_normals[allowed].T).max(axis=1)
                open_cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2

            # Score each hypothesis (only hyp constraints differ)
            for hyp_idx, hyp in enumerate(_hypotheses):
                hyp_cost = np.zeros(n_phi)
                for k, ci in enumerate(_hyp_indices):
                    ni = hyp[k]
                    bds = pbs_all[ci] @ unique_normals[ni]
                    hyp_cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2

                total = hyp_cost + open_cost
                mc = total.min()
                if mc < best_costs[hyp_idx]:
                    best_costs[hyp_idx] = mc
                    best_omegas[hyp_idx] = omega_test.copy()

    return best_costs, best_omegas

# Also run max-over-all baseline in same pass
def eval_direction_baseline(wi):
    """Standard max-over-allowed at all constraints (baseline)."""
    wd = omega_dirs[wi]
    best_cost = np.inf
    best_omega = None
    best_ni = -1
    best_phi_idx = -1
    for mag in omega_mags_search:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints, I_tensor)
        for ni, qa_xyzw in qa_anchor_sets:
            n_phi = len(qa_xyzw)
            R_anchors = Rotation.from_quat(qa_xyzw)
            cost = np.zeros(n_phi)
            for ci in range(n_constraints):
                dq = dqs[ci]
                R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
                R_all = R_anchors * R_delta
                pbs = R_all.apply(pab_at_constraints[ci])
                allowed = constraint_allowed[ci]
                bds = (pbs @ unique_normals[allowed].T).max(axis=1)
                cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2
            bi = int(np.argmin(cost))
            if cost[bi] < best_cost:
                best_cost = cost[bi]
                best_omega = omega_test.copy()
                best_ni = ni
                best_phi_idx = bi
    return best_cost, best_omega, best_ni, best_phi_idx

print(f"\n--- Hypothesis grid ({N_DIRS} dirs x {N_MAGS} mags, {n_hyp} hypotheses) ---")
t0 = time.time()
with Pool(N_WORKERS) as pool:
    hyp_results = pool.map(eval_direction_all_hyps, range(N_DIRS))
hyp_time = time.time() - t0
print(f"Hypothesis grid done in {hyp_time:.0f}s")

print(f"\n--- Baseline grid (max-over-allowed) ---")
t0 = time.time()
with Pool(N_WORKERS) as pool:
    base_results = pool.map(eval_direction_baseline, range(N_DIRS))
base_time = time.time() - t0
print(f"Baseline grid done in {base_time:.0f}s")


# ── Analyze each hypothesis ──────────────────────────────────────────

print(f"\n{'='*60}")
print("PHASE B RESULTS: Per-hypothesis grid analysis")
print(f"{'='*60}")

# Identify the true hypothesis
true_hyp = tuple(true_normals_at_constraints[i] for i in hyp_indices)
true_hyp_names = [group_names[ni] for ni in true_hyp]

hyp_summaries = []
for hyp_idx, hyp in enumerate(hypotheses):
    hyp_names = [group_names[ni] for ni in hyp]
    is_true = (hyp == true_hyp)

    grid_costs = np.array([r[0][hyp_idx] for r in hyp_results])
    grid_omegas = np.array([r[1][hyp_idx] for r in hyp_results])
    sorted_idx = np.argsort(grid_costs)

    dir_errors = np.array([omega_dir_err(grid_omegas[i], true_omega_anchor)
                           for i in range(N_DIRS)])
    closest = int(np.argmin(dir_errors))
    truth_rank = int(np.where(sorted_idx == closest)[0][0]) + 1

    winner_idx = sorted_idx[0]
    winner_err = dir_errors[winner_idx]
    winner_mag = np.rad2deg(np.linalg.norm(grid_omegas[winner_idx]))
    winner_mag_err = (winner_mag - true_omega_mag_dps) / true_omega_mag_dps * 100
    best_in_top10 = dir_errors[sorted_idx[:10]].min()

    hyp_summaries.append({
        'hyp_idx': hyp_idx, 'hyp': hyp, 'hyp_names': hyp_names,
        'is_true': is_true,
        'truth_rank': truth_rank,
        'winner_dir_err': winner_err,
        'winner_mag_err': winner_mag_err,
        'winner_cost': grid_costs[winner_idx],
        'best_top10_err': best_in_top10,
        'winner_omega': grid_omegas[winner_idx],
    })

# Also compute baseline
base_costs = np.array([r[0] for r in base_results])
base_omegas = np.array([r[1] for r in base_results])
base_sorted = np.argsort(base_costs)
base_dir_errors = np.array([omega_dir_err(base_omegas[i], true_omega_anchor)
                             for i in range(N_DIRS)])
base_closest = int(np.argmin(base_dir_errors))
base_truth_rank = int(np.where(base_sorted == base_closest)[0][0]) + 1
base_winner = base_sorted[0]
base_winner_err = base_dir_errors[base_winner]
base_winner_mag = np.rad2deg(np.linalg.norm(base_omegas[base_winner]))
base_winner_mag_err = (base_winner_mag - true_omega_mag_dps) / true_omega_mag_dps * 100

print(f"\nBaseline (max-over-all): winner w_dir={base_winner_err:.1f}° "
      f"w_mag={base_winner_mag_err:+.1f}% | truth rank #{base_truth_rank}")

# Sort hypotheses by winner omega error
hyp_summaries.sort(key=lambda x: x['winner_dir_err'])

print(f"\nAll {n_hyp} hypotheses sorted by winner omega dir error:")
for r in hyp_summaries:
    tag = " <-- TRUE" if r['is_true'] else ""
    print(f"  Hyp {r['hyp_idx']+1:2d}: {r['hyp_names']} | "
          f"winner w_dir={r['winner_dir_err']:.1f}° "
          f"w_mag={r['winner_mag_err']:+.1f}% "
          f"top10_best={r['best_top10_err']:.1f}° "
          f"truth_rank=#{r['truth_rank']}{tag}")


# ══════════════════════════════════════════════════════════════════════
# PHASE C: Lo-fi scoring of top hypotheses
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE C: Lo-fi peak matching for top hypotheses")
print(f"{'='*60}")

# Take the top 10 hypotheses by winner error, plus the true hypothesis,
# and the baseline. Generate lo-fi LC for each winner and score.
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

_satellite = CTX.satellite
_obs_times = obs_times
_obs_lc = observed_lc
_sun = CTX.sun_pos
_obs = CTX.obs_pos
_sat = CTX.sat_pos
_dist = CTX.obs_dist
_art = CTX.art_matrices
_I = I_tensor
_obs_peaks = peaks_idx

def eval_lofi(args):
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

# For each hypothesis winner, back-propagate to t=0 and evaluate lo-fi
lofi_test_hyps = hyp_summaries[:10]
# Ensure true hypothesis is included
if not any(h['is_true'] for h in lofi_test_hyps):
    true_entry = [h for h in hyp_summaries if h['is_true']][0]
    lofi_test_hyps.append(true_entry)

lofi_args = []
for i, h in enumerate(lofi_test_hyps):
    omega_winner = h['winner_omega']
    # Find best phi for this hypothesis winner
    dqs = propagate_delta_qs(omega_winner, dt_constraints, I_tensor)
    best_cost = np.inf
    best_qa = None
    for ni, qa_xyzw in qa_anchor_sets:
        n_phi = len(qa_xyzw)
        R_anchors = Rotation.from_quat(qa_xyzw)
        cost = np.zeros(n_phi)
        for ci in range(n_constraints):
            dq = dqs[ci]
            R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
            R_all = R_anchors * R_delta
            pbs = R_all.apply(pab_at_constraints[ci])
            # Use hypothesis normals at hyp constraints, max at open
            if ci in hyp_indices:
                k = list(hyp_indices).index(ci)
                bds_val = pbs @ unique_normals[h['hyp'][k]]
            else:
                allowed = constraint_allowed[ci]
                bds_val = (pbs @ unique_normals[allowed].T).max(axis=1)
            cost += CONSTRAINT_WEIGHT * (1.0 - bds_val) ** 2
        bi = int(np.argmin(cost))
        if cost[bi] < best_cost:
            best_cost = cost[bi]
            phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
            best_qa = anchor_q_from_phi(phi_arr[bi], unique_normals[ni],
                                         pab_j2000[anchor_idx])

    # Back-propagate to t=0
    bt = np.array([0.0, anchor_time])
    qb, ob = propagate_attitude(best_qa, -omega_winner, bt, "tumbling", I_tensor)
    q0_cand = qb[-1]
    w0_cand = -ob[-1]
    lofi_args.append((i, q0_cand, w0_cand))
    lofi_test_hyps[i]['q0'] = q0_cand
    lofi_test_hyps[i]['w0'] = w0_cand

# Also add baseline winner
base_winner_omega = base_omegas[base_winner]
dqs = propagate_delta_qs(base_winner_omega, dt_constraints, I_tensor)
best_cost = np.inf
best_qa = None
for ni, qa_xyzw in qa_anchor_sets:
    n_phi = len(qa_xyzw)
    R_anchors = Rotation.from_quat(qa_xyzw)
    cost = np.zeros(n_phi)
    for ci in range(n_constraints):
        dq = dqs[ci]
        R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
        R_all = R_anchors * R_delta
        pbs = R_all.apply(pab_at_constraints[ci])
        allowed = constraint_allowed[ci]
        bds_val = (pbs @ unique_normals[allowed].T).max(axis=1)
        cost += CONSTRAINT_WEIGHT * (1.0 - bds_val) ** 2
    bi = int(np.argmin(cost))
    if cost[bi] < best_cost:
        best_cost = cost[bi]
        phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
        best_qa = anchor_q_from_phi(phi_arr[bi], unique_normals[ni],
                                     pab_j2000[anchor_idx])
bt = np.array([0.0, anchor_time])
qb, ob = propagate_attitude(best_qa, -base_winner_omega, bt, "tumbling", I_tensor)
base_q0 = qb[-1]
base_w0 = -ob[-1]
base_idx = len(lofi_args)
lofi_args.append((base_idx, base_q0, base_w0))

print(f"\nEvaluating {len(lofi_args)} candidates with lo-fi...")
t0 = time.time()
with Pool(min(N_WORKERS, len(lofi_args))) as pool:
    lofi_results = pool.map(eval_lofi, lofi_args)
print(f"Lo-fi done in {time.time() - t0:.1f}s")

print(f"\nLo-fi results (sorted by peak match, then MSE):")
lofi_out = []
for idx, n_matched, mse in lofi_results:
    if idx < len(lofi_test_hyps):
        h = lofi_test_hyps[idx]
        q0_err = attitude_error_deg(h['q0'], true_q0)
        w_dir_err = omega_dir_err(h['w0'], true_omega0)
        w_mag = np.rad2deg(np.linalg.norm(h['w0']))
        w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100
        label = f"Hyp {h['hyp_names']}"
        if h['is_true']:
            label += " [TRUE]"
        lofi_out.append((n_matched, mse, label, q0_err, w_dir_err, w_mag_err))
    else:
        q0_err = attitude_error_deg(base_q0, true_q0)
        w_dir_err = omega_dir_err(base_w0, true_omega0)
        w_mag = np.rad2deg(np.linalg.norm(base_w0))
        w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100
        lofi_out.append((n_matched, mse, "BASELINE (max-over-all)", q0_err, w_dir_err, w_mag_err))

lofi_out.sort(key=lambda x: (-x[0], x[1]))
for matched, mse, label, q0e, wde, wme in lofi_out:
    tag = " <--" if wde < 10 else ""
    print(f"  {label}: matched={matched}/{len(peaks_idx)} mse={mse:.3f} | "
          f"q0={q0e:.1f}° w_dir={wde:.1f}° w_mag={wme:+.1f}%{tag}")


# ── Save ─────────────────────────────────────────────────────────────
total_time = time.time() - t_global
print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

np.savez(str(CKPT_DIR / "results.npz"),
         oracle_costs=oracle_costs, oracle_omegas=oracle_omegas,
         base_costs=base_costs, base_omegas=base_omegas,
         true_omega_anchor=true_omega_anchor,
         true_normals_at_constraints=np.array(true_normals_at_constraints),
         allow_pickle=True)

print(f"Saved to {CKPT_DIR}/")
