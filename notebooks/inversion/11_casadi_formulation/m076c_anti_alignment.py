#!/usr/bin/env python3
"""
m076c — Anti-alignment penalty test.

Take the top 20,000 candidates from the m076 scoring table (by current cost).
For each, propagate to all dim epochs (mag > 10) and compute an anti-alignment
penalty: if any normal is well-aligned with the PAB at a dim epoch, that's a
false glint prediction — penalize it.

New cost = W_peak * Σ(peak) (1 - best_dot)²
         + W_anti * Σ(dim)  max(0, best_dot - threshold)²

Test multiple W_anti values and thresholds. Report rank of best good candidate
under each combination.

Uses the stored 72M table for peak costs, only recomputes the anti-alignment
at dim epochs for the top 20K subset.
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
TABLE_DIR = RESULTS_DIR / "m076_scoring_table"
OUT_DIR = RESULTS_DIR / "m076c_anti_alignment"
OUT_DIR.mkdir(exist_ok=True)

SEED = 27
TOP_N = 20000
N_WORKERS = 24
DIM_THRESHOLD_MAG = 10.0  # epochs dimmer than this are "dim"


# ── Load data ──────────────────────────────────────────────────────────

print("=" * 60)
print(f"m076c — Anti-alignment penalty test (seed {SEED})")
print("=" * 60)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][SEED]
true_omega0 = master['omega0s'][SEED]
n_normals = len(unique_normals)
group_names = list(master['group_names'])

rng = np.random.default_rng(42)
observed_lc = master['mag_hifi'][SEED] + rng.normal(0, 0.05, 500)

# Identify spec peaks and dim epochs
peaks_idx, _ = find_peaks(-observed_lc, distance=5, prominence=0.3)
spec_peaks = peaks_idx[observed_lc[peaks_idx] < 9.0]
anchor_idx = int(spec_peaks[np.argmin(observed_lc[spec_peaks])])
anchor_time = obs_times[anchor_idx]

dim_mask = observed_lc > DIM_THRESHOLD_MAG
dim_epochs = np.where(dim_mask)[0]
n_dim = len(dim_epochs)
dt_dim = obs_times[dim_epochs] - anchor_time
pab_dim = pab_j2000[dim_epochs]

print(f"Anchor: ep {anchor_idx}")
print(f"Spec peaks: {len(spec_peaks)}, dim epochs (mag > {DIM_THRESHOLD_MAG}): {n_dim}")

# Load the 72M table and oracle errors
print("Loading scoring table...")
table = np.load(str(TABLE_DIR / "table.npz"))
all_costs = table['costs']
all_omega_vecs = table['omega_vecs']
all_normal_idxs = table['normal_idxs']
all_phi_idxs = table['phi_idxs']
all_q0s = table['q0s']

oracle = np.load(str(TABLE_DIR / "oracle_errors.npz"))
q0_errors = oracle['q0_errors']
w_dir_errors = oracle['w_dir_errors']

n_total = len(all_costs)
good_mask = (q0_errors < 20) & (w_dir_errors < 5)

# Take top N by current cost
sorted_idx = np.argsort(all_costs)
top_idx = sorted_idx[:TOP_N]
top_costs = all_costs[top_idx]
top_omegas = all_omega_vecs[top_idx]
top_q0s = all_q0s[top_idx]
top_q0_errors = q0_errors[top_idx]
top_w_errors = w_dir_errors[top_idx]
top_good = good_mask[top_idx]
top_normals = all_normal_idxs[top_idx]

n_good_in_top = top_good.sum()
print(f"Top {TOP_N}: {n_good_in_top} good candidates")
if n_good_in_top > 0:
    first_good_pos = int(np.where(top_good)[0][0])
    print(f"  First good at position #{first_good_pos + 1} (cost {top_costs[first_good_pos]:.6f})")


# ── Helper: anchor q from stored data ──────────────────────────────────

def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    q_xyzw = (R_twist * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

phi_xy = np.linspace(0, np.pi, 100, endpoint=False)
phi_z = np.linspace(0, 2 * np.pi, 200, endpoint=False)
z_normals = {4, 5}


# ── Compute anti-alignment at dim epochs for top N ────────────────────

ckpt = OUT_DIR / "anti_alignment.npz"

if ckpt.exists():
    print("Loading anti-alignment checkpoint...")
    aa = np.load(str(ckpt))
    dim_best_dots = aa['dim_best_dots']
else:
    print(f"Computing anti-alignment for top {TOP_N} candidates at {n_dim} dim epochs...")

    # For each candidate, we need the attitude at each dim epoch.
    # Candidate is defined by (omega_vec, anchor_normal, phi).
    # We reconstruct the anchor quaternion, then propagate delta-qs to dim epochs.

    # Pre-compute anchor quaternions for all candidates
    # (we need to reconstruct from normal_idx and phi_idx)
    phi_arrays = {}
    for ni in range(n_normals):
        phi_arrays[ni] = phi_z if ni in z_normals else phi_xy

    def propagate_delta_qs_batch(omega_vec, dt_arr):
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

    # Group candidates by omega vector to avoid redundant propagation
    # Candidates with the same omega only differ in (normal, phi) at anchor
    # The delta-qs at dim epochs depend only on omega, not on anchor choice
    # So we propagate once per unique omega, then sweep anchor for all candidates

    # Find unique omega vectors in top N
    # (candidates from the same grid direction + magnitude share omega)
    omega_hash = np.round(top_omegas * 1e6).astype(np.int64)
    _, unique_inv, unique_counts = np.unique(
        omega_hash, axis=0, return_inverse=True, return_counts=True)
    n_unique_omega = len(unique_counts)
    print(f"  {n_unique_omega} unique omega vectors in top {TOP_N}")

    # For each unique omega, propagate delta-qs to dim epochs
    # Then for each candidate with that omega, compute anchor attitude and dot products
    dim_best_dots = np.zeros((TOP_N, n_dim), dtype=np.float32)

    _dt_dim = dt_dim
    _pab_dim = pab_dim
    _normals = unique_normals
    _n_dim = n_dim

    def eval_omega_group(args):
        group_idx, omega_vec, cand_indices = args
        dqs = propagate_delta_qs_batch(omega_vec, _dt_dim)

        results = []
        for ci in cand_indices:
            ni = int(top_normals[ci])
            pi = int(all_phi_idxs[top_idx[ci]])
            phi_val = phi_arrays[ni][pi]
            qa_wxyz = anchor_q_from_phi(phi_val, _normals[ni], pab_j2000[anchor_idx])
            qa_xyzw = np.array([qa_wxyz[1], qa_wxyz[2], qa_wxyz[3], qa_wxyz[0]])
            R_anchor = Rotation.from_quat(qa_xyzw)

            best_dots = np.zeros(_n_dim)
            for di in range(_n_dim):
                dq = dqs[di]
                R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
                R_epoch = R_anchor * R_delta
                pb = R_epoch.apply(_pab_dim[di])
                best_dots[di] = max(pb @ _normals.T)

            results.append((ci, best_dots))
        return results

    # Build groups
    groups = {}
    for ci in range(TOP_N):
        gi = unique_inv[ci]
        if gi not in groups:
            groups[gi] = {'omega': top_omegas[ci], 'cands': []}
        groups[gi]['cands'].append(ci)

    group_args = [(gi, g['omega'], g['cands']) for gi, g in groups.items()]

    t0 = time.time()
    # Process in batches for progress
    BATCH = 200
    n_done = 0
    for batch_start in range(0, len(group_args), BATCH):
        batch = group_args[batch_start:batch_start + BATCH]
        with Pool(N_WORKERS) as pool:
            batch_results = pool.map(eval_omega_group, batch)
        for group_results in batch_results:
            for ci, best_dots in group_results:
                dim_best_dots[ci] = best_dots
        n_done += len(batch)
        elapsed = time.time() - t0
        print(f"  {n_done}/{len(group_args)} omega groups done ({elapsed:.0f}s)", flush=True)

    np.savez_compressed(str(ckpt), dim_best_dots=dim_best_dots)
    print(f"Anti-alignment computed in {time.time() - t0:.0f}s, saved checkpoint")


# ══════════════════════════════════════════════════════════════════════
# SCORING EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SCORING EXPERIMENTS")
print(f"{'='*60}")

# For reference: oracle check on truth
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

# Compute anti-alignment penalty for various thresholds and weights
# anti_penalty = Σ(dim epochs) max(0, best_dot - threshold)²
dot_thresholds = [0.90, 0.95, 0.98, 0.99, 0.995]
w_anti_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

print(f"\nBaseline: first good candidate at rank #{int(np.where(top_good)[0][0])+1}/{TOP_N}")

best_rank = TOP_N
best_config = ""

for dot_thresh in dot_thresholds:
    # Compute anti-penalty for all top N
    excess = np.maximum(0, dim_best_dots - dot_thresh)
    anti_penalty = np.sum(excess ** 2, axis=1)

    for w_anti in w_anti_values:
        combined = top_costs + w_anti * anti_penalty
        combined_sorted = np.argsort(combined)
        combined_good = top_good[combined_sorted]

        if combined_good.any():
            first_good_rank = int(np.where(combined_good)[0][0]) + 1
        else:
            first_good_rank = TOP_N

        if first_good_rank < best_rank:
            best_rank = first_good_rank
            best_config = f"thresh={dot_thresh}, w_anti={w_anti}"

        # Only print interesting ones
        if first_good_rank < 1000:
            gi = combined_sorted[np.where(combined_good)[0][0]]
            print(f"  thresh={dot_thresh:.3f} w={w_anti:5.1f}: "
                  f"good at #{first_good_rank:>5d}  "
                  f"q0={top_q0_errors[gi]:.1f}° w={top_w_errors[gi]:.1f}° "
                  f"peak_cost={top_costs[gi]:.4f} anti={anti_penalty[gi]:.4f}")

print(f"\nBest config: {best_config} -> rank #{best_rank}")

# Detailed view of the best config
if best_rank < TOP_N:
    dt, wa = float(best_config.split(',')[0].split('=')[1]), float(best_config.split(',')[1].split('=')[1])
    excess = np.maximum(0, dim_best_dots - dt)
    anti_penalty = np.sum(excess ** 2, axis=1)
    combined = top_costs + wa * anti_penalty
    combined_sorted = np.argsort(combined)

    print(f"\nTop 20 under best config ({best_config}):")
    for i in range(20):
        ci = combined_sorted[i]
        marker = " <-- GOOD" if top_good[ci] else ""
        print(f"  #{i+1}: combined={combined[ci]:.6f} "
              f"(peak={top_costs[ci]:.6f} + anti={wa * anti_penalty[ci]:.6f}) "
              f"q0={top_q0_errors[ci]:.1f}° w={top_w_errors[ci]:.1f}° "
              f"normal={group_names[top_normals[ci]]}{marker}")

# Also check: what does anti-penalty look like for good vs bad candidates?
if n_good_in_top > 0:
    print(f"\nAnti-penalty statistics (threshold=0.95):")
    excess_95 = np.maximum(0, dim_best_dots - 0.95)
    anti_95 = np.sum(excess_95 ** 2, axis=1)

    good_anti = anti_95[top_good]
    bad_anti = anti_95[~top_good]
    print(f"  Good candidates: mean={good_anti.mean():.4f} median={np.median(good_anti):.4f} "
          f"min={good_anti.min():.4f} max={good_anti.max():.4f}")
    print(f"  Bad candidates:  mean={bad_anti.mean():.4f} median={np.median(bad_anti):.4f} "
          f"min={bad_anti.min():.4f} max={bad_anti.max():.4f}")

    # How many dim epochs have alignment > 0.95 for good vs bad?
    n_aligned_good = np.mean(dim_best_dots[top_good] > 0.95, axis=1)
    n_aligned_bad = np.mean(dim_best_dots[~top_good] > 0.95, axis=1)
    print(f"\n  Fraction of dim epochs with alignment > 0.95:")
    print(f"    Good: mean={n_aligned_good.mean():.3f} ({n_aligned_good.mean()*n_dim:.0f}/{n_dim} epochs)")
    print(f"    Bad:  mean={n_aligned_bad.mean():.3f} ({n_aligned_bad.mean()*n_dim:.0f}/{n_dim} epochs)")

np.savez(str(OUT_DIR / "results.npz"),
    top_costs=top_costs, anti_penalties_95=anti_95 if n_good_in_top > 0 else np.array([]),
    top_q0_errors=top_q0_errors, top_w_errors=top_w_errors, top_good=top_good)

print(f"\nTotal time: {time.time() - t_global:.0f}s")
print(f"Results: {OUT_DIR}/")
