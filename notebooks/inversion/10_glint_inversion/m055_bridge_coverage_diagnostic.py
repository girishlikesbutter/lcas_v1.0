#!/usr/bin/env python3
"""Micro-55 -- Bridge coverage diagnostic.

Question: Does increasing phi density from 36 (10°) to 72 (5°) improve
the coverage of the two-glint bridge omega pool?

For 10 trajectories × 2 phi densities:
  1. Generate all bridge omegas (vectorized)
  2. Filter to top 10K by omega magnitude closeness to peak-count estimate
  3. Compute omega direction error vs true omega (inertial frame at anchor)
  4. Report: min error, count within 5°/10°/20° in full pool and top-10K

Also checks: for the top-10 closest bridges, propagate back to t=0 and
verify that the inertial-frame error translates to similar body-frame error.

No LC scoring — pure geometry diagnostic.  Runtime: ~10 min.
"""

import sys, os, time, json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from numpy.polynomial import polynomial as P

from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"

PEAK_COEFFS = np.array([0.0417, 0.0397])
MAX_BRIDGE = 10000


# =========================================================================
# Helpers
# =========================================================================

def anchor_q_from_phi(phi, n_body, pab):
    """Quaternion on PAB-alignment circle: normal n_body aligned with pab,
    twisted by angle phi around n_body.  Returns (w, x, y, z)."""
    R0, _ = Rotation.align_vectors([n_body], [pab])
    q_xyzw = (Rotation.from_rotvec(phi * n_body) * R0).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def batch_anchor_quats(normals, phi_vals, pab_vec):
    """Pre-compute all anchor quaternions for all normals × phi values.
    Returns array of shape (n_normals * n_phi, 4) in wxyz convention,
    and corresponding (hyp_idx, phi) arrays."""
    n_norm = len(normals)
    n_phi = len(phi_vals)
    quats = np.zeros((n_norm * n_phi, 4))
    hyps = np.zeros(n_norm * n_phi, dtype=int)
    phis = np.zeros(n_norm * n_phi)
    for h in range(n_norm):
        for pi, phi in enumerate(phi_vals):
            idx = h * n_phi + pi
            quats[idx] = anchor_q_from_phi(phi, normals[h], pab_vec)
            hyps[idx] = h
            phis[idx] = phi
    return quats, hyps, phis


def vectorized_bridges(c1_wxyz, c2_wxyz, dt_ab, omega_est, n_wind):
    """Generate all bridge omegas from two sets of anchor quaternions.

    Returns:
      omega_all: (N_total, 3) array of bridge omega vectors (inertial frame)
      mag_frac_err: (N_total,) fractional magnitude error vs omega_est
      pair_idx: (N_total, 2) indices into c1/c2 for each bridge
      winding: (N_total,) winding number for each bridge
    """
    n1 = len(c1_wxyz)
    n2 = len(c2_wxyz)

    # Convert to xyzw for scipy
    c1_xyzw = c1_wxyz[:, [1, 2, 3, 0]]
    c2_xyzw = c2_wxyz[:, [1, 2, 3, 0]]

    R1 = Rotation.from_quat(c1_xyzw)
    R2 = Rotation.from_quat(c2_xyzw)

    # All n1 × n2 pairs
    pairs_i = np.repeat(np.arange(n1), n2)
    pairs_j = np.tile(np.arange(n2), n1)

    # Vectorized bridge rotation: R2[j] * R1[i].inv()
    R_bridge = R2[pairs_j] * R1[pairs_i].inv()
    rv_all = R_bridge.as_rotvec()  # (n1*n2, 3) — inertial frame
    omega_base = rv_all / dt_ab

    # Winding corrections
    norms = np.linalg.norm(rv_all, axis=1, keepdims=True)
    directions = rv_all / (norms + 1e-30)
    valid = norms.ravel() > 1e-15

    all_omegas = []
    all_pairs_i = []
    all_pairs_j = []
    all_windings = []

    for w in range(n_wind + 1):
        if w == 0:
            omega_w = omega_base
        else:
            omega_w = omega_base.copy()
            omega_w[valid] += directions[valid] * (2 * np.pi * w / dt_ab)

        all_omegas.append(omega_w)
        all_pairs_i.append(pairs_i)
        all_pairs_j.append(pairs_j)
        all_windings.append(np.full(len(pairs_i), w))

    omega_all = np.vstack(all_omegas)
    pairs_all = np.column_stack([np.concatenate(all_pairs_i),
                                  np.concatenate(all_pairs_j)])
    winding_all = np.concatenate(all_windings)

    # Magnitude fractional error
    omega_mag = np.rad2deg(np.linalg.norm(omega_all, axis=1))
    mag_frac_err = np.abs(omega_mag - omega_est) / (omega_est + 1e-30)

    return omega_all, mag_frac_err, pairs_all, winding_all


def direction_errors(omega_candidates, omega_true):
    """Compute direction error (degrees) between each candidate and truth."""
    dots = np.sum(omega_candidates * omega_true[None, :], axis=1)
    norms_c = np.linalg.norm(omega_candidates, axis=1)
    norm_t = np.linalg.norm(omega_true)
    cos_angles = dots / (norms_c * norm_t + 1e-30)
    return np.rad2deg(np.arccos(np.clip(cos_angles, -1, 1)))


# =========================================================================
# Load dataset
# =========================================================================

print("=" * 70)
print("m055 -- Bridge coverage diagnostic")
print("=" * 70)
t_global = time.time()

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
normals = master['unique_normals']
I_tensor = master['inertia_tensor']
q0s = master['q0s']
omega0s = master['omega0s']
omega_mags = master['omega_mags']
quaternions = master['quaternions']
mag_hifi = master['mag_hifi']
group_frac_flux = master['group_frac_flux']
n_normals = len(normals)

# Select 10 trajectories: the 5 from m054 + 5 more spanning omega range
m054_set = {84, 70, 35, 22, 19}
omega_sorted = np.argsort(omega_mags)
candidates = [int(idx) for idx in omega_sorted
              if np.sum(mag_hifi[idx][find_peaks(-mag_hifi[idx], distance=5,
                        prominence=0.3)[0]] < 9.0) >= 3]
# Add 5 evenly spaced from candidates not already in m054_set
extra = [c for c in candidates if c not in m054_set]
sel_idx = np.linspace(0, len(extra) - 1, 5, dtype=int)
extra_5 = [extra[i] for i in sel_idx]
TEST = sorted(list(m054_set) + extra_5)
print(f"Test trajectories ({len(TEST)}): {TEST}")
print(f"Omega: {[f'{omega_mags[t]:.3f}' for t in TEST]}")

# =========================================================================
# Run diagnostic for each phi density
# =========================================================================

PHI_CONFIGS = {
    36: "10° spacing (baseline)",
    72: "5° spacing (dense)",
}

all_results = []

for n_phi, label in PHI_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"PHI DENSITY: {n_phi} ({label})")
    print(f"{'='*60}")

    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    for traj_idx in TEST:
        t0 = time.time()
        mags = mag_hifi[traj_idx]
        omega_true_body = omega0s[traj_idx]
        omega_mag_true = float(omega_mags[traj_idx])

        # Peak detection + anchor selection (same logic as m053c/54)
        peaks, _ = find_peaks(-mags, distance=5, prominence=0.3)
        bright = peaks[mags[peaks] < 9.0]
        n_pk = len(peaks)

        # Confident peaks (>77% flux from one group)
        ff = group_frac_flux[traj_idx]
        labs = [int(np.argmax(ff[:, p])) for p in bright]
        confs = [float(ff[labs[i], bright[i]]) for i in range(len(bright))]
        conf_peaks = bright[[i for i, c in enumerate(confs) if c > 0.77]]
        if len(conf_peaks) < 2:
            print(f"  Traj {traj_idx}: SKIP (< 2 confident peaks)")
            continue

        sorted_by_mag = conf_peaks[np.argsort(mags[conf_peaks])]
        a1, a2 = int(sorted_by_mag[0]), int(sorted_by_mag[1])
        dt_ab = obs_times[a2] - obs_times[a1]
        if abs(dt_ab) < 10:
            print(f"  Traj {traj_idx}: SKIP (anchors too close)")
            continue

        omega_est = float(P.polyval(n_pk, PEAK_COEFFS))
        n_wind = int(omega_est * abs(dt_ab) / 360) + 2

        # True omega in INERTIAL frame at anchor 1
        _, omega_hist = propagate_attitude(
            q0s[traj_idx], omega0s[traj_idx], obs_times, "tumbling", I_tensor)
        omega_body_a1 = omega_hist[a1]
        q_true_a1 = quaternions[traj_idx, a1]
        R_true_a1 = Rotation.from_quat(
            [q_true_a1[1], q_true_a1[2], q_true_a1[3], q_true_a1[0]]).as_matrix()
        omega_inertial_true = R_true_a1 @ omega_body_a1

        # Oracle: which normals are responsible for the 2 anchors?
        oracle_g1 = int(np.argmax(ff[:, a1]))
        oracle_g2 = int(np.argmax(ff[:, a2]))

        # Generate bridges (vectorized)
        c1_wxyz, h1s, p1s = batch_anchor_quats(normals, phi_vals, pab_j2000[a1])
        c2_wxyz, h2s, p2s = batch_anchor_quats(normals, phi_vals, pab_j2000[a2])

        omega_all, mag_frac_err, pairs, windings = vectorized_bridges(
            c1_wxyz, c2_wxyz, dt_ab, omega_est, n_wind)

        n_total = len(omega_all)

        # Direction errors vs true omega (inertial frame)
        dir_errs = direction_errors(omega_all, omega_inertial_true)

        # Stats for FULL pool
        min_err_full = float(np.min(dir_errs))
        n5_full = int(np.sum(dir_errs < 5))
        n10_full = int(np.sum(dir_errs < 10))
        n20_full = int(np.sum(dir_errs < 20))

        # Magnitude filter → top 10K
        top_10k_idx = np.argsort(mag_frac_err)[:MAX_BRIDGE]
        dir_errs_10k = dir_errs[top_10k_idx]

        min_err_10k = float(np.min(dir_errs_10k))
        n5_10k = int(np.sum(dir_errs_10k < 5))
        n10_10k = int(np.sum(dir_errs_10k < 10))
        n20_10k = int(np.sum(dir_errs_10k < 20))

        # For the best bridge in the pool, check which normal pair it uses
        best_idx = int(np.argmin(dir_errs))
        best_pair = pairs[best_idx]
        best_h1 = int(h1s[best_pair[0]])
        best_h2 = int(h2s[best_pair[1]])
        best_w = int(windings[best_idx])

        # For the best bridge in top-10K, propagate back to t=0 and check
        # body-frame omega error
        best_10k_local = int(np.argmin(dir_errs_10k))
        best_10k_global = top_10k_idx[best_10k_local]
        best_omega_inertial = omega_all[best_10k_global]
        best_q1_wxyz = c1_wxyz[pairs[best_10k_global, 0]]

        try:
            bt = np.array([0., obs_times[a1]])
            qb, ob = propagate_attitude(
                best_q1_wxyz, -best_omega_inertial, bt, "tumbling", I_tensor)
            q0_cand = qb[-1]
            o0_cand = -ob[-1]
            # Body-frame omega error at t=0
            d = np.dot(o0_cand, omega_true_body)
            n = np.linalg.norm(o0_cand) * np.linalg.norm(omega_true_body)
            body_dir_err_t0 = float(np.rad2deg(
                np.arccos(np.clip(d / (n + 1e-30), -1, 1)))) if n > 1e-15 else 180.0
        except Exception as e:
            body_dir_err_t0 = -1.0

        dt = time.time() - t0

        print(f"  Traj {traj_idx} (ω={omega_mag_true:.3f}, est={omega_est:.3f}, "
              f"dt={dt_ab:.0f}s, nw={n_wind}):")
        print(f"    Anchors: a1={a1} (G{oracle_g1}), a2={a2} (G{oracle_g2})")
        print(f"    Pool: {n_total:,} bridges")
        print(f"    FULL POOL: min={min_err_full:.1f}°, "
              f"<5°={n5_full}, <10°={n10_full}, <20°={n20_full}")
        print(f"    TOP-10K:   min={min_err_10k:.1f}°, "
              f"<5°={n5_10k}, <10°={n10_10k}, <20°={n20_10k}")
        print(f"    Best: h1=G{best_h1}, h2=G{best_h2}, w={best_w}")
        print(f"    Best-10K → body err at t=0: {body_dir_err_t0:.1f}°")
        print(f"    ({dt:.1f}s)")

        all_results.append({
            'n_phi': n_phi,
            'traj_idx': int(traj_idx),
            'omega_dps': omega_mag_true,
            'omega_est': omega_est,
            'dt_ab': float(dt_ab),
            'n_wind': n_wind,
            'anchor_groups': [oracle_g1, oracle_g2],
            'n_bridges': n_total,
            'full_pool': {
                'min_err': min_err_full,
                'n_within_5': n5_full,
                'n_within_10': n10_full,
                'n_within_20': n20_full,
            },
            'top_10k': {
                'min_err': min_err_10k,
                'n_within_5': n5_10k,
                'n_within_10': n10_10k,
                'n_within_20': n20_10k,
            },
            'best_normals': [best_h1, best_h2],
            'best_winding': best_w,
            'body_err_t0': body_dir_err_t0,
            'runtime_s': dt,
        })


# =========================================================================
# Summary
# =========================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for n_phi in PHI_CONFIGS:
    subset = [r for r in all_results if r['n_phi'] == n_phi]
    if not subset:
        continue
    min_errs_full = [r['full_pool']['min_err'] for r in subset]
    min_errs_10k = [r['top_10k']['min_err'] for r in subset]
    has_5_full = sum(1 for r in subset if r['full_pool']['n_within_5'] > 0)
    has_5_10k = sum(1 for r in subset if r['top_10k']['n_within_5'] > 0)
    has_10_full = sum(1 for r in subset if r['full_pool']['n_within_10'] > 0)
    has_10_10k = sum(1 for r in subset if r['top_10k']['n_within_10'] > 0)

    print(f"\n  {n_phi} phi ({PHI_CONFIGS[n_phi]}):")
    print(f"    Full pool min error: "
          f"median={np.median(min_errs_full):.1f}°, "
          f"range=[{min(min_errs_full):.1f}°, {max(min_errs_full):.1f}°]")
    print(f"    Top-10K min error:   "
          f"median={np.median(min_errs_10k):.1f}°, "
          f"range=[{min(min_errs_10k):.1f}°, {max(min_errs_10k):.1f}°]")
    print(f"    Trajs with <5° in full pool: {has_5_full}/{len(subset)}")
    print(f"    Trajs with <5° in top-10K:   {has_5_10k}/{len(subset)}")
    print(f"    Trajs with <10° in full pool: {has_10_full}/{len(subset)}")
    print(f"    Trajs with <10° in top-10K:   {has_10_10k}/{len(subset)}")

# Per-trajectory comparison
print(f"\n  Per-trajectory comparison (min err in top-10K):")
print(f"  {'Traj':>5} {'ω dps':>6} {'36φ':>6} {'72φ':>6} {'Δ':>6}")
for traj_idx in TEST:
    r36 = next((r for r in all_results
                if r['traj_idx'] == traj_idx and r['n_phi'] == 36), None)
    r72 = next((r for r in all_results
                if r['traj_idx'] == traj_idx and r['n_phi'] == 72), None)
    if r36 and r72:
        e36 = r36['top_10k']['min_err']
        e72 = r72['top_10k']['min_err']
        print(f"  {traj_idx:5d} {r36['omega_dps']:6.3f} "
              f"{e36:6.1f} {e72:6.1f} {e36-e72:+6.1f}")


# =========================================================================
# Save
# =========================================================================

json_path = RESULTS_DIR / "m055_bridge_coverage.json"
with open(str(json_path), 'w') as f:
    json.dump({
        'experiment': 'm055_bridge_coverage_diagnostic',
        'phi_configs': {str(k): v for k, v in PHI_CONFIGS.items()},
        'max_bridge': MAX_BRIDGE,
        'test_trajectories': TEST,
        'results': all_results,
        'total_time_s': time.time() - t_global,
    }, f, indent=2,
    default=lambda x: float(x) if isinstance(x, np.floating)
    else int(x) if isinstance(x, np.integer) else x)

print(f"\nSaved: {json_path}")
print(f"Total: {time.time() - t_global:.0f}s")
