#!/usr/bin/env python3
"""
m080 — Brightness-match cost function test on seed 018.

The alignment cost (1-dot)^2 is degenerate for IS-901 because +X, +WD, +ED
normals are only 15° apart (dot=0.966). When one aligns with the PAB, the
others co-align and score nearly identically.

The fix: use predicted brightness, not just alignment angle. Different normals
produce very different peak magnitudes due to area differences:
  ±X:  mag 4.93 at perfect alignment (large area: bus + panels)
  ±Z:  mag 5.93 (medium: bus face)
  ±Y:  mag 6.22 (medium: bus face)
  ±WD/±ED: mag 7.43 (small: dishes)

A brightness-match cost (predicted_mag - observed_mag)^2 naturally
discriminates normals that the alignment cost cannot.

Test: compare alignment-cost grid vs brightness-cost grid on seed 018.
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
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
CKPT_DIR = RESULTS_DIR / "m080_brightness_cost"
CKPT_DIR.mkdir(exist_ok=True)

SEED = 18
N_DIRS = 2000
N_MAGS = 20
N_PHI = 36
N_WORKERS = 24
Z_NORMALS = {4, 5}
PEAK_WINDOW = 3
LOFI_TOP = 20


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


# ── Build brightness model ───────────────────────────────────────────

print("=" * 60)
print(f"m080 — Brightness-match cost (seed {SEED})")
print("=" * 60)
t_global = time.time()

# Load trajectory data
master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
pab_j2000 = master['pab_j2000']
unique_normals = master['unique_normals']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][SEED]
true_omega0 = master['omega0s'][SEED]
true_omega_mag_dps = float(master['omega_mags'][SEED])
true_lc = master['mag_hifi'][SEED]
group_names = list(master['group_names'])
n_normals = len(unique_normals)

rng = np.random.default_rng(42 + SEED)
observed_lc = true_lc + rng.normal(0, 0.05, len(true_lc))

# Load satellite model for brightness calibration
print("Loading satellite model...", flush=True)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

# Calibrate: for each normal, compute lo-fi brightness at a grid of alignment
# angles (0°, 2°, 5°, 10°, 15°, 20°, 30°). Build interpolation tables.
print("Calibrating brightness model...", flush=True)

ref_ep = 250
sun_vec = CTX.sun_pos[ref_ep] - CTX.sat_pos[ref_ep]
sun_vec /= np.linalg.norm(sun_vec)
obs_vec = CTX.obs_pos[ref_ep] - CTX.sat_pos[ref_ep]
obs_vec /= np.linalg.norm(obs_vec)
ref_dist = CTX.obs_dist[ref_ep]
pab_ref = pab_j2000[ref_ep]

calib_angles_deg = np.array([0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60])
# mag_table[ni, ai] = magnitude for normal ni at misalignment calib_angles_deg[ai]
mag_table = np.zeros((n_normals, len(calib_angles_deg)))

for ni in range(n_normals):
    un = unique_normals[ni]
    perp = np.cross(un, np.array([0, 0, 1]))
    if np.linalg.norm(perp) < 0.1:
        perp = np.cross(un, np.array([0, 1, 0]))
    perp /= np.linalg.norm(perp)

    for ai, angle_deg in enumerate(calib_angles_deg):
        R_align, _ = Rotation.align_vectors([un], [pab_ref])
        if angle_deg > 0:
            R_off = Rotation.from_rotvec(np.deg2rad(angle_deg) * perp)
            R_align = R_off * R_align
        R_mat = R_align.as_matrix()
        k1 = (R_mat @ sun_vec).reshape(1, 3)
        k2 = (R_mat @ obs_vec).reshape(1, 3)

        lit = create_no_shadow_lit_status(CTX.satellite, 1)
        pred_mags, _, _, _, _, _ = generate_lightcurves(
            facet_lit_status_dict=lit, k1_vectors_array=k1,
            k2_vectors_array=k2, observer_distances=np.array([ref_dist]),
            satellite=CTX.satellite, epochs=np.array([0.0]),
            pre_computed_matrices=CTX.art_matrices, show_progress=False)
        mag_table[ni, ai] = pred_mags[0]

# Convert calibration to: dot value -> magnitude (for interpolation)
calib_dots = np.cos(np.deg2rad(calib_angles_deg))  # [1.0, 0.9998, 0.9994, ...]

print("Brightness model calibrated:")
print(f"  {'Normal':>6} | perfect  5°off  15°off  30°off")
for ni in range(n_normals):
    print(f"  {group_names[ni]:>6} | {mag_table[ni,0]:6.2f}  "
          f"{mag_table[ni,4]:6.2f}  {mag_table[ni,7]:6.2f}  {mag_table[ni,9]:6.2f}")


def predict_mag(ni, dot_val):
    """Interpolate calibrated magnitude for normal ni at alignment dot."""
    return float(np.interp(dot_val, calib_dots[::-1], mag_table[ni, ::-1]))


# ── Setup constraints ────────────────────────────────────────────────

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

# Oracle
_, w_hist = propagate_attitude(true_q0, true_omega0,
    np.array([0.0, anchor_time]), "tumbling", I_tensor)
true_omega_anchor = w_hist[1]

print(f"\nAnchor: ep {anchor_idx}, mag {anchor_mag:.2f}")
print(f"Constraints: {n_constraints}")
for ci in range(n_constraints):
    ep = non_anchor[ci]
    print(f"  ep {ep}: mag={constraint_mags[ci]:.2f}, "
          f"allowed={[group_names[i] for i in constraint_allowed[ci]]}")
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
# GRID A: Alignment cost (baseline)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("GRID A: Alignment cost (baseline)")
print(f"{'='*60}")

_constraint_allowed = constraint_allowed
CONSTRAINT_WEIGHT = 10.0

def eval_alignment_cost(wi):
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
                allowed = _constraint_allowed[ci]
                bds = (pbs @ unique_normals[allowed].T).max(axis=1)
                cost += CONSTRAINT_WEIGHT * (1.0 - bds) ** 2
            bi = int(np.argmin(cost))
            if cost[bi] < best_cost:
                best_cost = cost[bi]
                best_omega = omega_test.copy()
                best_ni = ni
                best_phi_idx = bi
    return best_cost, best_omega, best_ni, best_phi_idx

t0 = time.time()
with Pool(N_WORKERS) as pool:
    align_results = pool.map(eval_alignment_cost, range(N_DIRS))
print(f"Alignment grid done in {time.time()-t0:.0f}s")

align_costs = np.array([r[0] for r in align_results])
align_omegas = np.array([r[1] for r in align_results])
align_ni = np.array([r[2] for r in align_results], dtype=int)
align_phi = np.array([r[3] for r in align_results], dtype=int)
sorted_align = np.argsort(align_costs)

print(f"\nAlignment grid top-10:")
for i in range(10):
    ri = sorted_align[i]
    w_err = omega_dir_err(align_omegas[ri], true_omega_anchor)
    w_mag = np.rad2deg(np.linalg.norm(align_omegas[ri]))
    w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100
    print(f"  #{i+1}: cost={align_costs[ri]:.6f} | "
          f"w_dir={w_err:.1f}° w_mag={w_mag_err:+.1f}%")

dir_errs_a = np.array([omega_dir_err(align_omegas[i], true_omega_anchor) for i in range(N_DIRS)])
closest_a = int(np.argmin(dir_errs_a))
rank_a = int(np.where(sorted_align == closest_a)[0][0]) + 1
print(f"Truth rank: #{rank_a} (err={dir_errs_a[closest_a]:.1f}°)")


# ══════════════════════════════════════════════════════════════════════
# GRID B: Brightness-match cost
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("GRID B: Brightness-match cost")
print(f"{'='*60}")

# Pre-compute mag_table for use in workers (shared via fork)
_mag_table = mag_table
_calib_dots = calib_dots
_constraint_mags = constraint_mags

def _predict_mag_vec(ni, dots_arr):
    """Vectorized magnitude prediction for a single normal across all phis."""
    return np.interp(dots_arr, _calib_dots[::-1], _mag_table[ni, ::-1])

def eval_brightness_cost(wi):
    wd = omega_dirs[wi]
    best_cost = np.inf
    best_omega = None
    best_ni = -1
    best_phi_idx = -1
    for mag in omega_mags_search:
        omega_test = wd * mag
        dqs = propagate_delta_qs(omega_test, dt_constraints, I_tensor)
        for a_ni, qa_xyzw in qa_anchor_sets:
            n_phi = len(qa_xyzw)
            R_anchors = Rotation.from_quat(qa_xyzw)
            cost = np.zeros(n_phi)
            for ci in range(n_constraints):
                dq = dqs[ci]
                R_delta = Rotation.from_quat([dq[1], dq[2], dq[3], dq[0]])
                R_all = R_anchors * R_delta
                pbs = R_all.apply(pab_at_constraints[ci])  # (n_phi, 3)
                allowed = _constraint_allowed[ci]
                obs_mag = _constraint_mags[ci]
                # For each allowed normal, predict magnitude and compute brightness cost
                best_bright_cost = np.full(n_phi, np.inf)
                for ni in allowed:
                    dots = pbs @ unique_normals[ni]  # (n_phi,)
                    pred = _predict_mag_vec(ni, dots)  # (n_phi,)
                    bright_cost = (pred - obs_mag) ** 2
                    best_bright_cost = np.minimum(best_bright_cost, bright_cost)
                cost += best_bright_cost
            bi = int(np.argmin(cost))
            if cost[bi] < best_cost:
                best_cost = cost[bi]
                best_omega = omega_test.copy()
                best_ni = a_ni
                best_phi_idx = bi
    return best_cost, best_omega, best_ni, best_phi_idx

t0 = time.time()
with Pool(N_WORKERS) as pool:
    bright_results = pool.map(eval_brightness_cost, range(N_DIRS))
print(f"Brightness grid done in {time.time()-t0:.0f}s")

bright_costs = np.array([r[0] for r in bright_results])
bright_omegas = np.array([r[1] for r in bright_results])
bright_ni = np.array([r[2] for r in bright_results], dtype=int)
bright_phi = np.array([r[3] for r in bright_results], dtype=int)
sorted_bright = np.argsort(bright_costs)

print(f"\nBrightness grid top-10:")
for i in range(10):
    ri = sorted_bright[i]
    w_err = omega_dir_err(bright_omegas[ri], true_omega_anchor)
    w_mag = np.rad2deg(np.linalg.norm(bright_omegas[ri]))
    w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100
    print(f"  #{i+1}: cost={bright_costs[ri]:.4f} | "
          f"w_dir={w_err:.1f}° w_mag={w_mag_err:+.1f}%")

dir_errs_b = np.array([omega_dir_err(bright_omegas[i], true_omega_anchor) for i in range(N_DIRS)])
closest_b = int(np.argmin(dir_errs_b))
rank_b = int(np.where(sorted_bright == closest_b)[0][0]) + 1
print(f"Truth rank: #{rank_b} (err={dir_errs_b[closest_b]:.1f}°)")


# ══════════════════════════════════════════════════════════════════════
# LO-FI VALIDATION of brightness grid top candidates
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Lo-fi peak matching: top candidates from each grid")
print(f"{'='*60}")

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

# Back-propagate top candidates from both grids
def back_prop_candidates(grid_sorted, grid_omegas, grid_ni_arr, grid_phi_arr, top_n):
    cands = []
    for rank in range(min(top_n, len(grid_sorted))):
        gi = grid_sorted[rank]
        omega = grid_omegas[gi]
        ni = int(grid_ni_arr[gi])
        phi_idx = int(grid_phi_arr[gi])
        phi_arr = phi_coarse_z if ni in Z_NORMALS else phi_coarse_xy
        qa = anchor_q_from_phi(phi_arr[phi_idx], unique_normals[ni], pab_j2000[anchor_idx])
        bt = np.array([0.0, anchor_time])
        qb, ob = propagate_attitude(qa, -omega, bt, "tumbling", I_tensor)
        cands.append({'q0': qb[-1], 'w0': -ob[-1], 'omega_grid': omega, 'rank': rank})
    return cands

print("\nBack-propagating top-20 from each grid...")
align_cands = back_prop_candidates(sorted_align, align_omegas, align_ni, align_phi, LOFI_TOP)
bright_cands = back_prop_candidates(sorted_bright, bright_omegas, bright_ni, bright_phi, LOFI_TOP)

all_lofi_args = []
labels = []
for i, c in enumerate(align_cands):
    all_lofi_args.append((len(all_lofi_args), c['q0'], c['w0']))
    labels.append(('ALIGN', i, c))
for i, c in enumerate(bright_cands):
    all_lofi_args.append((len(all_lofi_args), c['q0'], c['w0']))
    labels.append(('BRIGHT', i, c))

print(f"Evaluating {len(all_lofi_args)} candidates with lo-fi...")
t0 = time.time()
with Pool(N_WORKERS) as pool:
    lofi_results = pool.map(eval_lofi, all_lofi_args)
print(f"Lo-fi done in {time.time()-t0:.1f}s")

# Report
for grid_name in ['ALIGN', 'BRIGHT']:
    print(f"\n  Top-10 {grid_name} grid candidates (lo-fi scored):")
    grid_items = []
    for idx, n_matched, mse in lofi_results:
        gname, rank, cand = labels[idx]
        if gname != grid_name:
            continue
        q0_err = attitude_error_deg(cand['q0'], true_q0)
        w_dir = omega_dir_err(cand['w0'], true_omega0)
        w_mag = np.rad2deg(np.linalg.norm(cand['w0']))
        w_mag_err = (w_mag - true_omega_mag_dps) / true_omega_mag_dps * 100
        grid_items.append((rank, n_matched, mse, q0_err, w_dir, w_mag_err))

    grid_items.sort(key=lambda x: (-x[1], x[2]))
    for rank, nm, mse, q0e, wde, wme in grid_items[:10]:
        tag = " <--" if wde < 10 else ""
        print(f"    grid#{rank+1}: matched={nm}/{len(peaks_idx)} mse={mse:.3f} | "
              f"q0={q0e:.1f}° w_dir={wde:.1f}° w_mag={wme:+.1f}%{tag}")


# ── Summary ──────────────────────────────────────────────────────────
total_time = time.time() - t_global
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Alignment grid:   truth rank #{rank_a}/{N_DIRS}")
print(f"  Brightness grid:  truth rank #{rank_b}/{N_DIRS}")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

np.savez(str(CKPT_DIR / "results.npz"),
         align_costs=align_costs, align_omegas=align_omegas,
         bright_costs=bright_costs, bright_omegas=bright_omegas,
         mag_table=mag_table, calib_dots=calib_dots,
         true_omega_anchor=true_omega_anchor)
print(f"Saved to {CKPT_DIR}/")
