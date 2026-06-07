#!/usr/bin/env python3
"""
m110 -- Sparse Hi-Fi Phi Sweep.

Tests whether hi-fi (shadow-inclusive) brightness evaluation at a sparse set of
peak epochs can discriminate the correct attitude twist angle (phi) for ATT_FAIL
seeds. Uses the TRUTH omega to isolate the phi problem.

Hypothesis:
  Lo-fi (no shadows) phi sweep CANNOT discriminate phi for ATT_FAIL seeds because
  the zero-phase brightness surface is too symmetric. Shadows (up to 4.6 mag) are
  the ONLY mechanism that breaks this symmetry.

Method:
  For each seed, using truth omega:
  1. Identify 10 brightest peaks from the observed (hi-fi) light curve.
  2. Sweep 72 phi values (5 deg spacing).
  3. For each phi:
     a. Compute candidate attitude (quaternion) at anchor epoch via anchor_q_from_phi.
     b. Propagate to eval epochs using precomputed delta-q from truth.
     c. Compute k1_body, k2_body at eval epochs.
     d. Hi-fi: compute_shadows + generate_lightcurves at eval epochs only.
     e. Lo-fi: create_no_shadow_lit_status + generate_lightcurves at eval epochs.
     f. Score: MSE of predicted vs observed magnitude at the 10 peaks.
  4. Report discrimination ratio and truth phi rank.

Key optimisation: evaluate all 10 peaks in a single call to compute_shadows /
generate_lightcurves per phi value. Per-phi cost ~0.5-2s (shadow engine).

Seeds: 27, 46, 58 (ATT_FAIL), 0, 75 (borderline), 93 (OK control)

Usage:
  MICRO110_SEED=27 python3 notebooks/inversion/12_brightness_surface/m110_hifi_phi.py
  # Or run all seeds:
  MICRO110_SEED=all python3 notebooks/inversion/12_brightness_surface/m110_hifi_phi.py
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, ExperimentContext
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.articulation import compute_rotation_matrices_from_angles

# ── Constants ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_DIR = RESULTS_DIR / "m110_hifi_phi"

ALL_SEEDS = [27, 46, 58, 0, 75, 93]
SEED_STATUS = {27: "ATT_FAIL", 46: "ATT_FAIL", 58: "ATT_FAIL",
               0: "ATT_FAIL_border", 75: "ATT_FAIL_border", 93: "OK"}

N_PHI = 72                         # phi values: 0, 5, 10, ..., 355 deg
PHI_DEG = np.arange(N_PHI) * (360.0 / N_PHI)
PHI_RAD = np.deg2rad(PHI_DEG)
N_EVAL_PEAKS = 10                  # brightest peaks to evaluate

NOISE_SEED = 42
NOISE_SIGMA = 0.05

# ── Logging ──────────────────────────────────────────────────────────
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


# ── Helpers ──────────────────────────────────────────────────────────
def anchor_q_from_phi(phi, n_body, pab):
    """Construct attitude quaternion for a given phi twist about n_body.

    Parameters
    ----------
    phi : float
        Twist angle in radians.
    n_body : (3,) array
        Body-frame normal (twist axis).
    pab : (3,) array
        PAB direction in J2000 at the anchor epoch.

    Returns
    -------
    q_wxyz : (4,) array
        Quaternion (w, x, y, z) that maps J2000 -> body frame with
        the PAB aligned to n_body (modulo the twist phi).
    """
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q_xyzw = R_total.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def find_truth_phi_idx(q_truth_anchor_wxyz, anchor_normal, pab_anchor_j2000):
    """Find which phi index (in PHI_RAD) best matches the truth attitude.

    Sweeps 3600 fine phi values, finds the closest, then snaps to the
    nearest index in the coarse PHI_RAD grid.

    Returns
    -------
    truth_phi_idx : int
        Index into PHI_DEG / PHI_RAD.
    truth_phi_deg : float
        Fine-resolution truth phi in degrees.
    min_err_deg : float
        Attitude error at the best fine phi.
    """
    R_truth = Rotation.from_quat([q_truth_anchor_wxyz[1], q_truth_anchor_wxyz[2],
                                   q_truth_anchor_wxyz[3], q_truth_anchor_wxyz[0]])
    fine_phi = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    best_err = 999.0
    best_phi = 0.0
    for p in fine_phi:
        q_cand = anchor_q_from_phi(p, anchor_normal, pab_anchor_j2000)
        R_cand = Rotation.from_quat([q_cand[1], q_cand[2], q_cand[3], q_cand[0]])
        err = np.rad2deg((R_cand.inv() * R_truth).magnitude())
        if err < best_err:
            best_err = err
            best_phi = p

    # Snap to coarse grid
    truth_phi_deg = np.rad2deg(best_phi) % 360.0
    coarse_dists = np.abs(PHI_DEG - truth_phi_deg)
    coarse_dists = np.minimum(coarse_dists, 360.0 - coarse_dists)
    truth_phi_idx = int(np.argmin(coarse_dists))

    return truth_phi_idx, truth_phi_deg, best_err


def compute_body_frame_vectors(quats_wxyz, sun_pos, obs_pos, sat_pos):
    """Compute k1_body and k2_body from quaternions and J2000 positions.

    Parameters
    ----------
    quats_wxyz : (N, 4) — quaternions in (w, x, y, z) order
    sun_pos : (N, 3) — sun positions in J2000
    obs_pos : (N, 3) — observer positions in J2000
    sat_pos : (N, 3) — satellite positions in J2000

    Returns
    -------
    k1_body : (N, 3) — sun direction in body frame (normalised)
    k2_body : (N, 3) — observer direction in body frame (normalised)
    """
    n = len(quats_wxyz)
    # Convert wxyz -> xyzw for scipy
    R_all = Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]])
    R_matrices = R_all.as_matrix()  # (N, 3, 3) — J2000 to body

    sun_vec = sun_pos - sat_pos
    sun_vec /= np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_vec = obs_pos - sat_pos
    obs_vec /= np.linalg.norm(obs_vec, axis=1, keepdims=True)

    k1_body = np.einsum('nij,nj->ni', R_matrices, sun_vec)
    k2_body = np.einsum('nij,nj->ni', R_matrices, obs_vec)

    return k1_body, k2_body


def discrimination_ratio(scores, truth_idx, lower_is_better=True):
    """Compute discrimination ratio. >1 means truth wins."""
    truth_score = scores[truth_idx]
    wrong_scores = np.concatenate([scores[:truth_idx], scores[truth_idx + 1:]])

    if lower_is_better:
        best_wrong = np.nanmin(wrong_scores)
        if truth_score <= 0 or np.isnan(truth_score):
            return np.nan
        return best_wrong / truth_score
    else:
        best_wrong = np.nanmax(wrong_scores)
        if best_wrong <= 0 or np.isnan(best_wrong):
            return np.nan
        return truth_score / best_wrong


def truth_rank(scores, truth_idx, lower_is_better=True):
    """Rank of truth phi among all phi values (1 = best)."""
    if lower_is_better:
        return int(np.sum(scores < scores[truth_idx])) + 1
    else:
        return int(np.sum(scores > scores[truth_idx])) + 1


# ══════════════════════════════════════════════════════════════════════
# PER-SEED PROCESSING
# ══════════════════════════════════════════════════════════════════════
def process_seed(seed, traj, ctx):
    """Run hi-fi + lo-fi phi sweep for one seed.

    Parameters
    ----------
    seed : int
    traj : NPZ archive (m046_trajectories)
    ctx : ExperimentContext (satellite, positions, art_matrices)

    Returns
    -------
    dict with all per-phi scores and diagnostics.
    """
    t0 = time.time()

    # ── Load truth data ──────────────────────────────────────────────
    q_truth_all = traj['quaternions'][seed]       # (500, 4) wxyz
    omega0_truth = traj['omega0s'][seed]           # (3,)
    pab_j2000 = traj['pab_j2000']                 # (500, 3)
    pab_body_truth = traj['pab_body'][seed]        # (500, 3)
    mag_hifi_truth = traj['mag_hifi'][seed]        # (500,)
    unique_normals = traj['unique_normals']        # (n_normals, 3)
    obs_times = traj['observation_times']          # (500,)
    I_tensor = traj['inertia_tensor']

    # Observed LC (truth + noise)
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.normal(0, NOISE_SIGMA, len(mag_hifi_truth))
    observed_lc = mag_hifi_truth + noise

    # ── Select 10 brightest peaks ────────────────────────────────────
    peak_mask = traj['peak_seeds'] == seed
    seed_peak_epochs = traj['peak_epochs'][peak_mask]

    if len(seed_peak_epochs) < N_EVAL_PEAKS:
        print(f"  WARNING: seed {seed} has only {len(seed_peak_epochs)} peaks, "
              f"using all of them")
        eval_epochs = seed_peak_epochs[np.argsort(mag_hifi_truth[seed_peak_epochs])]
    else:
        # Sort by brightness (lowest magnitude = brightest)
        sorted_by_mag = seed_peak_epochs[np.argsort(mag_hifi_truth[seed_peak_epochs])]
        eval_epochs = sorted_by_mag[:N_EVAL_PEAKS]

    n_eval = len(eval_epochs)
    eval_obs_mag = observed_lc[eval_epochs]
    print(f"  Eval epochs ({n_eval}): {eval_epochs.tolist()}")
    print(f"  Eval magnitudes (truth): "
          f"{mag_hifi_truth[eval_epochs].min():.2f} to "
          f"{mag_hifi_truth[eval_epochs].max():.2f}")

    # ── Determine anchor normal ──────────────────────────────────────
    # Anchor = brightest peak (first in sorted list)
    anchor_ep = eval_epochs[0]
    pab_at_anchor = pab_body_truth[anchor_ep]
    dots = unique_normals @ pab_at_anchor
    anchor_normal = unique_normals[np.argmax(dots)].copy()
    print(f"  Anchor: epoch {anchor_ep}, normal={anchor_normal}")

    # ── Precompute delta-q (truth relative rotations) ────────────────
    # R_truth(t) for all 500 epochs
    R_truth_all = Rotation.from_quat(q_truth_all[:, [1, 2, 3, 0]])
    R_anchor_truth = R_truth_all[anchor_ep]
    R_anchor_inv = R_anchor_truth.inv()
    # delta_R(t) = R_truth(t) * R_anchor^{-1}
    # So for candidate: R_cand(t) = delta_R(t) * R_cand_anchor
    delta_R_eval = R_truth_all[eval_epochs] * R_anchor_inv

    # ── Find truth phi ───────────────────────────────────────────────
    truth_phi_idx, truth_phi_deg, truth_phi_err = find_truth_phi_idx(
        q_truth_all[anchor_ep], anchor_normal, pab_j2000[anchor_ep])
    print(f"  Truth phi: {truth_phi_deg:.1f} deg (grid idx={truth_phi_idx}, "
          f"match err={truth_phi_err:.2f} deg)")

    # ── Sanity check: phi=truth reproduces truth quaternions at eval ──
    q_check = anchor_q_from_phi(PHI_RAD[truth_phi_idx], anchor_normal,
                                 pab_j2000[anchor_ep])
    R_check_anchor = Rotation.from_quat([q_check[1], q_check[2], q_check[3], q_check[0]])
    R_check_eval = delta_R_eval * R_check_anchor
    # Compare with truth at eval epochs
    R_truth_eval = R_truth_all[eval_epochs]
    max_att_err = max(np.rad2deg((R_check_eval[i].inv() * R_truth_eval[i]).magnitude())
                      for i in range(n_eval))
    print(f"  Delta-q reconstruction error at eval epochs: {max_att_err:.2f} deg "
          f"({'OK' if max_att_err < 3.0 else 'WARN'})")

    # ── Slice context data for eval epochs ───────────────────────────
    eval_sun = ctx.sun_pos[eval_epochs]
    eval_obs = ctx.obs_pos[eval_epochs]
    eval_sat = ctx.sat_pos[eval_epochs]
    eval_dist = ctx.obs_dist[eval_epochs]
    eval_art = {comp: matrices[eval_epochs] for comp, matrices in ctx.art_matrices.items()}

    # ── Phi sweep ────────────────────────────────────────────────────
    hifi_mse = np.full(N_PHI, np.nan)
    lofi_mse = np.full(N_PHI, np.nan)
    hifi_mag_all = np.full((N_PHI, n_eval), np.nan)
    lofi_mag_all = np.full((N_PHI, n_eval), np.nan)

    for pi, phi_rad in enumerate(PHI_RAD):
        t_phi = time.time()

        # Construct candidate attitude at anchor
        q_cand_anchor = anchor_q_from_phi(phi_rad, anchor_normal, pab_j2000[anchor_ep])
        R_cand_anchor = Rotation.from_quat([q_cand_anchor[1], q_cand_anchor[2],
                                             q_cand_anchor[3], q_cand_anchor[0]])

        # Propagate to eval epochs via delta-q
        R_cand_eval = delta_R_eval * R_cand_anchor  # (n_eval,) Rotation

        # Convert to wxyz quaternions
        q_eval_xyzw = R_cand_eval.as_quat()  # (n_eval, 4) xyzw
        q_eval_wxyz = np.column_stack([q_eval_xyzw[:, 3],
                                        q_eval_xyzw[:, 0],
                                        q_eval_xyzw[:, 1],
                                        q_eval_xyzw[:, 2]])

        # Compute k1_body, k2_body at eval epochs
        k1_body, k2_body = compute_body_frame_vectors(
            q_eval_wxyz, eval_sun, eval_obs, eval_sat)

        # ── Hi-fi: shadows + light curve ─────────────────────────────
        # Use precomputed articulation matrices (fixed approximation:
        # solar panel tracking angle depends weakly on attitude, <5 deg
        # change for typical phi errors, introducing <0.05 mag error,
        # negligible vs shadow effects of 0.1-4.6 mag)
        lit_hifi = compute_shadows(
            satellite=ctx.satellite, k1_vectors=k1_body,
            explicit_component_matrices=eval_art, show_progress=False)
        mag_hifi_pred, _, _, _, _, _ = generate_lightcurves(
            facet_lit_status_dict=lit_hifi,
            k1_vectors_array=k1_body,
            k2_vectors_array=k2_body,
            observer_distances=eval_dist,
            satellite=ctx.satellite,
            epochs=np.arange(n_eval, dtype=float),
            pre_computed_matrices=eval_art,
            generate_no_shadow=False, animate=False, show_progress=False)

        hifi_mag_all[pi] = mag_hifi_pred
        hifi_mse[pi] = float(np.mean((mag_hifi_pred - eval_obs_mag) ** 2))

        # ── Lo-fi: no shadows ────────────────────────────────────────
        lit_lofi = create_no_shadow_lit_status(ctx.satellite, n_eval)
        mag_lofi_pred, _, _, _, _, _ = generate_lightcurves(
            facet_lit_status_dict=lit_lofi,
            k1_vectors_array=k1_body,
            k2_vectors_array=k2_body,
            observer_distances=eval_dist,
            satellite=ctx.satellite,
            epochs=np.arange(n_eval, dtype=float),
            pre_computed_matrices=eval_art,
            generate_no_shadow=False, animate=False, show_progress=False)

        lofi_mag_all[pi] = mag_lofi_pred
        lofi_mse[pi] = float(np.mean((mag_lofi_pred - eval_obs_mag) ** 2))

        dt_phi = time.time() - t_phi
        if pi == 0 or (pi + 1) % 12 == 0:
            marker = " <-- TRUTH" if pi == truth_phi_idx else ""
            print(f"    phi={PHI_DEG[pi]:5.0f} deg: "
                  f"hifi_mse={hifi_mse[pi]:.4f}, "
                  f"lofi_mse={lofi_mse[pi]:.4f}  "
                  f"({dt_phi:.2f}s){marker}")

    elapsed = time.time() - t0
    print(f"  Phi sweep done in {elapsed:.1f}s "
          f"({elapsed / N_PHI:.2f}s per phi)")

    # ── Discrimination metrics ───────────────────────────────────────
    hifi_disc = discrimination_ratio(hifi_mse, truth_phi_idx, lower_is_better=True)
    lofi_disc = discrimination_ratio(lofi_mse, truth_phi_idx, lower_is_better=True)
    hifi_rank = truth_rank(hifi_mse, truth_phi_idx, lower_is_better=True)
    lofi_rank = truth_rank(lofi_mse, truth_phi_idx, lower_is_better=True)

    return {
        'seed': seed,
        'status': SEED_STATUS[seed],
        'phi_deg': PHI_DEG,
        'hifi_mse': hifi_mse,
        'lofi_mse': lofi_mse,
        'hifi_mag': hifi_mag_all,
        'lofi_mag': lofi_mag_all,
        'eval_epochs': eval_epochs,
        'eval_obs_mag': eval_obs_mag,
        'anchor_epoch': int(anchor_ep),
        'anchor_normal': anchor_normal,
        'truth_phi_idx': truth_phi_idx,
        'truth_phi_deg': truth_phi_deg,
        'truth_phi_err': truth_phi_err,
        'hifi_disc': hifi_disc,
        'lofi_disc': lofi_disc,
        'hifi_rank': hifi_rank,
        'lofi_rank': lofi_rank,
        'n_eval': n_eval,
        'elapsed_s': elapsed,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(str(OUT_DIR / "diagnostic.log"), "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    # ── Determine which seeds to run ─────────────────────────────────
    seed_env = os.environ.get('MICRO110_SEED', 'all')
    if seed_env.lower() == 'all':
        seeds = ALL_SEEDS
    else:
        seeds = [int(seed_env)]
        # Validate
        for s in seeds:
            if s not in SEED_STATUS:
                print(f"WARNING: seed {s} not in predefined set, "
                      f"status='UNKNOWN'")
                SEED_STATUS[s] = "UNKNOWN"

    print("=" * 70)
    print("m110 -- Sparse Hi-Fi Phi Sweep")
    print(f"  Seeds: {seeds}")
    print(f"  N_PHI: {N_PHI} (step = {360.0 / N_PHI:.0f} deg)")
    print(f"  N_EVAL_PEAKS: {N_EVAL_PEAKS}")
    print(f"  Noise: sigma={NOISE_SIGMA}, seed={NOISE_SEED}")
    print("=" * 70)
    t_global = time.time()

    # ── Load trajectory data ─────────────────────────────────────────
    print("\nLoading trajectory data...")
    t_load = time.time()
    traj = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
    print(f"  Loaded: {traj['n_trajectories']} trajectories, "
          f"{traj['n_obs']} epochs each ({time.time() - t_load:.1f}s)")

    # ── Setup experiment context (satellite, SPICE, positions) ───────
    print("\nSetting up experiment context (satellite + SPICE)...")
    t_ctx = time.time()
    ctx = setup_experiment(
        n_observations=500, noise_sigma=NOISE_SIGMA, random_seed=NOISE_SEED,
        true_omega_deg=(0.5, -0.3, 2.0),
        end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)
    print(f"  Context ready ({time.time() - t_ctx:.1f}s)")

    # ── Process each seed ────────────────────────────────────────────
    all_results = {}

    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"# Seed {seed} ({SEED_STATUS.get(seed, 'UNKNOWN')})")
        print(f"{'#' * 70}")

        result = process_seed(seed, traj, ctx)
        all_results[seed] = result

        # ── Print per-phi table ──────────────────────────────────────
        print(f"\n  {'phi':>5s} | {'hifi_mse':>10s} | {'lofi_mse':>10s} | {'note':>10s}")
        print(f"  {'-' * 45}")
        for pi in range(N_PHI):
            marker = " <-- TRUTH" if pi == result['truth_phi_idx'] else ""
            print(f"  {PHI_DEG[pi]:5.0f} | "
                  f"{result['hifi_mse'][pi]:10.4f} | "
                  f"{result['lofi_mse'][pi]:10.4f} |{marker}")

        # ── Per-seed summary ─────────────────────────────────────────
        print(f"\n  Summary (seed {seed}, {SEED_STATUS.get(seed, 'UNKNOWN')}):")
        print(f"    Truth phi:     {result['truth_phi_deg']:.1f} deg "
              f"(grid idx {result['truth_phi_idx']}, "
              f"match err {result['truth_phi_err']:.2f} deg)")
        print(f"    Hi-fi rank:    {result['hifi_rank']}/{N_PHI}  "
              f"disc={result['hifi_disc']:.3f}  "
              f"{'OK' if result['hifi_disc'] > 1.0 else 'FAIL'}")
        print(f"    Lo-fi rank:    {result['lofi_rank']}/{N_PHI}  "
              f"disc={result['lofi_disc']:.3f}  "
              f"{'OK' if result['lofi_disc'] > 1.0 else 'FAIL'}")
        print(f"    Hi-fi MSE at truth:   {result['hifi_mse'][result['truth_phi_idx']]:.4f}")
        print(f"    Hi-fi MSE best wrong: "
              f"{np.nanmin(np.concatenate([result['hifi_mse'][:result['truth_phi_idx']], result['hifi_mse'][result['truth_phi_idx']+1:]])):.4f}")
        print(f"    Lo-fi MSE at truth:   {result['lofi_mse'][result['truth_phi_idx']]:.4f}")
        print(f"    Lo-fi MSE best wrong: "
              f"{np.nanmin(np.concatenate([result['lofi_mse'][:result['truth_phi_idx']], result['lofi_mse'][result['truth_phi_idx']+1:]])):.4f}")

        # ── Save per-seed NPZ ────────────────────────────────────────
        npz_path = OUT_DIR / f"seed_{seed:03d}.npz"
        np.savez(str(npz_path),
                 seed=seed,
                 status=SEED_STATUS.get(seed, 'UNKNOWN'),
                 phi_deg=result['phi_deg'],
                 hifi_mse=result['hifi_mse'],
                 lofi_mse=result['lofi_mse'],
                 hifi_mag=result['hifi_mag'],
                 lofi_mag=result['lofi_mag'],
                 eval_epochs=result['eval_epochs'],
                 eval_obs_mag=result['eval_obs_mag'],
                 anchor_epoch=result['anchor_epoch'],
                 anchor_normal=result['anchor_normal'],
                 truth_phi_idx=result['truth_phi_idx'],
                 truth_phi_deg=result['truth_phi_deg'],
                 truth_phi_err=result['truth_phi_err'],
                 hifi_disc=result['hifi_disc'],
                 lofi_disc=result['lofi_disc'],
                 hifi_rank=result['hifi_rank'],
                 lofi_rank=result['lofi_rank'])
        print(f"  Saved: {npz_path}")

    # ══════════════════════════════════════════════════════════════════
    # GLOBAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: Hi-Fi vs Lo-Fi Phi Discrimination")
    print(f"{'=' * 80}")

    header = (f"{'Seed':>6s} | {'Status':>15s} | "
              f"{'HiFi rank':>10s} | {'HiFi disc':>10s} | "
              f"{'LoFi rank':>10s} | {'LoFi disc':>10s} | "
              f"{'Time':>6s}")
    print(header)
    print("-" * len(header))

    for seed in seeds:
        r = all_results[seed]
        hifi_flag = "OK" if r['hifi_disc'] > 1.0 else "FAIL"
        lofi_flag = "OK" if r['lofi_disc'] > 1.0 else "FAIL"
        print(f"{seed:6d} | {r['status']:>15s} | "
              f"{r['hifi_rank']:>5d}/{N_PHI:d} {hifi_flag:>4s} | "
              f"{r['hifi_disc']:10.3f} | "
              f"{r['lofi_rank']:>5d}/{N_PHI:d} {lofi_flag:>4s} | "
              f"{r['lofi_disc']:10.3f} | "
              f"{r['elapsed_s']:5.0f}s")

    # ── Verdict by category ──────────────────────────────────────────
    att_fail_seeds = [s for s in seeds if 'ATT_FAIL' in SEED_STATUS.get(s, '')]
    ok_seeds = [s for s in seeds if SEED_STATUS.get(s, '') == 'OK']

    if att_fail_seeds:
        print(f"\nATT_FAIL seeds ({len(att_fail_seeds)}):")
        hifi_wins = sum(1 for s in att_fail_seeds if all_results[s]['hifi_disc'] > 1.0)
        lofi_wins = sum(1 for s in att_fail_seeds if all_results[s]['lofi_disc'] > 1.0)
        print(f"  Hi-fi discriminates: {hifi_wins}/{len(att_fail_seeds)}")
        print(f"  Lo-fi discriminates: {lofi_wins}/{len(att_fail_seeds)}")

        if hifi_wins > 0:
            hifi_discs = [all_results[s]['hifi_disc'] for s in att_fail_seeds
                          if not np.isnan(all_results[s]['hifi_disc'])]
            print(f"  Hi-fi disc ratios: min={min(hifi_discs):.3f}, "
                  f"mean={np.mean(hifi_discs):.3f}, "
                  f"max={max(hifi_discs):.3f}")

    if ok_seeds:
        print(f"\nOK seeds ({len(ok_seeds)}):")
        for s in ok_seeds:
            r = all_results[s]
            print(f"  Seed {s}: hifi rank={r['hifi_rank']}/{N_PHI} "
                  f"disc={r['hifi_disc']:.3f}, "
                  f"lofi rank={r['lofi_rank']}/{N_PHI} "
                  f"disc={r['lofi_disc']:.3f}")

    # ── Final verdict ────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")

    if att_fail_seeds:
        hifi_wins = sum(1 for s in att_fail_seeds if all_results[s]['hifi_disc'] > 1.0)
        lofi_wins = sum(1 for s in att_fail_seeds if all_results[s]['lofi_disc'] > 1.0)

        if hifi_wins > lofi_wins:
            print(f"Hi-fi phi sweep OUTPERFORMS lo-fi for ATT_FAIL seeds: "
                  f"{hifi_wins}/{len(att_fail_seeds)} vs "
                  f"{lofi_wins}/{len(att_fail_seeds)}")
            if hifi_wins == len(att_fail_seeds):
                print("  --> Hi-fi sparse peak evaluation is a viable phi "
                      "discriminator for ATT_FAIL seeds.")
            else:
                failed_seeds = [s for s in att_fail_seeds
                                if all_results[s]['hifi_disc'] <= 1.0]
                print(f"  --> Partial success. Failed seeds: {failed_seeds}")
                print("  --> Investigate: are more eval peaks needed? "
                      "Different peak selection?")
        elif hifi_wins == lofi_wins:
            print(f"Hi-fi and lo-fi tie: {hifi_wins}/{len(att_fail_seeds)}")
            if hifi_wins == 0:
                print("  --> Neither method discriminates. "
                      "Shadow effects may not help at peak epochs.")
            else:
                print("  --> Both work equally well at peak epochs.")
        else:
            print(f"Unexpected: lo-fi OUTPERFORMS hi-fi ({lofi_wins} vs {hifi_wins})")
            print("  --> Investigate shadow artifacts or evaluation noise.")
    else:
        print("No ATT_FAIL seeds in this run.")

    elapsed_total = time.time() - t_global
    print(f"\nTotal runtime: {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")

    # ── Save summary JSON ────────────────────────────────────────────
    summary = {
        'seeds': seeds,
        'n_phi': N_PHI,
        'n_eval_peaks': N_EVAL_PEAKS,
        'results': {}
    }
    for seed in seeds:
        r = all_results[seed]
        summary['results'][str(seed)] = {
            'status': r['status'],
            'hifi_rank': r['hifi_rank'],
            'hifi_disc': float(r['hifi_disc']) if not np.isnan(r['hifi_disc']) else None,
            'lofi_rank': r['lofi_rank'],
            'lofi_disc': float(r['lofi_disc']) if not np.isnan(r['lofi_disc']) else None,
            'truth_phi_deg': float(r['truth_phi_deg']),
            'anchor_epoch': r['anchor_epoch'],
            'n_eval': r['n_eval'],
            'elapsed_s': round(r['elapsed_s'], 1),
        }
    summary_path = OUT_DIR / "summary.json"
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == '__main__':
    main()
