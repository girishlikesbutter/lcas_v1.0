#!/usr/bin/env python3
"""
m111b -- Hi-Fi Phi Sweep at Shadow-Rich Epochs.

Hypothesis:
  m110 failed to discriminate phi because it evaluated at the 10 BRIGHTEST
  peaks, which are all at ±X lobes where shadow effects are essentially zero
  (mean 0.004 mag). Shadows — the ONLY phi discriminator — are concentrated at
  ±Y lobes (mean 0.24-0.51 mag, max 4.6 mag). Evaluating hi-fi at shadow-rich
  epochs should provide the missing phi discrimination.

  Also tests whether the sheer NUMBER of epochs matters: 20 shadow-rich epochs
  vs m110's 10 brightest.

Method:
  For each ATT_FAIL seed with truth omega:
  1. Compute per-epoch shadow effect from truth data (|mag_hifi - mag_lofi|)
  2. Select 20 epochs with LARGEST shadow effects ("shadow-rich")
  3. Also select 20 epochs near ±Y normals ("Y-lobe")
  4. For 72 phi values: evaluate hi-fi + lo-fi at selected epochs
  5. Compare discrimination with m110 (10 brightest peaks)

Seeds: 27, 46, 58, 0, 75 (ATT_FAIL) + 93 (OK)

Usage:
  python3 notebooks/inversion/12_brightness_surface/m111b_shadow_epoch_hifi.py
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

from lib.experiment_setup import setup_experiment
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.articulation import compute_rotation_matrices_from_angles

# ── Constants ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_DIR = RESULTS_DIR / "m111b_shadow_hifi"

ALL_SEEDS = [27, 46, 58, 0, 75, 93]
SEED_STATUS = {27: "ATT_FAIL", 46: "ATT_FAIL", 58: "ATT_FAIL",
               0: "ATT_FAIL", 75: "ATT_FAIL", 93: "OK"}

N_PHI = 72
PHI_DEG = np.arange(N_PHI) * (360.0 / N_PHI)
PHI_RAD = np.deg2rad(PHI_DEG)

N_SHADOW_EPOCHS = 20   # epochs with largest shadow effects
N_Y_EPOCHS = 20        # epochs nearest ±Y normals

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
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q = R_total.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def find_truth_phi(q_truth_wxyz, anchor_normal, pab_anchor):
    R_truth = Rotation.from_quat([q_truth_wxyz[1], q_truth_wxyz[2],
                                   q_truth_wxyz[3], q_truth_wxyz[0]])
    fine_phi = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    best_err, best_phi = 999.0, 0.0
    for p in fine_phi:
        q = anchor_q_from_phi(p, anchor_normal, pab_anchor)
        R_c = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        e = np.rad2deg((R_c.inv() * R_truth).magnitude())
        if e < best_err:
            best_err, best_phi = e, p
    truth_deg = np.rad2deg(best_phi) % 360
    dists = np.minimum(np.abs(PHI_DEG - truth_deg), 360 - np.abs(PHI_DEG - truth_deg))
    return int(np.argmin(dists)), truth_deg, best_err


def compute_body_frame_vectors(quats_wxyz, sun_pos, obs_pos, sat_pos):
    R_all = Rotation.from_quat(quats_wxyz[:, [1, 2, 3, 0]])
    R_mats = R_all.as_matrix()
    sun_vec = sun_pos - sat_pos
    sun_vec /= np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_vec = obs_pos - sat_pos
    obs_vec /= np.linalg.norm(obs_vec, axis=1, keepdims=True)
    k1_body = np.einsum('nij,nj->ni', R_mats, sun_vec)
    k2_body = np.einsum('nij,nj->ni', R_mats, obs_vec)
    return k1_body, k2_body


def evaluate_phi_at_epochs(phi_rad, anchor_normal, pab_anchor, delta_R_eval,
                           eval_sun, eval_obs, eval_sat, eval_dist, eval_art,
                           satellite, eval_epochs_arr, observed_at_eval,
                           compute_hifi=True):
    """Evaluate hi-fi and lo-fi magnitude at selected epochs for one phi.

    Returns (hifi_mse, lofi_mse, hifi_mag, lofi_mag).
    """
    n_eval = len(eval_epochs_arr)

    # Construct attitude at anchor
    q_anchor = anchor_q_from_phi(phi_rad, anchor_normal, pab_anchor)
    R_anchor = Rotation.from_quat([q_anchor[1], q_anchor[2],
                                    q_anchor[3], q_anchor[0]])
    # Propagate to eval epochs
    R_eval = delta_R_eval * R_anchor
    q_eval_xyzw = R_eval.as_quat()
    q_eval_wxyz = np.column_stack([q_eval_xyzw[:, 3], q_eval_xyzw[:, :3]])

    k1_body, k2_body = compute_body_frame_vectors(
        q_eval_wxyz, eval_sun, eval_obs, eval_sat)

    # Articulation from k1
    sp_axis = np.array([1.0, 0.0, 0.0])
    proj = k1_body - np.outer(k1_body @ sp_axis, sp_axis)
    sp_angles = np.arctan2(proj[:, 2], proj[:, 1])
    ad_axis = np.array([0.0, 0.0, 1.0])
    proj_ad = k1_body - np.outer(k1_body @ ad_axis, ad_axis)
    ad_angles = np.arctan2(proj_ad[:, 1], proj_ad[:, 0])

    comp_angles = {
        'SP_North': sp_angles, 'SP_South': sp_angles,
        'AD_West': ad_angles, 'AD_East': ad_angles,
    }
    art_mats = compute_rotation_matrices_from_angles(comp_angles, satellite)

    dummy_epochs = np.arange(n_eval, dtype=float) * 120.0

    # Lo-fi
    lit_lofi = create_no_shadow_lit_status(satellite, n_eval)
    mag_lofi, _, _, _, _, _ = generate_lightcurves(
        lit_lofi, k1_body, k2_body, eval_dist, satellite, dummy_epochs,
        pre_computed_matrices=art_mats, show_progress=False
    )
    lofi_mse = float(np.mean((mag_lofi - observed_at_eval) ** 2))

    # Hi-fi
    hifi_mse = np.nan
    mag_hifi = np.full(n_eval, np.nan)
    if compute_hifi:
        lit_hifi = compute_shadows(satellite, k1_body,
                                    explicit_component_matrices=art_mats)
        mag_hifi, _, _, _, _, _ = generate_lightcurves(
            lit_hifi, k1_body, k2_body, eval_dist, satellite, dummy_epochs,
            pre_computed_matrices=art_mats, show_progress=False
        )
        hifi_mse = float(np.mean((mag_hifi - observed_at_eval) ** 2))

    return hifi_mse, lofi_mse, mag_hifi, mag_lofi


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(OUT_DIR / "diagnostic.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("m111b — Hi-Fi Phi Sweep at Shadow-Rich Epochs")
    print("=" * 60)
    t_start = time.time()

    ctx = setup_experiment(n_observations=500, noise_sigma=NOISE_SIGMA)
    traj = np.load(str(DATA_DIR / "m046_trajectories.npz"))

    unique_normals = traj['unique_normals']
    pab_j2000 = traj['pab_j2000']
    group_names = traj['group_names']

    results = {}

    for seed in ALL_SEEDS:
        t_seed = time.time()
        status = SEED_STATUS[seed]
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({status})")
        print(f"{'='*60}")

        q_truth = traj['quaternions'][seed]
        pab_body = traj['pab_body'][seed]
        mag_hifi_truth = traj['mag_hifi'][seed]
        mag_lofi_truth = traj['mag_lofi'][seed]
        best_group = traj['best_group'][seed]

        rng = np.random.default_rng(NOISE_SEED)
        observed = mag_hifi_truth + rng.normal(0, NOISE_SIGMA, 500)

        # ── Epoch selection strategies ───────────────────────────
        shadow_effect = np.abs(mag_hifi_truth - mag_lofi_truth)

        # Strategy 1: Top-20 by shadow effect
        shadow_order = np.argsort(-shadow_effect)
        shadow_epochs = shadow_order[:N_SHADOW_EPOCHS]

        # Strategy 2: Top-20 near ±Y normals
        y_dist = np.minimum(
            np.arccos(np.clip(pab_body @ unique_normals[2], -1, 1)),  # +Y
            np.arccos(np.clip(pab_body @ unique_normals[3], -1, 1))   # -Y
        )
        y_order = np.argsort(y_dist)
        y_epochs = y_order[:N_Y_EPOCHS]

        # Strategy 3: m110 style (brightest peaks)
        peak_mask = traj['peak_seeds'] == seed
        seed_peaks = traj['peak_epochs'][peak_mask]
        bright_order = np.argsort(mag_hifi_truth[seed_peaks])
        bright_peaks = seed_peaks[bright_order[:10]]

        # Strategy 4: Mixed — 10 shadow-rich + 10 brightest that aren't near ±X
        non_x_mask = np.ones(500, dtype=bool)
        for ep in range(500):
            if best_group[ep] in [0, 1, 6, 7, 8, 9]:  # ±X, ±WD, ±ED
                non_x_mask[ep] = False
        non_x_bright = np.where(non_x_mask)[0]
        non_x_bright = non_x_bright[np.argsort(mag_hifi_truth[non_x_bright])][:10]
        mixed_epochs = np.unique(np.concatenate([shadow_order[:10], non_x_bright]))

        print(f"  Epoch selection:")
        for name, eps in [("Shadow-rich", shadow_epochs),
                          ("±Y lobe", y_epochs),
                          ("Bright peaks", bright_peaks),
                          ("Mixed", mixed_epochs)]:
            lobes = [str(group_names[best_group[e]]) for e in eps]
            se = shadow_effect[eps]
            mags = mag_hifi_truth[eps]
            print(f"    {name:15s}: {len(eps)} eps, "
                  f"shadow [{se.min():.2f},{se.max():.2f}], "
                  f"mag [{mags.min():.1f},{mags.max():.1f}], "
                  f"lobes: {dict(zip(*np.unique(lobes, return_counts=True)))}")

        # ── Anchor + delta-R ─────────────────────────────────────
        anchor_ep = seed_peaks[np.argmin(mag_hifi_truth[seed_peaks])]
        pab_at_anchor = pab_body[anchor_ep]
        anchor_normal = unique_normals[np.argmax(unique_normals @ pab_at_anchor)]

        R_truth_all = Rotation.from_quat(q_truth[:, [1, 2, 3, 0]])
        R_anchor_inv = R_truth_all[anchor_ep].inv()

        truth_idx, truth_phi_deg, truth_err = find_truth_phi(
            q_truth[anchor_ep], anchor_normal, pab_j2000[anchor_ep])
        print(f"  Truth phi: {truth_phi_deg:.1f}° (idx {truth_idx}), "
              f"err {truth_err:.2f}°")

        # ── Sweep phis for each strategy ─────────────────────────
        strategies = {
            'shadow_rich': shadow_epochs,
            'y_lobe': y_epochs,
            'bright_peaks': bright_peaks,
            'mixed': mixed_epochs,
        }

        seed_results = {
            'status': status,
            'truth_phi_deg': float(truth_phi_deg),
            'truth_phi_idx': int(truth_idx),
            'truth_phi_err': float(truth_err),
        }

        for strat_name, eval_epochs in strategies.items():
            t_strat = time.time()
            n_eval = len(eval_epochs)

            # Precompute delta-R for eval epochs
            delta_R_eval = R_truth_all[eval_epochs] * R_anchor_inv

            eval_sun = ctx.sun_pos[eval_epochs]
            eval_obs = ctx.obs_pos[eval_epochs]
            eval_sat = ctx.sat_pos[eval_epochs]
            eval_dist = ctx.obs_dist[eval_epochs]

            observed_at_eval = observed[eval_epochs]

            hifi_scores = np.full(N_PHI, np.nan)
            lofi_scores = np.full(N_PHI, np.nan)

            for pi, phi_rad in enumerate(PHI_RAD):
                h, l, _, _ = evaluate_phi_at_epochs(
                    phi_rad, anchor_normal, pab_j2000[anchor_ep],
                    delta_R_eval,
                    eval_sun, eval_obs, eval_sat, eval_dist, None,
                    ctx.satellite, eval_epochs, observed_at_eval,
                    compute_hifi=True
                )
                hifi_scores[pi] = h
                lofi_scores[pi] = l

            # Compute metrics
            def compute_metrics(scores, truth_idx):
                rank = int(np.sum(scores < scores[truth_idx])) + 1
                wrong = np.concatenate([scores[:truth_idx], scores[truth_idx+1:]])
                disc = np.nanmin(wrong) / scores[truth_idx] if scores[truth_idx] > 0 else np.nan
                best_idx = np.argmin(scores)
                phi_err = min(abs(PHI_DEG[best_idx] - PHI_DEG[truth_idx]),
                              360 - abs(PHI_DEG[best_idx] - PHI_DEG[truth_idx]))
                return rank, disc, float(PHI_DEG[best_idx]), phi_err

            h_rank, h_disc, h_best, h_err = compute_metrics(hifi_scores, truth_idx)
            l_rank, l_disc, l_best, l_err = compute_metrics(lofi_scores, truth_idx)

            elapsed_strat = time.time() - t_strat
            print(f"\n  {strat_name} ({n_eval} epochs, {elapsed_strat:.0f}s):")
            print(f"    Hi-fi: rank {h_rank:2d}/72, disc {h_disc:.4f}, "
                  f"best {h_best:.0f}° (err {h_err:.0f}°)")
            print(f"    Lo-fi: rank {l_rank:2d}/72, disc {l_disc:.4f}, "
                  f"best {l_best:.0f}° (err {l_err:.0f}°)")

            seed_results[strat_name] = {
                'n_epochs': n_eval,
                'hifi': {'rank': h_rank, 'disc': h_disc, 'best_phi': h_best},
                'lofi': {'rank': l_rank, 'disc': l_disc, 'best_phi': l_best},
                'hifi_scores': hifi_scores.tolist(),
                'lofi_scores': lofi_scores.tolist(),
                'eval_epochs': eval_epochs.tolist(),
            }

        elapsed_seed = time.time() - t_seed
        print(f"\n  Seed total: {elapsed_seed:.0f}s ({elapsed_seed/60:.1f} min)")
        results[seed] = seed_results

        # Save per-seed
        np.savez(OUT_DIR / f"seed_{seed:03d}.npz", **{
            k: v for k, v in seed_results.items()
            if not isinstance(v, dict)
        })

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY — Hi-Fi Rank / Disc by Strategy")
    print(f"{'='*60}")
    print(f"{'Seed':>5s} {'Stat':>8s} | {'shadow_rich':>14s} | {'y_lobe':>14s} | "
          f"{'bright(m110)':>14s} | {'mixed':>14s}")
    print("-" * 80)
    for seed in ALL_SEEDS:
        r = results[seed]
        parts = [f"{seed:5d} {r['status']:>8s}"]
        for s in ['shadow_rich', 'y_lobe', 'bright_peaks', 'mixed']:
            h = r[s]['hifi']
            parts.append(f"r={h['rank']:2d} d={h['disc']:.3f}")
        print(" | ".join(parts))

    total = time.time() - t_start
    print(f"\nTotal: {total:.0f}s ({total/60:.1f} min)")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_DIR / 'summary.json'}")

    sys.stdout = sys.__stdout__
    log_file.close()
