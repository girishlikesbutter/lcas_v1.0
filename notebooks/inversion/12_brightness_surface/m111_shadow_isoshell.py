#!/usr/bin/env python3
"""
m111 -- Shadow-Corrected Isoshell Phi Discrimination.

Hypothesis:
  A precomputed hi-fi brightness surface (zero-phase, with shadows) can
  discriminate phi for ATT_FAIL seeds. Shadows break the lobe symmetry that
  makes lo-fi phi evaluation useless. By capturing shadows in a precomputed
  surface, we get hi-fi-quality phi discrimination at lo-fi computational cost.

Method:
  Phase 1: Precompute hi-fi brightness surface for IS-901
    - 1000 Fibonacci-spiral directions on the unit sphere
    - For each direction d: k1=k2=d, sun-tracking articulation
    - compute_shadows + generate_lightcurves -> B_hifi(d) and B_lofi(d)

  Phase 2: For each seed (truth omega), sweep 72 phi values:
    - Propagate attitude to all 500 epochs
    - Compute body-frame PAB direction at each epoch
    - Interpolate B_hifi_surface(PAB) from precomputed surface
    - Score MSE vs observed LC
    - Also score: B_lofi_surface, full lo-fi (actual k1/k2), pipeline lo-fi

  Phase 3: Compare discrimination across methods

Seeds: 27, 46, 58, 0, 75 (ATT_FAIL) + 93 (OK control)

Usage:
  python3 notebooks/inversion/12_brightness_surface/m111_shadow_isoshell.py
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.computation.brdf import BRDFManager
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.articulation import compute_rotation_matrices_from_angles
from lib.experiment_setup import setup_experiment

# ── Constants ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
OUT_DIR = RESULTS_DIR / "m111_shadow_isoshell"

ALL_SEEDS = [27, 46, 58, 0, 75, 93]
SEED_STATUS = {27: "ATT_FAIL", 46: "ATT_FAIL", 58: "ATT_FAIL",
               0: "ATT_FAIL", 75: "ATT_FAIL", 93: "OK"}

N_SURFACE_DIRS = 1000  # Fibonacci grid resolution
N_PHI = 72
PHI_DEG = np.arange(N_PHI) * (360.0 / N_PHI)
PHI_RAD = np.deg2rad(PHI_DEG)

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


# ── Fibonacci sphere ─────────────────────────────────────────────────
def fibonacci_sphere(n):
    """Generate n approximately uniform directions on unit sphere."""
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    phi = 2 * np.pi * i / golden
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])


# ── Precompute brightness surfaces ──────────────────────────────────
def precompute_brightness_surface(satellite, n_dirs):
    """Precompute hi-fi and lo-fi brightness at n_dirs directions (zero phase).

    Returns
    -------
    directions : (n_dirs, 3) unit vectors
    mag_hifi : (n_dirs,) hi-fi magnitudes (with shadows)
    mag_lofi : (n_dirs,) lo-fi magnitudes (no shadows)
    shadow_effect : (n_dirs,) mag_hifi - mag_lofi
    """
    print(f"\n{'='*60}")
    print(f"Precomputing brightness surface ({n_dirs} directions)...")
    print(f"{'='*60}")
    t0 = time.time()

    directions = fibonacci_sphere(n_dirs)
    k1 = directions.copy()
    k2 = directions.copy()

    # Articulation: solar panels track sun, dishes track sun
    sp_axis = np.array([1.0, 0.0, 0.0])
    proj = k1 - np.outer(k1 @ sp_axis, sp_axis)
    sp_angles = np.arctan2(proj[:, 2], proj[:, 1])

    ad_axis = np.array([0.0, 0.0, 1.0])
    proj_ad = k1 - np.outer(k1 @ ad_axis, ad_axis)
    ad_angles = np.arctan2(proj_ad[:, 1], proj_ad[:, 0])

    comp_angles = {
        'SP_North': sp_angles, 'SP_South': sp_angles,
        'AD_West': ad_angles, 'AD_East': ad_angles,
    }
    art_mats = compute_rotation_matrices_from_angles(comp_angles, satellite)

    obs_dist = np.full(n_dirs, 38000.0)
    epochs = np.arange(n_dirs, dtype=float) * 120.0  # dummy

    # Hi-fi
    t_shadow = time.time()
    lit_hifi = compute_shadows(satellite, k1, explicit_component_matrices=art_mats)
    print(f"  Shadow computation: {time.time()-t_shadow:.1f}s")

    mag_hifi, _, _, _, _, _ = generate_lightcurves(
        lit_hifi, k1, k2, obs_dist, satellite, epochs,
        pre_computed_matrices=art_mats, show_progress=False
    )

    # Lo-fi
    lit_lofi = create_no_shadow_lit_status(satellite, n_dirs)
    mag_lofi, _, _, _, _, _ = generate_lightcurves(
        lit_lofi, k1, k2, obs_dist, satellite, epochs,
        pre_computed_matrices=art_mats, show_progress=False
    )

    shadow_effect = mag_hifi - mag_lofi
    elapsed = time.time() - t0
    print(f"  Total: {elapsed:.1f}s ({elapsed/n_dirs*1000:.0f} ms/dir)")
    print(f"  Mag range: hifi [{mag_hifi.min():.2f}, {mag_hifi.max():.2f}], "
          f"lofi [{mag_lofi.min():.2f}, {mag_lofi.max():.2f}]")
    print(f"  Shadow effect: mean={shadow_effect.mean():.3f}, "
          f"max={shadow_effect.max():.3f}, |>0.1|={np.sum(np.abs(shadow_effect)>0.1)}/{n_dirs}")

    return directions, mag_hifi, mag_lofi, shadow_effect


# ��─ Surface interpolation ────────────────────────────────────────────
class BrightnessSurfaceLookup:
    """Fast nearest-neighbor lookup on the precomputed brightness surface."""

    def __init__(self, directions, mag_hifi, mag_lofi):
        self.tree = KDTree(directions)
        self.mag_hifi = mag_hifi
        self.mag_lofi = mag_lofi

    def query_hifi(self, query_dirs):
        """Look up hi-fi brightness for query directions."""
        _, idx = self.tree.query(query_dirs)
        return self.mag_hifi[idx]

    def query_lofi(self, query_dirs):
        """Look up lo-fi brightness for query directions."""
        _, idx = self.tree.query(query_dirs)
        return self.mag_lofi[idx]

    def query_both(self, query_dirs):
        """Return (hifi, lofi) at query directions."""
        _, idx = self.tree.query(query_dirs)
        return self.mag_hifi[idx], self.mag_lofi[idx]


# ── Phi helpers ─────────────────────────────────────────────��────────
def anchor_q_from_phi(phi, n_body, pab):
    R0, _ = Rotation.align_vectors([n_body], [pab])
    R_twist = Rotation.from_rotvec(phi * n_body)
    R_total = R_twist * R0
    q = R_total.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def find_truth_phi(q_truth_wxyz, anchor_normal, pab_anchor):
    R_truth = Rotation.from_quat([q_truth_wxyz[1], q_truth_wxyz[2],
                                   q_truth_wxyz[3], q_truth_wxyz[0]])
    fine_phi = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    best_err, best_phi = 999, 0
    for p in fine_phi:
        q = anchor_q_from_phi(p, anchor_normal, pab_anchor)
        R_c = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        e = np.rad2deg((R_c.inv() * R_truth).magnitude())
        if e < best_err:
            best_err, best_phi = e, p
    truth_deg = np.rad2deg(best_phi) % 360
    dists = np.minimum(np.abs(PHI_DEG - truth_deg), 360 - np.abs(PHI_DEG - truth_deg))
    return int(np.argmin(dists)), truth_deg, best_err


def discrimination_ratio(scores, truth_idx, lower_is_better=True):
    truth = scores[truth_idx]
    wrong = np.concatenate([scores[:truth_idx], scores[truth_idx + 1:]])
    if lower_is_better:
        return np.nanmin(wrong) / truth if truth > 0 else np.nan
    else:
        return truth / np.nanmax(wrong) if np.nanmax(wrong) > 0 else np.nan


# ── Full lo-fi evaluation (actual k1, k2) ───────────────────────────
def compute_full_lofi_mse(phi_rad, anchor_normal, pab_anchor, delta_R,
                          ctx, observed):
    """Compute lo-fi MSE over all 500 epochs using actual k1/k2 geometry."""
    q_anchor = anchor_q_from_phi(phi_rad, anchor_normal, pab_anchor)
    R_anchor = Rotation.from_quat([q_anchor[1], q_anchor[2],
                                    q_anchor[3], q_anchor[0]])
    R_all = delta_R * R_anchor
    R_mats = R_all.as_matrix()

    sun_vec = ctx.sun_pos - ctx.sat_pos
    sun_vec /= np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_vec = ctx.obs_pos - ctx.sat_pos
    obs_vec /= np.linalg.norm(obs_vec, axis=1, keepdims=True)

    k1_body = np.einsum('nij,nj->ni', R_mats, sun_vec)
    k2_body = np.einsum('nij,nj->ni', R_mats, obs_vec)

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
    art_mats = compute_rotation_matrices_from_angles(comp_angles, ctx.satellite)
    lit_status = create_no_shadow_lit_status(ctx.satellite, 500)

    mag, _, _, _, _, _ = generate_lightcurves(
        lit_status, k1_body, k2_body, ctx.obs_dist,
        ctx.satellite, ctx.epochs,
        pre_computed_matrices=art_mats, show_progress=False
    )
    return np.mean((mag - observed) ** 2)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(OUT_DIR / "diagnostic.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("m111 — Shadow-Corrected Isoshell Phi Discrimination")
    print("=" * 60)
    t_start = time.time()

    # ── Load satellite & context ─────────────────────────────────
    ctx = setup_experiment(n_observations=500, noise_sigma=NOISE_SIGMA)
    traj = np.load(str(DATA_DIR / "m046_trajectories.npz"))

    # ── Phase 1: Precompute brightness surface ───────────────────
    surface_file = OUT_DIR / "brightness_surface.npz"

    directions, mag_hifi_surf, mag_lofi_surf, shadow_eff = \
        precompute_brightness_surface(ctx.satellite, N_SURFACE_DIRS)

    np.savez(surface_file,
             directions=directions,
             mag_hifi=mag_hifi_surf,
             mag_lofi=mag_lofi_surf,
             shadow_effect=shadow_eff,
             n_dirs=N_SURFACE_DIRS)
    print(f"Saved: {surface_file}")

    lookup = BrightnessSurfaceLookup(directions, mag_hifi_surf, mag_lofi_surf)

    # ── Phase 2: Per-seed phi sweep ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2: Phi sweep on {len(ALL_SEEDS)} seeds")
    print(f"{'='*60}")

    results = {}
    unique_normals = traj['unique_normals']
    pab_j2000 = traj['pab_j2000']

    for seed in ALL_SEEDS:
        t_seed = time.time()
        status = SEED_STATUS[seed]
        print(f"\n--- Seed {seed} ({status}) ---")

        q_truth = traj['quaternions'][seed]
        pab_body = traj['pab_body'][seed]
        mag_hifi_truth = traj['mag_hifi'][seed]
        mag_lofi_truth = traj['mag_lofi'][seed]

        rng = np.random.default_rng(NOISE_SEED)
        observed = mag_hifi_truth + rng.normal(0, NOISE_SIGMA, 500)

        # Anchor
        peak_mask = traj['peak_seeds'] == seed
        seed_peaks = traj['peak_epochs'][peak_mask]
        anchor_ep = seed_peaks[np.argmin(mag_hifi_truth[seed_peaks])]
        pab_at_anchor = pab_body[anchor_ep]
        anchor_normal = unique_normals[np.argmax(unique_normals @ pab_at_anchor)]

        # Delta-R
        R_truth_all = Rotation.from_quat(q_truth[:, [1, 2, 3, 0]])
        R_anchor_inv = R_truth_all[anchor_ep].inv()
        delta_R = R_truth_all * R_anchor_inv

        # Truth phi
        truth_idx, truth_phi_deg, truth_err = find_truth_phi(
            q_truth[anchor_ep], anchor_normal, pab_j2000[anchor_ep])
        print(f"  Truth phi: {truth_phi_deg:.1f}° (idx {truth_idx}), "
              f"match err: {truth_err:.2f}°")
        print(f"  Anchor: ep {anchor_ep}, normal {anchor_normal}")

        # Shadow analysis for this seed
        shadow_this = np.abs(mag_hifi_truth - mag_lofi_truth)
        n_shadow = np.sum(shadow_this > 0.1)
        best_group = traj['best_group'][seed]

        # ── Sweep 72 phis ────────────────────────────────────────
        mse_surface_hifi = np.full(N_PHI, np.nan)
        mse_surface_lofi = np.full(N_PHI, np.nan)
        mse_full_lofi = np.full(N_PHI, np.nan)
        # Also: surface-corrected = lo-fi(actual) + shadow_correction(surface)
        mse_corrected = np.full(N_PHI, np.nan)

        for pi, phi_rad in enumerate(PHI_RAD):
            # Propagate attitude
            q_anchor = anchor_q_from_phi(phi_rad, anchor_normal,
                                          pab_j2000[anchor_ep])
            R_anchor = Rotation.from_quat([q_anchor[1], q_anchor[2],
                                            q_anchor[3], q_anchor[0]])
            R_all = delta_R * R_anchor
            R_mats = R_all.as_matrix()

            # Body-frame PAB directions
            sun_vec = ctx.sun_pos - ctx.sat_pos
            sun_vec /= np.linalg.norm(sun_vec, axis=1, keepdims=True)
            obs_vec = ctx.obs_pos - ctx.sat_pos
            obs_vec /= np.linalg.norm(obs_vec, axis=1, keepdims=True)
            pab_j = (sun_vec + obs_vec)
            pab_j /= np.linalg.norm(pab_j, axis=1, keepdims=True)

            pab_body_cand = np.einsum('nij,nj->ni', R_mats, pab_j)

            # Method A: Surface hi-fi lookup
            pred_hifi = lookup.query_hifi(pab_body_cand)
            mse_surface_hifi[pi] = np.mean((pred_hifi - observed) ** 2)

            # Method B: Surface lo-fi lookup
            pred_lofi = lookup.query_lofi(pab_body_cand)
            mse_surface_lofi[pi] = np.mean((pred_lofi - observed) ** 2)

            # Method C: Full lo-fi (actual k1, k2)
            mse_full_lofi[pi] = compute_full_lofi_mse(
                phi_rad, anchor_normal, pab_j2000[anchor_ep],
                delta_R, ctx, observed)

            # Method D: Corrected = full lo-fi + shadow offset from surface
            # shadow_offset(d) = B_hifi_surface(d) - B_lofi_surface(d)
            shadow_offset = pred_hifi - pred_lofi

            # Use the full lo-fi magnitudes for this phi (need to recompute)
            # Actually, approximate: corrected ≈ full_lofi + shadow_offset
            # But we'd need full_lofi magnitudes (not just MSE)...
            # Skip method D for now, it's complex

        # ── Results ──────────────────────────────────────────────
        def report_method(name, scores, truth_idx):
            rank = int(np.sum(scores < scores[truth_idx])) + 1
            disc = discrimination_ratio(scores, truth_idx)
            best_idx = np.argmin(scores)
            phi_err = min(abs(PHI_DEG[best_idx] - PHI_DEG[truth_idx]),
                          360 - abs(PHI_DEG[best_idx] - PHI_DEG[truth_idx]))
            print(f"  {name:30s}: rank {rank:2d}/72, disc {disc:.4f}, "
                  f"best phi {PHI_DEG[best_idx]:5.0f}° (err {phi_err:3.0f}°)")
            return rank, disc, float(PHI_DEG[best_idx])

        print(f"\n  Results (truth phi = {PHI_DEG[truth_idx]:.0f}°):")
        r_shifi, d_shifi, bp_shifi = report_method(
            "A: Surface hi-fi (NOVEL)", mse_surface_hifi, truth_idx)
        r_slofi, d_slofi, bp_slofi = report_method(
            "B: Surface lo-fi (zero-phase)", mse_surface_lofi, truth_idx)
        r_flofi, d_flofi, bp_flofi = report_method(
            "C: Full lo-fi (actual k1/k2)", mse_full_lofi, truth_idx)

        elapsed = time.time() - t_seed
        print(f"  Time: {elapsed:.1f}s")

        results[seed] = {
            'status': status,
            'truth_phi_deg': float(truth_phi_deg),
            'truth_phi_idx': int(truth_idx),
            'truth_phi_err': float(truth_err),
            'anchor_ep': int(anchor_ep),
            'anchor_normal': anchor_normal.tolist(),
            'n_shadow_epochs': int(n_shadow),
            'methods': {
                'surface_hifi': {'rank': r_shifi, 'disc': d_shifi, 'best_phi': bp_shifi},
                'surface_lofi': {'rank': r_slofi, 'disc': d_slofi, 'best_phi': bp_slofi},
                'full_lofi': {'rank': r_flofi, 'disc': d_flofi, 'best_phi': bp_flofi},
            },
            'mse_surface_hifi': mse_surface_hifi.tolist(),
            'mse_surface_lofi': mse_surface_lofi.tolist(),
            'mse_full_lofi': mse_full_lofi.tolist(),
        }

        # Save per-seed NPZ
        np.savez(OUT_DIR / f"seed_{seed:03d}.npz",
                 seed=seed, status=status,
                 phi_deg=PHI_DEG,
                 mse_surface_hifi=mse_surface_hifi,
                 mse_surface_lofi=mse_surface_lofi,
                 mse_full_lofi=mse_full_lofi,
                 truth_phi_idx=truth_idx,
                 truth_phi_deg=truth_phi_deg,
                 truth_phi_err=truth_err)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Seed':>5s} {'Status':>9s} | {'Surf-HiFi':>12s} | {'Surf-LoFi':>12s} | {'Full-LoFi':>12s}")
    print("-" * 60)
    for seed in ALL_SEEDS:
        r = results[seed]
        m = r['methods']
        print(f"{seed:5d} {r['status']:>9s} | "
              f"r={m['surface_hifi']['rank']:2d} d={m['surface_hifi']['disc']:.3f} | "
              f"r={m['surface_lofi']['rank']:2d} d={m['surface_lofi']['disc']:.3f} | "
              f"r={m['full_lofi']['rank']:2d} d={m['full_lofi']['disc']:.3f}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save summary JSON
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_DIR / 'summary.json'}")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Saved: {OUT_DIR / 'diagnostic.log'}")
