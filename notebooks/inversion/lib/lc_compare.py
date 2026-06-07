#!/usr/bin/env python3
"""
Light curve comparison tool — truth vs estimated overlay plot.

Generates one PNG per seed showing truth and estimated hi-fi light curves
overlaid on the same axes, with error metrics in the title.

Data sources:
  Truth LC:     m046_trajectories.npz (never regenerated)
  Estimated LC: pred_lc.npy in result dir (cached), or regenerated from
                result.json winner q0/w0 via full hi-fi (shadows + BRDF)
  Observed LC:  truth + N(0, 0.05) noise (optional, --show-observed)

Usage:
  python3 lc_compare.py 27               # single seed
  python3 lc_compare.py 6 24 27 93       # multiple seeds
  python3 lc_compare.py --prefix m093 27
  python3 lc_compare.py --show-observed 27
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from lib.traj_source import load_truth, VALID_SOURCES
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"


def generate_hifi_lc(q0_wxyz, w0_rad, obs_times, ctx):
    """Generate hi-fi light curve from initial state."""
    I_tensor = ctx['I_tensor']
    satellite = ctx['satellite']
    sun_pos = ctx['sun_pos']
    obs_pos = ctx['obs_pos']
    sat_pos = ctx['sat_pos']
    obs_dist = ctx['obs_dist']
    art_matrices = ctx['art_matrices']

    from src.computation.shadow_engine import compute_shadows
    from src.computation.lightcurve_generator import generate_lightcurves

    quats, _ = propagate_attitude(q0_wxyz, w0_rad, obs_times, "tumbling", I_tensor)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()

    sv = sun_pos[:n_ep] - sat_pos[:n_ep]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = obs_pos[:n_ep] - sat_pos[:n_ep]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)

    lit = compute_shadows(satellite=satellite, k1_vectors=k1,
                          explicit_component_matrices=art_matrices,
                          show_progress=False)
    pred_mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=obs_dist,
        satellite=satellite, epochs=np.arange(n_ep, dtype=float),
        pre_computed_matrices=art_matrices, show_progress=False)
    return pred_mags


def get_estimated_lc(seed, prefix, obs_times, ctx, candidate=0,
                     traj_source='m046'):
    """Get estimated LC — from cache or regenerate.

    Parameters
    ----------
    candidate : int
        Hi-fi candidate rank (0 = winner, 1 = rank #2, etc.).
        Cache is per-candidate: pred_lc.npy for winner, pred_lc_1.npy for rank #2, etc.
    traj_source : str
        'm046' or 'm048'. Reserved for future caching that keys on source; not
        currently used in the cache filename since prefix already disambiguates.
    """
    result_dir = RESULTS_DIR / f"{prefix}_seed{seed:03d}"
    cache_suffix = "" if candidate == 0 else f"_{candidate}"
    cache_path = result_dir / f"pred_lc{cache_suffix}.npy"

    # Try cache first
    if cache_path.exists():
        print(f"  Seed {seed} (candidate {candidate}): loading cached predicted LC")
        return np.load(str(cache_path))

    # Regenerate from result.json
    result_path = result_dir / "result.json"
    if not result_path.exists():
        print(f"  WARNING: {result_path} not found, skipping seed {seed}")
        return None

    with open(result_path) as f:
        result = json.load(f)

    if candidate == 0:
        source = result['winner']
    else:
        hifi = result.get('hifi_candidates', [])
        if candidate >= len(hifi):
            print(f"  WARNING: seed {seed} has only {len(hifi)} candidates, "
                  f"requested #{candidate}")
            return None
        source = hifi[candidate]
        if 'q0_wxyz' not in source:
            print(f"  WARNING: seed {seed} candidate {candidate} missing q0_wxyz "
                  f"(pipeline didn't save full state)")
            return None

    q0 = np.array(source['q0_wxyz'])
    w0 = np.array(source['w0_rad'])

    print(f"  Seed {seed} (candidate {candidate}): regenerating hi-fi LC from q0/w0...")
    pred_lc = generate_hifi_lc(q0, w0, obs_times, ctx)

    # Cache for future use
    np.save(str(cache_path), pred_lc)
    print(f"  Cached to {cache_path}")
    return pred_lc


def load_hifi_context(seed, traj_source='m046'):
    """Load satellite model and seed-specific geometry for hi-fi LC generation.

    For m046 the geometry is identical across all seeds (single fixed window)
    so a given ctx could be reused — but this function always builds per-call
    for simplicity; m048 REQUIRES a per-seed ctx since start_et varies.
    """
    print(f"Loading satellite model (source={traj_source}, seed={seed})...",
          flush=True)
    truth = load_truth(seed, traj_source)
    exp = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=42,
        true_omega_deg=(0.5, -0.3, 2.0),
        end_time_utc=truth['end_time_utc'],
        start_et=truth['start_et'],
        duration_s=truth['duration_s'],
        skip_true_lc=True)
    return {
        'satellite': exp.satellite,
        'sun_pos': exp.sun_pos,
        'obs_pos': exp.obs_pos,
        'sat_pos': exp.sat_pos,
        'obs_dist': exp.obs_dist,
        'art_matrices': exp.art_matrices,
        'I_tensor': truth['inertia_tensor'],
    }


def generate_lc_comparison(seeds, prefix='m090', show_observed=False,
                           output_dir=None, candidate=0, traj_source='m046'):
    """Generate LC comparison plots for the given seeds.

    Parameters
    ----------
    traj_source : str
        'm046' (legacy single-window) or 'm048' (per-seed random start times).
        Controls where truth q0/omega/mag_hifi come from and which start_et
        is threaded into setup_experiment for hi-fi regeneration.

    Returns list of saved PNG paths.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    # Per-seed truth load via traj_source (m046 all seeds share one obs_times,
    # but load_truth returns per-seed values so this works uniformly).
    cache_suffix = "" if candidate == 0 else f"_{candidate}"

    saved = []
    for seed in seeds:
        result_dir = RESULTS_DIR / f"{prefix}_seed{seed:03d}"
        result_path = result_dir / "result.json"
        if not result_path.exists():
            print(f"WARNING: {result_path} not found, skipping seed {seed}")
            continue

        with open(result_path) as f:
            result = json.load(f)

        # Truth + geometry from trajectory source.
        truth = load_truth(seed, traj_source)
        truth_lc = truth['mag_hifi']
        obs_times = truth['observation_times']
        hours = (obs_times - obs_times[0]) / 3600.0

        # Estimated LC: cached or regenerate via per-seed ctx.
        needs_regen = not (result_dir / f"pred_lc{cache_suffix}.npy").exists()
        ctx = load_hifi_context(seed, traj_source) if needs_regen else None
        est_lc = get_estimated_lc(seed, prefix, obs_times, ctx,
                                  candidate=candidate,
                                  traj_source=traj_source)
        if est_lc is None:
            continue

        # Error metrics — from the selected candidate
        if candidate == 0:
            w = result['winner']
        else:
            w = result['hifi_candidates'][candidate]
        q0_err = w['q0_err']
        w_dir_err = w['w0_err']
        w_mag_err = w.get('w_mag_err_pct', float('nan'))
        hifi_mse = w.get('hifi', np.mean((est_lc - truth_lc) ** 2))

        # Residual
        rms = np.sqrt(np.mean((est_lc - truth_lc) ** 2))

        # True and estimated state
        true_q0 = truth['q0_wxyz']
        true_w0 = truth['omega0_rad']
        est_q0 = np.array(w['q0_wxyz'])
        est_w0_dps = np.array(w['w0_dps']) if 'w0_dps' in w else np.rad2deg(np.array(w['w0_rad']))
        true_w0_dps = np.rad2deg(true_w0)

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={'height_ratios': [3, 1]},
                                        sharex=True)

        if show_observed:
            rng = np.random.default_rng(42)
            observed_lc = truth_lc + rng.normal(0, 0.05, len(truth_lc))
            ax1.plot(hours, observed_lc, '.', color='#cccccc', markersize=2,
                     label='Observed (noisy)', zorder=1)

        ax1.plot(hours, truth_lc, '.', color='#1f77b4', markersize=3,
                 label='Truth', zorder=2)
        ax1.plot(hours, est_lc, '-', color='#d62728', linewidth=1.0,
                 alpha=0.85, label='Estimate', zorder=3)

        ax1.set_ylabel('Apparent Magnitude')
        ax1.invert_yaxis()
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        experiment = result.get('experiment', prefix)
        cand_label = f" (candidate #{candidate})" if candidate > 0 else ""
        ax1.set_title(
            f'{experiment} seed {seed}{cand_label} — '
            f'q0 err: {q0_err:.1f}°, w_dir err: {w_dir_err:.1f}°, '
            f'w_mag err: {w_mag_err:+.1f}%, RMS: {rms:.3f} mag\n'
            f'True  q0=[{true_q0[0]:+.3f}, {true_q0[1]:+.3f}, {true_q0[2]:+.3f}, {true_q0[3]:+.3f}]  '
            f'w=[{true_w0_dps[0]:+.2f}, {true_w0_dps[1]:+.2f}, {true_w0_dps[2]:+.2f}] dps\n'
            f'Est   q0=[{est_q0[0]:+.3f}, {est_q0[1]:+.3f}, {est_q0[2]:+.3f}, {est_q0[3]:+.3f}]  '
            f'w=[{est_w0_dps[0]:+.2f}, {est_w0_dps[1]:+.2f}, {est_w0_dps[2]:+.2f}] dps',
            fontsize=9, fontfamily='monospace', loc='left')

        # Residual panel
        residual = est_lc - truth_lc
        ax2.plot(hours, residual, '-', color='#2ca02c', linewidth=0.8)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.fill_between(hours, residual, 0, alpha=0.2, color='#2ca02c')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Residual (mag)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-max(2.0, 1.2 * np.max(np.abs(residual))),
                      max(2.0, 1.2 * np.max(np.abs(residual))))

        plt.tight_layout()
        cand_suffix = f"_cand{candidate}" if candidate > 0 else ""
        out_path = Path(output_dir) / f"{prefix}_seed{seed:03d}_lc_compare{cand_suffix}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()
        saved.append(out_path)

    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Light curve comparison plot')
    parser.add_argument('seeds', nargs='+', type=int, help='Trajectory seeds')
    parser.add_argument('--prefix', default='m090',
                        help='Result directory prefix (default: m090)')
    parser.add_argument('--show-observed', action='store_true',
                        help='Also show the noisy observed LC')
    parser.add_argument('--candidate', type=int, default=0,
                        help='Hi-fi candidate rank (0=winner, 1=rank#2, etc.)')
    parser.add_argument('--traj-source', default='m046', choices=VALID_SOURCES,
                        help="Trajectory source: 'm046' (legacy single-window) "
                             "or 'm048' (per-seed random starts). Default m046.")
    args = parser.parse_args()
    generate_lc_comparison(args.seeds, prefix=args.prefix,
                           show_observed=args.show_observed,
                           candidate=args.candidate,
                           traj_source=args.traj_source)
