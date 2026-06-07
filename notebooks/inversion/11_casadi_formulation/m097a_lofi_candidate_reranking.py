#!/usr/bin/env python3
"""
m097a: Oracle-q0 Lo-fi Grid Re-ranking (100 seeds).

Question: With correct q0, does lo-fi full-curve MSE rank the correct omega
direction better than alignment cost within a structured grid?

Method: For each seed, take the 500 grid omegas from Exp 1 (oracle |w|),
evaluate lo-fi LC at (truth_q0, grid_omega), compute MSE against observed.
Compare alignment rank vs lo-fi rank.

Uses: m096_stage1 (truth q0, observed LC) + m096_exp1 (grid omegas, alignment costs)
"""

import sys, os, time, json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1 = RESULTS_DIR / "m096_stage1"
EXP1 = RESULTS_DIR / "m096_exp1_oracle_grid"
CKPT = RESULTS_DIR / "m097a_lofi_rerank"
CKPT.mkdir(exist_ok=True)

master = np.load(str(RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"),
                 allow_pickle=True)
I_tensor = master['inertia_tensor']

LOFI_WORKERS = 8

# ======================================================================
# CHECKPOINT SCHEMA (designed BEFORE computation code)
# ======================================================================
# Per seed: seed_{NNN}.npz
#   seed: int
#   valid: bool
#   alignment_costs: (N_DIRS,) float — from Exp 1
#   alignment_truth_rank: int — from Exp 1
#   lofi_mses: (N_DIRS,) float — NEW: full-curve lo-fi MSE per direction
#   lofi_sorted_idx: (N_DIRS,) int — argsort of lofi_mses
#   lofi_truth_rank: int — rank of first <5° direction by lo-fi MSE
#   lofi_truth_werr: float — omega dir error of that direction
#   lofi_win_mses: (N_DIRS,) float — windowed (360s around brightest) MSE
#   lofi_win_truth_rank: int — rank by windowed MSE
#   grid_omegas: (N_DIRS, 3) float — from Exp 1
#   true_omega_anchor: (3,) float
#   timing_s: float
#
# Summary: summary.json — per-seed comparison of alignment vs lo-fi rank
# ======================================================================


def omega_dir_err(w1, w2):
    d1 = w1 / np.linalg.norm(w1)
    d2 = w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


# Shared globals for pool workers (set once after model load)
_satellite = None
_sun = None
_obs = None
_sat = None
_dist = None
_art = None
_I = None


def _init_worker():
    """Workers inherit fork'd globals — no-op init."""
    pass


def eval_lofi_lc(args):
    """Generate lo-fi LC for a (q0, w0) state. Returns predicted magnitudes."""
    q0_wxyz, w0_rad, obs_times = args
    from src.computation.shadow_engine import create_no_shadow_lit_status as _no_shadow
    from src.computation.lightcurve_generator import generate_lightcurves as _gen_lc

    try:
        quats, _ = propagate_attitude(q0_wxyz, w0_rad, obs_times, "tumbling", _I)
        n_ep = len(quats)
        R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        sv = (_sun[:n_ep] - _sat[:n_ep])
        sv /= np.linalg.norm(sv, axis=1, keepdims=True)
        ov = (_obs[:n_ep] - _sat[:n_ep])
        ov /= np.linalg.norm(ov, axis=1, keepdims=True)
        k1 = np.einsum('nij,nj->ni', R_all, sv)
        k2 = np.einsum('nij,nj->ni', R_all, ov)
        lit = _no_shadow(_satellite, n_ep)
        pred, _, _, _, _, _ = _gen_lc(
            facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
            observer_distances=_dist, satellite=_satellite,
            epochs=np.arange(n_ep, dtype=float), pre_computed_matrices=_art,
            show_progress=False)
        return pred
    except Exception:
        return np.full(len(obs_times), 20.0)  # failure → dim = high MSE


if __name__ == '__main__':
    print("Loading satellite model...", flush=True)
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)

    _satellite = CTX.satellite
    _sun = CTX.sun_pos
    _obs = CTX.obs_pos
    _sat = CTX.sat_pos
    _dist = CTX.obs_dist
    _art = CTX.art_matrices
    _I = I_tensor

    ALL_SEEDS = list(range(100))
    summary = []
    t_total = time.time()

    print("=" * 70)
    print("m097a: Oracle-q0 Lo-fi Grid Re-ranking (500 dirs, 100 seeds)")
    print("=" * 70)

    for seed in ALL_SEEDS:
        # Load checkpoints
        s1_path = STAGE1 / f"seed_{seed:03d}.npz"
        e1_path = EXP1 / f"seed_{seed:03d}.npz"

        if not s1_path.exists() or not e1_path.exists():
            np.savez(str(CKPT / f"seed_{seed:03d}.npz"), seed=seed, valid=False)
            summary.append({'seed': seed, 'error': 'missing_checkpoint'})
            print(f"  seed {seed:3d}: SKIP (missing checkpoint)")
            continue

        s1 = np.load(str(s1_path), allow_pickle=True)
        e1 = np.load(str(e1_path), allow_pickle=True)

        if not bool(s1['valid']) or not bool(e1['valid']):
            np.savez(str(CKPT / f"seed_{seed:03d}.npz"), seed=seed, valid=False)
            summary.append({'seed': seed, 'error': 'invalid'})
            print(f"  seed {seed:3d}: SKIP (invalid)")
            continue

        true_q0 = s1['true_q0']            # wxyz
        obs_times = s1['obs_times']
        observed_lc = s1['observed_lc']
        true_omega_anchor = e1['true_omega_anchor']

        grid_omegas = e1['grid_omegas']     # (N_DIRS, 3)
        grid_costs = e1['grid_costs']       # (N_DIRS,)
        alignment_truth_rank = int(e1['truth_rank'])
        n_dirs = len(grid_omegas)

        # --- Lo-fi evaluation ---
        t0 = time.time()
        eval_args = [(true_q0, grid_omegas[wi], obs_times) for wi in range(n_dirs)]

        with Pool(LOFI_WORKERS) as pool:
            all_lcs = pool.map(eval_lofi_lc, eval_args)

        # Full-curve MSE
        lofi_mses = np.array([float(np.mean((lc - observed_lc) ** 2)) for lc in all_lcs])
        lofi_sorted_idx = np.argsort(lofi_mses)

        # Windowed MSE (360s around brightest peak)
        peaks_idx = s1['peaks_idx']
        peak_mags = s1['peak_mags']
        if len(peaks_idx) > 0:
            brightest_ep = peaks_idx[np.argmin(peak_mags)]
            brightest_time = obs_times[brightest_ep]
            win_mask = np.abs(obs_times - brightest_time) <= 180.0
            if win_mask.sum() > 5:
                lofi_win_mses = np.array([
                    float(np.mean((lc[win_mask] - observed_lc[win_mask]) ** 2))
                    for lc in all_lcs])
            else:
                lofi_win_mses = lofi_mses.copy()
        else:
            lofi_win_mses = lofi_mses.copy()

        lofi_win_sorted = np.argsort(lofi_win_mses)

        # Find truth rank by lo-fi MSE
        lofi_truth_rank = -1
        lofi_truth_werr = 999.0
        for i in range(len(lofi_sorted_idx)):
            ri = lofi_sorted_idx[i]
            werr = omega_dir_err(grid_omegas[ri], true_omega_anchor)
            if werr < 5.0:
                lofi_truth_rank = i + 1
                lofi_truth_werr = werr
                break

        # Find truth rank by windowed lo-fi MSE
        lofi_win_truth_rank = -1
        for i in range(len(lofi_win_sorted)):
            ri = lofi_win_sorted[i]
            werr = omega_dir_err(grid_omegas[ri], true_omega_anchor)
            if werr < 5.0:
                lofi_win_truth_rank = i + 1
                break

        timing = time.time() - t0

        # --- Save checkpoint ---
        np.savez(str(CKPT / f"seed_{seed:03d}.npz"),
                 seed=seed, valid=True,
                 alignment_costs=grid_costs,
                 alignment_truth_rank=alignment_truth_rank,
                 lofi_mses=lofi_mses,
                 lofi_sorted_idx=lofi_sorted_idx,
                 lofi_truth_rank=lofi_truth_rank,
                 lofi_truth_werr=lofi_truth_werr,
                 lofi_win_mses=lofi_win_mses,
                 lofi_win_truth_rank=lofi_win_truth_rank,
                 grid_omegas=grid_omegas,
                 true_omega_anchor=true_omega_anchor,
                 timing_s=timing)

        arrow = ""
        if alignment_truth_rank > 0 and lofi_truth_rank > 0:
            if lofi_truth_rank < alignment_truth_rank:
                arrow = f" IMPROVED ({alignment_truth_rank}→{lofi_truth_rank})"
            elif lofi_truth_rank > alignment_truth_rank:
                arrow = f" worse ({alignment_truth_rank}→{lofi_truth_rank})"

        print(f"  seed {seed:3d}: align={alignment_truth_rank:4d} "
              f"lofi={lofi_truth_rank:4d} "
              f"lofi_win={lofi_win_truth_rank:4d} "
              f"werr={lofi_truth_werr:5.1f}° "
              f"{timing:.1f}s{arrow}")

        summary.append({
            'seed': seed,
            'alignment_truth_rank': alignment_truth_rank,
            'lofi_truth_rank': lofi_truth_rank,
            'lofi_truth_werr': float(lofi_truth_werr),
            'lofi_win_truth_rank': lofi_win_truth_rank,
            'timing_s': float(timing),
        })

    save_results(str(CKPT / "summary.json"), summary)
    total_time = time.time() - t_total

    # ==================================================================
    # ANALYSIS
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS ({total_time:.0f}s total)")
    print(f"{'='*70}")

    valid = [r for r in summary if 'error' not in r]
    a_found = [r for r in valid if r['alignment_truth_rank'] > 0]
    l_found = [r for r in valid if r['lofi_truth_rank'] > 0]
    w_found = [r for r in valid if r['lofi_win_truth_rank'] > 0]

    print(f"\n  Valid seeds: {len(valid)}/100")
    print(f"  Truth in grid (any method): alignment={len(a_found)}, "
          f"lofi={len(l_found)}, windowed={len(w_found)}")

    for label, data, key in [("Alignment", a_found, 'alignment_truth_rank'),
                              ("Lo-fi MSE", l_found, 'lofi_truth_rank'),
                              ("Windowed MSE", w_found, 'lofi_win_truth_rank')]:
        if not data:
            continue
        ranks = [r[key] for r in data]
        print(f"\n  {label} rank (n={len(data)}):")
        print(f"    median={int(np.median(ranks))}, mean={np.mean(ranks):.0f}, "
              f"min={min(ranks)}, max={max(ranks)}")
        for t in [1, 3, 5, 10, 20, 50]:
            n = sum(1 for r in ranks if r <= t)
            print(f"    top-{t:2d}: {n:3d}/{len(data)} ({100*n/len(data):.0f}%)")

    # Head-to-head comparison (only seeds where both found truth)
    both = [r for r in valid
            if r['alignment_truth_rank'] > 0 and r['lofi_truth_rank'] > 0]
    if both:
        improved = sum(1 for r in both
                       if r['lofi_truth_rank'] < r['alignment_truth_rank'])
        worsened = sum(1 for r in both
                       if r['lofi_truth_rank'] > r['alignment_truth_rank'])
        tied = len(both) - improved - worsened
        print(f"\n  Head-to-head (n={len(both)} seeds where both found truth):")
        print(f"    Lo-fi better: {improved}")
        print(f"    Alignment better: {worsened}")
        print(f"    Tied: {tied}")

        # Median rank improvement
        a_med = np.median([r['alignment_truth_rank'] for r in both])
        l_med = np.median([r['lofi_truth_rank'] for r in both])
        print(f"    Median rank: alignment={a_med:.0f} → lofi={l_med:.0f}")

    print(f"\nSaved: {CKPT}/")
