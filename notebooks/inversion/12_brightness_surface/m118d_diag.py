#!/usr/bin/env python3
"""
m118 diag — Score all 5 cost variants against the seed-14 kernel,
save rich per-variant outputs, and emit diagnostic plots.

Reads kernel from m118_kernel, IPL data from isoshell_viewer/ipl_all_epochs.npz.

Outputs (per seed):
  seed_<s>/cost_<variant>.npz         — (D, M, C, P) cost tensor
  seed_<s>/topK_<variant>.npz         — top-10000 candidates
  seed_<s>/summary.json                — per-variant analytics
  seed_<s>/plots/<variant>_mollweide.png
  seed_<s>/plots/<variant>_topK_scatter.png
  seed_<s>/plots/truth_rank_summary.png

Usage:
  MICRO118_SEED=14 python3 m118_diag.py
"""

import sys
import os
import time
import json
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m118_costs as costs_lib
from m118_costs import (SCORERS, anchor_q_from_phi, _build_q_anchor_from_kernel)

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
IPL_NPZ = RESULTS_DIR / "isoshell_viewer" / "ipl_all_epochs.npz"

OUTPUT_DIR = Path(os.environ.get('MICRO118_OUTPUT_DIR',
                                 str(RESULTS_DIR / "m118")))

N_PHI = 360
TOP_K = 10000
BASIN_THRESH_DEG = 5.0

# --- Seed selection ---
_baseline10 = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]
_seeds_env = os.environ.get('MICRO118_SEEDS', os.environ.get('MICRO118_SEED', '14'))
if _seeds_env.lower() in ('all', 'baseline10', 'baseline'):
    SEEDS = list(_baseline10)
else:
    SEEDS = [int(s.strip()) for s in _seeds_env.split(',') if s.strip()]

# --- Variant selection ---
_all_variants = list(SCORERS.keys())
_variants_env = os.environ.get('MICRO118_VARIANTS', 'all')
if _variants_env.lower() == 'all':
    VARIANTS = list(_all_variants)
else:
    VARIANTS = [v.strip() for v in _variants_env.split(',') if v.strip()]
    unknown = [v for v in VARIANTS if v not in SCORERS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Available: {_all_variants}")

# --- Skip-if-exists ---
FORCE = os.environ.get('MICRO118_FORCE', '0') == '1'


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


# Per-seed log files opened inside process_seed; stdout is teed there.


def atomic_json_dump(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def omega_dir_err(w1, w2):
    n1 = np.linalg.norm(w1); n2 = np.linalg.norm(w2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 180.0
    d = float(np.clip(np.abs(np.dot(w1/n1, w2/n2)), 0, 1))
    return float(np.rad2deg(np.arccos(d)))


def reconstruct_q0(q_anchor_wxyz, omega_vec, anchor_time, I_tensor):
    """Backward-propagate q_anchor from t=anchor_time to t=0 under omega.

    Matches m103_hybrid lines 374-376: propagate with -omega across
    [0, anchor_time], take final state, then negate w.
    """
    bt = np.array([0.0, float(anchor_time)])
    qb, ob = propagate_attitude(q_anchor_wxyz, -omega_vec, bt,
                                "tumbling", I_tensor)
    return qb[-1].copy(), (-ob[-1]).copy()


def cluster_basins_by_omega(omega_vecs, threshold_deg):
    """Greedy clustering by direction angle. Returns list of cluster reps (indices)."""
    n = len(omega_vecs)
    if n == 0:
        return []
    reps = [0]
    for i in range(1, n):
        w = omega_vecs[i]
        ok = True
        for r in reps:
            if omega_dir_err(w, omega_vecs[r]) < threshold_deg:
                ok = False
                break
        if ok:
            reps.append(i)
    return reps


def mollweide_projection(dirs_xyz):
    """Convert unit direction vectors to (lon, lat) in Mollweide coords."""
    x, y, z = dirs_xyz[:, 0], dirs_xyz[:, 1], dirs_xyz[:, 2]
    lat = np.arcsin(np.clip(z, -1, 1))
    lon = np.arctan2(y, x)
    return lon, lat


def find_nearest_in_grid(omega_dirs, omega_mags, target_omega):
    """Return (dir_idx, mag_idx) closest to target omega."""
    tn = np.linalg.norm(target_omega)
    if tn < 1e-12:
        return 0, 0
    td = target_omega / tn
    # signed direction match (|dot| for twin-invariance would lose sign; use dot)
    dots = omega_dirs @ td
    di = int(np.argmax(dots))
    # magnitude
    mi = int(np.argmin(np.abs(omega_mags - tn)))
    return di, mi


def find_nearest_anchor_phi(q_anchor, truth_q0_at_anchor):
    """Return (c_idx, phi_idx) closest to truth attitude at anchor time."""
    # truth_q0_at_anchor: (4,) wxyz.  q_anchor: (C, P, 4).
    # Angle between quaternions q1, q2: 2*acos(|q1.q2|)
    dots = np.abs(q_anchor @ truth_q0_at_anchor)
    dots = np.clip(dots, 0, 1)
    ang = 2.0 * np.arccos(dots)  # (C, P)
    ci, pi = np.unravel_index(np.argmin(ang), ang.shape)
    return int(ci), int(pi), float(np.rad2deg(ang[ci, pi]))


def build_topK(cost_tensor, omega_dirs, omega_mags, q_anchor,
               anchor_time, I_tensor, truth_q0, truth_omega0, k=10000):
    D, M, C, P = cost_tensor.shape
    flat = cost_tensor.reshape(-1)
    k_eff = min(k, flat.size)
    # Use argpartition for speed then sort the subset
    idx_part = np.argpartition(flat, k_eff - 1)[:k_eff]
    idx_sorted = idx_part[np.argsort(flat[idx_part])]
    costs = flat[idx_sorted]

    multi = np.unravel_index(idx_sorted, (D, M, C, P))
    dir_idx, mag_idx, c_idx, phi_idx = multi

    out = {
        'dir_idx': dir_idx.astype(np.int32),
        'mag_idx': mag_idx.astype(np.int32),
        'c_idx':   c_idx.astype(np.int32),
        'phi_idx': phi_idx.astype(np.int32),
        'cost':    costs.astype(np.float32),
    }

    # Reconstruct q0, omega, errors
    q0s = np.zeros((k_eff, 4), dtype=np.float32)
    ws = np.zeros((k_eff, 3), dtype=np.float32)
    q0_err = np.zeros(k_eff, dtype=np.float32)
    w_dir_err = np.zeros(k_eff, dtype=np.float32)
    w_mag_err = np.zeros(k_eff, dtype=np.float32)

    true_w_mag = float(np.linalg.norm(truth_omega0))
    for i in range(k_eff):
        wd = omega_dirs[dir_idx[i]]
        wm = omega_mags[mag_idx[i]]
        omega = wd * wm
        q_anc = q_anchor[c_idx[i], phi_idx[i]]
        q0, w0 = reconstruct_q0(q_anc, omega, anchor_time, I_tensor)
        q0s[i] = q0
        ws[i] = w0
        q0_err[i] = attitude_error_deg(q0, truth_q0)
        w_dir_err[i] = omega_dir_err(w0, truth_omega0)
        if true_w_mag > 0:
            w_mag_err[i] = (np.linalg.norm(w0) - true_w_mag) / true_w_mag * 100
        else:
            w_mag_err[i] = 0.0

    out['q0']        = q0s
    out['omega']     = ws
    out['q0_err']    = q0_err
    out['w_dir_err'] = w_dir_err
    out['w_mag_err'] = w_mag_err
    return out


# --- Plotting ----------------------------------------------------------------
def plot_mollweide(cost_tensor, omega_dirs, omega_mags, variant, out_path):
    # Per-direction min over (mag, c, phi) then show
    D, M, C, P = cost_tensor.shape
    per_dir_min = cost_tensor.min(axis=(1, 2, 3))
    lon, lat = mollweide_projection(omega_dirs)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='mollweide')
    sc = ax.scatter(lon, lat, c=per_dir_min, s=8, cmap='viridis_r',
                    vmin=np.percentile(per_dir_min, 1),
                    vmax=np.percentile(per_dir_min, 99))
    ax.set_title(f"{variant}: min cost over (mag, c, phi) per direction")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='min cost')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_topK_scatter(topK, variant, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(topK['w_dir_err'], topK['q0_err'],
                    c=topK['cost'], s=3, cmap='viridis_r',
                    alpha=0.5,
                    vmin=np.percentile(topK['cost'], 1),
                    vmax=np.percentile(topK['cost'], 99))
    ax.set_xlabel('w_dir_err (deg)')
    ax.set_ylabel('q0_err (deg)')
    ax.set_title(f"{variant}: top-{len(topK['cost'])} error scatter")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='cost')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_truth_rank_summary(summary_per_variant, out_path):
    variants = list(summary_per_variant.keys())
    ranks = [summary_per_variant[v]['truth_rank'] for v in variants]
    cost_gaps = [summary_per_variant[v]['truth_cost_gap_pct'] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.barh(variants, ranks, color='steelblue')
    ax1.set_xscale('log')
    ax1.set_xlabel('truth rank (log scale)')
    ax1.set_title('Truth-neighbor rank per variant')
    ax1.grid(True, alpha=0.3, axis='x')

    ax2.barh(variants, cost_gaps, color='salmon')
    ax2.set_xlabel('cost gap to rank 1 (%)')
    ax2.set_title('Cost gap: truth-neighbor vs best')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# --- Main --------------------------------------------------------------------
def process_seed(seed, ipl, master, I_tensor):
    """Run all requested variants for one seed. Skips variants that already exist
    unless FORCE=1. Returns summary dict."""
    t_seed = time.time()
    ckpt_dir = OUTPUT_DIR / f"seed_{seed:03d}"
    plots_dir = ckpt_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed log — append so multiple invocations accumulate history
    _log_file = open(str(ckpt_dir / "diag_run.log"), "a")
    _saved_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, _log_file)

    print("=" * 70)
    print(f"m118 diag — seed {seed}")
    print(f"  CKPT_DIR={ckpt_dir}")
    print(f"  variants requested: {VARIANTS}")
    print(f"  FORCE={FORCE}")
    print("=" * 70)

    kernel_path = ckpt_dir / "kernel.npz"
    if not kernel_path.exists():
        print(f"  SKIP seed {seed}: kernel not found at {kernel_path}")
        return None
    print(f"Loading kernel: {kernel_path}")
    kernel = np.load(str(kernel_path), allow_pickle=True)

    omega_dirs = kernel['omega_dirs']
    omega_mags = kernel['omega_mags']
    anchor_time = float(kernel['anchor_time'])
    anchor_epoch = int(kernel['anchor_epoch'])
    truth_q0 = kernel['truth_q0']
    truth_omega0 = kernel['truth_omega0']

    D = len(omega_dirs)
    M = len(omega_mags)

    # Build anchor quaternion grid once (shared across variants)
    phi_arr = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    q_anchor = _build_q_anchor_from_kernel(kernel, phi_arr)  # (C, P, 4)
    C = q_anchor.shape[0]
    P = len(phi_arr)
    print(f"\nAnchor attitude grid: C={C}  P={P}")

    # Truth attitude at anchor time (for truth-nearest lookup in (c, phi) grid)
    q_truth_at_anchor_all, _ = propagate_attitude(
        truth_q0, truth_omega0, np.array([0.0, anchor_time]),
        "tumbling", I_tensor)
    truth_q_at_anchor = q_truth_at_anchor_all[-1]

    # Truth location in (dir, mag) grid
    truth_dir_idx, truth_mag_idx = find_nearest_in_grid(
        omega_dirs, omega_mags, truth_omega0)
    truth_c_idx, truth_phi_idx, truth_att_angle = find_nearest_anchor_phi(
        q_anchor, truth_q_at_anchor)
    print(f"\nTruth location in grid:")
    print(f"  omega: dir={truth_dir_idx} mag={truth_mag_idx}  "
          f"(true |omega|={np.linalg.norm(truth_omega0):.5f} rad/s, "
          f"nearest grid mag={omega_mags[truth_mag_idx]:.5f})")
    print(f"  attitude: c={truth_c_idx} phi={truth_phi_idx}  "
          f"(residual {truth_att_angle:.2f} deg)")

    # --- Score all variants ---
    summary_per_variant = {}
    for variant in VARIANTS:
        print(f"\n{'=' * 70}")
        print(f"VARIANT: {variant}")
        print(f"{'=' * 70}")

        scorer = SCORERS[variant]
        mode = 'extended' if variant == 'ipl_centroid_weighted_ext' else 'spec_peaks_only'
        cost_path = ckpt_dir / f"cost_{variant}.npz"
        topK_path = ckpt_dir / f"topK_{variant}.npz"

        if cost_path.exists() and not FORCE:
            print(f"  SKIP (cost exists): {cost_path}  — set MICRO118_FORCE=1 to recompute")
            cost_tensor = np.load(str(cost_path))['cost']
            t_score = 0.0
        else:
            t0 = time.time()
            cost_tensor = scorer(kernel, ipl, phi_arr, mode=mode)
            t_score = time.time() - t0
            print(f"  Scoring time: {t_score:.1f}s")
            print(f"  Shape: {cost_tensor.shape}  dtype={cost_tensor.dtype}  "
                  f"size={cost_tensor.nbytes / 1024 / 1024:.1f} MB")
            np.savez_compressed(str(cost_path), cost=cost_tensor)
            print(f"  Saved: {cost_path}")

        if topK_path.exists() and not FORCE:
            print(f"  Loading existing topK: {topK_path}")
            topK = dict(np.load(str(topK_path)))
        else:
            topK = build_topK(cost_tensor, omega_dirs, omega_mags, q_anchor,
                              anchor_time, I_tensor, truth_q0, truth_omega0, k=TOP_K)
            np.savez_compressed(str(topK_path), **topK)
            print(f"  Saved: {topK_path}")

        # Truth-neighbor rank
        truth_cost = float(cost_tensor[truth_dir_idx, truth_mag_idx,
                                       truth_c_idx, truth_phi_idx])
        flat = cost_tensor.reshape(-1)
        rank1_cost = float(flat.min())
        truth_rank = int((flat < truth_cost).sum()) + 1
        cost_gap_pct = (truth_cost - rank1_cost) / max(rank1_cost, 1e-12) * 100

        # Best omega in pool (min w_dir_err in top-K)
        best_omega_in_pool_idx = int(np.argmin(topK['w_dir_err']))
        best_omega_w_dir_err = float(topK['w_dir_err'][best_omega_in_pool_idx])
        best_omega_q0_err = float(topK['q0_err'][best_omega_in_pool_idx])
        best_omega_w_mag_err = float(topK['w_mag_err'][best_omega_in_pool_idx])

        # Basin count in top-1000
        top1000_omega = (omega_dirs[topK['dir_idx'][:1000]] *
                         omega_mags[topK['mag_idx'][:1000]][:, None])
        basin_reps = cluster_basins_by_omega(top1000_omega, BASIN_THRESH_DEG)
        n_basins = len(basin_reps)

        # Rank-1 info
        rank1_q0_err = float(topK['q0_err'][0])
        rank1_w_dir_err = float(topK['w_dir_err'][0])
        rank1_w_mag_err = float(topK['w_mag_err'][0])
        truth_basin_recovered = bool(best_omega_w_dir_err < BASIN_THRESH_DEG)

        print(f"\n  Rank 1: cost={rank1_cost:.4e}  q0_err={rank1_q0_err:.2f}  "
              f"w_dir_err={rank1_w_dir_err:.2f}  w_mag_err={rank1_w_mag_err:.2f}%")
        print(f"  Truth-neighbor: rank={truth_rank}  "
              f"cost={truth_cost:.4e} (gap {cost_gap_pct:.2f}%)")
        print(f"  Best omega in top-K: w_dir={best_omega_w_dir_err:.2f}°  "
              f"q0={best_omega_q0_err:.2f}°")
        print(f"  Basins in top-1000: {n_basins}  "
              f"(truth basin recovered: {truth_basin_recovered})")

        summary_per_variant[variant] = {
            'score_time_s': float(t_score),
            'cost_tensor_shape': list(cost_tensor.shape),
            'cost_tensor_mb': float(cost_tensor.nbytes / 1024 / 1024),
            'rank1_cost': rank1_cost,
            'rank1_q0_err_deg': rank1_q0_err,
            'rank1_w_dir_err_deg': rank1_w_dir_err,
            'rank1_w_mag_err_pct': rank1_w_mag_err,
            'truth_cost': truth_cost,
            'truth_rank': int(truth_rank),
            'truth_cost_gap_pct': float(cost_gap_pct),
            'best_omega_w_dir_err_deg': best_omega_w_dir_err,
            'best_omega_q0_err_deg': best_omega_q0_err,
            'best_omega_w_mag_err_pct': best_omega_w_mag_err,
            'n_basins_top1000': int(n_basins),
            'truth_basin_recovered': truth_basin_recovered,
            'mode': mode,
            'BEST_OMEGA_RANK': int(best_omega_in_pool_idx + 1),
            'TRUTH_OMEGA_BASIN_RECOVERED': truth_basin_recovered,
            'Q0_ERR_AT_RANK_1_DEG': rank1_q0_err,
        }

        # Per-variant plots
        plot_mollweide(cost_tensor, omega_dirs, omega_mags, variant,
                       plots_dir /f"{variant}_mollweide.png")
        plot_topK_scatter(topK, variant,
                          plots_dir /f"{variant}_topK_scatter.png")

        # Free memory
        del cost_tensor

    # --- Cross-variant summary ---
    plot_truth_rank_summary(summary_per_variant,
                            plots_dir /"truth_rank_summary.png")

    summary = {
        'seed': int(seed),
        'anchor_epoch': int(anchor_epoch),
        'anchor_time_s': float(anchor_time),
        'N_DIRS': int(D),
        'N_MAGS': int(M),
        'N_PHI': int(P),
        'N_ANCHOR_CENTROIDS': int(C),
        'TOP_K': int(TOP_K),
        'BASIN_THRESH_DEG': float(BASIN_THRESH_DEG),
        'truth_dir_idx': int(truth_dir_idx),
        'truth_mag_idx': int(truth_mag_idx),
        'truth_c_idx': int(truth_c_idx),
        'truth_phi_idx': int(truth_phi_idx),
        'truth_att_residual_deg': float(truth_att_angle),
        'variants': summary_per_variant,
        'total_time_s': float(time.time() - t_global),
    }

    summary_path = ckpt_dir / "summary.json"
    atomic_json_dump(summary, summary_path)
    print(f"\nSaved: {summary_path}")

    # --- Final classification printout ---
    print(f"\n{'=' * 70}")
    print(f"CLASSIFICATION (seed {seed})")
    print(f"{'=' * 70}")
    for v in VARIANTS:
        s = summary_per_variant[v]
        print(f"  {v:>32s}: "
              f"BEST_OMEGA_RANK={s['BEST_OMEGA_RANK']:<5d}  "
              f"TRUTH_BASIN_RECOVERED={str(s['TRUTH_OMEGA_BASIN_RECOVERED']):>5s}  "
              f"Q0_ERR_AT_RANK_1={s['Q0_ERR_AT_RANK_1_DEG']:.2f}°")

    print(f"\nSeed {seed} total: {time.time() - t_seed:.1f}s")
    sys.stdout = _saved_stdout
    _log_file.close()
    return summary


def main():
    t_global = time.time()
    print(f"m118 diag — seeds={SEEDS}  variants={VARIANTS}  FORCE={FORCE}")

    print(f"Loading IPL data: {IPL_NPZ}")
    ipl = np.load(str(IPL_NPZ), allow_pickle=True)
    master = np.load(str(RESULTS_DIR / "m046_trajectories" /
                         "m046_trajectories.npz"), allow_pickle=True)
    I_tensor = master['inertia_tensor']

    for seed in SEEDS:
        try:
            process_seed(seed, ipl, master, I_tensor)
        except Exception as e:
            print(f"\nERROR processing seed {seed}: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}\nAll seeds total: {time.time() - t_global:.1f}s\n{'=' * 70}")


if __name__ == '__main__':
    main()
    sys.stdout = sys.__stdout__
    _log_file.close()
