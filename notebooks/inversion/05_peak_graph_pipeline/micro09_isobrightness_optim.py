#!/usr/bin/env python3
"""
Micro-09 — L-BFGS-B Iso-brightness Optimization at Peak 1.

Tests whether gradient-based optimization finds attitudes closer to truth
than random SO(3) sampling (micro-06/07: nearest at 5.95 deg from 1M samples).

100 random SO(3) seeds → L-BFGS-B minimizing |B(q) - B_obs|² at peak 183.
Rodrigues vector parameterisation (3 DOF, unconstrained).
Lo-fi brightness (no shadows), same pipeline as micro-08.
"""
import sys, time, numpy as np
from pathlib import Path
from multiprocessing import get_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from lib.experiment_setup import setup_experiment, save_results, attitude_error_deg
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

# ── Config ──
SEED = 42
PEAK1 = 183
N_SEEDS = 100
N_WORKERS = 8
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# Module-level global (inherited by forked workers)
CTX = None


def lofi_single(q_wxyz, eidx):
    """Lo-fi brightness for one quaternion at one epoch."""
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    sv = CTX.sun_pos[eidx] - CTX.sat_pos[eidx]
    ov = CTX.obs_pos[eidx] - CTX.sat_pos[eidx]
    k1 = R @ sv; k1 /= np.linalg.norm(k1)
    k2 = R @ ov; k2 /= np.linalg.norm(k2)
    art = {c: m[eidx:eidx+1] for c, m in CTX.art_matrices.items()}
    lit = create_no_shadow_lit_status(CTX.satellite, 1)
    mags, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1.reshape(1, 3),
        k2_vectors_array=k2.reshape(1, 3),
        observer_distances=np.array([CTX.obs_dist[eidx]]),
        satellite=CTX.satellite, epochs=np.array([0.0]),
        pre_computed_matrices=art, generate_no_shadow=False,
        animate=False, show_progress=False)
    return float(mags[0])


def iso_objective(rotvec, target_mag, eidx):
    """Squared brightness error for L-BFGS-B."""
    q_sci = Rotation.from_rotvec(rotvec).as_quat()  # (x,y,z,w)
    q_wxyz = np.array([q_sci[3], q_sci[0], q_sci[1], q_sci[2]])
    return (lofi_single(q_wxyz, eidx) - target_mag) ** 2


def run_one_seed(args):
    """Run one L-BFGS-B from a random SO(3) seed."""
    si, rv_init, target_mag = args
    try:
        res = minimize(iso_objective, rv_init, args=(target_mag, PEAK1),
                       method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
        q_sci = Rotation.from_rotvec(res.x).as_quat()
        q_wxyz = np.array([q_sci[3], q_sci[0], q_sci[1], q_sci[2]])
        final_mag = lofi_single(q_wxyz, PEAK1)
        ae = attitude_error_deg(q_wxyz, CTX.true_quaternions[PEAK1])
        return {'seed': si, 'rotvec': res.x.tolist(), 'q_wxyz': q_wxyz.tolist(),
                'final_mag': float(final_mag), 'residual': float(abs(final_mag - target_mag)),
                'att_error_deg': float(ae), 'nfev': res.nfev, 'success': bool(res.success)}
    except Exception as e:
        return {'seed': si, 'error': str(e)}


if __name__ == '__main__':
    t0 = time.time()
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    print(f"Setup: {time.time()-t0:.1f}s", flush=True)

    # Reference brightness at peak 1
    obs_b1 = CTX.observed_lc[PEAK1]
    true_b1 = CTX.true_lc[PEAK1]
    print(f"Peak 1 (epoch {PEAK1}): obs={obs_b1:.4f}, true={true_b1:.4f}", flush=True)

    # Random SO(3) seeds
    np.random.seed(SEED + 9)
    seeds = Rotation.random(N_SEEDS)
    args_list = [(i, seeds[i].as_rotvec(), obs_b1) for i in range(N_SEEDS)]

    # ── Run L-BFGS-B in parallel ──
    print(f"\nRunning {N_SEEDS} L-BFGS-B optimizations ({N_WORKERS} workers)...", flush=True)
    t1 = time.time()
    results = []
    with get_context('fork').Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_one_seed, args_list, chunksize=4):
            results.append(r)
            if len(results) % 25 == 0:
                el = time.time() - t1
                good = sum(1 for x in results if x.get('residual', 999) < 0.01)
                print(f"  [{len(results)}/{N_SEEDS}] {good} converged, {el:.0f}s", flush=True)
    opt_time = time.time() - t1
    print(f"Optimization: {opt_time:.1f}s ({opt_time/N_SEEDS:.2f}s/seed)", flush=True)

    # ── Analysis ──
    converged = [r for r in results if 'error' not in r and r['residual'] < 0.01]
    failed = [r for r in results if 'error' in r]
    print(f"\nConverged (|resid| < 0.01 mag): {len(converged)}/{N_SEEDS}"
          f" ({len(failed)} errors)")

    if not converged:
        print("No converged results!"); sys.exit(1)

    att_errors = np.array([r['att_error_deg'] for r in converged])
    residuals = np.array([r['residual'] for r in converged])
    nfevs = np.array([r['nfev'] for r in converged])

    print(f"Att errors (deg): min={att_errors.min():.2f}, "
          f"median={np.median(att_errors):.1f}, max={att_errors.max():.1f}")
    print(f"Residuals (mag): min={residuals.min():.6f}, max={residuals.max():.6f}")
    print(f"Func evals: mean={nfevs.mean():.0f}, max={nfevs.max()}")

    # ── Deduplicate: cluster within 1 deg ──
    rotvecs = np.array([r['rotvec'] for r in converged])
    rots = Rotation.from_rotvec(rotvecs)
    n = len(rots)
    if n > 1:
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.rad2deg((rots[i].inv() * rots[j]).magnitude())
                dist[i, j] = dist[j, i] = d
        labels = fcluster(linkage(squareform(dist), method='complete'),
                          t=1.0, criterion='distance')
    else:
        labels = np.array([1])

    n_unique = len(set(labels))
    print(f"\nUnique attitudes (1° clustering): {n_unique}")
    print(f"\n{'Cl':>3} {'#':>3} {'AttErr':>8} {'Resid':>10} {'NfEv':>5}")
    for cl in sorted(set(labels)):
        members = [converged[i] for i, l in enumerate(labels) if l == cl]
        best = min(members, key=lambda x: x['residual'])
        print(f"  {cl:>3} {len(members):>3} {best['att_error_deg']:>8.2f} "
              f"{best['residual']:>10.6f} {best['nfev']:>5}")

    # ── Key comparison ──
    nearest = min(converged, key=lambda x: x['att_error_deg'])
    print(f"\n{'='*60}")
    print(f"NEAREST TO TRUTH: {nearest['att_error_deg']:.2f} deg "
          f"(residual={nearest['residual']:.6f})")
    print(f"Micro-06/07 random (1M samples): 5.95 deg nearest")
    if nearest['att_error_deg'] > 0:
        print(f"Improvement: {5.95/nearest['att_error_deg']:.1f}x closer "
              f"with {N_SEEDS} seeds vs 1M random")

    # Distribution
    bins = [0, 1, 2, 5, 10, 20, 45, 90, 180]
    print(f"\nAttitude error distribution:")
    for i in range(len(bins) - 1):
        count = int(np.sum((att_errors >= bins[i]) & (att_errors < bins[i + 1])))
        if count > 0:
            print(f"  [{bins[i]:>3}°, {bins[i+1]:>3}°): {count}")

    # ── Save checkpoint ──
    np.savez(RESULTS_DIR / 'micro09_isobrightness_optim.npz',
             converged_q_wxyz=np.array([r['q_wxyz'] for r in converged]),
             converged_rotvecs=rotvecs, att_errors=att_errors,
             residuals=residuals, cluster_labels=labels,
             peak_idx=PEAK1, obs_b1=obs_b1, seed=SEED)
    save_results(RESULTS_DIR / 'micro09_isobrightness_optim.json', {
        'config': {'peak': PEAK1, 'n_seeds': N_SEEDS, 'seed': SEED,
                   'method': 'L-BFGS-B', 'maxiter': 50},
        'n_converged': len(converged), 'n_unique_clusters': n_unique,
        'nearest_to_truth_deg': float(nearest['att_error_deg']),
        'att_error_stats': {'min': float(att_errors.min()),
                            'median': float(np.median(att_errors)),
                            'max': float(att_errors.max()),
                            'mean': float(att_errors.mean())},
        'comparison': {'micro07_random_1M_nearest_deg': 5.95,
                       'micro09_optim_nearest_deg': float(nearest['att_error_deg'])},
        'runtime_s': round(time.time() - t0, 1)})

    print(f"\nSaved to {RESULTS_DIR}/micro09_isobrightness_optim.*")
    print(f"Total runtime: {time.time()-t0:.1f}s")
