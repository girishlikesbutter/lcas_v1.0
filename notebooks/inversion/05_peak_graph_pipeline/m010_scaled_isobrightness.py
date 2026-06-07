#!/usr/bin/env python3
"""
Micro-10 — Scaled iso-brightness optimization (10,000 seeds).

Scale up from micro-09 (100 seeds → 71 basins, nearest 25.64 deg) to 10,000 seeds.
Save checkpoint compatible with micro-08 pipeline (m007_stage1.npz format).
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
SEED, PEAK1, PEAK2 = 42, 183, 260
N_SEEDS, N_WORKERS = 10_000, 8
DPHI = 1e-5
RESULTS_DIR = Path('data/results/inversion_diagnostics')
CTX = None  # module-level global (inherited by forked workers)


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


def lofi_batch(k1, k2, eidx):
    """Vectorised lo-fi brightness for N attitudes at one epoch."""
    N = len(k1)
    art = {c: np.tile(m[eidx:eidx+1], (N, 1, 1)) for c, m in CTX.art_matrices.items()}
    lit = create_no_shadow_lit_status(CTX.satellite, N)
    mags, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.full(N, CTX.obs_dist[eidx]),
        satellite=CTX.satellite, epochs=np.arange(N, dtype=float),
        pre_computed_matrices=art, generate_no_shadow=False,
        animate=False, show_progress=False)
    return mags


def iso_objective(rotvec, target_mag, eidx):
    """Squared brightness error for L-BFGS-B."""
    q_sci = Rotation.from_rotvec(rotvec).as_quat()  # xyzw
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
        return {'q_wxyz': q_wxyz, 'residual': float(abs(final_mag - target_mag)),
                'att_error_deg': float(ae), 'nfev': res.nfev}
    except Exception as e:
        return {'error': str(e)}


if __name__ == '__main__':
    t0 = time.time()
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    print(f"Setup: {time.time()-t0:.1f}s", flush=True)

    obs_b1 = CTX.observed_lc[PEAK1]
    print(f"Peak 1 (epoch {PEAK1}): obs={obs_b1:.4f}, true={CTX.true_lc[PEAK1]:.4f}")

    # ── 10,000 random SO(3) seeds ──
    np.random.seed(SEED + 10)
    seeds = Rotation.random(N_SEEDS)
    args_list = [(i, seeds[i].as_rotvec(), obs_b1) for i in range(N_SEEDS)]

    # ── L-BFGS-B in parallel ──
    print(f"\nRunning {N_SEEDS} L-BFGS-B ({N_WORKERS} workers)...", flush=True)
    t1 = time.time()
    results = []
    with get_context('fork').Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_one_seed, args_list, chunksize=16):
            results.append(r)
            if len(results) % 1000 == 0:
                el = time.time() - t1
                good = sum(1 for x in results if x.get('residual', 999) < 0.01)
                eta = el / len(results) * (N_SEEDS - len(results))
                print(f"  [{len(results):>5}/{N_SEEDS}] {good} converged, "
                      f"{el:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)
    opt_time = time.time() - t1
    print(f"Optimization: {opt_time:.1f}s ({opt_time/N_SEEDS:.3f}s/seed)")

    # ── Filter converged ──
    converged = [r for r in results if 'error' not in r and r['residual'] < 0.01]
    failed = sum(1 for r in results if 'error' in r)
    print(f"\nConverged: {len(converged)}/{N_SEEDS} ({failed} errors)")
    if not converged:
        print("No converged results!"); sys.exit(1)

    # ── Cluster within 1 deg (quaternion geodesic distance) ──
    print("Clustering...", flush=True)
    t2 = time.time()
    q_all = np.array([r['q_wxyz'] for r in converged])
    q_xyzw = q_all[:, [1, 2, 3, 0]]  # wxyz → xyzw for dot product
    gram = np.abs(q_xyzw @ q_xyzw.T)
    np.clip(gram, 0, 1, out=gram)
    np.arccos(gram, out=gram)
    gram *= 360.0 / np.pi  # 2 * rad2deg
    np.fill_diagonal(gram, 0.0)
    condensed = squareform(gram, checks=False)
    del gram
    labels = fcluster(linkage(condensed, method='complete'),
                      t=1.0, criterion='distance')
    del condensed
    n_unique = len(set(labels))
    print(f"  {n_unique} unique basins from {len(converged)} converged "
          f"({time.time()-t2:.1f}s)")

    # ── Best representative per cluster ──
    rep_idx = []
    for cl in sorted(set(labels)):
        members = [i for i, l in enumerate(labels) if l == cl]
        rep_idx.append(min(members, key=lambda i: converged[i]['residual']))
    rep_q = np.array([converged[i]['q_wxyz'] for i in rep_idx])

    # ── Brightness gradients (finite differences, same as micro-07) ──
    print(f"Computing gradients for {n_unique} unique attitudes...", flush=True)
    t3 = time.time()
    R_reps = Rotation.from_quat(rep_q[:, [1, 2, 3, 0]])
    Rm = R_reps.as_matrix()
    sv = CTX.sun_pos[PEAK1] - CTX.sat_pos[PEAK1]
    ov = CTX.obs_pos[PEAK1] - CTX.sat_pos[PEAK1]
    k1m = np.einsum('nij,j->ni', Rm, sv)
    k1m /= np.linalg.norm(k1m, axis=1, keepdims=True)
    k2m = np.einsum('nij,j->ni', Rm, ov)
    k2m /= np.linalg.norm(k2m, axis=1, keepdims=True)
    base_mags = lofi_batch(k1m, k2m, PEAK1)
    grads = np.zeros((n_unique, 3))
    axes = np.eye(3)
    for j in range(3):
        k1p = k1m + DPHI * np.cross(axes[j], k1m)
        k1p /= np.linalg.norm(k1p, axis=1, keepdims=True)
        k2p = k2m + DPHI * np.cross(axes[j], k2m)
        k2p /= np.linalg.norm(k2p, axis=1, keepdims=True)
        grads[:, j] = (lofi_batch(k1p, k2p, PEAK1) - base_mags) / DPHI
    grad_norms = np.linalg.norm(grads, axis=1)
    print(f"  Done ({time.time()-t3:.1f}s)")

    # ── Angular distances to truth ──
    R_tr = Rotation.from_quat(
        [CTX.true_q0[1], CTX.true_q0[2], CTX.true_q0[3], CTX.true_q0[0]])
    ang_dists = np.rad2deg((R_reps.inv() * R_tr).magnitude())

    # ── Summary ──
    i_near = np.argmin(ang_dists)
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_unique} unique iso-brightness basins from {N_SEEDS} seeds")
    print(f"Converged: {len(converged)}/{N_SEEDS} ({100*len(converged)/N_SEEDS:.1f}%)")
    print(f"\nNEAREST TO TRUTH: {ang_dists[i_near]:.2f} deg "
          f"(resid={converged[rep_idx[i_near]]['residual']:.6f} mag)")
    print(f"\nComparison:")
    print(f"  Random sampling (1M):       5.95 deg nearest")
    print(f"  Micro-09 L-BFGS-B (100):   25.64 deg nearest, 71 basins")
    print(f"  Micro-10 L-BFGS-B (10K):   {ang_dists[i_near]:.2f} deg nearest, "
          f"{n_unique} basins")
    if ang_dists[i_near] > 0:
        print(f"  vs random: {5.95/ang_dists[i_near]:.1f}x closer")

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nAttitude error percentiles (unique basins):")
    for p in pcts:
        print(f"  {p:>2}%ile: {np.percentile(ang_dists, p):>7.2f} deg")

    bins = [0, 1, 2, 5, 10, 20, 45, 90, 180]
    print(f"\nDistribution:")
    for i in range(len(bins) - 1):
        c = int(np.sum((ang_dists >= bins[i]) & (ang_dists < bins[i + 1])))
        if c > 0:
            print(f"  [{bins[i]:>3}°, {bins[i+1]:>3}°): {c}")

    # ── Save checkpoint (m007_stage1.npz compatible) ──
    obs_mag_p2 = CTX.observed_lc[PEAK2]
    ckpt_path = RESULTS_DIR / 'm010_optimized_candidates.npz'
    np.savez(ckpt_path, candidate_q_wxyz=rep_q, candidate_mags=base_mags,
             gradients=grads, gradient_norms=grad_norms,
             ang_dists_to_truth=ang_dists, truth_q0=CTX.true_q0,
             truth_omega0=CTX.true_omega0, peak1_idx=PEAK1, peak2_idx=PEAK2,
             obs_mag_peak2=obs_mag_p2, ref_mag_peak1=obs_b1, tol=0.01, seed=SEED)

    save_results(RESULTS_DIR / 'm010_scaled_isobrightness.json', {
        'config': {'peak1': PEAK1, 'peak2': PEAK2, 'n_seeds': N_SEEDS, 'seed': SEED,
                   'method': 'L-BFGS-B', 'maxiter': 50, 'cluster_deg': 1.0},
        'n_converged': len(converged), 'n_unique_basins': n_unique,
        'nearest_to_truth_deg': float(ang_dists[i_near]),
        'att_error_percentiles': {
            str(p): round(float(np.percentile(ang_dists, p)), 2) for p in pcts},
        'comparison': {'random_1M_nearest_deg': 5.95,
                       'm009_100seeds_nearest_deg': 25.64, 'm009_basins': 71,
                       'm010_nearest_deg': float(ang_dists[i_near]),
                       'm010_basins': n_unique},
        'runtime_s': round(time.time() - t0, 1)})

    print(f"\nSaved: {ckpt_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")
