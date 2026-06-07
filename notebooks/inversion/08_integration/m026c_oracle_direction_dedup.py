#!/usr/bin/env python3
"""Micro-26c — Oracle integration pipeline with direction-aware deduplication.

Clean rerun of m026b. Two omegas are duplicates only if BOTH |omega| within
0.05 deg/s AND angular distance between unit vectors < 5 deg.

Changes from m026b:
  - Renamed to m026c throughout
  - JSON output includes omega VECTORS for best and true pairs
  - JSON output includes omega direction error for true pair
"""
import sys, time, json, numpy as np
from pathlib import Path
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
import os; os.chdir(PROJECT_ROOT)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_STARTS = 10          # random starts per band
N_WORKERS = 8
ARRIVAL_THRESH = 1e-6  # bridge arrival error threshold
MAG_TOL = 0.05         # dedup: magnitude tolerance (deg/s)
DIR_TOL = 5.0          # dedup: angular distance tolerance (degrees)
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

_SHARED = {}  # populated in worker processes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_L(q_wxyz, omega_body, I):
    """Angular momentum in inertial frame: L = R(q) @ (I @ omega_body)."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])
    return R @ (I @ omega_body)


def _ang_dist_deg(a, b):
    """Angular distance between two vectors in degrees."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 180.0
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1., 1.))))


# ---------------------------------------------------------------------------
# Bridge solver (runs in worker processes)
# ---------------------------------------------------------------------------
def _init_worker(shared):
    global _SHARED
    _SHARED = shared


def _solve_band(args):
    """Solve bridge problem for one random start in one magnitude band."""
    w0, lb_rad, ub_rad = args
    qs = _SHARED['qs']
    qe = _SHARED['qe']
    dt = _SHARED['dt']
    I  = _SHARED['I']

    def obj(w):
        wn2 = float(np.dot(w, w))
        pen = (max(0., lb_rad**2 - wn2)**2 * 1e4
             + max(0., wn2 - ub_rad**2)**2 * 1e4)
        qp, _ = propagate_attitude(qs, w, np.array([0., dt]), "tumbling", I)
        d = np.clip(np.dot(qp[-1], qe), -1., 1.)
        return (1. - d*d) + pen

    res = minimize(obj, w0, method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-9})
    wk = res.x
    qp, _ = propagate_attitude(qs, wk, np.array([0., dt]), "tumbling", I)
    d = np.clip(np.dot(qp[-1], qe), -1., 1.)
    return {
        'omega': wk.tolist(),
        'mag_degs': float(np.rad2deg(np.linalg.norm(wk))),
        'arrival_err': float(1. - d*d),
    }


# ---------------------------------------------------------------------------
# Leg solver: band sweep + direction-aware dedup
# ---------------------------------------------------------------------------
def run_leg(q_start, q_end, dt, I, rng_seed):
    """Run band-sweep bridge solve for one leg, return deduplicated candidates."""
    bands = [(0.5 * i, 0.5 * (i + 1)) for i in range(13)]  # 0-0.5, ..., 6.0-6.5 deg/s
    rng = np.random.RandomState(rng_seed)

    # Build task list: N_STARTS random starts per band
    tasks = []
    for lb_d, ub_d in bands:
        lb_r = np.deg2rad(lb_d) + 1e-6
        ub_r = np.deg2rad(ub_d)
        for _ in range(N_STARTS):
            d = rng.randn(3)
            d /= np.linalg.norm(d)
            mag = lb_r + (ub_r - lb_r) * rng.rand()
            tasks.append((mag * d, lb_r, ub_r))

    shared = {'qs': q_start, 'qe': q_end, 'dt': dt, 'I': I}
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(shared,)) as pool:
        raw = pool.map(_solve_band, tasks, chunksize=4)

    # Keep only solutions with good arrival error
    valid = sorted(
        [r for r in raw if r['arrival_err'] < ARRIVAL_THRESH],
        key=lambda r: r['mag_degs']
    )

    # Direction-aware dedup: both magnitude AND direction must match to merge
    deduped = []
    for r in valid:
        w_r = np.array(r['omega'])
        is_dup = False
        for d in deduped:
            if abs(r['mag_degs'] - d['mag_degs']) < MAG_TOL:
                if _ang_dist_deg(w_r, np.array(d['omega'])) < DIR_TOL:
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(r)

    return deduped


# ---------------------------------------------------------------------------
# Find true candidate by closest omega vector
# ---------------------------------------------------------------------------
def _find_true_idx(candidates, true_omega):
    """Find candidate closest to truth (pre-filter by magnitude, then by vector distance)."""
    true_mag = float(np.rad2deg(np.linalg.norm(true_omega)))
    best_i, best_err = -1, 1e30
    for i, c in enumerate(candidates):
        mag_err = abs(c['mag_degs'] - true_mag)
        if mag_err < 0.1:  # pre-filter by magnitude
            vec_err = np.linalg.norm(np.array(c['omega']) - true_omega)
            if vec_err < best_err:
                best_err = vec_err
                best_i = i
    return best_i


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    t0 = time.time()

    # --- Setup ---
    CTX = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=SEED,
        true_omega_deg=(0.5, -0.3, 2.0), end_time_utc='2020-02-05T11:00:00'
    )
    I = CTX.inertia_tensor
    OBS_T = CTX.observation_times

    # Load peak indices from m013 stage 1
    PEAKS = [int(x) for x in np.load(RESULTS_DIR / "m013_stage1.npz")["peaks"]]
    print(f"Peak indices: {PEAKS}")

    # Propagate true attitude to peaks → oracle quaternions and omega at peaks
    times_p = np.array([0.0] + [float(OBS_T[p]) for p in PEAKS])
    q_traj, om_traj = propagate_attitude(CTX.true_q0, CTX.true_omega0, times_p, "tumbling", I)

    q_A, q_B, q_C = q_traj[1], q_traj[2], q_traj[3]
    om_A, om_B = om_traj[1], om_traj[2]   # true omega at peak A and B

    dt_0 = float(OBS_T[PEAKS[1]] - OBS_T[PEAKS[0]])
    dt_1 = float(OBS_T[PEAKS[2]] - OBS_T[PEAKS[1]])

    print(f"Peaks {PEAKS}, dt0={dt_0:.1f}s  dt1={dt_1:.1f}s, setup {time.time()-t0:.1f}s",
          flush=True)

    # --- Leg 0: q_A → q_B ---
    leg0 = run_leg(q_A, q_B, dt_0, I, SEED)
    print(f"Leg 0: {len(leg0)} candidates ({time.time()-t0:.1f}s)", flush=True)

    # --- Leg 1: q_B → q_C ---
    leg1 = run_leg(q_B, q_C, dt_1, I, SEED + 1000)
    print(f"Leg 1: {len(leg1)} candidates ({time.time()-t0:.1f}s)", flush=True)

    n0, n1 = len(leg0), len(leg1)

    # --- L-conservation scoring at peak B ---
    true_k = _find_true_idx(leg0, om_A)
    true_j = _find_true_idx(leg1, om_B)

    # Leg 0: propagate each candidate from q_A → compute arriving omega at q_B → L
    L_leg0 = []
    for k, c in enumerate(leg0):
        _, om_p = propagate_attitude(q_A, np.array(c['omega']),
                                     np.array([0., dt_0]), "tumbling", I)
        L_leg0.append(compute_L(q_B, om_p[-1], I))

    # Leg 1: compute L directly from departing omega at q_B
    L_leg1 = [compute_L(q_B, np.array(c['omega']), I) for c in leg1]

    # L-error matrix (n0 x n1)
    L_err = np.array([
        [np.linalg.norm(L_leg0[k] - L_leg1[j]) for j in range(n1)]
        for k in range(n0)
    ])

    # Rank analysis
    flat = L_err.ravel()
    ranks = flat.argsort().argsort()

    best_k, best_j = divmod(int(flat.argmin()), n1)
    true_flat = true_k * n1 + true_j if (true_k >= 0 and true_j >= 0) else -1
    true_rank = int(ranks[true_flat]) + 1 if true_flat >= 0 else -1

    sorted_flat = np.sort(flat)
    gap = float(sorted_flat[1] - sorted_flat[0]) if n0 * n1 > 1 else 0.0

    # Best pair errors
    best_om0 = np.array(leg0[best_k]['omega'])
    best_om1 = np.array(leg1[best_j]['omega'])
    err0 = float(np.rad2deg(np.linalg.norm(best_om0 - om_A)))
    err1 = float(np.rad2deg(np.linalg.norm(best_om1 - om_B)))
    dir0 = _ang_dist_deg(best_om0, om_A)
    dir1 = _ang_dist_deg(best_om1, om_B)

    # True pair errors (direction)
    if true_k >= 0:
        true_om0 = np.array(leg0[true_k]['omega'])
        true_dir0 = _ang_dist_deg(true_om0, om_A)
        true_vec_err0 = float(np.rad2deg(np.linalg.norm(true_om0 - om_A)))
    else:
        true_om0 = None
        true_dir0 = None
        true_vec_err0 = None

    if true_j >= 0:
        true_om1 = np.array(leg1[true_j]['omega'])
        true_dir1 = _ang_dist_deg(true_om1, om_B)
        true_vec_err1 = float(np.rad2deg(np.linalg.norm(true_om1 - om_B)))
    else:
        true_om1 = None
        true_dir1 = None
        true_vec_err1 = None

    # True pair L-conservation error
    true_L_err = float(L_err[true_k, true_j]) if (true_k >= 0 and true_j >= 0) else None

    runtime = time.time() - t0

    # Count occupied bands
    nb0 = len(set(int(c['mag_degs'] / 0.5) for c in leg0))
    nb1 = len(set(int(c['mag_degs'] / 0.5) for c in leg1))

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"=== m026c SUMMARY ===")
    print(f"Leg 0: {n0} omega candidates found ({nb0} bands non-empty)")
    print(f"Leg 1: {n1} omega candidates found ({nb1} bands non-empty)")
    print(f"Total pairs: {n0 * n1}")
    print(f"True pair: (k, j) = ({true_k}, {true_j})  rank = {true_rank}/{n0*n1}  "
          f"L_err = {true_L_err:.4e} kg*m^2/s" if true_L_err is not None else
          f"True pair: NOT FOUND")
    print(f"Best pair: (k, j) = ({best_k}, {best_j})  L_err = {float(flat.min()):.4e}")
    print(f"  Gap (2nd - 1st): {gap:.4e} kg*m^2/s")
    print(f"Best omega error: {err0:.4f} / {err1:.4f} deg/s "
          f"(direction: {dir0:.2f} / {dir1:.2f} deg)")
    if true_om0 is not None:
        print(f"True pair omega error: {true_vec_err0:.4f} / "
              f"{true_vec_err1:.4f} deg/s "
              f"(direction: {true_dir0:.2f} / {true_dir1:.2f} deg)")
    print(f"Runtime: {runtime:.1f}s")
    print(f"  cf. m026: 34/32 candidates, rank 423/1088, gap 1.80e+00")
    print(f"{'='*60}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Micro-26c: Integration pipeline — direction-aware dedup", fontsize=12)

    # Heatmap of L-error
    im = ax1.imshow(
        np.log10(np.where(L_err > 0, L_err, 1e-20)),
        aspect='auto', origin='lower', cmap='viridis_r'
    )
    plt.colorbar(im, ax=ax1, label='log10(||DL||)')
    ax1.set_xlabel('Leg 1 candidate j')
    ax1.set_ylabel('Leg 0 candidate k')
    ax1.set_title('L-error heatmap')
    if true_k >= 0 and true_j >= 0:
        ax1.plot(true_j, true_k, 'r*', ms=16, label='true')
    ax1.plot(best_j, best_k, 'go', ms=12, mfc='none', mew=2.5, label='best')
    ax1.legend(fontsize=8)

    # Bar chart of omega magnitudes
    mags0 = [c['mag_degs'] for c in leg0]
    mags1 = [c['mag_degs'] for c in leg1]
    x = np.arange(max(n0, n1))
    bw = 0.35
    if n0:
        colors0 = ['firebrick' if k == true_k else 'steelblue' for k in range(n0)]
        ax2.bar(x[:n0] - bw/2, mags0, bw, color=colors0,
                label='Leg 0', edgecolor='k', lw=0.5)
    if n1:
        colors1 = ['firebrick' if j == true_j else 'darkorange' for j in range(n1)]
        ax2.bar(x[:n1] + bw/2, mags1, bw, color=colors1,
                label='Leg 1', edgecolor='k', lw=0.5)
    ax2.set_xlabel('Candidate index')
    ax2.set_ylabel('|omega| (deg/s)')
    ax2.set_title('Omega magnitude (red=true)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_png = RESULTS_DIR / "m026c_oracle_direction_dedup.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

    # --- Save JSON results ---
    results = {
        'experiment': 'm026c',
        'description': 'Oracle integration pipeline with direction-aware dedup',
        'n_candidates_leg0': n0,
        'n_candidates_leg1': n1,
        'n_bands_occupied': {'leg0': nb0, 'leg1': nb1},
        'total_pairs': n0 * n1,
        'true_pair': {'k': true_k, 'j': true_j},
        'true_pair_rank': true_rank,
        'true_pair_L_err': round(true_L_err, 8) if true_L_err is not None else None,
        'best_pair': {'k': best_k, 'j': best_j},
        'best_pair_L_err': round(float(flat.min()), 8),
        'L_gap': gap,
        'best_omega_vectors': {
            'leg0_rad_s': best_om0.tolist(),
            'leg1_rad_s': best_om1.tolist(),
            'leg0_deg_s': np.rad2deg(best_om0).tolist(),
            'leg1_deg_s': np.rad2deg(best_om1).tolist(),
        },
        'best_omega_error_degs': {'leg0': round(err0, 5), 'leg1': round(err1, 5)},
        'best_direction_error_deg': {'leg0': round(dir0, 3), 'leg1': round(dir1, 3)},
        'true_omega_vectors': {
            'leg0_rad_s': true_om0.tolist() if true_om0 is not None else None,
            'leg1_rad_s': true_om1.tolist() if true_om1 is not None else None,
            'leg0_deg_s': np.rad2deg(true_om0).tolist() if true_om0 is not None else None,
            'leg1_deg_s': np.rad2deg(true_om1).tolist() if true_om1 is not None else None,
        },
        'true_omega_error_degs': {
            'leg0': round(true_vec_err0, 5) if true_vec_err0 is not None else None,
            'leg1': round(true_vec_err1, 5) if true_vec_err1 is not None else None,
        },
        'true_direction_error_deg': {
            'leg0': round(true_dir0, 3) if true_dir0 is not None else None,
            'leg1': round(true_dir1, 3) if true_dir1 is not None else None,
        },
        'ground_truth': {
            'omega_at_A_rad_s': om_A.tolist(),
            'omega_at_B_rad_s': om_B.tolist(),
            'omega_at_A_deg_s': np.rad2deg(om_A).tolist(),
            'omega_at_B_deg_s': np.rad2deg(om_B).tolist(),
        },
        'leg0_mags': [round(m, 4) for m in mags0],
        'leg1_mags': [round(m, 4) for m in mags1],
        'dedup': {'mag_tol_degs': MAG_TOL, 'dir_tol_deg': DIR_TOL},
        'peaks': PEAKS,
        'dt_0': round(dt_0, 2),
        'dt_1': round(dt_1, 2),
        'comparison_micro26': {
            'candidates': '34/32',
            'rank': '423/1088',
            'gap': 1.7953,
        },
        'runtime_s': round(runtime, 1),
    }
    out_json = RESULTS_DIR / "m026c_oracle_direction_dedup.json"
    save_results(out_json, results)
    print(f"Saved: {out_json}")
