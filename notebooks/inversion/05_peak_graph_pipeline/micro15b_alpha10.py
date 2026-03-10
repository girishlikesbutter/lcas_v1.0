#!/usr/bin/env python3
"""Micro-15b — min-|ω| with alpha=10 (stronger regularisation).

micro15 tested alpha=0.1, where ||ω||² ≈ 0.0013 for the true omega contributes
only ~0.00013 to the objective — negligible vs arrival error. With alpha=10 the
regularisation term is ~0.013, genuinely competing and pushing the zero-start
optimizer toward the minimum-magnitude solution.

Question: does stronger regularisation improve true-pair rank?
"""
import sys, json, time
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
ALPHA = 10.0  # 100x larger than micro15's 0.1


# ── Load micro13 data ────────────────────────────────────────────────────────
stage1 = np.load(RESULTS_DIR / "micro13_stage1.npz")
stage2 = np.load(RESULTS_DIR / "micro13_stage2.npz")
PEAKS     = stage1["peaks"]
CANDS     = [stage1["c0"], stage1["c1"], stage1["c2"]]
TRUTH_IDX = stage1["truth_idx"]
BDT       = stage2["bdt"]

CTX     = setup_experiment(n_observations=500, true_omega_deg=(0.5, -0.3, 2.0))
INERTIA = CTX.inertia_tensor


# ── Solvers ──────────────────────────────────────────────────────────────────
def bridge_axis_angle(q_start, q_end, dt, I):
    r1 = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r2 = Rotation.from_quat([q_end[1],   q_end[2],   q_end[3],   q_end[0]])
    w0 = (r1.inv() * r2).as_rotvec() / dt

    times = np.array([0.0, dt])
    def obj(w):
        qp, _ = propagate_attitude(q_start, w, times, "tumbling", I)
        d = np.clip(np.dot(qp[-1], q_end), -1.0, 1.0)
        return 1.0 - d * d

    res  = minimize(obj, w0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-14})
    w_opt = res.x
    qp, _ = propagate_attitude(q_start, w_opt, np.array([0.0, dt]), "tumbling", I)
    err  = attitude_error_deg(qp[-1], q_end)
    return w_opt, err


def bridge_min_omega(q_start, q_end, dt, I):
    """Min-|ω| bridge with zero initial guess, ALPHA=10."""
    w0    = np.zeros(3)
    times = np.array([0.0, dt])
    def obj(w):
        qp, _ = propagate_attitude(q_start, w, times, "tumbling", I)
        d       = np.clip(np.dot(qp[-1], q_end), -1.0, 1.0)
        arrival = 1.0 - d * d
        return arrival + ALPHA * np.dot(w, w)

    res  = minimize(obj, w0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14})
    w_opt = res.x
    qp, _ = propagate_attitude(q_start, w_opt, np.array([0.0, dt]), "tumbling", I)
    err  = attitude_error_deg(qp[-1], q_end)
    return w_opt, err


def worker(args):
    i, j, q_start, q_end, dt, I = args
    w_aa,  err_aa  = bridge_axis_angle(q_start, q_end, dt, I)
    w_min, err_min = bridge_min_omega( q_start, q_end, dt, I)
    return {
        'i': int(i), 'j': int(j),
        'w_aa':  w_aa.tolist(),  'err_aa':  float(err_aa),
        'w_min': w_min.tolist(), 'err_min': float(err_min),
        'mag_aa':  float(np.linalg.norm(w_aa)),
        'mag_min': float(np.linalg.norm(w_min)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    N_PAIRS = 100
    SEED    = 42
    rng     = np.random.default_rng(SEED)
    N0, N1  = len(CANDS[0]), len(CANDS[1])
    dt      = float(BDT[0])

    all_pairs  = [(i, j) for i in range(N0) for j in range(N1)]
    rng.shuffle(all_pairs)
    true_pair  = (int(TRUTH_IDX[0]), int(TRUTH_IDX[1]))
    sample_pairs = [true_pair] + [p for p in all_pairs if p != true_pair][:N_PAIRS]

    args_list = [
        (i, j, CANDS[0][i], CANDS[1][j], dt, INERTIA)
        for i, j in sample_pairs
    ]

    results = []
    for k, a in enumerate(args_list):
        results.append(worker(a))
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(args_list)} pairs done...", flush=True)

    runtime_s = time.time() - t0

    # ── Metrics ──────────────────────────────────────────────────────────────
    true_result    = results[0]
    random_results = results[1:]

    mag_min_all = np.array([r['mag_min'] for r in results])
    mag_aa_all  = np.array([r['mag_aa']  for r in results])

    true_omega_mag      = float(np.linalg.norm(CTX.true_omega0))
    rank_by_mag_min     = int(np.sum(mag_min_all <= true_result['mag_min']))

    w_min_true  = np.array(true_result['w_min'])
    cos_angle   = np.dot(w_min_true, CTX.true_omega0) / (
        np.linalg.norm(w_min_true) * np.linalg.norm(CTX.true_omega0) + 1e-15)
    omega_dir_error_deg = float(np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0.0, 1.0))))

    # ── Plot ─────────────────────────────────────────────────────────────────
    true_omega_mag_degs = np.degrees(true_omega_mag)
    mag_min_degs = np.degrees(mag_min_all)
    mag_aa_degs  = np.degrees(mag_aa_all)
    true_mag_min_degs = np.degrees(true_result['mag_min'])
    true_mag_aa_degs  = np.degrees(true_result['mag_aa'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'micro15b: min-|ω| with α={ALPHA}', fontsize=12)

    rand_mag_aa  = mag_aa_degs[1:]
    rand_mag_min = mag_min_degs[1:]
    ax1.scatter(rand_mag_aa, rand_mag_min, s=15, color='grey', alpha=0.6, label='random pairs')
    ax1.scatter(true_mag_aa_degs, true_mag_min_degs, s=120, color='red',
                marker='*', zorder=5, label='true pair')
    lim = max(mag_aa_degs.max(), mag_min_degs.max()) * 1.05
    ax1.plot([0, lim], [0, lim], 'k--', lw=0.8, label='y = x')
    ax1.axhline(true_omega_mag_degs, color='blue', lw=0.8, ls=':', label='|ω_true|')
    ax1.axvline(true_omega_mag_degs, color='blue', lw=0.8, ls=':')
    ax1.set_xlabel('|ω| axis-angle start (deg/s)')
    ax1.set_ylabel(f'|ω| zero start / min-|ω| α={ALPHA} (deg/s)')
    ax1.set_title('ω magnitude comparison')
    ax1.legend(fontsize=8)

    sorted_idx  = np.argsort(mag_min_degs)
    sorted_mags = mag_min_degs[sorted_idx]
    true_rank   = int(np.where(sorted_idx == 0)[0][0])

    ax2.plot(np.arange(len(sorted_mags)), sorted_mags, color='blue', lw=1.2)
    ax2.scatter([true_rank], [mag_min_degs[0]], color='red', s=80, zorder=5,
                label=f'true pair (rank {true_rank + 1}/{len(results)})')
    ax2.axhline(true_omega_mag_degs, color='green', lw=0.8, ls='--', label='|ω_true|')
    ax2.set_xlabel('rank (ascending |ω_min|)')
    ax2.set_ylabel('|ω_min| (deg/s)')
    ax2.set_title(f'Sorted min-|ω| — true pair rank: {true_rank + 1}/{len(results)}')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "micro15b_alpha10.png", dpi=150)
    plt.close()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results_out = {
        'config': {'leg': 0, 'peaks': PEAKS.tolist(), 'dt_s': dt,
                   'n_pairs': len(results), 'alpha': ALPHA},
        'true_pair': true_result,
        'true_omega_mag_rads': true_omega_mag,
        'true_omega_mag_degs': float(np.degrees(true_omega_mag)),
        'omega_dir_error_deg': omega_dir_error_deg,
        'rank_by_mag_min': rank_by_mag_min,
        'rank_by_mag_min_out_of': len(results),
        'mag_min_mean_random': float(np.mean([r['mag_min'] for r in random_results])),
        'mag_min_std_random':  float(np.std( [r['mag_min'] for r in random_results])),
        'mag_aa_mean_random':  float(np.mean([r['mag_aa']  for r in random_results])),
        'runtime_s': runtime_s,
    }
    save_results(RESULTS_DIR / "micro15b_alpha10.json", results_out)

    print(f"\n=== micro15b: min-|ω| α={ALPHA} ===")
    print(f"True ω magnitude: {np.degrees(true_omega_mag):.2f} deg/s")
    print(f"True pair:")
    print(f"  axis-angle |ω|:  {true_mag_aa_degs:.2f} deg/s  (err: {true_result['err_aa']:.3f} deg)")
    print(f"  zero-start min-|ω|: {true_mag_min_degs:.2f} deg/s  (err: {true_result['err_min']:.3f} deg)")
    print(f"  direction error vs truth: {omega_dir_error_deg:.1f} deg")
    print(f"  rank by min-|ω|: {rank_by_mag_min} / {len(results)}")
    print(f"Random (N={len(random_results)}):  mean |ω_min|={np.degrees(np.mean([r['mag_min'] for r in random_results])):.2f} deg/s")
    verdict = ("SUPPORTED" if rank_by_mag_min <= max(1, len(results) // 10) else "not supported")
    print(f"VERDICT: {verdict}")
    print(f"Runtime: {runtime_s:.1f}s")
