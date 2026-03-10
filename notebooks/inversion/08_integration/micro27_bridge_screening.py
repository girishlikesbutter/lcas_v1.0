"""micro27 — Single-bridge screening for candidate pair pruning.

Tests whether a single cheap bridge solve per pair provides a screening
signal to prune most pairs before expensive full band-sweep.
"""
import sys, os, time, json
import numpy as np
from pathlib import Path
from multiprocessing import Pool, set_start_method
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INVERSION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INVERSION_ROOT))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

# ── Helpers ────────────────────────────────────────────────────────────
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def make_nearby(q0, N=50, seed=42):
    """Generate N candidates: q0 itself + (N-1) random rotations of 1-3 deg."""
    rng = np.random.default_rng(seed)
    cands = np.zeros((N, 4)); cands[0] = q0
    for k in range(1, N):
        ax = rng.standard_normal(3); ax /= np.linalg.norm(ax)
        ang = np.deg2rad(rng.uniform(1.0, 3.0))
        dq = np.array([np.cos(ang/2), *(np.sin(ang/2) * ax)])
        cands[k] = qmul(dq, q0)
    return cands

def bridge_solve(qA, qB, dt, inertia):
    """Single L-BFGS-B bridge from zero omega initial guess."""
    def obj(w):
        qp, _ = propagate_attitude(qA, w, np.array([0.0, dt]), "tumbling", inertia)
        d = np.clip(np.dot(qp[-1], qB), -1.0, 1.0)
        return 1.0 - d * d
    res = minimize(obj, np.zeros(3), method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-9})
    return np.rad2deg(np.linalg.norm(res.x)), res.fun, res.nit

_G = {}
def _init(cA, cB, dt, inertia):
    _G.update(cA=cA, cB=cB, dt=dt, I=inertia)

def _worker(ij):
    i, j = ij
    return (*ij, *bridge_solve(_G['cA'][i], _G['cB'][j], _G['dt'], _G['I']))

def run_leg(cA, cB, dt, I_tensor, true_i, true_j, label=""):
    N = len(cA)
    pairs = [(i, j) for i in range(N) for j in range(N)]
    with Pool(8, initializer=_init, initargs=(cA, cB, dt, I_tensor)) as pool:
        results = pool.map(_worker, pairs)
    om = np.zeros((N, N)); res = np.zeros((N, N))
    for i, j, o, r, _ in results:
        om[i, j] = o; res[i, j] = r
    flat_om, flat_res = om.flatten(), res.flatten()
    true_om, true_res = om[true_i, true_j], res[true_i, true_j]
    rank_om = int((flat_om < true_om).sum()) + 1
    rank_res = int((flat_res < true_res).sum()) + 1
    print(f"  {label}: true |w|={true_om:.3f} deg/s rank {rank_om}/{N*N}, "
          f"residual={true_res:.2e} rank {rank_res}/{N*N}")
    return om, res, true_om, true_res, rank_om, rank_res


if __name__ == '__main__':
    set_start_method('fork')
    t0 = time.time()

    # ── Setup ──────────────────────────────────────────────────────────
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    I_tensor = CTX.inertia_tensor
    dt_samp = CTX.dt_sampling

    micro13 = np.load('data/results/inversion_diagnostics/micro13_stage1.npz')
    peaks = micro13['peaks']            # [183, 260, 360]
    truth_idx = micro13['truth_idx']    # [49, 49, 49]
    dt_legs = [(peaks[1] - peaks[0]) * dt_samp, (peaks[2] - peaks[1]) * dt_samp]
    q_oracle = CTX.true_quaternions[peaks]  # (3, 4) wxyz

    print(f"Setup done in {time.time()-t0:.1f}s. dt_legs={[f'{d:.1f}' for d in dt_legs]}s")

    # ── Part A: oracle + 1-3 deg nudge ─────────────────────────────────
    print("Part A: oracle-based candidates (1-3 deg nudge)...")
    cands_A = [make_nearby(q_oracle[k], 50, seed=42 + k) for k in range(3)]

    omA0, resA0, tA0_om, tA0_res, rA0_om, rA0_res = run_leg(
        cands_A[0], cands_A[1], dt_legs[0], I_tensor, 0, 0, "leg0")
    omA1, resA1, tA1_om, tA1_res, rA1_om, rA1_res = run_leg(
        cands_A[1], cands_A[2], dt_legs[1], I_tensor, 0, 0, "leg1")

    # ── Part B: micro13 real iso-brightness candidates ──────────────────
    print("Part B: micro13 iso-brightness candidates...")
    c_m = [micro13['c0'], micro13['c1'], micro13['c2']]
    ti = truth_idx

    omB0, resB0, tB0_om, tB0_res, rB0_om, rB0_res = run_leg(
        c_m[0], c_m[1], dt_legs[0], I_tensor, ti[0], ti[1], "leg0")
    omB1, resB1, tB1_om, tB1_res, rB1_om, rB1_res = run_leg(
        c_m[1], c_m[2], dt_legs[1], I_tensor, ti[1], ti[2], "leg1")

    runtime = time.time() - t0

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('micro27: Single-bridge screening for pair pruning', fontsize=14)

    panels = [
        (axes[0, 0], omA0, tA0_om, rA0_om, 'Part A leg 0 (oracle +/- 1-3 deg)'),
        (axes[0, 1], omB0, tB0_om, rB0_om, 'Part B leg 0 (micro13 candidates)'),
        (axes[1, 0], resA0, tA0_res, rA0_res, 'Part A leg 0 — arrival residual'),
        (axes[1, 1], omB1, tB1_om, rB1_om, 'Part B leg 1 (micro13 candidates)'),
    ]
    for ax, data, true_val, rank, title in panels:
        flat = data.flatten()
        is_residual = 'residual' in title
        xlabel = 'arrival residual (1-d^2)' if is_residual else 'min |w| (deg/s)'
        ax.hist(flat, bins=80, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(true_val, color='red', lw=2, ls='--',
                   label=f'true pair: {true_val:.4f}, rank {rank}/2500')
        ax.set_xlabel(xlabel); ax.set_ylabel('count')
        ax.set_title(title); ax.legend(fontsize=9)

    plt.tight_layout()
    out_plot = 'data/results/inversion_diagnostics/micro27_bridge_screening.png'
    plt.savefig(out_plot, dpi=150)
    print(f"Plot saved: {out_plot}")

    # ── Save JSON ──────────────────────────────────────────────────────
    ranks = [rA0_om, rA1_om, rB0_om, rB1_om]
    worst_pct = max(r / 2500 * 100 for r in ranks)
    threshold_pct = max(5, int(np.ceil(worst_pct * 2)))

    save_results('data/results/inversion_diagnostics/micro27_bridge_screening.json', {
        'part_A_leg0': dict(true_omega_mag=tA0_om, rank_by_omega=rA0_om, rank_by_residual=rA0_res),
        'part_A_leg1': dict(true_omega_mag=tA1_om, rank_by_omega=rA1_om, rank_by_residual=rA1_res),
        'part_B_leg0': dict(true_omega_mag=tB0_om, rank_by_omega=rB0_om, rank_by_residual=rB0_res),
        'part_B_leg1': dict(true_omega_mag=tB1_om, rank_by_omega=rB1_om, rank_by_residual=rB1_res),
        'n_pairs_leg0': 2500, 'n_pairs_leg1': 2500,
        'true_pair_rank_by_omega_mag': dict(A0=rA0_om, A1=rA1_om, B0=rB0_om, B1=rB1_om),
        'true_pair_rank_by_residual': dict(A0=rA0_res, A1=rA1_res, B0=rB0_res, B1=rB1_res),
        'screening_threshold_recommendation': f'keep top {threshold_pct}% by min-|omega|',
        'runtime_s': round(runtime, 1),
        'dt_legs_s': [round(d, 1) for d in dt_legs],
    })

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("=== micro27 SUMMARY ===")
    print(f"Part A (oracle q, 1-3 deg nudge):")
    print(f"  Leg 0: |w|={tA0_om:.3f} deg/s, rank {rA0_om}/2500 by |w|, {rA0_res}/2500 by res")
    print(f"  Leg 1: |w|={tA1_om:.3f} deg/s, rank {rA1_om}/2500 by |w|, {rA1_res}/2500 by res")
    print(f"Part B (micro13 candidates):")
    print(f"  Leg 0: |w|={tB0_om:.3f} deg/s, rank {rB0_om}/2500 by |w|, {rB0_res}/2500 by res")
    print(f"  Leg 1: |w|={tB1_om:.3f} deg/s, rank {rB1_om}/2500 by |w|, {rB1_res}/2500 by res")
    print(f"Screening: keep top {threshold_pct}% by min-|w| -> prune {100-threshold_pct}% of pairs")
    print(f"Runtime: {runtime:.1f}s")
    print('=' * 55)
