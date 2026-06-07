#!/usr/bin/env python3
"""Micro-16c — trough-constrained bridge.

Diagnosis from m016: the axis-angle bridge ω is optimised only for
peak-to-peak arrival. At t_trough (202s into the 554s leg) the propagated
attitude has large error → predicted trough brightness ≈ 12.5 mag vs observed
14.44 mag. True pair rank 65/101 — no signal.

New approach: embed the trough brightness constraint *inside* the bridge
objective so the optimizer is forced to find a ω that simultaneously:
  (a) takes q_A → q_B in dt_total  (arrival error)
  (b) passes through a trough-brightness orientation at t_trough  (lo-fi term)

Hypothesis: only the true pair has a ω satisfying both simultaneously.
Random pairs cannot satisfy both, so their minimum objective will be larger.

We score pairs by their minimum combined objective value (lower = better fit).
"""
import sys, time, json
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
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, brightness_single_epoch, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
SEED        = 42
N_PAIRS     = 100
ALPHA_TROUGH = 1.0   # weight on trough brightness residual² vs arrival error


# ── Load m013 data ────────────────────────────────────────────────────────
stage1   = np.load(RESULTS_DIR / "m013_stage1.npz")
stage2   = np.load(RESULTS_DIR / "m013_stage2.npz")
PEAKS    = stage1["peaks"]
CANDS    = [stage1["c0"], stage1["c1"], stage1["c2"]]
TRUTH_IDX = stage1["truth_idx"]
BDT      = stage2["bdt"]


def score_pair(i, j, q_start, q_end, dt, trough_idx, t_trough,
               lofi_trough_target, ctx, I):
    """Return minimum trough-constrained bridge objective for a (q_start, q_end) pair.

    Uses axis-angle ω as initial guess, then minimises:
        arrival_error²  +  ALPHA_TROUGH * (lofi_brightness_at_trough - target)²
    """
    # ── initial guess: axis-angle ω ──────────────────────────────────────────
    r1 = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r2 = Rotation.from_quat([q_end[1],   q_end[2],   q_end[3],   q_end[0]])
    w0 = (r1.inv() * r2).as_rotvec() / dt

    times = np.array([0.0, t_trough, dt])

    def obj(w):
        qp, _ = propagate_attitude(q_start, w, times, "tumbling", I)
        # arrival error
        d       = np.clip(np.dot(qp[-1], q_end), -1.0, 1.0)
        arrival = 1.0 - d * d
        # trough brightness residual (lo-fi, fast)
        pred_trough = brightness_single_epoch(qp[1], trough_idx, ctx,
                                              use_shadows=False)
        trough_res = (pred_trough - lofi_trough_target) ** 2
        return arrival + ALPHA_TROUGH * trough_res

    res = minimize(obj, w0, method='L-BFGS-B',
                   options={'maxiter': 150, 'ftol': 1e-12})

    # evaluate components separately at optimum
    qp_opt, _ = propagate_attitude(q_start, res.x, times, "tumbling", I)
    d_opt     = np.clip(np.dot(qp_opt[-1], q_end), -1.0, 1.0)
    arr_opt   = float(1.0 - d_opt * d_opt)
    pred_opt  = float(brightness_single_epoch(qp_opt[1], trough_idx, ctx,
                                               use_shadows=False))
    trough_err = float(abs(pred_opt - lofi_trough_target))

    return {
        'i': int(i), 'j': int(j),
        'obj': float(res.fun),
        'arrival_err': arr_opt,
        'trough_pred': pred_opt,
        'trough_err': trough_err,
        'w': res.x.tolist(),
        'ok': True,
    }


if __name__ == '__main__':
    t0 = time.time()

    CTX     = setup_experiment(n_observations=500, noise_sigma=0.05,
                               random_seed=SEED, true_omega_deg=(0.5, -0.3, 2.0),
                               end_time_utc='2020-02-05T11:00:00')
    INERTIA = CTX.inertia_tensor
    OBS_TIMES = CTX.observation_times
    print(f"Setup: {time.time()-t0:.1f}s", flush=True)

    # ── Find trough (leg 0: peak 183 → 260) ──────────────────────────────────
    PEAK_A = int(PEAKS[0])   # 183
    PEAK_B = int(PEAKS[1])   # 260
    seg    = CTX.observed_lc[PEAK_A:PEAK_B + 1]
    TROUGH_IDX = PEAK_A + int(np.argmax(seg))
    t_trough   = float(OBS_TIMES[TROUGH_IDX] - OBS_TIMES[PEAK_A])
    dt_total   = float(BDT[0])

    obs_trough  = float(CTX.observed_lc[TROUGH_IDX])
    true_trough = float(CTX.true_lc[TROUGH_IDX])

    # lo-fi reference at trough — use true LC brightness
    # (true_lc uses hi-fi, so approximate the lo-fi reference as the
    #  lo-fi brightness at the true attitude)
    lofi_trough_ref = float(brightness_single_epoch(
        CTX.true_q0_propagated[TROUGH_IDX], TROUGH_IDX, CTX, use_shadows=False
    )) if hasattr(CTX, 'true_q0_propagated') else obs_trough

    # Fallback: use observed brightness as target (will be biased but directional)
    # We compare lo-fi predictions against the lo-fi value at the true orientation:
    true_q_at_trough = None
    try:
        # propagate from true q0 to get true attitude at trough
        times_check = np.array([0.0, t_trough])
        q_true_start = CTX.true_q0
        omega_true   = CTX.true_omega0
        qp_true, _   = propagate_attitude(q_true_start, omega_true,
                                          times_check, "tumbling", INERTIA)
        true_q_at_trough = qp_true[-1]
        lofi_trough_ref = float(brightness_single_epoch(
            true_q_at_trough, TROUGH_IDX, CTX, use_shadows=False))
        print(f"Lo-fi trough reference (true attitude): {lofi_trough_ref:.4f}")
    except Exception as e:
        lofi_trough_ref = obs_trough
        print(f"Using observed trough as lo-fi reference: {lofi_trough_ref:.4f} (fallback)")

    print(f"Trough epoch {TROUGH_IDX}, t_from_peak={t_trough:.1f}s")
    print(f"Observed trough: {obs_trough:.4f}, True trough: {true_trough:.4f}", flush=True)

    # ── Sample pairs ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    N0, N1 = len(CANDS[0]), len(CANDS[1])
    all_pairs = [(i, j) for i in range(N0) for j in range(N1)]
    rng.shuffle(all_pairs)
    true_pair    = (int(TRUTH_IDX[0]), int(TRUTH_IDX[1]))
    sample_pairs = [true_pair] + [p for p in all_pairs if p != true_pair][:N_PAIRS]

    print(f"Scoring {len(sample_pairs)} pairs sequentially...", flush=True)
    t_score = time.time()

    results = []
    for k, (i, j) in enumerate(sample_pairs):
        r = score_pair(i, j,
                       CANDS[0][i], CANDS[1][j],
                       dt_total, TROUGH_IDX, t_trough,
                       lofi_trough_ref, CTX, INERTIA)
        results.append(r)
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(sample_pairs)} done...", flush=True)

    score_time = time.time() - t_score
    print(f"Scoring done: {score_time:.1f}s", flush=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    true_result = results[0]
    all_obj     = np.array([r['obj'] for r in results])
    rank        = int(np.sum(all_obj <= true_result['obj']))
    rand_obj    = all_obj[1:]
    rand_mean   = float(np.mean(rand_obj))
    rand_std    = float(np.std(rand_obj))

    all_trough_err = np.array([r['trough_err'] for r in results])
    rank_by_trough = int(np.sum(all_trough_err <= true_result['trough_err']))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'm016c: trough-constrained bridge (α_trough={ALPHA_TROUGH})', fontsize=11)

    # Panel 1: combined objective histogram
    ax = axes[0]
    ax.hist(rand_obj, bins=20, color='steelblue', alpha=0.8, label='Random pairs')
    ax.axvline(true_result['obj'], color='red', lw=2,
               label=f"True pair obj={true_result['obj']:.4f}")
    ax.axvline(rand_mean, color='grey', lw=1.5, ls='--',
               label=f"Mean random ({rand_mean:.4f})")
    ax.set_xlabel('Combined objective (arrival + α_trough × trough_res²)')
    ax.set_ylabel('Count')
    ax.set_title(f'Combined objective — true rank: {rank}/{len(results)}')
    ax.legend(fontsize=8)

    # Panel 2: trough prediction error histogram
    ax2 = axes[1]
    rand_te = all_trough_err[1:]
    ax2.hist(rand_te, bins=20, color='darkorange', alpha=0.8, label='Random pairs')
    ax2.axvline(true_result['trough_err'], color='red', lw=2,
                label=f"True pair err={true_result['trough_err']:.4f}")
    ax2.set_xlabel('|lo-fi predicted trough - lo-fi reference| (mag)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Trough residual alone — true rank: {rank_by_trough}/{len(results)}')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "m016c_trough_constrained.png", dpi=150)
    plt.close()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    runtime_s = float(time.time() - t0)
    results_out = {
        'config': {
            'leg': 0, 'peak_A': PEAK_A, 'peak_B': PEAK_B,
            'trough_idx': int(TROUGH_IDX), 't_trough_s': t_trough,
            'dt_total_s': dt_total, 'alpha_trough': ALPHA_TROUGH,
            'n_pairs': len(results),
        },
        'lofi_trough_ref': lofi_trough_ref,
        'obs_trough': obs_trough,
        'true_trough': true_trough,
        'true_pair_obj': float(true_result['obj']),
        'true_pair_rank_by_obj': rank,
        'true_pair_rank_by_trough': rank_by_trough,
        'true_pair_trough_err': float(true_result['trough_err']),
        'true_pair_arrival_err': float(true_result['arrival_err']),
        'random_obj_mean': rand_mean,
        'random_obj_std': rand_std,
        'runtime_s': round(runtime_s, 1),
    }
    save_results(RESULTS_DIR / "m016c_trough_constrained.json", results_out)

    # ── Console summary ───────────────────────────────────────────────────────
    verdict_obj = ("DISCRIMINATES" if rank <= max(5, len(results) // 10)
                   else "no signal")
    print(f"""
=== m016c: trough-constrained bridge ===
Leg 0: peak {PEAK_A} → {PEAK_B}, trough at epoch {TROUGH_IDX} (t={t_trough:.1f}s)
Lo-fi trough reference: {lofi_trough_ref:.4f}

True pair:
  combined obj: {true_result['obj']:.4f}  (arrival: {true_result['arrival_err']:.4f}, trough_err: {true_result['trough_err']:.4f})
  trough pred:  {true_result['trough_pred']:.4f}
  rank by obj:  {rank}/{len(results)}

Random (N={len(rand_obj)}):  mean obj={rand_mean:.4f}  std={rand_std:.4f}

VERDICT (combined obj): {verdict_obj}
VERDICT (trough only):  {'DISCRIMINATES' if rank_by_trough <= max(5, len(results) // 10) else 'no signal'} (rank {rank_by_trough}/{len(results)})
Runtime: {runtime_s:.1f}s
""")
