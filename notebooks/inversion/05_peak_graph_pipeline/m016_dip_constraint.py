#!/usr/bin/env python3
"""Micro-16 — Dip (trough) constraint experiment.

Scientific question: Do brightness troughs between peaks discriminate the true
path better than intermediate-epoch scoring (used in m013/14)?

Hypothesis: the brightness landscape is steeper at troughs, so hi-fi scoring
at the trough epoch should rank the true path higher than the ~13k/121k seen
with uniform intermediate-epoch scoring.
"""
import sys, time, json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, brightness_single_epoch, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
SEED = 42
N_PAIRS = 100
N_WORKERS = 8

# Module-level globals (forked workers inherit these)
CTX = None
INERTIA = None


def score_worker(args):
    i, j, q_start, omega, t_trough, trough_idx, obs_bright = args
    times = np.array([0.0, t_trough])
    try:
        qp, _ = propagate_attitude(q_start, omega, times, "tumbling", INERTIA)
        q_at_trough = qp[-1]
        pred_bright = brightness_single_epoch(q_at_trough, trough_idx, CTX, use_shadows=True)
        score = abs(pred_bright - obs_bright)
        return {'i': int(i), 'j': int(j), 'pred': float(pred_bright),
                'score': float(score), 'ok': True}
    except Exception as e:
        return {'i': int(i), 'j': int(j), 'pred': float('nan'),
                'score': float('inf'), 'ok': False}


if __name__ == '__main__':
    t0 = time.time()

    # ── Setup ─────────────────────────────────────────────────────────
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    INERTIA = CTX.inertia_tensor
    OBS_TIMES = CTX.observation_times
    print(f"Setup: {time.time()-t0:.1f}s\n", flush=True)

    # ── Load m013 checkpoints ───────────────────────────────────────
    stage1 = np.load(RESULTS_DIR / "m013_stage1.npz")
    stage2 = np.load(RESULTS_DIR / "m013_stage2.npz")

    PEAKS = stage1["peaks"]            # [183, 260, 360]
    CANDS = [stage1["c0"], stage1["c1"], stage1["c2"]]
    TRUTH_IDX = stage1["truth_idx"]
    BW = [stage2["bw_0"], stage2["bw_1"]]
    BDT = stage2["bdt"]

    # ── Find trough on leg 0 ──────────────────────────────────────────
    PEAK_A = int(PEAKS[0])   # 183
    PEAK_B = int(PEAKS[1])   # 260

    lc_segment = CTX.observed_lc[PEAK_A:PEAK_B + 1]
    trough_offset = int(np.argmax(lc_segment))  # argmax = maximum magnitude (= brightness trough)
    TROUGH_IDX = PEAK_A + trough_offset
    t_trough = float(OBS_TIMES[TROUGH_IDX] - OBS_TIMES[PEAK_A])
    obs_brightness_at_trough = float(CTX.observed_lc[TROUGH_IDX])
    true_brightness_at_trough = float(CTX.true_lc[TROUGH_IDX])

    print(f"Trough epoch: {TROUGH_IDX} (t_from_peak_A = {t_trough:.1f}s)")
    print(f"Observed brightness at trough: {obs_brightness_at_trough:.4f}")
    print(f"True brightness at trough:     {true_brightness_at_trough:.4f}")
    print(flush=True)

    # ── Sample pairs ─────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    N0, N1 = len(CANDS[0]), len(CANDS[1])
    all_indices = [(i, j) for i in range(N0) for j in range(N1)]
    rng.shuffle(all_indices)

    true_pair = (int(TRUTH_IDX[0]), int(TRUTH_IDX[1]))
    sample_pairs = [true_pair] + [p for p in all_indices if p != true_pair][:N_PAIRS]

    args_list = [
        (i, j, CANDS[0][i], BW[0][i, j], t_trough, TROUGH_IDX, obs_brightness_at_trough)
        for i, j in sample_pairs
    ]

    print(f"Scoring {len(args_list)} pairs (1 truth + {N_PAIRS} random) with {N_WORKERS} workers...",
          flush=True)
    t_score = time.time()

    fork_ctx = mp.get_context('fork')
    with fork_ctx.Pool(N_WORKERS) as pool:
        results = pool.map(score_worker, args_list)

    print(f"Scoring done: {time.time()-t_score:.1f}s\n", flush=True)

    # ── Analyse ───────────────────────────────────────────────────────
    true_result = results[0]
    all_scores = np.array([r['score'] for r in results])
    valid_mask = np.array([r['ok'] for r in results])

    true_score = true_result['score']
    # Rank = how many valid pairs score equal or better (lower = better match)
    rank = int(np.sum(all_scores[valid_mask] <= true_score))

    n_valid = int(valid_mask.sum())
    rand_scores = all_scores[valid_mask & (np.arange(len(results)) > 0)]
    rand_mean = float(np.mean(rand_scores)) if len(rand_scores) > 0 else float('nan')
    rand_std = float(np.std(rand_scores)) if len(rand_scores) > 0 else float('nan')

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: histogram of trough brightness prediction errors
    ax = axes[0]
    rand_valid_scores = all_scores[valid_mask][1:]  # skip true pair at index 0
    ax.hist(rand_valid_scores, bins=20, color='steelblue', alpha=0.8, label='Random pairs')
    ax.axvline(true_score, color='red', lw=2, label=f'True pair ({true_score:.4f})')
    if len(rand_valid_scores) > 0:
        ax.axvline(np.mean(rand_valid_scores), color='grey', lw=1.5, ls='--',
                   label=f'Mean random ({np.mean(rand_valid_scores):.4f})')
    ax.set_xlabel('|predicted − observed| at trough (mag)')
    ax.set_ylabel('Count')
    ax.set_title('Trough brightness error — hi-fi scoring')
    ax.set_title(f'Trough brightness error — hi-fi scoring\nTrue pair rank: {rank}/{n_valid}')
    ax.legend(fontsize=9)

    # Panel 2: predicted brightness vs target
    ax2 = axes[1]
    rand_preds = np.array([r['pred'] for r in results[1:] if r['ok']])
    rand_preds_sorted = np.sort(rand_preds)
    ax2.plot(rand_preds_sorted, np.arange(len(rand_preds_sorted)),
             'o', color='steelblue', markersize=4, alpha=0.7, label='Random pairs')
    if true_result['ok']:
        pred_rank = int(np.sum(rand_preds_sorted <= true_result['pred']))
        ax2.plot(true_result['pred'], pred_rank, 'r*', markersize=14, label='True pair')
    ax2.axvline(obs_brightness_at_trough, color='black', lw=1.5, ls='--',
                label=f'Observed trough ({obs_brightness_at_trough:.4f})')
    ax2.set_xlabel('Predicted brightness at trough (mag)')
    ax2.set_ylabel('Rank (ascending predicted brightness)')
    ax2.set_title('Predicted trough brightness vs target')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "m016_dip_constraint.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {fig_path}")

    # ── Save JSON ─────────────────────────────────────────────────────
    runtime_s = float(time.time() - t0)
    results_out = {
        'config': {
            'leg': 0,
            'peak_A': PEAK_A,
            'peak_B': PEAK_B,
            'trough_idx': int(TROUGH_IDX),
            't_trough_s': t_trough,
            'n_pairs_tested': n_valid,
            'n_workers': N_WORKERS,
        },
        'observed_brightness_at_trough': obs_brightness_at_trough,
        'true_brightness_at_trough': true_brightness_at_trough,
        'true_pair_score': float(true_score),
        'true_pair_rank': rank,
        'true_pair_rank_out_of': n_valid,
        'random_score_mean': rand_mean,
        'random_score_std': rand_std,
        'true_pair': {
            'i': true_result['i'], 'j': true_result['j'],
            'pred': true_result['pred'] if true_result['ok'] else None,
        },
        'runtime_s': round(runtime_s, 1),
    }
    json_path = RESULTS_DIR / "m016_dip_constraint.json"
    save_results(json_path, results_out)
    print(f"Results saved: {json_path}")

    # ── Console summary ───────────────────────────────────────────────
    verdict = ("trough DISCRIMINATES" if rank <= max(5, n_valid // 10)
               else "does NOT discriminate")
    print(f"""
=== m016: dip constraint experiment ===
Leg 0: peak {PEAK_A} → peak {PEAK_B}
Trough at epoch: {TROUGH_IDX} (t_from_peak = {t_trough:.1f}s)
Observed brightness at trough: {obs_brightness_at_trough:.4f}
True brightness at trough:     {true_brightness_at_trough:.4f}

True pair:
  Predicted brightness: {true_result['pred']:.4f}
  Score (error): {true_score:.4f} mag
  Rank: {rank} / {n_valid}

Random pairs (N={len(rand_scores)}):
  Mean score: {rand_mean:.4f}, Std: {rand_std:.4f}

VERDICT: {verdict}
  (m014 baseline: truth rank ~13,000/121,326)
Runtime: {runtime_s:.1f}s
""")
