#!/usr/bin/env python3
"""Micro-14 — Hi-fi rescore of micro-13 graph pipeline.

Loads micro-13 checkpoints (Stage 1 candidates + Stage 2 bridge omegas).
Replaces lo-fi intermediate brightness scoring with hi-fi (ray-traced shadows).
No new sampling or optimisation — only rescoring and graph search.
"""
import sys, time, numpy as np
from pathlib import Path
from multiprocessing import get_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from lib.experiment_setup import (setup_experiment, brightness_single_epoch,
                                  attitude_error_deg, save_results)
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ─────────────────────────────────────────────────────────
N_INTER, N_WORKERS = 10, 8
LAM_RATE, RATE_PRIOR_DEG, MM_PRUNE_DEG = 0.01, 5.0, 5.0
RD = Path('data/results/inversion_diagnostics')

# Module-level globals (forked workers inherit these)
CTX = None


def hifi_bridge_worker(args):
    """Score one bridge: propagate + 10 hi-fi brightness evals."""
    leg, i, j, q_start, omega, dt, inter_frac, peaks = args
    t_inter = np.array([0.0] + [dt * f for f in inter_frac] + [dt])
    qp, _ = propagate_attitude(q_start, omega, t_inter,
                                mode="tumbling", inertia_tensor=CTX.inertia_tensor)
    # Evaluate hi-fi brightness at each intermediate epoch
    ep_start, ep_end = peaks[leg], peaks[leg + 1]
    inter_ep = np.round(np.linspace(ep_start, ep_end, N_INTER + 2)[1:-1]).astype(int)
    mags = np.array([brightness_single_epoch(qp[1 + e], int(inter_ep[e]), CTX,
                                              use_shadows=True)
                     for e in range(N_INTER)])
    obs = CTX.observed_lc[inter_ep]
    rms = float(np.sqrt(np.mean((mags - obs) ** 2)))
    wdeg = float(np.rad2deg(np.linalg.norm(omega)))
    cost = rms + LAM_RATE * max(0.0, wdeg - RATE_PRIOR_DEG) ** 2
    return (leg, i, j, rms, cost)


# ── Main ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()

    # Setup
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    true_q, true_w = propagate_attitude(CTX.true_q0, CTX.true_omega0,
                                         CTX.observation_times, mode="tumbling",
                                         inertia_tensor=CTX.inertia_tensor)
    t_setup = time.time() - t0
    print(f"Setup: {t_setup:.1f}s\n", flush=True)

    # Load checkpoints
    s1 = np.load(RD / 'micro13_stage1.npz')
    s2 = np.load(RD / 'micro13_stage2.npz')
    PCANDS = [s1['c0'], s1['c1'], s1['c2']]
    peaks = list(s1['peaks'])
    truth_idx = list(s1['truth_idx'])
    bw = [s2['bw_0'], s2['bw_1']]
    bmm = [s2['bmm_0'], s2['bmm_1']]
    bdt = list(s2['bdt'])
    nc = [len(c) for c in PCANDS]
    print(f"Loaded: {nc} cands, peaks={peaks}, truth_idx={truth_idx}")
    for l in range(2):
        nf = int((bmm[l] < MM_PRUNE_DEG).sum())
        print(f"  Leg {l}: {nf} feasible bridges (bmm<{MM_PRUNE_DEG}°)")

    # Precompute intermediate fractions (same for both legs, relative)
    inter_frac = np.linspace(0, 1, N_INTER + 2)[1:-1].tolist()

    # ═══ Hi-fi scoring ═════════════════════════════════════════════
    brms = [np.full((nc[l], nc[l + 1]), np.inf) for l in range(2)]
    bcost = [np.full((nc[l], nc[l + 1]), np.inf) for l in range(2)]
    t_legs = []

    for leg in range(2):
        fij = np.argwhere(bmm[leg] < MM_PRUNE_DEG)
        nf = len(fij)
        if nf == 0:
            print(f"  Leg {leg}: 0 feasible"); t_legs.append(0); continue
        print(f"\nLeg {leg}: scoring {nf} bridges (hi-fi, {N_INTER} epochs each) ...",
              flush=True)
        ts = time.time()
        tasks = [(leg, int(ij[0]), int(ij[1]),
                  PCANDS[leg][ij[0]], bw[leg][ij[0], ij[1]],
                  bdt[leg], inter_frac, peaks)
                 for ij in fij]
        done = 0
        with get_context('fork').Pool(N_WORKERS) as pool:
            for r in pool.imap_unordered(hifi_bridge_worker, tasks, chunksize=4):
                _, i, j, rms, cost = r
                brms[leg][i, j] = rms
                bcost[leg][i, j] = cost
                done += 1
                if done % 500 == 0:
                    el = time.time() - ts
                    print(f"  [{done:>5}/{nf}] {el:.0f}s "
                          f"ETA {(nf - done) * el / done:.0f}s", flush=True)
        t_leg = time.time() - ts
        t_legs.append(t_leg)
        fin_rms = brms[leg][fij[:, 0], fij[:, 1]]
        print(f"  Leg {leg}: {nf} scored [{t_leg:.1f}s] "
              f"RMS med={np.median(fin_rms):.4f} min={fin_rms.min():.4f}", flush=True)

    # ═══ Graph search ══════════════════════════════════════════════
    ts4 = time.time()
    total = bcost[0][:, :, None] + bcost[1][None, :, :]
    flat = total.ravel()
    n_valid = int(np.isfinite(flat).sum())
    order = np.argsort(flat)
    ti, tj, tk = truth_idx
    tc = float(total[ti, tj, tk])
    tr = int((flat[np.isfinite(flat)] <= tc).sum()) if np.isfinite(tc) else -1
    t_graph = time.time() - ts4

    print(f"\n{'=' * 85}")
    print(f"MICRO-14 RESULTS — Hi-fi Rescore ({nc[0]}×{nc[1]}×{nc[2]} nodes)")
    print(f"{'=' * 85}")
    print(f"Valid paths: {n_valid}/{nc[0] * nc[1] * nc[2]:,}")
    ct_str = "INF" if not np.isfinite(tc) else f"{tc:.4f}"
    print(f"Truth path ({ti},{tj},{tk}): cost={ct_str}, rank={tr}/{n_valid}")
    print(f"\nTruth bridge detail:")
    print(f"  Leg 0: bmm={bmm[0][ti, tj]:.2f}° rms={brms[0][ti, tj]:.4f} "
          f"cost={bcost[0][ti, tj]:.4f}")
    print(f"  Leg 1: bmm={bmm[1][tj, tk]:.2f}° rms={brms[1][tj, tk]:.4f} "
          f"cost={bcost[1][tj, tk]:.4f}")

    hdr = (f"{'#':>3} {'i':>3}{'j':>4}{'k':>4} {'Cost':>8} "
           f"{'AE1':>6} {'AE2':>6} {'AE3':>6} "
           f"{'wE01':>7} {'wE12':>7} {'RMS01':>7} {'RMS12':>7} {'T':>3}")
    print(f"\nTop 10 paths:\n{hdr}")
    for r in range(min(10, n_valid)):
        fi = order[r]
        if not np.isfinite(flat[fi]):
            break
        i, j, k = np.unravel_index(fi, total.shape)
        ae = [attitude_error_deg(PCANDS[p][x], true_q[peaks[p]])
              for p, x in enumerate([i, j, k])]
        we0 = np.rad2deg(np.linalg.norm(bw[0][i, j] - true_w[peaks[0]]))
        we1 = np.rad2deg(np.linalg.norm(bw[1][j, k] - true_w[peaks[1]]))
        it = (i == ti and j == tj and k == tk)
        print(f"{r + 1:>3} {i:>3}{j:>4}{k:>4} {flat[fi]:>8.4f} "
              f"{ae[0]:>6.1f} {ae[1]:>6.1f} {ae[2]:>6.1f} "
              f"{we0:>7.2f} {we1:>7.2f} {brms[0][i, j]:>7.4f} {brms[1][j, k]:>7.4f} "
              f"{'<T' if it else '':>3}")

    t_total = time.time() - t0
    print(f"\nTiming: setup={t_setup:.1f}s leg0={t_legs[0]:.1f}s "
          f"leg1={t_legs[1]:.1f}s graph={t_graph:.2f}s total={t_total:.1f}s")

    save_results(RD / 'micro14_hifi_rescore.json', {
        'config': {'peaks': [int(p) for p in peaks], 'n_cands': nc,
                   'n_inter': N_INTER,
                   'mm_prune_deg': MM_PRUNE_DEG, 'lam_rate': LAM_RATE,
                   'rate_prior_deg': RATE_PRIOR_DEG, 'scoring': 'hi-fi'},
        'n_valid_paths': n_valid, 'truth_rank': tr,
        'truth_cost': tc if np.isfinite(tc) else None,
        'truth_idx': [int(x) for x in truth_idx], 'n_cands': nc,
        'leg_timing_s': [round(t, 1) for t in t_legs],
        'runtime_s': round(t_total, 1)})
    print(f"Saved: {RD}/micro14_hifi_rescore.json")
