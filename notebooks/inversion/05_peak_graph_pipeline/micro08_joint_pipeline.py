#!/usr/bin/env python3
"""
Micro-08 — Full Joint Constraint Pipeline.

Loads peak-1 candidates from micro07_stage1.npz checkpoint.
For each candidate: scan omega on plane perp to brightness gradient,
propagate to peak 2 with Euler dynamics, apply three joint filters:
  (a) Brightness match at peak 2 (3%)
  (b) Derivative constraint: |g2 · omega2| < threshold
  (c) Brightness match at peak 3 after further propagation (3%)

Uses multiprocessing.Pool(8) — ctx inherited via fork (module-level global).
"""
import sys, time, numpy as np
from pathlib import Path
from multiprocessing import Pool, get_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment, save_results
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
SEED = 42
PEAK1, PEAK2, PEAK3 = 183, 260, 360
TOL_B_PCT = 3
DERIV_THRESHOLDS = [0.1, 0.5, 1.0]
DPHI = 1e-5
N_MAG, N_ANG = 20, 20
N_WORKERS = 8
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# Module-level globals (inherited by forked workers)
CTX = None
CAND_Q = None
CAND_G = None
BRIDGE12 = None
OBS_B2 = None
TOL_B2 = None


def lofi_batch(k1, k2, eidx, ctx):
    N = len(k1)
    art = {c: np.tile(m[eidx:eidx+1], (N, 1, 1)) for c, m in ctx.art_matrices.items()}
    lit = create_no_shadow_lit_status(ctx.satellite, N)
    mags, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.full(N, ctx.obs_dist[eidx]),
        satellite=ctx.satellite, epochs=np.arange(N, dtype=float),
        pre_computed_matrices=art, generate_no_shadow=False, animate=False, show_progress=False)
    return mags


def bvecs_batch(q_wxyz_arr, eidx, ctx):
    R = Rotation.from_quat(q_wxyz_arr[:, [1, 2, 3, 0]]).as_matrix()
    sv = ctx.sun_pos[eidx] - ctx.sat_pos[eidx]
    ov = ctx.obs_pos[eidx] - ctx.sat_pos[eidx]
    k1 = np.einsum('nij,j->ni', R, sv); k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 = np.einsum('nij,j->ni', R, ov); k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    return k1, k2


def grad_batch(q_arr, eidx, ctx):
    k1, k2 = bvecs_batch(q_arr, eidx, ctx)
    base = lofi_batch(k1, k2, eidx, ctx)
    g = np.zeros((len(q_arr), 3))
    for j in range(3):
        ax = np.zeros(3); ax[j] = 1.0
        k1p = k1 + DPHI * np.cross(ax, k1); k1p /= np.linalg.norm(k1p, axis=1, keepdims=True)
        k2p = k2 + DPHI * np.cross(ax, k2); k2p /= np.linalg.norm(k2p, axis=1, keepdims=True)
        g[:, j] = (lofi_batch(k1p, k2p, eidx, ctx) - base) / DPHI
    return g


def omega_grid(g_vec):
    g_hat = g_vec / np.linalg.norm(g_vec)
    arb = np.array([1., 0., 0.]) if abs(g_hat[0]) < 0.9 else np.array([0., 1., 0.])
    e1 = np.cross(g_hat, arb); e1 /= np.linalg.norm(e1)
    e2 = np.cross(g_hat, e1)
    mags = np.deg2rad(np.logspace(np.log10(0.1), np.log10(5.0), N_MAG))
    angs = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)
    mm, aa = np.meshgrid(mags, angs, indexing='ij')
    dirs = np.cos(aa.ravel())[:, None] * e1 + np.sin(aa.ravel())[:, None] * e2
    return mm.ravel()[:, None] * dirs


def process_candidate(ci):
    """Process one candidate — runs in forked worker, reads module-level globals."""
    omegas = omega_grid(CAND_G[ci])
    nt = len(omegas)
    q2 = np.zeros((nt, 4)); w2 = np.zeros((nt, 3))
    for ti in range(nt):
        qp, wp = propagate_attitude(q0=CAND_Q[ci], omega0=omegas[ti],
                                    times=BRIDGE12, mode="tumbling",
                                    inertia_tensor=CTX.inertia_tensor)
        q2[ti], w2[ti] = qp[-1], wp[-1]

    k1p, k2p = bvecs_batch(q2, PEAK2, CTX)
    m2 = lofi_batch(k1p, k2p, PEAK2, CTX)
    res = np.abs(m2 - OBS_B2)
    hits = np.where(np.isfinite(res) & (res < TOL_B2))[0]
    survivors = []
    for ti in hits:
        survivors.append((ci, CAND_Q[ci].copy(), omegas[ti].copy(),
                          q2[ti].copy(), w2[ti].copy(), float(res[ti])))
    return survivors


if __name__ == '__main__':
    t0 = time.time()
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    _, true_omegas = propagate_attitude(q0=CTX.true_q0, omega0=CTX.true_omega0,
                                        times=CTX.observation_times, mode="tumbling",
                                        inertia_tensor=CTX.inertia_tensor)
    print(f"Setup: {time.time()-t0:.1f}s", flush=True)

    # Load checkpoint
    ckpt = np.load(RESULTS_DIR / 'micro07_stage1.npz')
    CAND_Q = ckpt['candidate_q_wxyz']
    CAND_G = ckpt['gradients']
    ang_dists = ckpt['ang_dists_to_truth']
    M = len(CAND_Q)

    OBS_B2 = CTX.observed_lc[PEAK2]
    obs_b3 = CTX.observed_lc[PEAK3]
    TOL_B2 = abs(OBS_B2) * TOL_B_PCT / 100
    tol_b3 = abs(obs_b3) * TOL_B_PCT / 100
    dt_12 = CTX.observation_times[PEAK2] - CTX.observation_times[PEAK1]
    dt_23 = CTX.observation_times[PEAK3] - CTX.observation_times[PEAK2]
    BRIDGE12 = np.array([0.0, dt_12])
    bridge23 = np.array([0.0, dt_23])

    print(f"Candidates: {M}")
    print(f"Peaks: {PEAK1}->{PEAK2}->{PEAK3}, dt12={dt_12:.1f}s, dt23={dt_23:.1f}s")
    print(f"B_obs: p2={OBS_B2:.4f}(+-{TOL_B2:.4f}), p3={obs_b3:.4f}(+-{tol_b3:.4f})")
    print(f"Grid: {N_MAG}x{N_ANG}={N_MAG*N_ANG}/cand, total={M*N_MAG*N_ANG:,} propagations")
    print(f"Workers: {N_WORKERS}\n", flush=True)

    # ── Stage 1: Propagate & brightness filter at peak 2 (parallel) ──
    t1 = time.time()
    all_survivors = []
    done = 0
    with get_context('fork').Pool(N_WORKERS) as pool:
        for result in pool.imap_unordered(process_candidate, range(M), chunksize=10):
            all_survivors.extend(result)
            done += 1
            if done % 200 == 0 or done == M:
                el = time.time() - t1
                rate = done / el if el > 0 else 1
                print(f"  [{done:>5}/{M}] {len(all_survivors)} B-survivors "
                      f"({el:.0f}s, ETA {(M-done)/rate:.0f}s)", flush=True)

    n_bs = len(all_survivors)
    print(f"\nBrightness filter (peak2, {TOL_B_PCT}%): {n_bs} survivors [{time.time()-t1:.0f}s]",
          flush=True)
    if n_bs == 0:
        print("No survivors.")
        sys.exit(0)

    # Unpack survivors
    s_ci = np.array([s[0] for s in all_survivors])
    s_q1 = np.array([s[1] for s in all_survivors])
    s_om1 = np.array([s[2] for s in all_survivors])
    s_q2 = np.array([s[3] for s in all_survivors])
    s_om2 = np.array([s[4] for s in all_survivors])
    s_res2 = np.array([s[5] for s in all_survivors])

    # ── Stage 2: Derivative filter at peak 2 ──
    print(f"Computing gradients at peak 2 ({n_bs} survivors)...", flush=True)
    g2 = grad_batch(s_q2, PEAK2, CTX)
    gdot = np.abs(np.sum(g2 * s_om2, axis=1))

    # ── Stage 3: Propagate all B-survivors to peak 3 ──
    print(f"Propagating {n_bs} survivors to peak 3...", flush=True)
    t3 = time.time()
    q3 = np.zeros((n_bs, 4))
    for i in range(n_bs):
        qp, _ = propagate_attitude(q0=s_q2[i], omega0=s_om2[i], times=bridge23,
                                   mode="tumbling", inertia_tensor=CTX.inertia_tensor)
        q3[i] = qp[-1]
    k1_3, k2_3 = bvecs_batch(q3, PEAK3, CTX)
    m3 = lofi_batch(k1_3, k2_3, PEAK3, CTX)
    res3 = np.abs(m3 - obs_b3)
    print(f"  Done ({time.time()-t3:.1f}s)", flush=True)

    # ── Results ──
    true_om_p1 = true_omegas[PEAK1]
    R_t2 = Rotation.from_quat([CTX.true_quaternions[PEAK2][1], CTX.true_quaternions[PEAK2][2],
                                CTX.true_quaternions[PEAK2][3], CTX.true_quaternions[PEAK2][0]])

    print(f"\n{'='*78}")
    print(f"JOINT FILTER RESULTS")
    print(f"{'='*78}")
    print(f"True |omega| at peak1 = {np.rad2deg(np.linalg.norm(true_om_p1)):.3f} deg/s")

    p3_pass = res3 < tol_b3
    print(f"\nReference (no derivative filter):")
    print(f"  B-filter (peak2):           {n_bs}")
    print(f"  B-filter (peak2) + peak3:   {int(p3_pass.sum())}")

    results = {}
    for th in DERIV_THRESHOLDS:
        dm = gdot < th
        n_d = int(dm.sum())
        jm = dm & p3_pass
        n_j = int(jm.sum())
        u_d = len(set(s_ci[dm].tolist())) if n_d > 0 else 0
        u_j = len(set(s_ci[jm].tolist())) if n_j > 0 else 0

        print(f"\n-- |g2*w2| < {th} --")
        print(f"  B-filter:    {n_bs}")
        print(f"  + deriv:     {n_d} ({u_d} unique candidates)")
        print(f"  + peak3:     {n_j} ({u_j} unique candidates)")

        results[str(th)] = {'n_bright': n_bs, 'n_deriv': n_d, 'n_joint': n_j,
                            'u_deriv': u_d, 'u_joint': u_j}

        if n_j > 0:
            jidx = np.where(jm)[0]
            comb = s_res2[jidx] + res3[jidx]
            order = np.argsort(comb)
            n_show = min(10, len(order))
            print(f"  {'#':>3} {'Ci':>5} {'AE1':>7} {'AE2':>7} {'|w|d/s':>7} "
                  f"{'wErr':>7} {'Res2':>7} {'Res3':>7} {'|g*w|':>7}")
            for r in range(n_show):
                si = jidx[order[r]]
                ae1 = float(ang_dists[s_ci[si]])
                R2 = Rotation.from_quat([s_q2[si][1], s_q2[si][2],
                                         s_q2[si][3], s_q2[si][0]])
                ae2 = np.rad2deg((R2.inv() * R_t2).magnitude())
                omd = np.rad2deg(np.linalg.norm(s_om1[si]))
                oerr = np.rad2deg(np.linalg.norm(s_om1[si] - true_om_p1))
                print(f"  {r+1:>3} {s_ci[si]:>5} {ae1:>7.1f} {ae2:>7.1f} {omd:>7.2f} "
                      f"{oerr:>7.2f} {s_res2[si]:>7.4f} {res3[jidx[order[r]]]:>7.4f} "
                      f"{gdot[si]:>7.3f}")

    # ── Save checkpoints ──
    np.savez(RESULTS_DIR / 'micro08_joint.npz',
             s_ci=s_ci, s_q1=s_q1, s_om1=s_om1, s_q2=s_q2, s_om2=s_om2,
             s_res2=s_res2, gdot=gdot, q3_pred=q3, res3=res3,
             true_omega_p1=true_om_p1, g2=g2)
    save_results(RESULTS_DIR / 'micro08_joint.json', {
        'config': {'peaks': [PEAK1, PEAK2, PEAK3], 'tol_pct': TOL_B_PCT,
                   'deriv_thresholds': DERIV_THRESHOLDS,
                   'grid': [N_MAG, N_ANG], 'seed': SEED},
        'n_candidates': M, 'n_brightness_survivors': n_bs,
        'results': results, 'runtime_s': round(time.time() - t0, 1)})
    print(f"\nSaved to {RESULTS_DIR}/micro08_joint.*")
    print(f"Total runtime: {time.time()-t0:.1f}s")
