#!/usr/bin/env python3
"""Micro-13 — Roberto's full graph pipeline (small scale).

Stage 1: 50 candidates at 3 peaks via 1M random SO(3) sampling.
Stage 2: L-BFGS-B bridge optimisation (ω) between consecutive peaks (Euler dynamics).
Stage 3: Score bridges by intermediate lo-fi brightness residual.
Stage 4: Shortest-path graph search across 3 peaks.
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
from lib.experiment_setup import setup_experiment, save_results, attitude_error_deg
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ─────────────────────────────────────────────────────────
SEED = 42
PEAKS = [183, 260, 360]
N_SAMP, BATCH_SZ = 1_000_000, 100_000
TOL_PCT, N_CAND = 1, 50
N_INTER, N_WORKERS = 10, 8
DPHI = 1e-5
LAM_GRAD, LAM_RATE = 0.1, 0.01
RATE_PRIOR_DEG, MM_PRUNE_DEG = 5.0, 5.0
RD = Path('data/results/inversion_diagnostics')

# Module-level globals (forked workers inherit these)
CTX = INERTIA = PCANDS = PGRADS = BDT = INTER_TIMES = None


def lofi_batch(k1, k2, eidx):
    """Lo-fi brightness for N attitudes at one epoch."""
    N = len(k1)
    art = {c: np.tile(m[eidx:eidx+1], (N, 1, 1)) for c, m in CTX.art_matrices.items()}
    lit = create_no_shadow_lit_status(CTX.satellite, N)
    mags, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.full(N, CTX.obs_dist[eidx]), satellite=CTX.satellite,
        epochs=np.arange(N, dtype=float), pre_computed_matrices=art,
        generate_no_shadow=False, animate=False, show_progress=False)
    return mags


def bvecs_batch(q_arr, eidx):
    """Body-frame sun/obs unit vectors for N wxyz quaternions at one epoch."""
    R = Rotation.from_quat(q_arr[:, [1, 2, 3, 0]]).as_matrix()
    sv, ov = CTX.sun_pos[eidx] - CTX.sat_pos[eidx], CTX.obs_pos[eidx] - CTX.sat_pos[eidx]
    k1 = np.einsum('nij,j->ni', R, sv); k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 = np.einsum('nij,j->ni', R, ov); k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    return k1, k2


def compute_grads(q_arr, eidx):
    """Brightness gradient dB/dφ for N candidates at one epoch."""
    k1, k2 = bvecs_batch(q_arr, eidx)
    base = lofi_batch(k1, k2, eidx)
    g = np.zeros((len(q_arr), 3))
    for j in range(3):
        ax = np.zeros(3); ax[j] = 1.0
        k1p = k1 + DPHI * np.cross(ax, k1); k1p /= np.linalg.norm(k1p, axis=1, keepdims=True)
        k2p = k2 + DPHI * np.cross(ax, k2); k2p /= np.linalg.norm(k2p, axis=1, keepdims=True)
        g[:, j] = (lofi_batch(k1p, k2p, eidx) - base) / DPHI
    return g


def bridge_worker(args):
    """L-BFGS-B optimisation of ω for one (leg, i, j) bridge pair."""
    leg, i, j = args
    q1, q2, g1, dt = PCANDS[leg][i], PCANDS[leg + 1][j], PGRADS[leg][i], BDT[leg]
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    w0, tb = (R1.inv() * R2).as_rotvec() / dt, np.array([0.0, dt])

    def obj(w):
        qp, _ = propagate_attitude(q1, w, tb, mode="tumbling", inertia_tensor=INERTIA)
        Rp = Rotation.from_quat([qp[-1][1], qp[-1][2], qp[-1][3], qp[-1][0]])
        return (Rp.inv() * R2).magnitude()**2 + LAM_GRAD * np.dot(g1, w)**2

    try:
        res = minimize(obj, w0, method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-10})
        qp, _ = propagate_attitude(q1, res.x, tb, mode="tumbling", inertia_tensor=INERTIA)
        Rp = Rotation.from_quat([qp[-1][1], qp[-1][2], qp[-1][3], qp[-1][0]])
        return (leg, i, j, res.x.copy(), np.rad2deg((Rp.inv() * R2).magnitude()))
    except Exception:
        return (leg, i, j, w0.copy(), 180.0)


def score_worker(args):
    """Propagate one bridge at intermediate times."""
    leg, i, j, omega = args
    qp, _ = propagate_attitude(PCANDS[leg][i], omega, INTER_TIMES[leg],
                                mode="tumbling", inertia_tensor=INERTIA)
    return (leg, i, j, qp[1:-1].copy())


# ── Main ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    INERTIA = CTX.inertia_tensor
    true_q, true_w = propagate_attitude(CTX.true_q0, CTX.true_omega0,
                                         CTX.observation_times, mode="tumbling",
                                         inertia_tensor=INERTIA)
    print(f"Setup: {time.time()-t0:.1f}s\n", flush=True)

    # ═══ STAGE 1: 50 candidates per peak ══════════════════════════
    PCANDS, PGRADS, truth_idx = [], [], []
    rng = np.random.default_rng(SEED)
    for pi, ep in enumerate(PEAKS):
        ts = time.time()
        obs_mag = CTX.observed_lc[ep]
        tol = abs(obs_mag) * TOL_PCT / 100
        sv = CTX.sun_pos[ep] - CTX.sat_pos[ep]
        ov = CTX.obs_pos[ep] - CTX.sat_pos[ep]
        mq, mm = [], []
        for _ in range(N_SAMP // BATCH_SZ):
            Rs = Rotation.random(BATCH_SZ, random_state=rng)
            Rm = Rs.as_matrix()
            k1 = np.einsum('nij,j->ni', Rm, sv); k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
            k2 = np.einsum('nij,j->ni', Rm, ov); k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
            mags = lofi_batch(k1, k2, ep)
            hit = np.abs(mags - obs_mag) < tol
            if hit.any():
                mq.append(Rs[hit].as_quat()); mm.append(mags[hit])
        all_xyzw = np.vstack(mq)
        all_wxyz = np.column_stack([all_xyzw[:, 3], all_xyzw[:, :3]])
        R_t = Rotation.from_quat([true_q[ep][1], true_q[ep][2], true_q[ep][3], true_q[ep][0]])
        ad = np.rad2deg((Rotation.from_quat(all_xyzw).inv() * R_t).magnitude())
        ni = int(np.argmin(ad))
        n_all = len(all_wxyz)
        if n_all > N_CAND:
            rng2 = np.random.default_rng(SEED + ep)
            ch = set(rng2.choice(n_all, N_CAND, replace=False).tolist())
            ch.discard(ni); ch = list(ch)[:N_CAND - 1] + [ni]
            chosen = np.array(ch)
        else:
            chosen = np.arange(n_all)
        cands = all_wxyz[chosen]; adists = ad[chosen]
        tidx = int(np.argmin(adists)); truth_idx.append(tidx)
        PCANDS.append(cands); PGRADS.append(compute_grads(cands, ep))
        print(f"Peak {pi+1} (ep {ep}): obs={obs_mag:.3f}±{tol:.3f}, "
              f"{n_all:,} hits → {len(cands)} cands, "
              f"nearest={adists[tidx]:.2f}° [{time.time()-ts:.1f}s]", flush=True)

    nc = [len(c) for c in PCANDS]
    print(f"\nTruth path indices: {truth_idx}  (cands per peak: {nc})")
    for pi, ep in enumerate(PEAKS):
        print(f"  Pk{pi+1}: q_err={attitude_error_deg(PCANDS[pi][truth_idx[pi]], true_q[ep]):.2f}°, "
              f"|ω|={np.rad2deg(np.linalg.norm(true_w[ep])):.3f}°/s")
    np.savez(RD / 'm013_stage1.npz',
             c0=PCANDS[0], c1=PCANDS[1], c2=PCANDS[2],
             g0=PGRADS[0], g1=PGRADS[1], g2=PGRADS[2],
             truth_idx=truth_idx, peaks=PEAKS)
    print(f"Stage 1 checkpoint saved.\n", flush=True)

    # ═══ STAGE 2: Bridge optimisation ═════════════════════════════
    ot = CTX.observation_times
    BDT = [ot[PEAKS[1]] - ot[PEAKS[0]], ot[PEAKS[2]] - ot[PEAKS[1]]]
    pairs = [(l, i, j) for l in range(2) for i in range(nc[l]) for j in range(nc[l+1])]
    print(f"Stage 2: {len(pairs)} pairs, dt=[{BDT[0]:.1f},{BDT[1]:.1f}]s", flush=True)
    bw = [np.zeros((nc[l], nc[l+1], 3)) for l in range(2)]
    bmm = [np.full((nc[l], nc[l+1]), np.inf) for l in range(2)]
    ts2, done = time.time(), 0
    with get_context('fork').Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(bridge_worker, pairs, chunksize=50):
            leg, i, j, w, mm = r
            bw[leg][i, j], bmm[leg][i, j] = w, mm
            done += 1
            if done % 500 == 0:
                el = time.time() - ts2
                print(f"  [{done:>5}/{len(pairs)}] {el:.0f}s "
                      f"ETA {(len(pairs)-done)*el/done:.0f}s", flush=True)
    for l in range(2):
        nf = int((bmm[l] < MM_PRUNE_DEG).sum())
        print(f"  Leg {l}: {nf}/{nc[l]*nc[l+1]} bridges <{MM_PRUNE_DEG}°")
    print(f"  Stage 2: {time.time()-ts2:.0f}s\n", flush=True)
    np.savez(RD / 'm013_stage2.npz', bw_0=bw[0], bw_1=bw[1],
             bmm_0=bmm[0], bmm_1=bmm[1], bdt=BDT)

    # ═══ STAGE 3: Intermediate brightness scoring ═════════════════
    print(f"Stage 3: intermediate brightness scoring", flush=True)
    inter_ep = [np.round(np.linspace(PEAKS[l], PEAKS[l+1], N_INTER+2)[1:-1]).astype(int)
                for l in range(2)]
    INTER_TIMES = [np.concatenate([[0.0], ot[inter_ep[l]] - ot[PEAKS[l]], [BDT[l]]])
                   for l in range(2)]
    bcost = [np.full((nc[l], nc[l+1]), np.inf) for l in range(2)]
    brms = [np.full((nc[l], nc[l+1]), np.inf) for l in range(2)]
    for leg in range(2):
        fij = np.argwhere(bmm[leg] < MM_PRUNE_DEG)
        Nf = len(fij)
        if Nf == 0:
            print(f"  Leg {leg}: 0 feasible"); continue
        ts3 = time.time()
        sa = [(leg, int(ij[0]), int(ij[1]), bw[leg][ij[0], ij[1]]) for ij in fij]
        iq = {}
        with get_context('fork').Pool(N_WORKERS) as pool:
            for r in pool.imap_unordered(score_worker, sa, chunksize=50):
                iq[(r[1], r[2])] = r[3]
        mag_arr = np.zeros((Nf, N_INTER))
        for e in range(N_INTER):
            qb = np.array([iq[(int(fij[b][0]), int(fij[b][1]))][e] for b in range(Nf)])
            k1, k2 = bvecs_batch(qb, inter_ep[leg][e])
            mag_arr[:, e] = lofi_batch(k1, k2, inter_ep[leg][e])
        rms = np.sqrt(np.mean((mag_arr - CTX.observed_lc[inter_ep[leg]][None, :])**2, axis=1))
        for b in range(Nf):
            i, j = int(fij[b][0]), int(fij[b][1])
            wm = np.rad2deg(np.linalg.norm(bw[leg][i, j]))
            brms[leg][i, j] = rms[b]
            bcost[leg][i, j] = rms[b] + LAM_RATE * max(0.0, wm - RATE_PRIOR_DEG)
        print(f"  Leg {leg}: {Nf} scored [{time.time()-ts3:.1f}s] "
              f"RMS med={np.median(rms):.4f} min={rms.min():.4f}", flush=True)

    # ═══ STAGE 4: Graph search ════════════════════════════════════
    print(f"\nStage 4: graph search", flush=True)
    total = bcost[0][:, :, None] + bcost[1][None, :, :]
    flat = total.ravel()
    n_valid = int(np.isfinite(flat).sum())
    order = np.argsort(flat)
    ti, tj, tk = truth_idx
    tc = float(total[ti, tj, tk])
    tr = int((flat[np.isfinite(flat)] <= tc).sum()) if np.isfinite(tc) else -1

    print(f"\n{'='*85}")
    print(f"MICRO-13 RESULTS — Graph Pipeline ({nc[0]}×{nc[1]}×{nc[2]} nodes)")
    print(f"{'='*85}")
    print(f"Valid paths: {n_valid}/{nc[0]*nc[1]*nc[2]:,}")
    ct_str = "INF" if not np.isfinite(tc) else f"{tc:.4f}"
    print(f"Truth path ({ti},{tj},{tk}): cost={ct_str}, rank={tr}/{n_valid}")
    print(f"\nTruth bridge detail:")
    print(f"  Leg 0: mm={bmm[0][ti,tj]:.2f}° rms={brms[0][ti,tj]:.4f} cost={bcost[0][ti,tj]:.4f}")
    print(f"  Leg 1: mm={bmm[1][tj,tk]:.2f}° rms={brms[1][tj,tk]:.4f} cost={bcost[1][tj,tk]:.4f}")

    hdr = (f"{'#':>3} {'i':>3}{'j':>4}{'k':>4} {'Cost':>8} "
           f"{'AE1':>6} {'AE2':>6} {'AE3':>6} "
           f"{'wE01':>7} {'wE12':>7} {'RMS01':>7} {'RMS12':>7} {'T':>3}")
    print(f"\nTop 10 paths:\n{hdr}")
    for r in range(min(10, n_valid)):
        fi = order[r]
        if not np.isfinite(flat[fi]):
            break
        i, j, k = np.unravel_index(fi, total.shape)
        ae = [attitude_error_deg(PCANDS[p][x], true_q[PEAKS[p]])
              for p, x in enumerate([i, j, k])]
        we0 = np.rad2deg(np.linalg.norm(bw[0][i, j] - true_w[PEAKS[0]]))
        we1 = np.rad2deg(np.linalg.norm(bw[1][j, k] - true_w[PEAKS[1]]))
        it = (i == ti and j == tj and k == tk)
        print(f"{r+1:>3} {i:>3}{j:>4}{k:>4} {flat[fi]:>8.4f} "
              f"{ae[0]:>6.1f} {ae[1]:>6.1f} {ae[2]:>6.1f} "
              f"{we0:>7.2f} {we1:>7.2f} {brms[0][i,j]:>7.4f} {brms[1][j,k]:>7.4f} "
              f"{'<T' if it else '':>3}")

    save_results(RD / 'm013_graph_pipeline.json', {
        'config': {'peaks': PEAKS, 'n_cand': N_CAND, 'n_samples': N_SAMP,
                   'tol_pct': TOL_PCT, 'n_inter': N_INTER, 'mm_prune_deg': MM_PRUNE_DEG,
                   'lam_grad': LAM_GRAD, 'lam_rate': LAM_RATE, 'seed': SEED},
        'n_valid_paths': n_valid, 'truth_rank': tr, 'truth_cost': tc,
        'truth_idx': truth_idx, 'n_cands': nc,
        'runtime_s': round(time.time() - t0, 1)})
    print(f"\nSaved: {RD}/m013_graph_pipeline.json")
    print(f"Total runtime: {time.time()-t0:.1f}s")
