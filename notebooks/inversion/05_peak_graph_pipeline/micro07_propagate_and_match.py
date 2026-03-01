#!/usr/bin/env python3
"""
Micro-07 — Propagate-and-Match Pipeline.

Stage 1: At peak 1, sample 1M random SO(3), filter by 1% brightness tolerance,
         compute brightness gradients, save checkpoint.
Stage 2: For each candidate, scan omega on constraint plane (perp to gradient),
         propagate to peak 2 with full Euler dynamics, filter by brightness match.

Uses sequential for-loop (no multiprocessing) to avoid pickling issues.
"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment, save_results, attitude_error_deg
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
SEED       = 42
PEAK1_IDX  = 183
PEAK2_IDX  = 260
TOL_PCT    = 1        # Stage 1 brightness tolerance %
TOL2_PCT   = 10       # Stage 2 brightness tolerance %
G_THRESH   = 0.5      # |g| threshold: below → 3D grid, above → 2D perp-plane
DPHI       = 1e-5     # finite-diff step for gradient
N_TOTAL    = 1_000_000
BATCH      = 100_000
RESULTS_DIR = Path('data/results/inversion_diagnostics')
MAX_CANDIDATES = None  # Set to 10 for testing; None = all

# Omega grid: 2D (perp to g) or 3D (isotropic)
N_MAG_2D = 10; N_ANG_2D = 10   # 10×10 = 100 trials
N_MAG_3D = 5;  N_ANG_3D = 5    # 5×5×5 = 125 trials


def lofi_batch(k1, k2, eidx, ctx):
    """Vectorised lo-fi brightness for N attitudes at one epoch."""
    N = len(k1)
    art = {c: np.tile(m[eidx:eidx+1], (N, 1, 1))
           for c, m in ctx.art_matrices.items()}
    lit = create_no_shadow_lit_status(ctx.satellite, N)
    mags, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.full(N, ctx.obs_dist[eidx]),
        satellite=ctx.satellite, epochs=np.arange(N, dtype=float),
        pre_computed_matrices=art,
        generate_no_shadow=False, animate=False, show_progress=False)
    return mags


def build_omega_trials(gradient, omega_mags_2d, omega_angles_2d,
                       omega_mags_3d, n_ang_3d):
    """Build (N,3) array of omega vectors for one candidate."""
    gn = np.linalg.norm(gradient)
    if gn >= G_THRESH:
        g_hat = gradient / gn
        arb = np.array([1.,0.,0.]) if abs(g_hat[0]) < 0.9 else np.array([0.,1.,0.])
        e1 = np.cross(g_hat, arb); e1 /= np.linalg.norm(e1)
        e2 = np.cross(g_hat, e1)
        out = []
        for wm in omega_mags_2d:
            for wa in omega_angles_2d:
                out.append(wm * (np.cos(wa) * e1 + np.sin(wa) * e2))
        return np.array(out)
    else:
        out = []
        for wm in omega_mags_3d:
            for i_th in range(n_ang_3d):
                theta = np.pi * (i_th + 0.5) / n_ang_3d
                for i_ph in range(n_ang_3d):
                    phi = 2 * np.pi * i_ph / n_ang_3d
                    out.append(wm * np.array([np.sin(theta)*np.cos(phi),
                                              np.sin(theta)*np.sin(phi),
                                              np.cos(theta)]))
        return np.array(out)


if __name__ == '__main__':
    t0 = time.time()
    ctx = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=SEED,
        true_omega_deg=(0.5, -0.3, 2.0),
        end_time_utc='2020-02-05T11:00:00',
    )
    print(f"Setup: {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════
    # STAGE 1 — Load or generate candidate checkpoint
    # ══════════════════════════════════════════════════════════════
    ckpt1_path = RESULTS_DIR / 'micro07_stage1.npz'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if ckpt1_path.exists():
        print(f"\nStage 1: loading checkpoint {ckpt1_path}")
        ckpt = np.load(ckpt1_path)
        cand_q   = ckpt['candidate_q_wxyz']    # (M,4) wxyz
        cand_mag = ckpt['candidate_mags']
        cand_g   = ckpt['gradients']            # (M,3)
        g_norm   = ckpt['gradient_norms']
        ang_dists = ckpt['ang_dists_to_truth']
        obs_mag_p2 = float(ckpt['obs_mag_peak2'])
    else:
        print(f"\nStage 1: generating candidates (1M samples at peak {PEAK1_IDX})")
        # Reference brightness at peak 1 (truth attitude, lo-fi)
        truth_q1 = ctx.true_quaternions[PEAK1_IDX]
        sv = ctx.sun_pos[PEAK1_IDX] - ctx.sat_pos[PEAK1_IDX]
        ov = ctx.obs_pos[PEAK1_IDX] - ctx.sat_pos[PEAK1_IDX]
        R_tr = Rotation.from_quat([truth_q1[1], truth_q1[2], truth_q1[3], truth_q1[0]])
        Rm_tr = R_tr.as_matrix()
        k1t = Rm_tr @ sv; k1t /= np.linalg.norm(k1t)
        k2t = Rm_tr @ ov; k2t /= np.linalg.norm(k2t)
        ref_mag = float(lofi_batch(k1t[None], k2t[None], PEAK1_IDX, ctx)[0])
        tol = abs(ref_mag) * TOL_PCT / 100.0
        obs_mag_p2 = ctx.observed_lc[PEAK2_IDX]
        print(f"  ref_mag={ref_mag:.4f}, tol=+-{tol:.4f}")

        rng = np.random.default_rng(SEED)
        match_q, match_m = [], []
        for bi in range(N_TOTAL // BATCH):
            t1 = time.time()
            Rs = Rotation.random(BATCH, random_state=rng)
            Rm = Rs.as_matrix()
            k1 = np.einsum('nij,j->ni', Rm, sv); k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
            k2 = np.einsum('nij,j->ni', Rm, ov); k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
            mags = lofi_batch(k1, k2, PEAK1_IDX, ctx)
            mask = np.abs(mags - ref_mag) < tol
            if mask.any():
                match_q.append(Rs[mask].as_quat())   # xyzw
                match_m.append(mags[mask])
            cum = sum(len(x) for x in match_m)
            print(f"  Batch {bi+1}/{N_TOTAL//BATCH}: {mask.sum():>5} hits (cumul {cum:>5}) [{time.time()-t1:.1f}s]")

        all_q_xyzw = np.vstack(match_q)
        cand_mag = np.concatenate(match_m)
        R_cands = Rotation.from_quat(all_q_xyzw)
        ang_dists = np.rad2deg((R_cands.inv() * R_tr).magnitude())

        # Brightness gradients (finite differences)
        print("  Computing gradients...")
        axes = np.eye(3)
        Rm_all = R_cands.as_matrix()
        k1m = np.einsum('nij,j->ni', Rm_all, sv); k1m /= np.linalg.norm(k1m, axis=1, keepdims=True)
        k2m = np.einsum('nij,j->ni', Rm_all, ov); k2m /= np.linalg.norm(k2m, axis=1, keepdims=True)
        M_s1 = len(cand_mag)
        cand_g = np.zeros((M_s1, 3))
        for j in range(3):
            k1p = k1m + DPHI * np.cross(axes[j], k1m)
            k1p /= np.linalg.norm(k1p, axis=1, keepdims=True)
            k2p = k2m + DPHI * np.cross(axes[j], k2m)
            k2p /= np.linalg.norm(k2p, axis=1, keepdims=True)
            cand_g[:, j] = (lofi_batch(k1p, k2p, PEAK1_IDX, ctx) - cand_mag) / DPHI
        g_norm = np.linalg.norm(cand_g, axis=1)

        cand_q = np.column_stack([all_q_xyzw[:, 3], all_q_xyzw[:, :3]])  # → wxyz
        np.savez(ckpt1_path, candidate_q_wxyz=cand_q, candidate_mags=cand_mag,
                 gradients=cand_g, gradient_norms=g_norm, ang_dists_to_truth=ang_dists,
                 truth_q0=ctx.true_q0, truth_omega0=ctx.true_omega0,
                 peak1_idx=PEAK1_IDX, peak2_idx=PEAK2_IDX,
                 obs_mag_peak2=obs_mag_p2, ref_mag_peak1=ref_mag, tol=tol, seed=SEED)
        print(f"  Checkpoint saved: {ckpt1_path}")

    M = len(cand_mag)
    n_low_g = int(np.sum(g_norm < G_THRESH))
    print(f"\n  Candidates: {M:,}")
    print(f"  Nearest to truth: {ang_dists.min():.2f} deg")
    for d in [1, 5, 10]:
        print(f"    <{d} deg: {int(np.sum(ang_dists < d))}")
    print(f"  |g|: min={g_norm.min():.3f} med={np.median(g_norm):.3f} max={g_norm.max():.3f}")
    print(f"  |g| < {G_THRESH}: {n_low_g}/{M} ({100*n_low_g/M:.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # STAGE 2 — Coarse omega scan (sequential — no pickling issues)
    # ══════════════════════════════════════════════════════════════
    t_s2 = time.time()
    M_run = min(M, MAX_CANDIDATES) if MAX_CANDIDATES else M

    dt_bridge = ctx.observation_times[PEAK2_IDX] - ctx.observation_times[PEAK1_IDX]
    sv2 = ctx.sun_pos[PEAK2_IDX] - ctx.sat_pos[PEAK2_IDX]
    ov2 = ctx.obs_pos[PEAK2_IDX] - ctx.sat_pos[PEAK2_IDX]
    tol2 = abs(obs_mag_p2) * TOL2_PCT / 100.0
    bridge_times = np.array([0.0, dt_bridge])

    omega_mags_2d  = np.deg2rad(np.logspace(np.log10(0.1), np.log10(5.0), N_MAG_2D))
    omega_angles_2d = np.linspace(0, 2 * np.pi, N_ANG_2D, endpoint=False)
    omega_mags_3d  = np.deg2rad(np.logspace(np.log10(0.1), np.log10(5.0), N_MAG_3D))

    n_2d = int(np.sum(g_norm[:M_run] >= G_THRESH))
    n_3d = M_run - n_2d
    total_prop = n_2d * N_MAG_2D * N_ANG_2D + n_3d * N_MAG_3D * N_ANG_3D**2

    print(f"\n{'='*60}")
    print(f"STAGE 2: Coarse omega scan ({M_run} candidates)")
    print(f"{'='*60}")
    print(f"  2D (|g|>={G_THRESH}): {n_2d} × {N_MAG_2D*N_ANG_2D} = {n_2d*N_MAG_2D*N_ANG_2D:,}")
    print(f"  3D (|g|< {G_THRESH}): {n_3d} × {N_MAG_3D*N_ANG_3D**2} = {n_3d*N_MAG_3D*N_ANG_3D**2:,}")
    print(f"  Total propagations: {total_prop:,}")
    print(f"  Bridge dt={dt_bridge:.1f}s, tol2={TOL2_PCT}% → ±{tol2:.4f} mag")
    print()

    all_survivors = []
    for ci in range(M_run):
        q1 = cand_q[ci]
        omega_trials = build_omega_trials(
            cand_g[ci], omega_mags_2d, omega_angles_2d, omega_mags_3d, N_ANG_3D)
        nt = len(omega_trials)

        # Propagate each omega trial from peak 1 → peak 2
        q2_arr = np.zeros((nt, 4))
        omega2_arr = np.zeros((nt, 3))
        for ti in range(nt):
            qp, wp = propagate_attitude(q0=q1, omega0=omega_trials[ti],
                                        times=bridge_times, mode="tumbling",
                                        inertia_tensor=ctx.inertia_tensor)
            q2_arr[ti] = qp[-1]
            omega2_arr[ti] = wp[-1]

        # Batch brightness evaluation at peak 2
        R2 = Rotation.from_quat(q2_arr[:, [1, 2, 3, 0]]).as_matrix()
        k1_p2 = np.einsum('nij,j->ni', R2, sv2)
        k1_p2 /= np.linalg.norm(k1_p2, axis=1, keepdims=True)
        k2_p2 = np.einsum('nij,j->ni', R2, ov2)
        k2_p2 /= np.linalg.norm(k2_p2, axis=1, keepdims=True)
        mags2 = lofi_batch(k1_p2, k2_p2, PEAK2_IDX, ctx)

        # Filter survivors
        resids = np.abs(mags2 - obs_mag_p2)
        for ti in np.where(np.isfinite(resids) & (resids < tol2))[0]:
            all_survivors.append(dict(
                ci=ci, q1=q1, omega1=omega_trials[ti],
                q2_pred=q2_arr[ti], omega2_pred=omega2_arr[ti],
                mag2_pred=float(mags2[ti]), resid=float(resids[ti])))

        if (ci + 1) % 100 == 0 or ci == M_run - 1 or ci == 0:
            el = time.time() - t_s2
            rate = (ci + 1) / el if el > 0 else 1
            eta = (M_run - ci - 1) / rate
            print(f"  [{ci+1:>5}/{M_run}] {len(all_survivors):>5} survivors "
                  f"({el:.0f}s elapsed, ETA {eta:.0f}s)")

    n_surv = len(all_survivors)
    t_s2_el = time.time() - t_s2

    # ── Save & report ──
    print(f"\n{'='*60}")
    print(f"STAGE 2 COMPLETE — {t_s2_el:.1f}s")
    print(f"{'='*60}")
    print(f"  Survivors: {n_surv}")
    unique_cands = len(set(s['ci'] for s in all_survivors)) if n_surv else 0
    print(f"  Unique candidates with survivors: {unique_cands}/{M_run}")

    if n_surv > 0:
        s2_q1 = np.array([s['q1'] for s in all_survivors])
        s2_om1 = np.array([s['omega1'] for s in all_survivors])
        s2_q2 = np.array([s['q2_pred'] for s in all_survivors])
        s2_om2 = np.array([s['omega2_pred'] for s in all_survivors])
        s2_mag = np.array([s['mag2_pred'] for s in all_survivors])
        s2_res = np.array([s['resid'] for s in all_survivors])
        s2_ci = np.array([s['ci'] for s in all_survivors])

        ckpt2_path = RESULTS_DIR / 'micro07_stage2.npz'
        np.savez(ckpt2_path, q1=s2_q1, omega1=s2_om1, q2_pred=s2_q2,
                 omega2_pred=s2_om2, mag2_pred=s2_mag, resid=s2_res,
                 candidate_idx=s2_ci, obs_mag_peak2=obs_mag_p2,
                 dt_bridge=dt_bridge)
        print(f"  Checkpoint: {ckpt2_path}")

        # Top 5 by residual with attitude error from truth
        truth_q2 = ctx.true_quaternions[PEAK2_IDX]
        R_truth2 = Rotation.from_quat([truth_q2[1], truth_q2[2], truth_q2[3], truth_q2[0]])
        sorted_s = sorted(all_survivors, key=lambda x: x['resid'])

        print(f"\n  Top 5 survivors (by residual):")
        print(f"  {'#':>3} {'Cand':>5} {'AE1°':>7} {'AE2°':>7} "
              f"{'|ω|°/s':>7} {'Resid':>8} {'Mag2':>8}")
        for rank, s in enumerate(sorted_s[:5]):
            ae1 = float(ang_dists[s['ci']])
            R2p = Rotation.from_quat([s['q2_pred'][1], s['q2_pred'][2],
                                      s['q2_pred'][3], s['q2_pred'][0]])
            ae2 = np.rad2deg((R2p.inv() * R_truth2).magnitude())
            om_d = np.rad2deg(np.linalg.norm(s['omega1']))
            print(f"  {rank+1:>3} {s['ci']:>5} {ae1:>7.1f} {ae2:>7.1f} "
                  f"{om_d:>7.2f} {s['resid']:>8.4f} {s['mag2_pred']:>8.4f}")

        # Omega distribution of survivors
        om_surv = np.rad2deg(np.linalg.norm(s2_om1, axis=1))
        true_om_mag = np.rad2deg(np.linalg.norm(ctx.true_omega0))
        print(f"\n  True |ω| = {true_om_mag:.2f} °/s")
        print(f"  Survivor |ω| range: {om_surv.min():.2f} – {om_surv.max():.2f} °/s")
    else:
        print("  No survivors — tolerance may be too tight")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
