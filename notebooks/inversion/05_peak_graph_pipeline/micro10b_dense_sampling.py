#!/usr/bin/env python3
"""
Micro-10b — Dense random sampling at peak 1 (10M samples).

Same approach as micro-07 Stage 1 but with 10× more samples.
Expected: ~13,600 candidates within 1% tolerance, nearest ~2.8 deg to truth
(N^{-1/3} scaling from micro-07's 5.95 deg at 1M).
"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

# ── Config ──
SEED       = 42
PEAK1_IDX  = 183
PEAK2_IDX  = 260
TOL_PCT    = 1          # brightness tolerance %
DPHI       = 1e-5       # finite-diff step for gradient
N_TOTAL    = 10_000_000
BATCH      = 100_000
RESULTS_DIR = Path('data/results/inversion_diagnostics')


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


if __name__ == '__main__':
    t0 = time.time()
    ctx = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=SEED,
        true_omega_deg=(0.5, -0.3, 2.0),
        end_time_utc='2020-02-05T11:00:00',
    )
    print(f"Setup: {time.time()-t0:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Reference brightness at peak 1 (truth attitude, lo-fi) ──
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
    print(f"\nPeak 1 (epoch {PEAK1_IDX}): ref_mag={ref_mag:.4f}, tol=±{tol:.4f}")

    # ── Random SO(3) sampling in batches ──
    n_batches = N_TOTAL // BATCH
    print(f"\nSampling {N_TOTAL:,} random SO(3) in {n_batches} batches of {BATCH:,}")
    rng = np.random.default_rng(SEED)
    match_q, match_m = [], []

    for bi in range(n_batches):
        t1 = time.time()
        Rs = Rotation.random(BATCH, random_state=rng)
        Rm = Rs.as_matrix()
        k1 = np.einsum('nij,j->ni', Rm, sv)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 = np.einsum('nij,j->ni', Rm, ov)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
        mags = lofi_batch(k1, k2, PEAK1_IDX, ctx)
        mask = np.abs(mags - ref_mag) < tol
        if mask.any():
            match_q.append(Rs[mask].as_quat())   # xyzw
            match_m.append(mags[mask])
        cum = sum(len(x) for x in match_m)
        print(f"  Batch {bi+1:>3}/{n_batches}: {mask.sum():>5} hits "
              f"(cumul {cum:>6}) [{time.time()-t1:.1f}s]")

    all_q_xyzw = np.vstack(match_q)
    cand_mag = np.concatenate(match_m)
    R_cands = Rotation.from_quat(all_q_xyzw)
    ang_dists = np.rad2deg((R_cands.inv() * R_tr).magnitude())

    # ── Brightness gradients (finite differences) ──
    print("\nComputing brightness gradients...")
    tg = time.time()
    axes = np.eye(3)
    Rm_all = R_cands.as_matrix()
    k1m = np.einsum('nij,j->ni', Rm_all, sv)
    k1m /= np.linalg.norm(k1m, axis=1, keepdims=True)
    k2m = np.einsum('nij,j->ni', Rm_all, ov)
    k2m /= np.linalg.norm(k2m, axis=1, keepdims=True)
    M = len(cand_mag)
    cand_g = np.zeros((M, 3))
    for j in range(3):
        k1p = k1m + DPHI * np.cross(axes[j], k1m)
        k1p /= np.linalg.norm(k1p, axis=1, keepdims=True)
        k2p = k2m + DPHI * np.cross(axes[j], k2m)
        k2p /= np.linalg.norm(k2p, axis=1, keepdims=True)
        cand_g[:, j] = (lofi_batch(k1p, k2p, PEAK1_IDX, ctx) - cand_mag) / DPHI
    g_norm = np.linalg.norm(cand_g, axis=1)
    print(f"  Gradients computed in {time.time()-tg:.1f}s")

    # ── Convert to wxyz and save ──
    cand_q = np.column_stack([all_q_xyzw[:, 3], all_q_xyzw[:, :3]])  # → wxyz
    ckpt_path = RESULTS_DIR / 'micro10b_dense_candidates.npz'
    np.savez(ckpt_path,
             candidate_q_wxyz=cand_q, candidate_mags=cand_mag,
             gradients=cand_g, gradient_norms=g_norm,
             ang_dists_to_truth=ang_dists,
             truth_q0=ctx.true_q0, truth_omega0=ctx.true_omega0,
             peak1_idx=PEAK1_IDX, peak2_idx=PEAK2_IDX,
             obs_mag_peak2=obs_mag_p2, ref_mag_peak1=ref_mag,
             tol=tol, seed=SEED)
    print(f"\nCheckpoint saved: {ckpt_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"MICRO-10b RESULTS — 10M random SO(3) at peak 1")
    print(f"{'='*60}")
    print(f"  Total candidates: {M:,}")
    print(f"  Hit rate: {100*M/N_TOTAL:.3f}%")
    print(f"  Nearest to truth: {ang_dists.min():.2f} deg")
    print(f"  Ang dist percentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    {p:>3}th: {np.percentile(ang_dists, p):.2f} deg")
    print(f"  <1 deg: {int(np.sum(ang_dists < 1))}")
    print(f"  <5 deg: {int(np.sum(ang_dists < 5))}")
    print(f"  <10 deg: {int(np.sum(ang_dists < 10))}")
    print(f"  |g|: min={g_norm.min():.3f} med={np.median(g_norm):.3f} max={g_norm.max():.3f}")

    # Comparison with micro-07 (1M)
    print(f"\n  Comparison with micro-07 (1M samples):")
    print(f"    micro-07: 1,360 candidates, nearest 5.95 deg")
    print(f"    micro-10b: {M:,} candidates, nearest {ang_dists.min():.2f} deg")
    print(f"    Ratio candidates: {M/1360:.1f}× (expected ~10×)")
    print(f"    Ratio nearest: {5.95/ang_dists.min():.1f}× (expected ~2.15× from N^{{-1/3}})")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
