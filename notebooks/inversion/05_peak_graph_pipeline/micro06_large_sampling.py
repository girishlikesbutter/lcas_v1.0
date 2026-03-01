#!/usr/bin/env python3
"""
Micro-06 — Large-Scale Random Sampling at Peaks (1M attitudes).

Can brute-force 1M random SO(3) sampling find the true attitude at
brightness peaks WITHOUT injecting truth?

Key outputs per peak:
- Brightness match count (1% tolerance, lo-fi)
- Nearest candidate angular distance to truth
- Derivative filter: unknown-omega vs oracle (known omega)
"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from lib.experiment_setup import setup_experiment, save_results
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
N_TOTAL   = 1_000_000
BATCH     = 100_000
SEED      = 42
TOL_PCT   = 1
DPHI      = 1e-5
N_PEAKS   = 3
OMEGA_MAX = np.deg2rad(5.0)   # rad/s bound for exists-omega test
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


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
# ══════════════════════════════════════════════════════════════

    t0 = time.time()
    ctx = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=SEED,
        true_omega_deg=(0.5, -0.3, 2.0),
        end_time_utc='2020-02-05T11:00:00',
    )
    _, omega_hist = propagate_attitude(
        ctx.true_q0, ctx.true_omega0, ctx.observation_times,
        mode="tumbling", inertia_tensor=ctx.inertia_tensor)
    dLdt_noise = ctx.noise_sigma / ctx.dt_sampling
    print(f"Setup: {time.time()-t0:.1f}s")

    peaks, props = find_peaks(-ctx.true_lc, prominence=0.05, distance=5)
    order = np.argsort(-props['prominences'])
    selected = sorted(peaks[order[:N_PEAKS]])
    print(f"\nSelected {N_PEAKS} peaks (of {len(peaks)}):")
    for pk in selected:
        print(f"  epoch {pk}, t={ctx.observation_times[pk]:.0f}s, "
              f"mag={ctx.true_lc[pk]:.4f}")

    rng = np.random.default_rng(SEED)
    all_results = {}
    axes = np.eye(3)

    for pi, peak_idx in enumerate(selected):
        peak_idx = int(peak_idx)
        truth_q = ctx.true_quaternions[peak_idx]
        omega_pk = omega_hist[peak_idx]
        sv = ctx.sun_pos[peak_idx] - ctx.sat_pos[peak_idx]
        ov = ctx.obs_pos[peak_idx] - ctx.sat_pos[peak_idx]

        # Reference lo-fi magnitude at truth attitude
        R_tr = Rotation.from_quat(
            [truth_q[1], truth_q[2], truth_q[3], truth_q[0]])
        Rm_tr = R_tr.as_matrix()
        k1t = Rm_tr @ sv; k1t /= np.linalg.norm(k1t)
        k2t = Rm_tr @ ov; k2t /= np.linalg.norm(k2t)
        ref_mag = float(lofi_batch(
            k1t.reshape(1, 3), k2t.reshape(1, 3), peak_idx, ctx)[0])
        tol = abs(ref_mag) * TOL_PCT / 100.0

        print(f"\n{'='*65}")
        print(f"Peak {pi+1}: epoch {peak_idx}, ref={ref_mag:.4f}, tol=+-{tol:.4f}")
        print(f"{'='*65}")

        # ── Sample 1M in batches ──
        mq, mm = [], []
        for bi in range(N_TOTAL // BATCH):
            t1 = time.time()
            Rs = Rotation.random(BATCH, random_state=rng)
            Rm = Rs.as_matrix()
            k1 = np.einsum('nij,j->ni', Rm, sv)
            k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
            k2 = np.einsum('nij,j->ni', Rm, ov)
            k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
            mags = lofi_batch(k1, k2, peak_idx, ctx)
            mask = np.abs(mags - ref_mag) < tol
            if mask.any():
                mq.append(Rs[mask].as_quat())   # xyzw convention
                mm.append(mags[mask])
            cum = sum(len(x) for x in mm)
            print(f"  Batch {bi+1}/{N_TOTAL//BATCH}: {mask.sum():>5} "
                  f"(cumul {cum:>6}) [{time.time()-t1:.1f}s]")

        all_q = np.vstack(mq) if mq else np.empty((0, 4))
        all_m = np.concatenate(mm) if mm else np.array([])
        n_match = len(all_m)
        print(f"\n  Brightness matches: {n_match:,} / {N_TOTAL:,} "
              f"({100*n_match/N_TOTAL:.3f}%)")

        # ── Angular distance to truth ──
        nearest_deg, nearest_idx = float('inf'), -1
        ang_dists = np.array([])
        if n_match > 0:
            R_cands = Rotation.from_quat(all_q)
            ang_dists = np.rad2deg((R_cands.inv() * R_tr).magnitude())
            nearest_idx = int(np.argmin(ang_dists))
            nearest_deg = float(ang_dists[nearest_idx])
            print(f"  Nearest to truth: {nearest_deg:.2f} deg")
            for d in [1, 5, 10, 30, 60, 90]:
                n_w = int(np.sum(ang_dists < d))
                if n_w > 0 or d <= 10:
                    print(f"    <{d:>3} deg: {n_w}")

        # ── Derivative filter ──
        gnorm_filt, oracle_filt = {}, {}
        g_stats = {}
        if n_match > 0:
            Rm = R_cands.as_matrix()
            k1m = np.einsum('nij,j->ni', Rm, sv)
            k1m /= np.linalg.norm(k1m, axis=1, keepdims=True)
            k2m = np.einsum('nij,j->ni', Rm, ov)
            k2m /= np.linalg.norm(k2m, axis=1, keepdims=True)
            g = np.zeros((n_match, 3))
            for j in range(3):
                k1p = k1m + DPHI * np.cross(axes[j], k1m)
                k1p /= np.linalg.norm(k1p, axis=1, keepdims=True)
                k2p = k2m + DPHI * np.cross(axes[j], k2m)
                k2p /= np.linalg.norm(k2p, axis=1, keepdims=True)
                g[:, j] = (lofi_batch(k1p, k2p, peak_idx, ctx) - all_m) / DPHI
            g_norm = np.linalg.norm(g, axis=1)
            g_stats = {'min': round(float(g_norm.min()), 3),
                       'med': round(float(np.median(g_norm)), 3),
                       'max': round(float(g_norm.max()), 3)}

            # Unknown-omega filter: exists omega with |omega|<=5 deg/s, |g.omega|<thr?
            # min_{|omega|<=w_max} |g.omega| = 0 (pick omega perp g) -> all pass
            print(f"\n  Derivative filter (exists omega, |omega|<="
                  f"{np.rad2deg(OMEGA_MAX):.0f} deg/s):")
            print(f"  |g|: min={g_stats['min']:.2f} med={g_stats['med']:.2f} "
                  f"max={g_stats['max']:.2f}")
            for mult, label in [(1, 'tight'), (5, 'medium'), (20, 'loose')]:
                thr = mult * dLdt_noise
                # exists omega perp g -> |g.omega|=0 < any thr -> all survive
                print(f"    |g.w|<{thr:.5f} ({label:>6}): "
                      f"{n_match:>6}/{n_match} survive, nearest OK")
                gnorm_filt[label] = {'threshold': round(float(thr), 6),
                    'n_survive': n_match, 'nearest_ok': True}
            print(f"  Reason: omega perp g always exists -> g.omega=0 trivially")

            # Oracle (known omega) for comparison
            gdotw = g @ omega_pk
            print(f"\n  Oracle (known omega) for reference:")
            for mult, label in [(1, '1x'), (5, '5x'), (20, '20x')]:
                thr = mult * dLdt_noise
                ns = int(np.sum(np.abs(gdotw) < thr))
                nr = bool(np.abs(gdotw[nearest_idx]) < thr)
                print(f"    |g.w|<{mult}x sigma/dt: "
                      f"{ns:>6}/{n_match}, nearest {'OK' if nr else 'OUT'}")
                oracle_filt[label] = {'n_survive': ns, 'nearest_ok': nr}

        all_results[f'peak_{pi}'] = {
            'epoch_idx': peak_idx,
            'time_s': float(ctx.observation_times[peak_idx]),
            'ref_mag': ref_mag, 'n_matches': n_match,
            'match_rate_pct': round(100 * n_match / N_TOTAL, 4),
            'nearest_deg': round(nearest_deg, 3),
            'ang_percentiles': {f'p{p}': round(float(np.percentile(ang_dists, p)), 2)
                for p in [10, 50, 90]} if n_match > 0 else {},
            'g_stats': g_stats,
            'unknown_omega_filter': gnorm_filt,
            'oracle_filter': oracle_filt,
        }

    # ── Summary ──
    total = time.time() - t0
    print(f"\n{'='*65}")
    print(f"SUMMARY: 1M Random SO(3) at {N_PEAKS} Peaks (lo-fi, {TOL_PCT}% tol)")
    print(f"{'='*65}")
    print(f"{'Peak':>5} {'Ep':>4} {'Matches':>9} {'Rate%':>8} {'Nearest':>9}")
    for i in range(N_PEAKS):
        r = all_results[f'peak_{i}']
        print(f"{i+1:>5} {r['epoch_idx']:>4} {r['n_matches']:>9,} "
              f"{r['match_rate_pct']:>8.3f} {r['nearest_deg']:>8.2f}°")
    print(f"\nKey finding: derivative filter with unknown omega is trivially")
    print(f"satisfied (omega perp g always exists). Known omega or multi-peak")
    print(f"bridging is needed for actual filtering power.")
    print(f"Total: {total:.1f}s")

    all_results['config'] = {
        'n_total': N_TOTAL, 'batch': BATCH, 'seed': SEED,
        'tol_pct': TOL_PCT, 'dphi': DPHI, 'n_peaks': N_PEAKS,
        'omega_max_deg_s': float(np.rad2deg(OMEGA_MAX)),
    }
    save_results(RESULTS_DIR / 'micro06_large_sampling.json', all_results)
    print(f"Saved to {RESULTS_DIR / 'micro06_large_sampling.json'}")
