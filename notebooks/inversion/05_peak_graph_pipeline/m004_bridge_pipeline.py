#!/usr/bin/env python3
"""
Micro-04 — Full Bridge Pipeline Test.

End-to-end test of peak-anchored bridging:
1. At two brightness peaks, generate attitude candidates
   (brightness match + dL/dt ≈ 0 derivative filter with oracle ω)
2. For every candidate pair across peaks, solve for the angular velocity
   that best bridges peak 1 → peak 2 via torque-free propagation
3. Score bridges by arrival attitude error
4. Check whether the true pair ranks #1 and measure separation
"""
import sys, time, numpy as np
import multiprocessing as mp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)

from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment, attitude_error_deg, save_results
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves

# ── Config ──
SEED = 42
N_SAMPLES = 10_000
BRIGHT_TOL_PCT = 3        # wider tolerance for lo-fi/hi-fi gap
DPHI = 1e-5
DERIV_MULT = 10           # lenient derivative filter (oracle ω)
MAX_CANDS = 20
N_STARTS = 3
MAXITER = 15
OMEGA_BOUND = 0.1         # rad/s ≈ 5.7 deg/s
RESULTS_DIR = Path('data/results/inversion_diagnostics')


# ── Helper functions (called from main process or fork workers) ──

def _solve_one(args):
    """Worker: find ω bridging q_A → q_B over dt via L-BFGS-B."""
    i, j, q_A, q_B, dt, I_t, seed = args
    rng = np.random.default_rng(seed)
    times = np.array([0.0, dt])
    bounds = [(-OMEGA_BOUND, OMEGA_BOUND)] * 3

    def obj(w):
        qp, _ = propagate_attitude(q_A, w, times, "tumbling", I_t)
        d = np.clip(np.dot(qp[-1], q_B), -1.0, 1.0)
        return 1.0 - d * d

    best_f, best_w = np.inf, np.zeros(3)
    for _ in range(N_STARTS):
        w0 = rng.uniform(-0.05, 0.05, 3)
        try:
            r = minimize(obj, w0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': MAXITER, 'ftol': 1e-14})
            if r.fun < best_f:
                best_f, best_w = r.fun, r.x.copy()
        except Exception:
            pass

    qp, _ = propagate_attitude(q_A, best_w, times, "tumbling", I_t)
    err = attitude_error_deg(qp[-1], q_B)
    return i, j, float(err), best_w.tolist(), float(best_f)


# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
# ══════════════════════════════════════════════════════════════════

    t0 = time.time()
    ctx = setup_experiment(
        n_observations=500, noise_sigma=0.05, random_seed=SEED,
        true_omega_deg=(0.5, -0.3, 2.0),
        end_time_utc='2020-02-05T11:00:00',
    )
    I_tensor = ctx.inertia_tensor
    _, omega_history = propagate_attitude(
        ctx.true_q0, ctx.true_omega0, ctx.observation_times,
        mode="tumbling", inertia_tensor=I_tensor,
    )
    dLdt_thr = DERIV_MULT * ctx.noise_sigma / ctx.dt_sampling
    print(f"Setup: {time.time() - t0:.1f}s")

    # ── Vectorised brightness eval ──
    def eval_bright_batch(k1, k2, eidx):
        N = len(k1)
        art = {c: np.tile(m[eidx:eidx+1], (N, 1, 1))
               for c, m in ctx.art_matrices.items()}
        lit = create_no_shadow_lit_status(ctx.satellite, N)
        mags, *_ = generate_lightcurves(
            facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
            observer_distances=np.full(N, ctx.obs_dist[eidx]),
            satellite=ctx.satellite, epochs=np.arange(N, dtype=float),
            pre_computed_matrices=art, generate_no_shadow=False,
            animate=False, show_progress=False)
        return mags

    # ── Candidate generation ──
    def generate_candidates(peak_idx, rng):
        """Brightness + derivative filter. Truth injected at index 0."""
        peak_mag = ctx.true_lc[peak_idx]
        omega_pk = omega_history[peak_idx]
        sun_v = ctx.sun_pos[peak_idx] - ctx.sat_pos[peak_idx]
        obs_v = ctx.obs_pos[peak_idx] - ctx.sat_pos[peak_idx]

        rots = Rotation.random(N_SAMPLES, random_state=rng)
        Rs = rots.as_matrix()
        k1 = np.einsum('nij,j->ni', Rs, sun_v)
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 = np.einsum('nij,j->ni', Rs, obs_v)
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)

        mags = eval_bright_batch(k1, k2, peak_idx)
        thr = abs(peak_mag) * BRIGHT_TOL_PCT / 100.0
        bidx = np.where(np.abs(mags - peak_mag) < thr)[0]
        n_bright = len(bidx)
        if n_bright == 0:
            return ctx.true_quaternions[peak_idx].reshape(1, 4), 0, 0

        # Derivative filter (oracle ω)
        k1c, k2c, mc = k1[bidx], k2[bidx], mags[bidx]
        g = np.zeros((n_bright, 3))
        for j in range(3):
            ax = np.eye(3)[j]
            k1p = k1c + DPHI * np.cross(ax, k1c)
            k1p /= np.linalg.norm(k1p, axis=1, keepdims=True)
            k2p = k2c + DPHI * np.cross(ax, k2c)
            k2p /= np.linalg.norm(k2p, axis=1, keepdims=True)
            g[:, j] = (eval_bright_batch(k1p, k2p, peak_idx) - mc) / DPHI
        dmask = np.abs(g @ omega_pk) < dLdt_thr
        didx = bidx[dmask]
        n_deriv = len(didx)

        qs = np.empty((0, 4)) if n_deriv == 0 else \
            rots[didx].as_quat()[:, [3, 0, 1, 2]]

        # Inject truth at index 0
        qs = np.vstack([ctx.true_quaternions[peak_idx].reshape(1, 4), qs])
        if len(qs) > MAX_CANDS:
            keep = rng.choice(np.arange(1, len(qs)), MAX_CANDS - 1, replace=False)
            qs = np.vstack([qs[0:1], qs[keep]])
        return qs, n_bright, n_deriv

    # ── Find two peaks: both prominent, 20-80 epochs apart ──
    peaks, props = find_peaks(-ctx.true_lc, prominence=0.05, distance=5)
    order = np.argsort(-props['prominences'])
    top_n = peaks[order[:min(15, len(order))]]

    best_pair, best_score = None, -1
    for a in top_n:
        for b in top_n:
            gap = b - a
            if 20 <= gap <= 80:
                score = min(props['prominences'][np.where(peaks == a)[0][0]],
                            props['prominences'][np.where(peaks == b)[0][0]])
                if score > best_score:
                    best_score = score
                    best_pair = (int(a), int(b))

    if best_pair is None:
        # Fallback: closest pair among top peaks with gap > 15
        for a in np.sort(top_n):
            for b in np.sort(top_n):
                if 15 < b - a:
                    best_pair = (int(a), int(b))
                    break
            if best_pair:
                break

    p1, p2 = best_pair
    dt_bridge = ctx.observation_times[p2] - ctx.observation_times[p1]
    print(f"\nPeak 1: epoch {p1} (t={ctx.observation_times[p1]:.0f}s, "
          f"mag={ctx.true_lc[p1]:.3f})")
    print(f"Peak 2: epoch {p2} (t={ctx.observation_times[p2]:.0f}s, "
          f"mag={ctx.true_lc[p2]:.3f})")
    print(f"Bridge: {dt_bridge:.0f}s ({p2 - p1} epochs)")

    # ── Generate candidates ──
    tc = time.time()
    c1, nb1, nd1 = generate_candidates(p1, np.random.default_rng(SEED + 1))
    c2, nb2, nd2 = generate_candidates(p2, np.random.default_rng(SEED + 2))
    N1, N2 = len(c1), len(c2)
    print(f"\nPeak 1 candidates: {nb1} bright → {nd1} deriv → {N1} final (incl truth)")
    print(f"Peak 2 candidates: {nb2} bright → {nd2} deriv → {N2} final (incl truth)")
    print(f"Total pairs: {N1 * N2}  (candidate gen: {time.time()-tc:.1f}s)")

    # ── Solve all bridges in parallel (fork to inherit globals) ──
    args_list = [(i, j, c1[i], c2[j], dt_bridge, I_tensor,
                  SEED + 100*i + j)
                 for i in range(N1) for j in range(N2)]
    print(f"\nSolving {len(args_list)} bridges with Pool(8)...")
    tb = time.time()
    fork_ctx = mp.get_context('fork')
    with fork_ctx.Pool(8) as pool:
        raw = pool.map(_solve_one, args_list)
    print(f"Bridge solving: {time.time()-tb:.1f}s")

    # ── Analyse ──
    err_mat = np.full((N1, N2), np.inf)
    omega_store = {}
    for i, j, err, w, cost in raw:
        err_mat[i, j] = err
        omega_store[(i, j)] = w

    true_err = err_mat[0, 0]
    true_w = np.array(omega_store[(0, 0)])
    true_w_actual = omega_history[p1]

    flat = err_mat.flatten()
    ranks = np.argsort(flat)
    true_rank = int(np.where(ranks == 0)[0][0]) + 1

    mask = np.ones_like(flat, dtype=bool); mask[0] = False
    false_errs = flat[mask]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  True pair (0,0) error:  {true_err:.4f} deg   "
          f"(rank {true_rank}/{len(flat)})")
    print(f"  Bridge ω (deg/s):  [{np.rad2deg(true_w[0]):.3f}, "
          f"{np.rad2deg(true_w[1]):.3f}, {np.rad2deg(true_w[2]):.3f}]")
    print(f"  Actual ω (deg/s):  [{np.rad2deg(true_w_actual[0]):.3f}, "
          f"{np.rad2deg(true_w_actual[1]):.3f}, {np.rad2deg(true_w_actual[2]):.3f}]")
    w_err = np.rad2deg(np.linalg.norm(true_w - true_w_actual))
    print(f"  ω recovery error:  {w_err:.4f} deg/s")

    print(f"\n  False pair stats (n={len(false_errs)}):")
    print(f"    Best:   {false_errs.min():.4f} deg")
    print(f"    Median: {np.median(false_errs):.4f} deg")
    print(f"    Worst:  {false_errs.max():.4f} deg")

    sep = false_errs.min() - true_err
    print(f"\n  Separation (best_false − true): {sep:+.4f} deg")
    if true_err > 1e-6:
        print(f"  Ratio best_false / true: {false_errs.min()/true_err:.1f}×")

    # Top-10 bridges
    print(f"\n  Top-10 bridges:")
    print(f"  {'#':>3}  {'(i,j)':>6}  {'Err(deg)':>9}  {'True?':>5}")
    print(f"  {'-'*28}")
    for r in range(min(10, len(flat))):
        fi = ranks[r]; ii, jj = fi // N2, fi % N2
        tag = " <<<" if ii == 0 and jj == 0 else ""
        print(f"  {r+1:>3}  ({ii:>2},{jj:>2})  {flat[fi]:>9.4f}{tag}")

    # Error distribution
    print(f"\n  Pairs below threshold:")
    for thr in [0.1, 1.0, 5.0, 10.0, 30.0]:
        n = int(np.sum(flat < thr))
        print(f"    < {thr:5.1f}°: {n:>4} / {len(flat)}")

    total = time.time() - t0
    print(f"\nTotal runtime: {total:.1f}s")

    # ── Save ──
    save_results(RESULTS_DIR / 'm004_bridge_pipeline.json', {
        'peaks': {'p1': p1, 'p2': p2, 'dt_s': float(dt_bridge)},
        'candidates': {
            'peak1': {'n_bright': nb1, 'n_deriv': nd1, 'n_final': N1},
            'peak2': {'n_bright': nb2, 'n_deriv': nd2, 'n_final': N2},
        },
        'true_pair': {
            'error_deg': float(true_err), 'rank': true_rank,
            'total_pairs': len(flat),
            'bridge_omega_deg_s': np.rad2deg(true_w).tolist(),
            'true_omega_deg_s': np.rad2deg(true_w_actual).tolist(),
            'omega_error_deg_s': float(w_err),
        },
        'false_pairs': {
            'min_deg': float(false_errs.min()),
            'median_deg': float(np.median(false_errs)),
        },
        'separation_deg': float(sep),
        'config': {
            'n_samples': N_SAMPLES, 'bright_tol_pct': BRIGHT_TOL_PCT,
            'deriv_mult': DERIV_MULT, 'max_cands': MAX_CANDS,
            'n_starts': N_STARTS, 'maxiter': MAXITER,
        },
    })
    print(f"Saved to {RESULTS_DIR / 'm004_bridge_pipeline.json'}")
