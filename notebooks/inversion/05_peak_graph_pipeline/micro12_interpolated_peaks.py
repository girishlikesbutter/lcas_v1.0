#!/usr/bin/env python3
"""
Micro-12 Part A — Interpolated peak times and g·ω verification.

Micro-11 failed because at discrete epoch 183, g·ω ≠ 0. The true omega has a
component along the gradient that the perpendicular-plane search misses. At the
exact peak time, dB/dt = g·ω = 0 by definition.

Key subtlety: the pipeline uses lo-fi gradients, but the truth lightcurve is
hi-fi (with shadows). The hi-fi and lo-fi peaks can occur at different times.
So we must find the LO-FI peak time and compute the lo-fi gradient there.

Approach for each peak:
  1. Hi-fi parabola: fit parabola to hi-fi truth LC → t*_hifi
  2. Lo-fi fine scan: evaluate lo-fi brightness on 500-point grid → t*_lofi
  3. Compute lo-fi gradient g·ω at three times:
     (a) discrete epoch, (b) hi-fi interpolated, (c) lo-fi fine-scan peak
  4. Show (c) gives g·ω ≈ 0, confirming the theory
"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment, save_results
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
SEED = 42
PEAK_EPOCHS = [183, 260, 360]
DPHI = 1e-5
N_FINE = 500
RESULTS_DIR = Path('data/results/inversion_diagnostics')


def lofi_brightness_single(k1, k2, obs_dist, ctx):
    """Lo-fi brightness (magnitude) for a single k1/k2 pair."""
    lit = create_no_shadow_lit_status(ctx.satellite, 1)
    art = {c: m[0:1] for c, m in ctx.art_matrices.items()}
    mags, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1.reshape(1, 3),
        k2_vectors_array=k2.reshape(1, 3), observer_distances=np.array([obs_dist]),
        satellite=ctx.satellite, epochs=np.array([0.0]),
        pre_computed_matrices=art, generate_no_shadow=False,
        animate=False, show_progress=False)
    return float(mags[0])


def body_vectors(q_wxyz, sun_pos, obs_pos, sat_pos):
    """Body-frame sun and observer unit vectors."""
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    k1 = R @ (sun_pos - sat_pos); k1 /= np.linalg.norm(k1)
    k2 = R @ (obs_pos - sat_pos); k2 /= np.linalg.norm(k2)
    return k1, k2


def brightness_gradient(q_wxyz, sun_pos, obs_pos, sat_pos, obs_dist, ctx):
    """3-component lo-fi gradient dB/dφ_j (same method as micro-11)."""
    k1, k2 = body_vectors(q_wxyz, sun_pos, obs_pos, sat_pos)
    base = lofi_brightness_single(k1, k2, obs_dist, ctx)
    g = np.zeros(3)
    for j in range(3):
        ax = np.zeros(3); ax[j] = 1.0
        k1p = k1 + DPHI * np.cross(ax, k1); k1p /= np.linalg.norm(k1p)
        k2p = k2 + DPHI * np.cross(ax, k2); k2p /= np.linalg.norm(k2p)
        g[j] = (lofi_brightness_single(k1p, k2p, obs_dist, ctx) - base) / DPHI
    return g


def interp_geometry(t, times, ctx):
    """Linearly interpolate geometry to arbitrary time."""
    idx = max(0, min(np.searchsorted(times, t) - 1, len(times) - 2))
    f = (t - times[idx]) / (times[idx + 1] - times[idx])
    return (ctx.sun_pos[idx] + f * (ctx.sun_pos[idx + 1] - ctx.sun_pos[idx]),
            ctx.obs_pos[idx] + f * (ctx.obs_pos[idx + 1] - ctx.obs_pos[idx]),
            ctx.sat_pos[idx] + f * (ctx.sat_pos[idx + 1] - ctx.sat_pos[idx]),
            ctx.obs_dist[idx] + f * (ctx.obs_dist[idx + 1] - ctx.obs_dist[idx]))


def fine_lofi_scan(ep, ctx, true_q0, true_omega0):
    """Evaluate lo-fi brightness on fine time grid around epoch ep (±1 sample)."""
    t_lo = ctx.observation_times[ep - 1]
    t_hi = ctx.observation_times[ep + 1]
    t_fine = np.linspace(t_lo, t_hi, N_FINE)
    q_fine, omega_fine = propagate_attitude(
        q0=true_q0, omega0=true_omega0, times=t_fine,
        mode="tumbling", inertia_tensor=ctx.inertia_tensor)
    R_fine = Rotation.from_quat(q_fine[:, [1, 2, 3, 0]]).as_matrix()
    sv = ctx.sun_pos[ep] - ctx.sat_pos[ep]
    ov = ctx.obs_pos[ep] - ctx.sat_pos[ep]
    k1 = np.einsum('nij,j->ni', R_fine, sv)
    k2 = np.einsum('nij,j->ni', R_fine, ov)
    k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    lit = create_no_shadow_lit_status(ctx.satellite, N_FINE)
    art = {c: np.tile(m[0:1], (N_FINE, 1, 1)) for c, m in ctx.art_matrices.items()}
    mag_fine, *_ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=np.full(N_FINE, ctx.obs_dist[ep]),
        satellite=ctx.satellite, epochs=np.arange(N_FINE, dtype=float),
        pre_computed_matrices=art, generate_no_shadow=False,
        animate=False, show_progress=False)
    return t_fine, mag_fine, q_fine, omega_fine


def parabola_vertex(t1, y1, t2, y2, t3, y3):
    """Fit parabola through 3 equally-spaced points, return (t*, y*, coeffs)."""
    dt = t2 - t1
    c, a = y2, (y1 + y3 - 2 * y2) / (2 * dt**2)
    b = (y3 - y1) / (2 * dt)
    tau = -b / (2 * a) if abs(a) > 1e-30 else 0.0
    return t2 + tau, a * tau**2 + b * tau + c, (a, b, c, t2)


if __name__ == '__main__':
    t0 = time.time()
    ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                           true_omega_deg=(0.5, -0.3, 2.0),
                           end_time_utc='2020-02-05T11:00:00')
    true_quats, true_omegas = propagate_attitude(
        q0=ctx.true_q0, omega0=ctx.true_omega0,
        times=ctx.observation_times, mode="tumbling",
        inertia_tensor=ctx.inertia_tensor)
    obs_times = ctx.observation_times
    dt = ctx.dt_sampling

    # Lo-fi truth lightcurve (all 500 epochs, batch)
    R_all = Rotation.from_quat(true_quats[:, [1, 2, 3, 0]]).as_matrix()
    k1_a = np.einsum('nij,nj->ni', R_all, ctx.sun_pos - ctx.sat_pos)
    k2_a = np.einsum('nij,nj->ni', R_all, ctx.obs_pos - ctx.sat_pos)
    k1_a /= np.linalg.norm(k1_a, axis=1, keepdims=True)
    k2_a /= np.linalg.norm(k2_a, axis=1, keepdims=True)
    lit_a = create_no_shadow_lit_status(ctx.satellite, 500)
    lofi_lc, *_ = generate_lightcurves(
        facet_lit_status_dict=lit_a, k1_vectors_array=k1_a, k2_vectors_array=k2_a,
        observer_distances=ctx.obs_dist, satellite=ctx.satellite, epochs=ctx.epochs,
        pre_computed_matrices=ctx.art_matrices, show_progress=False)
    print(f"Setup + lo-fi LC: {time.time()-t0:.1f}s, dt={dt:.2f}s")

    # Peak verification
    print(f"\nPeak verification (lower mag = brighter):")
    print(f"  {'Ep':>4} {'Hi-fi':>8} {'Lo-fi':>8} {'HF type':>8} {'LF type':>8}")
    for ep in PEAK_EPOCHS:
        hf, lf = ctx.true_lc, lofi_lc
        ht = "MIN" if hf[ep] < hf[ep-1] and hf[ep] < hf[ep+1] else "other"
        lt = "MIN" if lf[ep] < lf[ep-1] and lf[ep] < lf[ep+1] else "other"
        print(f"  {ep:>4} {hf[ep]:>8.3f} {lf[ep]:>8.3f} {ht:>8} {lt:>8}")

    # ── Process each peak ──
    results = {}
    for pi, ep in enumerate(PEAK_EPOCHS):
        print(f"\n{'='*70}")
        print(f"PEAK {pi+1} — Epoch {ep}")
        print(f"{'='*70}")

        t2 = obs_times[ep]

        # A) Hi-fi parabola interpolation
        t_hf, _, (a_hf, b_hf, c_hf, tc_hf) = parabola_vertex(
            obs_times[ep-1], ctx.true_lc[ep-1], t2, ctx.true_lc[ep],
            obs_times[ep+1], ctx.true_lc[ep+1])

        # B) Lo-fi fine scan (500 points)
        t1_scan = time.time()
        t_fine, mag_fine, q_fine, omega_fine = fine_lofi_scan(
            ep, ctx, ctx.true_q0, ctx.true_omega0)
        i_pk = np.argmin(mag_fine)  # brightness peak = magnitude minimum
        t_lf = t_fine[i_pk]
        print(f"  Hi-fi parabola peak: t={t_hf:.4f}s (shift={t_hf-t2:+.3f}s)")
        print(f"  Lo-fi fine-scan peak: t={t_lf:.4f}s (shift={t_lf-t2:+.3f}s)"
              f"  [{time.time()-t1_scan:.1f}s]")
        print(f"  Hi-fi vs Lo-fi peak offset: {t_lf-t_hf:+.3f}s")

        # C) g·ω at discrete epoch
        g_disc = brightness_gradient(true_quats[ep], ctx.sun_pos[ep], ctx.obs_pos[ep],
                                     ctx.sat_pos[ep], ctx.obs_dist[ep], ctx)
        gdot_disc = np.dot(g_disc, true_omegas[ep])

        # D) g·ω at hi-fi interpolated time
        q_hf_arr, om_hf_arr = propagate_attitude(
            q0=ctx.true_q0, omega0=ctx.true_omega0, times=np.array([0.0, t_hf]),
            mode="tumbling", inertia_tensor=ctx.inertia_tensor)
        sun_hf, obs_hf, sat_hf, dist_hf = interp_geometry(t_hf, obs_times, ctx)
        g_hf = brightness_gradient(q_hf_arr[-1], sun_hf, obs_hf, sat_hf, dist_hf, ctx)
        gdot_hf = np.dot(g_hf, om_hf_arr[-1])

        # E) g·ω at lo-fi fine-scan peak
        sun_lf, obs_lf, sat_lf, dist_lf = interp_geometry(t_lf, obs_times, ctx)
        g_lf = brightness_gradient(q_fine[i_pk], sun_lf, obs_lf, sat_lf, dist_lf, ctx)
        gdot_lf = np.dot(g_lf, omega_fine[i_pk])

        r_hf = abs(gdot_disc) / max(abs(gdot_hf), 1e-15)
        r_lf = abs(gdot_disc) / max(abs(gdot_lf), 1e-15)

        print(f"\n  g·ω comparison:")
        print(f"    Discrete (epoch {ep}):     g·ω = {gdot_disc:+.6f}")
        print(f"    Hi-fi parabola interp:    g·ω = {gdot_hf:+.6f}  ({r_hf:.1f}x reduction)")
        print(f"    Lo-fi fine-scan peak:     g·ω = {gdot_lf:+.6f}  ({r_lf:.1f}x reduction)")

        results[f'peak_{pi+1}'] = {
            'epoch': ep, 't_discrete': float(t2),
            't_hifi_interp': float(t_hf), 'dt_hifi': float(t_hf - t2),
            't_lofi_peak': float(t_lf), 'dt_lofi': float(t_lf - t2),
            'gdot_discrete': float(gdot_disc),
            'gdot_hifi_interp': float(gdot_hf),
            'gdot_lofi_peak': float(gdot_lf),
            'reduction_hifi': float(r_hf), 'reduction_lofi': float(r_lf),
        }

        # ── Plot ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Left: context (±10 epochs, both LCs)
        lo, hi_i = max(0, ep - 10), min(500, ep + 11)
        rng = np.arange(lo, hi_i)
        ax1.plot(obs_times[rng], ctx.true_lc[rng], 'o-', color='steelblue',
                 ms=4, lw=1, label='Hi-fi LC')
        ax1.plot(obs_times[rng], lofi_lc[rng], 's-', color='forestgreen',
                 ms=4, lw=1, label='Lo-fi LC')
        ax1.axvline(t2, color='gray', ls=':', alpha=0.5)
        ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Magnitude')
        ax1.set_title(f'Peak {pi+1}: Hi-fi vs Lo-fi (±10 epochs)')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        # Right: fine scan zoom
        ax2.plot(t_fine, mag_fine, '-', color='forestgreen', lw=1.5,
                 label=f'Lo-fi fine scan ({N_FINE} pts)')
        # Hi-fi parabola overlay
        tp = np.linspace(obs_times[ep-1], obs_times[ep+1], 200)
        yp = a_hf*(tp-tc_hf)**2 + b_hf*(tp-tc_hf) + c_hf
        ax2.plot(tp, yp, '--', color='steelblue', lw=1, alpha=0.7, label='Hi-fi parabola')
        # Discrete points
        ax2.plot(obs_times[ep-1:ep+2], ctx.true_lc[ep-1:ep+2], 'o', color='steelblue',
                 ms=8, zorder=4, label='Hi-fi discrete')
        ax2.plot(obs_times[ep-1:ep+2], lofi_lc[ep-1:ep+2], 's', color='forestgreen',
                 ms=8, zorder=4, label='Lo-fi discrete')
        # Peak markers
        ax2.axvline(t2, color='gray', ls=':', alpha=0.5, label=f'Epoch {ep}')
        ax2.axvline(t_hf, color='steelblue', ls='--', alpha=0.6, label='Hi-fi peak')
        ax2.axvline(t_lf, color='orangered', ls='--', alpha=0.6, label='Lo-fi peak')
        ax2.plot(t_lf, mag_fine[i_pk], '*', color='orangered', ms=14, zorder=5)

        yl, yh = ax2.get_ylim(); yr = yh - yl
        ax2.text(obs_times[ep-1] + 0.15*dt, yh - 0.08*yr,
                 f'Discrete:    g·ω = {gdot_disc:+.4f}', fontsize=9, color='gray',
                 fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax2.text(obs_times[ep-1] + 0.15*dt, yh - 0.17*yr,
                 f'Hi-fi interp: g·ω = {gdot_hf:+.4f}', fontsize=9, color='steelblue',
                 fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax2.text(obs_times[ep-1] + 0.15*dt, yh - 0.26*yr,
                 f'Lo-fi peak:  g·ω = {gdot_lf:+.4f}  ({r_lf:.0f}x)', fontsize=9,
                 color='orangered', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Magnitude')
        ax2.set_title(f'Peak {pi+1}: Fine Scan (lo-fi peak at {t_lf-t2:+.2f}s)')
        ax2.legend(loc='lower right', fontsize=7); ax2.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = RESULTS_DIR / f'micro12_peak_{pi+1}_interpolation.png'
        fig.savefig(plot_path, dpi=150); plt.close(fig)
        print(f"  Saved: {plot_path}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY — g·ω at discrete vs interpolated peak times")
    print(f"{'='*70}")
    print(f"{'Pk':>3} {'Ep':>4} {'Δt_hf':>7} {'Δt_lf':>7} {'|g·ω| disc':>11} "
          f"{'|g·ω| hf':>10} {'|g·ω| lf':>10} {'lf red':>7}")
    print(f"{'-'*70}")
    for pi, ep in enumerate(PEAK_EPOCHS):
        r = results[f'peak_{pi+1}']
        print(f"  {pi+1:>1} {ep:>4} {r['dt_hifi']:>+7.3f} {r['dt_lofi']:>+7.3f} "
              f"{abs(r['gdot_discrete']):>11.6f} {abs(r['gdot_hifi_interp']):>10.6f} "
              f"{abs(r['gdot_lofi_peak']):>10.6f} {r['reduction_lofi']:>7.1f}x")

    # ── Save ──
    save_results(RESULTS_DIR / 'micro12_interpolated_peaks.json', {
        'config': {'peaks': PEAK_EPOCHS, 'seed': SEED, 'dphi': DPHI,
                   'n_fine': N_FINE, 'n_observations': 500, 'dt_sampling': float(dt)},
        'results': results,
        'runtime_s': round(time.time() - t0, 1)
    })
    print(f"\nSaved: {RESULTS_DIR}/micro12_interpolated_peaks.json")
    print(f"Total runtime: {time.time()-t0:.1f}s")
