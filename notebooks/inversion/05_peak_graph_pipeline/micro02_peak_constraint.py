#!/usr/bin/env python3
"""
Micro-02 — Peak Derivative Constraint Test.

At brightness peaks (dL/dt ≈ 0), generate candidate attitudes matching the
peak brightness within 1%. Then compute the constraint vector g for each
candidate: dL/dt = g · ω, where g_j = ∂L/∂φ_j (brightness sensitivity to
rotation around body axis j).

At a peak, g · ω ≈ 0 ⟹ ω must be nearly perpendicular to g.

We test: given the TRUE ω, how many brightness-matching candidates also
satisfy the derivative constraint?  This measures the filtering power of
"dL/dt ≈ 0" on top of the brightness match.
"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from lib.experiment_setup import setup_experiment, brightness_single_epoch, save_results
from src.computation.shadow_engine import create_no_shadow_lit_status
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
N_SAMPLES = 10_000
SEED = 42
BRIGHTNESS_TOL_PCT = 1  # % brightness match for candidate selection
DPHI = 1e-5  # rad, finite-difference step for gradient
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(
    n_observations=500, noise_sigma=0.05, random_seed=SEED,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)
_, omega_history = propagate_attitude(
    ctx.true_q0, ctx.true_omega0, ctx.observation_times,
    mode="tumbling", inertia_tensor=ctx.inertia_tensor,
)
print(f"Setup: {time.time() - t0:.1f}s")


# ── Helper: vectorized brightness evaluation at one epoch ──
def eval_brightness_batch(k1_arr, k2_arr, epoch_idx):
    """Evaluate lo-fi brightness for arrays of body-frame vectors."""
    N = len(k1_arr)
    art_tiled = {
        c: np.tile(m[epoch_idx:epoch_idx + 1], (N, 1, 1))
        for c, m in ctx.art_matrices.items()
    }
    lit = create_no_shadow_lit_status(ctx.satellite, N)
    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit,
        k1_vectors_array=k1_arr,
        k2_vectors_array=k2_arr,
        observer_distances=np.full(N, ctx.obs_dist[epoch_idx]),
        satellite=ctx.satellite,
        epochs=np.arange(N, dtype=float),
        pre_computed_matrices=art_tiled,
        generate_no_shadow=False, animate=False, show_progress=False,
    )
    return mags


# ── Find brightness peaks (minima in magnitude = maxima in brightness) ──
peaks, props = find_peaks(-ctx.true_lc, prominence=0.05, distance=5)
order = np.argsort(-props['prominences'])  # most prominent first
print(f"\nFound {len(peaks)} brightness peaks")

# Pick the strongest glint and a moderate peak
peak_choices = []
if len(peaks) >= 1:
    peak_choices.append(('strongest_glint', int(peaks[order[0]])))
if len(peaks) >= 3:
    # Pick a middle-ranked peak for contrast
    mid = len(order) // 2
    peak_choices.append(('moderate_peak', int(peaks[order[mid]])))

# ── Sample random attitudes (shared across peaks) ──
rng = np.random.default_rng(SEED)
rotations = Rotation.random(N_SAMPLES, random_state=rng)
Rs = rotations.as_matrix()  # (N, 3, 3)

dt_samp = ctx.dt_sampling
sigma = ctx.noise_sigma
dLdt_threshold = sigma / dt_samp  # mag/s

all_results = {}

for label, peak_idx in peak_choices:
    peak_mag = ctx.true_lc[peak_idx]
    peak_t = ctx.observation_times[peak_idx]
    omega_at_peak = omega_history[peak_idx]

    print(f"\n{'='*60}")
    print(f"Peak: {label} — epoch {peak_idx} (t = {peak_t:.1f}s)")
    print(f"{'='*60}")
    print(f"  Magnitude:          {peak_mag:.4f} mag")
    print(f"  ω at peak (deg/s):  [{np.rad2deg(omega_at_peak[0]):.3f}, "
          f"{np.rad2deg(omega_at_peak[1]):.3f}, {np.rad2deg(omega_at_peak[2]):.3f}]")
    omega_norm = np.linalg.norm(omega_at_peak)

    # Body-frame vectors for all random attitudes at this epoch
    sun_vec = ctx.sun_pos[peak_idx] - ctx.sat_pos[peak_idx]
    obs_vec = ctx.obs_pos[peak_idx] - ctx.sat_pos[peak_idx]
    k1_all = np.einsum('nij,j->ni', Rs, sun_vec)
    k1_all /= np.linalg.norm(k1_all, axis=1, keepdims=True)
    k2_all = np.einsum('nij,j->ni', Rs, obs_vec)
    k2_all /= np.linalg.norm(k2_all, axis=1, keepdims=True)

    # Evaluate brightness
    mags_all = eval_brightness_batch(k1_all, k2_all, peak_idx)

    # Select candidates
    threshold = abs(peak_mag) * BRIGHTNESS_TOL_PCT / 100.0
    cand_idx = np.where(np.abs(mags_all - peak_mag) < threshold)[0]
    N_cand = len(cand_idx)
    print(f"  Brightness matches: {N_cand} / {N_SAMPLES} "
          f"(tol ±{threshold:.3f} mag)")

    if N_cand == 0:
        print("  No candidates — skipping gradient analysis")
        continue

    # ── Sanity check: truth attitude at this peak ──
    q_true = ctx.true_quaternions[peak_idx]
    truth_mag_lofi = brightness_single_epoch(q_true, peak_idx, ctx, use_shadows=False)
    print(f"  Truth lo-fi mag:    {truth_mag_lofi:.4f} (hi-fi: {peak_mag:.4f})")

    # ── Compute gradient g for each candidate ──
    k1_cand = k1_all[cand_idx]
    k2_cand = k2_all[cand_idx]
    mags_cand = mags_all[cand_idx]
    axes = np.eye(3)

    g_vectors = np.zeros((N_cand, 3))
    for j in range(3):
        cross_k1 = np.cross(axes[j], k1_cand)
        cross_k2 = np.cross(axes[j], k2_cand)
        k1_pert = k1_cand + DPHI * cross_k1
        k1_pert /= np.linalg.norm(k1_pert, axis=1, keepdims=True)
        k2_pert = k2_cand + DPHI * cross_k2
        k2_pert /= np.linalg.norm(k2_pert, axis=1, keepdims=True)
        mags_pert = eval_brightness_batch(k1_pert, k2_pert, peak_idx)
        g_vectors[:, j] = (mags_pert - mags_cand) / DPHI

    # ── Sanity check: truth gradient → should give dL/dt ≈ 0 ──
    R_true = Rotation.from_quat(
        [q_true[1], q_true[2], q_true[3], q_true[0]]).as_matrix()
    k1_true = R_true @ sun_vec
    k1_true /= np.linalg.norm(k1_true)
    k2_true = R_true @ obs_vec
    k2_true /= np.linalg.norm(k2_true)

    g_true = np.zeros(3)
    for j in range(3):
        k1p = k1_true + DPHI * np.cross(axes[j], k1_true)
        k1p /= np.linalg.norm(k1p)
        k2p = k2_true + DPHI * np.cross(axes[j], k2_true)
        k2p /= np.linalg.norm(k2p)
        mag_p = eval_brightness_batch(k1p.reshape(1, 3), k2p.reshape(1, 3), peak_idx)
        g_true[j] = (float(mag_p[0]) - truth_mag_lofi) / DPHI
    dLdt_true = float(g_true @ omega_at_peak)
    print(f"  Truth |g·ω|:        {abs(dLdt_true):.6f} mag/s "
          f"(threshold: {dLdt_threshold:.5f})")

    # ── Analyse candidates ──
    g_dot_omega = g_vectors @ omega_at_peak
    g_norms = np.linalg.norm(g_vectors, axis=1)
    cos_angle = np.where(
        g_norms > 1e-12,
        g_dot_omega / (g_norms * omega_norm),
        0.0,
    )
    dev_from_90 = np.rad2deg(np.arcsin(np.clip(np.abs(cos_angle), 0, 1)))

    print(f"\n  |g| range:          [{g_norms.min():.1f}, {g_norms.max():.1f}] mag/rad")
    print(f"  |g·ω| range:       [{np.abs(g_dot_omega).min():.5f}, "
          f"{np.abs(g_dot_omega).max():.5f}] mag/s")

    # dL/dt filtering
    print(f"\n  Filtering candidates with TRUE ω (oracle):")
    print(f"  {'Threshold':>22}  {'Pass':>5}  {'% of cands':>11}  {'Reduction':>10}")
    print(f"  {'-'*55}")
    for mult, thr in [(1, dLdt_threshold), (2, 2*dLdt_threshold),
                       (5, 5*dLdt_threshold), (10, 10*dLdt_threshold)]:
        n_pass = int(np.sum(np.abs(g_dot_omega) < thr))
        red = f"{N_cand/n_pass:.0f}×" if n_pass > 0 else "all cut"
        print(f"  {mult}× σ/dt = {thr:.5f}  {n_pass:>5d}  "
              f"{100*n_pass/N_cand:>10.1f}%  {red:>10}")

    # Angle from perpendicular
    print(f"\n  Deviation of g-ω angle from 90°:")
    print(f"  {'Dev < X°':>12}  {'Pass':>5}  {'% of cands':>11}  {'Random %':>9}")
    print(f"  {'-'*45}")
    for x in [5, 10, 20, 30, 45]:
        n_pass = int(np.sum(dev_from_90 < x))
        rand_pct = 100 * np.sin(np.deg2rad(x))
        print(f"  {x:>9d}°  {n_pass:>5d}  "
              f"{100*n_pass/N_cand:>10.1f}%  {rand_pct:>8.1f}%")

    # Store results
    n_pass_1x = int(np.sum(np.abs(g_dot_omega) < dLdt_threshold))
    n_pass_5x = int(np.sum(np.abs(g_dot_omega) < 5 * dLdt_threshold))
    all_results[label] = {
        'peak_epoch_idx': int(peak_idx),
        'peak_time_s': float(peak_t),
        'peak_mag': float(peak_mag),
        'omega_at_peak_deg_s': np.rad2deg(omega_at_peak).tolist(),
        'n_brightness_match': N_cand,
        'truth_dLdt': float(abs(dLdt_true)),
        'n_pass_1x_sigma_dt': n_pass_1x,
        'n_pass_5x_sigma_dt': n_pass_5x,
        'g_norm_median': round(float(np.median(g_norms)), 1),
        'dev_from_90_median': round(float(np.median(dev_from_90)), 1),
    }

# ── Grand summary ──
print(f"\n{'='*60}")
print(f"GRAND SUMMARY")
print(f"{'='*60}")
for label, peak_idx in peak_choices:
    if label not in all_results:
        continue
    r = all_results[label]
    print(f"\n{label} (epoch {r['peak_epoch_idx']}, {r['peak_mag']:.2f} mag):")
    print(f"  Brightness-only:   {r['n_brightness_match']:>5d} / {N_SAMPLES}")
    print(f"  + deriv (1×σ/dt):  {r['n_pass_1x_sigma_dt']:>5d} / {N_SAMPLES}")
    print(f"  + deriv (5×σ/dt):  {r['n_pass_5x_sigma_dt']:>5d} / {N_SAMPLES}")
    print(f"  Truth |g·ω|:       {r['truth_dLdt']:.6f} mag/s "
          f"(thresh = {dLdt_threshold:.5f})")

total_time = time.time() - t0
print(f"\nTotal runtime: {total_time:.1f}s")

all_results['config'] = {
    'n_samples': N_SAMPLES,
    'seed': SEED,
    'brightness_tol_pct': BRIGHTNESS_TOL_PCT,
    'dphi_rad': DPHI,
    'dLdt_threshold_mag_s': float(dLdt_threshold),
}
save_results(RESULTS_DIR / 'micro02_peak_constraint.json', all_results)
print(f"Results saved to {RESULTS_DIR / 'micro02_peak_constraint.json'}")
