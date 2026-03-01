#!/usr/bin/env python3
"""
Micro-07a — Omega Drift Measurement.

Sanity check: how much does angular velocity change between brightness
peaks under torque-free dynamics?  This determines whether constant-omega
is a valid approximation for bridging.

One forward propagation from truth, logging omega at every epoch.
No random sampling, no optimisation.
"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment, save_results
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
SEED = 42
PEAK_EPOCHS = [183, 260, 360]  # from prior experiments
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(
    n_observations=500, noise_sigma=0.05, random_seed=SEED,
    true_omega_deg=(0.5, -0.3, 2.0),
    end_time_utc='2020-02-05T11:00:00',
)
print(f"Setup: {time.time() - t0:.1f}s")

# ── Propagate truth to get omega history at every epoch ──
quats, omega_history = propagate_attitude(
    q0=ctx.true_q0, omega0=ctx.true_omega0,
    times=ctx.observation_times,
    mode="tumbling", inertia_tensor=ctx.inertia_tensor,
)

# ── Verify peaks ──
peaks_found, props = find_peaks(-ctx.true_lc, prominence=0.05, distance=5)
order = np.argsort(-props['prominences'])
top_peaks = np.sort(peaks_found[order[:6]])
print(f"Top brightness peaks found: {list(top_peaks)}")
print(f"Using peaks from prior experiments: {PEAK_EPOCHS}")

# ── Part A: Inertia tensor properties ──
print(f"\n{'='*60}")
print("Part A: Inertia Tensor Properties")
print(f"{'='*60}")

I = ctx.inertia_tensor
eigvals, eigvecs = np.linalg.eigh(I)
I1, I2, I3 = eigvals  # ascending order

print(f"\n  Inertia tensor (kg·m²):")
for row in range(3):
    print(f"    [{I[row,0]:12.2f}  {I[row,1]:12.2f}  {I[row,2]:12.2f}]")

print(f"\n  Principal moments (ascending):")
print(f"    I1 = {I1:.2f} kg·m²")
print(f"    I2 = {I2:.2f} kg·m²")
print(f"    I3 = {I3:.2f} kg·m²")

print(f"\n  Ratios:")
print(f"    I2/I1 = {I2/I1:.4f}")
print(f"    I3/I1 = {I3/I1:.4f}")
print(f"    I3/I2 = {I3/I2:.4f}")
print(f"    (I3 - I2)/(I3 - I1) = {(I3 - I2)/(I3 - I1):.4f}  "
      f"(0 = oblate, 1 = prolate, between = triaxial)")

# Asymmetry parameter
asym = (I3 - I2) / (I3 - I1)
if abs(asym) < 0.05 or abs(asym - 1) < 0.05:
    print(f"    → Nearly {'prolate' if abs(asym - 1) < 0.05 else 'oblate'} (near-axisymmetric)")
else:
    print(f"    → Triaxial (significantly asymmetric)")

# ── Part B: Omega at each peak ──
print(f"\n{'='*60}")
print("Part B: Angular Velocity at Peaks")
print(f"{'='*60}")

omega0_deg = np.rad2deg(ctx.true_omega0)
print(f"\n  Initial omega (deg/s): [{omega0_deg[0]:.4f}, {omega0_deg[1]:.4f}, {omega0_deg[2]:.4f}]")
print(f"  |omega0| = {np.linalg.norm(omega0_deg):.4f} deg/s")

print(f"\n  {'Epoch':>6}  {'Time (s)':>9}  {'omega_x':>10}  {'omega_y':>10}  "
      f"{'omega_z':>10}  {'|omega|':>10}  {'|Δω| (deg/s)':>13}  {'Δω/|ω₀| (%)':>12}")
print(f"  {'-'*88}")

peak_results = []
for ep in PEAK_EPOCHS:
    t_ep = ctx.observation_times[ep]
    w_ep = omega_history[ep]
    w_ep_deg = np.rad2deg(w_ep)
    w0_deg = np.rad2deg(ctx.true_omega0)
    delta_w = w_ep_deg - w0_deg
    delta_w_mag = np.linalg.norm(delta_w)
    w_mag = np.linalg.norm(w_ep_deg)
    frac_pct = delta_w_mag / np.linalg.norm(w0_deg) * 100

    peak_results.append({
        'epoch': int(ep),
        'time_s': float(t_ep),
        'omega_deg_s': [round(float(x), 6) for x in w_ep_deg],
        'omega_mag_deg_s': round(float(w_mag), 6),
        'delta_omega_deg_s': round(float(delta_w_mag), 6),
        'delta_omega_pct': round(float(frac_pct), 4),
    })

    print(f"  {ep:>6}  {t_ep:>9.1f}  {w_ep_deg[0]:>10.4f}  {w_ep_deg[1]:>10.4f}  "
          f"{w_ep_deg[2]:>10.4f}  {w_mag:>10.4f}  {delta_w_mag:>13.6f}  {frac_pct:>12.4f}")

# ── Part C: Omega drift over entire trajectory ──
print(f"\n{'='*60}")
print("Part C: Omega Drift Over Entire Trajectory")
print(f"{'='*60}")

omega_deg = np.rad2deg(omega_history)  # (N, 3)
omega0_tile = np.tile(np.rad2deg(ctx.true_omega0), (len(omega_history), 1))
delta_omega = omega_deg - omega0_tile
delta_mag = np.linalg.norm(delta_omega, axis=1)

max_drift_idx = np.argmax(delta_mag)
max_drift = delta_mag[max_drift_idx]
max_drift_pct = max_drift / np.linalg.norm(np.rad2deg(ctx.true_omega0)) * 100

print(f"\n  Max |Δω| over trajectory: {max_drift:.6f} deg/s "
      f"(at epoch {max_drift_idx}, t = {ctx.observation_times[max_drift_idx]:.1f}s)")
print(f"  Max Δω/|ω₀|:             {max_drift_pct:.4f}%")
print(f"  Mean |Δω|:               {delta_mag.mean():.6f} deg/s")
print(f"  Std |Δω|:                {delta_mag.std():.6f} deg/s")

# Per-component drift
print(f"\n  Per-component max drift:")
for ax_idx, ax_name in enumerate(['ωx', 'ωy', 'ωz']):
    comp_drift = np.abs(delta_omega[:, ax_idx])
    max_comp = comp_drift.max()
    print(f"    {ax_name}: max |Δ| = {max_comp:.6f} deg/s")

# ── Part D: Angular momentum conservation check ──
print(f"\n{'='*60}")
print("Part D: Conservation Checks")
print(f"{'='*60}")

# L should be constant in inertial frame
L_inertial = np.zeros((len(omega_history), 3))
T_kinetic = np.zeros(len(omega_history))
for i in range(len(omega_history)):
    R_i = Rotation.from_quat(
        [quats[i, 1], quats[i, 2], quats[i, 3], quats[i, 0]]).as_matrix()
    L_body = I @ omega_history[i]
    L_inertial[i] = R_i @ L_body
    T_kinetic[i] = 0.5 * omega_history[i] @ I @ omega_history[i]

L0 = L_inertial[0]
L_err = np.linalg.norm(L_inertial - L0, axis=1) / np.linalg.norm(L0)
T0 = T_kinetic[0]
T_err = np.abs(T_kinetic - T0) / T0

print(f"\n  |L| = {np.linalg.norm(L0):.4f} kg·m²/s")
print(f"  T   = {T0:.6f} J")
print(f"  Max |ΔL|/|L|: {L_err.max():.2e} (integrator precision)")
print(f"  Max |ΔT|/T:   {T_err.max():.2e} (integrator precision)")

# ── Part E: Theoretical precession analysis ──
print(f"\n{'='*60}")
print("Part E: Precession Analysis")
print(f"{'='*60}")

# Project omega onto principal axes
omega0_body = ctx.true_omega0  # rad/s in body frame
# Transform to principal frame
omega0_princ = eigvecs.T @ omega0_body
omega0_princ_deg = np.rad2deg(omega0_princ)

print(f"\n  omega0 in principal frame (deg/s):")
print(f"    ω₁ = {omega0_princ_deg[0]:.4f} (around I1={I1:.1f})")
print(f"    ω₂ = {omega0_princ_deg[1]:.4f} (around I2={I2:.1f})")
print(f"    ω₃ = {omega0_princ_deg[2]:.4f} (around I3={I3:.1f})")

# Identify dominant spin axis
omega_princ_abs = np.abs(omega0_princ_deg)
dom_idx = np.argmax(omega_princ_abs)
dom_labels = ['I1 (short)', 'I2 (intermediate)', 'I3 (long)']
omega_spin = omega0_princ[dom_idx]  # rad/s
omega_perp = np.sqrt(np.sum(omega0_princ**2) - omega_spin**2)
nutation_angle = np.rad2deg(np.arctan2(omega_perp, abs(omega_spin)))
print(f"\n  Dominant spin axis: {dom_labels[dom_idx]}")
print(f"    |ω_spin| = {np.rad2deg(abs(omega_spin)):.4f} deg/s")
print(f"    |ω_perp| = {np.rad2deg(omega_perp):.4f} deg/s")
print(f"    Nutation angle = {nutation_angle:.2f} deg")

# Linearized body-frame precession period (valid for near-axis rotation)
# For rotation mostly about axis j, transverse components oscillate at:
#   Ω² = ω_j² × (I_j - I_k)(I_j - I_m) / (I_k × I_m)
# where k, m are the other two axes
if dom_idx == 0:
    Ik, Im = I2, I3
elif dom_idx == 1:
    Ik, Im = I1, I3
else:
    Ik, Im = I1, I2

Omega_sq = omega_spin**2 * (eigvals[dom_idx] - Ik) * (eigvals[dom_idx] - Im) / (Ik * Im)
if Omega_sq > 0:
    Omega_prec = np.sqrt(Omega_sq)
    T_prec = 2 * np.pi / Omega_prec
    print(f"\n  Linearized body-frame precession:")
    print(f"    Ω_prec = {np.rad2deg(Omega_prec):.4f} deg/s")
    print(f"    T_prec = {T_prec:.1f} s")
    print(f"    Observation window = {ctx.observation_times[-1]:.1f} s")
    print(f"    Window / T_prec = {ctx.observation_times[-1] / T_prec:.2f} periods")
else:
    T_prec = None
    print(f"\n  Ω² < 0 — rotation about intermediate axis (unstable)")

# Verify by measuring actual precession from omega history
# Transform entire omega history to principal frame
omega_princ_history = (eigvecs.T @ omega_history.T).T  # (N, 3)
# The transverse components should oscillate
trans_axes = [i for i in range(3) if i != dom_idx]
omega_trans = omega_princ_history[:, trans_axes]
# Count zero crossings of first transverse component to estimate period
crossings = np.where(np.diff(np.sign(omega_trans[:, 0])))[0]
if len(crossings) >= 2:
    # Each half-period between crossings
    half_periods = np.diff(ctx.observation_times[crossings])
    T_measured = 2 * np.median(half_periods)
    print(f"\n  Measured precession (zero-crossings of ω_trans):")
    print(f"    T_measured = {T_measured:.1f} s")
    print(f"    {len(crossings)} zero-crossings in {ctx.observation_times[-1]:.0f}s")
    if T_prec is not None:
        print(f"    Agreement with linearized: {abs(T_measured - T_prec)/T_prec*100:.1f}%")

T_polhode = T_prec  # for summary

# ── Part F: Practical implications for bridging ──
print(f"\n{'='*60}")
print("Part F: Implications for Constant-Omega Bridging")
print(f"{'='*60}")

# Between peaks, how much does omega drift?
for i in range(len(PEAK_EPOCHS) - 1):
    ep_a, ep_b = PEAK_EPOCHS[i], PEAK_EPOCHS[i+1]
    dt = ctx.observation_times[ep_b] - ctx.observation_times[ep_a]
    w_a = np.rad2deg(omega_history[ep_a])
    w_b = np.rad2deg(omega_history[ep_b])
    drift = np.linalg.norm(w_b - w_a)
    drift_pct = drift / np.linalg.norm(w_a) * 100

    # If we propagated with constant omega from peak_a, what attitude error at peak_b?
    # Under constant omega: q(t) = q0 * exp(0.5*omega*dt)
    # Under Euler: q(t) = propagated
    q_a = quats[ep_a]
    omega_a = omega_history[ep_a]

    # Constant-omega propagation
    from src.dynamics.attitude_propagator import propagate_principal_axis
    bridge_times = np.array([0.0, dt])
    q_const = propagate_principal_axis(q_a, omega_a, bridge_times)
    q_const_end = q_const[-1]

    # Euler propagation (truth)
    q_euler_end = quats[ep_b]

    # Attitude error
    R_const = Rotation.from_quat([q_const_end[1], q_const_end[2], q_const_end[3], q_const_end[0]])
    R_euler = Rotation.from_quat([q_euler_end[1], q_euler_end[2], q_euler_end[3], q_euler_end[0]])
    att_err = np.rad2deg((R_const.inv() * R_euler).magnitude())

    print(f"\n  Peak {ep_a} → Peak {ep_b}  (Δt = {dt:.1f}s)")
    print(f"    ω drift: {drift:.6f} deg/s  ({drift_pct:.4f}%)")
    print(f"    Attitude error (const-ω vs Euler): {att_err:.4f} deg")

# ── Summary ──
total_time = time.time() - t0
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  |ω₀|            = {np.linalg.norm(np.rad2deg(ctx.true_omega0)):.4f} deg/s")
print(f"  Max ω drift      = {max_drift:.6f} deg/s ({max_drift_pct:.4f}%)")
if T_prec is not None:
    print(f"  Precession period = {T_prec:.1f}s")
print(f"  Observation window = {ctx.observation_times[-1]:.1f}s")
print(f"  Total runtime    = {total_time:.1f}s")

# ── Save ──
results = {
    'inertia': {
        'tensor': [[round(float(I[i,j]), 4) for j in range(3)] for i in range(3)],
        'principal_moments': [round(float(x), 4) for x in eigvals],
        'ratios': {
            'I2_I1': round(float(I2/I1), 6),
            'I3_I1': round(float(I3/I1), 6),
            'I3_I2': round(float(I3/I2), 6),
        },
        'asymmetry_param': round(float(asym), 6),
    },
    'omega_at_peaks': peak_results,
    'trajectory_drift': {
        'max_delta_omega_deg_s': round(float(max_drift), 6),
        'max_delta_omega_pct': round(float(max_drift_pct), 4),
        'max_drift_epoch': int(max_drift_idx),
        'mean_delta_omega_deg_s': round(float(delta_mag.mean()), 6),
    },
    'conservation': {
        'max_L_relative_error': float(L_err.max()),
        'max_T_relative_error': float(T_err.max()),
    },
    'precession': {
        'linearized_period_s': round(float(T_prec), 2) if T_prec else None,
        'measured_period_s': round(float(T_measured), 2) if len(crossings) >= 2 else None,
        'nutation_angle_deg': round(float(nutation_angle), 4),
        'dominant_axis': dom_labels[dom_idx],
    },
    'config': {
        'seed': SEED,
        'n_observations': 500,
        'true_omega_deg': [0.5, -0.3, 2.0],
        'peak_epochs': PEAK_EPOCHS,
        'observation_window_s': float(ctx.observation_times[-1]),
    },
}
save_results(RESULTS_DIR / 'micro07a_omega_drift.json', results)
print(f"\nResults saved to {RESULTS_DIR / 'micro07a_omega_drift.json'}")
