#!/usr/bin/env python3
"""Exp 01 — Forward Model Sanity. Does the forward model round-trip correctly?"""
import sys, time, json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, save_results
from src.inversion.objective_function import ObjectiveFunction, _quaternion_to_rotation_matrix
from src.inversion.quaternion_utils import axis_angle_to_quaternion, quaternion_to_axis_angle
from src.dynamics.attitude_propagator import propagate_attitude

# ── Config ──
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── L-param helpers ──
def omega_to_L(q_wxyz, omega_body, inertia_tensor):
    """Convert body-frame omega to inertial angular momentum.
    R(q) is body->inertial (propagator uses q_dot = 0.5*q*omega_quat)."""
    R = _quaternion_to_rotation_matrix(q_wxyz)
    L_body = inertia_tensor @ omega_body
    return R @ L_body  # body -> inertial

def L_to_omega(q_wxyz, L_inertial, inertia_tensor):
    """Convert inertial angular momentum to body-frame omega."""
    R = _quaternion_to_rotation_matrix(q_wxyz)
    L_body = R.T @ L_inertial  # inertial -> body
    return np.linalg.solve(inertia_tensor, L_body)

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
print(f"Setup: {time.time()-t0:.1f}s")

true_aa = quaternion_to_axis_angle(ctx.true_q0)
true_params = np.concatenate([true_aa, ctx.true_omega0])
I = ctx.inertia_tensor

def make_obj(shadows):
    return ObjectiveFunction(
        satellite=ctx.satellite,
        observation_times=ctx.observation_times,
        observed_lightcurve=ctx.observed_lc,
        sun_positions_j2000=ctx.sun_pos,
        observer_positions_j2000=ctx.obs_pos,
        satellite_positions_j2000=ctx.sat_pos,
        observer_distances=ctx.obs_dist,
        compute_shadows_flag=shadows,
        articulation_matrices=ctx.art_matrices,
        mode="tumbling", inertia_tensor=I, show_progress=False)

obj_lofi = make_obj(False)
obj_hifi = make_obj(True)

results = {}
all_pass = True

# ── Check 1: Lo-fi residual at truth ──
print("\n1. Lo-fi residual at truth:")
lofi_res = obj_lofi.evaluate(true_params)
print(f"   MSE = {lofi_res:.6f}")
print(f"   (noise sigma^2 = {ctx.noise_sigma**2:.4f})")
results['lofi_residual_at_truth'] = lofi_res

# ── Check 2: Hi-fi residual at truth ──
print("\n2. Hi-fi residual at truth:")
hifi_res = obj_hifi.evaluate(true_params)
print(f"   MSE = {hifi_res:.6f}")
print(f"   Expected ~ noise_sigma^2 = {ctx.noise_sigma**2:.4f}")
# Hi-fi should be close to noise^2 since observed LC was generated with hi-fi + noise
hifi_pass = abs(hifi_res - ctx.noise_sigma**2) < 0.01
print(f"   PASS: {hifi_pass}")
results['hifi_residual_at_truth'] = hifi_res
results['hifi_residual_pass'] = hifi_pass
if not hifi_pass:
    all_pass = False

# ── Check 3: Conservation (T and L) along true trajectory ──
print("\n3. Conservation check (T and L):")
quats, omegas = propagate_attitude(ctx.true_q0, ctx.true_omega0,
                                    ctx.observation_times, "tumbling", I)
check_epochs = [0, 50, 100, 250, 499]
T_vals = []
L_vecs = []
for idx in check_epochs:
    q = quats[idx]
    w = omegas[idx]
    # Kinetic energy: T = 0.5 * omega^T @ I @ omega
    T = 0.5 * w @ I @ w
    T_vals.append(T)
    # Angular momentum in inertial frame: L = R^T @ (I @ omega)
    L = omega_to_L(q, w, I)
    L_vecs.append(L)

T_vals = np.array(T_vals)
L_mags = np.array([np.linalg.norm(L) for L in L_vecs])

T_spread = (T_vals.max() - T_vals.min()) / T_vals.mean()
L_spread = (L_mags.max() - L_mags.min()) / L_mags.mean()
print(f"   T values:  {T_vals}")
print(f"   T relative spread: {T_spread:.2e}")
print(f"   |L| values: {L_mags}")
print(f"   |L| relative spread: {L_spread:.2e}")
T_pass = T_spread < 1e-6
L_pass = L_spread < 1e-6
print(f"   T conserved (< 1e-6): {T_pass}")
print(f"   |L| conserved (< 1e-6): {L_pass}")

# Also check L direction consistency
L_vecs = np.array(L_vecs)
L_dirs = L_vecs / L_mags[:, None]
dot_products = np.array([np.dot(L_dirs[0], L_dirs[i]) for i in range(len(L_dirs))])
L_dir_spread = 1.0 - dot_products.min()
print(f"   L direction spread (1-min_cos): {L_dir_spread:.2e}")
L_dir_pass = L_dir_spread < 1e-6
print(f"   L direction conserved (< 1e-6): {L_dir_pass}")

results['T_values'] = T_vals.tolist()
results['T_relative_spread'] = T_spread
results['L_magnitudes'] = L_mags.tolist()
results['L_relative_spread'] = L_spread
results['L_direction_spread'] = L_dir_spread
results['conservation_pass'] = bool(T_pass and L_pass and L_dir_pass)
if not (T_pass and L_pass and L_dir_pass):
    all_pass = False

# ── Check 4: L <-> omega roundtrip ──
print("\n4. L <-> omega roundtrip:")
L_true = omega_to_L(ctx.true_q0, ctx.true_omega0, I)
omega_recovered = L_to_omega(ctx.true_q0, L_true, I)
roundtrip_err = np.linalg.norm(omega_recovered - ctx.true_omega0)
print(f"   true omega:      {ctx.true_omega0}")
print(f"   recovered omega: {omega_recovered}")
print(f"   ||error||: {roundtrip_err:.2e}")
roundtrip_pass = roundtrip_err < 1e-12
print(f"   PASS (< 1e-12): {roundtrip_pass}")
results['L_omega_roundtrip_error'] = roundtrip_err
results['L_omega_roundtrip_pass'] = bool(roundtrip_pass)
if not roundtrip_pass:
    all_pass = False

# Also verify at a later epoch
L_later = omega_to_L(quats[250], omegas[250], I)
omega_recovered_later = L_to_omega(quats[250], L_later, I)
roundtrip_err_later = np.linalg.norm(omega_recovered_later - omegas[250])
print(f"   Roundtrip at epoch 250: ||error|| = {roundtrip_err_later:.2e}")

# ── Check 5: L-param vs omega-param objective at truth ──
print("\n5. L-param vs omega-param objective at truth:")
# omega-param: standard evaluate
omega_param_val = obj_lofi.evaluate(true_params)

# L-param: convert omega -> L, then define f_L that converts back
L_true = omega_to_L(ctx.true_q0, ctx.true_omega0, I)
def f_L(L_vec):
    """Evaluate objective with L parameterization."""
    omega_from_L = L_to_omega(ctx.true_q0, L_vec, I)
    params = np.concatenate([true_aa, omega_from_L])
    return obj_lofi.evaluate(params)

L_param_val = f_L(L_true)
diff = abs(omega_param_val - L_param_val)
print(f"   omega-param value: {omega_param_val:.10f}")
print(f"   L-param value:     {L_param_val:.10f}")
print(f"   |difference|:      {diff:.2e}")
L_param_pass = diff < 1e-10
print(f"   PASS (< 1e-10): {L_param_pass}")
results['omega_param_value'] = omega_param_val
results['L_param_value'] = L_param_val
results['L_omega_diff'] = diff
results['L_param_pass'] = bool(L_param_pass)
if not L_param_pass:
    all_pass = False

# ── Summary ──
print("\n" + "="*60)
print("SANITY CHECK SUMMARY")
print("="*60)
print(f"  Lo-fi residual at truth:    {lofi_res:.6f}")
print(f"  Hi-fi residual at truth:    {hifi_res:.6f}  (expected ~{ctx.noise_sigma**2:.4f})")
print(f"  T relative spread:          {T_spread:.2e}  {'PASS' if T_pass else 'FAIL'}")
print(f"  |L| relative spread:        {L_spread:.2e}  {'PASS' if L_pass else 'FAIL'}")
print(f"  L direction spread:         {L_dir_spread:.2e}  {'PASS' if L_dir_pass else 'FAIL'}")
print(f"  L<->omega roundtrip error:  {roundtrip_err:.2e}  {'PASS' if roundtrip_pass else 'FAIL'}")
print(f"  L-param vs omega-param:     {diff:.2e}  {'PASS' if L_param_pass else 'FAIL'}")
print(f"\n  ALL CHECKS: {'PASS' if all_pass else 'FAIL'}")

results['all_pass'] = all_pass
total_time = time.time() - t0
results['total_script_time_s'] = round(total_time, 1)
print(f"\nTotal script time: {total_time:.0f}s")

save_results(RESULTS_DIR / 'exp01_sanity.json', results)
print(f"Results saved to {RESULTS_DIR / 'exp01_sanity.json'}")
