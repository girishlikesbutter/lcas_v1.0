#!/usr/bin/env python3
"""Exp 00 — Timing Benchmark. How long does each operation take for our fast-tumbler test case?"""
import sys, time, json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, brightness_single_epoch, save_results
from src.inversion.objective_function import ObjectiveFunction
from src.inversion.quaternion_utils import quaternion_to_axis_angle
from src.dynamics.attitude_propagator import propagate_attitude
from scipy.optimize import minimize

# ── Config ──
N_REPS = 10
RESULTS_DIR = Path('data/results/inversion_diagnostics')

# ── Setup ──
t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
print(f"Setup: {time.time()-t0:.1f}s")

true_aa = quaternion_to_axis_angle(ctx.true_q0)
true_params = np.concatenate([true_aa, ctx.true_omega0])
I = ctx.inertia_tensor

# Build lo-fi and hi-fi objective functions
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

# ── Timing helper ──
def bench(func, n=N_REPS, label=""):
    times_ms = []
    for _ in range(n):
        t = time.perf_counter()
        func()
        times_ms.append((time.perf_counter() - t) * 1000)
    med = np.median(times_ms)
    print(f"  {label}: {med:.1f} ms (median of {n})")
    return med

results = {}

# ── 1. propagate_attitude ──
print("\n1. propagate_attitude (500 epochs, 3600s tumbling):")
results['propagate_attitude_ms'] = bench(
    lambda: propagate_attitude(ctx.true_q0, ctx.true_omega0, ctx.observation_times,
                               "tumbling", I),
    label="ODE only")

# ── 2. Lo-fi full LC eval ──
print("\n2. Lo-fi full LC eval (500 epochs, no shadows):")
results['lofi_full_lc_ms'] = bench(
    lambda: obj_lofi.evaluate(true_params),
    label="lo-fi evaluate")

# ── 3. Hi-fi full LC eval ──
print("\n3. Hi-fi full LC eval (500 epochs, with shadows):")
results['hifi_full_lc_ms'] = bench(
    lambda: obj_hifi.evaluate(true_params),
    label="hi-fi evaluate")

# ── 4. Single-epoch lo-fi ──
print("\n4. Single-epoch lo-fi brightness:")
results['single_epoch_lofi_ms'] = bench(
    lambda: brightness_single_epoch(ctx.true_q0, 0, ctx, use_shadows=False),
    label="single epoch lo-fi")

# ── 5. Single-epoch hi-fi ──
print("\n5. Single-epoch hi-fi brightness:")
results['single_epoch_hifi_ms'] = bench(
    lambda: brightness_single_epoch(ctx.true_q0, 0, ctx, use_shadows=True),
    label="single epoch hi-fi")

# ── 6. L-BFGS-B iteration timing (6 params, lo-fi, maxiter=10) ──
print("\n6. L-BFGS-B (6 params, lo-fi, maxiter=10) from truth + tiny perturbation:")
np.random.seed(99)
perturbed = true_params.copy()
perturbed[:3] += np.random.normal(0, 1e-4, 3)   # tiny attitude perturbation
perturbed[3:] += np.random.normal(0, 1e-6, 3)   # tiny omega perturbation

obj_lofi.n_evaluations = 0
t_start = time.perf_counter()
res = minimize(obj_lofi.evaluate, perturbed, method='L-BFGS-B',
               options={'maxiter': 10, 'disp': False})
t_lbfgsb = (time.perf_counter() - t_start) * 1000
n_evals = obj_lofi.n_evaluations
n_iters = res.nit
ms_per_iter = t_lbfgsb / max(n_iters, 1)
print(f"  Total: {t_lbfgsb:.0f} ms, {n_iters} iters, {n_evals} evals")
print(f"  Per iteration: {ms_per_iter:.0f} ms")
print(f"  Per evaluation: {t_lbfgsb/max(n_evals,1):.1f} ms")
results['lbfgsb_6param_total_ms'] = round(t_lbfgsb, 1)
results['lbfgsb_6param_n_iters'] = n_iters
results['lbfgsb_6param_n_evals'] = n_evals
results['lbfgsb_6param_ms_per_iter'] = round(ms_per_iter, 1)

# ── Summary ──
print("\n" + "="*60)
print("TIMING SUMMARY")
print("="*60)
print(f"  propagate_attitude (500 epochs): {results['propagate_attitude_ms']:.1f} ms")
print(f"  lo-fi full LC eval:              {results['lofi_full_lc_ms']:.1f} ms")
print(f"  hi-fi full LC eval:              {results['hifi_full_lc_ms']:.1f} ms")
print(f"  single-epoch lo-fi:              {results['single_epoch_lofi_ms']:.1f} ms")
print(f"  single-epoch hi-fi:              {results['single_epoch_hifi_ms']:.1f} ms")
print(f"  L-BFGS-B iteration (6p, lo-fi):  {results['lbfgsb_6param_ms_per_iter']:.0f} ms")
lofi_hifi_ratio = results['hifi_full_lc_ms'] / max(results['lofi_full_lc_ms'], 0.1)
print(f"  hi-fi / lo-fi ratio:             {lofi_hifi_ratio:.0f}x")
results['hifi_lofi_ratio'] = round(lofi_hifi_ratio, 1)

total_time = time.time() - t0
print(f"\nTotal script time: {total_time:.0f}s")
results['total_script_time_s'] = round(total_time, 1)

save_results(RESULTS_DIR / 'exp00_timing.json', results)
print(f"\nResults saved to {RESULTS_DIR / 'exp00_timing.json'}")
