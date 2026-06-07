#!/usr/bin/env python3
"""Micro-13b — Nudge test: how much does a small attitude error at peak 1 affect full-LC residual?"""
import sys, time, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

SEED = 42; PEAK = 183

t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=SEED,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00')
true_q, true_w = propagate_attitude(ctx.true_q0, ctx.true_omega0,
                                     ctx.observation_times, mode="tumbling",
                                     inertia_tensor=ctx.inertia_tensor)
print(f"Setup: {time.time()-t0:.1f}s\n")

def hifi_lc(quaternions):
    obj = ObjectiveFunction(satellite=ctx.satellite, observation_times=ctx.observation_times,
        observed_lightcurve=ctx.observed_lc, sun_positions_j2000=ctx.sun_pos,
        observer_positions_j2000=ctx.obs_pos, satellite_positions_j2000=ctx.sat_pos,
        observer_distances=ctx.obs_dist, compute_shadows_flag=True,
        articulation_matrices=ctx.art_matrices, mode="tumbling",
        inertia_tensor=ctx.inertia_tensor, show_progress=False)
    k1, k2 = obj._compute_body_frame_vectors(quaternions)
    lit = compute_shadows(ctx.satellite, k1, explicit_component_matrices=ctx.art_matrices,
                          show_progress=False)
    mags, *_ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=ctx.obs_dist, satellite=ctx.satellite,
        epochs=ctx.epochs, pre_computed_matrices=ctx.art_matrices, show_progress=False)
    return mags

rng = np.random.default_rng(SEED)
q_peak = true_q[PEAK]; w_peak = true_w[PEAK]
times_from_peak = ctx.observation_times - ctx.observation_times[PEAK]

print(f"{'Nudge':>7} {'RMS_truth':>10} {'RMS_nudge':>10} {'Ratio':>7} {'Time':>6}")
print("-" * 48)

# Truth trajectory from peak
t1 = time.time()
q_from_peak, _ = propagate_attitude(q_peak, w_peak, times_from_peak, mode="tumbling",
                                     inertia_tensor=ctx.inertia_tensor)
mags_truth = hifi_lc(q_from_peak)
rms_truth = np.sqrt(np.mean((mags_truth - ctx.observed_lc)**2))
print(f"{'0°':>7} {rms_truth:>10.4f} {'—':>10} {'—':>7} {time.time()-t1:>5.1f}s")

for nudge_deg in [2, 5, 10]:
    t1 = time.time()
    ax = rng.standard_normal(3); ax /= np.linalg.norm(ax)
    dR = Rotation.from_rotvec(np.deg2rad(nudge_deg) * ax)
    R_peak = Rotation.from_quat([q_peak[1], q_peak[2], q_peak[3], q_peak[0]])
    R_nudged = dR * R_peak
    q_nudged = R_nudged.as_quat()  # xyzw
    q_nudged_wxyz = np.array([q_nudged[3], q_nudged[0], q_nudged[1], q_nudged[2]])
    q_prop, _ = propagate_attitude(q_nudged_wxyz, w_peak, times_from_peak, mode="tumbling",
                                    inertia_tensor=ctx.inertia_tensor)
    mags_nudge = hifi_lc(q_prop)
    rms_nudge = np.sqrt(np.mean((mags_nudge - ctx.observed_lc)**2))
    print(f"{nudge_deg:>6}° {rms_truth:>10.4f} {rms_nudge:>10.4f} {rms_nudge/rms_truth:>7.2f}x {time.time()-t1:>5.1f}s")

print(f"\nTotal: {time.time()-t0:.1f}s")
