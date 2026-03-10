#!/usr/bin/env python3
"""Micro-13c — Compare truth vs 2°-nudged lightcurves from epoch 0."""
import sys, time, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports
import os; os.chdir(PROJECT_ROOT)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from lib.experiment_setup import setup_experiment
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

def hifi_lc(quats, ctx):
    obj = ObjectiveFunction(satellite=ctx.satellite, observation_times=ctx.observation_times,
        observed_lightcurve=ctx.observed_lc, sun_positions_j2000=ctx.sun_pos,
        observer_positions_j2000=ctx.obs_pos, satellite_positions_j2000=ctx.sat_pos,
        observer_distances=ctx.obs_dist, compute_shadows_flag=True,
        articulation_matrices=ctx.art_matrices, mode="tumbling",
        inertia_tensor=ctx.inertia_tensor, show_progress=False)
    k1, k2 = obj._compute_body_frame_vectors(quats)
    lit = compute_shadows(ctx.satellite, k1, explicit_component_matrices=ctx.art_matrices, show_progress=False)
    mags, *_ = generate_lightcurves(facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=ctx.obs_dist, satellite=ctx.satellite,
        epochs=ctx.epochs, pre_computed_matrices=ctx.art_matrices, show_progress=False)
    return mags

t0 = time.time()
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0), end_time_utc='2020-02-05T11:00:00')
q_truth, _ = propagate_attitude(ctx.true_q0, ctx.true_omega0, ctx.observation_times,
                                 mode="tumbling", inertia_tensor=ctx.inertia_tensor)
print(f"Setup: {time.time()-t0:.1f}s"); t1 = time.time()
lc_truth = hifi_lc(q_truth, ctx)
print(f"Truth LC: {time.time()-t1:.1f}s"); t1 = time.time()

rng = np.random.default_rng(42)
ax = rng.standard_normal(3); ax /= np.linalg.norm(ax)
dR = Rotation.from_rotvec(np.deg2rad(2.0) * ax)
R0 = Rotation.from_quat([ctx.true_q0[1], ctx.true_q0[2], ctx.true_q0[3], ctx.true_q0[0]])
q_n = (dR * R0).as_quat()  # xyzw
q0_nudged = np.array([q_n[3], q_n[0], q_n[1], q_n[2]])
q_nudged, _ = propagate_attitude(q0_nudged, ctx.true_omega0, ctx.observation_times,
                                  mode="tumbling", inertia_tensor=ctx.inertia_tensor)
lc_nudged = hifi_lc(q_nudged, ctx)
print(f"Nudged LC: {time.time()-t1:.1f}s")

rms = np.sqrt(np.mean((lc_truth - lc_nudged)**2))
print(f"\nRMS(truth vs nudged): {rms:.4f} mag")
print(f"RMS(truth vs observed): {np.sqrt(np.mean((lc_truth - ctx.observed_lc)**2)):.4f} mag")

fig, ax = plt.subplots(figsize=(14, 5))
t = ctx.observation_times
ax.plot(t, lc_truth, '-', lw=0.8, label='Truth q0')
ax.plot(t, lc_nudged, '-', lw=0.8, label='Nudged q0 (+2°)')
ax.plot(t, ctx.observed_lc, '.', ms=2, alpha=0.4, label='Observed (hi-fi+noise)')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Magnitude'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title(f'Truth vs 2°-nudged LC (RMS diff = {rms:.4f} mag)')
fig.tight_layout(); fig.savefig('/tmp/micro13c_lc_comparison.png', dpi=150); plt.close()
print(f"Plot saved: /tmp/micro13c_lc_comparison.png\nTotal: {time.time()-t0:.1f}s")
