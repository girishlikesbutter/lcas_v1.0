#!/usr/bin/env python3
"""
m091 — Hi-fi twin symmetry test.

Start from identity attitude with off-axis omega. For each of the 10 normal
group vectors, apply a 180° rotation to q0 (keeping same omega), propagate
the full attitude history, and generate a hi-fi light curve with shadows.

Compare each LC against the reference (unrotated) to identify which 180°
rotations are true optical twins.

Saves all LCs and quaternion histories to NPZ for plotting.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.dynamics.attitude_propagator import propagate_attitude
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m091_twin_test"
RESULTS_DIR.mkdir(exist_ok=True)

# Setup
print("Loading satellite model...", flush=True)
ctx = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)

I = ctx.inertia_tensor
obs_times = ctx.observation_times
q0_identity = np.array([1.0, 0.0, 0.0, 0.0])
w0 = np.deg2rad(np.array([0.5, -0.3, 2.0]))

master = np.load(str(PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" /
                      "m046_trajectories" / "m046_trajectories.npz"), allow_pickle=True)
normals = master['unique_normals']
names = list(master['group_names'])


def make_lc_hifi(q0_w, w0_w):
    """Generate hi-fi LC with shadows. Returns (magnitudes, quaternion_history)."""
    quats, _ = propagate_attitude(q0_w, w0_w, obs_times, "tumbling", I)
    n_ep = len(quats)
    R_all = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
    sv = ctx.sun_pos[:n_ep] - ctx.sat_pos[:n_ep]
    sv /= np.linalg.norm(sv, axis=1, keepdims=True)
    ov = ctx.obs_pos[:n_ep] - ctx.sat_pos[:n_ep]
    ov /= np.linalg.norm(ov, axis=1, keepdims=True)
    k1 = np.einsum('nij,nj->ni', R_all, sv)
    k2 = np.einsum('nij,nj->ni', R_all, ov)
    lit = compute_shadows(satellite=ctx.satellite, k1_vectors=k1,
                          explicit_component_matrices=ctx.art_matrices, show_progress=False)
    mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1, k2_vectors_array=k2,
        observer_distances=ctx.obs_dist, satellite=ctx.satellite,
        epochs=np.arange(n_ep, dtype=float), pre_computed_matrices=ctx.art_matrices,
        show_progress=False)
    return mags, quats


# Reference LC
print(f"q0 = identity, w0 = [0.5, -0.3, 2.0] deg/s", flush=True)
print(f"500 epochs over {obs_times[-1]:.0f}s, hi-fi with shadows", flush=True)
print(flush=True)

t0 = time.time()
lc_ref, quats_ref = make_lc_hifi(q0_identity, w0)
print(f"Reference LC: {time.time()-t0:.1f}s", flush=True)

# Store all results
all_lcs = {'reference': lc_ref}
all_quats = {'reference': quats_ref}
results = []

# Test each normal
print(flush=True)
print(f"{'Normal':6s}  {'axis':>30s}  {'LC RMS':>10s}  {'LC max':>10s}  {'quat RMS':>10s}  {'time':>5s}",
      flush=True)
print("-" * 80, flush=True)

for i, (name, n) in enumerate(zip(names, normals)):
    t1 = time.time()

    R_rot = Rotation.from_rotvec(np.pi * n)
    q_xyzw = R_rot.as_quat()
    q0_rotated = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    lc_rot, quats_rot = make_lc_hifi(q0_rotated, w0)

    lc_rms = np.sqrt(np.mean((lc_ref - lc_rot)**2))
    lc_max = np.max(np.abs(lc_ref - lc_rot))
    q_diff = np.linalg.norm(quats_ref - quats_rot, axis=1)
    q_rms = np.sqrt(np.mean(q_diff**2))

    dt = time.time() - t1
    tag = "  <<<" if lc_rms < 0.01 else ""
    print(f"{name:6s}  [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]  "
          f"{lc_rms:10.6f}  {lc_max:10.6f}  {q_rms:10.6f}  {dt:4.0f}s{tag}", flush=True)

    all_lcs[name] = lc_rot
    all_quats[name] = quats_rot
    results.append({
        'name': name, 'normal': n.tolist(),
        'lc_rms': float(lc_rms), 'lc_max': float(lc_max), 'quat_rms': float(q_rms),
    })

print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)

# Save everything
np.savez(str(RESULTS_DIR / "twin_lcs.npz"),
         obs_times=obs_times,
         lc_reference=lc_ref, quats_reference=quats_ref,
         **{f'lc_{name}': lc for name, lc in all_lcs.items() if name != 'reference'},
         **{f'quats_{name}': q for name, q in all_quats.items() if name != 'reference'},
         normals=normals, names=np.array(names))

import json
with open(str(RESULTS_DIR / "result.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {RESULTS_DIR}/", flush=True)
