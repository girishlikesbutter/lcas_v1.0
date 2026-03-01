"""
EXPLORATION: How constraining is a single brightness measurement?

Questions to answer in 20 min:
1. How many random attitudes match the first observed brightness within noise?
   → Tells us if step 1 filtering is useful at all.
2. How many match BOTH epoch 0 AND epoch 1 (static attitude, no propagation)?
   → Tests the "multi-epoch static" tightening.
3. For survivors of (2), how many survive propagation to epoch 5 with omega candidates?
   → Tests whether the sequential filter converges.
4. Among final survivors, how close are any to truth?
   → Is the true solution findable this way?

Budget: 50k attitudes for step 1, tight thresholds.
"""
import sys, os, time, json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scipy.spatial.transform import Rotation
from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.dynamics import propagate_attitude
from src.inversion import ObjectiveFunction, axis_angle_to_quaternion, quaternion_to_axis_angle
from src.computation.shadow_engine import compute_shadows
from src.computation.lightcurve_generator import generate_lightcurves

N_OBS = 50; NOISE_SIGMA = 0.05; SEED = 42
t0 = time.perf_counter()
def elapsed(): return time.perf_counter() - t0

print("=" * 70, flush=True)
print("EXPLORATION: BRIGHTNESS FILTERING FEASIBILITY", flush=True)
print("=" * 70, flush=True)

# SETUP
print("[1] Setup...", flush=True)
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
metakernel_path = config_manager.get_metakernel_path(config)
satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)
brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)
component_masses = {'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0, 'AD_East': 50.0, 'AD_West': 50.0}
inertia_result = compute_inertia_from_config(config=config, config_manager=config_manager, masses=component_masses,
                                              articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
inertia_tensor = inertia_result.inertia_tensor
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))
start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, N_OBS)
observation_times = epochs - epochs[0]
geometry_data = compute_observation_geometry(epochs=epochs, satellite_id=config.spice_config.satellite_id,
    observer_id=399999, spice_handler=spice_handler, config=config)
fixed_articulation_angles = {
    'SP_North': np.full(N_OBS, 0.0), 'SP_South': np.full(N_OBS, 0.0),
    'AD_East': np.full(N_OBS, 15.0), 'AD_West': np.full(N_OBS, 15.0),
}
articulation_matrices = compute_rotation_matrices_from_angles(fixed_articulation_angles, satellite)

true_axis = np.array([0.6, 0.3, 0.8]); true_axis /= np.linalg.norm(true_axis)
true_angle_rad = np.deg2rad(45.0)
true_q0 = np.array([np.cos(true_angle_rad/2), np.sin(true_angle_rad/2)*true_axis[0],
                     np.sin(true_angle_rad/2)*true_axis[1], np.sin(true_angle_rad/2)*true_axis[2]])
true_omega0 = np.deg2rad(np.array([0.005, -0.003, 0.05]))
true_axis_angle = quaternion_to_axis_angle(true_q0)
true_params = np.concatenate([true_axis_angle, true_omega0])

print("[2] Generating observed lightcurve...", flush=True)
true_quaternions, _ = propagate_attitude(q0=true_q0, omega0=true_omega0, times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor)
obj_gen = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=np.zeros(N_OBS), sun_positions_j2000=geometry_data['sun_positions'],
    observer_positions_j2000=geometry_data['obs_positions'],
    satellite_positions_j2000=geometry_data['sat_positions'],
    observer_distances=geometry_data['observer_distances'], compute_shadows_flag=True,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)
k1_vectors, k2_vectors = obj_gen._compute_body_frame_vectors(true_quaternions)
lit_status_dict = compute_shadows(satellite=satellite, k1_vectors=k1_vectors,
    explicit_component_matrices=articulation_matrices, show_progress=False)
true_lightcurve, _, _, _, _, _ = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict, k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors, observer_distances=geometry_data['observer_distances'],
    satellite=satellite, epochs=epochs, pre_computed_matrices=articulation_matrices,
    generate_no_shadow=False, animate=False, show_progress=False)
np.random.seed(SEED)
observed_lightcurve = true_lightcurve + np.random.normal(0, NOISE_SIGMA, N_OBS)
print(f"  Observed LC range: [{observed_lightcurve.min():.3f}, {observed_lightcurve.max():.3f}]", flush=True)
print(f"  First 5 observed: {observed_lightcurve[:5].round(3)}", flush=True)
print(f"  Timestep: {observation_times[1] - observation_times[0]:.1f}s", flush=True)
print(f"  True omega magnitude: {np.rad2deg(np.linalg.norm(true_omega0)):.4f}°/s", flush=True)
print(f"  Rotation per timestep: {np.rad2deg(np.linalg.norm(true_omega0)) * (observation_times[1]-observation_times[0]):.2f}°", flush=True)
print(f"  Done in {elapsed():.1f}s", flush=True)

# Helper: evaluate brightness for a given attitude at specific epochs (no propagation)
# We need ObjectiveFunction for each epoch subset. But creating many is slow.
# Instead, create one for first 5 epochs and extract per-epoch residuals.
obj_5ep = ObjectiveFunction(satellite=satellite, observation_times=observation_times[:5],
    observed_lightcurve=observed_lightcurve[:5], sun_positions_j2000=geometry_data['sun_positions'][:5],
    observer_positions_j2000=geometry_data['obs_positions'][:5],
    satellite_positions_j2000=geometry_data['sat_positions'][:5],
    observer_distances=geometry_data['observer_distances'][:5],
    compute_shadows_flag=False,
    articulation_matrices={k: v[:5] for k, v in articulation_matrices.items()},
    mode="tumbling", inertia_tensor=inertia_tensor, show_progress=False)

# Full lo-fi objective for later
obj_lofi = ObjectiveFunction(satellite=satellite, observation_times=observation_times,
    observed_lightcurve=observed_lightcurve, sun_positions_j2000=geometry_data['sun_positions'],
    observer_positions_j2000=geometry_data['obs_positions'],
    satellite_positions_j2000=geometry_data['sat_positions'],
    observer_distances=geometry_data['observer_distances'], compute_shadows_flag=False,
    articulation_matrices=articulation_matrices, mode="tumbling", inertia_tensor=inertia_tensor,
    show_progress=False)

def compute_att_error(aa):
    q_true = axis_angle_to_quaternion(true_axis_angle)
    q_x = axis_angle_to_quaternion(aa)
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    return np.rad2deg(2 * np.arccos(dot))

# ============================================================================
# Q1: How constraining is a single brightness measurement?
# ============================================================================
print(f"\n[Q1] How many of 50k random attitudes match observed brightness at epoch 0?", flush=True)
print(f"  Using 5-epoch lo-fi objective with omega=0...", flush=True)
# With omega=0, all 5 epochs see the same static attitude.
# The residual at epoch 0 tells us the brightness match.

N_TEST = 50000
rng = np.random.default_rng(SEED)
rand_quats = rng.normal(size=(N_TEST, 4))
rand_quats /= np.linalg.norm(rand_quats, axis=1, keepdims=True)
rand_quats[rand_quats[:, 0] < 0] *= -1

scores_5ep = np.zeros(N_TEST)
t_q1 = time.perf_counter()

for i in range(N_TEST):
    aa = quaternion_to_axis_angle(rand_quats[i])
    x = np.concatenate([aa, np.zeros(3)])  # omega=0
    scores_5ep[i] = obj_5ep.evaluate(x)
    
    if (i+1) % 10000 == 0:
        rate = (i+1) / (time.perf_counter() - t_q1)
        print(f"  [{i+1}/{N_TEST}] {rate:.0f}/s", flush=True)

t_q1 = time.perf_counter() - t_q1
print(f"  Screened {N_TEST} in {t_q1:.1f}s ({N_TEST/t_q1:.0f}/s)", flush=True)

# Score distribution
pcts = [1, 2, 5, 10, 25, 50]
for p in pcts:
    thresh = np.percentile(scores_5ep, p)
    n_pass = np.sum(scores_5ep <= thresh)
    # Check attitude errors of those passing
    pass_errs = []
    for idx in np.where(scores_5ep <= thresh)[0][:100]:  # sample up to 100
        aa = quaternion_to_axis_angle(rand_quats[idx])
        pass_errs.append(compute_att_error(aa))
    min_err = min(pass_errs) if pass_errs else 999
    med_err = np.median(pass_errs) if pass_errs else 999
    print(f"  Top {p:2d}%: threshold={thresh:.4f}, n={n_pass:5d}, min_att_err={min_err:.1f}°, median_att_err={med_err:.1f}°", flush=True)

# ============================================================================
# Q2: Does the true attitude survive filtering?
# ============================================================================
print(f"\n[Q2] Where does the true attitude rank?", flush=True)
x_true = np.concatenate([true_axis_angle, np.zeros(3)])
true_score = obj_5ep.evaluate(x_true)
rank = np.sum(scores_5ep < true_score) + 1
pct = 100.0 * rank / N_TEST
print(f"  True attitude score (ω=0): {true_score:.6f}", flush=True)
print(f"  Rank: {rank}/{N_TEST} (top {pct:.2f}%)", flush=True)
# Note: true attitude with ω=0 won't match perfectly because the real lightcurve
# was generated WITH rotation. The first few epochs barely rotate though.

# Also try true attitude with true omega
x_true_full = true_params.copy()
true_full_score = obj_5ep.evaluate(x_true_full)
rank_full = np.sum(scores_5ep < true_full_score) + 1
pct_full = 100.0 * rank_full / N_TEST
print(f"  True attitude+omega score: {true_full_score:.6f}", flush=True)
print(f"  Rank: {rank_full}/{N_TEST} (top {pct_full:.2f}%)", flush=True)

# ============================================================================
# Q3: Selectivity per epoch
# ============================================================================
print(f"\n[Q3] Per-epoch analysis — how quickly does sequential filtering converge?", flush=True)

# Create objectives for 2, 3, 4, 5 epochs
for n_ep in [2, 3, 5]:
    obj_nep = ObjectiveFunction(satellite=satellite, observation_times=observation_times[:n_ep],
        observed_lightcurve=observed_lightcurve[:n_ep],
        sun_positions_j2000=geometry_data['sun_positions'][:n_ep],
        observer_positions_j2000=geometry_data['obs_positions'][:n_ep],
        satellite_positions_j2000=geometry_data['sat_positions'][:n_ep],
        observer_distances=geometry_data['observer_distances'][:n_ep],
        compute_shadows_flag=False,
        articulation_matrices={k: v[:n_ep] for k, v in articulation_matrices.items()},
        mode="tumbling", inertia_tensor=inertia_tensor, show_progress=False)
    
    # Test top 1% from Q1 with true omega
    top1_idx = np.where(scores_5ep <= np.percentile(scores_5ep, 1))[0]
    scores_nep = []
    for idx in top1_idx:
        aa = quaternion_to_axis_angle(rand_quats[idx])
        x = np.concatenate([aa, true_omega0])  # give true omega
        s = obj_nep.evaluate(x)
        err = compute_att_error(aa)
        scores_nep.append((s, err, idx))
    
    scores_nep.sort(key=lambda r: r[0])
    top5 = scores_nep[:5]
    print(f"  {n_ep} epochs (true ω, top 1% attitudes):", flush=True)
    for j, (s, e, idx) in enumerate(top5):
        print(f"    {j+1}. score={s:.4f} att_err={e:.1f}°", flush=True)

# ============================================================================
# Q4: With omega grid, can we find truth among top 1% attitudes?
# ============================================================================
print(f"\n[Q4] Top 1% attitudes × omega grid → propagate 10 epochs, check match", flush=True)
t_q4 = time.perf_counter()

obj_10ep = ObjectiveFunction(satellite=satellite, observation_times=observation_times[:10],
    observed_lightcurve=observed_lightcurve[:10],
    sun_positions_j2000=geometry_data['sun_positions'][:10],
    observer_positions_j2000=geometry_data['obs_positions'][:10],
    satellite_positions_j2000=geometry_data['sat_positions'][:10],
    observer_distances=geometry_data['observer_distances'][:10],
    compute_shadows_flag=False,
    articulation_matrices={k: v[:10] for k, v in articulation_matrices.items()},
    mode="tumbling", inertia_tensor=inertia_tensor, show_progress=False)

# Omega grid: magnitude from 0.01 to 0.1°/s, 5 mags × 26 directions (icosahedron vertices)
omega_mags = np.deg2rad(np.array([0.01, 0.03, 0.05, 0.07, 0.1]))
# Simple: 6 axis directions + 8 octant diagonals = 14 directions
dirs = []
for s1 in [-1, 1]:
    for s2 in [-1, 1]:
        for s3 in [-1, 1]:
            d = np.array([s1, s2, s3], dtype=float)
            dirs.append(d / np.linalg.norm(d))
for ax in range(3):
    for s in [-1, 1]:
        d = np.zeros(3)
        d[ax] = s
        dirs.append(d)
dirs = np.array(dirs)  # 14 directions

omega_candidates = []
for mag in omega_mags:
    for d in dirs:
        omega_candidates.append(mag * d)
omega_candidates = np.array(omega_candidates)
print(f"  {len(top1_idx)} attitudes × {len(omega_candidates)} omegas = {len(top1_idx)*len(omega_candidates)} combos", flush=True)

best_combos = []
for i, idx in enumerate(top1_idx):
    aa = quaternion_to_axis_angle(rand_quats[idx])
    att_err = compute_att_error(aa)
    
    for omg in omega_candidates:
        x = np.concatenate([aa, omg])
        s = obj_10ep.evaluate(x)
        omega_err = np.rad2deg(np.linalg.norm(omg - true_omega0))
        best_combos.append((s, att_err, omega_err, x.copy()))
    
    if (i+1) % 100 == 0:
        best_combos.sort(key=lambda r: r[0])
        bs, bae, boe, _ = best_combos[0]
        print(f"  [{i+1}/{len(top1_idx)}] best: score={bs:.4f} att={bae:.1f}° ω={boe:.4f}°/s", flush=True)

t_q4 = time.perf_counter() - t_q4
best_combos.sort(key=lambda r: r[0])
print(f"  Q4 done: {t_q4:.1f}s", flush=True)
print(f"  Top 10:", flush=True)
for i, (s, ae, oe, x) in enumerate(best_combos[:10]):
    print(f"    {i+1}. score={s:.6f} att={ae:.1f}° ω={oe:.4f}°/s", flush=True)

# ============================================================================
# Q5: Do top Q4 survivors converge with full lo-fi?
# ============================================================================
print(f"\n[Q5] Full lo-fi (50 epochs) on top 5 Q4 candidates...", flush=True)
from scipy.optimize import minimize
omega_bound = np.deg2rad(0.15)
joint_bounds = [(-np.pi, np.pi)] * 3 + [(-omega_bound, omega_bound)] * 3

for i, (s0, ae0, oe0, x0) in enumerate(best_combos[:5]):
    rl = minimize(lambda x: obj_lofi.evaluate(x), x0, method='L-BFGS-B', bounds=joint_bounds,
                  options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-12})
    q_x = axis_angle_to_quaternion(rl.x[:3])
    q_true = axis_angle_to_quaternion(true_axis_angle)
    dot = min(np.abs(np.dot(q_true, q_x)), 1.0)
    ae_f = np.rad2deg(2 * np.arccos(dot))
    oe_f = np.rad2deg(np.linalg.norm(rl.x[3:] - true_omega0))
    success = ae_f < 5 and oe_f < 0.1
    sym = "✓" if success else "✗"
    print(f"  [{i+1}] {sym} init:{ae0:.1f}°/{oe0:.4f} → final:{ae_f:.2f}°/{oe_f:.4f}°/s (f={rl.fun:.4f})", flush=True)

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}", flush=True)
print(f"EXPLORATION COMPLETE: {elapsed():.0f}s ({elapsed()/60:.1f} min)", flush=True)
print(f"{'='*70}", flush=True)
print(f"\nKEY FINDINGS:", flush=True)
print(f"  Q1: Brightness selectivity at 5 epochs (static, ω=0)", flush=True)
print(f"  Q2: True attitude rank among 50k random", flush=True)
print(f"  Q3: Selectivity improves with more epochs + true omega", flush=True)
print(f"  Q4: Attitude × omega grid → best candidates at 10 epochs", flush=True)
print(f"  Q5: Full lo-fi convergence from top candidates", flush=True)
