#!/usr/bin/env python3
"""
m072c — Test non-glint peak count filter on m072 candidates.

Takes the 10 geo-refined candidates from m072, computes hi-fi LC for each
(parallelized), counts non-glint peaks (mag > 9.0), and tests a composite
score: hifi_residual * (1 + alpha * |n_cand - n_obs| / n_obs).
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment, attitude_error_deg
from src.dynamics.attitude_propagator import propagate_attitude
from src.inversion.objective_function import ObjectiveFunction

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
PROM = 0.2
MAG_THRESHOLD = 9.0
HIFI_WORKERS = 8


def omega_dir_err(w1, w2):
    d1, d2 = w1 / np.linalg.norm(w1), w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
print("Loading...", flush=True)
CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
obs_times = master['observation_times']
I_tensor = master['inertia_tensor']
true_q0 = master['q0s'][93]
true_omega0 = master['omega0s'][93]
true_mags = master['mag_hifi'][93]

rng = np.random.default_rng(42)
observed_lc = true_mags + rng.normal(0, 0.05, 500)

# Observed non-glint peak count
obs_peaks, _ = find_peaks(-observed_lc, prominence=PROM)
obs_nonglint = obs_peaks[observed_lc[obs_peaks] > MAG_THRESHOLD]
n_obs = len(obs_nonglint)
print(f"Observed LC: {n_obs} non-glint peaks (prom>{PROM}, mag>{MAG_THRESHOLD})")

# Load 10 geo-refined candidates
ckpt_geo = np.load(str(RESULTS_DIR / "m072_pipeline_seed093" / "geo_refined.npz"))
geo_costs = ckpt_geo['geo_costs']
q0s_ref = ckpt_geo['q0s_ref']
w0s_ref = ckpt_geo['w0s_ref']

# Module-level refs for multiprocessing
_satellite = CTX.satellite
_obs_times = obs_times
_obs_lc = observed_lc
_sun = CTX.sun_pos
_obs = CTX.obs_pos
_sat = CTX.sat_pos
_dist = CTX.obs_dist
_art = CTX.art_matrices
_I = I_tensor


# ══════════════════════════════════════════════════════════════════════
# PARALLEL HI-FI EVALUATION
# ══════════════════════════════════════════════════════════════════════

def eval_one(args):
    idx, q0_wxyz, w0_rad = args
    obj = ObjectiveFunction(
        satellite=_satellite, observation_times=_obs_times,
        observed_lightcurve=_obs_lc,
        sun_positions_j2000=_sun, observer_positions_j2000=_obs,
        satellite_positions_j2000=_sat, observer_distances=_dist,
        compute_shadows_flag=True, articulation_matrices=_art,
        mode="tumbling", inertia_tensor=_I, show_progress=False)

    # Propagate
    quats, _ = propagate_attitude(q0_wxyz, w0_rad, _obs_times, "tumbling", _I)
    k1, k2 = obj._compute_body_frame_vectors(quats)

    from src.computation.shadow_engine import compute_shadows
    from src.computation.lightcurve_generator import generate_lightcurves
    lit = compute_shadows(satellite=_satellite, k1_vectors=k1,
                          explicit_component_matrices=_art, show_progress=False)
    cand_mags, _, _, _, _, _ = generate_lightcurves(
        facet_lit_status_dict=lit, k1_vectors_array=k1,
        k2_vectors_array=k2, observer_distances=_dist,
        satellite=_satellite, epochs=np.arange(len(_obs_times), dtype=float),
        pre_computed_matrices=_art, show_progress=False)

    hifi_res = float(np.mean((cand_mags - _obs_lc) ** 2))

    cand_peaks, _ = find_peaks(-cand_mags, prominence=PROM)
    cand_nonglint = cand_peaks[cand_mags[cand_peaks] > MAG_THRESHOLD]
    n_ng = len(cand_nonglint)

    return idx, hifi_res, n_ng, cand_mags


print(f"\nRunning hi-fi for 10 candidates on {HIFI_WORKERS} cores...", flush=True)
t0 = time.time()
eval_args = [(i, q0s_ref[i].copy(), w0s_ref[i].copy()) for i in range(10)]
with Pool(HIFI_WORKERS) as pool:
    results = pool.map(eval_one, eval_args)
print(f"Done in {time.time() - t0:.1f}s\n")


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════

rows = []
for idx, hifi_res, n_ng, cand_mags in results:
    q0_err = attitude_error_deg(q0s_ref[idx], true_q0)
    w_err = omega_dir_err(w0s_ref[idx], true_omega0)
    rows.append({
        'idx': idx, 'q0_err': q0_err, 'w_err': w_err,
        'geo': geo_costs[idx], 'hifi_res': hifi_res, 'n_ng': n_ng,
    })

print(f"{'#':>3} {'q0_err':>7} {'w_err':>7} {'geo':>10} {'hifi':>10} {'n_ng':>5}  n_obs={n_obs}")
print("-" * 60)
for r in sorted(rows, key=lambda x: x['hifi_res']):
    tag = " <--" if r['w_err'] < 5 else ""
    print(f"{r['idx']+1:3d} {r['q0_err']:7.1f} {r['w_err']:7.1f} "
          f"{r['geo']:10.6f} {r['hifi_res']:10.4f} {r['n_ng']:5d}{tag}")

print()
print("=" * 70)
print("Composite score: hifi_res * (1 + alpha * |n_ng - n_obs| / n_obs)")
print("=" * 70)

for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
    scored = []
    for r in rows:
        penalty = 1.0 + alpha * abs(r['n_ng'] - n_obs) / n_obs
        score = r['hifi_res'] * penalty
        scored.append({**r, 'penalty': penalty, 'score': score})
    scored.sort(key=lambda x: x['score'])

    w = scored[0]
    tag = "CORRECT" if w['w_err'] < 5 else "WRONG"
    print(f"\nalpha={alpha:.1f} -> winner: #{w['idx']+1} q0={w['q0_err']:.1f} w={w['w_err']:.1f} [{tag}]")
    for rank, s in enumerate(scored[:5]):
        m = " <--" if s['w_err'] < 5 else ""
        print(f"  #{rank+1}: cand{s['idx']+1} hifi={s['hifi_res']:.4f} "
              f"n_ng={s['n_ng']} pen={s['penalty']:.2f} "
              f"score={s['score']:.4f} | q0={s['q0_err']:.1f} w={s['w_err']:.1f}{m}")
