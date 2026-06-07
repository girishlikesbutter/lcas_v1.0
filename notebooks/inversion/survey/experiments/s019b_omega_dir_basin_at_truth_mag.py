"""s019b — ω-direction basin width at pinned truth-ω-mag, 3 seeds.

Extends the s018c ω-grab harness (`scratch/s018c_omega_grab.py`) by
sweeping ω-direction at FINE perturbations {2°, 3°, 4°} at fixed
truth-ω-mag, on 3 seeds with classifiable peaks. The s018c
ω-grab measured the boundary at "between 1° and 5°" on seed 6 only —
this localises the boundary and tests cohort variation.

Pre-registered question:
  At what ω-direction perturbation does the LM-best q0 leave the basin
  (q0_err < 5°)? Does it vary across seeds?

Pre-registered prediction:
  ω-dir basin is between 2°-4° (consistent with s003's "1° tube"
  finding plus LM grab widening). Cohort variation at most 2× — i.e.,
  basin radius 1.5°-3° depending on seed.

Pipeline (per cell):
  - phi-sweep IC pool at this ω cell (truth-mag × perturbed dir).
  - surrogate-MSE score all ICs.
  - top-32 LM polish (max_nfev=60) on Pool(8).
  - record best q0_err, min q0_err, final_surr_MSE.

Wall budget: 3 seeds × 3 dir cells × ~3 min/cell = ~27 min (one Pool
init per cell — could be amortised but it's clean to keep cells
independent).

Pilot seeds (revised by tier coverage; same as s018c pilot tail):
  6   — control, 2 tiers, 4 classifiable peaks (T2:2, T3:2)
  28  — sub-Sobol-narrow, 3 tiers, 7 classifiable peaks
  44  — best-case, 4 tiers, 15 classifiable peaks
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR / "experiments"))

import s018c_phi_sweep_pilot as s018c

OUT_DIR = SURVEY_DIR / "results" / "s019b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PILOT_SEEDS = [6, 28, 44]
DIR_PERTURBATIONS_DEG = [2.0, 3.0, 4.0]


def perturb_dir(truth_dir, deg):
    if abs(deg) < 1e-9:
        return truth_dir.copy()
    ortho = (np.array([1.0, 0.0, 0.0]) if abs(truth_dir[0]) < 0.9
             else np.array([0.0, 1.0, 0.0]))
    axis = np.cross(truth_dir, ortho)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(np.radians(deg) * axis).as_matrix() @ truth_dir


def run_cell(seed, omega_cell, traj, seed_data, inertia):
    """Build phi-sweep ICs at this single ω-cell, surrogate-score, LM-polish top-32."""
    s018c.build_omega_grid = lambda omega=omega_cell: omega.reshape(1, 3)
    ics, meta = s018c.build_phi_sweep_ics_for_seed(
        seed, traj, omega_cell.reshape(1, 3),
    )
    if not ics:
        return None

    score_args = [(seed, ic['q0_wxyz'], ic['omega_rad']) for ic in ics]
    with Pool(processes=8, initializer=s018c.init_worker,
              initargs=(seed_data, inertia)) as pool:
        scored = pool.map(s018c.score_ic_worker, score_args, chunksize=32)
    scored = np.asarray(scored, dtype=float)
    finite = np.where(np.isfinite(scored))[0]
    if len(finite) == 0:
        return None
    order = finite[np.argsort(scored[finite])]
    topk = order[:s018c.TOP_K_LM]

    lm_args = []
    for idx in topk:
        ic = ics[int(idx)]
        lm_args.append((seed, ic['q0_wxyz'], ic['omega_rad'],
                        {'pool_idx': int(idx),
                         'peak_idx': ic['peak_idx'],
                         'peak_epoch': ic['peak_epoch'],
                         'face_idx': ic['face_idx'],
                         'phi_rad': ic['phi_rad'],
                         'tier': ic['tier']}))
    with Pool(processes=8, initializer=s018c.init_worker,
              initargs=(seed_data, inertia)) as pool:
        lm_results = pool.map(s018c.lm_polish_worker, lm_args, chunksize=4)

    finals = np.array([r['final_mse'] for r in lm_results])
    q0_errs = np.array([r['q0_err_deg'] for r in lm_results])
    wd_errs = np.array([r['omega_dir_err_deg'] for r in lm_results])
    wm_errs = np.array([r['omega_mag_err_pct'] for r in lm_results])
    i_best = int(np.argmin(finals))
    return {
        'best_final_mse': float(finals[i_best]),
        'best_q0_err_deg': float(q0_errs[i_best]),
        'best_omega_dir_err_deg': float(wd_errs[i_best]),
        'best_omega_mag_err_pct': float(wm_errs[i_best]),
        'min_q0_err_deg': float(q0_errs.min()),
        'n_ics': len(ics),
        'n_in_basin': int((q0_errs < 5).sum()),
    }


def main():
    print(f"=== s019b — ω-dir basin width at pinned truth-mag ===", flush=True)
    print(f"Pilot seeds: {PILOT_SEEDS}", flush=True)
    print(f"ω-dir perturbations (degrees from truth, mag at truth): "
          f"{DIR_PERTURBATIONS_DEG}", flush=True)

    # Override knobs for the cells
    s018c.N_PHI = 12
    s018c.TOP_K_LM = 32
    s018c.MAX_NFEV = 60

    master = np.load(s018c.M048_MASTER, allow_pickle=True)
    inertia = np.asarray(master['inertia_tensor'], dtype=float)

    all_results = {'pilot_seeds': PILOT_SEEDS,
                   'dir_perturbations_deg': DIR_PERTURBATIONS_DEG,
                   'per_seed': {}}
    t_total = time.time()
    for seed in PILOT_SEEDS:
        d = np.load(SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz")
        truth_omega = d['omega0_rad'].astype(float)
        truth_mag = float(np.linalg.norm(truth_omega))
        truth_dir = truth_omega / truth_mag
        traj = {
            'observation_times': d['observation_times'].astype(float),
            'sun_pos':           d['sun_pos'].astype(float),
            'obs_pos':           d['obs_pos'].astype(float),
            'sat_pos':           d['sat_pos'].astype(float),
            'obs_dist':          d['obs_dist'].astype(float),
            'mag_hifi':          d['mag_hifi'].astype(float),
            'pab_j2000':         d['pab_j2000'].astype(float),
            'hifi_peak_epochs':  d['hifi_peak_epochs'],
            'q0_wxyz':           d['q0_wxyz'].astype(float),
            'omega0_rad':        d['omega0_rad'].astype(float),
            'inertia_tensor':    inertia,
        }
        seed_data = {seed: traj}

        print(f"\n--- seed {seed} (truth |ω|={np.degrees(truth_mag):.4f} dps) ---",
              flush=True)
        seed_results = []
        for d_deg in DIR_PERTURBATIONS_DEG:
            new_dir = perturb_dir(truth_dir, d_deg)
            omega_cell = new_dir * truth_mag
            t_cell = time.time()
            r = run_cell(seed, omega_cell, traj, seed_data, inertia)
            wall = time.time() - t_cell
            if r is None:
                print(f"  d_dir={d_deg}°  (no ICs)", flush=True)
                seed_results.append({'d_dir_deg': d_deg, 'no_ics': True,
                                     'wall_s': wall})
                continue
            print(f"  d_dir={d_deg}°  | n_ics={r['n_ics']:4d} | "
                  f"best_q0_err={r['best_q0_err_deg']:7.3f}° | "
                  f"min_q0_err={r['min_q0_err_deg']:7.3f}° | "
                  f"n_in_basin={r['n_in_basin']:2d}/{s018c.TOP_K_LM} | "
                  f"best_mse={r['best_final_mse']:.3e} | "
                  f"wall={wall:.0f}s",
                  flush=True)
            r['d_dir_deg'] = d_deg
            r['wall_s'] = wall
            seed_results.append(r)

        all_results['per_seed'][seed] = seed_results

    print(f"\nTotal wall: {time.time()-t_total:.1f}s", flush=True)
    out = OUT_DIR / "summary.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
