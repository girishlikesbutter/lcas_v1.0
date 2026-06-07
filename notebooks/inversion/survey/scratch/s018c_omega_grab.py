"""Diagnostic: how far from truth-ω does s018c phi-sweep + LM still
converge to the truth basin?

Sweep ω perturbation along independent axes:
  - ω-dir at fixed truth-mag: 0°, 5°, 10°, 20°, 40°
  - ω-mag at fixed truth-dir: -50%, -20%, -10%, 0%, +10%, +20%, +50%

For each cell, evaluate phi-sweep ICs at that cell, take top-32 by
surrogate-MSE, LM polish. Record best q0_err_deg + final_surr_MSE.

Decisive output: at what dir / mag perturbation does the LM-best q0_err
exceed 5° (basin lost)?
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

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURVEY_DIR / "experiments"))

import s018c_phi_sweep_pilot as s018c


def main():
    seed = 6
    out_dir = SURVEY_DIR / "results" / "s018c_omega_grab_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load truth
    d = np.load(SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz")
    truth_omega = d['omega0_rad'].astype(float)
    truth_mag = float(np.linalg.norm(truth_omega))
    truth_dir = truth_omega / truth_mag
    print(f"Truth omega: |w|={truth_mag:.6f} rad/s = {np.degrees(truth_mag):.4f} dps; "
          f"dir={truth_dir}")

    master = np.load(s018c.M048_MASTER, allow_pickle=True)
    inertia = np.asarray(master['inertia_tensor'], dtype=float)

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

    # Override knobs
    s018c.N_PHI = 12
    s018c.TOP_K_LM = 32
    s018c.MAX_NFEV = 60

    # Build perturbation cells
    def perturb_dir(deg, axis_seed=0):
        rng = np.random.default_rng(axis_seed)
        if abs(deg) < 1e-9:
            return truth_dir.copy()
        # rotate around an arbitrary perpendicular axis
        ortho = np.array([1.0, 0.0, 0.0]) if abs(truth_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(truth_dir, ortho)
        axis /= np.linalg.norm(axis)
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(np.radians(deg) * axis).as_matrix() @ truth_dir

    cells = []
    for d_deg in [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 60.0]:
        new_dir = perturb_dir(d_deg)
        cells.append({
            'label': f"dir+{d_deg:g}deg_mag0%",
            'omega': new_dir * truth_mag,
            'd_deg': d_deg, 'd_mag_pct': 0.0,
        })
    for d_pct in [-50.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 50.0]:
        scale = 1.0 + d_pct / 100.0
        cells.append({
            'label': f"dir0deg_mag{d_pct:+g}%",
            'omega': truth_dir * truth_mag * scale,
            'd_deg': 0.0, 'd_mag_pct': d_pct,
        })

    print(f"\nTotal cells: {len(cells)}")
    print(f"Each cell: phi-sweep ICs + LM polish")
    print()

    results = []
    t_total = time.time()
    for cell in cells:
        omega_cell = cell['omega']
        # Build IC pool at this single cell
        traj_local = dict(traj)
        traj_local['inertia_tensor'] = inertia
        s018c.build_omega_grid = lambda omega=omega_cell: omega.reshape(1, 3)
        ics, meta = s018c.build_phi_sweep_ics_for_seed(seed, traj_local,
                                                       omega_cell.reshape(1, 3))
        if not ics:
            print(f"  {cell['label']:30s} | no ICs generated")
            continue

        # Score
        score_args = [(seed, ic['q0_wxyz'], ic['omega_rad']) for ic in ics]
        with Pool(processes=8, initializer=s018c.init_worker,
                  initargs=(seed_data, inertia)) as pool:
            scored = pool.map(s018c.score_ic_worker, score_args, chunksize=32)
        scored = np.asarray(scored, dtype=float)

        # Top-K
        finite = np.where(np.isfinite(scored))[0]
        order = finite[np.argsort(scored[finite])]
        topk = order[:s018c.TOP_K_LM]

        # LM
        lm_args = []
        for idx in topk:
            ic = ics[idx]
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
        cell_result = {
            **cell,
            'omega': cell['omega'].tolist(),
            'best_final_mse': float(finals[i_best]),
            'best_q0_err_deg': float(q0_errs[i_best]),
            'best_omega_dir_err_deg': float(wd_errs[i_best]),
            'best_omega_mag_err_pct': float(wm_errs[i_best]),
            'min_q0_err_deg': float(q0_errs.min()),
        }
        results.append(cell_result)
        print(f"  {cell['label']:30s} | n_ics={len(ics):4d} | "
              f"best_q0_err={q0_errs[i_best]:7.3f}° | "
              f"min_q0_err={q0_errs.min():7.3f}° | "
              f"best_mse={finals[i_best]:.4e}",
              flush=True)

    print(f"\nTotal wall: {time.time()-t_total:.1f}s")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({'seed': seed, 'truth_omega_dps': float(np.degrees(truth_mag)),
                   'truth_dir': truth_dir.tolist(),
                   'cells': results}, f, indent=2)
    print(f"Saved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
