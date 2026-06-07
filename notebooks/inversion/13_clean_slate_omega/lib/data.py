"""Shared data loading for the 13_clean_slate_omega series.

All three sub-experiments (a_spectral, b_differentiable, c_learned_inverse)
pull trajectories through this module so there's one source of truth for
the {truth, observed LC, inertial geometry, inertia tensor} bundle per seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAJ_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m048_trajectories" / "per_trajectory"
SHARED_NPZ = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m048_trajectories" / "m048_trajectories.npz"


def load_seed(seed: int) -> Dict:
    """Return everything needed to (a) observe and (b) forward-simulate seed `seed`.

    Keys:
        seed, q0_true (4,), omega0_true (3,), inertia_tensor (3,3),
        observation_times (500,), mag_hifi (500,),
        sun_j2k (500,3), obs_j2k (500,3), sat_j2k (500,3), obs_dist (500,),
        k1_body_true (500,3), k2_body_true (500,3), pab_body_true (500,3),
        omega_mag_dps (scalar), omega_theta_deg (scalar), omega_phi_deg (scalar),
        start_et, dt_sampling, duration_s.
    """
    per = np.load(TRAJ_DIR / f"traj_seed{seed:03d}.npz", allow_pickle=True)
    shared = np.load(SHARED_NPZ, allow_pickle=True)
    I = np.asarray(shared["inertia_tensor"], dtype=np.float64)

    return {
        "seed": int(per["seed"]),
        "q0_true": np.asarray(per["q0_wxyz"], dtype=np.float64),
        "omega0_true": np.asarray(per["omega0_rad"], dtype=np.float64),
        "inertia_tensor": I,
        "observation_times": np.asarray(per["observation_times"], dtype=np.float64),
        "mag_hifi": np.asarray(per["mag_hifi"], dtype=np.float64),
        "sun_j2k": np.asarray(per["sun_pos"], dtype=np.float64),
        "obs_j2k": np.asarray(per["obs_pos"], dtype=np.float64),
        "sat_j2k": np.asarray(per["sat_pos"], dtype=np.float64),
        "obs_dist": np.asarray(per["obs_dist"], dtype=np.float64),
        "k1_body_true": np.asarray(per["k1_body"], dtype=np.float64),
        "k2_body_true": np.asarray(per["k2_body"], dtype=np.float64),
        "pab_body_true": np.asarray(per["pab_body"], dtype=np.float64),
        "omega_mag_dps": float(per["omega_mag_dps"]),
        "omega_theta_deg": float(per["omega_theta_deg"]),
        "omega_phi_deg": float(per["omega_phi_deg"]),
        "start_et": float(per["start_et"]),
        "dt_sampling": float(per["dt_sampling"]),
        "duration_s": float(per["dt_sampling"]) * (len(per["observation_times"]) - 1),
        "phase_angle_3d": np.asarray(per["phase_angle_3d"], dtype=np.float64),
    }


def all_seeds() -> List[int]:
    return sorted(int(p.name[len("traj_seed"):-len(".npz")]) for p in TRAJ_DIR.glob("traj_seed*.npz"))
