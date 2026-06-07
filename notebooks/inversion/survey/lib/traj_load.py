"""Trajectory loader for the survey workspace.

Loads canonical post-fix m048 trajectory NPZs from the symlinked
`data/trajectories/` directory. The buggy `_buggy.npz` siblings (pre-2026-04-30
conjugation bug) and the `_s066buggy_archive/` siblings (pre-2026-05-12
q-kinematic-sign bug) are NOT exposed by this module by design — they live
in the parent project and should only be loaded with explicit absolute paths
if a pre-fix-vs-post-fix comparison is the actual goal.

"Post-fix" in this module means: post both the 2026-04-30 conjugation fix
(commit f7fabbe) AND the 2026-05-12 q-kinematic-sign fix (commit d5705ff).
The current cohort was regenerated 2026-05-12 against both fixes.

Each NPZ contains the cached forward-model outputs at truth (q0, ω0):
    seed, q0_wxyz, omega0_rad, omega_mag_dps, observation_times,
    quaternions, sun_pos, obs_pos, sat_pos, obs_dist,
    k1_body, k2_body, pab_body, phase_angle_3d,
    mag_hifi, mag_lofi, hifi_peak_epochs, ...

Because k1_body and k2_body are cached at truth, scoring any cost surface
"at truth" is a pure file-load — no propagation needed.
"""

from pathlib import Path
import numpy as np

SURVEY_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "trajectories"


def load_truth(seed: int) -> dict:
    """Load post-fix truth NPZ for an m048 seed.

    Returns a dict with all NPZ fields. Caller picks what they need.
    """
    path = SURVEY_DATA_DIR / f"traj_seed{seed:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No post-fix trajectory NPZ for seed {seed} at {path}")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def list_seeds() -> list[int]:
    """All m048 seeds available in the survey workspace."""
    return sorted(
        int(p.stem.replace("traj_seed", ""))
        for p in SURVEY_DATA_DIR.glob("traj_seed[0-9][0-9][0-9].npz")
    )


def truth_state(seed: int) -> dict:
    """Compact accessor for the most-used fields.

    Returns:
      q0_wxyz: (4,) truth quaternion in (w, x, y, z) order
      omega0_rad: (3,) truth body-frame angular velocity in rad/s
      observation_times: (N,) ET seconds
      mag_hifi: (N,) cached hi-fi magnitude at truth
      mag_lofi: (N,) cached lo-fi magnitude at truth
      k1_body: (N, 3) cached truth body-frame sun direction (unit)
      k2_body: (N, 3) cached truth body-frame observer direction (unit)
      obs_dist: (N,) observer distance in km
      phase_angle_3d: (N,) phase angle in degrees
    """
    d = load_truth(seed)
    return {
        "q0_wxyz": d["q0_wxyz"],
        "omega0_rad": d["omega0_rad"],
        "omega_mag_dps": float(d["omega_mag_dps"]),
        "observation_times": d["observation_times"],
        "mag_hifi": d["mag_hifi"],
        "mag_lofi": d["mag_lofi"],
        "k1_body": d["k1_body"],
        "k2_body": d["k2_body"],
        "obs_dist": d["obs_dist"],
        "phase_angle_3d": d["phase_angle_3d"],
    }
