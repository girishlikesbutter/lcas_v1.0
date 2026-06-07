"""Trajectory source abstraction for the inversion pipeline.

Unifies loading of ground-truth states + observed light curves from either:
  - m046_trajectories.npz   (100 seeds, single fixed observation window
                             2020-02-05 10:00-11:00 UTC)
  - m048_trajectories       (100 seeds, per-seed random start times,
                             each a 1-hour window, phase angle 9°-95°)

Consumers call `load_truth(seed, source)` and get a uniform dict. Pass the
resulting `start_et` + `end_time_utc` through to
`lib.experiment_setup.setup_experiment` as kwargs — back-compat for m046 is
preserved because the m046 branch returns (None, '2020-02-05T11:00:00').

`observed_lc` is the canonical noisy light curve every pipeline stage must
consume. Computed here once as `mag_hifi + default_rng(noise_seed).normal(
0, noise_sigma, N)` so m103/m115/m126 cannot drift to different noise
realisations (previously m126 used legacy np.random.seed which yielded
different noise than default_rng with the same seed).
"""

from pathlib import Path
from typing import Optional, Dict

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"

M046_PATH = RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"
M048_DIR = RESULTS_DIR / "m048_trajectories"
M048_PER_TRAJ_DIR = M048_DIR / "per_trajectory"


VALID_SOURCES = ("m046", "m048")

CANONICAL_NOISE_SIGMA = 0.05
CANONICAL_NOISE_SEED = 42


def canonical_observed_lc(
    mag_hifi: NDArray,
    noise_seed: int = CANONICAL_NOISE_SEED,
    noise_sigma: float = CANONICAL_NOISE_SIGMA,
) -> NDArray:
    """Single source of truth for the noisy observed LC.

    Uses the new-style Generator API (np.random.default_rng) so all stages
    see bit-identical noise for the same seed. Pre-fix, m126 used legacy
    np.random.seed which produced different numbers from default_rng for
    the same seed — this caused m115 and m126 to fit slightly different
    noise realisations of the same truth.
    """
    mag_hifi = np.asarray(mag_hifi, dtype=np.float64)
    rng = np.random.default_rng(noise_seed)
    return mag_hifi + rng.normal(0.0, noise_sigma, size=mag_hifi.shape[0])


def load_truth(
    seed: int,
    source: str = "m046",
    noise_seed: int = CANONICAL_NOISE_SEED,
    noise_sigma: float = CANONICAL_NOISE_SIGMA,
) -> Dict:
    """Load ground-truth state + truth LC + canonical observed LC for one seed.

    Returns a dict with:
        q0_wxyz          (4,)          scalar-first unit quaternion
        omega0_rad       (3,)          body-frame angular velocity rad/s
        mag_hifi         (500,)        truth light curve (noise-free)
        observed_lc      (500,)        canonical noisy LC (consume everywhere)
        observation_times (500,)       seconds relative to epoch start
        inertia_tensor   (3, 3)
        start_et         Optional[float]   only set for m048
        end_time_utc     Optional[str]     only set for m046
        source           str               'm046' or 'm048'
        seed             int
        noise_seed       int
        noise_sigma      float

    For m046: start_et=None, end_time_utc='2020-02-05T11:00:00'.
    For m048: start_et=<per-seed>, end_time_utc=None, duration_s=3600.0.

    Pass `start_et` / `end_time_utc` into setup_experiment — only ONE of the
    two is non-None, and setup_experiment accepts both kwargs.
    """
    if source not in VALID_SOURCES:
        raise ValueError(
            f"source must be one of {VALID_SOURCES}; got {source!r}")

    truth = _load_m046(seed) if source == "m046" else _load_m048(seed)
    truth["observed_lc"] = canonical_observed_lc(
        truth["mag_hifi"], noise_seed=noise_seed, noise_sigma=noise_sigma)
    truth["noise_seed"] = int(noise_seed)
    truth["noise_sigma"] = float(noise_sigma)
    return truth


def _load_m046(seed: int) -> Dict:
    m = np.load(str(M046_PATH), allow_pickle=True)
    if seed < 0 or seed >= int(m["n_trajectories"]):
        raise ValueError(
            f"seed {seed} out of range for m046 (n={int(m['n_trajectories'])})")
    return {
        "q0_wxyz": m["q0s"][seed].astype(np.float64).copy(),
        "omega0_rad": m["omega0s"][seed].astype(np.float64).copy(),
        "mag_hifi": m["mag_hifi"][seed].astype(np.float64).copy(),
        "observation_times": m["observation_times"].astype(np.float64).copy(),
        "inertia_tensor": m["inertia_tensor"].astype(np.float64).copy(),
        "start_et": None,
        "end_time_utc": "2020-02-05T11:00:00",
        "duration_s": 3600.0,
        "source": "m046",
        "seed": int(seed),
    }


def _load_m048(seed: int) -> Dict:
    path = M048_PER_TRAJ_DIR / f"traj_seed{seed:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"m048 per-trajectory file missing for seed {seed}: {path}")
    z = np.load(str(path), allow_pickle=True)
    # inertia_tensor lives in the top-level m048_trajectories.npz; the
    # per-trajectory files don't carry a copy. Load once (cheap — 3×3 scalar).
    shared = np.load(str(M048_DIR / "m048_trajectories.npz"), allow_pickle=True)
    return {
        "q0_wxyz": z["q0_wxyz"].astype(np.float64).copy(),
        "omega0_rad": z["omega0_rad"].astype(np.float64).copy(),
        "mag_hifi": z["mag_hifi"].astype(np.float64).copy(),
        "observation_times": z["observation_times"].astype(np.float64).copy(),
        "inertia_tensor": shared["inertia_tensor"].astype(np.float64).copy(),
        "start_et": float(z["start_et"]),
        "end_time_utc": None,
        "duration_s": 3600.0,
        "source": "m048",
        "seed": int(seed),
    }


def seeds_for_source(source: str) -> NDArray:
    """Return available seed indices for a given source."""
    if source == "m046":
        m = np.load(str(M046_PATH), allow_pickle=True)
        return m["seeds"].astype(np.int64).copy()
    if source == "m048":
        shared = np.load(str(M048_DIR / "m048_trajectories.npz"),
                         allow_pickle=True)
        return shared["seeds"].astype(np.int64).copy()
    raise ValueError(f"source must be one of {VALID_SOURCES}")
