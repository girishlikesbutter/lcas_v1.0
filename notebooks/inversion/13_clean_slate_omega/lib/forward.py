"""Shared NumPy forward model for the 13_clean_slate_omega series.

Given (q0, omega, inertia, epoch geometry) produce a predicted magnitude
light curve via the v2 surrogate. This is the scoring forward used by
Ideas 1 (spectral ω_dir sweep), 2 (diff inversion warm-start candidate
checks), and 3 (learned-inverse candidate validation).

For differentiable inversion, a separate PyTorch forward lives in
b_differentiable/torch_forward.py (mirrors this math, backprops through v2).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import sys

sys.path.insert(0, str(Path.home() / "surrogate_model" / "surrogate_model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from surrogate import SurrogateModel  # noqa: E402
from src.dynamics.attitude_propagator import propagate_attitude  # noqa: E402


_MODEL_CACHE: dict = {}


def get_surrogate(variant: str = "v2") -> SurrogateModel:
    """Cached surrogate loader. variant='v2' (default) or 'v1' for legacy.

    v1 loads the top-level ~/surrogate_model/s10_5M_*.npz files; v2 loads
    the nested ensemble. See memory `project_surrogate_model.md`.
    """
    if variant in _MODEL_CACHE:
        return _MODEL_CACHE[variant]
    if variant == "v2":
        m = SurrogateModel.load_default()
    elif variant == "v1":
        root = Path.home() / "surrogate_model"
        m = SurrogateModel(
            weights_paths=[str(root / "s10_5M_weights.npz")],
            normalization_path=str(root / "s10_5M_normalization.npz"),
            geometry_path=str(root / "surrogate_model" / "s11_geometry.npz"),
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    _MODEL_CACHE[variant] = m
    return m


def body_vectors_from_attitude(
    quaternions: np.ndarray,
    sun_j2k: np.ndarray,
    obs_j2k: np.ndarray,
    sat_j2k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate J2000 sun/obs vectors into the body frame using q_body_from_inertial.

    Scalar-first quaternion convention (w, x, y, z). Matches the convention in
    src/inversion/objective_function.py and src/dynamics/attitude_propagator.py.

    Returns (k1_body, k2_body) as unit vectors.
    """
    n = len(quaternions)
    k1_body = np.empty((n, 3))
    k2_body = np.empty((n, 3))

    sun_vec_j2k = sun_j2k - sat_j2k
    obs_vec_j2k = obs_j2k - sat_j2k

    for i in range(n):
        w, x, y, z = quaternions[i]
        # This matrix maps inertial-frame vectors to body-frame components.
        # Empirically verified against m048 seed 0: k1_body = R_ib @ (sun_j2k - sat_j2k).
        # (The propagator + LCAS pipeline treat the quaternion as the inertial->body rotation.)
        R_ib = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
        k1 = R_ib @ sun_vec_j2k[i]
        k2 = R_ib @ obs_vec_j2k[i]
        k1_body[i] = k1 / (np.linalg.norm(k1) + 1e-30)
        k2_body[i] = k2 / (np.linalg.norm(k2) + 1e-30)
    return k1_body, k2_body


def predict_lc(
    q0: np.ndarray,
    omega0: np.ndarray,
    inertia_tensor: np.ndarray,
    observation_times: np.ndarray,
    sun_j2k: np.ndarray,
    obs_j2k: np.ndarray,
    sat_j2k: np.ndarray,
    obs_dist: np.ndarray,
    panel_deg: float = 0.0,
    dish_deg: float = 15.0,
    surrogate_variant: str = "v2",
    mode: str = "tumbling",
) -> np.ndarray:
    """Predicted magnitude light curve for a candidate (q0, omega0) on this seed's geometry.

    Times must be strictly increasing; `observation_times[0]` is the epoch at
    which q0/omega0 apply. Returns (N,) array of magnitudes.
    """
    model = get_surrogate(surrogate_variant)
    # Propagate attitude; supply relative times (propagator expects t=0 at q0).
    t_rel = observation_times - observation_times[0]
    quats, _ = propagate_attitude(
        q0=q0.astype(np.float64),
        omega0=omega0.astype(np.float64),
        times=t_rel.astype(np.float64),
        mode=mode,
        inertia_tensor=inertia_tensor.astype(np.float64) if inertia_tensor is not None else None,
    )
    k1_body, k2_body = body_vectors_from_attitude(quats, sun_j2k, obs_j2k, sat_j2k)
    mag = model.predict_magnitude(
        k1_body, k2_body, panel_deg, dish_deg, obs_dist,
    )
    return np.asarray(mag, dtype=np.float64)
