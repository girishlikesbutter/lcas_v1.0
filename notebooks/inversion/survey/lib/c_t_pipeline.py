"""C_t pipeline primitives — per-epoch q-cloud projection + survival.

Factored from `experiments/s048b_per_epoch_spread_v1.py:106-150`. The
operations here are general enough to be reused by any C_t-style
experiment (single-epoch survivors, joint multi-epoch consistency
filters, animation viewers).

Pipeline at a glance:

    pool = sample_so3_pool(n_samples, sample_seed)
    sun_unit, obs_unit = compute_j2000_units(sun_pos, obs_pos, sat_pos)
    for ep in epoch_indices:
        k1, k2 = project_directions(pool['R_cache'], sun_unit[ep], obs_unit[ep])
        pred, keep = survive_at_epoch(model, k1, k2, obs_dist[ep], sp, ad,
                                       mag_measured[ep], tol)

    closest_idx, closest_deg = nearest_in_pool_to_truths(pool['q_pool_wxyz'],
                                                          q_truth_at)

The surrogate object passed to `survive_at_epoch` only needs to expose
`predict_magnitude(k1, k2, sp_deg, ad_deg, obs_dist_km) → mag`. Both v1
and v2 surrogates satisfy this.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def sample_so3_pool(n_samples: int, sample_seed: int = 42) -> dict:
    """Uniform Haar-measure samples on SO(3).

    Returns a dict with:
      R_cache       (N, 3, 3) float64 — body→inertial rotation matrices
      q_pool_wxyz   (N, 4)    float64 — quaternions, scalar-first
      q_pool_xyzw   (N, 4)    float64 — quaternions, scalar-last (scipy native)
      rotvec_pool   (N, 3)    float32 — rotvec coords, |.|≤π
    """
    rng = np.random.default_rng(int(sample_seed))
    R = Rotation.random(int(n_samples), random_state=rng)
    q_xyzw = R.as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    return {
        "R_cache": R.as_matrix(),
        "q_pool_wxyz": q_wxyz,
        "q_pool_xyzw": q_xyzw,
        "rotvec_pool": R.as_rotvec().astype(np.float32),
    }


def compute_j2000_units(
    sun_pos: np.ndarray, obs_pos: np.ndarray, sat_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sun and observer unit vectors in the J2000 inertial frame, per epoch.

    All inputs are (n_obs, 3) J2000 positions in km. Standard pattern from
    `s048b_per_epoch_spread_v1.py:80-83`.
    """
    sun_vec = np.asarray(sun_pos, float) - np.asarray(sat_pos, float)
    obs_vec = np.asarray(obs_pos, float) - np.asarray(sat_pos, float)
    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_unit = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)
    return sun_unit, obs_unit


def project_directions(
    R_cache: np.ndarray, sun_unit_ep: np.ndarray, obs_unit_ep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate one epoch's J2000 sun/observer unit vectors into body frame
    for every candidate q in the pool.

    R_cache shape (N, 3, 3), sun_unit_ep / obs_unit_ep shape (3,).
    Returns k1_body, k2_body each shape (N, 3).
    """
    k1_body = np.einsum("nij,j->ni", R_cache, sun_unit_ep)
    k2_body = np.einsum("nij,j->ni", R_cache, obs_unit_ep)
    return k1_body, k2_body


def survive_at_epoch(
    model,
    k1_body: np.ndarray,
    k2_body: np.ndarray,
    obs_dist_ep: float,
    sp_deg: float,
    ad_deg: float,
    measured_mag: float,
    tolerance_mag: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate prediction and survival mask at one epoch.

    Returns:
      pred (N,) float32 — predicted magnitude per candidate
      keep (N,) bool    — True where |pred - measured| < tolerance_mag
    """
    n = k1_body.shape[0]
    obs_dist_arr = np.full(n, float(obs_dist_ep))
    pred = model.predict_magnitude(
        k1_body, k2_body, float(sp_deg), float(ad_deg), obs_dist_arr
    )
    keep = np.abs(pred - float(measured_mag)) < float(tolerance_mag)
    return pred.astype(np.float32, copy=False), keep


def nearest_in_pool_to_truth(
    q_pool_wxyz: np.ndarray, q_truth_wxyz: np.ndarray
) -> tuple[float, int]:
    """Geodesic distance (deg) and pool index of the candidate closest to
    a single truth quaternion. Antipodal-aware via |dot|.
    """
    dots = np.abs(q_pool_wxyz @ np.asarray(q_truth_wxyz, float))
    idx = int(np.argmax(dots))
    deg = float(np.degrees(2.0 * np.arccos(np.clip(dots[idx], 0.0, 1.0))))
    return deg, idx


def nearest_in_pool_to_truths(
    q_pool_wxyz: np.ndarray, q_truths_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batched: for every truth quaternion, return the closest pool index
    and geodesic distance in degrees.

    q_pool_wxyz shape (N, 4); q_truths_wxyz shape (n_ep, 4).
    Returns (closest_deg (n_ep,), closest_idx (n_ep,)).
    """
    dots = np.abs(q_pool_wxyz @ np.asarray(q_truths_wxyz, float).T)  # (N, n_ep)
    closest_idx = np.argmax(dots, axis=0).astype(np.int64)
    max_dots = dots[closest_idx, np.arange(dots.shape[1])]
    closest_deg = np.degrees(2.0 * np.arccos(np.clip(max_dots, 0.0, 1.0)))
    return closest_deg, closest_idx


def quats_to_rotvec_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert (M, 4) wxyz quaternions to (M, 3) rotation vectors. Float32."""
    q_xyzw = np.asarray(q_wxyz, float)[:, [1, 2, 3, 0]]
    return Rotation.from_quat(q_xyzw).as_rotvec().astype(np.float32)


# --------------------------------------------------------------------------
# Rotvec continuity helpers
#
# The canonical rotvec chart `|r| ≤ π` has a discontinuity on the boundary
# sphere |r|=π — antipodal boundary points represent the same rotation.
# These helpers let downstream visualizations pick the alternative rotvec
# representation `r' = (1 - 2π/|r|)·r` when it preserves continuity (closer
# to a reference like truth-rotvec, or to the previous frame's pick).
# --------------------------------------------------------------------------


def alt_rotvec_rep(r: np.ndarray) -> np.ndarray:
    """Alternate rotvec representation of the same rotation.

    For canonical rotvec `r = θ·n̂` with `θ ∈ [0, π]`, the alt rep is the
    rotation expressed as `(2π − θ)` around `−n̂`, i.e.
    `(1 − 2π/θ)·r` with magnitude `2π − θ` and direction `−n̂`.

    Identity (`|r|=0`) is returned unchanged — it has no alt.
    Works elementwise on (..., 3) arrays.
    """
    r = np.asarray(r, float)
    norm = np.linalg.norm(r, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-10, norm, 1.0)
    out = (1.0 - 2.0 * np.pi / safe) * r
    return np.where(norm > 1e-10, out, r)


def anchor_rotvec_to(r: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    """Pick whichever of {r, alt_rotvec_rep(r)} is closer in 3D to `r_ref`.

    `r` and `r_ref` broadcast over the last axis (..., 3).
    """
    r = np.asarray(r, float)
    r_ref = np.asarray(r_ref, float)
    r_alt = alt_rotvec_rep(r)
    d_can = np.linalg.norm(r - r_ref, axis=-1, keepdims=True)
    d_alt = np.linalg.norm(r_alt - r_ref, axis=-1, keepdims=True)
    return np.where(d_alt < d_can, r_alt, r)


def temporal_unwrap_rotvec(rotvec_seq: np.ndarray) -> np.ndarray:
    """Sequentially pick rotvec rep at each step to minimize 3D jump from
    the previous step.

    Lets a trajectory walk past `|r|=π` into the outer shell `|r| ∈ (π, 2π)`
    instead of teleporting at the boundary. The first sample is kept
    canonical; each subsequent sample picks canonical or alt rep
    based on Euclidean distance to the previously chosen rep.
    """
    seq = np.asarray(rotvec_seq, float)
    out = np.empty_like(seq)
    if len(seq) == 0:
        return out
    out[0] = seq[0]
    for f in range(1, len(seq)):
        out[f] = anchor_rotvec_to(seq[f], out[f - 1])
    return out
