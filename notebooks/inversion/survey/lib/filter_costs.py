"""Necessary-condition filter costs for the survey.

Two metrics treated as one-sided rejection filters (high = good):

  alignment_cost(q0, omega, truth_data)
      Did the candidate produce a peak at each truth-peak epoch?
      Score = matched / total truth-peaks. Per-peak match = candidate
      surrogate LC has a local minimum (bright peak) within ±W epochs
      of the truth peak.

  geo_cost(q0, omega, truth_data, face_normals, tier_table)
      At each truth spec-event peak, does the candidate place a
      tier-allowed face within a small angle of body-frame PAB?
      Score = matched / total spec-event peaks. Per-peak match = at
      least one face in the magnitude-implied tier shortlist falls
      within `geo_threshold_deg` of pab_body_candidate(t).

Both scores live in [0, 1]; truth = 1.0 by construction.

Use as filters, NOT as optimisation targets.
"""

from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from src.dynamics.attitude_propagator import propagate_attitude

from .surrogate_eval import predict as surrogate_predict


SURVEY_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SURVEY_ROOT.parent.parent.parent
M048_MASTER = (
    PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
    / "m048_trajectories" / "m048_trajectories.npz"
)

GROUP_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z", "+WD", "-WD", "+ED", "-ED"]
D_REF_KM = 38649.2


def load_static_geometry():
    """Load the body-frame face normals + cohort inertia tensor.

    Returns dict with:
      face_normals: (10, 3) — body-frame group normals (SP=0, AD=15)
      group_names: (10,) array of strings
      inertia_tensor: (3, 3) cohort inertia tensor
    """
    d = np.load(M048_MASTER)
    return {
        "face_normals": np.array(d["unique_normals"]),
        "group_names": np.array(d["group_names"]),
        "inertia_tensor": np.array(d["inertia_tensor"]),
    }


def load_tier_table():
    """Load the s018b face-identity tier table.

    Returns dict with:
      tier_labels: (4,) ['T1_X', 'T2_YZ', 'T3_any', 'T4_D']
      tier_mag_lo, tier_mag_hi: (4,) mag_abs band edges per tier
      tier_face_idx: list of 4 arrays, each containing face indices in 0..9
                     allowed for that tier.
    """
    d = np.load(SURVEY_ROOT / "results" / "s018b" / "face_tiers.npz")
    cand = d["tier_candidate_indices"]
    sizes = d["tier_shortlist_sizes"]
    tier_face_idx = [cand[i, : sizes[i]].astype(int) for i in range(len(sizes))]
    return {
        "tier_labels": d["tier_labels"],
        "tier_mag_lo": d["tier_mag_abs_lo"],
        "tier_mag_hi": d["tier_mag_abs_hi"],
        "tier_face_idx": tier_face_idx,
    }


def assign_tier(mag_abs: np.ndarray, tier_table: dict) -> np.ndarray:
    """Map mag_abs values to tier indices (0..3) or -1 if outside any tier."""
    tier = np.full(mag_abs.shape, -1, dtype=np.int8)
    for ti in range(4):
        lo = tier_table["tier_mag_lo"][ti]
        hi = tier_table["tier_mag_hi"][ti]
        m = (mag_abs >= lo) & (mag_abs < hi)
        tier[m] = ti
    return tier


def precompute_seed_filter_data(
    truth: dict,
    tier_table: dict,
    spec_threshold_deg: float = 5.0,
    bright_mag_threshold: float = 11.0,
):
    """Precompute per-seed filter inputs from cached truth NPZ.

    Returns:
      truth_peak_idx: (n_peaks,) indices of all truth peaks.
      bright_peak_idx: (n_bright,) subset of truth_peak_idx with
                       mag_hifi[idx] < bright_mag_threshold. Used by
                       alignment cost — faint peaks are noise.
      spec_event_idx: (n_spec,) subset of truth_peak_idx that are spec
                      events (min_ang_dist < spec_threshold_deg AND
                      mag_abs < 9). Used by geo cost.
      spec_tier:      (n_spec,) tier index (0..3) per spec event.
      pab_j2000:      (N_obs, 3) body-frame-independent PAB in J2000.
      sun_vec_j2000:  (N_obs, 3) sun_pos - sat_pos.
      obs_vec_j2000:  (N_obs, 3) obs_pos - sat_pos.
      observation_times, obs_dist_km, mag_hifi: cached arrays.
    """
    obs_times = np.asarray(truth["observation_times"], dtype=float)
    sun_pos = np.asarray(truth["sun_pos"], dtype=float)
    obs_pos = np.asarray(truth["obs_pos"], dtype=float)
    sat_pos = np.asarray(truth["sat_pos"], dtype=float)
    obs_dist_km = np.asarray(truth["obs_dist"], dtype=float)
    mag_hifi = np.asarray(truth["mag_hifi"], dtype=float)
    min_ang_dist = np.asarray(truth["min_ang_dist"], dtype=float)
    hifi_peak_epochs = np.asarray(truth["hifi_peak_epochs"], dtype=int)

    sun_vec = sun_pos - sat_pos
    obs_vec = obs_pos - sat_pos

    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_unit = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)
    pab_j2000 = sun_unit + obs_unit
    pab_j2000 /= np.linalg.norm(pab_j2000, axis=1, keepdims=True)

    # Bright peaks for alignment cost — the salient "peaks where they should be"
    is_bright = mag_hifi[hifi_peak_epochs] < bright_mag_threshold
    bright_peak_idx = hifi_peak_epochs[is_bright]

    # Spec events: tier-classifiable + truth-PAB-aligned (geo-constrained)
    mag_abs_at_peak = mag_hifi[hifi_peak_epochs] - 5.0 * np.log10(
        obs_dist_km[hifi_peak_epochs] / D_REF_KM
    )
    tier_at_peak = assign_tier(mag_abs_at_peak, tier_table)
    is_spec = (
        (min_ang_dist[hifi_peak_epochs] < spec_threshold_deg)
        & (tier_at_peak >= 0)
    )
    spec_event_idx = hifi_peak_epochs[is_spec]
    spec_tier = tier_at_peak[is_spec]

    return {
        "truth_peak_idx": hifi_peak_epochs,
        "bright_peak_idx": bright_peak_idx,
        "spec_event_idx": spec_event_idx,
        "spec_tier": spec_tier,
        "pab_j2000": pab_j2000,
        "sun_vec_j2000": sun_vec,
        "obs_vec_j2000": obs_vec,
        "observation_times": obs_times,
        "obs_dist_km": obs_dist_km,
        "mag_hifi": mag_hifi,
    }


def propagate_candidate(
    q0_wxyz: np.ndarray,
    omega0_rad: np.ndarray,
    seed_data: dict,
    inertia_tensor: np.ndarray,
):
    """Propagate (q0, omega0) and build candidate body-frame state.

    Returns:
      k1_body: (N_obs, 3) body-frame sun direction (unit).
      k2_body: (N_obs, 3) body-frame observer direction (unit).
      pab_body: (N_obs, 3) body-frame PAB direction (unit).
    """
    obs_times = seed_data["observation_times"]
    sun_vec = seed_data["sun_vec_j2000"]
    obs_vec = seed_data["obs_vec_j2000"]
    pab_j2000 = seed_data["pab_j2000"]

    quats, _ = propagate_attitude(
        q0=q0_wxyz, omega0=omega0_rad,
        times=obs_times, mode="tumbling",
        inertia_tensor=inertia_tensor,
    )

    # R = R_inertial_to_body (conv-(a)). pab_body = R @ pab_j2000.
    qxyzw = quats[:, [1, 2, 3, 0]]
    R = Rotation.from_quat(qxyzw).as_matrix()  # (N, 3, 3) — i→b
    sun_unit = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    obs_unit = obs_vec / np.linalg.norm(obs_vec, axis=1, keepdims=True)
    k1_body = np.einsum("nij,nj->ni", R, sun_unit)
    k2_body = np.einsum("nij,nj->ni", R, obs_unit)
    pab_body = np.einsum("nij,nj->ni", R, pab_j2000)
    return k1_body, k2_body, pab_body


def alignment_cost(
    candidate_mag_pred: np.ndarray,
    bright_peak_idx: np.ndarray,
    window_epochs: int = 3,
    bright_mag_threshold: float = 11.0,
) -> float:
    """Score = (truth bright peaks where candidate has a local-min bright value
    within ±window) / total truth bright peaks.

    Per-peak hit criterion: in window [tp - W, tp + W], the candidate's
    surrogate magnitude has a local minimum at some index `i` AND
    candidate_mag_pred[i] < bright_mag_threshold.

    No prominence requirement — handles the s018a case where many truth
    "peaks" are tiny dips in dark sky (mag 14-16, prominence ~0.01) that
    are not the salient bright peaks the metric is asking about.

    Args:
      candidate_mag_pred: (N_obs,) candidate surrogate LC.
      bright_peak_idx: indices of truth bright peaks (mag_hifi < threshold).
      window_epochs: tolerance for peak-position match.
      bright_mag_threshold: peak must have surrogate mag below this to count.
    """
    if bright_peak_idx.size == 0:
        return float("nan")
    N = candidate_mag_pred.shape[0]
    W = window_epochs
    hits = 0
    for tp in bright_peak_idx:
        lo = max(0, int(tp) - W)
        hi = min(N, int(tp) + W + 1)
        window = candidate_mag_pred[lo:hi]
        # Argmin position in window
        local_min_idx = lo + int(np.argmin(window))
        # Confirm it's a true local min (not at window edge with no neighbors)
        is_min = True
        if local_min_idx > 0:
            is_min &= candidate_mag_pred[local_min_idx] <= candidate_mag_pred[local_min_idx - 1]
        if local_min_idx < N - 1:
            is_min &= candidate_mag_pred[local_min_idx] <= candidate_mag_pred[local_min_idx + 1]
        if is_min and candidate_mag_pred[local_min_idx] < bright_mag_threshold:
            hits += 1
    return hits / float(bright_peak_idx.size)


def geo_cost(
    candidate_pab_body: np.ndarray,
    spec_event_idx: np.ndarray,
    spec_tier: np.ndarray,
    face_normals: np.ndarray,
    tier_face_idx: list,
    geo_threshold_deg: float = 5.0,
) -> float:
    """Score = fraction of spec events where a tier-allowed face is < threshold from PAB.

    Args:
      candidate_pab_body: (N_obs, 3) candidate body-frame PAB direction (unit).
      spec_event_idx: indices into observation_times of truth spec events.
      spec_tier: (n_spec,) tier index (0..3) per spec event.
      face_normals: (10, 3) body-frame face normals (unit).
      tier_face_idx: list of 4 arrays — face indices allowed per tier.
      geo_threshold_deg: angular tolerance for a face to count as PAB-aligned.
    """
    if spec_event_idx.size == 0:
        return float("nan")
    cos_thresh = np.cos(np.deg2rad(geo_threshold_deg))
    hits = 0
    for k in range(spec_event_idx.size):
        i = spec_event_idx[k]
        ti = int(spec_tier[k])
        allowed = tier_face_idx[ti]
        # cos(angle) between pab and each allowed face normal — max over allowed
        dots = candidate_pab_body[i] @ face_normals[allowed].T
        if np.max(dots) >= cos_thresh:
            hits += 1
    return hits / float(spec_event_idx.size)


def evaluate_candidate(
    q0_wxyz: np.ndarray,
    omega0_rad: np.ndarray,
    seed_data: dict,
    inertia_tensor: np.ndarray,
    face_normals: np.ndarray,
    tier_face_idx: list,
    *,
    sp_angle_deg: float = 0.0,
    ad_angle_deg: float = 15.0,
    align_window_epochs: int = 3,
    align_bright_mag: float = 11.0,
    geo_threshold_deg: float = 5.0,
):
    """Convenience: propagate + score both filters.

    Returns dict with score_alignment, score_geo, mag_pred (for downstream use).
    """
    k1_body, k2_body, pab_body = propagate_candidate(
        q0_wxyz, omega0_rad, seed_data, inertia_tensor
    )
    mag_pred = surrogate_predict(
        k1_body, k2_body, seed_data["obs_dist_km"],
        sp_angle_deg=sp_angle_deg, ad_angle_deg=ad_angle_deg,
    )
    s_align = alignment_cost(
        mag_pred, seed_data["bright_peak_idx"],
        window_epochs=align_window_epochs,
        bright_mag_threshold=align_bright_mag,
    )
    s_geo = geo_cost(
        pab_body, seed_data["spec_event_idx"], seed_data["spec_tier"],
        face_normals, tier_face_idx, geo_threshold_deg=geo_threshold_deg,
    )
    return {
        "score_alignment": s_align,
        "score_geo": s_geo,
        "mag_pred": mag_pred,
    }
