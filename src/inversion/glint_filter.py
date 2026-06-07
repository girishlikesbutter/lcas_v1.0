"""
Glint alignment filter for fast candidate rejection in lightcurve inversion.

A glint occurs when a facet normal aligns closely with the Phase Angle
Bisector (PAB). This module provides a fast geometric filter that checks
whether a candidate attitude trajectory produces alignment dips at the
times where the observed lightcurve shows brightness spikes.

The alignment check is ~3x faster than a lo-fi LC evaluation and ~770x
faster than hi-fi, making it an effective pre-filter for rejection.

Usage:
    filter = GlintFilter(unique_normals, pab_j2000, observed_glint_epochs)
    score = filter.evaluate(q0, omega0, times, inertia_tensor)
    if score < filter.rejection_threshold:
        # candidate cannot explain the observed glints — reject
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_prominences

from ..dynamics.attitude_propagator import propagate_attitude


def extract_glint_epochs(
    observed_magnitudes: NDArray,
    min_prominence: float = 0.3,
) -> tuple[NDArray, NDArray]:
    """
    Identify glint epochs from an observed lightcurve.

    Finds all local brightness maxima (magnitude minima) and returns
    their epoch indices and prominences.

    Parameters
    ----------
    observed_magnitudes : (N,) array
        Observed apparent magnitude time series.
    min_prominence : float
        Minimum prominence (in magnitudes) to classify as a confident
        glint. Peaks below this are still returned but flagged.

    Returns
    -------
    glint_epochs : (K,) int array
        Epoch indices of detected brightness peaks.
    prominences : (K,) float array
        Prominence of each peak in magnitudes.
    """
    peak_indices, _ = find_peaks(-observed_magnitudes, distance=3)
    if len(peak_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    proms, _, _ = peak_prominences(-observed_magnitudes, peak_indices)
    return peak_indices, proms


def compute_pab_j2000(
    sun_pos: NDArray, obs_pos: NDArray, sat_pos: NDArray,
) -> NDArray:
    """
    Compute Phase Angle Bisector in J2000 frame.

    PAB = normalize(sun_direction + observer_direction).

    Parameters
    ----------
    sun_pos, obs_pos, sat_pos : (N, 3) arrays
        Positions in J2000 (km).

    Returns
    -------
    pab : (N, 3) array
        Unit PAB vectors in J2000.
    """
    k1 = sun_pos - sat_pos
    k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
    k2 = obs_pos - sat_pos
    k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
    pab = k1 + k2
    pab /= np.linalg.norm(pab, axis=1, keepdims=True)
    return pab


class GlintFilter:
    """
    Fast geometric filter for candidate trajectory rejection.

    For a candidate (q0, omega0), propagates attitude and checks whether
    any normal family achieves close PAB alignment at the observed glint
    epochs. Candidates that cannot explain the observed glints are rejected.

    Parameters
    ----------
    unique_normals : (G, 3) array
        Unit normal vectors for each face family in body frame.
    pab_j2000 : (N, 3) array
        PAB unit vectors in J2000 at each observation epoch.
    glint_epochs : (K,) int array
        Epoch indices where observed lightcurve has brightness peaks.
    glint_prominences : (K,) float array, optional
        Prominence of each glint (used for weighting).
    alignment_threshold_deg : float
        Maximum angular distance (degrees) for a normal to "explain"
        a glint. Default 4.0 (roughly the Ashikhmin-Shirley specular
        lobe width for n_phong ~ 200-300).
    """

    def __init__(
        self,
        unique_normals: NDArray,
        pab_j2000: NDArray,
        glint_epochs: NDArray,
        glint_prominences: NDArray = None,
        alignment_threshold_deg: float = 4.0,
    ):
        self.unique_normals = np.asarray(unique_normals, dtype=np.float64)
        self.pab_j2000 = np.asarray(pab_j2000, dtype=np.float64)
        self.glint_epochs = np.asarray(glint_epochs, dtype=int)
        self.glint_prominences = (
            np.asarray(glint_prominences, dtype=np.float64)
            if glint_prominences is not None
            else np.ones(len(glint_epochs))
        )
        self.cos_threshold = np.cos(np.radians(alignment_threshold_deg))
        self.alignment_threshold_deg = alignment_threshold_deg
        self.n_groups = len(unique_normals)
        self.n_glints = len(glint_epochs)

        # PAB vectors at glint epochs only (for fast evaluation)
        self.pab_at_glints = self.pab_j2000[self.glint_epochs]  # (K, 3)

    def evaluate(
        self,
        q0: NDArray,
        omega0: NDArray,
        times: NDArray,
        inertia_tensor: NDArray,
        mode: str = "tumbling",
    ) -> dict:
        """
        Evaluate a candidate trajectory against observed glints.

        Parameters
        ----------
        q0 : (4,) array, wxyz
            Initial attitude quaternion.
        omega0 : (3,) array, rad/s
            Initial angular velocity in body frame.
        times : (N,) array
            Observation times (seconds from epoch 0).
        inertia_tensor : (3, 3) array
            Body inertia tensor.
        mode : str
            Propagation mode ("tumbling" or "principal_axis").

        Returns
        -------
        dict with:
            score : float
                Fraction of observed glints explained (0.0 to 1.0).
                1.0 = all glints have a matching alignment dip.
            weighted_score : float
                Prominence-weighted fraction explained.
            n_explained : int
                Number of glints with alignment < threshold.
            n_total : int
                Total number of observed glints.
            min_angles_deg : (K,) array
                Minimum angular distance across all groups at each glint epoch.
            best_groups : (K,) int array
                Which normal group is closest at each glint epoch.
            n_false_alarms : int
                Number of deep alignment dips (< threshold) at non-glint epochs.
        """
        # Propagate attitude
        quaternions, _ = propagate_attitude(
            q0=q0, omega0=omega0, times=times,
            mode=mode, inertia_tensor=inertia_tensor,
        )

        from scipy.spatial.transform import Rotation

        n_obs = len(times)

        # Compute R^T (body→J2000) at ALL epochs
        R_all = np.zeros((n_obs, 3, 3))
        for i in range(n_obs):
            q = quaternions[i]
            R_all[i] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T

        # Min angular distance across all groups at every epoch
        min_ang_all = np.full(n_obs, 180.0)
        best_group_all = np.full(n_obs, -1, dtype=int)

        for g in range(self.n_groups):
            n_j2000 = R_all @ self.unique_normals[g]  # (N, 3)
            cos_ang = np.sum(n_j2000 * self.pab_j2000, axis=1)
            ang_deg = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            closer = ang_deg < min_ang_all
            min_ang_all[closer] = ang_deg[closer]
            best_group_all[closer] = g

        # --- Predicted glints: epochs where alignment < threshold ---
        predicted_mask = min_ang_all < self.alignment_threshold_deg
        # Collapse contiguous runs into single predicted events
        # (a dip spanning 3 epochs = 1 predicted glint, not 3)
        predicted_epochs = np.where(predicted_mask)[0]
        predicted_events = []
        if len(predicted_epochs) > 0:
            splits = np.where(np.diff(predicted_epochs) > 3)[0] + 1
            for chunk in np.split(predicted_epochs, splits):
                # Use epoch with minimum angular distance as the event center
                best_idx = chunk[np.argmin(min_ang_all[chunk])]
                predicted_events.append(best_idx)
        predicted_events = np.array(predicted_events, dtype=int)
        n_predicted = len(predicted_events)

        # --- Check each predicted glint against observed LC peaks ---
        EPOCH_TOLERANCE = 3  # predicted and observed must be within ±3 epochs
        n_confirmed = 0
        confirmed_mask = np.zeros(n_predicted, dtype=bool)

        glint_set = set(self.glint_epochs)
        for i, pred_ep in enumerate(predicted_events):
            # Is there an observed LC peak within ±tolerance?
            for offset in range(-EPOCH_TOLERANCE, EPOCH_TOLERANCE + 1):
                if (pred_ep + offset) in glint_set:
                    n_confirmed += 1
                    confirmed_mask[i] = True
                    break

        precision = n_confirmed / n_predicted if n_predicted > 0 else 0.0

        # --- Also compute recall: how many observed glints are explained ---
        n_recalled = 0
        for obs_ep in self.glint_epochs:
            for pred_ep in predicted_events:
                if abs(pred_ep - obs_ep) <= EPOCH_TOLERANCE:
                    n_recalled += 1
                    break
        recall = n_recalled / self.n_glints if self.n_glints > 0 else 0.0

        # F1 score
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_predicted': n_predicted,
            'n_confirmed': n_confirmed,
            'n_observed': self.n_glints,
            'n_recalled': n_recalled,
            'predicted_epochs': predicted_events,
            'confirmed_mask': confirmed_mask,
            'min_ang_all': min_ang_all,
            'best_group_all': best_group_all,
        }

    def _count_false_alarms(self, quaternions: NDArray) -> int:
        """
        Count alignment dips below threshold at non-glint epochs.

        A false alarm is an epoch where the candidate predicts a glint
        (some normal < threshold from PAB) but no observed glint exists.
        """
        from scipy.spatial.transform import Rotation
        n_obs = len(quaternions)

        # Compute min angular distance across all groups at every epoch
        min_ang_all = np.full(n_obs, 180.0)
        R_all = np.zeros((n_obs, 3, 3))
        for i in range(n_obs):
            q = quaternions[i]
            R_all[i] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T

        for g in range(self.n_groups):
            n_j2000 = R_all @ self.unique_normals[g]
            cos_ang = np.sum(n_j2000 * self.pab_j2000, axis=1)
            ang_deg = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            min_ang_all = np.minimum(min_ang_all, ang_deg)

        # Epochs with deep dips
        deep_dip_epochs = set(np.where(min_ang_all < self.alignment_threshold_deg)[0])

        # Remove epochs near observed glints (within ±3 epochs)
        glint_set = set()
        for ep in self.glint_epochs:
            for offset in range(-3, 4):
                glint_set.add(ep + offset)

        false_alarms = deep_dip_epochs - glint_set
        return len(false_alarms)

    def quick_reject(
        self,
        q0: NDArray,
        omega0: NDArray,
        times: NDArray,
        inertia_tensor: NDArray,
        min_precision: float = 0.5,
        mode: str = "tumbling",
    ) -> bool:
        """
        Fast rejection test: returns True if candidate should be REJECTED.

        Propagates attitude once, finds all predicted alignment dips,
        and checks what fraction are confirmed by observed LC peaks.

        Parameters
        ----------
        min_precision : float
            Minimum precision to pass. Default 0.5 (at least half the
            predicted glints must correspond to observed LC peaks).

        Returns
        -------
        reject : bool
            True if candidate fails the glint test and should be rejected.
        """
        result = self.evaluate(q0, omega0, times, inertia_tensor, mode=mode)

        # If no predictions at all, trajectory is too far off
        if result['n_predicted'] == 0:
            return True

        return result['precision'] < min_precision
