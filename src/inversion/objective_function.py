"""
Objective function wrapper for lightcurve inversion.

This module provides the ObjectiveFunction class that wraps the LCAS
forward model (generate_lightcurves) for use in optimization algorithms.

The objective function computes the chi-squared residual between predicted
and observed lightcurves given a set of initial attitude parameters.
"""

import logging
from typing import Any, Dict, Optional
import numpy as np
from numpy.typing import NDArray

from ..io.stl_loader import Satellite
from ..dynamics.attitude_propagator import propagate_attitude
from ..computation.shadow_engine import compute_shadows, create_no_shadow_lit_status
from ..computation.lightcurve_generator import generate_lightcurves
from .quaternion_utils import axis_angle_to_quaternion

logger = logging.getLogger(__name__)


def _quaternion_to_rotation_matrix(q: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Convert quaternion to 3x3 rotation matrix.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Quaternion (w, x, y, z) - scalar first convention.

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.
    """
    w, x, y, z = q

    # Rotation matrix from quaternion (scalar-first convention)
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


class ObjectiveFunction:
    """
    Objective function wrapper for lightcurve inversion optimization.

    This class wraps the LCAS generate_lightcurves forward model to compute
    residuals between predicted and observed lightcurves for a given set
    of initial attitude parameters.

    Parameters
    ----------
    satellite : Satellite
        Satellite model with STL components and BRDF properties.
    observation_times : ndarray
        Array of observation times (seconds from reference epoch).
    observed_lightcurve : ndarray
        Array of observed magnitudes.
    sun_positions_j2000 : ndarray
        Sun positions in J2000 frame (N, 3) in km.
    observer_positions_j2000 : ndarray
        Observer positions in J2000 frame (N, 3) in km.
    satellite_positions_j2000 : ndarray
        Satellite positions in J2000 frame (N, 3) in km.
    observer_distances : ndarray
        Observer-to-satellite distances (N,) in km.
    uncertainties : ndarray, optional
        Array of measurement uncertainties for weighted chi-squared.
    compute_shadows : bool, optional
        Whether to compute shadows via ray tracing. Default True.
    articulation_matrices : dict, optional
        Pre-computed rotation matrices for articulated components.
        Dict mapping component names to (N, 4, 4) transformation matrices.
        Used for fixed articulation angles (e.g., solar panels at 0°,
        antenna dishes at 15°).

    Attributes
    ----------
    n_evaluations : int
        Number of times evaluate() has been called.
    _last_progress_print : int
        Last evaluation count when progress was printed.
    """

    # Class-level constant for progress printing interval
    PROGRESS_PRINT_INTERVAL = 100

    def __init__(
        self,
        satellite: Satellite,
        observation_times: NDArray[np.floating],
        observed_lightcurve: NDArray[np.floating],
        sun_positions_j2000: NDArray[np.floating],
        observer_positions_j2000: NDArray[np.floating],
        satellite_positions_j2000: NDArray[np.floating],
        observer_distances: NDArray[np.floating],
        uncertainties: Optional[NDArray[np.floating]] = None,
        compute_shadows_flag: bool = True,
        articulation_matrices: Optional[Dict[str, NDArray[np.floating]]] = None,
    ) -> None:
        """Initialize the objective function with observation data."""
        # Validate inputs
        n_obs = len(observation_times)
        if len(observed_lightcurve) != n_obs:
            raise ValueError(
                f"observed_lightcurve length ({len(observed_lightcurve)}) "
                f"must match observation_times length ({n_obs})"
            )
        if len(sun_positions_j2000) != n_obs:
            raise ValueError(
                f"sun_positions_j2000 length ({len(sun_positions_j2000)}) "
                f"must match observation_times length ({n_obs})"
            )
        if len(observer_positions_j2000) != n_obs:
            raise ValueError(
                f"observer_positions_j2000 length ({len(observer_positions_j2000)}) "
                f"must match observation_times length ({n_obs})"
            )
        if len(satellite_positions_j2000) != n_obs:
            raise ValueError(
                f"satellite_positions_j2000 length ({len(satellite_positions_j2000)}) "
                f"must match observation_times length ({n_obs})"
            )
        if len(observer_distances) != n_obs:
            raise ValueError(
                f"observer_distances length ({len(observer_distances)}) "
                f"must match observation_times length ({n_obs})"
            )
        if uncertainties is not None and len(uncertainties) != n_obs:
            raise ValueError(
                f"uncertainties length ({len(uncertainties)}) "
                f"must match observation_times length ({n_obs})"
            )

        # Store satellite model
        self.satellite = satellite

        # Store observation data
        self.observation_times = np.asarray(observation_times, dtype=np.float64)
        self.observed_lightcurve = np.asarray(observed_lightcurve, dtype=np.float64)
        self.uncertainties = (
            np.asarray(uncertainties, dtype=np.float64)
            if uncertainties is not None
            else None
        )

        # Store J2000 positions for body-frame vector computation
        self.sun_positions_j2000 = np.asarray(sun_positions_j2000, dtype=np.float64)
        self.observer_positions_j2000 = np.asarray(
            observer_positions_j2000, dtype=np.float64
        )
        self.satellite_positions_j2000 = np.asarray(
            satellite_positions_j2000, dtype=np.float64
        )
        self.observer_distances = np.asarray(observer_distances, dtype=np.float64)

        # Shadow computation flag
        self.compute_shadows_flag = compute_shadows_flag

        # Articulation matrices for fixed component angles
        # Dict mapping component names to (N, 4, 4) rotation matrices
        self.articulation_matrices = articulation_matrices if articulation_matrices else {}

        # Statistics
        self.n_evaluations = 0
        self._last_progress_print = 0

        logger.debug(
            f"ObjectiveFunction initialized: {n_obs} observations, "
            f"shadows={'on' if compute_shadows_flag else 'off'}, "
            f"articulation_components={list(self.articulation_matrices.keys()) if self.articulation_matrices else 'none'}"
        )

    def _compute_body_frame_vectors(
        self,
        quaternions: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute sun and observer direction vectors in body frame.

        Parameters
        ----------
        quaternions : ndarray, shape (N, 4)
            Attitude quaternions for each observation time.

        Returns
        -------
        k1_vectors : ndarray, shape (N, 3)
            Sun direction vectors in body frame (normalized).
        k2_vectors : ndarray, shape (N, 3)
            Observer direction vectors in body frame (normalized).
        """
        n_obs = len(self.observation_times)
        k1_vectors = np.zeros((n_obs, 3), dtype=np.float64)
        k2_vectors = np.zeros((n_obs, 3), dtype=np.float64)

        for i in range(n_obs):
            # Get rotation matrix from quaternion (J2000 to body frame)
            # This matrix transforms vectors FROM J2000 TO body frame
            q = quaternions[i]
            R_j2000_to_body = _quaternion_to_rotation_matrix(q)

            # Sun vector in J2000: from satellite to sun
            sun_vec_j2000 = self.sun_positions_j2000[i] - self.satellite_positions_j2000[i]
            sun_vec_body = R_j2000_to_body @ sun_vec_j2000
            sun_norm = np.linalg.norm(sun_vec_body)
            if sun_norm > 1e-12:
                k1_vectors[i] = sun_vec_body / sun_norm
            else:
                k1_vectors[i] = np.array([1.0, 0.0, 0.0])

            # Observer vector in J2000: from satellite to observer
            obs_vec_j2000 = (
                self.observer_positions_j2000[i] - self.satellite_positions_j2000[i]
            )
            obs_vec_body = R_j2000_to_body @ obs_vec_j2000
            obs_norm = np.linalg.norm(obs_vec_body)
            if obs_norm > 1e-12:
                k2_vectors[i] = obs_vec_body / obs_norm
            else:
                k2_vectors[i] = np.array([0.0, 1.0, 0.0])

        return k1_vectors, k2_vectors

    def _generate_predicted_lightcurve(
        self,
        k1_vectors: NDArray[np.floating],
        k2_vectors: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Generate predicted lightcurve using the forward model.

        Parameters
        ----------
        k1_vectors : ndarray, shape (N, 3)
            Sun direction vectors in body frame.
        k2_vectors : ndarray, shape (N, 3)
            Observer direction vectors in body frame.

        Returns
        -------
        magnitudes : ndarray, shape (N,)
            Predicted magnitudes.
        """
        n_obs = len(self.observation_times)

        # Compute or skip shadows
        if self.compute_shadows_flag:
            lit_status_dict = compute_shadows(
                satellite=self.satellite,
                k1_vectors=k1_vectors,
                explicit_component_matrices=self.articulation_matrices,
                show_progress=False,
            )
        else:
            lit_status_dict = create_no_shadow_lit_status(self.satellite, n_obs)

        # Generate lightcurve using forward model
        magnitudes, _, _, _, _, _ = generate_lightcurves(
            facet_lit_status_dict=lit_status_dict,
            k1_vectors_array=k1_vectors,
            k2_vectors_array=k2_vectors,
            observer_distances=self.observer_distances,
            satellite=self.satellite,
            epochs=self.observation_times,
            pre_computed_matrices=self.articulation_matrices,
            show_progress=False,
        )

        return magnitudes

    def evaluate(
        self,
        params: NDArray[np.floating],
    ) -> float:
        """
        Evaluate the objective function for given parameters.

        Parameters
        ----------
        params : ndarray, shape (6,)
            Parameter array: [axis_angle(3), omega(3)]
            - axis_angle: Initial orientation as axis-angle (3 params)
            - omega: Initial angular velocity in body frame (3 params, rad/s)

        Returns
        -------
        float
            Chi-squared residual between predicted and observed lightcurve.
            Lower values indicate better fit.
        """
        self.n_evaluations += 1

        # Print progress every PROGRESS_PRINT_INTERVAL evaluations
        if self.n_evaluations - self._last_progress_print >= self.PROGRESS_PRINT_INTERVAL:
            print(f"      [Evaluations: {self.n_evaluations}]", flush=True)
            self._last_progress_print = self.n_evaluations

        params = np.asarray(params, dtype=np.float64)

        if len(params) != 6:
            raise ValueError(f"params must have 6 elements, got {len(params)}")

        # Extract parameters
        axis_angle = params[:3]
        omega = params[3:6]

        # Convert axis-angle to initial quaternion
        q0 = axis_angle_to_quaternion(axis_angle)

        # Propagate attitude (principal axis mode - constant omega)
        quaternions, _ = propagate_attitude(
            q0=q0,
            omega0=omega,
            times=self.observation_times,
            mode="principal_axis",
        )

        # Compute sun/observer vectors in body frame
        k1_vectors, k2_vectors = self._compute_body_frame_vectors(quaternions)

        # Generate predicted lightcurve
        predicted = self._generate_predicted_lightcurve(k1_vectors, k2_vectors)

        # Compute residual (chi-squared or MSE)
        chi_sq = self._compute_chi_squared(predicted)

        return chi_sq

    def _compute_chi_squared(
        self,
        predicted: NDArray[np.floating],
    ) -> float:
        """
        Compute chi-squared residual between predicted and observed.

        Observations with infinite magnitude (satellite not visible) are
        excluded from the calculation.

        Parameters
        ----------
        predicted : ndarray, shape (N,)
            Predicted magnitudes.

        Returns
        -------
        float
            Chi-squared or MSE residual.
        """
        # Create mask for valid observations (finite magnitudes)
        valid_obs = np.isfinite(self.observed_lightcurve)
        valid_pred = np.isfinite(predicted)
        valid_mask = valid_obs & valid_pred

        # If no valid observations, return large value
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            return 1e10

        # Extract valid data
        obs_valid = self.observed_lightcurve[valid_mask]
        pred_valid = predicted[valid_mask]
        residuals = obs_valid - pred_valid

        if self.uncertainties is not None:
            # Weighted chi-squared: sum((obs - pred)^2 / sigma^2)
            sigma_valid = self.uncertainties[valid_mask]
            # Avoid division by zero
            sigma_safe = np.where(sigma_valid > 1e-12, sigma_valid, 1e-12)
            chi_sq = np.sum(residuals**2 / sigma_safe**2)
        else:
            # Unweighted: MSE (mean squared error)
            chi_sq = np.sum(residuals**2) / n_valid

        return float(chi_sq)
