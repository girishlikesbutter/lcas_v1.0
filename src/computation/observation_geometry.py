"""
Observation geometry computation for satellite light curve generation.

Computes satellite positions, orientations, and observer vectors for light curve simulation.
Handles SPICE integration for precise orbital mechanics calculations.
"""

import time
import logging
from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray

from ..spice.spice_handler import SpiceHandler
from ..config.rso_config_schemas import RSO_Config

logger = logging.getLogger(__name__)


def compute_observation_geometry(
    epochs: NDArray[np.float64],
    satellite_id: int,
    observer_id: int,
    spice_handler: SpiceHandler,
    config: RSO_Config,
    attitude_keyframes: Optional[Dict[str, any]] = None
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute observation geometry for all time epochs.

    Calculates satellite positions, orientations, sun/observer vectors, and distances
    for the complete simulation time series. Uses SPICE for precise orbital mechanics
    or custom attitude interpolation when provided.

    Args:
        epochs: Array of time epochs in ephemeris time (ET)
        satellite_id: SPICE satellite ID for position calculations
        observer_id: SPICE ID for the ground observer/station (e.g., 399999 for DST)
        spice_handler: Initialized SPICE handler with loaded kernels
        config: Configuration object containing body frame information
        attitude_keyframes: Optional custom attitude keyframes for interpolation
            Format: {
                'times': [...],  # UTC strings or ET floats
                'time_format': 'utc' or 'et',
                'attitudes': [...],  # quaternions or matrices
                'format': 'quaternion' or 'matrix'
            }

    Returns:
        Dictionary containing geometry data:
        - 'k1_vectors': Sun direction vectors in body frame (N, 3)
        - 'k2_vectors': Observer direction vectors in body frame (N, 3)
        - 'observer_distances': Distances from satellite to observer in km (N,)
        - 'sun_positions': Sun positions in J2000 frame (N, 3)
        - 'sat_positions': Satellite positions in J2000 frame (N, 3)
        - 'obs_positions': Observer positions in J2000 frame (N, 3)
        - 'sat_att_matrices': Attitude matrices for body frame (N, 3, 3)

    Raises:
        RuntimeError: If SPICE calculations fail
        ValueError: If input parameters are invalid
    """
    # Input validation
    if len(epochs) == 0:
        raise ValueError("Epochs array cannot be empty")
    if satellite_id == 0:
        raise ValueError(f"Invalid satellite ID: {satellite_id}")

    logger.info("Computing observation geometry...")
    geometry_start = time.time()

    # Initialize storage arrays
    sun_positions = []
    sat_positions = []
    sat_att_matrices = []
    obs_positions = []
    k1_vectors = []  # Sun-to-sat vectors in body frame
    k2_vectors = []  # Observer-to-sat vectors in body frame
    observer_distances = []  # Observer-to-satellite distances in km

    # Get attitude matrices (either from SPICE or custom interpolation)
    if attitude_keyframes is not None:
        logger.info("Using custom attitude interpolation (SLERP)...")
        from ..interpolation.attitude_interpolator import interpolate_attitudes_only

        # Get interpolated attitude matrices for all epochs
        att_matrices_array = interpolate_attitudes_only(
            attitude_keyframes=attitude_keyframes,
            epochs=epochs,
            spice_handler=spice_handler
        )
        logger.info(f"Interpolated {len(att_matrices_array)} attitude matrices via SLERP")
    else:
        logger.info("Using SPICE attitude kernels...")
        att_matrices_array = None  # Will query SPICE in the loop

    # Compute geometry for each epoch
    for i, epoch in enumerate(epochs):
        try:
            # Get positions using SPICE (always needed)
            sun_pos, _ = spice_handler.get_body_position("SUN", epoch, "J2000", "EARTH")
            sat_pos, _ = spice_handler.get_body_position(str(satellite_id), epoch, "J2000", "EARTH")
            obs_pos, _ = spice_handler.get_body_position(str(observer_id), epoch, "J2000", "EARTH")

            # Get attitude matrix from either custom interpolation or SPICE
            if att_matrices_array is not None:
                att_matrix = att_matrices_array[i]
            else:
                att_matrix = spice_handler.get_target_orientation("J2000", config.spice_config.body_frame, epoch)

        except Exception as e:
            logger.error(f"SPICE calculation failed for satellite {satellite_id} at epoch {i}: {e}")
            raise RuntimeError(f"Geometry computation failed at epoch {i}: {e}") from e

        # Store position and attitude data
        sun_positions.append(sun_pos)
        sat_positions.append(sat_pos)
        obs_positions.append(obs_pos)
        sat_att_matrices.append(att_matrix)

        # Compute k1/k2 vectors (same for both SPICE and custom attitudes)
        sun_vector_j2000 = sun_pos - sat_pos
        k1_vector_body = att_matrix @ sun_vector_j2000
        k1_norm = np.linalg.norm(k1_vector_body)
        if k1_norm > 0:
            k1_vector_body = k1_vector_body / k1_norm
        else:
            k1_vector_body = np.array([1.0, 0.0, 0.0])  # Default direction
        k1_vectors.append(k1_vector_body)

        obs_vector_j2000 = obs_pos - sat_pos
        k2_vector_body = att_matrix @ obs_vector_j2000
        k2_norm = np.linalg.norm(k2_vector_body)
        if k2_norm > 0:
            k2_vector_body = k2_vector_body / k2_norm
        else:
            k2_vector_body = np.array([0.0, 1.0, 0.0])  # Default direction
        k2_vectors.append(k2_vector_body)

        # Calculate observer distance (used in both surrogate and ray tracing paths)
        # Note: SPICE positions are in km, so obs_vector_j2000 is in km
        distance_km = np.linalg.norm(obs_vector_j2000)  # Already in km from SPICE
        observer_distances.append(distance_km)

    # Convert lists to numpy arrays for efficient processing
    geometry_data = {
        'k1_vectors': np.array(k1_vectors),
        'k2_vectors': np.array(k2_vectors),
        'observer_distances': np.array(observer_distances),
        'sun_positions': np.array(sun_positions),
        'sat_positions': np.array(sat_positions),
        'obs_positions': np.array(obs_positions),
        'sat_att_matrices': np.array(sat_att_matrices)
    }

    geometry_time = time.time() - geometry_start
    logger.info(f"Geometry computed for {len(epochs)} epochs in {geometry_time:.2f}s")

    return geometry_data
