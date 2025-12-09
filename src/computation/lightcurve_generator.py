"""
Light curve generation for satellite models.

Provides vectorized light curve generation with BRDF calculations,
shadow handling, and animation data collection for satellite simulations.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ..io.stl_loader import Satellite
from .brdf import BRDFCalculator, convert_flux_to_magnitude
from .facet_data_extractor import (
    extract_facet_arrays,
    apply_articulation_to_arrays,
    apply_articulation_to_vertices,
    lit_status_to_flat_array,
    FacetArrays
)

logger = logging.getLogger(__name__)


def _compute_per_facet_flux(
    normals: np.ndarray,
    areas: np.ndarray,
    r_d: np.ndarray,
    r_s: np.ndarray,
    n_phong: np.ndarray,
    sun_direction: np.ndarray,
    observer_direction: np.ndarray,
    lit_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-facet flux values using vectorized operations.

    Args:
        normals: Facet normals (M, 3)
        areas: Facet areas (M,)
        r_d, r_s, n_phong: BRDF parameters (M,) each
        sun_direction: Sun direction vector (3,)
        observer_direction: Observer direction vector (3,)
        lit_mask: Boolean mask for lit facets (M,)

    Returns:
        Tuple of:
        - per_facet_flux: Flux per facet (M,), 0 for inactive facets
        - back_culled: Boolean mask of back-culled facets (M,)
        - n_dot_k2: Observer visibility values (M,)
    """
    from .brdf import calculate_brdf_vectorized

    M = len(normals)

    # Vectorized dot products
    n_dot_k1 = normals @ sun_direction      # (M,)
    n_dot_k2 = normals @ observer_direction  # (M,)

    # Back-culled = facing away from observer
    back_culled = n_dot_k2 <= 0

    # Visibility mask: facet must face both sun and observer
    visible = (n_dot_k1 > 0) & (n_dot_k2 > 0)

    # Combined mask: visible AND lit (not shadowed)
    active = visible & lit_mask

    # Initialize per-facet flux to zero
    per_facet_flux = np.zeros(M, dtype=np.float64)

    if not np.any(active):
        return per_facet_flux, back_culled, n_dot_k2

    # Halfway vector
    halfway = sun_direction + observer_direction
    halfway_norm = np.linalg.norm(halfway)
    if halfway_norm < 1e-10:
        return per_facet_flux, back_culled, n_dot_k2
    halfway = halfway / halfway_norm

    h_dot_k1 = np.dot(halfway, sun_direction)
    n_dot_h = normals @ halfway

    # Vectorized BRDF
    rho = calculate_brdf_vectorized(
        n_dot_k1, n_dot_k2, h_dot_k1, n_dot_h,
        r_d, r_s, n_phong
    )

    # Compute flux for all facets
    flux = rho * areas * n_dot_k1 * n_dot_k2

    # Only keep active facets
    per_facet_flux[active] = flux[active]

    return per_facet_flux, back_culled, n_dot_k2


def generate_lightcurves(
    facet_lit_status_dict: Dict[str, NDArray[np.bool_]],
    k1_vectors_array: NDArray[np.float64],
    k2_vectors_array: NDArray[np.float64],
    observer_distances: NDArray[np.float64],
    satellite: Satellite,
    epochs: NDArray[np.float64],
    brdf_calculator: Optional[BRDFCalculator] = None,
    pre_computed_matrices: Optional[Dict[str, NDArray[np.float64]]] = None,
    generate_no_shadow: bool = False,
    animate: bool = False,
    show_progress: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], NDArray[np.float64], Optional[List[Dict[str, Any]]]]:
    """
    Generate light curves with vectorized BRDF calculations.

    Args:
        facet_lit_status_dict: Dictionary mapping component names to boolean lit status arrays (N, num_facets)
        k1_vectors_array: Pre-computed sun vectors in body frame (N, 3)
        k2_vectors_array: Pre-computed observer vectors in body frame (N, 3)
        observer_distances: Pre-computed observer distances in km (N,)
        satellite: Satellite model
        epochs: Array of time epochs in ephemeris time (ET)
        brdf_calculator: Optional BRDFCalculator (accepted for API compatibility)
        pre_computed_matrices: Pre-computed rotation matrices (N, 4, 4) per component
        generate_no_shadow: If True, also generate non-shadowed curve
        animate: If True, collect animation data for visualization
        show_progress: If True, display tqdm progress bar

    Returns:
        Tuple containing:
        - magnitudes_shadowed: Shadowed light curve magnitudes (N,)
        - total_flux_shadowed: Shadowed total flux values (N,)
        - magnitudes_no_shadow: Non-shadowed magnitudes if requested, None otherwise
        - total_flux_no_shadow: Non-shadowed flux values if requested, None otherwise
        - observer_distances_out: Observer distances array (N,)
        - animation_data: Animation frame data if animate=True, None otherwise
    """
    # Input validation
    num_epochs = len(epochs)
    if len(k1_vectors_array) != num_epochs:
        raise ValueError("k1_vectors_array length must match epochs length")
    if len(k2_vectors_array) != num_epochs:
        raise ValueError("k2_vectors_array length must match epochs length")
    if len(observer_distances) != num_epochs:
        raise ValueError("observer_distances length must match epochs length")

    logger.info(f"Generating light curves for {num_epochs} epochs")

    # Extract facet data once (preprocessing)
    facet_arrays = extract_facet_arrays(satellite)
    logger.info(f"Extracted {facet_arrays.total_facets} facets into contiguous arrays")

    # Initialize output arrays
    magnitudes_shadowed = np.zeros(num_epochs)
    total_flux_shadowed = np.zeros(num_epochs)
    magnitudes_no_shadow = np.zeros(num_epochs) if generate_no_shadow else None
    total_flux_no_shadow = np.zeros(num_epochs) if generate_no_shadow else None
    observer_distances_out = observer_distances.copy()

    # Initialize animation data collection
    animation_data: Optional[List[Dict[str, Any]]] = [] if animate else None

    # Default to empty matrices if none provided
    if pre_computed_matrices is None:
        pre_computed_matrices = {}

    # Progress bar
    epoch_iter = range(num_epochs)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Generating light curves", unit="epoch", mininterval=0.5)

    for i in epoch_iter:
        sun_dir = k1_vectors_array[i]
        obs_dir = k2_vectors_array[i]
        distance = observer_distances[i]

        # Apply articulation to get transformed normals/centers
        transformed_normals, transformed_centers = apply_articulation_to_arrays(
            facet_arrays, pre_computed_matrices, i, satellite
        )

        # Convert lit_status to flat array for this epoch
        lit_mask = lit_status_to_flat_array(facet_lit_status_dict, facet_arrays, i)

        # Compute per-facet flux (vectorized)
        per_facet_flux, back_culled, n_dot_k2 = _compute_per_facet_flux(
            normals=transformed_normals,
            areas=facet_arrays.areas,
            r_d=facet_arrays.r_d,
            r_s=facet_arrays.r_s,
            n_phong=facet_arrays.n_phong,
            sun_direction=sun_dir,
            observer_direction=obs_dir,
            lit_mask=lit_mask
        )

        # Sum flux for magnitude calculation
        flux_sum = np.sum(per_facet_flux)

        # Store results
        total_flux_shadowed[i] = flux_sum
        if flux_sum > 1e-20:
            log_flux = np.log10(flux_sum)
            magnitudes_shadowed[i] = convert_flux_to_magnitude(log_flux, distance, mode='surrogate')
        else:
            magnitudes_shadowed[i] = np.inf

        # Collect animation data
        if animate and animation_data is not None:
            # Transform vertices using vectorized function (no deep copy)
            transformed_vertices = apply_articulation_to_vertices(
                facet_arrays, pre_computed_matrices, i, satellite
            )

            # Convert per-facet data to dict format for animation
            facet_flux_dict = {}
            back_culled_dict = {}
            flat_idx = 0
            for component in satellite.components:
                for facet_idx in range(len(component.facets)):
                    facet_key = f"{component.name}_{facet_idx}"
                    facet_flux_dict[facet_key] = per_facet_flux[flat_idx]
                    back_culled_dict[facet_key] = back_culled[flat_idx]
                    flat_idx += 1

            epoch_data = {
                'epoch_index': i,
                'epoch': epochs[i],
                'transformed_vertices': transformed_vertices,
                'sun_direction': sun_dir.copy(),
                'observer_direction': obs_dir.copy(),
                'lit_status': copy.deepcopy(facet_lit_status_dict),
                'back_culled_facets': back_culled_dict,
                'facet_flux': facet_flux_dict
            }
            animation_data.append(epoch_data)

        # No-shadow calculation (ignore shadow status)
        if generate_no_shadow and magnitudes_no_shadow is not None and total_flux_no_shadow is not None:
            all_lit_mask = np.ones(facet_arrays.total_facets, dtype=bool)

            per_facet_flux_ns, _, _ = _compute_per_facet_flux(
                normals=transformed_normals,
                areas=facet_arrays.areas,
                r_d=facet_arrays.r_d,
                r_s=facet_arrays.r_s,
                n_phong=facet_arrays.n_phong,
                sun_direction=sun_dir,
                observer_direction=obs_dir,
                lit_mask=all_lit_mask
            )

            flux_sum_no_shadow = np.sum(per_facet_flux_ns)
            total_flux_no_shadow[i] = flux_sum_no_shadow
            if flux_sum_no_shadow > 1e-20:
                log_flux_ns = np.log10(flux_sum_no_shadow)
                magnitudes_no_shadow[i] = convert_flux_to_magnitude(log_flux_ns, distance, mode='surrogate')
            else:
                magnitudes_no_shadow[i] = np.inf

    logger.info(f"Light curve generation completed for {num_epochs} epochs")
    if animate:
        logger.info(f"Animation data collected: {len(animation_data) if animation_data else 0} frames")

    if generate_no_shadow:
        return (magnitudes_shadowed, total_flux_shadowed,
                magnitudes_no_shadow, total_flux_no_shadow, observer_distances_out, animation_data)
    else:
        return (magnitudes_shadowed, total_flux_shadowed, None, None, observer_distances_out, animation_data)
