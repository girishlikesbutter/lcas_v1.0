"""
Facet data extraction for vectorized computation.

Extracts satellite facet data into contiguous NumPy arrays for efficient
vectorized BRDF and shadow calculations. This module enables ~50-200x speedup
on light curve generation by eliminating Python loop overhead.

Key functions:
- extract_facet_arrays: Extract all facet data into contiguous arrays
- apply_articulation_to_arrays: Apply rotation matrices to normals/centers
- lit_status_to_flat_array: Convert lit_status_dict to flat array
- flat_array_to_lit_status_dict: Convert flat array back to dict format
"""

import numpy as np
import quaternion
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

from ..io.stl_loader import Satellite, Component

logger = logging.getLogger(__name__)


@dataclass
class FacetArrays:
    """
    Pre-extracted facet data in contiguous arrays.

    All arrays have shape (total_facets, ...) where total_facets
    is the sum of facets across all components.

    Attributes:
        normals: Unit normal vectors (M, 3)
        areas: Facet areas (M,)
        centers: Facet centroid positions (M, 3)
        vertices: Facet vertex positions (M, 3, 3) - 3 vertices per triangle facet
        r_d: Diffuse reflectivity (M,)
        r_s: Specular reflectivity (M,)
        n_phong: Phong exponent (M,)
        component_slices: Maps component name to index slice
        component_names: Ordered list of component names
        total_facets: Total number of facets M
    """
    normals: np.ndarray       # (M, 3)
    areas: np.ndarray         # (M,)
    centers: np.ndarray       # (M, 3)
    vertices: np.ndarray      # (M, 3, 3) - [facet_idx, vertex_idx, xyz]
    r_d: np.ndarray           # (M,)
    r_s: np.ndarray           # (M,)
    n_phong: np.ndarray       # (M,)
    component_slices: Dict[str, slice]
    component_names: List[str]
    total_facets: int


def extract_facet_arrays(satellite: Satellite) -> FacetArrays:
    """
    Extract all facet data from satellite into contiguous arrays.

    This pre-processing step enables vectorized BRDF and shadow calculations
    by organizing facet data for efficient NumPy operations.

    Args:
        satellite: Satellite model with components and facets

    Returns:
        FacetArrays containing all facet data in contiguous arrays
    """
    # Count total facets
    total_facets = sum(len(c.facets) for c in satellite.components)

    if total_facets == 0:
        logger.warning("Satellite has no facets to extract")
        return FacetArrays(
            normals=np.zeros((0, 3), dtype=np.float64),
            areas=np.zeros(0, dtype=np.float64),
            centers=np.zeros((0, 3), dtype=np.float64),
            vertices=np.zeros((0, 3, 3), dtype=np.float64),
            r_d=np.zeros(0, dtype=np.float64),
            r_s=np.zeros(0, dtype=np.float64),
            n_phong=np.zeros(0, dtype=np.float64),
            component_slices={},
            component_names=[],
            total_facets=0
        )

    # Pre-allocate arrays
    normals = np.zeros((total_facets, 3), dtype=np.float64)
    areas = np.zeros(total_facets, dtype=np.float64)
    centers = np.zeros((total_facets, 3), dtype=np.float64)
    vertices = np.zeros((total_facets, 3, 3), dtype=np.float64)  # M facets x 3 verts x 3 coords
    r_d = np.zeros(total_facets, dtype=np.float64)
    r_s = np.zeros(total_facets, dtype=np.float64)
    n_phong = np.zeros(total_facets, dtype=np.float64)

    component_slices = {}
    component_names = []

    idx = 0
    for component in satellite.components:
        start_idx = idx
        component_names.append(component.name)

        for facet in component.facets:
            normals[idx] = facet.normal
            areas[idx] = facet.area
            # Store vertices and calculate center
            if len(facet.vertices) >= 3:
                vertices[idx, 0] = facet.vertices[0]
                vertices[idx, 1] = facet.vertices[1]
                vertices[idx, 2] = facet.vertices[2]
                centers[idx] = np.mean(facet.vertices[:3], axis=0)
            r_d[idx] = facet.material_properties.r_d
            r_s[idx] = facet.material_properties.r_s
            n_phong[idx] = facet.material_properties.n_phong
            idx += 1

        component_slices[component.name] = slice(start_idx, idx)

    logger.debug(f"Extracted {total_facets} facets from {len(component_names)} components")

    return FacetArrays(
        normals=normals,
        areas=areas,
        centers=centers,
        vertices=vertices,
        r_d=r_d,
        r_s=r_s,
        n_phong=n_phong,
        component_slices=component_slices,
        component_names=component_names,
        total_facets=total_facets
    )


def _get_component_transform(
    component: Component,
    component_matrices: Dict[str, np.ndarray],
    epoch_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the combined rotation matrix and body position for a component.

    This is the single source of truth for how articulation transforms
    are computed. Both apply_articulation_to_arrays and
    apply_articulation_to_vertices use this function.

    Args:
        component: Satellite component
        component_matrices: Dict mapping component names to (N, 4, 4) matrices
        epoch_idx: Current epoch index

    Returns:
        Tuple of (combined_rot, body_pos):
        - combined_rot: (3, 3) rotation matrix (body_frame @ articulation)
        - body_pos: (3,) translation vector
    """
    # Get body frame transform from component's relative orientation/position
    body_rot = quaternion.as_rotation_matrix(component.relative_orientation)
    body_pos = component.relative_position

    # Get articulation rotation for this epoch (if any)
    if component.name in component_matrices:
        matrices_array = component_matrices[component.name]
        if epoch_idx < len(matrices_array):
            art_matrix = matrices_array[epoch_idx]
            art_rot = art_matrix[:3, :3]
        else:
            art_rot = np.eye(3)
    else:
        art_rot = np.eye(3)

    # Combined rotation: body_frame @ articulation
    combined_rot = body_rot @ art_rot

    return combined_rot, body_pos


def apply_articulation_to_arrays(
    facet_arrays: FacetArrays,
    component_matrices: Dict[str, np.ndarray],
    epoch_idx: int,
    satellite: Satellite
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply articulation transforms to facet normals and centers.

    Instead of copying the entire satellite structure, this applies
    rotation matrices directly to the extracted arrays. This is more
    efficient for vectorized computation.

    Args:
        facet_arrays: Pre-extracted facet data
        component_matrices: Dict mapping component names to (N, 4, 4) matrices
        epoch_idx: Current epoch index
        satellite: Original satellite for body frame transforms

    Returns:
        Tuple of (transformed_normals, transformed_centers) both (M, 3)
    """
    transformed_normals = np.zeros_like(facet_arrays.normals)
    transformed_centers = np.zeros_like(facet_arrays.centers)

    for component in satellite.components:
        comp_slice = facet_arrays.component_slices.get(component.name)
        if comp_slice is None:
            continue

        # Get transform from shared helper
        combined_rot, body_pos = _get_component_transform(
            component, component_matrices, epoch_idx
        )

        # Transform normals (rotation only, vectorized)
        local_normals = facet_arrays.normals[comp_slice]
        transformed_normals[comp_slice] = (combined_rot @ local_normals.T).T

        # Transform centers (rotation + translation, vectorized)
        local_centers = facet_arrays.centers[comp_slice]
        transformed_centers[comp_slice] = (combined_rot @ local_centers.T).T + body_pos

    return transformed_normals, transformed_centers


def apply_articulation_to_vertices(
    facet_arrays: FacetArrays,
    component_matrices: Dict[str, np.ndarray],
    epoch_idx: int,
    satellite: Satellite
) -> np.ndarray:
    """
    Apply articulation transforms to facet vertices for animation.

    Uses the same transformation logic as apply_articulation_to_arrays
    (via shared _get_component_transform helper) but returns transformed
    vertices in a flat format suitable for Plotly mesh visualization.

    Args:
        facet_arrays: Pre-extracted facet data including vertices
        component_matrices: Dict mapping component names to (N, 4, 4) matrices
        epoch_idx: Current epoch index
        satellite: Original satellite for body frame transforms

    Returns:
        Flattened vertex array (M*3, 3) where M is total facets.
        Vertices are ordered: [facet0_v0, facet0_v1, facet0_v2, facet1_v0, ...]
    """
    M = facet_arrays.total_facets
    # Output: flattened vertices for Plotly (M*3 vertices, each with xyz)
    transformed_vertices_flat = np.zeros((M * 3, 3), dtype=np.float64)

    for component in satellite.components:
        comp_slice = facet_arrays.component_slices.get(component.name)
        if comp_slice is None:
            continue

        # Get transform from shared helper (same as apply_articulation_to_arrays)
        combined_rot, body_pos = _get_component_transform(
            component, component_matrices, epoch_idx
        )

        # Get local vertices for this component: (num_facets_in_comp, 3, 3)
        local_verts = facet_arrays.vertices[comp_slice]  # shape: (N_comp, 3, 3)

        # Transform each vertex: rotated + translated
        # Reshape to (N_comp * 3, 3) for batch transformation
        local_verts_flat = local_verts.reshape(-1, 3)  # (N_comp*3, 3)
        transformed = (combined_rot @ local_verts_flat.T).T + body_pos  # (N_comp*3, 3)

        # Store in output array at correct position
        out_start = comp_slice.start * 3
        out_end = comp_slice.stop * 3
        transformed_vertices_flat[out_start:out_end] = transformed

    return transformed_vertices_flat


def lit_status_to_flat_array(
    lit_status_dict: Dict[str, np.ndarray],
    facet_arrays: FacetArrays,
    epoch_idx: int
) -> np.ndarray:
    """
    Convert lit_status_dict to flat boolean array for single epoch.

    Args:
        lit_status_dict: Dict mapping component names to (N, num_facets) bool arrays
        facet_arrays: FacetArrays for index mapping
        epoch_idx: Current epoch index

    Returns:
        Flat boolean array (M,) where M is total facets
    """
    lit_flat = np.zeros(facet_arrays.total_facets, dtype=bool)

    for comp_name, comp_slice in facet_arrays.component_slices.items():
        if comp_name in lit_status_dict:
            lit_data = lit_status_dict[comp_name]
            # Handle both shapes: (N, num_facets) and mismatched sizes
            if epoch_idx < lit_data.shape[0]:
                slice_size = comp_slice.stop - comp_slice.start
                data_size = min(slice_size, lit_data.shape[1])
                lit_flat[comp_slice.start:comp_slice.start + data_size] = lit_data[epoch_idx, :data_size]

    return lit_flat


def flat_array_to_lit_status_dict(
    lit_flat: np.ndarray,
    facet_arrays: FacetArrays,
    num_epochs: int = 1
) -> Dict[str, np.ndarray]:
    """
    Convert flat lit status array back to dict format.

    Args:
        lit_flat: Boolean array (M,) or (N, M)
        facet_arrays: FacetArrays for index mapping
        num_epochs: Number of epochs if 2D array

    Returns:
        Dict mapping component names to boolean arrays
    """
    result = {}

    if lit_flat.ndim == 1:
        lit_flat = lit_flat.reshape(1, -1)

    for comp_name, comp_slice in facet_arrays.component_slices.items():
        result[comp_name] = lit_flat[:, comp_slice].copy()

    return result


def extract_facet_arrays_with_transform(
    satellite: Satellite,
    component_matrices: Dict[str, np.ndarray],
    epoch_idx: int
) -> FacetArrays:
    """
    Extract facet arrays with articulation already applied.

    Convenience function that extracts and transforms in one step.
    Useful when you only need data for a single epoch.

    Args:
        satellite: Satellite model
        component_matrices: Pre-computed rotation matrices
        epoch_idx: Current epoch index

    Returns:
        FacetArrays with transformed normals and centers
    """
    # First extract base arrays
    facet_arrays = extract_facet_arrays(satellite)

    # Then apply articulation
    transformed_normals, transformed_centers = apply_articulation_to_arrays(
        facet_arrays, component_matrices, epoch_idx, satellite
    )

    # Return new FacetArrays with transformed data
    # Note: vertices kept as base (untransformed) - use apply_articulation_to_vertices
    # separately if you need transformed vertices for animation
    return FacetArrays(
        normals=transformed_normals,
        areas=facet_arrays.areas,
        centers=transformed_centers,
        vertices=facet_arrays.vertices,
        r_d=facet_arrays.r_d,
        r_s=facet_arrays.r_s,
        n_phong=facet_arrays.n_phong,
        component_slices=facet_arrays.component_slices,
        component_names=facet_arrays.component_names,
        total_facets=facet_arrays.total_facets
    )
