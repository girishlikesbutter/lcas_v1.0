"""
Inertia tensor calculation from STL mesh geometry.

This module provides functions to compute:
- Mesh volume using the signed tetrahedra (divergence theorem) method
- Mesh watertightness validation
- Component inertia tensors assuming homogeneous mass distribution
- Full satellite inertia from multiple STL components

All calculations assume the mesh is in a consistent coordinate frame.
"""

import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import trimesh

logger = logging.getLogger(__name__)


def compute_mesh_volume(mesh: trimesh.Trimesh) -> float:
    """
    Compute the volume of a mesh using the signed tetrahedra method.

    This method uses the divergence theorem to compute volume by summing
    the signed volumes of tetrahedra formed between each face and the origin.
    For each triangular face with vertices (v0, v1, v2), the signed volume
    contribution is (1/6) * v0 · (v1 × v2).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to compute volume for.

    Returns
    -------
    float
        The computed volume. For a properly oriented closed mesh, this will
        be positive. For non-watertight meshes, the result may be inaccurate.

    Notes
    -----
    If the mesh is not watertight (closed), a warning is logged and the
    computed volume may not be physically meaningful.
    """
    if not is_mesh_watertight(mesh):
        logger.warning(
            "Mesh is not watertight (not closed). Volume calculation may be inaccurate."
        )

    # Get vertices and faces
    vertices = mesh.vertices  # Shape (N_vertices, 3)
    faces = mesh.faces  # Shape (N_faces, 3), indices into vertices

    # Extract the three vertices for each face
    v0 = vertices[faces[:, 0]]  # Shape (N_faces, 3)
    v1 = vertices[faces[:, 1]]  # Shape (N_faces, 3)
    v2 = vertices[faces[:, 2]]  # Shape (N_faces, 3)

    # Compute signed volume using the scalar triple product:
    # signed_volume = (1/6) * v0 · (v1 × v2)
    # This is equivalent to the determinant of the 3x3 matrix [v0, v1, v2]
    cross_product = np.cross(v1, v2)  # v1 × v2
    signed_volumes = np.sum(v0 * cross_product, axis=1) / 6.0

    # Total volume is the sum of all signed contributions
    total_volume = np.abs(np.sum(signed_volumes))

    return float(total_volume)


def is_mesh_watertight(mesh: trimesh.Trimesh) -> bool:
    """
    Check if a mesh is watertight (closed).

    A watertight mesh has no holes - each edge is shared by exactly two faces.
    This is required for accurate volume and inertia calculations.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to check.

    Returns
    -------
    bool
        True if the mesh is watertight (closed), False otherwise.
    """
    # trimesh provides a built-in watertight check
    return bool(mesh.is_watertight)
