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
from dataclasses import dataclass
from typing import List, Tuple, Union

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


def compute_component_inertia(
    mesh: trimesh.Trimesh, mass: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the inertia tensor for a single STL component with given mass.

    Assumes homogeneous mass distribution throughout the volume. The inertia
    tensor is computed about the component's center of mass.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh representing the component geometry.
    mass : float
        The total mass of the component in consistent units.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        A tuple (inertia_tensor, center_of_mass) where:
        - inertia_tensor is a 3x3 numpy array (moment of inertia tensor about CoM)
        - center_of_mass is a (3,) numpy array with the CoM coordinates

    Notes
    -----
    For a homogeneous solid, the inertia tensor is computed by:
    1. Computing the volume using signed tetrahedra method
    2. Deriving density as mass / volume
    3. Computing volume integrals of position products over tetrahedra
    4. Scaling by density to get mass-based inertia tensor

    The inertia tensor elements are defined as:
    - Diagonal: I_xx = integral(y^2 + z^2 dm), etc.
    - Off-diagonal: I_xy = -integral(x*y dm), etc.

    If the mesh is not watertight, a warning is logged.
    """
    if not is_mesh_watertight(mesh):
        logger.warning(
            "Mesh is not watertight. Inertia calculation may be inaccurate."
        )

    # Compute volume
    volume = compute_mesh_volume(mesh)
    if volume < 1e-12:
        raise ValueError("Mesh has zero or near-zero volume, cannot compute inertia.")

    # Compute density from mass and volume
    density = mass / volume

    # Get mesh data
    vertices = mesh.vertices  # Shape (N_vertices, 3)
    faces = mesh.faces  # Shape (N_faces, 3)

    # Extract the three vertices for each face
    v0 = vertices[faces[:, 0]]  # Shape (N_faces, 3)
    v1 = vertices[faces[:, 1]]  # Shape (N_faces, 3)
    v2 = vertices[faces[:, 2]]  # Shape (N_faces, 3)

    # Compute center of mass using volume-weighted centroids of tetrahedra
    # Each tetrahedron formed by origin and a face has centroid at (v0 + v1 + v2) / 4
    # and signed volume (1/6) * v0 · (v1 × v2)
    cross_product = np.cross(v1, v2)
    signed_volumes = np.sum(v0 * cross_product, axis=1) / 6.0
    centroids = (v0 + v1 + v2) / 4.0  # Shape (N_faces, 3)

    # Volume-weighted center of mass
    total_signed_volume = np.sum(signed_volumes)
    center_of_mass = np.sum(
        signed_volumes[:, np.newaxis] * centroids, axis=0
    ) / total_signed_volume

    # Now compute the inertia tensor about the origin using canonical formulas
    # for tetrahedra, then translate to center of mass

    # For each tetrahedron with vertices at origin (0), v0, v1, v2,
    # we need to compute integrals of x^2, y^2, z^2, xy, xz, yz over the volume.
    # Using the formula for volume integrals over tetrahedra with one vertex at origin:
    #
    # For a tetrahedron with vertices (0, a, b, c), the volume integrals are:
    # integral(x^2) = V * (a_x^2 + b_x^2 + c_x^2 + a_x*b_x + a_x*c_x + b_x*c_x) / 10
    # integral(xy)  = V * (2*a_x*a_y + 2*b_x*b_y + 2*c_x*c_y +
    #                      a_x*b_y + a_y*b_x + a_x*c_y + a_y*c_x + b_x*c_y + b_y*c_x) / 20
    # where V is the signed volume of the tetrahedron.

    # Components of vertices
    ax, ay, az = v0[:, 0], v0[:, 1], v0[:, 2]
    bx, by, bz = v1[:, 0], v1[:, 1], v1[:, 2]
    cx, cy, cz = v2[:, 0], v2[:, 1], v2[:, 2]

    # Volume integrals for each tetrahedron (scaled by signed volume)
    # integral(x^2) over all tetrahedra
    int_x2 = signed_volumes * (ax**2 + bx**2 + cx**2 + ax*bx + ax*cx + bx*cx) / 10.0
    int_y2 = signed_volumes * (ay**2 + by**2 + cy**2 + ay*by + ay*cy + by*cy) / 10.0
    int_z2 = signed_volumes * (az**2 + bz**2 + cz**2 + az*bz + az*cz + bz*cz) / 10.0

    # integral(xy) over all tetrahedra
    int_xy = signed_volumes * (
        2*ax*ay + 2*bx*by + 2*cx*cy +
        ax*by + ay*bx + ax*cy + ay*cx + bx*cy + by*cx
    ) / 20.0
    int_xz = signed_volumes * (
        2*ax*az + 2*bx*bz + 2*cx*cz +
        ax*bz + az*bx + ax*cz + az*cx + bx*cz + bz*cx
    ) / 20.0
    int_yz = signed_volumes * (
        2*ay*az + 2*by*bz + 2*cy*cz +
        ay*bz + az*by + ay*cz + az*cy + by*cz + bz*cy
    ) / 20.0

    # Sum over all tetrahedra
    total_x2 = np.sum(int_x2)
    total_y2 = np.sum(int_y2)
    total_z2 = np.sum(int_z2)
    total_xy = np.sum(int_xy)
    total_xz = np.sum(int_xz)
    total_yz = np.sum(int_yz)

    # Inertia tensor about origin (with unit density)
    # I_xx = integral(y^2 + z^2 dV), I_xy = -integral(xy dV), etc.
    I_xx_origin = total_y2 + total_z2
    I_yy_origin = total_x2 + total_z2
    I_zz_origin = total_x2 + total_y2
    I_xy_origin = -total_xy
    I_xz_origin = -total_xz
    I_yz_origin = -total_yz

    # Scale by density to get mass-based inertia
    I_xx_origin *= density
    I_yy_origin *= density
    I_xz_origin *= density
    I_zz_origin *= density
    I_xy_origin *= density
    I_yz_origin *= density

    inertia_origin = np.array([
        [I_xx_origin, I_xy_origin, I_xz_origin],
        [I_xy_origin, I_yy_origin, I_yz_origin],
        [I_xz_origin, I_yz_origin, I_zz_origin]
    ], dtype=np.float64)

    # Translate from origin to center of mass using parallel axis theorem (inverse)
    # I_cm = I_origin - m * (d^2 * I - outer(d, d))
    # where d is the displacement from origin to CoM
    d = center_of_mass
    d_squared = np.dot(d, d)
    parallel_axis_term = mass * (d_squared * np.eye(3) - np.outer(d, d))
    inertia_cm = inertia_origin - parallel_axis_term

    return inertia_cm, center_of_mass.astype(np.float64)


def translate_inertia(
    I_cm: NDArray[np.float64],
    mass: float,
    displacement: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Translate an inertia tensor from center of mass to a new reference point.

    Applies the parallel axis theorem to translate a component's inertia tensor
    (given about its center of mass) to the body frame origin.

    Parameters
    ----------
    I_cm : NDArray[np.float64]
        The 3x3 inertia tensor about the component's center of mass.
    mass : float
        The mass of the component.
    displacement : NDArray[np.float64]
        Vector from body frame origin to component center of mass, shape (3,).
        This is the translation vector d in the parallel axis theorem.

    Returns
    -------
    NDArray[np.float64]
        The 3x3 inertia tensor about the body frame origin.

    Notes
    -----
    The parallel axis theorem states:
        I_body = I_cm + m * (d² * I - outer(d, d))

    where:
    - I_body is the inertia tensor about the new reference point (body origin)
    - I_cm is the inertia tensor about the center of mass
    - m is the mass of the component
    - d is the displacement vector from body origin to center of mass
    - d² = dot(d, d) is the squared magnitude of d
    - I is the 3x3 identity matrix
    - outer(d, d) is the outer product of d with itself

    This is the forward parallel axis theorem (moving away from CoM).
    To go from a point to CoM, subtract the parallel axis term instead.
    """
    displacement = np.asarray(displacement, dtype=np.float64)
    d_squared = np.dot(displacement, displacement)

    # Parallel axis term: m * (d² * I - outer(d, d))
    parallel_axis_term = mass * (d_squared * np.eye(3) - np.outer(displacement, displacement))

    # I_body = I_cm + parallel_axis_term
    I_body = np.asarray(I_cm, dtype=np.float64) + parallel_axis_term

    return I_body


@dataclass
class InertiaResult:
    """
    Result container for satellite inertia calculation.

    Contains all computed inertia properties from a multi-component satellite
    model, including the total mass, center of mass location, full inertia
    tensor, and principal axis decomposition.

    Attributes
    ----------
    total_mass : float
        Total mass of the satellite (sum of all component masses).
    center_of_mass : NDArray[np.float64]
        Position of the satellite center of mass in body frame, shape (3,).
    inertia_tensor : NDArray[np.float64]
        3x3 inertia tensor about the satellite center of mass.
    principal_moments : NDArray[np.float64]
        Principal moments of inertia (eigenvalues), shape (3,), sorted ascending.
    principal_axes : NDArray[np.float64]
        Principal axes (eigenvectors) as columns of a 3x3 matrix.
        principal_axes[:, i] corresponds to principal_moments[i].
    """

    total_mass: float
    center_of_mass: NDArray[np.float64]
    inertia_tensor: NDArray[np.float64]
    principal_moments: NDArray[np.float64]
    principal_axes: NDArray[np.float64]


@dataclass
class STLComponent:
    """
    An STL component with its mesh, position, and mass.

    This is a convenience wrapper for passing STL data to the inertia
    calculator. It allows the same mesh to be used multiple times at
    different positions (e.g., symmetric components).

    Attributes
    ----------
    mesh : trimesh.Trimesh
        The triangular mesh geometry of the component.
    position : NDArray[np.float64]
        Position of the component origin in body frame, shape (3,).
    mass : float
        Mass of this component instance.
    """

    mesh: trimesh.Trimesh
    position: NDArray[np.float64]
    mass: float


def compute_inertia_from_stl(
    stl_components: List[Union[STLComponent, Tuple[trimesh.Trimesh, NDArray[np.float64], float]]],
    masses: Union[List[float], None] = None,
) -> InertiaResult:
    """
    Compute the total satellite inertia from multiple STL components with masses.

    Computes the combined inertia tensor of a satellite composed of multiple
    STL mesh components, each with its own position and mass. The same STL file
    can be used multiple times at different positions (e.g., for symmetric parts).

    Parameters
    ----------
    stl_components : List[Union[STLComponent, Tuple[trimesh.Trimesh, NDArray, float]]]
        List of STL components. Each element can be either:
        - An STLComponent dataclass instance
        - A tuple of (mesh, position, mass) where:
          - mesh: trimesh.Trimesh object
          - position: component origin position in body frame, shape (3,)
          - mass: component mass
    masses : List[float] | None, optional
        Alternative way to specify masses. If provided, must have same length
        as stl_components and will override masses in the components.
        Deprecated: prefer using STLComponent or tuples with mass included.

    Returns
    -------
    InertiaResult
        Dataclass containing:
        - total_mass: Sum of all component masses
        - center_of_mass: Satellite CoM in body frame
        - inertia_tensor: 3x3 inertia tensor about satellite CoM
        - principal_moments: Eigenvalues (principal moments)
        - principal_axes: Eigenvectors (principal axes) as column vectors

    Raises
    ------
    ValueError
        If stl_components is empty or if masses length doesn't match components.

    Notes
    -----
    The algorithm:
    1. Compute each component's inertia tensor about its own center of mass
    2. Translate each component's CoM to its position in body frame
    3. Translate each component's inertia to body frame origin using parallel axis theorem
    4. Sum all contributions to get total inertia about body origin
    5. Compute total satellite CoM from mass-weighted component positions
    6. Translate total inertia from body origin to satellite CoM
    7. Compute principal axes via eigendecomposition

    Examples
    --------
    Using STLComponent dataclass:

    >>> mesh = trimesh.load("component.stl")
    >>> components = [
    ...     STLComponent(mesh=mesh, position=np.array([1, 0, 0]), mass=10.0),
    ...     STLComponent(mesh=mesh, position=np.array([-1, 0, 0]), mass=10.0),
    ... ]
    >>> result = compute_inertia_from_stl(components)

    Using tuples:

    >>> components = [
    ...     (mesh1, np.array([0, 0, 0]), 50.0),
    ...     (mesh2, np.array([2, 0, 0]), 5.0),
    ... ]
    >>> result = compute_inertia_from_stl(components)
    """
    if not stl_components:
        raise ValueError("stl_components list cannot be empty")

    # Normalize input to list of (mesh, position, mass) tuples
    normalized_components: List[Tuple[trimesh.Trimesh, NDArray[np.float64], float]] = []

    for i, comp in enumerate(stl_components):
        if isinstance(comp, STLComponent):
            mesh = comp.mesh
            position = np.asarray(comp.position, dtype=np.float64)
            mass = comp.mass
        elif isinstance(comp, tuple) and len(comp) == 3:
            mesh, pos, m = comp
            position = np.asarray(pos, dtype=np.float64)
            mass = float(m)
        else:
            raise ValueError(
                f"Component {i} must be STLComponent or (mesh, position, mass) tuple"
            )

        # Override mass if masses list is provided
        if masses is not None:
            if len(masses) != len(stl_components):
                raise ValueError(
                    f"masses list length ({len(masses)}) must match "
                    f"stl_components length ({len(stl_components)})"
                )
            mass = float(masses[i])

        normalized_components.append((mesh, position, mass))

    # Compute total mass
    total_mass = sum(m for _, _, m in normalized_components)

    if total_mass < 1e-12:
        raise ValueError("Total mass is zero or near-zero")

    # Step 1-3: For each component, compute inertia and translate to body origin
    # Also track component CoMs for total satellite CoM calculation
    component_coms: List[NDArray[np.float64]] = []
    component_masses: List[float] = []
    total_inertia_origin = np.zeros((3, 3), dtype=np.float64)

    for mesh, position, mass in normalized_components:
        # Compute component inertia about its own center of mass
        I_cm, com_local = compute_component_inertia(mesh, mass)

        # Component CoM in body frame = local CoM + component position
        com_body = com_local + position
        component_coms.append(com_body)
        component_masses.append(mass)

        # Translate inertia from component CoM to body origin
        I_at_origin = translate_inertia(I_cm, mass, com_body)
        total_inertia_origin += I_at_origin

    # Step 4-5: Compute total satellite center of mass
    center_of_mass = np.zeros(3, dtype=np.float64)
    for com, m in zip(component_coms, component_masses):
        center_of_mass += m * com
    center_of_mass /= total_mass

    # Step 6: Translate total inertia from body origin to satellite CoM
    # This is the inverse parallel axis theorem: I_cm = I_origin - m*(d²I - outer(d,d))
    d = center_of_mass
    d_squared = np.dot(d, d)
    parallel_axis_term = total_mass * (d_squared * np.eye(3) - np.outer(d, d))
    inertia_tensor = total_inertia_origin - parallel_axis_term

    # Step 7: Compute principal axes via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

    # eigh returns eigenvalues in ascending order
    principal_moments = eigenvalues.astype(np.float64)
    principal_axes = eigenvectors.astype(np.float64)

    return InertiaResult(
        total_mass=total_mass,
        center_of_mass=center_of_mass,
        inertia_tensor=inertia_tensor,
        principal_moments=principal_moments,
        principal_axes=principal_axes,
    )
