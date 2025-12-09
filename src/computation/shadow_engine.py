#!/usr/bin/env python3
"""
Shadow Engine
=============

Unified module for shadow computation using ray tracing.
Contains the shadow engine class and high-level computation functions.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ..io.stl_loader import Satellite

logger = logging.getLogger(__name__)


class ShadowEngine:
    """
    Shadow engine for ray-based self-occlusion computation.

    Uses batched ray-mesh intersection for efficient shadow calculation.
    """

    def __init__(self):
        """Initialize shadow engine."""
        self.stats = {
            'total_rays': 0,
            'total_time': 0.0
        }
        logger.info("Shadow Engine initialized")

    def _calculate_articulation_rotation(self, component, sun_vector_body, offset_deg=0.0,
                                        articulation_engine=None, earth_vector_body=None, epoch=None):
        """
        Calculate rotation matrix for component articulation.

        Args:
            component: Component to articulate
            sun_vector_body: Sun direction vector in body frame (normalized)
            offset_deg: Additional offset angle in degrees for testing
            articulation_engine: ArticulationEngine instance (required for articulation)
            earth_vector_body: Earth direction vector in body frame (optional)
            epoch: Current epoch time (optional)

        Returns:
            4x4 homogeneous transformation matrix for articulation
        """
        if articulation_engine is not None:
            rotation_matrix = articulation_engine.calculate_articulation_rotation(
                component, sun_vector_body,
                earth_vector_body if earth_vector_body is not None else np.array([0, 0, -1]),
                epoch if epoch is not None else 0.0,
                offset_deg
            )
            if rotation_matrix is not None:
                return rotation_matrix

        # No articulation for this component
        return np.eye(4)

    def create_shadow_mesh_with_explicit_angles(
        self,
        satellite,
        component_angles: Dict[str, float]
    ) -> Optional[trimesh.Trimesh]:
        """
        Create a single shadow mesh with explicit articulation angles.

        Args:
            satellite: Satellite model
            component_angles: Dict mapping component names to angles in degrees

        Returns:
            Trimesh object with articulation applied
        """
        import quaternion

        component_meshes = []

        for component in satellite.components:
            if not component.facets:
                continue

            # Build mesh from facets
            vertices_list = []
            faces_list = []
            vertex_offset = 0

            for facet in component.facets:
                vertices_list.extend(facet.vertices)
                faces_list.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                vertex_offset += 3

            if not faces_list:
                continue

            # Create local mesh
            local_mesh = trimesh.Trimesh(
                vertices=np.array(vertices_list),
                faces=np.array(faces_list),
                process=False
            )

            # Transform to body frame
            transform = np.eye(4)
            transform[:3, :3] = quaternion.as_rotation_matrix(component.relative_orientation)
            transform[:3, 3] = component.relative_position

            # Apply articulation if component has an angle specified
            if component.name in component_angles:
                angle_deg = component_angles[component.name]
                angle_rad = np.radians(angle_deg)

                # Create rotation matrix for articulation (around Z-axis typically)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                articulation_rot_3x3 = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])

                # Apply articulation to rotation part only (match ArticulationTransformer)
                body_rot_3x3 = transform[:3, :3]
                transform[:3, :3] = body_rot_3x3 @ articulation_rot_3x3  # Combined rotation

            # Apply transform and add to list
            local_mesh.apply_transform(transform)
            component_meshes.append(local_mesh)

        # Combine all component meshes
        if component_meshes:
            combined_mesh = trimesh.util.concatenate(component_meshes)
            return combined_mesh

        return None

    def _identify_static_components(
        self,
        satellite,
        explicit_component_matrices: Dict[str, List[np.ndarray]]
    ) -> Tuple[List, List]:
        """
        Identify which components are static vs articulated based on provided matrices.

        Args:
            satellite: Satellite model
            explicit_component_matrices: Dict mapping component names to lists of transformation matrices

        Returns:
            Tuple of (static_components, articulated_components)
        """
        articulated_names = set(explicit_component_matrices.keys())
        static_components = []
        articulated_components = []

        for component in satellite.components:
            if component.name in articulated_names:
                articulated_components.append(component)
            else:
                static_components.append(component)

        logger.debug(f"Identified {len(static_components)} static and {len(articulated_components)} articulated components")
        return static_components, articulated_components

    def _create_static_mesh(
        self,
        static_components: List
    ) -> Optional[trimesh.Trimesh]:
        """
        Create a single mesh for all static components with body frame transforms applied.
        This mesh will be reused across all epochs.

        Args:
            static_components: List of static satellite components

        Returns:
            Combined trimesh of all static components, or None if no static components
        """
        import quaternion

        if not static_components:
            return None

        component_meshes = []

        for component in static_components:
            if not component.facets:
                continue

            # Build mesh from facets
            vertices_list = []
            faces_list = []
            vertex_offset = 0

            for facet in component.facets:
                vertices_list.extend(facet.vertices)
                faces_list.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                vertex_offset += 3

            if not faces_list:
                continue

            # Create local mesh
            local_mesh = trimesh.Trimesh(
                vertices=np.array(vertices_list),
                faces=np.array(faces_list),
                process=False
            )

            # Apply only body frame transform (no articulation)
            transform = np.eye(4)
            transform[:3, :3] = quaternion.as_rotation_matrix(component.relative_orientation)
            transform[:3, 3] = component.relative_position

            local_mesh.apply_transform(transform)
            component_meshes.append(local_mesh)

        if component_meshes:
            combined_mesh = trimesh.util.concatenate(component_meshes)
            logger.debug(f"Created static mesh with {len(combined_mesh.vertices)} vertices")
            return combined_mesh

        return None

    def _create_articulated_mesh_for_epoch(
        self,
        articulated_components: List,
        component_matrices: Dict[str, np.ndarray],
        epoch_idx: int,
        explicit_component_matrices: Dict[str, List[np.ndarray]]
    ) -> Optional[trimesh.Trimesh]:
        """
        Create mesh for articulated components at a specific epoch.

        Args:
            articulated_components: List of articulated satellite components
            component_matrices: Pre-extracted matrices for this epoch
            epoch_idx: Current epoch index
            explicit_component_matrices: Full matrix arrays for validation

        Returns:
            Combined trimesh of articulated components for this epoch
        """
        import quaternion

        component_meshes = []

        for component in articulated_components:
            if not component.facets:
                continue

            # Build mesh from facets
            vertices_list = []
            faces_list = []
            vertex_offset = 0

            for facet in component.facets:
                vertices_list.extend(facet.vertices)
                faces_list.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                vertex_offset += 3

            if not faces_list:
                continue

            # Create local mesh
            local_mesh = trimesh.Trimesh(
                vertices=np.array(vertices_list),
                faces=np.array(faces_list),
                process=False
            )

            # Transform to body frame
            transform = np.eye(4)
            transform[:3, :3] = quaternion.as_rotation_matrix(component.relative_orientation)
            transform[:3, 3] = component.relative_position

            # Apply articulation matrix for this epoch
            if component.name in component_matrices:
                articulation_transform = component_matrices[component.name]
                # Apply articulation to rotation part only
                body_rot_3x3 = transform[:3, :3]
                articulation_rot_3x3 = articulation_transform[:3, :3]
                transform[:3, :3] = body_rot_3x3 @ articulation_rot_3x3

            local_mesh.apply_transform(transform)
            component_meshes.append(local_mesh)

        if component_meshes:
            return trimesh.util.concatenate(component_meshes)

        return None

    def create_shadow_meshes_with_explicit_matrices_optimized(
        self,
        satellite,
        sun_vectors_body: np.ndarray,
        explicit_component_matrices: Dict[str, List[np.ndarray]],
        show_progress: bool = True
    ) -> List[Optional[trimesh.Trimesh]]:
        """
        Optimized mesh creation that reuses static component meshes.

        Separates static and articulated components, creating the static
        mesh once and only updating articulated components per epoch.

        Args:
            satellite: Satellite model
            sun_vectors_body: Array of sun vectors in body frame
            explicit_component_matrices: Dict mapping component names to lists of transformation matrices
            show_progress: If True, display tqdm progress bar (default True)

        Returns:
            List of trimesh objects, one per epoch
        """
        logger.info(f"Creating {len(sun_vectors_body)} shadow meshes with optimized pre-computed matrices...")
        start_time = time.time()

        # Separate static and articulated components
        static_components, articulated_components = self._identify_static_components(
            satellite, explicit_component_matrices
        )

        # Create static mesh once (will be reused for all epochs)
        static_mesh = self._create_static_mesh(static_components)
        static_creation_time = time.time() - start_time
        logger.info(f"Created static mesh in {static_creation_time:.3f}s")

        num_epochs = len(sun_vectors_body)
        shadow_meshes = []
        articulated_start = time.time()

        # Create iterator with optional progress bar
        epoch_iter = range(num_epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Creating shadow meshes", unit="epoch", mininterval=0.5)

        for epoch_idx in epoch_iter:
            # Get matrices for this epoch for each articulated component
            epoch_matrices = {}
            for comp_name, matrices_array in explicit_component_matrices.items():
                if epoch_idx < len(matrices_array):
                    epoch_matrices[comp_name] = matrices_array[epoch_idx]

            # Create articulated mesh for this epoch
            articulated_mesh = self._create_articulated_mesh_for_epoch(
                articulated_components, epoch_matrices, epoch_idx, explicit_component_matrices
            )

            # Combine static and articulated meshes
            if static_mesh and articulated_mesh:
                # Make a copy of static mesh vertices/faces to avoid modifying the original
                combined_mesh = trimesh.util.concatenate([static_mesh, articulated_mesh])
            elif static_mesh:
                # All components are static
                combined_mesh = static_mesh
            elif articulated_mesh:
                # All components are articulated
                combined_mesh = articulated_mesh
            else:
                raise RuntimeError(f"Could not create shadow mesh for epoch {epoch_idx}")

            shadow_meshes.append(combined_mesh)

        articulated_time = time.time() - articulated_start
        total_time = time.time() - start_time

        logger.info(f"Mesh creation complete: static={static_creation_time:.3f}s, articulated={articulated_time:.3f}s, total={total_time:.3f}s")
        logger.info(f"  Static components: {len(static_components)}, Articulated components: {len(articulated_components)}")

        return shadow_meshes

    def create_shadow_meshes_with_explicit_angles(
        self,
        satellite,
        sun_vectors_body: np.ndarray,
        explicit_component_angles: Dict[str, np.ndarray]
    ) -> List[trimesh.Trimesh]:
        """
        Create shadow meshes using pre-computed explicit angles for each component.

        Args:
            satellite: Satellite model
            sun_vectors_body: Sun vectors in body frame (N, 3) - only for logging
            explicit_component_angles: Pre-computed angles per component per epoch
                Dict[component_name, array of angles in degrees]

        Returns:
            List of trimesh objects, one per epoch
        """
        logger.info(f"Creating {len(sun_vectors_body)} shadow meshes with explicit angles...")
        start_time = time.time()

        num_epochs = len(sun_vectors_body)
        shadow_meshes = []

        for epoch_idx in range(num_epochs):
            # Get angles for this epoch for each component
            epoch_angles = {}
            for comp_name, angles_array in explicit_component_angles.items():
                if epoch_idx < len(angles_array):
                    epoch_angles[comp_name] = angles_array[epoch_idx]

            # Create mesh with explicit angles for this epoch
            mesh = self.create_shadow_mesh_with_explicit_angles(satellite, epoch_angles)
            if mesh is None:
                raise RuntimeError(f"Could not create shadow mesh for epoch {epoch_idx}")
            shadow_meshes.append(mesh)

            if (epoch_idx + 1) % 50 == 0:
                logger.info(f"  Created {epoch_idx + 1}/{num_epochs} meshes...")

        mesh_time = time.time() - start_time
        logger.info(f"Created {num_epochs} meshes in {mesh_time:.3f}s")

        return shadow_meshes

    def create_shadow_mesh(self, satellite, sun_vector_body=None, articulation_offset=0.0,
                          articulation_engine=None, earth_vector_body=None, epoch=None) -> Optional[trimesh.Trimesh]:
        """Create optimized mesh for shadow ray tracing."""
        import quaternion

        logger.info("Creating shadow mesh...")
        start_time = time.time()

        component_meshes = []

        for component in satellite.components:
            if not component.facets:
                continue

            # Build mesh from facets
            vertices_list = []
            faces_list = []
            vertex_offset = 0

            for facet in component.facets:
                vertices_list.extend(facet.vertices)
                faces_list.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                vertex_offset += 3

            if not faces_list:
                continue

            # Create local mesh
            local_mesh = trimesh.Trimesh(
                vertices=np.array(vertices_list),
                faces=np.array(faces_list),
                process=False
            )

            # Transform to body frame
            transform = np.eye(4)
            transform[:3, :3] = quaternion.as_rotation_matrix(component.relative_orientation)
            transform[:3, 3] = component.relative_position

            # Apply articulation if sun vector and articulation engine are provided
            if sun_vector_body is not None and articulation_engine and articulation_engine.is_component_articulated(component):
                articulation_transform = self._calculate_articulation_rotation(
                    component, sun_vector_body, articulation_offset,
                    articulation_engine, earth_vector_body, epoch)

                # Debug: Log articulation application
                angle_rad = np.arctan2(articulation_transform[1, 0], articulation_transform[0, 0])
                angle_deg = np.degrees(angle_rad)
                logger.info(f"  Applying articulation to {component.name}: {angle_deg:.2f}° "
                           f"(sun={sun_vector_body}, offset={articulation_offset}°)")

                # Apply articulation to rotation part only (match validation method)
                # Extract 3x3 rotation and apply articulation
                comp_rot_3x3 = transform[:3, :3]
                articulation_rot_3x3 = articulation_transform[:3, :3]
                transform[:3, :3] = comp_rot_3x3 @ articulation_rot_3x3

            body_mesh = local_mesh.copy()
            body_mesh.apply_transform(transform)
            component_meshes.append(body_mesh)

        if not component_meshes:
            logger.error("No valid components for shadow mesh")
            return None

        # Combine all meshes
        combined_mesh = trimesh.util.concatenate(component_meshes)

        # Minimal processing to preserve component boundaries
        # Don't call combined_mesh.process() or merge_vertices() as they destroy
        # component geometry by merging vertices from different components
        # Only remove truly degenerate faces (zero area)
        face_areas = combined_mesh.area_faces
        non_degenerate = face_areas > 1e-12
        if not np.all(non_degenerate):
            combined_mesh.update_faces(non_degenerate)

        mesh_time = time.time() - start_time
        logger.info(f"Shadow mesh created: {len(combined_mesh.vertices)} vertices, "
                   f"{len(combined_mesh.faces)} faces ({mesh_time:.3f}s)")

        return combined_mesh

    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        return {
            'total_rays': self.stats['total_rays'],
            'total_time': self.stats['total_time'],
            'average_rays_per_second': self.stats['total_rays'] / self.stats['total_time']
                                     if self.stats['total_time'] > 0 else 0
        }

    # =========================================================================
    # Vectorized Shadow Computation Methods
    # =========================================================================

    def _extract_mesh_geometry_vectorized(
        self,
        satellite,
        mesh: trimesh.Trimesh
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, slice]]:
        """
        Extract all facet centers and normals from mesh in vectorized manner.

        The mesh vertices are ordered as: component1_facet1_v0, v1, v2, facet2_v0, ...
        This matches the order components appear in satellite.components.

        Args:
            satellite: Satellite model
            mesh: Combined trimesh with all components

        Returns:
            Tuple of:
            - centers: (M, 3) facet centroid positions
            - normals: (M, 3) facet unit normal vectors
            - component_slices: Dict mapping component name to slice in arrays
        """
        # Count total facets
        total_facets = sum(len(c.facets) for c in satellite.components if c.facets)

        if total_facets == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), {}

        # Pre-allocate arrays
        centers = np.zeros((total_facets, 3), dtype=np.float64)
        normals = np.zeros((total_facets, 3), dtype=np.float64)
        component_slices = {}

        # Get all vertices at once (M*3, 3)
        mesh_vertices = mesh.vertices

        # Extract geometry for each component
        vertex_offset = 0
        facet_idx = 0

        for component in satellite.components:
            if not component.facets:
                continue

            num_facets = len(component.facets)
            start_idx = facet_idx

            # Calculate indices for this component's vertices
            v_start = vertex_offset
            v_end = vertex_offset + num_facets * 3

            if v_end > len(mesh_vertices):
                logger.warning(f"Vertex index out of range for {component.name}")
                vertex_offset += num_facets * 3
                facet_idx += num_facets
                continue

            # Get all vertices for this component (num_facets*3, 3)
            comp_vertices = mesh_vertices[v_start:v_end]

            # Reshape to (num_facets, 3, 3) - [facet, vertex_in_triangle, xyz]
            comp_vertices_reshaped = comp_vertices.reshape(num_facets, 3, 3)

            # Vectorized centroid calculation: mean of 3 vertices per facet
            centers[start_idx:start_idx + num_facets] = np.mean(comp_vertices_reshaped, axis=1)

            # Vectorized normal calculation
            v0 = comp_vertices_reshaped[:, 0, :]  # (num_facets, 3)
            v1 = comp_vertices_reshaped[:, 1, :]
            v2 = comp_vertices_reshaped[:, 2, :]

            edge1 = v1 - v0  # (num_facets, 3)
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)  # (num_facets, 3)
            norms = np.linalg.norm(cross, axis=1, keepdims=True)  # (num_facets, 1)

            # Avoid division by zero for degenerate faces
            safe_norms = np.where(norms > 1e-12, norms, 1.0)
            normals[start_idx:start_idx + num_facets] = cross / safe_norms

            component_slices[component.name] = slice(start_idx, start_idx + num_facets)
            vertex_offset += num_facets * 3
            facet_idx += num_facets

        return centers, normals, component_slices

    def _process_epoch_vectorized(
        self,
        mesh: trimesh.Trimesh,
        centers: np.ndarray,
        normals: np.ndarray,
        sun_vector: np.ndarray,
        epsilon: float = 1e-2
    ) -> np.ndarray:
        """
        Process single epoch with vectorized ray tracing.

        Instead of tracing one ray per facet, batches ALL rays into a single
        trimesh intersection call. This is the key optimization for ~10-50x speedup.

        Args:
            mesh: Shadow mesh for this epoch
            centers: Facet centers (M, 3)
            normals: Facet normals (M, 3)
            sun_vector: Sun direction in body frame (3,)
            epsilon: Offset from surface to avoid self-intersection

        Returns:
            lit_status: (M,) boolean array, True = lit, False = shadowed
        """
        M = len(centers)

        if M == 0:
            return np.array([], dtype=bool)

        # 1. Vectorized back-face culling
        n_dot_sun = normals @ sun_vector  # (M,)
        front_facing = n_dot_sun > 0

        # Initialize result: back-facing facets are automatically shadowed
        lit_status = np.zeros(M, dtype=bool)

        # Get indices of front-facing facets
        active_indices = np.where(front_facing)[0]
        K = len(active_indices)

        if K == 0:
            # All facets are back-facing
            return lit_status

        # 2. Prepare batched rays
        active_centers = centers[active_indices]  # (K, 3)
        active_normals = normals[active_indices]  # (K, 3)

        # Ray origins: offset from surface along normal to avoid self-intersection
        ray_origins = active_centers + epsilon * active_normals  # (K, 3)

        # All rays point toward sun (same direction for all)
        ray_directions = np.tile(sun_vector, (K, 1))  # (K, 3)

        # 3. BATCH RAY INTERSECTION - single call for all K rays!
        ray_intersector = RayMeshIntersector(mesh)

        try:
            locations, ray_indices, tri_indices = ray_intersector.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions
            )
        except Exception as e:
            logger.warning(f"Vectorized ray intersection failed: {e}")
            # Fallback: assume all front-facing facets are lit
            lit_status[active_indices] = True
            return lit_status

        # 4. Process results
        # All front-facing facets start as lit
        lit_status[active_indices] = True

        # Mark facets with intersections as shadowed
        if len(ray_indices) > 0:
            # ray_indices contains indices into our ray array (0 to K-1)
            # Multiple hits per ray are possible, so get unique ray indices
            hit_rays = np.unique(ray_indices)

            # Map back to facet indices
            shadowed_facet_indices = active_indices[hit_rays]
            lit_status[shadowed_facet_indices] = False

        return lit_status

    def compute_facet_shadows_multi_mesh_vectorized(
        self,
        satellite,
        shadow_meshes: List[trimesh.Trimesh],
        sun_vectors_body: np.ndarray,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Vectorized shadow computation for multiple epochs.

        Batches all facet rays per epoch into a single trimesh intersection call,
        providing ~10-50x speedup over the per-facet approach.

        Args:
            satellite: Satellite model
            shadow_meshes: List of shadow meshes (one per epoch)
            sun_vectors_body: Sun vectors in body frame (N, 3)
            show_progress: If True, display progress bar

        Returns:
            Dict[component_name, np.ndarray] - Boolean arrays (num_epochs, num_facets)
        """
        num_epochs = len(shadow_meshes)
        total_facets = sum(len(c.facets) for c in satellite.components if c.facets)

        logger.info(f"Vectorized shadow computation: {num_epochs} epochs × {total_facets} facets")

        # Allocate flat result array (N, M)
        lit_status_flat = np.zeros((num_epochs, total_facets), dtype=bool)

        # Build component slices (same across all epochs since satellite structure is constant)
        component_slices = {}
        idx = 0
        for component in satellite.components:
            if component.facets:
                num_facets = len(component.facets)
                component_slices[component.name] = slice(idx, idx + num_facets)
                idx += num_facets

        # Progress bar
        epoch_iter = range(num_epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Computing shadows", unit="epoch", mininterval=0.3)

        for i in epoch_iter:
            mesh = shadow_meshes[i]
            sun_vector = sun_vectors_body[i]

            # Extract geometry from this epoch's mesh (includes articulation)
            centers, normals, _ = self._extract_mesh_geometry_vectorized(satellite, mesh)

            # Process epoch with batched ray intersection
            lit_status_flat[i] = self._process_epoch_vectorized(
                mesh, centers, normals, sun_vector
            )

        # Convert flat array to dict format for compatibility
        result = {}
        for comp_name, comp_slice in component_slices.items():
            result[comp_name] = lit_status_flat[:, comp_slice].copy()

        return result

    def compute_facet_shadows_multi_mesh(self, satellite, shadow_meshes, sun_vectors_body, show_progress: bool = True):
        """
        Compute facet-level shadows using multiple meshes (articulated per epoch).

        Uses vectorized ray batching for efficient computation.

        Args:
            satellite: Satellite model
            shadow_meshes: List of shadow meshes (one per epoch)
            sun_vectors_body: Sun vectors in body frame
            show_progress: If True, display tqdm progress bar (default True)

        Returns:
            Dict[component_name, np.ndarray] - Boolean arrays of shape (num_epochs, num_facets)
        """
        # Use vectorized implementation
        return self.compute_facet_shadows_multi_mesh_vectorized(
            satellite, shadow_meshes, sun_vectors_body, show_progress
        )

    def compute_facet_shadows_single_mesh(self, satellite, shadow_mesh, sun_vectors_body, show_progress: bool = True):
        """
        Compute facet-level shadows using single mesh (no articulation).

        Uses vectorized ray batching. Since the mesh is static, geometry is extracted
        once and reused for all epochs.

        Args:
            satellite: Satellite model
            shadow_mesh: Static shadow mesh (no articulation)
            sun_vectors_body: Sun vectors in body frame (N, 3)
            show_progress: If True, display progress bar

        Returns:
            Dict[component_name, np.ndarray] - Boolean arrays of shape (num_epochs, num_facets)
        """
        num_epochs = len(sun_vectors_body)
        total_facets = sum(len(c.facets) for c in satellite.components if c.facets)

        logger.info(f"Vectorized shadow computation (single mesh): {num_epochs} epochs × {total_facets} facets")

        # Extract geometry once since mesh is static
        centers, normals, component_slices = self._extract_mesh_geometry_vectorized(satellite, shadow_mesh)

        # Allocate flat result array
        lit_status_flat = np.zeros((num_epochs, total_facets), dtype=bool)

        # Progress bar
        epoch_iter = range(num_epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Computing shadows", unit="epoch", mininterval=0.3)

        for i in epoch_iter:
            sun_vector = sun_vectors_body[i]

            # Process epoch with batched ray intersection (reuse geometry)
            lit_status_flat[i] = self._process_epoch_vectorized(
                shadow_mesh, centers, normals, sun_vector
            )

        # Convert flat array to dict format
        result = {}
        for comp_name, comp_slice in component_slices.items():
            result[comp_name] = lit_status_flat[:, comp_slice].copy()

        return result


# Global shadow engine instance
_shadow_engine = None

def get_shadow_engine() -> ShadowEngine:
    """Get or create global shadow engine instance."""
    global _shadow_engine
    if _shadow_engine is None:
        _shadow_engine = ShadowEngine()
    return _shadow_engine


def get_lit_status(
    satellite,
    pre_computed_k1_vectors: np.ndarray,
    shadow_meshes: List[trimesh.Trimesh],
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Ray tracing for shadow computation.

    Only performs ray tracing with pre-created meshes - no vector computation or mesh creation.

    Args:
        satellite: Satellite model
        pre_computed_k1_vectors: Sun vectors in body frame (N, 3), normalized
        shadow_meshes: Pre-created shadow meshes with articulation applied
        show_progress: If True, display tqdm progress bar (default True)

    Returns:
        facet_lit_status: Dict[component_name, np.ndarray]
            - Shape: (num_epochs, num_facets)
            - Values: Boolean (True=lit, False=shadowed)
    """
    logger.info("Ray tracing for shadow computation starting")

    start_time = time.time()
    num_epochs = len(pre_computed_k1_vectors)

    # Get shadow engine
    engine = get_shadow_engine()

    # Use pre-computed sun vectors
    sun_vectors_body = pre_computed_k1_vectors

    # Perform ray tracing with provided meshes
    if len(shadow_meshes) > 1:
        # Multiple meshes - articulated case
        facet_lit_status = engine.compute_facet_shadows_multi_mesh(
            satellite, shadow_meshes, sun_vectors_body, show_progress=show_progress)
    else:
        # Single mesh - no articulation
        facet_lit_status = engine.compute_facet_shadows_single_mesh(
            satellite, shadow_meshes[0], sun_vectors_body, show_progress=show_progress)

    total_time = time.time() - start_time
    total_facets = sum(len(comp.facets) for comp in satellite.components if comp.facets)
    total_rays = num_epochs * total_facets

    logger.info(f"Ray tracing complete: {total_time:.3f}s total "
               f"({total_rays:,} rays, {total_rays/total_time/1000:.1f}K rays/sec)")

    return facet_lit_status


# =============================================================================
# High-level shadow computation functions
# =============================================================================

def compute_shadows(
    satellite: Satellite,
    k1_vectors: NDArray[np.float64],
    explicit_component_angles: Optional[Dict[str, NDArray[np.float64]]] = None,
    explicit_component_matrices: Optional[Dict[str, NDArray[np.float64]]] = None,
    articulation_offset: float = 0.0,
    show_progress: bool = True
) -> Dict[str, NDArray[np.bool_]]:
    """
    Compute shadows using ray tracing.

    Uses ray-mesh intersection to compute precise shadow masks
    for each satellite component at all time epochs.

    Args:
        satellite: Satellite model with mesh geometry
        k1_vectors: Sun direction vectors in body frame (N, 3)
        explicit_component_angles: Component articulation angles by name (for logging)
        explicit_component_matrices: PRE-COMPUTED rotation matrices (N, 4, 4) per component
        articulation_offset: Solar panel articulation offset in degrees
        show_progress: If True, display tqdm progress bars (default True)

    Returns:
        Dictionary mapping component names to boolean lit status arrays (N, num_facets)

    Raises:
        RuntimeError: If shadow computation fails
        ValueError: If input parameters are invalid
    """
    # Input validation
    if len(k1_vectors) == 0:
        raise ValueError("k1_vectors array cannot be empty")

    logger.info("Computing shadows via ray tracing")

    if articulation_offset != 0.0:
        logger.info(f"Solar panel articulation: {articulation_offset:.1f} degree offset")

    shadow_start = time.time()

    try:
        # Get shadow engine to create meshes
        shadow_engine = get_shadow_engine()

        # Create shadow meshes using pre-computed matrices (no redundant angle→matrix conversion)
        if explicit_component_matrices:
            logger.info("Using pre-computed rotation matrices for shadow meshes (optimized)")
            shadow_meshes = shadow_engine.create_shadow_meshes_with_explicit_matrices_optimized(
                satellite=satellite,
                sun_vectors_body=k1_vectors,
                explicit_component_matrices=explicit_component_matrices,
                show_progress=show_progress
            )
        elif explicit_component_angles:
            # Fallback: angles provided but not matrices (compute matrices here)
            logger.warning("Explicit angles provided without matrices, computing matrices in shadow stage")
            shadow_meshes = shadow_engine.create_shadow_meshes_with_explicit_angles(
                satellite=satellite,
                sun_vectors_body=k1_vectors,
                explicit_component_angles=explicit_component_angles
            )
        else:
            # No articulation - create single mesh
            single_mesh = shadow_engine.create_shadow_mesh(satellite)
            if single_mesh is not None:
                shadow_meshes = [single_mesh]
            else:
                raise RuntimeError("Failed to create shadow mesh")

        # Ray tracing with pre-computed vectors and meshes
        lit_status_dict = get_lit_status(
            satellite=satellite,
            pre_computed_k1_vectors=k1_vectors,
            shadow_meshes=shadow_meshes,
            show_progress=show_progress
        )

    except Exception as e:
        logger.error(f"Shadow computation failed: {e}")
        raise RuntimeError(f"Ray tracing failed: {e}") from e

    shadow_time = time.time() - shadow_start
    logger.info(f"Shadows computed ({shadow_time:.2f}s)")

    return lit_status_dict


def create_no_shadow_lit_status(satellite: Satellite, num_epochs: int) -> Dict[str, NDArray[np.bool_]]:
    """
    Create lit status dictionary for no-shadow mode.

    Creates a dictionary where all facets are marked as fully lit
    for all time epochs. Used in no-shadow mode comparisons.

    Args:
        satellite: Satellite model with component structure
        num_epochs: Number of time epochs

    Returns:
        Dictionary mapping component names to boolean arrays (N, num_facets) all True

    Raises:
        ValueError: If satellite has no components or facets
    """
    if not satellite.components:
        raise ValueError("Satellite must have components")

    logger.info("No shadows mode - all faces fully lit, skipping ray tracing")

    # Create lit_status_dict with all facets set to True (facet-level format)
    lit_status_dict: Dict[str, NDArray[np.bool_]] = {}

    for component in satellite.components:
        if component.facets:
            # All facets are lit for all epochs
            num_facets = len(component.facets)
            lit_status_dict[component.name] = np.ones((num_epochs, num_facets), dtype=bool)
        else:
            logger.warning(f"Component {component.name} has no facets")

    if not lit_status_dict:
        raise ValueError("No components with facets found in satellite")

    return lit_status_dict
