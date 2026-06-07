"""
STL file loader and core model definitions for satellite light curve generation.

This module provides:
- Core data structures: Satellite, Component, Facet, BRDFMaterialProperties
- STL mesh loading via trimesh
- Satellite model assembly from configuration

The STL workflow loads pre-tessellated meshes directly - facets come from STL triangles.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

try:
    import quaternion
    Quaternion = quaternion.quaternion
except ImportError:
    Quaternion = None

import trimesh

from src.config.rso_config_schemas import RSO_Config, BRDFParameters, ArticulationCapability

logger = logging.getLogger(__name__)

# Type alias for 3D vectors
Vector3D = np.ndarray  # 3-element array: np.array([x, y, z])


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class BRDFMaterialProperties:
    """
    BRDF material properties for a facet, based on a Phong reflection model.

    Attributes:
        r_d: Diffuse reflectivity coefficient (0 to 1).
        r_s: Specular reflectivity coefficient (0 to 1).
        n_phong: Phong exponent (shininess).
    """
    r_d: float = 0.0
    r_s: float = 0.0
    n_phong: float = 1.0


@dataclass
class Facet:
    """
    A single triangular facet of a satellite component.

    Facets are loaded directly from STL mesh triangles. Each facet has
    pre-computed vertices, normal, and area from trimesh.

    Attributes:
        id: Unique identifier for the facet.
        vertices: List of 3 vertex positions (np.ndarray) defining the triangle.
        normal: Outward-pointing unit normal vector (np.ndarray).
        area: Surface area of the triangle.
        material_properties: BRDF properties for light curve calculation.
    """
    id: str
    vertices: List[Vector3D] = field(default_factory=list)
    normal: Vector3D = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    area: float = 0.0
    material_properties: BRDFMaterialProperties = field(default_factory=BRDFMaterialProperties)


@dataclass
class Component:
    """
    A distinct component of a satellite (e.g., bus, solar panel, antenna).

    Components are loaded from STL files. Each component contains a list of
    triangular facets and optional articulation parameters for moving parts.

    Attributes:
        id: Unique identifier for the component.
        name: Descriptive name (e.g., "SP_North", "Bus").
        facets: List of triangular Facet objects from STL mesh.
        relative_position: Position of component origin in body frame [x, y, z].
        relative_orientation: Orientation quaternion relative to body frame.
        articulation_parameters: Rotation axis/center for articulating components.
        default_material: Default BRDF properties applied to all facets.
    """
    id: str
    name: str
    facets: List[Facet] = field(default_factory=list, repr=False)
    relative_position: Vector3D = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    relative_orientation: Quaternion = field(default_factory=lambda: np.quaternion(1, 0, 0, 0))
    articulation_parameters: Optional[ArticulationCapability] = None
    default_material: Optional[BRDFMaterialProperties] = None


@dataclass
class Satellite:
    """
    Complete satellite model composed of multiple components.

    Defines the satellite's body-fixed reference frame and contains all
    geometric components loaded from STL files.

    Attributes:
        id: Unique identifier (typically SPICE satellite ID).
        name: Descriptive name of the satellite.
        components: List of Component objects making up the satellite.
        body_frame_name: SPICE reference frame name for the satellite body.
    """
    id: str
    name: str
    components: List[Component] = field(default_factory=list)
    body_frame_name: str = ""


# =============================================================================
# STL Loading
# =============================================================================

class STLLoader:
    """Loads STL mesh files and converts them to LCAS Satellite/Component objects."""

    @staticmethod
    def load_component_from_stl(
        stl_path: Path,
        component_name: str,
        component_id: str,
        position: np.ndarray,
        orientation: List[float],  # Quaternion as list [w, x, y, z]
        brdf_params: Optional[BRDFParameters] = None
    ) -> Component:
        """
        Load a component from an STL file.

        Args:
            stl_path: Path to the STL file.
            component_name: Name of the component.
            component_id: Unique ID for the component.
            position: Position in body frame [x, y, z].
            orientation: Orientation quaternion [w, x, y, z].
            brdf_params: BRDF material parameters.

        Returns:
            Component object with facets from STL triangles.
        """
        logger.info(f"Loading STL component '{component_name}' from: {stl_path}")

        # Load the STL mesh using trimesh
        mesh = trimesh.load(stl_path, force='mesh')

        # Ensure it's a Trimesh object (not a Scene)
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, 'dump'):
                meshes = mesh.dump()
                if meshes:
                    mesh = meshes[0]
                else:
                    raise ValueError(f"No valid mesh found in {stl_path}")
            else:
                raise ValueError(f"Could not extract mesh from {stl_path}")

        logger.info(f"  Loaded mesh: {len(mesh.faces)} triangles, {len(mesh.vertices)} vertices")
        logger.debug(f"  Mesh bounds: {mesh.bounds}")
        logger.debug(f"  Mesh volume: {mesh.volume:.6f} m³")

        # Create BRDF material if provided
        material = None
        if brdf_params:
            material = BRDFMaterialProperties(
                r_d=brdf_params.r_d,
                r_s=brdf_params.r_s,
                n_phong=brdf_params.n_phong
            )

        # Create Facet objects from mesh triangles
        facets = []
        for i, face_indices in enumerate(mesh.faces):
            # Get the three vertices of this triangle
            vertices = [mesh.vertices[idx] for idx in face_indices]

            # Get the pre-computed normal from trimesh (already normalized)
            normal = mesh.face_normals[i]

            # Get the pre-computed area
            area = mesh.area_faces[i]

            facet = Facet(
                id=f"{component_id}_facet_{i}",
                vertices=vertices,
                normal=normal,
                area=area,
                material_properties=material
            )
            facets.append(facet)

        # Convert orientation from list to quaternion
        orientation_quat = np.quaternion(
            orientation[0],  # w
            orientation[1],  # x
            orientation[2],  # y
            orientation[3]   # z
        )

        component = Component(
            id=component_id,
            name=component_name,
            facets=facets,
            relative_position=position,
            relative_orientation=orientation_quat,
        )

        logger.info(f"  Created component '{component_name}' with {len(facets)} facets")
        return component

    @staticmethod
    def create_satellite_from_stl_config(
        config: RSO_Config,
        config_manager
    ) -> Satellite:
        """
        Create a complete satellite from RSO configuration.

        Args:
            config: RSO configuration object.
            config_manager: RSO_ConfigManager instance for path resolution.

        Returns:
            Satellite object with all components loaded from STL files.
        """
        logger.info(f"Creating satellite '{config.name}' from STL files")

        components = []

        for comp_name, comp_def in config.components.items():
            # Get the full component path from the manager
            stl_path = config_manager.get_component_path(comp_name, config)

            # Get BRDF parameters for this component
            brdf_params = config.brdf_mappings.get(comp_name)
            if not brdf_params:
                logger.warning(f"No BRDF parameters for component '{comp_name}', using defaults")
                brdf_params = BRDFParameters()

            # Create unique component ID
            component_id = f"{config.name.upper().replace(' ', '_')}_{comp_name.upper()}"

            # Load the component from STL
            component = STLLoader.load_component_from_stl(
                stl_path=stl_path,
                component_name=comp_name,
                component_id=component_id,
                position=np.array(comp_def.position),
                orientation=comp_def.orientation,
                brdf_params=brdf_params
            )

            # Assign articulation capabilities if available
            if comp_name in config.articulation_capabilities:
                capability = config.articulation_capabilities[comp_name]
                component.articulation_parameters = capability
                logger.debug(f"  Assigned articulation capability to '{comp_name}': "
                           f"axis={capability.rotation_axis}, center={capability.rotation_center}")

            components.append(component)

        satellite = Satellite(
            id=str(config.spice_config.satellite_id),
            name=config.name,
            components=components,
            body_frame_name=config.spice_config.body_frame
        )

        total_facets = sum(len(comp.facets) for comp in components)
        logger.info(f"Created satellite '{config.name}' with {len(components)} components, {total_facets} total facets")

        return satellite
