"""
Articulation engine for satellite component movement.

Provides:
- ArticulationEngine: Manages behaviors and component registration
- calculate_angles_from_behaviors: Get angles from registered behaviors
- compute_rotation_matrices_from_angles: Convert angle arrays to rotation matrices
"""

import numpy as np
import quaternion
import logging
from typing import Dict, Optional
from numpy.typing import NDArray

from src.config.rso_config_schemas import RSO_Config
from src.io.stl_loader import Satellite, Component
from .behaviors.base_behavior import ArticulationBehavior

logger = logging.getLogger(__name__)


# =============================================================================
# ARTICULATION ENGINE CLASS
# =============================================================================

class ArticulationEngine:
    """Manages articulation behaviors and applies transformations to components."""

    def __init__(self, config: Optional[RSO_Config] = None):
        """
        Initialize articulation engine with optional configuration.

        Args:
            config: Optional RSO configuration containing articulation rules
        """
        self.config = config
        # Store runtime-assigned behaviors (component name -> behavior instance)
        self.component_behaviors: Dict[str, ArticulationBehavior] = {}

    def register_component_behavior(self, component_name: str, behavior: ArticulationBehavior) -> None:
        """
        Register a behavior instance for a specific component at runtime.

        Args:
            component_name: Name of the component
            behavior: ArticulationBehavior instance to assign
        """
        self.component_behaviors[component_name] = behavior
        logger.debug(f"Registered {type(behavior).__name__} for component '{component_name}'")

    def get_component_behavior(self, component: Component) -> Optional[ArticulationBehavior]:
        """
        Get articulation behavior for a component.

        Args:
            component: Component to get behavior for

        Returns:
            ArticulationBehavior instance or None if no articulation
        """
        return self.component_behaviors.get(component.name)

    def is_component_articulated(self, component: Component) -> bool:
        """
        Check if a component has articulation.

        Args:
            component: Component to check

        Returns:
            True if component has articulation, False otherwise
        """
        return component.name in self.component_behaviors


# =============================================================================
# BATCH ANGLE CALCULATION
# =============================================================================

def calculate_angles_from_behaviors(
    satellite: Satellite,
    k1_vectors: NDArray[np.float64],
    articulation_engine: ArticulationEngine,
    articulation_offset: float = 0.0
) -> Dict[str, NDArray[np.float64]]:
    """
    Calculate rotation angles for components using their assigned behaviors.

    This function calculates angles without computing matrices, allowing you to
    modify angles (add offsets, interpolate) before converting to matrices.

    Args:
        satellite: Satellite model with components
        k1_vectors: Sun-to-satellite vectors in body frame (N, 3), normalized
        articulation_engine: ArticulationEngine with component behaviors
        articulation_offset: Offset angle in degrees to add to computed angles

    Returns:
        component_angles: Dict[component_name, angles_array(N,)] in degrees
    """
    logger.debug(f"Calculating angles from behaviors for {satellite.name}")

    num_epochs = len(k1_vectors)
    component_angles = {}

    for component in satellite.components:
        behavior = articulation_engine.get_component_behavior(component)
        if not behavior:
            continue

        # Calculate angle for each epoch
        angles = np.zeros(num_epochs)
        for i in range(num_epochs):
            base_angle = behavior.calculate_rotation_angle(
                sun_vector_body=k1_vectors[i],
                earth_vector_body=None,
                epoch=float(i),
                offset_deg=0.0
            )
            angles[i] = base_angle + articulation_offset

        component_angles[component.name] = angles

    logger.debug(f"Calculated angles for {len(component_angles)} components")
    return component_angles


# =============================================================================
# BATCH MATRIX COMPUTATION
# =============================================================================

def compute_rotation_matrices_from_angles(
    component_angles: Dict[str, NDArray[np.float64]],
    satellite: Satellite,
    articulation_engine: Optional[ArticulationEngine] = None
) -> Dict[str, NDArray[np.float64]]:
    """
    Convert angle arrays to rotation matrix arrays.

    Gets rotation axis and center from Component.articulation_parameters.

    Args:
        component_angles: Dict mapping component names to angle arrays (N,) in degrees
        satellite: Satellite model with components
        articulation_engine: Optional ArticulationEngine (for fallback to behavior params)

    Returns:
        component_matrices: Dict mapping component names to matrix arrays (N, 4, 4)
    """
    from src.utils.geometry_utils import build_rotation_matrix

    logger.debug(f"Computing rotation matrices for {len(component_angles)} components")

    component_matrices = {}

    for component_name, angles_array in component_angles.items():
        # Find the component in the satellite model
        component = next((c for c in satellite.components if c.name == component_name), None)
        if not component:
            logger.warning(f"Component {component_name} not found in satellite model, skipping")
            continue

        # Get rotation axis from component
        rotation_axis = None

        if component.articulation_parameters is not None:
            rotation_axis = np.array(component.articulation_parameters.rotation_axis)
        elif articulation_engine is not None:
            # Fallback: try to get from behavior
            behavior = articulation_engine.get_component_behavior(component)
            if behavior:
                rotation_axis = behavior.rotation_axis

        if rotation_axis is None:
            logger.warning(f"No rotation axis found for component {component_name}, skipping")
            continue

        # Convert each angle to a rotation matrix
        num_epochs = len(angles_array)
        matrices = np.zeros((num_epochs, 4, 4))

        for i, angle_deg in enumerate(angles_array):
            matrices[i] = build_rotation_matrix(angle_deg, rotation_axis)

        component_matrices[component_name] = matrices

    logger.debug(f"Computed matrices for {len(component_matrices)} components")
    return component_matrices
