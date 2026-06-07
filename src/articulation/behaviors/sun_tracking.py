"""
Sun tracking articulation behavior.
Implements solar panel sun tracking functionality.
"""

import numpy as np
from typing import Dict, Any
from .base_behavior import ArticulationBehavior
from src.utils.geometry_utils import calculate_sun_pointing_rotation


class SunTrackingBehavior(ArticulationBehavior):
    """Sun tracking articulation behavior for solar panels."""
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize sun tracking behavior.
        
        Args:
            parameters: Dictionary containing articulation parameters
        """
        super().__init__(parameters)
    
    def calculate_rotation_angle(self, 
                               sun_vector_body: np.ndarray,
                               earth_vector_body: np.ndarray,
                               epoch: float,
                               offset_deg: float = 0.0) -> float:
        """
        Calculate rotation angle to track the sun.
        
        Args:
            sun_vector_body: Sun vector in body frame
            earth_vector_body: Earth vector in body frame (unused for sun tracking)
            epoch: Current epoch time (unused for sun tracking)
            offset_deg: Additional offset in degrees
            
        Returns:
            Rotation angle in degrees
        """
        # Use existing sun pointing rotation calculation
        panel_angle_deg = calculate_sun_pointing_rotation(
            sun_vector_body, 
            self.rotation_axis, 
            self.reference_normal
        )
        
        # Apply offset
        final_angle = panel_angle_deg + offset_deg
        
        # Apply limits
        final_angle = self.apply_limits(final_angle)
        
        return final_angle