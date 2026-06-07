"""
Articulation module for satellite component movement.

Main components:
- ArticulationEngine: Manages behaviors and component registration
- SunTrackingBehavior: Implements sun-tracking for solar panels
- calculate_angles_from_behaviors: Get angles from registered behaviors
- compute_rotation_matrices_from_angles: Convert angles to rotation matrices
"""

from .articulation_engine import (
    ArticulationEngine,
    calculate_angles_from_behaviors,
    compute_rotation_matrices_from_angles,
)
from .behaviors import ArticulationBehavior, SunTrackingBehavior

__all__ = [
    'ArticulationEngine',
    'ArticulationBehavior',
    'SunTrackingBehavior',
    'calculate_angles_from_behaviors',
    'compute_rotation_matrices_from_angles',
]
