"""
Author: Girish Narayanan
RSO configuration schemas for satellite light curve generation.
Unified configuration system for STL-based satellite models.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class BRDFParameters:
    """BRDF material parameters."""
    r_d: float = 0.1
    r_s: float = 0.1
    n_phong: float = 10.0


@dataclass
class SpiceConfig:
    """SPICE configuration parameters."""
    satellite_id: int = -126824
    observer_id: int = 399999
    metakernel_path: str = ""
    body_frame: str = "IS901_BUS_FRAME"


@dataclass
class SimulationDefaults:
    """Default simulation parameters."""
    subdivision_level: int = 3
    start_time: str = "2020-02-05T10:00:00"
    end_time: str = "2020-02-05T16:00:00"
    output_dir: str = "lightcurve_results"


@dataclass
class ComponentDefinition:
    """Definition for a component with STL mesh file."""
    stl_file: str  # Relative to config file directory
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # Quaternion [w,x,y,z]


@dataclass
class ArticulationCapability:
    """
    Component articulation capability and behavior.
    Defines both physical constraints (from config) and runtime behavior assignment.
    """
    # Physical constraints (defined in config file)
    rotation_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    limits: Dict[str, float] = field(default_factory=lambda: {'min_angle': -180.0, 'max_angle': 180.0})

    # Behavior assignment (assigned at runtime, defaults to "none")
    rule_type: str = "none"
    reference_normal: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    parameters: Dict = field(default_factory=dict)


@dataclass
class RSO_Config:
    """
    Complete RSO configuration for satellite light curve generation.
    Supports STL-based satellite models.
    """
    # Basic info
    name: str = "Unknown Satellite"

    # Standard configs
    spice_config: SpiceConfig = field(default_factory=SpiceConfig)
    simulation_defaults: SimulationDefaults = field(default_factory=SimulationDefaults)

    # Component definitions (STL-based)
    components: Dict[str, ComponentDefinition] = field(default_factory=dict)
    brdf_mappings: Dict[str, BRDFParameters] = field(default_factory=dict)
    articulation_capabilities: Dict[str, ArticulationCapability] = field(default_factory=dict)

    def get_output_directory(self, project_root: Path) -> Path:
        """Get the full output directory path under data/results/."""
        return project_root / "data" / "results" / self.simulation_defaults.output_dir
