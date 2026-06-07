"""
Author: Girish Narayanan
RSO configuration manager for satellite light curve generation.
Handles loading and management of RSO satellite configurations.
"""

import yaml
import logging
from pathlib import Path
from typing import Union, Any

# Project Imports
from .rso_config_schemas import (
    RSO_Config,
    ComponentDefinition,
    ArticulationCapability,
    SpiceConfig,
    SimulationDefaults,
    BRDFParameters
)

logger = logging.getLogger(__name__)


class RSO_ConfigManager:
    """
    Configuration manager for RSO satellite models.
    Handles configuration loading and path resolution.
    """

    def __init__(self, project_root: Path = None):
        """Initialize the RSO configuration manager."""
        # Initialize project_root (default to project structure)
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        # Initialize models directory
        self.models_dir = self.project_root / "data" / "models"

        # Configuration attributes
        self.config_directory = None  # Will store the directory containing the config file
        self.component_info = {}  # Will store component info

    def load_config(self, config_path: Union[str, Path]) -> RSO_Config:
        """
        Load RSO configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file

        Returns:
            RSO_Config: Loaded RSO configuration
        """
        config_path = Path(config_path)

        # Resolve relative paths
        if not config_path.is_absolute():
            config_path = self.models_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Store the config directory - component files are in the same directory
        self.config_directory = config_path.parent
        logger.info(f"Loading RSO config from: {config_path}")
        logger.info(f"Component files will be loaded from: {self.config_directory}")

        # Load YAML
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Create RSO_Config
        rso_config = RSO_Config()

        # Basic info
        rso_config.name = config_data.get('name', 'Unknown Satellite')

        # SPICE config
        if 'spice_config' in config_data:
            spice = config_data['spice_config']
            rso_config.spice_config = SpiceConfig(
                satellite_id=spice.get('satellite_id', -126824),
                metakernel_path=spice.get('metakernel_path', ''),
                body_frame=spice.get('body_frame', 'IS901_BUS_FRAME')
            )

        # Simulation defaults with output directory handling
        if 'simulation_defaults' in config_data:
            sim = config_data['simulation_defaults']
            rso_config.simulation_defaults = SimulationDefaults(
                subdivision_level=sim.get('subdivision_level', 3),
                start_time=sim.get('start_time', '2020-02-05T10:00:00'),
                end_time=sim.get('end_time', '2020-02-05T16:00:00'),
                output_dir=sim.get('output_dir', 'lightcurve_results')
            )

        # Component definitions
        if 'components' in config_data:
            for comp_name, comp_data in config_data['components'].items():
                rso_config.components[comp_name] = ComponentDefinition(
                    stl_file=comp_data.get('stl_file', f'{comp_name}.stl'),
                    position=comp_data.get('position', [0.0, 0.0, 0.0]),
                    orientation=comp_data.get('orientation', [1.0, 0.0, 0.0, 0.0])
                )
                logger.debug(f"  Component '{comp_name}': STL={comp_data.get('stl_file')}")

        # Component BRDF
        if 'component_brdf' in config_data:
            for comp_name, brdf_data in config_data['component_brdf'].items():
                rso_config.brdf_mappings[comp_name] = BRDFParameters(
                    r_d=brdf_data.get('r_d', 0.1),
                    r_s=brdf_data.get('r_s', 0.1),
                    n_phong=brdf_data.get('n_phong', 10.0)
                )

        # Articulation capabilities (physical constraints, not behaviors)
        if 'articulation_capabilities' in config_data:
            for comp_name, artic_data in config_data['articulation_capabilities'].items():
                rso_config.articulation_capabilities[comp_name] = ArticulationCapability(
                    rotation_center=artic_data.get('rotation_center', [0.0, 0.0, 0.0]),
                    rotation_axis=artic_data.get('rotation_axis', [0.0, 0.0, 1.0]),
                    limits=artic_data.get('limits', {'min_angle': -180.0, 'max_angle': 180.0})
                )
                logger.debug(f"  Component '{comp_name}' has articulation capability")

        logger.info(f"Loaded RSO config: {rso_config.name} with {len(rso_config.components)} components")
        return rso_config

    def get_component_path(self, component_name: str, config: RSO_Config) -> Path:
        """
        Get the full path to a component file (STL mesh).
        Component files are in the same directory as the config file.

        Args:
            component_name: Name of the component
            config: The RSO configuration

        Returns:
            Path: Full path to the component file
        """
        if component_name not in config.components:
            raise ValueError(f"Component '{component_name}' not found in configuration")

        component_file = config.components[component_name].stl_file
        component_path = self.config_directory / component_file

        if not component_path.exists():
            raise FileNotFoundError(f"Component file not found: {component_path}")

        return component_path

    def get_metakernel_path(self, config: Union[RSO_Config, Any]) -> Path:
        """Get absolute path to SPICE metakernel."""
        metakernel_path = config.spice_config.metakernel_path

        # Handle relative paths
        if not Path(metakernel_path).is_absolute():
            metakernel_path = self.project_root / metakernel_path

        return Path(metakernel_path)

    def get_output_directory(self, config: RSO_Config) -> Path:
        """
        Get the output directory path.
        Creates it under data/results/<output_dir_name>.
        """
        output_dir = self.project_root / "data" / "results" / config.simulation_defaults.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
