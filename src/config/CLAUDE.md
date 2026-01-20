# src/config - Configuration Management

This module handles YAML configuration loading and validation for LCAS satellite models.

## Module Overview

| File | Purpose |
|------|---------|
| `rso_config_manager.py` | Main configuration loader with path resolution |
| `rso_config_schemas.py` | Dataclass schemas for configuration validation |

## Key Classes

### RSO_ConfigManager
Central manager for loading and resolving satellite configurations.

```python
from src.config.rso_config_manager import RSO_ConfigManager

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")

# Path resolution
metakernel_path = config_manager.get_metakernel_path(config)
output_dir = config_manager.get_output_directory(config)
stl_path = config_manager.get_component_path("Bus", config)
```

**Key attributes:**
- `project_root`: Base path for relative path resolution
- `models_dir`: `project_root / "data" / "models"`
- `config_directory`: Directory containing loaded config (for STL file resolution)

### Configuration Schemas (rso_config_schemas.py)

```python
from src.config.rso_config_schemas import (
    RSO_Config,           # Complete satellite configuration
    ComponentDefinition,   # STL file + position/orientation
    ArticulationCapability,# Rotation axis, center, limits
    SpiceConfig,          # satellite_id, metakernel_path, body_frame
    SimulationDefaults,   # start_time, end_time, output_dir
    BRDFParameters        # r_d, r_s, n_phong
)
```

## RSO_Config Structure

```python
@dataclass
class RSO_Config:
    name: str                                    # "Intelsat 901"
    spice_config: SpiceConfig                    # SPICE settings
    simulation_defaults: SimulationDefaults      # Time range, output
    components: Dict[str, ComponentDefinition]   # STL files per component
    brdf_mappings: Dict[str, BRDFParameters]     # Material properties
    articulation_capabilities: Dict[str, ArticulationCapability]  # Movement rules
```

## YAML Config Format

```yaml
name: "Intelsat 901"

spice_config:
  satellite_id: -126824
  metakernel_path: "data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm"
  body_frame: "IS901_BUS_FRAME"

simulation_defaults:
  start_time: "2020-02-05T10:00:00"
  end_time: "2020-02-05T16:00:00"
  output_dir: "intelsat_901_results"

components:
  Bus:
    stl_file: "bus.stl"
    position: [0.0, 0.0, 0.0]
    orientation: [1.0, 0.0, 0.0, 0.0]  # Quaternion [w, x, y, z]
  SP_North:
    stl_file: "sp_north.stl"
    position: [0.0, 5.0, 0.0]
    orientation: [1.0, 0.0, 0.0, 0.0]

component_brdf:
  Bus:
    r_d: 0.02
    r_s: 0.5
    n_phong: 300.0
  SP_North:
    r_d: 0.026
    r_s: 0.3
    n_phong: 200.0

articulation_capabilities:
  SP_North:
    rotation_center: [0.0, 5.0, 0.0]
    rotation_axis: [0.0, 1.0, 0.0]
    limits:
      min_angle: -180.0
      max_angle: 180.0
```

## Usage Pattern

```python
# In notebooks/scripts
from src.config.rso_config_manager import RSO_ConfigManager

PROJECT_ROOT = Path.cwd().parent
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")

# Access configuration
satellite_id = config.spice_config.satellite_id
start_time = config.simulation_defaults.start_time
components = config.components  # Dict[str, ComponentDefinition]
brdf = config.brdf_mappings     # Dict[str, BRDFParameters]
articulation = config.articulation_capabilities  # Dict[str, ArticulationCapability]
```

## Cross-References

- **Config files**: Located in `data/models/<satellite>/`
- **STL loading**: Uses paths via `src/io/stl_loader.py`
- **BRDF application**: Used by `src/computation/brdf.py`
