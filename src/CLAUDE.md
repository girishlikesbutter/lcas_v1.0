# LCAS Core Library Development - src/ Module Patterns

This file provides specialized guidance for developing LCAS core library modules in the `src/` directory. These patterns apply when working on satellite light curve generation, ray tracing, SPICE orbital mechanics, and BRDF material modeling.

## Scope & Context

**When this applies**: Working on any module within `src/` directory
**Domain focus**: Scientific computing for satellite light curve generation

## Core Architecture & Data Flow

### Module Structure
The codebase is organized around the STL-based satellite light curve pipeline:

```
src/
├── config/
│   ├── rso_config_manager.py     # YAML configuration loading
│   └── rso_config_schemas.py     # Pydantic schemas for validation
├── io/
│   ├── stl_loader.py             # STL mesh loading + core data structures
│   └── data_writer.py            # CSV export for light curve data
├── spice/
│   └── spice_handler.py          # SPICE orbital mechanics integration
├── articulation/
│   ├── articulation_engine.py    # Component movement + transformations
│   └── behaviors/                # Articulation behavior implementations
│       ├── base_behavior.py      # Abstract base class
│       └── sun_tracking.py       # Solar panel sun-tracking
├── computation/
│   ├── shadow_engine.py          # Ray tracing & shadow computation
│   ├── brdf.py                   # BRDF calculations + material manager
│   ├── observation_geometry.py   # Observation geometry computation
│   └── lightcurve_generator.py   # Light curve generation
├── interpolation/
│   ├── quaternion_interpolator.py  # Angle + quaternion SLERP interpolation
│   └── attitude_interpolator.py    # Attitude-only interpolation helper
├── utils/
│   └── geometry_utils.py         # Rotation matrix utilities
└── visualization/
    ├── lightcurve_plotter.py     # Light curve plotting + CSV saving
    ├── plotly_animation_generator.py  # Interactive 3D Plotly animation
    └── plot_styling.py           # Shared plot constants
```

### Core Data Structures (io/stl_loader.py)

```python
from src.io.stl_loader import Satellite, Component, Facet, BRDFMaterialProperties

# Satellite: Complete model with components
# Component: Single part (bus, solar panel, antenna) with facets
# Facet: Single triangular mesh face with vertices, normal, area
# BRDFMaterialProperties: r_d, r_s, n_phong for Ashikhmin-Shirley BRDF
```

### Typical Pipeline Flow

```python
# 1. Load configuration
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("satellite/config.yaml")

# 2. Load satellite from STL files
satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

# 3. Initialize SPICE for orbital mechanics
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(metakernel_path)

# 4. Compute observation geometry (sun/observer vectors in body frame)
geometry_data = compute_observation_geometry(epochs, satellite_id, spice_handler, config)

# 5. Calculate articulation angles and rotation matrices
component_angles = calculate_angles_from_behaviors(satellite, k1_vectors, articulation_engine)
component_matrices = compute_rotation_matrices_from_angles(component_angles, satellite)

# 6. Compute shadows using ray tracing
lit_status_dict = compute_shadows(satellite, k1_vectors,
                                  explicit_component_matrices=component_matrices)

# 7. Generate light curves
magnitudes, flux, *_ = generate_lightcurves(
    lit_status_dict, k1_vectors, k2_vectors, distances,
    satellite, epochs, pre_computed_matrices=component_matrices
)
```

## Domain-Specific Patterns

### SPICE Integration
```python
from src.spice.spice_handler import SpiceHandler

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))

# Time conversions
et = spice_handler.utc_to_et("2024-01-01T12:00:00")
utc = spice_handler.et_to_utc(et, "C", 0)

# Position calculations
position, light_time = spice_handler.get_body_position(target, et, frame, observer)
```

### Articulation System
```python
from src.articulation import (
    ArticulationEngine, SunTrackingBehavior,
    calculate_angles_from_behaviors, compute_rotation_matrices_from_angles
)

# Create engine and register behaviors
engine = ArticulationEngine()
sun_tracking = SunTrackingBehavior({
    'rotation_center': [0, 0, 0],
    'rotation_axis': [0, 0, 1],
    'reference_normal': [1, 0, 0],
    'limits': [-180, 180]
})
engine.register_component_behavior('SP_North', sun_tracking)

# Calculate angles from behaviors
angles_dict = calculate_angles_from_behaviors(satellite, k1_vectors, engine)

# Or create angles directly with interpolation
from src.interpolation import create_angle_interpolator
angles = create_angle_interpolator(
    keyframe_times_utc=['2024-01-01T12:00:00', '2024-01-01T13:00:00'],
    keyframe_values=[0, 45],
    transitions=['linear'],
    epochs=epochs,
    utc_to_et=spice_handler.utc_to_et
)

# Convert to rotation matrices
matrices = compute_rotation_matrices_from_angles(angles_dict, satellite)
```

### Shadow Computation
```python
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status

# Ray tracing with pre-computed matrices
lit_status = compute_shadows(
    satellite=satellite,
    k1_vectors=k1_vectors,
    explicit_component_matrices=component_matrices
)

# For comparison without shadows
lit_status_no_shadow = create_no_shadow_lit_status(satellite, num_epochs)
```

### BRDF Material Management
```python
from src.computation.brdf import BRDFCalculator, BRDFManager

brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)
```

## Type Hints

```python
import numpy as np
from numpy.typing import NDArray

def compute_observation_geometry(
    epochs: NDArray[np.float64],
    satellite_id: int,
    spice_handler: SpiceHandler
) -> Dict[str, NDArray[np.float64]]:
    """
    Returns dict with:
    - 'k1_vectors': (N, 3) sun vectors in body frame
    - 'k2_vectors': (N, 3) observer vectors in body frame
    - 'observer_distances': (N,) distances in km
    """
```

## Import Conventions

### Within src/ (relative imports)
```python
from ..io.stl_loader import Satellite, Component
from ..spice.spice_handler import SpiceHandler
from .brdf import BRDFCalculator
```

### From notebooks/scripts (absolute imports)
```python
from src.io.stl_loader import STLLoader
from src.computation.shadow_engine import compute_shadows
from src.articulation import ArticulationEngine, SunTrackingBehavior
```

## Error Handling

```python
import spiceypy
import logging

logger = logging.getLogger(__name__)

try:
    result = spiceypy.spkpos(target, epoch, frame, abcorr, observer)
except spiceypy.utils.exceptions.SpiceyError as e:
    logger.error(f"SPICE error: {e}")
    raise RuntimeError(f"Orbital calculation failed: {e}") from e
```

## Logging Pattern

```python
import logging
logger = logging.getLogger(__name__)

def expensive_operation(data):
    logger.info(f"Starting operation with {len(data)} items")
    start = time.time()

    result = process(data)

    logger.info(f"Operation completed in {time.time() - start:.2f}s")
    return result
```
