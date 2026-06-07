# src/io - Data I/O and Core Data Structures

This module handles STL mesh loading, core data structures, and CSV export for light curve data.

## Module Overview

| File | Purpose |
|------|---------|
| `stl_loader.py` | STL mesh loading + core data structures (Satellite, Component, Facet) |
| `data_writer.py` | CSV export for light curve results |

## Core Data Structures

### Satellite
Complete satellite model composed of multiple components.

```python
@dataclass
class Satellite:
    id: str                      # SPICE satellite ID as string
    name: str                    # "Intelsat 901"
    components: List[Component]  # All satellite parts
    body_frame_name: str         # "IS901_BUS_FRAME"
```

### Component
Single part (bus, solar panel, antenna) with mesh geometry.

```python
@dataclass
class Component:
    id: str                                    # "INTELSAT_901_BUS"
    name: str                                  # "Bus"
    facets: List[Facet]                        # Triangular mesh faces
    relative_position: np.ndarray              # [x, y, z] in body frame
    relative_orientation: quaternion.quaternion # Orientation quaternion
    articulation_parameters: ArticulationCapability  # Rotation axis/center
    default_material: BRDFMaterialProperties   # BRDF params
```

### Facet
Single triangular mesh face from STL.

```python
@dataclass
class Facet:
    id: str                                    # "INTELSAT_901_BUS_facet_0"
    vertices: List[np.ndarray]                 # 3 vertices [v0, v1, v2]
    normal: np.ndarray                         # Outward unit normal
    area: float                                # Triangle area
    material_properties: BRDFMaterialProperties # r_d, r_s, n_phong
```

### BRDFMaterialProperties
Material properties for Ashikhmin-Shirley BRDF.

```python
@dataclass
class BRDFMaterialProperties:
    r_d: float = 0.0     # Diffuse reflectivity (0-1)
    r_s: float = 0.0     # Specular reflectivity (0-1)
    n_phong: float = 1.0 # Phong exponent (shininess)
```

## STL Loading

### STLLoader Class

```python
from src.io.stl_loader import STLLoader

# Load complete satellite from config
satellite = STLLoader.create_satellite_from_stl_config(
    config=config,
    config_manager=config_manager
)

# Load individual component (internal use)
component = STLLoader.load_component_from_stl(
    stl_path=Path("bus.stl"),
    component_name="Bus",
    component_id="INTELSAT_901_BUS",
    position=np.array([0, 0, 0]),
    orientation=[1, 0, 0, 0],  # [w, x, y, z] quaternion
    brdf_params=brdf_params
)
```

### Loading Process
1. Reads STL file using `trimesh`
2. Extracts triangular faces as Facet objects
3. Applies component position/orientation
4. Assigns articulation_parameters from config
5. Sets BRDF material properties

## CSV Export

### save_lightcurve_data

```python
from src.io.data_writer import save_lightcurve_data

data_path = save_lightcurve_data(
    output_dir=output_dir,
    epochs=epochs,
    time_hours=time_hours,
    magnitudes=magnitudes,
    phase_angles=phase_angles,
    observer_distances=observer_distances,
    utc_times=utc_times,
    compare_shadows=False,    # If True, includes both mag columns
    use_shadows=True,         # Affects filename
    num_points=100,
    magnitudes_no_shadow=None,
    timestamp="1430"          # Optional HHMM prefix
)
```

### CSV Format

Single mode:
```csv
UTC_Time,ET_Seconds,Hours_Since_Start,Apparent_Magnitude,Phase_Angle_Deg,Observer_Distance_1000km
```

Comparison mode:
```csv
UTC_Time,ET_Seconds,Hours_Since_Start,Magnitude_Shadowed,Magnitude_No_Shadow,Phase_Angle_Deg,Observer_Distance_1000km
```

## Usage in Pipeline

```python
from src.io.stl_loader import STLLoader, Satellite, Component, Facet

# Load satellite
satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

# Access components
for component in satellite.components:
    print(f"{component.name}: {len(component.facets)} facets")

    # Access articulation parameters
    if component.articulation_parameters:
        axis = component.articulation_parameters.rotation_axis
        center = component.articulation_parameters.rotation_center

# Access facets
for facet in component.facets:
    center = np.mean(facet.vertices, axis=0)
    normal = facet.normal
    area = facet.area
```

## Cross-References

- **Config loading**: `src/config/rso_config_manager.py`
- **BRDF usage**: `src/computation/brdf.py`
- **Shadow computation**: `src/computation/shadow_engine.py`
- **Articulation**: `src/articulation/articulation_engine.py`
