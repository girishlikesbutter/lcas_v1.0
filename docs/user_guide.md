# LCAS User Guide

A comprehensive guide to using the Light Curve Analysis Suite for satellite brightness simulation.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Understanding Satellite Models](#2-understanding-satellite-models)
3. [Coordinate Systems](#3-coordinate-systems)
4. [Articulation](#4-articulation)
5. [Attitude Control](#5-attitude-control)
6. [BRDF Materials](#6-brdf-materials)
7. [Shadow Computation](#7-shadow-computation)
8. [Light Curves](#8-light-curves)
9. [3D Animations](#9-3d-animations)
10. [Configuration Files](#10-configuration-files)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quick Start

Generate your first light curve in under 5 minutes using the torus_plate test model.

### Prerequisites

1. Complete installation: `python install_dependencies.py`
2. Activate virtual environment:
   ```bash
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

### Run the Example

```python
import sys
from pathlib import Path

# Setup
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.lightcurve_generator import generate_lightcurves
from src.computation.shadow_engine import compute_shadows
from src.computation.brdf import BRDFCalculator, BRDFManager

# Load configuration
config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("torus_plate/torus_plate_config.yaml")

# Load satellite model
satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)

# Initialize SPICE
spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(config_manager.get_metakernel_path(config)))

# Generate time points
start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et(config.simulation_defaults.end_time)
epochs = np.linspace(start_et, end_et, 10)  # 10 time points

# Compute geometry
geometry_data = compute_observation_geometry(
    epochs, config.spice_config.satellite_id, 399999, spice_handler, config
)

# Compute shadows
lit_status_dict = compute_shadows(satellite, geometry_data['k1_vectors'])

# Generate light curve
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, BRDFManager(config))

magnitudes, flux, *_ = generate_lightcurves(
    lit_status_dict, geometry_data['k1_vectors'], geometry_data['k2_vectors'],
    geometry_data['observer_distances'], satellite, epochs
)

print("Magnitudes:", magnitudes)
```

### Expected Output

An array of apparent magnitudes (typically 10-15 for geostationary satellites).

---

## 2. Understanding Satellite Models

LCAS uses STL (stereolithography) mesh files to define satellite geometry.

### What Are STL Files?

STL files contain triangular meshes - collections of triangles that approximate 3D surfaces. Each triangle (facet) has:
- Three vertices (corner points)
- A normal vector (perpendicular to the surface)

### Component Structure

Satellites are divided into components, each with its own STL file:

```
data/models/intelsat_901/
├── intelsat_901_config.yaml   # Main configuration
├── bus.stl                    # Main satellite body
├── sp.stl                     # Solar panel
├── ad.stl                     # Antenna dish
```

### Component Properties

Each component has:

| Property | Description |
|----------|-------------|
| **Geometry** | STL mesh defining shape |
| **Position** | Location relative to satellite origin (meters) |
| **Orientation** | Initial rotation (quaternion) |
| **BRDF** | Material optical properties |
| **Articulation** | Movement capability (optional) |

### Directory Requirements

When creating a new satellite model:
1. Create folder: `data/models/<satellite_name>/`
2. Add STL files (binary or ASCII format)
3. Create configuration YAML file
4. Units must be in meters

---

## 3. Coordinate Systems

Understanding coordinate systems is essential for working with LCAS.

### J2000 Inertial Frame

The J2000 frame is a standard astronomical reference frame:
- Origin: Solar system barycenter (or Earth center for Earth satellites)
- X-axis: Points toward vernal equinox (March 21, 2000)
- Z-axis: Points toward celestial north pole
- Y-axis: Completes right-handed system

SPICE uses J2000 for orbital calculations.

### Body Frame (Satellite-Fixed)

The body frame moves with the satellite:
- Origin: Satellite center of mass
- Axes: Defined by the satellite's attitude
- STL geometry is defined in this frame

### Key Vectors

LCAS computes two critical direction vectors for each time point:

**k1_vectors (Sun Direction)**
- Unit vector pointing from satellite toward the sun
- Expressed in body frame coordinates
- Shape: (N, 3) where N = number of time points

**k2_vectors (Observer Direction)**
- Unit vector pointing from satellite toward observer
- Expressed in body frame coordinates
- Shape: (N, 3)

### Phase Angle

The phase angle is the angle between k1 and k2:

```
phase_angle = arccos(k1 · k2)
```

- 0°: Observer sees fully illuminated face (like full moon)
- 90°: Observer sees half-illuminated face (like quarter moon)
- 180°: Observer is between sun and satellite (like new moon)

Phase angle strongly affects apparent brightness.

---

## 4. Articulation

Articulation allows satellite components to move, such as solar panels tracking the sun.

### Articulation Capabilities

Defined in the config file, capabilities specify what movements are physically possible:

```yaml
articulation_capabilities:
  SP_North:
    rotation_center: [0.0, 0.0, 0.0]    # Point on rotation axis
    rotation_axis: [0.0, 0.0, 1.0]       # Direction of axis (normalized)
    limits:
      min_angle: -180.0                   # Rotation limits (degrees)
      max_angle: 180.0
```

### Option 1: SunTrackingBehavior

Automatically rotates components to face the sun:

```python
from src.articulation import ArticulationEngine, SunTrackingBehavior

engine = ArticulationEngine()

sun_tracking = SunTrackingBehavior({
    'rotation_center': [0.0, 0.0, 0.0],
    'rotation_axis': [0.0, 0.0, 1.0],
    'reference_normal': [1.0, 0.0, 0.0],  # Normal when angle=0
    'limits': {'min_angle': -180, 'max_angle': 180}
})

engine.register_component_behavior('SP_North', sun_tracking)
```

The behavior computes optimal angles from k1_vectors automatically.

### Option 2: Angle Interpolation

For explicit control over angles, use `create_angle_interpolator`:

```python
from src.interpolation import create_angle_interpolator

angles = create_angle_interpolator(
    keyframe_times_utc=['2020-02-05T10:00:00', '2020-02-05T14:00:00'],
    keyframe_values=[0.0, 90.0],     # Rotate from 0° to 90°
    transitions=['linear'],           # Linear interpolation
    epochs=epochs,
    utc_to_et=spice_handler.utc_to_et
)
```

### Transition Types

| Type | Description | Example Use |
|------|-------------|-------------|
| `'constant'` | Hold value until next keyframe | Fixed angle |
| `'linear'` | Smooth interpolation | Gradual rotation |
| `'step'` | Jump at specified fraction | Command sequences |

### Step Transition Example

```python
angles = create_angle_interpolator(
    keyframe_times_utc=['10:00:00', '12:00:00', '14:00:00'],
    keyframe_values=[0.0, 45.0, 90.0],
    transitions=['linear', 'step'],
    step_params=[None, 0.9],  # Step at 90% through second interval
    epochs=epochs,
    utc_to_et=spice_handler.utc_to_et
)
```

### Composing Angles

Add offset angles to sun-tracking:

```python
# Base sun-tracking angles
sun_angles = calculate_angles_from_behaviors(satellite, k1_vectors, engine)

# Add time-varying offset
offset = create_angle_interpolator(...)
final_angles = sun_angles['SP_North'] + offset
```

### Converting to Matrices

Angles must be converted to rotation matrices for shadow computation:

```python
from src.articulation import compute_rotation_matrices_from_angles

explicit_component_angles = {'SP_North': angles}
matrices = compute_rotation_matrices_from_angles(explicit_component_angles, satellite)
```

---

## 5. Attitude Control

Attitude determines how the satellite is oriented in space.

### SPICE-Based Attitudes (Default)

When attitude kernels (CK files) are available, SPICE provides accurate orientations:

```python
geometry_data = compute_observation_geometry(
    epochs, satellite_id, observer_id, spice_handler, config
)
# Attitudes are automatically applied to k1/k2 vectors
```

This is the preferred method for operational satellites with real attitude data.

### Custom Quaternion Attitudes

For prototyping or scenarios without attitude kernels, define custom orientations:

```python
import quaternion

# Define quaternion keyframes
# Quaternions use scalar-first format: (w, x, y, z)
custom_quaternions = [
    quaternion.quaternion(1.0, 0.0, 0.0, 0.0),    # Identity (no rotation)
    quaternion.quaternion(0.707, 0.707, 0.0, 0.0)  # 90° rotation
]
custom_times = [start_time_utc, end_time_utc]

# Build keyframes dict
attitude_keyframes = {
    'times': custom_times,
    'time_format': 'utc',
    'attitudes': custom_quaternions,
    'format': 'quaternion'
}

# Pass to geometry computation
geometry_data = compute_observation_geometry(
    epochs, satellite_id, observer_id, spice_handler, config,
    attitude_keyframes=attitude_keyframes  # Triggers SLERP interpolation
)
```

### SLERP Interpolation

When using custom attitudes, LCAS performs Spherical Linear Interpolation (SLERP) between keyframes. SLERP ensures smooth, constant-angular-velocity rotation between orientations.

### When to Use Each Method

| Scenario | Recommended Approach |
|----------|---------------------|
| Real satellite with CK kernels | SPICE attitudes |
| Prototype/concept | Custom quaternions |
| Sensitivity analysis | Custom quaternions |
| What-if scenarios | Custom quaternions |

---

## 6. BRDF Materials

BRDF (Bidirectional Reflectance Distribution Function) defines how light reflects from surfaces.

### Ashikhmin-Shirley Model

LCAS uses the Ashikhmin-Shirley BRDF with three parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| **r_d** | 0.0-1.0 | Diffuse reflectance (matte, Lambertian) |
| **r_s** | 0.0-1.0 | Specular reflectance (glossy, mirror-like) |
| **n_phong** | 1-1000+ | Phong exponent (surface roughness) |

### Material Examples

| Material | r_d | r_s | n_phong | Description |
|----------|-----|-----|---------|-------------|
| Solar Panel | 0.026 | 0.3 | 250 | Dark, moderately glossy |
| White MLI | 0.5 | 0.2 | 100 | Bright, slightly glossy |
| Gold Foil | 0.02 | 0.7 | 500 | Very glossy, low diffuse |
| Black Paint | 0.02 | 0.05 | 50 | Very dark, matte |
| Aluminum | 0.1 | 0.5 | 300 | Metallic appearance |

### Configuration

Define BRDF in the config file:

```yaml
component_brdf:
  Bus:
    r_d: 0.02
    r_s: 0.5
    n_phong: 300
  SP_North:
    r_d: 0.026
    r_s: 0.3
    n_phong: 250
```

### Programmatic Assignment

Override BRDF at runtime:

```python
from src.computation.brdf import BRDFManager, BRDFCalculator

brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

# Or set directly on components:
for facet in satellite.get_component('Bus').facets:
    facet.r_d = 0.05
    facet.r_s = 0.4
    facet.n_phong = 200
```

### Physical Interpretation

**Higher r_d**: Brighter in all viewing geometries, matte appearance

**Higher r_s**: Brighter specular highlights, more mirror-like

**Higher n_phong**: Sharper, more concentrated specular highlights

**Lower n_phong**: Broader, softer specular highlights

---

## 7. Shadow Computation

LCAS computes self-shadowing using ray tracing.

### How It Works

For each facet at each time point:
1. Cast a ray from facet center toward the sun (k1 direction)
2. Check if ray intersects any other facet
3. Mark facet as lit (no intersection) or shadowed (intersection)

### Facet States

| State | Description | Color in Animation |
|-------|-------------|-------------------|
| **Lit** | Receives direct sunlight | Yellow |
| **Shadowed** | Blocked by another part | Dark Blue |
| **Back-culled** | Facing away from sun | Purple |

### Using Shadow Computation

```python
from src.computation.shadow_engine import compute_shadows

# With articulation
lit_status_dict = compute_shadows(
    satellite=satellite,
    k1_vectors=k1_vectors,
    explicit_component_angles=angles_dict,
    explicit_component_matrices=matrices_dict
)

# Without articulation
lit_status_dict = compute_shadows(
    satellite=satellite,
    k1_vectors=k1_vectors
)
```

### No-Shadow Mode

For comparison or debugging, create "always lit" status:

```python
from src.computation.shadow_engine import create_no_shadow_lit_status

lit_status_unshadowed = create_no_shadow_lit_status(satellite, num_epochs)
```

---

## 8. Light Curves

Light curves show satellite brightness over time.

### Generating Light Curves

```python
from src.computation.lightcurve_generator import generate_lightcurves

magnitudes, flux, _, _, distances, animation_data = generate_lightcurves(
    facet_lit_status_dict=lit_status_dict,
    k1_vectors_array=k1_vectors,
    k2_vectors_array=k2_vectors,
    observer_distances=observer_distances,
    satellite=satellite,
    epochs=epochs,
    pre_computed_matrices=matrices,  # Optional: for articulation
    animate=True  # Collect animation data
)
```

### Output Arrays

| Output | Shape | Description |
|--------|-------|-------------|
| `magnitudes` | (N,) | Apparent magnitude at each epoch |
| `flux` | (N,) | Total flux (arbitrary units) |
| `distances` | (N,) | Observer distance (km) |
| `animation_data` | list | Per-frame facet data for animation |

### Understanding Magnitudes

Apparent magnitude is a logarithmic brightness scale:
- Smaller = brighter (magnitude 10 is brighter than 15)
- Each magnitude step = 2.512x brightness change
- Typical GEO satellites: magnitude 10-15
- Typical LEO satellites: magnitude 5-10

### Plotting Light Curves

```python
from src.visualization.lightcurve_plotter import create_light_curve_plot

create_light_curve_plot(
    time_hours=time_hours,
    epochs=epochs,
    magnitudes=magnitudes_shadowed,
    phase_angles=phase_angles,
    utc_times=utc_times,
    satellite_name=satellite.name,
    plot_mode="comparison",
    output_dir=output_dir,
    magnitudes_no_shadow=magnitudes_unshadowed,
    observer_distances=distances,
    save=True
)
```

### Shadowed vs Unshadowed

Comparing shadowed and unshadowed light curves reveals:
- Shadow timing and duration
- Impact of self-occlusion on brightness
- Glint events from specular surfaces

---

## 9. 3D Animations

Interactive 3D animations visualize satellite illumination over time.

### Creating Animations

```python
from src.visualization.plotly_animation_generator import create_interactive_3d_animation

animation_path = create_interactive_3d_animation(
    animation_data=animation_data,
    magnitudes=magnitudes,
    time_hours=time_hours,
    geometry_data=geometry_data,
    satellite_name=satellite.name,
    output_dir=output_dir,
    show_j2000_frame=True,
    show_body_frame=True,
    show_sun_vector=True,
    show_observer_vector=True,
    frame_duration_ms=100,
    color_mode='flux',
    save=True
)
```

### Color Modes

**`'lit_status'`**: Discrete coloring
- Yellow: Lit facets
- Dark Blue: Shadowed facets
- Purple: Back-culled facets

**`'flux'`**: Continuous brightness
- Dark Red → Orange → Yellow → White
- Shows relative brightness per facet
- Reveals specular highlights

### Viewing Animations

1. Open the generated `.html` file in a web browser
2. Use mouse to rotate, zoom, and pan
3. Play/pause with animation controls
4. Hover over facets for component names

### Animation Options

| Option | Description |
|--------|-------------|
| `show_j2000_frame` | Display inertial reference axes |
| `show_body_frame` | Display satellite body axes |
| `show_sun_vector` | Show sun direction arrow |
| `show_observer_vector` | Show observer direction arrow |
| `frame_duration_ms` | Time per frame (milliseconds) |

---

## 10. Configuration Files

LCAS uses YAML configuration files for satellite definitions.

### Main Configuration Structure

```yaml
name: "Satellite Name"

spice_config:
  satellite_id: -126824              # NAIF ID
  body_frame: "SATELLITE_BODY_FRAME"

simulation_defaults:
  start_time: "2020-02-05T10:00:00"  # UTC
  end_time: "2020-02-05T16:00:00"
  output_dir: "satellite_results"

components:
  Bus:
    stl_file: "bus.stl"
    position: [0.0, 0.0, 0.0]
    orientation: [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
  SP_North:
    stl_file: "sp_north.stl"
    position: [0.0, 5.0, 0.0]
    orientation: [1.0, 0.0, 0.0, 0.0]

component_brdf:
  Bus:
    r_d: 0.02
    r_s: 0.5
    n_phong: 300
  SP_North:
    r_d: 0.026
    r_s: 0.3
    n_phong: 250

articulation_capabilities:
  SP_North:
    rotation_center: [0.0, 5.0, 0.0]
    rotation_axis: [0.0, 1.0, 0.0]
    limits:
      min_angle: -180.0
      max_angle: 180.0
```

### SPICE Metakernel

Metakernels list all SPICE data files to load:

```
KPL/MK

\begindata

PATH_VALUES = ('../../generic', './kernels')
PATH_SYMBOLS = ('GEN', 'MISSION')

KERNELS_TO_LOAD = (
    '$GEN/lsk/naif0012.tls.pc'
    '$GEN/spk/de440.bsp'
    '$MISSION/satellite.bsp'
    '$MISSION/satellite_attitude.bc'
)

\begintext
```

### Adding a New Satellite

1. Create `data/models/<name>/` directory
2. Add STL files for each component
3. Create `<name>_config.yaml`
4. Add SPICE kernels to `data/spice_kernels/missions/<mission>/`
5. Create metakernel file

---

## 11. Troubleshooting

### Common Errors and Solutions

**"Metakernel not found"**
```
FileNotFoundError: Metakernel not found at ...
```
- Check that the metakernel path in config is correct
- Verify the `.tm` file exists
- Ensure SPICE kernels were downloaded: `python install_dependencies.py`

**"SPICE kernel file not found"**
```
SpiceKERNEL_NOT_FOUND: The specified kernel file was not found
```
- Check metakernel references correct files
- Verify de440.bsp exists in `data/spice_kernels/generic/spk/`
- Re-run installer without `--skip-spice`

**"STL file not found"**
```
FileNotFoundError: STL file not found: ...
```
- Check STL filename in config matches actual file
- Verify STL files are in correct model directory
- Check file extensions (.stl, .STL)

**Import errors**
```
ModuleNotFoundError: No module named 'src...'
```
- Ensure virtual environment is activated
- Check you're running from project root
- Verify sys.path includes PROJECT_ROOT

### Validation Checklist

Before running a simulation:

- [ ] Virtual environment activated
- [ ] Running from project root directory
- [ ] SPICE kernels present (`python quickstart.py`)
- [ ] STL files exist for all components
- [ ] Metakernel references valid paths
- [ ] Time range is within kernel coverage

### Getting Help

1. Run `python quickstart.py` to verify installation
2. Review example scripts in `examples/`
3. Examine existing notebooks for working code

### Performance Tips

- Reduce `num_points` for faster iteration during development
- Use `animate=False` if you don't need 3D visualization
- Start with torus_plate model (simpler, faster)
