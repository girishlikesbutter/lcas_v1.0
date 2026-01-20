# LCAS Data Configuration - Models, SPICE & BRDF

This file provides specialized guidance for configuring LCAS data files, including satellite models, SPICE kernels, BRDF materials, and articulation rules. These patterns apply when creating or modifying configuration files in the `data/` directory.

## Scope & Context

**When this applies**: Working with configuration files in the `data/` directory  
**Inherits from**: `/CLAUDE.md` for general project standards  
**Consumed by**: `src/config/` modules - see `/src/CLAUDE.md` for usage patterns  
**Domain focus**: Satellite model configuration and orbital mechanics data

## Data Directory Structure

### Directory Organization
```
data/
├── models/                      # Satellite model configurations
│   ├── intelsat_901_config.yaml     # Main configuration (master)
│   ├── intelsat_901_model.yaml      # Geometric model definition
│   ├── intelsat_901_brdf.yaml       # Material properties
│   ├── intelsat_901_articulation.yaml # Articulation rules
│   ├── pyramid_dumbbell_config.yaml # Simple test model
│   └── pyramid_dumbbell_*.yaml      # Component files for test model
├── spice_kernels/               # SPICE orbital mechanics data
│   ├── generic/                      # Standard SPICE kernels
│   │   ├── lsk/                     # Leap second kernels
│   │   ├── pck/                     # Planetary constants
│   │   └── spk/                     # Ephemeris data
│   └── missions/                    # Mission-specific kernels
│       ├── dst-is901/               # Intelsat 901 mission data
│       └── mmt-galaxy4r/            # MMT Galaxy 4R mission data
└── training/                    # Training data for surrogate models
    └── *.csv                        # Shadow training datasets
```

### File Relationships
- **Config files** (`*_config.yaml`) reference model, BRDF, and articulation files
- **Model files** (`*_model.yaml`) define satellite geometry using PyYAML tags
- **BRDF files** (`*_brdf.yaml`) specify material properties for components
- **Articulation files** (`*_articulation.yaml`) define movement behaviors

## Satellite Model Configuration Patterns

### Master Configuration Files
```yaml
# intelsat_901_config.yaml - Master configuration pattern
model_info:
  name: "Intelsat 901"
  model_file: "intelsat_901_model.yaml"         # Geometric definition
  brdf_file: "intelsat_901_brdf.yaml"           # Material properties
  articulation_file: "intelsat_901_articulation.yaml"  # Movement rules

spice_config:
  satellite_id: -126824                          # NAIF ID for SPICE
  metakernel_path: "data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm"
  body_frame: "IS901_BUS_FRAME"                 # Satellite body-fixed frame

simulation_defaults:
  subdivision_level: 3                          # Mesh refinement level
  start_time: "2020-02-05T10:00:00"            # UTC simulation start
  end_time: "2020-02-05T16:00:00"              # UTC simulation end
  output_dir: "lightcurve_results"              # Default output location
```

### Geometric Model Files
```yaml
# intelsat_901_model.yaml - Satellite geometry pattern
!src.models.model_definitions.Satellite
id: '-126824'
name: Intelsat 901
components:
- !src.models.model_definitions.Component
  id: IS901_BUS
  name: Bus
  conceptual_face_definitions:
    Bus_Face_X_Positive:
    - [2.8, -1.75, -1.4]    # Vertex coordinates (meters)
    - [2.8,  1.75, -1.4]
    - [2.8,  1.75,  1.4]
    - [2.8, -1.75,  1.4]
    Bus_Face_X_Negative:
    - [-2.8, -1.75,  1.4]
    - [-2.8,  1.75,  1.4]
    - [-2.8,  1.75, -1.4]
    - [-2.8, -1.75, -1.4]
    # Additional faces...
  position: [0.0, 0.0, 0.0]          # Component position (meters)
  rotation: [0.0, 0.0, 0.0]          # Component rotation (radians)
  parent_component: null              # Parent for hierarchical models
  children: []                        # Child components
```

### BRDF Material Configuration
```yaml
# intelsat_901_brdf.yaml - Material properties pattern
brdf_mappings:
  Bus:                               # Component name from model
    r_d: 0.02                       # Diffuse reflectance (0.0-1.0)
    r_s: 0.5                        # Specular reflectance (0.0-1.0)
    n_phong: 300.0                  # Phong exponent (>0)
  Solar_Panel_North:
    r_d: 0.026                      # Higher diffuse for solar panels
    r_s: 0.3                        # Lower specular
    n_phong: 200.0                  # Different surface roughness
  Solar_Panel_South:
    r_d: 0.026
    r_s: 0.3
    n_phong: 200.0
  Antenna_Dish_West:
    r_d: 0.01                       # Low diffuse for metal dishes
    r_s: 0.4                        # Higher specular reflection
    n_phong: 200.0
  # Additional component materials...
```

### Articulation Rules Configuration
```yaml
# intelsat_901_articulation.yaml - Movement behavior pattern
articulation_rules:
  Solar_Panel_North:
    behavior_type: "sun_tracking"    # Behavior class name
    rotation_center: [0.0, 3.5, 0.0]  # Rotation axis point (meters)
    rotation_axis: [0.0, 0.0, 1.0]     # Rotation axis vector (normalized)
    limits:
      min_angle: -180.0             # Minimum rotation (degrees)
      max_angle: 180.0              # Maximum rotation (degrees)
    parameters:
      tracking_efficiency: 0.95     # Solar tracking efficiency (0.0-1.0)
      slew_rate: 1.0                # Degrees per second
  Solar_Panel_South:
    behavior_type: "sun_tracking"
    rotation_center: [0.0, -3.5, 0.0]
    rotation_axis: [0.0, 0.0, 1.0]
    limits:
      min_angle: -180.0
      max_angle: 180.0
    parameters:
      tracking_efficiency: 0.95
      slew_rate: 1.0
  # Additional articulated components...
```

## SPICE Kernel Organization

### Mission Directory Structure
```
spice_kernels/missions/dst-is901/
├── INTELSAT_901-metakernel.tm      # SPICE metakernel (kernel list)
├── INTELSAT_901-metakernel-v2.tm   # Updated metakernel format
└── kernels/                        # Individual kernel files
    ├── DST_OBS.bsp                 # Observer location (ground station)
    ├── IS901.bsp                   # Satellite ephemeris
    ├── IS901-clock.tsc             # Spacecraft clock correlation
    ├── IS901_BUS-frame.tf          # Reference frame definitions
    ├── IS901_BUS-orientation.bc    # Attitude/orientation data
    └── IS901_SP-*.bc               # Solar panel orientation kernels
```

### Metakernel File Patterns
```
# INTELSAT_901-metakernel.tm - SPICE kernel loading pattern
KPL/MK

\begindata

   PATH_VALUES = ( 'data/spice_kernels' )

   PATH_SYMBOLS = ( 'KERNELS' )

   KERNELS_TO_LOAD = (
      '$KERNELS/generic/lsk/naif0012.tls',           # Leap seconds
      '$KERNELS/generic/pck/pck00011.tpc',           # Planetary constants
      '$KERNELS/generic/spk/de440.bsp',              # Planetary ephemeris
      '$KERNELS/missions/dst-is901/kernels/DST_OBS.bsp',      # Ground station
      '$KERNELS/missions/dst-is901/kernels/IS901.bsp',        # Satellite orbit
      '$KERNELS/missions/dst-is901/kernels/IS901-clock.tsc',  # Clock correlation
      '$KERNELS/missions/dst-is901/kernels/IS901_BUS-frame.tf',     # Frames
      '$KERNELS/missions/dst-is901/kernels/IS901_BUS-orientation.bc', # Attitude
   )

\begintext
```

### SPICE ID Conventions
```yaml
# SPICE NAIF ID assignments for LCAS satellites
spice_ids:
  intelsat_901: -126824             # Intelsat 901 satellite
  galaxy_4r: -999999               # Galaxy 4R (example)
  ground_station_dst: 399001       # Deep Space Telescope
  ground_station_mmt: 399002       # McMath-Pierce Solar Telescope
```

## Configuration Validation Patterns

### Required Configuration Elements
```yaml
# Minimum required configuration for any satellite model
required_fields:
  model_info:
    - name                          # Human-readable satellite name
    - model_file                    # Geometry definition file
    - brdf_file                     # Material properties file
  spice_config:
    - satellite_id                  # NAIF ID for SPICE operations
    - metakernel_path              # Path to SPICE metakernel
  simulation_defaults:
    - subdivision_level             # Mesh refinement (1-4)
    - start_time                    # ISO 8601 UTC time string
    - end_time                      # ISO 8601 UTC time string
```

### Component Naming Conventions
```yaml
# Standard component naming patterns for LCAS satellites
component_naming:
  bus_components:
    - "Bus"                         # Main satellite body
    - "Bus_Primary"                 # Primary bus section
    - "Bus_Secondary"               # Secondary bus section
  solar_panels:
    - "Solar_Panel_North"           # +Y axis solar panel
    - "Solar_Panel_South"           # -Y axis solar panel  
    - "Solar_Panel_East"            # +X axis solar panel
    - "Solar_Panel_West"            # -X axis solar panel
  antennas:
    - "Antenna_Dish_West"           # Communication dish
    - "Antenna_Array_Main"          # Antenna array
    - "Antenna_HGA"                 # High gain antenna
  instruments:
    - "Instrument_Camera"           # Optical instruments
    - "Instrument_Spectrometer"     # Scientific instruments
```

## File Naming Conventions

### Configuration File Naming
```
# Pattern: {satellite_name}_{file_type}.yaml
intelsat_901_config.yaml           # Master configuration
intelsat_901_model.yaml             # Geometric model
intelsat_901_brdf.yaml              # Material properties
intelsat_901_articulation.yaml      # Movement rules

# For alternative models:
pyramid_dumbbell_config.yaml        # Simple test satellite
pyramid_dumbbell_model.yaml
pyramid_dumbbell_brdf.yaml
pyramid_dumbbell_articulation.yaml
```

### SPICE File Organization
```
# Pattern: missions/{mission_name}/
missions/dst-is901/                 # Deep Space Telescope - Intelsat 901
missions/mmt-galaxy4r/              # McMath-Pierce - Galaxy 4R

# Metakernel naming:
{SATELLITE_NAME}-metakernel.tm      # INTELSAT_901-metakernel.tm
{SATELLITE_NAME}-metakernel-v2.tm   # Updated format version
```

## Configuration Inheritance Patterns

### Base Configuration Override
```yaml
# Advanced pattern: configuration inheritance
base_config: "satellite_base_config.yaml"    # Base configuration file

# Override specific values
model_info:
  name: "Custom Satellite"                    # Override name
  # Other fields inherited from base

spice_config:
  satellite_id: -999999                       # Override SPICE ID
  # Other SPICE settings inherited

# simulation_defaults fully inherited from base
```

### Multi-Mission Support
```yaml
# Support multiple missions for the same satellite
missions:
  dst_observation:
    observer_location: "DST"
    observation_period: ["2020-02-05T10:00:00", "2020-02-05T16:00:00"]
    metakernel_path: "data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm"
  mmt_observation:
    observer_location: "MMT"
    observation_period: ["2020-03-15T08:00:00", "2020-03-15T14:00:00"]
    metakernel_path: "data/spice_kernels/missions/mmt-galaxy4r/GALAXY_4R.bsp"
```

## Data Validation Rules

### Geometric Constraints
```yaml
# Validation rules for satellite geometry
geometry_validation:
  coordinate_system: "body_fixed"            # Right-handed coordinate system
  units: "meters"                           # All distances in meters
  face_winding: "counter_clockwise"         # Face normal direction
  vertex_precision: 0.001                  # Minimum vertex spacing (mm)
  max_component_size: 50.0                  # Maximum component dimension (m)
```

### BRDF Property Ranges
```yaml
# Valid ranges for BRDF material properties
brdf_validation:
  r_d: [0.0, 1.0]                          # Diffuse reflectance range
  r_s: [0.0, 1.0]                          # Specular reflectance range
  n_phong: [1.0, 1000.0]                   # Phong exponent range
  total_reflectance: [0.0, 1.0]            # r_d + r_s <= 1.0
```

### Articulation Limits
```yaml
# Validation for articulation parameters
articulation_validation:
  rotation_angle_limits: [-360.0, 360.0]   # Rotation range (degrees)
  slew_rate_limits: [0.1, 10.0]           # Movement speed (deg/s)
  tracking_efficiency: [0.0, 1.0]          # Efficiency factor
  rotation_axis_magnitude: [0.99, 1.01]    # Normalized axis vector
```

## Cross-References

- **Configuration loading**: See `/src/CLAUDE.md` for `src/config/` module patterns
- **Testing configurations**: See `/tests/CLAUDE.md` for validation test patterns
- **General project standards**: See `/CLAUDE.md`
- **Script usage**: See `/scripts/CLAUDE.md` for data loading in scripts

Remember: All configuration files support the satellite light curve generation mission. Maintain consistency with SPICE conventions and ensure geometric models are physically realistic.