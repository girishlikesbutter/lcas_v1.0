# data/models - Satellite Model Configurations

This folder contains satellite model definitions including STL mesh files, BRDF material properties, and articulation capabilities.

## Folder Structure

```
models/
├── intelsat_901/              # Geostationary communications satellite
│   ├── intelsat_901_config.yaml  # Main configuration
│   ├── bus.stl                   # Main body mesh
│   ├── sp_north.stl              # North solar panel
│   ├── sp_south.stl              # South solar panel
│   ├── ad.stl                    # Antenna dish (used twice: east/west)
│   ├── ad_east.stl               # East antenna dish (alternative)
│   └── ad_west.stl               # West antenna dish (alternative)
├── torus_plate/               # Simple test model
│   ├── torus_plate_config.yaml   # Main configuration
│   ├── torus.stl                 # Torus mesh
│   └── plate.stl                 # Plate mesh
└── CLAUDE.md                  # This file
```

## Adding a New Model

1. Create folder: `models/<model_name>/`
2. Add STL files for each component
3. Create `<model_name>_config.yaml` with:
   - Model name and SPICE settings
   - Component definitions (STL file, position, orientation)
   - BRDF material properties
   - Articulation capabilities

### Config Template

```yaml
name: "My Satellite"

spice_config:
  satellite_id: -999999        # Unique SPICE ID
  metakernel_path: "data/spice_kernels/missions/<mission>/metakernel.tm"
  body_frame: "SATELLITE_BODY_FRAME"

simulation_defaults:
  start_time: "2024-01-01T00:00:00"
  end_time: "2024-01-01T06:00:00"
  output_dir: "my_satellite_results"

components:
  Bus:
    stl_file: "bus.stl"
    position: [0.0, 0.0, 0.0]
    orientation: [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z] quaternion
  Solar_Panel:
    stl_file: "solar_panel.stl"
    position: [0.0, 5.0, 0.0]
    orientation: [1.0, 0.0, 0.0, 0.0]

component_brdf:
  Bus:
    r_d: 0.02     # Diffuse reflectivity
    r_s: 0.5      # Specular reflectivity
    n_phong: 300  # Phong exponent
  Solar_Panel:
    r_d: 0.026
    r_s: 0.3
    n_phong: 250

articulation_capabilities:
  Solar_Panel:
    rotation_center: [0.0, 5.0, 0.0]
    rotation_axis: [0.0, 1.0, 0.0]
    limits:
      min_angle: -180.0
      max_angle: 180.0
```

## STL File Requirements

- **Format**: Binary or ASCII STL
- **Units**: Meters
- **Origin**: Component origin at rotation center (for articulated parts)
- **Normals**: Outward-facing (counter-clockwise winding)

## Cross-References

- **Config loading**: `src/config/rso_config_manager.py`
- **STL loading**: `src/io/stl_loader.py`
- **BRDF application**: `src/computation/brdf.py`
- **General data patterns**: See `/data/CLAUDE.md`
