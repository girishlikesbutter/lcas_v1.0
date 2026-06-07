# Torus Plate Test Model

Simple test model for validating shadow computation and pipeline functionality.

## Components

| Component | STL File | Description |
|-----------|----------|-------------|
| Torus | `torus.stl` | Torus ring (articulated, casts shadows) |
| Plate | `plate.stl` | Flat plate (receives shadows) |

## Purpose

This model demonstrates:
- Self-shadowing: Torus casts shadows on the plate
- Articulation: Torus rotates around Z-axis
- Simple geometry for debugging/validation

## Articulation

### Torus
- **Rotation Axis**: Z-axis `[0, 0, 1]`
- **Rotation Center**: Origin `[0, 0, 0]`
- **Range**: -180° to +180°

### Plate
- **Static** (no articulation capability defined)

## BRDF Materials

| Component | r_d | r_s | n_phong |
|-----------|-----|-----|---------|
| Torus | 0.02 | 0.5 | 300 |
| Plate | 0.026 | 0.3 | 250 |

## Configuration Notes

- Uses Intelsat 901 SPICE kernels (for testing convenience)
- Output to `data/results/torus_plate_results/`

## Usage

```python
config = config_manager.load_config("torus_plate/torus_plate_config.yaml")
satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)
```
