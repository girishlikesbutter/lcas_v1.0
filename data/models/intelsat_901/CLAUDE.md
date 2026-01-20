# Intelsat 901 Model

Geostationary communications satellite model for light curve simulation.

## Components

| Component | STL File | Description |
|-----------|----------|-------------|
| Bus | `bus.stl` | Main satellite body |
| SP_North | `sp_north.stl` | North solar panel (articulated) |
| SP_South | `sp_south.stl` | South solar panel (articulated) |
| AD_West | `ad.stl` | West antenna dish (articulated) |
| AD_East | `ad.stl` | East antenna dish (articulated) |

## Articulation

### Solar Panels (SP_North, SP_South)
- **Rotation Axis**: Z-axis `[0, 0, 1]`
- **Rotation Center**: Origin `[0, 0, 0]`
- **Range**: -180° to +180° (full rotation)
- **Typical Behavior**: Sun tracking

### Antenna Dishes (AD_West, AD_East)
- **Rotation Axis**: Z-axis (opposite signs)
- **Rotation Center**: Offset from origin
- **Range**: 0° to 90°
- **Note**: Can be animated with step/linear transitions

## BRDF Materials

| Component | r_d | r_s | n_phong | Description |
|-----------|-----|-----|---------|-------------|
| Bus | 0.02 | 0.5 | 300 | Low diffuse, high specular |
| SP_North/South | 0.026 | 0.3 | 250 | Solar cells |
| AD_West/East | 0.01 | 0.4 | 200 | Metal dish |

## SPICE Configuration

- **Satellite ID**: -126824
- **Body Frame**: IS901_BUS_FRAME
- **Metakernel**: `data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm`

## Default Simulation

- **Start**: 2020-02-05T10:00:00 UTC
- **End**: 2020-02-05T16:00:00 UTC
- **Output**: `data/results/intelsat_901_results/`

## Usage

```python
from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)
```
