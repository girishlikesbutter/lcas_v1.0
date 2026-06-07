# data/results - Generated Output Files

This folder contains generated light curves, plots, and animations organized by satellite model.

## Folder Structure

```
results/
├── intelsat_901_results/        # Intelsat 901 outputs
│   ├── 251126/                  # Date folder (YYMMDD)
│   │   ├── 1430_lc_comparison_100pts.png
│   │   ├── 1430_lc_comparison_100pts.csv
│   │   └── 1430_animation_100pts.html
│   └── 251127/
├── torus_plate_results/         # Torus Plate outputs
│   └── ...
└── CLAUDE.md                    # This file
```

## Output File Naming

Files are organized in date folders (`YYMMDD`) with timestamp prefixes (`HHMM`):

| Pattern | Example | Description |
|---------|---------|-------------|
| `HHMM_lc_comparison_<N>pts.png` | `1430_lc_comparison_100pts.png` | Light curve plot (both curves) |
| `HHMM_lc_shadowed_<N>pts.png` | `0900_lc_shadowed_50pts.png` | Light curve plot (shadows only) |
| `HHMM_lc_<N>pts.csv` | `1430_lc_100pts.csv` | Light curve data |
| `HHMM_animation_<N>pts.html` | `1430_animation_100pts.html` | Interactive 3D animation |

## CSV Format

### Comparison Mode
```csv
UTC_Time,ET_Seconds,Hours_Since_Start,Magnitude_Shadowed,Magnitude_No_Shadow,Phase_Angle_Deg,Observer_Distance_1000km
```

### Single Mode
```csv
UTC_Time,ET_Seconds,Hours_Since_Start,Apparent_Magnitude,Phase_Angle_Deg,Observer_Distance_1000km
```

## Output Configuration

Results are saved based on `simulation_defaults.output_dir` in the model config:

```yaml
simulation_defaults:
  output_dir: "intelsat_901_results"  # Creates data/results/intelsat_901_results/
```

## Programmatic Access

```python
from src.config.rso_config_manager import RSO_ConfigManager

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")

# Get output directory path
output_dir = config_manager.get_output_directory(config)
# Returns: PROJECT_ROOT / "data" / "results" / "intelsat_901_results"
```

## Cross-References

- **Plotting**: `src/visualization/lightcurve_plotter.py`
- **Animation**: `src/visualization/plotly_animation_generator.py`
- **Data export**: `src/io/data_writer.py`
