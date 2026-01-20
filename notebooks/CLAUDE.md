# LCAS Notebooks - Workflow Reference

This folder contains Jupytext-linked notebooks demonstrating the LCAS light curve generation pipeline. The `.py` files are the source of truth and sync with `.ipynb` files.

## Available Notebooks

### lcas_stl_pipeline-is901-3.py (Primary Reference)
The most complete and up-to-date workflow demonstrating:
- STL-based satellite model loading
- Runtime articulation behavior assignment
- Custom quaternion-based attitude interpolation (SLERP)
- Time-varying articulation angle interpolation with step/linear transitions
- Shadow computation with pre-computed rotation matrices
- Light curve generation with animation data collection
- Interactive 3D Plotly visualization

**Key Features:**
- `CUSTOM_QUATS` toggle for SPICE vs custom attitude interpolation
- `create_angle_interpolator()` for complex articulation profiles
- Explicit component angle/matrix workflow for full control

### lcas_stl_pipeline-is901-2.py
Similar to -3.py but with custom attitudes enabled by default. Demonstrates:
- Custom quaternion keyframes with SLERP interpolation
- Angle interpolation with `spice_handler` parameter (legacy API)

### lcas_stl_pipeline.py
Simpler workflow for the torus_plate test model. Good for:
- Basic pipeline understanding without complex articulation
- Testing new behaviors (sun_tracking, fixed_angle, oscillating examples)

## Jupytext Usage

These notebooks use Jupytext's percent format for version control:

```bash
# Sync .py to .ipynb (generates notebook from Python file)
jupytext --sync notebooks/lcas_stl_pipeline-is901-3.py

# Open .py directly in Jupyter (requires jupytext extension)
jupyter notebook notebooks/lcas_stl_pipeline-is901-3.py
```

## Pipeline Cell Structure

All notebooks follow this cell organization:

| Cell | Purpose |
|------|---------|
| 1 | Configuration and Imports |
| 2 | Load Satellite from STL Files |
| 3 | Runtime Articulation Assignment |
| 4 | SPICE Setup |
| 5 | Compute Observation Geometry |
| 6 | Angle Interpolation (optional) |
| 7 | Shadow Computation |
| 8 | Light Curve Generation |
| 9 | Comparison Plot |
| 10 | Interactive 3D Animation |

## Key Import Patterns

```python
# Configuration
from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager

# Model Loading
from src.io.stl_loader import STLLoader

# SPICE
from src.spice.spice_handler import SpiceHandler

# Articulation
from src.articulation import (
    ArticulationEngine,
    SunTrackingBehavior,
    calculate_angles_from_behaviors,
    compute_rotation_matrices_from_angles
)

# Interpolation
from src.interpolation import create_angle_interpolator

# Computation
from src.computation.observation_geometry import compute_observation_geometry
from src.computation import generate_lightcurves
from src.computation.shadow_engine import compute_shadows, create_no_shadow_lit_status

# Visualization
from src.visualization.lightcurve_plotter import create_light_curve_plot
from src.visualization.plotly_animation_generator import create_interactive_3d_animation
```

## Custom Attitude Workflow (from -3.py)

```python
import quaternion

# Toggle between SPICE and custom attitudes
CUSTOM_QUATS = True

# Define quaternion keyframes
custom_quaternions = [
    quaternion.quaternion(0.707, 0.707, 0.0, 0.0),  # Start
    quaternion.quaternion(0.707, 0.0, 0.707, 0.0)   # End
]
custom_attitude_times = [start_time_utc, end_time_utc]

# Build keyframes dict for interpolation
attitude_keyframes = {
    'times': custom_attitude_times,
    'time_format': 'utc',
    'attitudes': custom_quaternions,
    'format': 'quaternion'
}

# Pass to geometry computation
geometry_data = compute_observation_geometry(
    epochs, satellite_id, spice_handler, config,
    attitude_keyframes=attitude_keyframes  # Triggers SLERP
)
```

## Angle Interpolation Workflow (from -3.py)

```python
from src.interpolation import create_angle_interpolator

# Create time-varying offset with transitions
offset_angles = create_angle_interpolator(
    keyframe_times_utc=['2020-02-05T10:00:00', '2020-02-05T10:30:00', '2020-02-05T14:30:00'],
    keyframe_values=[5.5, 8.2, 13.5],
    transitions=['linear', 'step'],
    step_params=[None, 0.99],  # Step at 99% through interval
    epochs=epochs,
    utc_to_et=spice_handler.utc_to_et
)

# Compose with sun-tracking angles
sp_angles = sun_angles + offset_angles

# Build explicit angles dict
explicit_component_angles = {
    'SP_North': sp_angles,
    'SP_South': sp_angles,
    'AD_West': ad_angles,
    'AD_East': ad_angles
}

# Convert to matrices
explicit_component_matrices = compute_rotation_matrices_from_angles(
    explicit_component_angles, satellite
)
```

## Output Files

Results are saved to `data/results/<satellite>_results/<yymmdd>/`:
- `HHMM_lc_comparison_<N>pts.png` - Light curve plot
- `HHMM_lc_comparison_<N>pts.csv` - Light curve data
- `HHMM_animation_<N>pts.html` - Interactive 3D animation

## Cross-References

- **Root overview**: See `/CLAUDE.md`
- **Source modules**: See `/src/CLAUDE.md`
- **Data configuration**: See `/data/CLAUDE.md`
