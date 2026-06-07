# src/visualization - Plotting and Animation

This module provides visualization capabilities for light curves and interactive 3D satellite animations.

## Module Overview

| File | Purpose |
|------|---------|
| `lightcurve_plotter.py` | Matplotlib light curve plots with CSV export |
| `plotly_animation_generator.py` | Interactive 3D Plotly animations |
| `plot_styling.py` | Shared constants and backend configuration |

## Key Exports

```python
from src.visualization import (
    create_light_curve_plot,
    create_interactive_3d_animation,
    setup_matplotlib_backend,
    PLOT_DPI,
    FIGURE_SIZE,
    TITLE_MAPPING
)
```

## Light Curve Plotting

### create_light_curve_plot
Generate publication-quality light curve plots.

```python
from src.visualization import create_light_curve_plot

plot_time = create_light_curve_plot(
    time_hours=time_hours,
    epochs=epochs,
    magnitudes=magnitudes,
    phase_angles=phase_angles,
    utc_times=utc_times,
    satellite_name="Intelsat 901",
    plot_mode='comparison',           # 'shadowed', 'no_shadows', 'comparison'
    output_dir=output_dir,
    magnitudes_no_shadow=mag_no_shadow,
    observer_distances=observer_distances,
    no_plot=False,                    # True to save without display
    save=True                         # Save PNG and CSV
)
```

### Plot Modes
- `'shadowed'`: Single curve with shadows
- `'no_shadows'`: Single curve without shadows
- `'comparison'`: Both curves overlaid

### Output Files
Saved to `output_dir/<yymmdd>/`:
- `HHMM_lc_<mode>_<N>pts.png` - Light curve plot
- `HHMM_lc_<mode>_<N>pts.csv` - Data in CSV format

### Plot Features
- Dual x-axes (UTC time + phase angle)
- Inverted y-axis (brighter = lower magnitude = top)
- 5 evenly-spaced time labels
- Legend for comparison mode

## Interactive 3D Animation

### create_interactive_3d_animation
Generate interactive Plotly HTML animation.

```python
from src.visualization import create_interactive_3d_animation

output_path = create_interactive_3d_animation(
    animation_data=animation_data,    # From generate_light_curves_optimized
    magnitudes=magnitudes,
    time_hours=time_hours,
    geometry_data=geometry_data,
    satellite_name="Intelsat 901",
    output_dir=output_dir,
    show_j2000_frame=True,            # J2000 reference axes
    show_body_frame=True,             # Body-fixed axes
    show_sun_vector=True,             # Gold sun direction arrow
    show_observer_vector=True,        # Cyan observer arrow
    frame_duration_ms=100,            # Animation speed
    save=True
)
```

### Animation Features
- **3D Satellite View**: Facet-level illumination coloring
- **Light Curve Panel**: Synchronized magnitude marker
- **Play/Pause Controls**: Animation playback
- **Frame Slider**: Manual frame navigation
- **Interactive Camera**: Rotate, zoom, pan

### Facet Coloring
- **Yellow**: Lit by sun (visible + illuminated)
- **Dark Blue**: In shadow (occluded by other geometry)
- **Purple**: Back-culled (facing away from observer)
- **Gray**: No data

### Output
Saved to `output_dir/<yymmdd>/HHMM_animation_<N>pts.html`

## Backend Configuration

### setup_matplotlib_backend
Configure matplotlib for different environments.

```python
from src.visualization import setup_matplotlib_backend

# For animation generation (non-interactive)
setup_matplotlib_backend(animate_flag=True)  # Sets 'Agg' backend

# For interactive display
setup_matplotlib_backend(animate_flag=False)  # Uses default
```

## Shared Constants

```python
from src.visualization import PLOT_DPI, FIGURE_SIZE, TITLE_MAPPING

PLOT_DPI = 150                          # Publication quality
FIGURE_SIZE = (12, 8)                   # Standard figure size
TITLE_MAPPING = {
    'shadowed': 'Light Curve (With Shadows)',
    'no_shadows': 'Light Curve (No Shadows)',
    'comparison': 'Light Curve Comparison'
}
```

## Usage in Pipeline

```python
# Generate light curves with animation data
magnitudes, flux, mag_ns, flux_ns, distances, animation_data = \
    generate_light_curves_optimized(
        ...,
        animate=True  # Enable animation data collection
    )

# Create plot
create_light_curve_plot(
    time_hours=time_hours,
    epochs=epochs,
    magnitudes=magnitudes,
    phase_angles=phase_angles,
    utc_times=utc_times,
    satellite_name=config.name,
    plot_mode='comparison',
    output_dir=config_manager.get_output_directory(config),
    magnitudes_no_shadow=mag_ns,
    observer_distances=distances
)

# Create animation (if animate=True was used)
if animation_data:
    create_interactive_3d_animation(
        animation_data=animation_data,
        magnitudes=magnitudes,
        time_hours=time_hours,
        geometry_data=geometry_data,
        satellite_name=config.name,
        output_dir=config_manager.get_output_directory(config)
    )
```

## Cross-References

- **Animation data**: Collected by `src/computation/lightcurve_generator.py`
- **Data export**: CSV via `src/io/data_writer.py`
- **Geometry data**: From `src/computation/observation_geometry.py`
