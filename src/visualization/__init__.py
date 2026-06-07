"""
LCAS Visualization Module
========================

Provides visualization functionality for satellite light curve generation,
including plotting and animation capabilities for validation.

Features:
- Professional light curve plotting with dual-axis support
- Interactive 3D Plotly animations with facet-level illumination
- Consistent styling and backend configuration
- Memory-efficient matplotlib handling

Modules:
- lightcurve_plotter: Main plotting functionality
- plotly_animation_generator: Interactive 3D animations (Plotly-based)
- plot_styling: Shared constants and backend configuration
"""

from .lightcurve_plotter import create_light_curve_plot
from .plotly_animation_generator import create_interactive_3d_animation
from .plot_styling import setup_matplotlib_backend, PLOT_DPI, FIGURE_SIZE, TITLE_MAPPING

__all__ = [
    'create_light_curve_plot',
    'create_interactive_3d_animation',
    'setup_matplotlib_backend',
    'PLOT_DPI',
    'FIGURE_SIZE',
    'TITLE_MAPPING'
]