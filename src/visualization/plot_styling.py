"""
Plot Styling and Configuration Module
====================================

Provides shared styling constants and matplotlib backend configuration
for consistent visualization across the LCAS project.

This module centralizes plot styling decisions and handles backend configuration
to support both interactive and animation use cases.

Constants:
    PLOT_DPI: High DPI for publication-quality plots
    FIGURE_SIZE: Consistent figure dimensions
    TITLE_MAPPING: Plot mode to title mapping

Functions:
    setup_matplotlib_backend: Configure matplotlib backend based on requirements
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Plot configuration constants
PLOT_DPI = 150
FIGURE_SIZE = (12, 8)

# Title mapping for different plot modes
TITLE_MAPPING: Dict[str, str] = {
    'surrogate': 'Neural Network Surrogate Light Curve',
    'shadowed': 'Light Curve (With Shadows)',
    'no_shadows': 'Light Curve (No Shadows)',
    'comparison': 'Light Curve Comparison'
}


def setup_matplotlib_backend(animate_flag: bool) -> None:
    """
    Configure matplotlib backend based on animation requirements.
    
    For animation generation, uses 'Agg' backend to ensure compatibility
    with remote/server environments. For regular plotting, allows default
    backend for interactive use.
    
    Args:
        animate_flag: True if animation will be generated, False otherwise
        
    Note:
        This function should be called early in the application lifecycle,
        before any matplotlib imports that depend on the backend choice.
    """
    if animate_flag:
        logger.info("Configuring matplotlib for animation (non-interactive backend)")
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for remote/server environments
        except Exception as e:
            logger.warning(f"Failed to set matplotlib backend to 'Agg': {e}")
            logger.warning("Continuing with default backend - animations may not work in headless environments")
    else:
        logger.debug("Using default matplotlib backend for interactive plotting")
        # Allow default backend for interactive use