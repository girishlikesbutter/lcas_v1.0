"""
Light Curve Plotting Module
===========================

Provides professional light curve plotting functionality with consistent styling
and proper memory management.

This module handles the extraction of plotting logic from the main script,
providing a clean separation between computation and visualization.

Classes:
    None

Functions:
    create_light_curve_plot: Generate professional light curve plots with dual axes
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

# Import the shared plotting constants
from .plot_styling import FIGURE_SIZE, PLOT_DPI, TITLE_MAPPING

logger = logging.getLogger(__name__)


def _get_date_output_dir(output_dir: Path) -> Path:
    """
    Create and return a date-based subdirectory (yymmdd format) within output_dir.

    Args:
        output_dir: Base output directory from config

    Returns:
        Path to date-based subdirectory (created if it doesn't exist)
    """
    date_folder = datetime.now().strftime("%y%m%d")
    date_output_dir = output_dir / date_folder
    date_output_dir.mkdir(parents=True, exist_ok=True)
    return date_output_dir


def _get_timestamp_prefix() -> str:
    """
    Get current time as HHMM string for filename prefix.

    Returns:
        4-digit time string (e.g., '1740' for 5:40pm, '0900' for 9:00am)
    """
    return datetime.now().strftime("%H%M")


def _get_lc_filenames(plot_mode: str, num_points: int, timestamp: str) -> tuple[str, str]:
    """
    Generate consistent PNG and CSV filenames based on plot mode.

    Filenames include HHMM timestamp prefix for uniqueness.

    Args:
        plot_mode: One of 'comparison', 'shadowed', 'no_shadows'
        num_points: Number of time points
        timestamp: HHMM timestamp string

    Returns:
        Tuple of (png_filename, csv_filename)
    """
    if plot_mode == 'comparison':
        base = f"{timestamp}_lc_comparison_{num_points}pts"
    elif plot_mode == 'shadowed':
        base = f"{timestamp}_lc_shadowed_{num_points}pts"
    else:
        base = f"{timestamp}_lc_no_shadows_{num_points}pts"

    return f"{base}.png", f"{base}.csv"


def create_light_curve_plot(
    time_hours: np.ndarray,
    epochs: np.ndarray,
    magnitudes: np.ndarray,
    phase_angles: np.ndarray,
    utc_times: List[str],
    satellite_name: str,
    plot_mode: str,
    output_dir: Path,
    magnitudes_no_shadow: Optional[np.ndarray] = None,
    observer_distances: Optional[np.ndarray] = None,
    no_plot: bool = False,
    save: bool = True,
    compare_tier2: bool = False
) -> float:
    """
    Create professional light curve plot with consistent styling.

    By default, automatically saves both PNG and CSV to a date-based subdirectory
    (yymmdd format) within output_dir. Set save=False to display without saving.

    Args:
        time_hours: Time array in hours since start
        epochs: SPICE epoch times (ET)
        magnitudes: Primary magnitude values to plot
        phase_angles: Phase angle array in degrees
        utc_times: Pre-computed UTC time strings for each epoch
        satellite_name: Satellite name for title
        plot_mode: One of 'surrogate', 'shadowed', 'no_shadows', 'comparison'
        output_dir: Base output directory (date subfolder created automatically)
        magnitudes_no_shadow: Additional magnitudes for comparison mode
        observer_distances: Observer distances in km (required for CSV saving)
        no_plot: If True, save plot but don't display
        save: If True (default), save PNG and CSV to date-based folder.
              If False, only display plot without saving any files.
        compare_tier2: If True, use Tier 2 comparison labels

    Returns:
        Time taken to generate plot in seconds

    Raises:
        ValueError: If input arrays have inconsistent lengths or invalid values
        RuntimeError: If plot generation fails
    """
    logger.info("Creating light curve plot...")
    plot_start = time.time()

    # Input validation
    try:
        if not np.isfinite(magnitudes).any():
            raise ValueError("No finite magnitude values provided")
        if len(time_hours) != len(magnitudes):
            raise ValueError("Time and magnitude arrays must have same length")
        if len(epochs) != len(magnitudes):
            raise ValueError("Epochs and magnitude arrays must have same length")
        if save and observer_distances is None:
            raise ValueError("observer_distances required when save=True")

    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return 0.0

    # Determine output paths if saving
    png_path = None
    csv_path = None
    timestamp = None
    if save:
        date_dir = _get_date_output_dir(output_dir)
        num_points = len(epochs)
        timestamp = _get_timestamp_prefix()  # Capture timestamp for both PNG and CSV
        png_filename, csv_filename = _get_lc_filenames(plot_mode, num_points, timestamp)
        png_path = date_dir / png_filename
        csv_path = date_dir / csv_filename
        logger.info(f"Output directory: {date_dir}")
    
    fig = None
    try:
        finite_mask = np.isfinite(magnitudes)
        
        # Create single magnitude plot with dual x-axes
        fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
        
        # Get title from mapping
        if compare_tier2 and plot_mode == 'comparison':
            title = f'Tier 2 Model vs Ray-Traced Comparison - {satellite_name}'
        else:
            title = TITLE_MAPPING.get(plot_mode, f'Light Curve - {satellite_name}')
            if plot_mode not in TITLE_MAPPING:
                title = f'{title} - {satellite_name}'
            
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot curves based on mode
        if plot_mode == 'comparison' and magnitudes_no_shadow is not None:
            # Plot both curves - labels depend on comparison type
            finite_mask_shadowed = np.isfinite(magnitudes)
            finite_mask_no_shadow = np.isfinite(magnitudes_no_shadow)
            
            if compare_tier2:
                # Tier2 comparison mode
                ax.scatter(time_hours[finite_mask_shadowed], magnitudes[finite_mask_shadowed], 
                          c='b', alpha=0.8, s=8, label='Ray-Traced')
                ax.scatter(time_hours[finite_mask_no_shadow], magnitudes_no_shadow[finite_mask_no_shadow], 
                          c='r', alpha=0.8, s=8, label='Tier 2 Model')
            else:
                # Shadow comparison mode
                ax.scatter(time_hours[finite_mask_shadowed], magnitudes[finite_mask_shadowed], 
                          c='b', alpha=0.8, s=8, label='With Shadows')
                ax.scatter(time_hours[finite_mask_no_shadow], magnitudes_no_shadow[finite_mask_no_shadow], 
                          c='r', alpha=0.8, s=8, label='No Shadows')
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Use shadowed curve for phase angle axis (they should be identical)
            phase_mask = finite_mask_shadowed
            phase_magnitudes = magnitudes
        else:
            # Plot single curve - use scatter for consistency
            ax.scatter(time_hours[finite_mask], magnitudes[finite_mask], c='b', alpha=0.8, s=8)
            
            phase_mask = finite_mask
            phase_magnitudes = magnitudes
        
        # Set axes labels
        ax.set_ylabel('Apparent Magnitude', fontsize=12)
        ax.invert_yaxis()  # Brighter magnitudes (smaller values) at top
        ax.grid(True, alpha=0.3)
        
        # Create second x-axis for phase angle if phase angles are available
        if phase_angles is not None:
            ax2 = ax.twiny()
            ax2.scatter(phase_angles[phase_mask], phase_magnitudes[phase_mask], alpha=0, s=0)  # invisible
            ax2.set_xlabel('Phase Angle (degrees)', fontsize=12)
            ax2.set_xlim(ax2.get_xlim()[::-1])  # Reverse to match time progression
        
        # Format time labels
        # Show 5 evenly spaced time labels
        num_labels = min(5, len(time_hours))
        label_indices = np.linspace(0, len(time_hours)-1, num_labels, dtype=int)
        label_times = time_hours[label_indices]
        label_utc = []
        for i in range(len(label_indices)):
            epoch_index = label_indices[i]
            utc_time = utc_times[epoch_index]
            # Handle both ISO format (with T) and calendar format (with space)
            if 'T' in utc_time:
                time_part = utc_time.split('T')[1][:5]  # Get HH:MM part
            else:
                # Calendar format: "2020 FEB 05 10:00:00"
                parts = utc_time.split()
                time_part = parts[-1][:5] if len(parts) >= 4 else utc_time[:5]
            label_utc.append(time_part)
        
        # Replace main x-axis labels with UTC time
        ax.set_xticks(label_times)
        ax.set_xticklabels(label_utc, rotation=0, ha='center')
        ax.set_xlabel('UTC Time', fontsize=12)
        
        # Adjust layout to prevent label overlap
        plt.subplots_adjust(bottom=0.15)

        # Save PNG and CSV if save=True
        if save and png_path is not None:
            plt.savefig(png_path, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"Plot saved: {png_path}")

            # Save CSV data with same timestamp as PNG
            from src.io.data_writer import save_lightcurve_data
            save_lightcurve_data(
                output_dir=png_path.parent,  # Use the date directory
                epochs=epochs,
                time_hours=time_hours,
                magnitudes=magnitudes,
                phase_angles=phase_angles,
                observer_distances=observer_distances,
                utc_times=utc_times,
                compare_shadows=(plot_mode == 'comparison'),
                use_shadows=(plot_mode == 'shadowed'),
                num_points=len(epochs),
                magnitudes_no_shadow=magnitudes_no_shadow,
                timestamp=timestamp
            )
            logger.info(f"CSV saved: {csv_path}")

        plot_time = time.time() - plot_start
        logger.info(f"Light curve plot generated ({plot_time:.2f}s)")

        # Handle display/no-display with proper memory management
        if not no_plot:
            plt.show()

        return plot_time
        
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        return 0.0
        
    finally:
        # CRITICAL: Always close figure to prevent memory leaks
        if fig is not None:
            plt.close(fig)
        else:
            plt.close('all')  # Fallback cleanup
            
        # Optional: Force garbage collection for memory cleanup
        import gc
        gc.collect()