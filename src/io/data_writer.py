"""
Data export utilities for light curve results.

Provides CSV export functionality with mode-specific formatting for shadowed,
non-shadowed, and comparison data. Maintains exact format compatibility
for downstream validation and analysis tools.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def save_lightcurve_data(
    output_dir: Path,
    epochs: NDArray[np.float64],
    time_hours: NDArray[np.float64],
    magnitudes: NDArray[np.float64],
    phase_angles: NDArray[np.float64],
    observer_distances: NDArray[np.float64],
    utc_times: List[str],
    compare_shadows: bool = False,
    use_shadows: bool = True,
    num_points: int = 300,
    magnitudes_no_shadow: Optional[NDArray[np.float64]] = None,
    timestamp: Optional[str] = None
) -> Path:
    """
    Save light curve data to CSV with mode-specific format.

    Preserves exact CSV format, headers, and data precision for compatibility
    with existing validation scripts and downstream analysis tools.

    Args:
        output_dir: Directory to save CSV file
        epochs: SPICE epoch times (ET) for CSV output
        time_hours: Time array in hours since start
        magnitudes: Primary magnitude values (shadowed or standard)
        phase_angles: Phase angle array in degrees
        observer_distances: Observer distance array in km
        utc_times: Pre-computed UTC time strings for each epoch
        compare_shadows: If True, save comparison mode with both magnitude sets
        use_shadows: If True, use shadowed naming convention
        num_points: Number of points for filename generation
        magnitudes_no_shadow: Additional magnitudes for comparison mode
        timestamp: Optional HHMM timestamp string. If not provided, generates current time.

    Returns:
        Path to the saved CSV file

    Raises:
        ValueError: If input arrays have inconsistent lengths or invalid values
        RuntimeError: If CSV saving fails
    """
    # Input validation
    if len(epochs) == 0:
        raise ValueError("Empty epochs array provided")
    if len(epochs) != len(magnitudes):
        raise ValueError("Epochs and magnitudes arrays must have same length")
    if len(epochs) != len(time_hours):
        raise ValueError("Epochs and time_hours arrays must have same length")
    if len(epochs) != len(phase_angles):
        raise ValueError("Epochs and phase_angles arrays must have same length")
    if len(epochs) != len(observer_distances):
        raise ValueError("Epochs and observer_distances arrays must have same length")
    if len(epochs) != len(utc_times):
        raise ValueError("Epochs and utc_times arrays must have same length")
    if compare_shadows and magnitudes_no_shadow is None:
        raise ValueError("Comparison mode requires magnitudes_no_shadow parameter")

    logger.info("Saving light curve data to CSV...")

    try:
        # Generate filename with HHMM timestamp prefix
        if timestamp is None:
            timestamp = datetime.now().strftime("%H%M")
        if compare_shadows:
            data_filename = f"{timestamp}_lc_comparison_{num_points}pts.csv"
        elif use_shadows:
            data_filename = f"{timestamp}_lc_shadowed_{num_points}pts.csv"
        else:
            data_filename = f"{timestamp}_lc_no_shadows_{num_points}pts.csv"

        data_path = output_dir / data_filename
        
        # Build data array and header based on mode
        if compare_shadows:
            # Comparison mode: include both shadowed and non-shadowed results
            assert magnitudes_no_shadow is not None  # Already validated above
            data_array = np.column_stack([
                np.array(utc_times, dtype=object),
                epochs,
                time_hours,
                magnitudes,  # shadowed magnitudes
                magnitudes_no_shadow,
                phase_angles,
                observer_distances / 1000  # Convert to 1000km units
            ])
            header = "UTC_Time,ET_Seconds,Hours_Since_Start,Magnitude_Shadowed,Magnitude_No_Shadow,Phase_Angle_Deg,Observer_Distance_1000km"
        else:
            # Single mode: either shadowed or non-shadowed
            data_array = np.column_stack([
                np.array(utc_times, dtype=object),
                epochs,
                time_hours,
                magnitudes,
                phase_angles,
                observer_distances / 1000  # Convert to 1000km units
            ])
            header = "UTC_Time,ET_Seconds,Hours_Since_Start,Apparent_Magnitude,Phase_Angle_Deg,Observer_Distance_1000km"
        
        # Save using exact numpy.savetxt parameters from original code
        np.savetxt(data_path, data_array, delimiter=',', header=header, fmt='%s', comments='')
        
        logger.info(f"Data saved: {data_path}")
        return data_path
        
    except Exception as e:
        logger.error(f"Failed to save light curve data: {e}")
        raise RuntimeError(f"CSV export failed: {e}") from e