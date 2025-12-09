"""
Quaternion and angle interpolation utilities for LCAS.

Provides multiple interpolation methods:
- SLERP for quaternions (attitudes)
- Step, linear, constant for angles (articulation)
- Support for UTC and ET time formats
"""

import numpy as np
from typing import Dict, List, Union, Optional, Any, Callable
import quaternion


def slerp(q1: Union[np.ndarray, quaternion.quaternion],
          q2: Union[np.ndarray, quaternion.quaternion],
          t: float) -> np.ndarray:
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions.

    Parameters:
    -----------
    q1, q2 : array-like or quaternion
        Input quaternions in scalar-first format [w, x, y, z] or quaternion objects
    t : float
        Interpolation parameter between 0 and 1
        t = 0 returns q1, t = 1 returns q2

    Returns:
    --------
    array
        Interpolated quaternion in scalar-first format [w, x, y, z]
    """
    # Convert quaternion objects to arrays if needed
    if isinstance(q1, quaternion.quaternion):
        q1 = np.array([q1.w, q1.x, q1.y, q1.z])
    if isinstance(q2, quaternion.quaternion):
        q2 = np.array([q2.w, q2.x, q2.y, q2.z])

    # Ensure unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calculate cosine of angle between quaternions
    dot_product = np.sum(q1 * q2)

    # If quaternions are very close, just do linear interpolation
    if dot_product > 0.9995:
        result = (1 - t) * q1 + t * q2
        return result / np.linalg.norm(result)

    # If dot product is negative, take shorter path
    if dot_product < 0:
        q2 = -q2
        dot_product = -dot_product

    # Calculate interpolation parameters
    theta = np.arccos(np.clip(dot_product, -1.0, 1.0))
    sin_theta = np.sin(theta)

    # Perform SLERP
    ratio1 = np.sin((1 - t) * theta) / sin_theta
    ratio2 = np.sin(t * theta) / sin_theta

    # Calculate interpolated quaternion
    result = ratio1 * q1 + ratio2 * q2

    # Ensure unit quaternion
    return result / np.linalg.norm(result)


def step_interpolate(val1: float, val2: float, t: float, step_at: float = 0.5) -> float:
    """
    Step function interpolation: holds val1 until step_at, then jumps to val2.

    Parameters:
    -----------
    val1, val2 : float
        Values to interpolate between (typically angles in degrees)
    t : float
        Interpolation parameter between 0 and 1
    step_at : float
        Fraction of interval at which step occurs (default 0.5 for midpoint)
        e.g., 0.9 means hold val1 for 90% of interval, then jump to val2

    Returns:
    --------
    float
        Either val1 or val2 depending on t relative to step_at
    """
    return val1 if t < step_at else val2


def linear_interpolate(val1: float, val2: float, t: float) -> float:
    """
    Linear interpolation between two values.

    Parameters:
    -----------
    val1, val2 : float
        Values to interpolate between (typically angles in degrees)
    t : float
        Interpolation parameter between 0 and 1

    Returns:
    --------
    float
        Linearly interpolated value
    """
    return (1 - t) * val1 + t * val2


def constant_interpolate(val1: float, val2: float, t: float) -> float:
    """
    Constant (order 0) interpolation: always returns val1.

    Parameters:
    -----------
    val1 : float
        Value to return (typically angle in degrees)
    val2 : float
        Ignored (included for consistent interface)
    t : float
        Ignored (included for consistent interface)

    Returns:
    --------
    float
        Always returns val1
    """
    return val1


def parse_keyframe_times(times: List[Union[str, float]],
                        time_format: str,
                        spice_handler: Optional[Any] = None) -> np.ndarray:
    """
    Convert keyframe times to ephemeris time (ET).

    Parameters:
    -----------
    times : List[Union[str, float]]
        Time values in specified format
    time_format : str
        'utc' for UTC strings, 'et' for ephemeris time
    spice_handler : Optional[SpiceHandler]
        Required if time_format is 'utc' for conversion

    Returns:
    --------
    np.ndarray
        Times in ephemeris time (ET) seconds
    """
    if time_format == 'utc':
        if spice_handler is None:
            raise ValueError("spice_handler required for UTC to ET conversion")
        return np.array([spice_handler.utc_to_et(t) for t in times])
    elif time_format == 'et':
        return np.array(times, dtype=np.float64)
    else:
        # Auto-detect format
        if isinstance(times[0], str):
            if 'T' in times[0] or ':' in times[0]:
                # Looks like UTC format
                if spice_handler is None:
                    raise ValueError("spice_handler required for UTC to ET conversion")
                return np.array([spice_handler.utc_to_et(t) for t in times])
        # Assume ET
        return np.array(times, dtype=np.float64)


def create_multi_order_interpolator(keyframes: Dict[str, Any],
                                   spice_handler: Optional[Any] = None) -> Callable:
    """
    Create an interpolator supporting multiple interpolation methods per segment.

    Parameters:
    -----------
    keyframes : dict
        Dictionary containing:
        - 'times': List of time values (UTC strings or ET floats)
        - 'time_format': 'utc' or 'et' (optional, auto-detected if not provided)
        - 'values': List of values to interpolate (angles in degrees)
        - 'methods': List of interpolation methods per segment
          Options: 'constant', 'step', 'linear'
        - 'method_params': List of parameter dicts per segment
          e.g., [{'step_at': 0.9}, {}, ...]
    spice_handler : Optional[SpiceHandler]
        Required if using UTC time format

    Returns:
    --------
    function
        Interpolation function that takes time(s) and returns interpolated value(s)

    Example:
    --------
    keyframes = {
        'times': ['2024-01-01T12:00:00', '2024-01-01T13:00:00', '2024-01-01T14:00:00'],
        'time_format': 'utc',
        'values': [0, 90, 0],
        'methods': ['step', 'linear'],
        'method_params': [{'step_at': 0.9}, {}]
    }
    interpolator = create_multi_order_interpolator(keyframes, spice_handler)
    angle_at_t = interpolator(some_et_time)
    """
    # Parse times to ET
    time_format = keyframes.get('time_format', 'auto')
    times_et = parse_keyframe_times(keyframes['times'], time_format, spice_handler)
    values = np.array(keyframes['values'])
    methods = keyframes.get('methods', ['linear'] * (len(times_et) - 1))
    method_params = keyframes.get('method_params', [{}] * len(methods))

    # Validate inputs
    if len(values) != len(times_et):
        raise ValueError(f"Number of values ({len(values)}) must match number of times ({len(times_et)})")
    if len(methods) != len(times_et) - 1:
        raise ValueError(f"Number of methods ({len(methods)}) must be number of times - 1 ({len(times_et) - 1})")
    if len(method_params) != len(methods):
        raise ValueError(f"Number of method_params ({len(method_params)}) must match number of methods ({len(methods)})")

    # Map method names to functions
    method_funcs = {
        'constant': constant_interpolate,
        'step': step_interpolate,
        'linear': linear_interpolate
    }

    def interpolator(t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Interpolate value at time t using multi-order interpolation.

        Parameters:
        -----------
        t : float or array-like
            Time(s) at which to interpolate (in ET)

        Returns:
        --------
        float or array
            Interpolated value(s)
        """
        if np.isscalar(t):
            # Find the segment
            if t <= times_et[0]:
                return values[0]
            if t >= times_et[-1]:
                return values[-1]

            idx = np.searchsorted(times_et, t) - 1
            idx = max(0, min(idx, len(methods) - 1))

            # Get interpolation parameters for this segment
            t1, t2 = times_et[idx], times_et[idx + 1]
            v1, v2 = values[idx], values[idx + 1]
            fraction = (t - t1) / (t2 - t1) if t2 != t1 else 0

            # Apply the appropriate interpolation method
            method = methods[idx]
            params = method_params[idx]

            if method not in method_funcs:
                raise ValueError(f"Unknown interpolation method: {method}")

            func = method_funcs[method]

            # Apply method with parameters
            if method == 'step':
                step_at = params.get('step_at', 0.5)
                return func(v1, v2, fraction, step_at)
            else:
                return func(v1, v2, fraction)
        else:
            # Handle array of times
            return np.array([interpolator(ti) for ti in t])

    return interpolator


def create_angle_interpolator(
    keyframe_times_utc: List[str],
    keyframe_values: List[float],
    transitions: List[str],
    epochs: np.ndarray,
    utc_to_et: Callable[[str], float],
    step_params: Optional[List[float]] = None
) -> np.ndarray:
    """
    Simple angle interpolator that returns an array matching epochs.

    Parameters:
    -----------
    keyframe_times_utc : List[str]
        UTC times when angles change (e.g., ['2024-01-01T12:00:00', '2024-01-01T12:15:00'])
    keyframe_values : List[float]
        Angle values at those times in degrees
    transitions : List[str]
        How to transition between each pair of keyframes
        Options: 'step', 'linear', 'constant'
        Length should be len(keyframe_times) - 1
    epochs : np.ndarray
        The main time array in ET (ephemeris time)
    utc_to_et : Callable[[str], float]
        Function to convert UTC time string to ET (e.g., spice_handler.utc_to_et)
    step_params : Optional[List[float]]
        Step_at values for 'step' transitions (default 0.5)

    Returns:
    --------
    np.ndarray
        Array of interpolated angles matching the length of epochs

    Example:
    --------
    angles = create_angle_interpolator(
        keyframe_times_utc=['2024-01-01T12:00:00', '2024-01-01T12:15:00', '2024-01-01T12:30:00'],
        keyframe_values=[0, 45, 90],
        transitions=['step', 'linear'],
        step_params=[0.95, None],
        epochs=epochs,
        utc_to_et=spice_handler.utc_to_et
    )
    """
    # Convert UTC times to ET
    keyframe_times_et = np.array([utc_to_et(t) for t in keyframe_times_utc])
    keyframe_values = np.array(keyframe_values)

    # Validate inputs
    if len(transitions) != len(keyframe_times_et) - 1:
        raise ValueError(f"transitions length ({len(transitions)}) must be len(times) - 1 ({len(keyframe_times_et) - 1})")

    # Create angle array
    angles = np.zeros(len(epochs))

    # Default step params if not provided
    if step_params is None:
        step_params = [0.5] * len(transitions)

    # Interpolate for each epoch
    for i, t in enumerate(epochs):
        # Find which interval we're in
        if t <= keyframe_times_et[0]:
            angles[i] = keyframe_values[0]
        elif t >= keyframe_times_et[-1]:
            angles[i] = keyframe_values[-1]
        else:
            # Find the bracketing keyframes
            idx = np.searchsorted(keyframe_times_et, t) - 1
            idx = max(0, min(idx, len(transitions) - 1))

            t1, t2 = keyframe_times_et[idx], keyframe_times_et[idx + 1]
            v1, v2 = keyframe_values[idx], keyframe_values[idx + 1]

            # Calculate interpolation fraction
            if t2 != t1:
                fraction = (t - t1) / (t2 - t1)
            else:
                fraction = 0

            # Apply the transition method
            method = transitions[idx]

            if method == 'constant':
                angles[i] = v1
            elif method == 'step':
                step_at = step_params[idx] if step_params[idx] is not None else 0.5
                angles[i] = v1 if fraction < step_at else v2
            elif method == 'linear':
                angles[i] = v1 + fraction * (v2 - v1)
            else:
                raise ValueError(f"Unknown transition method: {method}")

    return angles


def create_quaternion_interpolator(sample_times: Union[List, np.ndarray],
                                  sample_quats: Union[List, np.ndarray],
                                  time_format: str = 'et',
                                  spice_handler: Optional[Any] = None) -> Callable:
    """
    Create SLERP interpolator for quaternions (always uses SLERP for attitudes).

    Parameters:
    -----------
    sample_times : array-like
        Times at which quaternions are sampled (UTC strings or ET floats)
    sample_quats : array-like
        Array of quaternion objects from numpy-quaternion
    time_format : str
        'utc' or 'et' (default 'et')
    spice_handler : Optional[SpiceHandler]
        Required if time_format is 'utc'

    Returns:
    --------
    function
        Interpolation function that takes time(s) and returns interpolated quaternion(s)
    """
    # Parse times to ET
    times_et = parse_keyframe_times(sample_times, time_format, spice_handler)

    # Convert quaternion objects to numpy array of [w, x, y, z] components
    if isinstance(sample_quats[0], quaternion.quaternion):
        quat_components = np.array([[q.w, q.x, q.y, q.z] for q in sample_quats])
    else:
        quat_components = np.array(sample_quats)

    def interpolator(t: Union[float, np.ndarray]) -> Union[quaternion.quaternion, np.ndarray]:
        """
        Return SLERP-interpolated quaternion at time t.

        Parameters:
        -----------
        t : float or array-like
            Time(s) at which to interpolate (in ET)

        Returns:
        --------
        quaternion or array of quaternions
            Interpolated quaternion(s) at requested time(s)
        """
        if np.isscalar(t):
            # Find bracketing times
            if t <= times_et[0]:
                q = quat_components[0]
                return quaternion.quaternion(q[0], q[1], q[2], q[3])
            if t >= times_et[-1]:
                q = quat_components[-1]
                return quaternion.quaternion(q[0], q[1], q[2], q[3])

            idx = np.searchsorted(times_et, t) - 1
            t1, t2 = times_et[idx], times_et[idx + 1]
            fraction = (t - t1) / (t2 - t1) if t2 != t1 else 0

            # SLERP between quaternions
            q_interp = slerp(quat_components[idx], quat_components[idx + 1], fraction)

            # Convert back to quaternion object
            return quaternion.quaternion(q_interp[0], q_interp[1], q_interp[2], q_interp[3])
        else:
            # Handle array of times
            return np.array([interpolator(ti) for ti in t])

    return interpolator