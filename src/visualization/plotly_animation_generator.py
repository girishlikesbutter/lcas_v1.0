"""
Interactive 3D animation generator using Plotly for satellite visualization.

This module provides functionality to create interactive HTML animations showing
satellite facet-level illumination over time with accompanying light curves.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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


def _get_animation_filename(num_points: int) -> str:
    """
    Generate animation filename with HHMM timestamp prefix.

    Args:
        num_points: Number of time points in the animation

    Returns:
        Filename string (e.g., '0900_animation_150pts.html')
    """
    timestamp = datetime.now().strftime("%H%M")
    return f"{timestamp}_animation_{num_points}pts.html"


# Color scheme configuration
FACET_COLORS = {
    'lit': 'yellow',         # Facets illuminated by sun
    'shadowed': 'darkblue',  # Facets in shadow
    'back_culled': 'purple', # Facets facing away from observer
    'no_data': 'gray'        # Facets with no data available
}


def create_interactive_3d_animation(
    animation_data: List[Dict],
    magnitudes: np.ndarray,
    time_hours: np.ndarray,
    geometry_data: Dict,
    satellite_name: str = "Satellite",
    output_dir: Path = Path("."),
    show_j2000_frame: bool = True,
    show_body_frame: bool = True,
    show_sun_vector: bool = True,
    show_observer_vector: bool = True,
    frame_duration_ms: int = 100,
    j2000_opacity: float = 0.3,
    body_opacity: float = 0.9,
    save: bool = True,
    color_mode: str = 'lit_status'
) -> Optional[Path]:
    """
    Create an interactive 3D animation with satellite visualization and light curve.

    By default, saves to a date-based subdirectory (yymmdd format) within output_dir
    with filename format: HHMM_animation_<numpoints>pts.html

    Args:
        animation_data: List of frame data dictionaries containing satellite geometry,
                       sun/observer directions, and lit status
        magnitudes: Array of magnitude values for the light curve
        time_hours: Array of time values in hours
        geometry_data: Dictionary containing satellite attitude matrices
        satellite_name: Name of the satellite for display
        output_dir: Base output directory (date subfolder created automatically)
        show_j2000_frame: Whether to display J2000 reference frame
        show_body_frame: Whether to display body reference frame
        show_sun_vector: Whether to display sun direction vector
        show_observer_vector: Whether to display observer direction vector
        frame_duration_ms: Duration of each frame in milliseconds
        j2000_opacity: Opacity for J2000 reference frame axes
        body_opacity: Opacity for body reference frame axes
        save: If True (default), save HTML to date-based folder.
              If False, only display animation without saving.
        color_mode: Facet coloring mode. Options:
            - 'lit_status': Discrete coloring based on shadow status (default).
              Yellow=lit, dark blue=shadowed, purple=back-culled.
            - 'flux': Continuous coloring based on facet brightness.
              Uses warm gradient (dark red→orange→yellow→white) for lit facets,
              with log scaling and global normalization across all frames.

    Returns:
        Path to the saved HTML file, or None if Plotly is not available or save=False
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed. Run: pip install plotly")
        print("Skipping animation generation")
        return None

    # Validate input
    if animation_data is None or len(animation_data) == 0:
        print("No animation data available. Re-run light curve generation with animate=True")
        return None

    # Validate color_mode
    valid_color_modes = ['lit_status', 'flux']
    if color_mode not in valid_color_modes:
        raise ValueError(f"Invalid color_mode '{color_mode}'. Must be one of: {valid_color_modes}")

    print("Creating interactive Plotly animation...")
    print(f"Preparing animation with {len(animation_data)} frames")
    print(f"Color mode: {color_mode}")
    print()

    # Prepare mesh topology and get component face ranges
    # Pass the full frame_data dict to support both new and old formats
    vertices, faces, component_face_ranges = _prepare_mesh_topology(animation_data[0])

    # Pre-compute colors and vertices for all frames
    frame_colors, frame_vertices_list, frame_stats, flux_range = _precompute_frame_data(
        animation_data, component_face_ranges, color_mode
    )

    # Create the base figure with subplots
    fig = _create_base_figure()

    # Add the main 3D mesh trace
    _add_mesh_trace(fig, vertices, faces, frame_colors[0])

    # Add colorbar for flux mode
    if color_mode == 'flux' and flux_range is not None:
        _add_flux_colorbar(fig, flux_range)

    # Add reference frames if requested
    # trace_count tracks number of traces for animation frame updates
    trace_count = 1  # Mesh trace
    if color_mode == 'flux' and flux_range is not None:
        trace_count += 1  # Colorbar trace (invisible, not updated in animation)
    j2000_trace_indices = []
    if show_j2000_frame:
        j2000_trace_indices = _add_j2000_frame(
            fig, geometry_data['sat_att_matrices'][0], j2000_opacity, start_trace_idx=trace_count
        )
        trace_count += 3

    if show_body_frame:
        _add_body_frame(fig, body_opacity)
        trace_count += 3

    # Add sun and observer vectors if requested
    sun_trace_idx = None
    obs_trace_idx = None
    if show_sun_vector:
        sun_trace_idx = trace_count
        _add_sun_vector(fig, animation_data[0]['sun_direction'])
        trace_count += 1

    if show_observer_vector:
        obs_trace_idx = trace_count
        _add_observer_vector(fig, animation_data[0]['observer_direction'])
        trace_count += 1

    # Add light curve traces
    lc_marker_idx = _add_light_curve_traces(fig, time_hours, magnitudes)

    # Generate animation frames
    frames = _generate_animation_frames(
        animation_data,
        frame_colors,
        frame_vertices_list,
        geometry_data['sat_att_matrices'],
        time_hours,
        magnitudes,
        j2000_trace_indices,
        sun_trace_idx,
        obs_trace_idx,
        lc_marker_idx,
        show_j2000_frame,
        show_sun_vector,
        show_observer_vector
    )

    fig.frames = frames

    # Add animation controls
    _add_animation_controls(fig, frame_duration_ms)

    # Apply final layout styling
    _apply_final_layout(
        fig,
        satellite_name,
        len(faces),
        len(animation_data),
        magnitudes
    )

    # Save as HTML file if save=True
    output_path = None
    if save:
        date_dir = _get_date_output_dir(Path(output_dir))
        num_points = len(animation_data)
        output_filename = _get_animation_filename(num_points)
        output_path = date_dir / output_filename
        fig.write_html(str(output_path), auto_play=False, include_plotlyjs='cdn')
        print(f"Animation saved as HTML file: {output_path}")
        print(f"Open this file in your web browser to view the interactive animation!")
        print(f"Command: firefox {output_path} &")
        print()

    # Try to display in notebook if running in Jupyter
    try:
        from IPython.display import display
        display(fig)
        print("Animation displayed successfully in notebook!")
    except:
        pass  # Not in a notebook environment

    print()
    print("="*70)
    print("VISUALIZATION ELEMENTS:")
    if color_mode == 'lit_status':
        print("  Yellow facets: Lit by sun")
        print("  Blue facets: In shadow")
        print("  Purple facets: Back-facing (away from observer)")
        print("  Gray facets: No data")
    else:  # flux mode
        print("  Warm gradient (dark red → orange → yellow → white): Facet brightness")
        print("    Brighter facets (including specular glints) appear yellow/white")
        print("    Dimmer facets appear orange/red")
        print("  Blue facets: In shadow")
        print("  Purple facets: Back-facing (away from observer)")
        if flux_range is not None:
            print(f"  Flux range: log10 = [{flux_range[0]:.2f}, {flux_range[1]:.2f}]")
    print("="*70)

    return output_path


def _prepare_mesh_topology(first_frame_data) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Prepare unified mesh topology from animation frame data.

    Args:
        first_frame_data: First frame dictionary containing 'transformed_vertices'

    Returns:
        vertices: Initial vertex positions (M*3, 3)
        faces: Face indices (M, 3)
        component_face_ranges: Mapping of component names to face index ranges
    """
    vertices = first_frame_data['transformed_vertices']
    num_vertices = len(vertices)
    num_faces = num_vertices // 3

    # Build simple sequential face topology: [0,1,2], [3,4,5], etc.
    faces = np.arange(num_vertices, dtype=np.int32).reshape(num_faces, 3)

    # Get component face ranges from lit_status keys
    component_face_ranges = {}
    lit_status = first_frame_data.get('lit_status', {})
    face_idx = 0
    for comp_name in lit_status.keys():
        num_facets_in_comp = lit_status[comp_name].shape[1]
        comp_start_face = face_idx
        comp_end_face = face_idx + num_facets_in_comp
        component_face_ranges[comp_name] = (comp_start_face, comp_end_face)
        print(f"  {comp_name}: faces {comp_start_face}-{comp_end_face}")
        face_idx = comp_end_face

    print(f"Total: {num_vertices} vertices, {num_faces} faces")
    return vertices, faces, component_face_ranges


def _precompute_frame_data(animation_data, component_face_ranges, color_mode='lit_status'):
    """
    Pre-compute colors and vertices for all animation frames.

    Args:
        animation_data: List of frame data dictionaries
        component_face_ranges: Dictionary mapping component names to face index ranges
        color_mode: 'lit_status' for discrete coloring, 'flux' for continuous brightness

    Returns:
        frame_colors: List of color arrays for each frame
        frame_vertices_list: List of vertex arrays for each frame
        frame_stats: List of statistics dictionaries for each frame
        flux_range: Tuple of (log_min, log_max) for flux mode, None for lit_status mode
    """
    print("Pre-computing frame colors...")

    frame_vertices_list = []
    frame_stats = []
    flux_range = None

    # First pass: collect vertices and (for flux mode) find global flux range
    all_flux_values = []  # For computing global normalization

    for frame_idx, frame_data in enumerate(animation_data):
        frame_vertices = frame_data['transformed_vertices']
        frame_vertices_list.append(frame_vertices)

        # Collect flux values for global normalization (flux mode only)
        if color_mode == 'flux' and 'facet_flux' in frame_data:
            facet_flux = frame_data['facet_flux']
            back_culled_facets = frame_data['back_culled_facets']
            lit_status = frame_data['lit_status']

            # Iterate through components using face ranges (works with both formats)
            for comp_name, (start_face, end_face) in component_face_ranges.items():
                for local_facet_idx in range(end_face - start_face):
                    facet_key = f"{comp_name}_{local_facet_idx}"

                    # Only collect flux for lit, front-facing facets
                    is_back_culled = facet_key in back_culled_facets and back_culled_facets[facet_key]
                    is_lit = (comp_name in lit_status and
                              local_facet_idx < lit_status[comp_name].shape[1] and
                              lit_status[comp_name][frame_idx, local_facet_idx])

                    if not is_back_culled and is_lit and facet_key in facet_flux:
                        flux_val = facet_flux[facet_key]
                        if flux_val > 0:
                            all_flux_values.append(flux_val)

    # Compute global log flux range for normalization
    if color_mode == 'flux' and len(all_flux_values) > 0:
        log_flux_values = np.log10(all_flux_values)
        log_min = np.min(log_flux_values)
        log_max = np.max(log_flux_values)
        flux_range = (log_min, log_max)
        print(f"  Flux range: log10 = [{log_min:.3f}, {log_max:.3f}]")
    elif color_mode == 'flux':
        print("  Warning: No valid flux values found for flux mode")
        flux_range = (-10, 0)  # Fallback range

    # Second pass: compute colors for each frame
    frame_colors = []

    for frame_idx, frame_data in enumerate(animation_data):
        lit_status = frame_data['lit_status']
        back_culled_facets = frame_data['back_culled_facets']
        facet_flux = frame_data.get('facet_flux', {})

        # Initialize color array for this frame
        colors = []
        n_lit = 0
        n_shadowed = 0
        n_back_culled = 0

        # Process each component's facets (using face ranges, works with both formats)
        for comp_name, (start_face, end_face) in component_face_ranges.items():
            for local_facet_idx in range(end_face - start_face):
                facet_key = f"{comp_name}_{local_facet_idx}"

                # Check back-culling first
                if facet_key in back_culled_facets and back_culled_facets[facet_key]:
                    colors.append(FACET_COLORS['back_culled'])
                    n_back_culled += 1
                else:
                    # Check lit status
                    is_lit = (comp_name in lit_status and
                              local_facet_idx < lit_status[comp_name].shape[1] and
                              lit_status[comp_name][frame_idx, local_facet_idx])

                    if is_lit:
                        if color_mode == 'lit_status':
                            colors.append(FACET_COLORS['lit'])
                        else:  # flux mode
                            flux_val = facet_flux.get(facet_key, 0)
                            color = _flux_to_color(flux_val, flux_range)
                            colors.append(color)
                        n_lit += 1
                    else:
                        colors.append(FACET_COLORS['shadowed'])
                        n_shadowed += 1

        frame_colors.append(colors)
        frame_stats.append({
            'lit': n_lit,
            'shadowed': n_shadowed,
            'back_culled': n_back_culled,
            'total': len(colors)
        })

    print(f"Pre-computed colors for {len(frame_colors)} frames")

    return frame_colors, frame_vertices_list, frame_stats, flux_range


def _flux_to_color(flux_value: float, flux_range: tuple) -> str:
    """
    Convert a flux value to an RGB color string using warm gradient.

    Gradient: dark red → red → orange → yellow → white

    Args:
        flux_value: Raw flux value (will be log-scaled)
        flux_range: Tuple of (log_min, log_max) for normalization

    Returns:
        RGB color string like 'rgb(255, 200, 100)'
    """
    if flux_value <= 0:
        # Fallback for zero/negative flux (shouldn't happen for lit facets)
        return 'rgb(80, 0, 0)'  # Very dark red

    log_min, log_max = flux_range
    log_val = np.log10(flux_value)

    # Normalize to 0-1 range
    if log_max > log_min:
        normalized = (log_val - log_min) / (log_max - log_min)
    else:
        normalized = 0.5

    # Clamp to [0, 1]
    normalized = max(0.0, min(1.0, normalized))

    # Warm gradient colormap: dark red → red → orange → yellow → white
    # Using 5 control points for smooth transition
    colormap = [
        (0.0, (80, 0, 0)),      # Dark red (dimmest)
        (0.25, (180, 30, 0)),   # Red
        (0.5, (255, 120, 0)),   # Orange
        (0.75, (255, 220, 50)), # Yellow
        (1.0, (255, 255, 255))  # Pure white (brightest/glints)
    ]

    # Interpolate between colormap points
    r, g, b = _interpolate_colormap(normalized, colormap)

    return f'rgb({int(r)}, {int(g)}, {int(b)})'


def _interpolate_colormap(t: float, colormap: list) -> tuple:
    """
    Interpolate between colormap control points.

    Args:
        t: Normalized value in [0, 1]
        colormap: List of (position, (r, g, b)) tuples

    Returns:
        Tuple of (r, g, b) values
    """
    # Find the two control points to interpolate between
    for i in range(len(colormap) - 1):
        t0, color0 = colormap[i]
        t1, color1 = colormap[i + 1]

        if t0 <= t <= t1:
            # Linear interpolation between these two points
            if t1 > t0:
                local_t = (t - t0) / (t1 - t0)
            else:
                local_t = 0.0

            r = color0[0] + local_t * (color1[0] - color0[0])
            g = color0[1] + local_t * (color1[1] - color0[1])
            b = color0[2] + local_t * (color1[2] - color0[2])

            return (r, g, b)

    # Fallback: return last color if t > 1
    return colormap[-1][1]


def _add_flux_colorbar(fig, flux_range: tuple):
    """
    Add a colorbar legend for flux mode visualization.

    Creates an invisible scatter trace with a colorbar that shows the
    warm gradient colorscale and log flux values.

    Args:
        fig: Plotly figure object
        flux_range: Tuple of (log_min, log_max) for the flux scale
    """
    log_min, log_max = flux_range

    # Define the warm colorscale matching _flux_to_color
    # Format: list of [position, 'rgb(r,g,b)'] pairs
    warm_colorscale = [
        [0.0, 'rgb(80, 0, 0)'],       # Dark red (dimmest)
        [0.25, 'rgb(180, 30, 0)'],    # Red
        [0.5, 'rgb(255, 120, 0)'],    # Orange
        [0.75, 'rgb(255, 220, 50)'],  # Yellow
        [1.0, 'rgb(255, 255, 220)']   # Near-white (brightest)
    ]

    # Create invisible scatter trace just for the colorbar
    # Using a single point at the origin that won't be visible
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                size=0,
                color=[0.5],  # Dummy value
                colorscale=warm_colorscale,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(
                        text='Facet Brightness<br>(log₁₀ flux)',
                        font=dict(color='white', size=12)
                    ),
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=[
                        f'{log_min:.1f}',
                        f'{log_min + 0.25*(log_max-log_min):.1f}',
                        f'{log_min + 0.5*(log_max-log_min):.1f}',
                        f'{log_min + 0.75*(log_max-log_min):.1f}',
                        f'{log_max:.1f}'
                    ],
                    tickfont=dict(color='white'),
                    x=0.02,
                    xpad=10,
                    len=0.4,
                    y=0.8,
                    bgcolor='rgba(10, 10, 30, 0.8)',
                    bordercolor='white',
                    borderwidth=1
                ),
                showscale=True
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )


def _create_base_figure():
    """Create the base Plotly figure with subplots."""
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{'type': 'mesh3d'}, {'type': 'scatter'}]],
        subplot_titles=('3D Satellite View (Interactive)', 'Light Curve')
    )
    return fig


def _add_mesh_trace(fig, vertices, faces, initial_colors):
    """Add the main 3D mesh trace."""
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            facecolor=initial_colors,
            hovertemplate='Facet %{text}<extra></extra>',
            text=[f'{i}' for i in range(len(faces))],
            name='Satellite',
            showscale=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
            lightposition=dict(x=100, y=100, z=100)
        ),
        row=1, col=1
    )


def _add_j2000_frame(fig, initial_att_matrix, opacity=0.3, start_trace_idx=1):
    """
    Add J2000 reference frame axes that rotate in body frame view.

    Args:
        fig: Plotly figure
        initial_att_matrix: Initial attitude matrix
        opacity: Opacity for the axes
        start_trace_idx: Starting trace index for these axes

    Returns:
        List of trace indices for J2000 axes
    """
    j2000_length = 9.0  # Increased from 3.0 for better visibility
    trace_indices = []

    # Get initial transformation: body-to-J2000 (transpose of J2000-to-body)

    # J2000 X-axis (Red)
    j2000_x = initial_att_matrix.T @ np.array([j2000_length, 0, 0])
    fig.add_trace(
        go.Scatter3d(
            x=[0, j2000_x[0]], y=[0, j2000_x[1]], z=[0, j2000_x[2]],
            mode='lines+text',
            line=dict(color='red', width=3),
            text=['', 'X_J2000'],
            textposition='top center',
            opacity=opacity,
            name='J2000 X',
            showlegend=False,
            hovertemplate='J2000 X-axis<extra></extra>'
        ),
        row=1, col=1
    )
    trace_indices.append(start_trace_idx)

    # J2000 Y-axis (Green)
    j2000_y = initial_att_matrix.T @ np.array([0, j2000_length, 0])
    fig.add_trace(
        go.Scatter3d(
            x=[0, j2000_y[0]], y=[0, j2000_y[1]], z=[0, j2000_y[2]],
            mode='lines+text',
            line=dict(color='green', width=3),
            text=['', 'Y_J2000'],
            textposition='top center',
            opacity=opacity,
            name='J2000 Y',
            showlegend=False,
            hovertemplate='J2000 Y-axis<extra></extra>'
        ),
        row=1, col=1
    )
    trace_indices.append(start_trace_idx + 1)

    # J2000 Z-axis (Blue)
    j2000_z = initial_att_matrix.T @ np.array([0, 0, j2000_length])
    fig.add_trace(
        go.Scatter3d(
            x=[0, j2000_z[0]], y=[0, j2000_z[1]], z=[0, j2000_z[2]],
            mode='lines+text',
            line=dict(color='blue', width=3),
            text=['', 'Z_J2000'],
            textposition='top center',
            opacity=opacity,
            name='J2000 Z',
            showlegend=False,
            hovertemplate='J2000 Z-axis<extra></extra>'
        ),
        row=1, col=1
    )
    trace_indices.append(start_trace_idx + 2)

    return trace_indices


def _add_body_frame(fig, opacity=0.9):
    """Add body frame axes that remain static in body frame view."""
    body_length = 8.0  # Increased from 2.0 for better visibility

    # Body X-axis (Red) - always [1,0,0]
    body_x = np.array([body_length, 0, 0])
    fig.add_trace(
        go.Scatter3d(
            x=[0, body_x[0]], y=[0, body_x[1]], z=[0, body_x[2]],
            mode='lines+text',
            line=dict(color='red', width=5),
            text=['', 'X_body'],
            textposition='top center',
            opacity=opacity,
            name='Body X',
            showlegend=False,
            hovertemplate='Body X-axis<extra></extra>'
        ),
        row=1, col=1
    )

    # Body Y-axis (Green) - always [0,1,0]
    body_y = np.array([0, body_length, 0])
    fig.add_trace(
        go.Scatter3d(
            x=[0, body_y[0]], y=[0, body_y[1]], z=[0, body_y[2]],
            mode='lines+text',
            line=dict(color='green', width=5),
            text=['', 'Y_body'],
            textposition='top center',
            opacity=opacity,
            name='Body Y',
            showlegend=False,
            hovertemplate='Body Y-axis<extra></extra>'
        ),
        row=1, col=1
    )

    # Body Z-axis (Blue) - always [0,0,1]
    body_z = np.array([0, 0, body_length])
    fig.add_trace(
        go.Scatter3d(
            x=[0, body_z[0]], y=[0, body_z[1]], z=[0, body_z[2]],
            mode='lines+text',
            line=dict(color='blue', width=5),
            text=['', 'Z_body'],
            textposition='top center',
            opacity=opacity,
            name='Body Z',
            showlegend=False,
            hovertemplate='Body Z-axis<extra></extra>'
        ),
        row=1, col=1
    )


def _add_sun_vector(fig, sun_direction):
    """Add sun direction vector visualization."""
    vector_length = 10.0  # Increased from 3.5 for better visibility
    sun_end = sun_direction * vector_length

    fig.add_trace(
        go.Scatter3d(
            x=[0, sun_end[0]],
            y=[0, sun_end[1]],
            z=[0, sun_end[2]],
            mode='lines+markers+text',
            line=dict(color='gold', width=8),
            marker=dict(size=[0, 10], color='gold', symbol='diamond'),
            text=['', 'SUN'],
            textposition='top center',
            textfont=dict(size=14, color='gold'),
            name='Sun Vector',
            hovertemplate='Sun Direction<extra></extra>'
        ),
        row=1, col=1
    )


def _add_observer_vector(fig, observer_direction):
    """Add observer direction vector visualization."""
    vector_length = 10.0  # Increased from 3.5 for better visibility
    obs_end = observer_direction * vector_length

    fig.add_trace(
        go.Scatter3d(
            x=[0, obs_end[0]],
            y=[0, obs_end[1]],
            z=[0, obs_end[2]],
            mode='lines+markers+text',
            line=dict(color='cyan', width=8),
            marker=dict(size=[0, 10], color='cyan', symbol='diamond'),
            text=['', 'OBSERVER'],
            textposition='top center',
            textfont=dict(size=14, color='cyan'),
            name='Observer Vector',
            hovertemplate='Observer Direction<extra></extra>'
        ),
        row=1, col=1
    )


def _add_light_curve_traces(fig, time_hours, magnitudes):
    """
    Add light curve plot and current position marker.

    Returns:
        Index of the marker trace
    """
    # Add static light curve
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=magnitudes,
            mode='lines',
            name='Light Curve',
            line=dict(color='black', width=2),
            hovertemplate='Time: %{x:.2f}h<br>Mag: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add current position marker
    fig.add_trace(
        go.Scatter(
            x=[time_hours[0]],
            y=[magnitudes[0]],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Current',
            hovertemplate='Current<br>Time: %{x:.2f}h<br>Mag: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Return the index of the marker trace (last added trace)
    return len(fig.data) - 1


def _generate_animation_frames(
    animation_data,
    frame_colors,
    frame_vertices_list,
    sat_att_matrices,
    time_hours,
    magnitudes,
    j2000_trace_indices,
    sun_trace_idx,
    obs_trace_idx,
    lc_marker_idx,
    show_j2000_frame,
    show_sun_vector,
    show_observer_vector
):
    """Generate animation frames for all time steps."""
    frames = []
    j2000_length = 9.0  # Increased from 3.0 for better visibility
    vector_length = 10.0  # Increased from 3.5 for better visibility

    for frame_idx in range(len(animation_data)):
        frame_data = animation_data[frame_idx]

        # Update mesh colors and geometry
        frame_verts = frame_vertices_list[frame_idx]
        mesh_update = go.Mesh3d(
            x=frame_verts[:, 0],
            y=frame_verts[:, 1],
            z=frame_verts[:, 2],
            facecolor=frame_colors[frame_idx]
        )

        # Build list of data updates and trace indices
        data_updates = [mesh_update]
        trace_indices = [0]  # Mesh is always trace 0

        # Update J2000 frame axes if shown
        if show_j2000_frame:
            att_matrix = sat_att_matrices[frame_idx]

            j2000_x = att_matrix.T @ np.array([j2000_length, 0, 0])
            j2000_frame_x = go.Scatter3d(
                x=[0, j2000_x[0]], y=[0, j2000_x[1]], z=[0, j2000_x[2]]
            )

            j2000_y = att_matrix.T @ np.array([0, j2000_length, 0])
            j2000_frame_y = go.Scatter3d(
                x=[0, j2000_y[0]], y=[0, j2000_y[1]], z=[0, j2000_y[2]]
            )

            j2000_z = att_matrix.T @ np.array([0, 0, j2000_length])
            j2000_frame_z = go.Scatter3d(
                x=[0, j2000_z[0]], y=[0, j2000_z[1]], z=[0, j2000_z[2]]
            )

            data_updates.extend([j2000_frame_x, j2000_frame_y, j2000_frame_z])
            trace_indices.extend(j2000_trace_indices)

        # Update sun vector if shown
        if show_sun_vector:
            sun_dir = frame_data['sun_direction']
            sun_end = sun_dir * vector_length
            sun_vector = go.Scatter3d(
                x=[0, sun_end[0]],
                y=[0, sun_end[1]],
                z=[0, sun_end[2]]
            )
            data_updates.append(sun_vector)
            trace_indices.append(sun_trace_idx)

        # Update observer vector if shown
        if show_observer_vector:
            obs_dir = frame_data['observer_direction']
            obs_end = obs_dir * vector_length
            obs_vector = go.Scatter3d(
                x=[0, obs_end[0]],
                y=[0, obs_end[1]],
                z=[0, obs_end[2]]
            )
            data_updates.append(obs_vector)
            trace_indices.append(obs_trace_idx)

        # Update light curve marker
        lc_marker = go.Scatter(
            x=[time_hours[frame_idx]],
            y=[magnitudes[frame_idx]]
        )
        data_updates.append(lc_marker)
        trace_indices.append(lc_marker_idx)

        # Create frame
        frame = go.Frame(
            data=data_updates,
            name=str(frame_idx),
            traces=trace_indices
        )
        frames.append(frame)

    return frames


def _add_animation_controls(fig, frame_duration_ms):
    """Add slider and play/pause controls to the figure."""

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    # Add slider
    sliders = [{
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.05,
        "y": 0,
        "steps": [
            {
                "args": [[f.name], frame_args(0)],
                "label": f"Frame {i+1}",
                "method": "animate",
            }
            for i, f in enumerate(fig.frames)
        ],
        "currentvalue": {
            "font": {"size": 12},
            "prefix": "Frame: ",
            "visible": True,
            "xanchor": "center"
        },
    }]

    # Add play/pause buttons
    updatemenus = [{
        "buttons": [
            {
                "args": [None, frame_args(frame_duration_ms)],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [[None], frame_args(0)],
                "label": "Pause",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 70},
        "type": "buttons",
        "x": 0.05,
        "y": 0.05,
    }]

    fig.update_layout(sliders=sliders, updatemenus=updatemenus)


def _apply_final_layout(fig, satellite_name, num_faces, num_frames, magnitudes):
    """Apply final layout styling and configuration."""

    # Calculate Y-axis range for inverted magnitude plot
    mag_min = np.min(magnitudes)
    mag_max = np.max(magnitudes)
    mag_range = mag_max - mag_min
    mag_padding = mag_range * 0.05  # 5% padding

    fig.update_layout(
        title=dict(
            text=f"<b>Interactive STL Satellite Animation</b><br>" +
                 f"<sub>{satellite_name} | {num_faces} facets | {num_frames} frames</sub>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=dict(x=3, y=3, z=3),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(showbackground=False, showgrid=False, title='', visible=False),
            yaxis=dict(showbackground=False, showgrid=False, title='', visible=False),
            zaxis=dict(showbackground=False, showgrid=False, title='', visible=False),
            bgcolor='rgb(10, 10, 30)'
        ),
        xaxis2=dict(title='Time (hours)', color='white', gridcolor='gray'),
        yaxis2=dict(
            title='Magnitude',
            color='white',
            gridcolor='gray',
            range=[mag_max + mag_padding, mag_min - mag_padding]  # Inverted for magnitudes
        ),
        height=700,
        showlegend=True,
        legend=dict(
            x=0.6,
            y=0.5,
            xanchor='right',
            yanchor='middle',
            bgcolor='rgba(0, 0, 0, 0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        paper_bgcolor='rgb(10, 10, 30)',
        font=dict(color='white')
    )

    # Force y-axis inversion for magnitude plot (brighter = more negative = top)
    fig.update_yaxes(autorange='reversed', row=1, col=2)