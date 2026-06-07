#!/usr/bin/env python3
"""Micro-44 -- Normal-family sphere visualization.

Plots a unit sphere showing the relationship between facet normal families
and the Phase Angle Bisector (PAB) trajectory.

Two frame modes:
  body:      Normals are fixed points, PAB sweeps across the sphere.
  inertial:  Normals trace trajectories as the body tumbles, PAB is nearly fixed.

Glints are marked where PAB passes near (or crosses through) a normal family.
A side-by-side mode shows both views together.

Uses saved m041 results for glint identification (hi-fi oracle labels).
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for projection='3d')
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation

from src.config.rso_config_manager import RSO_ConfigManager
from src.computation.brdf import BRDFManager, BRDFCalculator
from src.io.stl_loader import STLLoader
from src.spice.spice_handler import SpiceHandler
from src.computation.observation_geometry import compute_observation_geometry
from src.computation.inertia_calculator import compute_inertia_from_config
from src.articulation import compute_rotation_matrices_from_angles
from src.computation.facet_data_extractor import extract_facet_arrays, apply_articulation_to_arrays
from src.dynamics.attitude_propagator import propagate_attitude

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m044_normal_sphere"

# ===== Configuration =====
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # CLI or default
MODE = 'ndot_pab'                  # ..., 'ndot_pab', 'ndot_compare', or 'correlation'
SEED_B = 0                         # second trajectory for comparison
OMEGA_MAG_RANGE_DPS = (0.5, 5.0)   # same as m041
# Ordered: +X, -X, +Y, -Y, +Z, -Z, +WD, -WD, +ED, -ED
SHOW_GROUPS = [13, 0, 8, 5, 7, 6, 11, 2, 12, 1]
PAB_FRACTION = None                # fraction of trajectory to show (None = all)

# ===== Color scheme =====
COMPONENT_PALETTE = {
    frozenset(['AD_East']):  '#d62728',   # red
    frozenset(['AD_West']):  '#ff7f0e',   # orange
}
BUS_SP_COLOR = '#1f77b4'                  # blue  (Bus + solar panels)
ALL_COMP_COLOR = '#9467bd'                # purple (±Z faces, all 5 components)
PAB_COLOR = '#FFD700'                     # bright yellow


def get_group_color(components):
    """Assign color based on component membership."""
    key = frozenset(components)
    if key in COMPONENT_PALETTE:
        return COMPONENT_PALETTE[key]
    if len(key) >= 4:
        return ALL_COMP_COLOR
    return BUS_SP_COLOR


GROUP_NAMES = {
    0: '-X', 1: '-ED', 2: '-WD', 5: '-Y', 6: '-Z',
    7: '+Z', 8: '+Y', 11: '+WD', 12: '+ED', 13: '+X',
}


def get_group_label(components, group_id=None):
    """Short human-readable label for a normal group."""
    if group_id is not None and group_id in GROUP_NAMES:
        return GROUP_NAMES[group_id]
    s = set(components)
    if s == {'AD_East'}:
        return 'ED'
    if s == {'AD_West'}:
        return 'WD'
    if len(s) >= 4:
        return '±Z'
    return 'Bus'


# ===========================================================================
# Drawing helpers
# ===========================================================================
def draw_sphere(ax, n_lat=20, n_lon=40, grid_alpha=0.15, grid_color='grey',
                surface_color='#e8e8e8', surface_alpha=0.85):
    """Draw an opaque sphere with light grid lines on the surface."""
    # Opaque surface
    u = np.linspace(0, 2 * np.pi, n_lon + 1)
    v = np.linspace(0, np.pi, n_lat + 1)
    x = np.outer(np.sin(v), np.cos(u))
    y = np.outer(np.sin(v), np.sin(u))
    z = np.outer(np.cos(v), np.ones_like(u))
    ax.plot_surface(x, y, z, color=surface_color, alpha=surface_alpha,
                    shade=False, zorder=1, antialiased=True)
    # Grid lines on surface (slightly outside to avoid z-fighting)
    r = 1.002
    n_grid_lon, n_grid_lat = 12, 6
    u_grid = np.linspace(0, 2 * np.pi, 100)
    v_grid = np.linspace(0, np.pi, 100)
    for i in range(n_grid_lon):
        phi = 2 * np.pi * i / n_grid_lon
        ax.plot(r * np.sin(v_grid) * np.cos(phi),
                r * np.sin(v_grid) * np.sin(phi),
                r * np.cos(v_grid),
                color=grid_color, alpha=grid_alpha, linewidth=0.4, zorder=2)
    for j in range(1, n_grid_lat):
        theta = np.pi * j / n_grid_lat
        ax.plot(r * np.sin(theta) * np.cos(u_grid),
                r * np.sin(theta) * np.sin(u_grid),
                r * np.cos(theta) * np.ones_like(u_grid),
                color=grid_color, alpha=grid_alpha, linewidth=0.4, zorder=2)


def draw_axis_labels(ax, length=1.35, fontsize=7, alpha=0.4):
    """Draw faint axis direction labels at the tips."""
    for axis, label in [([length, 0, 0], '+X'), ([0, length, 0], '+Y'),
                        ([0, 0, length], '+Z')]:
        ax.text(*axis, label, fontsize=fontsize, alpha=alpha, ha='center')


def format_sphere_axes(ax, title=None):
    """Standard formatting for sphere axes."""
    lim = 1.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X_body', fontsize=8, labelpad=1)
    ax.set_ylabel('Y_body', fontsize=8, labelpad=1)
    ax.set_zlabel('Z_body', fontsize=8, labelpad=1)
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title, fontsize=11, pad=10)


# ===========================================================================
# Core plotting functions
# ===========================================================================
def plot_body_frame(ax, unique_normals, group_info, pab_body,
                    glint_epochs, glint_groups, glint_mags,
                    show_groups=None, pab_fraction=None):
    """
    Body-frame sphere: normals are fixed dots, glints shown at PAB location.

    Glints appear as stars at the PAB position at the glint epoch, with
    dotted lines to the responsible normal dot.

    show_groups: list of group IDs to plot, or None for all.
    """
    draw_sphere(ax)
    draw_axis_labels(ax)

    n_groups = len(unique_normals)
    group_colors = [get_group_color(g['components']) for g in group_info]
    max_area = max(g['total_area'] for g in group_info)
    groups_to_show = show_groups if show_groups is not None else list(range(n_groups))

    # ----- Normal vectors (arrows from origin, protruding past sphere) -----
    VEC_LENGTH = 1.25  # extend past r=1 surface
    for g in groups_to_show:
        n = unique_normals[g]
        color = group_colors[g]
        area = group_info[g]['total_area']
        lw = 1.0 + 2.5 * (area / max_area)
        tip = n * VEC_LENGTH
        ax.quiver(0, 0, 0, tip[0], tip[1], tip[2],
                  color=color, arrow_length_ratio=0.06,
                  linewidth=lw, alpha=0.9, zorder=10)
        # Dot at tip
        tip_size = 30 + 120 * (area / max_area)
        ax.scatter(*tip, s=tip_size, c=color, edgecolors='black', linewidth=0.5,
                   zorder=11, alpha=0.95)
        label = get_group_label(group_info[g]['components'], g)
        ax.text(n[0] * 1.35, n[1] * 1.35, n[2] * 1.35,
                f'G{g} {label}', fontsize=7, ha='center',
                va='center', fontweight='bold', alpha=0.8, zorder=12)

    # ----- PAB trace projected onto sphere surface -----
    SURF_R = 1.005  # tiny offset above surface to prevent z-fighting
    n_obs = len(pab_body)
    pab_end = int(n_obs * pab_fraction) if pab_fraction is not None else n_obs
    if pab_end > 0:
        pab_slice = pab_body[:pab_end]
        # Project onto sphere: normalize and scale to SURF_R
        pab_surf = pab_slice / np.linalg.norm(pab_slice, axis=1, keepdims=True) * SURF_R
        ax.plot(pab_surf[:, 0], pab_surf[:, 1], pab_surf[:, 2],
                color=PAB_COLOR, linewidth=2.0, alpha=0.9, zorder=8)
        # Start / end markers
        ax.scatter(*pab_surf[0], s=100, c='lime', edgecolors='black',
                   linewidth=1.2, marker='o', zorder=9)
        ax.scatter(*pab_surf[-1], s=100, c='crimson', edgecolors='black',
                   linewidth=1.2, marker='s', zorder=9)

    # ----- Glint markers on sphere surface -----
    for epoch_idx, grp_id, gmag in zip(glint_epochs, glint_groups, glint_mags):
        if show_groups is not None and grp_id not in show_groups:
            continue
        if epoch_idx >= pab_end:
            continue
        pab_pt = pab_body[epoch_idx]
        pab_pt_surf = pab_pt / np.linalg.norm(pab_pt) * SURF_R
        ax.scatter(*pab_pt_surf, s=250, c=group_colors[grp_id],
                   edgecolors='black', linewidth=1.5, marker='*', zorder=13)
        # Dotted line from glint star to normal vector tip
        n_tip = unique_normals[grp_id] * VEC_LENGTH
        ax.plot([pab_pt_surf[0], n_tip[0]], [pab_pt_surf[1], n_tip[1]],
                [pab_pt_surf[2], n_tip[2]],
                color=group_colors[grp_id], linewidth=0.8, linestyle=':',
                alpha=0.5, zorder=9)

    frac_label = f' (first {pab_fraction:.0%})' if pab_fraction is not None else ''
    format_sphere_axes(ax, f'Body Frame{frac_label}\n(normals fixed, PAB sweeps)')


def plot_inertial_frame(ax, quaternions, unique_normals, group_info,
                        sun_pos, obs_pos, sat_pos,
                        glint_epochs, glint_groups, glint_mags,
                        show_groups=None):
    """
    Inertial-frame sphere: normals trace curves, PAB is nearly fixed.

    Each body-fixed normal is rotated to J2000 at each epoch, producing
    a trajectory on the sphere. The PAB moves slowly (orbital timescale).

    show_groups: list of group IDs to plot, or None for all.
    """
    draw_wireframe_sphere(ax)

    n_obs = len(quaternions)
    n_groups = len(unique_normals)
    group_colors = [get_group_color(g['components']) for g in group_info]

    # Precompute R^T (body→inertial) for every epoch
    R_body_to_inertial = np.zeros((n_obs, 3, 3))
    for i in range(n_obs):
        q = quaternions[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        R_body_to_inertial[i] = R.T

    # ----- Normal family trajectories -----
    groups_to_show = show_groups if show_groups is not None else list(range(n_groups))
    for g in groups_to_show:
        n_body = unique_normals[g]
        color = group_colors[g]
        # n_J2000(t) = R^T(t) @ n_body
        trajectory = (R_body_to_inertial @ n_body)  # (n_obs, 3)

        lw = 1.8 if len(groups_to_show) <= 4 else 0.7
        alpha = 0.8 if len(groups_to_show) <= 4 else 0.45
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, linewidth=lw, alpha=alpha, zorder=3,
                label=f'G{g} {get_group_label(group_info[g]["components"], g)}')
        # Start / end
        ms = 60 if len(groups_to_show) <= 4 else 25
        ax.scatter(*trajectory[0], s=ms, c='lime', edgecolors=color,
                   linewidth=1.0, marker='o', zorder=5)
        ax.scatter(*trajectory[-1], s=ms, c='crimson', edgecolors=color,
                   linewidth=1.0, marker='s', zorder=5)

    # ----- PAB in inertial frame (slow) -----
    k1_J2000 = sun_pos - sat_pos
    k1_J2000 /= np.linalg.norm(k1_J2000, axis=1, keepdims=True)
    k2_J2000 = obs_pos - sat_pos
    k2_J2000 /= np.linalg.norm(k2_J2000, axis=1, keepdims=True)
    pab_unnorm = k1_J2000 + k2_J2000
    pab_J2000 = pab_unnorm / np.linalg.norm(pab_unnorm, axis=1, keepdims=True)

    ax.plot(pab_J2000[:, 0], pab_J2000[:, 1], pab_J2000[:, 2],
            color=PAB_COLOR, linewidth=2.5, alpha=0.9, zorder=4)

    # Start / end
    ax.scatter(*pab_J2000[0], s=100, c='lime', edgecolors='black',
               linewidth=1.2, marker='o', zorder=6)
    ax.scatter(*pab_J2000[-1], s=100, c='crimson', edgecolors='black',
               linewidth=1.2, marker='s', zorder=6)

    # ----- Glint markers on PAB (only for visible groups) -----
    for epoch_idx, grp_id, gmag in zip(glint_epochs, glint_groups, glint_mags):
        if show_groups is not None and grp_id not in show_groups:
            continue
        ax.scatter(*pab_J2000[epoch_idx], s=250, c=group_colors[grp_id],
                   edgecolors='black', linewidth=1.5, marker='*', zorder=7)

    ax.set_xlabel('X_J2000', fontsize=8, labelpad=1)
    ax.set_ylabel('Y_J2000', fontsize=8, labelpad=1)
    ax.set_zlabel('Z_J2000', fontsize=8, labelpad=1)
    lim = 1.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(labelsize=6)
    ax.set_title('Inertial Frame\n(normals sweep, PAB nearly fixed)', fontsize=11, pad=10)


def xyz_to_lonlat(xyz):
    """Convert unit vectors (N,3) or (3,) to (lon, lat) in radians.

    lon in [-pi, pi], lat in [-pi/2, pi/2].
    Convention: X=front, Y=left, Z=up  →  lon=atan2(y,x), lat=asin(z).
    """
    xyz = np.atleast_2d(xyz)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lat = np.arcsin(np.clip(xyz[:, 2], -1, 1))
    return lon, lat


def plot_mollweide(ax, unique_normals, group_info, pab_body,
                   glint_epochs, glint_groups, glint_mags,
                   show_groups=None, pab_fraction=None, pab_end_epoch=None,
                   title=None, show_axis_refs=True, label_normals=True):
    """
    Mollweide projection of the body-frame sphere.

    Normal families shown as markers, PAB trace as a curve on the map.
    No occlusion — every point on the sphere is visible.

    pab_end_epoch: if set, show PAB up to this epoch index (overrides pab_fraction).
    """
    group_colors = [get_group_color(g['components']) for g in group_info]
    max_area = max(g['total_area'] for g in group_info)
    groups_to_show = show_groups if show_groups is not None else list(range(len(unique_normals)))

    # ----- PAB trace -----
    n_obs = len(pab_body)
    if pab_end_epoch is not None:
        pab_end = pab_end_epoch + 1  # inclusive
    elif pab_fraction is not None:
        pab_end = int(n_obs * pab_fraction)
    else:
        pab_end = n_obs

    pab_end = min(pab_end, n_obs)

    if pab_end > 0:
        pab_slice = pab_body[:pab_end]
        pab_lon, pab_lat = xyz_to_lonlat(pab_slice)

        # Build per-epoch colour: yellow normally, component colour near glints
        GLINT_RADIUS = 3  # epochs either side of glint to recolour
        seg_color = np.full(pab_end, PAB_COLOR, dtype=object)
        for epoch_idx, grp_id, gmag in zip(glint_epochs, glint_groups, glint_mags):
            if epoch_idx >= pab_end:
                continue
            if show_groups is not None and grp_id not in show_groups:
                continue
            c = group_colors[grp_id]
            lo = max(0, epoch_idx - GLINT_RADIUS)
            hi = min(pab_end - 1, epoch_idx + GLINT_RADIUS)
            seg_color[lo:hi + 1] = c

        # Draw each segment (pair of consecutive points)
        dlon = np.abs(np.diff(pab_lon))
        for k in range(pab_end - 1):
            if dlon[k] > np.pi:  # skip lon wraps
                continue
            ax.plot(pab_lon[k:k + 2], pab_lat[k:k + 2],
                    color=seg_color[k], linewidth=1.5, alpha=0.8, zorder=3)

        # Start / end
        ax.scatter(pab_lon[0], pab_lat[0], s=20, c='lime',
                   edgecolors='black', linewidth=0.5, marker='o', zorder=6)
        ax.scatter(pab_lon[-1], pab_lat[-1], s=20, c='crimson',
                   edgecolors='black', linewidth=0.5, marker='s', zorder=6)

    # ----- Normal vectors as small markers -----
    for g in groups_to_show:
        n = unique_normals[g]
        color = group_colors[g]
        lon, lat = xyz_to_lonlat(n)
        ax.scatter(lon, lat, s=10, c=color, edgecolors='black',
                   linewidth=0.3, zorder=7, alpha=0.9)
        if label_normals:
            label = get_group_label(group_info[g]['components'], g)
            ax.annotate(f'G{g} {label}', (lon[0], lat[0]),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=5, fontweight='bold', alpha=0.7, zorder=8)

    # ----- Body-frame axis reference labels -----
    if show_axis_refs:
        axis_refs = [
            (0, 0, '+X'),
            (np.pi, 0, '-X'),
            (np.pi / 2, 0, '+Y'),
            (-np.pi / 2, 0, '-Y'),
            (0, np.pi / 2, '+Z'),
            (0, -np.pi / 2, '-Z'),
        ]
        for lon_r, lat_r, txt in axis_refs:
            ax.annotate(txt, (lon_r, lat_r), fontsize=7, alpha=0.3,
                        ha='center', va='center', fontweight='bold', zorder=1)

    # Formatting
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=9, pad=6)


def tangent_plane_project(vectors, center):
    """Project unit vectors onto a tangent plane centered on `center`.

    Returns (x, y) in degrees on the tangent plane.
    Uses orthographic projection (exact for small angles).

    Parameters
    ----------
    vectors : (N, 3) unit vectors
    center : (3,) unit vector defining the tangent point

    Returns
    -------
    x, y : (N,) arrays in degrees
    """
    c = center / np.linalg.norm(center)

    # Build orthonormal basis on the tangent plane
    # e1 ~ "east", e2 ~ "north"
    if abs(c[2]) < 0.9:
        up = np.array([0, 0, 1.0])
    else:
        up = np.array([1.0, 0, 0])
    e1 = np.cross(c, up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    e2 /= np.linalg.norm(e2)

    # Project: component perpendicular to c
    v = np.atleast_2d(vectors)
    x = np.degrees(v @ e1)
    y = np.degrees(v @ e2)
    return x, y


def plot_pab_zoom(ax, quaternions, unique_normals, group_info,
                  sun_pos, obs_pos, sat_pos,
                  glint_epochs, glint_groups, glint_mags,
                  observation_times, show_groups=None, pad_deg=2.0):
    """
    Inertial frame zoomed onto the PAB arc.

    Projects everything onto a tangent plane centered on the mean PAB.
    Normal family trajectories appear as curves sweeping through;
    crossings of the PAB arc are glints.
    """
    n_obs = len(quaternions)
    group_colors = [get_group_color(g['components']) for g in group_info]
    groups_to_show = show_groups if show_groups is not None else list(range(len(unique_normals)))

    # ----- PAB in J2000 -----
    k1_J2000 = sun_pos - sat_pos
    k1_J2000 /= np.linalg.norm(k1_J2000, axis=1, keepdims=True)
    k2_J2000 = obs_pos - sat_pos
    k2_J2000 /= np.linalg.norm(k2_J2000, axis=1, keepdims=True)
    pab_unnorm = k1_J2000 + k2_J2000
    pab_J2000 = pab_unnorm / np.linalg.norm(pab_unnorm, axis=1, keepdims=True)

    # Tangent-plane center = mean PAB direction
    pab_center = pab_J2000.mean(axis=0)
    pab_center /= np.linalg.norm(pab_center)

    # Project PAB arc
    pab_x, pab_y = tangent_plane_project(pab_J2000, pab_center)

    # ----- Precompute rotations -----
    R_body_to_inertial = np.zeros((n_obs, 3, 3))
    for i in range(n_obs):
        q = quaternions[i]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        R_body_to_inertial[i] = R.T

    # ----- Determine zoom extent from PAB arc -----
    pab_xrange = pab_x.max() - pab_x.min()
    pab_yrange = pab_y.max() - pab_y.min()
    half_size = max(pab_xrange, pab_yrange) / 2 + pad_deg
    cx, cy = (pab_x.max() + pab_x.min()) / 2, (pab_y.max() + pab_y.min()) / 2

    # ----- Plot PAB arc -----
    ax.plot(pab_x, pab_y, color=PAB_COLOR, linewidth=3.0, alpha=0.9,
            zorder=4, label='PAB arc')
    ax.scatter(pab_x[0], pab_y[0], s=50, c='lime', edgecolors='black',
               linewidth=0.8, marker='o', zorder=6)
    ax.scatter(pab_x[-1], pab_y[-1], s=50, c='crimson', edgecolors='black',
               linewidth=0.8, marker='s', zorder=6)

    # ----- Glint arcs: interpolated to resolve actual crossing duration -----
    from scipy.spatial.transform import Slerp as ScipySlerp

    GLINT_THRESHOLD_DEG = 3.0  # angular distance defining "in glint"
    cos_thresh = np.cos(np.radians(GLINT_THRESHOLD_DEG))
    N_INTERP = 200  # sub-samples across the ±2 epoch window

    plotted_labels = set()
    for epoch_idx, grp_id, gmag in zip(glint_epochs, glint_groups, glint_mags):
        if show_groups is not None and grp_id not in show_groups:
            continue
        color = group_colors[grp_id]
        n_body = unique_normals[grp_id]

        # Coarse window: ±2 epochs around glint
        lo = max(0, epoch_idx - 2)
        hi = min(n_obs - 1, epoch_idx + 2)
        coarse_times = observation_times[lo:hi + 1]
        coarse_quats_wxyz = quaternions[lo:hi + 1]

        # Convert to scipy Rotation for SLERP
        coarse_rots = Rotation.from_quat(
            coarse_quats_wxyz[:, [1, 2, 3, 0]])  # wxyz → xyzw
        slerp = ScipySlerp(coarse_times, coarse_rots)

        # Fine time grid
        fine_times = np.linspace(coarse_times[0], coarse_times[-1], N_INTERP)
        fine_rots = slerp(fine_times)
        # R = J2000→body, so R^T = body→J2000 = inv(R)
        fine_R_inv = np.array([r.as_matrix().T for r in fine_rots])

        # Fine normal trajectory in J2000
        fine_n_J2000 = (fine_R_inv @ n_body)  # (N_INTERP, 3)

        # Fine PAB (interpolate linearly — it barely moves)
        fine_pab = np.zeros((N_INTERP, 3))
        for dim in range(3):
            fine_pab[:, dim] = np.interp(fine_times,
                                         observation_times, pab_J2000[:, dim])
        fine_pab /= np.linalg.norm(fine_pab, axis=1, keepdims=True)

        # Angular distance to PAB
        cos_ang = np.sum(fine_n_J2000 * fine_pab, axis=1)
        in_glint = cos_ang > cos_thresh

        if not np.any(in_glint):
            # Widen threshold for this glint if it doesn't meet 3°
            local_thresh = np.cos(np.radians(
                np.degrees(np.arccos(np.clip(cos_ang.max(), -1, 1))) + 1.0))
            in_glint = cos_ang > local_thresh

        if not np.any(in_glint):
            continue

        # Find contiguous glint region
        changes = np.diff(in_glint.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if in_glint[0]:
            starts = np.concatenate([[0], starts])
        if in_glint[-1]:
            ends = np.concatenate([ends, [N_INTERP]])

        # Pick the segment containing the glint epoch
        glint_fine_idx = np.argmin(np.abs(fine_times - observation_times[epoch_idx]))
        best_seg = None
        for s, e in zip(starts, ends):
            if s <= glint_fine_idx < e:
                best_seg = (s, e)
                break
        if best_seg is None:
            best_seg = (starts[0], ends[0])

        s, e = best_seg
        tx, ty = tangent_plane_project(fine_n_J2000[s:e], pab_center)

        # Label only first occurrence of each group
        label_str = None
        if grp_id not in plotted_labels:
            label_str = f'G{grp_id} {get_group_label(group_info[grp_id]["components"], grp_id)}'
            plotted_labels.add(grp_id)

        # Arc line
        ax.plot(tx, ty, color=color, linewidth=1.8, alpha=0.85,
                zorder=5, label=label_str)

        # Start dot
        ax.scatter(tx[0], ty[0], s=20, c=color, edgecolors='black',
                   linewidth=0.4, marker='o', zorder=6)

        # End arrow
        if len(tx) >= 3:
            ax.annotate('', xy=(tx[-1], ty[-1]), xytext=(tx[-3], ty[-3]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.5, mutation_scale=10),
                        zorder=6)

    # Formatting
    ax.set_xlim(cx - half_size, cx + half_size)
    ax.set_ylim(cy - half_size, cy + half_size)
    ax.set_aspect('equal')
    ax.set_xlabel('Tangent-plane X (deg)', fontsize=9)
    ax.set_ylabel('Tangent-plane Y (deg)', fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(fontsize=7, loc='upper left', ncol=2, framealpha=0.8)


def build_legend(fig):
    """Create a shared legend at the bottom of the figure."""
    elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BUS_SP_COLOR,
               markersize=10, markeredgecolor='black', markeredgewidth=0.5,
               label='Bus + SP'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markersize=10, markeredgecolor='black', markeredgewidth=0.5,
               label='AD_East'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
               markersize=10, markeredgecolor='black', markeredgewidth=0.5,
               label='AD_West'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ALL_COMP_COLOR,
               markersize=10, markeredgecolor='black', markeredgewidth=0.5,
               label='All comps (±Z)'),
        Line2D([0], [0], color=PAB_COLOR, linewidth=2.5, label='PAB'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',
               markersize=14, markeredgecolor='black', markeredgewidth=0.5,
               label='Glint'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='t = 0'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='crimson',
               markersize=8, markeredgecolor='black', markeredgewidth=0.5,
               label='t = end'),
    ]
    fig.legend(handles=elements, loc='lower center', ncol=4, fontsize=9,
               frameon=True, fancybox=True, shadow=True)


# ===========================================================================
# Main
# ===========================================================================
print("=" * 70)
print(f"m044 -- Normal-family sphere (seed={SEED}, mode={MODE})")
print("=" * 70)

t0 = time.time()

# --- Lightweight setup (model + geometry only, NO lightcurve generation) ---
N_OBS = 500
print("Loading satellite model and SPICE geometry (no LC generation)...")

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
metakernel_path = config_manager.get_metakernel_path(config)
satellite = STLLoader.create_satellite_from_stl_config(
    config=config, config_manager=config_manager)

brdf_manager = BRDFManager(config)
brdf_calc = BRDFCalculator()
brdf_calc.update_satellite_brdf_with_manager(satellite, brdf_manager)

component_masses = {
    'Bus': 1532.0, 'SP_North': 170.0, 'SP_South': 170.0,
    'AD_East': 50.0, 'AD_West': 50.0,
}
inertia_result = compute_inertia_from_config(
    config=config, config_manager=config_manager,
    masses=component_masses,
    articulation_angles={'SP_North': 0.0, 'SP_South': 0.0})
inertia_tensor = inertia_result.inertia_tensor

spice_handler = SpiceHandler()
spice_handler.load_metakernel_programmatically(str(metakernel_path))

start_et = spice_handler.utc_to_et(config.simulation_defaults.start_time)
end_et = spice_handler.utc_to_et('2020-02-05T11:00:00')
epochs = np.linspace(start_et, end_et, N_OBS)
observation_times = epochs - epochs[0]
n_obs = N_OBS
dt_sampling = observation_times[-1] / (N_OBS - 1)

geometry_data = compute_observation_geometry(
    epochs=epochs,
    satellite_id=config.spice_config.satellite_id,
    observer_id=399999,
    spice_handler=spice_handler,
    config=config)
sun_pos = geometry_data['sun_positions']
obs_pos = geometry_data['obs_positions']
sat_pos = geometry_data['sat_positions']
obs_dist = geometry_data['observer_distances']

art_angles = {
    'SP_North': np.full(N_OBS, 0.0),
    'SP_South': np.full(N_OBS, 0.0),
    'AD_East': np.full(N_OBS, 15.0),
    'AD_West': np.full(N_OBS, 15.0),
}
art_matrices = compute_rotation_matrices_from_angles(art_angles, satellite)
print(f"  {n_obs} observations, dt = {dt_sampling:.2f}s")


# --- Extract unique normal families ---
print("Extracting facet normal families...")
facet_arrays = extract_facet_arrays(satellite)
art_normals, _ = apply_articulation_to_arrays(
    facet_arrays, art_matrices, 0, satellite
)
rounded_normals = np.round(art_normals, 4)
unique_normals, inverse_indices = np.unique(
    rounded_normals, axis=0, return_inverse=True
)
n_groups = len(unique_normals)
print(f"  {n_groups} unique normal groups, {facet_arrays.total_facets} total facets")

# Build group metadata
group_info = []
for g in range(n_groups):
    mask = (inverse_indices == g)
    facet_indices = np.where(mask)[0]
    components_in_group = set()
    for comp_name, comp_slice in facet_arrays.component_slices.items():
        comp_indices = np.arange(comp_slice.start, comp_slice.stop)
        if np.any(np.isin(facet_indices, comp_indices)):
            components_in_group.add(comp_name)
    group_info.append({
        'group_id': g,
        'normal': unique_normals[g].tolist(),
        'components': sorted(components_in_group),
        'total_area': float(facet_arrays.areas[mask].sum()),
    })


# --- Load glint identification from m041 saved results ---
print(f"\nLoading m041 results for seed={SEED}...")
json_path = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "unsorted" / "m041_glint_classification.json"
with open(json_path) as f:
    m041_data = json.load(f)

traj_data = None
for t in m041_data['trajectories']:
    if t['seed'] == SEED:
        traj_data = t
        break

if traj_data is None:
    raise ValueError(f"Seed {SEED} not found in m041 results")

glint_epochs = [p['epoch_idx'] for p in traj_data['peak_details']]
glint_groups = [p['oracle_group'] for p in traj_data['peak_details']]
glint_mags = [p['magnitude'] for p in traj_data['peak_details']]
print(f"  {len(glint_epochs)} bright peaks from hi-fi oracle labels")
for p in traj_data['peak_details']:
    grp = p['oracle_group']
    comps = p['oracle_components']
    print(f"    epoch {p['epoch_idx']:3d}: mag={p['magnitude']:.1f}  "
          f"G{grp}  n·PAB={p['oracle_n_dot_pab']:.4f}  comps={comps}")


# --- Generate trajectory ---
print(f"\nPropagating attitude (seed={SEED})...")
rng = np.random.RandomState(SEED)
q0_scipy = Rotation.random(random_state=rng)
q0_xyzw = q0_scipy.as_quat()
q0_wxyz = np.array([q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]])

omega_dir = rng.randn(3)
omega_dir /= np.linalg.norm(omega_dir)
omega_mag_dps = rng.uniform(*OMEGA_MAG_RANGE_DPS)
omega0_dps = omega_mag_dps * omega_dir
omega0_rad = np.deg2rad(omega0_dps)

quaternions, _ = propagate_attitude(
    q0=q0_wxyz, omega0=omega0_rad,
    times=observation_times,
    mode="tumbling", inertia_tensor=inertia_tensor,
)
print(f"  omega = [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, {omega0_dps[2]:.2f}] deg/s  "
      f"(|omega| = {omega_mag_dps:.2f} deg/s)")


# --- Compute body-frame sun/observer vectors ---
print("Computing body-frame vectors...")
k1_body = np.zeros((n_obs, 3))
k2_body = np.zeros((n_obs, 3))
for i in range(n_obs):
    q = quaternions[i]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    sun_vec = sun_pos[i] - sat_pos[i]
    k1_body[i] = R @ sun_vec / np.linalg.norm(sun_vec)
    obs_vec = obs_pos[i] - sat_pos[i]
    k2_body[i] = R @ obs_vec / np.linalg.norm(obs_vec)

# PAB in body frame
pab_unnorm = k1_body + k2_body
pab_body = pab_unnorm / np.linalg.norm(pab_unnorm, axis=1, keepdims=True)


# ===========================================================================
# Plot
# ===========================================================================
print(f"\nGenerating {MODE} plot...")

if MODE == 'both':
    fig = plt.figure(figsize=(20, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    plot_body_frame(ax1, unique_normals, group_info, pab_body,
                    glint_epochs, glint_groups, glint_mags,
                    show_groups=SHOW_GROUPS, pab_fraction=PAB_FRACTION)
    plot_inertial_frame(ax2, quaternions, unique_normals, group_info,
                        sun_pos, obs_pos, sat_pos,
                        glint_epochs, glint_groups, glint_mags,
                        show_groups=SHOW_GROUPS)
    # Match viewing angle
    ax1.view_init(elev=25, azim=-60)
    ax2.view_init(elev=25, azim=-60)

    build_legend(fig)
    fig.suptitle(
        f'Micro-44: Normal-Family Sphere — Seed {SEED}\n'
        f'$\\omega$ = [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, '
        f'{omega0_dps[2]:.2f}] deg/s  '
        f'(|$\\omega$| = {omega_mag_dps:.2f} deg/s)  •  '
        f'{len(glint_epochs)} glints',
        fontsize=13, fontweight='bold')
    plt.subplots_adjust(bottom=0.12, top=0.86, wspace=0.05)

elif MODE in ('body', 'inertial'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=-60)

    if MODE == 'body':
        plot_body_frame(ax, unique_normals, group_info, pab_body,
                        glint_epochs, glint_groups, glint_mags,
                        show_groups=SHOW_GROUPS, pab_fraction=PAB_FRACTION)
    else:
        plot_inertial_frame(ax, quaternions, unique_normals, group_info,
                            sun_pos, obs_pos, sat_pos,
                            glint_epochs, glint_groups, glint_mags,
                            show_groups=SHOW_GROUPS)

    build_legend(fig)
    fig.suptitle(
        f'Micro-44: Normal-Family Sphere ({MODE} frame) — Seed {SEED}\n'
        f'$\\omega$ = [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, '
        f'{omega0_dps[2]:.2f}] deg/s  '
        f'(|$\\omega$| = {omega_mag_dps:.2f} deg/s)  •  '
        f'{len(glint_epochs)} glints',
        fontsize=13, fontweight='bold')
    plt.subplots_adjust(bottom=0.10, top=0.88)

elif MODE == 'mollweide':
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='mollweide')

    plot_mollweide(ax, unique_normals, group_info, pab_body,
                   glint_epochs, glint_groups, glint_mags,
                   show_groups=SHOW_GROUPS, pab_fraction=PAB_FRACTION)

    build_legend(fig)
    fig.suptitle(
        f'Micro-44: Normal-Family Sphere — Seed {SEED}\n'
        f'$\\omega$ = [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, '
        f'{omega0_dps[2]:.2f}] deg/s  '
        f'(|$\\omega$| = {omega_mag_dps:.2f} deg/s)  •  '
        f'{len(glint_epochs)} glints',
        fontsize=13, fontweight='bold')
    plt.subplots_adjust(bottom=0.10, top=0.85)

elif MODE == 'correlation':
    # Hi-fi lightcurve from cache
    from src.computation.shadow_engine import compute_shadows
    from src.computation.lightcurve_generator import generate_lightcurves
    lc_cache_path = RESULTS_DIR / f"hifi_lc_seed{SEED:02d}.npz"
    if lc_cache_path.exists():
        print(f"  Loading cached hi-fi LC from {lc_cache_path.name}")
        mag_lc = np.load(lc_cache_path)['mag']
    else:
        print("  Computing hi-fi lightcurve (will cache)...")
        lit = compute_shadows(
            satellite=satellite, k1_vectors=k1_body,
            explicit_component_matrices=art_matrices, show_progress=False)
        mag_lc, _, _, _, _, _ = generate_lightcurves(
            facet_lit_status_dict=lit, k1_vectors_array=k1_body,
            k2_vectors_array=k2_body, observer_distances=obs_dist,
            satellite=satellite, epochs=epochs,
            pre_computed_matrices=art_matrices,
            generate_no_shadow=False, animate=False, show_progress=False)
        np.savez_compressed(str(lc_cache_path), mag=mag_lc)

    # PAB in J2000 and R^T
    k1_J2000 = sun_pos - sat_pos
    k1_J2000 /= np.linalg.norm(k1_J2000, axis=1, keepdims=True)
    k2_J2000 = obs_pos - sat_pos
    k2_J2000 /= np.linalg.norm(k2_J2000, axis=1, keepdims=True)
    pab_unnorm_J = k1_J2000 + k2_J2000
    pab_J2000 = pab_unnorm_J / np.linalg.norm(pab_unnorm_J, axis=1, keepdims=True)

    R_body_to_inertial = np.zeros((n_obs, 3, 3))
    for i in range(n_obs):
        q = quaternions[i]
        R_body_to_inertial[i] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T

    groups_to_plot = SHOW_GROUPS if SHOW_GROUPS is not None else list(range(len(unique_normals)))
    n_panels = len(groups_to_plot)
    group_colors = [get_group_color(g['components']) for g in group_info]
    time_minutes = observation_times / 60.0

    # Convert magnitude to flux-like quantity (brighter = higher)
    # Use -mag so that brightness peaks are positive peaks
    brightness = -mag_lc

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.0 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    for panel_idx, grp_id in enumerate(groups_to_plot):
        ax = axes[panel_idx]
        color = group_colors[grp_id]
        label = get_group_label(group_info[grp_id]['components'], grp_id)

        # Angular distance
        traj_J2000 = R_body_to_inertial @ unique_normals[grp_id]
        ndot = np.sum(traj_J2000 * pab_J2000, axis=1)
        ang_dist = np.degrees(np.arccos(np.clip(ndot, -1, 1)))

        # Alignment signal: use -ang_dist so that close alignment = high value
        # (matches brightness convention: high = interesting)
        alignment = -ang_dist

        # Sliding-window Pearson correlation
        from numpy.lib.stride_tricks import sliding_window_view
        WINDOW = 21  # epochs (~150s)
        half_w = WINDOW // 2

        if len(brightness) >= WINDOW:
            b_win = sliding_window_view(brightness, WINDOW)
            a_win = sliding_window_view(alignment, WINDOW)

            # Pearson r for each window
            b_mean = b_win.mean(axis=1)
            a_mean = a_win.mean(axis=1)
            b_std = b_win.std(axis=1)
            a_std = a_win.std(axis=1)

            # Avoid division by zero
            safe_mask = (b_std > 1e-10) & (a_std > 1e-10)
            corr = np.full(len(b_mean), np.nan)
            corr[safe_mask] = np.sum(
                (b_win[safe_mask] - b_mean[safe_mask, None]) *
                (a_win[safe_mask] - a_mean[safe_mask, None]),
                axis=1) / (WINDOW * b_std[safe_mask] * a_std[safe_mask])

            # Pad to full length
            t_corr = time_minutes[half_w: half_w + len(corr)]

            ax.plot(t_corr, corr, color=color, linewidth=0.8, alpha=0.9)
            ax.axhline(0, color='grey', linewidth=0.4, linestyle='-', alpha=0.3)
            ax.fill_between(t_corr, corr, 0, where=corr > 0,
                            color=color, alpha=0.15)
            ax.fill_between(t_corr, corr, 0, where=corr < 0,
                            color='grey', alpha=0.08)

        ax.set_ylabel('r', fontsize=8, rotation=0, labelpad=10)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f'{label}', fontsize=10, loc='left',
                     color=color, fontweight='bold')
        ax.grid(True, alpha=0.15, linewidth=0.3)

    axes[-1].set_xlabel('Time (minutes)', fontsize=10)
    fig.suptitle(
        f'Micro-44: Sliding correlation (alignment vs brightness) — Seed {SEED}\n'
        f'Window = {WINDOW} epochs ({WINDOW * dt_sampling:.0f}s)  •  '
        f'$\\omega$ = [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, '
        f'{omega0_dps[2]:.2f}] deg/s',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

elif MODE == 'ndot_pab':
    # Compute PAB in J2000
    k1_J2000 = sun_pos - sat_pos
    k1_J2000 /= np.linalg.norm(k1_J2000, axis=1, keepdims=True)
    k2_J2000 = obs_pos - sat_pos
    k2_J2000 /= np.linalg.norm(k2_J2000, axis=1, keepdims=True)
    pab_unnorm_J = k1_J2000 + k2_J2000
    pab_J2000 = pab_unnorm_J / np.linalg.norm(pab_unnorm_J, axis=1, keepdims=True)

    # R^T for all epochs
    R_body_to_inertial = np.zeros((n_obs, 3, 3))
    for i in range(n_obs):
        q = quaternions[i]
        R_body_to_inertial[i] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T

    # Hi-fi lightcurve: load from cache or compute and save
    from src.computation.shadow_engine import compute_shadows
    from src.computation.lightcurve_generator import generate_lightcurves
    lc_cache_path = RESULTS_DIR / f"hifi_lc_seed{SEED:02d}.npz"
    if lc_cache_path.exists():
        print(f"  Loading cached hi-fi LC from {lc_cache_path.name}")
        mag_lc = np.load(lc_cache_path)['mag']
    else:
        print("  Computing hi-fi lightcurve (will cache for reuse)...")
        lit = compute_shadows(
            satellite=satellite, k1_vectors=k1_body,
            explicit_component_matrices=art_matrices, show_progress=False)
        mag_lc, _, _, _, _, _ = generate_lightcurves(
            facet_lit_status_dict=lit, k1_vectors_array=k1_body,
            k2_vectors_array=k2_body, observer_distances=obs_dist,
            satellite=satellite, epochs=epochs,
            pre_computed_matrices=art_matrices,
            generate_no_shadow=False, animate=False, show_progress=False)
        np.savez_compressed(str(lc_cache_path), mag=mag_lc)
        print(f"  Cached to {lc_cache_path.name}")

    # Detect ALL local brightness peaks (magnitude minima = brightness maxima)
    # Use find_peaks on -mag to find minima, with prominence info
    from scipy.signal import find_peaks
    peak_indices_raw, peak_props = find_peaks(-mag_lc, distance=3)
    # Compute prominence on the negated signal
    from scipy.signal import peak_prominences
    prominences, _, _ = peak_prominences(-mag_lc, peak_indices_raw)
    PROM_THRESHOLD = 0.3  # magnitudes — below this = low confidence
    all_peak_indices = peak_indices_raw
    peak_prom = prominences
    n_high = np.sum(peak_prom >= PROM_THRESHOLD)
    n_low = np.sum(peak_prom < PROM_THRESHOLD)
    print(f"  {len(all_peak_indices)} local minima detected "
          f"({n_high} prominent, {n_low} low-prominence)")

    # Compute angular distance for all groups at all epochs
    groups_to_plot = SHOW_GROUPS if SHOW_GROUPS is not None else list(range(len(unique_normals)))
    n_panels = len(groups_to_plot)
    group_colors = [get_group_color(g['components']) for g in group_info]
    time_minutes = observation_times / 60.0

    ang_dist_all = {}  # grp_id -> (n_obs,) angular distance array
    for grp_id in groups_to_plot:
        traj_J2000 = R_body_to_inertial @ unique_normals[grp_id]
        ndot = np.sum(traj_J2000 * pab_J2000, axis=1)
        ang_dist_all[grp_id] = np.degrees(np.arccos(np.clip(ndot, -1, 1)))

    # For each LC peak, assign to the group with smallest angular distance
    peak_groups = []
    for pidx in all_peak_indices:
        best_grp = min(groups_to_plot, key=lambda g: ang_dist_all[g][pidx])
        peak_groups.append(best_grp)
    peak_groups = np.array(peak_groups)

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.0 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    for panel_idx, grp_id in enumerate(groups_to_plot):
        ax = axes[panel_idx]
        color = group_colors[grp_id]
        label = get_group_label(group_info[grp_id]['components'], grp_id)
        ang_dist = ang_dist_all[grp_id]

        # Alignment curve
        ax.plot(time_minutes, ang_dist, color=color, linewidth=0.8, alpha=0.9)

        # Minima of alignment curve (angular distance dips)
        align_minima, _ = find_peaks(-ang_dist, distance=3)
        ax.scatter(time_minutes[align_minima], ang_dist[align_minima],
                   s=12, c='black', zorder=4, alpha=0.7)

        # Red vertical lines at ALL LC peaks, markers by prominence
        for i_pk, (pidx, pgrp) in enumerate(zip(all_peak_indices, peak_groups)):
            prom = peak_prom[i_pk]
            is_prominent = prom >= PROM_THRESHOLD
            ax.axvline(time_minutes[pidx], color='red',
                       linewidth=0.5 if is_prominent else 0.3,
                       alpha=0.3 if is_prominent else 0.12, zorder=2)
            # Mark on this group's curve if it's the closest group
            if pgrp == grp_id:
                if is_prominent:
                    ax.scatter(time_minutes[pidx], ang_dist[pidx],
                               s=30, c=color, edgecolors='black', linewidth=0.5,
                               marker='v', zorder=5)
                else:
                    ax.scatter(time_minutes[pidx], ang_dist[pidx],
                               s=20, facecolors='none', edgecolors=color,
                               linewidth=0.6, marker='v', zorder=5)

        ax.set_ylabel('°', fontsize=8, rotation=0, labelpad=10)
        ax.set_ylim(0, 180)
        ax.set_yscale('symlog', linthresh=5)
        ax.set_yticks([0, 1, 5, 30, 90, 180])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
        ax.set_title(f'{label}', fontsize=10, loc='left',
                     color=color, fontweight='bold')
        ax.grid(True, alpha=0.15, linewidth=0.3)

        # Lightcurve overlay on right y-axis
        ax2 = ax.twinx()
        ax2.plot(time_minutes, mag_lc, color='black', linewidth=0.4, alpha=0.3)
        ax2.set_ylabel('mag', fontsize=7, alpha=0.4)
        ax2.invert_yaxis()
        ax2.tick_params(labelsize=6, colors='grey')

        # Sliding correlation overlay on a third axis
        from numpy.lib.stride_tricks import sliding_window_view
        CORR_WINDOW = 21
        half_w = CORR_WINDOW // 2
        brightness = -mag_lc
        align_signal = -ang_dist
        if len(brightness) >= CORR_WINDOW:
            b_win = sliding_window_view(brightness, CORR_WINDOW)
            a_win = sliding_window_view(align_signal, CORR_WINDOW)
            b_mean = b_win.mean(axis=1)
            a_mean = a_win.mean(axis=1)
            b_std = b_win.std(axis=1)
            a_std = a_win.std(axis=1)
            safe = (b_std > 1e-10) & (a_std > 1e-10)
            corr = np.full(len(b_mean), 0.0)
            corr[safe] = np.sum(
                (b_win[safe] - b_mean[safe, None]) *
                (a_win[safe] - a_mean[safe, None]),
                axis=1) / (CORR_WINDOW * b_std[safe] * a_std[safe])
            t_corr = time_minutes[half_w: half_w + len(corr)]

            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('axes', 1.08))
            ax3.plot(t_corr, corr, color=color, linewidth=0.6, alpha=0.4,
                     linestyle='-')
            ax3.fill_between(t_corr, corr, 0, where=corr > 0,
                             color=color, alpha=0.08)
            ax3.fill_between(t_corr, corr, 0, where=corr < 0,
                             color='grey', alpha=0.05)
            ax3.set_ylim(-1.05, 1.05)
            ax3.set_ylabel('r', fontsize=7, alpha=0.4)
            ax3.tick_params(labelsize=5, colors='grey')
            ax3.axhline(0, color='grey', linewidth=0.3, alpha=0.3)

    axes[-1].set_xlabel('Time (minutes)', fontsize=10)
    fig.suptitle(
        f'Micro-44: n · PAB vs time — Seed {SEED}\n'
        f'$\\omega$ = [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, '
        f'{omega0_dps[2]:.2f}] deg/s  '
        f'(|$\\omega$| = {omega_mag_dps:.2f} deg/s)',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

elif MODE == 'ndot_compare':
    # Helper: compute PAB J2000 and R_body_to_inertial for a given seed
    def compute_trajectory_alignment(seed, unique_normals, observation_times,
                                     inertia_tensor, sun_pos, obs_pos, sat_pos):
        """Return ang_dist (n_groups, n_obs) and glint data for a seed."""
        rng_loc = np.random.RandomState(seed)
        q0_sc = Rotation.random(random_state=rng_loc)
        q0_xyzw_loc = q0_sc.as_quat()
        q0_wxyz_loc = np.array([q0_xyzw_loc[3], q0_xyzw_loc[0],
                                q0_xyzw_loc[1], q0_xyzw_loc[2]])
        od = rng_loc.randn(3); od /= np.linalg.norm(od)
        om = rng_loc.uniform(0.5, 5.0)
        omega0_rad_loc = np.deg2rad(om * od)

        quats, _ = propagate_attitude(
            q0=q0_wxyz_loc, omega0=omega0_rad_loc,
            times=observation_times, mode="tumbling",
            inertia_tensor=inertia_tensor)

        n_obs_loc = len(observation_times)
        R_inv = np.zeros((n_obs_loc, 3, 3))
        for i in range(n_obs_loc):
            q = quats[i]
            R_inv[i] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T

        # PAB in J2000
        k1 = sun_pos - sat_pos
        k1 /= np.linalg.norm(k1, axis=1, keepdims=True)
        k2 = obs_pos - sat_pos
        k2 /= np.linalg.norm(k2, axis=1, keepdims=True)
        pab_u = k1 + k2
        pab_loc = pab_u / np.linalg.norm(pab_u, axis=1, keepdims=True)

        # Angular distance for all groups
        n_groups_loc = len(unique_normals)
        ang_dist_all = np.zeros((n_groups_loc, n_obs_loc))
        for g in range(n_groups_loc):
            traj = R_inv @ unique_normals[g]
            ndot = np.sum(traj * pab_loc, axis=1)
            ang_dist_all[g] = np.degrees(np.arccos(np.clip(ndot, -1, 1)))

        return ang_dist_all, om

    # Compute for both seeds
    ang_A, omega_A = compute_trajectory_alignment(
        SEED, unique_normals, observation_times, inertia_tensor,
        sun_pos, obs_pos, sat_pos)
    ang_B, omega_B = compute_trajectory_alignment(
        SEED_B, unique_normals, observation_times, inertia_tensor,
        sun_pos, obs_pos, sat_pos)

    # Load glint data for both seeds
    glint_data = {}
    for s in [SEED, SEED_B]:
        for t in m041_data['trajectories']:
            if t['seed'] == s:
                glint_data[s] = t['peak_details']
                break

    groups_to_plot = SHOW_GROUPS if SHOW_GROUPS is not None else list(range(len(unique_normals)))
    n_panels = len(groups_to_plot)
    group_colors_list = [get_group_color(g['components']) for g in group_info]
    time_minutes = observation_times / 60.0

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.0 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    for panel_idx, grp_id in enumerate(groups_to_plot):
        ax = axes[panel_idx]
        color = group_colors_list[grp_id]
        label = get_group_label(group_info[grp_id]['components'], grp_id)

        # Red vertical lines for ALL glints from both trajectories
        for s, marker_style, mk_color, mk_label in [
            (SEED, 'v', color, f'Seed {SEED}'),
            (SEED_B, '^', 'grey', f'Seed {SEED_B}'),
        ]:
            ang = ang_A if s == SEED else ang_B
            for p in glint_data.get(s, []):
                ep = p['epoch_idx']
                gid = p['oracle_group']
                gmag = p['magnitude']
                # Thin vertical line
                ax.axvline(time_minutes[ep], color='red' if s == SEED else 'blue',
                           linewidth=0.5, alpha=0.25, zorder=1)
                # Marker on this group's curve
                if gid == grp_id:
                    ax.scatter(time_minutes[ep], ang[grp_id, ep],
                               s=45, c=mk_color, edgecolors='black',
                               linewidth=0.6, marker=marker_style, zorder=5)
                    ax.annotate(f'{gmag:.1f}',
                                (time_minutes[ep], ang[grp_id, ep]),
                                textcoords='offset points', xytext=(4, 6),
                                fontsize=5, alpha=0.7)

        ax.set_ylabel('°', fontsize=8, rotation=0, labelpad=10)
        ax.set_ylim(0, 180)
        ax.set_yscale('symlog', linthresh=5)
        ax.set_yticks([0, 1, 5, 30, 90, 180])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
        ax.set_title(f'{label}', fontsize=10, loc='left',
                     color=color, fontweight='bold')
        ax.grid(True, alpha=0.15, linewidth=0.3)

    # Legend for the two seeds
    from matplotlib.lines import Line2D as L2D
    legend_els = [
        L2D([0], [0], marker='v', color='w', markerfacecolor='grey',
            markersize=8, markeredgecolor='black', markeredgewidth=0.5,
            label=f'Seed {SEED} (|ω|={omega_A:.1f}°/s)'),
        L2D([0], [0], marker='^', color='w', markerfacecolor='grey',
            markersize=8, markeredgecolor='black', markeredgewidth=0.5,
            label=f'Seed {SEED_B} (|ω|={omega_B:.1f}°/s)'),
        L2D([0], [0], color='red', linewidth=1, alpha=0.5,
            label=f'Seed {SEED} glint time'),
        L2D([0], [0], color='blue', linewidth=1, alpha=0.5,
            label=f'Seed {SEED_B} glint time'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=4, fontsize=8,
               frameon=True, fancybox=True)

    axes[-1].set_xlabel('Time (minutes)', fontsize=10)
    fig.suptitle(
        f'Micro-44: Glint alignment comparison — Seeds {SEED} & {SEED_B}\n'
        f'Angular distance to PAB at glint epochs',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

elif MODE == 'pab_zoom':
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    plot_pab_zoom(ax, quaternions, unique_normals, group_info,
                  sun_pos, obs_pos, sat_pos,
                  glint_epochs, glint_groups, glint_mags,
                  observation_times, show_groups=SHOW_GROUPS)

    fig.suptitle(
        f'Micro-44: Inertial PAB Zoom — Seed {SEED}\n'
        f'$\\omega$ = [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, '
        f'{omega0_dps[2]:.2f}] deg/s  '
        f'(|$\\omega$| = {omega_mag_dps:.2f} deg/s)  •  '
        f'{len(glint_epochs)} glints',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])

elif MODE == 'glint_panels':
    N_PANELS = 5
    n_glints = min(N_PANELS, len(glint_epochs))
    fig, axes = plt.subplots(1, n_glints, figsize=(4.5 * n_glints, 4),
                             subplot_kw={'projection': 'mollweide'})
    if n_glints == 1:
        axes = [axes]

    group_colors = [get_group_color(g['components']) for g in group_info]

    for i in range(n_glints):
        ax = axes[i]
        ep = glint_epochs[i]
        grp = glint_groups[i]
        mag_val = glint_mags[i]
        comps = group_info[grp]['components']
        t_sec = ep * dt_sampling

        # Show PAB history up to and including this glint
        # Only show glints up to and including this one
        plot_mollweide(
            ax, unique_normals, group_info, pab_body,
            glint_epochs[:i + 1], glint_groups[:i + 1], glint_mags[:i + 1],
            show_groups=SHOW_GROUPS, pab_end_epoch=ep,
            label_normals=False, show_axis_refs=True,
            title=f'Glint {i+1}: ep {ep}  mag {mag_val:.1f}\n'
                  f'{get_group_label(comps, grp)}  t={t_sec:.0f}s',
        )

    build_legend(fig)
    fig.suptitle(
        f'Micro-44: PAB approach to glints — Seed {SEED}\n'
        f'$\\omega$ = [{omega0_dps[0]:.2f}, {omega0_dps[1]:.2f}, '
        f'{omega0_dps[2]:.2f}] deg/s  '
        f'(|$\\omega$| = {omega_mag_dps:.2f} deg/s)',
        fontsize=13, fontweight='bold')
    plt.subplots_adjust(bottom=0.12, top=0.78, wspace=0.08)

else:
    raise ValueError(f"Unknown MODE: {MODE}")

plot_path = RESULTS_DIR / f"seed{SEED:02d}_{MODE}_omega{omega_mag_dps:.1f}dps.png"
fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved: {plot_path}")

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
