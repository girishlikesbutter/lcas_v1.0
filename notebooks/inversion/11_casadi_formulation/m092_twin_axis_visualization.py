#!/usr/bin/env python3
"""
m092 — Visualise true vs estimated attitude and omega per seed.

One PNG per seed (6, 24, 36). Each PNG is a 2x2 grid:
  Top-left:     True attitude — satellite mesh + body axes rotated by true q0
  Top-right:    Est attitude  — satellite mesh + body axes rotated by est q0
  Bottom-left:  True omega    — satellite mesh in body frame + true omega arrow
  Bottom-right: Est omega     — satellite mesh in body frame + est omega arrow
"""

import sys
import json
import numpy as np
import quaternion as quat_mod
from pathlib import Path
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.utils.geometry_utils import build_rotation_matrix

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
SEEDS = [6, 24, 36]

COMP_COLORS = {
    'Bus': '#aaaaaa',
    'SP_North': '#224488',
    'SP_South': '#224488',
    'AD_East': '#cc8844',
    'AD_West': '#cc8844',
}


def load_satellite():
    cm = RSO_ConfigManager(PROJECT_ROOT)
    cfg = cm.load_config('intelsat_901/intelsat_901_config.yaml')
    return STLLoader.create_satellite_from_stl_config(config=cfg, config_manager=cm)


def get_body_frame_faces(satellite):
    """Extract all facet triangles in body frame, with articulation applied.

    Dishes (AD_East, AD_West) are articulated at 15° to match the pipeline.
    Solar panels at 0°. This matches setup_experiment() in experiment_setup.py.
    """
    art_angles = {'AD_East': 15.0, 'AD_West': 15.0}

    components = []
    for comp in satellite.components:
        R_body = quat_mod.as_rotation_matrix(comp.relative_orientation)
        pos = np.array(comp.relative_position)

        # Apply articulation if this component has a non-zero angle
        angle = art_angles.get(comp.name, 0.0)
        if angle != 0.0 and comp.articulation_parameters is not None:
            rot_axis = np.array(comp.articulation_parameters.rotation_axis)
            R_art = build_rotation_matrix(angle, rot_axis)[:3, :3]
        else:
            R_art = np.eye(3)

        R_combined = R_body @ R_art
        triangles = []
        for facet in comp.facets:
            verts = np.array(facet.vertices)  # (3, 3)
            transformed = (R_combined @ verts.T).T + pos
            triangles.append(transformed)
        components.append({
            'name': comp.name,
            'faces': np.array(triangles),  # (N, 3, 3)
        })
    return components


def render_satellite(ax, comp_data, R=None):
    """Render satellite mesh. If R given, rotate all vertices by R."""
    for comp in comp_data:
        faces = comp['faces'].copy()
        if R is not None:
            # R @ each vertex: faces is (N, 3, 3) where last dim is xyz
            faces = np.einsum('ij,nkj->nki', R, faces)
        poly = Poly3DCollection(faces, alpha=0.7, linewidth=0.15,
                                edgecolor='#333333')
        poly.set_facecolor(COMP_COLORS.get(comp['name'], '#999999'))
        ax.add_collection3d(poly)


def draw_body_axes(ax, R, length=8.0):
    """Draw body +X, +Y, +Z axes rotated by R."""
    colors = {'X': '#d62728', 'Y': '#2ca02c', 'Z': '#1f77b4'}
    for i, name in enumerate(['X', 'Y', 'Z']):
        direction = np.zeros(3)
        direction[i] = 1.0
        rotated = R @ direction * length
        ax.quiver(0, 0, 0, *rotated, color=colors[name],
                  arrow_length_ratio=0.06, linewidth=2.5)
        ax.text(rotated[0]*1.15, rotated[1]*1.15, rotated[2]*1.15,
                f'+{name}', fontsize=9, color=colors[name],
                fontweight='bold', ha='center', va='center')


def draw_observer_vector(ax, obs_dir_j2000, length=8.0):
    """Draw observer direction vector in J2000 frame."""
    vec = obs_dir_j2000 * length
    ax.quiver(0, 0, 0, *vec, color='#00bfbf',
              arrow_length_ratio=0.06, linewidth=2.5)
    ax.text(vec[0]*1.15, vec[1]*1.15, vec[2]*1.15,
            'obs', fontsize=9, color='#00bfbf',
            fontweight='bold', ha='center', va='center')


def draw_omega_arrow(ax, omega_dir, label, length=12.0):
    """Draw omega direction arrow in body frame."""
    vec = omega_dir * length
    ax.quiver(0, 0, 0, *vec, color='#9467bd',
              arrow_length_ratio=0.05, linewidth=3.0)
    ax.text(vec[0]*1.15, vec[1]*1.15, vec[2]*1.15,
            label, fontsize=9, color='#9467bd',
            fontweight='bold', ha='center', va='center')


def setup_ax(ax, title, lim=12.0):
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.set_title(title, fontsize=10, pad=10)
    ax.tick_params(labelsize=6)


def quat_wxyz_to_rotmat(q_wxyz):
    r = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return r.as_matrix()


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def error_axis_angle(q_est, q_true):
    q_err = quat_multiply(q_est, quat_conj(q_true))
    if q_err[0] < 0:
        q_err = -q_err
    angle = 2 * np.arccos(np.clip(q_err[0], -1, 1))
    sin_half = np.sin(angle / 2)
    if sin_half < 1e-10:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = q_err[1:] / sin_half
        axis /= np.linalg.norm(axis)
    return axis, np.rad2deg(angle)


def main():
    print("Loading satellite model...", flush=True)
    satellite = load_satellite()
    comp_data = get_body_frame_faces(satellite)
    print(f"  {sum(len(c['faces']) for c in comp_data)} total faces")

    master = np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)

    results = {}
    for seed in SEEDS:
        with open(RESULTS_DIR / f"m090_seed{seed:03d}" / "result.json") as f:
            results[seed] = json.load(f)

    view_elev, view_azim = 25, -60

    for seed in SEEDS:
        true_q0 = master['q0s'][seed]
        true_w0 = master['omega0s'][seed]  # rad/s
        est_q0 = np.array(results[seed]['winner']['q0_wxyz'])
        est_w0 = np.deg2rad(np.array(results[seed]['winner']['w0_dps']))

        R_true = quat_wxyz_to_rotmat(true_q0)
        R_est = quat_wxyz_to_rotmat(est_q0)

        # Observer direction in J2000 at epoch 0
        # k2_body was computed with true attitude: k2_body = R_true^T @ k2_j2000
        k2_body_ep0 = master['k2_body'][seed, 0, :]
        obs_dir_j2000 = R_true @ k2_body_ep0
        obs_dir_j2000 /= np.linalg.norm(obs_dir_j2000)

        true_w_dir = true_w0 / np.linalg.norm(true_w0)
        est_w_dir = est_w0 / np.linalg.norm(est_w0)
        true_w_mag = np.rad2deg(np.linalg.norm(true_w0))
        est_w_mag = np.rad2deg(np.linalg.norm(est_w0))

        err_axis, err_angle = error_axis_angle(est_q0, true_q0)
        dot_x = abs(np.dot(err_axis, [1, 0, 0]))
        q0_err = results[seed]['winner']['q0_err']
        w_dir_err = results[seed]['winner']['w0_err']
        w_mag_err = results[seed]['winner']['w_mag_err_pct']

        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(
            f'Seed {seed} — q0 err: {q0_err:.1f}°, '
            f'w_dir err: {w_dir_err:.1f}°, w_mag err: {w_mag_err:+.1f}%\n'
            f'Error rotation axis: [{err_axis[0]:+.2f}, {err_axis[1]:+.2f}, '
            f'{err_axis[2]:+.2f}], dot(axis, +X) = {dot_x:.3f}',
            fontsize=12, y=0.98)

        # ── Top-left: True attitude ────────────────────────────────────
        ax1 = fig.add_subplot(221, projection='3d')
        render_satellite(ax1, comp_data, R=R_true)
        draw_body_axes(ax1, R_true)
        draw_observer_vector(ax1, obs_dir_j2000)
        setup_ax(ax1, f'True attitude (q0)')
        ax1.view_init(elev=view_elev, azim=view_azim)

        # ── Top-right: Estimated attitude ──────────────────────────────
        ax2 = fig.add_subplot(222, projection='3d')
        render_satellite(ax2, comp_data, R=R_est)
        draw_body_axes(ax2, R_est)
        draw_observer_vector(ax2, obs_dir_j2000)
        setup_ax(ax2, f'Estimated attitude (q0 err = {q0_err:.1f}°)')
        ax2.view_init(elev=view_elev, azim=view_azim)

        # Observer direction in body frame at epoch 0 (for omega plots)
        obs_dir_body = k2_body_ep0 / np.linalg.norm(k2_body_ep0)

        # ── Bottom-left: True omega (body frame) ──────────────────────
        ax3 = fig.add_subplot(223, projection='3d')
        render_satellite(ax3, comp_data, R=None)
        draw_omega_arrow(ax3, true_w_dir,
                         f'true w\n({true_w_mag:.2f} dps)')
        draw_observer_vector(ax3, obs_dir_body)
        setup_ax(ax3, f'True omega (body frame)')
        ax3.view_init(elev=view_elev, azim=view_azim)

        # ── Bottom-right: Estimated omega (body frame) ────────────────
        ax4 = fig.add_subplot(224, projection='3d')
        render_satellite(ax4, comp_data, R=None)
        draw_omega_arrow(ax4, est_w_dir,
                         f'est w\n({est_w_mag:.2f} dps)')
        draw_observer_vector(ax4, obs_dir_body)
        setup_ax(ax4, f'Estimated omega (w_dir err = {w_dir_err:.1f}°)')
        ax4.view_init(elev=view_elev, azim=view_azim)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = RESULTS_DIR / f"m092_seed{seed:03d}_attitude_viz.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()


if __name__ == '__main__':
    main()
