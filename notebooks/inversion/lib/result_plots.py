"""
Result visualisation for inversion pipeline.

Produces two 3D plots:
  1. Attitude (q0): body frame axes in inertial space — true vs estimates
  2. Omega direction: unit vectors in body frame — true vs estimates

Usage:
    from lib.result_plots import plot_result_comparison
    plot_result_comparison(
        true_q0, true_omega0,
        estimates=[
            {'label': 'Old beta', 'q0': q0_old, 'w0': w0_old, 'color': 'orange'},
            {'label': 'm083', 'q0': q0_new, 'w0': w0_new, 'color': 'blue'},
        ],
        title='Seed 35',
        save_path='output.png',
    )
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


def attitude_error_deg(q1, q2):
    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    R2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    return float(np.rad2deg((R1.inv() * R2).magnitude()))


def omega_dir_err(w1, w2):
    d1 = w1 / np.linalg.norm(w1)
    d2 = w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def _draw_arc(ax, v1, v2, color, n=50):
    """Draw great circle arc between two unit vectors."""
    angles = np.linspace(0, 1, n)
    points = np.array([(1 - t) * v1 + t * v2 for t in angles])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points /= norms
    ax.plot(points[:, 0], points[:, 1], points[:, 2],
            color=color, linewidth=1.5, linestyle='--', alpha=0.6)


def _q_to_R(q_wxyz):
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def plot_result_comparison(true_q0, true_omega0, estimates, title='', save_path=None):
    """
    Plot attitude and omega direction comparison.

    Parameters
    ----------
    true_q0 : array (4,) wxyz quaternion
    true_omega0 : array (3,) rad/s
    estimates : list of dicts, each with:
        'label': str
        'q0': array (4,) wxyz
        'w0': array (3,) rad/s
        'color': matplotlib color
    title : str, prefix for plot titles
    save_path : str or None, if provided saves to this path
    """
    true_omega_mag = np.rad2deg(np.linalg.norm(true_omega0))
    body_axes = np.eye(3)
    axis_labels = ['+X', '+Y', '+Z']
    axis_colors = ['firebrick', 'forestgreen', 'royalblue']

    fig = plt.figure(figsize=(16, 7))

    # ── Left: Attitude (q0) ──
    ax1 = fig.add_subplot(121, projection='3d')
    R_true = _q_to_R(true_q0)

    # True body axes — solid, thick
    for i in range(3):
        v = R_true @ body_axes[i]
        ax1.quiver(0, 0, 0, *v, color=axis_colors[i], linewidth=2.5,
                   arrow_length_ratio=0.1,
                   label=f'True {axis_labels[i]}' if i == 0 else f'      {axis_labels[i]}')

    # Estimated body axes — same colour, dashed, thinner
    subtitle_parts = []
    for est in estimates:
        R_est = _q_to_R(est['q0'])
        q_err = attitude_error_deg(est['q0'], true_q0)
        subtitle_parts.append(f"{est['label']}: {q_err:.1f}°")
        for i in range(3):
            v = R_est @ body_axes[i]
            ax1.quiver(0, 0, 0, *v, color=axis_colors[i], linewidth=1.2,
                       arrow_length_ratio=0.08, linestyle='dashed',
                       label=f"{est['label']} {axis_labels[i]}" if i == 0
                       else None)

    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])
    ax1.set_xlabel('X (J2000)')
    ax1.set_ylabel('Y (J2000)')
    ax1.set_zlabel('Z (J2000)')
    ax1.set_title(f'{title} — Attitude (q0)\n' + '  |  '.join(subtitle_parts),
                  fontsize=11)
    ax1.legend(fontsize=7, loc='upper left')

    # ── Right: Omega direction (body frame) ──
    ax2 = fig.add_subplot(122, projection='3d')

    # Unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax2.plot_wireframe(xs, ys, zs, alpha=0.05, color='gray', linewidth=0.3)

    # True omega direction
    w_true_dir = true_omega0 / np.linalg.norm(true_omega0)
    ax2.quiver(0, 0, 0, *w_true_dir, color='black', linewidth=3,
               arrow_length_ratio=0.08, label='True w dir')

    # Estimated omega directions
    subtitle_parts_w = []
    for est in estimates:
        w_dir = est['w0'] / np.linalg.norm(est['w0'])
        # Flip if negative dot (sign ambiguity)
        if np.dot(w_dir, w_true_dir) < 0:
            w_dir = -w_dir

        w_err = omega_dir_err(est['w0'], true_omega0)
        w_mag = np.rad2deg(np.linalg.norm(est['w0']))
        w_mag_err = (w_mag - true_omega_mag) / true_omega_mag * 100
        subtitle_parts_w.append(
            f"{est['label']}: {w_err:.1f}° dir, {w_mag_err:+.1f}% mag")

        ax2.quiver(0, 0, 0, *w_dir, color=est['color'], linewidth=2,
                   arrow_length_ratio=0.08,
                   label=f"{est['label']}: {w_err:.1f}° dir, {w_mag_err:+.1f}% mag")
        _draw_arc(ax2, w_true_dir, w_dir, est['color'])

    # Body frame axes for reference (faint)
    for i in range(3):
        ax2.quiver(0, 0, 0, *(body_axes[i] * 0.3), color='gray', linewidth=0.5,
                   arrow_length_ratio=0.15, alpha=0.4)
        ax2.text(*(body_axes[i] * 0.35), axis_labels[i],
                 fontsize=7, color='gray', alpha=0.5)

    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2])
    ax2.set_zlim([-1.2, 1.2])
    ax2.set_xlabel('X (body)')
    ax2.set_ylabel('Y (body)')
    ax2.set_zlabel('Z (body)')
    ax2.set_title(f'{title} — Omega direction (body frame)\n'
                  + '  |  '.join(subtitle_parts_w), fontsize=11)
    ax2.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.close(fig)
    return fig
