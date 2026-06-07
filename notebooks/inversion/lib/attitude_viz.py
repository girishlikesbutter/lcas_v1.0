#!/usr/bin/env python3
"""
Attitude visualisation tool — 2x2 comparison of two arbitrary states.

Generates one PNG showing:
  Top-left:     State A attitude — satellite mesh + body axes rotated by q0
  Top-right:    State B attitude — satellite mesh + body axes rotated by q0
  Bottom-left:  State A omega — satellite mesh in body frame + omega arrow
  Bottom-right: State B omega — satellite mesh in body frame + omega arrow
  All panels:   Observer direction vector (cyan) at epoch 0

States can come from:
  - Trajectory database (true state for a seed)
  - Result JSON (winner or any hi-fi candidate)
  - Explicit q0/omega values

Usage:
  # Truth vs winner (default labels)
  python3 attitude_viz.py 27

  # Truth vs specific candidate
  python3 attitude_viz.py 27 --candidate 1

  # Compare winner vs candidate with custom labels
  python3 attitude_viz.py 27 --a winner --b candidate:1 --label-a "Winner" --label-b "Rank #2"

  # Different experiment prefix
  python3 attitude_viz.py --prefix m093 6 24 36
"""

import sys
import json
import argparse
import numpy as np
import quaternion as quat_mod
from pathlib import Path
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
import os; os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.utils.geometry_utils import build_rotation_matrix

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
DATA_DIR = RESULTS_DIR / "m046_trajectories"
VALID_TRAJ_SOURCES = ("m046", "m048")

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
    """Extract all facet triangles in body frame, with articulation applied."""
    art_angles = {'AD_East': 15.0, 'AD_West': 15.0}
    components = []
    for comp in satellite.components:
        R_body = quat_mod.as_rotation_matrix(comp.relative_orientation)
        pos = np.array(comp.relative_position)
        angle = art_angles.get(comp.name, 0.0)
        if angle != 0.0 and comp.articulation_parameters is not None:
            rot_axis = np.array(comp.articulation_parameters.rotation_axis)
            R_art = build_rotation_matrix(angle, rot_axis)[:3, :3]
        else:
            R_art = np.eye(3)
        R_combined = R_body @ R_art
        triangles = []
        for facet in comp.facets:
            verts = np.array(facet.vertices)
            transformed = (R_combined @ verts.T).T + pos
            triangles.append(transformed)
        components.append({'name': comp.name, 'faces': np.array(triangles)})
    return components


def render_satellite(ax, comp_data, R=None):
    for comp in comp_data:
        faces = comp['faces'].copy()
        if R is not None:
            faces = np.einsum('ij,nkj->nki', R, faces)
        poly = Poly3DCollection(faces, alpha=0.7, linewidth=0.15, edgecolor='#333333')
        poly.set_facecolor(COMP_COLORS.get(comp['name'], '#999999'))
        ax.add_collection3d(poly)


def draw_body_axes(ax, R, length=8.0):
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


def draw_observer_vector(ax, obs_dir, length=8.0):
    vec = obs_dir * length
    ax.quiver(0, 0, 0, *vec, color='#00bfbf',
              arrow_length_ratio=0.06, linewidth=2.5)
    ax.text(vec[0]*1.15, vec[1]*1.15, vec[2]*1.15,
            'obs', fontsize=9, color='#00bfbf',
            fontweight='bold', ha='center', va='center')


def draw_omega_arrow(ax, omega_dir, label, length=12.0):
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


def omega_dir_err(w1, w2):
    d1 = w1 / np.linalg.norm(w1)
    d2 = w2 / np.linalg.norm(w2)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(np.dot(d1, d2)), 0, 1))))


def parse_inline_spec(spec):
    """Parse 'inline:q0=w,x,y,z;w0=wx,wy,wz' into (q0_array, w0_array).

    Returns None if spec is not an inline string.
    """
    if not isinstance(spec, str) or not spec.startswith('inline:'):
        return None
    body = spec[len('inline:'):]
    parts = {}
    for kv in body.split(';'):
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        parts[k.strip()] = [float(x) for x in v.strip().split(',')]
    if 'q0' not in parts or 'w0' not in parts:
        raise ValueError(
            f"inline spec must have q0=w,x,y,z;w0=wx,wy,wz; got {spec!r}")
    q = np.array(parts['q0'])
    w = np.array(parts['w0'])
    if q.shape != (4,) or w.shape != (3,):
        raise ValueError(
            f"inline spec dimensions wrong (q0 needs 4, w0 needs 3); got {spec!r}")
    return q, w


def resolve_state(spec, seed, prefix, master, result):
    """Resolve a state specification to (q0_wxyz, omega_rad).

    spec can be:
      'truth'        — true state from trajectory database
      'winner'       — winner from result.json
      'candidate:N'  — hi-fi candidate N from result.json
      'inline:q0=w,x,y,z;w0=wx,wy,wz' — literal (q0, ω) pair (rad/s)
      or a dict with 'q0_wxyz' and 'w0_rad' keys (for programmatic use)
    """
    if isinstance(spec, dict):
        return np.array(spec['q0_wxyz']), np.array(spec['w0_rad'])

    inline = parse_inline_spec(spec)
    if inline is not None:
        return inline

    if spec == 'truth':
        # master here is the dict returned by load_truth (or legacy npz for m046)
        if isinstance(master, dict) and 'q0_wxyz' in master:
            return np.asarray(master['q0_wxyz']), np.asarray(master['omega0_rad'])
        return master['q0s'][seed], master['omega0s'][seed]

    if spec == 'winner':
        w = result['winner']
        return np.array(w['q0_wxyz']), np.deg2rad(np.array(w['w0_dps']))

    if spec.startswith('candidate:'):
        idx = int(spec.split(':')[1])
        cands = result.get('hifi_candidates', [])
        if idx >= len(cands):
            raise ValueError(f"Candidate {idx} not found (only {len(cands)} available)")
        c = cands[idx]
        if 'q0_wxyz' not in c:
            raise ValueError(f"Candidate {idx} missing q0_wxyz (pipeline didn't save full state)")
        return np.array(c['q0_wxyz']), np.deg2rad(np.array(c['w0_dps']))

    raise ValueError(f"Unknown state spec: {spec}")


def _needs_result_json(spec):
    """True if a state spec requires loading result.json."""
    if isinstance(spec, dict):
        return False
    if not isinstance(spec, str):
        return False
    if spec == 'winner' or spec.startswith('candidate:'):
        return True
    return False


def _per_seed_obs_dir_j2000(seed, traj_source, master_legacy):
    """Return observer direction in J2000 at epoch 0 for the given seed.

    For m046 we read the precomputed k2_body and rotate by truth attitude.
    For m048 we compute (obs_pos - sat_pos) at epoch 0 directly via SPICE.
    """
    if traj_source == 'm046':
        # Same as the legacy code: world-frame obs dir at epoch 0 is
        # R_b2w_truth @ k2_body[seed, 0, :].
        from lib.traj_source import load_truth as _load_truth
        truth = _load_truth(seed, 'm046')
        k2_body_ep0 = master_legacy['k2_body'][seed, 0, :]
        R_truth = quat_wxyz_to_rotmat(truth['q0_wxyz'])
        v = R_truth @ k2_body_ep0
        return v / np.linalg.norm(v)
    # m048: use experiment_setup ctx to get sat_pos / obs_pos at epoch 0
    from lib.lc_compare import load_hifi_context as _lhc
    ctx = _lhc(seed, traj_source='m048')
    obs0 = ctx['obs_pos'][0]
    sat0 = ctx['sat_pos'][0]
    v = obs0 - sat0
    return v / np.linalg.norm(v)


def generate_attitude_viz(seeds, prefix='m090', output_dir=None,
                          state_a='truth', state_b='winner',
                          label_a='Truth', label_b='Estimate',
                          traj_source='m046'):
    """Generate attitude comparison plots for the given seeds.

    Parameters
    ----------
    seeds : list of int
        Trajectory seeds to visualise.
    prefix : str
        Result directory prefix (e.g. 'm090', 'm093').
    output_dir : Path or None
        Where to save PNGs. Defaults to RESULTS_DIR.
    state_a, state_b : str or dict
        State specifications. See resolve_state() for options.
    label_a, label_b : str
        Labels for the two states in plot titles.
    traj_source : str
        'm046' (legacy single-window) or 'm048' (per-seed random starts).

    Returns
    -------
    list of Path
        Paths to saved PNG files.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    if traj_source not in VALID_TRAJ_SOURCES:
        raise ValueError(f"traj_source must be one of {VALID_TRAJ_SOURCES}; got {traj_source!r}")

    print("Loading satellite model...", flush=True)
    satellite = load_satellite()
    comp_data = get_body_frame_faces(satellite)
    print(f"  {sum(len(c['faces']) for c in comp_data)} total faces")

    # m046 keeps the legacy master npz path for k2_body lookup; m048 doesn't
    # have an equivalent precomputed table, so we'll compute obs_dir on demand.
    master_legacy = (np.load(str(DATA_DIR / "m046_trajectories.npz"), allow_pickle=True)
                     if traj_source == 'm046' else None)

    needs_rj = _needs_result_json(state_a) or _needs_result_json(state_b)
    results = {}
    for seed in seeds:
        if needs_rj:
            rpath = RESULTS_DIR / f"{prefix}_seed{seed:03d}" / "result.json"
            if not rpath.exists():
                print(f"  WARNING: {rpath} not found, skipping seed {seed}")
                continue
            with open(rpath) as f:
                results[seed] = json.load(f)
        else:
            results[seed] = None  # placeholder so the loop runs

    view_elev, view_azim = 25, -60
    saved = []

    for seed in seeds:
        if seed not in results:
            continue

        # For 'truth' specs, load_truth returns a per-seed dict that resolve_state
        # can use; for legacy m046 we also keep the master npz to feed k2_body.
        from lib.traj_source import load_truth as _load_truth
        truth_dict = _load_truth(seed, traj_source)

        q0_a, w0_a = resolve_state(state_a, seed, prefix, truth_dict, results[seed])
        q0_b, w0_b = resolve_state(state_b, seed, prefix, truth_dict, results[seed])

        R_a = quat_wxyz_to_rotmat(q0_a)
        R_b = quat_wxyz_to_rotmat(q0_b)

        # Observer direction in J2000 (world frame). Independent of attitude;
        # computed from the SPICE geometry at epoch 0.
        obs_dir_j2000 = _per_seed_obs_dir_j2000(seed, traj_source, master_legacy)
        # Observer direction in each state's body frame. Under convention (a)
        # (post-fix), R = R_J2000→body, so v_body = R @ v_J2000.
        obs_dir_body_a = R_a @ obs_dir_j2000
        obs_dir_body_a /= np.linalg.norm(obs_dir_body_a)
        obs_dir_body_b = R_b @ obs_dir_j2000
        obs_dir_body_b /= np.linalg.norm(obs_dir_body_b)

        w_a_dir = w0_a / np.linalg.norm(w0_a)
        w_b_dir = w0_b / np.linalg.norm(w0_b)
        w_a_mag = np.rad2deg(np.linalg.norm(w0_a))
        w_b_mag = np.rad2deg(np.linalg.norm(w0_b))

        # Error metrics between A and B
        err_axis, err_angle = error_axis_angle(q0_b, q0_a)
        dot_x = abs(np.dot(err_axis, [1, 0, 0]))
        q_dist = err_angle
        w_dir_dist = omega_dir_err(w0_a, w0_b)

        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(
            f'Seed {seed} — {label_a} vs {label_b}\n'
            f'q0 distance: {q_dist:.1f}°, '
            f'w_dir distance: {w_dir_dist:.1f}°, '
            f'Error axis: [{err_axis[0]:+.2f}, {err_axis[1]:+.2f}, '
            f'{err_axis[2]:+.2f}], dot(axis, +X) = {dot_x:.3f}',
            fontsize=12, y=0.98)

        # Top-left: State A attitude
        ax1 = fig.add_subplot(221, projection='3d')
        render_satellite(ax1, comp_data, R=R_a)
        draw_body_axes(ax1, R_a)
        draw_observer_vector(ax1, obs_dir_j2000)
        setup_ax(ax1, f'{label_a} attitude (q0)')
        ax1.view_init(elev=view_elev, azim=view_azim)

        # Top-right: State B attitude
        ax2 = fig.add_subplot(222, projection='3d')
        render_satellite(ax2, comp_data, R=R_b)
        draw_body_axes(ax2, R_b)
        draw_observer_vector(ax2, obs_dir_j2000)
        setup_ax(ax2, f'{label_b} attitude (q0 dist = {q_dist:.1f}°)')
        ax2.view_init(elev=view_elev, azim=view_azim)

        # Bottom-left: State A omega (body frame)
        ax3 = fig.add_subplot(223, projection='3d')
        render_satellite(ax3, comp_data, R=None)
        draw_omega_arrow(ax3, w_a_dir, f'{label_a} w\n({w_a_mag:.2f} dps)')
        draw_observer_vector(ax3, obs_dir_body_a)
        setup_ax(ax3, f'{label_a} omega (body frame)')
        ax3.view_init(elev=view_elev, azim=view_azim)

        # Bottom-right: State B omega (body frame)
        ax4 = fig.add_subplot(224, projection='3d')
        render_satellite(ax4, comp_data, R=None)
        draw_omega_arrow(ax4, w_b_dir, f'{label_b} w\n({w_b_mag:.2f} dps)')
        draw_observer_vector(ax4, obs_dir_body_b)
        setup_ax(ax4, f'{label_b} omega (w_dir dist = {w_dir_dist:.1f}°)')
        ax4.view_init(elev=view_elev, azim=view_azim)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = Path(output_dir) / f"{prefix}_seed{seed:03d}_attitude_viz.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()
        saved.append(out_path)

    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attitude comparison visualisation')
    parser.add_argument('seeds', nargs='+', type=int, help='Trajectory seeds')
    parser.add_argument('--prefix', default='m090',
                        help='Result directory prefix (default: m090)')
    parser.add_argument('--a', default='truth', dest='state_a',
                        help='State A spec: truth, winner, candidate:N')
    parser.add_argument('--b', default='winner', dest='state_b',
                        help='State B spec: truth, winner, candidate:N')
    parser.add_argument('--label-a', default=None,
                        help='Label for state A (default: derived from spec)')
    parser.add_argument('--label-b', default=None,
                        help='Label for state B (default: derived from spec)')
    parser.add_argument('--traj-source', default='m046', choices=VALID_TRAJ_SOURCES,
                        help="Trajectory source: 'm046' (legacy single-window) "
                             "or 'm048' (per-seed random starts). Default m046.")
    args = parser.parse_args()

    # Default labels from spec
    def default_label(spec):
        if spec == 'truth': return 'Truth'
        if spec == 'winner': return 'Winner'
        if spec.startswith('candidate:'): return f'Candidate #{spec.split(":")[1]}'
        if spec.startswith('inline:'): return 'Inline state'
        return spec
    la = args.label_a or default_label(args.state_a)
    lb = args.label_b or default_label(args.state_b)

    generate_attitude_viz(args.seeds, prefix=args.prefix,
                          state_a=args.state_a, state_b=args.state_b,
                          label_a=la, label_b=lb,
                          traj_source=args.traj_source)
