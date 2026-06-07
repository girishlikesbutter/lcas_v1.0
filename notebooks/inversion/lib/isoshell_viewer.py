#!/usr/bin/env python3
"""
Interactive isoshell light curve viewer.

Extends the brightness surface visualizer with:
  - Animated iso-brightness contours (isoshells) from an observed light curve
  - True body-frame PAB path overlay
  - Light curve subplot with epoch cursor
  - Time scrubbing and playback

Usage:
    python isoshell_viewer.py [--seed 28] [-o output.html]
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from notebooks.inversion.lib.brightness_surface import (
    load_satellite, extract_component_data, _build_sliders, _build_brdf_defaults,
    FAMILY_NAMES, FAMILY_COLORS,
)


def load_trajectory(seed):
    """Load trajectory data for a given seed from the 100-trajectory database."""
    traj_path = _PROJ / "data" / "results" / "inversion_diagnostics" / \
        "m046_trajectories" / "m046_trajectories.npz"
    traj = dict(np.load(str(traj_path), allow_pickle=True))

    n_obs = int(traj['n_obs'])
    dt = float(traj['dt_sampling'])

    # Body-frame PAB at each epoch (precomputed with true attitude)
    pab_body = traj['pab_body'][seed]  # (500, 3)

    # Magnitudes
    mag_hifi = traj['mag_hifi'][seed]  # (500,)
    mag_lofi = traj['mag_lofi'][seed]  # (500,)

    # Peak info
    mask = traj['peak_seeds'] == seed
    peak_epochs = traj['peak_epochs'][mask].tolist()

    # Omega info for display
    omega_mag_dps = float(traj['omega_mags'][seed])

    return {
        'seed': int(seed),
        'n_obs': n_obs,
        'dt': round(dt, 4),
        'pab_body': np.round(pab_body, 6).tolist(),
        'mag_hifi': np.round(mag_hifi, 4).tolist(),
        'mag_lofi': np.round(mag_lofi, 4).tolist(),
        'peak_epochs': peak_epochs,
        'omega_mag_dps': round(omega_mag_dps, 4),
    }


def _generate_icosphere(subdivisions):
    """Generate icosphere vertices and faces (mirrors the JS implementation)."""
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],
        [0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],
        [11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],
        [3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=int)

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []
        verts_list = list(verts)
        for f in faces:
            mids = []
            for e in [(f[0],f[1]), (f[1],f[2]), (f[2],f[0])]:
                key = (min(e), max(e))
                if key not in edge_midpoints:
                    mid = (verts_list[e[0]] + verts_list[e[1]]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints[key] = len(verts_list)
                    verts_list.append(mid)
                mids.append(edge_midpoints[key])
            a, b, c = mids
            new_faces.extend([
                [f[0], a, c], [f[1], b, a], [f[2], c, b], [a, b, c]
            ])
        verts = np.array(verts_list)
        faces = np.array(new_faces, dtype=int)

    return verts, faces


def _compute_brdf(nk1, nk2, hk1, nh, r_d, r_s, n_phong):
    """Ashikhmin-Shirley BRDF (vectorized over arrays)."""
    a = 1 - nk1 / 2
    b = 1 - nk2 / 2
    fr = r_s + (1 - r_s) * (1 - hk1)**5
    rd = (28 * r_d / (23 * np.pi)) * (1 - r_s) * (1 - a**5) * (1 - b**5)
    dn = hk1 * np.maximum(nk1, nk2)
    rs = np.where(dn > 1e-10, ((n_phong + 1) / (8 * np.pi)) * nh**n_phong / dn * fr, 0.0)
    return rd + rs


def _compute_surface_flux(verts, components, ad_angle_deg=15.0):
    """Compute lo-fi zero-phase flux at each icosphere vertex.

    Mirrors the JS computeFlux: for each vertex direction d,
    flux = sum over facet groups of BRDF(n·d, n·d, 1, n·d, ...) * area * (n·d)^2

    Parameters
    ----------
    verts : (N, 3) unit sphere directions
    components : list of component dicts from extract_component_data
    ad_angle_deg : antenna dish angle (degrees), default 15

    Returns
    -------
    flux : (N,) array of flux values
    """
    N = len(verts)
    flux = np.zeros(N)

    for comp in components:
        # Apply articulation if present
        art = comp['articulation']
        ang_rad = 0.0
        if art is not None:
            label = comp['name']
            if label.startswith('AD'):
                ang_rad = np.radians(ad_angle_deg)

        for g in comp['groups']:
            normal = np.array(g['normal'], dtype=float)
            if art is not None and abs(ang_rad) > 1e-6:
                axis = np.array(art['axis'], dtype=float)
                c, s = np.cos(ang_rad), np.sin(ang_rad)
                d = np.dot(axis, normal)
                cr = np.cross(axis, normal)
                normal = normal * c + cr * s + axis * d * (1 - c)

            nd = verts @ normal  # (N,)
            mask = nd > 0
            if not np.any(mask):
                continue
            nd_m = nd[mask]
            brdf = _compute_brdf(nd_m, nd_m, 1.0, nd_m,
                                 g['r_d'], g['r_s'], g['n_phong'])
            flux[mask] += brdf * g['area'] * nd_m * nd_m

    return flux


def _extract_contour_loops(verts, faces, flux, target_flux):
    """Extract iso-flux contour loops via marching triangles.

    Returns list of loops, each loop is (M, 3) array of 3D points on the
    surface (at the radial distance corresponding to target_flux).
    """
    log_flux = np.where(flux > 0, np.log10(flux), np.nan)
    log_target = np.log10(target_flux) if target_flux > 0 else np.nan
    if np.isnan(log_target):
        return []

    log_min = np.nanmin(log_flux)
    log_max = np.nanmax(log_flux)
    span = log_max - log_min
    R_MIN, R_MAX = 0.1, 1.0
    t_norm = (log_target - log_min) / span if span > 1e-10 else 1.0
    t_norm = np.clip(t_norm, 0, 1)
    r = R_MIN + t_norm * (R_MAX - R_MIN)

    # Find edge crossings for each face
    segments = []
    for face in faces:
        lf = [log_flux[face[0]], log_flux[face[1]], log_flux[face[2]]]
        crossings = []
        for e in range(3):
            i0, i1 = e, (e + 1) % 3
            if np.isnan(lf[i0]) or np.isnan(lf[i1]):
                continue
            d0 = lf[i0] - log_target
            d1 = lf[i1] - log_target
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                direction = verts[face[i0]] + t * (verts[face[i1]] - verts[face[i0]])
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                crossings.append(direction * r)
        if len(crossings) == 2:
            segments.append(crossings)

    if not segments:
        return []

    # Chain segments into loops via spatial hashing
    tol = 1e-4
    def pk(p):
        return (round(p[0]/tol), round(p[1]/tol), round(p[2]/tol))

    adj = {}
    for i, seg in enumerate(segments):
        for e in range(2):
            k = pk(seg[e])
            adj.setdefault(k, []).append((i, e))

    used = np.zeros(len(segments), dtype=bool)
    loops = []

    for si in range(len(segments)):
        if used[si]:
            continue
        used[si] = True
        chain = [segments[si][0], segments[si][1]]

        # Extend tail
        while True:
            k = pk(chain[-1])
            nb = adj.get(k, [])
            found = False
            for seg_i, end_i in nb:
                if used[seg_i]:
                    continue
                used[seg_i] = True
                chain.append(segments[seg_i][1 - end_i])
                found = True
                break
            if not found:
                break

        # Extend head
        while True:
            k = pk(chain[0])
            nb = adj.get(k, [])
            found = False
            for seg_i, end_i in nb:
                if used[seg_i]:
                    continue
                used[seg_i] = True
                chain.insert(0, segments[seg_i][1 - end_i])
                found = True
                break
            if not found:
                break

        if len(chain) >= 3:
            loops.append(np.array(chain))

    return loops


def _loop_centroid_dir(loop):
    """Centroid direction of a loop on the unit sphere."""
    norms = np.linalg.norm(loop, axis=1, keepdims=True)
    dirs = loop / np.maximum(norms, 1e-10)
    avg = dirs.mean(axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 1e-10 else np.array([0, 0, 1.0])


def _loop_length(loop):
    """Total arc length of a loop."""
    diffs = np.diff(loop, axis=0)
    return float(np.sqrt((diffs**2).sum(axis=1)).sum())


def _ang_dist_deg(a, b):
    """Angular distance in degrees between two unit vectors."""
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def _compute_target_flux(pab_dir, components, ad_angle_deg):
    """Compute zero-phase-angle flux at a single PAB direction."""
    target_flux = 0.0
    for comp in components:
        art = comp['articulation']
        ang_rad = 0.0
        if art is not None and comp['name'].startswith('AD'):
            ang_rad = np.radians(ad_angle_deg)
        for g in comp['groups']:
            normal = np.array(g['normal'], dtype=float)
            if art is not None and abs(ang_rad) > 1e-6:
                axis = np.array(art['axis'], dtype=float)
                c, s = np.cos(ang_rad), np.sin(ang_rad)
                d = np.dot(axis, normal)
                cr = np.cross(axis, normal)
                normal = normal * c + cr * s + axis * d * (1 - c)
            nd = np.dot(normal, pab_dir)
            if nd <= 0:
                continue
            brdf = float(_compute_brdf(
                np.array([nd]), np.array([nd]), 1.0, np.array([nd]),
                g['r_d'], g['r_s'], g['n_phong'])[0])
            target_flux += brdf * g['area'] * nd * nd
    return target_flux


def precompute_ipl_data(components, pab_body, subdiv=6, ad_angle_deg=15.0):
    """Precompute all IPL data for all epochs.

    Returns dict with:
      ipl_lengths: [float] — total IPL set length per epoch
      ipl_loop_counts: [int] — number of IPLs per epoch
      ipl_loop_lengths: [[float]] — per-loop lengths per epoch
      ipl_ang_dist: [float|null] — angular distance PAB↔active centroid (deg)
      ipl_centroids: [[{dir, isActive}]] — all loop centroids per epoch
      ipl_active_centroid_dir: [[float]|null] — unit vector of active centroid per epoch
      ipl_minima: [{ep, angDist}] — detected local minima epochs
    """
    import time
    t0 = time.time()

    verts, faces = _generate_icosphere(subdiv)
    flux = _compute_surface_flux(verts, components, ad_angle_deg=ad_angle_deg)
    n_obs = len(pab_body)

    # Precompute log flux range (constant across epochs)
    log_flux = np.where(flux > 0, np.log10(flux), np.nan)
    log_min = float(np.nanmin(log_flux))
    log_max = float(np.nanmax(log_flux))
    span = log_max - log_min

    pab_arr = np.array(pab_body)  # (n_obs, 3)

    ipl_lengths = []
    ipl_loop_counts = []
    ipl_loop_lengths = []
    ipl_ang_dist = []
    ipl_centroids = []
    ipl_active_centroid_dir = []

    for ep in range(n_obs):
        pab = pab_arr[ep]
        pab_dir = pab / np.linalg.norm(pab)

        target_flux = _compute_target_flux(pab_dir, components, ad_angle_deg)
        loops = _extract_contour_loops(verts, faces, flux, target_flux)

        # Per-loop lengths
        loop_lens = [round(_loop_length(l), 6) for l in loops]
        ipl_lengths.append(round(sum(loop_lens), 6))
        ipl_loop_counts.append(len(loops))
        ipl_loop_lengths.append(loop_lens)

        # Surface radius for this brightness
        log_target = np.log10(target_flux) if target_flux > 0 else 0
        t_n = (log_target - log_min) / span if span > 1e-10 else 1.0
        t_n = float(np.clip(t_n, 0, 1))
        r_pab = 0.1 + t_n * 0.9
        pab_surf = pab_dir * r_pab

        # Find active loop + compute ALL centroids
        best_loop = -1
        best_dist = float('inf')
        for li, loop in enumerate(loops):
            d = float(np.linalg.norm(loop - pab_surf, axis=1).min())
            if d < best_dist:
                best_dist = d
                best_loop = li

        centroids = []
        for li, loop in enumerate(loops):
            cdir = _loop_centroid_dir(loop)
            centroids.append({
                'dir': np.round(cdir, 4).tolist(),
                'isActive': li == best_loop,
            })

        ipl_centroids.append(centroids)

        if best_loop >= 0:
            active_cdir = _loop_centroid_dir(loops[best_loop])
            ipl_ang_dist.append(round(_ang_dist_deg(pab_dir, active_cdir), 4))
            ipl_active_centroid_dir.append(np.round(active_cdir, 6).tolist())
        else:
            ipl_ang_dist.append(None)
            ipl_active_centroid_dir.append(None)

    # Find local minima
    W = 3
    ipl_minima = []
    for ep in range(W, n_obs - W):
        is_min = all(ipl_lengths[ep] <= ipl_lengths[ep + d]
                     for d in range(-W, W + 1) if d != 0)
        if not is_min:
            continue
        baseline = max(ipl_lengths[max(0, ep-10)], ipl_lengths[min(n_obs-1, ep+10)])
        if ipl_lengths[ep] > baseline * 0.9:
            continue
        ipl_minima.append({
            'ep': int(ep),
            'angDist': ipl_ang_dist[ep],
        })

    elapsed = time.time() - t0
    print(f'  IPL precomputation: {elapsed:.1f}s ({n_obs} epochs, '
          f'subdiv {subdiv}, {len(verts)} verts, {len(ipl_minima)} minima)')

    return {
        'ipl_lengths': ipl_lengths,
        'ipl_loop_counts': ipl_loop_counts,
        'ipl_loop_lengths': ipl_loop_lengths,
        'ipl_ang_dist': ipl_ang_dist,
        'ipl_centroids': ipl_centroids,
        'ipl_active_centroid_dir': ipl_active_centroid_dir,
        'ipl_minima': ipl_minima,
    }


def generate(seed=28, output=None, title=None):
    """Generate the isoshell viewer HTML file."""
    satellite = load_satellite()
    components = extract_component_data(satellite)
    sliders = _build_sliders(components)
    brdf_defaults = _build_brdf_defaults(components)
    traj_data = load_trajectory(seed)

    # Precompute IPL data
    ipl_data = precompute_ipl_data(components, traj_data['pab_body'])

    if title is None:
        title = f'IS-901 Isoshell Viewer — Seed {seed}'

    if output is None:
        output = Path(__file__).parent / f'isoshell_seed{seed:03d}.html'

    data = {
        'components': components,
        'sliders': sliders,
        'brdfDefaults': brdf_defaults,
        'familyNames': FAMILY_NAMES,
        'familyColors': FAMILY_COLORS,
        'title': title,
        'trajectory': traj_data,
        'iplData': ipl_data,
    }

    template_path = Path(__file__).parent / 'isoshell_template.html'
    html = template_path.read_text()
    html = html.replace('__SURFACE_DATA__', json.dumps(data, separators=(',', ':')))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    size_kb = output.stat().st_size / 1024
    print(f'Saved: {output} ({size_kb:.0f} KB)')
    print(f'  Seed {seed}, {traj_data["n_obs"]} epochs, '
          f'|ω|={traj_data["omega_mag_dps"]:.3f} deg/s')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate isoshell viewer')
    parser.add_argument('--seed', type=int, default=28)
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()
    generate(seed=args.seed, output=args.output)
