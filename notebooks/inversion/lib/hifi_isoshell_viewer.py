#!/usr/bin/env python3
"""
Non-phase-invariant PAB Manifold viewer using the hi-fi MLP surrogate.

Generates an interactive HTML animation showing:
  - Non-phase-invariant PAB Manifold in body frame — deforms per epoch as the
    sun direction k1_body(t) sweeps through the body frame.
  - Brightness shell at m_obs(t) — sphere that expands/contracts with the LC.
  - Loops (intersections of shell and manifold) — evolve with topology changes.
  - True PAB direction as a marker on the manifold (lies on a loop by construction).
  - k1_body marker on the unit sphere for reference.
  - Loop count / total loop length / PAB↔active-centroid panels.

Unlike the lofi viewer (which assumes k1 = k2 = PAB → phase-invariant), this
evaluates brightness via the hi-fi surrogate `B(k1_body, k2_body, panel, dish,
obs_dist)`. For each candidate PAB direction h on an icosphere, k2 is derived
as the reflection of k1 about h: k2 = 2(k1·h)h − k1. With k1_body(t) and the
observed mag_hifi(t) taken from the truth trajectory, the animation is a
diagnostic / oracle tool (truth attitude required).

Usage:
    python hifi_isoshell_viewer.py --seed 0
"""

import base64
import json
import sys
import argparse
import time
import numpy as np
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

_SURROGATE_DIR = Path('/home/girish/surrogate_model')
if str(_SURROGATE_DIR) not in sys.path:
    sys.path.insert(0, str(_SURROGATE_DIR))

from notebooks.inversion.lib.brightness_surface import (
    load_satellite, extract_component_data, _build_sliders, _build_brdf_defaults,
    FAMILY_NAMES, FAMILY_COLORS,
)
from notebooks.inversion.lib.isoshell_viewer import (
    _generate_icosphere, _loop_centroid_dir, _loop_length, _ang_dist_deg,
)

from surrogate import SurrogateModel

# IS-901 inversion convention — see notebooks/inversion/wiki/wiki/concepts/surrogate-model.md
PANEL_DEG_CONSTANT = 0.0
DISH_DEG_CONSTANT = 15.0
SURROGATE_WEIGHTS = _SURROGATE_DIR / 's10_5M_weights.npz'
SURROGATE_NORM = _SURROGATE_DIR / 's10_5M_normalization.npz'

R_MIN, R_MAX = 0.1, 1.0


def load_m048_trajectory(seed):
    """Load a per-seed m048 trajectory NPZ."""
    traj_path = _PROJ / "data" / "results" / "inversion_diagnostics" / \
        "m048_trajectories" / "per_trajectory" / f"traj_seed{seed:03d}.npz"
    return dict(np.load(str(traj_path), allow_pickle=True))


def compute_manifold_hifi(k1_body, pab_verts, panel_deg, dish_deg, obs_dist_km, model):
    """Evaluate the non-phase-invariant PAB Manifold at every icosphere vertex.

    For fixed k1_body at this epoch, at each candidate PAB direction h = pab_verts[i]
    we derive k2 = 2(k1·h)h − k1 (reflection of k1 about h), then evaluate the
    surrogate magnitude at (k1, k2).
    """
    k1h = pab_verts @ k1_body                       # (N,)
    k2 = 2.0 * k1h[:, None] * pab_verts - k1_body   # (N, 3), unit by construction
    k1_batch = np.tile(k1_body[None, :], (len(pab_verts), 1))
    return model.predict_magnitude(k1_batch, k2, panel_deg, dish_deg, obs_dist_km)


def _radial_scale(log_val, log_min, log_max):
    """Map a log-flux value to a radial distance in [R_MIN, R_MAX]."""
    span = log_max - log_min
    if span < 1e-10:
        return R_MAX
    t = (log_val - log_min) / span
    t = float(np.clip(t, 0.0, 1.0))
    return R_MIN + t * (R_MAX - R_MIN)


def _extract_loops_global(verts, faces, flux, target_flux, log_min, log_max):
    """Marching triangles at fixed target flux; radial placement uses global log range.

    Variant of isoshell_viewer._extract_contour_loops that takes external log_min /
    log_max so loops are placed at radial heights consistent across epochs.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        log_flux = np.where(flux > 0, np.log10(flux), np.nan)
    if target_flux <= 0:
        return []
    log_target = float(np.log10(target_flux))
    r = _radial_scale(log_target, log_min, log_max)

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
            found = False
            for seg_i, end_i in adj.get(k, []):
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
            found = False
            for seg_i, end_i in adj.get(k, []):
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


def _active_loop_index(loops, pab_surf):
    """Index of the loop whose nearest vertex is closest to the PAB surface point."""
    best, best_d = -1, float('inf')
    for i, loop in enumerate(loops):
        d = float(np.linalg.norm(loop - pab_surf, axis=1).min())
        if d < best_d:
            best_d = d
            best = i
    return best


def _pack_uint16(arr):
    """Pack a numpy uint16 array to base64 (little-endian)."""
    return base64.b64encode(arr.astype('<u2').tobytes()).decode('ascii')


def precompute_hifi_data(traj_data, model, subdiv=5,
                         panel_deg=PANEL_DEG_CONSTANT,
                         dish_deg=DISH_DEG_CONSTANT):
    """Precompute per-epoch PAB Manifold, loops, and diagnostics.

    Ships static icosphere (verts + face indices once) + per-epoch log-flux
    packed as uint16 base64. The client computes xyz and face intensity on
    demand per epoch. This keeps ultra-quality (subdiv=5 or 6) under ~100 MB.

    Returns a JSON-serializable dict; see the `return {...}` block for fields.
    """
    t0 = time.time()

    verts, faces = _generate_icosphere(subdiv)
    n_v = len(verts)
    n_f = len(faces)

    k1_body_arr = np.asarray(traj_data['k1_body'], dtype=float)
    pab_body_arr = np.asarray(traj_data['pab_body'], dtype=float)
    obs_dist = np.asarray(traj_data['obs_dist'], dtype=float)
    mag_hifi = np.asarray(traj_data['mag_hifi'], dtype=float)
    n_obs = len(k1_body_arr)

    # Stage 1: evaluate the manifold over the icosphere for every epoch
    print(f'    stage 1: surrogate eval over {n_obs} epochs x {n_v} verts...')
    mags_per_epoch = np.empty((n_obs, n_v), dtype=np.float64)
    # Backlit mask: h's where phase(k1, k2_derived) > 90°, i.e. k1·k2_derived < 0.
    # Using cos(phase) = 2(k1·h)^2 − 1, this is (k1·h)^2 < 0.5. At those h's the
    # observer is on the opposite side of the satellite from the sun → no facet
    # has both n·k1 > 0 AND n·k2 > 0, so apparent brightness is physically zero.
    # We KEEP the surrogate value (so the manifold is still visible — in real
    # inversion we won't know which directions are shadowed) but tag each vertex
    # backlit/lit so the JS can color them distinctly.
    backlit_mask = np.empty((n_obs, n_v), dtype=bool)
    for ep in range(n_obs):
        k1 = k1_body_arr[ep]
        mags_per_epoch[ep] = compute_manifold_hifi(
            k1, verts,
            panel_deg, dish_deg, float(obs_dist[ep]),
            model,
        )
        k1h = verts @ k1
        backlit_mask[ep] = (k1h ** 2) < 0.5

    # Pseudo-flux (smaller mag → brighter → larger flux). Keep backlit values.
    flux_per_epoch = 10 ** (-mags_per_epoch / 2.5)
    with np.errstate(invalid='ignore', divide='ignore'):
        log_flux_per_epoch = np.where(flux_per_epoch > 0,
                                      np.log10(flux_per_epoch),
                                      np.nan)

    # GLOBAL log range across all epochs → one radius on the plot corresponds to
    # one brightness value, consistently. Under this convention the shell truly
    # expands/contracts with m_obs(t) and the manifold deforms because its values
    # change, not because the axis rescales. Backlit verts are handled below via
    # a dedicated q=0 sentinel (mapped to r=0 in JS).
    log_min = float(np.nanmin(log_flux_per_epoch))
    log_max = float(np.nanmax(log_flux_per_epoch))
    span = max(log_max - log_min, 1e-10)

    # Stage 2: per-epoch loops + diagnostics (surface data shipped as packed log-flux)
    print(f'    stage 2: loop extraction + diagnostics...')
    loops_per_epoch = []
    active_loop_per_epoch = np.zeros(n_obs, dtype=int)
    pab_surf_per_epoch = np.zeros((n_obs, 3), dtype=np.float32)
    k1_surf_per_epoch = np.zeros((n_obs, 3), dtype=np.float32)

    ipl_lengths = []
    ipl_loop_counts = []
    ipl_loop_lengths = []
    ipl_ang_dist = []
    ipl_centroids = []
    ipl_active_centroid_dir = []

    for ep in range(n_obs):
        flux_field = flux_per_epoch[ep]

        # Loop extraction at target = m_obs(t), using the GLOBAL log range so
        # loops placed at the same brightness sit at the same radius at every
        # epoch (one radius = one brightness across the whole animation).
        m_obs = float(mag_hifi[ep])
        target_flux = 10 ** (-m_obs / 2.5)
        loops = _extract_loops_global(verts, faces, flux_field, target_flux,
                                      log_min, log_max)
        loops_per_epoch.append(
            [np.round(l, 4).astype(np.float32).tolist() for l in loops]
        )

        loop_lens = [round(_loop_length(l), 6) for l in loops]
        ipl_lengths.append(round(sum(loop_lens), 6))
        ipl_loop_counts.append(len(loops))
        ipl_loop_lengths.append(loop_lens)

        # PAB marker position on manifold — r at mag_hifi log-flux (global range)
        r_pab = _radial_scale(np.log10(target_flux) if target_flux > 0 else log_min,
                              log_min, log_max)
        pab_dir = pab_body_arr[ep]
        pab_dir = pab_dir / max(np.linalg.norm(pab_dir), 1e-10)
        pab_surf = pab_dir * r_pab
        pab_surf_per_epoch[ep] = pab_surf.astype(np.float32)

        # k1_body marker on unit sphere (slightly outside R_MAX for visibility)
        k1_dir = k1_body_arr[ep]
        k1_dir = k1_dir / max(np.linalg.norm(k1_dir), 1e-10)
        k1_surf_per_epoch[ep] = (k1_dir * (R_MAX + 0.15)).astype(np.float32)

        # Active loop = the loop PAB is closest to
        if loops:
            active_idx = _active_loop_index(loops, pab_surf)
        else:
            active_idx = -1
        active_loop_per_epoch[ep] = active_idx

        # Centroids per epoch (matches lofi schema)
        centroids = []
        for li, loop in enumerate(loops):
            cdir = _loop_centroid_dir(loop)
            centroids.append({
                'dir': np.round(cdir, 4).tolist(),
                'isActive': bool(li == active_idx),
            })
        ipl_centroids.append(centroids)

        if active_idx >= 0:
            active_cdir = _loop_centroid_dir(loops[active_idx])
            ipl_ang_dist.append(round(_ang_dist_deg(pab_dir, active_cdir), 4))
            ipl_active_centroid_dir.append(np.round(active_cdir, 6).tolist())
        else:
            ipl_ang_dist.append(None)
            ipl_active_centroid_dir.append(None)

    # Local-minima detection on IPL total length (same heuristic as lofi)
    W = 3
    ipl_minima = []
    for ep in range(W, n_obs - W):
        is_min = all(ipl_lengths[ep] <= ipl_lengths[ep + d]
                     for d in range(-W, W + 1) if d != 0)
        if not is_min:
            continue
        baseline = max(ipl_lengths[max(0, ep - 10)],
                       ipl_lengths[min(n_obs - 1, ep + 10)])
        if ipl_lengths[ep] > baseline * 0.9:
            continue
        ipl_minima.append({'ep': int(ep), 'angDist': ipl_ang_dist[ep]})

    # --- Pack data for efficient HTML embedding ---
    # GLOBAL quantize: log-flux normalized to [log_min, log_max], mapped to uint16.
    # We ship the manifold in full (backlit vertices at their surrogate-predicted
    # radius — in real inversion we don't know a priori which directions are
    # shadowed), and ship a separate per-vertex backlit flag so the JS can color
    # shadowed faces distinctly.
    with np.errstate(invalid='ignore'):
        t = (log_flux_per_epoch - log_min) / span
    t = np.clip(t, 0.0, 1.0)
    q = t * 65535.0
    q_int = np.clip(q, 0, 65535).astype(np.uint16)
    log_flux_b64 = _pack_uint16(q_int)
    # Pack backlit mask as one byte per (epoch, vertex)
    backlit_b64 = base64.b64encode(
        backlit_mask.astype(np.uint8).tobytes()
    ).decode('ascii')

    # Static icosphere verts packed as int16 in [-32767, 32767] mapping to [-1, 1]
    verts_int = np.clip(verts * 32767.0, -32767, 32767).astype(np.int16)
    verts_b64 = base64.b64encode(verts_int.astype('<i2').tobytes()).decode('ascii')

    # Face indices as uint32 concatenated (verts count fits uint32 easily)
    faces_int = faces.astype(np.uint32)
    faces_b64 = base64.b64encode(faces_int.astype('<u4').tobytes()).decode('ascii')

    elapsed = time.time() - t0
    mb_logflux = len(log_flux_b64) / (1024 * 1024)
    mb_loops = sum(sum(len(l) for l in ep) for ep in loops_per_epoch) * 12 / (1024 * 1024)
    print(f'  Hifi manifold precomputation: {elapsed:.1f}s '
          f'({n_obs} epochs, subdiv {subdiv}, {n_v} verts, '
          f'{len(ipl_minima)} minima)')
    print(f'    packed log-flux: {mb_logflux:.1f} MB base64, '
          f'~loops data: {mb_loops:.1f} MB')

    return {
        'log_flux_b64': log_flux_b64,
        'backlit_b64': backlit_b64,
        'verts_b64': verts_b64,
        'faces_b64': faces_b64,
        'loops_per_epoch': loops_per_epoch,
        'active_loop_per_epoch': active_loop_per_epoch.tolist(),
        'pab_surf_per_epoch': pab_surf_per_epoch.tolist(),
        'k1_surf_per_epoch': k1_surf_per_epoch.tolist(),
        'n_verts': n_v,
        'n_faces': n_f,
        'n_obs': n_obs,
        'subdiv': subdiv,
        'log_min': log_min,
        'log_max': log_max,
        'R_MIN': R_MIN,
        'R_MAX': R_MAX,
        'ipl_lengths': ipl_lengths,
        'ipl_loop_counts': ipl_loop_counts,
        'ipl_loop_lengths': ipl_loop_lengths,
        'ipl_ang_dist': ipl_ang_dist,
        'ipl_centroids': ipl_centroids,
        'ipl_active_centroid_dir': ipl_active_centroid_dir,
        'ipl_minima': ipl_minima,
    }


def build_trajectory_payload(traj_data, seed):
    """Condense trajectory data to what the HTML needs."""
    n_obs = len(traj_data['k1_body'])
    dt = float(traj_data['dt_sampling'])
    peak_epochs = np.asarray(traj_data['hifi_peak_epochs']).astype(int).tolist()
    omega_mag_dps = float(traj_data['omega_mag_dps'])
    return {
        'seed': int(seed),
        'n_obs': n_obs,
        'dt': round(dt, 4),
        'pab_body': np.round(np.asarray(traj_data['pab_body']), 6).tolist(),
        'k1_body': np.round(np.asarray(traj_data['k1_body']), 6).tolist(),
        'k2_body': np.round(np.asarray(traj_data['k2_body']), 6).tolist(),
        'mag_hifi': np.round(np.asarray(traj_data['mag_hifi']), 4).tolist(),
        'mag_lofi': np.round(np.asarray(traj_data['mag_lofi']), 4).tolist(),
        'phase_angle_deg': np.round(np.asarray(traj_data['phase_angle_3d']), 3).tolist(),
        'obs_dist_km': np.round(np.asarray(traj_data['obs_dist']), 3).tolist(),
        'peak_epochs': peak_epochs,
        'omega_mag_dps': round(omega_mag_dps, 4),
    }


def generate(seed=0, output=None, title=None, subdiv=5, traj_source='m048'):
    """Generate the hifi isoshell viewer HTML file for one seed."""
    if traj_source != 'm048':
        raise NotImplementedError("Only m048 supported in this PoC.")

    t_total = time.time()

    print(f'[seed {seed}] loading satellite + surrogate...')
    satellite = load_satellite()
    components = extract_component_data(satellite)
    sliders = _build_sliders(components)
    brdf_defaults = _build_brdf_defaults(components)

    model = SurrogateModel(str(SURROGATE_WEIGHTS), str(SURROGATE_NORM))

    print(f'[seed {seed}] loading m048 trajectory...')
    traj_data = load_m048_trajectory(seed)

    print(f'[seed {seed}] precomputing non-phase-invariant manifold...')
    hifi_data = precompute_hifi_data(traj_data, model, subdiv=subdiv)
    traj_payload = build_trajectory_payload(traj_data, seed)

    if title is None:
        title = f'IS-901 Hi-Fi PAB Manifold — Seed {seed} (m048)'

    if output is None:
        output = Path(__file__).parent / f'hifi_isoshell_seed{seed:03d}.html'

    data = {
        'components': components,
        'sliders': sliders,
        'brdfDefaults': brdf_defaults,
        'familyNames': FAMILY_NAMES,
        'familyColors': FAMILY_COLORS,
        'title': title,
        'trajectory': traj_payload,
        'hifi': hifi_data,
    }

    template_path = Path(__file__).parent / 'hifi_isoshell_template.html'
    html = template_path.read_text()
    html = html.replace('__SURFACE_DATA__', json.dumps(data, separators=(',', ':')))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    size_kb = output.stat().st_size / 1024
    elapsed = time.time() - t_total
    print(f'Saved: {output} ({size_kb:.0f} KB, total {elapsed:.1f}s)')
    print(f'  Seed {seed}, {traj_payload["n_obs"]} epochs, '
          f'|omega|={traj_payload["omega_mag_dps"]:.3f} deg/s, '
          f'phase_angle range {min(traj_payload["phase_angle_deg"]):.1f}-'
          f'{max(traj_payload["phase_angle_deg"]):.1f} deg')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate hifi PAB Manifold viewer')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--subdiv', type=int, default=5,
                        help='icosphere subdivisions (3=642, 4=2562, 5=10242, 6=40962 verts). '
                             'Default 5 matches old "Ultra" setting. 6 is "Ultra 2x".')
    args = parser.parse_args()
    generate(seed=args.seed, output=args.output, subdiv=args.subdiv)
