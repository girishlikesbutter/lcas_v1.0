#!/usr/bin/env python3
"""
Interactive 3D brightness surface generator.

Produces a self-contained HTML file showing a satellite's "brightness
fingerprint" — a radial surface where each direction on the unit sphere
encodes the lo-fi flux produced when the PAB points in that body-frame
direction.  Articulation angles are adjustable in real time via sliders.

Usage:
    python brightness_surface.py [-o output.html]
"""

import json
import sys
import numpy as np
import quaternion as quat_mod
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[3]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader
from src.computation.brdf import BRDFManager
from src.computation.facet_data_extractor import extract_facet_arrays

# ── Reference normals for display arrows ──────────────────────────────
FAMILY_COLORS = ['#ff4444', '#44ff44', '#4488ff', '#ff8800', '#cc44ff']
FAMILY_NAMES = ['X', 'Y', 'Z', 'WD', 'ED']


def load_satellite():
    """Load IS-901 satellite with BRDF parameters applied."""
    config_manager = RSO_ConfigManager(_PROJ)
    config = config_manager.load_config("intelsat_901/intelsat_901_config.yaml")
    satellite = STLLoader.create_satellite_from_stl_config(config, config_manager)
    BRDFManager(config).update_satellite_brdf_parameters(satellite)
    return satellite


def extract_component_data(satellite):
    """Extract per-component facet groups with un-articulated normals.

    Returns list of component dicts ready for JSON embedding.
    Normals have the component's body-frame rotation applied but NO
    articulation, so JS can rotate them live.
    """
    fa = extract_facet_arrays(satellite)

    components = []
    for comp in satellite.components:
        cs = fa.component_slices.get(comp.name)
        if cs is None:
            continue

        # Apply body-frame rotation only (no articulation)
        body_rot = quat_mod.as_rotation_matrix(comp.relative_orientation)
        local_normals = fa.normals[cs]
        body_normals = (body_rot @ local_normals.T).T

        # Group by unique normal within this component
        rounded = np.round(body_normals, 4)
        unique_normals, inverse = np.unique(rounded, axis=0, return_inverse=True)

        groups = []
        for ui in range(len(unique_normals)):
            mask = inverse == ui
            groups.append({
                'normal': np.round(unique_normals[ui], 6).tolist(),
                'area': round(float(fa.areas[cs][mask].sum()), 6),
                'r_d': float(fa.r_d[cs][mask][0]),
                'r_s': float(fa.r_s[cs][mask][0]),
                'n_phong': float(fa.n_phong[cs][mask][0]),
            })

        # Articulation metadata (if any)
        art = None
        if comp.articulation_parameters is not None:
            ap = comp.articulation_parameters
            art = {
                'axis': [round(float(x), 6) for x in ap.rotation_axis],
                'range': [float(ap.limits['min_angle']),
                          float(ap.limits['max_angle'])],
            }

        components.append({
            'name': comp.name,
            'groups': groups,
            'articulation': art,
        })

    return components


def _build_sliders(components):
    """Auto-detect articulation slider groups from component data.

    Groups components by prefix (e.g. SP_, AD_) that share the same
    articulation axis and range.  Returns slider definitions.
    """
    import re
    buckets = {}  # prefix → {axis, range, components}
    for comp in components:
        art = comp['articulation']
        if art is None:
            continue
        prefix = re.match(r'^([A-Z]+_)', comp['name'])
        key = prefix.group(1) if prefix else comp['name']
        if key not in buckets:
            buckets[key] = {
                'axis': art['axis'],
                'range': art['range'],
                'components': [],
            }
        buckets[key]['components'].append(comp['name'])

    # Friendly labels
    label_map = {'SP_': 'Solar Panels', 'AD_': 'Antenna Dishes'}

    sliders = []
    for prefix, info in buckets.items():
        sliders.append({
            'label': label_map.get(prefix, prefix),
            'components': info['components'],
            'range': info['range'],
            'default': 0.0,
            'step': 1,
        })
    return sliders


def _build_brdf_defaults(components):
    """Extract default BRDF values per component type (Bus, SP, AD).

    Groups components by prefix and takes the BRDF from the first group
    of the first component in each bucket.  Returns a dict keyed by
    a short label with {r_d, r_s, n_phong, components: [names]}.
    """
    import re
    buckets = {}
    label_map = {'Bus': 'Bus', 'SP_': 'Solar Panels', 'AD_': 'Antenna Dishes'}

    for comp in components:
        # Match prefix with underscore (SP_, AD_) or full name (Bus)
        prefix = re.match(r'^([A-Z]+_)', comp['name'])
        key = prefix.group(1) if prefix else comp['name']
        if key not in buckets:
            g0 = comp['groups'][0]
            buckets[key] = {
                'label': label_map.get(key, key),
                'r_d': g0['r_d'],
                'r_s': g0['r_s'],
                'n_phong': g0['n_phong'],
                'components': [],
            }
        buckets[key]['components'].append(comp['name'])

    return list(buckets.values())


def generate(output='brightness_surface.html', title=None):
    """Generate the brightness surface HTML file."""
    satellite = load_satellite()
    components = extract_component_data(satellite)
    sliders = _build_sliders(components)
    brdf_defaults = _build_brdf_defaults(components)

    n_groups = sum(len(c['groups']) for c in components)

    if title is None:
        title = f'IS-901 Brightness Surface'

    data = {
        'components': components,
        'sliders': sliders,
        'brdfDefaults': brdf_defaults,
        'familyNames': FAMILY_NAMES,
        'familyColors': FAMILY_COLORS,
        'title': title,
    }

    template_path = Path(__file__).parent / 'brightness_surface_template.html'
    html = template_path.read_text()
    html = html.replace('__SURFACE_DATA__', json.dumps(data, separators=(',', ':')))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    size_kb = output.stat().st_size / 1024
    print(f'Saved: {output} ({size_kb:.0f} KB)')
    print(f'  {len(components)} components, {n_groups} facet groups, {len(sliders)} sliders')
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate brightness surface')
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(__file__).parent / 'brightness_surface.html'
    generate(output=args.output)
