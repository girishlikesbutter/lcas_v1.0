#!/usr/bin/env python3
"""
Extract full per-epoch IPL centroid data from all 100 HTML isoshell viewers.

For each seed, extracts at ALL 500 epochs:
  - ipl_centroids: list of centroid directions per loop
  - ipl_loop_counts: number of loops
  - ipl_lengths: total IPL set length
  - ipl_ang_dist: angular distance from active centroid to true PAB

Also extracts peak epochs and computes crossing arcs at each peak.

Output: data/results/inversion_diagnostics/isoshell_viewer/ipl_all_epochs.npz
"""

import json
import re
import sys
import numpy as np
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parents[3] / \
    "data" / "results" / "inversion_diagnostics" / "isoshell_viewer"
OUT_PATH = VIEWER_DIR / "ipl_all_epochs.npz"


def extract_data_from_html(html_path):
    """Extract the full JSON data blob from an isoshell HTML file."""
    text = html_path.read_text()
    m = re.search(r'const D\s*=\s*(\{.+?\});\s*$', text, re.MULTILINE)
    if not m:
        return None
    return json.loads(m.group(1))


def process_seed(seed, data):
    """Extract per-epoch IPL data + peak crossing info."""
    ipl = data['iplData']
    traj = data['trajectory']

    n_obs = traj['n_obs']
    peak_epochs = traj['peak_epochs']

    # Per-epoch data (already computed in the HTML)
    lengths = ipl['ipl_lengths']           # (n_obs,) float
    loop_counts = ipl['ipl_loop_counts']   # (n_obs,) int
    ang_dists = ipl['ipl_ang_dist']        # (n_obs,) float or None
    centroids_raw = ipl['ipl_centroids']   # (n_obs,) list of {dir, isActive}
    active_dirs = ipl['ipl_active_centroid_dir']  # (n_obs,) [x,y,z] or None

    # Pack centroids into a structured format:
    # For each epoch: list of (x,y,z) unit vectors
    # Store as ragged array via object dtype
    all_centroids = []
    for ep in range(n_obs):
        dirs = [c['dir'] for c in centroids_raw[ep]]
        all_centroids.append(np.array(dirs, dtype=np.float32) if dirs else np.zeros((0, 3), dtype=np.float32))

    # Active centroid directions (the one closest to truth PAB)
    active_centroid_dirs = np.zeros((n_obs, 3), dtype=np.float32)
    for ep in range(n_obs):
        if active_dirs[ep] is not None:
            active_centroid_dirs[ep] = active_dirs[ep]

    # Angular distances (None → NaN)
    ang_dist_arr = np.array([d if d is not None else np.nan for d in ang_dists], dtype=np.float32)

    return {
        'lengths': np.array(lengths, dtype=np.float32),
        'loop_counts': np.array(loop_counts, dtype=np.int16),
        'ang_dists': ang_dist_arr,
        'active_centroid_dirs': active_centroid_dirs,
        'all_centroids': all_centroids,  # list of (n_loops, 3) arrays
        'peak_epochs': np.array(peak_epochs, dtype=np.int16),
    }


def compute_crossing_arcs(seed_data, half_window=3):
    """For each peak, compute the centroid trajectory across flanking epochs.

    Returns per-peak:
      - peak_ep: epoch index
      - arc_dirs: (2*half_window+1, 3) active centroid directions across the peak
      - arc_valid: (2*half_window+1,) bool — whether centroid data exists
      - entry_dir: unit vector from flanking centroid toward peak centroid (approach)
      - exit_dir: unit vector from peak centroid toward flanking centroid (departure)
      - crossing_dir: entry→exit unit vector on the sphere (the crossing direction)
    """
    n_obs = len(seed_data['lengths'])
    peaks = seed_data['peak_epochs']
    active = seed_data['active_centroid_dirs']
    ang_dists = seed_data['ang_dists']
    W = half_window

    arcs = []
    for peak_ep in peaks:
        peak_ep = int(peak_ep)
        if peak_ep < W or peak_ep >= n_obs - W:
            continue

        # Collect active centroid directions across the window
        arc = np.zeros((2 * W + 1, 3), dtype=np.float32)
        valid = np.zeros(2 * W + 1, dtype=bool)
        for i, ep in enumerate(range(peak_ep - W, peak_ep + W + 1)):
            d = active[ep]
            if np.linalg.norm(d) > 0.5:  # valid centroid
                arc[i] = d / np.linalg.norm(d)
                valid[i] = True

        if not valid[W]:  # no centroid at peak center
            continue

        center = arc[W]

        # Entry direction: average of pre-peak centroids offset from center
        pre_offsets = []
        for i in range(W):
            if valid[i]:
                offset = arc[i] - center
                if np.linalg.norm(offset) > 1e-6:
                    pre_offsets.append(offset / np.linalg.norm(offset))

        # Exit direction: average of post-peak centroids offset from center
        post_offsets = []
        for i in range(W + 1, 2 * W + 1):
            if valid[i]:
                offset = arc[i] - center
                if np.linalg.norm(offset) > 1e-6:
                    post_offsets.append(offset / np.linalg.norm(offset))

        entry = np.mean(pre_offsets, axis=0) if pre_offsets else np.zeros(3)
        exit_d = np.mean(post_offsets, axis=0) if post_offsets else np.zeros(3)

        # Crossing direction: from entry side to exit side
        crossing = exit_d - entry
        cn = np.linalg.norm(crossing)
        crossing = crossing / cn if cn > 1e-6 else np.zeros(3)

        arcs.append({
            'peak_ep': peak_ep,
            'center': center,
            'arc_dirs': arc,
            'arc_valid': valid,
            'entry_dir': entry,
            'exit_dir': exit_d,
            'crossing_dir': crossing,
            'peak_ang_dist': float(ang_dists[peak_ep]),
            'n_valid_flanks': sum(valid) - 1,  # exclude center
        })

    return arcs


def main():
    all_seed_data = {}

    for seed in range(100):
        html_path = VIEWER_DIR / f"seed_{seed:03d}.html"
        if not html_path.exists():
            print(f"  SKIP seed {seed}: no HTML file")
            continue
        print(f"  Extracting seed {seed}...", end='\r')
        data = extract_data_from_html(html_path)
        if data is None:
            print(f"  SKIP seed {seed}: parse failed")
            continue

        seed_data = process_seed(seed, data)
        arcs = compute_crossing_arcs(seed_data)
        seed_data['crossing_arcs'] = arcs

        all_seed_data[seed] = seed_data

    print(f"\n{'=' * 70}")
    print(f"Extracted IPL data for {len(all_seed_data)} seeds")
    print(f"{'=' * 70}\n")

    # Save as NPZ with object arrays (ragged centroids)
    save_dict = {}
    for seed, sd in all_seed_data.items():
        prefix = f"s{seed:03d}_"
        save_dict[prefix + 'lengths'] = sd['lengths']
        save_dict[prefix + 'loop_counts'] = sd['loop_counts']
        save_dict[prefix + 'ang_dists'] = sd['ang_dists']
        save_dict[prefix + 'active_centroid_dirs'] = sd['active_centroid_dirs']
        save_dict[prefix + 'peak_epochs'] = sd['peak_epochs']
        # Pack centroids as object array (ragged)
        save_dict[prefix + 'centroids'] = np.array(sd['all_centroids'], dtype=object)
        # Pack crossing arcs
        n_arcs = len(sd['crossing_arcs'])
        if n_arcs > 0:
            save_dict[prefix + 'arc_peak_eps'] = np.array([a['peak_ep'] for a in sd['crossing_arcs']], dtype=np.int16)
            save_dict[prefix + 'arc_centers'] = np.array([a['center'] for a in sd['crossing_arcs']], dtype=np.float32)
            save_dict[prefix + 'arc_crossing_dirs'] = np.array([a['crossing_dir'] for a in sd['crossing_arcs']], dtype=np.float32)
            save_dict[prefix + 'arc_entry_dirs'] = np.array([a['entry_dir'] for a in sd['crossing_arcs']], dtype=np.float32)
            save_dict[prefix + 'arc_exit_dirs'] = np.array([a['exit_dir'] for a in sd['crossing_arcs']], dtype=np.float32)
            save_dict[prefix + 'arc_ang_dists'] = np.array([a['peak_ang_dist'] for a in sd['crossing_arcs']], dtype=np.float32)
            save_dict[prefix + 'arc_n_valid'] = np.array([a['n_valid_flanks'] for a in sd['crossing_arcs']], dtype=np.int16)

    np.savez_compressed(str(OUT_PATH), **save_dict)
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"Saved: {OUT_PATH} ({size_mb:.1f} MB)")

    # Summary statistics on crossing arcs
    print(f"\n{'=' * 70}")
    print(f"CROSSING ARC SUMMARY")
    print(f"{'=' * 70}")

    all_crossing_norms = []
    all_n_valid = []
    seeds_with_arcs = 0
    total_arcs = 0

    for seed, sd in all_seed_data.items():
        arcs = sd['crossing_arcs']
        if arcs:
            seeds_with_arcs += 1
            total_arcs += len(arcs)
            for a in arcs:
                cn = np.linalg.norm(a['crossing_dir'])
                all_crossing_norms.append(cn)
                all_n_valid.append(a['n_valid_flanks'])

    all_crossing_norms = np.array(all_crossing_norms)
    all_n_valid = np.array(all_n_valid)

    print(f"Seeds with crossing arcs: {seeds_with_arcs}/100")
    print(f"Total arcs: {total_arcs}")
    print(f"Arcs with detectable crossing (|crossing_dir| > 0.1): "
          f"{np.sum(all_crossing_norms > 0.1)}/{total_arcs} "
          f"({100 * np.sum(all_crossing_norms > 0.1) / max(total_arcs, 1):.0f}%)")
    print(f"Valid flanking epochs per arc: median={np.median(all_n_valid):.0f}, "
          f"mean={np.mean(all_n_valid):.1f}")

    # Per-seed breakdown for ATT_FAIL seeds
    att_fail_seeds = [0, 11, 27, 46, 58, 75]
    print(f"\n=== ATT_FAIL seeds: crossing arc quality ===")
    for seed in att_fail_seeds:
        if seed not in all_seed_data:
            print(f"  Seed {seed}: no data")
            continue
        arcs = all_seed_data[seed]['crossing_arcs']
        good = [a for a in arcs if np.linalg.norm(a['crossing_dir']) > 0.1
                and a['peak_ang_dist'] < 10]
        print(f"  Seed {seed}: {len(arcs)} total arcs, {len(good)} with detectable crossing + tight centroid (<10°)")
        for a in good[:5]:
            print(f"    ep={a['peak_ep']}: ang_dist={a['peak_ang_dist']:.1f}°, "
                  f"|crossing|={np.linalg.norm(a['crossing_dir']):.3f}, "
                  f"n_flanks={a['n_valid_flanks']}")


if __name__ == '__main__':
    main()
