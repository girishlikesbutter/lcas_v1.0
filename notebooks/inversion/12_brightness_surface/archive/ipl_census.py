#!/usr/bin/env python3
"""
IPL Census — Extract IPL data from 100 precomputed isoshell HTML viewers.

For each seed, extracts:
  - Number of IPL-length local minima
  - At each minimum: epoch, loop count, centroid angular distance to true PAB,
    number of loops, all centroid directions
  - Population statistics: what fraction have ≥2, ≥3 tight minima?

Output: JSON summary + printed table.
"""

import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter

VIEWER_DIR = Path(__file__).resolve().parents[3] / \
    "data" / "results" / "inversion_diagnostics" / "isoshell_viewer"


def extract_data_from_html(html_path):
    """Extract the __SURFACE_DATA__ JSON from an isoshell HTML file."""
    text = html_path.read_text()
    # The data is injected as: const D = {...};  (single line, minified JSON)
    m = re.search(r'const D\s*=\s*(\{.+?\});\s*$', text, re.MULTILINE)
    if not m:
        return None
    return json.loads(m.group(1))


def analyse_seed(data):
    """Analyse IPL data for one seed."""
    ipl = data['iplData']
    traj = data['trajectory']

    minima = ipl['ipl_minima']
    lengths = ipl['ipl_lengths']
    loop_counts = ipl['ipl_loop_counts']
    ang_dists = ipl['ipl_ang_dist']
    centroids = ipl['ipl_centroids']
    peak_epochs = traj['peak_epochs']
    omega_dps = traj['omega_mag_dps']
    n_obs = traj['n_obs']

    # For each minimum, gather detailed info
    min_info = []
    for m in minima:
        ep = m['ep']
        ad = m['angDist']  # angular distance PAB↔active centroid
        lc = loop_counts[ep]
        length = lengths[ep]
        ep_centroids = centroids[ep]

        # Is this minimum near a peak epoch?
        near_peak = any(abs(ep - pe) <= 5 for pe in peak_epochs)

        min_info.append({
            'ep': ep,
            'ang_dist_deg': ad,
            'loop_count': lc,
            'ipl_length': length,
            'near_peak': near_peak,
            'centroids': ep_centroids,
        })

    # Count tight minima (centroid within various thresholds)
    tight_5 = sum(1 for m in min_info if m['ang_dist_deg'] is not None
                  and m['ang_dist_deg'] < 5.0)
    tight_10 = sum(1 for m in min_info if m['ang_dist_deg'] is not None
                   and m['ang_dist_deg'] < 10.0)
    tight_15 = sum(1 for m in min_info if m['ang_dist_deg'] is not None
                   and m['ang_dist_deg'] < 15.0)

    # Count minima near peaks
    near_peak_count = sum(1 for m in min_info if m['near_peak'])

    # Average centroid distance at minima
    valid_dists = [m['ang_dist_deg'] for m in min_info
                   if m['ang_dist_deg'] is not None]
    avg_dist = np.mean(valid_dists) if valid_dists else None
    min_dist = min(valid_dists) if valid_dists else None

    return {
        'seed': traj['seed'],
        'omega_dps': omega_dps,
        'n_peaks': len(peak_epochs),
        'n_minima': len(minima),
        'n_tight_5': tight_5,
        'n_tight_10': tight_10,
        'n_tight_15': tight_15,
        'n_near_peak': near_peak_count,
        'avg_centroid_dist': round(avg_dist, 2) if avg_dist is not None else None,
        'min_centroid_dist': round(min_dist, 2) if min_dist is not None else None,
        'minima_detail': min_info,
    }


def main():
    results = []
    for seed in range(100):
        html_path = VIEWER_DIR / f"seed_{seed:03d}.html"
        if not html_path.exists():
            print(f"  SKIP seed {seed}: no HTML file")
            continue
        print(f"  Extracting seed {seed}...", end='\r')
        data = extract_data_from_html(html_path)
        if data is None:
            print(f"  SKIP seed {seed}: couldn't parse JSON")
            continue
        info = analyse_seed(data)
        results.append(info)

    print(f"\n{'='*80}")
    print(f"IPL CENSUS — {len(results)} seeds analysed")
    print(f"{'='*80}\n")

    # Population statistics
    n = len(results)
    print("=== Minima counts ===")
    min_counts = Counter(r['n_minima'] for r in results)
    for k in sorted(min_counts):
        print(f"  {k} minima: {min_counts[k]} seeds ({100*min_counts[k]/n:.0f}%)")
    print(f"  Mean: {np.mean([r['n_minima'] for r in results]):.1f}")

    print("\n=== Tight minima (centroid < 5°) ===")
    tight5_counts = Counter(r['n_tight_5'] for r in results)
    for k in sorted(tight5_counts):
        print(f"  {k} tight: {tight5_counts[k]} seeds ({100*tight5_counts[k]/n:.0f}%)")
    ge2_tight5 = sum(1 for r in results if r['n_tight_5'] >= 2)
    ge3_tight5 = sum(1 for r in results if r['n_tight_5'] >= 3)
    print(f"  ≥2 tight: {ge2_tight5} seeds ({100*ge2_tight5/n:.0f}%)")
    print(f"  ≥3 tight: {ge3_tight5} seeds ({100*ge3_tight5/n:.0f}%)")

    print("\n=== Tight minima (centroid < 10°) ===")
    tight10_counts = Counter(r['n_tight_10'] for r in results)
    for k in sorted(tight10_counts):
        print(f"  {k} tight: {tight10_counts[k]} seeds ({100*tight10_counts[k]/n:.0f}%)")
    ge2_tight10 = sum(1 for r in results if r['n_tight_10'] >= 2)
    ge3_tight10 = sum(1 for r in results if r['n_tight_10'] >= 3)
    print(f"  ≥2 tight: {ge2_tight10} seeds ({100*ge2_tight10/n:.0f}%)")
    print(f"  ≥3 tight: {ge3_tight10} seeds ({100*ge3_tight10/n:.0f}%)")

    print("\n=== Tight minima (centroid < 15°) ===")
    ge2_tight15 = sum(1 for r in results if r['n_tight_15'] >= 2)
    ge3_tight15 = sum(1 for r in results if r['n_tight_15'] >= 3)
    print(f"  ≥2 tight: {ge2_tight15} seeds ({100*ge2_tight15/n:.0f}%)")
    print(f"  ≥3 tight: {ge3_tight15} seeds ({100*ge3_tight15/n:.0f}%)")

    print("\n=== Centroid distance at minima ===")
    all_dists = [m['ang_dist_deg'] for r in results for m in r['minima_detail']
                 if m['ang_dist_deg'] is not None]
    if all_dists:
        print(f"  Overall: median={np.median(all_dists):.1f}°, "
              f"mean={np.mean(all_dists):.1f}°, "
              f"P25={np.percentile(all_dists, 25):.1f}°, "
              f"P75={np.percentile(all_dists, 75):.1f}°")

    # Best minima per seed (smallest centroid distance)
    print("\n=== Best minimum per seed (smallest centroid distance) ===")
    best_dists = [r['min_centroid_dist'] for r in results
                  if r['min_centroid_dist'] is not None]
    if best_dists:
        print(f"  median={np.median(best_dists):.1f}°, "
              f"mean={np.mean(best_dists):.1f}°, "
              f"P10={np.percentile(best_dists, 10):.1f}°, "
              f"P90={np.percentile(best_dists, 90):.1f}°")
        under5 = sum(1 for d in best_dists if d < 5)
        under10 = sum(1 for d in best_dists if d < 10)
        print(f"  Best < 5°: {under5}/{n} ({100*under5/n:.0f}%)")
        print(f"  Best < 10°: {under10}/{n} ({100*under10/n:.0f}%)")

    # Near-peak minima
    print("\n=== Minima near peaks (within ±5 epochs) ===")
    near_peak_counts = Counter(r['n_near_peak'] for r in results)
    for k in sorted(near_peak_counts):
        print(f"  {k} near-peak minima: {near_peak_counts[k]} seeds")
    ge2_near = sum(1 for r in results if r['n_near_peak'] >= 2)
    print(f"  ≥2 near-peak: {ge2_near} seeds ({100*ge2_near/n:.0f}%)")

    # Loop count distribution at minima
    print("\n=== Loop counts at minima ===")
    loop_counts_at_min = [m['loop_count'] for r in results
                          for m in r['minima_detail']]
    lc_dist = Counter(loop_counts_at_min)
    for k in sorted(lc_dist):
        print(f"  {k} loops: {lc_dist[k]} minima ({100*lc_dist[k]/len(loop_counts_at_min):.0f}%)")

    # Correlation: omega magnitude vs number of tight minima
    print("\n=== Omega magnitude vs tight minima (< 10°) ===")
    omegas = [r['omega_dps'] for r in results]
    tights = [r['n_tight_10'] for r in results]
    corr = np.corrcoef(omegas, tights)[0, 1]
    print(f"  Correlation: {corr:.3f}")

    # Top 20 best seeds for IPL approach
    print("\n=== Top 20 seeds (by n_tight_10, tiebreak by min_centroid_dist) ===")
    ranked = sorted(results, key=lambda r: (-r['n_tight_10'],
                                             r['min_centroid_dist'] or 999))
    print(f"  {'Seed':>4} {'ω(dps)':>7} {'#peaks':>6} {'#min':>4} "
          f"{'<5°':>3} {'<10°':>4} {'<15°':>4} {'best_d':>6}")
    for r in ranked[:20]:
        print(f"  {r['seed']:4d} {r['omega_dps']:7.3f} {r['n_peaks']:6d} "
              f"{r['n_minima']:4d} {r['n_tight_5']:3d} {r['n_tight_10']:4d} "
              f"{r['n_tight_15']:4d} {r['min_centroid_dist']:6.1f}°")

    # Bottom 10 (worst for IPL)
    print("\n=== Bottom 10 seeds (fewest tight minima) ===")
    for r in ranked[-10:]:
        bd = f"{r['min_centroid_dist']:.1f}°" if r['min_centroid_dist'] is not None else "N/A"
        print(f"  {r['seed']:4d} {r['omega_dps']:7.3f} {r['n_peaks']:6d} "
              f"{r['n_minima']:4d} {r['n_tight_5']:3d} {r['n_tight_10']:4d} "
              f"{r['n_tight_15']:4d} {bd:>6}")

    # The key m103 seeds
    m103_seeds = [0, 1, 6, 11, 12, 14, 19, 24, 27, 28, 33, 36, 44, 46, 58,
                      73, 74, 75, 93]
    print(f"\n=== m103 + m106 tested seeds ===")
    print(f"  {'Seed':>4} {'ω(dps)':>7} {'#peaks':>6} {'#min':>4} "
          f"{'<5°':>3} {'<10°':>4} {'<15°':>4} {'best_d':>6}")
    for r in results:
        if r['seed'] in m103_seeds:
            bd = f"{r['min_centroid_dist']:.1f}°" if r['min_centroid_dist'] is not None else "N/A"
            print(f"  {r['seed']:4d} {r['omega_dps']:7.3f} {r['n_peaks']:6d} "
                  f"{r['n_minima']:4d} {r['n_tight_5']:3d} {r['n_tight_10']:4d} "
                  f"{r['n_tight_15']:4d} {bd:>6}")

    # Save full results
    out_path = VIEWER_DIR / "ipl_census.json"
    # Strip centroids from detail to keep file size reasonable
    save_results = []
    for r in results:
        sr = dict(r)
        sr['minima_detail'] = [{
            'ep': m['ep'],
            'ang_dist_deg': m['ang_dist_deg'],
            'loop_count': m['loop_count'],
            'ipl_length': m['ipl_length'],
            'near_peak': m['near_peak'],
            'n_centroids': len(m['centroids']),
            'centroid_dirs': [c['dir'] for c in m['centroids']],
        } for m in r['minima_detail']]
        save_results.append(sr)

    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
