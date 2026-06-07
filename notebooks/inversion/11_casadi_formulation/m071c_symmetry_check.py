#!/usr/bin/env python3
"""
m071c — Check if IS-901 has X-Y plane symmetry (z → -z).

For each component, check if every facet has a mirror partner
under z-reflection. Report per-component and overall.
"""

import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config.rso_config_manager import RSO_ConfigManager
from src.io.stl_loader import STLLoader

config_manager = RSO_ConfigManager(PROJECT_ROOT)
config = config_manager.load_config('intelsat_901/intelsat_901_config.yaml')
satellite = STLLoader.create_satellite_from_stl_config(config=config, config_manager=config_manager)

print("Checking IS-901 X-Y plane symmetry (z → -z)")
print("=" * 70)

for comp in satellite.components:
    verts = np.array([f.vertices for f in comp.facets])  # (N, 3, 3)
    normals = np.array([f.normal for f in comp.facets])   # (N, 3)
    areas = np.array([f.area for f in comp.facets])       # (N,)

    # Centroid of each facet
    centroids = verts.mean(axis=1)  # (N, 3)

    # Reflect through X-Y plane: z → -z
    ref_centroids = centroids.copy()
    ref_centroids[:, 2] *= -1
    ref_normals = normals.copy()
    ref_normals[:, 2] *= -1

    max_centroid_err = 0
    max_normal_err = 0
    unmatched = 0

    for i in range(len(centroids)):
        dists = np.linalg.norm(centroids - ref_centroids[i], axis=1)
        closest = np.argmin(dists)
        d = dists[closest]
        if d > 0.1:
            unmatched += 1
            continue
        max_centroid_err = max(max_centroid_err, d)
        ndiff = np.linalg.norm(normals[closest] - ref_normals[i])
        max_normal_err = max(max_normal_err, ndiff)

    sym = "SYMMETRIC" if unmatched == 0 and max_centroid_err < 0.01 else "ASYMMETRIC"
    print(f"  {comp.name:15s} | {len(comp.facets):4d} facets | "
          f"unmatched={unmatched} | centroid_err={max_centroid_err:.4f}m | "
          f"normal_err={max_normal_err:.4f} | {sym}")

print()
print("Also checking unique normals:")
un = np.load('data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz',
             allow_pickle=True)['unique_normals']
gn = np.load('data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz',
             allow_pickle=True)['group_names']
for i in range(len(un)):
    ref = un[i].copy()
    ref[2] *= -1
    # Find match
    dists = [np.linalg.norm(un[j] - ref) for j in range(len(un))]
    closest = np.argmin(dists)
    print(f"  {str(gn[i]):>4s} [{un[i][0]:+.4f},{un[i][1]:+.4f},{un[i][2]:+.4f}] "
          f"→ z-flip → [{ref[0]:+.4f},{ref[1]:+.4f},{ref[2]:+.4f}] "
          f"= {str(gn[closest]):>4s} (dist={dists[closest]:.4f})")
