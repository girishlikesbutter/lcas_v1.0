#!/usr/bin/env python3
"""
m071d — Check IS-901 symmetry in the BODY frame.

Transform all facets to body frame (applying component positions and
articulation), then check symmetry under 180° rotation about +X.
"""

import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
from src.articulation import compute_rotation_matrices_from_angles

CTX = setup_experiment(n_observations=500, noise_sigma=0.05, random_seed=42,
                       true_omega_deg=(0.5, -0.3, 2.0),
                       end_time_utc='2020-02-05T11:00:00',
                       skip_true_lc=True)

satellite = CTX.satellite

# Get articulation matrices at epoch 0 (static — panels at 0°, dishes at 15°)
art_angles = {
    'SP_North': np.array([0.0]),
    'SP_South': np.array([0.0]),
    'AD_East': np.array([15.0]),
    'AD_West': np.array([15.0]),
}
art_matrices = compute_rotation_matrices_from_angles(art_angles, satellite)

# Collect ALL facets in body frame
all_centroids = []
all_normals = []
all_areas = []
all_comp_names = []

for comp in satellite.components:
    # Get component transform (position + articulation at epoch 0)
    if comp.name in art_matrices:
        T = art_matrices[comp.name][0]  # (4, 4) transform at epoch 0
    else:
        T = np.eye(4)

    for f in comp.facets:
        # Transform vertices to body frame
        verts_local = f.vertices  # (3, 3)
        verts_body = np.zeros_like(verts_local)
        for vi in range(3):
            v4 = np.append(verts_local[vi], 1.0)
            v4_body = T @ v4
            verts_body[vi] = v4_body[:3]

        centroid = verts_body.mean(axis=0)

        # Transform normal
        R = T[:3, :3]
        normal_body = R @ f.normal
        normal_body = normal_body / np.linalg.norm(normal_body)

        all_centroids.append(centroid)
        all_normals.append(normal_body)
        all_areas.append(f.area)
        all_comp_names.append(comp.name)

centroids = np.array(all_centroids)
normals = np.array(all_normals)
areas = np.array(all_areas)
comp_names = np.array(all_comp_names)

print(f"Total facets in body frame: {len(centroids)}")
print(f"Components: {np.unique(comp_names)}")

# Check 180° rotation about +X: [x, y, z] → [x, -y, -z]
print("\n" + "=" * 60)
print("180° rotation about +X symmetry check")
print("=" * 60)

rot_centroids = centroids.copy()
rot_centroids[:, 1] *= -1
rot_centroids[:, 2] *= -1
rot_normals = normals.copy()
rot_normals[:, 1] *= -1
rot_normals[:, 2] *= -1

matched = 0
unmatched_list = []
max_c_err = 0
max_n_err = 0

for i in range(len(centroids)):
    dists = np.linalg.norm(centroids - rot_centroids[i], axis=1)
    closest = np.argmin(dists)
    d = dists[closest]
    if d < 0.05:
        matched += 1
        max_c_err = max(max_c_err, d)
        ndiff = np.linalg.norm(normals[closest] - rot_normals[i])
        max_n_err = max(max_n_err, ndiff)
    else:
        unmatched_list.append((i, d, comp_names[i]))

print(f"\nMatched: {matched}/{len(centroids)}")
print(f"Unmatched: {len(unmatched_list)}")
print(f"Max centroid error: {max_c_err:.6f} m")
print(f"Max normal error: {max_n_err:.6f}")

if unmatched_list:
    print(f"\nUnmatched facets by component:")
    from collections import Counter
    counts = Counter(item[2] for item in unmatched_list)
    for comp_name, count in counts.items():
        print(f"  {comp_name}: {count}")

# Per-component check
print("\nPer-component breakdown:")
for cname in np.unique(comp_names):
    mask = comp_names == cname
    c = centroids[mask]
    n = normals[mask]
    rc = c.copy(); rc[:, 1] *= -1; rc[:, 2] *= -1
    rn = n.copy(); rn[:, 1] *= -1; rn[:, 2] *= -1

    m = 0
    for i in range(len(c)):
        # Search against ALL facets (cross-component matching)
        dists = np.linalg.norm(centroids - rc[i], axis=1)
        closest = np.argmin(dists)
        if dists[closest] < 0.05:
            m += 1
    print(f"  {cname:15s}: {m}/{len(c)} matched (against all facets)")

print(f"\nSYMMETRIC: {'YES' if len(unmatched_list) == 0 else 'NO'}")
