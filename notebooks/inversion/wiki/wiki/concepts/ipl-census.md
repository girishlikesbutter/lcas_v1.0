---
title: "IPL Census — 100-Seed Population Statistics"
type: concept
sources: ["data/results/inversion_diagnostics/isoshell_viewer/ipl_census.json"]
related: ["[[pab-contour-isoshell]]", "[[ipl-candidate-generation]]", "[[alignment-cost]]", "[[m096_exp1_oracle_grid]]"]
created: 2026-04-12
updated: 2026-04-16
confidence: high
---

# IPL Census — 100-Seed Population Statistics

## Overview

Census of IPL (ico-PAB-loop) minima across all 100 seeds, quantifying how many seeds have tight enough constraints for IPL-based candidate generation.

Census script: `notebooks/inversion/12_brightness_surface/archive/ipl_census.py`
Results: `data/results/inversion_diagnostics/isoshell_viewer/ipl_census.json`

## Key Statistics

- 100 seeds analysed, mean 21.1 IPL minima per seed
- **83% have >=2 tight minima (centroid < 5 deg)** — enough for IPL candidate generation
- **89% have >=2 tight minima (< 10 deg)**
- **92% have best centroid distance < 5 deg** — median best = 1.2 deg
- 98% have >=2 near-peak minima (within +/-5 epochs of a peak)

## Loop Count Distribution at Minima

- Dominant loop count: 4 (55%), then 8 (16%), 6 (11%)
- 4 loops = +/-X and +/-YZ-combo (approx (0, +/-0.815, +/-0.58)) directions

## Omega vs Constraint Tightness

- **Strong omega vs tight minima correlation (rho = 0.746)** — slow tumblers (< 0.25 dps) have fewest constraints
- 6 seeds with 0 tight minima (< 10 deg): mostly slow tumblers (omega < 0.25 dps) + seed 81 outlier

## Discrimination Gap (Critical Finding)

For grid-failure seeds 27 and 33, at the tightest IPL minima the truth PAB sits ~35 deg from the nearest standard normal (between Y and Z). The standard alignment cost is flat there. But the IPL centroid captures truth within 1-3 deg. This explains why these seeds fail: the cost function has no discrimination at the epochs where the PAB is between normals.

## Plateau Observation

Constant-magnitude phases mean the PAB is circling a single IPL loop around a lobe. This provides continuous body-frame constraints during plateaus (not just point constraints at peaks). Only meaningful at bright-to-moderate magnitude levels where IPLs are tight.
