---
title: "PAB-Contour and Isoshell Framework"
type: concept
sources: ["data/results/inversion_diagnostics/isoshell_viewer/"]
related: ["[[brightness_surface_path_matching]]", "[[alignment-cost]]", "[[grid-search]]", "[[ipl-candidate-generation]]", "[[m106_pairwise_vec_ipl]]"]
created: 2026-04-12
updated: 2026-04-16
confidence: high
---

# PAB-Contour and Isoshell Framework

## Core Abstraction

The satellite's brightness can be represented as a 3D radial surface in a spherical coordinate system (azimuth, elevation, brightness) attached to the body frame — the **PAB-contour** (pab-c). For every direction (a, e) on the body-frame sphere, the brightness value is computed assuming k1 = k2 = h (zero phase angle, no shadows). This is the satellite's "brightness fingerprint."

Given an observed magnitude at any epoch, converting to the pab-c brightness scale defines an **isoshell** — a sphere at that brightness radius. Wherever the isoshell intersects the pab-c surface, we get a set of closed curves: the **ico-PAB-loop (IPL) set**. Each IPL is a closed contour on the body-frame sphere representing all directions from which the satellite would appear at that brightness.

## The Light Curve as Evolving Shells

A light curve is a time series of magnitudes → a time series of isoshells → a time series of IPL sets. At bright peaks the IPLs contract into tight loops around lobe centres. At dim epochs they expand into large curves spanning the sphere. The true body-frame PAB trajectory threads through the IPL sets at every epoch.

**The inversion problem becomes:** find R(t) — consistent with Euler dynamics — such that R(t)^{-1} · p_I(t) lies on the correct IPL at every epoch.

## Key Quantities (precomputed for all 100 seeds)

For each epoch:
- **IPL set length** — total arc length of all loops. Measures constraint tightness (short = well constrained).
- **IPL count** — number of separate loops. Fewer = less ambiguity.
- **IPL centroids** — unit-vector direction of each loop's centroid on the body-frame sphere.
- **Active IPL** — the loop the true PAB belongs to.
- **PAB↔centroid angular distance** — how well the centroid approximates the true PAB direction.

## Population Statistics (100-Seed Census)

Census script: `notebooks/inversion/12_brightness_surface/archive/ipl_census.py`
Results: `data/results/inversion_diagnostics/isoshell_viewer/ipl_census.json`

- 83% of seeds have >=2 tight minima (centroid < 5 deg from truth)
- Median best centroid distance: 1.2 deg
- Dominant loop families: +/-X (from SP lobes) and +/-YZ-combo approx (0, +/-0.815, +/-0.58)
- See [[ipl-census]] for full statistics.

## What a Tight IPL Gives You

When an IPL is tight (small arc length, low centroid distance):
- The centroid is a good approximation of the body-frame PAB direction.
- Knowing the body-frame PAB constrains attitude R(t) to a **1-DOF family** (twist ψ about the PAB axis). This is because spinning the satellite about the PAB axis doesn't change where the PAB points.
- The twist DOF can be resolved by examining the peak shape over 3-5 epochs (the crossing direction constrains ψ), or by requiring dynamical consistency with another epoch.

With N tight IPLs at one epoch, we get N candidate PAB directions → N twist-parameterized attitude families. Between two such epochs: N₁ × N₂ combinations, each requiring a 1D consistency check under Euler dynamics. This is **orders of magnitude fewer candidates** than the 40K grid search.

## Error Sources

1. **Phase angle (minor for centroids, significant for phi discrimination):** pab-c assumes k1=k2=h. Real GEO observations have α ≈ 5-30°. Effect on lobe positions: centroid error << 1°. BUT for phi discrimination, the phase angle breaks the zero-phase symmetry — actual lo-fi (with k1≠k2) shows disc=1.067 while zero-phase shows disc<1 (m110 vs m109).
2. **Shadows (CRITICAL for phi):** pab-c has no shadows. Shadows can suppress lobes by up to 4.6 mag (±Y lobes, solar panel occlusion). 40-44% of epochs have |shadow| > 0.1 mag. See [[shadow-asymmetry]] and [[isoshell-phi-limits]].

## Visualisation Tool

Interactive HTML viewer at `data/results/inversion_diagnostics/isoshell_viewer/seed_XXX.html` for all 100 seeds. Shows:
- 3D pab-c surface (black-to-white greyscale, Ultra 2x resolution: 40,962 vertices)
- Animated IPL contours (cyan = active loop, orange = inactive)
- PAB dot + fading trail
- Centroid markers (diamond dots) at every epoch
- Angle arc at IPL-length local minima
- Three synced plots: light curve, IPL set length (colour-coded by loop count), PAB↔centroid angular distance

Generator: `notebooks/inversion/lib/isoshell_viewer.py --seed N`
Batch: `notebooks/inversion/lib/generate_all_isoshells.py --workers 4`

## Key Files
- `notebooks/inversion/lib/brightness_surface.py` — original pab-c surface generator
- `notebooks/inversion/lib/brightness_surface_template.html` — original template
- `notebooks/inversion/lib/isoshell_viewer.py` — isoshell viewer generator (Python precomputation)
- `notebooks/inversion/lib/isoshell_template.html` — isoshell viewer template
- `notebooks/inversion/lib/generate_all_isoshells.py` — batch generator for all 100 seeds
- `data/results/inversion_diagnostics/isoshell_viewer/` — 100 precomputed HTML viewers + summary JSON
