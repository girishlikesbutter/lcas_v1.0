---
title: "IPL-Based Candidate Generation"
type: branch
sources: ["data/results/inversion_diagnostics/isoshell_viewer/"]
related: ["[[pab-contour-isoshell]]", "[[crossing-geometry-scoring]]", "[[alignment-cost]]", "[[grid-search]]", "[[candidate-selection]]", "[[m106_pairwise_vec_ipl]]"]
created: 2026-04-12
updated: 2026-04-16
confidence: medium
---

# IPL-Based Candidate Generation

## Status: #dead-end

## Question

Can IPL centroids from tight isoshells replace or augment the omega direction grid search, generating attitude/omega candidates directly from the light curve's constraint structure?

## Motivation

The current pipeline uses a 2000-direction × 20-magnitude grid (40,000 candidates) to search for omega. This is brute force — it doesn't use the light curve's structure to guide the search. The grid has ~1-3° spacing, which limits precision and causes grid failures (3/13 seeds in m103 had truth outside the candidate pool).

The [[pab-contour-isoshell]] framework reveals that at epochs where the IPL set length is at a local minimum, the body-frame PAB is tightly constrained to one of a small number (typically 2-4) of candidate directions (the IPL centroids). Each centroid constrains attitude to a 1-DOF family (twist about the PAB). The twist is resolved by peak shape over 3-5 epochs or by dynamical consistency across multiple peaks.

## Proposed Approach

### Per-peak candidate generation
1. At each IPL-length local minimum (detected from precomputed data), extract the IPL centroids.
2. Each centroid gives a body-frame PAB direction → constrains R(t) to 1-DOF (twist ψ).
3. The peak shape over the 3-5 epochs spanning the peak constrains the crossing direction, which resolves ψ to a small discrete set.
4. Result: at each peak, O(2-4) centroids × O(1-3) twist candidates = O(2-12) full attitude candidates.

### Multi-peak consistency
5. For each pair of peaks, propagate attitude candidates from peak 1 forward under Euler dynamics to peak 2.
6. Check which propagated candidates are consistent with peak 2's candidate set.
7. With 2-4 candidates per peak × 2-3 peaks = 4-12 combinations, each requiring a 1D consistency check. Compare: current grid search tests 40,000 candidates.
8. Each additional peak further prunes false matches. With N peaks, the system is massively overdetermined.

### Omega recovery
9. A consistent trajectory through multiple peaks determines both q₀ and ω₀. The omega is implicit in the propagation that links the peaks.

## Why This Is Different from m106 (Pairwise Alignment)

m106 tested pairwise peak alignment using grid-level omega candidates (~1° precision) and failed because delta-q sensitivity amplifies the error over the observation timespan. The IPL approach is fundamentally different:

- **m106:** Start from an omega grid → propagate → check if peaks align. Fails because 1° omega error → 60-90° quaternion error at late peaks.
- **IPL approach:** Start from peak constraints (centroids) → generate attitude candidates → find omega by requiring dynamical consistency. No omega grid needed. Precision comes from the tightness of the IPL centroids, not from grid resolution.

## Precomputed Data Available

All 100 seeds have precomputed IPL data at `data/results/inversion_diagnostics/isoshell_viewer/`:
- Per-epoch: IPL lengths, loop counts, centroids (all loops), active centroid direction, PAB↔centroid angular distance
- Detected local minima with centroid sets
- Interactive HTML viewers for visual inspection

### Census Results (2026-04-12)

Census script: `notebooks/inversion/12_brightness_surface/archive/ipl_census.py`

- **Population coverage is strong:** 83% of seeds have >=2 tight minima (< 5 deg)
- **Discrimination gap confirmed:** At tight IPL minima where truth PAB is between standard normals (~35 deg from nearest), IPL centroids are within 1-3 deg of truth. This is the mechanism behind grid failures in seeds 27, 33.
- **m107 proposed:** IPL centroid cost function to replace standard alignment cost at grid level.

### m107-108 Results (2026-04-13)

IPL centroids as grid cost replacement: **FAILED.** Two root causes:
1. Delta-q amplification: 0.67° grid error -> 30° PAB error at distant epochs
2. Many centroids -> false positives: max-over-12-centroids is non-discriminating

Failure mode reclassification revealed the real bottleneck is ATT_FAIL (correct omega, wrong attitude/phi), not omega finding. See [[att-fail-diagnosis]].

### Revised Direction

The IPL framework's value is in:
1. **Epoch selection** -- identifying when the PAB is near lobe centers (tight IPLs)
2. **Phi discrimination** -- using peak-crossing and plateau data to resolve the twist angle
3. **Continuous tracking** -- plateau phases where PAB circles a lobe provide extended constraints

NOT in replacing the grid cost function.

## Key Questions to Resolve

1. ~~**Peak shape → twist resolution:**~~ m104 refuted direct extraction. Multi-anchor constraint also fails at grid/NM precision (m109 session analysis). **CLOSED.**
2. **Dynamical consistency check cost:** Trivially cheap, but delta-q amplification limits window to ~70 epochs from anchor.
3. **Population coverage:** Census DONE — 83% have ≥2 tight minima, but tight-centroid epochs are essentially alignment-cost equivalents. **See [[isoshell-phi-limits]].**
4. **Phase angle correction:** Not tested, but theoretical analysis suggests it brings isoshell to lo-fi level (still insufficient for phi discrimination).

## Updated Assessment (2026-04-13)

Comprehensive analysis showed IPL centroids ≈ standard normals at tight epochs and non-discriminating at dim epochs. The IPL framework's value for phi discrimination is nil — the zero-phase brightness surface is fundamentally too symmetric. See [[isoshell-phi-limits]].

The framework remains valuable for:
- **Understanding** failure modes (discrimination gap, lobe symmetry)
- **Epoch quality metrics** (IPL set length, loop count)
- **Visualization** (interactive PAB-brightness viewers)

## Revised Value (2026-04-13, second session)

The IPL framework's primary pipeline value is now in **anchor epoch selection** for phi discrimination, not in grid replacement or direct phi discrimination:
- IPL centroid distance to standard normals = anchor alignment error metric
- The optimal anchor epoch is where this distance is minimized
- Population study: 97/100 seeds have sub-0.5° best anchor at some IPL minimum
- This directly informs m112 (best-anchor phi re-sweep)

## Next Steps

1. ~~Census~~ DONE — 83% have ≥2 tight minima
2. ~~Grid cost replacement~~ DEAD END — delta-q amplification + false positives
3. **m112** — use IPL centroid distance as anchor quality metric in post-NM phi re-sweep
