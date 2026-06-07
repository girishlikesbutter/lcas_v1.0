# Stage 1 Findings — 100-Seed Constraint Census

> **Date:** 2026-04-08
> **Script:** `m096_stage1_constraints.py` (identification), `m096_stage1_analysis.py` (analysis)
> **Checkpoint:** `data/results/inversion_diagnostics/m096_stage1/seed_{NNN}.npz` (100 files, 59 keys each)

---

## Population Segmentation

The 100 seeds fall into 5 natural tiers based on what the pipeline can work with:

| Tier | Description | Seeds | Median \|w\| | Example seeds |
|------|-------------|-------|-------------|---------------|
| A | Rich: ≥6 constraints, ≥2 bright | 10 | 1.24 dps | 0, 12, 14, 24, 27, 36, 74, 93 |
| B | Moderate: ≥4 constraints, ≥1 bright | 15 | 1.13 dps | 33, 50, 54, 73, 92, 94, 95 |
| C | Sparse: ≥2 constraints, 0 bright | 55 | 0.71 dps | 2, 8, 17, 20, 21, 55, 56... |
| D | Minimal: 1 constraint only | 10 | 0.24 dps | 1, 5, 9, 16, 30, 42, 87, 89 |
| X | Invalid: <2 specular peaks | 10 | 0.16 dps | 3, 10, 13, 31, 43, 53, 72 |

**The pipeline was designed and validated on Tier A (10% of the dataset). Tier C (55%) is the majority and has zero bright constraints.**

---

## Key Findings

### 1. Tumble rate is the dominant feature

|w| correlates with n_spec (r=+0.74), dt_range (r=+0.74), and n_bright (r=+0.38). Faster tumblers produce more peaks, more specular peaks, and more geometric diversity. There is no hidden feature that compensates for slow tumble rate.

| \|w\| range | Seeds | Invalid | Median spec | Median constraints | Median bright |
|------------|-------|---------|------------|-------------------|--------------|
| 0.0–0.3 | 21 | 7 | 2 | 2 | 0 |
| 0.3–0.6 | 23 | 1 | 5 | 4 | 0 |
| 0.6–0.9 | 12 | 1 | 5 | 4 | 0 |
| 0.9–1.2 | 24 | 0 | 7 | 6 | 0 |
| 1.2–1.6 | 20 | 1 | 10 | 9 | 0 |

### 2. Constraint quality is poor for most seeds

Of 510 total constraints across 90 valid seeds:
- **8.4%** (43) are ±X only (2 allowed normals) — strongest discrimination
- **6.5%** (33) are ±X/±Z (4 allowed normals)
- **25.5%** (130) are bus faces (6 allowed normals)
- **59.6%** (304) allow all 10 normals — essentially no discrimination

The pipeline's alignment cost `(1 - max_dot)^2` with max over 10 normals is nearly always close to zero — it provides no useful gradient.

### 3. 68% of peaks are currently discarded

Of 1883 total peaks across all seeds:
- 606 (32%) are "specular" (mag < 9) — currently used
- 1277 (68%) are non-specular (mag ≥ 9) — currently discarded

The discarded peaks break down as:
- 138 at mag 9–10 (detectable, moderate SNR)
- 145 at mag 10–11
- 252 at mag 11–12
- 742 at mag 12+ (faint, low SNR)

Lowering the specular threshold from 9.0 to 12.0 would make 96/100 seeds viable (vs 90) and roughly double the constraint count per seed.

### 4. Omega magnitude estimation has a long error tail

Median error is 12.8%, but:
- 15 seeds have > 20% error
- 9 seeds have true |w| outside the ±30% grid range
- The worst errors are systematically Z-dominant slow tumblers (peak count overcounts)
- At ±20% grid range: only 72/90 seeds covered
- At ±30% grid range: 81/90 seeds covered

### 5. PAB degeneracy is universal

Every seed has PAB min dot > 0.992. This is structural for GEO satellites observed over 1 hour — the sun-observer geometry barely changes. All alignment constraints point in nearly the same direction regardless of epoch.

This means: alignment-based costs cannot discriminate omega direction using geometric spread alone. The information must come from the DYNAMICS (how the satellite rotates between constraint epochs), not the GEOMETRY (where the constraints point).

### 6. Omega direction affects constraint type

| Dominant axis | Seeds | Median spec | Median bright | Median \|w\| err |
|--------------|-------|------------|--------------|-----------------|
| X | 26 | 8 | 0 | 11.6% |
| Y | 32 | 7 | 0 | 12.7% |
| Z | 32 | 5 | 0 | 15.1% |

Z-dominant tumblers have fewer specular peaks and worse |w| estimation. This makes physical sense: Z-axis rotation brings the large flat bus faces (±Z) into alignment less dramatically than X/Y rotation brings the solar panels/antennas.

### 7. Anchor selection works well

57/90 anchors are bright ±X (mag < 5.9). No anchor is dimmer than 7.21. The SG-smoothed ranking with tie-breaking is stable (only 5 ties in 90 seeds).

### 8. Non-specular peaks are an untapped resource

Seeds with the worst spec/non-spec ratio have 4–7 spec peaks but 20–28 non-spec peaks. These dim peaks carry attitude information (the satellite IS rotating through these orientations) even though the normal ambiguity is higher.

---

## Implications for Pipeline Design

1. **The pipeline must work with 0 bright constraints as the default case.** Tier C (55 seeds) has no ±X-only constraints. The grid search cost function needs to extract information from 6-normal and 10-normal constraints.

2. **Consider lowering the specular threshold to 11 or 12.** This adds ~5 constraints per seed on average, at the cost of more normal ambiguity (all peaks at mag > 7.3 allow all 10 normals). The question is whether more weak constraints beat fewer strong ones.

3. **The omega magnitude grid needs widening.** ±20% misses 18/90 seeds. ±30% misses 9/90. Consider ±40% or adaptive range based on peak-count confidence.

4. **The information is in the dynamics, not the geometry.** PAB degeneracy means all alignment constraints are geometrically equivalent. What distinguishes omega directions is HOW the satellite rotates between constraints — the delta-q propagation already captures this, but the cost function (max-over-normals alignment) doesn't exploit it well.

5. **Tier D and X seeds (20%) may be fundamentally unsolvable** with specular-constraint-based approaches. They need either: longer observation windows, higher SNR, or a completely different approach (e.g., full-curve template matching without peak anchoring).
