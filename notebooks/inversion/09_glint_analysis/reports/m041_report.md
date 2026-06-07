# Micro-41: Comprehensive Glint-to-Normal Classification Across 30 Trajectories

## Question

How reliably can we identify which body-frame normal group is responsible for each glint, across diverse trajectories (random q0, random omega)? Specifically: (a) magnitude-based classification accuracy, (b) failure modes, (c) confidence metrics.

## Method

- 30 trajectories with random q0 (uniform SO(3)) and omega0 (random direction, magnitude uniform in [0.5, 5.0] deg/s), seeds 0-29.
- For each trajectory: propagate attitude, compute hi-fi LC with animate=True, aggregate per-facet flux by 14 unique normal groups.
- Detect bright peaks (mag < 9, argrelmin order=5).
- Oracle label: dominant normal group (argmax of fractional flux).
- Rule-based classifier: 3 magnitude bands (mag < 7.5, 7.5-8.5, 8.5-9.0). Map each band to the oracle group that appears most often in that band (majority vote).
- Confidence metric: distance from nearest band boundary, normalized by band width.
- Overlap flag: second-largest group has frac_flux > 0.15.
- Weak-glint flag: dominant frac_flux < 0.77.

## Results

### Overall statistics

| Metric | Value |
|--------|-------|
| Total trajectories | 30 |
| Total bright peaks | 439 |
| Overall accuracy | 96/439 = 21.9% |
| Overlap rate | 34/439 = 7.7% |
| Weak-glint rate | 24/439 = 5.5% |
| Runtime | 2778s (46.3 min) |

### Rule-class-to-oracle mapping (majority vote)

| Band | Magnitude range | Mapped to | Components | Area |
|------|----------------|-----------|------------|------|
| B0 | mag < 7.5 | Group 0 | Bus, SP_North, SP_South | 97.30 |
| B1 | 7.5 - 8.5 | Group 1 | AD_East | 9.80 |
| B2 | 8.5 - 9.0 | Group 1 | AD_East | 9.80 |

Bands B1 and B2 both mapped to the same oracle group (Group 1, AD_East), meaning the rule-based classifier effectively has only 2 classes, not 3.

### Accuracy by oracle group

| Group | Components | N peaks | N correct | Accuracy |
|-------|-----------|---------|-----------|----------|
| G0 | Bus, SP_North, SP_South | 80 | 55 | 68.8% |
| G1 | AD_East | 45 | 41 | 91.1% |
| G2 | AD_West | 34 | 0 | 0.0% |
| G5 | Bus, SP_North, SP_South | 41 | 0 | 0.0% |
| G6 | AD_East, AD_West, Bus, SP_North, SP_South | 57 | 0 | 0.0% |
| G7 | AD_East, AD_West, Bus, SP_North, SP_South | 43 | 0 | 0.0% |
| G8 | Bus, SP_North, SP_South | 30 | 0 | 0.0% |
| G11 | AD_West | 26 | 0 | 0.0% |
| G12 | AD_East | 22 | 0 | 0.0% |
| G13 | Bus, SP_North, SP_South | 61 | 0 | 0.0% |

10 distinct oracle groups produce glints. The rule-based classifier only ever maps to 2 of them (G0 and G1), so the remaining 8 groups are always misclassified.

### Accuracy by omega magnitude

| Omega bin (deg/s) | N peaks | N correct | Accuracy |
|-------------------|---------|-----------|----------|
| 0.5 - 1.5 | 73 | 16 | 21.9% |
| 1.5 - 3.0 | 94 | 17 | 18.1% |
| 3.0 - 5.0 | 272 | 63 | 23.2% |

No significant dependence on rotation speed. Accuracy is uniformly poor across all omega bins.

### Confusion matrix

```
  Group              Components       B0<7.5  B1<8.5  B2<9.0  Total
  G0     Bus,SP_North,SP_South          55      19       6      80
  G1                   AD_East           4      30      11      45
  G2                   AD_West           0      27       7      34
  G5     Bus,SP_North,SP_South          19      12      10      41
  G6     AD_East,AD_West,Bus,...        34      18       5      57
  G7     AD_East,AD_West,Bus,...        30       9       4      43
  G8     Bus,SP_North,SP_South          17       9       4      30
  G11                  AD_West           0      20       6      26
  G12                  AD_East           0      17       5      22
  G13    Bus,SP_North,SP_South          47      11       3      61
```

### Confidence distribution

| Prediction type | Count | Mean confidence | Median | Std |
|----------------|-------|----------------|--------|-----|
| Correct | 96 | 0.472 | 0.410 | 0.320 |
| Incorrect | 343 | 0.326 | 0.311 | 0.226 |

Correct predictions have slightly higher confidence on average, but the distributions overlap heavily. Confidence is not a reliable discriminator.

### Overlap and weak-glint rates

- Overlap rate (2nd group > 15% of flux): 7.7% overall. Tends to be higher at larger omega (up to 22% at omega = 4.13 dps, seed 14).
- Weak-glint rate (dominant group < 77% of flux): 5.5% overall. Correlated with overlap. Most trajectories have 0-2 weak glints.

## Key Findings

1. **The 3-band magnitude rule-based classifier fails badly (21.9% accuracy).** This is because glints from 10 distinct normal groups span overlapping magnitude ranges. Magnitude alone cannot distinguish them.

2. **The fundamental problem is degeneracy in magnitude bands.** Multiple normal groups with very different orientations produce glints in the same magnitude range:
   - Band B0 (mag < 7.5): Groups 0, 5, 6, 7, 8, 13 all produce bright glints. These are the large-area z-faces (G0, G13), the x/y-faces (G5, G8), and the mixed groups (G6, G7).
   - Band B1 (7.5-8.5): Groups 1, 2, 11, 12 (antenna dishes from both East and West) plus Bus groups all appear here.
   - Only Group 0 (68.8%) and Group 1 (91.1%) are classified reasonably, because they happen to be the plurality in their respective bands.

3. **Glint flux is dominated by a single group (94.5% of peaks have dom_frac > 0.77).** The physics is clean -- each glint IS caused by one normal group. The problem is purely that peak magnitude is a poor feature for distinguishing WHICH group.

4. **Overlap and weak-glint cases are relatively rare (7.7% and 5.5%).** These are not the main failure mode. The classifier fails on clean, well-separated glints.

5. **No dependence on rotation speed.** Accuracy is ~20% regardless of omega magnitude (0.5-5.0 deg/s).

6. **Confidence metric has weak discriminative power.** Mean confidence for correct predictions (0.47) is only slightly higher than for incorrect (0.33). The classifier is not "confidently wrong" -- it is simply using the wrong feature space.

## Implications for Inversion

- Magnitude-based classification cannot solve the normal-group identification problem across diverse trajectories. A 3-band magnitude classifier achieves only 22% accuracy when 10 groups produce glints.
- The information needed to distinguish groups lies in temporal features (recurrence period, glint width) or geometric features (PAB alignment), not in peak brightness alone.
- Future work should explore the recurrence-based approach (m039/40 direction) or a forward-model approach that tests each group hypothesis against the observed LC shape.

## Files

- Script: `notebooks/inversion/09_glint_analysis/m041_glint_classification.py`
- Results JSON: `data/results/inversion_diagnostics/m041_glint_classification.json`
- Results NPZ: `data/results/inversion_diagnostics/m041_glint_classification.npz`
- Plot PNG: `data/results/inversion_diagnostics/m041_glint_classification.png`
