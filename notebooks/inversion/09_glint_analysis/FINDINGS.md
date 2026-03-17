# Series 09 — Glint Analysis: Findings

## Overview

This series exploits specular glint physics for attitude inversion. A glint occurs when a facet normal aligns with the Phase Angle Bisector (PAB = normalize(k1 + k2)). The series progressed through three phases:

1. **Series 09a (micro34-40):** Established that brightness peaks are specular glints, quantified PAB alignment, tested PAB-circle seeding, demonstrated component identification from magnitude.
2. **Series 09b (micro44-48):** Deep classification — built visualization tools, generated 200 trajectory datasets at realistic omega, quantified specular vs diffuse thresholds, built ML classifier and geometric filter for inversion.

---

## The Specular Threshold

The single most important finding: **mag < 6.0 = 100% specular purity.**

Every brightness peak brighter than magnitude 6.0 is caused by specular PAB-normal alignment (< 5°). Zero exceptions across ~6000 peaks, 200 trajectories, all phase angles.

| Purity | Threshold | Peaks | Contaminants |
|--------|-----------|-------|-------------|
| 90% | mag < 6.16 | 127 | 12 |
| 95% | mag < 6.05 | 108 | 3 |
| 100% | mag < 6.02 | 104 | 0 |

Above mag 9.0: zero specular peaks. The transition zone (mag 6-9) is mixed.

This threshold is invariant to equatorial phase angle (tested 0-97°).

---

## Normal Group Identification from Brightness

For specular peaks, the dominant normal group maps to face area via brightness:

| Group | Area (m²) | Median mag at glint | Identification |
|-------|-----------|-------------------|----------------|
| ±X (Bus) | 97.3 | 5.5 | Brightest — unambiguous |
| ±Y (Bus) | 16.9 | 6.9 | Medium |
| ±Z (All) | 22.8 | 6.7 | Medium |
| ±WD/ED (Dishes) | 9.8 | 7.6-8.0 | Dimmest specular |

Distinguishing +X from -X within a single glint is hard (identical area/BRDF), but:
- They alternate perfectly (r = -1.000 correlation in alignment curves)
- +X glints at larger equatorial phase angles than -X (median 51° vs 35°)
- The temporal alternation pattern constrains which is which

---

## ML Classification (LC Morphology Only)

Gradient Boosting classifier, 14 features, 5-fold CV:

| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Logistic Regression | 0.823 | 0.702 | 0.995 |
| Random Forest | 0.891 | 0.876 | 0.908 |
| **Gradient Boosting** | **0.898** | **0.887** | **0.911** |

Feature importance (permutation):
1. **peak_mag: 0.854** — overwhelmingly dominant
2. **max_slope: 0.154** — the only other feature that matters
3. Everything else: < 0.02

Confusion matrix (GB): TN=2507, FP=49, FN=37, TP=376.

---

## Geometric Glint Filter (`src/inversion/glint_filter.py`)

Fast candidate rejection for inversion. Given a candidate (q0, omega0):
1. Propagate attitude (77ms)
2. Find all alignment dips < threshold
3. Check if predicted glints match observed LC peaks

**Scoring: precision = n_confirmed / n_predicted**
- "When this candidate predicts a glint, is there actually one in the LC?"

| Threshold | True trajectory | Random trajectory |
|-----------|----------------|-------------------|
| 4° | prec=1.00 (8/8) | prec=0.50 (2/4) |
| 10° | prec=0.94 (15/16) | prec=0.53 (9/17) |
| 15° | prec=0.91 (20/22) | prec=0.45 (13/29) |

Basin shape: clear gradient for attitude, visible for omega magnitude, noisy for omega direction.

---

## Phase Angle Effects

Equatorial phase angle = angle between sun and observer projected onto the equatorial plane. Reaches 0° at ~09:57 UTC for this geometry. Observable window: 08:30-16:40 UTC.

- **Does NOT improve specular/diffuse separation.** Optimal phase correction coefficient = 0.000. The magnitude gap (~6.5 mag) is constant across all phase angles.
- **DOES affect which group glints.** Different normal groups activate preferentially at different equatorial phase angles. This is a potential feature for group identification in the inversion pipeline.
- **Excess brightness** (peak relative to local running median) generalises better than raw magnitude across varied observation conditions.

---

## Opposite-Pair Anti-Correlation

All opposite-face pairs are perfectly anti-correlated in their alignment curves:

| Pair | Pearson r |
|------|-----------|
| +X / -X | -1.000 |
| +Y / -Y | -1.000 |
| +Z / -Z | -1.000 |
| +WD / -WD | -1.000 |
| +ED / -ED | -1.000 |

This is a geometric identity: if n points toward the PAB, -n points away. The +X dominance fraction across trajectories is 0.50 ± 0.12 — they alternate symmetrically.

---

## Diffuse Bump Characterisation

Diffuse bumps (alignment > 5°) are NOT a mixture of many faces:
- 79% have one group contributing > 70% of flux
- Always the large faces (±X, ±Z dominate 72% of diffuse peaks)
- Same mechanism as specular (one face drives brightness), just the broad Lambertian component at 20-40° alignment instead of sharp specular at < 5°
- Dishes almost never dominate diffuse bumps (< 2% each)

---

## Realistic Omega Range

Literature survey (Binz et al. 2014, AMOS) shows retired GEO satellites tumble well below 1°/s. The original experimental range [0.5, 5.0] deg/s was unrealistically fast.

Corrected to [0.1, 1.5] deg/s for micro46/48 datasets. Key implications:
- Slower tumblers have fewer, more widely spaced glints (more distinctive for fingerprinting)
- Faster (unrealistic) tumblers produce too many glints, making the glint filter basin flatter
- 62% of realistic trajectories have ≥1 guaranteed specular anchor (mag < 6)

---

## Datasets

| Dataset | Location | Trajectories | Omega range | Start time | Phase angles | Size |
|---------|----------|-------------|-------------|------------|-------------|------|
| micro46 | `micro46_trajectories/` | 100 | [0.11, 1.48] | Fixed 10:00 UTC | ~9-17° | 13.5 MB |
| micro48 | `micro48_trajectories/` | 100 | [0.11, 1.48] | Random 08:35-15:35 | 0-97° | 19.0 MB |

Both store per trajectory: quaternions, k1/k2, PAB (body + J2000 + equatorial in micro48), phase angles, hi-fi + lo-fi magnitudes, per-group flux + fractional flux, alignment angles, peak detection.

---

## File Inventory

| File | Purpose |
|------|---------|
| `micro34_pab_alignment.py` | PAB alignment diagnostic (oracle) |
| `micro35_multi_trajectory_pab.py` | 30-trajectory robustness |
| `micro36_pab_candidate_filter.py` | PAB as iso-brightness post-filter |
| `micro37_brdf_glint_profile.py` | BRDF specular lobe characterisation |
| `micro38_pab_seeded_candidates.py` | PAB-circle seeding (failed) |
| `micro39_glint_identification.py` | Component ID from glint shape |
| `micro41_glint_classification.py` | 30-trajectory glint classification |
| `micro42_glint_anchored_nlp.py` | Glint-anchored NLP (exploration) |
| `micro42b_anchor_hypothesis_sweep.py` | Anchor hypothesis sweep |
| `micro43_focused_pab_seeding.py` | Focused PAB seeding |
| `micro44_normal_sphere.py` | **Normal-family sphere visualization (multi-mode)** |
| `micro45_glint_filter_basin.py` | Glint filter basin shape analysis |
| `micro46_generate_trajectories.py` | 100 fixed-time trajectory dataset |
| `micro47_glint_statistics.py` | Exhaustive glint statistics + ML |
| `micro48_generate_trajectories_v2.py` | 100 varied-time trajectory dataset |
| `src/inversion/glint_filter.py` | Geometric filter module |
