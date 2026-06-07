---
title: "s018b — Face-identity tier classifier from peak brightness, post-fix m048"
type: experiment
sources:
  - "results/s018b/face_tiers.npz"
  - "results/s018b/summary.json"
  - "results/s018b_run.log"
related:
  - "[[s018a_bright_band_calibration]]"
  - "[[s001_cost_at_truth_cohort]]"
  - "[[concepts/known_pathologies_to_revalidate]]"
created: 2026-05-02
updated: 2026-05-02
confidence: high
---

## TL;DR

**Per-face spec-event mag distributions are well-separated by face area class — the brightness alone partitions the 10 face groups into 4 clean tiers.** The buggy-era pipeline was using a tiered face-identity classifier that this experiment recovers under correct truth (and sharpens):

| tier | mag_abs band | candidate face groups | shortlist size | purity |
|---|---|---|---|---|
| T1_X | < 6 | ±X | 2 | **99.4%** |
| T2_YZ | 6–7 | ±Y, ±Z | 4 | **97.7%** |
| T3_any | 7–8 | any non-±X (8 faces) | 8 | **100%** |
| T4_D | 8–9 | dishes (±WD, ±ED) | 4 | **97.4%** |

mag_abs ≥ 9 is non-spec (no PAB-alignment guarantee).

**This is the structured IC-generation primitive the survey was missing**: for any peak in an observed LC, magnitude alone shortlists 2–8 body-frame candidate locations for PAB at that epoch. Two well-separated peaks in different tiers ⇒ joint shortlist ≤ T_a × T_b candidate face pairs. Each pair constrains (q0, ω) via the body-frame trajectory of PAB.

**Cohort coverage on pre-detected hi-fi peaks (100 seeds):** 45 with ≥1 T1 (best — 2-face anchor); 73 with ≥2 distinct tiers hit (multi-peak intersection viable); 49 with ≥3 distinct tiers (strong constraint set); 19 with zero classifiable peaks (`mag_abs < 9` empty — Sobol fallback).

**Distance normalisation:** Tier thresholds are stored as `mag_abs` referenced to `D_REF_KM = 38649.2` (cohort median slant range for IS-901 at GEO observed from the m048 ground station). Within m048 the apparent-vs-absolute correction is empirically <0.004 mag, but the absolute formulation makes the table portable to other observer geometries.

## What

Build the formal tiered face-identity classifier the buggy era was using (recovered under correct truth from cached NPZ fields):

  given an observed LC peak with magnitude M, return a candidate face-group set
  {g : g is plausibly responsible for this peak's specular alignment}

The classifier consumes `mag_abs` (distance-normalised), looks up which tier the peak falls in, and returns the candidate face index list. Downstream consumers (phi-sweep IC generator, peak-spacing ω-mag prior) use the candidate face normals to generate geometrically-constrained q0 ICs that the survey's uniform Sobol-Shoemake had no equivalent of.

## How

Pure cohort aggregation on cached NPZ fields, building on s018a's calibration:

1. For each of 100 m048 seeds, load `mag_hifi`, `obs_dist`, `min_ang_dist`, `best_group`, `hifi_peak_epochs`.
2. Convert apparent → absolute magnitude: `mag_abs = mag_hifi − 5 log10(obs_dist / D_REF_KM)` with `D_REF_KM = 38649.2` (cohort median).
3. Pool 100 × 500 = 50,000 epochs.
4. At spec events (`min_ang_dist < 5°`, n=885), compute per-face mag_abs percentiles (p5, p10, p25, p50, p75, p90, p95).
5. Pre-register the 4 tiers based on s018a's per-face distributions and cross-checked against the inverse classifier `P(best_group = g | mag_abs in band, spec5)`.
6. Validate purity: for each tier, compute `P(best_group ∈ candidate_set | mag_abs in band, spec5)`.
7. Per-seed coverage: count pre-detected hi-fi peaks falling in each tier.

No propagation. No surrogate. No rendering. Wall: 0.4 s.

## Result

**Per-face mag_abs distribution at spec events (`min_ang_dist < 5°`):**

| face | area (m²) | n | p10 | p50 | p90 |
|---|---|---|---|---|---|
| +X | 97.3 | 88 | 5.10 | 5.54 | 5.95 |
| -X | 97.3 | 76 | 5.10 | 5.42 | 5.82 |
| +Z | 22.8 | 104 | 6.21 | 6.73 | 7.19 |
| -Z | 22.8 | 113 | 6.22 | 6.76 | 7.36 |
| +Y | 16.9 | 104 | 6.62 | 7.10 | 7.51 |
| -Y | 16.9 | 77 | 6.63 | 7.25 | 7.78 |
| -WD | 9.8 | 87 | 7.51 | 7.82 | 8.20 |
| -ED | 9.8 | 77 | 7.50 | 7.83 | 8.17 |
| +ED | 9.8 | 75 | 7.68 | 7.97 | 8.53 |
| +WD | 9.8 | 84 | 7.69 | 8.05 | 8.92 |

The p10–p90 ranges barely overlap between area classes. Brightest spec mag is monotonic in face area: ±X (97 m²) → ±Z (23 m²) → ±Y (17 m²) → dishes (10 m²).

**Tier validation:** T1 99.4% (n=160), T2 97.7% (n=213), T3 100% (n=384), T4 97.4% (n=117).

**Cohort coverage on hi-fi peaks (100 seeds):**

- 45/100 seeds: ≥1 T1 peak (best — 2-face shortlist available)
- 73/100 seeds: ≥2 distinct tiers hit (joint multi-peak intersection viable)
- 49/100 seeds: ≥3 distinct tiers hit (strong constraint set)
- 63/100 seeds: ≥4 classifiable peaks total
- 19/100 seeds: zero classifiable peaks (mag_abs < 9 empty) — fallback territory

## Why this matters

This recovers the buggy-era brightness-tier face-identity classifier under correct truth and validates it cohort-wide. Three load-bearing implications:

1. **Phi-sweep IC generation has a principled foundation.** For any seed, walk its bright peaks; for each, look up the candidate face set; for each candidate face, phi-sweep around the body-frame face-normal-to-PAB cone (1-DOF rotation). This generates a structured pool of q0 candidates anchored on actual LC features — a fundamentally different IC-generation strategy from the survey's uniform Sobol-Shoemake.

2. **Trajectory-class-specific architecture switching is now operational.** The cohort splits into:
   - **T1-rich seeds (45/100):** 2-face anchor available; phi-sweep generates very narrow IC cone.
   - **Multi-tier seeds (73/100):** can intersect across tiers; even-stronger IC sets.
   - **Sub-3-peak seeds (~37/100):** insufficient classifiable peaks; fall back to Sobol.
   - **Zero-peak seeds (19/100):** no spec structure at all; Sobol-only territory.

3. **The face-identity table is portable across observer geometries** via the `D_REF_KM` distance correction. Future inversions on different orbits or ground stations apply `mag_abs = mag_apparent − 5 log10(d_obs / D_REF_KM)` to incoming peaks before tier lookup.

The shortlist-size column matters for compute budgeting: T1 (2 candidates) is most informative; T3 (8 candidates) is least informative; T4 (4 candidates) is comparable to T2.

## Numbers (canonical)

- Cohort wall: 0.4 s on cached arrays.
- 885 spec events (geometric ground truth).
- Tier purities: T1 99.4%, T2 97.7%, T3 100%, T4 97.4%.
- D_REF_KM = 38649.2 km, distance correction max |Δ| = 0.004 mag within m048.
- Cohort coverage: 73/100 multi-tier, 19/100 zero-classifiable.

## Artefacts

- Script: `experiments/s018b_face_identity_tiers.py` (~330 lines).
- NPZ: `results/s018b/face_tiers.npz` — tier definitions, per-face mag_abs percentiles, inverse classifier matrix, per-seed tier counts, `d_ref_km`.
- JSON: `results/s018b/summary.json` — tier definitions, validation purities, per-face stats, cohort coverage, distance-normalisation formula.
- Plots: `results/s018b/per_face_mag_cdf.png` (CDF per face), `results/s018b/inverse_classifier_heatmap.png` (P(face|mag-band)), `results/s018b/per_seed_tier_coverage.png` (stacked bars).
- Run log: `results/s018b_run.log`.

## Out of scope

- Implementing the phi-sweep IC generator that consumes this tier table (intended as s018c).
- Bright-peak ω-mag prior (s007 closed full-LC LS-peak spacing; bright-peak-only spacing remains untested — would belong in a parallel s018c-class experiment).
- Per-phase-angle re-binning of the tier table. Spec events at different phase angles have different specular-lobe geometries; the cohort marginalises over PA. If downstream architecture wants tighter per-PA bands, this can be added.
- Validation that the tier classifier's q0 IC pool actually contains truth on a held-out test seed. That is the s018c+ inversion validation.

## Cross-references

- [[s018a_bright_band_calibration]] — calibrated bright threshold (mag_abs < 8) and per-group spec frequencies.
- [[s001_cost_at_truth_cohort]] — surrogate-MSE cost-at-truth honesty cohort-wide.
- [[concepts/known_pathologies_to_revalidate]] — buggy-era `mag < 11` filter was inappropriate under correct truth.
- m103 phi-sweep (parent project, contraband) — the buggy-era IC generator this experiment provides the calibrated foundation to rebuild.
- m096 buggy-era 100-seed census (parent project wiki, frozen) — the "87% lack ±X" claim now formally softened to 56% under correct truth.
