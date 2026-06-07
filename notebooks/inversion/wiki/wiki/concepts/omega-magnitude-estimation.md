---
title: "Omega Magnitude Estimation"
type: concept
sources:
  - "raw/inversion_diagnostics/"
related:
  - "[[glint-physics]]"
  - "[[grid-search]]"
  - "[[m138_seed47_lombscargle_bracket]]"
  - "[[constraint-poor-regime]]"
created: 2026-03-19
updated: 2026-04-29
confidence: medium
---

# Omega Magnitude Estimation

The spin rate |omega| can be estimated from observable light curve features before running the full inversion. **Status (2026-04-29):** point estimators (peak-count, LS-top1) all fail in the constraint-poor regime; spectrum-spanning strategies (LS-bracket, harmonic-division) cover truth across the failure cohort.

## Peak-Count Regression (m052)

A simple linear fit to the number of brightness peaks:

```
|omega| = 0.040 * n_peaks + 0.042  (deg/s)
```

Population (m052, 100 trajectories): Pearson ρ = 0.954, median error 13.2%, within ±20% for 75/100. Works because faster-spinning satellites sweep more facet normals through PAB alignment per unit time, producing more [[glint-physics|glints]].

**Limitation discovered 2026-04-29 ([[m138_seed47_lombscargle_bracket]]):** in the constraint-poor regime, peak-count systematically reads the k-th harmonic of the rotation rate, where k is set by the geometry's rotational symmetry (not noise). On the m138 failure cohort {47, 91, 51, 79, 84, 89}:

| seed | truth \|ω\| | peak-count est | ratio |
|---:|---:|---:|---:|
| 47 | 0.00800 | 0.01571 | 1.96× |
| 91 | 0.02489 | 0.04538 | 1.82× |
| 51 | 0.01616 | 0.02269 | 1.40× |
| 79 | 0.00211 | 0.01047 | **4.97×** |
| 84 | 0.00884 | 0.01222 | 1.38× |
| 89 | 0.00419 | 0.00524 | 1.25× |

Ratios cluster around 2× but are not exactly 2× — they're whatever k-th harmonic the LC's facet symmetry produces. m052's 13.2% median error masks this regime because the population averages over harmonic-clean and harmonic-confused seeds.

## Alternatives Tested

| Method | rho | Median error | Verdict |
|--------|-----|-------------|---------|
| Peak count regression | 0.954 | 13.2% | Best point estimator (population); fails in constraint-poor regime |
| Lomb-Scargle top-1 peak | 0.740 | 31.1% | Weak point estimator (m052) — confirmed on 6 failure-cohort seeds |
| Glint recurrence interval | 0.146 | -- | Useless |

Glint recurrence is useless because polhode geometry breaks periodicity: the spin axis precesses around the angular momentum vector, so glint intervals are irregular even at constant |omega|.

## Spectrum-Spanning Strategies (NEW, 2026-04-29)

m052's "LS is weak" verdict was about LS reduced to a single number (top-1 peak frequency). Treating LS as a *bracket* or *multi-hypothesis* uses information from ALL significant peaks. Probed on 6 failure-cohort seeds (`m138_ls_bracket_probe.py`):

### LS-bracket
Find all LS peaks with power ≥ 0.1 × peak_max. Build a single |ω|-grid spanning `[0.5 × min_peak, 2.0 × max_peak]` at 5% step. Grid size 75-92 mags.

### LS-multi-hypothesis
Treat each significant LS peak as a candidate base |ω|; build a `[0.3, 3.0] × base` grid around each. Total grid = union of K such grids. Grid size 100-440.

### LS-harmonic-division
For LS top-1 peak f_LS, hypothesise that the observed peak is the k-th harmonic of truth, k ∈ {1, 2, 3, 4}. Test 4 candidate bases `f_LS / k` each with `[0.3, 3.0]` × 20 mags. Grid size ~80 (with overlap collapse).

### Coverage of truth across failure cohort

| seed | pc baseline (20) | LS-top1 (20) | **Bracket** (~80) | **Multi-hyp** (100-440) | **Harmonic-div** (~80) |
|---:|---|---|---|---|---|
| 47 | in grid, 4.36% off | NOT in, 28.6% | **in, 2.15%** | **in, 0.18%** | **in, 0.09%** |
| 91 | in grid, 0.24% | NOT in, 101% | in, 1.12% | in, 0.41% | in, 0.39% |
| 51 | in grid, 1.63% | NOT in, 12.2% | in, 1.00% | in, 0.67% | in, 1.43% |
| 79 | **NOT in, 49.1%** | NOT in, 268% | in, 0.50% | in, 1.03% | in, 3.76% |
| 84 | in grid, 3.18% | in, 2.70% | in, 1.16% | in, 0.25% | in, 1.89% |
| 89 | in grid, 1.05% | NOT in, 135% | in, 1.50% | in, 1.13% | in, 0.26% |

Bracket and harmonic-division cover truth on every seed at sub-2.2% nearest grid offset. **Harmonic-division is the cheapest at fixed grid size 80; on seed 47 it places truth within 0.09% of a grid point.**

## Why the spectrum-spanning approach works

The LC of a tumbling rigid body has rich harmonic content by construction (multiple facets glinting at different phase angles, plus shadowing). Truth's frequency is one peak among several, often not the dominant one. Point estimators discard this structure; bracket/multi-hyp/harmonic-division retain it.

The LC's harmonic content is a **fingerprint of the geometry**, not noise — which is why no smoothing or noise-floor argument can help. The fix is to test all harmonic hypotheses, not to find a "principled" frequency-picker.

## Role in Pipeline

The estimator sets the N_MAGS search range in [[grid-search]] and the |ω|-mag grid in `m138_isoshell_h1.estimate_omega_mag_grid` (the [[surrogate-attitude-isoshell]] H1 cost). With pc-only baseline at 20 mags, truth misses the grid on seed 79 entirely and barely-clings on seed 47. With bracket / harmonic-division at ~80 mags, truth is in-grid with sub-2% step on all 6 failure-cohort seeds.

**Open question (as of 2026-04-29):** whether moving truth into the grid is sufficient to recover seed 47 in H1, or whether the grid-coverage fix exposes a separate cost-shape bias toward high-|ω| candidates ([[m138_seed47_lombscargle_bracket]] Finding B). Pending: H1 re-run on seed 47 with harmonic-division grid.
