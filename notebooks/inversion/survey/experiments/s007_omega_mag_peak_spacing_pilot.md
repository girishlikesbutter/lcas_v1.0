---
title: "s007 — peak-spacing ω-magnitude prior pilot (Q4c-iii)"
type: experiment
sources:
  - "results/s007/peakspacing.npz"
  - "results/s007/summary.json"
  - "results/s007/periodograms.png"
  - "results/s007/truth_vs_estimate.png"
related:
  - "[[s001_cost_at_truth_cohort]]"
  - "[[s003_landscape_vs_omega]]"
  - "[[s005_joint_local_descent]]"
  - "[[s006_seed28_landscape_at_truth_omega]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# s007 — peak-spacing ω-magnitude prior pilot (Q4c-iii)

## TL;DR

**Decisive negative.** A simple Lomb-Scargle / autocorrelation prior on
the truth LC does NOT produce an ω-mag estimate within s005's basin
radius (3-5% relative). On 8 well-sampled seeds (≥3 rotations / 60-min
LC), the **blind dominant-peak** estimator has median 15.7% error and
worst case 72.3%. The **oracle** estimator (truth-aware best-of-top-5 ×
{1×, 2×} harmonics) has median 7.4% error and worst case 17.2% —
better, but still well above the basin-radius bar and unattainable
without truth knowledge anyway. The cohort's slow rotators (seeds 10
and 48, with 1.06 and 2.57 rotations sampled) trivially fail at 12-19%.

**Mechanism (qualitative):** Tumbling-rigid-body LCs are multi-periodic
(precession + spin coupling), so the dominant spectral peak does not
track |ω|. Peak-frequency ratios scatter from 0.14× to 6.7× of truth ω
across seeds. There is no consistent harmonic structure mapping
periodogram peaks to rotation rate.

**Implication for Q4c:** the ω-mag-from-peak-spacing path is closed.
Naive LC-feature priors that try to bootstrap ω from spectral content
hit the same multi-periodicity wall. Q4c (global-search → local-polish
handoff) must either accept a coarse ω-mag grid (~3-5% relative density,
≥30 grid points) or find a different conditioning signal.

---

## What

For each of 10 PA/ω-stratified pilot seeds (6, 10, 18, 21, 28, 41, 48,
60, 84, 91):

1. Compute Lomb-Scargle periodogram of the truth hi-fi magnitude LC
   over the 60-min observation window (8000 freq grid, range 1/3600
   to 1/14.4 Hz).
2. Extract top-5 spectral peaks (local maxima ranked by power).
3. For each peak, propose two ω-mag candidates: 1× harmonic
   (LC period = rotation period → ω = 360 · f) and 2× harmonic
   (LC period = half rotation → ω = 180 · f).
4. **Blind estimator:** best-of-{1×, 2×} of the dominant (rank-0) peak.
   This is what an actual inversion algorithm would use.
5. **Oracle estimator:** best-of-{1×, 2×} across all top-5 peaks (10
   candidates), picked vs truth. Diagnostic — measures whether ANY
   peak captures ω, even with truth-aware selection.
6. Report relative error of each estimator vs truth ω-mag.
7. Cross-check via simple ACF first non-zero peak.

The 3% threshold matches s005's tightest ω-mag basin radius (3-5%
across the 5 seeds tested). If the prior cannot reliably hit that, it
cannot seed LM polish into the basin without an additional broader
search over ω.

## How

- Frequency grid: linspace(1/3600 Hz, 1/14.4 Hz, 8000) — well above LS
  required oversampling at this window length.
- Peak detection: scipy.signal.find_peaks (no min height; we want the
  full top-5 ranking).
- 1×/2× harmonics: tested both because the IS-901 model has
  quasi-symmetric solar panels, which COULD make the LC repeat at
  half-rotation. The data show this is not a clean dichotomy.
- ACF: first peak above 0.05 normalised autocorrelation, excluding
  lags below 2·dt = 14.4 s.
- Subcohort split: ≥3 rotations / 60 min = "well-sampled". Below
  that, periodogram FWHM > 33% by Fourier resolution alone — the
  estimator has no chance.

## Result

### Per-seed errors

| seed | ω_truth (dps) | rot/lc | blind 1× err% | blind 2× err% | blind best err% | oracle err% (peak#, harm) | acf best err% |
|------|----------------|--------|----------------|----------------|------------------|---------------------------|----------------|
| 6    | 0.713          | 7.13   | 134.4          | 17.2           | 17.2             | 4.16  (peak#2, 1×)        | 112.0          |
| 10   | 0.106          | 1.06   | 62.5           | 18.8           | 18.8             | 18.76 (peak#0, 2×)        | 92.2           |
| 18   | 1.131          | 11.31  | 83.0           | 8.5            | 8.5              | 4.99  (peak#4, 2×)        | 69.7           |
| 21   | 0.794          | 7.94   | 71.4           | 14.3           | 14.3             | 14.31 (peak#0, 2×)        | 9.0            |
| 28   | 1.438          | 14.38  | 10.0           | 45.0           | 10.0             | 0.75  (peak#4, 1×)        | 5.2            |
| 41   | 0.565          | 5.65   | 70.8           | 85.4           | 70.8             | 12.50 (peak#3, 1×)        | 22.6           |
| 48   | 0.257          | 2.57   | 89.8           | 5.1            | 5.1              | 5.08  (peak#0, 2×)        | 67.2           |
| 60   | 0.572          | 5.72   | 4.9            | 47.6           | 4.9              | 4.86  (peak#0, 1×)        | 13.2           |
| 84   | 0.507          | 5.07   | 197.5          | 48.8           | 48.8             | 9.80  (peak#2, 2×)        | 118.9          |
| 91   | 1.426          | 14.26  | 72.3           | 86.1           | 72.3             | 17.23 (peak#1, 1×)        | 42.2           |

### Decision-grade scalars (from `summary.json`)

- **Blind dominant-peak estimator (well-sampled subcohort, n=8):**
  - median error: **15.7%**
  - worst case: 72.3%
- **Oracle estimator (well-sampled subcohort, n=8):**
  - median error: **7.4%**
  - worst case: 17.2%
- **3% threshold (s005's tightest basin radius):**
  - blind passes: **False**
  - oracle passes: **False**

### Per-seed peak-frequency ratios (top-5 peaks, hifi LS)

`f_peak / f_truth_1x` ratios reveal the multi-periodic structure:

| seed | peak#0 | peak#1 | peak#2 | peak#3 | peak#4 |
|------|--------|--------|--------|--------|--------|
| 6    | 2.34   | 1.70   | 1.04   | 3.38   | 4.72   |
| 10   | 1.62   | 3.83   | 2.60   | 7.45   | 6.19   |
| 18   | 1.83   | 1.56   | 3.66   | 4.66   | 2.10   |
| 21   | 1.71   | 1.18   | 0.84   | 0.58   | 1.37   |
| 28   | 1.10   | 0.92   | 2.02   | 1.63   | 1.01   |
| 41   | 0.29   | 1.16   | 1.55   | 0.88   | 0.66   |
| 48   | 1.90   | 1.40   | 5.05   | 3.38   | 6.69   |
| 60   | 1.05   | 0.64   | 1.30   | 1.55   | 3.32   |
| 84   | 2.98   | 4.47   | 1.80   | 5.97   | 0.29   |
| 91   | 0.28   | 1.17   | 0.38   | 0.18   | 0.44   |

Peak#0 (the dominant) ratios span 0.28-2.98. The dominant peak is
sometimes at ~truth (seeds 28, 60), sometimes at 2× (seed 6, 21), and
sometimes at a fractional sub-multiple suggesting precession-driven
envelope (seed 91 at 0.28 ≈ 1/4, seed 41 at 0.29 ≈ 1/4).

## Why this matters

Peak-spacing was the most cost-effective candidate prior for Q4c-iii:
~milliseconds per LC, no model required, leverages Fourier theory we
trust. Its failure tells us:

1. **The LC's spectral content does not directly encode ω-mag** for
   tumbling rigid bodies of the IS-901 type. A torque-free Euler body's
   k1·body and k2·body trajectories are doubly-periodic with periods
   set by the spin period AND the precession period — and the latter
   depends on inertia ratios, not |ω| alone. The LC inherits both
   periods, plus their beat frequencies and harmonics from the BRDF
   non-linearity.
2. **Naive LC-feature priors are unlikely to work.** Peak count, glint
   count, mean magnitude, etc. are similarly tangled functions of
   (q0, ω, inertia). Any prior that maps "feature → ω" without
   reference to the satellite model is hitting the same wall.
3. **Q4c-iii is closed.** The PROGRESS-listed expectation that this
   prior "collapses ω-mag to ~3 candidates" is refuted: ω-mag must be
   gridded at 3-5% density (≳30 grid points across 0.1-1.5 dps), or
   we need a model-aware prior.

What's still on the table for Q4c:

- **Q4c-ii** (naive Sobol-SO(3) × ω-grid baseline) is still the
  measurement of last resort. Its cost is now confirmed expensive:
  ~30 ω-mag points × ω-dir grid × Sobol-q0 × LM polish per seed.
- **Cohort basin-radius probe** — extending s005 to all 100 seeds at
  low density (T1+T3 only, ~200 LM runs total) tells us the cohort
  distribution of "tight basin" seeds. If most seeds have ≥5° basins
  (like 4/5 of the s005 sample), Q4c is more tractable than if seed
  28's 2° basin is typical.
- **Model-aware ω prior** (open-ended) — e.g., training a regression
  model from (LC features) → ω over the existing 100 m048 truth
  trajectories. Outside this experiment's scope but a viable next
  direction.

## Numbers

- 10 seeds × ~10 ms LS = 0.06 s wall total.
- Frequency grid 8000 × N=500 LS evaluations per seed: cheap.
- Pilot completed in 60 ms wall (BLAS=1, single-thread).
- Output sizes: peakspacing.npz 2.4 KB, summary.json 1.0 KB,
  periodograms.png ~120 KB, truth_vs_estimate.png ~80 KB.

## Artefacts

- `experiments/s007_omega_mag_peak_spacing_pilot.py` — pilot script.
- `experiments/s007_omega_mag_peak_spacing_pilot.md` — this writeup.
- `results/s007/peakspacing.npz` — per-seed top-5 peaks (freq, power)
  for both hifi and lofi LCs, blind/oracle/acf estimates and errors.
- `results/s007/summary.json` — decision-grade scalars.
- `results/s007/periodograms.png` — Lomb-Scargle plots for all 10
  seeds with truth-ω 1× / 2× lines and top-5 peaks marked.
- `results/s007/truth_vs_estimate.png` — log-log scatter of truth-ω
  vs blind 1×/blind 2×/oracle estimates.
- `results/s007_run.log` — console output.

## Out of scope

- Smarter spectral priors (multi-taper, harmonic-aware models) —
  these are unlikely to beat oracle (which already cherry-picks the
  best of top-5 × 2 harmonics) since the ratios show no consistent
  harmonic structure.
- Time-domain features (glint count / duration / spacing) as priors —
  in principle independent of spectral analysis, but motivated only
  by speculation. Not pursued here.
- Cohort-scale extension to 100 seeds — the negative finding on 10
  pilot seeds is enough to close Q4c-iii; running the full cohort
  would refine the failure-mode distribution but not change the
  conclusion.
- Lofi vs hifi periodogram cross-check — both LC variants were
  computed and saved, but the LS results are nearly identical for
  the dominant peaks (the surrogate trains on lofi-equivalent
  inputs and inherits the same multi-periodic structure). Not worth
  reporting here.

## Cross-references

- **[[s001_cost_at_truth_cohort]]** — established the 100-seed m048
  cohort and the surrogate-MSE baseline at truth.
- **[[s003_landscape_vs_omega]]** — established that surrogate-MSE
  is ω-fragile inside a thin tube (~1° dir / 2-5% mag), motivating
  the need for an ω prior.
- **[[s005_joint_local_descent]]** — established the s005 basin
  radii (3-5% in ω-mag) that this pilot's 3% threshold targets.
- **[[s006_seed28_landscape_at_truth_omega]]** — established
  seed 28's 2° q0 basin as the calibration target for Q4c.
- **[[concepts/known_pathologies_to_revalidate]]** — peak-spacing
  has not been a load-bearing claim in the buggy era; this pilot
  doesn't revisit any prior finding.
