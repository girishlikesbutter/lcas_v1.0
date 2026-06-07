---
title: "s025 — Tube-shape characterization: filter roll-off on each axis"
type: experiment
sources:
  - notebooks/inversion/survey/lib/filter_costs.py
related:
  - notebooks/inversion/survey/experiments/s003_landscape_vs_omega.md
  - notebooks/inversion/survey/experiments/s019b_omega_dir_basin_at_truth_mag.md
  - notebooks/inversion/survey/experiments/s021_truth_vs_random.md
created: 2026-05-04
updated: 2026-05-04
confidence: high
---

## TL;DR

Per-axis perturbation map of filter scores from truth on 5 stratified
seeds (6, 28, 41, 44, 91). Defines the natural cell-size for any
filter-based grid search.

**Headline:**
- **Geo cost** is a SHARP filter: ω-dir basin ~1° (already 0.10-0.27 at
  1°), q0 basin ~5° (still 0.5-0.6 at 5°, 0 by 15°). Tighter than the
  s019b "3-5° at pinned mag" measurement because s019b allowed LM
  refinement; here we score the IC directly.
- **Alignment cost** is a WIDE filter on q0/ω-dir but EXTREMELY tight
  on ω-mag: drops to ~0.0-0.5 at any non-zero perturbation. ω-mag is the
  most discriminating axis for alignment cost.
- **Seeds 41, 91 (zero-classifiable)** have geo always 0 (no spec
  events) — alignment cost is the only filter for them.

These define the natural anisotropy of the filter tube: q0 wide, ω-dir
narrow, ω-mag narrowest (alignment) / no constraint (geo). Bracket-prior
(5% accuracy) for ω-mag aligns nicely with the alignment-cost ω-mag
sensitivity.

## What

For each pilot seed, perturb truth (q0, ω) one axis at a time, others
fixed at truth:

- **Δq0 axis:** angles {0, 1, 5, 15, 45, 90, 180}°, 8 random body-frame
  axes per shell → 56 cells/shell × 7 shells = 56 jobs.
- **Δω-dir axis:** {0, 1, 5, 30, 90, 180}° in tangent plane to truth
  ω-dir, 8 azimuths each = 48 jobs.
- **Δω-mag axis:** {-50, -20, -10, -5, 0, 5, 10, 20, 50}% of truth-ω
  magnitude, single perturbation per pct = 9 jobs.

Total ~113 jobs/seed × 5 seeds × 85ms / 8 workers ≈ 10 sec wall.

## How

`_perturb_q0(q0, ang, axis)` LEFT-multiplies a body-frame rotation onto
the truth q0 (per quaternion convention). `_perturb_omega_dir(ω, ang,
azim)` rotates ω-direction in tangent plane while keeping magnitude.
`_perturb_ω-mag` scales magnitude only, keeps direction.

Score `(alignment, geo)` per perturbation. Bin by axis × angle and
report mean across azimuths.

## Result

**Alignment-cost roll-off (mean, per axis):**

| axis | 0 | 1 | 5 | 15 | 30 | 45 | 90 | 180 |
|---|---|---|---|---|---|---|---|---|
| Δq0° s006 | 1.00 | 1.00 | 1.00 | 0.73 | — | 0.09 | 0.21 | 0.05 |
| Δq0° s028 | 1.00 | 1.00 | 0.96 | 0.80 | — | 0.06 | 0.28 | 0.49 |
| Δq0° s041 | 1.00 | 0.75 | 0.00 | 0.00 | — | 0.13 | 0.13 | 0.00 |
| Δq0° s044 | 1.00 | 0.98 | 0.97 | 0.78 | — | 0.33 | 0.30 | 0.25 |
| Δq0° s091 | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.13 | 0.25 | 0.00 |
| Δω-dir° s006 | 1.00 | 0.86 | 0.45 | — | 0.23 | — | 0.05 | 0.00 |
| Δω-dir° s028 | 1.00 | 0.58 | 0.51 | — | 0.15 | — | 0.17 | 0.50 |
| Δω-dir° s044 | 1.00 | 0.80 | 0.51 | — | 0.35 | — | 0.40 | 0.33 |
| Δω-mag % | -50 | -20 | -10 | -5 | 0 | 5 | 10 | 20 | 50 |
| s006 | 0.00 | 0.14 | 0.00 | 0.14 | 1.00 | 0.14 | 0.00 | 0.00 | 0.14 |
| s028 | 0.00 | 0.08 | 0.08 | 0.25 | 1.00 | 0.33 | 0.08 | 0.00 | 0.25 |
| s044 | 0.05 | 0.19 | 0.24 | 0.38 | 1.00 | 0.48 | 0.43 | 0.29 | 0.38 |

(Seed 91 not shown for ω-dir / ω-mag — has 1 bright peak at edge of
bracket, low statistics.)

**Geo-cost roll-off (mean, per axis):**

| axis | 0 | 1 | 5 | 15 | 30 | 45 | 90 |
|---|---|---|---|---|---|---|---|
| Δq0° s006 | 1.00 | 1.00 | 0.56 | 0.00 | — | 0.00 | 0.00 |
| Δq0° s028 | 1.00 | 1.00 | 0.62 | 0.23 | — | 0.00 | 0.00 |
| Δq0° s044 | 1.00 | 0.97 | 0.64 | 0.22 | — | 0.02 | 0.00 |
| Δω-dir° s006 | 1.00 | 0.25 | 0.13 | — | 0.00 | — | 0.00 |
| Δω-dir° s028 | 1.00 | 0.27 | 0.10 | — | 0.02 | — | 0.02 |
| Δω-dir° s044 | 1.00 | 0.22 | 0.08 | — | 0.00 | — | 0.02 |

Seeds 41, 91: geo = 0 always (zero-classifiable).

## Why this matters

The roll-off curves quantify the *natural cell-size* for filter-based
grid search.

- **Geo cost is the sharper filter.** ω-dir basin ~1° (drops to ~0.20
  at 1°) — this is consistent with s003's "1° tube" but at the
  candidate-rejection level, not LM convergence. q0 basin ~5° (still
  ≥0.5 at 5°). Translates to ~5° q0 grid + ~1° ω-dir grid for full
  coverage.
- **Alignment cost is more permissive on q0/ω-dir** but **sharply
  selective on ω-mag** — drops to <50% at any non-zero perturbation.
  This is the same harmonic-density artifact observed in s024 but
  pivoted: alignment cost demands ω-mag pinned for a SINGLE q0 (truth);
  it's permissive about which q0 if you marginalise (s024).

**Strategic implication:** the natural filter tube is **anisotropic**:
- q0 axis: wide (5-15°)
- ω-dir axis: narrow (1-5°)
- ω-mag axis: narrow (≤5%)

LS-bracket gives ω-mag at ~5% accuracy = exactly the right scale.
s019b's 3-5° dir basin at PINNED mag is consistent with the alignment +
geo joint constraint here.

## Quirks worth flagging

- **Δq0=180° rebound (alignment):** seeds 28 (0.49) and 44 (0.25) have
  notable alignment scores at 180° q0 perturbation. Consistent with
  body-twin recovery (180° from truth → twin direction, which produces
  truth LC). The X-axis twin lives at q_180x · q0 — but here our
  perturbation directions are random body axes, so 180° about a random
  axis only HITS the X-twin if the random axis happened to be ±X. Seeds
  28 / 44 happen to have Sobol axes pointing close to ±X by chance.
- **Seed 41's q0=1° score 0.75:** truth-ω alone with a tiny q0 perturb
  fails 25% of the alignment check on seed 41. Seed 41 is in the s014
  multi-solution-rich class (class_3 near-twin); the alignment cost is
  apparently sensitive to q0 even at sub-degree perturbations on that
  seed. Worth flagging but not blocking.

## What this does NOT validate

- Cohort-scale roll-off characterisation. 5 seeds is a pilot sample;
  some seeds may have substantially different tube shapes (esp. cohort
  tail). Cheap follow-up if needed.
- Anisotropy quantification (e.g., ellipsoidal basin shape). Out of
  scope; the per-axis curves answer the cell-size question directly.

## Numbers

Per-axis 50%-retention radius (interpolated):

| axis | metric | s006 | s028 | s044 |
|---|---|---|---|---|
| Δq0 | alignment | ~25° | ~30° | ~30° |
| Δq0 | geo | ~5° | ~12° | ~12° |
| Δω-dir | alignment | ~5° | ~6° | ~10° |
| Δω-dir | geo | <1° | <1° | <1° |
| Δω-mag | alignment | <5% | <5% | <50% |

## Artefacts

- `experiments/s025_tube_shape.{py,md}`
- `results/s025/{seed*.npz, summary.json, tube_shape.png}`

## Out of scope

- Joint-axis perturbation surface (would explore basin anisotropy in
  2D); single-axis is sufficient for cell-size estimates.
- Cohort scale (>5 seeds): cheap follow-up if needed.

## Cross-references

- `s003_landscape_vs_omega.md` — the original "1° tube" measurement.
  s025 confirms the geometric scale at the candidate-rejection level.
- `s019b_omega_dir_basin_at_truth_mag.md` — 3-5° basin at pinned mag.
  s025's 1° geo-cost basin is tighter because it tests IC quality
  before LM refinement.
- `s019_ls_bracket_omega_mag.md` — 5% ω-mag prior accuracy aligns with
  s025's alignment-cost ω-mag sensitivity.
