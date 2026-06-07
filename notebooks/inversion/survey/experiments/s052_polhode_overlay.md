---
title: s052 — polhode-prior cannot rescue cascade noise on seed 14 (three diagnostics)
type: experiment
sources:
  - experiments/s049_cascade_seed14.md
  - experiments/s050a_cluster_geometry.md
  - experiments/s050b_validation_stress.md
  - experiments/s050cde_c1_concentration_dead.md
  - experiments/s051_polhode_observation.md
  - concepts/polhode_prior.md
related:
  - project_polhode_prior.md
  - feedback_visual_artefacts_unlock_insight.md
created: 2026-05-07
updated: 2026-05-07
confidence: high
---

# TL;DR

**The s051 polhode-conditioned ω-prior cannot rescue the s049 cascade
pool as a downstream filter on seed 14.** Three independent geometric
diagnostics, each via a static viewer + image read-back:

1. **Polhode overlay (s052)** — visual: cascade pool + 153 truth-q_a
   hypotheses + truth polhode in body-frame ω-space. Truth polhode is a
   substantial closed curve (1.97 dps L2 diameter, ~105° great-circle
   arc on the |ω|=1.235 dps sphere, NOT a point). Cascade pool extends
   ~1.86 dps p90 from centroid — comparable to the polhode itself, so
   the polhode is not "small relative to noise" as an order-of-magnitude
   guess might suggest.
2. **Polhode-tangent projection (s052c)** — quantitative: project each
   truth-q_a hypothesis onto the nearest polhode point, measure
   pre/post error vs ω_truth_t0. **Median improvement factor = 0.77**
   (23% reduction). Projection moves hypotheses onto the polhode but at
   *random phase positions* — not concentrated near ω_truth_t0. Modest,
   not structural.
3. **Polhode-label preservation (s052d)** — quantitative: cascade noise
   scrambles the polhode label `D = 2T·I_b/|L|²`. Truth `D = 1.0885`
   (just above separatrix — confirms s051 prediction that high-|ω|
   seeds are near-separatrix). Cascade truth-q_a hypotheses span
   `D ∈ [0.996, 1.367]` p10–p90, freely crossing the separatrix.
   **Concentration enrichment over pool: 1.33× at ±0.01 in D, 1.0× at
   ±0.05.** `|L|` itself varies 2× across truth-qa hypotheses (truth=660,
   span 305-1183 kg·m²/s).

The cascade noise is large enough to occupy *multiple polhodes
simultaneously*, so polhode constraints applied as downstream filters
are statistically impotent. **This refutes the polhode-rescue
formulation of s051**, but does NOT refute the polhode prior itself —
it constrains where the prior is applicable (upstream sampling, LM
polish, LC feature extraction; NOT cascade-pool filtering).

The s051 promotion gate prediction (high-|ω| seeds are near-separatrix
polhodes) IS confirmed by the truth measurement on seed 14. So the
polhode prior may still be the correct architectural reframe — it just
isn't a fix to the broken cascade.

# What

For seed 14 (m048 cohort tail, |ω|=1.229 dps, the binding case for s042
basin-width measurements) we re-tested the s051 reframe by directly
visualising and quantifying:

- The truth polhode (ω(t) trajectory in body frame) vs the s049 cascade
  pool (141706 (q_a, ω) hypotheses) and the 153 truth-q_a hypotheses
  (rows of the pool whose q_a equals the truth's discrete sample).

Three sub-tests:

- **s052** — overlay viewer + 6-panel diagnostic figure.
- **s052c** — pre/post-projection error decomposition with PCA polhode-
  plane projection figure.
- **s052d** — polhode-label `(|L|, 2T, D)` preservation test.

# How

`experiments/s052_polhode_overlay.py`:
- Build seed-14 context, propagate truth → ω(t) at 500 observation
  epochs.
- Load cascade pool, identify 153 truth-qa rows via antipode-aware
  `qA_kept @ truth_q_a > 1 - 1e-9`.
- Subsample 5000 non-truth-qa pool rows.
- Plotly 3D scatter: polhode (line+markers), pool (grey), truth-qa
  (colored by polhode-distance), truth target + cascade derivation.
- Save NPZ checkpoint with all arrays.

`experiments/s052b_polhode_diagnostic.py`:
- Load NPZ, compute polhode pairwise diameter, cluster extents,
  noise/polhode ratios.
- 6-panel matplotlib: 3D wide, 3D tight (5×polhode diameter), 3
  orthogonal 2D projections, |ω|(t) line, polhode-distance histogram.

`experiments/s052c_projection_test.py`:
- Project each truth-qa onto nearest polhode point.
- Decompose noise (ω_h - ω_p) into tangent vs perpendicular components.
- Plot: pre vs post error scatter; tangent vs perp magnitudes; histogram
  of improvement factors; PCA polhode-plane view with projection arrows.

`experiments/s052d_polhode_label_test.py`:
- Compute `(|L|, 2T)` for each ω hypothesis.
- Compute polhode label `D = 2T·I_b / |L|²`.
- Plot: (|L|, 2T) plane; D histogram (full + zoomed); D vs |L| scatter.
- Concentration test: fraction within ±s of D_truth for s ∈ {0.001,
  0.01, 0.05, 0.1}; enrichment vs pool.

All figures rendered with matplotlib at 110 dpi; viewers also produced
as Plotly HTML (`overlay.html`, kept open for user verification).

# Result

## Inertia tensor for m048 (seed-independent)
```
I_a = 7749    I_b = 37985    I_c = 38306    (kg m²)
I_a/I_b = 0.204   I_b/I_c = 0.992   I_c/I_a = 4.94
```
Body is essentially prolate (I_b ≈ I_c, both ~5× I_a). Long axis along
principal axis a; near-symmetric around it.

## Truth polhode for seed 14 (post-propagation, 500 epochs over 3600s)
- Truth `(|L|, 2T) = (660.37 kg m²/s, 12.50 J)` — conserved exactly to
  numerical precision over the 500 samples.
- Truth `D = 2T·I_b / |L|² = 1.0885` — **just above separatrix (D=1)**.
  Confirms s051's structural prediction for the high-|ω| cohort tail.
- Polhode L2 pairwise diameter: **1.97 dps** (substantial closed curve;
  ~105° arc on |ω|=1.235 sphere via 2·arcsin(d/(2|ω|))).
- |ω|(t) span 0.013 dps (std/mean 0.37%) — magnitude near-conserved
  even though the body is asymmetric, because (I_b ≈ I_c) makes the
  precession quasi-axisymmetric.

## Sub-test 1: visual overlay (s052)
Body-frame 3D: polhode is a clean blue ellipse near origin; truth-qa
(Viridis cloud) extends ~1.5 dps from polhode; pool (grey haze) extends
~2 dps. Polhode is comparable in size to the noise extent, not point-
like. Visual at `results/s052_polhode_overlay/diagnostic.png`.

## Sub-test 2: projection error reduction (s052c)
- Pre-projection error to ω_truth_t0: p10/p50/p90 = 0.66 / 1.61 / 2.70 dps
- Post-projection error to ω_truth_t0: p10/p50/p90 = 0.26 / 1.53 / 1.94 dps
- Improvement factor (post/pre): p50 = 0.77 (23% reduction). p90 = 1.18
  (cases where projection is worse).
- PCA-plane visualisation: polhode is a clean ellipse; projected truth-qa
  points are scattered around the entire ellipse — not concentrated near
  the truth phase position.
- **Conclusion**: projection-to-polhode ≠ projection-to-truth. The cascade
  noise has a large tangent component along the polhode (random phase),
  and the perpendicular component (which projection nullifies) is only
  half the total noise.

## Sub-test 3: polhode-label preservation (s052d)
Truth: `|L|=660.37, 2T=12.50, D=1.0885`.

Cascade truth-qa hypotheses (n=153):
| metric  |    p10  |    p50  |    p90  |
|---------|---------|---------|---------|
| `|L|`   |  304.96 |  806.58 | 1183.13 |
| `D`     |    0.996 |   1.034 |   1.367 |

Pool (n=5000): similar broad spread, slightly less skewed.

Concentration enrichment of truth-qa over pool around D_truth:
| window  | truth-qa | pool    | enrich |
|---------|----------|---------|--------|
| ±0.001  |   0.65%  |   0.44% |  1.49× |
| ±0.01   |   5.23%  |   3.92% |  1.33× |
| ±0.05   |  22.88%  |  22.84% |  1.00× |
| ±0.1    |  81.70%  |  88.60% |  0.92× |

**Polhode label is NOT preserved by cascade noise.** Cascade truth-qa
hypotheses fluently span across the separatrix (D from 1.0 below to
1.4 above truth). |L| varies 2× across the cluster.

## Synthesis

Three structurally distinct tests, three negative results for the
polhode prior as a *downstream filter on the s049 cascade pool*. The
cascade noise is large enough that the polhode prior cannot recover
the truth ω from cascade hypotheses by any direct constraint
(projection, label-filtering, |L|-filtering).

# Why this matters

**Refines the s051 reframe** without refuting it:

- Confirmed: seed 14 IS on a near-separatrix polhode (D_truth = 1.088).
  This validates the s051 prediction that high-|ω| cohort-tail seeds
  are near-separatrix.
- Refuted: applying the polhode prior as a filter on cascade-derived
  ω hypotheses recovers the truth ω. Cascade noise is too large.
- The polhode prior remains viable for *upstream* applications:
  (a) direct sampling in `(q_a, |L|, polhode_label, polhode_phase)`
  space — replaces uniform Sobol over `(q_a, ω_x, ω_y, ω_z)` and
  guarantees every sample is dynamics-admissible;
  (b) LM-polish ω-tangent constraint near truth — reduces 3-DOF to
  1-DOF in ω updates (polhode phase only); plausible LM convergence
  win;
  (c) LC feature → polhode-period extraction — independent of cascade,
  could provide a 1D cohort-wide polhode-label classifier from LC
  alone.

The cascade was supposed to be a "seed generator" for LM polish (s049);
s050a showed the seed count is 32k clusters at default radius (163×
over budget); s050b/c/d/e closed every per-q_a re-scoring strategy;
s052 closes the polhode-rescue strategy. **The cascade is not the
right architecture for seed 14.** Time to pivot to direct polhode-
conditioned sampling or to LC-based polhode-label extraction.

# Numbers

Run details:
- Seed: 14 (binding cohort case, |ω| = 1.229 dps).
- Cascade pool: 141,706 (q_a, ω) hypotheses (s049, tol=0.10/K=3 not
  yet applied — the raw pool).
- Truth-qa subset: 153 rows.
- Pool subsample for plots: 5000 rows.
- Polhode samples: 500 (full observation window, 3600s, ~12 polhode
  loops).

Wall time:
- s052_polhode_overlay.py: ~2s (mostly satellite STL load).
- s052b_polhode_diagnostic.py: ~3s (pairwise distance on 153×500).
- s052c_projection_test.py: ~3s.
- s052d_polhode_label_test.py: ~2s.
Total: <15s end-to-end.

# Artefacts

- `experiments/s052_polhode_overlay.py` — script (overlay + NPZ).
- `experiments/s052b_polhode_diagnostic.py` — 6-panel diagnostic.
- `experiments/s052c_projection_test.py` — projection-leverage test.
- `experiments/s052d_polhode_label_test.py` — polhode-label test.
- `experiments/s052_polhode_overlay.md` — this writeup.
- `results/s052_polhode_overlay/polhode.npz` — truth ω(t) + cascade
  subsets.
- `results/s052_polhode_overlay/overlay.html` — interactive Plotly 3D.
- `results/s052_polhode_overlay/overlay_default.png` — initial
  screenshot for image read-back.
- `results/s052_polhode_overlay/diagnostic.png` — 6-panel matplotlib.
- `results/s052_polhode_overlay/projection_test.png` — pre/post +
  PCA-plane.
- `results/s052_polhode_overlay/polhode_label_test.png` — (|L|, 2T)
  + D distributions.
- `results/s052_polhode_overlay/summary.json` — quantitative summary.

# Out of scope

- Cohort-wide polhode label scan for all 100 seeds (cheap, ~30s; would
  test "are all high-|ω| seeds near-separatrix?" cohort-wide). Not done
  in this experiment but obvious next step.
- Visual rendering of polhodes for seeds 17, 68, 81 (the rest of the
  high-|ω| cohort tail, per s051 promotion gate). Not done; can be
  produced via the s048c+ viewer with new trajectories.
- Polhode-conditioned direct sampling test (sample in
  `(|L|, polhode_label, polhode_phase)` space, validate at fixed q_a).
  This is a sample-efficiency study, not done here.
- LC feature extraction for polhode period τ_p. Not done; needs LC
  Lomb-Scargle + correlation with truth polhode period.

# Cross-references

- `concepts/polhode_prior.md` — the geometric framing.
- `experiments/s051_polhode_observation.md` — viewer-driven discovery.
- `experiments/s050cde_c1_concentration_dead.md` — sister cascade-
  rescue closure (Hough / consensus / q-space all closed negative).
- `experiments/s042_basin_radius_cohort.md` — cohort-wide ω-mag basin
  width measurements; D_truth = 1.088 is consistent with seed 14's
  narrow basin.
- `feedback_visual_artefacts_unlock_insight.md` — methodology lesson
  re-validated: each of the three findings here surfaced from a
  matplotlib figure or Plotly HTML + image read-back rather than from
  numbers alone.
