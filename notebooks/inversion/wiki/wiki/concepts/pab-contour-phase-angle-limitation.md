---
title: "PAB-contour phase-angle limitation"
type: concept
sources:
  - "raw/inversion_diagnostics/m118/seed_014/summary.json"
  - "raw/inversion_diagnostics/isoshell_viewer/ipl_all_epochs.npz"
related:
  - "[[pab-contour-isoshell]]"
  - "[[surrogate-model]]"
  - "[[shadow-asymmetry]]"
  - "[[alignment-cost]]"
  - "[[m118_cost_comparison]]"
  - "[[surrogate-attitude-isoshell]]"
created: 2026-04-15
updated: 2026-04-16
confidence: high
---

# PAB-contour phase-angle limitation

## Core claim

The PAB-contour (`B(n_body)`) assumes `k1 = k2 = h`. At real IS-901 geometries with phase angle 30-60°, this approximation introduces a systematic **~25° offset** between truth PAB and its nearest IPL centroid. No cost function built on IPL centroid proximity can beat this noise floor.

## The math of the approximation

Per [[pab-contour-isoshell]], the PAB-contour is defined as:

> For every direction `(a, e)` on the body-frame sphere, the brightness value is computed assuming `k1 = k2 = h` (zero phase angle, no shadows).

So `B(n_body)` is a scalar: the brightness if sun and observer are both at direction `n_body` in the body frame.

The real brightness function is `B_real(k1_body, k2_body, panel, dish, distance)` — four more inputs, captures phase angle + shadows + articulation + range.

The pab-contour is the restriction of `B_real` to the diagonal slice where `k1 = k2`. Everywhere else on the `(k1, k2)` sphere × sphere, the real brightness disagrees with the pab-contour — sometimes by large amounts.

## Empirical evidence (m118, seed 14) [inline measurement]

The IPL `ang_dists[ep]` is "angular distance from truth PAB to its nearest centroid at epoch ep." If the pab-contour were a faithful model, this would be ~0 — truth would be on the centroid.

**Actual distribution at seed 14's 255 constraint epochs** (measured by inline Bash `python3 -c` on `isoshell_viewer/ipl_all_epochs.npz` + `m118/seed_014/kernel.npz` — not a committed script output):

| bucket | fraction |
|--------|---------|
| < 1° | 0% |
| 1° - 5° | 4% |
| 5° - 10° | 7% |
| 10° - 30° | 49% |
| 30° - 60° | 33% |
| 60° - 180° | 7% |

Median **25.7°**, mean 31.1°, max 81.6°. Only 4% of epochs have ang_dist under 5°.

The truth trajectory does not pass through the IPL centroids. They are a systematically shifted approximation.

## Implications

### Why IPL-centroid alignment cost fails (m118 evidence)

A cost of the form `Σ_epochs (1 − PAB_body · centroid)²` can never give 0 at truth on this seed, because truth doesn't align with the centroids. The minimum-possible truth cost is bounded below by the ang_dist distribution.

Moreover, wrong (ω, q0) solutions can *also* align poorly with the centroids — and sometimes align *better* than truth does, purely by accident. That's why m118's rank 1 is always a non-truth basin across all variants.

### Why ring cost didn't help (hypothesis)

`(PAB · centroid − cos(ang_dist))²` is the "correct target" version — it expects truth to be at angle ang_dist from the centroid. In principle truth cost = 0 under this.

In practice m118 showed near-identical results for ring vs bullseye. The **proposed mechanism (not separately measured)**:
- Our discretised grid doesn't contain exact truth; the grid-nearest truth candidate has ~1% magnitude error which over 2500 s propagation plausibly produces tens of degrees of attitude error at distant epochs
- So the scored "truth" cell in the tensor is not actually on the truth trajectory at distant epochs, and both cost forms produce residuals dominated by that discretisation error rather than the cost-form difference

The ring cost idea is geometrically correct; seed 14 doesn't show it off because the grid-truth-nearest-candidate's propagation error is likely comparable to or larger than the ring-vs-bullseye difference. A follow-up test would compute ring cost at the EXACT truth trajectory (not the grid cell) and verify it goes to zero. Not yet done.

### Why shadows also matter

Per [[shadow-asymmetry]], self-occlusion blocks up to 4.6 mag at ±Y epochs for IS-901. The pab-contour is lo-fi (no shadows). So even if the phase-angle issue were fixed, the pab-contour at a given direction claims brightness that's wildly wrong at some attitudes. The observed magnitude therefore doesn't pin PAB to a well-defined isoshell — the mapping from observed mag to pab-contour-level-set is itself noisy.

## What to do about it

Two paths ([[surrogate-attitude-isoshell]] fleshes both out):

1. **Use the surrogate as `B(k1_body, k2_body, panel, dish, distance)`.** No pab-contour assumption needed. At each epoch, the allowed attitudes are `{R : |surrogate(R k1_J2000, R k2_J2000, panel, dish, distance) − observed_mag| < σ}` — a 2D submanifold of SO(3). This is a proper generalisation of IPL with correct physics.
2. **Accept IPL as candidate-generator-only.** Do not use centroid proximity as a cost. Use IPL tightness / structure as an input to anchor selection and candidate seeding, but score final candidates with the full forward model (surrogate-DE, m115 pattern).

Option 1 is the active branch. Option 2 already works (m115, m117 inline test).

## Related wiki

- [[pab-contour-isoshell]] — the original framework
- [[surrogate-model]] — the drop-in replacement function
- [[shadow-asymmetry]] — the other reason pab-contour is approximate
- [[surrogate-attitude-isoshell]] — the proposed successor branch
- [[m118_cost_comparison]] — the experiment that measured this systematically
