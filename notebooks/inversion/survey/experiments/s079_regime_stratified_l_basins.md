---
title: "s079 — regime-stratified L-basin re-score: the equal-|L|/free-direction class is a long-axis-mode phenomenon"
type: experiment
sources:
  - results/s079/summary.json
  - results/s077/basin_l_metrics.npz  (640 converged basins, 10 pilot seeds)
  - results/s077/summary.json  (cross-check: truth_regime via omega_jacobi)
related:
  - experiments/s077_l_vector_basin_sweep.md
  - experiments/s073f_cluster457_local_geometry.md
  - experiments/s073d_polhode_match_cluster457.md
created: 2026-05-14
updated: 2026-05-14
confidence: medium (clean split on the 10-seed pilot, every number traced to results/s079/summary.json; but n=10 and the basin gate is the loose surrogate final_mse<0.5, not a hi-fi rho-band)
---

# TL;DR

Re-scored s077's 640 converged basins, stratified by the truth seed's tumbling regime. The
"equal-|L|, free-direction" competing-basin class that s077 found cohort-wide is **almost
entirely a long-axis-mode (LAM) phenomenon** on these 10 pilot seeds: the 6 LAM seeds host
**96 of the 104** competing low-MSE basins (median 14/seed, range 3–42); the 4 short-axis-mode
(SAM) seeds host **8** (median 2.5/seed, range 0–3). The lowest LAM count equals the highest
SAM count. Separatrix proximity (k²) is *not* the driver — the most near-separatrix seed of
all (seed 28, SAM, k²=0.944) has zero competing basins. This supports the user's reframe that
the multi-solution structure depends on the *type* of tumbling motion. Pure re-score, no new
compute.

# What

s077 found, among 104 competing low-MSE basins, that `|L|` magnitude is pinned to truth
(median 0.73%) while `L_J2000` direction is free (median 90.8°) — discrete clumps, strongly
seed-dependent. s073f then showed the lone seed-89 example is a *locally isolated* point, not
a soft sheet. The open question the user posed: the equal-|L|/free-direction structure is not
expected on every seed — does it track the tumbling regime, i.e. short-axis mode (SAM,
rotation about the max-inertia axis) vs long-axis mode (LAM, rotation about the min-inertia
axis), and separatrix proximity?

This experiment is the regime-stratified cut of s077's cached basins.

# How

`experiments/s079_regime_stratified_l_basins.py`. Pure re-score of
`results/s077/basin_l_metrics.npz` — no propagation, no rendering. For each of the 10 pilot
seeds, classify the truth state's tumbling regime from `(ω0, I)`:

- Principal inertias ascending `I_1 ≤ I_2 ≤ I_3`; `disc = 2T·I_2 − |L|²` with `2T`, `|L|²`
  computed basis-independently (`ω·Iω`, `|Iω|²`).
- `disc ≥ 0` → **regime A / LAM** (polhode encloses the min-inertia axis `I_1`).
- `disc < 0` → **regime B / SAM** (polhode encloses the max-inertia axis `I_3`).
- `k²` (elliptic modulus) → 1 at the separatrix, → 0 at a pure-axis spin.

Formulas mirror `lib/jacobi_propagator.py::_build_omega_func`. A self-check asserts s079's
per-seed regime label reproduces s077's `truth_regime` (which came from the validated
`omega_jacobi` path) — **gate passed for all 10 seeds**. Then cross the per-seed competing-
basin count, distinct-polhode count, `L`-direction spread, and `|L|`-magnitude pinning
against the regime label and k².

A first version of `classify_regime` had a category-B bug — it computed `2T`/`|L|²` from
body-frame ω components against principal-axis inertia *values*, which is only valid if the
body frame coincides with the principal-axis frame (it does not here). Fixed to the
basis-independent form; the s077 cross-check gate now passes.

# Result

| seed | mode | regime | k² | n_comp | distinct polhodes | L-dir median (°) |
|---|---|---|---|---|---|---|
| 6  | LAM | A | 0.228 | 15 | 2  | 90.8  |
| 10 | LAM | A | 0.521 | 42 | 24 | 83.6  |
| 21 | SAM | B | 0.390 | 2  | 2  | 143.4 |
| 28 | SAM | B | 0.944 | 0  | 0  | —     |
| 41 | SAM | B | 0.548 | 3  | 2  | 48.8  |
| 44 | LAM | A | 0.074 | 18 | 2  | 91.7  |
| 48 | LAM | A | 0.015 | 3  | 3  | 92.6  |
| 60 | LAM | A | 0.391 | 13 | 1  | 136.5 |
| 84 | LAM | A | 0.040 | 5  | 3  | 116.8 |
| 91 | SAM | B | 0.681 | 3  | 1  | 5.8   |

**The regime split is clean.** LAM (6 seeds): **96** competing basins total, median 14/seed,
every LAM seed ≥ 3. SAM (4 seeds): **8** competing basins total, median 2.5/seed, every SAM
seed ≤ 3. The lowest LAM count (3, seed 48) equals the highest SAM count (3, seeds 41 & 91) —
the distributions barely touch.

**Separatrix proximity is not the driver.** Seed 28 (SAM) has the highest k² of all seeds
(0.944, essentially *at* the separatrix) and hosts **zero** competing basins. Within LAM
there is a weak positive trend — the richest seed (10, 42 basins / 24 distinct polhodes) has
the highest LAM k² (0.52), and the two lowest-k² LAM seeds (48, 84) have the fewest LAM
basins (3, 5) — but it is not monotonic (seed 44, k²=0.074, has 18). The first-order axis is
**regime**, not k².

**Pinning still holds inside LAM.** Pooled over the 96 LAM competing basins, `|L|`-magnitude
rel diff median is **0.70%** (pinned) and `L_J2000` direction offset median is **91°** (free,
range 8–167°) — i.e. the s077 cohort headline is really the LAM headline. The 8 SAM basins
are too few to characterise (pooled `|L|` rel diff 1.7%, L-dir median 49°).

# Why this matters

The cat-4 / equal-|L|-free-direction multi-solution class is, on this pilot, a **long-axis-
mode property**. That is a sharper and more useful statement than "seed-dependent": it ties
the structure to a physically meaningful, *a-priori-knowable* property of the truth state
(LAM vs SAM is decided by `disc = 2T·I_2 − |L|²`, which an inversion would estimate alongside
`(q0, ω)`). If it holds up, it says *where* to expect the hard multi-solution degeneracy and
where not to.

A plausible mechanism (hypothesis, not tested here): the m048 inertia tensor is
near-axisymmetric — principal inertias `[7749, 37986, 38306]` kg·m², with `I_2` and `I_3`
only 0.84% apart and `I_1` ~5× smaller. LAM tumbles about the *unique* small axis; SAM
tumbles about one of the *near-degenerate* large-axis pair, which is closer to an
axisymmetric top and may simply admit fewer geometrically distinct attitudes that fit the LC.
This would predict the split is specific to near-axisymmetric bodies — worth keeping in mind
before generalising.

# What this does NOT establish

- **n = 10, and 6/4 split** — this is a first-look correlation on the pilot, not a cohort
  claim. A real regime study needs Band A∪B multi-sols on more seeds, which is new inversion
  compute (the user explicitly scoped this round to the 10 pilot seeds).
- **The basin gate is loose.** "competing low-MSE" = surrogate `final_mse < 0.5` (the s011
  convention), not a per-basin hi-fi ρ-band. The counts conflate genuine LC-equivalent
  solutions with multi-start basin-of-attraction geometry. Hi-fi ρ-band validation of these
  basins is the natural next gate.
- **No causal claim.** Regime *correlates* with competing-basin count here; the mechanism
  paragraph above is an untested hypothesis.
- **`n_comp` is not a pure cat-4 metric** — it counts low-MSE non-truth basins, which is the
  necessary substrate for the equal-|L|/free-direction class but also picks up ordinary
  multi-start spread.

# Numbers

- 640 basins, 10 seeds, 104 competing low-MSE (cohort).
- LAM: seeds 6,10,44,48,60,84 — 96 competing basins, median 14.0/seed, range 3–42; pooled
  `|L|`-mag rel diff median 0.703%, `L`-dir offset median 91.2° [8.2, 166.8].
- SAM: seeds 21,28,41,91 — 8 competing basins, median 2.5/seed, range 0–3; pooled `|L|`-mag
  rel diff median 1.685%, `L`-dir offset median 48.8° [5.7, 146.2].
- Principal inertias (ascending): [7749.01, 37985.16, 38305.71] kg·m².
- s077 regime cross-check gate: PASS on all 10 seeds.
- Compute wall: ~seconds (re-score only).
- Source for all of the above: `results/s079/summary.json`.

# Out of scope

- Hi-fi ρ-band validation of the 104 competing basins (queued; the loose-gate caveat above).
- Extending the regime cut beyond the 10 pilot seeds (needs new inversion runs).
- Testing the near-axisymmetry mechanism hypothesis (would need a non-axisymmetric satellite
  model or a perturbed-inertia study).
- The cat-4 theorem half-page and [RF74] acquisition (s073e deferred items).

# Cross-references

- `experiments/s077_l_vector_basin_sweep.md` — the cohort sweep this re-scores; |L| pinned,
  direction free across 104 competing basins.
- `experiments/s073f_cluster457_local_geometry.md` — closed the polish-residual reading on
  seed 89's lone Band-A attractor; cleared the way for this regime cut.
- `experiments/s073d_polhode_match_cluster457.md` — the ~1.1% Casimir mismatch on seed 89.
- `lib/jacobi_propagator.py` — `_build_omega_func` regime A/B and k² formulas.
