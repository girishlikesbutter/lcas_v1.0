---
title: "s004 — m103 alignment-cost landscape under ω-misspecification (3 seeds × 24 ω perturbations)"
type: experiment
sources:
  - data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed*.npz (post-fix; commit ac1fdf4)
  - data/results/inversion_diagnostics/m048_trajectories/m048_trajectories.npz (inertia tensor, unique_normals)
  - lib/forward.py (q0+ω → quaternions; round-trips at machine precision)
  - experiments/s003_landscape_vs_omega.py (sweep shape — same 24 perturbations, same Sobol seed)
related:
  - experiments/s001_cost_at_truth_cohort.md (Q1 — alignment cost at truth, cohort-scale)
  - experiments/s003_landscape_vs_omega.md (Q3 — surrogate landscape under ω-misspecification)
  - concepts/known_pathologies_to_revalidate.md (m103 alignment cost sphere of applicability)
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# s004 — m103 alignment-cost landscape under ω-misspecification

## TL;DR

Same sweep shape as s003 (3 seeds × 24 ω perturbations × 512 candidate q0)
but evaluating the m103 alignment cost (peak-bisector dot product against
allowed normals at constraint epochs) instead of surrogate full-LC MSE. The
question: does the alignment cost have a wider truth-ω tube than the
surrogate, and could it therefore feed an outer ω-search the surrogate can't?

**Answer: No. The alignment cost is strictly worse than surrogate-MSE on
every metric measured, and on one of the three seeds it fails the most
basic correctness check (argmin not at truth-q0 even at truth-ω).**

Specifically:

- **5/72 cells (7%) keep argmin within 10° of truth-q0**; s003 was 11/72
  (15%). **65/72 cells (90%)** have argmin ≥30° away (s003: 50/72, 69%);
  **43/72 (60%)** have argmin ≥100° away (s003: 39%).
- **Per-seed truth-ω tubes are zero or near-zero in the mag direction
  on all 3 seeds**, narrower than s003's surrogate tube which extended to
  ~2-5% mag.
- **Seed 28 has argmin at 176.8° from truth even at baseline ω = ω_truth**,
  with 2 sobol candidates beating truth-q0. The alignment cost is structurally
  ambiguous at this seed: multiple dissimilar attitudes satisfy the
  constraint-epoch dot products comparably well. The surrogate-MSE was
  argmin = truth on 8/8 seeds at baseline (s002).
- **Alignment cost is structurally undefined on 24/100 seeds** (those with
  ≤1 spec peak after the m103 anchor strip — already known from s001).
  This includes seed 91, the m115 failure anchor.

Decision: **m103 alignment cost adds no ω-search value the surrogate
doesn't already provide.** It is narrower in tube, structurally undefined on
24% of the cohort (including seed 91), and on at least one seed not even
correctly minimised at truth. Any inversion architecture using alignment cost
as an outer ω-search around a surrogate q0-inner is dead. **Joint (q0, ω)
local descent (Q4) remains the only viable architecture.**

## What

For each of 3 seeds (6, 18, 28 — see seed-selection note below), apply 24
ω perturbations and probe the q0 alignment-cost landscape under each. Same
perturbation ladder, same Sobol-Shoemake quaternion grid, and same truth-q0
embedded at index 0 as s003 — so cell-by-cell comparison between s003
(surrogate-MSE) and s004 (alignment-cost) is well-defined for the
intersecting seed (6).

Cost evaluated per candidate (q0_c, ω_p):

```
cost = sum_{ci in constraints} w * (1 - max_a (pab_body[ci] · normals[allowed_a]))^2
       (w = 10.0, allowed = m103 magnitude-banded normals at observed[ci])
```

with `pab_body[ci] = R(q_propagated[ci]) @ pab_J2000[ci]`, where pab_J2000 is
derived once per seed by inverting the cached truth pab_body using the cached
truth quaternions (bypasses the bisector sign convention).

### Seed-selection note

s003 used seeds 6, 41, 91. **41 and 91 cannot be reused** for s004: both have
≤1 spec peak after the m103 anchor strip, so the constraint set is empty and
the alignment cost is structurally undefined. Replacements were chosen to
preserve the s003 PA-stratification spirit while having a usable constraint
set:

| seed | role             | PA_med | ω_mag    | n_constraints | s001 align_cost |
|------|------------------|--------|----------|---------------|-----------------|
|  6   | carry from s003  | 62.5°  | 0.71°/s  | 3             | 4.370e-04       |
| 18   | PA-low (← was 41)| 14.2°  | 1.13°/s  | 14            | 1.713e-03       |
| 28   | PA-high (← was 91)| 87.3° | 1.44°/s  | 6             | 7.745e-04       |

That 41 and 91 are inalignable is itself a structural finding: alignment cost
cannot be the seeding layer for any inversion that needs to be defined on the
full cohort. (Re-stated from s001 but operationalised here.)

## How

- **Cost machinery:** Copied from s001 (which itself copied from
  `m103_hybrid.py`); not imported. Same constants: `CONSTRAINT_WEIGHT=10.0`,
  noise rng=42 / σ=0.05, spec threshold mag<9.0, peak-finder distance=5
  prominence=0.3, savgol(7,3) smoothing, magnitude-banded
  `get_allowed_normals` lookup.
- **Smoke test (passed all 3 seeds):** at (q0_truth, ω_truth) using cached
  pab_body_truth, the alignment cost reproduces s001's recorded value to all
  digits — 4.3704e-04 (seed 6), 1.7133e-03 (seed 18), 7.7450e-04 (seed 28).
  The cached-pab path is sound; the propagated path matches in the baseline
  perturbation (seeds 6, 18) modulo the seed 28 ambiguity discussed below.
- **Propagation:** `src.dynamics.attitude_propagator.propagate_attitude`
  (post-fix) per candidate; rotations applied only at constraint epochs
  (much fewer than the full N epochs); scipy `Rotation.from_quat` with
  (w,x,y,z)→(x,y,z,w) reshuffle.
- **Grid:** identical to s003 — `qmc.Sobol(d=3, scramble=True, seed=42)`,
  511 sobol + truth-q0 at index 0 = 512 candidates. Identical Sobol candidate
  geometry across all (seed, perturbation) cells.
- **Workers:** Pool(8), one Pool per seed (reused across 24 perturbations).
  BLAS=1 in workers. No surrogate forward pass per candidate, so each score
  is dominated by the propagator (~25 ms per candidate vs s003's ~89 ms with
  surrogate).
- **Wall:** seed 6 = 51.1 s; seed 18 = 87.6 s; seed 28 = 69.3 s; total ~3.5 min.

## Result

### Cell-count summary across 72 (seed, ω-perturbation) cells

| metric                                 | s003 (surrogate-MSE) | s004 (alignment cost) |
|----------------------------------------|----------------------|------------------------|
| argmin within 5° of truth-q0           | 11 / 72  (15%)       | **5 / 72  (7%)**       |
| argmin within 10° of truth-q0          | 11 / 72  (15%)       | **5 / 72  (7%)**       |
| argmin within 30° of truth-q0          | 22 / 72  (31%)       | **7 / 72  (10%)**      |
| argmin ≥30° from truth-q0              | 50 / 72  (69%)       | **65 / 72 (90%)**      |
| argmin ≥100° from truth-q0             | 28 / 72  (39%)       | **43 / 72 (60%)**      |

The alignment cost is **strictly worse** in every count.

### Per-seed truth-ω tubes (largest perturbation at which argmin = truth-q0)

| seed | s004 ω-dir tube | s004 ω-mag tube | s003 ω-dir tube  | s003 ω-mag tube |
|------|------------------|-----------------|--------------------|-----------------|
|   6  | **0°** (escapes at dir_0.5°) | **0** (escapes at mag×0.5; even mag×0.95 lands 53° off) | 1° | < 5% (mag×0.95 already 130° off) |
|  18  | 1° (escapes at dir_2°) | **0** (escapes at mag×0.5) | n/a (different seed) | n/a |
|  28  | **NO TUBE** — argmin 176.8° from truth even at ω = ω_truth | **NO TUBE** | n/a (different seed) | n/a |

For the only overlapping seed (6), the alignment cost has a **strictly
narrower** dir tube (0° vs 1°) and the same essentially-zero mag tube. Where
s003's surrogate-MSE reliably localised truth-q0 inside the small tube, s004
has truth-q0 winning only at exact-baseline-ω.

### Per-seed argmin-stays-within-5° matrix (1 = within tube, dot = escaped)

```
                     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
                    base 0.5x .8x .9x .95 1.05 1.1 1.2 1.5 2.0 d.5 d1 d2 d5 d10 d30 d60 d90 (combined)
  seed   6:          1  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  seed  18:          1  .  .  .  .  .  .  .  .  .  1  1  .  .  .  .  .  1  .  .  .  .  .  .
  seed  28:          .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
```

Seed 18's perturbation 17 (dir_90°) = "1" is almost certainly a numerical
coincidence: at dir_90° the alignment cost is huge for everyone
(truth_q0_cost = 2.982e+0 vs baseline 1.713e-3) and truth-q0 happens to be
the least-bad candidate of a sea of bad candidates, with cost numerically
identical to the argmin's. This is statistical noise, not a recovered tube.

### truth-q0 cost growth as ω drifts (seed 6, illustrative)

| perturbation       | s001 truth_align_cost | s004 baseline-equiv | growth factor |
|--------------------|------------------------|--------------------|----------------|
| baseline (ω = ω_truth) | 4.37e-04           | 4.37e-04           | 1×             |
| ω-dir +0.5°            |                    | 3.66e-03           | 8×             |
| ω-dir +1°              |                    | 2.57e-02           | 59×            |
| ω-dir +2°              |                    | 2.34e-01           | 535×           |
| ω-dir +5°              |                    | 1.06e+00           | 2400×          |
| ω-mag ×0.95            |                    | 1.97e+00           | 4500×          |
| ω-mag ×0.5             |                    | 9.70e-01           | 2200×          |

Cost growth pattern is similar to s003's surrogate (3-4 OOM under small
perturbation), but the argmin's position carries no useful information — it
wanders to random sobol candidates 50°–180° from truth, indicating the
landscape is just noisy in the off-baseline regime.

### Special pathology: seed 28 baseline failure

At seed 28 baseline (ω = ω_truth, the candidate set includes truth-q0):
- truth_q0_cost = 7.745e-04
- argmin_cost = 4.480e-04 at a sobol candidate **176.8° geodesic from truth**
- n_sobol_below_truth = 2

Two sobol-Shoemake candidates have lower alignment cost than truth-q0 itself.
The same pattern persists across the small-dir perturbations on seed 28
(dir_0.5° / dir_1° / dir_2° / dir_5° all show n_sobol_below_truth ∈ {2, 3, 4}
and argmin between 93° and 177° from truth).

The alignment cost on this seed has multiple low-cost basins competing with
truth at very similar cost values. The surrogate-MSE (s002) had argmin =
truth-q0 on 8/8 seeds at baseline; alignment cost fails this property here.

The mechanism is geometric: at seed 28 the 6 constraint epochs and their
allowed-normal sets admit multiple very different attitudes whose
constraint-epoch pab_body vectors all align comparably well with normals in
the allowed set. The full-LC surrogate cost penalises every epoch (not just
constraint epochs) and so distinguishes between these "constraint-twin"
attitudes; the alignment cost cannot.

## Why this matters

1. **Q5 has a decisive negative answer.** The alignment cost has no wider
   ω-tube than the surrogate; on 2/3 seeds it has a strictly narrower tube,
   and on the third seed it has no tube at all. There is no architecture in
   which alignment cost helps the ω-search where surrogate-MSE has already
   failed (s003).
2. **The "alignment-outer ω + surrogate-inner q0" hybrid is dead.** Any
   hybrid that tries to use alignment cost to localise ω first and then
   surrogate cost to refine q0 is doomed: alignment cost has no
   monotonic-toward-truth-ω signal in 90% of the (seed, perturbation) cells
   tested, and on a non-trivial subset of seeds it doesn't even pick truth
   at exactly-correct ω. The "alignment cost provides a structural prior"
   intuition (carryover from m103-era thinking) does NOT survive post-fix
   measurement.
3. **24% of the cohort is structurally inalignable.** This is already
   recorded in s001 but it bites here: any architecture relying on alignment
   cost cannot be defined on seeds with ≤1 spec peak. That includes seed 91,
   the m115 failure anchor — even setting aside ω-fragility, alignment cost
   could never have helped m115 on its own anchor seed.
4. **m103 alignment cost may have multi-basin pathology even at baseline.**
   Seed 28 demonstrates that a sufficiently small constraint set with
   permissive allowed-normal lookups can admit multiple low-cost attitudes
   structurally indistinguishable to the alignment cost. This is a
   fundamental discrimination ceiling, not a solver issue. To verify how
   widespread this is, a follow-up cohort-scale "alignment-cost argmin =
   truth-q0 at baseline" check on all 76 seeds where it's defined would be
   useful (cheap experiment — call it Q5b).
5. **Joint (q0, ω) local descent (Q4) is now the only viable architecture
   left standing.** s003 ruled out decoupled (ω-outer surrogate / q0-inner
   surrogate) because of the narrow surrogate tube. s004 rules out
   decoupled (ω-outer alignment / q0-inner surrogate) because the
   alignment-cost surface is even less informative. Every "outer search
   over ω" architecture is now closed. Q4a — joint local descent from a
   near-truth seed — is the next experiment.

## Numbers

- Seeds: 6, 18, 28 (PA spread 14° to 87°; n_constraints ∈ {3, 14, 6}).
- Perturbations per seed: 24 (1 baseline + 9 mag-only + 8 dir-only + 6 combined; identical to s003).
- Candidates per perturbation: 512 (truth-q0 + 511 Sobol-Shoemake; identical Sobol seed to s003).
- Total alignment-cost evaluations: 3 × 24 × 512 = 36 864.
- Wall: 51.1 + 87.6 + 69.3 ≈ 208 s scoring + ~5 s save phase = ~3.5 min total.
- Baseline (ω = ω_truth) argmin = truth on 2/3 seeds (seed 28 fails with
  argmin at 176.8° from truth, 2 sobol candidates lower-cost than truth).
- Truth-ω tube width (largest perturbation keeping argmin within 5° of truth):
  seed 6 = baseline only; seed 18 = dir up to 1°, no mag; seed 28 = none.
- Cells with argmin ≥30° from truth-q0: **65/72 (90%)**, vs s003's 50/72 (69%).
- Cells with argmin ≥100° from truth-q0: **43/72 (60%)**, vs s003's 28/72 (39%).
- Seeds in cohort where alignment cost is structurally defined: 76/100
  (24/100 seeds have ≤1 spec peak after anchor strip, including seed 91).

## Artefacts

- `experiments/s004_alignment_landscape_vs_omega.py` — the script.
- `experiments/s004_alignment_landscape_vs_omega.md` — this writeup.
- `results/s004/per_seed_omega.npz` — per-seed × per-perturbation full_cost
  (512), truth_q0_cost, argmin_idx, argmin_geo_to_truth, n_sobol_below_truth,
  constraint_epochs arrays.
- `results/s004/summary.json` — population-level cell counts + per-seed
  per-perturbation summary table + seed-selection note.
- `results/s004/argmin_drift.png` — 4-panel: argmin geodesic and
  truth_q0_cost, vs ω-mag and vs ω-dir.
- `results/s004/landscape_panels.png` — 3 seeds × 6 representative
  perturbations grid of cost-vs-q0_geo scatter; visual confirmation of
  no-basin / multi-basin behaviour.

## Out of scope

- **Q5b — cohort-scale "alignment cost argmin = truth-q0 at baseline" check.**
  Seed 28's failure at baseline is striking but n=1; need to know how many
  of the 76 alignable seeds share this pathology. Cheap experiment (no
  ω-perturbation, just one Sobol grid per seed): probably ~5 min wall.
- **Alignment cost with stricter spec threshold.** Tighter `SPEC_THRESHOLD`
  reduces the number of constraint epochs but constrains them harder; could
  in principle change the multi-basin pathology on seed 28. Probably a side
  diagnostic, not a main thread.
- **Anchor sensitivity.** All cost values here use the m103 anchor selection
  (savgol-smoothed brightest spec peak with epoch-tiebreak). A different
  anchor changes the constraint set and so the cost. Out of scope for Q5.
- **Joint (q0, ω) local-descent validation (Q4a).** This is the obvious
  next experiment given s003 + s004's combined verdict.

## Cross-references

- **s001 (Q1):** alignment cost at truth is well-behaved on the 76 seeds
  where it's defined (median 1.5e-4 per constraint, similar growth pattern
  to surrogate). s004 confirms the *value at truth* is faithful (smoke test
  passes), but shows the *argmin location* is unreliable.
- **s002 (Q2):** at fixed truth-ω, surrogate-MSE argmin = truth-q0 on 8/8
  PA-stratified seeds. s004 demonstrates alignment cost FAILS this property
  on seed 28 (argmin 176.8° from truth even at truth-ω) — a direct
  qualitative gap with surrogate-MSE.
- **s003 (Q3):** narrow surrogate-MSE truth-ω tube (~1° dir, ~2-5% mag).
  s004 shows the alignment cost tube is strictly narrower (0° dir on
  overlapping seed 6, 1° dir on seed 18, no tube at all on seed 28) and
  has more cells far from truth.
- **concepts/known_pathologies_to_revalidate.md:** "m103 alignment cost
  bridge from peak structure to attitude" — re-validated negative under
  post-fix truth: alignment cost provides no useful seeding signal for ω
  search, and on a fraction of seeds it does not even minimise at truth at
  exactly-correct ω. The known-pathologies entry should be updated to mark
  this as RESOLVED-NEGATIVE.
- **memory: feedback_stop_cost_shape_engineering.md** — reinforced. Even
  the structurally motivated alignment cost (peak bisector dot) does not
  outperform surrogate-MSE; engineering more cost shapes from peak structure
  is unlikely to recover what the surrogate cannot.
