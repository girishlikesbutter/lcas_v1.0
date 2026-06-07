---
title: "Known pathologies to re-validate — buggy-era claims, NOT inherited as fact"
type: concept
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# Known pathologies — to re-validate, NOT to inherit

This is the catalogue of inversion-side claims that were established under the **buggy forward model** (pre-2026-04-30) and have NOT been re-validated under correct truth. Each is treated as an open question. Do not cite any of these as fact in the survey's experiment write-ups.

The purpose of this page is to prevent the survey from accidentally re-discovering the same claims as fresh insights, while also preventing them from being smuggled in as starting assumptions. If the survey wants to re-test one, the entry below tells you what to measure and where the buggy-era reference lives.

---

## Cost-surface claims

### 1. "Alignment cost is anti-truth on failure seeds" (m135)

**Buggy-era claim:** On 5 random-cohort failure seeds (47, 51, 79, 84, 89), the m103 alignment cost reported `cost(geo_best) << cost(truth)` by 4–22 orders of magnitude, with `geo_best` 8–82° from truth. Conclusion at the time: alignment cost is anti-correlated with truth, denser sampling cannot fix it.

**Status post-fix:** Open. m141 confirmed the pathology survives on seed 6 under correct truth on a single-seed probe; cohort-scale extension is exactly what survey question Q1 (s001) measures.

**To re-validate:** Score alignment cost at `(q0_truth, ω_truth)` for each of the 100 seeds; compare to the cost surface's argmin location. If `cost(truth) > cost(argmin)` and the argmin is geometrically far from truth, the pathology is confirmed under correct truth.

### 2. "Surrogate-MSE rerank rescues seed 91 to rank-1 at 2.76°" (m135 lofi-pool)

**Buggy-era claim:** Re-ranking m103's lofi-300 pool by surrogate full-LC MSE places truth-near at rank 1, vs alignment cost's rank-1 anti-truth.

**Status post-fix:** Refuted as standalone patch. m143 found that under correct truth on seed 91, alignment cost already places jointly-truth-near at rank 2 and surrogate-MSE rerank DEMOTES it. Surrogate-rerank is reliable PRE-truncation (m144) but not generally post-truncation.

**Implication:** Do not use surrogate-MSE rerank as the survey's first hammer without re-measuring its argmin behaviour against truth.

### 3. "DE over q0 at fixed truth-ω converges reliably" (Roberto Step 4 / m115 / m126)

**Buggy-era claim:** Holding ω near truth, surrogate-MSE q0-search via DE + L-BFGS-B converges reliably to truth-q0.

**Status post-fix:** Refuted on seed 91 (m145). 0/100 DE runs at 10 truth-near ω inputs landed q0_err < 10°; the surrogate landscape's global minimum was at q0_err ≈ 135°. m126 6-DOF polish drifted ω from 3.46° to 41.32°.

**Open question:** Is the deceptive q0=135° attractor a seed-91 specific pathology or a population-wide structural feature of the surrogate landscape under correct truth? Survey question Q2 (surrogate landscape probe) measures this on PA-stratified seeds.

---

## Basin / Hessian claims

### 4. "Truth-basin widths: q0 ≳ 5°, ω-dir < 0.1°, ω-mag < 0.25%" (m121)

**Buggy-era claim:** Surrogate-residual cost basin widths around truth on 5 baseline seeds.

**Status post-fix:** Open. The basin definition assumes the surrogate has its argmin at truth, which m145 questions on seed 91. Pre-conditions for re-running m121 under correct truth: confirm via Q2 that surrogate's argmin IS at truth on the seed in question.

### 5. "Truth is essentially the surrogate minimum" (m123)

**Buggy-era claim:** L-BFGS polish from truth converges back to truth on all 5 seeds tested.

**Status post-fix:** Refuted by m145 on seed 91 (truth is NOT the surrogate's global minimum under correct truth there). Open elsewhere.

---

## Bridging-radius claims

### 6. "m115's q0-bridging radius is < 30° empirical, ω-bridging marginal at 15°, reliable at 5°" (m134)

**Buggy-era claim:** Empirical bounds on how close DE input ω needs to be to truth-ω for the q-from-ω solver to land truth-q0.

**Status post-fix:** The Q-FROM-ω part of this is refuted (m145 — DE fails even at w_err = 3.4°). The ω-bridging-radius half is open and probably also wrong.

**Implication:** Do NOT use these bounds as a constraint on survey design.

---

## "Solved seed" claims

### 7. "Seed 6 is solved (m098 → m126 wrapped, hi-fi 87% improvement)"

**Buggy-era claim:** Seed 6's m126 wrapped pipeline produced a Band-A solution.

**Status post-fix:** Refuted by m141. Under correct truth, seed 6 is upstream-FAIL: m103's geo_cost ranking buries the truth-near ω at rank 9 behind 8 candidates with w_err 53–74°.

### 8. "Seed 91 is solved (Play-1 textbook win, m115 sort_by=surr_q0polish_mse)"

**Buggy-era claim:** End-to-end Band-A on seed 91.

**Status post-fix:** Refuted by m145. Under correct truth, seed 91 is downstream-FAIL — patched m103 surfaces truth basin to m115, but q-from-ω solver fails.

### 9. "8/11 baseline seeds wrapped-pipeline-improved (m126 cohort)"

**Buggy-era claim:** Wrapped pipeline default over plain m115 on cohort.

**Status post-fix:** Open. The cohort was selected and judged under buggy truth. Re-classification under correct truth is one of the cleanest "audit" experiments the survey could do — re-render the recorded `(q0_winner, ω_winner)` from each cached `result.json` under post-fix forward model and re-score against post-fix truth. Most will fall to Band D. The survivors (if any) are real signals.

---

## Cost-shape engineering claims (m136–m138)

### 10. "min-over-anchor washes signal" (m136)

**Buggy-era observation:** The min-over-anchor pattern in m103's phi-sweep produces a noise attractor when the candidate pool is large and the kept-point K is small.

**Status post-fix:** Open. The observation is a property of the cost-shape arithmetic, not the forward model directly, but it interacts with which costs are "honest" under correct truth. Defer until Q1 + Q2 give the cost-surface map.

### 11. "Kept-point UNION beats centroids; mag-grid ≤5% spacing" (m138)

**Buggy-era observation:** Best practices for cost-shape design discovered in the H1 isoshell pilot.

**Status post-fix:** Open. Same caveat as item 10. The methodology may transfer; the seed-by-seed numerics certainly do not.

---

## What's NOT in this list (because it survives the bug fix)

- The convention bug itself, the q→q* per epoch claim, the renderer formula. See `quaternion_convention.md`.
- The surrogate's bridge-independence and its training-time MAE. See `surrogate_model.md`.
- The ρ-band convention. See `rho_band.md`.
- The (q0, ω) coupling and time-stretch invariance of |ω|. See `q_omega_coupling.md`.
- The structural existence of ω-sign degeneracy and IS-901 ±X twin degeneracy (the STRUCTURAL claim, not the buggy-era seed-specific numbers). See respective concept pages.
- All methodology rules in `~/.claude/.../memory/MEMORY.md` (BLAS threading, save-intermediate, etc.).
- The phase-angle modulation of the bug effect (m146): high-PA seeds had larger bug delta, low-PA smaller. This is a property of the bug + geometry, validated under post-fix vs buggy comparison; it survives.

## How to use this page

When designing a survey experiment, scan this list for related claims. If any are relevant, decide explicitly: am I re-testing this claim under correct truth? Or am I avoiding it because it's not on the critical path? Either is fine; what's not fine is implicitly inheriting the buggy-era number as a starting assumption.

---

## ADDENDUM (2026-05-12) — second propagator sign bug fixed; s001-s066 numerics need replication

A second sign bug in the propagator was discovered and fixed on 2026-05-12 (commits — discovery: s066; fix + validation: s067 / d5705ff). Pre-fix `dq/dt = +0.5 ω ⊗ q (LEFT)` had the opposite sign of the textbook conv-(a) kinematic for the codebase's q-interpretation, producing trajectories where L_J2000 drifted 36-137% over m048's 60-min LCs. Real torque-free physics requires exact conservation. Independent textbook integrator + s067's 4-gate regression test confirm the post-fix propagator now matches SPICE pxform (1e-10/s) and conserves L_J2000 (DOP853 noise floor).

**Implication for this page:** every "buggy-era claim" listed above was measured under the pre-2026-04-30 conjugation bug. Almost all were re-measured between 2026-04-30 and 2026-05-12 under the second sign bug — those re-measurements are also stale. Anything cited as "post-fix" between those dates (e.g. m139, m140, m141, m143, m144, m145, m146, s001..s066) is post-conjugation-fix but pre-physics-fix.

**Replication scope (decided 2026-05-12):**

The minimal architecture-validation set (s068-s071) re-runs:
- s011 cohort baseline (density-recoverable trajectory class fraction)
- s059k Band A on cohort, no oracle injection
- s064 Jacobi-coord LM polish
- s062a Jacobi closed-form ω(t) cohort validation

Other s001-s066 findings are deferred. When a future experiment cites a pre-fix numerical claim, replicate that specific finding then.

**What survives without replication:**
- Methodology rules (surrogate-first, multi-mag-start, polhode-basis polish, ρ-band, multi-solution acceptance, body-twin halving, Pool BLAS=1, no-oracle).
- Forward-model qualitative findings (multi-sol structure exists, polhode invariants exist, basin scaling with |ω|/pol_diam, polhode tangent is the soft direction).
- Surrogate trainability and bridge-independence (`(k1_body, k2_body) → mag` doesn't depend on the propagator).
- The s067 4-gate regression test as the canonical pre-merge check for any future propagator change.

**What does NOT survive:**
- Specific seed numerical values (q0_err, ρ, basin widths, polhode diameters, cluster IDs, rank positions).
- Quantitative comparisons between pre-fix and post-fix data (apples to two different oranges).
- Any architecture-yield headline that rests on specific seed numerics ("Band A on seed 89", "5/50 unique clusters", etc.) — qualitatively likely to replicate; numerically unverified.
