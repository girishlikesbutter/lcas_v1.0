# LCAS Inversion — Dead Ends

> Approaches that were explored and conclusively failed. Each entry includes *why* it fails, so future sessions don't waste time revisiting these ideas.

---

## Global Optimizers on 6D Joint Space (Series 01)

**Tried:** Differential Evolution, CMA-ES, dual annealing, basin-hopping, multi-start L-BFGS-B, alternating att/omega, brute-force grid (13,824 points), decoupled grid. Every standard global method. Up to 50k evaluations, 45 min wall time.

**Why it fails:** Joint convergence basin is ~5° attitude × 0.02 dps omega. In 6D, the basin volume fraction is negligibly small (~10^-8). No global optimizer can find it without structural exploitation.

**Replaced by:** Physics-informed candidate generation (iso-brightness, glint constraints).

---

## Sequential Cascade Filtering (Series 02)

**Tried:** Multi-epoch brightness screening. Each epoch filters ~100x, chain K epochs for 100^K reduction.

**Why it fails:** Combinatorial explosion. 100 survivors at each of K epochs = 100^K candidate pairs. Even with FFT-based omega bounding and triplet consistency checks, pairing is prohibitive.

**Replaced by:** Peak-anchored approaches (use ~3 brightness peaks, not all 500 epochs).

---

## Lo-fi / Hi-fi Intermediate Brightness Scoring (Series 05, micro13-14)

**Tried:** Score graph paths by brightness residual at intermediate epochs between peaks.

**Why it fails:** Lo-fi scores are nearly uniform across all feasible paths — no discrimination. Truth path ranked ~10th percentile. Hi-fi rescoring also fails (truth rank ~13k/121k). The problem: arbitrary omega vectors can produce plausible-looking brightness at individual epochs; only the full trajectory shape discriminates.

**Replaced by:** L-conservation filter (avoids brightness evaluation entirely).

---

## Multi-Epoch LC Shape for Winding Discrimination (Series 07a, micro21-22a)

**Tried:** Score staircase winding solutions by lo-fi MSE at all ~78 intermediate epochs.

**Why it fails:** All staircase omegas have 13-25° direction error. The bridge constrains only the two endpoints, NOT the rotation axis. Every winding's intermediate trajectory is wrong (MSE 2.0-4.0 vs 0.16 for truth). Multi-epoch scoring IS discriminating between windings, but NONE match the observed LC.

**Key insight:** The problem is omega *direction*, not winding *number*. Scoring is sound only when the omega direction is already correct.

---

## Sequential Staircase Heuristic (Series 07, micro17-18)

**Tried:** Find winding solutions sequentially: find ω_k, set lower bound = |ω_k|, guess next = ω_k + (2π/dt) × axis.

**Why it fails:** Heuristic assumes constant rotation axis across windings. Under Euler dynamics with triaxial inertia, the axis shifts with |ω|. On leg 1 (dt=721s), jumped from 0.25 to 3.28 dps, missing 5 of 12 families including the true omega at ~2.08 dps.

**Replaced by:** Band-sweep with multi-start random directions (micro20). 10 random starts per 0.5 dps band finds all families.

---

## Magnitude-Only Dedup for Bridge Solutions (Series 08, micro26)

**Tried:** After band-sweep, dedup by |ω| within 0.05 dps — keep one solution per magnitude.

**Why it fails:** At each |ω|, multiple directionally-distinct bridge solutions exist. Keeping one arbitrary direction discards the correct one. L-conservation then fails (rank 423/1088 instead of 1).

**Replaced by:** Direction-aware dedup (angular distance < 5° AND |ω| within 0.05 dps).

---

## PA-Mode Bridge Substitution (Series 08, micro28)

**Tried:** Replace Euler dynamics with principal-axis closed-form propagation (763× faster).

**Why it fails:** PA-mode produces different omega directions and winding topology than Euler dynamics. 0/30 trials correct. IS-901 has triaxial asymmetry = 0.556 — not axisymmetric enough.

---

## PAB-Circle Seeding (Series 09, micro38 + micro43)

**Tried (micro38):** Generate 30K seeds on PAB circles for all 14 normals, run L-BFGS-B iso-brightness. **Result:** 10× worse than random SO(3) seeding (5.0° vs 0.5°). Seeds diluted across 14 normals.

**Tried (micro43):** Focused single-normal seeding with oracle normal. **Result:** Still 18× worse (12.4° vs 0.69°). L-BFGS-B immediately leaves the PAB circle — the iso-brightness objective doesn't respect the geometric constraint.

**Why both fail:** Unconstrained optimization discards PAB structure. The PAB circle is a *hard constraint*, not a good starting point for unconstrained search. **The phi-sweep approach (micro42b) is the correct way to exploit PAB circles** — it stays on the circle by construction.

---

## Bridge-LC Omega Selection Pipeline (Series 10b, micro52-59)

**Tried:** Bridge between two glint-circle candidates → derive omega → score by LC residual → select correct omega.

**Why it fails — the chicken-and-egg problem:** The bridge generates correct omega (0.3° direction error in the pool), but LC scoring ranks it at #34K/90K because the bridge's anchor attitude (q1) is always far from truth (14-76°). Wrong q1 → wrong predicted brightness → bad LC score, regardless of omega quality. Finding a good q1 requires the correct omega (the phi sweep only works with known omega, per micro51b). **This circularity is fundamental, not fixable by scoring variants.**

**Confirmed across 9 experiments in micro59:** stratified winding selection, anchor-centered scoring, phi-sweep-improved q1, alignment filtering, grid search, combined specular+anti-glint scoring, focused bridging, arrival error filtering — all fail for the same root cause.

---

## Brightness Profile Shape Scoring (Series 10b, micro59 step 6)

**Tried:** For a candidate omega direction, propagate from anchor attitude, predict brightness at ±5 epochs around nearby peaks, compare shape with observed.

**Result:** True omega at rank #64/400 with 3 peaks at ~10° grid spacing. Close peaks help, distant peaks (>1000s) add noise due to attitude drift.

**Why it's insufficient:** Rank #64/400 means ~64 NM starts needed. micro50c already showed 0/200 random NM starts converge. Shape scoring narrows the search ~6× but the basin is ~2°, so you'd need ~5° accuracy to have a shot. At 10° grid spacing, the residual error after shape ranking is still ~5-10° — marginal at best.

**Status:** Explored but insufficient as standalone. Could potentially contribute as a component in a denser search, but the CasADi formulation is the more promising path.

---

## Glint Recurrence for Omega Estimation (micro52b)

**Tried:** Use timing between same-group glint recurrences to estimate |ω|.

**Why it fails:** rho=0.146, p=0.28. The relationship between recurrence interval and |ω| is NOT 2π/|ω| because of triaxial polhode geometry — the body-frame omega vector precesses, so the time between successive alignments of the same normal with PAB is not periodic.

---

## Single-Bridge Pair Screening (Series 08, micro27)

**Tried:** Use one cheap bridge solve per candidate pair to screen N² combinations.

**Why it fails:** Full bridge solves at dt~500s cost ~11s each (not 160ms as at dt=50s — cost scales with ODE integration steps). Cheap alternatives tested: geodesic metric (instant but no discrimination), PA-mode (5ms but wrong solutions), bounded L-BFGS-B (275ms but no convergence). No viable cheap metric found.

---

## Brightness Surface Path Matching — Direct Extraction (micro104)

**Tried:** Extract ω-direction from peak crossing geometry — measure how the body-frame PAB traverses brightness lobes, use crossing velocity to constrain ω₀. Proposed 10-100× search space reduction via DIRECT EXTRACTION from peak shapes.

**Why direct extraction fails:**

1. **Kinematic error in concept:** dp_B/dt = Ω_L × p_B where Ω_L = R(t)·ω_body(t) — NOT ω_body. Cannot extract ω₀ without knowing the attitude.

2. **Peak FWHM too noisy:** 27% median error on crossing speed from discrete 7.2s sampling.

**Constraint satisfaction variant ALSO fails (micro105-106):** The revised approach (check if a candidate ω₀ produces alignment at multiple peaks via delta-q + ψ sweep) appeared to work in a POC [inline] with exact truth omega (14/14 peaks aligned), but micro106 showed it FAILS at grid-level precision: a 1.16° ω direction error → only 6/14 peaks aligned. The quaternion error from ~1° omega error accumulates to 60-90° over the ~3600s observation, making late peaks unmatchable. This approach requires ω precision (~0.01°) unavailable from grid search. See [[crossing-geometry-scoring]] wiki branch (#dead-end).

**Data:** `data/results/inversion_diagnostics/micro104_crossing_diagnostic.npz`

---

## Buggy Results: L-Parameterization Basin (Series 03)

**exp_conservation_and_L_param.py** Part A labels "attitude held at truth" but does NOT actually fix attitude — all 6 params are free. The conclusion "L basin NOT wider" is **INVALID**. The experiment needs to be redone properly. This is a significant gap: the physics argument for L-parameterization is compelling (L errors don't compound) but the empirical test was never completed.
