---
title: "Surrogate-driven attitude isoshells (option 2 + option 3)"
type: branch
sources:
  - "raw/inversion_diagnostics/m118/seed_014/summary.json"
  - "raw/inversion_diagnostics/m119v2/seed_014/summary.json"
  - "raw/inversion_diagnostics/m120/seed_000/summary.json"
  - "raw/inversion_diagnostics/m120/seed_014/summary.json"
  - "raw/inversion_diagnostics/m120/seed_027/summary.json"
  - "raw/inversion_diagnostics/m120/seed_046/summary.json"
  - "raw/inversion_diagnostics/m120/seed_058/summary.json"
  - "raw/inversion_diagnostics/m120/seed_075/summary.json"
related:
  - "[[m118_cost_comparison]]"
  - "[[m119_attitude_isoshell]]"
  - "[[m119v2_attitude_isoshell]]"
  - "[[m120_tumbling_competitors]]"
  - "[[m121_basin_width_metric]]"
  - "[[m122_hessian_curvature]]"
  - "[[pab-contour-isoshell]]"
  - "[[pab-contour-phase-angle-limitation]]"
  - "[[surrogate-model]]"
  - "[[surrogate-de-search]]"
  - "[[shadow-asymmetry]]"
  - "[[multi-solution-philosophy]]"
  - "[[surrogate-truth-offset]]"
created: 2026-04-15
updated: 2026-04-29
confidence: high
---

# Branch: Surrogate-driven attitude isoshells

## Status: #parked-2026-04-29 — ranking discrimination validated, but cost-shape engineering on top of this substrate is suspended

**2026-04-29 strategic reframe (read first):** This branch's per-epoch attitude-isoshell substrate is a real geometric object, and the multi-seed rank-0 result against tumbling competitors ([[m120_tumbling_competitors]]) is real. But the algorithmic shape that was being engineered ON TOP of it (kernel-consistency in m136, H1 epoch-density in m138) keeps producing costs that are **anti-correlated with truth** — same pathology class as alignment cost (`project_alignment_cost_anti_truth.md`, `feedback_min_over_anchor_pattern.md`, `feedback_h1_cost_shape.md`). The only thing that actually works on the m138 H1 grid pool is **substituting surrogate full-LC MSE** for the H1 cost (m138 surr-rerank, 2026-04-29) — at which point you're using the H1 grid as a sampling scheme, not a discriminator.

**Don't design more aggregator costs over this substrate.** The production path for this branch — if it ever gets revived — is `H1 grid → surrogate full-LC MSE rerank`, not `H1 grid → some_new_cost`. The cheaper, equally-validated path that does NOT need this branch is `m103's existing 26-candidate pool → surrogate full-LC MSE rerank` (m133, single-seed validation in m134/m135). Acceptance bar is ρ < 4 (Band A∪B); see `feedback_rho_band_yield_metric.md`.

**Original validation status preserved below for context, but it is NOT a green light to keep building cost surfaces here.**

## Status (historical): #validated (multi-seed, ATT_FAIL cohort) — post-[[m120_tumbling_competitors]] multi-seed

## 2026-04-15 — multi-seed test PASSED

Truth rank 0/10004 under all 13 cost variants for all 6 seeds {0, 14, 27, 46, 58, 75} covering the full ATT_FAIL cohort plus seed 14 as the single-seed reference. See [[m120_tumbling_competitors]] for the multi-seed table.

Key multi-seed findings beyond seed 14:
- **close_omega bucket ranks 8001–8004 across all seeds** — 2° ω perturbation with random q0 is universally the worst bucket; not a seed-14 artifact.
- **near_truth bucket ranks 1 across all seeds** — basin width is ≤20° in both q0 and ω regardless of seed.
- **m115 basin ranks scale with ω error, not q0 twin-distance** — seed 14 (ω_err 0.34°) m115 basins rank 17–83, seed 0 (ω_err 2.9°) basins rank 7–10, seed 27 (ω_err 3.08°) basins rank 227–1861. Twin (±X q0 mirror) cannot be distinguished from truth when ω is tight (known [[twin-degeneracy]]).
- **Truth mean_L1 score 0.053–0.057 across all seeds** — surrogate noise floor is seed-independent.

Multi-seed gate cleared. Branch usable for ATT_FAIL-cohort inversion at single-candidate cost (pool scoring). Still needs integration with search algorithm (see Next).

## 2026-04-15 afternoon — [[m120_tumbling_competitors]] tumbling-competitor test PASSED

The v2 weakness (static ω=0 grid vs tumbling truth) was addressed by [[m120_tumbling_competitors]]: 10004 TUMBLING (q0, ω) candidates, including the truth, three m115 basins, and 10000 random/perturbed candidates across 4 buckets (uniform 6-DOF, close-ω + random-q0, close-q0 + random-ω, near-truth perturbations at {2°, 5°, 10°, 20°}). Each propagated through 500 epochs, surrogate-scored, residualised, 13 cost variants.

**Truth rank = 0 / 10004 under ALL 13 cost variants.** This is the first honest discrimination result for this branch. Key structural findings:

- **near_truth bucket best rank 1** — small (q0, ω) perturbations approach but do not beat truth. The cost landscape has a narrow deep basin around truth.
- **close_omega bucket (tight ω, random q0) best rank 8004 / 10004** — knowing ω exactly without q0 is WORSE than random 6-DOF tumbling. Evidence the cost is jointly sensitive to (q0, ω), not separable.
- **close_q0 bucket (tight q0, random ω) best rank 137-208** — small q0 error with random ω also badly ranked. Good discrimination.
- **uniform bucket best rank 13-69** — random 6-DOF tumbling is comfortably far from truth.
- **m115 basins** (b0 near-truth q0_err=2.3°, b1 twin q0_err=178°, b2 alt q0_err=179°): ranks 17-571. Consistent with their hi-fi MSE of 0.16 — they are wrong-basin competitors that truth beats.

The branch can now be trusted at single-seed confidence. Multi-seed validation is the gate to `#validated` without qualification.

## 2026-04-15 evening — honest re-scope after v1 retraction

The original [[m119_attitude_isoshell]] POC was **retracted** because it mixed `m046_trajectories.npz` (1-hr window) with `setup_experiment()` (6-hr window). The session's "threshold REFUTED / score CONFIRMED" verdict was built on corrupted geometry.

[[m119v2_attitude_isoshell]] re-ran with internally-consistent geometry at today's 6-hr grid. Findings:

- **Truth residual at honest geometry:** median 0.043 mag, p90 0.112 mag, max 0.549 mag. The surrogate is actually *accurate* at truth when the inputs are consistent. The v1 claim that threshold framing was refuted was itself an artifact of mixed geometry — the dim-epoch residual explosion was mostly bogus k2.
- **Truth rank = 0 / 60,000 under all 13 cost variants.** Headline survived the fix.
- **But the POC design is weak.** Every grid "candidate" is a static ω=0 rotation. Truth is a 500-epoch tumbling trajectory. Truth beating 60k static competitors is near-tautological (500 DoF vs 1). This does NOT validate surrogate-residual cost as a discriminator against the real enemy — nearby WRONG tumbling trajectories.

The branch stays open at **lower confidence** until a tumbling-competitor test runs. Previous threshold/score/level-set conclusions should be disregarded; they were conclusions about corrupted geometry.

## The question

Can we replace the pab-contour-based IPL framework with a physics-correct attitude-isoshell framework using the trained surrogate?

## Motivation

[[m118_cost_comparison]] showed that IPL-centroid alignment costs have a fundamental ~25° noise floor at seed 14 because the pab-contour assumes `k1 = k2 = h` (zero phase) and lo-fi (no shadows). Truth does not pass through IPL centroids — median offset is 25.7°. No cost variant over IPL centroids puts truth at rank 1.

The surrogate model (see [[surrogate-model]]) takes `(k1_body, k2_body, panel_deg, dish_deg, distance_km) → magnitude` and was trained on full hi-fi (ray-traced shadows + Ashikhmin-Shirley BRDF). It *is* the correct `B(k1_body, k2_body, ...)` function that the pab-contour approximates.

## The proposal

### Option 2: use surrogate as the brightness function

The pab-contour is `B(n_body)` — one scalar per body-frame direction, with `k1 = k2 = n_body`. Replace this with `B(k1_body, k2_body)` — two separate inputs on the 2-sphere, correct phase angle, correct shadows.

We don't need to materialise this as a table; the surrogate already computes it in ~5 ms per eval, batch throughput much better.

### Option 3: per-epoch attitude isoshells

For each constraint epoch t with known `(k1_J2000(t), k2_J2000(t), panel(t), dish(t), distance(t))` and `observed_mag(t)`:

1. Sample ~60k attitudes `R_i` on a uniform SO(3) grid.
2. For each `R_i`: compute `k1_body = R_i @ k1_J2000(t)`, `k2_body = R_i @ k2_J2000(t)`.
3. Batch-call surrogate on all 60k `(k1_body, k2_body, panel, dish, distance)` tuples.
4. Level set: `{R_i : |surrogate_pred − observed_mag(t)| < σ}` — this is a 2D manifold in SO(3), the proper attitude-space version of an IPL.

Store the surviving `R_i` indices (or explicit rotations) per epoch.

### Multi-epoch intersection

A candidate `(q0, ω)` is consistent with the observations iff its propagated attitude trajectory passes through the isoshell at every constraint epoch (within tolerance). Multiple intersection components correspond to degenerate solutions — the multi-solution picture is made explicit geometrically.

## Cost picture

- **One-time per satellite:** nothing needs to be pre-computed (surrogate is already trained). The satellite-level B function is "call surrogate."
- **Per-seed, per-epoch:** 60k surrogate evals batched. README estimate: 500 evals in ~5 ms ⇒ 60k ≈ 600 ms per epoch.
- **Per seed at 255 constraint epochs:** ~150 s of isoshell extraction. Plus SO(3) grid setup and intersection logic.
- **Storage:** per-epoch isoshell is a list of kept R_i indices. With ~10% keep rate at a 60k SO(3) grid, ~6k rotations per epoch × 255 epochs × small int = a few MB per seed.

## Why this is interesting

- Same mental model as IPL: constraint sets, isoshells, intersection.
- Built on physics that isn't lying (phase angle + shadows correct via surrogate).
- Multi-solution enumeration comes for free — degeneracies appear as multiple connected components in the intersection.
- Complements [[surrogate-de-search]]: DE is point-estimate iterative search; isoshells are constraint-set geometric.

## Caveats

- **Shadows in the surrogate:** we believe the surrogate was trained against ray-traced shadows, per its README. Should verify the training data covers the full `(k1, k2)` × panel × dish space densely enough for the shadow boundaries to be captured accurately.
- **SO(3) grid resolution:** 60k samples on SO(3) gives ~5° angular resolution. Tight enough for coarse level-set extraction; may need adaptive refinement near boundaries.
- **Intersection strategy:** the intersection of 255 isoshells is not trivially a set operation. Either:
  (a) propagate trajectory from each candidate `(q0, ω)` and check each epoch — puts the intersection into a scoring framework, still compute-cheap
  (b) full set-intersection with SO(3) indexing — more machinery, gives a geometric picture

## Next experiment (post-[[m121_basin_width_metric]])

Branch validated multi-seed ([[m120_tumbling_competitors]]) and the basin has been characterised ([[m121_basin_width_metric]]). The remaining open question is **search integration**: how do we go from "cost correctly ranks truth" to "search algorithm finds (q0, ω) from scratch"?

Priority order:

1. **Search integration POC.** Pair the isoshell cost with a search scheme. Two candidates:
   - **(a) SO(3) × (ω-grid) isoshell intersection.** Sample ~60k SO(3) × N_ω candidates, score via surrogate, extract level-set, intersect across epochs. Concrete geometric story; multi-solution-friendly.
   - **(b) Surrogate-DE with the new cost.** Replace m115's per-epoch hi-fi-MSE cost with surrogate-residual mean_L1 and see if DE converges more often. Cheaper to implement (one-line swap).
   Pick based on which aligns with the strategist's near-term plan for handling ω-direction (which is the tight-basin axis per m121).

2. **Basin-informed DE mutation strategy.** m121 showed ω-direction basin is a narrow SLAB with seed-specific preferred axis. A DE mutation that weights ω-direction perturbations toward the principal-axis direction should outperform isotropic mutation. **Hessian-at-truth now available: [[m122_hessian_curvature]] saved `hessian.npz` for seeds {14, 27, 46, 74, 93}**, giving a 6×6 symmetric mass matrix usable directly as a DE-mutation preconditioner (scale mutations by `eigenvector_i / sqrt(eig_i)`) or HMC inverse-mass. Online estimator to recover the mass matrix *without* truth is still open — local cost-curvature samples during DE's first generations would give a field estimate. Caveat from m122: the Hessian-derived stiff axis does NOT match m121's finite-scale preferred axis for 2/3 ATT_FAIL seeds (scale-dependent anisotropy). Mass-matrix preconditioning from m122's Hessian is the right fit for *local* polishing; DE global-phase mutation may want the m121 finite-scale anisotropy instead.

3. ~~**OK-cohort generalisation of basin width.**~~ **DONE** — [[m122_hessian_curvature]] ran Hessians on OK-cohort seeds 74 and 93. Basin widths qualitatively identical to ATT_FAIL (eigenvalue spread within 2.11×; hyp3 CONFIRMED). Branch generalises beyond ATT_FAIL. ATT_FAIL classification is NOT explained by tighter local basin geometry.

Deferred (done or no longer blocking):
- ~~Tumbling-competitor test~~ → [[m120_tumbling_competitors]] passed multi-seed.
- ~~Surrogate-at-truth fidelity audit~~ → inline 4-seed audit complete, MAE ~0.03 mag.
- ~~m115 re-fix~~ → audit resolved, m115 was not buggy (see 2026-04-15 AUDIT-RETRACTED entry in log).

## Decision tree (post-v2)

- **If (1) shows truth rank 0–10 under tumbling competitors:** real signal, proceed to multi-seed.
- **If (1) shows truth mid-pack vs tumbling competitors:** the static-grid rank was tautological and the branch closes as a cost-landscape story rather than a discrimination tool.
- **If (1) shows truth top-1 but gap is narrow (distance < 2× noise floor):** the discrimination is real but weak; keep as complement to [[surrogate-de-search]] rather than standalone.

## 2026-04-28 — m136 lesson on the algorithmic shape

[[m136_kernel_consistency_failure]] tried a shortcut on this branch: m118-style kernel-factorisation with a single-anchor q-set sampled at one constraint epoch (Q=54 on seed 91), evaluating surrogate-MSE consistency at K=40 OTHER constraint epochs and **min-collapsing** over the 54 anchors per (ω-dir, |ω|) grid point. Result: rank-1 at 165° from truth, Spearman ρ ≈ 0.

The lesson generalises and matters for any successor algorithm on this branch. **The per-epoch attitude level set L(t) is super-disconnected** — a disjoint union of small components, each a different physical hypothesis (different facet aligned to PAB, different shadowing config, ±X twin partner, etc.). m118 already encoded this via `loop_count` per epoch on the lo-fi side.

So the proper formulation of "intersect across epochs" is **not** anchor-fitted MSE summed over epochs — it's **trajectory-level component membership**. See [[attitude-level-set-disconnection]] for the full concept. Concretely:

```
For each candidate (ω-dir, |ω|):
    propagate identity to each constraint epoch t  →  q_world(t)
    cost = Σ over t of  dist( q_world(t), nearest component of L(t) )
                                              ^^^^^^^^^^^^^^^^^^^^^^^
                                NO anchor enumeration, NO min over a chosen anchor set
```

The candidate (q0, ω) trajectory either threads through the union of per-epoch component sets or it doesn't. The ±X twin survives at glint epochs but fails at mid-brightness epochs where its propagated q lies outside every component of L(t) — that's the multi-epoch discriminator.

Cost: ~5s/epoch surrogate scan × ~100 epochs ≈ 8-10 min/seed Step-1 + clustering, then ~1 min scoring on the kernel. Tractable.

This note should be the first thing the next iteration on this branch reads after the m119/m120/m121 history below, because it directly informs the *cost-shape* design (which the multi-seed validation didn't pin down).

## 2026-04-29 — H1 instantiation: cost-shape pathology bypassed by surrogate-MSE substitution

[[m138_isoshell_h1_pilot]] tested an H1 cost ("densest spot in q0-rewind cloud, count distinct epochs in the ball around it") on the failure cohort. Worked on seed 91 (truth at rank 2, 1.76°), failed on seed 47 — the cost rewards extreme-|ω| candidates at either end (slow-spin pile-up from cloud collapse, fast-spin pile-up from sweep). [[m138_seed47_lombscargle_bracket]] confirmed: 12 truth-near joint candidates exist in the 80k harmdiv pool but H1 ranks the best at position 22,231/80,000.

[[m138_seed47_surr_rerank]] (today) ran Round A on the cached pool: substitute surrogate full-LC MSE for H1 cost. Result: joint truth-near candidate at **rank 5**, twin-q0 candidate at **rank 1** (ω_dir 13.9°, q0_to_twin 26°). Spearman ρ(H1 cost, surr_MSE) = -0.30, p≈0 — H1 cost is anti-correlated with truth-fit quality, which is the second documented case of this pattern after [[alignment-cost]].

Implication for the algorithmic shape: **the trajectory-level component-membership cost from the 2026-04-28 note above is NOT needed.** Surrogate-MSE on the H1-pool is a positive-definite cost that surfaces the right candidates without needing to redesign the cost-shape. The H1 cost's epoch-density mechanism still has standalone value as an independent ω-evaluation signal (parked for later, as a tiebreaker or volume-normalised variant), but the production path for this branch is now:

```
H1 grid (harmdiv-LS |ω|-base + 1000-dir Fibonacci sphere + eps_cluster=10°)
  → Stage 3 produces 80k (q0_centroid, ω) candidates
  → Surrogate-MSE re-rank (Pool(8), 16 min/seed)
  → top-K (K=5..30) hand-off to m115 hi-fi DE polish
  → m126 hi-fi convergence
```

q0_centroid is unreliable across all ranks (joint candidates show q0_to_truth scattered 47°–177°) — m115 must re-optimise q0 from scratch per ω candidate.

Pending: hi-fi m115 verification on seed 47 + cohort {51, 79, 84, 89} generalisation (~64 min total wall).

## Historical context (pre-m119)

Original motivation below is preserved; the level-set framing is what m119 refuted. The discrimination evidence still holds and is strengthened by m119's rank-0 mean_L1 result.
