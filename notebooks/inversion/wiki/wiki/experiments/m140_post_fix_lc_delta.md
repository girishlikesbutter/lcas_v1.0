---
title: "m140 — Post-fix LC delta on m048 seed 6: quantifying the convention-bug magnitude error"
type: experiment
sources:
  - "data/results/inversion_diagnostics/m140_post_fix_lc_delta/m140_lc_delta.py"
  - "data/results/inversion_diagnostics/m140_post_fix_lc_delta/m140_seed006_delta.npz"
  - "data/results/inversion_diagnostics/m140_post_fix_lc_delta/m140_seed006_lc_overlay.png"
  - "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed006_buggy.npz"
  - "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed006.npz"
  - "data/results/inversion_diagnostics/convention_bug_audit/step3_seed006_lc_compare.npz"
related:
  - "[[m139_convention_bug_fix]]"
  - "[[quaternion-convention]]"
  - "[[m048-migration]]"
  - "[[surrogate-model]]"
  - "[[basin-of-attraction]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# m140 — Post-fix LC delta on m048 seed 6

#validated #post-bug-fix

## TL;DR

Render m048 seed 6 end-to-end with the FIXED forward model (post-[[m139_convention_bug_fix]]) and compare against the cached buggy `mag_hifi` from `traj_seed006.npz`. **The delta is large:** RMS 2.18 mag, max 8.46 mag, ρ = 43.6 (Band D). Bright epochs took 5–8 mag hits; dim epochs ~1 mag. The bug was not a small perturbation — it was producing physically scrambled LCs. Every previous "OK" classification on seed 6 (and by extension every basin classification across the m070→m138 cohort) was sorting candidates against a target ~11× outside the ρ < 4 acceptance bar.

## What

Seed 6 was the canonical Step-3 validation seed in [[m139_convention_bug_fix]] and is also the textbook "previously-solved" seed in the m115/m126 cohort — m098 rescued it from PARTIAL → OK; m126 reported 87% hi-fi MSE improvement under the wrapped pipeline. Choosing it for m140 lets us measure the bug's effect on a known-good seed: if the delta is large *here*, the bug is large everywhere.

The comparison artefacts already existed before m140 was written:
- `step3_seed006_lc_compare.npz.mag_way_a` — post-fix correct LC (Way A renderer, m139 Step 3)
- `traj_seed006.npz.mag_hifi` — pre-fix buggy LC (cached by m048 generator)

Both were rendered with identical SPICE state, identical `(q0, ω0, I_tensor)`, identical articulation convention (`SP_North/South=0°, AD_East/West=15°`), identical observation_times grid (500 epochs over 1-hr window starting `et=634182313.41`), identical shadow ray-tracing engine, and identical Ashikhmin–Shirley BRDF. **The only difference is which q-trajectory was fed to the renderer**: pre-fix `q_b` (conv-(b)) vs post-fix `q_a` (conv-(a)).

m140 loads both arrays, computes per-epoch delta + summary statistics, and produces an overlay plot.

## Numbers

| metric | value |
|---|---|
| RMS (fixed − buggy) | **2.181 mag** |
| max \|diff\| | **8.457 mag**  (epoch 196, t=1414 s) |
| median \|diff\| | 1.088 mag |
| 90th pct \|diff\| | 3.412 mag |
| 99th pct \|diff\| | 6.872 mag |
| hifi MSE | 4.758 mag² |
| **ρ = √MSE / 0.05** | **43.63 (Band D)** |

LC ranges differ structurally (different bright/dim extremes, not the same trajectory perturbed):
- buggy: `[7.5309, 15.9123] mag`
- fixed: `[6.5169, 16.4320] mag`

Per-magnitude-band breakdown (binned by buggy `mag_hifi`):

| band | n | median \|diff\| | p90 | max |
|---|---|---|---|---|
| bright (mag<8) | 7 | 5.32 | 7.78 | 7.84 |
| mid (8–12) | 41 | 3.69 | 5.86 | 7.34 |
| dim (≥12) | 452 | 0.95 | 2.94 | 8.46 |

Damage concentrates on the bright specular peaks — exactly the epochs that drive every alignment-cost surface, every MSE basin, every glint-filter discrimination. The dim epochs have lower median delta but the absolute max (8.46 mag) lives in the dim regime, indicating the buggy LC's "dim" regime occasionally lit up dramatically under the correct geometry.

The peak-count metadata in the regenerated truth NPZ (m140 follow-up, 2026-04-30 same session) shows the buggy NPZ had 32 hi-fi peaks while the fixed NPZ has 30 — i.e., the bug was creating/eliminating ~2 spurious peaks on this seed alone.

## Why this matters

### 1. The buggy attractor structure was not just a relabeling

The framing in [[m139_convention_bug_fix]] is "internally consistent buggy forward model": both truth-generator and candidate-renderer applied the same bug, so candidates that matched buggy truth really were attractors of the buggy forward model. m140 confirms that framing — the bug is at the propagator boundary, not in the cost or scoring layers. But the *attractor structure* of the buggy forward model is meaningfully different from the real forward model. With ρ = 43.6 between the two LCs, no candidate's basin under the buggy model can be assumed to correspond to a basin under the real model. **All previous basin classifications, all twin/decoy diagnoses, all alignment-cost-anti-truth findings (m135) are now empirical questions, not established facts.**

### 2. Bright-peak displacement breaks every cost-shape result

Cost surfaces built from MSE, alignment, IPL centroids, and surrogate residuals are dominated by bright-peak terms. The bug displaced bright peaks by 5–8 mag — far larger than any noise threshold, basin width, or cost-shape gradient ever measured. The "alignment cost is anti-truth" finding ([[m135_alignment_cost_forensics_constrained_anchor]]), the H1 cost-shape pathology ([[m138_seed47_lombscargle_bracket]]), the cost-at-truth probes — all were measured against displaced peaks. The probes may or may not survive on correct truth.

### 3. The surrogate is fine

A clarification that emerged from the post-m140 discussion: the surrogate maps `(k1_body, k2_body) → mag` (or similar attitude-frame inputs), NOT `(q0, ω, t) → mag`. There is no propagator in the surrogate's training pipeline. So the surrogate's accuracy on body-frame coordinates is *unaffected* by the convention bug. What was wrong pre-fix was the bridge from `(q0, ω, t)` to `(k1_body, k2_body)` — the propagator + renderer chain — not the surrogate itself. Post-fix, the bridge produces correct body-frame coordinates and the surrogate evaluates them accurately. **No surrogate retraining is required.** This collapses one of the original "scheduled follow-ons" from m139.

## Implementation

`m140_lc_delta.py` is a 110-line read-and-compare script — no rendering, no SPICE, no satellite model. Loads `traj_seed006_buggy.npz.mag_hifi` and `step3_seed006_lc_compare.npz.mag_way_a`, asserts shape + grid alignment + q0 match (max diff 1.11e-16), computes diff statistics and per-magnitude-band breakdown, plots overlay. ~5 sec wall.

## What this enables

m140 unblocks the next step in the post-fix recovery: regenerate the seed-6 truth NPZ in canonical form (so downstream pipelines find it via `lib/traj_source.load_truth`) and re-run the inversion pipeline against it. That work happens immediately after m140 in the same session — see the post-fix seed-6 pipeline run for outcomes. The truth-NPZ regeneration policy chosen: rename the buggy NPZ to `traj_seed006_buggy.npz` and let `m048_generate_trajectories_v2.py` run via its `MICRO48_WORKER_SEED` env-var single-seed mode. 53 sec wall.

## Out of scope here

- **Generalisation to other seeds.** m140 measured the delta on one mid-phase, "previously solved" seed. Whether the delta scales similarly on high-phase / constraint-poor / surrogate-DE-failure seeds is unmeasured. Likely seed-dependent; some seeds may have smaller deltas (if the conv-(a)/(b) flip happens to land closer to truth for that geometry), some larger.
- **Full m048 cohort regeneration.** Out of scope — costs ~4 worker-hours on Pool(4), scheduled for a follow-on session.
- **m046 cohort regeneration.** Same — separate session.

## Cross-references

- [[m139_convention_bug_fix]] — the audit + fix experiment that m140 follows up on.
- [[quaternion-convention]] — the concept page covering conv-(a) vs conv-(b).
- [[m048-migration]] — the cohort whose truth NPZs are now subject to regeneration.
- [[surrogate-model]] — clarification: surrogate is bridge-independent, no retraining needed.
- [[basin-of-attraction]] — basin widths characterised in m121/m122 are buggy-model properties; revalidation against correct truth is open.
