---
title: "m136 — Kernel-consistency cost on constraint-epoch subset (FAILED)"
type: experiment
sources:
  - "notebooks/inversion/score_kernel_consistency.py"
  - "data/results/inversion_diagnostics/kernel_consistency/seed_091/result.npz"
  - "data/results/inversion_diagnostics/kernel_consistency/seed_091/summary.json"
related:
  - "[[m118_cost_comparison]]"
  - "[[kernel-factorization]]"
  - "[[m135_alignment_cost_forensics_constrained_anchor]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[twin-degeneracy]]"
created: 2026-04-28
updated: 2026-04-28
confidence: high
---

# m136 — Kernel-consistency cost on constraint-epoch subset (FAILED)

#dead-end (this specific shape) — but lesson is high value.

## Hypothesis

m118's kernel-factorisation pattern (`q_delta[N_DIRS, N_MAGS, K_EPS, 4]`
precomputed once per seed) is cost-agnostic. m118 used it for IPL/alignment
costs that all failed. **Hypothesis**: feeding it the surrogate full-LC MSE
*restricted to a K-epoch constraint set* gives us m118's kernel speedup +
m135's surrogate-cost ranking power.

Algorithm:
1. Step 1 anchor selection (from m135 v4): mid-30% × `mag<11` for surrogate
   validity → most-constrained epoch by Q-count → q_anchor_set (Q=54 on
   seed 91, ep273 mag=6.05).
2. Constraint epochs = spec_peaks ∪ (uniform-sampled informative-mid-LC),
   target 100 — 40 actual on seed 91 (50% mag 9-11, 47% mag 5-9).
3. Kernel: 2000 dirs × 20 mags × 40 eps × 4 quat. Forward+backward branch
   (constraint epochs span before/after anchor). Built in 187s on Pool(8).
4. Score per (dir, mag): for each q_anchor in Q_a, propagate q_world[ep] =
   q_anchor · q_delta[dir, mag, ep], surrogate-mag at all K eps, MSE
   residual. Take **min over q_anchor** → cost(dir, mag) → min over mag →
   cost(dir).
5. Rank.

## Result (seed 91 only)

| metric | value |
|---|---|
| rank-1 by cost: ω-direction error | **165.23°** |
| ranks 2-5 by cost: ω-direction errors | 116°, 45°, 126°, 128° |
| closest-to-truth ω-direction in grid | 2.18° (rank **1587/2000** by cost) |
| Spearman ρ(cost, ω-direction error) | **+0.005** |
| `cost(truth-near) - cost(rank-1)` gap | 19.63 vs 9.87 (~2× higher) |
| total wall | 352 s (kernel 187s + Step1 5s + score 159s) |

The cost surface is ~uncorrelated with truth-direction. NOT anti-correlated
(that would be a negative signal, exploitable by sign flip); it's *noise*.
For comparison on the same seed:

| cost / source | rank-1 ω-direction error |
|---|---|
| m103 alignment cost (current production) | 174.83° |
| m103 `lofi_mse` (full hi-fi LC MSE, 300 candidates) | 174.00° |
| m135 surrogate full-LC MSE on lofi-300 pool | **2.76°** |
| m135 constrained-anchor (oracle |ω|, full-LC MSE) | **2.18°** |
| **m136 kernel-consistency (this experiment)** | **165.23°** |

## Why it failed — TWO compounding causes (deeper than first claimed)

### (a) Order-statistic noise attractor at large |Q_a|

`cost(dir, mag) = min over q_anchor ∈ Q_a of MSE_K(q_anchor · q_delta[dir, mag, eps])`. With |Q_a|=54 and K=40 epochs, the min-of-54-noisy-MSEs is several σ below the mean by pure order statistics — best-of-54 picks up enough random fits per (dir, mag) that the cost-surface variance across grid points is dominated by the noise-floor realisation, not by truth-correlated structure. m118-style anchor sets (2-4 IPL centroids per epoch) don't trigger this because min-of-4 has only ~0.5σ bias.

### (b) The deeper cause: Q_a samples across DISCONNECTED COMPONENTS

The per-epoch attitude level set `L(t) = {q : m(q; t) ≈ observed_mag(t)}` is **not a connected blob in SO(3)** — it's a disjoint union of small localised regions, each a different physical hypothesis:

- different facet aligned to PAB (±X panel glint, dish glint, etc.)
- different shadowing configuration (which faces are lit, which are self-shadowed)
- ±X (and other body-symmetry) twin partners
- diffuse-mix configurations of multiple partly-illuminated faces

m118 already encoded this with `loop_count` per epoch (count of pab-contour connected components — many epochs have 2-4+ loops). The Q=54 q-set on seed 91 ep273 is a discrete sample drawn FROM SEVERAL OF THESE COMPONENTS, not a coherent local blob. Treating the set as a single locus to anchor from — and min-collapsing across it — picks "the hypothesis that happens to look least bad under this (dir, mag)", which is essentially a random choice when no hypothesis is correct for the given grid point.

There is no natural "single anchor to fit from"; each component is its own *mutually-exclusive theory* about the satellite's physical configuration at the anchor epoch. The right algorithmic shape recognises this and intersects component memberships across epochs, not anchor-fitted MSEs.

### What m135 Finding 2 does differently

m103's lofi pool produces UNIQUE (q0, ω) candidates from grid search + lofi peak-match re-rank — no post-hoc anchor fitting, no min-over-set. Each candidate is its own row carrying full signal. Re-ranking by surrogate full-LC MSE works because every (q0, ω) is independently scored against all 500 epochs.

### The pattern that breaks

```
cost(dir, mag) = min over q_anchor ∈ Q_a of f(q_anchor · q_delta[dir, mag, eps])
```
fails when:
- |Q_a| is large (order-statistic noise floor)
- AND/OR Q_a is sampled from disconnected components (hypothesis conflation under the min)
- AND K is small relative to |Q_a| (insufficient epochs for the noise to average out per anchor before the min)

## What this rules out

The shape "use the q_anchor SET from Step 1 + multi-epoch consistency cost"
is dead at this Q size. To resurrect it would require:
- Drastically shrink |Q_a| (e.g. cluster the 54 to ~5 representatives via
  quaternion k-means), OR
- Drastically grow K (constraint epochs) to >> Q so the "best of Q" min
  doesn't wash out signal — likely K=200+ epochs per seed.
- Either path is more analysis-heavy and less likely to deliver than the
  proven m135 Finding 2 (full-LC MSE on m103's existing lofi pool).

## What this does NOT rule out

- m135 Finding 2 (single-line m103 patch: re-rank lofi-300 by surrogate
  full-LC MSE) — **still the highest-leverage open option**, untested on
  the 5 failure seeds.
- The kernel-factorisation pattern itself — still useful for any cost where
  post-anchor fitting is small (m118-style 2-4 centroids) or absent
  (m135-style unique-candidate re-rank).
- **The proper m136 successor: surrogate-attitude-isoshell with per-epoch
  L(t) clustering + trajectory-membership cost.** See [[surrogate-attitude-isoshell]]
  and [[attitude-level-set-disconnection]]. Concretely:
  1. **Per-epoch L(t) construction** by surrogate forward sampling (64k q's
     spanning SO(3), threshold against observed mag). One scalar function
     m(q; t) per epoch — no body-frame "many-candidate-manifolds"
     ambiguity in attitude coordinates. ~5 sec/epoch on Pool(8).
  2. **Cluster L(t) into connected components** C_1(t), ..., C_k(t). Each
     component is a hypothesis (specific facet aligned to PAB, specific
     shadowing config, twin partner, etc.). Quaternion DBSCAN or geodesic
     k-means.
  3. **Trajectory-membership cost** per (ω-direction, |ω|): propagate
     identity to each constraint epoch t; cost = sum over t of
     dist(q_world(t), nearest component of L(t)). NO anchor enumeration.
     Twin-survives-at-glint-but-not-mid-brightness IS the discriminator.
  4. Total: ~8-10 min/seed Step-1 + clustering + ~1 min scoring on the
     kernel. Tractable.

## Files

- `notebooks/inversion/score_kernel_consistency.py` — algorithm
- `data/results/inversion_diagnostics/kernel_consistency/seed_091/result.npz`
- `data/results/inversion_diagnostics/kernel_consistency/seed_091/summary.json`
- Run log: `data/results/inversion_diagnostics/kernel_consistency/seed_091_run.log`

## Bugs caught during the run (fixed in commit)

1. **First-attempt anchor selection picked ep283, mag=15.73 (Q=21)** — same
   v3 dim-saturation failure mode the original m135 ran into. Cause: I
   removed the bright filter on the misreading that "bright = least
   constrained" implied "no brightness guard at all". The user clarified:
   bright filter ≠ surrogate-validity dim guard. Re-added `max_obs_mag=11`
   default with documented rationale (surrogate-validity, not constraint).
2. **Kernel propagation crashed** on negative `dt` (constraint epochs
   before anchor): `propagate_attitude` rejects out-of-span eval points.
   Fixed via forward/backward branching: backward branch propagates
   `(q_id, -ω, |dt|)` per the involution `q(-t, ω) = q(t, -ω)`.
3. `score_constrained_anchor.py` `--include-truth-mag` default was True
   AND argparse signature made it impossible to disable — flipped to
   default-False with paired `--no-include-truth-mag` flag.

## Next move (post-context-budget)

- Run m135 Finding 2 (`score_lofi_surrogate.py`) on the 5 failure seeds
  (47, 51, 79, 84, 89). All have fresh `lofi_ckpt.npz` from this session's
  re-runs. ~4 min total. This is the highest-leverage open lever.
- If Finding 2 holds on failure seeds: 10-line patch to m103's NM_TOP
  re-rank, end-to-end test on Band-D seeds.
- If Finding 2 does NOT hold: revisit the per-epoch q-set version of m136
  (no min-over-anchor at anchor; per-epoch consistency check).
