---
title: "s061c — multi-step threading on seed 28 (high-\|ω\| where s059j ω-grid failed)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s061c_seed28_thread.py
  - notebooks/inversion/survey/results/s061c_seed28_thread/seed028/summary.json
related:
  - s061a — 1-step smoke
  - s061b — multi-step on slow seed 89 (no cluster death)
  - s059j — single-anchor ω-grid FAILED on seed 28 due to grid quantization
  - s060_anchor_topology — seed 28 anchor cluster structure
created: 2026-05-10
updated: 2026-05-10
confidence: medium-high (clear cluster-death signal; truth-lineage tracking
  reveals an architectural failure mode)
---

## TL;DR

s061c stresses the threading architecture on seed 28: high |ω|=1.43 dps,
narrow basin, where s059j FAILED due to ω-grid quantization (7° spacing
→ 5° truth offset → ρ=38 at exact q_a). The threading approach replaces
the ω-grid with the connectability tube; ω falls out of per-pair
geodesic. Anchor at t=312 (|C_t|=43 post-canonicalize, 19 clusters),
ω_max=1.5 dps (tube/truth ratio = 1.04 — barely sufficient).

**Two findings, opposite signs:**

1. ✅ **Cluster-disappearance IS REAL on seed 28.** Right thread:
   19 → 14 anchor clusters alive at step 20 (5 die). Left thread:
   19 → 11 (8 die). Brightness drops to 0.1% at step 11 right; the LC's
   information density on this fast tumbler kills inconsistent
   trajectory hypotheses. **The user's idea is validated as an
   architectural mechanism.**

2. ⚠️ **Truth-LINEAGE tracking decouples from spatial truth.** Truth
   cluster (id=6, size 3 at anchor, 19.6° from any survivor) survives
   in BOTH threads but its descendants wander to ≥150° from real truth.
   At many steps, truth is nearer to a NON-truth-cluster lineage's
   descendants. Random-walk uniform tube sampling causes lineage tags
   to drift relative to actual SO(3) position. The lineage tag is NOT
   a reliable proxy for "is this thread following truth?"

## What

The same threading pipeline as s061b, but on seed 28's high-|ω| regime
where the tube is barely big enough to admit truth's motion:

- Pool: N=100k (10× s061b — needed because |C_t=312| is small and
  truth is ~20° from any anchor survivor at this seed).
- Anchor t=312: dim-extreme (mag=20.62, dimmest LC point), s060
  diagnostic shows |C_t|=5 at N=25k. With N=100k pool we get |C_t|=43
  (post body-twin canonicalize), 19 clusters at 40° threshold.
- ω_max=1.5 dps: cohort-safe upper bound. Truth |ω|=1.43 dps. Tube
  radius 10.82°; truth step 10.37°. Ratio 1.04.
- Steps: 20 each direction.

## How

```bash
python experiments/s061c_seed28_thread.py \
    --seed 28 --anchor-t 312 --n-pool 100000 \
    --omega-max-dps 1.5 \
    --n-per-anchor 100 --max-per-step 2000 \
    --n-steps 20
```

Same primitives as s061b (`thread_step`, `body_twin_canonicalize_q`,
`greedy_cluster`); n_per_anchor bumped 30→100 because anchor has fewer
survivors and we want enough child density per anchor.

## Result

### Anchor diagnostics

| metric | value |
|---|---|
| pool size | 100,000 |
| raw \|C_t=312\| | 43 (post-canon) |
| n_clusters at 40° | 19 |
| cluster sizes top-6 | [5, 5, 4, 3, 3, 3] |
| truth in cluster | 6 (size 3) |
| truth dist to nearest survivor | **19.56°** |

The anchor cloud is poorly-resolved on seed 28: even at N=100k pool,
truth is 19.56° from any survivor — much worse than seed 89's 8.6°.
This reflects seed 28's narrower basin (|ω|=1.43 dps, tighter sensitivity
to attitude). Doubling pool to N=200k+ would help.

### Cluster lifelines

| step | t | clusters alive (right) | t | clusters alive (left) |
|---|---|---|---|---|
| 1  | 313 | 18/19 | 311 | 13/19 |
| 5  | 317 | 18/19 | 307 | 13/19 |
| 10 | 322 | 17/19 | 302 | 12/19 |
| 11 | 323 | 14/19 | 301 | 12/19 |
| 15 | 327 | 14/19 | 297 | 12/19 |
| 16 | 328 | 14/19 | 296 | 11/19 |
| 20 | 332 | 14/19 | 292 | 11/19 |

**Right: 5/19 clusters die by step 20.** **Left: 8/19 die.** The deaths
cluster at brightness-extreme epochs (step 11 right t=323 sees 200,000
candidates → 243 survivors = 0.12% retention; step 16 left t=296 sees
~1% retention). At those moments, multiple anchor clusters' tubes have
NO brightness-consistent point at the next epoch — kill.

Compared to seed 89 where 0/5 clusters die over 30 steps each, seed 28
shows the architecture's discrimination working as designed.

### Brightness retention extrema (right thread)

| step | t | retention |
|---|---|---|
| 4-9 | 316-321 | 9-38% |
| **11** | **323** | **0.12%** |
| 12 | 324 | 4.0% |
| 16 | 328 | 5.7% |
| 17 | 329 | 3.0% |

t=323 is the cut-point where 99.88% of tube candidates fail brightness.
Across multiple anchor clusters, this is enough to drop several to zero
descendants.

### Truth-lineage tracking (the failure mode)

| direction | step | d_truth (any cloud) | d_truth (truth-cluster lineage only) |
|---|---|---|---|
| right | 1 | 24.5° | 24.5° |
| right | 5 | 22.6° | 22.6° |
| right | 10 | 30.6° | 57.7° |
| right | 11 | 32.8° | **115.7°** |
| right | 15 | 29.2° | **137.7°** |
| right | 20 | 9.9° | **127.7°** |
| left | 1 | 26.4° | 26.4° |
| left | 10 | 34.6° | 86.2° |
| left | 14 | **1.4°** | 131.7° |
| left | 16 | **5.3°** | INF (lineage dead) |
| left | 20 | 10.8° | INF |

The story: truth moves through SO(3) over the threading window. SOME
anchor cluster's lineage tracks truth well at each step (`d_truth` is
the min over all surviving lineages, often only 1-30°). But the
TRUTH-cluster's lineage descendants do NOT track truth — they wander
50-150° away by step 10+. By step 16 left, the truth cluster has zero
surviving descendants entirely (its tube emptied), yet some other
cluster's lineage is 5° from truth.

**Root cause: random uniform tube sampling causes lineage tags to drift
relative to SO(3) position.** Each step, every anchor survivor produces
n_per_anchor random children in its tube; over k steps the "spatial
neighborhood of cluster c" becomes a random-walk path of length k·r_max.
With r_max ≈ truth motion, neighborhoods rapidly overlap; "what cluster
did this descendant come from" becomes lossy as a proxy for "where on
SO(3) is it now."

## Why this matters

### Architectural mechanism: validated

- **Cluster-disappearance is a real, measurable signal on hard seeds.**
  Seed 28 thread kills 5-8 of 19 anchor clusters; this prunes the
  candidate space by ~30-40% via dynamics-consistency alone, no
  ω-grid needed.
- **Brightness has high information density on fast tumblers.** Single
  epochs (step 11 right, step 8 left) kill 99% of tube candidates.
  Threading concentrates the cumulative constraint power.

### Implementation: lineage tag is broken at this regime

- The lineage-tracking proxy used by s061b (and partially by s061c) for
  "is the truth cluster still alive" fails on hard seeds because the
  tube allows random-walk drift. A surviving descendant of cluster_id=6
  may be 150° from truth at step 15, while a surviving descendant of
  cluster_id=2 may be 1° from truth.

- The user's geometric framing — "truth must thread continuously" — is
  correct, but the random-walk realization of it doesn't preserve the
  cluster identity. **A constant-ω propagation per (q_a, ω) hypothesis
  would preserve cluster identity by tying each thread to a specific
  trajectory.**

### Reframe for the next experiment

The threading architecture wants two changes after s061c:

1. **Constant-ω propagation per (q_a, ω) candidate.** Instead of random
   walk tube at each step, sample (q_a, ω) PAIRS from the anchor × tube
   product space, propagate each as a fixed (q_a, ω) trajectory through
   N epochs, brightness-filter at every step. Survivors are
   (q_a, ω) pairs consistent with ALL N epochs.

2. **Anchor tighter ω_max via polhode prior.** s055a gives per-seed
   |ω| at ~25% MAPE. With a prior estimate of e.g. 1.4 dps for seed 28,
   ω_max = 1.05 × 1.75 = 1.84 dps still safely covers truth, but each
   tube concentrates around the prior. Per-pair derived |ω| (in
   constant-ω propagation) lands inside a much smaller cell.

3. **Truth-cluster identification via spatial proximity at every step,
   not via lineage tag.** At each step, find the cluster CONTAINING the
   nearest survivor to truth-q. That's the "true thread" at that step.
   The lineage tag at the anchor remains useful as a SEED for hypotheses
   but isn't a reliable identifier downstream.

These changes turn s061 from a random-walk particle filter into a
candidate-trajectory enumerator — closer to s060's design but with ω
sampled in a tube rather than Newton-shot from a pair.

## Numbers

- Anchor: seed 28, t=312, mag=20.62, |C_t|=43 (canon), 19 clusters.
- ω_max: 1.5 dps (cohort-safe top), truth 1.43 dps, ratio 1.04.
- Truth dist to nearest anchor survivor: 19.56°.
- Truth cluster: id=6, size 3.
- Steps: 20 each direction.
- Compute wall: ~10 minutes total.
- Cluster deaths: right 5/19 (26%), left 8/19 (42%).
- Brightness retention extreme: 0.12% (right step 11, t=323).
- Truth-lineage final distance: right 127.7°, left dead by step 16.

## Artefacts

- `experiments/s061c_seed28_thread.py` — script.
- `results/s061c_seed28_thread/seed028/summary.json` — per-step lineage
  + truth-distance data.

## Out of scope

- Constant-ω propagation variant (the natural follow-up — see "Reframe").
- Hi-fi rendering of (q_a, ω) candidates that survive threading. Without
  constant-ω structure, no single (q_a, ω) is identifiable.
- Polhode-prior ω_max tightening.
- Newton-shoot ω from anchor pair (s060 architecture, distinct path).
- Cluster-shape extent (variance/spread) tracking — would supplement
  count-based death detection.
- Cohort run.

## Cross-references

- s061b — same architecture on slow seed 89; no cluster death. The
  contrast establishes that the architecture's discrimination scales
  with tube/truth ratio AND with seed |ω| (LC information density).
- s059j_cloud_data_omega_grid.md — the ω-grid architecture this is
  trying to replace; failed on seed 28 due to 7° quantization.
- s060_multi_anchor_design.md — Newton-shoot ω from a pair; uses large
  Δt instead of small.
- `feedback_finite_diff_small_geodesic_noise.md` — s057f failure at small
  Δt at v1 substrate. v2 substrate + brightness-doubly-filtered
  endpoints does NOT have the same noise floor; s061c shows the
  multi-step constraint accumulates real discrimination here.
