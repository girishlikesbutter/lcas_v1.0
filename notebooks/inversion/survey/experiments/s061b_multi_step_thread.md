---
title: "s061b — multi-step cloud threading on seed 89 (30 epochs each direction)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s061b_multi_step_thread.py
  - notebooks/inversion/survey/results/s061b_multi_step_thread/seed089/summary.json
related:
  - s061a — 1-step smoke (substrate)
  - s061c — same architecture on seed 28 (high-\|ω\| contrast case)
created: 2026-05-10
updated: 2026-05-10
confidence: high (mechanism + truth-tracking measurement on slow seed)
---

## TL;DR

Extends s061a from 1 step to 30 steps each direction on seed 89. Each
step: re-tube around current cloud, brightness-filter, subsample to
MAX_PER_STEP=2000 to keep wall bounded. Track which anchor cluster each
survivor descends from (lineage tag).

**Result on slow seed 89**: with ω_max=0.5 dps (~2× truth |ω|=0.24 dps),
all 5 anchor clusters retain descendants through 30 steps in BOTH
directions. Truth-cluster lineage tracks truth at 4-9° (left) growing to
44° (right) over 30 steps. Brightness filtering does real work at some
epochs (step 9 right kills 95.5% of tube candidates), but per-cluster
contributions are even enough that no cluster's count drops to zero.

**Cluster-disappearance does not occur on slow seeds with a 2×-truth
budget.** The connectability + brightness filter is too generous when
ω_max is multiples of truth |ω|. Need tighter budget OR harder seed for
the architecture to discriminate.

## What

Multi-step extension of s061a:

```
for k = 1..N_steps:
    target_t = anchor_t + sign·k
    for each q in current_cloud:
        sample n_per perturbations in tube radius ω_max·|target_t-prev_t|
    children = q ⊗ delta_q
    survivors = brightness_filter(children, target_t)
    if |survivors| > MAX_PER_STEP:
        subsample randomly to MAX_PER_STEP
    current_cloud = survivors
    track lineage (which anchor cluster each survivor descends from)
```

Subsampling necessary because |C| × n_per_anchor explodes (~30^k without
cap). Capping to 2000/step means every step has at most 60,000 candidates
(2000 × 30/anchor) regardless of history.

## How

```bash
python experiments/s061b_multi_step_thread.py \
    --seed 89 --anchor-t 208 \
    --omega-max-dps 0.5 \
    --n-per-anchor 30 --max-per-step 2000 \
    --n-steps 30
```

Anchor: t=208 (clean-dim from s060, |C_t|=48, 5 clusters).
ω_max: 0.5 dps = ~2× seed 89 |ω|=0.24 dps. Tube radius 3.6° per step.
Truth step on seed 89: 1.75° per step.

## Result

### Cluster lifelines

| direction | anchor clusters | step 30 alive | truth alive | truth d_min final |
|---|---|---|---|---|
| RIGHT (t=208 → t=238) | 5 | 5 / 5 | YES | 44.1° |
| LEFT  (t=208 → t=178) | 5 | 5 / 5 | YES | 9.2° |

All 5 anchor clusters retain at least one descendant through 30 steps in
both directions. No cluster-disappearance.

### Truth-distance over steps (right thread)

| step | t | n_surv | truth_cl_count | min dist to truth |
|---|---|---|---|---|
| 1 | 209 | 1188 | 561 | 7.2° |
| 5 | 213 | 2000 | 822 | 9.3° |
| 10 | 218 | 1306 | 631 | 18.2° |
| 15 | 223 | 2000 | 1020 | 21.4° |
| 20 | 228 | 2000 | 955 | 26.8° |
| 25 | 233 | 2000 | 920 | 36.3° |
| 30 | 238 | 2000 | 986 | 44.1° |

Truth-distance grows monotonically — the cloud is spreading roughly
linearly with step count. With ω_max=0.5 dps, the tube radius is 3.6°
per step. Cumulative cloud radius after N steps ~ N · 3.6°. By step 30
that's 108° — not yet saturating SO(3) (max 180°).

### Brightness retention (% candidates passing per step, right thread)

| step | t | retention |
|---|---|---|
| 3 | 211 | 87% |
| 8 | 216 | 76% |
| **9** | **217** | **4.5%** ← extreme |
| 10 | 218 | 2.2% |
| 11 | 219 | 4.6% |
| 14 | 222 | 4.6% |
| 25 | 233 | 20% |

Step 9-14 sees brightness retention crash to 1-5%. This is real
discrimination — that LC region (t=217-222) carries strong attitude
info. Yet all 5 clusters survive because each contributes ~600-900
descendants (subsampled) per step; even at 4.5% retention, each cluster
keeps 30+ survivors after subsampling.

## Why this matters

### What works

1. **Truth tracks across 60 epochs (~7 minutes wall clock of trajectory).**
   On left thread, the truth-cluster lineage stays within 4-10° of truth
   throughout 30 steps. Right thread degrades to 44° by step 30 but
   never loses truth.

2. **Brightness IS doing real work at selected epochs.** Steps 9-14
   right thread show 1-5% retention — that's 95-99% of tube candidates
   killed by brightness alone. The LC has strong information at those
   moments.

3. **Subsampling+lineage tracking keeps wall bounded.** ~5 sec/step
   Pool(1) × 30 steps × 2 directions = 5 minutes total. Fast enough
   for cohort-scale.

### What doesn't

1. **No cluster-disappearance with this budget.** With ω_max=0.5 dps =
   2× truth |ω|, the tube is wide enough that every anchor cluster
   finds SOME brightness-consistent descendant at every step. The
   architecture's "kill clusters whose tube has no brightness-consistent
   neighbor" mechanism fires zero times.

2. **Lineage decoupling at higher step counts.** Random uniform tube
   sampling means a survivor's lineage tag (which anchor cluster it
   descends from) decouples from its actual SO(3) position over time.
   By step 20+ the truth-cluster lineage is no longer spatially
   localized; tracking lineage stops being a proxy for "truth's current
   position."

3. **Slow tumblers are easy mode.** Brightness barely changes between
   adjacent epochs (60-90% retention typical). The connectability
   constraint at this regime is doing very little; brightness alone
   would produce nearly the same survivor set without the tube
   restriction.

### Implications

The architecture's discriminating power is bounded by:
1. **Tube size relative to truth motion** — closer to 1× = more pruning.
2. **Adjacent-epoch brightness change** — proportional to LC information
   content at that time.
3. **Multi-step accumulation of constraints** — but with random-walk
   tube sampling this saturates because each step's constraint is local
   to the previous cloud.

For seed 89 specifically (slow, 5 clean clusters at anchor), threading
preserves truth but also preserves multi-solution candidates. The
multi-solution acceptance criterion is intact.

## Numbers

- Anchor: seed 89, t=208, |C_t|=48, 5 clusters, top-3 mass 0.92.
- Tube: ω_max=0.5 dps × Δt=7.2s = 3.6° radius per step.
- Truth motion: 1.75°/step (|ω|=0.24 dps).
- Tube/truth ratio: 2.05.
- Steps: 30 each direction.
- Compute wall: ~3 minutes total.
- Subsample: 2000/step.
- Cluster surviving counts at step 30: right 5/5, left 5/5.
- Brightness retention range: 2.2-93%.
- Truth-min-distance range: 4.0° (left step 5) - 44.1° (right step 30).

## Artefacts

- `experiments/s061b_multi_step_thread.py` — script.
- `results/s061b_multi_step_thread/seed089/summary.json` — full per-step
  state including lineage counts.

## Out of scope

- Tighter ω_max (e.g. 0.27 dps = 1.1× truth — would test cluster death).
- Constant-ω propagation per (q_a, ω) hypothesis (s061 architectural
  variant — eliminates random-walk drift).
- N=100+ step threading (long-baseline; expensive).
- Cohort-scale thread on the s011 9-seed pilot.
- Cluster-shape spread tracking (what's the SO(3) extent of each
  cluster's descendants over steps).
- ω-derivation quality once lineage decouples — not measured here.

## Cross-references

- s061a — 1-step substrate.
- s061c — same architecture, hard seed 28 where cluster death DOES happen.
- s060_multi_anchor_design.md — Newton-shoot variant (no random walk).
- s059k — production architecture being compared against (yields Band A
  on seed 89 via single-anchor + ω-grid + multi-mag-start).
