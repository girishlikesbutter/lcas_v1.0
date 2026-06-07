---
title: "m143 — Seed 91 generality check: m103 ranks truth-ω at rank 1, but no truth-q0 anchor exists in either pool"
type: experiment
sources:
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/multi_phi_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/lofi_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/lofi_surr_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/geo_surr_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/lofi_surr_ckpt.npz"
  - "notebooks/inversion/score_geo_surrogate.py"
  - "notebooks/inversion/score_lofi_surrogate.py"
related:
  - "[[m141_seed6_postfix_pipeline]]"
  - "[[m142_seed6_postfix_surrogate_rerank]]"
  - "[[m139_convention_bug_fix]]"
  - "[[m135_alignment_cost_forensics_constrained_anchor]]"
  - "[[m138_seed47_lombscargle_bracket]]"
  - "[[m115_de_bridging_radius]]"
  - "[[surrogate-rerank]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# m143 — Seed 91 generality check + multi-seed lofi-pool pathology

#post-fix-generality #seed91 #seed6 #lofi-pool-truth-coverage

## TL;DR

Repeated the m141 + m142 protocol (truth regen → m103-only pipeline → surrogate-MSE rerank) on m048 seed 91 — the [[m135_alignment_cost_forensics_constrained_anchor]] textbook Play-1 win under buggy truth (ρ=0.22 Band-A). Two distinct findings emerged that together flip the strategic picture:

1. **Seed 91's m103 ALREADY ranks the truth-ω at rank 1** (w_err=4.65° at multi_phi rank 1, surr_mse rank 1). Surrogate-MSE rerank gives the **same** answer as alignment cost on seed 91. **Completely different qualitative pattern from seed 6** (where alignment ranks the truth-near ω at rank 9/26 and surrogate-MSE demotes it to 15/26).
2. **But on both seeds, m103's phi-sweep fails to find a truth-q0 anchor**: seed-91's 4 phi anchors for the truth-ω all have q0_err=120–162°; seed-6's single phi anchor for the rank-9 truth-ω has q0_err=130°. **m103 is finding truth-ω fine on at least some seeds, but its q0-parameterisation cannot reach truth-q0 even when it knows the right ω.**

A second-level diagnostic: re-ran `score_lofi_surrogate.py` on the lofi-300 pools for both seeds (the m135 rescue lever). On **seed 91**, the lofi-300 pool contains **exactly one jointly-truth-near candidate** (q0_err=25.24°, w_err=28.83°) — at alignment-cost rank 2 but **surrogate-MSE rank 26** (DEMOTED). On **seed 6**, the lofi-300 pool contains **zero** jointly-truth-near candidates (loosest criterion q0<60° AND w<30°). **The buggy-truth m135 finding ("surrogate-rerank promotes truth on seed 91 from align-rank-1-anti-twin to surr-rank-1-truth-near") does NOT generalise to the post-fix forward model.** Under correct truth, seed 91's alignment cost is actually correctly placing the joint-truth-near candidate near the top, and surrogate-rerank is HARMFUL.

Combined with m142, the picture is now: surrogate-MSE rerank is a **lever that worked under buggy truth and stops working (or inverts) under correct truth**. The decisive bottleneck is upstream of any ranking question — **m103's phi-sweep parameterisation reaches a small subset of q0-space**, and on seed 6 even the broader lofi-300 pool doesn't contain the truth basin. **Play 3 ([[upstream-redesign-6dof-surrogate-de]]) becomes the structural fix; surrogate-rerank as a standalone patch is dead.**

## What

[[m142_seed6_postfix_surrogate_rerank]] established that on seed 6 (post-fix, correct truth), surrogate-MSE rerank fails to promote the rank-9 truth-near ω. The session prompt's generality check then asks: does this generalise across seeds, or is seed 6 atypical?

m143 runs the same protocol on seed 91 — chosen because it was the m135 / strategic-reframe / Play-1 textbook win under buggy truth (q0=0.05° / w_dir=0.01° / ρ=0.22 Band-A), so a clean negative or clean positive on it is maximally informative. Plus a follow-on lofi-pool diagnostic on both seeds 6 and 91 to test whether the m135 rescue mechanism survives under correct truth.

## How

### Pipeline (seed 91)

1. Backup buggy artefacts (`traj_seed091.npz` → `_buggy.npz`; 5 result dirs renamed).
2. Regen truth: `MICRO48_WORKER_SEED=91 python3 notebooks/inversion/09_glint_analysis/m048_generate_trajectories_v2.py`. 38s wall.
3. m103 only (skip m115/m126/lc-compare): `python3 notebooks/inversion/invert.py --seed 91 --traj-source m048 --skip-m115 --skip-m126 --skip-lc-compare`. **Step 4 (Geo) timed out at 480s** — `geo_ckpt.npz` not produced; `multi_phi_ckpt.npz` is the highest-stage available pool. Total m103 wall: 763s.
4. Surrogate-MSE rerank on the multi_phi pool: `notebooks/inversion/score_geo_surrogate.py --seeds 91 --traj-source m048` (modified to fall back to `multi_phi_ckpt.npz` when geo is missing). 2.6s scoring wall.
5. Lofi-pool surrogate rerank on **both** seeds: `notebooks/inversion/score_lofi_surrogate.py --seeds 6 91 --traj-source m048`. 27s + 32s scoring wall. Saves `lofi_surr_ckpt.npz` per seed.
6. Joint-truth-near analysis script (one-shot inline) — q0_err and w_dir_err computed against canonical `q0s` / `omega0s` from `m048_trajectories.npz`.

### Why no `geo_ckpt.npz` for seed 91

m103's Step 4 (Geo) refines the 26 multi-phi candidates with a per-candidate optimization. On seed 91 it exceeded the 480s timeout. The multi_phi candidates' q0/ω/q0_err/w_err are slightly less refined than geo's would be, but the fields are equivalent for a ranking-test. (Whether the geo step would have moved the rankings substantially is a separate question — likely no, given how multi-phi q0 errors are 120–162° regardless of refinement.)

## Result 1 — Seed 91 multi_phi pool, surrogate-rerank

```
 surr_rk  geo_rk    surr_mse    surr_brt  glint_cost     w_err    q0_err
       1       1      3.5533     24.3624  1.1558e-01      4.65    137.75
       2      23      3.6027     23.9501  8.5208e-01      4.65    120.42
       3      24      3.8323     26.6665  8.6468e-01      4.65    161.67
       4       2      3.9419     27.0909  1.3048e-01      4.65    143.87
       5       7      5.2838     26.8604  3.2851e-01      7.73    129.99
       6       6      5.5407     25.4663  3.1289e-01     28.75     24.22
      ...
```

Headline: surr_mse rank-1 is the truth-near ω (w_err=4.65°). Both alignment cost and surrogate MSE place it at rank 1. The next 3 candidates (surr_rk 2–4) are also w_err=4.65° (same ω) — they're the 4 phi anchors m103 produced for the top-1 ω, all with q0_err 120–162°.

But: **surr_rk 6 has q0_err=24.22°, w_err=28.75°** — the only jointly-truth-near candidate in the multi_phi pool. m103's alignment cost has it at multi_phi rank 6, surrogate-MSE has it at surr_mse rank 6. Same ranking either way.

## Result 2 — Lofi-300 pool, both seeds

| pool stat | seed 6 | seed 91 |
|---|---|---|
| pool min `q0_err` | 21.53° | 10.22° |
| pool min `w_dir_err` | 4.39° | 0.73° |
| `surr_mse` min | 4.17 | 3.79 |
| `surr_mse` median | 7.65 | 7.85 |
| jointly-truth-near (q0<60° AND w<30°) | **0** / 300 | **1** / 300 |

The single jointly-truth-near candidate in seed 91's lofi-300 is at:
- `q0_err=25.24°`, `w_err=28.83°`
- `align_cost` rank: **2** (m103's default ranking)
- `surr_mse` rank: **26** (surrogate-MSE rerank DEMOTES it)

Seed 91 top-5 by `align_cost` (m103's pool-feed ranking):

```
align=2.7309e-01  surr_mse=3.9501  q0_err=141.92  w_err= 13.31
align=3.1835e-01  surr_mse=5.5092  q0_err= 25.24  w_err= 28.83  ← jointly truth-near
align=3.5833e-01  surr_mse=4.4581  q0_err=143.26  w_err= 16.36
align=4.1956e-01  surr_mse=5.5834  q0_err=127.53  w_err=174.00
align=4.2011e-01  surr_mse=7.3927  q0_err= 70.80  w_err= 24.17
```

Seed 91 top-5 by `surr_mse` (the m135 rescue lever):

```
align=1.0531e+00  surr_mse=3.7896  q0_err= 29.74  w_err= 65.52
align=2.7309e-01  surr_mse=3.9501  q0_err=141.92  w_err= 13.31
align=7.2773e-01  surr_mse=3.9868  q0_err=137.36  w_err= 27.67
align=9.1013e-01  surr_mse=4.1490  q0_err=126.97  w_err=159.69
align=7.4248e-01  surr_mse=4.2997  q0_err=146.19  w_err=  9.40
```

Surrogate-rerank's top-5 contains the truth-near ω twice (w_err 13.31, 9.40) but with q0_err 142°, 146° — same q0-anchor problem as before. The only jointly-truth-near candidate (rank 2 by alignment) is missing from surr_mse's top-5; it's at surr_mse rank 26.

Seed 6 lofi-300: same picture but worse — there is **NO** jointly-truth-near candidate in the entire 300-pool by even the loosest joint criterion (`q0<60° AND w<30°`). Both alignment-cost rerank and surrogate-rerank fail because the candidate that would be promoted simply doesn't exist.

## Why this matters

### 1. The buggy-truth m135 finding does not generalise

[[m135_alignment_cost_forensics_constrained_anchor]] reported, under buggy truth: "Lofi-300 surrogate ranking on seed 91: rank-1 lands 2.76° from truth (alignment-cost rank-1 is 174.83° anti-truth ±X twin)". Under correct truth (m143):

- Alignment-cost rank-1 is q0_err=141.92°, w_err=13.31° — NOT an ±X twin (twin would be ~180°), but still bad on q0.
- Surrogate-MSE rank-1 is q0_err=29.74°, w_err=65.52° — better on q0 but bad on ω.
- The single jointly-truth-near candidate is alignment-rank 2 (NOT rank 1 anti-twin), surr_mse rank 26.

Under buggy truth, the alignment cost surface had a strong attractor at the ±X twin which surrogate-rerank correctly identified as wrong. Under correct truth, alignment cost is messier but actually places jointly-truth-near at rank 2, and surrogate-rerank pushes it down to rank 26. **The exact mechanism that worked at buggy truth is inverted at correct truth on this seed.**

### 2. The two seeds reveal two distinct failure modes

| | seed 6 | seed 91 |
|---|---|---|
| Lofi-300 jointly-truth-near | 0 | 1 (at align rank 2) |
| Multi_phi rank-1 ω accuracy | bad (w_err=58°) | good (w_err=4.65°) |
| Multi_phi best q0_err for the truth-ω | 130° | 138° |
| Why m103 fails | both ω and q0 wrong | ω right, q0 reachable nowhere |
| Surrogate-rerank effect | demotes truth (no truth in pool to demote) | demotes the one truth candidate |

Seed 6 is **pool-deficient**: even the broader 300-candidate lofi pool doesn't contain a jointly-truth-near candidate. Re-ranking can't help.

Seed 91 is **q0-anchor-deficient**: m103 finds the truth ω, but its phi-sweep parameterisation reaches a 4-anchor subset of q0-space at q0_err 120–162°. The lofi-300 pool DOES contain a jointly-truth-near candidate, but it's at align rank 2 (not promotable by either align or surr_mse to a position where m115's K=3 default would consume it without further intervention).

### 3. Surrogate-rerank as a standalone patch is dead

Across the two seeds, surrogate-MSE rerank:
- Demotes the truth-near ω on seed 6 (m142).
- Agrees with alignment-cost on seed 91's truth-near ω (no improvement, no harm).
- Demotes the only jointly-truth-near candidate on seed 91 (lofi pool).

The m133/m134/m138 single-seed wins on this lever were under buggy truth; under correct truth, the lever is unreliable. **The strategic-reframe Play 1 ("consolidate the random m048 25-seed cohort using the m133 3-cost union ω-ranking + K=7 to m115") is now in question, because Play 1's m133 3-cost union depends on surrogate-MSE being a reliable signal. Generality across more seeds is needed to confirm, but the seed-91 negative datum is enough to suspend the Play-1 production direction until we understand it.**

### 4. m103's phi-sweep is the structural bottleneck

Both seeds share the failure: m103's 4-phi-per-ω parameterisation does not reach truth-q0 even when the truth-ω is in the candidate set. The phi-sweep is a 1-parameter family — it parameterises q0 by a single rotation angle around an axis determined by the anchor epoch. With only 4 anchors per ω, it's at best a sparse sample of q0-space.

**Two recovery directions, neither cheap:**

1. **Patch m103 phi-sweep**: denser anchor count (8? 16? 64?), wider angular footprint, multi-axis sweep, or surrogate-MSE-driven rejection. Local fix; might be enough on q0-anchor-deficient seeds (like 91) but cannot help pool-deficient seeds (like 6) where the truth basin isn't even in the 300-pool.
2. **Replace m103 with Play 3** ([[upstream-redesign-6dof-surrogate-de]]): 6-DOF DE over (q0, ω) driven by surrogate full-LC MSE from the start, no m103 substrate. Structural fix that addresses both pool-deficiency AND q0-anchor-deficiency.

m143 evidence does NOT distinguish between (1) and (2) yet. A diagnostic on more seeds (especially constraint-poor and high-phase from the random m048 cohort) would help — but the prompt explicitly limits this session to two additional seeds and seed 47 has not been run yet.

### 5. m115 K=1 rerun on seed 91 is also not triggered

Even though seed 91's surrogate-rerank places truth-near ω at rank 1, the q0_err of that candidate is 137.75°. m115's 3-DOF DE around an input ω has q0-bridging that's empirically <30° (per [[m115_de_bridging_radius]]) and in practice often less. 137° is far outside. The m115 K=1 rerun would not bridge.

The lofi-300 jointly-truth-near candidate (q0_err=25°, w_err=29°) is more interesting: the q0 is bridgeable but the ω is at the marginal edge (m134 says 15° is marginal, 28.75° is outside). m115 from this seed would be a marginal test. Out of scope this session.

## Numbers

| | seed 6 | seed 91 |
|---|---|---|
| Truth regen wall | 53 sec | 38 sec |
| m103-only wall | ~13 min (full pipeline killed mid-m115) | 763 sec (Step 4 timed out at 480s) |
| Multi_phi pool size | 26 | 26 |
| Multi_phi pool min `w_dir_err` | 16.86° (rank-9) | 4.65° (rank-1, x4 phi anchors) |
| Multi_phi pool min `q0_err` for the truth-ω | 129.58° | 120.42° |
| Lofi-300 pool min `q0_err` | 21.53° | 10.22° |
| Lofi-300 pool min `w_dir_err` | 4.39° | 0.73° |
| Lofi-300 jointly-truth-near (q0<60° AND w<30°) | 0 / 300 | 1 / 300 |
| Lofi-300 surr_mse min | 4.17 | 3.79 |
| Lofi-300 align cost rank of jointly-truth-near | n/a (none) | 2 |
| Lofi-300 surr_mse rank of jointly-truth-near | n/a (none) | 26 |
| Surrogate-rerank verdict | DEMOTES truth (m142) | DEMOTES the one truth candidate; ω-ranking unchanged |

## Artefacts

- `notebooks/inversion/score_geo_surrogate.py` — modified to fall back to `multi_phi_ckpt.npz` when `geo_ckpt.npz` is missing.
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/` — post-fix m103 outputs (Step 4 timed out; multi_phi available, geo not).
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091_buggy/` and 4 other `_buggy` seed-91 dirs.
- `data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed091.npz` (post-fix correct truth).
- `data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed091_buggy.npz` (preserved).
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/geo_surr_ckpt.npz` (m143 surrogate-MSE rerank on multi_phi pool).
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/lofi_surr_ckpt.npz` and `seed_006/lofi_surr_ckpt.npz` (lofi-300 surrogate scores for both seeds).
- Seed-47 buggy artefacts also preserved as `_buggy` (1 NPZ + 5 result dirs); seed 47 not regenerated this session.

## Out of scope here

- **Seed 47 generality check** — not run; backed up but truth not regenerated. Could be added in a follow-on session if context permits.
- **NM-pool surrogate rerank** (the diagnostic next step). NM-300 pool on seed 6 has 7 jointly-truth-near candidates buried at cost ranks 127–267. Re-ranking by surrogate-MSE before multi-phi could surface them, IF the lofi `surr_mse` NaN issue is fixed (see open items in CURRENT_STATE).
- **Play 3 design and prototyping**. Out of scope this session.
- **m103 phi-sweep parameterisation patch**. Out of scope; design needs more thought.
- **Generality on more random m048 seeds** (e.g., 4 more from the 25-seed Play-1 cohort). Cohort regeneration is a separate session per the discipline rules.

## Cross-references

- [[m141_seed6_postfix_pipeline]] — the original m141 finding on seed 6 that motivated this generality check.
- [[m142_seed6_postfix_surrogate_rerank]] — m142 surrogate-MSE rerank on seed 6 (the failure case that m143 is testing for generality).
- [[m139_convention_bug_fix]] — the propagator fix two levels upstream.
- [[m135_alignment_cost_forensics_constrained_anchor]] — the buggy-truth lofi-pool surrogate rerank rescue that does NOT generalise to the post-fix forward model.
- [[m138_seed47_lombscargle_bracket]] — the seed-47 surrogate-rerank precedent under buggy truth; status under correct truth still untested.
- [[m115_de_bridging_radius]] — the empirical q0/ω bridging radius constraints.
- [[surrogate-rerank]] — the lever now characterised as buggy-truth-specific.
- [[upstream-redesign-6dof-surrogate-de]] — the structural fix candidate.
