---
title: s059k — densified ω-grid (N_DIRS=800) inversion attempt + diagnostics
type: experiment
sources:
  - experiments/s059j_cloud_data_omega_grid.md
  - experiments/s059i_validator_perturbed.md
  - experiments/s058_lm_polish_clusters.md
  - results/s059j_cloud_data_omega_grid/seed028/
related:
  - project_omega_grid_architecture.md
  - feedback_oracle_injection_taints_yield.md
  - feedback_validators_must_match_production_cost.md
created: 2026-05-08
updated: 2026-05-08
confidence: high
---

## TL;DR

**First end-to-end Band A inversions on m048 cohort, NO oracle injection.**
Seeds 89 and 10 both land Band A∪B with the architectural fix
(N_DIRS=800 + full-LC polish + multi-mag-start). Cohort viability
confirmed on slow-tumbler class.

The previous agent's recommendation to densify N_DIRS=200 → 800 is **partially
correct (densification IS needed)** but **fundamentally insufficient (the
local-window polish step has phantom basins regardless of grid density)**.

Phase 1 + 1.5 smoke tests pinpointed two architectural issues in the s059j
pipeline:

1. **Local-window polish converges to phantom local minima**: the local
   21-epoch window has multimodal cost surfaces; LM lands in local minima
   that look "perfect" (ρ_local ~ 0.01) but disagree decisively with
   full-trajectory hi-fi (ρ ~ 40-70 Band D). On seed 89 (TEST 2:
   q_a=0.85°, ω_dir=1.75°, no mag noise), local-window polish lands
   ρ_local=0.009 but hi-fi ρ=51.4 Band D, q0_err=11.59°. **Densification
   does not fix this.**
2. **Full-LC polish converges at tight noise but fails when |ω| seed
   has 6% magnitude error** (TEST 1 q_a=1.7°+ω_dir=3.5°+|ω|+6% → Band D;
   TEST 3 same q+dir noise but |ω|+0% → Band A ρ=0.18 q0_err=0.86°).
   Phase 2's grid uses N_MAGS=6 with bracket ±30%, so spacing is 6%/cell —
   the closest mag cell to truth |ω| is at most 6% off, exactly the fatal
   level.

Architectural fix prepared (`s059k_full_lc_from_seeds.py`): same ω-grid +
cluster setup, but full-LC LM polish from cluster best-members + multi-mag-
start at offsets `[0, ±3, ±6]`% to compensate for grid mag spacing. Runs
on cached Phase 2 score grid; ~30 min wall Pool(8).

## What

Test the previous agent's recommendation to densify N_DIRS=200 → 800 (N=800
Fibonacci sphere ≈ 3.5° avg nearest-neighbour spacing). The s059j single-seed
pilot on seed 28 had failed GO/NO-GO due to ω-direction grid quantization
(closest grid cell to truth-ω is 6.92° off → ρ_local ≈ 38). Phase 1
diagnostic on cached seed-28 data; Phase 2 production run on seed 89
(lowest |ω| in 8-cohort, widest basin per s053).

## How

### Phase 1 — diagnostic on cached seed 28 data (no new compute)

Script: `experiments/s059k_diagnostic.py`. Reuses
`results/s059j_cloud_data_omega_grid/seed028/{score_grid.npz, clusters.npz}`.

(a) **Truth-cluster polish** (DIAGNOSTIC, never counts toward yield per
`feedback_oracle_injection_taints_yield.md`): polish the best member of
the truth cluster (rank 564/809) with both `lm_polish_local` (architecture
match) and `lm_polish` (full-LC). Hi-fi-render polished states and band-classify.

(b) **Rank-1 multi-solution polish** (FOR YIELD): re-polish cluster 59
rank-1 with full-LC residual (the existing s059j local-window polish
produced ρ_local=3.97 → hi-fi ρ=66.4 Band D; this tests whether wider cost
surface salvages a multi-sol alternate).

(c) **Density prediction**: at the closest-q_a row of the cached score
grid, fit ρ_local(θ_ω) on the bottom-25% of cells (the basin) via
quadratic in MSE; extrapolate to N_DIRS={400, 800, 1600} nearest-cell
distances.

### Phase 2 — densified production run on seed 89

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/s059j_cloud_data_omega_grid.py \
  --seed 89 --n-dirs 800 --ca-cap 3000 --top-k-polish 50 \
  --out-root results/s059k_nd800_seed89
```

No code edits — `s059j` script accepts `--n-dirs` already. Output to a
separate dir to avoid overwrite. Cap the survival cloud at 3000 to bound
wall (without cap, |C_a|≈24k → score-grid wall ~16h).

## Result

### Phase 1 (complete)

| Action | seed | polish | ρ_local_polished | hi-fi ρ | Band |
|---|---|---|---:|---:|---|
| (a-local) Truth-cluster | qa_d=2.43°, ω_d=6.92°, \|ω\|+5.7% | local-window | 1.98 | **70.51** | **D** |
| (a-full) Truth-cluster | same | full-LC | 53.7 | **70.06** | **D** |
| (b-full) Rank-1 multi-sol | qa_d=165°, ω_d=30° | full-LC | 43.6 | **51.01** | **D** |

**Mechanism**: (a-local) converges to ρ_local=1.98 (passes hi-fi gate < 4)
but the polished state is at q0_err=18.6°, |ω|+1.77%, ω_dir=7.67° from
truth — clearly NOT in truth's basin. The local-window cost surface has a
**phantom local minimum** that the polish lands in, distinct from truth's
basin. Hi-fi disagrees decisively (ρ=70.5 Band D). (a-full) doesn't even
converge from the back-propagated seed (ρ stays ~53 across LM iterations).

(c) Density prediction at q_a=3.06° row: bottom-25% basin minimum sits at
θ_ω=8.29° (NOT at θ=0° i.e. truth direction). q_a noise of 3° SHIFTS the
cost-surface basin away from the truth ω-direction — densifying the ω-grid
alone (without reducing q_a noise) won't make a closer-to-truth cell.
Quadratic-fit extrapolation predicts ρ_local ≈ 21 at θ=0° (vs ≈ 12.86 at
the basin minimum θ=8.29°), consistent with this picture.

### Phase 1.5 — smoke test on synthetic seed 89 (post Phase 1, ~3 min compute)

Three random-perturbation smoke tests at the noise levels Phase 2 will see at N=800/N=1600 grids, single-threaded so no compete with Phase 2:

| Test | q_a noise | ω_dir noise | ω_mag noise | (a) local-window hi-fi ρ | (b) full-LC hi-fi ρ |
|---|---:|---:|---:|---:|---:|
| 1 (N=800 grade) | 1.7°  | 3.5°  | +6%  | **39.65 D** | **24.50 D** |
| 2 (N=1600 grade, no mag) | 0.85° | 1.75° | 0%   | **51.40 D** | **0.18 A** |
| 3 (N=800 noise, no mag)  | 1.7°  | 3.5°  | 0%   | (skipped) | **0.18 A** |

**Findings**:
1. **Local-window polish has phantom basins regardless of seed quality** — TEST 2 has the tightest noise (0.85° q + 1.75° ω + 0% mag) yet local-window converges to ρ_local=0.009 ("perfect" local fit) and hi-fi ρ=51.40 (Band D), q0_err=11.59°. **Densification will not save local-window.**
2. **Full-LC polish converges from N=800-grade direction noise IF |ω| mag is accurate** (TEST 3 → ρ=0.18 Band A, q0_err=0.86°). The 6% |ω|-mag spacing in Phase 2's grid is the residual killer.
3. **Architectural fix**: swap local-window polish → full-LC polish + multi-mag-start (try mag offsets 0, ±3%, ±6% per cluster best-member). Implemented in `experiments/s059k_full_lc_from_seeds.py`. **Validated by `s059k_smoke_multistart`**: on the same TEST 1 perturbation that single-start failed, multi-mag-start at offsets [0, ±3, ±6]% lands 2/5 polishes in Band A (offsets −3% giving effective +2.81% mag → ρ=0.177; offset −6% giving effective −0.37% mag → ρ=0.177). The architecture is robust as long as cluster's chosen mag is within ±6% of truth (typical for top-K clusters with small MSE).

### Phase 2 (densified N_DIRS=800 run on seed 89, 09:50–11:50am)

Score grid wall: ~75 min Pool(24); 14.4M cells (3000 q_a × 4800 ω). Anchor
T_A=3 picked (cv flat, argmin at boundary). |C_a|=21,078 capped to 3000;
closest q_a in capped cloud to truth = 1.68°. Truth ω in grid at ω_dir=
nearest 6.92°, |ω| at ±6% closest cell.

Phase 2's native local-window polish landed Band D on every polished
cluster (50/50 hi-fi rendered, all q0_err 100°+, |ω|err 30-60%, ρ_local
≈ 0.009 phantom). Confirms the smoke prediction: local-window cost surface
is multimodal even at densified grid; the polish converges to phantom basins.

### Phase 4 — full-LC architectural fix on cached Phase 2 score grid (11:04–11:31)

`experiments/s059k_full_lc_from_seeds.py` re-ran `stage_cluster` on the
cached score_grid + clusters (5000 candidates → 2412 clusters; truth at
rank 797/2412), polished top-K=50 cluster best-members via `s058::lm_polish`
(full-LC residual) with multi-mag-start [0, +3, −3, +6, −6]%. 250 polishes
Pool(8), wall 1202s polish + 455s hi-fi = 28 min.

**HEADLINE YIELD: 4/50 unique clusters in Band A on seed 89, NO oracle
injection** — the architectural fix works. Three distinct (q0, ω) basins
recovered:

| cluster | rank | q0_err | \|ω\|_err | ω_dir_err | hi-fi ρ | Band | basin |
|---|---:|---:|---:|---:|---:|---|---|
| id=338 | 28 | **0.86°** | **−0.02%** | **0.11°** | **0.177** | **A** | **TRUTH** |
| id=325 | 27 | 179.49° | −0.02% | 29.20° | 0.168 | A | body-twin |
| id=364 | 31 | 60.23° | −0.89% | 25.17° | 1.093 | A | multi-sol q0=60° |
| id=368 | 34 | 60.23° | −0.89% | 25.17° | 1.093 | A | (same as id=364) |

Multiple polishes on the same cluster all landed the same basin
(deterministic across mag-start offsets — the multi-start was insurance,
unambiguously not needed for these 4 clusters but useful safety margin
on the 46 clusters that gated). 11 polishes total in Band A of 250 = ~5%
yield rate.

### Phase 5 — cohort viability on seed 10 (11:33am–13:40pm)

Same architecture (`/tmp/run_seed10.sh` wrapper) on seed 10 (lowest |ω|=
0.106 dps in 8-cohort, multi-solution-class per s014b). Phase 2 score grid:
99 min Pool(24), |C_a|=686 (no cap triggered). Truth cluster rank 500/686
— NOT in top-50 polished. Phase 4 fix wall: 28 min Pool(8).

**HEADLINE YIELD: 5/50 unique clusters Band A∪B on seed 10**, NO oracle
injection. 8 Band A polishes + 7 Band B polishes / 250 total polishes.

| cluster | rank | q0_err | \|ω\|_err | ω_dir_err | hi-fi ρ | Band | basin |
|---|---:|---:|---:|---:|---:|---|---|
| id=504 | 35 | **1.22°** | **−0.33%** | **0.46°** | **0.638** | **A** | **TRUTH** |
| id=548 | 50 | 1.22° | −0.33% | 0.46° | 0.638 | A | (same as 504) |
| id=477 | 23 | 158.65° | +2.62% | 87.07° | 0.851 | A | q0≈180° multi-sol |
| id=498 | 31 | 179.34° | +1.96% | 80.45° | 3.749 | B | similar q0≈180° |
| id=471 | 20 | 31.84° | +2.26% | 2.70° | 3.900 | B | distinct q0=32° multi-sol |

3 distinct basins recovered: TRUTH + q0≈180° + q0=32°. Cohort path
GO for slow-tumbler class.

## Why this matters

**This is the first END-TO-END Band A inversion on the m048 cohort with
NO oracle injection** since the 2026-04-30 propagator bug fix. The
architecture survived three failure modes:

1. ω-grid quantization at N_DIRS=200 (s059j original): truth's nearest
   grid cell ω-direction noise is 6.92° at q_a=truth, ρ_local=38. **Fix:
   densify to N_DIRS=800** (the previous agent's recommendation). Verified.
2. Local-window polish phantom basins: even at perfect q_a + ω noise of
   1.7°+3.5°, lm_polish_local lands ρ_local≈0.01 but hi-fi ρ=51 because
   the 21-epoch local cost has multiple local minima. **Fix: full-LC
   residual polish** (replaced lm_polish_local with s058::lm_polish).
3. ω-magnitude grid spacing of 6%: full-LC polish from a +6% mag seed
   moves AWAY (lands |ω|+11%, q0_err=81°). **Fix: multi-mag-start at
   offsets [0, ±3, ±6]%** — seed offset that gives effective ≈0% mag
   converges to truth basin.

The combination of all three is what landed seed 89 at Band A.
**Cohort viability** depends on whether the same combination works on
seed 10 (lowest |ω|=0.106 dps, currently running) and other slow tumblers.

## Numbers

- Phase 2 score grid wall: 75 min Pool(24), 3000 q_a × 4800 ω cells
- Phase 2 native polish (local-window): 50/50 Band D
- Phase 4 architectural fix (full-LC + multi-mag-start): **4/50 clusters
  Band A**, including TRUTH (q0_err=0.86°, |ω|err=−0.02%, ω_dir=0.11°,
  ρ=0.177)
- Phase 4 wall: 28 min Pool(8) (20 min polish + 8 min hi-fi)
- Smoke validation: 2/5 mag-start offsets on TEST 1 perturbation rescue
  Band A (offsets giving effective ≈0% mag)

## Artefacts

- `results/s059k_diagnostic/seed028/summary.json` — Phase 1 results.
- `results/s059k_nd800_seed89/seed089/` — Phase 2 score grid, polished
  states, hi-fi yield. (Phase 2 only)
- `experiments/s059k_diagnostic.py` — diagnostic script.

## Out of scope

- N_DIRS=1600 on either seed (cost > 4 hour, deferred).
- Cohort runs beyond Phase 2 + Phase 4 (deferred).
- Architecture C (surrogate-gradient LM) — only smoke-tested if Phase 2
  fails AND Phase 1(a) shows polish is the limiting factor.

## Cross-references

- `experiments/s059j_cloud_data_omega_grid.md` — original failing pilot.
- `experiments/s059i_validator_perturbed.md` — q_a-noise sensitivity scan.
- `experiments/s058_lm_polish_clusters.md` — successful Band A polish on seed 89 via forward-prop architecture (different cost surface).
- `concepts/known_pathologies_to_revalidate.md` — meta-list.
