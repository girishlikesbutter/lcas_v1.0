# m048 random-cohort baseline (2026-04-22)

## Purpose

First representative estimate of pipeline yield on the m048 trajectory database. All prior cohorts (Phase-A, Phase-B) were cherry-picked from rounds of iteration — they do not represent the true distribution.

## Method

- **Seeds**: 25 drawn reproducibly via `np.random.default_rng(42).choice(100, 25)` → `[6, 7, 8, 11, 16, 17, 34, 35, 42, 45, 47, 48, 51, 57, 59, 64, 67, 69, 71, 78, 79, 84, 89, 91, 99]`.
- **Pipeline**: `invert.py` → m103 (geo harvest, SKIP_HIFI) → m115 (surrogate-DE 3×10 = 30 starts) → m126 (L-BFGS polish + hi-fi before/after min) → wrappedbest.
- **Honest noise realisation**: all three stages consume `truth['observed_lc']` from `lib/traj_source.canonical_observed_lc` (`mag_hifi + default_rng(42).normal(0, 0.05)`). Pre-fix, m126 used legacy `np.random.seed(42)` → different noise than m115 for the same seed → polish fitting a shifted observed_lc.
- **Graceful kill path**: m103 Step 4 `pool.map_async().get(timeout=480)` writes `geo_timeout.flag` + `status=geo_timeout`, exits 0. Batch driver tags seed and moves on.
- **Reporting**: ρ = √hifi_MSE / 0.05 (σ_noise = 0.05 mag synthetic noise). Bands A < 2, B 2-4, C 4-8, D ≥ 8 (Roberto convention).

## Results

| Band | n/25 | % | seeds |
|------|------|---|-------|
| A | 4 | 16% | 6, 17, 45, 91 |
| B | 0 | 0%  | — |
| C | 1 | 4%  | 79 |
| D | 17 | 68% | 7, 8, 11, 16, 34, 47, 48, 51, 57, 59, 64, 67, 71, 78, 84, 89, 99 |
| geo_timeout | 2 | 8% | 35, 69 |
| m103 crash | 1 | 4% | 42 |

Median per-seed wall: ~12 min. Full cohort wall: **4.79 h**.

### Band A winners

| seed | ρ | q₀ err | ω dir err | ω mag err % |
|------|---|--------|-----------|-------------|
| 6  | 0.96 | 179.98° (twin) | 0.07° | +0.01 |
| 91 | 0.98 | 0.05°          | 0.01° | −0.00 |
| 17 | 0.98 | 179.99° (twin) | 0.06° | +0.00 |
| 45 | 1.14 | 179.80° (twin) | 0.40° | −0.00 |

All four recoveries have ω direction error < 0.4° — when the pipeline works, it nails it. Three of four land on the ±X twin (180°-about-X degenerate state), which is a known valid solution mode for IS-901 given its symmetry.

## Dominant failure mode

**Every single Band D seed has ω direction error > 19° at the winner basin.** Most are in the 50°-170° range.

m103 ranks its L-BFGS-refined ω candidates by `geo_costs` (alignment cost at brightness peaks). For random m048 seeds, the lowest-geo-cost ω candidates are routinely far from truth — the alignment cost surface has plenty of false minima in observation geometries with few bright peaks or high phase angles.

m115 DE and m126 polish cannot recover when the candidate pool is wrong. The polish step is robust IF given a truth-adjacent ω (seed 49's noise-fix ρ drop from 1.67 → 1.03 confirms this); it's useless otherwise.

## Contrast with Phase-B 8-seed cohort

Phase-B (2026-04-17) reported **4A + 1B + 0C + 1D + 2 upstream-fails** = 67% at/near noise floor. Those seeds were the ones the pipeline got right in prior iterations. Selection bias.

Random 25-seed sample: **16% A + 0% B + 4% C + 68% D + 12% upstream-fail**.

**The Roberto 2026-04-17 report's "12 of 16 at noise floor" headline is not a pipeline-yield estimate; it is a biased subset.** Any forward-looking performance claim needs to reference the random-cohort number, not the cherry-picked one.

## Noise-realisation fix

Seed 49, same truth, same m115 output, same m103 ω candidates — only the noise realisation downstream changed (m126 switched from Mersenne Twister to PCG64 via `traj_source.canonical_observed_lc`):

| | ρ | hifi MSE | q₀ err | ω dir err |
|-|---|----------|--------|-----------|
| pre-fix | 1.67 | 0.00697 | 1.15° | 0.89° |
| post-fix | 1.03 | 0.00266 | 0.21° | 0.094° |

Real quality lift on well-behaved seeds. The inconsistency was a correctness bug, not a cosmetic issue.

## Open problems this baseline exposes

1. **m103 ω-pool miss-rate on random m048 is 68%+** — surrogate-first front-end rewrite (Roberto path-forward #1) now strongly motivated. m103's alignment-cost grid is the obvious replacement target: a coarse 6-DOF surrogate DE on `(q₀, ω)` seeded by peak-count |ω|_est should replace m103 Steps 1-3.5 outright.
2. **Seed 42 crash at m103 line 209** — `anchor_rank = sr[0]` fails when `spec_peaks` is empty (no LC peaks below mag 9.0). Needs graceful-skip path similar to geo_timeout.
3. **m115 Step 1 is serial** — nested `for ic (3 ω) × for si (10 starts)` with `differential_evolution(..., workers=1)`. 5-7× speedup available by wrapping in `Pool.map`.

## Artefacts

- `data/results/inversion_diagnostics/batch_m048_v1/batch_summary.json` — aggregate.
- `data/results/inversion_diagnostics/batch_m048_v1/batch_log.jsonl` — per-seed rows.
- Per-seed: `wrappedbest_m048_seed{NNN}/result.json`, `pred_lc.npy`, `.noise_fix_v1.done` marker.
- Stage checkpoints: `m115_surrogate_pipeline_m048/seed_{NNN}/{step1_de.npz, step2_hifi.npz, result.json}`, `m126_wrapped_m048/seed_{NNN}/{polish_ckpt.npz, hifi_ckpt.npz, result.json}`.
- Graceful-kill markers: `m103_hybrid_m048/seed_{028,035,069}/geo_timeout.flag`.
- Validation run (pre-batch): `batch_m048_v1_validation/` — confirmed both known-hang seeds (28, 69) hit the 8-min cap at 480.0s and 480.1s respectively.

## Commits

- `36e724c` — pipeline code (canonical noise, m103 graceful timeout, run_m048_batch.py).
- `db46756` — data artifacts (overnight batch + validation + updated seed_049 under canonical noise).

## See also

- [bridge-lc-selection](bridge-lc-selection.md) — m115 + m126 pipeline overview.
- [multi-solution-philosophy](multi-solution-philosophy.md) — ρ-band reporting convention rationale.
- Roberto 2026-04-17 weekly report at `reports/2026-04-17-roberto/report.pdf` — context for why that 67% number is not pipeline-yield.
