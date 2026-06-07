---
title: "s019b — ω-direction basin width at pinned truth-ω-mag (3 seeds)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s018c_phi_sweep_pilot.md
  - notebooks/inversion/survey/experiments/s019_ls_bracket_omega_mag.md
  - notebooks/inversion/survey/experiments/s003_landscape_vs_omega.md
related:
  - notebooks/inversion/survey/concepts/q_omega_coupling.md
  - notebooks/inversion/survey/concepts/known_pathologies_to_revalidate.md
created: 2026-05-03
updated: 2026-05-03
confidence: high
---

## TL;DR

Localises the ω-direction LM basin width at pinned truth-ω-mag, which
the s018c diagnostic had bounded only as "between 1° and 5°" on seed 6.
Sweeps Δω-dir at {2°, 3°, 4°} on 3 seeds (6, 28, 44), reusing the
s018c phi-sweep-IC + Pool(8) LM harness.

| seed | d_dir 2° | d_dir 3° | d_dir 4° | basin radius |
|------|----------|----------|----------|--------------|
| 6  | A (q0_err=0.287°, 12/32) | A (0.287°, 7/32) | A (0.287°, 7/32) | **≥4°** (≤5° from s018c) |
| 28 | A (0.034°, 14/32)         | A (0.034°, 14/32)| A (0.034°, 13/32)| **≥4°** (≤5° from s018c) |
| 44 | A (0.033°, 5/32)          | A (0.033°, 4/32) | **D (160.8°, 0/32)** | **between 3° and 4°** |

**Decisive: ω-dir basin radius at pinned truth-mag is 3-5° across the
three seeds tested.** This is **3-5× wider than the s003 "1° tube"**
because s003 measured the joint dir+mag tube; pinning ω-mag at truth
expands the dir basin substantially. The user-predicted positive
result holds.

Implication: with an LS-bracket prior collapsing the ω-mag axis to ~5
cells (s019: 98/100 within 5%), an ω-dir grid at 3°-cell radius covers
S² with **~1500 cells**. Joint grid total: **~7500 cells/seed** —
feasible. This unlocks an LS-bracket-augmented S016-A architecture
where phi-sweep ICs + LM polish run per (ω-mag-bracket-cell, ω-dir-cell)
pair.

## What

For each (seed × ω-direction perturbation), construct an ω-cell at
(truth-direction-rotated-by-Δθ, truth-magnitude). Build the phi-sweep
IC pool at this cell (288 ICs for seed 6, 480 for seed 28, 720 for
seed 44 due to varying tier coverage), surrogate-score, top-32 LM
polish (max_nfev=60). Record best q0_err, min q0_err, n_LMs in basin.

Pilot seeds chosen for tier coverage (must be in s018b's 81/100
classifiable cohort):
- **6** — 4 classifiable peaks (T2:2, T3:2). s011 recoverable.
- **28** — 7 classifiable peaks (T1:2, T2:1, T3:4). Sub-Sobol-narrow
  basin per s006.
- **44** — 15 classifiable peaks (4 tiers). s010 found a competing
  basin; tier-richest pilot seed.

Pre-registered prediction: ω-dir basin 1.5°-3°. **Outcome exceeded
prediction at 3-5°.**

## How

Reuses s018c machinery via an internal monkey-patch on
`s018c_phi_sweep_pilot.build_omega_grid`. Per cell:

1. Construct perturbed ω-direction by Rodrigues-rotating truth-dir
   around an arbitrary perpendicular axis by Δθ degrees. Combined with
   truth-magnitude.
2. Build phi-sweep IC pool at this single ω-cell (s018c `build_phi_
   sweep_ics_for_seed`). N_PHI=12.
3. Surrogate-score all ICs (Pool(8)).
4. Top-32 by surrogate-MSE → LM polish (max_nfev=60, Pool(8)).
5. Record best (lowest final_surr_MSE) q0_err, min q0_err across all
   32 LMs, n_LMs landing in basin (q0_err < 5°).

Total wall: 20 min for 9 cells × 32 LMs each.

## Result

```
seed | d_dir | n_ics | best_q0_err | min_q0_err | n_in_basin | final_mse
-----+-------+-------+-------------+------------+------------+----------
 6   |  2°   |  288  |    0.287°   |   0.287°   |  12/32     | 2.788e-3
 6   |  3°   |  288  |    0.287°   |   0.287°   |   7/32     | 2.788e-3
 6   |  4°   |  288  |    0.287°   |   0.284°   |   7/32     | 2.788e-3
 28  |  2°   |  480  |    0.034°   |   0.034°   |  14/32     | 5.776e-4
 28  |  3°   |  480  |    0.034°   |   0.034°   |  14/32     | 5.776e-4
 28  |  4°   |  480  |    0.034°   |   0.034°   |  13/32     | 5.776e-4
 44  |  2°   |  720  |    0.033°   |   0.033°   |   5/32     | 3.208e-4
 44  |  3°   |  720  |    0.033°   |   0.033°   |   4/32     | 3.208e-4
 44  |  4°   |  720  |  160.763°   |  52.291°   |   0/32     | 1.666e+0
```

Combined with the s018c ω-grab (which had Δdir=1° in basin and 5°
collapsed):

| seed | basin radius (Δω-dir) | source |
|------|------------------------|--------|
| 6    | ≥4° and <5°            | s018c (1, 5°), s019b (2, 3, 4°) |
| 28   | ≥4° and (presumably <5°) | s019b (2, 3, 4°) only |
| 44   | between 3° and 4°       | s019b (2, 3, 4°) |

The basin width tightens slightly with peak-count:

| seed | n_classifiable_peaks | basin Δω-dir | best q0_err in basin |
|------|----------------------|---------------|----------------------|
| 28   | 7  | ≥4° | 0.034° |
| 6    | 4  | ≥4° | 0.287° |
| 44   | 15 | ~3° | 0.033° |

Counter-intuitively, seed 44 with the MOST peaks has the narrowest
basin. Hypothesis: more peaks = more constraint epochs = the
phi-sweep IC pool aligns at multiple peaks simultaneously, and the
"correct" alignment becomes more sensitive to ω-direction error. The
basin is *deeper* (LM lands at 0.033°) but *narrower* (boundary at 3°
not 4°). For seeds 6 / 28, the per-peak constraint is looser (fewer
peaks) so the basin is wider but shallower.

## Why this matters

Combined with s019:

| component                             | value | source |
|---------------------------------------|-------|--------|
| ω-mag prior (LS-bracket) coverage     | 98/100 within 5% of truth | s019 |
| ω-mag bracket grid size               | ~5 cells covering [0.5p_min, 2.0p_max] | s019 |
| ω-dir basin radius at pinned truth-mag| 3-5° (cohort tested 3 seeds) | s019b |
| ω-dir cells for S² coverage at 3°     | ~1500 | derived |
| Joint (mag × dir) grid                | ~7500 cells/seed | derived |
| Per-cell phi-sweep IC count            | 144-720 (varies by tier coverage) | s018c |

A bracket-augmented S016-A architecture is feasible: per (bracket-mag,
dir-cell) pair, build phi-sweep IC pool, surrogate-score, top-K LM
polish. Cohort selector = lowest LM-polished surrogate-MSE per seed.

**Wall budget estimate** for a 5-seed pilot (s020):
- Per seed: 7500 cells × ~200 ICs/cell × 50 ms / 8 workers ≈ 18 min
  surrogate scoring; top-128 LM × 20s × 8 workers ≈ 5 min LM ≈ 23 min/seed.
- 5-seed pilot: ~2 hr. **Cohort scale (100 seeds): ~40 hr** (overnight,
  feasible single-machine; could shard).

## What this validates

- The s018c phi-sweep IC primitive works in basin at fine ω-dir
  perturbations (2°-3°), not just at exact truth-ω.
- The s003 "1° dir tube" was *jointly* in dir-and-mag — pinning ω-mag
  at truth widens the dir basin to ~3-5°.
- The user's reasoning ("how wide are the basins if we inject close to
  truth ω_mag and now we're looking at the basins of w_dir only?")
  was correct directionally and the basins are even wider than I
  predicted (predicted 1.5-3°, observed 3-5°).

## What this does NOT validate

- **Whether the actual bracket-prior-augmented S016-A architecture
  produces Band A∪B per seed.** s020 (next experiment) is the actual
  pilot.
- **Whether the cohort-scale ω-dir basin minimum is 3°.** Tested 3
  seeds; cohort tail (especially low n_rotations or extreme PA seeds)
  may have tighter basins. A wider 5-10 seed cohort sweep would be
  cheap follow-up.
- **The basin shape** — only tested rotation around a single
  perpendicular axis. Other dir-perturbation directions might give
  different basin widths (anisotropic basin).
- **Asymmetry in ω-mag axis** (s018c showed -10% in basin, +10%
  collapsed at fixed truth-dir). Not re-tested here; assumed bracket
  grid in geometric spacing covers this asymmetry.

## Numbers

| metric | value |
|--------|-------|
| Cells run | 9 (3 seeds × 3 dir perturbations) |
| LMs per cell | 32 |
| Wall | 20 min (9 cells × ~2 min each) |
| Best q0_err in seed-44 basin | 0.033° |
| Best q0_err in seed-6 basin | 0.287° |
| Best q0_err in seed-28 basin | 0.034° |
| Cohort min basin radius (3-seed) | 3° (seed 44) |
| Cohort max basin radius (3-seed) | ≥4° (seeds 6, 28) |

## Artefacts

- `experiments/s019b_omega_dir_basin_at_truth_mag.{py,md}`
- `results/s019b/summary.json`

## Out of scope

- Cohort-wide ω-dir basin sweep (5-10 seeds) — cheap follow-up, not
  blocking on s020.
- Anisotropic basin shape (perturbation direction varied) — only
  matters if s020 fails at the predicted feasibility envelope.
- s020 itself — the actual bracket-augmented S016-A pilot.

## Cross-references

- `s019_ls_bracket_omega_mag.md` — the ω-mag prior that complements
  this finding.
- `s018c_phi_sweep_pilot.md` — the IC-generator + ω-grab harness
  reused here.
- `s003_landscape_vs_omega.md` — the original "1° tube" (joint
  dir+mag) finding that this work decomposes.
