---
title: "s036: Multi-seed generalisation pilot — density+polish chain"
type: experiment
sources:
  - experiments/s035_hifi_confirm_seed89.md
  - experiments/s033_density_sensitivity.md
  - experiments/s034_lm_polish_seed89.md
  - results/s032_cohort_fast/
related:
  - s035_hifi_confirm_seed89
  - s034_lm_polish_seed89
  - s033_density_sensitivity
  - s011_sobol_shoemake_pilot
  - s006_narrow_basin_seeds
created: 2026-05-05
updated: 2026-05-05
confidence: definitive
---

## TL;DR

The s033→s034→s035 density+polish chain (validated end-to-end on seed 89) does
**NOT** generalise to representative cohort seeds. 0/3 pilot seeds produce any
Band A∪B candidates. Two distinct failure modes: (1) phi-sweep ICs too far from
truth → ρ=50 starting point → LM diverges (seed 23); (2) narrow basin unreachable
by phi-sweep → zero filter survivors (seed 28). The **phi-sweep IC primitive is
the architectural bottleneck** — bracket quality and LM polish are downstream.

## What

Full density+polish pipeline (N_dir=2000, N_mag=20, N_phi=12 + LM top-50 +
hi-fi confirm) on three cohort-representative seeds:
- Seed 23: good-bracket (nearest_cell=5.6%), zero-survivor at default density
- Seed 57: highest-survivor (19,847), nearest_cell=49% (far bracket)
- Seed 28: mid-bracket (nearest_cell=7.9%), known sub-Sobol-narrow basin (s006)

## How

Single script (`s036_multi_seed_pilot.py`) calling s020 as subprocess for the
density pass, then inline LM polish (s034 recipe) + hi-fi confirm (s035 recipe)
per seed.

## Result

**0/3 seeds produce Band A∪B candidates.**

| Seed | nearest_cell | Survivors | Best surr ρ | Polish result | Failure mode |
|------|-------------|-----------|-------------|---------------|--------------|
| 89 (ref) | 24.6% | 4,170 | 9.98 | 24/50 Band A | success |
| 23 | 5.6% | 160 | 50.1 | 0/50, best ρ=38.7 | IC quality |
| 57 | 49% | timeout (>30m) | — | — | bracket + scale |
| 28 | 7.9% | 0 | — | — | narrow basin |

### Seed 23 (IC quality failure)
- Bracket is inside the s003 tube (5.6% from truth ω-mag)
- 160 survivors after density pass (vs 0 at default)
- BUT best survivor ρ=50.1 in surrogate — q0 is ~50°+ from truth
- LM polish: 8398s wall (168s/candidate, all hit max_nfev=500)
- Best polished: ρ=38.7, q0→truth=97.5°, ω-dir=32.5°, ω-mag=+427%
- LM diverged — no gradient toward truth from ρ=50

### Seed 57 (bracket + scale failure)
- nearest_cell=49% — structurally outside s003 tube (~2-5%)
- Density pass timed out at 1800s (N_dir=2000 × many cells × high survivor count)
- Even if completed, bracket misses truth by 49% → same ρ-floor as seed 23 or worse

### Seed 28 (narrow basin failure)
- Bracket is inside the tube (7.9% from truth)
- 19.2M candidates generated at N_dir=2000, N_mag=20
- **100% rejected by filter** — zero survivors
- Known s006 narrow-basin seed (radius ~2° in q0)
- Phi-sweep ICs cannot place q0 within ~2° of truth at any phi step

## Why this matters

1. **Phi-sweep IC is the bottleneck.** The density+polish chain requires starting
   ρ < ~15 for LM to converge. Seed 89 achieved this (ρ=10, q0~8° from truth);
   seed 23 did not (ρ=50, q0~50° from truth). The bracket and polish are
   downstream — fixing them doesn't help if ICs are too far.

2. **The off-circle distance varies by seed.** Seed 89: phi-sweep floor at 7.86°
   (bridgeable by LM). Seed 23: phi-sweep floor apparently >>30° (unbridgeable).
   The off-circle distance correlates with whether truth-q0 happens to lie near
   a phi-circle anchored on a classifiable bright peak.

3. **Narrow-basin seeds are structurally unreachable.** Seed 28's ~2° basin is
   smaller than the phi-sweep angular spacing at any tested density. Only ICs
   within ~2° of truth pass the filter, and phi-sweep cannot achieve that
   precision on arbitrary q0 values.

4. **s011's 9/10 Sobol-Shoemake success at truth-ω remains the strongest
   positive result.** Sobol-Shoemake on SO(3) at N=64 bypasses the phi-circle
   constraint entirely. The path forward is likely: bracket provides ω-cells,
   Sobol-Shoemake provides q0 ICs at each cell, LM polishes jointly.

## Numbers

- Seed 23 wall: ~2.5 hr (816s density + 8398s polish + 0s hifi)
- Seed 57 wall: >1800s (timeout during density)
- Seed 28 wall: 531s density, 0s polish (no survivors)
- Total pilot wall: ~3 hr

## Artefacts

- `experiments/s036_multi_seed_pilot.py` — combined pipeline script
- `results/s036_multi_seed_pilot/seed023_result.json` — seed 23 full results
- `results/s036_multi_seed_pilot/seed{023,028}/` — density pass outputs

## Out of scope

- Sobol-Shoemake IC replacement (next experiment)
- Characterising the off-circle distance distribution across the cohort
- Seed 57 at reduced density (would still fail due to bracket position)

## Cross-references

- s035: hi-fi confirmation that validated the chain on seed 89
- s034: LM polish recipe (works from ρ<~15, fails from ρ>~30)
- s033: density sensitivity showing seed 89's off-circle floor = 7.86°
- s011: Sobol-Shoemake N=64 at truth-ω → 9/10 seeds recovered
- s006: seed 28 narrow-basin characterisation
- s003: ω-mag tube width (~2-5% for coherent surrogate landscape)
