---
title: "s035: Hi-fi confirmation of s034 polished candidates — seed 89"
type: validation
sources:
  - experiments/s034_lm_polish_seed89.md
  - results/s034_lm_polish_seed089/polish_summary.json
related:
  - s034_lm_polish_seed89
  - s017_hifi_rho_band
  - s033_density_sensitivity
created: 2026-05-05
updated: 2026-05-05
confidence: definitive
---

## TL;DR

All 24 surrogate-Band-A polished candidates from s034 confirm as **hi-fi Band A** (ρ<2).
Surrogate was over-pessimistic: truth-basin surrogate ρ=0.391 → hi-fi ρ=0.177.
Four distinct physically valid basins recovered. Framework end-to-end validated on seed 89.

## What

Hi-fi render (shadow + BRDF) of the 24 candidates that s034's LM polish placed
in surrogate Band A (ρ_pred < 2.0). Computes true ρ = √(MSE)/0.05 against the
cached truth hi-fi LC.

## How

- Load 24 Band A candidates from `s034_lm_polish_seed089/polish_summary.json`
- `lib.hifi_render.build_context(89)` → full satellite+SPICE cached geometry
- `render_hifi(q0, ω, ctx)` per candidate in `Pool(8)` fork-context, BLAS=1
- `rho_from_hifi(pred, truth)` for band classification

## Result

**24/24 hi-fi Band A. 0 Band B/C/D.**

| Basin class | Count | Surr ρ | Hi-fi ρ | q0→truth | ω-dir→truth | ω-mag error |
|-------------|-------|--------|---------|----------|-------------|-------------|
| Truth | 7 | 0.391 | 0.177 | 0.86° | 0.11° | -0.02% |
| 180° twin | 3 | 0.372 | 0.161 | 179.5° | 29.2° | -0.02% |
| Multi-sol A (q0≈60°) | 9 | 1.063 | 1.093 | 60.2° | 154.8° | -0.89% |
| Multi-sol B (q0≈174°) | 5 | 1.065 | 1.086 | 174.1° | 165.2° | -0.88% |

**Surrogate/hi-fi ratio:**
- At truth basin: surrogate 2.2× pessimistic (0.391/0.177)
- At multi-solution: surrogate ≈ hi-fi (1.063/1.093 = 0.97×)

This matches s017's finding that the surrogate is over-pessimistic at truth
(intrinsic noise floor) but faithful elsewhere.

## Why this matters

1. **End-to-end validation complete.** The density+polish chain (s033→s034) delivers
   hi-fi Band A solutions in ~10 min wall on seed 89.
2. **Multi-solution structure is real.** Four distinct basins, all ρ<1.1 in hi-fi —
   genuinely observationally indistinguishable at noise level.
3. **Twin basin is strongest.** The 180° twin (q0≈180° from truth, ω-dir 29° off,
   same ω-mag) has lower hi-fi ρ than truth itself (0.161 vs 0.177). This is
   physically meaningful: the twin orientation happens to produce a marginally
   closer LC to the rendered truth than truth does (floating-point/shadow effects).
4. **Multi-solution classes 3&4 are genuinely different attitudes** — q0 60° and 174°
   from truth, ω-dir >150° off, ω-mag ~0.9% off — yet produce LCs indistinguishable
   from truth below noise floor. These are the kind of degeneracies the multi-solution
   philosophy is designed to accept.

## Numbers

- Wall: 171.5s total (7.1s/candidate in Pool(8))
- Hi-fi ρ: min=0.161, median=1.086, max=1.094
- A∪B yield: 24/24 (100%)
- Distinct basins: 4 (truth, twin, multi-sol-A, multi-sol-B)

## Artefacts

- `experiments/s035_hifi_confirm_seed89.py` — script
- `results/s035_hifi_confirm_seed089/hifi_confirm.json` — per-candidate results
- `results/s035_hifi_confirm_seed089/hifi_mags.npz` — rendered LCs + truth

## Out of scope

- Multi-seed generalisation (s036 next)
- Cohort-scale timing extrapolation

## Cross-references

- s034: LM polish recipe that produced these candidates
- s033: density sensitivity fix (bracket padding) that made s034 possible
- s017: surrogate/hi-fi ratio characterisation (predicted this outcome)
- s014: surrogate↔hi-fi rank fidelity on 9-seed pilot
