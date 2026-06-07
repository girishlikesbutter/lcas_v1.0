---
title: "Observational indistinguishability — ρ < 2 is a valid solution regardless of twin status"
type: concept
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# Observational indistinguishability

## The principle

If a candidate `(q0_cand, ω_cand)` produces a hi-fi LC that matches the observed LC at ρ < 2 (Band A; below the noise floor by ~2×), then **the candidate is observationally indistinguishable from truth**, regardless of whether `(q0_cand, ω_cand)` is geometrically near `(q0_truth, ω_truth)`.

Concretely: if the candidate is a twin attractor (per `twin_degeneracy.md`) or an ω-sign degenerate (per `omega_sign_degeneracy.md`) and produces an LC that fits, the data does not distinguish. Subjectively saying "but it's not THE truth" is meaningful for cataloguing geometric distance to known truth, but it is not a quality judgement on the inversion — the inversion has produced a state that explains the observation.

## Implication for survey scoring AND inversion success criterion

Do NOT define success as "truth basin recovered." Define success as **"return all candidates with ρ < 4 (Band A∪B), and retain ρ < 8 (Band C) flagged as suboptimal."** A survey question like "what fraction of seeds admits a ρ < 4 inversion under cost surface X?" is well-posed; "what fraction of seeds recovers the truth basin under cost surface X?" is ambiguous and may understate solver quality.

The workspace's inversion endgame inherits this convention directly:

- **Band A∪B candidates** are the inversion's primary output. Multiple per seed are expected and admitted; we want as many as physically valid given the LC information content.
- **Band C candidates** are kept and reported but flagged as "less than ideal" — visually presentable, not publishable.
- **Band D candidates** are rejected.
- **Geometric truth recovery is a desirable subset of the goal, NOT the gating criterion.** The truth basin is one valid `(q0, ω)` candidate among possibly several; the inversion should return it when LC information uniquely determines attitude (most of the cohort) and return multiple valid candidates when LC information under-determines it (e.g. low-rotation tumblers — seed 10; multi-solution-rich seeds — 28 / 41 / 48 / 84 per s014).

When reporting, distinguish:
- `n_basins_below_rho_X` — observational fits below threshold X (use ρ < 2, < 4, < 8)
- `n_truth_near` — additionally requires q0_err < some bound and ω_err < some bound
- `n_twin_near` — same but anchored on the symmetry-related twin
- `n_other_attractors` — ρ < 4 but NOT geometrically near truth or any known degenerate state — these are the real "discoveries" that confirm multi-solution under-determination

## Where this is most important

For the survey's failure-mode taxonomy (question 3), a seed that has multiple ρ < 2 attractors at distinct (q0, ω) points is genuinely ambiguous — the data does not pin down the geometric truth. Such a seed is NOT a solver failure; it's an information-theoretic limit of the observation. The survey should flag these explicitly so that any downstream inversion approach knows when to report "multi-modal solution" rather than "best single answer."

## Cross-references

- Auto-memory: `feedback_observational_indistinguishability.md`, `feedback_multi_solution.md` (multi-solution philosophy)
- Survey concepts: `rho_band.md`, `twin_degeneracy.md`, `omega_sign_degeneracy.md`
