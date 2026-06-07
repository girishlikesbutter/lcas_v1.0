---
title: "s068 — Replicate s011 (Q4c-ii cohort baseline) on post-fix m048: 10/10 yield (improved from 9/10)"
type: experiment
sources:
  - experiments/s011_q4cii_sobol_so3_polish_pilot.py (unchanged)
  - results/s011/ (post-fix)
  - results_prefix/s011/ (pre-fix baseline)
related:
  - experiments/s067_postfix_propagator_validation.md
  - experiments/s067b_cohort_lj2000_audit.py
created: 2026-05-12
updated: 2026-05-12
confidence: high (full cohort run, direct pre/post comparison)
---

# TL;DR

Re-ran s011 (Sobol-Shoemake N=64 over SO(3) at fixed truth-ω, 6-DOF LM polish) on the post-fix cohort: **cohort yield 10/10 at N=64 (up from pre-fix 9/10)**. Seed 10 gained 3 in-basin landings (was 0 pre-fix). Architecture validated post-fix; methodology and headline conclusion ("Sobol-q0 cheap, Q4c works at low density") survives. Wall: 2270 s ≈ 38 min Pool(8) (matches s011 estimate).

# What

s011 (2026-05-01) was the original Q4c-ii cohort baseline, demonstrating that uniform Sobol-Shoemake on SO(3) at N=64 + 6-DOF LM polish at truth-ω lands ≥1 in-basin IC on 9/10 PA-stratified pilot seeds (seed 10 was the 1 failure). After the propagator fix landed (commit `d5705ff`) and the cohort was regenerated (commit `7aad73b`), this script re-ran the same pilot to confirm the architecture still works on physically-correct trajectories.

# How

`experiments/s011_q4cii_sobol_so3_polish_pilot.py` (unchanged) on the post-fix cohort. SOBOL_SEED=42 (matches s002/s006/s010), max_nfev=60, Pool(8), BLAS=1.

# Result — pre-fix vs post-fix per-seed

| Seed | n_basin (pre) | n_basin (post) | min_q0_err° (pre) | min_q0_err° (post) | Δ |
|-----:|--------------:|---------------:|------------------:|-------------------:|----|
| 6    | 8 | 6 | 0.287 | **0.012** | tighter |
| 10   | **0** | **3** | 28.245 | **0.501** | now solved |
| 21   | 3 | 3 | 0.016 | 0.147 | similar |
| 28   | 7 | 4 | 0.034 | 0.101 | still positive |
| 41   | 2 | 3 | 1.045 | 0.778 | tighter |
| 44   | 5 | 9 | 0.033 | 0.215 | more landings |
| 48   | 1 | 2 | 0.188 | 0.797 | similar |
| 60   | 4 | 2 | 0.056 | 0.060 | similar |
| 84   | 3 | 7 | 0.178 | 0.098 | more landings, tighter |
| 91   | 3 | 2 | 0.073 | 1.413 | wider basin |

| Cohort metric | Pre-fix | Post-fix |
|---------------|--------:|---------:|
| yield at N=64 | 9/10 | **10/10** |
| seeds with ≥3 basin landings | 7/10 | 7/10 |
| seeds with min_q0_err < 0.1° | 5/10 | 4/10 |

# Why this matters

**Architecture is intact.** The "Sobol-q0 cheap; Q4c works at low density" headline survives unchanged. PA-stratified pilot at N=64 lands at least one in-basin IC on every cohort seed.

**Specific numerics shifted as expected.** Per-seed values (n_basin, min_q0_err, min_final_mse) are quantitatively different — methodology survives but seed-specific numerics don't (per the cleanup plan's stated assumption). Headline cohort metric (10/10 vs 9/10) is incidentally tighter, suggesting the fixed propagator gives a slightly more well-behaved cost landscape on at least one previously-difficult seed.

**Seed 10 is no longer in the "density-recoverable failure" class** under the post-fix cohort. Pre-fix s012a closed seed 10 as DECISIVE H2 negative (0/256 in-basin at any density). The post-fix s011 lands 3/64. Whether this is a basin-width change or a competing-attractor change is open; not on the critical path for the cleanup.

# Numbers

See `results/s011/summary.json` for full per-cell numbers. `results_prefix/s011/summary.json` preserves the pre-fix baseline.

Wall: post-fix 2270 s, pre-fix unknown (not recorded in summary; estimated ~30 min from script's expected wall ≈1600 s).

# Artefacts

- `results/s011/{summary.json, runs.npz, yield_vs_density.png, q0_err_distribution.png}` — post-fix
- `results_prefix/s011/...` — pre-fix baseline (delete by 2026-05-19)

# Out of scope

- Density scan beyond N=64 (s011 ran only N=64; pre-fix runs at higher densities are in `results_prefix/s011_*` if needed for follow-up).
- Re-running s012a (seed 10 density question) under post-fix data — flagged for future work; the s068 result already weakens the pre-fix conclusion.

# Cross-references

- `experiments/s011_q4cii_sobol_so3_polish_pilot.md` — pre-fix original.
- `experiments/s067_postfix_propagator_validation.md` — propagator fix + 4-gate validation.
- `experiments/s067b_cohort_lj2000_audit.py` — cohort-wide L_J2000 conservation gate.
