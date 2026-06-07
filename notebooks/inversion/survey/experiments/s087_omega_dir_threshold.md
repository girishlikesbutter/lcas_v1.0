---
title: "s087 — ω-direction tolerance for fast-seed polish convergence (~1.25°)"
type: experiment
sources:
  - experiments/s087_omega_dir_threshold.py
  - results/s087/summary.json
related:
  - experiments/s086_omega_refine_rescue.md
  - experiments/s085_anchor_accuracy_gate.md
  - experiments/s082_joint_grid_pivot.md
  - feedback_body_twin_search_space_halving.md
created: 2026-05-21
updated: 2026-05-21
confidence: medium-high (5 axes/level, 3 fast LAM seeds; sharp cliff on 119, noisier on 108/100; surrogate-band with hi-fi spot-check)
---

# TL;DR
**The fast-tumbler polish recovers truth only when the ω-direction error is ≲ 1.25°.** Sweeping ω-direction error at a fixed realistic 2° q-anchor with |ω| held exact (s086 showed magnitude is negligible), seed **119 shows a sharp cliff**: truth-recovery rate (q0_err<5° AND ρ_surr<4) is **5/5 through 1.25°, then collapses to 1/5 at 1.50°** (source: `results/s087/summary.json` `aggregate.119`). Seed 108 is consistent (full recovery to 1.25°, 0.80 at 1.50°). Seed 100 is multi-solution-contaminated — full *truth* recovery only ≤0.25°, but Band A∪B (the plan's primary metric) holds to 1.25° because the polish lands non-truth twins (q0_err 80–134°) in between. **Implication:** the N_dir=2000 Fibonacci grid (s082: ω-dir 0.94–2.69°, median ~1.5°) sits **at the cliff** — it must be ~4–5× denser (N_dir ≈ 9–10k, ~5k unique after the body-X twin halving) to guarantee worst-case ≤1.25°, OR a local ω-direction refine must tighten the coarse grid's direction before the joint polish.

# What
s086 isolated the binding constraint on fast tumblers as the ω-direction error (1.5° → ~13° q0_seed_err through back-propagation; magnitude buys only ~0.2°). This experiment measures the *threshold*: at what ω-direction error does the polish stop recovering truth, across the fast-seed class — which sets the required ω-direction grid density for the cohort run.

# How
Entrypoint `experiments/s087_omega_dir_threshold.py`, Pool(24), seeds 119/108/100 (fast LAM).
- |ω| held at truth (mag 0.0%) to isolate direction; q perturbed by a fixed 2.0° (≈ the s085 800k anchor delivery), held constant.
- Sweep ω-direction error over {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}°, 5 random (q-axis, ω-dir-axis) pairs per level.
- back_propagate → `lm_polish_jacobi` (s064) → band from ρ_surr (s081); hi-fi spot-check the best truth-recovered per seed.
- Metrics per (seed, dir): truth_recovery_rate (q0_err<5° AND ρ_surr<4) and band_AB_rate (ρ_surr<4, includes twins/multi-sols). Controlled perturbation about truth (s005/s064-gate2), not a blind yield claim.

# Result

**Truth-recovery rate (q0_err<5° AND ρ<4) vs ω-direction error:**

| dir° | 119 | 108 | 100 |
|---:|---:|---:|---:|
| 0.00 | 1.00 | 1.00 | 1.00 |
| 0.25 | 1.00 | 1.00 | 1.00 |
| 0.50 | 1.00 | 1.00 | 0.40 |
| 0.75 | 1.00 | 1.00 | 0.80 |
| 1.00 | 1.00 | 0.60 | 0.80 |
| 1.25 | 1.00 | 1.00 | 0.80 |
| 1.50 | **0.20** | 0.80 | 0.40 |
| 2.00 | 0.40 | 0.60 | 0.00 |

- **Seed 119**: clean — full recovery to 1.25° (q0_seed_err 4.8°), collapse at 1.50° (q0_seed_err jumps to 15.5°, ρ_med 26.2 Band D). The cliff is where the back-propagated q0_seed_err exceeds the ~8° polish grab radius.
- **Seed 108**: ≥0.60 throughout, mostly fine to 1.25°; the 1.0° dip (0.60) is 5-axis sampling noise (the amplification is axis-direction-dependent, s086 range 11–17° at fixed 1.5°).
- **Seed 100**: Band A∪B rate stays 1.00 to 1.25° but *truth* recovery is ragged (multi-sol rich LAM seed per s079/s081 — small direction errors send the polish to competing Band-A basins at q0_err 80–134°).

hi-fi spot-checks reconfirm the ρ_surr band proxy: 119 (dir 1.25°) ρ_hifi 0.072 (A); 108 (1.25°) 0.114 (A); 100 (0.25°) 0.046 (A).

# Why this matters
- **Sets the ω-direction grid requirement.** For reliable *truth* recovery on fast tumblers, the grid must guarantee ω-direction within ~1.25°. s082 measured N_dir=2000 delivering 0.94–2.69° (worst-case 2.69°), so the grid is ~2× too coarse in the worst case. Fibonacci nearest-point spacing scales ~N^(−1/2), so worst-case 2.69° → 1.25° needs N_dir ≈ 2000·(2.69/1.25)² ≈ **9300** (≈4600 unique after the s043 body-X twin halving). [derived from the s082 worst-case + sphere-packing scaling, not directly measured.]
- **Or skip the dense global grid:** a coarse grid (N=2000) + a *local* ω-direction refine (gradient or small local grid) to tighten direction below 1.25° before the joint polish is cheaper than a 5× global densification — the natural s088 candidate. A denser global grid multiplies the joint-candidate count (and the surrogate scoring wall, the s082 bottleneck) by ~4.6×, pressuring the 15-min budget unless the v3 multi-anchor pre-prune absorbs it.
- **Multi-solution caveat (seed 100):** even with a tight direction grid, multi-sol-rich LAM seeds will land non-truth Band-A basins. Under the plan's hybrid metric (≥1 Band A∪B primary, truth secondary) that is acceptable, but "truth recovery" specifically needs the tighter tolerance.

# Numbers
- Seed 119 truth-recovery: 1.00 for dir ≤1.25°, 0.20 at 1.50°, 0.40 at 2.00°; q0_seed_err median 4.8° (1.25°) → 15.5° (1.50°) (source: `summary.json` `aggregate.119.per_level`).
- Full-recovery thresholds: 119 = 1.25°, 108 = 1.25°, 100 = 0.25°; 80%-recovery: 119 = 1.25°, 108 = 1.50°, 100 = 1.25°.
- Compute wall: 341 s (5.7 min), Pool(24).

# Artefacts
- `experiments/s087_omega_dir_threshold.{py,md}`
- `results/s087/{summary.json, dir_threshold.png, full_run.log}`

# Out of scope
- The actual N_dir is *derived* from sphere-packing scaling, not measured by re-gridding — a direct N_dir sweep (or a local direction-refine prototype) is the follow-on.
- |ω| held exact and q held at 2°; the true blind pipeline carries both errors jointly (s086 says |ω| adds little, but the joint q+dir interaction at the cliff is unmapped).
- Slow / SAM tumblers (they tolerate looser direction per the s084 regime split — untested here).
- Body-twin folding: truth-recovery uses raw geodesic to truth-q0; an exact body-X twin would read as ~180° and not count, slightly understating recovery on twin-landing axes.

# Cross-references
- `experiments/s086_omega_refine_rescue.md` — established direction (not magnitude) as the fast-seed lever.
- `experiments/s082_joint_grid_pivot.md` — the N_dir=2000 grid delivery (0.94–2.69°) this threshold is compared against.
- `feedback_body_twin_search_space_halving.md` — the 2× halving applied to the derived N_dir.
