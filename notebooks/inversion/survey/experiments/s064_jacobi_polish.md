---
title: "s064 — Jacobi-coord LM polish: 5/50 Band A on seed 89 vs s059k baseline 4/50, with NEW truth-basin cluster id=125"
type: experiment
sources:
  - experiments/s058_lm_polish_clusters.py
  - experiments/s059k_full_lc_from_seeds.py
  - experiments/s059k_densify_ndirs.md
  - experiments/s063b_polhode_curvature.md
  - lib/jacobi_propagator.py
related:
  - project_jacobi_propagation_priority.md
  - feedback_local_window_polish_phantom_basins
created: 2026-05-12
updated: 2026-05-12
confidence: high (Gates 1, 2, 3 PASS on seed 89; new Band A cluster id=125 not in s059k baseline)
---

# TL;DR

Reparameterising LM polish's ω onto the **polhode basis** (Gram-Schmidt from grad 2T and grad L² at ω_seed, per s063b) lifts seed-89 Band A yield from **4/50 → 5/50 unique clusters**. The new cluster id=125 lands on the truth basin (q0_err=0.86°, |ω|err=-0.02%, ω_dir=0.11°, hi-fi ρ=0.176 — same basin s059k recovers via cluster id=338 only). Smoke tests on basin-reachable perturbations converge **21-27× faster** in wall (3.1-4.0 s vs 85.0-85.9 s for s059k's full-LC polish on the same q_a/ω perturbations). Method stays `scipy.least_squares(method='lm')`; the only change is the residual closure, which substitutes raw ω-components with polhode-basis coefficients.

# What

s063b showed: at truth (q₀, ω₀), the cost surface curvature along the polhode tangent (Euler direction `I^-1((I·ω)×ω)`) is **4.2-17.5× lower** than along the polhode normals (`grad 2T = 2 I·ω` and `grad L² = 2 I²·ω`, Gram-Schmidt). This implies LM polish takes inefficient steps in mixed-curvature coordinates: too aggressive along normals (overshoot the sharp basin), too cautious along tangent (under-explore the soft direction). Reparameterising ω → (y_tangent, y_normal_E, y_normal_L) decouples the soft and sharp directions; LM's internal damping can balance them.

s064 implements this as a drop-in for `s058::lm_polish` and tests it against s059k's seed-89 Phase 4 pipeline.

# How

Three gates were defined in the plan:

**Gate 1 — Truth invariance.** From truth `(q0_truth, ω0_truth)` on seed 89, the polish must converge near the surrogate floor (`ρ_pol ≈ ρ_seed`), not move ω in the polhode basis (`|y_pol| ~ machine epsilon`), and land Band A in hi-fi.

**Gate 2 — Smoke parity.** Match s059k_smoke_seed89's three perturbations (test1: q_a=1.7°/ω_dir=3.5°/|ω|+6%; test2: q_a=0.85°/ω_dir=1.75°/|ω|+0%; test3: q_a=1.7°/ω_dir=3.5°/|ω|+0%). Compare to cached s059k full-LC baselines. Pass criterion: test2 + test3 must hit Band A (they did in s059k).

**Gate 3 — Cohort regression.** Re-run s059k_full_lc_from_seeds Phase 4 with `lm_polish_jacobi` substituted for `lm_polish`. Pass criterion: **≥ 4/50 unique Band A clusters** (= s059k baseline). Stretch: identify a NEW Band A cluster not in the s059k baseline.

# Result

**Gate 1: PASS.** ρ_seed_surr = 0.404 → ρ_pol_surr = 0.391 (Δ = -0.013, LM converged not regressed). |y_pol| = 8.04e-6 (ω basically unchanged in polhode coords). Hi-fi ρ = 0.177 Band A. Wall: 4.9 s.

**Gate 2: PASS.** Polished ρ matches s059k baseline exactly (basin-reachable tests 2 & 3 both land at ρ=0.177 Band A; basin-unreachable test 1 lands at ρ=24.5 Band D, same as s059k — mag-noise too large, motivates multi-mag-start). Wall comparison:

| Test | Perturbation | s059k full-LC ρ_hifi | s064 ρ_hifi | s059k wall | s064 wall | Speedup |
|------|--------------|------:|------:|------:|------:|------:|
| 1    | q_a=1.7°, ω_dir=3.5°, \|ω\|+6% | 24.495 (D) | 24.490 (D) | 102.4 s | 24.7 s | 4.1× |
| 2    | q_a=0.85°, ω_dir=1.75°, \|ω\|+0% | **0.177 (A)** | **0.177 (A)** | 85.0 s | 3.1 s | **27×** |
| 3    | q_a=1.7°, ω_dir=3.5°, \|ω\|+0% | **0.177 (A)** | **0.176 (A)** | 85.9 s | 4.0 s | **21×** |

In-basin convergence is dramatically faster. Out-of-basin polishes are similar or modestly faster.

**Gate 3: PASS, with stretch.** Seed-89 Phase 4 pipeline (50 clusters × 5 mag-starts = 250 polishes, Pool(8)):

- **5/50 unique Band A clusters** (s059k baseline: 4/50). 14 Band A polishes total (s059k: 11).
- Polish wall: 19.97 min Pool(8). Hi-fi gating: 9.75 min.
- Unique Band A clusters found: **125 (NEW)**, 325 (body-twin basin), 338 (truth basin), 364, 368 (q0=60° multi-sol basin).
- Cluster **id=125 (rank 7)** is the new finding: q0_err=0.86°, ω_dir=0.11°, |ω|err=-0.02%, hi-fi ρ=0.176. Same (q0, ω) basin as the truth basin (id=338), but reached from a different cluster representative that s059k's standard polish did not converge from.

Truth cluster rank: 797/2412 (= not in top-50; truth was not directly polished). Gate 3 yield is genuine architecture output, not oracle injection.

# Why this matters

**Drop-in win.** s064 is a 60-line patch to `s058::lm_polish` (basis computation at the seed + linear change of variables; the rest of the LM loop is unchanged). It produces an extra Band A cluster on the canonical benchmark seed without changing the score grid, score function, clustering, or candidate set.

**Validates the s063b finding.** The curvature ratio (4-17×) wasn't just an interesting geometry fact — it translates directly into convergence speed. The 21-27× smoke speedup on in-basin polishes is in the ballpark of `(ratio_normal / ratio_tangent)²` since LM's nfev scales roughly with the condition number.

**Architectural implication.** The polhode basis should become the **default** parameterisation for any LM polish step in the inversion pipeline. The Gram-Schmidt construction is O(1); the basis storage is 3×3; the parameter mapping is linear. No reason to keep raw ω-components.

**Limitation.** Polishes that start outside the truth basin still drift around in Band D — the polhode basis improves convergence WITHIN a basin but doesn't increase the basin radius. Multi-mag-start (s059k Phase 4 contribution) and ω-grid densification (s059k Phase 2) remain necessary to GET into a basin. s064 makes the IN-basin step efficient.

# Numbers

Gates 1-3 summarized:

| Gate | Description | Result | Wall |
|------|-------------|--------|------|
| 1    | Truth invariance, seed 89 | PASS (ρ_pol_surr=0.391, hi-fi Band A) | 4.9 s |
| 2    | s059k smoke parity (3 tests) | PASS (test2/3 Band A, test1 Band D = s059k) | 32 s total |
| 3    | Seed-89 cohort regression | **PASS+stretch: 5/50 Band A clusters** (s059k: 4/50) | ~30 min Pool(8) total |

In-basin smoke speedup: 21-27× (4.0 s vs 85.9 s on test3).

# Artefacts

- `experiments/s064_jacobi_polish.py` — script: gates 1, 2, 3 selectable via `--gate1 / --gate2 / --gate3 --in-dir ...`. Reuses `s058::lm_polish` quat helpers and `s063b::gram_schmidt_basis` without modification.
- `results/s064_jacobi_polish/gate1_truth_invariance.json`
- `results/s064_jacobi_polish/gate2_smoke_parity.json`
- `results/s064_jacobi_polish/gate3_seed089_summary.json`

# Out of scope

- Cohort generalisation (other seeds beyond 89). The 4-17× curvature ratio holds on seeds 28 + 14 + 89 (s063b cohort), so s064 should generalise; left as next chunk if needed.
- Re-running the basis at each LM iteration (a "nonlinear reparameterisation"). The fixed-at-seed basis works for in-basin polishes (|y_pol|/|ω_seed| < 5% in successful Gate 2 cases). If basin-jumping polishes become a use case, swap in dynamic basis recomputation (s064b).
- Switching `method='lm'` to `method='trf', x_scale='jac'`. LM's internal damping handles the 4-17× ratio adequately; no need for auto-scaling.
- Per-column rescaling of B by analytical curvature scales. Not needed for the Gate 3 result; deferred if future cohort tests show under/overstepping along a specific direction.
- Substituting `propagate_jacobi` for `propagate_attitude` in the residual. The polish path is ~50-200 evals per polish, where DOP853 is cheap enough; the Jacobi speedup is only meaningful for ~100k-eval enumerators (s062d). Out of scope.

# Cross-references

- `experiments/s058_lm_polish_clusters.py` — vanilla LM polish substrate.
- `experiments/s059k_full_lc_from_seeds.py` — Phase 4 pipeline being benchmarked against.
- `experiments/s059k_densify_ndirs.md` — s059k's headline 4/50 baseline.
- `experiments/s063b_polhode_curvature.md` — the curvature finding that motivated s064.
- `lib/jacobi_propagator.py` — Jacobi closed-form ω + hybrid q (used here implicitly via `gram_schmidt_basis` which uses textbook Euler dω/dt formula, NOT the propagator's q ODE — so s064 is unaffected by the s066 L_J2000 non-conservation finding).
