---
title: "s084 — is 1-D |ω|-refine viable? (aliasing + realistic-anchor budget)"
type: experiment
sources:
  - experiments/s084_omega_refine_viability.py
  - results/s084/summary.json
related:
  - experiments/s083_nmag_sensitivity.md
  - experiments/s082_joint_grid_pivot.md
  - project_q_wdir_coupling.md
  - project_omega_mag_basin_scales_with_omega.md
created: 2026-05-21
updated: 2026-05-21
confidence: high (1-D full-bracket scan, 3 seeds; q-perturbation over 3 random axes per error level)
---

# TL;DR
**The 1-D |ω|-refine idea is viable.** Over the *full* LS-bracket (8.6–142× wide) with q+dir at truth, truth-|ω| is the **unique global minimum** — exactly **1 local minimum below ρ=8, ZERO aliases** on all three seeds (source: `results/s084/summary.json` `part1_global_structure`). So a 1-D scan/descent has nothing to fall into. The refine recovers the **correct |ω| robustly even at 15° q-error** (recovered |ω|-err ≤3% slow, ≤0.4% fast). **But q-error sets a regime-dependent floor on the achievable fit:** the slow seed reaches Band B even at 15° q-error, while the fast seed needs q within **~1–2°** (3° q-error → only ρ 6.09, Band C) even with |ω| perfect. So |ω| is solved as a separable axis; the binding constraint moves to **anchor q-accuracy**, which on fast tumblers is tighter than the 5–15° LM grab radius the plan assumes.

# What
s083 showed |ω| has a sharp, unique V given truth q/dir, motivating "pull |ω| out of the joint grid and refine it in 1-D." Two unknowns decided whether that's sound: (1) aliasing — is truth the *global* min over the full bracket, or are there competing dips? (2) realistic anchor — does a findable truth-|ω| min survive when q is only anchor-quality (a few degrees off), and at what q-error does it stop reaching Band B?

# How
Entrypoint `experiments/s084_omega_refine_viability.py`, 3 s082 seeds, elliprj Path 2 + surrogate-v2 full-LC residual, Pool(24), 15375 candidates.
- **Part 1** (aliasing): scan |ω| over the full `ls_bracket` [lo,hi] at 1500 log-spaced points (+ exact truth) holding q+dir at truth; detect local minima (scipy `find_peaks` on −ρ) and flag any with ρ<8 more than 3% from truth-|ω|.
- **Part 2** (anchor budget): perturb q0 by 3/5/10/15° about 3 random axes each (geodesic error = the stated angle), dir at truth, scan |ω| over ±15% of truth (301 pts); report min ρ and the |ω| at the min.
Truth used only to centre the sweep and perturb q — basin/sensitivity probe, not a search-yield claim.

# Result

**Part 1 — global structure (q+dir = truth):**

| Seed | bracket span | global-min \|ω\|-err | global-min ρ | local minima <ρ8 | aliases >3% off | deep aliases <ρ4 |
|---|---:|---:|---:|---:|---:|---:|
| 116 | 8.6× | 0.00% | 0.20 | 1 | 0 | 0 |
| 119 | 142× | 0.00% | 0.36 | 1 | 0 | 0 |
| 103 | 54.9× | 0.00% | 0.33 | 1 | 0 | 0 |

**Part 2 — anchor q-budget (min ρ_surr median over 3 axes / recovered |ω|-err median):**

| Seed | 3° q-err | 5° | 10° | 15° |
|---|---|---|---|---|
| 116 LAM-slow | 0.55 (A) / −0.6% | 0.88 (A) | 1.74 (A) | 2.59 (B) / −2.9% |
| 103 SAM | 5.78 (C) / −0.1% | 9.32 (D) | 16.60 (D) | 18.86 (D) / −0.4% |
| 119 LAM-fast | 6.09 (C) / 0.0% | 8.10 (D) | 13.06 (D) | 16.41 (D) / −0.2% |

# Why this matters
- **The |ω| axis is solved as a component.** No aliasing → a coarse 1-D scan + local descent over the bracket nails |ω|; the recovered |ω| is correct (≤0.4% on fast/SAM) regardless of q-error magnitude. Build the 1-D refine into v3: *anchor q → grid ω-dir → 1-D refine |ω| → joint q+|ω| polish.*
- **The binding constraint moved to anchor q-accuracy, and it's regime-dependent.** Fast tumblers (more revolutions = more independent looks) are more uniquely determined (no aliases, tight basin) but need a ~1–3° anchor to clear Band B even with perfect |ω| — tighter than the 5–15° LM grab radius (s005). Slow tumblers tolerate ~15°. The open v3 question is therefore "**can anchoring deliver q within ~1–3° on fast seeds?**"
- This sharpens (does not contradict) the plan: |ω| is fixable cheaply, but the plan's anchor-accuracy assumptions need a fast-tumbler-specific check.

# Numbers
- Part 1: global-min |ω|-err 0.00% all seeds; ρ 0.20/0.36/0.33; n_local_minima<ρ8 = 1; n_aliases>3%-off<ρ8 = 0; n_deep_aliases<ρ4 = 0 (source: `results/s084/summary.json:seeds.*.part1_global_structure`).
- Part 2 (median min ρ): 116 → 0.55/0.88/1.74/2.59; 119 → 6.09/8.10/13.06/16.41; 103 → 5.78/9.32/16.60/18.86 for 3/5/10/15° (source: same, `part2_anchor_budget`). Per-axis spread wide: e.g. 119@5° ρ range 5.11–12.06.
- Recovered |ω|-err always small: ≤2.9% (116), ≤0.2% (119), ≤0.4% (103) across all q-errors.
- Compute wall: 91.8 s, Pool(24).

# Artefacts
- `experiments/s084_omega_refine_viability.{py,md}`
- `results/s084/summary.json`
- `results/s084/seed_{116,119,103}_refine_viability.png` (P1 full-bracket scan + P2 |ω| scan vs q-error)

# Out of scope
- Whether v3 anchoring can hit ~1–3° q on fast seeds (the new binding question — untested).
- P2 held ω-direction at truth (real anchors add ~1° dir error → floor slightly higher).
- Aliasing structure at off-truth q (Part 1 used truth q; Part 2 confirms the *near-truth* min survives but didn't re-map the full bracket at each q-error).
- Hi-fi confirmation (surrogate-only; s081-validated bands).

# Cross-references
- `experiments/s083_nmag_sensitivity.md` — measured the basin this refine exploits.
- `project_q_wdir_coupling.md` — "only ω-mag is truly independent" — the basis for treating |ω| as a separable 1-D axis.
- `report/blind_inversion_15min_plan_2026-05-20.md` §4.2 — v3 anchoring stage, which now carries the ~1–3° fast-seed q-budget.
