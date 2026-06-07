---
title: "s086 — does a 1-D |ω| refine rescue seed 119 at L1? (no: the failure is direction-driven)"
type: experiment
sources:
  - experiments/s086_omega_refine_rescue.py
  - results/s086/summary.json
related:
  - experiments/s085_anchor_accuracy_gate.md
  - experiments/s084_omega_refine_viability.md
  - experiments/s087_omega_dir_threshold.md
  - project_q_wdir_coupling.md
created: 2026-05-21
updated: 2026-05-21
confidence: high (Layer 1 decomposition over 6 q_pert×axis is deterministic; Layer 2 polish over 6 axes/treatment, 2 seeds, hi-fi spot-checked)
---

# TL;DR
**No — a 1-D |ω| refine cannot rescue the s085 L1 failure on fast seed 119, because the failure is driven by the 1.5° ω-DIRECTION error, not the 0.5% |ω|-magnitude error.** Back-propagation error decomposition (source: `results/s086/summary.json` `layer1_decomposition`): for seed 119 the median q0_seed_err is **12.90°** with both errors, **12.66°** with magnitude removed (the ceiling of a perfect refine — barely changed), but drops to **4.03°** with the direction removed. Layer 2 confirms three ways: (a) oracle-perfect |ω| still fails to recover truth on 119 (0/6 converge; its one Band A∪B hit is a **169.62° twin** at ρ_hifi 1.36, not truth); (b) the 1-D refine is non-robust to the direction error — pulled to |ω| **−15%**, landing **122° off** in q0 (a real effect: at the wrong 1.5° direction, ρ(−15%)=31.8 < ρ(truth-|ω|)=36.5, but both Band D); (c) seed 100 (which already passed L1) sees the refine recover |ω| to +0.65% and slightly help — so refine robustness is seed-dependent, and it fails on the seed that actually needs it. **The lever for fast tumblers is a tighter ω-direction grid (<1.5°), not a |ω| refine.**

# What
s085 showed seed 119 fails the realistic-blind-ω level (L1: dir 1.5°, |ω| 0.5%) — only 1/3 axes reach Band A∪B because the ω error amplifies into a 12–20° q0_seed_err through 738° of back-propagation. PROGRESS's revised stage order (`anchor q → grid ω-dir → 1-D refine |ω| → joint polish`) proposes refining |ω| first. The decisive uncertainty: of that 12–20° q0_seed_err, how much comes from the 0.5% magnitude (which the refine fixes) vs the 1.5° direction (which it does not)?

# How
Entrypoint `experiments/s086_omega_refine_rescue.py`, Pool(24), seeds 119 + 100, q_pert {2,3}° × 3 axes. Same anchor T_A as s085; rng scheme matches s085 layer_b exactly so qa_p / om_p-direction are identical.
- **Layer 1 (analytical, back_propagate only):** measure q0_seed_err under 4 ω-seed treatments — (dir 1.5, mag 0.5) baseline, (dir 1.5, mag 0.0) oracle_mag = refine ceiling, (dir 0.0, mag 0.5) oracle_dir, (dir 0.0, mag 0.0) = L0.
- **Layer 2 (polish + hi-fi):** three treatments — baseline (reproduces s085 L1), oracle_mag (perfect |ω|, wrong dir), refined (|ω| from a 2-stage 1-D full-LC surrogate-MSE scan over the ls_bracket, the s084 method, holding the wrong direction). back_propagate → `lm_polish_jacobi` → hi-fi classify the best converged per (seed, treatment).

# Result

**Layer 1 — q0_seed_err (median over 6 q_pert×axis):**

| treatment | seed 119 | seed 100 |
|---|---:|---:|
| baseline `dir1.5 mag0.5` | 12.90° | 15.10° |
| oracle_mag `dir1.5 mag0.0` (refine ceiling) | 12.66° | 12.59° |
| oracle_dir `dir0.0 mag0.5` | 4.03° | 10.42° |
| L0 `dir0.0 mag0.0` | 2.50° | 2.50° |

On 119, removing magnitude buys ~0.2°; removing direction buys ~9°. (On 100 both contribute and the per-axis range is large, 9.6–41°.)

**Layer 2 — polish outcome (median ρ_surr; Band A∪B / converged out of 6):**

| treatment | seed 119 | seed 100 |
|---|---|---|
| baseline | ρ 25.24 (D); 2/6 A∪B, 2/6 conv | ρ 0.61 (A); 6/6 A∪B, 3/6 conv |
| oracle_mag | ρ 27.49 (D); 1/6 A∪B (=169° twin), 0/6 conv | ρ 0.22 (A); 6/6 A∪B, 5/6 conv |
| refined | ρ 26.75 (D); 0/6 A∪B, \|ω\|err −15% | ρ 0.22 (A); 6/6 A∪B, 4/6 conv, \|ω\|err +0.65% |

hi-fi spot-checks: 119 baseline best = ρ_hifi 0.072 (A), q0_err 0.10° (**truth**); 119 oracle_mag best = ρ_hifi 1.362 (A) but q0_err **169.62°** (**twin, not truth**); 100 all treatments → ρ_hifi 0.046 (A), q0_err 0.25°.

# Why this matters
- **Kills the "refine |ω| to rescue fast seeds" hypothesis.** The fast-seed L1 failure is direction-error amplification; |ω| is already a solved/separable axis (s084) and refining it is at best neutral, at worst harmful (the 122° pull-off).
- **The refine is non-robust to a direction error.** s084 proved truth-|ω| is the unique global min *given truth direction*; with a 1.5° direction error the MSE-minimizing |ω| shifts (seed 119: −15%), because no |ω| fits a wrong-direction trajectory well on a fast tumbler — the whole neighborhood is Band D. Robustness is seed-dependent (100 is fine, 119 is not).
- **Redirects the architecture:** the v3 stage order is sound but the *lever* for fast seeds is ω-direction grid density / a direction refine, not the |ω| refine. → s087 measures the required direction tolerance.
- Body-twin caveat: oracle_mag's lone Band A∪B on 119 is a 169.62° multi-solution. Under the plan's hybrid metric a ρ<4 multi-sol counts, but it is NOT truth recovery; report the two separately.

# Numbers
- Layer 1 seed 119: 12.90 / 12.66 / 4.03 / 2.50° (baseline/oracle_mag/oracle_dir/L0) (source: `summary.json` `layer1_decomposition.119`).
- Refine pull-off verified: ρ(−15% |ω|, wrong dir 1.5°, qa 3°) = 31.8 < ρ(truth-|ω|, same) = 36.5, both Band D (in-session check, reproducible from `summary.json` `rows[].refine_info`).
- Compute wall: 366 s (6.1 min), Pool(24).

# Artefacts
- `experiments/s086_omega_refine_rescue.{py,md}`
- `results/s086/{summary.json, seed119_refine_rescue.png, seed100_refine_rescue.png}`

# Out of scope
- Whether tightening the ω-direction grid below 1.5° rescues 119 — that is s087.
- Slow / SAM seeds (s086 ran two fast LAM seeds).
- A *joint* direction+magnitude refine or a direction-gradient step before the polish (untested alternatives).

# Cross-references
- `experiments/s085_anchor_accuracy_gate.md` — produced the L1 failure this decomposes.
- `experiments/s084_omega_refine_viability.md` — refine robustness was proven only with truth direction; s086 shows the direction-error gap.
- `experiments/s087_omega_dir_threshold.md` — measures the required ω-direction tolerance.
- `project_q_wdir_coupling.md` — q and ω-dir jointly determine the LC; only |ω| is separable. s086 is the empirical edge of that coupling on fast tumblers.
