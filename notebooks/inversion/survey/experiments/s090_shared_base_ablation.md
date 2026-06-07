---
title: "s090 — shared-base ablation: floating q_a (even on the brightness isophote) does NOT lower the fast-seed ω-dir floor"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s090_shared_base_ablation.py
  - notebooks/inversion/survey/results/s090/ablation.json
  - notebooks/inversion/survey/lib/shoot.py
related:
  - experiments/s089_third_anchor_overdetermination.md
created: 2026-05-22
updated: 2026-05-22
confidence: high (paired within-run comparison, N=300; self-checks exact)
---

# TL;DR
Tested s089's shared-base hypothesis: let the base orientation q_a float in the joint solve so it can slide along its observed-brightness isophote and shed common-mode noise. On seed 119 it **never beats pinning** — at σ_q=2.5° the pinned solve gives ω-dir 1.33° (ac=92ep), free-unconstrained 4.41° (square system, overfits noisy targets), brightness-isophote-constrained 2.24° (source: `results/s090/ablation.json`). **Architectural consequence: the shared-base noise is NOT an averageable lever; the fast-seed accuracy floor is set by anchor quality, not solver freedom.**

# What
s089 found the third anchor disambiguates but doesn't lower the fast-seed noise floor, hypothesising the shared base orientation carries common noise that can't average out. Test: free q_a as extra unknowns. (User correction mid-design: freeing q_a unconstrained makes the geometry-only system square → over-fits noise; the fix is to keep q_a on its brightness isophote — the same |Δmag|<TOL bound that defined the cloud.)

# How
`experiments/s090_shared_base_ablation.py`, seed 119, σ_q ∈ {2.0, 2.5}°, N=300, init-at-truth, Pool(12), v2 surrogate for the brightness residual. Three arms: (1) PINNED q_a (= s089 joint); (2) FREE q_a, no brightness; (3) FREE q_a + brightness-isophote residual `(mag(q_a)−mag_obs)/TOL_MAG`, ML-weighted. New solver `lib/shoot.py::shoot_multianchor_freebase` (residual-callback design, keeps shoot.py surrogate-free).

# Result
ω-dir error (median, deg), seed 119:
| geometry | σ_q | PIN | FREE | BRI |
|---|---|---:|---:|---:|
| ac=92ep | 2.0° | 1.23 | 4.15 | 1.98 |
| ac=92ep | 2.5° | 1.33 | 4.41 | 2.24 |
| ac=123ep | 2.0° | 1.28 | 1.83 | 1.34 |
| ac=123ep | 2.5° | 1.39 | 2.16 | 1.54 |

PIN wins every cell. BRI-float moves q_a 3.1–6.0° (further from truth than its σ_q start). FREE overfits (square system).

# Why this matters
- **Shared-base hypothesis REFUTED.** No solver freedom on q_a helps.
- **Why even the correct constraint fails:** the brightness isophote pins only 1 of q_a's 3 rotational DOF; the other 2 (tangent to the isophote) stay free at *any* weight and get spent fitting endpoint noise. Closes the sensitivity question analytically — no weighting makes floating beat pinning.
- The fast-seed ω-dir floor is set by anchor quality (s085 @800k → 1.56°) or downstream polish — confirms the lever is denser sampling, not the solver.

# Numbers
- σ_q=2.5°, ac=92: PIN 1.33 / FREE 4.41 / BRI 2.24°; ac=123: 1.39 / 2.16 / 1.54° (source: `results/s090/ablation.json`).
- BRI base-shift 3.8–6.0°; |ω| incidental (≤0.44%). Wall 10s.

# Artefacts
- `results/s090/ablation.json`; `lib/shoot.py::shoot_multianchor_freebase`.

# Out of scope
- Whether denser anchors get fast-seed ω-dir < 1.25° (the s087 gate) — the standing density lever.

# Cross-references
- `s089_third_anchor_overdetermination.md` — the hypothesis tested here.
- `s087` — the 1.25° fast-seed gate; `s085` — anchor q-accuracy.
