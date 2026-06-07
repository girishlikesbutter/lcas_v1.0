---
title: "s085 — anchor q-accuracy go/no-go gate on fast tumblers"
type: experiment
sources:
  - experiments/s085_anchor_accuracy_gate.py
  - results/s085/summary.json
related:
  - experiments/s084_omega_refine_viability.md
  - experiments/s086_omega_refine_rescue.md
  - project_anchor_selection_coarse_then_dense.md
created: 2026-05-21
updated: 2026-05-21
confidence: high (blind anchor delivery; controlled-perturbation polish ladder, 3 fast + 1 slow seed, 3 axes/level)
---

# TL;DR
**Anchor delivery is NOT the blocker; the blind ω-seed quality feeding the polish is.** The sharp-|C_t| anchoring stage delivers the initial orientation `q` to **1.65–3.30° @400k Sobol** on the three fast seeds (119→3.30°, tightening to 1.56° @800k; 108→2.20°; 100→1.65°), comfortably near the s084 1–3° budget (source: `results/s085/summary.json`, `headline_anchor_q_geo_deg` + `density_scan`). With **truth ω (L0)** the joint LM polish is rock-solid — every seed × {2,3,5}° perturbation → 3/3 Band A∪B, hi-fi Band A. But with a **realistic blind ω (L1: dir off 1.5°, |ω| off 0.5%)** the fast seed **119 fails** (1/3 Band A∪B, median ρ≈25 Band D): the small ω error amplifies into a 12–20° q0 error through 738° of back-propagation, beyond the polish grab radius. Seeds 108/100/116 still pass L1.

# What
s084 moved the binding constraint to "anchor q-accuracy" (fast 119 needs q within ~1–3° even with perfect |ω|), but that was a *static* ρ-floor with q held fixed. The v3 pipeline polishes q *jointly*. This gate asks the handoff's literal question — can sharp-|C_t| anchoring deliver ~1–3° q on a fast seed? — and whether the joint polish bridges any residual gap, separated into a blind Layer A and a controlled-perturbation Layer B.

# How
Entrypoint `experiments/s085_anchor_accuracy_gate.py`, Pool(24). Seeds 119/108/100 (fast LAM, |ω| 1.35–1.48 dps) + 116 (slow control, 0.13 dps).
- **Layer A (blind anchor delivery):** coarse 50k Haar pool over all epochs → `T_A = argmin |C_t|` (sharpest survival set); dense Haar pool at `T_A`, density scan {100k,200k,400k,800k}, measure nearest *survivor* to truth-q(T_A). Truth is the comparison target only — never inserted into the pool.
- **Layer B (controlled-perturbation grab radius, s005/s064-gate2 method):** perturb truth-q(T_A) by {2,3,5}° over 3 random axes; seed ω at two realism levels — L0 (truth ω, isolates pure q grab radius) and L1 (dir +1.5°, |ω| +0.5%, realistic blind ω). back_propagate → `lm_polish_jacobi` (s064) → band from ρ_surr (s081: 145/145 surrogate↔hi-fi); hi-fi spot-check the best converged per seed.

# Result

**Layer A — blind anchor delivery (best surviving q_geo to truth, deg):**

| Seed | \|ω\| dps | T_A | rotation to anchor | @100k | @400k | @800k |
|---|---:|---:|---:|---:|---:|---:|
| 119 fast | 1.482 | 69 | 738° | 3.30 | 3.30 | **1.56** |
| 108 fast | 1.367 | 123 | 1213° | 8.54 | 2.20 | 2.20 |
| 100 fast | 1.348 | 198 | 1925° | 1.65 | 1.65 | 1.65 |
| 116 slow | 0.134 | 7 | 7° | 3.58 | 1.47 | 1.47 |

**Layer B — polish convergence (Band A∪B over 3 axes; median ρ_surr):**

| Seed | L0 (truth ω) | L1 (blind ω) |
|---|---|---|
| 119 fast | 3/3 (ρ 0.35, A) all q_pert | **1/3 (ρ 25, D)** |
| 108 fast | 3/3 (ρ 0.32, A) | 3/3 (ρ 0.32, A) |
| 100 fast | 3/3 (ρ 0.22, A) | 3/3 (ρ 0.22, A; conv 1–2/3) |
| 116 slow | — | 3/3 (ρ 0.20, A) |

hi-fi spot-checks all land Band A (119 L0 ρ_hifi 0.072 q0_err 0.10°; 108 L1 0.114; 100 L1 0.046; 116 L1 0.041).

# Why this matters
- **Anchoring works.** The sharp-|C_t| coarse-find then dense-resample stage (project_anchor_selection_coarse_then_dense) delivers fast-seed q to within the s084 1–3° budget, especially at 800k. This was the PROGRESS #1 open question — answered yes.
- **The bottleneck moved upstream, to the blind ω estimate.** With truth ω the polish never fails; with a realistic 1.5°/0.5% ω seed it fails on the fastest seed. The failure is *amplification*: back-propagating a small ω error across hundreds of degrees of accumulated rotation magnifies it into a large q0 error. This sets up s086 (which component of the ω error drives it?) and s087 (how tight must ω be?).
- It is NOT monotone in rotation angle: seed 100 (1925° rotation) passes L1, seed 119 (738°) fails — the per-seed basin shape matters, not just the angle.

# Numbers
- Anchor q_geo @400k: 3.30/2.20/1.65/1.47° (119/108/100/116); 119 @800k = 1.56° (source: `summary.json` per-seed `density_scan` + `headline_anchor_q_geo_deg`).
- L1 seed 119: 1/3 Band A∪B, q0_seed_err median 17.1°, ρ_surr median 25.2 (source: `summary.json` seed 119 `layer_b_summary`, om_level `L1_realistic`).
- Compute wall: 2066 s (34.4 min), Pool(24); per-seed hi-fi spot-check ~40–55 s.

# Artefacts
- `experiments/s085_anchor_accuracy_gate.{py,md}`, `scratch/s085_convention_check.py`
- `results/s085/{summary.json, seed{119,108,100,116}_anchor_gate.png, seed*_data.npz}`

# Out of scope
- Layer B is a controlled perturbation about truth, not a blind end-to-end run; Layer A's anchor delivery IS blind but the two are not yet chained.
- Which component of the L1 ω error (direction vs magnitude) drives the 119 amplification — see s086.
- Only fast LAM + one slow control; SAM fast tumblers untested.

# Cross-references
- `experiments/s084_omega_refine_viability.md` — set up the anchor-q-accuracy question this gate answers.
- `experiments/s086_omega_refine_rescue.md` — decomposes the L1 failure (direction-driven).
- `project_anchor_selection_coarse_then_dense.md` — the Layer A method.
