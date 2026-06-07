---
title: "s088 — two-point attitude BVP shoot: validates the primitive, maps multiplicity, and beats finite-diff ω on fast tumblers"
type: experiment
sources:
  - notebooks/inversion/survey/lib/shoot.py
  - notebooks/inversion/survey/experiments/s088_gateA1_truth_pair.py
  - notebooks/inversion/survey/experiments/s088_gateA2_multiplicity.py
  - notebooks/inversion/survey/experiments/s088_phaseB_conditioning.py
  - notebooks/inversion/survey/experiments/s088_phaseB_sigma.py
  - notebooks/inversion/survey/results/s088/{gateA1,gateA2,phaseB,phaseB_sigma}.json
  - notebooks/inversion/survey/results/s088/phaseB_conditioning.png
related:
  - s060_multi_anchor_design — designed "multi-anchor + ω-from-shoot" but never built it (pre-fix)
  - s063d_polhode_identity_filter — "killed" finite-diff ω; tested the finite-diff VALUE, not a BVP solve
  - s057c_truth_pair_omega / s057d_dt_sweep — finite-diff ω noise floor + Δt sweep (the baseline beaten here)
  - s087_omega_dir_threshold — fast-seed truth recovery needs ω-dir ≲ 1.25°
  - s085_anchor_accuracy_gate — anchors deliver q to 1.47–3.30°
  - project_jacobi_propagation_priority — the closed-form propagator this is built on
created: 2026-05-22
updated: 2026-05-22
confidence: high (truth-pair gates exact; conditioning measured on 2 regime-extreme seeds, N=200–300 noise)
---

## TL;DR

Built and validated a two-point attitude boundary-value solver (`lib/shoot.py`):
given two body-frame orientations q_a (t=0) and q_b (t=Δt) and the inertia,
LM-solve for the body-frame ω_a whose torque-free trajectory connects them. This
is the post-fix realisation of s060's never-built "ω-from-shoot," and it is **not**
the dead finite-diff of s063d — it removes the constant-ω/polhode-drift error by
solving the real dynamics.

Three findings on seeds 119 (fast, |ω|=1.48 dps) + 116 (slow, |ω|=0.134 dps):

1. **Primitive validated (Gate A1).** Propagation round-trip vs cached truth =
   1.7e-6°; residual at true ω = 0 exactly; the solve always drives the geodesic
   residual to 0. A *single* finite-diff-init shoot recovers truth only up to
   |ω|·Δt ≈ 116–160°, then converges to a lower-|ω| "short-way" alias.
2. **Multiplicity is real but tamed by the |ω| prior (Gate A2).** Unconstrained,
   the connecting-ω family explodes (9→399 roots as Δt grows; 53 even at 14° for
   the slow seed — high-winding roots). With a realistic ±30% |ω| prior the count
   collapses: **slow seed → truth UNIQUE at every Δt tested (to 241°); fast seed →
   truth unique to |ω|·Δt ≈ 160°**, then a small same-direction winding ladder
   (4 at 321°, 10–11 at 642–1284°). Truth is recovered by multi-start in every cell.
3. **BVP-shoot decisively beats finite-diff on fast tumblers; the s087 gate is met
   at good-anchor accuracy (Phase B).** Under q-cloud noise σ_q=2.5°, BVP ω-dir
   error stays ~1.5–2.4° on the fast seed where finite-diff blows to 80–127°
   (polhode-drift + short-way alias). σ-sweep at each seed's sweet-spot Δt:
   **BVP clears the s087 1.25° gate (median) for σ_q ≲ 2.0°** on both seeds, with
   ω-dir error scaling ≈ linearly in σ_q.

The cross-cloud bridging idea is a **viable grid-free route to ω-direction**,
on the edge of the s087 cliff at realistic noise — sharpened by a tighter anchor
(s085 @800k) or by a third anchor (over-determination, not yet tested).

## What

Tests the user-proposed architecture (this session): cross two highly-constrained
anchor clouds, and for a candidate (q_a, q_b) pair solve for the connecting ω with
the post-fix closed-form propagator, instead of finite-differencing it. The BVP is
3 unknowns (ω_a) ↔ 3 constraints (q_b ∈ SO(3)) → generically isolated roots. Open
questions going in: (i) is the solve well-posed and convention-correct, (ii) how
many roots connect a pair and is truth recoverable, (iii) is a large-Δt solve
better conditioned against anchor noise than a small-Δt finite-diff (s057c/d).

## How

`lib/shoot.py`: residual = rotation-vector of q_b⁻¹⊗propagate_jacobi_path2(q_a,ω,[0,Δt])[-1];
`scipy.least_squares(method='lm')` (damped, robust to the non-convex many-root
residual, returns residual floor + Jacobian conditioning). `finite_diff_omega`
init from q_b⊗q_a⁻¹ (validated body-frame, sign-checked against truth in A1).

- **Gate A1** (`s088_gateA1_truth_pair.py`): exact truth pairs, anchor epoch 100,
  Δt ∈ {3..250} ep. Round-trip, residual@true, finite-diff init error, shoot recovery.
- **Gate A2** (`s088_gateA2_multiplicity.py`): multi-start over Fibonacci(200) ×
  geomspace(0.04–2.2 dps, 11) inits (2200/pair), dedup connected roots, count
  within |ω| prior bands ±{10,30,50}%. Pool(24).
- **Phase B** (`s088_phaseB_conditioning.py`): perturb truth (q_a,q_b) by σ_q=2.5°
  (N=200), shoot init-at-truth (isolates truth-branch conditioning) vs finite-diff
  on identical noisy endpoints; ω-dir + |ω| error vs Δt. Plot saved.
- **Phase B σ-sweep** (`s088_phaseB_sigma.py`): σ_q ∈ {1,1.5,2,2.5,3.3}° at each
  seed's sweet-spot Δt (119→60 ep, 116→120 ep), N=300.

Seeds 119 (fast LAM, the s085/s087 binding case) + 116 (slow, worst-case
conditioning). Pure cached-truth + closed-form propagation; no surrogate.

## Result

**Gate A1 (PASS).** roundtrip max geo err 1.7e-6°; residual@true ω = 0.0 at all Δt;
shoot geodesic residual = 0 at all Δt. Single finite-diff-init recovery: truth to
|ω|·Δt ≤ 160° (seed 119) / ≤ 116° (seed 116); aliases to a low-|ω| short-way root
beyond (e.g. seed 119 Δt=30: dir 127°, |ω| −82%). (`results/s088/gateA1.json`)

**Gate A2 — distinct roots, and roots within ±30% |ω| prior:**

| seed | Δt (ep) | \|ω\|·Δt | distinct | within ±30% | truth |
|---|---:|---:|---:|---:|---|
| 119 | 15 | 160° | 9 | **1 (unique)** | Y |
| 119 | 30 | 321° | 94 | 4 | Y |
| 119 | 60 | 642° | 128 | 10 | Y |
| 119 | 120 | 1284° | 99 | 11 | Y |
| 119 | 250 | 2674° | 399 | 98 | Y |
| 116 | 15 | 14° | 53 | **1 (unique)** | Y |
| 116 | 30 | 29° | 93 | **1 (unique)** | Y |
| 116 | 60 | 58° | 177 | **1 (unique)** | Y |
| 116 | 120 | 116° | 250 | **1 (unique)** | Y |
| 116 | 250 | 241° | 393 | **1 (unique)** | Y |

Mechanism: winding spacing in |ω| ≈ (2π/Δt)/(rotation per |ω|); for slow seeds the
±30% absolute band is narrower than the spacing → unique; for fast seeds the band
admits a same-direction winding ladder that grows with |ω|·Δt. (`results/s088/gateA2.json`)

**Phase B (σ_q=2.5°, init-at-truth, N=200) — BVP ω-dir error (median/p90), deg:**

| seed | Δt | \|ω\|Δt | BVP dir | finite-diff dir | BVP \|ω\|% | FD \|ω\|% |
|---|---:|---:|---:|---:|---:|---:|
| 119 | 7 | 75° | 2.34/4.00 | 5.40/7.80 | 1.75 | 1.70 |
| 119 | 15 | 160° | 2.42/6.06 | 11.54/12.80 | 1.05 | 1.13 |
| 119 | 30 | 321° | 2.08/4.44 | **126.9** | 0.41 | 82.0 |
| 119 | 60 | 642° | **1.45/3.59** | 114.8 | 0.25 | 81.3 |
| 119 | 120 | 1284° | 28.8/44.4 | 78.1 | 1.08 | 86.2 |
| 116 | 7 | 7° | 19.0/36.1 | 19.1/36.2 | 21.4 | 21.4 |
| 116 | 60 | 58° | 2.62/4.26 | 3.05/5.02 | 2.97 | 2.96 |
| 116 | 120 | 116° | **1.32/2.13** | 3.25/4.73 | 1.15 | 1.10 |

Fast seed: BVP beats finite-diff at every Δt, by 1–2 orders past 320° (finite-diff
hits the short-way alias). Best at Δt=60 (sweet spot); degrades at Δt=120 as
winding-aliases crowd within the 2.5° noise ball — the conditioning↔multiplicity
crossover, empirically. Slow seed: BVP ≈ finite-diff (near-constant-ω regime,
T_pol≫LC), both monotone-improving; BVP edges ahead at large Δt and has no aliasing
ceiling. (`results/s088/phaseB.json`, `phaseB_conditioning.png`)

**Phase B σ-sweep — BVP ω-dir median (deg) at sweet-spot Δt:**

| σ_q | seed 119 (Δt=60) | seed 116 (Δt=120) |
|---:|---:|---:|
| 1.0° | 0.62 | 0.55 |
| 1.5° | 0.76 | 0.78 |
| 2.0° | 1.20 | 1.06 |
| 2.5° | 1.48 | 1.36 |
| 3.3° | 2.08 | 1.71 |

**Gate (1.25° median) clears for σ_q ≲ 2.0°** on both seeds; ω-dir error ≈ linear
in σ_q (~0.5–0.6° per degree of anchor noise). (`results/s088/phaseB_sigma.json`)

## Why this matters

- **The s063d "finite-diff ω is dead" verdict does not apply to the BVP solve.**
  s063d filtered cached finite-diff *values* (87× |L|² spread); the BVP *solves* the
  real dynamics and removes the polhode-drift term s057d named. On the fast seed at
  Δt=30–60, finite-diff is 80–127° off while BVP is ~1.5–2°. s060's pair-shoot was
  the right idea, just never built post-fix.
- **The multiplicity you'd worry about is governed by |ω|·Δt and is tamed by the
  |ω| prior** — unique for slow tumblers, a small same-direction ladder for fast
  ones at large rotation. The shoot's job is to return the small in-prior family;
  over-determination (3rd anchor / intermediate brightness) picks among them.
- **Cross-cloud bridging is a viable grid-free ω-direction source**, sitting right
  at the s087 1.25° cliff at σ_q=2.5° and clearing it at σ_q≲2.0° — i.e. it works at
  the better end of s085 anchor accuracy (1.56° @800k for seed 119), no ω-direction
  grid densification needed.
- **L-conservation is automatic here** (torque-free solve), not a filter — confirms
  the session's earlier conclusion; the discriminator is connectability +
  over-determination, not angular momentum.

## Numbers

- Inertia: I_principal_ascending = (7749.01, 37985.16, 38305.71) kg·m² (`results/s062/...json`).
- Anchor epoch 100; seed 119 T_pol=1473s (204 ep), seed 116 T_pol=3388s (470 ep).
- Compute wall: Gate A1 ~seconds; Gate A2 ~tens of s Pool(24); Phase B + σ-sweep ~tens of s Pool(24) each.

## Out of scope (next moves)

- **Phase C — connectability as a filter:** for NON-truth (q_a × q_b) cross-cloud
  pairs, what fraction admit an in-prior connecting ω? The over-determination claim.
- **Third anchor / over-determination** to beat the noise floor below 1.25° at
  σ_q≥2.5° (average down endpoint noise; resolve the fast-seed winding ladder).
- **Real (blind) anchor clouds** (s085 machinery) instead of perturbed truth pairs;
  finite-diff-init multi-start under the |ω| prior end-to-end.
- **SAM seed + separatrix** behaviour (elliprj sensitivity near k²→1; only seed 7).
- **Cohort** across the 10 stratified seeds.

## Artefacts

- `lib/shoot.py` — the BVP primitive (`shoot`, `finite_diff_omega`, `polhode_period`, helpers).
- `experiments/s088_gateA1_truth_pair.py`, `s088_gateA2_multiplicity.py`,
  `s088_phaseB_conditioning.py`, `s088_phaseB_sigma.py`.
- `results/s088/{gateA1,gateA2,phaseB,phaseB_sigma}.json`,
  **`results/s088/phaseB_conditioning.png`**.

## Cross-references

- `s060_multi_anchor_design.md` — the design this implements (open Q#3 = the
  multiplicity now measured in A2).
- `s063d_polhode_identity_filter.md` — finite-diff-value death; this shows the
  *solve* sidesteps it.
- `s057c/s057d` — finite-diff ω baseline beaten in Phase B.
- `s087_omega_dir_threshold.md` — the 1.25° gate; `s085_anchor_accuracy_gate.md` — anchor q-accuracy.
