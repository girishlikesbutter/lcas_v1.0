---
title: "s089 — third-anchor over-determination: a disambiguator, not a noise-reducer"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s089_third_anchor_overdetermination.py
  - notebooks/inversion/survey/lib/shoot.py
  - notebooks/inversion/survey/results/s089/{conditioning,ladder}.json
  - notebooks/inversion/survey/results/s089/conditioning.png
related:
  - experiments/s088_bvp_shoot_conditioning.md
  - experiments/s087_omega_dir_threshold.md
  - experiments/s085_anchor_accuracy_gate.md
created: 2026-05-22
updated: 2026-05-22
confidence: high (self-checks exact; ladder collapse is a large decisive effect on 3 seeds / 2 regimes; the shared-base-anchor noise MECHANISM is interpretation, not ablated)
---

# TL;DR

Added a third anchor to the s088 two-point attitude BVP, turning it into an
over-determined joint solve (`lib/shoot.py::shoot_multianchor`, 3 unknowns ω_a
vs 3K geodesic constraints). Two predictions, split result:
**Q2 (multiplicity) confirmed decisively** — the third anchor collapses the
fast-seed winding ladder from **26 in-prior roots → 1 unique truth** (seed 119),
**4 → 1** (SAM 103), and rescues an otherwise-aliased long baseline from
**39.6° → 1.38°** (SAM 103 @ σ_q=2.5°). **Q1 (noise floor) refuted on the binding
seed** — on fast LAM 119 the joint 3-pt tracks the 2-pt almost exactly
(1.49° vs 1.57° @ σ_q=2.5°), still above the s087 1.25° gate; both clear it only
at σ_q≲2.0°, the *same* threshold as s088's 2-point (source:
`results/s089/conditioning.json`). **Architectural consequence:** the third
anchor is a *disambiguator* (unique truth + unlocks long baselines), **not** a
noise-reducer; the fast-tumbler accuracy gate is still set by base-anchor quality,
not by over-determination.

# What

s088 left two open issues on fast tumblers: (Q1) ω-dir 1.45° @ σ_q=2.5°, just
above the s087 1.25° cliff, clearing only at σ_q≲2.0°; and (Q2) a winding ladder
(11 in-prior roots at the long baseline) plus a conditioning↔multiplicity
crossover where the long, well-conditioned baseline blows up to 28.8° even
init-at-truth. The session hypothesis: a third intermediate anchor (a) averages
endpoint noise down (Q1) and (b) pins the winding so the long baseline no longer
aliases (Q2). This experiment tests both, on both tumbling regimes.

# How

`experiments/s089_third_anchor_overdetermination.py`, two parts, Pool(24),
pure closed-form Jacobi propagation + cached truth (no surrogate). Anchors built
by re-propagating truth with `propagate_jacobi_path2` (matches s088; equal to
cached DOP853 to 1.7e-6° per s088 Gate A1). Perturb each anchor by σ_q (random
axis, fixed angle).

- **Part 1 (conditioning, init-at-truth):** σ_q ∈ {1.5,2.0,2.5,3.3}°, N=300, three
  estimators on identical noisy triples — short 2-pt (sweet spot), long 2-pt
  (aliases), joint 3-pt (`shoot_multianchor`). Reports ω-dir median/p90 + |ω|%.
- **Part 2 (ladder, blind multi-start):** 2200 inits (Fibonacci-200 dirs × 11
  geomspace |ω|), dedup connected roots, count distinct + in-prior(±30% |ω|) +
  truth-uniqueness — long 2-pt vs joint 3-pt.

Anchor placement = fractions of each seed's polhode period T_pol (B at 0.30·T_pol,
C at {0.45,0.60}·T_pol), spanning the LC if T_pol>remaining-LC. This reproduces
s088's measured 119 geometry (Δt=60≈0.30·T_pol, Δt=120≈0.60·T_pol). Seeds: 119
(LAM-fast, binding), 116 (LAM-slow, continuity), 103 (SAM, regime control). A
per-seed noiseless self-check (init +5%, must snap back) validates the stacked
residual. q0_err and ρ-band are NOT measured — this is the upstream spin-solve
conditioning stage; downstream recovery is out of scope.

# Result

Self-check PASS on all 3 seeds: dir_err = 0.0°, geo_max = 0.0° (init +5%).

**Q1 — conditioning, ω-dir median (deg), short / long / JOINT, at σ_q=2.5°:**

| seed | regime | geometry (ab,ac ep) | short 2-pt | long 2-pt | JOINT 3-pt | gate 1.25° |
|---|---|---|---:|---:|---:|---|
| 119 | LAM-fast | (61, 92) | 1.57 | 1.60 | **1.49** | fail |
| 119 | LAM-fast | (61, 123) | 1.57 | 2.94 | 1.67 | fail |
| 116 | LAM-slow | (120, 239) | 1.35 | 0.69 | 0.71 | PASS |
| 103 | SAM | (120, 180) | 1.47 | 2.10 | 1.37 | indic |
| 103 | SAM | (120, 239) | 1.47 | **39.64** | **1.38** | indic |

Seed 119: joint ≈ short (no noise-floor gain); both clear 1.25° only at σ_q≲2.0°
(σ=2.0 → joint 1.07–1.12°; σ=2.5 → 1.49–1.67°) — same as the s088 2-point.
Seed 116/103: the long baseline genuinely helps accuracy (116: 1.35→0.71; 103:
1.47→1.38), and on SAM 103 the joint *rescues* a 39.6°-aliased long baseline to
1.38° by pinning the winding. |ω| is incidental everywhere (joint ≤0.33%).

**Q2 — multiplicity (blind multi-start, ±30% |ω| prior), long 2-pt vs JOINT 3-pt:**

| seed | regime | long: distinct / in-prior / unique | JOINT: distinct / in-prior / unique |
|---|---|---|---|
| 119 | LAM-fast | 167 / **26** / ✗ | 1 / **1** / ✓ |
| 116 | LAM-slow | 383 / 1 / ✓ | 1 / 1 / ✓ |
| 103 | SAM | 175 / **4** / ✗ | 1 / **1** / ✓ |

The third anchor collapses every ladder to a unique truth, on both regimes.

# Why this matters

- **The s088 winding-ladder open question is CLOSED (yes).** A single intermediate
  anchor turns a family of in-prior connecting-ω solutions into the unique truth
  and unlocks long, well-conditioned baselines that alias on their own (SAM 103:
  39.6°→1.38°). This is the uniqueness mechanism the cross-cloud architecture
  needed — you can act on one candidate, not 26.
- **The fast-tumbler accuracy gate is NOT relaxed by over-determination.** The
  predicted Q1 win did not land: joint ≈ short on seed 119. The fast-seed noise
  floor is still set by anchor quality (clears 1.25° only at σ_q≲2.0°), so the
  lever there remains tighter base anchors (s085 @800k → 1.56°) or a downstream
  LM polish that absorbs the residual — not more anchors.
- **Interpretation [hypothesis, not ablated]:** all anchors share the base
  orientation q_a, whose noise is common to every constraint and cannot average
  out. Over-determination reduces far-endpoint noise but not the shared-base term,
  which dominates the fast-seed ω-dir error. A test: let q_a float as 3 more
  unknowns and see if the fast-seed floor drops.

# Numbers

- Self-check (init +5%): dir_err 0.0° / |ω|_err ≤9.3e-8% / geo_max 0.0° all seeds (run stdout).
- Seed 119 (|ω|=1.4819 dps, T_pol=1473s): joint ω-dir @σ_q=2.5° = 1.49° (ac=92) / 1.67° (ac=123); ladder 26→1 (source: `results/s089/ladder.json`).
- Seed 116 (|ω|=0.1336 dps, T_pol=3388s>LC): joint ω-dir @σ_q=2.5° = 0.71°; ladder 1→1.
- Seed 103 (SAM, |ω|=0.5300 dps, T_pol=5051s>LC): joint ω-dir @σ_q=2.5° = 1.38°; long 2-pt 39.64° (ac=239); ladder 4→1 (source: `results/s089/conditioning.json`, `ladder.json`).
- Compute wall: 33 s total, Pool(24).

# Artefacts

- `lib/shoot.py::shoot_multianchor` — the over-determined joint solver.
- `experiments/s089_third_anchor_overdetermination.py`.
- `results/s089/conditioning.json`, `results/s089/ladder.json`, **`results/s089/conditioning.png`**.

# Out of scope

- **Shared-base ablation** — let q_a float; test whether the fast-seed noise floor drops (the Q1 mechanism check).
- **Connectability filter (Phase C)** — fraction of NON-truth cross-cloud pairs admitting an in-prior connecting ω; the cheap prune for cloud combinatorics, still untested.
- **End-to-end blind** — real (s085) anchor clouds, finite-diff-init multi-start under the |ω| prior; downstream q0 polish + hi-fi ρ-band.
- **SAM threshold** — the 1.25° gate is LAM-119-derived; SAM 103 needs its own s087-style cliff before a hard pass/fail.
- **Cohort** — 10 stratified seeds, separatrix (elliprj near k²→1).

# Cross-references

- `s088_bvp_shoot_conditioning.md` — the 2-point baseline + the open Q1/Q2 this answers.
- `s087_omega_dir_threshold.md` — the 1.25° fast-seed gate.
- `s085_anchor_accuracy_gate.md` — the σ_q≈1.56° @800k anchor accuracy that now sets the binding floor.
