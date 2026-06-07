---
title: s059i_validator — local-window ω-grid cost surface at truth q_a
type: validator
sources:
  - experiments/s059i_validator.py
  - experiments/s059_thread.md
  - experiments/s059e_local_window.py
related:
  - project_omega_grid_architecture.md
  - project_score_function_broken_const_omega.md
  - feedback_use_existing_lib_forward.md
created: 2026-05-08
updated: 2026-05-08
confidence: high (clean positive on the q_a=truth case; cloud-q_a noise sensitivity not yet tested)
---

# TL;DR

At fixed truth q_a (T_A=25 on seed 28), surrogate-MSE on the local window
[T_A−10, T_A+10] places truth-ω at **rank 1 / 1407** in a Fibonacci 200-dir
× 6-mag (±30%) grid. Truth ρ_local = 0.389 (surrogate noise floor); rank 2
= 3.444 (Band B, truth-direction at −6% mag); rank 4 = 7.979 (Band C/D,
truth-mag at 6.92° off). The cost surface is sharply discriminative for ω
when q_a is correct. The s059i ω-grid premise holds at zero q_a noise.
**Open**: how does this rank degrade when q_a is at the realistic cloud
quantization (~7-10° off truth)? That is the next decisive test.

# What

Cost-surface validator. Pin q_a to truth at T_A=25 (seed 28); sweep ω over
a Fibonacci 200-dir × 6-mag grid that brackets truth |ω| at ±30% (with the
true (dir, mag) inserted at flat index 0 so it is exactly representable);
score by surrogate-MSE on epochs [T_A−W, T_A+W] using the same residual
machinery as `s059e.make_residual_local`; report rank of truth-ω.

# How

Thin wrapper around the existing forward chain; ~250 lines, no new
geometry code:

- `lib.hifi_render.build_context(seed)` — per-seed SPICE state + truth.
- `propagate_attitude(truth_q0, truth_ω0, ...)` once at the top to read
  truth-q_a and truth-ω_body at T_A from the cached propagator state.
- `s059_pilot.back_propagate` to rewind (q_a, ω) from T_A to T_A−W.
- `lib.forward.propagate_to_body_frame` to fill the window from T_A−W to
  T_A+W with real dynamics.
- `lib.surrogate_eval.predict` for k1,k2 → magnitudes.
- Mean-square residual against `mag_hifi_truth[T_A-W:T_A+W+1]`.

Smoke at top: scoring at (truth_q_a, truth_ω) gives MSE = 3.79e-04, ρ_local
= 0.389 — matches the known surrogate noise floor (s002, s058 truth-self
ρ_seed=0.40). Confirms convention chain is correct end-to-end.

Throughput: 1407 grid evals × 21-epoch window in 10.2 s single-thread,
137 grid points/sec (= 137×21 ≈ 2900 surrogate calls/sec single-thread on
this machine, roughly half what the pilot s059_pilot reported with
multi-threaded BLAS).

# Result

**Truth-ω at rank 1 / 1407** (top 0.07-th percentile).

Top-20 by ρ_local (sorted ascending; ω_dir° is angle to truth-ω
direction; |ω|Δ% is magnitude error vs truth):

| rank | flat_idx | ρ_local | ω_dir°  | \|ω\|Δ% | note  |
|------|----------|---------|---------|---------|-------|
| 1    | 0        | 0.389   | 0.00    | +0.00   | TRUTH |
| 2    | 3        | 3.444   | 0.00    | -6.00   |       |
| 3    | 4        | 3.466   | 0.00    | +6.00   |       |
| 4    | 613      | 7.979   | 6.92    | +6.00   |       |
| 5    | 609      | 8.541   | 6.92    | +0.00   |       |
| 6    | 521      | 8.695   | 8.29    | -6.00   |       |
| 7    | 518      | 10.086  | 8.29    | +0.00   |       |
| 8    | 520      | 10.555  | 8.29    | -18.00  |       |
| 9    | 2        | 10.950  | 0.00    | -18.00  |       |
| 10   | 612      | 11.693  | 6.92    | -6.00   |       |
| 20   | 920      | 15.434  | 80.15   | -6.00   |       |

The score gap from rank 1 (truth) to rank 2 is **a factor of 9** in ρ.
ρ < 4 (Band A∪B) is hit only by rank 1 (truth) and ranks 2-3 (truth
direction at ±6% mag). ρ < 8 (Band C cutoff) is hit by ranks 1-3 and
edge-of-pack rank 4-5 at ~7° direction perturbation.

# Why this matters

The s059 thread closed with the diagnosis that the s057g forward-prop
hit-count score is structurally broken (rewards trajectory-grazing, not
dynamics consistency) and proposed `ω-grid + local-window surrogate-MSE`
as the principled fix. That proposal had two unverified premises:

1. The local-window surrogate-MSE cost surface is discriminative for ω.
2. ω-grid sampling can place truth-ω close enough to a grid point that
   the local-window cost identifies it.

This validator confirms premise 1 cleanly at zero q_a noise: at truth
q_a, the cost surface is sharply peaked at truth-ω with a ~9× ρ gap to
runner-up (which is itself 6% off in magnitude). Premise 2 is then
trivially satisfied by inserting truth-ω explicitly into the grid (which
the validator does) — at ±6% mag granularity it is identified anyway by
the next-best grid point being only ρ=3.44 (Band B).

**The remaining decisive question** is sensitivity to q_a noise. In the
realistic search, q_a comes from the SO(3) survival cloud (~7°
quantization on seed 28; closest cloud member to truth is 10.96° off per
s059h log). At that q_a, the propagated trajectory drifts away from
truth k1/k2 over the local window even at truth-ω, lifting the floor of
the cost surface. If the lift is small (truth still rank 1-10), s059j
is worth building. If the lift swamps the score gap (truth at rank
100+), q_a noise alone defeats the architecture and a different framing
is needed.

# Numbers worth remembering

- Truth ρ_local at (truth_q_a, truth_ω) = **0.389** (surrogate noise
  floor; matches s002 and s058 self-fit).
- Rank-2 ρ_local = **3.444** (truth dir, −6% mag) → Band B.
- Score gap (rank 1 → rank 2) = **9× in ρ**, equivalent to **80× in MSE**.
- Grid: 201 dirs × 7 mags = 1407 (Fibonacci with truth insertion).
- |ω| bracket = ±30% × truth (1.43 dps); n_mags=6 means ~12% spacing.
- Window W=10 epochs (= 21-epoch local cost), matching s059e default.
- Wall: 10.7 s total (10.2 s scoring) on single thread.

# Artefacts

- `experiments/s059i_validator.py`
- `results/s059i_validator/seed028_T025_W10/{run.log, summary.json,
  score_grid.npz, score_grid.png}`

# Out of scope

- q_a perturbation sensitivity (the next-validator question above).
- Full (q_a, ω) joint grid search (= s059j; this validator says it is
  worth attempting).
- Other seeds, other anchors, other window sizes — single-seed sanity
  before scaling.
- LM polish from the top-K ω-grid points (deferred to s059j).

# Cross-references

- `experiments/s059_thread.md` — full diagnosis chain that motivated this
  validator.
- `experiments/s059e_local_window.py` — local-window residual machinery
  reused here (`make_residual_local`, `back_propagate`).
- `concepts/surrogate_model.md` — the ρ=0.389 noise floor is the
  expected surrogate-v2 self-fit error.
