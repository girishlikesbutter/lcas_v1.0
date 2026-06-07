---
title: "s114 — Anchor-direct omega-multistart recovers truth BLIND & hi-fi-confirmed on fast-aliased seed 119 (Band A), but the cost lever is unsolved"
type: experiment
sources:
  - experiments/s114_anchor_direct_omega_multistart.py
  - experiments/s114_hifi.py
  - experiments/s114_plot.py
  - experiments/s114_plot_errors.py
  - results/s114/seed119/summary.json
  - results/s114/seed119/hifi.json
  - results/s114/cost_probe.json
related:
  - experiments/s106_hybrid_loss_polish.md
  - experiments/s107_discrimination_test.md
  - experiments/s108_cross_cloud_multistart_cost_wall.md
  - experiments/s100_5step_coverage_proto.md
created: 2026-06-05
updated: 2026-06-05
confidence: high on the blind hi-fi-confirmed Band A (N=1 seed, 6/6 hi-fi A+B, 0 phantoms); high on the two-stage-prune refutation (truth-near rep ranked 1118/2000, deterministic); medium on generalisation (seed 19 NOT run)
---

# TL;DR
Drop the IA Cloud Cross/q_b entirely (it only seeded omega, badly — 104° off, s100/s107 — and its soft-B term is inert, s106) and instead multistart an omega-dir × |omega| GRID directly at each cloud q_a rep, rank by coarse-K full-LC surrogate RMSE, polish the top-K over the s106 A→C window. On fast-aliased seed 119 this **recovers truth blind**: the winner (rep 192, q_a 2.27° off truth) polishes to **surr-ρ 1.526 / hi-fi-ρ 1.534 Band A, ω-dir 1.64°, |ω| 1.5022 deg/s** — the **first fully-blind, hi-fi-confirmed Band A on this seed** (s106 was oracle-seeded + surrogate-only; s107's blind tries were all hi-fi Band D; s108's blind path was 100× over budget). 26 surrogate Band A∪B attractors, **0 phantoms**, and hi-fi confirms 6/6 A∪B with surr-ρ matching hifi-ρ to ~0.05 (the surrogate is honest off-truth here, unlike s107). **omega is uniquely determined** (all 26 within ω-dir ≤2.67°, Δ|ω| ≤0.005 deg/s of truth); the **attitude is multi-valued** into two families — near-truth (q0 2.3–21°) and near the body-X-twin *attitude* (q0 3–14° from the twin) — and NONE are exact body-twins (a twin carries the reflected ω R_180x·ω, ~154° from truth ω; every found solution carries ~truth ω). **What failed: cost.** The contract's v1 two-stage rep-prune (coarse-dir filter all reps → top-M → dense-dir) is refuted on its binding prediction — the truth-NEAREST rep (idx 31, 1.02°) ranked 1118/2000 at the coarse Stage A and was dropped — and the run took 31.8 min/seed, over the 15-min budget. So truth was recovered (via the 2.27° surviving rep at ρ 1.53, not the 1.02° rep that would have given ~ρ 0.03), but the cheap rep-discriminator does not work and the basin-catching single-stage is ~97 min/seed.

# What
After s108 (the blind windowed-polish path works but is ~100× over the 15-min budget because the IA Cloud Cross multistart runs over 3.17M q_a×q_b pairs), the align contract reframed the branch: since q_b only ever seeded omega and the soft-B term is inert, **drop the cross** and multistart omega directly at each of the ~2000 cloud q_a reps. That removes the q_a×q_b combinatorics (a ~1585× base-eval reduction). The question: does anchor-direct reach a blind, phantom-free Band A∪B attractor within the 15-min/seed budget on fast seeds 119 (and 19)?

# How
Script `experiments/s114_anchor_direct_omega_multistart.py` (seed 119 uses the cached s100 repA, ep_a=69/ep_c=377; Pool(24), BLAS pinned, fork CoW, v2 surrogate). Pipeline (v1, two-stage rep-prune — see Result for why two-stage):
1. **Stage A** — coarse-dir filter ALL 2000 reps: at each rep, score an n_dir=64 × n_mag=5 omega grid (|omega| ∈ [0.1,1.5] deg/s) by `s100._coarse_rmse` (coarse-K=50 full-LC surrogate RMSE), keep the best root, rank reps, take top-M=200.
2. **Stage B** — dense-dir on the top-200: n_dir=1000 × n_mag=10 grid (nearest grid point 1.11° from truth-ω-dir), keep best root per rep, rank.
3. **Polish** the top-K=150 over the s106 abc window (ep_a → ep_c+60 = 369 ep) with `least_squares(lm)`, band by ρ = √MSE/0.05.
4. **Hi-fi** (`s114_hifi.py`, serial trimesh, gc between renders) on the winner + surr-Band-A∪B set + truth control — DIAGNOSTIC (acceptance is surrogate-v2 per the contract).

Past-error self-check (CLAUDE.md): times[0]==0 gauge asserted on the abc window (`assert _T_SEL[0]==0.0`); truth used only for dir/geo LABELS (oracle_clean=false — cached cloud + labels); BLAS=1 before Pool; killed-at-2× armed via a calibration print.

The v0 single-stage contract was AMENDED to v1 mid-run: the seed-119 smoke calibration measured **~6.83 ms/coarse_rmse**, not the 0.478 ms the s114 cost probe assumed (the probe conflated s108 shoot-ATTEMPTS — most short-circuiting on the connectability filter before any coarse eval — with coarse evals). Single-stage n_dir=1000 is therefore ~97 min/seed; two-stage was the budget-restoring redesign (approved amendment v1).

# Result

## The win — blind, hi-fi-confirmed Band A (source: results/s114/seed119/hifi.json, summary.json)
| solution | q0 err | ω-dir err | Δ|ω| dps | surr-ρ | hi-fi-ρ | band |
|---|---|---|---|---|---|---|
| winner (rep 192) | **2.27°** | **1.64°** | ~0.000 | 1.526 | **1.534** | **A** |
| rep 187 | 12.06° | 1.92° | 0.001 | 2.130 | 2.125 | B |
| truth control | 0 | 0 | 0 | — | 0.000 | A |

- 26 surrogate Band A∪B, 2 Band A, **0 phantoms** across 150 polishes (`summary.json`: n_bandAB 26, n_phantom 0).
- Hi-fi: **6/6 A∪B confirmed, 0 surrogate-phantoms**; surr-ρ ≈ hifi-ρ to ~0.05 (`hifi.json`).
- **omega uniquely determined**: all 26 A∪B have ω-dir ≤ 2.67° and Δ|ω| ≤ 0.005 deg/s vs truth (`s114_plot_errors.py` table).
- **attitude multi-valued**: 10 near-truth-attitude (q0 2.3–21°), 16 near the body-X-twin attitude (q0 3–14° from the twin). NOT body-twins — the twin's ω (R_180x·ω) is ~154° from truth ω, and every solution carries ~truth ω.

## The failures (source: summary.json)
- **Two-stage prune refuted (binding prediction):** truth-nearest rep idx 31 (1.02°) ranked **1118/2000** at Stage A — its best coarse root was a spurious ω 58.6° off truth (coarse 1.43, no better than junk) because the n_dir=64 coarse grid has no ω near truth-dir. Dropped from top-M=200. The cheap rep-discriminator cannot tell the right q_a from the wrong ones without a dense ω-grid.
- **Over budget:** wall_search 1906 s = **31.8 min** (Stage B 24.4 min, 2.6× its projection), `within_budget: false`.
- **Precision degraded, not recovery lost:** the prune dropped the 1.02° rep (would-be ~ρ 0.03, the s106 value), so the best SURVIVING near-truth recovery is rep 192 at ρ 1.53 — still truth, still Band A, just looser.

# Why this matters
The hard-shoot trap is broken BLIND and hi-fi-confirmed on the fast-aliased class — the first time (s106 was oracle+surrogate; s107/s108 blind attempts failed on hi-fi or budget). It also shows the LC pins ω hard but leaves attitude genuinely ambiguous (two basins) — a multi-solution acceptance is the right frame, not single-truth recovery. The unsolved piece is purely COST: rep-discrimination needs a dense per-rep ω-grid (~97 min single-stage), and the cheap two-stage shortcut drops the truth-near rep. The next branch is a cost lever — a rep pre-filter that survives the truth-near rep without a dense ω search.

# Numbers
- coarse_rmse cost 6.83 ms/eval (`summary.json` coarse_ms_per_eval); single-stage n_dir=1000 ≈ 97 min/seed.
- winner rep 192: surr-ρ 1.526, hi-fi-ρ 1.534, ω-dir 1.64°, |ω| 1.5022 deg/s, q0 2.27° (`hifi.json`, `summary.json`).
- truth |ω| 1.4819 deg/s (`data/trajectories/traj_seed119.npz` omega_mag_dps).
- truth-near rep idx 31 stageA rank 1118/2000 (`summary.json` truth_rep_stageA_rank).
- 26 A∪B, 0 phantoms, 6/6 hi-fi-confirmed (`summary.json`, `hifi.json`).

# Artefacts
- `results/s114/seed119/{multistart.npz, polish.npz, summary.json, hifi.json}`
- `results/s114/seed119/{attractors.png, winner_lc.png, joint_errors.png}` (on the plot-stream)
- `results/s114/cost_probe.json` (align-stage cost projection)

# Out of scope
- **Seed 19 NOT run** — with the v1 two-stage method refuted (drops nearest rep, over budget), re-running it on 19 was a foregone failure on cost; generality of the multi-sol win to a 2nd fast seed is untested.
- The cost lever (cheap rep pre-filter / sub-15-min architecture) — next branch.
- Single-seed (N=1); acceptance is surrogate-v2 with hi-fi as a recorded diagnostic.

# Cross-references
- [[s106_hybrid_loss_polish]] (the windowed-photometry polish, oracle-seeded) · [[s107_discrimination_test]] (phantom-free but production-inaccessible) · [[s108_cross_cloud_multistart_cost_wall]] (cross multistart 100× over budget) · [[s100_5step_coverage_proto]] (the cloud + anchors).
- contract_anchor-direct-omega-multistart (v1 amended) · claim_windowed-photometry-breaks-hard-shoot-trap.
