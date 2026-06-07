---
title: s059 thread — cloud-data architecture cohort validation, diagnosis, and reframe
type: experiment-thread
sources:
  - s058_lm_polish_clusters.md (precursor)
  - s059_pilot.py
  - s059b_diagnose_seed28.py
  - s059c_anchor_polish.py
  - s059d_early_anchor.py
  - s059e_local_window.py
  - s059f_score_diagnostic.py
  - s059g_realdyn_score.py
  - s059h_omega_grid.py
related:
  - feedback_oracle_injection_taints_yield.md
  - feedback_use_existing_lib_forward.md
  - project_score_function_broken_const_omega.md
  - project_omega_grid_architecture.md
created: 2026-05-08
updated: 2026-05-08
confidence: high (negative result on architecture-as-was; positive on the diagnostic chain)
---

# TL;DR

The cloud-data forward-prop architecture promoted from s058 as "PROVEN-on-89"
was actually proven only with **truth-cluster oracle injection** in the
polish set. With injection removed, the score function (Σ validator hits with
const-ω propagation) fails to rank truth in any top-K we'd polish at
inference: seed 89 truth at rank 107/325, seed 28 at rank 390/25k. The
score function is structurally broken — it rewards const-ω trajectories
that graze dense validator regions, not dynamics-consistent (q, ω)
candidates. Real dynamics alone doesn't fix it (truth ranks worse) because
the candidate's finite-diff ω is ~17% off truth's instantaneous ω on
high-|ω| seeds. The principled reframe is **ω-grid + real-dynamics +
surrogate-MSE on local window** (s059h sketched, s059i to write); ω is
gridded directly so we don't depend on noisy finite-diff, and the cost is
the same surrogate-MSE residual that s059e's LM polish already used
successfully. Wind-down with no successful seed 28 inversion, but with the
next-agent architecture clear and the over-engineering trap (raw-numpy
quat math) flagged so the next attempt stays a thin wrapper around
`lib.forward.propagate_to_body_frame`.

# Sub-experiments

## s059_pilot.py — cohort validation pilot on seed 28 (T_A=312)

- OOM diagnosed and fixed (chunked validator scoring + free 7.2 GB R_cache
  after cloud generation). Pre-fix anon-rss 24.8 GB on a 30 GB box.
- After fix, ran clean in 2.9 min on cached cloud (5 min cold).
- Forward-prop: 184k candidates, discrimination 6524× over null, truth
  rank 6/4331 by sum-score. **But all 6 polishes (top-5 + truth-cluster
  injection) Band D**: truth-cluster polish ρ=53, q0_err=124°, |ω|_err=−17%.
  0/6 Band A∪B even with oracle.

## s059b_diagnose_seed28.py — hi-fi sanity on the polished states (killed mid-run)

- R1/R2 sanity passed: hi-fi ρ=0 at exact truth, surrogate ρ=0.48
- R3 (back-prop seed): hi-fi ρ=66.8, surrogate ρ=66.8 (agree to 0.1%)
- R4 (truth cluster polished): surrogate ρ=53.0, hi-fi ρ=63.6 — surrogate
  underestimates by 20% in this region, polish drifted away from truth
- Killed before sweep finished; saved data path was sufficient

## s059c_anchor_polish.py — LM in (rotvec_a, ω_body) at the anchor instead of (q_0, ω_0)

- Hypothesis: starting LM from a 8.5°-off seed in anchor coords (vs 113°
  off in t=0 coords) allows convergence
- **Refuted**: same Band D outcome on seed 28. The Jacobian still has
  back-prop sensitivity (gradient w.r.t. ω_a involves rewinding from t_a
  to t=0). With |ω|·t_a = 1.4 dps × 2250 s = 3240° accumulated rotation,
  the cost surface is non-convex.

## s059d_early_anchor.py — pick T_A near t=0 (killed before completion)

- Picks anchor as argmin |C_t| in epochs [3, 30) to shrink back-prop time
  ~30×
- Killed before scoring completed; superseded by s059e

## s059e_local_window.py — surrogate-MSE on local window only

- Window W=10 epochs (±72s around T_A=25). Bounds Jacobian sensitivity to
  |ω|·W = ~100° instead of |ω|·T = 5000°+
- Truth cluster (rank 390/25,348 by score, oracle-injected into polish
  set) → polished local ρ=0.31 (q0_err=0.14°, |ω|_err=−0.03%, ω_dir=0.02°);
  full-traj hi-fi ρ=3.799 → **Band B**
- Other 5 (top-5 by score) all Band D, q0_err ≥ 86°
- **CRITICAL CAVEAT**: The Band B success on seed 28 only happens because
  we deliberately polished the truth cluster after looking up its
  cluster_id. Without that oracle injection, truth was at rank 390 — the
  pipeline would never select it.
- Wall: 17.2 min (clustering 932k candidates was 6.5 min; polish 6×~3s;
  hi-fi 6×~50s)

## s059f_score_diagnostic.py — why doesn't the score rank truth highly?

- Top-20 by score on seed 28 T_A=312: 17/20 sit at qa_d ≈ 178° from truth
  (canonical-distance still 178° after twin-collapse — these are NOT
  truth's body-twins, they're independent (q, ω) competitors)
- Tier-1 validators (smallest |C_v|, highest weight): truth hits **0/8**
  while top-1 hits 8/8. Entire score gap is in tier 1.
- Diagnosis: truth's finite-diff ω over Δ_gen=15 has 17.33% magnitude
  error vs truth's instantaneous ω. Const-ω propagation with this wrong
  ω lands ~30° off truth-q at validators (Δω·dt over 10 epochs), missing
  the 5° hit threshold. Top scorers happen to const-ω-graze dense
  validator regions; their (q, ω) is dynamics-irrelevant.
- Pool-quantization compounds: 100k Sobol on SO(3) has ~7° spacing; 5°
  hit threshold is below noise floor → even truth-q + truth-ω would clip.

## s059g_realdyn_score.py — replace const-ω with real rigid-body dynamics

- Per-candidate Pool(8) call to `propagate_attitude(mode="tumbling")` with
  ω_body = q_a^* · ω_inertial · q_a, rtol=1e-6
- **Worse**: truth rank 1988 → 15502 under real dynamics. Hits dropped
  10/30 → 5/30. Real dynamics with WRONG initial ω diverges nonlinearly;
  const-ω at least linearly approximated displacement.
- Top-20 now dominated by three different (q, ω) basins (qa_d ≈ 158°,
  117°, 177°) — real-dynamics-consistent false positives.
- **Fundamental insight**: The candidate ω from finite-diff is the
  bottleneck, not const-ω vs real-dynamics. To fix the score, we must
  decouple ω from finite-diff.

## s059h_omega_grid.py — ω-grid + ψ-factorization + surrogate-MSE on local window

- Architectural reform attempt: sample ω_body directly from a Fibonacci-
  direction × magnitude grid covering the polhode-prior bracket. Truth-ω
  is in the grid by construction.
- Per ω: compute ψ_body(t) once via real-dynamics (identity, ω) propagated
  over local window. Then q(t) = q_a ⊗ ψ(t) batched across all q_a in
  C_a. Score by surrogate-MSE on local window.
- **Multiple iterations of speed/correctness debugging**:
  - Pool size 8 → 16 → 24 (for the 32-core box)
  - SURROGATE_BATCH_OMEGA tuning (200 was too big; 25 was small enough
    for visible progress)
  - scipy.Rotation overhead → raw-numpy quat math (helped a bit)
  - **Convention bug**: forgot `sun_pos - sat_pos` and normalisation;
    used `quat_inv_rotate` (q^* sandwich) instead of `quat_rotate`
    (q sandwich). Caught only after first tiny grid returned MSE = 3e+19.
- After conventions fixed: tiny grid (50 dirs × 4 mags × W=3, 76,600
  candidates) gives sane MSE [0.024, 41.3], truth-closest grid point
  (qa_d=8.5°, ω_d ≈ small) at MSE=1.05 ρ=20.5, rank 4439/76600 (top
  5.8%). Top-K ≠ truth — coarse grid resolution + multi-solution
  competitors.
- **Throughput**: 76600 × 7 epochs in 8.5s with Pool(24) = ~63k evals/sec
  aggregate. For 5-min wall budget = ~19M evals.
- **Over-engineering retraction (this thread's main lesson)**: I rebuilt
  raw-numpy quaternion math + ψ-factorization + custom broadcasting
  when `lib.forward.propagate_to_body_frame` already does the body-frame
  rotation correctly, and `s059e.make_residual_local` already evaluates
  surrogate-MSE on a local window. **Next iteration (s059i) should be a
  thin wrapper around the existing residual function**, not a custom
  rebuild.

# Numbers worth remembering

| seed | T_A | n_omega | n_window | total evals | wall   | truth rank        | best truth ρ_local | best truth band hi-fi |
|------|-----|---------|----------|-------------|--------|-------------------|--------------------|-----------------------|
| 28   | 312 | 184k pairs | 30 (validators) | const-ω hit-count | 5 min  | 1502/184k (score), 6/4331 (cluster sum) | 53 (full-traj)         | D (oracle-injected only) |
| 28   | 25  | 932k pairs | local W=10 | s059e local cost  | 17 min | 390/25k (cluster sum)                 | 0.31 (local)           | B (ρ=3.80, oracle-injected) |
| 28   | 25  | 200 grid    | local W=3  | s059h tiny       | 8.5 s  | 4439/76600 (grid)                     | 20.5 (local, coarse)   | not run                |

Surrogate eval rate (single-thread): 12k/sec; aggregate Pool(24): ~63k/sec.

# Why this matters / next move

The next agent should write **s059i** as a thin wrapper:
- Reuse `lib.forward.propagate_to_body_frame` for the body-frame transform
- Reuse `s059e.make_residual_local`'s exact residual computation (not
  re-implemented)
- Iterate over the (q_a, ω) grid; per pair, evaluate the residual at
  rotvec=0 and ω=ω_grid; score = mean(residual²)
- Pool(24) over chunks of (q_a, ω) pairs
- Top-K → LM polish (`s059e.lm_polish` exists) → hi-fi check
- **Honest yield**: count Band A∪B from top-K only; report
  truth-grid-injection result separately as DIAGNOSTIC

Sizing for 5-min budget: ~19M surrogate evals total. With |C_a|=383 and
W=5 (n_window=11): n_omega ≤ 4500. With W=10 (n_window=21): n_omega ≤
2350. Recommend the larger window (more discrimination per pair) and
800 dir × 6 mag = 4800 ω, W=5 — fits at ~5 min.

# Out of scope

- s058 reproduction with oracle removed (would just re-confirm seed 89
  truth rank 107/325 → 0/5 yield without injection; no new info)
- Hi-fi-cost LM polish (surrogate is sufficient; hi-fi reserved for final
  ρ-band check per `feedback_lm_cost_use_surrogate.md`)
- Cohort-scale runs across other seeds — wait for s059i to validate
  architecture on a single seed first

# Cross-references

- `feedback_oracle_injection_taints_yield.md` — methodology lesson
- `feedback_use_existing_lib_forward.md` — over-engineering lesson
- `project_score_function_broken_const_omega.md` — load-bearing diagnosis
- `project_omega_grid_architecture.md` — the reframe
