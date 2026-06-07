---
title: "m138 — H1 surrogate-attitude-isoshell pilot: q0-hypothesis cluster cost"
type: experiment
sources:
  - "notebooks/inversion/m138_isoshell_h1.py"
  - "data/results/inversion_diagnostics/m138_isoshell_h1/seed_091/"
related:
  - "[[m136_kernel_consistency_failure]]"
  - "[[m137_lofi_surr_sort_pipeline]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[attitude-level-set-disconnection]]"
  - "[[surrogate-model]]"
  - "[[twin-degeneracy]]"
  - "[[omega-sign-degeneracy]]"
  - "[[multi-solution-philosophy]]"
created: 2026-04-28
updated: 2026-04-28
confidence: medium
---

# m138 — H1 surrogate-attitude-isoshell pilot

#open (single-seed validation; failure-cohort generalisation pending)

## What

First implementation of H1: an upstream replacement for m103's grid+NM+geo
that builds per-epoch attitude level sets `L(t)` from the surrogate, then
scores ω-candidates by **q0-hypothesis cluster tightness** in the union of
`L(t) · conj(q_world(t))` across constraint epochs. No alignment cost
anywhere.

Algorithm in three stages:

1. **Stage 1 — per-epoch L(t)**: pick ~30 bright (mag<11) constraint epochs.
   Sample 64k uniform-random SO(3) attitudes (shared grid). For each constraint
   epoch t, compute surrogate prediction at every sample; threshold
   `|pred - obs| < 2σ` (σ=0.06 mag combined) → kept indices per epoch.
   Wall: ~140s/seed (Pool-able but currently sequential).
2. **Stage 2 — DBSCAN clustering** at 15° eps, min_samples=5 → per-epoch
   components. Median ~11 components per epoch on seed 91 (consistent with
   the disconnected-level-set picture). Sub-second.
3. **Stage 3 — ω-grid scoring**. For each candidate ω in {Fibonacci dirs} ×
   {log-spaced |ω| around peak-count base}: propagate q0=I forward to all
   constraint epochs (custom batched RK4, validated to 1e-5° vs the project's
   `propagate_attitude`), compute the q0-hypothesis cloud
   `q0_hat[t,i] = q_kept[t,i] · conj(q_world(t))` over ALL kept points (not
   centroids), then **cost = -(max unique-epoch count within 8° geodesic
   ball)** in the cloud. Best q0 cluster centroid is also returned as a
   warm-start for downstream m115.

Why "unique-epoch count" not "raw density": at correlated nearby epochs the
same kept-q sets repeat, so raw density inflates artificially. Counting
unique epochs in each query's neighbourhood rewards trajectories that
**thread through the disconnected-component union across many distinct t**.
This is the precise algorithmic shape the
[[attitude-level-set-disconnection]] concept page argues for.

## Single-seed result (seed 91, m048)

Configuration: 64k SO(3), 1000 Fibonacci dirs × 20 log-spaced mags = 20k
candidates. 30 constraint epochs.

| metric | value |
|---|---:|
| pool_min ω-dir err in top-30 | **1.76°** |
| rank-1 ω-dir err | 176.04° (sign-flipped, |ω|-err +0.2%) |
| rank-2 ω-dir err | **1.76°** (truth-direction, |ω|-err +0.2%) |
| rank-8 ω-dir err | 6.28° (q0_est 10.8° from twin) |
| Stage 1 wall | 143 s |
| Stage 3 wall | 450 s |
| Total wall | 594 s |

Rank-1 is a known [[omega-sign-degeneracy]] artefact (anti-truth direction
with truth's exact magnitude — produces a kinematically-flipped trajectory
that intersects the same per-epoch L(t) sets). Rank-2 is the truth direction
to grid resolution.

**Cost-shape diagnostic (offline on saved levelset)**: feeding exact
truth ω into the cost gave 28 unique epochs (of 30) in the densest ball,
vs 14 for a random ω of the same magnitude. The cost has a clean signal-
to-noise ratio of ~2× at this resolution, but the discrete grid "snaps" to
the wrong sign (176° ω-flip) when the magnitude grid is fine enough to
resolve truth's |ω| exactly — both ω and -ω hit equally-strong density
clusters.

## What this shows

1. **The cost shape works** on a Band-A seed where alignment cost has been
   shown to be anti-correlated with truth (m135 Finding 1: cost(geo_best) <<
   cost(truth) by 1.85–17 orders of magnitude). H1's cost ranks truth-near
   at top-2; the m103 alignment-cost path ranked truth at 174°.
2. **The pool quality at H1's hand-off point is acceptable for m115's 5°
   bridging radius**: top-30 contains a 1.76° candidate, well within the
   bridging radius from `m115_de_bridging_radius`. The 6.28° rank-8 candidate
   is borderline.
3. **No alignment cost anywhere in the pipeline up to this point** — H1's
   output goes directly to m115 + m126, both of which use surrogate / hi-fi
   costs. The Step-4-Geo trap that destroyed m137's M2 sort patch is bypassed.
4. **Magnitude-grid resolution matters**. The earlier 5-mag grid at 1.5x
   spacing collapsed the signal (truth-near grid cand was -8.9% off in |ω|,
   producing wrong q_world(t) drift over the 1-hour window). 20-mag log
   grid at 1.13x spacing recovered the signal. Magnitude grid spacing should
   be ≤5% to preserve density.

## What this rules out (so far)

- **m136-style anchor min-collapse**: this cost has no `min over Q_a`. The
  unique-epoch-count over the union of kept points respects component
  disconnection structurally.
- **Single-stage patches on m103 (m137 ceiling)**: H1 entirely replaces
  the grid+NM+geo trio with one Stage-3 scoring pass.

## What this does NOT validate

- **Generalisation to the failure cohort**: seed 91 has high-quality
  constraints (55 bright epochs, glint-rich). Seeds 47 / 51 / 79 / 84 / 89
  are pending. Especially worried about 51 / 79 (16°-pool-min in m103;
  if their L(t) sets are too small / ill-formed, the q0-cluster signal
  may collapse).
- **m115 hand-off integration**: H1 produces a top-30 (q0_est, ω) pool with
  q0 estimates near the twin (rank-2) and varied (rank-1, 8). Whether m115
  + m126 successfully closes from this pool to a hi-fi-validated solution
  is the next gate.
- **Wall-time scalability**: 594 s/seed at 1k×20 grid. 16k×20 = 16× would
  give ~160 min. Stage 3's per-candidate Python loop (BallTree build +
  unique-epoch counting) is the bottleneck. Vectorisation across candidates
  would close most of the gap.

## Bugs caught

1. **`PROJECT_ROOT = parents[1]`** in m138 — same shape as the
   `audit_failure_seed_battery.py` bug (resolved to `notebooks/` not project
   root). Fixed to `parents[2]`.
2. **Naive cluster-centroid cost gave 2× weaker signal** than the kept-point
   union approach (centroids drift along the L(t) manifold strip across
   epochs, producing a wandering q0_hat sequence even at correct ω).
3. **5-mag grid was too sparse** for |ω| resolution; signal collapsed at
   8% magnitude error.
4. **Raw-density cost double-counts correlated kept points across nearby
   epochs**. Switching to unique-epoch counting breaks the spurious
   inflation.

## Compute budget

- Stage 1: 64k × 30 epochs × surrogate v2 (~75 µs/sample) ≈ 144 s, sequential
  in single process. Pool(8) parallelism per epoch would give ~20 s but
  requires per-process surrogate load (~5 s overhead per worker).
- Stage 3: 20k candidates × ~22 ms = 440 s sequential. Per-candidate cost
  is BallTree(7000 points) build + `query_radius` + `np.unique` over
  neighbourhoods.

Memory: per-candidate cloud is 7k × 4 floats = 220 KB; chunk of 2000 cand
× 30 epochs × 4 quat = 1.9 MB live. All fine.

## Files

- `notebooks/inversion/m138_isoshell_h1.py` — single-script driver
  (Stage 1 + Stage 2 + Stage 3 + audit).
- `notebooks/inversion/m138_check_residuals.py` — diagnostic: surrogate-
  at-truth residuals across constraint epochs (max 0.10 mag at bright,
  0% over 0.12 tol band on seed 91).
- `notebooks/inversion/m138_validate_propagation.py` — RK4 vs project's
  `propagate_attitude`: matches to 1e-5° (left-multiply convention
  `q_truth(t) = q0_truth · q_world(t)`).
- `notebooks/inversion/m138_debug_cost.py` — interactive cost-shape probe
  (truth ω vs random ω vs wrong-mag).
- `data/results/inversion_diagnostics/m138_isoshell_h1/seed_091/` —
  `levelset_ckpt.npz`, `components_ckpt.npz`, `isoshell_ckpt.npz`,
  `result.json`.

## Next

Run seeds 47 / 51 / 79 / 84 / 89 at the same configuration. Then:

- If pool_min in top-30 ≤ 5° on the failure cohort → integrate as drop-in
  geo_ckpt-format input to `m115.load_omega_candidates`. Test end-to-end
  pipeline performance.
- If pool_min > 5° on some seeds → diagnose: is L(t) too sparse (constraint-
  epoch starvation)? Is the cost double-counting from a different artefact?
  Bump SO(3) sample count, mag-grid density, or constraint-epoch criterion.
