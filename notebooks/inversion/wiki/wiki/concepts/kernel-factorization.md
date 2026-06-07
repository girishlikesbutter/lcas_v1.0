---
title: "Kernel factorisation for cost-variant iteration"
type: concept
sources:
  - "raw/inversion_diagnostics/m118/seed_014/kernel.npz"
related:
  - "[[m118_cost_comparison]]"
  - "[[grid-search]]"
  - "[[alignment-cost]]"
created: 2026-04-15
updated: 2026-04-16
confidence: high
---

# Kernel factorisation for cost-variant iteration

## The core idea

Every alignment-style cost the project has used can be written as

```
cost(dir, mag, centroid_choice, phi) =
    Σ_epochs w_ep · f( PAB_body(dir, mag, centroid_choice, phi, epoch), targets_ep )
```

The **body-frame PAB at each constraint epoch** depends only on the grid point (dir, mag), anchor attitude (centroid_choice, phi), and kinematics (ω-propagation). It does **not** depend on which cost function `f` or which target set `targets_ep` we choose.

Factor the computation accordingly:

- **Kernel** (expensive, cost-agnostic): propagate `q_delta[dir, mag, epoch]` for the whole grid once. Stores propagated delta quaternions per (direction, magnitude, constraint epoch). Save to disk.
- **Scorer** (cheap, cost-specific): load kernel + target directions + weight scheme → evaluate cost on the grid. Sub-30-second operation per variant with direction-chunk parallelism.

## Why this matters

Before m118, testing a new cost variant required re-running grid+NM+phi-sweep — ~15 min per seed. With kernel factorisation, a new cost variant is a ~20 s batch on an existing kernel. Two orders of magnitude faster iteration.

This changes what experiments are worth running. Trying 5 cost variants on 10 seeds is now a ~20 min compute window, not a 12-hour batch.

## What's in the kernel (m118 schema)

Per seed, `kernel.npz`:

- `q_delta` — `(N_DIRS, N_MAGS, N_CONSTRAINT_EPOCHS, 4)` float32, propagated delta quaternions (wxyz)
- `omega_dirs`, `omega_mags` — grid definitions
- `constraint_epochs` — epoch indices used
- `pab_j2000_at_constraints` — PAB in J2000 at each epoch
- `anchor_epoch`, `anchor_centroids`, `anchor_ipl_length`, `anchor_loop_count`, `anchor_time`, `pab_anchor_j2000`
- `anchor_selection_trace` — per-tier cascade info (which loop counts had how many candidates below the q75 length threshold)
- `anchor_length_q75`
- `truth_q0`, `truth_omega0`, `truth_pab_body_at_constraints` — ground truth

## What's NOT in the kernel (cost-variant invariants)

- The cost function itself (bullseye vs ring vs weighted vs ...)
- The target directions at each epoch (facet normals vs IPL centroids vs single active centroid vs sampled loop points)
- The per-epoch weights (uniform vs 1/IPL_length)

## Anchor choice is baked in

The kernel is built for one specific anchor epoch. Changing the anchor requires a new kernel (`dt_constraints` changes). But this is cheap: ~95 s to build on Pool(24). So exploring different anchor selection strategies costs ~95 s per anchor tried.

## Constraint epoch set is baked in

Same story: changing which epochs count as constraints (spec peaks only, spec peaks ∪ tight-IPL, all 500) requires a new kernel.

## Parallelism notes

- Kernel build: multiprocessing Pool over directions, worker count = `MICRO118_POOL_SIZE`.
- Scorer: multiprocessing Pool over direction chunks inside the scorer, count = `MICRO118_COST_WORKERS`. Essential — the numpy operations in the scorer don't parallelise at the BLAS level (the inner einsum's contracted dimension is size 3, too small for threaded BLAS), so explicit process-level chunking is what saturates cores. Without this the scorer pins one thread.

## Invocation pattern (from [[m118_cost_comparison]])

```
# Build kernel (once per seed per anchor choice)
MICRO118_SEED=14 MICRO118_POOL_SIZE=24 python3 m118_kernel_computation.py

# Score one or many variants on that kernel
MICRO118_SEEDS=14 MICRO118_VARIANTS=facet_normal,ipl_active_ring \
  MICRO118_COST_WORKERS=16 python3 m118_diagnostic_mode.py

# Set MICRO118_FORCE=1 to re-score existing variants (default: skip-if-exists)
```

## Reusability for future experiments

The kernel pattern generalises beyond alignment costs. Any cost of the form `Σ_ep f(PAB_body_at_ep, targets_at_ep)` slots in. This includes:

- Ring / bullseye / hybrid forms
- IPL-centroid / facet-normal / IPL-loop-sampled targets
- Tightness-weighted variants
- Extended epoch sets
- Magnitude-band variants with different thresholds

Costs that don't fit: anything needing full forward physics (e.g. predicting magnitude given attitude), which is what [[surrogate-de-search]] and [[surrogate-attitude-isoshell]] use instead.

## Success criterion for future additions

A new cost variant gets added by:
1. Writing a `score_<name>` function in `m118_cost_comparison.py` with the standard `(kernel_data, ipl_data, phi_arr, **kw)` signature.
2. Registering it in the `SCORERS` dict.
3. Running `MICRO118_VARIANTS=<name>` — it will score only the new variant and reuse everything else.
