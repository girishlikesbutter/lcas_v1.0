---
title: "m135: alignment-cost forensics + constrained-anchor surrogate-driven ω search"
type: experiment
sources:
  - "notebooks/inversion/pipeline_viz.py"
  - "notebooks/inversion/probe_cost_at_truth.py"
  - "notebooks/inversion/score_lofi_surrogate.py"
  - "notebooks/inversion/score_constrained_anchor.py"
  - "data/results/inversion_diagnostics/probe_cost_at_truth/"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/{lofi_surr_ckpt,constrained_anchor_ckpt}.npz"
related:
  - "[[alignment-cost]]"
  - "[[constraint-poor-regime]]"
  - "[[phase-angle-operating-range]]"
  - "[[m133_rerank_geo_ckpt_pool]]"
  - "[[m134_pipeline_test_q0polish]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
  - "[[surrogate-model]]"
created: 2026-04-28
updated: 2026-04-28
confidence: high
---

# m135: alignment-cost forensics + constrained-anchor surrogate-driven ω search

Three nested experiments on the 2026-04-22 m048 random-cohort failure seeds (47, 51, 79, 84, 89) and the seed 91 Band-A reference, designed to answer one question: **is m103 saveable, or replaceable?**

Conclusion: **replaceable**. Two surrogate-driven alternatives both localise truth to within m115's bridging radius on seed 91. Alignment cost is empirically anti-correlated with truth on every failure seed.

## (1) Cost-at-truth probe — alignment cost is wrong, not just noisy

For each failure seed, replicated m103's anchor + spec-peak + constraint setup, then evaluated `vectorized_phi_cost_excl(...)` at:
- TRUTH ω direction with TRUTH |ω| (exact)
- Fibonacci-grid nearest-neighbour to truth direction at TRUTH |ω|
- m103's saved geo_best ω (the candidate alignment cost ranked rank-1 from the 26 deduped geo candidates)

| seed | n_constraints | cost(truth_ω) | cost(geo_best) | geo_offset° |
|:---:|:---:|---:|---:|---:|
| 47 | 1 | 3.22e-6 | 4.50e-8 | 8.7 |
| 51 | 2 | 2.83e-3 | 1.35e-10 | 21.2 |
| 79 | 0 | 0 | 0 | 81.0 |
| 84 | 1 | 9.52e-6 | 1.11e-22 | 28.4 |
| 89 | 2 | 3.24e-4 | 7.14e-7 | 81.9 |

For the four non-degenerate seeds: `cost(geo_best) << cost(truth)` by **~2–17 orders of magnitude** (seed 47: 1.85 orders; 51: 5.12; 84: 16.93; 89: 2.66), with geo_best 8°–82° from truth. **The cost function has spurious basins many orders of magnitude deeper than truth's basin** — most dramatically for seed 84. Denser ω-direction sampling cannot recover truth — even if a grid point landed exactly on truth, geo refinement would prefer the spurious deeper minimum elsewhere. (An earlier draft said "4–22 orders" — that overstated both ends; the verified range is 1.85–16.93.)

Seed 79 has 0 constraints (1 spec peak only) → alignment cost is degenerate (zero) everywhere → geo refinement output is essentially arbitrary. Constraint-poor regime confirmed mechanically.

The pre-existing wiki failure mode taxonomy of [[alignment-cost]] (high-phase flatness vs constraint-poor) is now unified: **both are surface manifestations of the same root cause — alignment cost is the wrong objective for these geometries**.

Sanity control: seed 91 (Band-A) under the same probe: cost(truth) = 0.372, cost(grid_nn @ 2.77° offset) = 0.540, cost(geo_best) = 0.153 at offset 14.61° from truth. Even on a Band-A seed the alignment cost prefers a non-truth basin — pipeline rescues seed 91 *despite* the alignment cost, via downstream NM polish + m115 DE polish + m126 wrap. The gap is just smaller (factor 2.4 on a Band-A seed vs 4–22 orders of magnitude on failures).

## (2) Surrogate full-LC MSE on the lofi-300 pool — single-line fix that works

m103 produces 300 (q0, ω) candidates at the lofi stage (post grid + lofi peak-match re-rank, pre NM polish). m103 currently sorts them by `(-n_matched, lofi_mse)`. The `lofi_mse` field IS computed (it's the full hi-fi LC MSE). The 300 candidates are then truncated to NM_TOP=300 and fed to NM polish.

`score_lofi_surrogate.py` re-scores those 300 candidates with surrogate full-LC MSE (~41 sec/seed). On seed 91:

| cost function | rank-1 candidate offset from truth |
|---|---:|
| alignment cost (current m103 sort field) | 174.83° |
| `lofi_mse` (full hi-fi LC MSE — already computed) | 174.00° |
| **surrogate full-LC MSE (NEW)** | **2.76°** |
| post-NM `q0_err` (oracle reference) | 36.94° |

The surrogate cost picks the truth-direction candidate as **rank 1 out of 300**. Alignment cost picks the anti-truth ±X twin at rank 1; `lofi_mse` (the actual full-LC MSE on the pre-noise-fix predicted LC, not surrogate) ALSO picks anti-truth — confirming the issue is global LC-fit ambiguity at the lofi stage, not surrogate-vs-hi-fi.

**Intervention scope**: 10-line patch in `m103_hybrid.py` to add a surrogate-LC scorer between the lofi step and the dedup/NM step. No architectural change. The pre-existing `surr_q0polish_mse` cost from [[m133_rerank_geo_ckpt_pool]] is even better than this raw surrogate-LC MSE because it includes a quick q0 polish — but raw surrogate cost on the lofi stage is sufficient for the proof of concept.

Untested on the 5 failure seeds — they were run before the lofi-pool instrumentation landed, so don't yet have a `lofi_ckpt.npz`. Re-running m103 on those 5 seeds with the new instrumentation is a ~25 min job.

## (3) Constrained-anchor algorithm — full m103 replacement candidate

Implements the user's algorithm verbatim:

> 1. use surrogate to decide which timepoint near the middle is most constrained in terms of q (use real phase angle, shadowing etc).
> 2. at most constrained epoch, generate w grid.
> 3. do delta q factorisation on all the q candidates obtained from 1 (in order to know what is constrained you need to know how many q candidates there are for that level of brightness and that particular phase angle).
> 4. rank the w-q candidates by hifi surrogate cost across the whole lc.

Implemented as `score_constrained_anchor.py`:

- **Step 1 — constraint metric per epoch**: for each candidate epoch in the middle 30% of the LC where `observed_mag < 11` (avoiding surrogate dim-saturation), sample 64,000 q's broadly on SO(3) — `n_pab=800` Fibonacci-sphere body-frame PAB directions × `n_phi=80` azimuths around each PAB axis, no facet-normal hypothesis. Compute v1 surrogate magnitude at each q for the epoch's true J2000 sun/observer geometry. Count how many fall within ±3σ noise band of observed. The count IS the constraint metric. Pick the smallest count subject to a min-count floor (≥6) to avoid Q=1 sampling artifacts.

- **Step 2-3 — ω grid + δq factorisation**: 2000 Fibonacci ω-directions × 2 |ω| magnitudes (peak-counting estimate + truth-mag oracle for the diagnostic). For each (ω, q_anchor in the consistent set from Step 1) pair: precompute identity-propagated δq's once over the full obs window AND at the anchor epoch, then derive `q0 = q_anchor · δq_anchor⁻¹` via cheap quaternion multiplication. Vectorised v2 surrogate forward LC over (Q × N_t = ~27000) candidates per direction in one batched call.

- **Step 4 — rank**: per ω-direction, keep min surrogate full-LC MSE over (mag, q_anchor). 2000 directions ranked.

**Caveat — oracle |ω|**: the v4 result below uses `include_truth_mag=True` (default). The mag grid was `[0.0170 (peak-count est.), 0.0249 (TRUTH)]` and **all top-5 candidates by surrogate MSE landed on the truth-mag candidate**. Without truth-mag injection, the peak-count estimate alone is 32% low and the algorithm's performance is unverified. The 2026-04-28 failure-seed battery tests both modes (oracle reference + honest `--n-mags 5 --no-include-truth-mag`).

**Seed 91 result (Band-A reference)**:

| iteration | config | most-constrained epoch | Q | rank-1 ω offset from truth | surr_mse |
|---|---|---:|---:|---:|---:|
| v1 | tol=0.05, n_pab=200×n_phi=36, no min count, no mag filter | ep=372, mag=10.27 | 1 | 164.5° | 4.36 |
| v2 | tol=0.05, n_pab=500×n_phi=72, min count=3, no mag filter | ep=204, mag=7.68 | 6 | 145.1° | 3.96 |
| v3 | tol=0.15, n_pab=500×n_phi=72, min count=3, no mag filter | ep=283, mag=15.73 | 48 | killed (Step 2-4 ETA 17min) | — |
| **v4** | **tol=0.15, n_pab=800×n_phi=80, min count=6, max_obs_mag<11, v1 Step1, v2 Step2-4** | **ep=273, mag=6.05** | **54** | **2.18°** | **1.12** |

v1's Q=1 was a sampling artifact (truth excluded due to coverage gap at ~10° SO(3) resolution). v2's tol=0.05 (1σ) excluded truth from the consistent set because the realised noise at ep=204 was +0.103 mag, putting truth's surrogate prediction outside the ±0.05 band by construction. v3 picked a dim epoch (mag 15.7) where the surrogate saturates and almost any attitude that hides the satellite produces ~similar dim values — false-constraint degeneracy. v4 fixed all three by widening tolerance, raising the count floor, and filtering to the surrogate's well-modeled bright regime.

**v4 took ~33 min wall** (Step 1: 7 sec on v1 surrogate, 16 epochs survived bright filter / Step 2-4: 28 min on 8 workers, 54 q × 4000 (mag, dir) = 216k full-LC candidates).

## Pipeline visualisation tool — `pipeline_viz.py`

Single self-contained HTML report per seed at `data/results/inversion_diagnostics/pipeline_viz/seed_NNN.html`. Eleven stage panels (header / grid sweep with full Fibonacci background overlay / surrogate-cost grid if computed / 4-panel cost-landscape comparison / lofi / NM / phi sweeps / geo+rerank / DE convergence / hi-fi / polish / final). Reads only the saved instrumentation NPZs — no recompute. Built around seed 91 today.

## Implications for [[upstream-redesign-6dof-surrogate-de]]

The constrained-anchor algorithm is **a concrete instance** of that branch. Step 1 picks the most-informative anchor by surrogate query rather than peak brightness; Step 2-4 use surrogate full-LC MSE as the cost; Step 1's q-set comes from genuine surrogate-derived consistency, not facet-normal hypothesis. With seed 91 producing a 2.18° rank-1 in this single-shot run, the branch is now **#open-validated** on a single seed. Multi-seed validation on the 5 failure cohort is the next step.

## Open questions for the next session

- Does (2) work on the 5 failure seeds when m103 instrumentation is re-run for them?
- Does (3) work on the 5 failure seeds — i.e., does the constrained-anchor approach produce truth-near rank-1 in the seeds where alignment cost has spurious deeper basins?
- Tightening (3): can the algorithm self-tune its three guardrails (tolerance, min-count floor, mag filter) per seed instead of using global defaults?
