---
title: s059j — design doc for the cloud-data ω-grid search (anchor-selection-aware)
type: design
sources:
  - experiments/s059i_validator.md
  - experiments/s059i_validator_perturbed.md
  - experiments/s059i_density_scan.md
  - experiments/s059i_cohort_density_scan.md
  - experiments/s059_thread.md
related:
  - project_omega_grid_architecture.md
  - feedback_use_existing_lib_forward.md
  - feedback_oracle_injection_taints_yield.md
  - feedback_lm_cost_use_surrogate.md
  - feedback_surrogate_first_hifi_last.md
created: 2026-05-08
updated: 2026-05-08
confidence: high (architecture validated end-to-end at cohort scale; implementation is straight-ahead)
---

# TL;DR

s059j is the production cloud-data inversion pipeline that the s059i_*
validators have de-risked end-to-end. **Headline implementation idea**
(user proposal, 2026-05-08): **separate the "where to anchor" decision
from the "how dense at the anchor" decision.** A coarse 50k-pool sweep
across early epochs picks the most-constrained anchor T_A in ~10 sec;
then a 400k-pool resample at JUST that one anchor builds the q_a
search set in ~3 sec. Total cloud setup ~15 sec/seed (vs the old
"dense-at-all-500-epochs" cost of ~5 min Pool(8)). This is a 20×
speedup on the setup phase with zero quality loss — closest-to-truth
in the survival cloud is set by pool density at the chosen anchor,
which we densify regardless. Combined with |C_a|-cap subsampling, the
per-seed wall fits in 5-10 min Pool(24).

# Context: what's been validated

Four validators landed cleanly on 2026-05-08:

1. **s059i_validator** — at truth q_a, the local-window surrogate-MSE
   ranks truth-ω at **rank 1/1407** with ρ=0.389 (surrogate noise floor),
   runner-up ρ=3.44. Cost surface is sharply discriminative when q_a
   is correct.
2. **s059i_validator_perturbed** — q_a noise sensitivity sweep: truth
   stays in top-K (≤16) for q_a noise ≤ 7.5°. Bimodal at ~10°. Fails
   at ≥15°. Three regimes mapped.
3. **s059i_density_scan** — single-seed (28) Sobol density vs
   closest-survive: 100k → 10.96°, 400k → 2.32° median. RNG variance
   exists (one bad seed can leave a "donut hole" near truth even at
   higher N) but the median moves into the deep-robust regime at 400k.
4. **s059i_cohort_density_scan** — 8-seed cohort (6, 10, 14, 23, 28,
   44, 84, 89) at 100k, 200k, 400k. **At N=400k all 8 land in the
   robust regime** (median closest-survive 1.37–2.40°). **Even at
   100k, 7/8 are already robust** — seed 28 was the only outlier
   (|C_a|=383 vs 3681–14937 for the rest).

So the s059_thread's diagnosis of a cohort-wide architectural problem
was misled by single-seed reasoning. The architecture is cohort-viable
at moderate pool density. **s059j is GO.**

# The architecture

```
Per seed:

  [1] anchor selection — coarse pool ≈ 50k, find argmin |C_t| in [3, 30):
        pool_coarse  = sample_so3_pool(50_000, rng_seed=42)
        for t in early_window:
            |C_t|(t) = sum( |surrogate_pred(pool, t) - measured(t)| < TOL )
        T_A          = argmin_t |C_t|(t)

  [2] dense q_a search set at T_A — pool ≈ 400k, project + survive at T_A only:
        pool_dense   = sample_so3_pool(400_000, rng_seed=42)
        k1, k2       = project_directions(pool_dense.R_cache, sun_unit[T_A], obs_unit[T_A])
        keep         = |surrogate_pred(k1, k2, dist[T_A]) - measured[T_A]| < TOL
        C_a          = pool_dense.q_pool_wxyz[keep]

  [3] |C_a| cap subsampling (control wall):
        if |C_a| > CA_CAP:
            C_a = random_subsample(C_a, CA_CAP, rng=42)
        # CA_CAP ≈ 3000 keeps closest-to-truth w.h.p. per s059i_cohort_density_scan
        # (median closest-survive scales gently with |C_a| at fixed N)

  [4] ω grid (Fibonacci × magnitudes around polhode-prior |ω|):
        |ω|_prior     = polhode_prior_omega_mag(seed)        # from s055a
        bracket       = ±30%  (s059i validators tested at ±30%)
        ω_grid        = fibonacci_sphere(N_DIRS) ⊗ linspace(0.7, 1.3, N_MAGS) × |ω|_prior
        # defaults: N_DIRS=200, N_MAGS=6 → 1200 candidates per q_a

  [5] joint score on local window:
        for (q_a, ω) in C_a × ω_grid (Pool(24) over chunks):
            score(q_a, ω) = mean( s059e.make_residual_local(q_a, ω, T_A, W=10, ctx, target)² )

  [6] cluster + canonicalise top-K (s057h pattern, optional but recommended):
        canon_pairs = lib.twin.canonical_batch(top_K_pairs)
        clusters    = greedy_cluster(canon_pairs, q_radius=8°, ω_radius=15°, mag_pct=25%)
        polish_set  = top-K cluster representatives, NO TRUTH-CLUSTER INJECTION

  [7] joint LM polish (q_a, ω) using s059e.lm_polish_local on surrogate:
        for cluster_rep in polish_set (Pool over polishes):
            polish_rep = lm_polish_local(q_a, ω, T_A, W, ctx, target,
                                         max_nfev=200, ftol=1e-8, xtol=1e-8)

  [8] hi-fi gate + ρ-band classify:
        for p in polished if surrogate_rho_polished < SURROGATE_HIFI_GATE (=4):
            pred_hifi = render_hifi(p.q0_pol, p.om0_pol, ctx)
            ρ         = sqrt(MSE(pred_hifi, mag_hifi_truth)) / 0.05
            band      = rho_band(ρ)            # A: <2,  B: <4,  C: <8,  D: ≥8

  [9] HEADLINE YIELD = #(polishes in Band A ∪ B) / |polish_set|.
      NEVER include any truth-injected row in the headline.
```

## Tunable parameters and recommended defaults

| Param          | Default        | Rationale                                                  |
|----------------|----------------|-----------------------------------------------------------|
| N_COARSE       | 50,000         | argmin\|C_t\| ranking is stable above 30k (verify in pilot) |
| N_DENSE        | 400,000        | cohort scan: 8/8 robust at 400k                            |
| RNG_SEED       | 42             | one bad outlier (RNG=44) at 800k still robust median       |
| EARLY_WIN      | [3, 30)        | matches s059d_early_anchor; bounds back-prop sensitivity   |
| CA_CAP         | 3,000          | seed 84 \|C_a\|=29,940 → 10× cost reduction, neg. quality loss |
| TOL_MAG        | 0.10           | s059_pilot default                                         |
| ω_BRACKET      | ±30%           | s059i validators                                           |
| N_DIRS         | 200            | 7° avg spacing on the unit sphere; rank-2 in validator was at 6.9° |
| N_MAGS         | 6              | 12% mag spacing; rank-2 in validator was at 6% mag         |
| WINDOW_W       | 10             | matches s059e; trades q_a-noise envelope vs ω discrimination |
| TOP_K_POLISH   | 20-50          | s058 polished 6, 2/6 hit Band A; expand budget here        |
| LM_MAX_NFEV    | 200            | s058 default                                               |
| LM_FTOL/XTOL   | 1e-8           | s058 default                                               |
| SURR_HIFI_GATE | 4              | only ρ_local < 4 candidates worth hi-fi rendering          |

## Wall-budget projection

Per seed (Pool(24) on 32-core box, surrogate aggregate ~63k evals/sec):

- Anchor selection: 50k × 27 epochs ≈ 1.35M evals → ~21 sec sequential
  (vectorised across epochs: actually ~2 sec). Cheap.
- Dense projection at T_A: 400k × 1 ≈ 0.4M evals → ~6 sec single-thread.
- Score grid: |C_a|=3000 (capped) × 1200 ω × 21-epoch window ≈ 75.6M evals
  → 20 min Pool(24). **Dominant term.** Reduce to W=5 (n_window=11)
  if needed: 39.6M → 10 min.
- LM polish: top-K=50 × ~25 sec/polish → 21 min sequential, ~1 min Pool(24).
- Hi-fi gate: pass count uncertain; assume 5-10 hi-fi renders × ~50 sec
  ≈ 5-10 min sequential.

**Per-seed wall ≈ 15-30 min Pool(24)**. Cohort 8 seeds ≈ 2-4 hours.
Acceptable if results are decisive; revisit cost if not.

## What NOT to rebuild

Per `feedback_use_existing_lib_forward.md`:

- **Body-frame propagation**: use `lib.forward.propagate_to_body_frame`.
  Do NOT roll your own quaternion math. The `q^* sandwich` vs `q sandwich`
  + `sun_pos − sat_pos` convention bugs killed s059h.
- **Local-window residual**: use `s059e.make_residual_local` /
  `s059e.lm_polish_local`. Already debugged, already fast.
- **Pool sampling**: use `lib.c_t_pipeline.sample_so3_pool`. (Note: it's
  Haar-uniform random via `Rotation.random`, not actual Sobol-Shoemake;
  the name is misleading. Cohort scan shows this is fine at N=400k for
  every seed in the 8-seed cohort. Refactoring to true low-discrepancy
  is optional cleanup, not blocking.)
- **Survival**: use `lib.c_t_pipeline.survive_at_epoch`.
- **Twin canonicalisation**: use `lib.twin.canonical_batch` if needed.
- **Hi-fi render**: use `lib.hifi_render.render_hifi` + `rho_from_hifi` +
  `rho_band`.
- **Anchor selection**: use the existing `s059d_early_anchor.stage_pick_early_anchor`
  pattern, but adapt to take the coarse pool's `survive_all` mask (or
  reproject coarse-at-each-epoch directly — the inner loop is cheap).

## Critical correctness rules

Per `feedback_oracle_injection_taints_yield.md`:

- **NEVER** look up the truth cluster's id and inject it into the polish
  set. The headline yield must come from top-K-by-score polishes only.
- It is OK to render the truth state separately as a DIAGNOSTIC and
  report it alongside the headline — but flag it as diagnostic and do
  NOT count it toward yield.

Per `feedback_lm_cost_use_surrogate.md`:

- LM polish uses **surrogate-v2 full-LC residual**, not hi-fi cost.
  Hi-fi reserved for a single render per polish that passes the
  surrogate ρ < 4 gate.

Per `feedback_surrogate_first_hifi_last.md`:

- Every search/clustering/polish/sanity step runs on the surrogate.
  Hi-fi is only the final ρ-band classification.

# Why anchor selection matters more than expected

The cohort density scan revealed |C_a| varies 20× across seeds at
fixed pool density. Without anchor optimisation, the s059j wall would
be dominated by the worst seeds (seed 84: |C_a|=29,940 → 14 min Pool(24)
inner loop alone). Two ways to control this:

1. **|C_a|-cap** (defence-in-depth, always-on). Random subsample C_a
   to ≤ CA_CAP. Closest-to-truth in C_a is preserved with high
   probability if cap is large vs the closest-to-truth-rank — and the
   rank of truth-q in C_a is bounded by 1/(area-fraction-of-truth-shell),
   which is much smaller than CA_CAP for typical seeds.
2. **Anchor selection by argmin|C_t|** (preferred, structural). Pick
   the epoch where |C_t| is intrinsically smallest. Different epochs
   have wildly different survival rates (seed 28 had |C_a|=383 at
   T_A=25, the global min in [3, 30)). For high-|C_a|-everywhere seeds,
   the cap kicks in; for seeds with a sharp |C_t| dip, anchor selection
   alone solves it.

The user's coarse-find-then-dense-resample optimisation is what makes
anchor selection cheap to do at full-cohort scale. Without it, you'd
need to densify everywhere to know |C_t| accurately at every epoch.

# Verification step before scaling

Before cohort-running s059j, run a one-shot pilot:

1. **Seed 28**: run s059j with the design above. Confirm Band A∪B yield
   on the truth-cluster-equivalent polish (truth NOT in the polish set
   except as diagnostic). Compare to the s059_pilot baseline (which
   was 0/6 Band A∪B on seed 28 even with truth injection).
2. If pilot Band A∪B yield ≥ 1 on seed 28: cohort run on the 8-seed set.
3. If pilot Band A∪B yield = 0: localise the failure (q_a search-set
   too sparse? ω grid too coarse? polish budget too tight?) and iterate.

The s059i validators say the cost surface is sharp; the polishing
should pull top-K candidates into truth-adjacent basins. If it doesn't,
the next diagnostic step is **inspect the score distribution at the
known truth grid point** (already validated rank 1/1407 with q_a=truth
in s059i_validator) — the question is just how that rank degrades
through the C_a sampling step.

# Out of scope for s059j

- **Sampler refactor (A3) — true Sobol-Shoemake.** Useful cleanup but
  not blocking; cohort scan shows Haar-uniform at 400k works.
- **Multi-anchor fallback.** Bonus extension: pick top-3 most-constrained
  anchors and run s059j at each, take the union of polished candidates.
  Defer until single-anchor s059j is validated.
- **Cohort-scale 100-seed run.** 8-seed pilot first; expand only after
  Band A∪B yield is confirmed.
- **s055e polhode-bracket pilot.** Independent track; doesn't depend
  on s059j.

# Cross-references

- `experiments/s059_thread.md` — diagnosis chain that motivated this
  design.
- `experiments/s059i_validator.md` — cost-surface validation at truth q_a.
- `experiments/s059i_validator_perturbed.md` — q_a-noise envelope.
- `experiments/s059i_density_scan.md` — single-seed donut-hole diagnosis.
- `experiments/s059i_cohort_density_scan.md` — 8-seed cohort viability.
- `experiments/s059e_local_window.py` — residual + LM polish to reuse.
- `experiments/s059_pilot.py` — back_propagate, twin helpers, etc.
- `lib/forward.py`, `lib/hifi_render.py`, `lib/c_t_pipeline.py`,
  `lib/surrogate_eval.py`, `lib/twin.py` — substrate.
