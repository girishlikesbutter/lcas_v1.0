---
title: "m142 — Surrogate-MSE re-rank on m141's seed-6 m103 pool: rank-9 ω-promotion REFUTED"
type: experiment
sources:
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/geo_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/geo_surr_ckpt.npz"
  - "notebooks/inversion/score_geo_surrogate.py"
related:
  - "[[m141_seed6_postfix_pipeline]]"
  - "[[m140_post_fix_lc_delta]]"
  - "[[m139_convention_bug_fix]]"
  - "[[m134_pipeline_test_q0polish]]"
  - "[[m138_seed47_lombscargle_bracket]]"
  - "[[m115_de_bridging_radius]]"
  - "[[surrogate-rerank]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# m142 — Surrogate-MSE re-rank on m141's seed-6 m103 pool

#post-fix-rerank-fails #seed6 #upstream-pool-deficient

## TL;DR

Re-ranked m141's saved 26-candidate `geo_ckpt.npz` pool by surrogate full-LC MSE (post-fix propagator, ~4 sec scoring wall via `score_geo_surrogate.py`, an adapter mirroring `score_lofi_surrogate.py`). **Headline finding: the rank-9 truth-near ω (w_err=16.86°) is DEMOTED to surr_mse rank 15/26**, not promoted. Pool-min surr_mse is 3.94 (ρ≈8.9, still Band D). The new surr_mse rank-1 has w_err=58.32° and q0_err=85.61° — neither truth-near. Surrogate **bright-epoch** MSE happens to *agree* with m103's geo_cost on this seed (the four lowest `surr_bright_mse` values 4.01/4.74/4.83/5.26 sit at m103 ranks 1/2/3/4) but those candidates are 53–57° from truth in ω. Neither full nor bright surrogate MSE can rescue this seed. The m103 pool simply does not contain a jointly truth-near `(q0, ω)` candidate — m103's phi-sweep failed to find a truth-q0 anchor for the rank-9 ω (its q0_err is 129.58° from truth). **Re-ranking cannot fix a pool that is missing the truth basin entirely.** This is the prompt's "failure case" — Play 3 (6-DOF surrogate DE upstream, replacing m103) becomes the next concrete escalation. m115 K=1 rerun NOT triggered (gating condition unmet).

## What

[[m141_seed6_postfix_pipeline]] established that under post-fix correct truth, m103's 26-candidate ω pool contains the truth-near ω at rank 9 (w_err=16.86°), but m103's `geo_cost` ranking buries it under 8 candidates with w_err 53–74°. The session-prompt question for m142 is whether **surrogate full-LC MSE re-ranking** of that same pool can promote the rank-9 candidate to top-K=3 — the precedent being [[m134_pipeline_test_q0polish]] (seed 67 bridged 14.63° to a Band-B q0-twin) and [[m138_seed47_lombscargle_bracket]] (seed 47 rescued from rank 22231 to rank 5 by the same lever).

This is the first surrogate-rerank test on **physically correct truth**.

## How

`notebooks/inversion/score_geo_surrogate.py` — a 130-line thin adapter for `score_lofi_surrogate.py`. Differences from the lofi version:

1. Reads `geo_ckpt.npz` (26 post-Step-4 candidates with `q0_refs`, `w0_refs`, `geo_costs`, `q0_ref_errs`, `w0_ref_errs`) instead of `lofi_ckpt.npz` (300 candidates).
2. Echoes the geo metadata into the output `geo_surr_ckpt.npz` for downstream convenience.
3. Prints a sorted ranking table (full pool, sorted by `surr_mse` ascending, columns: `surr_rk`, `geo_rk`, `surr_mse`, `surr_bright_mse`, `geo_cost`, `w_err`, `q0_err`).
4. Prints a headline summary of where the truth-nearest ω lands by both `geo_cost` and `surr_mse` rank.

Otherwise identical scoring chain: propagate via the **post-fix** `propagate_attitude` (LEFT-mult Hamilton, conv-(a)), build `R = scipy.from_quat(quats[xyzw]).as_matrix()`, compute `k1 = R @ sun_dir`, `k2 = R @ obs_dir`, surrogate-evaluate per epoch, MSE against `canonical_observed_lc(true_lc)` (the post-fix regenerated truth from `traj_seed006.npz`).

Wall budget: setup ~1 min (SPICE + surrogate load), scoring 4.0 sec for 26 candidates. No m103 re-run, no m115 launch.

## Result table (full 26-candidate pool, sorted by surr_mse ascending)

```
 surr_rk  geo_rk    surr_mse    surr_brt    geo_cost     w_err    q0_err
       1      26      3.9405     40.5743  1.7601e+00     58.32     85.61
       2      22      4.0567     42.3987  6.6990e-01     68.36     70.56
       3      13      4.0702     37.6906  1.7603e-01     67.54     98.71
       4      18      4.9497     20.3204  3.7594e-01     66.23    157.55
       5      11      5.1143     23.8413  1.2142e-01     80.24     58.63
       6      15      5.4422     27.2459  2.5968e-01     74.90    125.35
       7      21      5.4757     29.0239  5.2691e-01     85.65     31.20
       8      24      5.5732     35.1319  9.5051e-01     36.81    100.38
       9       6      6.1764     11.9106  3.4159e-02     63.97     64.78
      10      14      6.3911     35.1204  2.0308e-01     78.06    108.26
      11      17      6.6198     31.6836  3.3523e-01     36.16    135.27
      12      12      7.1412     13.1186  1.4363e-01     74.67     90.85
      13      25      7.1815     31.8637  1.3306e+00     63.19     39.47
      14      23      7.3147     33.8503  9.0785e-01     68.10    116.00
      15       9      7.7849     14.2361  8.3912e-02     16.86    129.58   ← truth-near ω
      16      10      8.6421     17.4759  9.1164e-02     81.31    137.19
      17      19      9.6784     21.9109  3.8206e-01     49.09    150.20
      18      20      9.7957     26.1925  5.0786e-01     47.95    169.08
      19      16     10.2827     22.3799  3.3436e-01     87.12    149.01
      20       4     12.8003      5.2645  9.9610e-03     52.85    161.40
      21       8     13.4248     17.6997  7.7542e-02     74.13    126.50
      22       2     13.5223      4.7357  7.6090e-03     53.56    174.87
      23       1     14.5011      4.0077  4.8672e-03     56.99    171.58   ← m103 geo-rank 1
      24       3     15.1410      4.8274  8.4396e-03     54.57    145.06
      25       5     17.7350      6.4318  3.1669e-02     67.28    153.85
      26       7     18.1521     16.2666  5.8768e-02     67.74    129.29
```

## Key observations

### 1. Truth-near ω demoted from m103-rank 9 → surr_mse-rank 15

Not promoted to top-K=3. Not even retained at its m103 position. The candidate's `surr_mse` is 7.78 — middling. Its `surr_bright_mse` is 14.24 — also middling. Re-ranking by either metric fails to surface it.

### 2. The structural bottleneck is q0, not ω

The truth-near ω candidate's q0 is 129.58° from truth. Propagating from a q0 that wrong, even with correct ω, produces a body-frame attitude history that never matches truth — so the LC predicted by the surrogate at correct (k1, k2) is far from observed. The surrogate is doing its job correctly; the *input attitude history* is wrong.

This makes the m115 K=1 rerun moot: even if we could feed `(q0=rank9, ω=rank9)` to a single-omega DE, m115's 3-DOF DE searches q0 around the input candidate. With q0 already 129° wrong, m115 is being asked to bridge ~129° in q0 and ~17° in ω simultaneously. Empirical bridging radius for q0 in 3-DOF DE is ~10–30° per [[m115_de_bridging_radius]] / [[m134_pipeline_test_q0polish]]; 129° is well outside.

The m141 finding is now sharper: **m103's phi-sweep** (the step that explores q0 around each ω candidate) didn't find a truth-q0 anchor for the rank-9 ω. That anchor should exist — for correct ω, there's by construction a basin of q0s that produce the correct LC — but the phi-sweep didn't sample it.

### 3. Bright-LC MSE and full-LC MSE disagree on this seed

`surr_bright_mse` (epochs with observed mag < 9) and `surr_mse` (full LC) point at different parts of the pool:

- m103 geo_cost ranks 1/2/3/4 (the four candidates with the lowest alignment cost) have the **lowest** `surr_bright_mse` (4.01 / 4.74 / 4.83 / 5.26) but the **highest** `surr_mse` (14.50 / 13.52 / 15.14 / 12.80, table rows 23 / 22 / 24 / 20).
- The rank-9 truth-near candidate has middling values on both (`surr_mse`=7.78, rank 15; `surr_bright_mse`=14.24, rank 8 by ascending bright MSE).
- No candidate dominates both metrics. The two cost surfaces favour disjoint subsets of the pool.

Interpreted: m103's alignment cost is doing what it always did — fitting the bright peaks. The surrogate evaluation confirms that those candidates *do* match observed bright epochs reasonably (low `surr_brt`), but their full-LC fit is bad because the dim regime is wrong. The truth-near ω candidate is the inverse — neither bright nor dim regime is a strong fit, because q0 is 129° wrong and the trajectory is geometrically scrambled relative to truth.

The [[m138_seed47_lombscargle_bracket]] precedent (surrogate full-LC MSE as the decisive single discriminator) does NOT generalise here. On seed 6, no single re-rank metric over the existing 26-pool surfaces the truth basin, because the 26-pool does not contain a jointly truth-near `(q0, ω)` candidate.

### 4. Pool-min surr_mse 3.94 = ρ ≈ 8.9 (Band D)

Even if we declare the surr_mse rank-1 candidate (w_err 58.32°, q0_err 85.61°) the "best", it's nowhere near the ρ < 4 acceptance bar. The buggy-truth wins on this seed (m098 PARTIAL→OK, m126 87% improvement, m131 ω-flip Band-A) had hi-fi MSE around 0.05–0.5 = ρ ≈ 1–3 (Band A∪B). The post-fix corrected-truth pool min is 80× worse.

The pool **as it stands** does not contain a Band-A or Band-B candidate. Re-ranking is an ordering operation; it cannot manufacture a candidate that the upstream stage didn't produce.

## Why this matters

### The "bug was helping m103" thesis is reinforced, not just suggested

[[feedback_bug_was_helping_m103]] (auto-memory created in m141) hypothesised that the bug's bright-peak displacement may have *coincidentally* aligned m103's noisy ranking with buggy-truth's ω on this seed. m142 rules out a softer alternative: that the buggy-truth wins were driven by the alignment-cost ranking finding the truth and the surrogate-rerank not being needed. Under correct truth, the surrogate rerank — the validated lever from m133/m134/m138 — also fails. The seed-6 buggy-truth "OK" was structurally a different basin.

### m103's phi-sweep needs revisiting, OR m103 needs replacing

The phi-sweep (Step 4 geo refinement, the multi-phi exploration around each ω candidate) was supposed to surface multiple q0 anchors per ω. On seed 6 under correct truth, it failed to find a truth-q0 anchor for the rank-9 truth-near ω. Two recovery directions:

1. **Patch m103's phi-sweep**: denser phi grid, larger angular footprint per ω, or re-evaluate q0 candidates by surrogate full-LC MSE during the phi-sweep instead of by alignment-cost. This is local and might salvage seed 6 specifically.
2. **Replace m103 entirely** with [[upstream-redesign-6dof-surrogate-de]] (Play 3): a 6-DOF DE over `(q0, ω)` driven by surrogate full-LC MSE from the start, no m103 substrate at all. This is the structural fix and addresses both the alignment-cost-anti-truth pathology AND the phi-sweep-misses-truth pathology.

The seed-6 evidence does not let us choose between (1) and (2) yet. A generality check on a second seed (seed 91, the textbook Play-1 win under buggy truth, or seed 47, the m138 surrogate-rerank rescue) is the natural next step.

### m115 K=1 rerun NOT triggered

Per the session-prompt gating: m115 K=1 was conditional on the rerank promoting the truth-near ω to top-K. It didn't. Skipping the K=1 rerun saves ~5 min wall and (more importantly) avoids spending time on a bridging task that was already known to be marginal even in the best case (16.86° in ω + 129° in q0 is well outside the m134 bridging radius).

## Numbers

| metric | value |
|---|---|
| Wall (rerank only) | 4.0 sec scoring + ~1 min SPICE/surrogate setup |
| n_candidates rescored | 26 (m141 geo_ckpt) |
| Pool min `surr_mse` | 3.9405 |
| Pool median `surr_mse` | 7.2481 |
| Pool max `surr_mse` | 18.1521 |
| Pool min `surr_bright_mse` | 4.0077 (at m103-rank 1) |
| Pool min `w0_ref_err` | 16.86° (at m103-rank 9, NEW surr_mse-rank 15) |
| Pool min `q0_ref_err` | 31.20° (at m103-rank 21, NEW surr_mse-rank 7) |
| ρ from pool-min `surr_mse` | √(3.94/0.05) ≈ 8.88 (Band D) |
| Acceptance bar | ρ < 4 (Band A∪B) — **MISSED** |

## Artefacts

- `notebooks/inversion/score_geo_surrogate.py` — the adapter (130 lines).
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/geo_surr_ckpt.npz` — `n`, `surr_mse`, `surr_bright_mse`, echoed `geo_costs`, `q0_ref_errs`, `w0_ref_errs`, `elapsed_s`.

## Out of scope here

1. **Generality check on seed 91 / seed 47** — running concurrently with the m142 write-up (background regen + m103 dispatch). If the seed-91 textbook win still solves under correct truth, m142's failure is seed-specific. If it doesn't, the buggy-truth ranking-luck explanation is universal and Play 3 is urgently needed.
2. **Play 3 (6-DOF surrogate DE upstream)** — out of scope for this session per the prompt. m142 establishes the empirical motivation; the implementation is a separate session.
3. **m103 phi-sweep diagnostic** — open question whether a denser / wider phi-sweep (or surrogate-driven q0 refinement) would surface the truth-q0 anchor for the rank-9 ω. Out of scope; testable cheaply on the saved `phi_sweeps_ckpt.npz` if the data is rich enough.
4. **Alternative re-rank metrics** — `surr_bright_mse` doesn't help on this seed either. Exotic metrics (per-epoch chi-square, Lomb-Scargle harmonic match, etc.) are not justified here; the structural issue is missing-truth-basin, not metric choice.

## Cross-references

- [[m141_seed6_postfix_pipeline]] — the m103 pool that m142 re-ranks.
- [[m140_post_fix_lc_delta]] — the LC delta that motivated the entire post-fix re-validation.
- [[m139_convention_bug_fix]] — the propagator fix two levels upstream.
- [[m134_pipeline_test_q0polish]] — establishes m115's bridging radius (~3–5° reliable, ~15° marginal in ω).
- [[m138_seed47_lombscargle_bracket]] — the surrogate-rerank rescue that DID work under buggy truth; m142 shows it does not generalise to seed 6 post-fix.
- [[surrogate-rerank]] — the lever; m142 is the first negative datum.
- [[upstream-redesign-6dof-surrogate-de]] — the next escalation if generality check confirms.
