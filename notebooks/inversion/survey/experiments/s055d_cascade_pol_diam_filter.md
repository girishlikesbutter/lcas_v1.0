---
title: "s055d — Cascade-pool pol_diam filter on cached s049 seed-14 pool"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s055d_cascade_pol_diam_filter.py
  - notebooks/inversion/survey/results/s049_cascade_seed14/cascade.npz
  - notebooks/inversion/survey/results/s055a_pol_diam_lc_regression/regression.npz
  - notebooks/inversion/survey/results/s053_cohort_polhode_survey/cohort.npz
related:
  - s055a — pol_diam regression (✓ COARSE_USEFUL @ 25% holdout MAPE)
  - s055b — L̂ direction regression (✗ WEAK @ 74° holdout)
  - s055c — |cos(L̂,PAB)| + polhode-binary class (✗ both WEAK)
  - s049 — cascade architecture works on seed 14 (truth-q_a survives)
  - s050a — cascade survivor pool 0.59× anti-enrichment, Phase-2 gated out
  - s050b/c/d/e — re-aggregate cascade pool: closed negative across mag-agreement, multipair, q-space, peak-stationarity formulations
created: 2026-05-07
updated: 2026-05-07
confidence: high (negative result, two independent scenarios both close)
status: WEAK — best enrichment 1.24× (K3 ∩ pol_diam at 25% threshold), well below 3× operational gate. Pol_diam is NOT orthogonal to mag-agreement in the cascade pool. With s055a + s055b + s055c + s055d collectively, the polhode prior is operational ONLY on |ω| basin width via adaptive bracket — not on cascade-pool truth-discrimination, not on ω-direction priors.
---

## TL;DR

**Pol_diam disagreement is not a useful filter on the s049 cascade pool for seed 14.** Best enrichment of truth-q_a hypotheses = 1.24× at the tightest 25% threshold (K=3 ∩ pol_diam scenario); pol_diam-only on the raw 141k pool is essentially flat at every threshold (0.87-1.02×). The histogram shows truth-q_a survivors (n=30, median pol_diam 1.37 dps) and the full survivor pool (n=39591, median 1.79 dps) overlap substantially — both ranges span [~0.4, ~3.5] dps. **Mechanism**: the cascade derives ω from anchor-pair finite-diff `(q_b ⊗ q_a^{-1})/Δt`, and the resulting ω's have pol_diams determined more by the (q_a, q_b) geometry of the anchor pair than by whether q_a is truth-adjacent. Random q_a hypotheses with appropriate q_b's reproduce truth-like pol_diams as easily as truth-q_a ones. **Decision: WEAK.** This closes the last remaining "extract from existing cascade pool" hypothesis, joining s050b/c/d/e in the closed-negative pile.

## What

After s055a confirmed pol_diam IS LC-recoverable (~25% holdout MAPE) and s055b/c found ω-direction priors are NOT recoverable at the s008+PAB feature granularity, this experiment tested the remaining "polhode prior is useful" claim: **does pol_diam disagreement act as a non-trivial filter on the s049 cascade survivor pool?** The hypothesis: each (q_a, ω) hypothesis implies a polhode of computable size (integrate Euler's equations over 1 hour from ω in body frame, measure max pairwise L2 distance). Reject hypotheses whose implied pol_diam disagrees with the LC-predicted pol_diam from s055a by more than X%. Orthogonal to mag-agreement (which s050b-e closed); should attack the s050a 0.59× anti-enrichment in the 35k-survivor pool.

## How

**Inputs** (all cached, no rendering):
- `cascade.npz` from s049 on seed 14: 141,706 raw (q_a, ω) hypotheses with delta_mag at 5 validation epochs.
- s055a's LOO predictions for seed 14 (since seed 14 ∈ cohort, the LOO prediction simulates the hold-out): rf-log10 = 1.819 dps (truth = 1.969 dps; 7.7% LC-pred error vs truth).
- Inertia tensor I_body from `lib.hifi_render._build_model()`: eigvals 7749 / 37985 / 38306.

**Pipeline**:
1. Identify truth-q_a mask via geodesic q-distance < 0.5° to truth_q_a → 153 hypotheses in raw 141k.
2. Apply tol=0.10 / K=3 mag-agreement filter on `delta_mag` → **39,591 survivors** (vs s049's 35,835; small `<` vs `<=` discrepancy that doesn't change the qualitative finding); 30 truth-q_a survivors among them (vs s050a's 23).
3. For each of the 141k raw hypotheses, integrate Euler's equations on body-frame ω from `om_kept[i]` (rad/s, NOT dps despite the variable name — verified via `om_truth_at_t0` cross-check) for 1 hour, sample 500 points, compute pol_diam in dps as max pairwise L2 distance. scipy `solve_ivp` DOP853, rtol=1e-8 atol=1e-11. **Pool(8), wall ~400s for 141k integrations (2.84 ms/hyp).**
4. Test pol_diam filter at thresholds {10, 15, 25, 50, 100, 150, 200, 300, 500, 1000}% of LC prediction in two scenarios:
   - **(A) pol_diam-only on raw 141k** — orthogonality test (does pol_diam alone discriminate?).
   - **(B) K=3 ∩ pol_diam** — combined test (does pol_diam add to K=3?).
5. Compute truth-q_a enrichment factor: `(truth_after/total_after) / (truth_before/total_before)`.

## Result

### Both scenarios negative

| Threshold | (A) raw 141k pool | enrichment | (B) K3 ∩ pol_diam | enrichment |
|-----------|-------------------|------------|-------------------|------------|
| 10%       | 17,120 / 141,706 (12.1%); 16/153 truth | 0.87× | 4,472 / 39,591; 1/30 truth | **0.30× (anti)** |
| 15%       | 25,765 (18.2%); 25/153 truth | 0.90× | 6,911; 4/30 truth | 0.76× |
| **25%**   | 42,821 (30.2%); 44/153 truth | 0.95× | **11,661; 11/30 truth** | **1.24×** ← best |
| 50%       | 82,328 (58.1%); 84/153 truth | 0.94× | 23,093; 17/30 truth | 0.97× |
| 100%      | 136,142 (96.1%); 148/153 truth | 1.01× | 38,928; 30/30 truth | 1.02× |
| 200%+     | ~141,706 (100%); 153/153 truth | 1.00× | ~39,591; 30/30 truth | 1.00× |

- **Pol_diam-only filter on raw 141k (scenario A)**: enrichment ranges 0.87 to 1.02 across all thresholds — **flat, no signal**.
- **K=3 ∩ pol_diam (scenario B)**: best enrichment 1.24× at 25% threshold (11/30 truth-q_a kept among 11,661 of 39,591 survivors). At tighter thresholds (10-15%), filter is anti-enriching (0.30×, 0.76×) — pol_diam pushes truth-q_a OUT.
- **Reference**: s050a found K=3 alone gives 0.59× anti-enrichment (this run: 0.70× with slightly different filter; same qualitative).

### Histogram tells the mechanism

The pol_diam histogram on K=3 survivors (`pol_diam_hist.png`) shows:
- All survivors (n=39,591): density spread across [~0.1, ~4.0] dps, median 1.79 (close to LC pred 1.82, truth 1.97).
- Truth-q_a survivors (n=30): density spread across [~0.4, ~3.5] dps, median 1.37 — actually skewed *below* LC pred and truth.
- The two distributions overlap heavily in the [1, 3] dps band.

**Mechanism**: cascade-derived ω = `log(q_b * q_a^{-1}) / Δt` is dominated by the anchor-pair geometry (q_a, q_b). For truth-q_a hypotheses, ω is "near-truth" only with the noise level of finite-diff (~17% per s049 stage4) — and this is enough perturbation in body-frame ω to push pol_diam off truth by 30-50%. For random q_a hypotheses with random q_b's, ω can also reproduce a truth-like pol_diam as long as |L| and 2T (which determine pol_diam together with I) happen to be in the right range. **Pol_diam under-discriminates because the cascade's noise level is large enough to scramble it on truth-q_a, AND random hypotheses can reproduce truth-like pol_diam by chance.**

### Truth-q_a is below LC prediction in median pol_diam

A subtle finding: median pol_diam of truth-q_a survivors (1.37 dps) is *below* both LC pred (1.82) and truth (1.97). The 30 truth-q_a survivors have q_a = truth_q_a but ω derived from various q_b — these ω's are spread around truth ω with cascade noise. The asymmetry (median below) suggests cascade noise systematically reduces the implied polhode amplitude for truth-q_a hypotheses, possibly because the q_b survivors at validation epochs are slightly inside the polhode's body-frame trace rather than at extrema.

## Why this matters

**Closes the polhode prior as a cascade-pool filter cleanly.** Combined with:
- s050b: validation-step stress (mag-agreement re-scoring, 30+ filter variants) — closed.
- s050c: Hough voting / multipair concentration — closed.
- s050d: distinct-t_k consensus — closed.
- s050e: q-space agreement at validation — closed.
- **s055d (this experiment): pol_diam orthogonal filter — closed.**

**The s049 cascade pool is fundamentally uniform-thinning under every LC-derived re-aggregation tested. The information bottleneck is the cascade's ω-noise level (17% per s049), not the choice of post-cascade filter.** s055d is the natural completion of the s050 series with the polhode prior added.

**Implications for the polhode-prior architecture:**

| Application | Status |
|---|---|
| Adaptive bracket on |ω| via pol_diam | ✓ s055a OPERATIONAL |
| ω-direction grid pruning via L̂ or topology | ✗ s055b/c CLOSED |
| Cascade-pool truth-discrimination via pol_diam | ✗ s055d CLOSED (this) |

**The polhode framework is real and useful, but its operational scope is narrowly the |ω| basin width.** It does NOT rescue the cascade architecture as currently formulated.

**Forward path (sharpened by this result):**

1. **s055e** — oracle adaptive-bracket pilot on holdout 100..119 with pol_diam from s055a. Upper-bound the polhode-prior architecture given pol_diam estimator quality. ~6-10 h Pool(4). **Highest-leverage remaining experiment.**
2. **Cascade architecture rethink (Phase 3 ω-noise reduction)**: the cascade pool as currently constructed has 17% ω-noise that washes out structure under EVERY LC-derived filter (mag-agreement, multipair, q-space, peak-stationarity, pol_diam). Either (a) reduce ω-noise via multi-pair averaging (sqrt-N reduction with N anchor pairs) or peak-stationarity (`∇B(q_peak) · ω = 0` at LC extrema, geometric not statistical), or (b) abandon the cascade in favour of the s011-class joint q × ω architecture under pol_diam-adaptive bracket. The latter is **what s055e tests**.
3. **Closed-now-clearly**: the cascade pool, as constructed, cannot be enriched by *any* LC-derived re-aggregation. Don't propose more "filter X on the cached pool" experiments.

## Numbers

- seed = 14, |ω|_truth = 1.241 dps, pol_diam_truth = 1.969 dps
- LC-pred for seed 14 (s055a rf-log10 LOO) = 1.819 dps; LC error vs truth = 7.7%
- Filter: tol_mag = 0.10, K_required = 3 → 39,591 survivors (vs s049 reported 35,835)
- truth_q_a tolerance = 0.5° geodesic; 153 truth-q_a in raw 141k, 30 in K=3 survivor pool
- Wall: 403s (Pool(8), 141k integrations × 2.84ms each)
- **Best enrichment**: 1.24× (K3 ∩ pol_diam @ 25% threshold)
- Pol_diam-only enrichment range across thresholds: 0.87 to 1.02
- Decision label: **WEAK**
- Reference s050a: K=3 alone enrichment 0.59× (this run: 0.70×)

## Artefacts

- `notebooks/inversion/survey/experiments/s055d_cascade_pol_diam_filter.py`
- `notebooks/inversion/survey/results/s055d_cascade_pol_diam_filter/`:
  - `pool.npz` — survivor masks, qA_dist_deg, om_kept_survivors, truth_q_a, etc.
  - `pol_diam.npz` — implied pol_diam (raw 141k + K=3 survivors), LC pred, truth, wall_s
  - `enrichment.json` — full filter sweep results, scenarios A & B, decision
  - `pol_diam_hist.png` — distribution overlap of all-survivors vs truth-q_a-survivors with LC pred + truth markers
  - `enrichment_curve.png` — 2-scenario enrichment + total counts vs threshold

## Out of scope

- Filtering on the K=2 pool (46,546 survivors, less restrictive). Strictly looser filter; if K=3 ∩ pol_diam fails, K=2 ∩ pol_diam is also unlikely to help. Not tested.
- Pol_diam filter using truth pol_diam (1.97) instead of LC pred (1.82). Would represent oracle-mode upper bound — may marginally tighten enrichment but doesn't change architecture conclusion.
- Joint pol_diam + |L| or pol_diam + D filter — fishing for any non-trivial discriminator. Closed by the same mechanism: cascade ω-noise scrambles all polhode invariants.
- Phase 3 ω-noise reduction (multi-pair averaging, peak-stationarity at LC extrema). Real candidate for follow-up but separate architecture.

## Cross-references

- s055a — `s055a_pol_diam_lc_regression.{py,md}`: pol_diam recoverable @ 25% MAPE; basis for adaptive bracket.
- s055b — `s055b_lhat_lc_regression.{py,md}`: L̂ direction WEAK.
- s055c — `s055c_aux_priors.{py,md}`: |cos(L̂,PAB)| + polhode-binary WEAK.
- s050a-e — cascade-pool re-aggregation closed across mag-agreement, multipair, q-space, peak-stationarity formulations.
- s049 — `s049_cascade_seed14.{py,md}`: cascade architecture (works as seed-generator on seed 14).
- s053 — `s053_cohort_polhode_survey.{py,md}`: pol_diam ρ=−0.94 vs basin width.
- `feedback_first_principles_first.md` + `feedback_stop_cost_shape_engineering.md`: don't keep proposing post-hoc filters when the underlying signal is the bottleneck.
