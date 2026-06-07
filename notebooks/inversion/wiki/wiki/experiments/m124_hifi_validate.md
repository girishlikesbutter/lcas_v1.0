---
title: "m124 — Hi-fi validation of m123 polished candidates"
type: experiment
sources:
  - "raw/inversion_diagnostics/m124/summary.json"
related:
  - "[[m123_lbfgs_polish]]"
  - "[[m122_hessian_curvature]]"
  - "[[m115_surrogate_pipeline]]"
  - "[[gradient-based-inversion]]"
  - "[[surrogate-model]]"
  - "[[dark-mag-saturation]]"
created: 2026-04-16
updated: 2026-04-17
confidence: high
---

> ## ✅ 2026-04-17 — RE-RUN ON CORRECT 1-HOUR WINDOW (Option A)
>
> `m124_hifi_validate.py:219` patched 2026-04-17 (commit `5d5938f`). Re-run chain m122 → m123 → m124 on correct window (commit `b906691`). Headline verdict **flips** from REFUTED → **PARTIAL**: 9/15 basins agree within ±30% (was 3/15 on wrong window; frac 0.60 in [0.25, 0.75] → PARTIAL).
>
> New per-basin hi-fi MSE on the 15 polished DE basins:
>
> | seed | basin | hi-fi MSE after polish |
> |:---:|:---:|---:|
> | 14 | 0 / 1 / 2 | (cand_001) / 0.0563 / 0.2221 |
> | 27 | 0 / 1 / 2 | 0.2183 / **0.0222** / 0.2870 |
> | 46 | 0 / 1 / 2 | 0.6285 / 0.6301 / 0.1416 |
> | 74 | 0 / 1 / 2 | **0.0109** / 0.3360 / 0.6593 |
> | 93 | 0 / 1 / 2 | 0.0430 / **0.0073** / 0.1839 |
>
> (cand_001 for seed 14 basin 0 is not in the tail excerpt above — see `m124/summary.json` for full numbers.) Truth-polish: all 5 seeds within 1σ noise floor at 0.0024 hi-fi (same as truth-ref 0.0024; polish does not damage truth).
>
> The original April-16 "seed 27 catastrophic 12–16× WORSENS" finding does NOT replicate on the correct window — seed 27 basin 1 now polishes from surrogate 1.0 to hi-fi 0.0222 (PARTIAL). The "dark-mag-saturation catastrophe" narrative needs revisiting: much of what looked like polish-makes-it-worse on the wrong window was actually polish-moves-toward-truth-but-we-were-scoring-against-a-different-window.
>
> State-error angles unchanged (window-independent). Surrogate ratios are on new correct-window surrogate evaluations. See `notebooks/inversion/DATA_INTEGRITY_BUG.md` (Bug 2) for audit.

# m124 — Hi-fi validation of m123 polished candidates

## Hypothesis

**Hi-fi MSE reduction factor tracks surrogate MSE reduction factor to within ±30% (in log-ratio) across the 12 DE-basin polished candidates of [[m123_lbfgs_polish]].**

- CONFIRMED ⇔ ≥75% within threshold → L-BFGS polish on the surrogate produces real hi-fi improvement → [[gradient-based-inversion]] viable as a stand-alone refiner.
- REFUTED ⇔ <25% within threshold → polish is surrogate-internal cosmetic, can hurt hi-fi.
- PARTIAL in between.

Secondary safety check: polishing from truth must not move any seed's hi-fi MSE significantly above the noise floor σ² = (0.05 mag)² = 0.0025.

## Method

- Seeds: {14, 27, 46, 74, 93}.
- Candidates: 17 (5 truth-starts + 12 DE-basin starts) — exact post-L-BFGS-B `(q0, ω)` outputs from [[m123_lbfgs_polish]].
- Hi-fi forward model: full ray-traced shadow + Ashikhmin-Shirley BRDF on 500-epoch trajectory; `mse = mean((mag_pred - mag_obs)^2)` against noisy observed magnitudes.
- Reference: hi-fi MSE evaluated at exact truth on each seed (`hifi_at_truth`) as the noise floor.
- For each DE-basin candidate, also re-evaluate hi-fi at the pre-polish (start) state to get `hifi_mse_before`, so the hi-fi reduction `hifi_ratio = hifi_before / hifi_after` can be compared to `surr_ratio = surr_cost_before / surr_cost_after`.
- Agreement metric: `ratio_agreement_log = |log10(surr_ratio) - log10(hifi_ratio)|`. `within_30pct ⇔ log_diff ≤ 0.3`.

## Truth-start safety check (5/5 PASS)

| seed | hifi_at_truth | hifi_after_truth_polish | Δ |
|:----:|:-------------:|:-----------------------:|:-:|
| 14 | 0.002402 | 0.002402 | < 1e-6 |
| 27 | 0.002402 | 0.002400 | -2.7e-6 |
| 46 | 0.002402 | 0.002515 | +1.1e-4 |
| 74 | 0.002402 | 0.002432 | +2.9e-5 |
| 93 | 0.002402 | 0.002416 | +1.4e-5 |

All 5 truth-polished hi-fi MSEs sit at the noise floor (~0.0024 ≈ σ²). **Polishing near truth is safe.** Worst displacement (seed 46) is +5% on MSE, still well below 1σ. This preserves [[gradient-based-inversion]]'s viability as a *local-refinement* tool.

## Per-basin DE-start results (12/12)

| seed | basin | surr_ratio | hifi_before | hifi_after | hifi_ratio | log_diff | within ±30% |
|:----:|:-----:|:----------:|:-----------:|:----------:|:----------:|:--------:|:-----------:|
| 14 | 0 | 10.94× | 0.161 | 0.0243 | 6.64× | 0.499 | ✗ |
| 14 | 1 | 12.88× | 0.161 | 0.0162 | 9.98× | 0.254 | ✓ |
| 14 | 2 | 4.59× | 0.290 | 0.245 | 1.18× | 1.354 | ✗ |
| 27 | 0 | 1.18× | 0.310 | **4.521** | **0.069×** | 2.840 | ✗ |
| 27 | 1 | 1.14× | 0.311 | **5.115** | **0.061×** | 2.933 | ✗ |
| 27 | 2 | 1.16× | 0.405 | **4.899** | **0.083×** | 2.640 | ✗ |
| 74 | 0 | 4.63× | 0.376 | 0.261 | 1.44× | 1.166 | ✗ |
| 74 | 1 | 4.66× | 0.376 | 0.253 | 1.48× | 1.144 | ✗ |
| 74 | 2 | 3.21× | 0.692 | 0.550 | 1.26× | 0.935 | ✗ |
| 93 | 0 | 7.48× | 0.063 | 0.0189 | 3.31× | 0.816 | ✗ |
| 93 | 1 | 7.53× | 0.063 | 0.0187 | 3.34× | 0.812 | ✗ |
| 93 | 2 | 6.35× | 0.233 | 0.0333 | 6.99× | -0.096 | ✓ |

**Aggregate: 2/12 within ±30% → fraction = 0.17 < 0.25 → REFUTED.**

## Classification

**REFUTED.** The surrogate's MSE-reduction factor is NOT a reliable proxy for hi-fi MSE-reduction factor on off-truth candidates. Three failure regimes:

1. **Catastrophic mis-direction (seed 27, 3/3 basins):** surrogate reports modest improvement (~1.15×); hi-fi WORSENS by 12–16× (0.31 → 4.5–5.1 mag²; mean |residual| jumps from 0.56 mag to ~2.1 mag). L-BFGS-B drives candidates AWAY from any hi-fi minimum.
2. **Surrogate over-statement (seeds 74, 93 mostly):** surrogate claims 3–7× improvement; hi-fi delivers 1.2–3.3×. Polish helps, but the surrogate gradient overstates the gain by 2–3× (log_diff 0.8–1.2).
3. **Rare agreement (2/12: seed 14 basin_1, seed 93 basin_2):** surrogate and hi-fi ratios within 30% in log space.

## What we learned

### Truth-polish is universally safe (4/5 lesson preserved from [[m123_lbfgs_polish]])

All 5 truth starts converge to hi-fi MSE within 5% of the noise floor σ². The surrogate IS faithful at truth, and L-BFGS-B near truth is a no-op refinement. This is the only regime where pure surrogate-gradient polish is unconditionally safe.

### Surrogate gradient is unreliable OFF truth — seed 27 mechanism

Seed 27's three DE basins are at q0 angles 100–180° from truth (they are NOT exact ±X twins; the q0 quaternions are e.g. `[0.890, -0.149, 0.398, 0.166]` vs truth `[0.262, 0.850, -0.201, 0.410]` — a different attractor entirely). At these distant attractors:

- The surrogate sees a smooth shallow ω-basin (the [[dark-mag-saturation]] plateau is shallowly sloped in ω, as discovered in [[m123_lbfgs_polish]]). L-BFGS happily descends it (`surr_cost_before ≈ 1.58 → surr_cost_after ≈ 1.34`).
- Hi-fi sees something completely different: post-polish ω lands in a regime where the hi-fi ray-traced shadow geometry produces predictions ~2 mag off observed (RMS), 4× worse than pre-polish (which already had ~0.56 mag RMS).
- **Mechanism (hypothesis):** the surrogate's ω-direction "gradient" on the saturated plateau encodes its OWN modelling-error landscape, not the physical light-curve gradient. The surrogate's MAE of 0.03 mag at truth ([[surrogate-model]]) hides systematic errors of ~2 mag at distant-attractor geometries that don't appear in the training distribution. When L-BFGS chases the surrogate gradient at off-truth points, it descends a *surrogate artifact*, not a physical minimum.
- The [[dark-mag-saturation]] page already noted that the plateau gradient is "small but nonzero in ω". m124 reveals that this small nonzero gradient is *uncorrelated with the hi-fi gradient direction* on at least seed 27.

### Seed 14 vs seed 27: why the asymmetry?

- Seed 14 basin_1 (the only basin with both surr & hi-fi >10× and within-30% agreement) starts with `q0 = [0.992, 0.028, 0.117, -0.048]` — the ±X-twin of truth (`q_180x · q0_true`). Twin attractors share most physics with truth and the surrogate's twin-region gradient remains physical.
- Seed 14 basin_0 is essentially the sign-flipped truth (`q ≡ -q`), pre-polish hi-fi already 0.16; both surr and hi-fi find ~7-10× gain, log_diff = 0.50 (close to threshold).
- Seed 14 basin_2 is a far-q0 attractor with shaky polish (1.2× hi-fi vs 4.6× surr, log_diff 1.35).
- Seed 27's attractors are all far from truth AND far from any twin axis; the surrogate has nothing well-trained nearby.

The pattern: **polish quality correlates with proximity of attractor to truth or to a known symmetry**. Pure-far-attractor basins (seed 27 and seed 14 basin_2) are where the surrogate misleads.

### Architecture verdict for [[gradient-based-inversion]]

L-BFGS-B polish is a *conditional* improvement, not a guaranteed one. Required wrapper:

1. DE on surrogate cost → enumerate q0 attractors (`m115` pattern).
2. L-BFGS-B polish each basin on surrogate cost.
3. **Hi-fi evaluate BOTH pre-polish and post-polish** for every basin.
4. Keep whichever has lower hi-fi MSE per basin.

This is essentially what [[m115_surrogate_pipeline]] already does (it uses hi-fi scoring to rank DE basins). The marginal contribution of adding L-BFGS polish, given that the wrapper must hi-fi-validate to catch seed-27-style catastrophes, is small — and the polish risks moving a candidate from a partial-fit basin into a worse one.

## Open questions

- **Why is seed 27 special?** Is it a population statistic (any seed whose DE basins are all >100° from truth fails the same way), or a specific surrogate-training-coverage gap? Could test by running m124 on more ATT_FAIL seeds (0, 58, 75) where DE basins were also far from truth.
- **Can a "best-of-pre-and-post" wrapper recover the seed-14 / seed-93 polish wins without seed 27's catastrophes?** Yes by construction (hi-fi keeps the better of two evals), but it adds 5 hi-fi evals per seed → ~5 min/seed extra cost. Worth it if it delivers ≥1 seed of improvement on the ATT_FAIL cohort.
- **Is the surrogate's off-truth modelling error correctable?** [[surrogate-model]] training was 5M samples with random `(k1, k2)`. A retrain weighted toward the under-represented "wrong-attitude" manifold might close the gap. Beyond current scope.
- **For [[surrogate-attitude-isoshell]]:** does the surrogate-isoshell cost suffer the same off-truth fragility? m120 only tested at-truth ranking; the off-truth cost-correctness is unknown.
