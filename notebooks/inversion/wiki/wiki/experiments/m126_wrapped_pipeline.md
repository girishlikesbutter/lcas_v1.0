---
title: "m126 — keep_better wrapper on 6 untested baseline seeds (0, 6, 12, 24, 33, 36)"
type: experiment
sources:
  - "raw/inversion_diagnostics/m126_wrapped/batch_summary.json"
  - "raw/inversion_diagnostics/m126_wrapped/seed_000/result.json"
  - "raw/inversion_diagnostics/m126_wrapped/seed_006/result.json"
  - "raw/inversion_diagnostics/m126_wrapped/seed_012/result.json"
  - "raw/inversion_diagnostics/m126_wrapped/seed_024/result.json"
  - "raw/inversion_diagnostics/m126_wrapped/seed_033/result.json"
  - "raw/inversion_diagnostics/m126_wrapped/seed_036/result.json"
  - "raw/inversion_diagnostics/m115_surrogate_pipeline/batch_summary.json"
related:
  - "[[m115_surrogate_pipeline]]"
  - "[[m123_lbfgs_polish]]"
  - "[[m124_hifi_validate]]"
  - "[[m125_keep_better_inline]]"
  - "[[gradient-based-inversion]]"
  - "[[basin-of-attraction]]"
  - "[[dark-mag-saturation]]"
  - "[[surrogate-model]]"
  - "[[multi-solution-philosophy]]"
  - "[[omega-sign-degeneracy]]"
created: 2026-04-16
updated: 2026-04-17
confidence: high
---

> ## ✅ 2026-04-17 — HI-FI NUMBERS CORRECTED (Option A re-run on 1-hour window)
>
> The original April-16 m126 run used the config 6-hour default window instead of m046's 1-hour truth window. `m126_wrapped_pipeline.py:347` patched 2026-04-17 (commit `5d5938f`); full re-run completed (commit `b906691`). New numbers in the tables below come from `m126_wrapped/batch_summary.json` dated 2026-04-17.
>
> **Corrected headline:** 5/6 of THIS page's seeds improved ≥10% (was 3/6); 18/18 basins helped (was 9/18); wall time 766 s (was 1469 s — correct window halves the forward-model cost). The [[gradient-based-inversion]] `#validated` status has been REINSTATED based on these numbers plus the [[m125_keep_better_inline]] cohort; see that branch page for the full 11-seed table.
>
> The "flipped-ω" narrative about seed 33 SURVIVES on correct window: seed 33 pre-polish hi-fi is 0.0796 (vs previously reported 0.0817; essentially unchanged) — the retrograde-ω basin is a real physical degeneracy, not a bug artefact.

# m126 — keep_better wrapper on 6 untested baseline seeds

## Hypothesis

On the 6 m115 baseline seeds that had never been run through the keep_better wrapper ({0, 6, 12, 24, 33, 36}), the wrapped pipeline `DE basins → L-BFGS polish on surrogate cost → hi-fi(before, after) → keep_min` improves the seed-level best hi-fi MSE by ≥10% on ≥3 of 6 seeds. Confirms → promote [[gradient-based-inversion]] branch to `#validated`.

## Method

Per seed:

1. Load m115 context: satellite, truth `(q0, ω)`, upstream ω candidates, 3 DE basins (q0, ω).
2. L-BFGS-B polish each basin on surrogate `mean_L1` cost (6-DOF parameter vector: quaternion tangent + ω tangent + fractional ω-magnitude). `ftol=1e-6, gtol=1e-3, maxiter=100, maxfun=500`.
3. Hi-fi-evaluate **both** the pre-polish and post-polish `(q0, ω)` per basin.
4. Per basin: `hifi_wrapped = min(hifi_before, hifi_after)`.
5. Seed-level best: `min(hifi_wrapped across basins)`.
6. Compare to plain [[m115_surrogate_pipeline]] `best_hifi_mse`.

All basin-level state (q0/ω before & after, surr_mse, q0_err, w_dir_err, w_mag_err_pct, iteration counts, wall times) saved to `polish_ckpt.npz` and `hifi_ckpt.npz` per seed.

## Upstream-candidate caveat (load-bearing)

Per inline audit of m115 upstream-ω sources:

| seed | upstream | n_ω | w_dir upstream | DE n_basins |
|-----:|:--------:|:---:|:--------------:|:-----------:|
| 0  | geo_ckpt  | 3 | 2.9°–3.1° | 3 |
| 6  | result_npz | 1 | 3.08° | 3 |
| 12 | result_npz | 1 | 8.0° | 3 |
| 24 | geo_ckpt  | 3 | 0.6°–4.9° | 3 |
| 33 | result_npz | 1 | 18.46° (axis-angle; equivalently 161.5° signed) | 3 |
| 36 | result_npz | 1 | 10.7° | 3 |

Seeds {6, 12, 33, 36} inherit a single ω-direction from m102 — all 3 DE basins share the same ω-dir. Polish can only refine ω-magnitude on a fixed ω-direction when the upstream ω is wrong. This is the structural limit for the 4 `result_npz` seeds.

## Per-seed predictions vs results

| seed | predicted | actual | verdict |
|-----:|-----------|:------:|:-------:|
| 0  | IMPROVES | 20.0% | CONFIRMED |
| 6  | IMPROVES (like seed 93) | 86.7% | CONFIRMED strong |
| 12 | MARGINAL | 0.0% (wrapper rejects catastrophe) | break-even as predicted |
| 24 | TIGHTENS | 44.7% | CONFIRMED |
| 33 | BREAK-EVEN (degeneracy) | 0.0% | CONFIRMED |
| 36 | BREAK-EVEN | 0.0% | CONFIRMED |

The above table preserves the April-16 strategist predictions verbatim. Post-fix outcomes: **5/6 seeds ≥10% improved, 0/6 regressed, 18/18 basins helped.** Seed 12's "wrapper rejects catastrophe" prediction inverted to 99.2% OK-class improvement. Seed 24's prediction ("TIGHTENS 44.7%") underestimated — actual 69%. Only seed 33's break-even prediction held.

## Headline results

| seed | m115_best | wrapped | improv% | class | basins helped/total |
|---:|---:|---:|---:|:---:|:---:|
| 0  | 0.1399 | **0.00367** | 97.4% | **OK** | 3/3 |
| 6  | 0.1299 | **0.00291** | 97.8% | **OK** | 3/3 |
| 12 | 0.3271 | **0.00265** | 99.2% | **OK** | 3/3 |
| 24 | 0.0443 | **0.01363** | 69.2% | PARTIAL | 3/3 |
| 33 | 0.0817 | 0.07965 | 2.5% | PARTIAL | 3/3 |
| 36 | 0.6018 | **0.39371** | 34.6% | FAIL | 3/3 |

**Aggregate (corrected):** 5/6 improved ≥10%, 0/6 regressed, **18/18 basins helped (no hurts)**. Wall: 766 s (43 s/basin ctx-build + 36 s/basin polish + 39 s/basin hi-fi — 2× faster than the wrong-window version).

### Contrast with wrong-window result

The original (April-16) claim was "3/6 improved ≥10%, 9/18 basins helped, 9/18 hurt (7 of 9 hurt from seeds 12+33+36)." The wrong window inflated `hifi_after` values because the 6-hour observed LC was being compared against states whose 3600 s propagation only matched 1 hour of it. In particular seeds 12 and 36 looked like "wrapper-rescues-catastrophe" on wrong window; on correct window they are clean successes (12) or still-FAIL-but-improved (36). Seed 33 is the one genuine wrapper-rescue: its flipped-ω attractor is a real physical degeneracy ([[omega-sign-degeneracy]]).

## Signed vs axis-angle w_dir convention (load-bearing flag)

The m126 `w_dir_err` column uses the **SIGNED** angle between ω vectors (0–180°). [[m115_surrogate_pipeline]]'s `w_dir_err` uses **axis-angle** (acos of absolute dot product, 0–90°). They differ by `180° − x` when ω is antipodal.

- Seed 33 shows `w_dir_err = 161.5°` in m126 vs `18.46°` in m115. Same physical ω, different display convention.
- The signed angle is honest: seed 33's DE basin has ω rotating BACKWARDS relative to truth (`(ω_cand · ω_true) / (|ω||ω|) < 0`). See [[omega-sign-degeneracy]].

Throughout this page, we use the SIGNED convention to expose the flipped-ω finding. Cross-referencing with [[m115_surrogate_pipeline]] requires the map `axis_angle = min(signed, 180° − signed)`.

## Per-basin polish motions (correct 1-hour window)

The per-seed mechanism prose from the April-16 wrong-window run has been retired; it misread polish behaviour (most seeds were described as "locked at DE attractor" when in fact the correct-window polish is genuinely global in both q0 and ω for seeds 0/6/12). Full per-basin table from `polish_ckpt.npz` + `hifi_ckpt.npz`:

| seed | basin | q0_err before → after | Δq0 | ω-dir before → after | ω-mag % before → after | hi-fi before → after |
|:---:|:---:|:---|:---:|:---|:---|:---|
| 0  | 0 | 176.30° → 176.30° | +0.00° | 2.90° → 2.90° | +0.23 → +0.17 | 0.140 → 0.090 |
| 0  | 1 |   6.68° →   0.54° | **−6.14°** | 2.90° → 0.02° | +0.23 → +0.01 | 0.140 → **0.004** |
| 0  | 2 | 177.51° → 177.51° | +0.00° | 2.90° → 2.90° | +0.23 → +0.18 | 0.302 → 0.231 |
| 6  | 0 | 176.76° → 179.59° | **+2.83°** | 3.08° → 0.20° | −0.15 → +0.01 | 0.130 → **0.003** |
| 6  | 1 |   5.10° →   0.40° | **−4.70°** | 3.08° → 0.11° | −0.15 → −0.03 | 0.130 → **0.003** |
| 6  | 2 | 176.89° → 179.76° | **+2.88°** | 3.08° → 0.25° | −0.15 → +0.01 | 0.432 → 0.333 |
| 12 | 0 |  10.24° →   0.14° | **−10.10°** | 8.01° → 0.03° | +0.21 → −0.01 | 0.327 → **0.003** |
| 12 | 1 | 170.76° → 179.79° | **+9.04°** | 8.01° → 0.03° | +0.21 → +0.01 | 0.501 → 0.240 |
| 12 | 2 | 172.79° → 179.14° | **+6.35°** | 8.01° → 0.53° | +0.21 → −0.04 | 0.650 → 0.440 |
| 24 | 0 | 179.53° → 179.85° | +0.32° | 1.93° → 1.35° | −0.01 → −0.00 | 0.044 → 0.014 |
| 24 | 1 |   2.16° →   2.07° | −0.09° | 1.93° → 1.45° | −0.01 → −0.01 | 0.044 → 0.015 |
| 24 | 2 | 179.98° → 179.89° | −0.09° | 0.64° → 0.52° | −0.06 → −0.03 | 0.080 → 0.078 |
| 33 | 0 |  98.53° →  98.53° | −0.00° | **161.54° → 161.54°** (retrograde) | −0.70 → −0.67 | 0.082 → 0.080 |
| 33 | 1 | 136.32° → 136.32° | +0.00° | **161.54° → 161.54°** (retrograde) | −0.70 → −0.67 | 0.082 → 0.080 |
| 33 | 2 | 178.63° → 178.63° | +0.00° | **161.54° → 161.54°** (retrograde) | −0.70 → −0.68 | 0.258 → 0.256 |
| 36 | 0 | 169.76° → 169.59° | −0.17° | 10.73° → 8.96° | +0.27 → +0.12 | 0.602 → 0.394 |
| 36 | 1 |  14.13° →  14.27° | +0.14° | 10.73° → 8.90° | +0.27 → +0.11 | 0.602 → 0.396 |
| 36 | 2 | 170.97° → 171.10° | +0.12° | 10.73° → 9.10° | +0.27 → +0.12 | 0.687 → 0.484 |

## What we learned (correct-window edition)

1. **The wrapper is production-safe and a genuine full-pipeline upgrade.** 33/33 basins helped on correct window; only seed 33's flipped-ω basins are "locked" (because they're at a separate valid attractor). The April-16 "wrapper catches catastrophes" framing was an artefact of the wrong window making polish look broken.
2. **Polish is genuinely global in q0 + ω for several seeds.** Seeds 0/6/12 each have at least one basin where polish moves q0 by multiple degrees (up to 10°) and ω-direction from ~3–8° down to sub-0.5°. The earlier "q0 is locked at the DE attractor" claim was wrong-window; in reality the correct-window surrogate gradient on the 1-hour observation pulls q0 meaningfully.
3. **Seed 33's flipped-ω attractor is still a real observational degeneracy** — its polish is the ONLY truly locked motion in the cohort, and the hi-fi (0.0796) is barely distinguishable from the original 0.0817. That makes [[omega-sign-degeneracy]] a physics claim rather than a numerical artefact.
4. **Seeds 36 and 46 remain FAIL** (hi-fi 0.39 and 0.14 respectively; 46 is in [[m125_keep_better_inline]]). Upstream ω-direction error ~10° is the bottleneck — polish tightens it to ~9° but that's still outside the <0.1° ω-dir basin. The `geo_ckpt` harvest lever ([[harvester-optimization]]) would provide multiple ω candidates for these seeds; still the natural next move.
5. **The 1469 s → 766 s wall-time halving** (same Pool(8), same 18 basins) reflects the 6× → 1× observation epoch count. The original wrong-window pipeline was spending 5/6 of its forward-model time on observations outside m046's truth window.

## Open questions

- **Population frequency of flipped-ω degeneracy.** Seed 33 is the first clear case. Is this the same phenomenon that put seed 46 at w_dir 6.39° with 3 wrong-q0 basins? Check signed-angle sign on 46's ω and the 100-seed cohort.
- **geo_ckpt harvest for {12, 33, 36, 46}.** Would adding multiple ω-direction candidates rescue these seeds? Estimated cost: 6 min surrogate-DE × 3 ω per seed × 4 seeds = ~1.2 hr.
- **Skip-polish-on-flagged-basin heuristic.** With the two predictors (signed w_dir > 90° OR upstream-shared ω + q0_err > 10° in all basins), we could skip polish on ~6/18 basins and keep the same improvement rate. Saves ~300 s/run.
