---
title: "Dark-mag saturation of surrogate-residual cost"
type: concept
sources:
  - "raw/inversion_diagnostics/m120/seed_000/residuals.npz"
  - "raw/inversion_diagnostics/m120/seed_014/residuals.npz"
  - "raw/inversion_diagnostics/m120/seed_027/residuals.npz"
  - "raw/inversion_diagnostics/m120/seed_046/residuals.npz"
related:
  - "[[surrogate-model]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[m120_tumbling_competitors]]"
  - "[[m123_lbfgs_polish]]"
  - "[[m124_hifi_validate]]"
  - "[[m126_wrapped_pipeline]]"
  - "[[gradient-based-inversion]]"
  - "[[basin-of-attraction]]"
created: 2026-04-15
updated: 2026-04-16
confidence: high
---

# Dark-mag saturation

## One-line

When a candidate attitude trajectory keeps the body's bright facets pointed away from sun/observer, the surrogate predicts magnitudes in its **dark-mag ceiling** (~23–28 mag across seeds). Observed truth mags are 5–17 mag, so the residual saturates at ≈10 and the cost becomes *locally flat* in attitude parameters.

## Empirical evidence (2026-04-15, inline analysis of m120 data)

Analysis on 4 seeds (0, 14, 27, 46) of the `close_omega` bucket (N=2000 candidates each; ω within 2° / 5% of truth, q0 uniform on SO(3)). Results:

| seed | truth pred range | close_ω pred median | close_ω pred max | frac epochs > 20 mag |
|-----:|-----------------:|--------------------:|-----------------:|---------------------:|
| 0  | 5.4–17.1 | 23.3 | 27.1 | 91.6% |
| 14 | 5.5–17.4 | 23.1 | 27.4 | ~100% |
| 27 | 5.0–18.2 | 23.7 | 28.3 | 93.4% |
| 46 | 5.0–17.2 | 23.6 | 27.2 | 96.2% |

Uniform bucket (random ω, random q0) predicts >20 mag in **0%** of epochs across all seeds — the saturation is specific to the close_omega geometry, not random candidates.

## Mechanism

1. Ω = ω_true fixes the precession axis and rate. The body-frame sun direction `k1_body(t) = R_candidate(t)^T · k1_J2000(t)` traces a cone structured identically to truth's cone, just rigidly rotated by the fixed SO(3) offset between q0_candidate and q0_true.
2. Truth's ω is, by physical construction, *tuned* to sweep a cone that includes IS-901's bright facets (±X panels). That's why truth produces peaks.
3. A uniformly random q0 rigidly rotates the cone. For most random rotations the rotated cone misses the bright lobes → every epoch the candidate shows a dark face → surrogate output saturates near its training-ceiling (~27 mag, IS-901 eclipse/edge-on training examples).
4. Residual = pred (≈23 mag typical) − obs (≈13 mag typical) ≈ +10 mag.

## Why this matters

### Local flatness in cost landscape

Inside the dark-side basin the surrogate's gradient w.r.t. q0 is near zero (it predicts ~dark for all nearby q0). This is a **locally flat region of the cost landscape around the "accurate-ω / random-q0" manifold**. Implications:

- **Gradient-based inversion (see [[gradient-based-inversion]]).** Adam/L-BFGS from a dark-side start will not improve — gradient is noise. Any gradient-based pipeline needs an initialisation strategy that escapes the dark-mag region (e.g., require predicted median mag < ~18 at init, or use random-restart until a bright candidate is found). **Update 2026-04-16:** See [[m123_lbfgs_polish]]: L-BFGS-B polishes DE basins by ~0.05° ω-dir and ~0.2% ω-mag despite being in the saturated regime — the gradient on the plateau is SMALL but NONZERO in ω (q0 IS locally flat as claimed here, but ω is not). Earlier "locally flat" language was an overstatement; the plateau is shallowly sloped in ω and flat only in q0.
- **Update 2026-04-16 (from [[m124_hifi_validate]]):** the small-but-nonzero ω gradient that m123 found on the saturated plateau is **uncorrelated with the hi-fi gradient direction** on at least seed 27. All 3 of seed 27's far-q0 DE basins polish to a surrogate-cost-improvement state where hi-fi MSE is 12–16× WORSE (0.31 → 4.5–5.1 mag²). The plateau gradient is information about the surrogate's own modelling-error landscape, not about physical reality. Practical implication: NEVER trust a plateau-gradient polish without hi-fi validation. The "shallowly sloped plateau" is real but the slope direction is unphysical at far-from-truth attractors.
- **Update 2026-04-16 (from [[m126_wrapped_pipeline]]):** catastrophe predictor confirmed on 2 additional seeds. Seed 12 (upstream ω-dir 8°, single candidate, all 3 q0_err ≥ 10°): polish drives hi-fi 0.33 → 3.5–4.2 on all 3 basins. Seed 36 (upstream ω-dir 10.7°, single candidate, q0_err 14°, 170°, 171°): polish drives hi-fi 0.60 → 4.05–4.40 on 2/3 basins. Seed 46 from earlier [[m125_keep_better_inline]] data matches the same pattern (ω-dir 6.39°, single candidate, wrong-q0 cluster, polish 0.65 → 3.24). The **catastrophe predictor is now "upstream ω-dir ≥ 5° + single upstream + all q0 basins far from truth"** — three constraints, all checkable at DE time before polish runs. The wrapped pipeline (`keep_min`) catches all these catastrophes at the cost of one extra hi-fi eval per basin.
- **DE with surrogate MSE cost (see [[surrogate-de-search]]).** DE's mutation strategy still works because it's mutation-selection, not gradient-following — as long as *some* population members land in bright regions, they dominate and pull the population. The 10-start multi-start strategy used in [[m115_surrogate_pipeline]] empirically works; this is why.
- **Basin-width characterisation ([[m121_basin_width_metric]]).** When m121 varies perturbation scale about truth, the cost curve at large scales may plateau at the dark-mag saturation value rather than growing unboundedly. Report cost relative to this saturation value (cost ≈ 10 = "fully dark", cost < 10 = "partial visibility").

### Interpretation of m120 close_omega rank

The "close_omega best rank 8004/10004" result from [[m120_tumbling_competitors]] is **consistent with but not uniquely explained by** joint (q0, ω) coupling. The saturation mechanism says close_omega ranks worst because its cost is near-uniformly at the dark-mag ceiling while uniform candidates' costs are spread over a wider range (some lucky ones hit bright geometry). This is an **upper-bound** ranking — if the surrogate had a more accurate dark-side prediction (rather than saturation), close_omega's cost spread would widen and the rank might be different.

## What it does NOT imply

- The cost is still a correct discriminator *at truth and near truth* (residuals there are 0.04–0.11 mag, well below saturation). Confirmed by [[m120_tumbling_competitors]]'s truth rank 0 result.
- Uniform candidates are not "truly better" than close_omega — they just have more variance in cost due to chance bright-hits. Truth still beats both.
- Not a property of the ground-truth physics — it's a property of the **surrogate's extrapolation behaviour** at dark-side inputs. An exact hi-fi simulator would likely also predict very dark (maybe mag 30+) but the cost spread would depend on the hi-fi's dark-regime fidelity.

## Open questions

- Is the ~23 mag ceiling a hard limit of the surrogate MLP output, or the population mean of its training-set dark examples? A quick weights-inspection or saturation-probe (feed in known-dark `(k1, k2)` pairs) would answer this.
- How does the saturation interact with [[m121_basin_width_metric]] basin-width characterisation? At large q0-only perturbations, do cost values plateau at ~10 (saturation) or grow past it (beyond dark-mag)?
- Can we exploit this? A "light-detector" pre-filter (reject candidates whose predicted median mag > some threshold) would cheaply eliminate the entire dark-side manifold before cost evaluation.
