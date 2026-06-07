---
title: "ρ-band convention — the survey's primary classification metric"
type: concept
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# ρ-band convention

## Definition

```
ρ = √(hifi_MSE / 0.05²)   =   √(hifi_MSE) / 0.05
```

Where `hifi_MSE` is the mean squared error in mag² between the candidate's hi-fi LC and the (post-fix) observed truth LC. The denominator `0.05` is the assumed observational noise floor in mag.

## Bands

| Band | Range            | Meaning                                                           |
|------|------------------|-------------------------------------------------------------------|
| A    | ρ < 2            | Truth-grade fit, well below noise floor                           |
| B    | 2 ≤ ρ < 4        | Acceptable fit, within ~2× noise floor                            |
| C    | 4 ≤ ρ < 8        | Marginal — visually presentable LC, not a publishable inversion   |
| D    | ρ ≥ 8            | Failure — LCs do not agree                                        |

**Acceptance bar for the survey: ρ < 4 (Band A∪B).** This is the headline acceptance criterion for any inversion approach the survey eventually proposes.

## Why this metric is bug-fix-invariant

ρ is a function of `hifi_MSE`, which is a function of `(predicted_LC, observed_LC)`. The bug fix changed both LCs proportionally (the cohort-wide bug effect was uniform Band D), so the ρ formula itself is unchanged. Only the **numerator value** (hifi_MSE on a regenerated truth LC) changes. The bands and the acceptance bar transfer cleanly.

## Reporting convention

For every candidate state the survey scores, always report:
- `q0_err` in degrees (geodesic angle between candidate and truth quaternion)
- `ω_dir_err` in degrees (signed; see `omega_sign_degeneracy.md`)
- `ω_mag_err` as a percentage of truth `|ω|`
- `hifi_MSE` in mag²
- `ρ` and band

Do NOT collapse to a single "OK / PARTIAL / FAIL" verdict. The four-error tuple plus ρ-band is the contract.

## Surrogate-MSE has no ρ-band

The surrogate's full-LC MSE is NOT directly ρ-band classifiable, because the surrogate has its own intrinsic offset from hi-fi. Use ρ-band ONLY for hi-fi comparisons. For surrogate scoring, use raw MSE or a separate normalisation.

## When ρ < 2 is "valid" but not "truth"

See `observational_indistinguishability.md`. A candidate with ρ < 2 may be the truth basin OR a twin / degenerate attractor. ρ-band classifies LC fit quality; it does NOT decide between attractors. Use the (q0_err, ω_err) tuple in addition to ρ to disambiguate.

## Cross-references

- Auto-memory: `feedback_rho_band_convention.md`, `feedback_observational_indistinguishability.md`
- Survey concept: `observational_indistinguishability.md` (when ρ < 2 is valid regardless of truth)
