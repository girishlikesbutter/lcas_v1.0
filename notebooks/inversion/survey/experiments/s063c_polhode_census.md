---
title: "s063c — Cohort polhode census via closed-form Jacobi (120 m048 seeds)"
type: experiment
sources:
  - lib/jacobi_propagator.py
related:
  - project_jacobi_propagation_priority.md
  - project_polhode_prior.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (cohort-complete; closed-form is exact)
---

# TL;DR

The m048 cohort sits comfortably FAR from the separatrix. **Only 1 of 120 seeds has k² ≥ 0.99** (seed 7, at k² = 1.0000 to printed digits — genuine separatrix candidate). Median k² is **0.135** (small modulation off the dominant principal axis). Polhode period median is **1594 s**, range [360, 29315 s]; no seeds with T_pol < 60 s. Regime split 104 Case A (encloses I₁) / 16 Case B (encloses I₃) — confirms s062a. pol_diam (magnitude variation along polhode) is small for most seeds (median 6e-3 dps; tail extends to 0.5 dps near separatrix).

For s062b Path 2 (closed-form q(t)): **only seed 7 is at risk** for elliptic-integral conditioning issues, so a separatrix fallback is needed for exactly one seed and the main Path 2 implementation can target the bulk distribution.

# What

For each of 120 cohort seeds: load `omega0_rad`, eigendecompose inertia, build polhode info dict via `_build_omega_func`, extract:
- regime (Case A: 2T·I₂ > L²; Case B: 2T·I₂ < L²)
- k² (elliptic modulus squared; separatrix at k² = 1)
- |L|, 2T (conserved invariants)
- |τ_dot| (Jacobi rate)
- T_pol = 4·K(k²)/|τ_dot| (full polhode period)
- amplitudes (a₁, a₂, a₃)
- |ω| range along polhode → pol_diam_ω in dps (magnitude variation, NOT s055d Euclidean diameter)

# How

`experiments/s063c_polhode_census.py`. Pure linear algebra + scipy.special.ellipj/ellipk; no integration. Wall 0.03 s for 120 seeds.

# Result

Cohort statistics (n=120):

| Metric | min | p05 | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| k² | 0.0001 | 0.005 | **0.135** | 0.785 | **1.0000** |
| T_pol [s] | 360 | 415 | **1594** | 5814 | 29315 |
| |ω₀| [dps] | 0.106 | 0.176 | 0.742 | 1.398 | 1.482 |
| pol_diam_ω [dps] | 1e-5 | 9e-5 | **0.006** | 0.108 | 0.535 |

Regime split: **104 Case A / 16 Case B** (matches s062a exactly).

**Near-separatrix seeds**: only seed 7 at k² > 0.99 (it sits AT the separatrix, k² = 1.0000 to printed digits). No seeds have k² ∈ (0.99, 0.999).

**Short polhode-period (T_pol < 60 s)**: none. Minimum T_pol is 360 s on a high-|ω| Case B seed.

Plot: `results/s063c/cohort_polhode_census.png` — six-panel scatter/histogram of (k², T_pol, |ω|, pol_diam_ω) cross-products colored by regime.

# Why this matters

Three downstream implications:

1. **s062b risk assessment.** Path 2 (precession + nutation, needs `scipy.special.elliprj` for Π) becomes ill-conditioned as k² → 1 because K(k²) diverges and the elliptic integral of the third kind picks up a 1/(1-k²) singularity. Only **1 cohort seed** is affected. The main Path 2 implementation can target k² ∈ (0, 0.99) (119/120 seeds) and use a hybrid (DOP853 on q for this one seed) as a fallback. This is a small, well-defined exception, not a design constraint on the bulk of the architecture.

2. **s061 constant-ω budget.** The s061 cloud-threading reframe propagates each (q_a, ω) candidate as a fixed trajectory (constant ω) over the full LC. The "constant ω" approximation is only valid over a fraction of the polhode period. With median T_pol = 1594 s and LC duration ~3000–4000 s, constant-ω is *not* valid over a full LC for most cohort seeds — the polhode phase wraps. **The s062b path is non-optional for s061** if we want full-LC propagation; for short windows (s059j's 21-epoch local-window polish ≈ 200 s), constant-ω is OK.

3. **Polhode magnitude is small for bulk cohort.** pol_diam_ω median is 6e-3 dps on |ω| median 0.74 dps — a *0.8% relative* magnitude oscillation. Most cohort seeds are nearly pure-axis rotators with small polhode modulation. Only the high-k² tail (8 seeds with pol_diam > 0.1 dps) is what s053's "across-polhode basin width" effect is dominated by. This is consistent with s053's polhode-DIAMETER (Euclidean) being the load-bearing predictor rather than label D — Euclidean diameter is small for low-k² seeds even if the polhode topology is "interesting".

# Numbers

`results/s063c/summary.json` (cohort statistics + identified outliers)
`results/s063c/per_seed.json` (per-seed (m_k_sq, regime, T_pol, |L|, 2T, pol_diam_ω))

Notable seeds:
- **Separatrix candidate**: seed 7 (k² = 1.0000, T_pol = 29315 s)
- **Highest pol_diam_ω**: seed 7 (0.535 dps)
- **Shortest T_pol**: seed 0 (734 s, |ω| = 1.45 dps, k² = 0.21)
- **Slowest |ω₀|**: seed 32 (0.106 dps) — basin candidate via s053 pol_diam rule
- **Fastest |ω₀|**: seed 91 (1.482 dps)

# Out of scope

- pol_diam definition cross-check: I used max|ω(t)|-min|ω(t)| (magnitude variation). s055d/s053 used max pairwise Euclidean L2 distance of ω(t) trajectory in 3D. The two differ; for the bulk-cohort distribution shape they're correlated, but the absolute numbers are not directly comparable. The s055d definition is the one to use for filter retests (see s063d).
- LC-recoverability of (|L|, 2T) separately. s055a confirmed pol_diam (scalar) recoverable at 25% MAPE; |L| and 2T may have different recoverability properties and are worth their own regression study.

# Artefacts

- `experiments/s063c_polhode_census.py`
- `results/s063c/summary.json`
- `results/s063c/per_seed.json`
- `results/s063c/cohort_polhode_census.png`. **Saved: `notebooks/inversion/survey/results/s063c/cohort_polhode_census.png`**

# Cross-references

- `experiments/s062_jacobi.md` — same regime split (104 / 16)
- `project_polhode_prior.md` — s053 pol_diam basin-width prediction
- `experiments/s063a_polhode_bijectivity.md` — verified the parameterization used here
