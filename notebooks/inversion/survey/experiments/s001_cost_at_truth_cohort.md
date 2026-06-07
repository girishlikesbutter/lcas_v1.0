---
title: "s001 — cost-at-truth, cohort-scale (100 m048 seeds, post-fix)"
type: experiment
sources:
  - data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed*.npz (post-fix; commit ac1fdf4)
  - data/results/inversion_diagnostics/m048_trajectories/m048_trajectories.npz (master, for unique_normals)
  - ~/surrogate_model (v2 residual ensemble; bridge-independent)
related:
  - concepts/known_pathologies_to_revalidate.md (alignment-cost anti-truth, m135 buggy-era)
  - concepts/surrogate_model.md (intrinsic accuracy, m145 caveat)
  - concepts/rho_band.md
  - concepts/q_omega_coupling.md
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# s001 — cost-at-truth, cohort-scale

## TL;DR

Across all 100 m048 seeds under post-fix truth, four cost surfaces evaluated at `(q0_truth, ω_truth)`:

- **Surrogate full-LC MSE is universally applicable and small.** All 100 seeds: MSE ≤ 3.6×10⁻³ mag² (ρ-equivalent ≤ 1.20). Median 3.0×10⁻⁴ (ρ-equiv 0.35). The surrogate's noise floor at truth sits in Band A on every seed.
- **m103 alignment cost is structurally inapplicable on 24/100 seeds.** Those seeds have ≤1 spec peak (mag<9), so the m103 anchor/constraint construction yields zero non-anchor constraints. Includes seed 91 (the m145 "deceptive landscape" seed).
- **Where alignment cost IS defined (76 seeds): it is small at truth.** Per-constraint median 1.5×10⁻⁴, max 1.5×10⁻³ (seed 59, n=1 constraint). The buggy-era "argmin << truth by 4–22 OOM" rank claim is NOT decided by this experiment — it requires search-side data (survey Q3).
- **Lofi peak-match is good at truth, with 2 hard exceptions.** Median 100% match fraction; 55 seeds match all observed peaks. Seeds 3 (2/5) and 10 (0/4) fail at truth — the lofi prediction's peaks DO NOT line up with the noisy observed peaks even when fed truth attitude. These are intrinsic lofi-model failures, not solver issues.

The decisive cohort-scale finding: **surrogate-MSE is the only cost surface defined and small on all 100 seeds**. m103 alignment cost is restricted to ~3/4 of the cohort by construction. Lofi peak-match is universal-but-broken on a 2-seed minority.

## What

Survey question Q1: for each candidate cost surface, score `cost(q0_truth, ω_truth | observed_postfix)` across the full m048 cohort. Identify which surfaces place truth at-or-near a small value (necessary but insufficient condition for honest argmin), and which surfaces are structurally inapplicable on subsets of the cohort.

Four cost surfaces in scope:
1. **Surrogate full-LC MSE** — `MSE(surrogate.predict(k1_body_truth, k2_body_truth, obs_dist), mag_hifi)`.
2. **Surrogate bright-MSE** — same with mask `mag_hifi < 11`.
3. **m103 alignment cost** — copy-adapted from `notebooks/inversion/11_casadi_formulation/m103_hybrid.py:269–281`. At truth, reduces to a pure dot-product sum over non-anchor bright peaks.
4. **Lofi peak-match** — copy-adapted from m103_hybrid.py:359–376 (count of observed peaks matched by predicted peaks within ±3 epochs).

## How

Pure post-hoc scoring; no propagation, no search. Each cached `traj_seedXXX.npz` already stores truth-attitude body-frame quantities (`k1_body`, `k2_body`, `pab_body`, `mag_hifi`, `mag_lofi`). Verified machine-precision agreement `pab_body == R(q_truth) @ pab_J2000` on seed 6 before relying on it.

Workflow per seed:
1. Generate canonical observed LC: `mag_hifi + rng(seed=42).normal(0, 0.05, N)`. (Reproduces `lib/traj_source.canonical_observed_lc` from the parent project.)
2. Find peaks in `observed_lc` (distance=5, prominence=0.3). Spec subset: `observed_lc < 9.0`.
3. If ≥2 spec peaks: select anchor (savgol-smoothed brightest, with ±0.05-mag tiebreak on earlier epoch). Constraint epochs = spec peaks excluding anchor.
4. **Alignment cost at truth:** Σ_{ci ∈ constraints} 10.0·(1 − max(pab_body[ci]·normals[allowed]))², with `get_allowed_normals(observed_lc[ci])` from m103_hybrid.py:104.
5. **Surrogate MSE:** `surrogate.predict(k1_body, k2_body, obs_dist, sp=0°, ad=15°)`, score against both `mag_hifi` (primary, per PROGRESS.md) and `observed_lc` (secondary; inverter-perspective floor).
6. **Lofi peak-match:** `find_peaks(-mag_lofi, distance=3, prominence=0.2)`; count observed peaks matched within ±3 epochs.

All inputs cached → 5.6 s wall for 100 seeds, single-threaded.

Anti-import discipline: m103-side formulas were transcribed from `m103_hybrid.py` after reading source, NOT imported. Constants (`CONSTRAINT_WEIGHT=10.0`, `PEAK_WINDOW=3`, prominence/distance pairs) are reproduced verbatim. Smoke-tested on seed 6 vs the surrogate concept page's reference value — match.

## Result

### Population statistics (100 seeds)

| Cost surface | n_finite | min | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|
| Surrogate full-LC MSE [mag²] | 100 | 5.3e-5 | 2.0e-4 | 3.0e-4 | 4.6e-4 | 1.6e-3 | 3.6e-3 |
| Surrogate bright-MSE [mag²]  | 87  | 1.1e-7 | 1.7e-4 | 3.2e-4 | 6.2e-4 | 1.2e-3 | 3.6e-3 |
| Alignment cost (total, w=10) | 81  | 0.0    | 2.5e-4 | 1.0e-3 | 1.6e-3 | 3.3e-3 | 4.2e-3 |
| Alignment cost / constraint  | 76  | 1.6e-6 | 1.0e-4 | 1.5e-4 | 2.4e-4 | 4.4e-4 | 1.5e-3 |
| Lofi match-fraction          | 100 | 0.0    | 0.91   | 1.0    | 1.0    | 1.0    | 1.0    |
| n_spec_peaks (m103 anchors)  | 100 | 0      | 2      | 5.5    | 10     | 15     | 24    |
| n_constraints                | 100 | 0      | 1      | 4.5    | 9      | 14     | 23    |

### ρ-equivalent of surrogate MSE at truth

`ρ_surr = √(MSE)/0.05` (NOT a true ρ-band — surrogate has its own offset from hi-fi — but informative as a per-seed cost-surface noise floor):

| seed | surr full-MSE | ρ_surr | PA_med (deg) | n_spec |
|---|---|---|---|---|
| 10 | 3.59e-3 | 1.20 | 40.2 | 0  |
| 92 | 3.31e-3 | 1.15 | 72.9 | 14 |
|  6 | 2.92e-3 | 1.08 | 62.5 | 4  |
| 97 | 2.06e-3 | 0.91 | 79.9 | 10 |
|  8 | 1.88e-3 | 0.87 | 60.1 | 6  |

Top 5 surrogate-noisiest seeds at truth cluster at high PA (median 62°) — consistent with surrogate v2 having a mildly elevated floor in specular-dominated regimes. Still all Band A.

### Failure-by-construction subsets

- **24 seeds w/ alignment cost undefined:** 2, 3, 4, 5, 9, 10, 30, 32, 33, 41, 43, 48, 51, 52, 54, 57, 66, 70, 71, 81, 82, 86, 91, 99 (≤1 spec peak, no non-anchor constraints).
- **13 seeds w/ surrogate bright-MSE undefined:** mag_hifi never drops below 11 on these — uniformly dim throughout the observation window.
- **2 seeds w/ lofi peak-match < 50% at truth:** seed 3 (2/5; PA_med 35°), seed 10 (0/4; PA_med 40°). Truth attitude does NOT reproduce the observed peak pattern under lofi — the lofi MODEL is the failure mode here, not the search.
- **5 seeds w/ alignment cost = 0 exactly at truth:** PAB at every constraint epoch perfectly aligns with one of the satellite normals.

### Cross-references against buggy-era claims (re-validation)

| seed | buggy-era claim (m135/m141/m145) | post-fix s001 | what it tells us |
|---|---|---|---|
| 6   | "solved" (m126 wrapped, hi-fi 87% improvement) | surr_full=2.9e-3 (ρ_surr 1.08; 3rd noisiest), align/n=1.5e-4, lofi 20/21 | Surrogate floor is elevated here; m141's upstream-FAIL diagnosis is consistent. Truth is plausibly the surrogate's argmin but only at ~Band A noise floor. |
| 47  | alignment cost "anti-truth" (cost(geo_best)<<cost(truth)) | n_con=1, align=1.5e-5 (3rd lowest) | cost(truth) is essentially zero; rank claim still requires argmin probe (Q3). |
| 51  | alignment cost "anti-truth" | n_spec=0 → undefined | Pathology untestable here; m103 cost surface inapplicable. |
| 79  | alignment cost "anti-truth" | n_con=1, align=7.7e-5 | Small cost-at-truth; rank claim deferred. |
| 84  | alignment cost "anti-truth" | n_con=3, align=1.5e-3 (top-5 per-constraint highest) | Cost-at-truth is elevated relative to cohort; the 4-22 OOM gap to argmin is plausible IF argmin is near zero (Q3). |
| 89  | alignment cost "anti-truth" | n_con=2, align=2.1e-4 | Small cost-at-truth; rank claim deferred. |
| 91  | "DE q0-search fails; surrogate landscape's argmin at q0=135°" (m145) | n_spec=0 → align cost undefined; surr_full=2.0e-4 (Band A); lofi 16/16 | Seed 91's surrogate at TRUTH is fine. The m145 deceptive-landscape claim is about NON-truth surrogate-MSE attractors, not a truth-floor problem. Q2 (landscape probe) is the right next step. |

Note: none of these comparisons re-validate the buggy-era claims — most of the m135 claims are about ranks vs argmin, which s001 does not measure. They DO show that the cost-at-truth absolute value is small everywhere it is defined, so any "anti-truth" claim implies the argmin is essentially numerical zero (4–22 OOM below 1e-3 = 1e-25 to 1e-7) — testable in survey Q3.

## Why this matters

s001 narrows the candidate cost surfaces for the survey:

- **Surrogate full-LC MSE survives Q1.** It is defined and small on all 100 seeds. Truth is plausibly its argmin everywhere — the question of whether it actually IS (i.e., no deceptive distant attractor under correct truth, generalising m145 beyond seed 91) is exactly survey Q2 (landscape probe).
- **m103 alignment cost is restricted by construction.** 24% of the cohort cannot even be evaluated. Any inversion approach that depends on m103-style anchor/constraint structure inherits this restriction. This is not a bug-era artefact — it's a property of the cost surface's definition (no bright peaks ⇒ no constraints).
- **Lofi peak-match is universal-but-broken on a small minority.** 2/100 seeds are intrinsic failures. Useful as a co-cost but not as a sole criterion.

This narrows survey Q2 (landscape probe) to: probe surrogate full-LC MSE landscape on PA-stratified seeds (since surrogate noise floor mildly correlates with PA), at minimum including seed 91 (m145 deceptive-attractor) and seed 6 (m141 upstream-FAIL, elevated surrogate floor at truth).

Survey Q3 (failure-mode taxonomy) inherits the question of where each cost surface's argmin lives relative to the small-but-finite cost-at-truth values measured here.

## Numbers (selected, for cross-reference)

- Surrogate full-LC MSE at truth, full distribution: median 3.0×10⁻⁴ mag², p95 1.6×10⁻³, max 3.6×10⁻³. ρ-equiv: median 0.35, p95 0.79, max 1.20. **All Band A.**
- Alignment-cost-per-constraint at truth, where defined (76 seeds): median 1.5×10⁻⁴ (unitless, w=10), max 1.5×10⁻³, min 1.6×10⁻⁶.
- Lofi peak-match at truth: 55/100 seeds match all observed peaks, median fraction 1.00, p05 0.71.
- Spec-peak count distribution (m103 prerequisite): median 5.5 spec peaks per seed, 19 seeds with zero, 5 with one.
- Bright-window epoch availability (mag_hifi<11): 87/100 seeds have at least one bright epoch; 13 seeds are uniformly dim.

## Artefacts

- `experiments/s001_cost_at_truth_cohort.py` — script
- `results/s001/per_seed.csv` — one row per seed: seed, ω-magnitude, PA min/med/max, peak counts, all four cost values vs both targets (`mag_hifi` primary, `observed_lc` secondary)
- `results/s001/summary.json` — population statistics + outlier seeds
- `results/s001/cost_at_truth_distributions.png` — 4-panel histogram (the primary visual artefact)

## Out of scope

- **Where each cost surface's argmin lives relative to truth.** s001 measures cost(truth), NOT rank(truth, argmin). The buggy-era "alignment cost is anti-truth" rank claim requires search; it is survey Q3.
- **Surrogate landscape shape around truth.** Whether the surrogate's argmin is at truth-q0 (Band-A floor here would be the floor everywhere) or whether m145's deceptive q0=135° attractor generalises across seeds is survey Q2.
- **Twin / ω-sign degeneracies.** Cost-at-truth is also small at the twin if BRDF asymmetry is small (per `concepts/twin_degeneracy.md`). s001 does NOT score the twin state. Worth a side-quest probe.
- **Lofi-failure root cause on seeds 3 and 10.** Why the lofi model's peak pattern does not match the observed peak pattern at truth is not diagnosed here.

## Cross-references

- `concepts/known_pathologies_to_revalidate.md` items 1, 7, 8 (alignment cost anti-truth, "solved" seeds 6 and 91)
- `concepts/surrogate_model.md` (validates the per-seed Band-A floor claim cohort-wide)
- `concepts/rho_band.md` (surrogate-MSE has no true ρ-band — used "ρ-equivalent" for cohort-comparable scaling only)
- Frozen-reference experiments superseded for cohort-scale on this question:
  - parent wiki `m135_cost_at_truth_probe.md` (single-seed, buggy-era)
  - parent wiki `m141_seed6_postfix.md` (single-seed, post-fix on seed 6 only)
  - parent wiki `m145_seed91_postfix_full_patch_chain.md` (single-seed, post-fix on seed 91 only)
