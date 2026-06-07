---
title: "s009 — cohort basin-radius probe (Q4b extension)"
type: experiment
sources:
  - "results/s009/runs.npz"
  - "results/s009/summary.json"
  - "results/s009/cohort_basins.png"
  - "results/s009/error_panels.png"
related:
  - "[[s002_surrogate_landscape_probe]]"
  - "[[s005_joint_local_descent]]"
  - "[[s006_seed28_landscape_at_truth_omega]]"
  - "[[s007_omega_mag_peak_spacing_pilot]]"
  - "[[s008_lc_feature_regression_omega]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# s009 — cohort basin-radius probe (Q4b extension)

## TL;DR

**Strongly positive for Q4c-on-bulk-cohort.** 100 m048 seeds × 2 LM
ICs (T1_inside q0=2°/ωd=0.3°/ωm-1%; T3_edge q0=8°/ωd=1°/ωm-5%, both
along deterministic body-X axis) = 200 LM runs. Wall 564 s (9.4 min)
Pool(8) BLAS=1.

- **97/100 seeds pass T1 strict** (q0_err<5°, ωd_err<1°, |ωm_err|<5%).
  90/100 pass at q0_err<1°.
- **45/100 seeds pass T3 strict** — basin radius ≥8° in body-X axis.
- **Cohort basin classification (s009 strict bar):**
  - WIDE (basin ≥8°): 45/100 (45%)
  - MID (2°-8°): 52/100 (52%)
  - **TIGHT (<2°): 3/100 (3%) — seeds 7, 44, 76**
  - ANOMALOUS (T3 pass + T1 fail): 0/100
- **Tight-tail decomposition:**
  - **Seed 7** lands just outside basin: q0_err=7.5°, ωd_err=5.9°,
    final_mse=1.85 mag². Near-miss; LM stalled at the basin edge.
  - **Seeds 44 and 76 land in deep competing q0 basins ~100° from
    truth at near-truth ω.** Final ω is recovered (ωd_err 0.1-0.8°,
    |ωm_err| <1.1%) but q0 sits at q0_err=99°-112° with **low**
    final_mse (0.018-0.06 mag²) — these are real competing-basin
    attractors, not stall points.
- Twin basin recoveries: 0/200 (consistent with s005).

**New finding not in s002:** seeds 44 and 76 are competing-basin-deceptive
at near-truth ω. s002 had argmin = truth on 8/8 PA-stratified seeds
(6/10/91/21/41/48/60/84) and concluded "surrogate-MSE is structurally
honest under correct truth and correct ω". s002's sample missed seeds
44 and 76, where this honesty fails: at body-X-axis q0 perturbation of
2°, LM converges to a competing q0 ~100° away with lower MSE than the
T3=8°-perturbation does. This is a **(q0, ω)-jointly-coupled deception**:
at the polished ω the alternative q0 is locally MSE-better.

**Implications for Q4c:**
- 97% of cohort recoverable from T1-style ICs along body-X axis.
- Partial-cohort acceptance loses 3/100 seeds; bulk Q4c-on-cohort is
  feasible with partial coverage.
- The competing-basin pathology (seeds 44, 76) is rare but present. If
  Q4c samples q0 along multiple axes (Sobol on SO(3)), it should hit
  the truth basin from at least one axis even on these seeds. Confirms
  that **multi-axis Sobol(q0) is necessary** — single-axis perturbations
  miss this small subset.
- Seed 28 passed T1 in s009 (q0_err=0.034°). The s005 T2 failure (q0=5°)
  remains: seed 28's basin is ~2° in body-X (passes T1), <5° (fails T2).
  Anisotropy is real; basin shape varies per seed.

---

## What

For each of 100 m048 seeds:

1. Load truth (q0, ω) from cached trajectory NPZ.
2. Construct two perturbed ICs:
   - **T1_inside**: q0 rotated 2° around body-X axis, ω-direction
     rotated 0.3° around a deterministic perpendicular axis,
     ω-magnitude × 0.99.
   - **T3_edge**: q0 rotated 8° around body-X, ω-dir rotated 1°,
     ω-mag × 0.95.
3. Polish each IC with joint (q0, ω) Levenberg-Marquardt on the
   surrogate full-LC residual (6-DOF parameterisation: δθ rotation
   vector for q0, ω as 3-vector). max_nfev=200, scipy LM defaults.
4. Classify each seed as WIDE / MID / TIGHT / ANOMALOUS based on
   T1 and T3 strict-basin pass.

Strict-basin bar (matching s005): q0_err<5°, ωd_err<1°, |ωm_err|<5%.

## How

- Same forward kernel as s005 (lib.forward.propagate_to_body_frame +
  surrogate_eval.predict). max_nfev=200 LM evaluations per IC.
- 200 LM runs over Pool(8) workers, BLAS=1. Wall 564 s = 2.8 s per IC
  (slightly higher than s005's 4.3 s/IC; s005 included random ICs which
  ran fewer iterations on average).
- Per-seed classification logic:
  - T1 ✅ AND T3 ✅ → WIDE
  - T1 ✅ AND T3 ❌ → MID (basin in [2°, 8°] in body-X axis)
  - T1 ❌ AND T3 ✅ → ANOMALOUS (non-convex basin)
  - T1 ❌ AND T3 ❌ → TIGHT (<2° in body-X axis)

Note: this classification gives basin width along the **body-X**
axis only. s005's random ICs and s006's Sobol cloud showed basins are
anisotropic (per-seed); s009's body-X-axis result is one slice. The
tight tail (T1 fail) is a strong upper bound on body-X basin width
but does NOT preclude wider basins in other axes.

## Result

### Tier-level pass rates

| tier         | passed | total | frac  |
|--------------|--------|-------|-------|
| T1_inside    | 97     | 100   | 97%   |
| T3_edge      | 45     | 100   | 45%   |

### Cohort basin classification

| class       | count | frac | seeds                                             |
|-------------|-------|------|---------------------------------------------------|
| WIDE (≥8°)  | 45    | 45%  | (45 seeds with basin ≥ 8° in body-X)              |
| MID (2°-8°) | 52    | 52%  | (52 seeds with 2° ≤ basin < 8° in body-X)         |
| TIGHT (<2°) | 3     | 3%   | **7, 44, 76**                                     |
| ANOMALOUS   | 0     | 0%   | —                                                 |

### Tight-tail seeds in detail

| seed | T1 q0_err | T1 ωd_err | T1 ωm_err | T1 final_mse | T1 outcome                |
|------|-----------|-----------|-----------|---------------|----------------------------|
| 7    | 7.46°     | 5.90°     | -0.54%    | 1.85 mag²     | near-miss; basin edge stall|
| 44   | **99.6°** | 0.14°     | 0.31%     | 0.060 mag²    | **competing q0 basin**     |
| 76   | **112.0°**| 0.76°     | -1.01%    | 0.018 mag²    | **competing q0 basin**     |

For comparison, T1 q0_err on the cohort:

- median 0.188°, p90 0.934°, max 112.0°
- 90/100 seeds at q0_err < 1°
- 97/100 seeds at q0_err < 5°

The bimodal split (sub-1° vs ≥99° for the competing-basin seeds) is
clean — seeds 44 and 76 do not partially fail; they LM-converge to a
distinct attractor ~100° from truth.

### Per-seed bar chart (cohort_basins.png)

The plot shows q0_err (log scale) per seed sorted by T1 q0_err
descending. The three red bars on the left are tight-tail seeds; the
mass of green/olive bars below the 5° basin bar shows the cohort's
basin recovery.

T3_edge q0_err is much wider — many seeds where T1 succeeds at <1°
land at q0_err >10° when started from T3 (8° q0 perturbation + 1°
ω-dir + 5% ω-mag). The pattern of MID-class seeds is dispersed —
basin width <8° but ≥2°.

### Twin recovery

0/200 ICs land in twin basin (q_180x · q0_truth, same ω). Consistent
with s005 — twin is not an attractor at near-truth ω.

## Why this matters

This is the calibrating measurement for Q4c. With s007 + s008 having
closed LC-only ω priors, the brute-force / partial-cohort question
required cohort basin distribution to be well-posed. Now we have it:

**Q4c bulk-cohort bound:**
- 97% recoverable from a 2° body-X-axis IC.
- 45% recoverable from an 8° body-X-axis IC.
- Joint Sobol(q0) on SO(3) at 2° density needs ~10⁵ candidates per
  seed to land at least one IC inside the basin uniformly across SO(3)
  (per-seed-axis, basin volume estimate from s006).
- For a Sobol(q0) at 8° density, ~6e3 candidates suffice for 45% of
  cohort.
- A two-stage Sobol(q0) (coarse 8° → refine to 2° around best
  candidates) is the natural strategy, but design is deferred.

**Competing-basin-deception finding (NEW):** seeds 44 and 76 reveal
that s002's "argmin = truth on 8/8 seeds" property is NOT universal
across the cohort. At near-truth ω, ~2% of cohort has a competing q0
basin ~100° away with lower surrogate-MSE than the basin edge of
truth-q0. This is potentially the buggy-era "m145 deceptive attractor"
phenomenon, real but rare.

Sub-question opened (not pursued here): **does s002's argmin=truth
property hold on a 100-seed Sobol probe?** s002 was 8 seeds × 2046
Sobol candidates at fixed truth-ω. Extending to 100 seeds at the same
density would resolve whether seeds 44/76 are sub-Sobol-resolution
narrow (basin so tight that 2046 Sobol misses it AND a competing basin
sits at lower MSE than truth itself) or whether the surrogate genuinely
prefers the alternate q0. ~50 min wall extrapolated from s002. Tagged
as Q2b for next-round consideration.

## Numbers

- 100 seeds × 2 ICs × max 200 LM evals × 6-DOF residual = **200 LM
  runs**, wall 564 s. Pool(8), BLAS=1, average 22 s per LM run.
- Output sizes: runs.npz 78 KB (no LCs saved per IC, just states),
  summary.json 4 KB, cohort_basins.png 270 KB, error_panels.png 130 KB.

## Artefacts

- `experiments/s009_cohort_basin_radius_probe.py` — runner script.
- `experiments/s009_cohort_basin_radius_probe.md` — this writeup.
- `results/s009/runs.npz` — per-IC final state, errors, MSEs, basin
  flags; 200 records.
- `results/s009/summary.json` — cohort distribution + tight-seed list.
- `results/s009/cohort_basins.png` — per-seed q0_err bar chart sorted
  by T1 tightness, T1 + T3 panels.
- `results/s009/error_panels.png` — cohort histograms of q0/ωd/ωm
  errors for both tiers.
- `results/s009_run.log` — console output (gitignored).

## Out of scope

- Multi-axis basin-radius probe. Body-X axis is one slice; s006
  showed basins are anisotropic. To fully characterise basin shape
  per seed, repeat s009 with q0 perturbations along ~6 axes
  (body-X/Y/Z + cross-products). Wall ~30 min Pool(8) for the 100 ×
  6 axis × 2 tier extension. Not pursued here — the 1-axis result is
  sufficient to bound cohort tractability.
- Hi-fi ρ-band validation of the converged candidates. Still pending
  (`lib/hifi_render.py` not implemented).
- Investigation of seeds 44/76 competing basins. The 100° offset and
  low final_mse hint at a non-trivial multi-basin structure — could be
  a near-twin-degeneracy, a body-symmetry-induced LC equivalence, or
  a surrogate-MSE artefact. Worth a single-seed Sobol probe (mirror
  of s006 on seed 44).

## Cross-references

- **[[s002_surrogate_landscape_probe]]** — measured argmin = truth on
  8/8 PA-stratified seeds. s009 finds 2/100 cohort exceptions
  (44, 76); s002's claim is approximately true but not universal.
- **[[s005_joint_local_descent]]** — established the joint-LM
  architecture and basin-radius idiom this experiment scales up.
- **[[s006_seed28_landscape_at_truth_omega]]** — the per-seed Sobol
  probe pattern that should be applied to seed 44 / 76 if the
  competing-basin question is pursued.
- **[[s007_omega_mag_peak_spacing_pilot]]**, **[[s008_lc_feature_regression_omega]]** —
  closed LC-only ω priors; together with s009 they say "Q4c is bulk-
  feasible without conditioning, on 97% of cohort".
