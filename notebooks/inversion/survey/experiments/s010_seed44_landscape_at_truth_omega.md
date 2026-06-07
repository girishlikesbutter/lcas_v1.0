---
title: "s010 — seed-44 surrogate landscape probe at fixed truth-ω"
type: experiment
sources:
  - "results/s010/landscape.npz"
  - "results/s010/summary.json"
  - "results/s010_run.log"
  - "results/s009/runs.npz"
related:
  - "[[s002_surrogate_landscape_probe]]"
  - "[[s006_seed28_landscape_at_truth_omega]]"
  - "[[s009_cohort_basin_radius_probe]]"
  - "[[concepts/twin_degeneracy]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

## TL;DR

Seed 44's s009-LM-found competing basin (~100° from truth at near-truth ω,
final_mse 0.06 mag²) is **case (A): sub-Sobol-resolution narrow basin** at
2046-Sobol-Shoemake density. Truth is still the global argmin
(truth_mse 3.36e-4 mag², matches s001 cache); n_sobol_below_truth = 0;
n_sobol_below_lm_mse = 0; **no Sobol point lies within 10° of the LM-landing
q0** (sobol_min_geo_to_lm_landing = 10.06°). The competing basin is real
(LM-discoverable from inside-tube ICs along body-X axis) but invisible at
2046-density Sobol — same regime as seed 28's truth basin in s006. The
LM-landing q0 is **not at any clean rigid-body symmetry pole** (closest is
"180°y · truth" at 81.89°), so this is a genuine surrogate-MSE local
minimum, not a body-symmetry artefact. Wall 33.4 s.

## What

Single-seed surrogate-MSE landscape probe on seed 44 at fixed truth-ω,
2046 Sobol-Shoemake quaternions + truth + twin (2048 candidates total),
SOBOL_SEED = 42 (matches s002 / s006). Diagnostics:

- All s006 metrics: argmin location, n_sobol_below_K_truth for
  K ∈ {2, 5, 10, 100}, sobol-min-geo-to-truth.
- **NEW** — geo_to_lm_landing_deg for every candidate. Ring bins
  ([0,5), [5,10), [10,20), [20,30), [30,60), [60,180]°) around the s009
  T1_inside LM-landing q0; per-ring MSE percentiles + count of
  candidates with mse below LM-landing's 0.06 mag².
- Decision-case mapping (A / B / C) printed at end.
- Auxiliary post-hoc: distance from LM-landing to body-symmetry poles
  of truth (computed in run log, not in NPZ).

## How

Clone of s006 with SEED=44 and the LM-landing-ring diagnostic added.
Candidate set construction, scoring, output schema otherwise identical.
The s009 T1_inside row for seed 44 is loaded from `results/s009/runs.npz`
to give the exact LM-converged q0 and ω used as the "competing basin"
reference point.

Pseudocode:
```
q0_lm  ← s009.runs[(seed=44, tier=T1_inside)].q0_final_wxyz
q0_truth ← traj_load.load_truth(44).q0_wxyz
ω_truth  ← traj_load.load_truth(44).omega0_rad
q_grid   ← [q0_truth, q_twin(q0_truth), Sobol-Shoemake(2046)]
for q in q_grid (Pool(8)):
    k1b, k2b ← propagate_to_body_frame(q, ω_truth, ...)
    full_mse ← surrogate.full_lc_mse(predict(k1b, k2b), mag_hifi)
diagnostics:
    n_sobol_below_truth, K-thresholds, sobol-min-geo-to-truth,
    NEW: per-ring-around-q0_lm percentiles, n_below_lm_mse
decide A/B/C
```

## Result

```
truth_full_mse           = 3.3565e-04 mag²
twin_full_mse            = 1.0657e+00 mag²
lm_landing_final_mse     = 6.0051e-02 mag²  (s009 T1_inside)

argmin                   = truth (idx 0)  — argmin_mse = 3.3565e-04
sobol_min_geo_truth      = 16.35°
sobol_min_geo_lm_landing = 10.06°
n_sobol_below_truth      =     0 / 2046
n_sobol_below_lm_mse     =     0 / 2046

competing  K2    (mse < 2× truth_mse):     0 sobol
competing  K5    (mse < 5× truth_mse):     0 sobol
competing  K10   (mse < 10× truth_mse):    0 sobol
competing  K100  (mse < 100× truth_mse):   0 sobol

LM-landing ring probe — geodesic to s009 T1 q0_final (= 99.6° from truth):
  ring         n   mse_min    mse_p10    mse_p50    mse_p90    n<lm_mse
  [0, 5)       0  —          —          —          —          0
  [5, 10)      0  —          —          —          —          0
  [10, 20)     6  5.88e-01   1.51e+00   2.70e+00   3.36e+00   0
  [20, 30)     9  1.33e+00   1.57e+00   3.47e+00   4.32e+00   0
  [30, 60)   102  1.52e+00   3.49e+00   5.01e+00   6.71e+00   0
  [60, 180) 1929  1.97e-01   3.94e+00   5.86e+00   7.37e+00   0

DECISION: case (A) — sub-Sobol-resolution narrow basin
```

Symmetry-pole post-hoc (LM-landing q0 = [-0.04445, 0.51733, -0.83657, -0.17479]):
```
geo to truth                 99.60°
geo to twin (180°x·truth)   177.68°
geo to 180°y·truth           81.89°  ← closest, but not at a pole
geo to 180°z·truth          167.19°
geo to 90°x·truth           123.84°
geo to 90°y·truth           171.09°
geo to 90°z·truth           135.64°

|ω_LM - ω_truth| / |ω_truth| < 1%  (ω-recovery essentially perfect)
```

## Why this matters

1. **s002's "argmin = truth on 8/8" claim survives at cohort scale on
   seed 44.** Even on the s009 tight-tail seed where joint LM finds a
   competing basin, truth is still the argmin among 2048 Sobol+truth
   candidates at fixed truth-ω. The s009 phenomenon is not a refutation
   of s002 — it's an additional cost-surface feature that LM finds
   from specific ICs but Sobol(2046) doesn't see.

2. **The competing basin is real but tighter than s006's seed-28 truth
   basin.** Seed 28 truth basin: sobol_min_geo = 10.06° (s006). Seed 44
   competing basin: 0 Sobol points within 10°, full first-bin emptiness;
   closest Sobol bin (10-20°) sits at MSE p10 = 1.5 mag², ~25× above the
   LM-landing's 0.06 mag². The basin width is well below 10° in q0_geo.

3. **The competing basin is NOT a body-symmetry artefact.** No clean
   rigid-body pole of truth lies within 80° of the LM-landing. The
   closest (180°y · truth) is 81.89° away — far enough that the basin
   is a real surrogate-MSE feature, not a 180°-twin-style structural
   degeneracy. Twin recovery rate on seed 44 (s009: 0/2 ICs) is also
   consistent.

4. **Q4c-cohort architecture choice is unaffected.** The s009 cohort
   measurement (45% wide / 52% mid / 3% tight) was made with LM-polished
   ICs; this s010 result confirms that bare Sobol density is not what
   determines basin coverage — LM polish per Sobol candidate is doing
   the actual basin-finding. Seeds 44 and 76 join seed 28 as tight-tail
   seeds requiring multi-axis Sobol(q0) (not just body-X) to land an
   IC inside the truth basin.

5. **Mechanism (speculative, not measured):** the LM-landing has
   ω-recovery essentially perfect (LM-ω matches truth-ω to <1%), but
   q0 sits in an entirely different SO(3) region. This is consistent
   with a glint-multiplicity-degenerate q0 — two distinct orientations
   producing similar body-frame (k1, k2) trajectories under the same
   ω. Confirming this would require a hi-fi LC overlay (truth vs LM-
   landing) — left to follow-up using the future `lib/hifi_render.py`.

## Numbers (cached)

- truth_mse = 3.3565e-04 mag² (s001 cache match)
- twin_mse = 1.0657 mag² (twin not a competing basin)
- LM-landing mse = 6.0051e-02 mag² (~180× truth_mse, ~50× lower than
  Sobol p10 in 30-60° ring)
- sobol_min_geo_to_truth = 16.35° (truth basin sub-Sobol, like seed 28)
- sobol_min_geo_to_lm_landing = 10.06°
- n_sobol_below_truth = 0; n_sobol_below_K100_truth = 0
- decision = case (A)
- wall 33.4 s, Pool(8), BLAS=1

## Artefacts

- `experiments/s010_seed44_landscape_at_truth_omega.py` (script)
- `results/s010/landscape.npz` (q_grid, kind, full_mse, bright_mse,
  geo_to_truth_deg, geo_to_twin_deg, geo_to_truth_or_twin_deg,
  geo_to_lm_landing_deg, q0_lm_wxyz, ring_edges_deg)
- `results/s010/summary.json` (argmin, decision case, all diagnostics)
- `results/s010/landscape.png` (Sobol scatter MSE vs geo-to-truth)
- `results/s010/lm_landing_ring.png` (Sobol scatter MSE vs geo-to-LM-
  landing, with ring p10-p90 band overlay; truth & LM-landing reference
  lines)
- `results/s010_run.log` (gitignored — full stdout)

## Out of scope

- **Hi-fi ρ-band validation of the LM-landing state** — requires
  `lib/hifi_render.py`. Would test whether 0.06 mag² surrogate MSE
  translates to a ρ-band-A/B/C/D classification on hi-fi, and whether
  truth-LC vs LM-landing-LC overlay reveals which features differ.
- **Seed 76 Sobol probe** — would confirm whether the seeds-44/76
  pair is structurally similar (same regime: case A, sub-Sobol narrow
  competing basin, ~100° q0 separation, near-perfect ω recovery) or
  different. ~30 s wall; cheap follow-up.
- **Sub-10° basin geometry** — Sobol resolution is the natural floor
  here; tightening it (e.g. 16k Sobol = ~3°) would narrow the
  basin-width estimate but not change the case (A) decision.
- **Multi-axis q0-perturbation tier sweep on seed 44** — would tell us
  which IC axes LM-polish into truth vs into the LM-landing basin.
  Likely required for the Q4c-ii cohort pilot to handle seeds 44/76
  cleanly. ~5 min Pool(8) for 6 axes × 2 tiers.
- **ω-misspecified landscape on seed 44** — s003 sampled 6/41/91 only.
  s003-style sweep on seed 44 (24 ω perturbations × 512 candidates
  ≈ 5 min) would tell us whether the competing basin survives ω
  perturbation, or is a knife-edge feature at exactly truth-ω.

## Cross-references

- `s002` — surrogate landscape on 8 PA-stratified seeds (seed 44 NOT
  in the s002 sample). s010 is the seed-44 extension; argmin=truth
  property holds, consistent with s002's 8/8 pattern.
- `s006` — seed-28 single-seed Sobol probe. Same idiom, same Sobol seed,
  same SOBOL_SEED=42, same N=2046. s006 was case (A) for seed 28's truth
  basin; s010 is case (A) for seed 44's competing basin. The two
  experiments together establish that sub-Sobol-resolution narrow basins
  (both truth and competing) are a real cost-surface feature on at
  least 2 of 100 seeds (seeds 28, 44).
- `s009` — cohort basin-radius probe that opened this sub-question.
  s010 closes the case-A/B/C decision for seed 44 specifically.
- `concepts/twin_degeneracy.md` — twin convention used here. Twin is
  far from the LM-landing (177.68° from it) so this is not twin-related.
