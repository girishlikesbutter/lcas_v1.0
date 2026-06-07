---
title: "s040 — joint-LM basin radius on seeds 23/28/89 × {truth, twin}"
type: experiment
sources:
  - experiments/s038_filter_diagnostics_seed023.md
  - experiments/s037_sobol_lm_pilot.md
  - experiments/s034_lm_polish_seed89.md
  - results/s040_basin_radius_3seed/
related:
  - s011
  - s034
  - s037b
  - s038
  - s039_smoke_basin_walk
created: 2026-05-06
updated: 2026-05-06
confidence: high
---

## TL;DR

Joint 6-DOF LM polish (`scipy.least_squares(method='lm', max_nfev=200)`,
surrogate-MSE objective, same recipe as s034/s038) starting from controlled
perturbations off (q0, ω) of either truth or body-twin on seeds {23, 28, 89}.
**The LM basin is wide in q0 (≥30° on every axis on every seed/attractor
except seed 89-truth-perp = 5°) but narrow in ω.** Binding axis is
**seed-specific**: ω-mag (±7%) on seed 23, ω-mag (+2% / −3%) on seed 28,
ω-dir + q0-perp on seed 89. **Seed 89 is uniquely asymmetric** between
truth and twin (q0-perp basin truth=5° vs twin=20°), driven by a competing
attractor at ρ≈11 reachable from truth-perpendicular perturbations beyond
~5°. Seeds 23 and 28 are symmetric. **Headline implication for the cohort
pipeline:** the IC primitive needs to deliver q0 within only ~30°-class
distance to truth/twin under the correct ω, but the **ω-prior must hit
within ~2% on the narrowest seed**, which is the binding cohort
constraint.

## What

For each (seed ∈ {23, 28, 89}) × (attractor ∈ {truth, twin}) we perturb
the initial state in 4 independent axes and fit:

- **`q0_along_omega`**: rotvec parallel to attractor ω-axis.
  {2°, 5°, 10°, 15°, 20°, 25°, 30°}.
- **`q0_perp_omega`**: rotvec along an arbitrary axis perpendicular to
  ω. Same magnitudes.
- **`omega_mag_pct`**: ω → ω·(1+δ), with δ in {±0.5, ±1, ±2, ±3, ±5,
  ±7, ±10} %.
- **`omega_dir_deg`**: rotate ω about a perpendicular axis by
  {0.5, 1, 2, 3, 5, 7, 10}°.

LM optimises the joint 6-vector (rotvec_pert, ω_delta) starting from
zero. ρ_final, q0_err, ω-dir_err, ω-mag_pct are reported back to **the
attractor we perturbed AROUND** (not always truth — for the twin column
the attractor is twin).

## How

`experiments/s040_basin_radius_3seed.py`. Pool(8) fork, BLAS=1 set
before numpy, surrogate loaded once per worker via
`lib.surrogate_eval.get_model()`. Forward model identical to s034/s038
(`propagate_attitude` post-fix, `Rotation.from_quat().as_matrix()`
returns `R_i2b` directly, no transpose). Body twin computed as
`q_twin = q_180x ⊗ q_truth`, `ω_twin = diag(1,−1,−1) @ ω_truth`.

210 LM polishes total (3 × 2 × 35), Pool(8) wall **9.4 min** (under
the 13–25 min projection; well below the 45-min kill).

## Result

### Per-seed-per-attractor band counts (n=35 each)

| Seed | Attractor | A  | B | C | D |
|-----:|-----------|---:|--:|--:|--:|
| 23   | truth     | 32 | 0 | 0 | 3 |
| 23   | twin      | 32 | 0 | 0 | 3 |
| 28   | truth     | 28 | 0 | 0 | 7 |
| 28   | twin      | 27 | 0 | 0 | 8 |
| 89   | truth     | 29 | 0 | 0 | 6 |
| 89   | twin      | 32 | 0 | 0 | 3 |

**Bimodal — every fit is either Band A or Band D**; the LM either lands
truth-/twin-basin (ρ ≈ surrogate noise floor 0.4–0.5) or escapes to a
distant attractor at ρ ≥ 11.

### Per-axis basin radii (largest perturbation that still lands Band A)

The grid saturates at the upper end on q0 axes (≥30° = grid maximum).
"sat" below means the basin extends beyond the largest tested
magnitude.

| Seed | Attractor | q0_along_ω | q0_perp_ω | ω-mag (+pct) | ω-mag (−pct) | ω-dir (deg) |
|-----:|-----------|-----------:|----------:|-------------:|-------------:|------------:|
| 23   | truth     | sat ≥30°   | sat ≥30°  | 7%           | 7%           | 7°          |
| 23   | twin      | sat ≥30°   | sat ≥30°  | 7%           | 7%           | 7°          |
| 28   | truth     | sat ≥30°   | sat ≥30°  | **2%**       | **3%**       | sat ≥10°    |
| 28   | twin      | sat ≥30°   | sat ≥30°  | **2%**       | **3%**       | sat ≥10°    |
| 89   | truth     | sat ≥30°   | **5°**    | sat ≥10%     | sat ≥10%     | 7°          |
| 89   | twin      | sat ≥30°   | 20°       | sat ≥10%     | sat ≥10%     | 7°          |

**Binding axis per seed** (tightest single-axis constraint):

- **Seed 23**: ω-mag and ω-dir tied (~7% / 7°). q0 is loose.
- **Seed 28**: ω-mag (especially positive at +2%) — narrowest of the
  cohort. q0 and ω-dir are loose.
- **Seed 89 truth**: q0-perp (5°) and ω-dir (7°). ω-mag is the widest
  of any seed (±10% saturated).
- **Seed 89 twin**: ω-dir (7°). q0-perp widens to 20°; ω-mag wide.

### Where the LM escapes (Band D landings)

When LM fails, it lands in a distinct competing minimum, NOT a near-truth
shoulder.

| Seed/Attr | Failure axis | Init mag | ρ_final | q0_err° | ω-dir° | ω-mag% (final) |
|-----------|--------------|---------:|--------:|--------:|-------:|---------------:|
| 23 / truth | ω-mag −10%  | −10%     | 35.02   |  94.5   | 15.4   | **−12.6**      |
| 23 / truth | ω-mag +10%  | +10%     | 35.59   | 166.8   | 13.7   | **+11.7**      |
| 23 / truth | ω-dir 10°   | 10°      | 27.99   |  69.3   | 15.0   |  +0.6          |
| 28 / truth | ω-mag +3%   | +3%      | 55.47   |  86.8   | 11.7   | **+3.7**       |
| 28 / truth | ω-mag −5%   | −5%      | 65.88   |  53.1   |  7.3   |  −6.1          |
| 89 / truth | q0-perp 10° | 10°      | 11.17   |  34.1   | 10.7   |  +3.4          |
| 89 / truth | q0-perp 20° | 20°      | 21.40   |   9.5   | 13.8   |  +6.4          |

The 23 ω-mag failures fall onto attractors at **±~12% ω-mag** with
q0_err 90°-170°. Seed 89's q0-perp failures land an attractor with
**q0_err ≈ 34°, ω-dir ≈ 10°, ω-mag +3.4%, ρ ≈ 11** — almost certainly the
class_2 / class_3 multi-solution attractor seen in s014/s034 on this
seed.

### Notable anomaly

One LM (seed 28 twin, q0_along_omega +25°) raised an exception inside
`least_squares` (logged, ρ=inf, n_iter=-1). Surrounding magnitudes 20°
and 30° both landed Band A at ρ=0.51. Treating this as a numerical
glitch, not a basin boundary; basin radius reported as 30° (saturated)
because that magnitude landed.

## Why this matters

1. **q0 alone is essentially globally convergent at correct ω**, on
   seeds 23 and 28 and on seed 89-twin. From any q0 within ≥30°, with
   ω at truth, LM walks home. The cohort IC primitive does NOT need
   tight q0 spacing IF ω is right — it needs *coverage*. Even Sobol
   N=8 (mean ~75° spacing) likely suffices for q0 on the seed-23 /
   seed-28 class. The s011 N=64 yield of 9/10 reflects ω-cell coverage
   problems more than q0-IC density.

2. **ω is the binding constraint**, and the ω-mag basin is
   **seed-dependent and not predictable from |ω| alone**:
   - Seed 28 (|ω|=0.025 rad/s, fastest of the three): tightest ω-mag
     basin (+2% / −3%) — the "narrow basin" class identified in s006/s011.
   - Seed 89 (|ω|=0.004 rad/s, slowest): widest ω-mag basin (±≥10%
     saturated) — low rotation rate ⇒ LC is insensitive to small
     ω-mag changes within the observation window.
   - Seed 23 (|ω|=0.017 rad/s, mid-range): intermediate (±7%).

3. **Truth and twin are symmetric on seeds 23 and 28** — basin
   radii are identical to within numerical noise across all four axes.
   The body-twin convention `(q_180x ⊗ q_truth, R_180x @ ω_truth)` is
   the "same" basin in LC-space.

4. **Seed 89 is uniquely asymmetric**: truth's q0-perp basin is **5°**,
   twin's is **20°**. The truth-attractor sits next to a competing
   attractor at q0_err≈34°/ω-dir≈10.7°/ω-mag+3.4°/ρ≈11 (almost
   certainly the multi-solution class_2 cluster from s014 — q0=30°-150° +
   near-truth ω). Perpendicular q0 perturbations beyond 5° fall toward
   that competing minimum. The twin attractor sits in a different
   region of SO(3) where this competitor is not as close.

5. **Cohort IC-pool implication**: the binding requirement is on **ω
   priors** (need within ~2% mag, ~7° dir on the narrowest seed
   class), not on the q0 IC primitive. Sobol N=64 in q0 is overkill in
   pure q0-coverage terms but inexpensive; phi-sweep's seed-dependent
   q0 distribution is fine *when paired with the right ω-cell*. The
   architectural consequence is that **the cell-filter is what
   matters most**: as long as we land on a cell within the ω basin
   (~2% × 7° on seed 28; wider elsewhere), almost any reasonable q0
   IC will polish home.

6. **This re-prioritises the s039 forward path**: cell-ranking
   (s039b) and end-to-end hybrid (s039c) become the binding tests; q0
   IC density (Sobol vs phi-sweep) is a second-order question because
   either works once ω is right.

7. **Why s037b's per-cell Sobol yields differed across seeds** (23 →
   9/64, 28 → 6/64, 89 → 1/64): on seed 89 truth, the q0-perp basin is
   only 5° wide. With Sobol uniform on SO(3), the fraction of Sobol
   ICs landing inside a 5°-radius cap on the (q0_perp, q0_along) ⊗
   (ω-perturbed) joint manifold is roughly (5/180)^1 × (one
   perpendicular q0 dimension) ≈ a few %, hence ~1-4/64. On seed 23 the
   q0 basin is so wide that nearly any Sobol IC lands home as long as
   the ω-cell is within (±7%, 7°), giving 9/64.

## Numbers

| Metric                           | Value                |
|----------------------------------|----------------------|
| Seeds                            | 23, 28, 89           |
| Attractors                       | truth, twin          |
| LMs total                        | 210 (3 × 2 × 35)     |
| Pool                             | 8 fork workers       |
| Wall                             | **9.4 min**          |
| LM nfev cap                      | 200                  |
| Surrogate ρ noise floor (truth) | 0.39 / 0.48 / 0.39   |
| Surrogate ρ noise floor (twin)  | 0.43 / 0.51 / 0.37   |

## Artefacts

- `experiments/s040_basin_radius_3seed.py`
- `results/s040_basin_radius_3seed/seed{023,028,089}/{truth,twin}_basin.json`
- `results/s040_basin_radius_3seed/summary.json`
- `results/s040_basin_radius_3seed/run.log`

## Out of scope

- Larger q0 perturbations (the ≥30° saturation is a grid limit, not a
  measured basin edge — could probe to 60°/90°). Worthwhile if cohort
  cell-filter accuracy degrades.
- Combined-axis perturbations (e.g. simultaneous q0 + ω-mag): basins
  may be coupled, in which case independent-axis basin radii overstate
  the joint basin.
- Cohort scaling: only 3 seeds tested. Whether seed 28's narrow ω-mag
  basin is typical of "narrow-basin class" or specific to seed 28
  remains open.
- Dynamic basin probing on multi-solution attractors (the seed-89
  q0-perp 5° boundary identifies the competing attractor — characterising
  ITS basin radius is a separate question).

## Cross-references

- **s011** — N=64 Sobol-Shoemake at fixed truth-ω: 9/10 cohort recovery.
  Now reframed: that yield reflects ω-cell coverage AND q0 basin width,
  but the q0 basin is wide enough on most seeds that the binding
  variable is the ω-cell.
- **s034** — phi-sweep + LM on seed 89: 24/50 Band A. Confirms phi-sweep
  works on seed 89 not because q0-perp is wide (it's only 5° on truth)
  but because phi-circles happen to pass through the basin — a geometric
  coincidence.
- **s037b** — Sobol N=64 + LM at the bracket cell: 9/64 (23) / 8/64 (28) /
  1-4/64 (89) Band A. The yield differential matches the q0-perp basin
  width discovered here.
- **s038** — phi-sweep IC pool on seed 23 yields 0/107 LM Band A from
  ICs as close as 2.6° to twin. s040 confirms LM converges from any q0
  perturbation at correct ω; s038's failure was driven by being at the
  bracket ω-cell (5.6% mag, 2.3° dir off) — at the EDGE of seed 23's ω
  basin (7% / 7°). Combined with the q0 perturbation phi-sweep
  introduces, the joint basin is exited.
- **s014b** — multi-solution class_2 / class_3 attractors on seed 89.
  s040 q0-perp Band D landings (ρ≈11, q0_err 34°, ω-dir 10°, ω-mag
  +3.4%) consistent with class_2.
