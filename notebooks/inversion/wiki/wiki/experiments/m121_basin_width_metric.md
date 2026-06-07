---
title: "m121 — Basin-width characterisation for surrogate-residual cost"
type: experiment
sources:
  - "raw/inversion_diagnostics/m121/seed_014/summary.json"
  - "raw/inversion_diagnostics/m121/seed_027/summary.json"
  - "raw/inversion_diagnostics/m121/seed_046/summary.json"
related:
  - "[[m120_tumbling_competitors]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[gradient-based-inversion]]"
  - "[[basin-of-attraction]]"
  - "[[dark-mag-saturation]]"
  - "[[surrogate-model]]"
created: 2026-04-15
updated: 2026-04-16
confidence: high
---

# m121 — Basin-width characterisation for surrogate-residual cost

## Hypothesis

The surrogate-residual cost basin-of-attraction around truth is much tighter than the 20° upper bound set by [[m120_tumbling_competitors]]. Expected at the <1° perturbation scale: cost ≈ surrogate noise floor (~0.055 mag mean_L1). Expected 5-10°: cost discriminates clearly. Expected geometry: ELONGATED with ω-magnitude the WIDEST axis (weakest constraint) and q0/ω-direction narrower.

Falsifiable: if cost at 20° is indistinguishable from cost at 0.25°, there is no local basin structure and search integration is infeasible.

## Actual outcome — split verdict

1. **q0 basin is the WIDEST, not ω-magnitude.** q0 cost grows smoothly and only reaches 5× truth cost at ~5°; still gradient-bearing to ~20°.
2. **ω-magnitude basin is the NARROWEST** (the hypothesis got the direction wrong). At 0.25% |ω| perturbation the cost is already saturated at ~23× truth.
3. **ω-direction basin is also narrow**, saturated at ~17× truth by 0.1° perturbation on all three seeds.
4. **Strong anisotropy WITHIN the ω-direction axis** — 15-25× cost ratio between best-direction and worst-direction perturbations at 0.5°. The basin is a narrow *slab* in SO(3), not a sphere. (Confirmed on all three seeds; preferred axis is seed-specific.)

## Method

Script: `notebooks/inversion/12_brightness_surface/m121_basin_width_metric.py`.

### Pool construction (N = 6751 per seed)

| bucket | count | construction |
|--------|------:|--------------|
| truth | 1 | (q0_true, ω_true) |
| q0 axis | 250 × 8 scales | q0 perturbed by `scale_deg` via random rotation axis; ω held at truth. Scales {0.1, 0.25, 0.5, 1, 2, 5, 10, 20} deg. |
| omega_dir axis | 250 × 8 scales | ω direction perturbed by `scale_deg` via random rotation axis; \|ω\| and q0 held at truth. Same scale list. |
| omega_mag axis | 250 × 7 scales | \|ω\| multiplied by (1 + N(0, scale_pct)); direction and q0 held at truth. Scales {0.25, 0.5, 1, 2, 5, 10, 20}%. |
| joint axis | 250 × 4 scales | (q0, ω) jointly perturbed at `scale_deg` (same perturbation scale applied coherently). Scales {0.25, 0.5, 1, 2}°. |

Total = 1 + 8×250 + 8×250 + 7×250 + 4×250 = 6751.

### Scoring

Each candidate → 500-epoch attitude propagation (reuses `setup.npz` from m119v2/m120) → 500 surrogate magnitudes → residual vs observed → cost variants `mean_L1` and `mean_L2` over 500 epochs. Residuals + cost stored per-candidate (no summarisation).

### Noise-floor reference

`sigma_L1` = stddev of the `mean_L1` cost across the smallest-q0 bucket (250 samples at 0.1° q0 perturbation) — treats this as "surrogate + obs noise" random jitter.

## Headline numbers

### Truth cost and rank

| seed | truth mean_L1 | truth rank / 6751 | σ_L1 noise floor |
|-----:|--------------:|------------------:|-----------------:|
| 14 | 0.05545 | 9 | 6.1×10⁻⁴ |
| 27 | 0.05394 | 62 | 5.6×10⁻⁴ |
| 46 | 0.05283 | 43 | 4.4×10⁻⁴ |

The 9–62 ranks (non-zero) are expected and non-problematic: at the 0.1° q0 scale, ~250 random perturbations will sometimes beat truth by chance given that noise σ_L1 ≈ 5×10⁻⁴ while perturbation-induced cost bump at that scale is ~8×10⁻⁴ for q0. This is the noise floor, not a ranking failure. Compare [[m120_tumbling_competitors]] where truth ranked 0/10004 because that pool had no <1° perturbations (smallest was 2°).

### Seed 14 cost vs scale (mean_L1 median, percent of truth cost)

| scale | q0 | omega_dir | omega_mag (%) | joint |
|------:|----:|----:|----:|----:|
| 0.1° | +1.4% | **+1596%** | — | — |
| 0.25° | +26% | +2803% | **+2279%** | **+2508%** |
| 0.5° | +26% | +3589% | +3081% | +3353% |
| 1° | +73% | +3514% | +3749% | +3556% |
| 2° | +188% | +3561% | +3391% | +3602% |
| 5° | +534% | +3612% | +3605% | — |
| 10° | +1148% | +3608% | +3580% | — |
| 20° | +2048% | +3585% | +3665% | — |

Median-cost saturation (asymptote ≈ 2.05 in mean_L1 units):
- q0: saturates around 20° (2048% of truth — still below asymptote)
- omega_dir: saturates by 0.5° (3589% of truth ≈ 2.04)
- omega_mag: saturates by 0.5–1% (3081–3749% ≈ 1.75–2.13)
- joint: saturates by 0.25° (2508% ≈ 1.44 — dominated by ω component)

Seeds 27 and 46 show the same qualitative pattern with minor scale shifts (cost asymptotes slightly lower at ~1.65–1.75 vs 2.05 because observation geometry differs).

### Basin-boundary k's (factor ×σ)

For each axis, smallest `scale` such that (median − truth) > k·σ:

| axis | seed 14 (k=1) | seed 27 | seed 46 | interpretation |
|------|:------------:|:-------:|:-------:|----------------|
| q0 | 0.1° | 0.25° | 0.1° | Cost lifts off noise floor at the smallest scale tested. |
| omega_dir | 0.1° | 0.1° | 0.1° | Already saturated well above noise at 0.1°. |
| omega_mag | 0.25% | 0.25% | 0.25% | Same. |
| joint | 0.25° | 0.25° | 0.25° | Dominated by ω component. |

The boundary is below resolution for ω-direction and ω-magnitude on all three seeds — we hit cost saturation before we can measure the basin edge. Finer-grained sampling (e.g. 0.01-0.1° ω-direction, 0.01-0.25% ω-magnitude) is needed to pinpoint the actual edge.

## Mechanism: why q0 is broad and ω is narrow

q0 error is a **constant rotational offset** — at every epoch, the propagated attitude differs from truth by the same rotation matrix. Body-frame sun/observer vectors rotate rigidly. The brightness prediction error is bounded and roughly scale-linear at small angles (via Ashikhmin-Shirley smoothness). Residuals accumulate slowly.

ω error is a **constant angular-velocity offset** — attitude divergence grows LINEARLY with time. Over the 1-hr constraint window (obs_times span with 500 epochs), the accumulated drift is:

- For an ω-direction error of θ_dir rad, the divergence at time t is approximately `|ω| · t · θ_dir` in radians (small-angle cross-product). IS-901 |ω_true| ≈ 0.0215 rad/s, t_window = 3600 s:
  - 0.1° (= 1.75e-3 rad) ω-dir error → drift of `0.0215 · 3600 · 1.75e-3 = 0.135 rad = 7.8°` by end of window. With a light curve that has ~114 peaks across 500 epochs (peak every ~30 s ≈ 5–10° of attitude travel), this is enough to offset peak timing by 1–2 peak widths at late epochs.
  - 0.5° ω-dir → 39° drift → attitude is essentially decorrelated from truth for the second half of the window. Cost saturates.

- For an ω-magnitude error of δ (fractional), phase drift is `|ω| · t · δ`:
  - 0.25% → `0.0215 · 3600 · 0.0025 = 0.194 rad = 11.1°` end-of-window drift. Same regime: enough to scramble peak timing, saturate cost.
  - 0.5% → 22° end-of-window → saturated.

- For a q0 error of θ_q0 rad: the SAME offset at every epoch. The body-frame sun/observer vectors are rotated by a fixed matrix. Light-curve perturbation is bounded by local BRDF gradient times θ_q0:
  - 1° → bright/dark contrast perturbation at each epoch, not cumulative → at a typical IS-901 facet-alignment moment, roughly 1° rotation of sun vector produces ~0.01-0.05 mag change. mean_L1 bump ≈ 0.04 mag ≈ +73% of truth cost. Monotonic, survivable, gradient-bearing.

So the asymmetry is a **time-integral effect**: ω errors compound, q0 errors don't. The same absolute angular error (say 1°) produces vastly different trajectory deviations depending on whether it's applied once (q0) or continuously (ω).

This is reinforced by the joint axis: at scale 0.25° the joint cost is already saturated at 2508% of truth, which is slightly *worse* than the ω-direction cost at the same scale (2803%) because q0 error adds a small additional penalty. But the dominant term is ω.

## ω-direction anisotropy: seed-specific preferred axis

At ω_dir scale = 0.1°, the best-cost 3 and worst-cost 3 of 250 random rotation axes were compared. Rotation axes computed as `axis_inertial = normalise(ω̂_true × ω̂_perturbed)`, then mapped into the body frame by applying `inv(q0_true)`.

| seed | best body-frame axis (approx) | worst body-frame axis | cost ratio @ 0.5° (worst/best) |
|-----:|-------------------------------|-----------------------|:-----:|
| 14 | `[+0.39, +0.19, +0.90]` (mostly body +Z) | `[+0.15, −0.98, +0.09]` (body −Y) | 25.0× |
| 27 | `[+1.00, +0.01, +0.03]` (body +X) | `[+0.02, −0.89, −0.45]` (−Y/−Z) | 17.4× |
| 46 | `[−0.86, −0.36, +0.37]` (body −X, some +Z) | `[+0.00, +0.73, +0.68]` (+Y/+Z) | 15.3× |

**Observations:**
1. **Anisotropy is universal (15-25× ratio across all three seeds).** The ω-direction basin is NOT a spherical ball — it's a narrow slab.
2. **Preferred body-frame axis is NOT identical across seeds.** Seed 14 prefers body +Z, seed 27 prefers body +X, seed 46 prefers body −X with some +Z. This is observation-window / seed-specific, not a pure IS-901 geometry property.
3. **Common thread: worst axis always has dominant body ±Y component.** Body Y is the solar-panel spin axis (Intelsat 901 convention). Perturbing ω direction *around* the Y axis swaps bright +X facet lobes between observer and sun — maximally decorrelating the LC. Perturbing ω around the preferred axis (parallel to the dominant bright-facet normal at that epoch's geometry) keeps the bright/dark structure of the LC approximately intact.

The seed-specificity of the preferred axis tells us: *which* body direction is "safe" depends on where IS-901 is in its tumble and where the sun/observer are during the window. This is not a flaw in the cost — it's a property of the LC inversion problem itself.

## What we learned

1. **Basin is elongated, q0 widest** — confirmed elongated, but axis assignment was wrong in the hypothesis.
2. **ω-direction basin < 0.1° at the 1σ noise-floor test.** Gradient-based search needs ω initialised within 0.1° or the gradient points nowhere useful because the cost is saturated. Current classical pipeline ([[m102_fullmse]]) achieves 0.3–3° ω-direction error on "OK" seeds → short by a factor of 3–30.
3. **ω-magnitude basin < 0.25%.** Current [[omega-magnitude-estimation]] achieves ~13% median error (peak-count regression) and m102 achieves 0.1–0.7% after NM+geo → borderline-adequate on "OK" seeds, insufficient on ATT_FAIL seeds.
4. **q0 basin ≳ 5°** (cost at 5° is only 5× truth; gradient still meaningful).
5. **Basin anisotropy in SO(3) is universal but not axis-universal.** Each seed has its own preferred ω-rotation direction (governed by the observation-window bright-facet geometry). Implication: a DE mutation that anisotropically favours rotations around the seed-specific preferred axis would outperform isotropic mutation.
6. **Dark-mag saturation is the floor.** The ~2.05 mean_L1 asymptote matches the [[dark-mag-saturation]] mechanism: a wildly wrong ω produces all-dark surrogate predictions, residual ≈ 10 mag, mean_L1 = 10 × (fraction of epochs saturated). Consistent with [[m120_tumbling_competitors]] close_omega rank 8004 story — same mechanism, different pool.

## Limitations / what this does NOT establish

- **Three seeds (14, 27, 46) from the ATT_FAIL cohort only.** Not tested on OK-cohort seeds (74, 93) — the basin shape may be wider there if the LC has more discriminating peaks.
- **Basin edges for ω-direction and ω-magnitude are below resolution.** We know the cost saturates by 0.1° / 0.25%, but cannot tell whether the true basin edge is at 0.01° / 0.01% or 0.08° / 0.2% without finer sampling.
- **Hessian/curvature not measured directly.** All statements come from random-sampling in perturbation shells, not from local derivatives. A direct finite-difference Hessian at truth would be one-shot and definitive.
- **No search-algorithm integration tested.** Inferring gradient-based feasibility from basin geometry alone. Concrete test: run surrogate-DE from perturbed inits at known scales and count convergence rate per axis.

## Next steps

1. **Finer-grained ω-direction slice (0.01–0.1°)** on seed 14 — pinpoint the actual basin edge within the current resolution. Cheap (~250 extra candidates on saved setup).
2. **Hessian at truth** via finite differences — 12-parameter central-difference Hessian on (q0, ω) gives the local quadratic basin shape in one shot. Eigenvalues of this matrix ARE the basin widths along principal axes, no sampling required. Estimated cost: ~150 surrogate evals = seconds.
3. **OK-cohort generalisation** — rerun m121 on seeds 74 and 93. If basin widths are comparable, the narrow-ω conclusion generalises. If wider, the ATT_FAIL classification is partly explained by tighter basins.
4. **Direct search test** — run `surrogate-DE` from perturbed inits at known scales (0.05°, 0.1°, 0.5°, 2° ω-direction) and measure convergence basin as function of init scale. This is the operational question for [[gradient-based-inversion]].

## Plots

- `data/results/inversion_diagnostics/m121/seed_014/plots/cost_vs_scale_per_axis.png`
- `data/results/inversion_diagnostics/m121/seed_014/plots/basin_boundary.png`
- (same for seeds 027 and 046)
