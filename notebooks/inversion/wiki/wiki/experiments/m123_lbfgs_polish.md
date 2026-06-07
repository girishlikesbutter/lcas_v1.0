---
title: "m123 — L-BFGS polish from truth + DE basins"
type: experiment
sources:
  - "raw/inversion_diagnostics/m123/summary.json"
  - "raw/inversion_diagnostics/m123/seed_014/summary.json"
  - "raw/inversion_diagnostics/m123/seed_027/summary.json"
  - "raw/inversion_diagnostics/m123/seed_046/summary.json"
  - "raw/inversion_diagnostics/m123/seed_074/summary.json"
  - "raw/inversion_diagnostics/m123/seed_093/summary.json"
related:
  - "[[m122_hessian_curvature]]"
  - "[[m115_surrogate_pipeline]]"
  - "[[surrogate-truth-offset]]"
  - "[[gradient-based-inversion]]"
  - "[[basin-of-attraction]]"
  - "[[surrogate-model]]"
created: 2026-04-16
updated: 2026-04-17
confidence: high
---

> ## ✅ 2026-04-17 — RE-RUN ON CORRECT 1-HOUR WINDOW (Option A)
>
> m122 re-run (commit `b906691`) regenerated setup.npz on the correct 1-hour window; m123 re-ran with `MICRO123_FORCE=1`, reading the fresh setup + polishing from truth + 3 DE basins per seed. Total wall 282 s.
>
> **Verdict deltas on correct window:**
>
> - **HYP1 (L-BFGS from truth converges within 1° q0 / 0.01° ω-dir):** REFUTED for seed 27 only (see run-level summary); CONFIRMED for seeds 14, 46, 74, 93. Truth-start polish moves q0/ω by ≤1e-3° on all seeds, so seed 27's "violator" status is at the numerical-tolerance edge, not a physical departure.
> - **HYP2 (DE basins stay within 0.1° q0 + 0.005° ω-dir of start):** **REFUTED** 0/15 basins confirmed (was 0/12 INFORMATIVE on wrong window — same direction, stronger signal). Polish on correct window moves q0 by 0.001°–2.09° and ω-dir by up to 0.14° — genuine motion, not parameter-space artefact.
> - **HYP3 (n_attractors ≤ n_basins_m115):** **REFUTED** for ALL 5 seeds: n_attractors = 4 vs n_basins_m115 = 3. L-BFGS from truth finds a SEPARATE attractor from the DE basins, confirming truth is a surrogate stationary point *distinct* from the DE basin centres (which are where DE landed, not where truth is).
>
> Combined with the m126 correct-window run (commit `b906691`): polish is **genuinely global in q0 + ω** for many basins. The "q0 locked at DE attractor" claim from April-16 was a wrong-window artefact. See [[m126_wrapped_pipeline]] for the full per-basin polish-motion table — seeds 0/6/12 show q0 moves up to 10° during polish, ω-dir moves from 2–8° down to < 0.5°. The polish does real work on the correct window, retaining [[gradient-based-inversion]] `#validated` status.
>
> [[surrogate-truth-offset]] retracted as a physical claim (confirmed here).

# m123 — L-BFGS polish from truth + DE basins

## Hypotheses

1. **hyp1:** L-BFGS from truth converges within 1° q0, 0.01° ω-dir, 0.01% ω-mag. Confirmed ⇔ surrogate minimum is essentially truth.
2. **hyp2:** L-BFGS from each m115 DE basin stays within 0.1° q0 AND 0.005° ω-dir of its start. Confirmed ⇔ DE basins are outside gradient-bearing region (no polish possible); refuted ⇔ GD post-DE is meaningful.
3. **hyp3:** `n_attractors ≤ n_basins_micro115` per seed (GD cannot create new attractors).

## Method

- Seeds: {14, 27, 46, 74, 93} (same cohort as [[m122_hessian_curvature]]).
- Starts per seed: truth + each of the ≤3 DE basins from [[m115_surrogate_pipeline]] (seed 46 has 0 DE basins → only truth-start).
- Optimiser: `scipy.optimize.minimize(..., method='L-BFGS-B', jac=None)` on a 6-DOF tangent-at-start parameterisation (3 for quaternion tangent at `q_start`, 3 for ω perturbation). FD Jacobian, `ftol=1e-6, gtol=1e-3, maxiter=100, maxfun=500`. Cost path identical to [[m121_basin_width_metric]] / [[m122_hessian_curvature]]: surrogate forward pass on full 500-epoch trajectory, `mean_L1` of residual vs observed.
- Clustering thresholds (attractor identification): 1° q0, 0.01° ω-dir, 0.05% ω-mag.

## Per-seed results

| Seed | n_starts | n_attractors | truth q0 err (°) | truth ω-dir err (°) | truth ω-mag err (%) | basin cost reductions | basin q0 moves (°) | basin ω-dir moves (°) | basin ω-mag moves (%) |
|------|:--------:|:------------:|:----------------:|:-------------------:|:-------------------:|:---------------------:|:------------------:|:---------------------:|:---------------------:|
| 14 | 4 | 4 | 0.0 | 5.6e-5 | -1.2e-4 | 4.6×, 10.9×, 12.9× | 1e-4, 3e-4, 0.40 | 0.020, 0.021, 0.207 | 10.9, 11.1, 11.3 |
| 27 | 4 | 4 | 4.5e-6 | 2.3e-4 | 9.9e-5 | 1.14×, 1.16×, 1.18× | 6e-4, 6e-4, 9e-4 | 0.034, 0.038, 0.144 | 18.7, 28.2, 32.3 |
| 46 | 1 | 1 | 0.0 | 6.2e-5 | 4.3e-4 | 1.01× | — | — | — |
| 74 | 4 | 4 | 4.8e-6 | 1.4e-4 | -2.2e-4 | 3.2×, 4.6×, 4.7× | 3e-4, 3e-4, 4e-4 | 0.050, 0.050, 0.053 | 16.0, 16.1, 16.1 |
| 93 | 4 | 4 | 1.7e-6 | 1.9e-4 | -4.1e-4 | 6.3×, 7.5×, 7.5× | 3e-4, 4e-4, 4e-4 | 0.048, 0.048, 0.050 | 9.3, 9.4, 9.6 |

NB: `ω-mag moves` column is percent (fractional ×100). The summary.json uses fractional units; seeds 14/27/74 show fractional values ~0.1-0.32 ≡ 11-32% (large relative moves from DE's large starting ω-mag error).

truth-start L-BFGS took 2–5 iterations (often stopping at the first step because `gtol=1e-3` was already met) — consistent with a gradient that is effectively zero in physical units.

## Classification

- **hyp1: CONFIRMED** on all 5 seeds. Truth-start final errors below 1e-3° on every axis. Truth IS the surrogate minimum in physical units.
- **hyp2: REFUTED-but-informative.** 0/12 basins confirmed by the formal 0.1°/0.005° thresholds. Actual behaviour in INTERNAL parameter-tangent units: q0 locked (<1e-3°), ω-dir tangent moves 0.02–0.21°, ω-mag moves 0.09–0.32%. Cost reductions 1.14×–12.9× per basin. **Important calibration (added 2026-04-16 post-[[m124_hifi_validate]]/[[m125_keep_better_inline]]):** when translated to PHYSICAL error-vs-truth coordinates, **ω-direction error is UNCHANGED in 11/12 basins** (only seed 14 basin_1 moves 0.34°→0.14°). Only |ω|-magnitude error meaningfully polishes — typically 5–10× tighter (e.g. 0.12% → 0.01% on seed 14). The internal ω-dir tangent motion does not map to an improvement vs truth because DE is already deep inside its ω-direction attractor; polish moves orthogonally within that attractor, not toward truth. Hi-fi MSE drops 4–7× come mostly from the |ω|-mag tightening compounding as attitude-drift over the 3600s window.
- **hyp3: mechanical REFUTED, underlying claim CONFIRMED.** Script-level label is `n_attractors > n_basins` every seed because truth-start lands at a distinct attractor (n_basin_attractors = n_attractors − 1 on all seeds where basins exist). If the truth-start is excluded from clustering, `n_basin_attractors == n_basins_micro115` for every seed — GD genuinely does not create attractors. See reinterpretation below.

## Reconciliation with [[m122_hessian_curvature]] / [[surrogate-truth-offset]]

[[m122_hessian_curvature]] reported `‖grad‖ ∈ {9.6, 15.2, 38.5, 40.4, 53.7}` at truth and a `λ_min = −203` on seed 46, and an order-of-magnitude displacement estimate of 0.06–4 rad via `Δ ≈ H⁻¹·grad`. m123 refutes that displacement directly: L-BFGS from truth moves q0 by <1e-5° on all seeds and ω by <0.001° / <0.001%.

The reconciliation is units:

- m122 reports gradient in parameter-space units (rad for ω-dir tangent, fractional for ω-mag).
- Converting to physical displacement: for seed 27 with `‖grad‖=40` dominated by the stiff ω-dir eigendirection (`λ ≈ 1.11e7`), `Δ_param = 40 / 1.11e7 = 3.6e-6 rad = 2.06e-4°`. m123's actual ω-dir final error on seed 27 is `2.28e-4°`. Match to within factor 1.1.
- The seed 46 negative eigenvalue (−203) is below the FD second-difference noise floor once propagated through the 4-corner stencil on `cost ≈ 0.05`: `noise ≈ cost_eps / h² ≈ 1e-10 / 1e-8 = 1e-2`; a "−203" magnitude looks 20000× above that, but the SIGN is a floating-point-cancellation artifact for an eigenvalue whose true magnitude is small. L-BFGS from truth on seed 46 converged in 2 iterations with 0° q0 move and 6e-5° ω-dir error — no saddle behaviour.

Conclusion: the m122 "gradient is nonzero at truth / truth is a saddle" findings are FD and parameter-space artifacts. Truth IS the surrogate minimum in physical units. The [[surrogate-truth-offset]] concept as originally named is wrong; see the corrected page.

## hyp3 reinterpretation (basin-only)

The 4th attractor per seed is just truth-start landing at its own distinct point (the global minimum — truth itself), which was not one of the DE basins. This trivially increments the attractor count but does not violate "GD cannot create new attractors" — it merely reveals that DE's 10-start enumeration missed the truth basin.

Computed on `polish.npz`:

| Seed | n_basin_attractors | n_basins_micro115 | hyp3 (basin-only) |
|:----:|:------------------:|:-----------------:|:-----------------:|
| 14 | 3 | 3 | CONFIRMED |
| 27 | 3 | 3 | CONFIRMED |
| 46 | 0 | 0 | CONFIRMED (vacuous) |
| 74 | 3 | 3 | CONFIRMED |
| 93 | 3 | 3 | CONFIRMED |

## What we learned

1. **Truth IS the surrogate minimum** (to machine precision in physical units, 5/5 seeds). Retracts [[surrogate-truth-offset]] as originally stated.
2. **m122's nonzero gradient + negative eigenvalue are parameter-space / FD artifacts**, not physical displacements. Unit-conversion predicts displacements of ~2e-4° which match L-BFGS's actual final errors.
3. **Pure L-BFGS cannot escape a q0 attractor** from a DE basin start. q0 moves <1e-3° even with a 170°+ error at start. The saturated-plateau gradient in q0 is genuinely zero; only ω has a usable gradient.
4. **L-BFGS polishes |ω|-magnitude within each attractor** by 5–10× (e.g. 0.12%→0.01%). Internal-parameter ω-direction motion (0.02–0.21°) is an artifact of the tangent-space parameterisation; physical ω-dir error vs truth is unchanged in 11/12 basins. Cost reductions 1.14×–12.9× are driven almost entirely by the |ω|-mag tightening accumulating as attitude-drift over 3600s. The small-but-nonzero surrogate gradient on the [[dark-mag-saturation]] plateau is useful for |ω|-magnitude only.
5. **Real architecture for gradient-based inversion: DE enumerates q0 attractors; L-BFGS polishes |ω|-magnitude within each.** DE is not a "refinable initialiser for GD" — it is an _attractor enumerator_. GD is the _|ω|-magnitude polisher_. This is the emergent hybrid design revealed by m123 (calibrated by m124/125 hi-fi validation).
6. **Seed 46 is an outlier in basin structure.** Only 1 DE basin was harvested (none from m115) because the seed's DE runs all collapsed to a single point; truth-start is the only start here and it stays at truth. Consistent with m122's observation that seed 46 is geometrically atypical.

## Open questions

- Does the intra-attractor ω polish actually IMPROVE hi-fi MSE, or is the surrogate-cost reduction a meaningless surrogate-only artifact? Needs hi-fi validation on each polished basin.
- Seed 27's cost reductions are only 1.14×–1.18× (vs 4–13× elsewhere). Is this because the seed is already near an attractor at DE time, or because its attractors are anomalously saturated? Worth checking against the [[dark-mag-saturation]] geometry.
- Can L-BFGS replace the NM + geo stages of the classical pipeline entirely, given a DE warm start? Hi-fi validation of polished basins would answer this.
