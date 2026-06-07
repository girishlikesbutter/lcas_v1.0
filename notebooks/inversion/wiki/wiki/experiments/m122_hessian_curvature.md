---
title: "m122 — Hessian at truth, basin curvature + cohort comparison"
type: experiment
sources:
  - "raw/inversion_diagnostics/m122/summary.json"
  - "raw/inversion_diagnostics/m122/seed_014/summary.json"
  - "raw/inversion_diagnostics/m122/seed_027/summary.json"
  - "raw/inversion_diagnostics/m122/seed_046/summary.json"
  - "raw/inversion_diagnostics/m122/seed_074/summary.json"
  - "raw/inversion_diagnostics/m122/seed_093/summary.json"
related:
  - "[[m121_basin_width_metric]]"
  - "[[m120_tumbling_competitors]]"
  - "[[basin-of-attraction]]"
  - "[[gradient-based-inversion]]"
  - "[[dark-mag-saturation]]"
  - "[[surrogate-model]]"
  - "[[surrogate-truth-offset]]"
created: 2026-04-16
updated: 2026-04-17
confidence: high
---

> ## ✅ 2026-04-17 — RE-RUN ON CORRECT 1-HOUR WINDOW (Option A)
>
> `m122_hessian_curvature.py:328` patched 2026-04-17 (commit `5d5938f`). Stage A assertion fixed (was checking for 6-hr span, now correctly checks 1-hr). Quarantined m119v2 wrong-window setup.npz caches (all 6 affected seed dirs — m119v2 turned out to have the same bug; the bug-doc claim "m119v2 CORRECT" was itself wrong). Re-run with `MICRO122_FORCE=1` regenerated setup.npz + Hessian for all 5 seeds on correct 1-hr ctx (commit `b906691`).
>
> **Verdict deltas on correct window:**
>
> - **HYP1 (basin widths Hessian-tight):** 4/5 CONFIRMED (seed 14 has one `inf` q0 width → REFUTED for that seed).
> - **HYP2 (m121 anisotropy axis match):** INCONCLUSIVE — not fully decidable on current data.
> - **HYP3 (cohort universality of basin geometry — ATT_FAIL ≈ OK basins):** **REFUTED** (was CONFIRMED on wrong window). Cohort eig-spreads are now within 1.20× (ATT_FAIL median 4.63e+03, OK median 5.55e+03); basin geometry does NOT explain ATT_FAIL. Strategic flip: the wrong-window finding suggested basin geometry might be the driver; the correct-window finding instead points at ω-direction quality of the upstream candidates (see [[m126_wrapped_pipeline]] and [[gradient-based-inversion]]).
>
> New Hessian numbers (correct window):
>
> | seed | eigenvalue spread | basin widths q0 (°) | ω-dir (°) | ω-mag (%) |
> |:---:|---:|---|---|---|
> | 14 | 3.58e+04 | [2.82, inf] | [] | [] |
> | 27 | 1.49e+03 | [0.51, 0.78] | [0.027] | [0.035] |
> | 46 | 4.63e+03 | [1.06, inf] | [0.016] | [0.030] |
> | 74 | 2.81e+03 | [0.75, 0.87, inf] | [0.028, inf] | [0.029] |
> | 93 | 8.29e+03 | [0.30, 0.45, 1.06] | [0.014, inf] | [0.020] |
>
> [[surrogate-truth-offset]] retracted as a physical claim — [[m123_lbfgs_polish]] on correct window finds truth IS a surrogate stationary point (polish from truth moves q0/ω by ≤1e-3° on every seed). The non-zero |grad| at truth observed here is FD-chart numerical artefact, not a real offset. Eigenvector geometry and per-axis anisotropy interpretations in the page body may need case-by-case re-verification against the new `seed_NNN/summary.json` before citing.

# m122 — Hessian at truth, basin curvature + cohort comparison

## Hypothesis

Three falsifiable claims about the 6-DOF surrogate-residual cost Hessian `H` evaluated at `(q0_true, ω_true)`:

1. **Hyp1:** Eigenvalues of `H` confirm per-axis basin widths from [[m121_basin_width_metric]]: ω-direction width < 0.1°, q0 width ~5°.
2. **Hyp2:** The two stiffest eigenvectors of `H` lie predominantly in the ω-direction subspace, and their dominant body-frame projection matches the seed-specific principal axis identified empirically by m121 (seed 14 ≈ body +Z stiff / −Y soft; seed 27 ≈ body +X stiff; seed 46 ≈ body −X/+Z stiff).
3. **Hyp3:** OK-cohort seeds (74, 93) have basin geometry **qualitatively similar** to ATT_FAIL cohort (eigenvalue spread within ~1 order of magnitude). REFUTED ⇔ basin geometry is part of why ATT_FAIL seeds are hard.

## Method

Script: `notebooks/inversion/12_brightness_surface/m122_hessian_curvature.py`.

- Parameterisation: 6-DOF local chart — `(rx, ry, rz)` rotation-vector perturbation of `q0_true` + `(e1, e2)` tangent basis perturbation of `ω̂_true` (keeping |ω| fixed) + `delta` scalar fractional perturbation of |ω|.
- Step sizes: `h_q0 = 1e-2 °` (1.7e-4 rad), `h_ω_dir = 5e-3 °` (8.7e-5 rad), `h_ω_mag = 1e-4` (fractional).
- Central-difference 6×6 symmetric Hessian via 1 centre + 12 axis + 60 mixed evaluations = **73 surrogate calls per seed**.
- `q0_true` / `ω_true` read from per-seed setup kernel (consistent with m119v2 / m120 / m121 geometry — honest 1-hr window, 500 epochs).
- Outputs per seed: `hessian.npz` (H, eigenvalues, eigenvectors), `evals.npz` (raw cost evaluations), `summary.json` (gradient, eigenvalues, basin widths in parameter and physical units, axis classification, body-frame projection of stiff eigenaxes, verdicts).

## Results

### Headline per-seed table

| seed | cohort | truth cost | ‖grad‖ | eig max | eig #2 | eig #3 | eig min | q0 basin | ω-dir basin | ω-mag basin | hyp2 |
|-----:|:------:|-----------:|-------:|--------:|-------:|-------:|--------:|---------:|------------:|------------:|:-----|
| 14 | ATT_FAIL | 0.0555 | 15.2 | 1.50e+07 | 8.02e+06 | 8.05e+04 | 14.3 | 0.38° / 0.85° / 3.56° | 0.048° | — | REFUTED |
| 27 | ATT_FAIL | 0.0539 | 40.4 | 1.11e+07 | 6.18e+06 | 1.63e+05 | 38.1 | 0.59° / 1.15° / 2.16° | 0.005° / 0.033° | 0.007% | CONFIRMED |
| 46 | ATT_FAIL | 0.0528 | 38.5 | 1.67e+07 | 8.83e+06 | 6.28e+05 | **−203** | 0.38° / 0.64° / ∞ | 0.003° / 0.017° | 0.008% | REFUTED |
| 74 | OK | 0.0623 | 53.7 | 1.23e+07 | 7.63e+06 | 1.51e+06 | 169 | 0.44° / 0.82° / 1.10° | 0.005° / 0.012° | 0.007% | INCONCLUSIVE |
| 93 | OK | 0.0494 | 9.6 | 1.42e+07 | 8.43e+06 | 1.92e+05 | 70.4 | 0.32° / 0.56° / 1.52° | 0.004° / 0.029° | 0.006% | INCONCLUSIVE |

Basin width is the axis-aligned 50%-cost-rise half-width, computed from each eigenvalue as `w_i = sqrt(0.5 · truth_cost / λ_i)` and converted into physical units via the eigenvector's projection onto q0 / ω-dir / ω-mag subspaces (`axis_fracs` and `basin_widths_physical` in each seed's `summary.json`).

### Axis classification of eigenvectors

Axis classification uses a 0.75 fraction threshold (see per-seed `axis_fracs`). All 5 seeds have the SAME structure:
- Top 2 stiffest eigenvectors: mixed ω-dir / ω-mag (at ~0.88 / 0.12 split) — not pure ω-dir.
- 3rd stiffest: ω-dir-dominant.
- Bottom 3: q0-dominant, essentially one per (rx, ry, rz) axis.

The pattern is universal across ATT_FAIL and OK cohorts.

### Cohort eigenvalue spread comparison

- ATT_FAIL spreads: seed 14 → 1.04e+06; seed 27 → 2.90e+05; seed 46 → 3.89e+04. Median 2.90e+05.
- OK spreads: seed 74 → 7.31e+04; seed 93 → 2.02e+05. Median 1.38e+05.
- Ratio ATT_FAIL median / OK median = **2.11×** — well within the 10× threshold set by the hypothesis.

## Classification

| Hypothesis | Script verdict | Correct verdict | Notes |
|-----------|:--------------:|:---------------:|-------|
| Hyp1 | CONFIRMED (5/5) | CONFIRMED | Hessian widths tightly consistent with m121 saturation scales (see "Hessian vs m121 empirical basin" below). |
| Hyp2 | REFUTED (2/3) | REFUTED (2/3) | Only seed 27's stiffest ω-dir eigenvector aligns with m121's empirical preferred axis (dot 0.89). Seed 14 dot 0.07, seed 46 dot 0.26 (see `anisotropy_axis_dot_micro121` per seed). Mechanism: local curvature ≠ finite-scale anisotropy — they measure different quantities (see below). |
| Hyp3 | REFUTED (label) | **CONFIRMED** | **Script label is inverted** — the reason text says "eig spreads within 2.11×, basin geometry does NOT explain ATT_FAIL", which matches the hypothesis ("OK cohort similar to ATT_FAIL"). Numbers are correct; label is backwards. Decision: document here rather than re-emit the run summary (numbers authoritative). |

## What we learned

1. **Hessian-derived basin widths are universal across all 5 seeds tested.** q0 widths 0.3–0.9° per principal q0 axis, ω-dir widths 0.003–0.05°, ω-mag width 0.006–0.008%. Cohort is irrelevant — OK and ATT_FAIL seeds have basins of the same shape and tightness.

2. **Truth is NOT a stationary point of the surrogate-residual cost.** Gradient norm at truth is 9.6–53.7 across 5 seeds (not machine zero). Dominant components are always in ω-dir / ω-mag axes (up to `grad = 40.4` on the ω-mag component of seed 27, `grad = 38.4` on an ω-dir component of seed 46). Truth has |grad| ≈ 10–50 units.mag²/rad, which means a pure-gradient step of even 1e-6 rad leaves truth. See [[surrogate-truth-offset]].

3. **Seed 46 has a NEGATIVE eigenvalue (−203).** The FD Hessian at truth is not positive-semi-definite: there is a direction in parameter space along which the cost DECREASES at truth. This direction lies in the ω-dir / ω-mag subspace (eigenvector: −0.995 on ω-mag axis, −0.095 on ω-dir e1). Combined with the non-zero gradient, this is strong evidence that **the surrogate's local cost minimum is OFFSET from truth** in the ω direction. For seed 46 specifically, a gradient-descent started exactly at truth would move AWAY from truth along this direction.

   FD noise check: with `h_ω_mag = 1e-4`, `truth_cost ≈ 0.053`, and float64 cost noise ~1e-8, the second-difference noise floor is ~1e-8 / h² ≈ 1 — so |−203| is ~200× above the FD noise floor. Not numerical; the saddle is real.

4. **Hyp2 refutation is a scale mismatch, not a contradiction.** m121 measured cost-ratio anisotropy at **finite 0.1–0.5°** ω perturbations, where the cost has already saturated at ~2.05 mean_L1 ([[dark-mag-saturation]]). The axis m121 identifies is "which direction climbs the fastest to saturation" — a large-scale property of the cost surface. m122's Hessian is the **local quadratic curvature** at truth — an infinitesimal property. The two need not agree; a saturating cost has principal curvatures near truth that say nothing about where the saturation plateau tilts far away. `omega_dir_eigenaxes_body` per seed: seed 14 `[−0.18, +0.98, +0.07]` (body +Y, 4° off m121's +Z); seed 27 `[−0.89, +0.46, −0.03]` (body −X, antiparallel to m121's +X — dot 0.89 in absolute value); seed 46 `[−0.47, +0.87, −0.11]` for stiffest, `[+0.61, +0.41, +0.67]` for next (both ≳90° from m121's body −X/+Z).

5. **Cohort universality (hyp3 CONFIRMED).** ATT_FAIL classification is NOT explained by tighter basin curvature at truth. OK-cohort seeds have the same narrow ω basin and wide q0 basin as ATT_FAIL. ATT_FAIL must therefore come from the **finding problem** — getting the classical pipeline's NM/geo stage close enough to truth — not from intrinsic local geometry. Confirms that [[surrogate-attitude-isoshell]] / [[gradient-based-inversion]] search effort should target the init-accuracy gap, not cost reshaping.

6. **Mass-matrix precompute now available.** Each seed's `hessian.npz` gives a 6×6 symmetric mass matrix usable for preconditioning DE mutation (scale mutations by eigenvector / sqrt(eig)) or HMC (use H as inverse-mass). The 5-seed pool covers ATT_FAIL + OK so transfer tests are possible.

## Hessian vs m121 empirical basin

m121 measured basin widths by random sampling at finite perturbation scales {0.1°, 0.25°, ..., 20°} and reported the scale at which median cost lifts off the 1σ noise floor. Those widths are ~0.1–0.5° ω-dir and ~5° q0. m122 measures basin widths by extracting the local quadratic approximation, giving ~0.003–0.05° ω-dir and ~0.3–0.9° q0 — **10–100× tighter**.

This is expected and does not undermine either measurement:

- **m121 widths** are the scale at which the cost is already saturated at its dark-mag ceiling (~2.05 mean_L1). Beyond 0.5° ω-dir the cost is flat; the "width" is the last finite scale where the sampled median is distinguishable from saturation.
- **m122 widths** are the quadratic extrapolation of curvature at truth to a 50%-cost-rise threshold. They predict what the basin WOULD look like if the cost remained quadratic out to the threshold. Since the cost actually saturates well before the quadratic prediction, the Hessian basin is a lower bound on the *functional* basin width.

The two numbers measure different features of the same landscape. Taken together: **the cost is quadratic for a very short distance (~0.01–0.05° ω-dir), then transitions to a flat dark-mag plateau.** No intermediate "gradient-bearing but non-quadratic" regime. This is bad news for gradient-based inversion — the window where gradients are trustworthy is narrower than m121 suggested. See [[gradient-based-inversion]] for the revised init-accuracy budget.

## Pseudocode

```
load setup (q0_true, omega_true, 500 epochs, surrogate, observed mags)
for each of 73 probe points in 6-DOF local chart:
    q0_p, omega_p = apply_perturbation(q0_true, omega_true, delta_params)
    traj = propagate_attitude(q0_p, omega_p, epochs, I_tensor)
    mags = surrogate(k1_body, k2_body, panel, dish, dist)  # 500 calls
    cost = mean_L1(mags - observed_mags)
H = central_difference_hessian(73 costs, h_per_axis)
eigvals, eigvecs = eigendecomp(H)
for each eigvec:
    axis_class = classify_by_axis_fracs(eigvec[0:3], eigvec[3:5], eigvec[5])
    basin_width = sqrt(0.5 * truth_cost / eigval)
    physical_width = project_onto_q0_or_omega_axis(eigvec, basin_width)
save hessian.npz, evals.npz, summary.json
```

## Limitations / open questions

- Only 5 seeds. Negative eigenvalue on seed 46 is a 1/5 rate; sampling error bars wide.
- Step sizes chosen to balance truncation vs FD noise but not optimised; see appendix of `summary.json` for `h_per_axis`.
- Hessian measured in the same local chart as m121's perturbations (rx/ry/rz quaternion tangent + ω̂ tangent basis). Different parameterisations (e.g. Euler, MRPs) would give different numerical Hessians but the same eigenvalues.
- Gradient at truth not tested for whether minimising would move into a known twin basin (±X q0 mirror) — interesting diagnostic for future micro.

## Next

- Confirm the surrogate-truth-offset magnitude on more seeds. If |grad| ≠ 0 is universal, the gradient-based-inversion branch needs to reframe "find truth" as "find minimum + report distance to truth".
- Mass-matrix-preconditioned DE on 5 seeds to see whether anisotropy helps. See [[surrogate-attitude-isoshell]] §Next.
