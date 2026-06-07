---
title: "s078 — RF25 NLL residual A/B: does not help on seed 89, do not adopt into the m048 cost path"
type: experiment
sources:
  - results/s059k_nd800_seed89/seed089/polished_states.npz  (20 cached s069 polished states)
  - results/s059k_nd800_seed89/seed089/clusters.npz          (5000 score-grid candidates)
  - experiments/s073e_robinson_frueh_2025_full_audit.md      (RF25 Eq. 3 spec)
related:
  - experiments/s073e_robinson_frueh_2025_full_audit.md
  - experiments/s077_l_vector_basin_sweep.md
created: 2026-05-14
updated: 2026-05-14
confidence: high for the negative result on seed 89 (3-stage A/B with a proper full-LC-MSE control); N=1 seed — see Out of scope
---

# TL;DR

Transplanted Robinson & Frueh 2025's Eq. 3 NLL loss (flux-space negative
log-likelihood with σ_k weighting + ‖S‖/‖Ŝ‖ signal rescaling) as an **opt-in**
`lib/surrogate_eval.py::nll_residual` / `nll_cost` primitive — production default
left unchanged — and ran the staged A/B the user asked for. **It does not help on
seed 89, and the A/B says: do not adopt it into the m048 production cost path.**

- **Step 1a (re-score 20 cached polished states):** plain magnitude MSE tracks the
  hi-fi ρ ground truth perfectly (Spearman 1.000); NLL-rescale tracks it *worse*
  (0.555). NLL adds a free flux-scale DOF that decorrelates the score from hi-fi ρ.
- **Step 1b (re-score 5000 score-grid candidates):** the truth-nearest grid cell
  ranks #1710 under the cached local-window score, and *worse* under every full-LC
  score — #3649 (full-LC MSE), #3262 (NLL-rescale), #2929 (NLL-no-rescale). NLL is
  marginally less-bad than full-LC MSE but does not rescue truth.
- **Step 2 (re-polish 20 non-Band-A states, NLL vs full-LC-MSE control):** the
  full-LC-MSE control beats the NLL polish on surrogate ρ in **19/20** cases.
  Neither reaches band A/B (0/20 both); the MSE control gets one state to band C,
  NLL gets none past D.

**Mechanism (matches the primitive's own scope note):** with the m048 *fixed*
0.05-mag noise floor, σ_k comes out flux-proportional, which makes the σ-weighted
χ² ≈ plain magnitude MSE to first order — the σ-weighting contributes nothing. The
‖S‖/‖Ŝ‖ rescaling — the genuine lever — *hurts* here because seed-89's phantom
basins differ from truth in LC *shape*, not *scale*; removing a global scale DOF
just lets bad fits look less bad. RF25's NLL needs a genuine per-timestep varying
σ_k (their full noise model, RF25 §3.5.1) to deliver — and that is the
`lib/noise_rf.py` piece the s073e audit explicitly scoped out as cohort-invalidating.

The primitive stays in `lib/` as opt-in for a future real-data thread. The
staged-A/B-before-adoption discipline did its job: a 30-line transplant that looked
attractive on paper is shown not to earn a production swap.

# What

s073e's Part 1 #1 ranked RF25 Eq. 3 the single highest-value implementation
transplant — "directly addresses bright-peak overfitting empirically seen in
s058/s059." The user asked for a **minimal** version (opt-in primitive + A/B only,
no production swap) and a **staged** validation (re-score cached states first,
re-polish only flagged cases).

# How

**Primitive.** `lib/surrogate_eval.py::nll_residual(pred_mag, truth_mag,
sigma_mag=0.05, rescale=True)` and `nll_cost(...)`, re-exported from
`lib/cost_surfaces.py`. Implements RF25 Eq. 3 on relative flux
`S = 10**(-0.4·mag)`: σ_k from the fixed mag floor propagated to flux
(`σ_k = S_k · ln(10)/2.5 · σ_mag`, flux-proportional), optional ‖S‖/‖Ŝ‖ rescale of
the prediction, least-squares residual `r_k = √½ · (S_k − Ŝ_k)/σ_k`. The `rescale`
flag isolates the σ-weighting effect from the rescaling effect. Unit-checked:
zero residual at perfect match; with `rescale=True`, `nll_cost` is invariant to a
global magnitude offset of the prediction; first-order `Σr² ≈ ½·magMSE·N/σ²`
(confirmed to 0.5%). Production default (`full_lc_mse`) untouched.

**Staged A/B (all on seed 89, post-fix cached data):**
- *Step 1a* — re-score the 20 cached s069 polished states (`pred_hifi` cached →
  pure file-load) with plain MSE / NLL-rescale / NLL-no-rescale; Spearman vs the
  cached `rho_polished_hifi`.
- *Step 1b* — render all 5000 `clusters.npz` score-grid candidates (Pool(24)) and
  re-score; locate the truth-nearest grid cell (`truth_idx_diagnostic = 1709`)
  under each score; cached `mse_top` (local-window) is the control.
- *Step 2* — re-polish the non-Band-A polished states with `scipy.least_squares`,
  **both** the NLL residual and a full-LC-MSE residual control (the cached states
  were *local-window* MSE polished, so the MSE control is required to attribute any
  change to NLL rather than to full-LC-vs-local-window).

Script: `experiments/s078_nll_residual_ab.py`. Compute wall: ~11 min total
(Step 1b 31.5 s + Step 2 621.5 s, both Pool(24)).

# Result

**Step 1a — re-score 20 cached polished states (all band D, hi-fi ρ 27–48):**

| score | Spearman vs hi-fi ρ |
|---|---|
| plain magnitude MSE | **1.000** |
| NLL (rescale) | 0.555 |
| NLL (no rescale) | 0.878 |

Plain MSE on the surrogate perfectly rank-orders the hi-fi ρ ground truth on these
20 states; NLL-rescale scatters it. (`s078_step1a_score_vs_hifi.png`.)

**Step 1b — truth-nearest grid cell rank (of 5000):**

| score | truth-cell rank |
|---|---|
| cached local-window MSE (control) | **#1710** |
| full-LC magnitude MSE | #3649 |
| full-LC NLL (rescale) | #3262 |
| full-LC NLL (no rescale) | #2929 |

Every full-LC score ranks truth *worse* than the cached local-window score. NLL is
marginally less-bad than full-LC MSE but nowhere near useful.
(`s078_step1b_truth_rank.png`.)

**Step 2 — re-polish 20 non-Band-A states, NLL vs full-LC-MSE control:**

- Full-LC-MSE control beats NLL on surrogate ρ in **19/20** states.
- Band A/B reached: NLL **0/20**, MSE control **0/20** (the MSE control reaches
  band C on one state, cl38: ρ 43.5 → 6.99; NLL on the same state → 9.14, still D).
- Both improve on the cached local-window polish (grey bars in
  `s078_step2_repolish.png`) — but that improvement is the full-LC-vs-local-window
  effect, and the MSE control captures it better than NLL does.

# Why this matters

1. **Clean negative result with a proper control.** The first (control-free) run
   showed NLL "improving" 16/20 states, which looked positive. Adding the full-LC-MSE
   control showed the improvement was the full-LC-vs-local-window effect, not NLL —
   and that NLL is in fact *worse* than plain full-LC MSE (19/20). Without the
   control this would have been a false positive.

2. **The mechanism was predicted.** The primitive's scope note already stated that a
   flux-proportional σ_k (the only kind the m048 fixed noise floor can give) makes
   the σ-weighted χ² ≈ magnitude MSE. s078 confirms it empirically. The only
   non-trivial part of the transplant — the ‖S‖/‖Ŝ‖ rescaling — actively hurts on
   seed-89 phantom basins, which are *shape*-wrong not *scale*-wrong: a free global
   scale DOF lets a bad fit look less bad (Step 1a 1.000 → 0.555).

3. **It tells us what the real lever would be.** RF25's NLL pays off when σ_k
   genuinely varies per timestep (their Fig. 6c: 4–5× envelope variation across one
   LC). That needs RF25's full noise model — `lib/noise_rf.py` — which the s073e
   audit already flagged as `evaluate`, scoped to real-data, cohort-invalidating.
   s078 confirms there is no shortcut: you cannot get RF25's benefit from the fixed
   m048 floor alone.

4. **Decision: do not adopt into the m048 production cost path.** `full_lc_mse`
   stays the default. `nll_residual` / `nll_cost` remain opt-in primitives for the
   future real-data thread (paired with `lib/noise_rf.py` if that is ever built).
   The ρ-band convention is untouched. This is exactly the
   `feedback_stop_cost_shape_engineering` outcome: validate before swapping.

# Numbers

All from `results/s078/summary.json` and the console log unless noted.

- Step 1a: Spearman(score, hi-fi ρ) — MSE 1.000, NLL-rescale 0.555, NLL-no-rescale
  0.878; all 20 cached states band D (hi-fi ρ 27.24–48.15)
  (source: `results/s059k_nd800_seed89/seed089/polished_states.npz` `band`,
  `rho_polished_hifi`).
- Step 1b: 5000/5000 candidates finite; truth cell idx 1709; rank #1710
  (cached local-window) / #3649 (full-LC MSE) / #3262 (NLL-rescale) / #2929
  (NLL-no-rescale); render wall 31.5 s on Pool(24).
- Step 2: 20 flagged states, 40 polishes, wall 621.5 s on Pool(24); NLL beats
  MSE-control on surrogate ρ in 1/20; band A/B reached 0/20 (NLL) and 0/20
  (MSE-control); cl38 MSE-control ρ 43.46 → 6.99 (band C), NLL → 9.14 (band D).
- Primitive unit checks: perfect-match residual max-abs 0.0; `nll_cost` with
  rescale invariant to a +0.7-mag global offset (−13.0728 both); first-order
  `Σr² = 9.119` vs `½·magMSE·N/σ² = 9.160`.

# Out of scope

- **N=1 seed.** s078 ran on seed 89 only — and seed 89's polished states are all
  band-D phantom basins, an extreme case where plain MSE was already known to
  struggle. It is *possible* NLL helps on a seed whose error is genuinely a global
  *scale* error (BRDF albedo/area) rather than a shape error. But the dominant
  mechanism here — flux-proportional σ_k ⇒ σ-weighting ≈ MSE — is general, not
  seed-specific, so a broader sweep is low priority unless the real noise model is
  on the table.
- **RF25's full per-timestep noise model** (`lib/noise_rf.py`, RF25 §3.5.1). This is
  the change that would actually make NLL pay off — explicitly scoped out by s073e
  as `evaluate`, real-data only, cohort-invalidating. Not built here.
- **Production swap of the cost path.** Per the user decision, s078 was opt-in
  primitive + A/B only. The A/B says don't swap; the swap was never in scope.
- **Re-deriving the ρ-band thresholds for NLL.** Not needed — NLL is not adopted.

# Cross-references

- `experiments/s073e_robinson_frueh_2025_full_audit.md` — Part 1 #1 proposed this
  transplant; s078 is the A/B that evaluates it. The audit's own caveat (fixed
  floor vs per-timestep σ_k) is confirmed load-bearing.
- `lib/surrogate_eval.py` — `nll_residual` / `nll_cost` (the opt-in primitive, with
  the scope note); `lib/cost_surfaces.py` — re-exports.
- Memory: `feedback_stop_cost_shape_engineering.md` — s078 is the disciplined
  version: a published, principled loss, transplanted as opt-in, A/B'd against a
  proper control before any production decision.
- Memory: `feedback_surrogate_first_hifi_last.md` — A/B ran entirely on cached
  predictions + surrogate renders; no hi-fi optimisation.
