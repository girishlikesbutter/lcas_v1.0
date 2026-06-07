---
title: "Gradient-based inversion — reframe the whole pipeline"
type: branch
sources:
  - "raw/inversion_diagnostics/m119v2/seed_014/summary.json"
  - "raw/inversion_diagnostics/m120/seed_000/summary.json"
  - "raw/inversion_diagnostics/m120/seed_014/summary.json"
  - "raw/inversion_diagnostics/m120/seed_027/summary.json"
  - "raw/inversion_diagnostics/m120/seed_046/summary.json"
  - "raw/inversion_diagnostics/m120/seed_058/summary.json"
  - "raw/inversion_diagnostics/m120/seed_075/summary.json"
  - "raw/inversion_diagnostics/m122/summary.json"
  - "raw/inversion_diagnostics/m123/summary.json"
  - "raw/inversion_diagnostics/m124/summary.json"
  - "raw/inversion_diagnostics/m125_keep_better/summary.json"
  - "raw/inversion_diagnostics/m126_wrapped/batch_summary.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/batch_summary.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_033/stage_a_grid.npz"
  - "raw/inversion_diagnostics/m128_warmstart_polish/batch_summary.json"
  - "raw/inversion_diagnostics/m129_densegrid/batch_summary.json"
  - "raw/inversion_diagnostics/m129_densegrid/seed_033/result.json"
related:
  - "[[differentiable-inversion]]"
  - "[[surrogate-model]]"
  - "[[m119v2_attitude_isoshell]]"
  - "[[m120_tumbling_competitors]]"
  - "[[m121_basin_width_metric]]"
  - "[[m122_hessian_curvature]]"
  - "[[m123_lbfgs_polish]]"
  - "[[m124_hifi_validate]]"
  - "[[m125_keep_better_inline]]"
  - "[[m126_wrapped_pipeline]]"
  - "[[m127_flipped_omega_search]]"
  - "[[m128_warmstart_polish]]"
  - "[[m129_dense_grid_eval]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[surrogate-de-search]]"
  - "[[multi-solution-philosophy]]"
  - "[[surrogate-truth-offset]]"
  - "[[dark-mag-saturation]]"
  - "[[omega-sign-degeneracy]]"
created: 2026-04-15
updated: 2026-04-17
confidence: high on m046 cohort (re-validated 2026-04-17 on correct 1-hour window; see "REINSTATED" block below). Generalization to diverse observation geometries is the open question — see [[m048-migration]] Phase 2.
---

> ## ✅ 2026-04-17 — `#validated` REINSTATED after Option A re-run on correct 1-hour window
>
> The April-16 retraction cited corrupted hi-fi MSE numbers. Option A (fix `end_time_utc` in m122/m124/m126 → re-run the m122 → m123 → m124 → m125 → m126 chain on the 1-hour window that matches `m46['mag_hifi']`) executed 2026-04-17. See `notebooks/inversion/DATA_INTEGRITY_BUG.md` and commits `5d5938f` → `b906691`.
>
> **The bug hid BETTER results, not worse.** Correct-window combined 11-seed cohort:
>
> | cohort | old (wrong-window) | new (correct-window) |
> |--------|---|---|
> | improved ≥10% | 6/11 | **10/11** |
> | break-even | 5/11 | 1/11 (seed 33 only) |
> | regressed | 0/11 | 0/11 |
> | basins helped | 18/33 | **33/33** |
>
> New per-seed classifications (hi-fi MSE): 5 OK (0, 6, 12, 74, 93), 4 PARTIAL (14, 24, 27, 33), 2 FAIL (36, 46). The wrapped pipeline is now a strict Pareto improvement *and* lifts most seeds into the OK/PARTIAL band. The m127/m128/m129 NOTE blocks below were always correctly windowed and are unchanged.
>
> Seed 33's flipped-ω attractor survives at hi-fi 0.0796 (basically unchanged from the wrong-window 0.0817 — confirming that degeneracy is a real physical phenomenon, not a bug artefact). Seed 33 is now the ONLY break-even seed in the cohort.


> # 2026-04-16 — NOTE: L-BFGS polish from coarse SO(3) grid is UNDERPOWERED for narrow basins ([[m127_flipped_omega_search]])
>
> [[m127_flipped_omega_search]] attempted to find flipped-ω compensating-q0 attractors on the 11-seed baseline cohort by running a 60k super-Fibonacci SO(3) grid (~3° median spacing) + top-20 L-BFGS-B polish per seed with ω fixed at `−ω_true`. The positive control (seed 33, known basin at hi-fi 0.082 from [[m126_wrapped_pipeline]]) **FAILED**: nearest top-20 grid quaternion was 32.7° from the known basin target; L-BFGS-B could not jump into a basin whose q0-width is sub-0.001° (per m126: `Δq0 ≤ 0.0004°`).
>
> **Decision-tree consequence.** Any future gradient-based approach on this landscape must either:
>
> - **(a) Warm-start from DE basins** (the proven-effective path — this is what [[m125_keep_better_inline]] / [[m126_wrapped_pipeline]] already do). Pure grid-then-polish loses narrow basins by construction.
> - **(b) Densify the grid 10–100×** (600k–6M points). Cost scales linearly in surrogate evals; not physically impossible but expensive.
> - **(c) Run DE before polish** on any ω-slice where narrow attractors are suspected. DE's population-based search is width-agnostic in a way gradient-based search fundamentally isn't on this cost landscape.
>
> [[m127_flipped_omega_search]] also produced one WIDE-basin confirmation: seed 12 at (q0_err 140.36°, ω=−ω_true), hi-fi 0.171, visible at 3° grid spacing. So the L-BFGS-from-grid method is not useless — it finds wide basins. It just cannot be used as a narrow-basin enumerator.
>
> ---
>
> # 2026-04-16 — NOTE: warm-start polish with negated-ω also cannot find flipped-ω basins ([[m128_warmstart_polish]])
>
> [[m128_warmstart_polish]] tested option (a) above with ω negated: 3-DOF L-BFGS-B at `(q0_m115basin, −basin_ω)` on 3 basins × 11 seeds. Verdict: **REFUTED on 10 valid-test seeds; seed 33 INVALID by construction** (census in [[omega-sign-degeneracy]]: seed 33's m115 basin ω is already retrograde, so `−basin_ω` is forward — not a flipped-ω test). All 10 valid tests hit FLIPPED_FAIL (best_hifi 1.42–4.73; sanity-control seed 12 missed its 0.3 target with actual 1.42).
>
> **Polish mechanics verified OK** (inline spot-check re-eval at `(q0_m115basin_0, +basin_ω)` for seed 33 reproduces m115's 0.082 within 0.97%).
>
> **Broader implication for the branch:** option (a) — warm-starting from known basins — works **only when the warm-start state and the target attractor are on the same cost manifold**. For forward-ω → flipped-ω transitions the two attractor families are on different ω-slices of the surrogate cost surface, and the 3-DOF q0-only polish cannot cross between slices. Even the 6-DOF polish would likely be trapped at the nearer local minimum because DE warm-starts are specifically NEAR forward-ω basins. Reaching flipped-ω attractors requires starting **already inside** a flipped-ω basin — which requires a population-based or grid-based enumerator on the `−ω_true` slice. See [[omega-sign-degeneracy]] Open-Questions for m129/m130 plans.
>
> Neither m127 nor m128 invalidates the wrapped pipeline as a default for forward-ω inversion; both only close specific flipped-ω-enumeration strategies. The `#validated` status of the wrapped pipeline from [[m126_wrapped_pipeline]] stands.
>
> ---
>
> # 2026-04-16 — NOTE: 10× SO(3) grid density does NOT fix grid-then-polish on narrow basins ([[m129_dense_grid_eval]])
>
> [[m129_dense_grid_eval]] ran a 600 000-point super-Fibonacci SO(3) grid (~1.4° median spacing, 10× denser than [[m127_flipped_omega_search]]'s 60k ~3° grid) + top-50 L-BFGS-B polish on seed 33 alone, ω fixed at `−ω_true`. Outcome: **REFUTED** — winner hi-fi 2.6186 (FLIPPED_FAIL, threshold was 0.15).
>
> Critical numbers from `seed_033/result.json` and `batch_summary.json`:
>
> | quantity | m127 (60k) | m129 (600k) |
> |---|---|---|
> | Stage A best grid surr | 1.1833 | 1.1800 |
> | Stage B best polish surr | 0.978 | 0.9804 |
> | Hi-fi winner MSE | 2.578 | 2.6186 |
> | winner q0_err | 91.43° | 110.25° |
>
> 10× density delivered 0.3% improvement in the best grid score. The known basin (q0-width sub-0.001°) is still ~1000× narrower than the densest grid cell; the landscape around the basin is saturated at surrogate cost ~1.0 (see [[dark-mag-saturation]]), so polish from any nearby grid point walks to a different local minimum (basin 1 at q0_err 91.55° is only ~7° in q0-geodesic from the known 98.53° target, yet its surrogate cost is 0.9989 vs the known 0.086 — no gradient connection).
>
> **Decision-tree consequence: option (b) (densify grid 10–100×) is REFUTED as a narrow-basin-enumeration strategy.** Arithmetically: reaching the known 0.001°-wide basin via uniform SO(3) sampling would require ~600 million grid points (~60 GB of surrogate-forward scratch) — not a practical path. Option (c) — population-based search (DE) on the `−ω_true` slice — is now the **only remaining enumerator** for narrow flipped-ω basins. m130 (DE-over-q0 at ω=−ω_true, 11 seeds, pop 200, 10 restarts/seed, ~30–45 min) is the required next experiment. If m130 also fails, the wiki should close the flipped-ω investigation with "narrow flipped-ω basins are not enumerable by any tested search method." Seed 12's WIDE basin from [[m127_flipped_omega_search]] remains the only search-reachable flipped-ω attractor on the 11-seed cohort.
>
> **Wrapped pipeline unaffected.** `#validated` status stands — m129 is narrow-basin-focused and only closes one specific strategy.
>
> ---
>
> # 2026-04-17 — PROMOTED to `#validated` via [[m126_wrapped_pipeline]] on corrected 1-hour window
>
> Option A re-run of the m122 → m123 → m124 → m125 → m126 chain on the correct
> 1-hour window completed 2026-04-17 (commit `b906691`). Architecture
> `DE basins → L-BFGS polish → hi-fi(before, after) → keep_min per basin → rank`
> is the recommended default pipeline.
>
> Full 11-seed baseline cohort (correct-window numbers):
>
> | cohort | seeds | count |
> |--------|-------|------:|
> | improved ≥10% | 0 (97%), 6 (98%), 12 (99%), 14 (90%), 24 (69%), 27 (93%), 36 (35%), 46 (78%), 74 (97%), 93 (88%) | **10/11** |
> | break-even | 33 (2.5% — flipped-ω degeneracy) | 1/11 |
> | regressed | — | 0/11 |
>
> All 33 polish basins improved (or wrapper caught the one case per basin
> where polish hurt). Hi-fi classification distribution: **5 OK** (0, 6, 12, 74, 93),
> **4 PARTIAL** (14, 24, 27, 33), **2 FAIL** (36, 46).
>
> Seed 33 is the ONLY break-even case: its m115 basin sits at the flipped-ω
> attractor (see [[omega-sign-degeneracy]]) where polish deepens a surrogate
> modelling-error pocket rather than the physical truth. The wrapper keeps
> pre-polish 0.0796 and the degeneracy persists on correct window — confirming
> it's a real observational symmetry, not a bug artefact.
>
> Seed 36 and 46 remain FAIL (hi-fi 0.39 and 0.14 respectively) — upstream
> ω-direction error of ~5–10° keeps these off any truth-adjacent attractor.
> Wrapped pipeline cannot fix bad DE; the next lever is a better ω-candidate
> harvest ([[harvester-optimization]]) for the seeds where m102 gave only one ω.
>
> Mechanism per class:
>
> - **Standard ω-mag refinement (8 basins across 5 seeds):** L-BFGS tightens
>   |ω|-magnitude by 5–14× (e.g. seed 0: 22.99%→8.1% uniform across 3 basins).
>   q0 locked at DE attractor, ω-direction saturated. Hi-fi drop driven by reduced
>   attitude drift over the 3600 s window.
> - **Gradient-bearing edge case (seed 6 basin 1):** atypical — polish moves
>   q0 by 3.46°, ω-dir by 2.05°, ω-mag by 14.6% jointly. Starts from q0_err=5.1°
>   (inside the q0 gradient-bearing band per [[basin-of-attraction]]) and
>   upstream ω-dir 3.08° (plateau but steeply-sloped in this direction).
>   End state: q0=1.64°, ω-dir=1.03°, ω-mag=0.6% → hi-fi 0.017.
> - **Dark-mag catastrophe (seeds 12, 36 — 5 basins):** surrogate-cost gradient
>   on the saturated plateau is anti-correlated with hi-fi gradient → polish
>   drives hi-fi 0.3–0.7 → 3.5–4.4 (10× worse). `keep_min` rescues.
> - **Flipped-ω degeneracy (seed 33):** backwards ω (signed 161.5°) with
>   compensating q0 produces near-OK hi-fi fit (0.082). Polish deepens a
>   different surrogate local minimum, hi-fi marginally worsens. See
>   [[omega-sign-degeneracy]].
>
> **Convention note (load-bearing):** m126's signed w_dir (0–180°) vs
> [[m115_surrogate_pipeline]]'s axis-angle (0–90°) differ when ω is retrograde. Seed 33 was
> 18.5° (axis-angle) but 161.5° (signed). The signed convention is recommended
> for all forward work.
>
> ---
>
> # 2026-04-16 — REINSTATED via [[m125_keep_better_inline]] [inline] wrapper analysis
>
> [[m124_hifi_validate]]'s ratio-agreement metric was the WRONG product-level question. The
> right question is "does `polish + hi-fi(before, after) + keep_min` improve the
> seed-level best hi-fi MSE vs plain [[m115_surrogate_pipeline]]?". [[m125_keep_better_inline]] [inline]
> re-scored the existing m124 data with this wrapper:
>
> | seed | m115 best | wrapped best | improvement |
> |-----:|----------:|-------------:|:-----------:|
> | 14 | 0.1613 | **0.0162** | **90%** |
> | 27 | 0.3105 | 0.3105 | break-even (wrapper rejects seed-27 catastrophe) |
> | 74 | 0.3760 | **0.2534** | **33%** |
> | 93 | 0.0626 | **0.0187** | **70%** |
>
> 9/12 basins helped by polish, 3/12 hurt (all seed 27, wrapper rejects them).
> Status restored to `#open-active`. The correct architecture is DE → polish →
> hi-fi(before, after) → keep_min per basin → rank. This is a strict improvement
> on plain [[m115_surrogate_pipeline]] on 3/4 seeds tested.
>
> **Mechanism (honest, calibrated 2026-04-16):** L-BFGS polish moves ONLY the
> |ω|-magnitude error. q0 error is unchanged in 12/12 basins (DE's attractor
> selection is respected). ω-direction error is unchanged in 11/12 basins
> (only seed 14 basin_1 moved: 0.34°→0.14°). |ω|-magnitude error tightens
> 5-10× (e.g. 0.12% → 0.01% on seed 14), which compounds as attitude-drift
> over the 3600s window and drives the hi-fi MSE reduction. Earlier "polish
> ω by ~0.05° direction + ~0.2% magnitude" phrasing was a parameterisation
> artifact — internal tangent-space motion not matching physical error-vs-truth.
>
> **Honest per-seed solution count (hi-fi MSE < 0.1, truth excluded), full 11-seed cohort post-[[m126_wrapped_pipeline]]:**
>
> | seed | wrapped best | basins < 0.1 | notes |
> |-----:|-------------:|:------------:|-------|
> | 0    | 0.112 | 0/3 | q0_err=6.7° on truth-adj basin → residual dominated by q0 offset; ω-mag tightened by polish but q0 locked |
> | 6    | 0.017 | 1/3 | basin_1 polish escaped to OK class (unique gradient-bearing case); twins stay ~0.08–0.09 |
> | 12   | 0.327 | 0/3 | plateau catastrophe on all 3; upstream ω-dir=8° |
> | 14   | 0.016 | 2/3 | truth-adj + ±X twin |
> | 24   | 0.025 | 2/3 | basins 0, 1 both tight (twin + truth-adj); basin 2 wrapper-rejected |
> | 27   | 0.311 | 0/3 | plateau catastrophe 3/3; all wrong-q0 |
> | 33   | 0.082 | 1/3 (flipped-ω) | flipped-ω attractor hi-fi 0.082 on basins 0, 1 per [[omega-sign-degeneracy]] — valid under [[multi-solution-philosophy]] but not truth-adjacent |
> | 36   | 0.602 | 0/3 | plateau catastrophe 2/3; upstream ω-dir=10.7° |
> | 46   | 0.651 | 0/3 | plateau-like, all wrong-q0 at 21°, 162°, 171°; polish hurt 0.65 → 3.24 |
> | 74   | 0.253 | 0/3 | ~100× noise; upstream ω 4.88° off |
> | 93   | 0.019 | 3/3 | truth-adj + two twin-like |
>
> Summed basin count below 0.1: **9/33 basins** across the 11 seeds. OK-class (≤ 0.01): **0/33** (tightest is seed 14 basin_1 at 0.016). PARTIAL-class (≤ 0.1): 9/33.
>
> [[m124_hifi_validate]]'s REFUTED verdict stands for its stated hypothesis (ratio
> agreement); the product-level verdict is DIFFERENT.

> # Superseded: 2026-04-16 — m124 "REFUTED" framing
> The earlier banner below described the ratio-agreement finding; it is historically accurate but was not the right product-level metric. See the m125 banner above. Kept for provenance.
>
> [[m124_hifi_validate]] hi-fi-validated the 12 polished DE-basin candidates from [[m123_lbfgs_polish]]. Result: only 2/12 basins had hi-fi MSE reduction agreeing with surrogate MSE reduction within ±30% (log-ratio). On seed 27, all 3 basins were CATASTROPHIC: surrogate reported 1.15× cost reduction while hi-fi MSE got **12–16× WORSE** (0.31 → 4.5–5.1 mag²). The L-BFGS-B polish drove candidates AWAY from any hi-fi minimum.
>
> Truth-start polish remains SAFE on all 5 seeds (hi-fi MSE within 5% of σ² noise floor). The "DE+GD hybrid as a *local* refiner with hi-fi safety" interpretation is what survived; m125 confirms that architecture.

> # 2026-04-15: un-retracted
> The earlier retraction was correct at the time: m119 v1 had a time-mismatch bug. Since then [[m119v2_attitude_isoshell]] fixed the geometry (truth rank 0/60000 at honest 6-hr grid) and [[m120_tumbling_competitors]] extended the result to the multi-seed ATT_FAIL cohort (truth rank 0/10004 for all 6 seeds {0, 14, 27, 46, 58, 75} under all 13 cost variants). The cost-function prerequisite is now met. The branch stays #open-horizon because the remaining prerequisites (autograd path, differentiable ODE) are implementation work, not research gates.

# Branch: gradient-based inversion

## Status: #validated — 10/11 improved ≥10%, 0/11 regressed, 33/33 basins helped on correct 1-hr window (2026-04-17 re-run)

> **2026-04-17 (evening) — Phase B Phase 1 landed.** [[m048-migration]] opened as its own branch. Phase 1 plumbing (`lib.traj_source`, `setup_experiment(start_et=...)`, `invert.py` driver, TRAJ_SOURCE env threaded through m115/m126/lc_compare) is complete; back-compat on m046 verified bitwise. Every `#validated` cohort claim below still reads "100 trajectories on ONE observation night" until [[m048-migration]] Phase 2 pilot lands. Phase 2 is blocked by `m115.load_omega_candidates` having no m048 fallback (m103/m102 harvests are m046-only). See [[observation-geometry-sources]] for the m046/m048 dichotomy.

**The "promoted to `#validated` after [[m126_wrapped_pipeline]]" claim below is on hold.** m126's hi-fi MSE numbers were scored on a 6-hour observed LC instead of the 1-hour `m46` truth (DATA_INTEGRITY_BUG.md Bug 2). Until m126 is re-run on the correct window (Option A) or the architecture migrates to m48 (Option B), this branch's status reverts to `#open-active` — the wrapped pipeline still RUNS (geometry is correct), but the "Pareto improvement" claim is unverified.

**Original promotion text below — preserved as a record of what the bug let us conclude.** The wrapped pipeline
`DE → polish → hi-fi(before, after) → keep_min` is a strict Pareto improvement
over plain [[m115_surrogate_pipeline]]: across the full 11-seed baseline cohort (seeds 0, 6,
12, 14, 24, 27, 33, 36, 46, 74, 93), **6/11 improve ≥10%, 5/11 break-even,
0/11 regress**. The wrapper turns a previously-dangerous polish step into a
safe, consistently-non-regressive addition to the pipeline.

Usage:

- **Default for all surrogate-DE-based pipelines.** Plain polish (take
  `hifi_after`) is REFUTED ([[m124_hifi_validate]]). Wrapped polish (`keep_min(before,
  after)`) is VALIDATED — always use the wrapper.
- **Predictable failure mode for polish itself:** plateau catastrophe
  (upstream ω-dir ≥ 5°, all basins share ω, all q0 far from truth) → polish
  drives hi-fi 10× worse. The wrapper catches these without cost.
- **Flipped-ω is a separate degeneracy** (seed 33; see
  [[omega-sign-degeneracy]]). Not a wrapper failure — a class of valid
  alternative solutions that happens to satisfy hi-fi threshold <0.1 while
  ω is retrograde.

The prior "DE+GD replaces NM+geo+hi-fi" framing is still OUT — the JAX port
for 6-DOF gradient descent replacing the full pipeline is not on the table.
Validated scope is narrower: polish is an **intra-attractor ω-magnitude
refiner** that occasionally (seed 6 basin 1) escapes an attractor when
initial conditions place it on a gradient-bearing plateau slope.

## The question

Can the entire inversion pipeline (grid + NM + geo + phi + hi-fi) be replaced by direct gradient-based optimisation on the surrogate cost, since the surrogate is differentiable?

## Why this is worth tracking now

- [[m119v2_attitude_isoshell]] confirmed at honest 6-hr geometry that `mean_L1` of surrogate residual over 255 epochs puts truth at rank 0/60000 (seed 14, static-rotation pool; tautological but geometry-correct).
- [[m120_tumbling_competitors]] removed the staticness concern with a tumbling-competitor pool and proved truth rank 0/10004 across 6 seeds of the ATT_FAIL cohort under all 13 cost variants.
- The surrogate is a neural network — differentiable in principle.
- The five-stage pipeline was designed for a black-box forward model. That assumption is obsolete.

See [[differentiable-inversion]] for the full framing.

## Prerequisites

1. ~~**Multi-seed m119**~~ **DONE** — [[m120_tumbling_competitors]] landed 2026-04-15, multi-seed ATT_FAIL cohort, truth rank 0/10004 for all 6 seeds under `mean_L1` and 12 other variants. The cost function is a universal scorer on the seeds where it matters most.
2. ~~**Surrogate fidelity audit** against noiseless hi-fi~~ **DONE (inline, 4 seeds)** — MAE ~0.03 mag, dim-regime MAE lower than bright. See [[surrogate-model]] concept page.
3. **Autograd path for the surrogate.** Currently `surrogate.py` is pure numpy. Options:
   - Port forward pass to JAX (easiest; a 239k-param MLP is trivial)
   - Port to PyTorch (most familiar tooling for inverse problems)
   - Derive analytical gradients through the MLP weight matrices (no extra dep, but custom code for ReLU/tanh/etc.)
4. **Differentiable attitude propagation.** `propagate_attitude(q0, ω, t, I_tensor)` integrates an ODE. For gradients w.r.t. (q0, ω), either:
   - Use an adjoint method (standard for neural ODEs)
   - Derive analytical gradients for the tumbling case (might be tractable given known I_tensor)
   - Use a simpler kinematic model where applicable (pure rotation around a fixed axis when inertia tensor is effectively spherical — not our case)

## What it would replace

See [[differentiable-inversion]] for the full table. Collapses five stages to one.

## What it would NOT replace

- **Constraint epoch selection.** Choosing which subset of the 500 observation epochs to trust / weight remains an open question independent of the optimiser.
- **Multi-solution enumeration.** Gradient starts or MCMC give this, but it's still a distinct concern.
- **Hi-fi validation.** We might still want one hi-fi eval on final candidates as a sanity check.

## Proof-of-concept spec (once prerequisites are met)

Label: **m001XX — gradient-based inversion POC on seed 14.**

Setup: JAX-ify the surrogate forward pass. Wrap propagate_attitude with a differentiable ODE solver (jax.experimental.ode or diffrax).

Cost: `f(q0_wxyz, ω_rad) = mean_L1 of (surrogate_magnitude - observed_mag)` over constraint epochs.

Experiment: start L-BFGS or Adam from 100 random (q0, ω) seeds. Record which starts converge to which basins. Compare basin count and basin quality against [[m115_surrogate_pipeline]]'s 10-start DE (~4 basins/seed). Report wall-clock time, number of forward evals, and hi-fi MSE at each basin.

Success: fewer forward evals than DE, more basins found, comparable or better hi-fi MSE.

## Decision tree

- ~~**Multi-seed m119 validates the cost**~~ **passed via [[m120_tumbling_competitors]]** → activate this branch when autograd+diffrax infra is ready.
- ~~**Basin-width characterisation pending**~~ **[[m121_basin_width_metric]] landed 2026-04-15.** Results change the feasibility story (see below).
- **JAX port trivial, POC faster + more basins than DE** → replace [[surrogate-de-search]] as the primary attitude-finder.
- **JAX port works but POC comparable or worse than DE** → keep DE; the marginal win isn't worth the new infra.

## Basin-width prerequisite (new 2026-04-15, from [[m121_basin_width_metric]])

[[m121_basin_width_metric]] measured the cost basin directly on 3 ATT_FAIL seeds. Headline:

- **ω-direction basin < 0.1°** (1σ noise-floor test). Cost saturated by 0.5°.
- **ω-magnitude basin < 0.25%.** Cost saturated by 1%.
- **q0 basin ≳ 5°** — gradient-bearing to ~20°.
- **ω-direction basin is anisotropic** (15-25× cost ratio across rotation axes) with seed-specific preferred direction.

### Implication for gradient-based init

Gradient-based search (L-BFGS, Adam, HMC) requires the initial (q0, ω) to lie within the basin where the cost gradient points toward truth. Outside the basin the cost saturates at ~2.0 mean_L1 and the gradient is ~zero (no signal). The basin widths above set the **required accuracy of the initialiser**:

| Parameter | Required init accuracy | Classical pipeline ([[m102_fullmse]]) achieves | Gap |
|-----------|:---:|:---:|:---:|
| ω-direction | < 0.1° | 0.3°–3° (OK seeds), 5°+ (ATT_FAIL) | 3× to 30× short |
| ω-magnitude | < 0.25% | 0.1%–0.7% | adequate ⟷ marginal |
| q0 | < 5° | 0.15°–3° (OK seeds) | adequate |

**This is a HARD prerequisite, not an optimisation.** Gradient descent from a random start will spend most of its compute on the flat saturated plateau and will not find the basin without a warm start. Concretely:

- Pure L-BFGS cold-start from random (q0, ω): expect failure. Gradient is degenerate.
- L-BFGS warm-started from classical pipeline NM output: will converge on q0 and ω-magnitude, may fail on ω-direction.
- L-BFGS after surrogate-DE refinement: likely works — DE basins in [[m115_surrogate_pipeline]] land ω-direction at ~0.3-3° which is still outside the surrogate basin. Need a post-DE polishing stage specifically tuned for ω-direction (finer DE or 1-DOF search around ω̂).
- HMC / MCMC: sampler gets stuck on the saturated plateau same as gradient descent. Would need mass-matrix anisotropy tuning informed by the basin anisotropy (seed-specific). Not a turnkey fix.

### Post-m123: surrogate minimum IS truth (correcting m122)

**Retraction of the earlier "Truth is NOT the cost minimum" claim.** [[m123_lbfgs_polish]] directly measured the displacement that [[m122_hessian_curvature]] predicted: run L-BFGS from truth on all 5 seeds and see where it converges. Result: q0 moves ≤5e-6°, ω-dir moves ≤2.3e-4°, ω-mag moves ≤4.3e-4% — at or below the FD noise floor. Truth IS the surrogate's local minimum in physical units.

The [[m122_hessian_curvature]] gradient and negative-eigenvalue findings were artifacts of the parameterisation, not physical offsets:

- Gradient in parameter-space units (rad for quaternion/ω-dir tangent, fractional for ω-mag) translates to physical displacement via `Δ = |grad| / λ`, using eigenvalues in the matching units.
- Example: seed 27 has `‖grad‖ = 40` dominated by the stiff ω-dir eigendirection (`λ ≈ 1.11e7`). Predicted `Δ = 40 / 1.11e7 = 3.6e-6 rad ≈ 2.06e-4°`. [[m123_lbfgs_polish]] measured seed-27 truth-start ω-dir final error at `2.28e-4°`. Match.
- Seed 46's `λ_min = −203` is below the FD second-difference noise floor on `cost ≈ 0.05` with `h = 1e-4`: noise floor `~cost_eps / h² ≈ 1e-2`. Magnitude is 20000× above, but the sign is a floating-point-cancellation artifact — L-BFGS from truth on seed 46 showed no saddle behaviour (2 iterations, zero q0 move).

See [[surrogate-truth-offset]] for the corrected concept page.

**HMC / Bayesian sampling does NOT inherit an offset** (we were worried it would; it does not). Posterior mode is at truth modulo the surrogate's 0.03 mag MAE, which is below the discriminating threshold.

### The real architecture: DE + GD hybrid (new 2026-04-16, from [[m123_lbfgs_polish]])

[[m123_lbfgs_polish]] ran L-BFGS-B from the m115 DE basins in addition to truth, on 5 seeds. Key findings:

- **DE enumerates q0 attractors.** Each m115 basin sits at a distinct q0 well separated from truth (and from other basins) by 2°–180°. L-BFGS-B cannot move between q0 attractors: q0 moves <1e-3° across all 12 basin starts. Pure GD is trapped.
- **L-BFGS polishes ω within each attractor.** From DE basins with ω-dir error ~3° and ω-mag error 10–50%, L-BFGS reduces ω-dir error by ~0.02–0.21° and ω-mag error by ~0.09–0.32% per start. Cost reductions 1.14×–12.9× per basin (most 4–7×).
- **Polish works despite being OUTSIDE the [[basin-of-attraction]] ω-dir basin** (<0.1° at the 1σ noise floor). The [[dark-mag-saturation]] plateau cost of ~2.0 is not truly flat — it has a small-but-nonzero gradient in ω. L-BFGS with FD Jacobian follows that gradient. q0 doesn't move because (a) q0 is already at an attractor where its gradient is zero by definition, and (b) q0's gradient-direction component is smaller in the saturated regime than ω's.

**Working architecture for gradient-based inversion on this problem:**

1. DE on surrogate cost → enumerate q0 attractors (m115 pattern).
2. L-BFGS-B polish on each DE basin → refine ω to the surrogate's local optimum within that attractor.
3. Hi-fi validation on the polished candidates → rank and select.

DE is NOT a "refinable initialiser for GD". It is an _attractor enumerator_. GD is the _intra-attractor polisher_. The two are complementary, not sequential-in-the-usual-sense.

### Restated decision tree (post-m126)

- **Cheap feasibility test before JAX port:** ~~take m115 DE basins → polish with pure-numpy surrogate-MSE gradient descent~~ **DONE in [[m123_lbfgs_polish]]. Passed at the surrogate level.**
- **Hi-fi validation of polished basins:** **DONE in [[m124_hifi_validate]] + [[m125_keep_better_inline]] + [[m126_wrapped_pipeline]]. WRAPPED answer: YES.** Naive polish (take `hifi_after`) fails (2/12 agree within ±30% in [[m124_hifi_validate]]). Wrapped polish (`keep_min(hifi_before, hifi_after)`) across the 11-seed baseline cohort: **6/11 improved ≥10%, 0/11 regressed**. Strict Pareto improvement.
- **Recommendation:** make the wrapped pipeline the default for surrogate-DE attitude finding. Cost: 1 extra hi-fi eval/basin (~80 s/basin wall). Gain: up to 90% hi-fi MSE reduction where polish works; zero cost where it doesn't.
- **Catastrophe predictor (free at DE time):** upstream ω-dir ≥ 5° + single upstream candidate + all 3 basin q0_err ≥ 10° → expect plateau catastrophe; could skip polish to save time if the wrapper is deemed unnecessary for confirmed-catastrophe basins. (Wrapper still catches it, so optional optimisation.)
- **Flipped-ω detection (free at DE time):** signed `θ = acos((ω̂_cand · ω̂_true))` > 90° → retrograde ω attractor. This is a valid local minimum of the surrogate under its own modelling-error landscape; a separate degeneracy from truth. See [[omega-sign-degeneracy]]. Exists in at least seed 33; seed 46 check pending.
- **JAX port remains not-load-bearing.** Justification was 6-DOF gradient descent replacing the pipeline. With the wrapped pipeline validated as the default, JAX is optional (faster polish, HMC exploration of the multi-solution posterior, etc.) but not required for the primary inversion task.
- **Next research question:** can the ω-direction basin itself be widened by re-training the surrogate on wrong-attitude examples, so polish can escape more attractors (not just seed 6's edge case)? Pre-question: does the surrogate's `(k1, k2)` training distribution have coverage holes at the geometries hit by seeds 12, 36, 46 basins? Tracked under [[surrogate-model]].

## Literature pointers

- Mitsuba 3 (differentiable physically-based renderer)
- NeRF + inverse rendering variants (attitude-from-images is conceptually similar to pose-from-images)
- Simulation-based inference packages: `sbi` (PyTorch), `BayesFlow`
- Hamiltonian Monte Carlo: NumPyro, PyMC, Stan
- Neural ODEs and adjoint methods: `diffrax`, `torchdiffeq`
