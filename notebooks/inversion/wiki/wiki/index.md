# LCAS Inversion Research Wiki — Index

> ## 🚨 2026-04-30 — CRITICAL CONVENTION BUG IN INVERSION-SIDE PROPAGATOR (FIXED, awaiting downstream review)
>
> The inversion-side propagator (`src/dynamics/attitude_propagator.py`) was integrating `dq/dt = (1/2) q ⊗ ω̄_body` (RIGHT mult) — the kinematic for **convention (b)** (`R(q) = R_b→i`, body→J2000). The renderer everywhere downstream (lc_compare.py, m048_generate_trajectories_v2.py, the main pipeline custom-quaternion path) computes `k1 = R(q) @ sun_J2000`, which is correct only under **convention (a)** (`R(q) = R_i→b`, J2000→body). Conventions (a) and (b) are inverses (`q_a = q_b*`). Fix is a two-line swap in `propagate_principal_axis` (line 167→173) and `propagate_euler` (line 287→298): RIGHT-mult → LEFT-mult Hamilton. After the fix, the inversion-side renderer matches the canonical main pipeline at machine precision (RMS 6.26e-15 mag) on m048 seed 6.
>
> **Implication:** every truth NPZ (`m046_trajectories.npz`, `m048_trajectories/per_trajectory/*.npz`), every `pred_lc.npy` cache, every `mag_hifi` ever produced on the inversion side, and every basin/ρ-band classification ever made against those was generated under the bug. The numbers are internally consistent with the buggy forward model but do not correspond to physical observations. **All previous classifications and pipeline verdicts are currently in suspended status until the truth NPZs are regenerated and inversions re-validated.**
>
> **What is NOT in scope here:** truth-NPZ regeneration, inversion-code review, surrogate-retraining, or convention runtime gates — all are scheduled for follow-on sessions. The current session's job was the audit + fix + validation only.
>
> **Read first:** [[m139_convention_bug_fix]] (the audit), [[quaternion-convention]] (the concept), and `CURRENT_STATE.md` (the next-session plan).

> ## 🎯 2026-04-29 — STRATEGIC REFRAME (read CURRENT_STATE.md first)
>
> The recent m136/m137/m138 cost-shape engineering thread is **suspended**. Designing new aggregator costs over alignment-cost / IPL-centroid / attitude-isoshell substrates produces anti-truth costs (m135 forensics, m136 dead-end, m138 cost-shape pathology). The validated fix is **surrogate full-LC MSE re-ranking of m103's existing pool** — three independent single-seed validations (m133 9/17→15/17, m135 lofi-pool, m138 surr-rerank). The m115 q-finder is not broken; m103's *ranking* of its own pool is.
>
> **Acceptance bar is ρ < 4 (Band A∪B), not ρ < 2.** Band C is publishable as an LC fit but flag the partial-state caveat. Headline cohort metric is **% seeds with ≥1 (A∪B) basin**.
>
> **Next-session priority:** Play 1 (consolidate the random m048 25-seed cohort using the m133 3-cost union ω-ranking + K=7 to m115 + ρ-band yield reporting). Plays 2 and 3 follow if Play 1 doesn't close the gap. Full plan in `notebooks/inversion/CURRENT_STATE.md`.
>
> **Don't:** design new aggregator costs over per-epoch attitude level sets / IPL centroids / kernel factorisations / harmonic divisions. Read `feedback_stop_cost_shape_engineering.md` first if tempted.

> ## ✅ 2026-04-17 — Bug 2 FIXED via Option A; all m122/m123/m124/m125/m126 numbers re-computed on correct 1-hour window
>
> Two compounding data-integrity bugs were discovered 2026-04-16. See `notebooks/inversion/DATA_INTEGRITY_BUG.md`.
>
> **Bug 1 (open):** all pipelines still use `m046_trajectories.npz` (fixed 10–11 UTC shared window) instead of the intended `m048_trajectories.npz` (per-seed random start 08:35–15:35 UTC, 1-hr duration). Every "population" / "cohort-universal" claim remains single-geometry. Option B (migrate m46→m48 + regenerate the arc) is the honest fix; deferred pending a separate decision.
>
> **Bug 2 (fixed):** `m122/123/124/126` were missing `end_time_utc='2020-02-05T11:00:00'` and silently ran on the 6-hour default. `m119v2` turned out to share the bug (bug doc's "m119v2 CORRECT" claim was wrong). Scripts patched 2026-04-17 (commit `5d5938f`); m122 → m123 → m124 → m125 → m126 chain re-run on correct window (commit `b906691`). The bug hid **better** results:
>
> | cohort | old (wrong-window) | new (correct-window) |
> |--------|---|---|
> | improved ≥10% | 6/11 | **10/11** |
> | break-even | 5/11 | 1/11 (seed 33 only — flipped-ω real) |
> | regressed | 0/11 | 0/11 |
> | basins helped | 18/33 | **33/33** |
>
> [[gradient-based-inversion]] #validated reinstated. All referenced wiki pages below carry "post-fix" numbers. Seed 0 m126: claimed 0.112 → regenerated 0.325 → fresh-polish on correct window 0.00367 (OK-class). Seed 33's flipped-ω 0.082 persists as 0.0796 post-fix — the degeneracy is physical.
>
> **Bug 1 open; Option B ([[m048-migration]]) Phase 2 pilot effectively complete on 8 seeds 2026-04-17.** 4 OK + 1 PARTIAL + 1 downstream-FAIL + 2 upstream-FAILs. Two distinct failure modes disentangled: high-phase (>65°) alignment-cost flatness (seeds 28, 69 — both had ≥10 spec peaks, not constraint-poor) and constraint-poor (seed 23 — only 2 spec peaks, 1 alignment constraint, geo_cost degenerate). Reliable band ≈ 38-55° phase with ≥4 spec peaks. **Phase 3 (100-seed batch) blocked on design decision**: patch m103 (expected 30-40% upstream-FAIL) vs replace with 6-DOF surrogate DE ([[upstream-redesign-6dof-surrogate-de]] #proposed). See [[phase_B_m048_cohort]] for the central results page.

## Narrative

[[journey]] — chronological research arc from m001 → m129, organised by era with revivable-dead-end flags and cross-era idea connections.

## Experiments

| Page | Summary | Status |
|------|---------|--------|
| [[m070_full_pipeline]] | Full pipeline success — q0=1.94°, w=0.14° in 7.8 min (seed 93) | Done |
| [[m073_alpha_pipeline]] | Multi-seed alpha — 4/6 omega <5°, 1/6 full success | Done |
| [[m087_m088_fast_grid]] | Speed opts — SLERP unsafe with NM, relaxed tol safe (~3×) | Done |
| [[m090_robust_peak_selection]] | 10-seed validation baseline — 4 OK + 3 PARTIAL + 3 FAIL | Done |
| [[m091_twin_state_test]] | Hi-fi twin test — only ±X produces identical LCs | Done |
| [[m092_twin_axis_visualization]] | Twin-axis diagnosis — seeds 6/24/36 are NOT valid +X twins | Done |
| [[m093_expected_dot_nelder_mead]] | BRDF cost discovery — 3 constraints per epoch, we use 1 | Done |
| [[m094_brdf_cost_function]] | Adaptive BRDF/alignment cost — 8000 dirs, multi-phi | Done |
| [[m095_grid_cost_diagnostic]] | NM_TOP=200 + multi-window hi-fi — rescues seeds 12, 27 | Done |
| [[m096_exp1_oracle_grid]] | 100-seed census + diagnostics — 87% lack bright ±X constraints | Done |
| [[m097_candidate_ranking]] | Lo-fi re-ranking — can't replace alignment cost at grid level | Done |
| [[m098_m099_nm_grid_pipeline]] | NM_TOP=300 + 2000-dir grid — seed 6 rescued | Done |
| [[m100_m101_batch_multi_phi]] | Multi-phi + selection — full-MSE > vote, multi-phi risky | Done |
| [[m102_fullmse]] | **Previous best:** NM_TOP=300 + full-window MSE — 3 OK + 3 PARTIAL + 4 FAIL | Done |
| [[m103_hybrid]] | Top-2 multi-phi + hybrid selection — superseded by surrogate-DE pipeline | Done |
| [[m104_crossing_diagnostic]] | Peak crossing geometry diagnostic — hypothesis refuted | Done |
| [[m106_pairwise_vec_ipl]] | Vectorized pairwise alignment — negative result, 1° omega error → 6/14 peaks | Done |
| [[m107_m108_ipl_cost]] | IPL centroid grid cost — FAILED: false positives from many centroids | Done |
| [[m109_ipl_phi_diagnostic]] | IPL phi discrimination diagnostic — ALL centroid metrics fail | Done |
| [[m110_hifi_phi_study]] | Sparse hi-fi phi sweep — seed 27 works, 4/5 others fail (wrong peaks) | Done |
| [[m111_shadow_isoshell]] | Anchor alignment error discovery — cos^250 amplifies 2° to 100-1200× noise | Done |
| [[m112_bestanchor_selection]] | Best-anchor phi re-sweep — FAILED: omega error corrupts distant-epoch PABs | Done |
| [[m113_de_attitude_search]] | 3-DOF DE attitude search — 0.6° ceiling w/ truth omega, paradigm shift | Done |
| [[m114_surrogate_multistart]] | Surrogate multi-start + 6-DOF cold start — 3-DOF positive, 6-DOF negative | Done |
| [[m115_surrogate_pipeline]] | **BREAKTHROUGH:** Surrogate multi-start DE — 10/10 valid, 6/10 truth-adjacent | Done |
| [[m117_result_harvester]] | Harvester sanity — Lever 2 (skip geo) REFUTED: NM-only pool misses truth basin (seed 14: 48.8° vs 0.34°) | Done |
| [[m118_cost_comparison]] | **Kernel-factored IPL cost diagnostic (seed 14)** — 6 cost variants, none put truth at rank 1; root cause is pab-contour phase-angle approximation (median 25° truth-to-centroid mismatch) | Done |
| [[m119_attitude_isoshell]] | **RETRACTED** — mixed m046 (1-hr) + setup_experiment (6-hr) geometry; superseded by [[m119v2_attitude_isoshell]] | Retracted |
| [[m119v2_attitude_isoshell]] | **Fixed POC (seed 14)** — truth rank 0/60000 under all 13 variants; honest residuals (median 0.043, p90 0.112 mag); static-grid test tautological, next step is tumbling-competitor rerun | Done |
| [[m120_tumbling_competitors]] | **Tumbling-competitor test (multi-seed: 0, 14, 27, 46, 58, 75)** — truth rank 0/10004 under ALL 13 cost variants for ALL 6 seeds. near_truth best 1 (all seeds), close_ω best 8001–8004 (all seeds), close_q0 best 51–158, uniform best 18–135. Surrogate-attitude-isoshell branch validated across full ATT_FAIL cohort | Done |
| [[m121_basin_width_metric]] | **Basin-width characterisation (seeds 14, 27, 46)** — surrogate-residual cost basin: q0 ≳ 5° (widest, gradient-bearing to 20°), ω-direction < 0.1° (narrow, saturated by 0.5°), ω-magnitude < 0.25% (narrowest). ω-direction is ANISOTROPIC (15-25× cost ratio across rotation axes) with seed-specific preferred body-frame axis. Hypothesis (ω-magnitude widest) REFUTED — opposite direction | Done |
| [[m122_hessian_curvature]] | **Hessian-at-truth (seeds 14, 27, 46, 74, 93)** — 6-DOF FD Hessian at truth. Hyp1 CONFIRMED (basin widths Hessian-tight, 10-100× narrower than m121 empirical — quadratic vs saturation). Hyp2 REFUTED 2/3 (local curvature ≠ finite-scale anisotropy). Hyp3 CONFIRMED (ATT_FAIL/OK basins within 2.11× — cohort-universal). Seed 46 has negative eigenvalue (−203); ‖grad‖ = 9.6–53.7 at truth on all 5 seeds → surrogate-truth offset (later retracted by [[m123_lbfgs_polish]]) | Done |
| [[m123_lbfgs_polish]] | **L-BFGS polish from truth + m115 DE basins (seeds 14, 27, 46, 74, 93)** — hyp1 CONFIRMED (truth IS the surrogate minimum in physical units, retracting the [[surrogate-truth-offset]] concept); hyp2 REFUTED-but-informative (DE basins polish ω by 0.02–0.21° and 0.09–0.32%, cost drops 1.14×–12.9×, q0 LOCKED at attractor); hyp3 mechanical-refuted / underlying-CONFIRMED (basin-only attractors match DE basins). Reveals DE+GD hybrid architecture: DE enumerates q0 attractors, L-BFGS polishes ω within each. | Done |
| [[m124_hifi_validate]] | **Hi-fi validation of m123 polished candidates (seeds 14, 27, 46, 74, 93)** — REFUTED (2/12 basins agree within ±30% log-ratio; threshold 75%). Seed 27 CATASTROPHIC (3/3 basins: hi-fi WORSENS 12-16×). Truth-polish SAFE. | Done |
| [[m125_keep_better_inline]] [inline] | **keep_better wrapper re-scoring of m124** — product-level metric: `polish + hi-fi(before, after) + keep_min` improves seed-level best hi-fi by 90%/33%/70% on seeds 14/74/93; seed 27 break-even. 9/12 basins helped. [[gradient-based-inversion]] REINSTATED #open-active. | Done |
| [[m126_wrapped_pipeline]] | **Wrapped pipeline on 6 untested baseline seeds (0, 6, 12, 24, 33, 36)** — CONFIRMED 3/6 improved ≥10% (seeds 0: 20%, 6: 87%, 24: 45%), 3/6 break-even (12, 33, 36), 0/6 regressed. Combined with [[m125_keep_better_inline]] → full 11-seed cohort: 6/11 improved, 5/11 break-even, 0/11 regressed. Seed 33 reveals [[omega-sign-degeneracy]] (ω rotating backwards, hi-fi 0.082). [[gradient-based-inversion]] promoted to **#validated**. | Done |
| [[m127_flipped_omega_search]] | **Flipped-ω compensating-q0 search on 11 baseline seeds** — 60k SO(3) + L-BFGS polish with ω=−ω_true. MIXED / UNDERPOWERED: seed 12 = new WIDE flipped-ω attractor (hi-fi **0.171** at q0_err 140°, independent of any m115 basin); **positive control seed 33 FAILED** (known basin's q0-width <0.001° vs grid spacing ~3°, miss by ~10000×). Counts: VALID=0, PARTIAL=1, FAIL=10. Seed 12's compensation axis is body ≈ +X (NOT +Z like seed 33) — flipped-ω compensation is seed-specific in both angle and axis. | Done |
| [[m128_warmstart_polish]] | **Warm-start flipped-ω polish from m115 DE basins (11 seeds)** — 3-DOF L-BFGS-B at `(q0_basin, −basin_ω)`. Verdict REFUTED on 10 valid-test seeds (all FLIPPED_FAIL, best_hifi 1.42–4.73); script emitted INVALID on seed-33 control because seed 33's basin ω is already retrograde (signed 161.54°, census this page) → warm-start produced forward-ω, not a flipped-ω test. Polish mechanics proven OK (inline spot-check: `+basin_ω` re-eval reproduces m115's 0.082 within 0.97%). Rules out "warm-start + negate-ω + local polish" as a flipped-ω enumerator. Does NOT update [[omega-sign-degeneracy]] confidence. | Done |
| [[m129_dense_grid_eval]] | **REFUTED** — densified 600k SO(3) grid on seed 33 gives only 0.3% improvement in best grid cost vs 60k (1.180 vs 1.183); hi-fi winner 2.62 (FLIPPED_FAIL); narrow flipped-ω basins are NOT grid-density-recoverable. Pivot to DE (m130). | Done |
| [[m131_refresh_wrappedbest]] | **Plumbing** — rebuild 11 `wrappedbest_seed*/` dirs with post-Option-A winners + cached hi-fi LCs. Zero hi-fi regeneration — reuses `hifi_mags_after` from m126 `hifi_ckpt.npz` and `hifi_mags` from m124 `hifi_results.npz`. Correction to prior claim: seed 6 winner is basin_0 (±X twin, q0_err 179.6°), not basin_1 (truth-adjacent); twin beats truth by 9e-5 on hi-fi. | Done |
| [[m132_solution_count_by_band]] | **Multi-solution yield metric, 11-seed post-fix cohort** — 33 polished basins stratified by hi-fi band. Headline: **16/33 below 0.1 (48% valid)**, 5/33 below 0.01, 9/33 FAIL ≥0.3. Winner-class counts oversell reliability — 6/11 seeds have 2/3 basins in the FAIL band. Bottleneck is DE basin enumeration (upstream), not keep_better polish (which helped on 33/33 basins). | Done |
| [[phase_B_m048_cohort]] | **NEW 2026-04-17** — Phase-B 8-seed m048 cohort (15°–87° phase). 4 OK + 1 PARTIAL + 1 downstream-FAIL + 2 upstream-FAILs. Two distinct upstream failure modes exposed; reliable band ≈ 38-55° phase with ≥4 spec peaks. Forces a design decision before Phase 3. | Done |
| [[m103_seed028_geo_hang]] | **NEW 2026-04-17** — Investigation of the 87°-phase Pool(24) geo hang + serial rescue attempts. Mechanism: alignment cost surface flat at high phase, L-BFGS-B burns unbounded function evals per iter, workers eventually lost. Truth absent from NM pool regardless. | Done |
| [[m139_convention_bug_fix]] | 🚨 **NEW 2026-04-30 — CRITICAL** — Quaternion convention bug at the propagator/renderer boundary. Propagator was outputting convention (b) (`R(q) = R_b→i`) via RIGHT-mult Hamilton kinematic; renderer expects convention (a) (`R(q) = R_i→b`) for `k1 = R @ sun_J2000`. Fix: two-line LEFT-mult swap in `propagate_principal_axis` (line 167→173) and `propagate_euler` (line 287→298). Validated on m048 seed 6: inversion-side renderer matches main pipeline at RMS 6.26e-15 mag. **All previous truth NPZs and inversion verdicts are buggy-forward-model classifications; downstream re-validation is scheduled.** | #validated |
| [[m140_post_fix_lc_delta]] | **NEW 2026-04-30** — Post-fix LC delta on m048 seed 6 (a known-good "previously solved" seed). RMS 2.18 mag, max 8.46 mag, ρ=43.6 (Band D) between buggy and correct hi-fi LCs. Bright peaks took 5–8 mag hits; LC ranges differ structurally. Bug is geometrically scrambled (R^T vs R), not a small perturbation. Establishes that all prior basin/ρ-band classifications were sorting against targets ~11× outside the acceptance bar. Surrogate model unaffected (it maps body-frame coords to mag, no propagator dependency in training). | #validated |
| [[m141_seed6_postfix_pipeline]] | **NEW 2026-04-30** — First end-to-end inversion pipeline rerun on a previously-solved seed (m048 seed 6) under post-fix correct truth. Result: **UPSTREAM-FAIL.** m103 26-candidate pool's best ω is at rank 9/26 (w_err=16.86°), buried by geo_cost ranking under 8 worse candidates (53–74° errors); top-3 fed to m115 are 53–57° from truth, far outside m115's bridging radius. **m135 alignment-cost-anti-truth pathology survives the convention fix.** Convention bug likely was masking m103 ranking failures by buggy-LC accident; "previously-solved" baseline-cohort status is ranking-luck artefact. Surrogate-rerank thread (m133/m134) is the unblocked next lever. | #upstream-fail |
| [[m142_seed6_postfix_surrogate_rerank]] | **NEW 2026-04-30** — Surrogate full-LC MSE re-rank of m141's 26-candidate seed-6 pool (4-sec scoring via `score_geo_surrogate.py`). Truth-near ω at m103-rank 9 (w_err=16.86°) is **DEMOTED to surr_mse rank 15/26** — not promoted. Pool min surr_mse=3.94 (ρ≈8.9, Band D); no Band-A/B candidate exists in the pool. Bright-MSE and full-MSE rankings disagree (m103-top candidates dominate surr_brt; truth-near candidate dominates neither). Structural blocker: the truth-near ω has q0_err=129.58° — m103's phi-sweep failed to find a truth-q0 anchor. Re-ranking cannot fix a pool missing the truth basin; m115 K=1 rerun NOT triggered. Reinforces [[feedback_bug_was_helping_m103]]; the m138/m134 surrogate-rerank rescue does NOT generalise to seed 6 post-fix. **Next escalation: Play 3 ([[upstream-redesign-6dof-surrogate-de]]) OR phi-sweep patch — pending generality check.** | #post-fix-rerank-fails |
| [[m143_seed91_postfix_generality]] | **NEW 2026-04-30** — Generality check on seed 91 (the m135 textbook Play-1 win under buggy truth, ρ=0.22 Band-A). Two flagship findings: **(1)** Seed 91's m103 ALREADY ranks the truth-ω at rank 1 (w_err=4.65°), agreeing with surrogate-MSE rerank — qualitatively different from seed 6 where ranks disagree. **(2)** But on BOTH seeds, m103's phi-sweep fails to find a truth-q0 anchor (q0_err 120–162° for the truth-ω across all 4 phi anchors). Lofi-300 pool diagnostic: seed 91 has exactly **1** jointly-truth-near candidate (q0_err=25°, w_err=29°) at align rank 2 / surr_mse rank 26 — surrogate-rerank DEMOTES it. Seed 6 has **0** jointly-truth-near candidates in lofi-300 (pool-deficient). The m135 buggy-truth surrogate-rerank rescue does NOT generalise to correct truth on seed 91 either — alignment cost places truth correctly at rank 2; surrogate-rerank harms it. **Two distinct failure modes: pool-deficient (seed 6) vs q0-anchor-deficient (seed 91).** Structural conclusion: surrogate-rerank as a standalone patch is dead under correct truth; m103's phi-sweep is the structural bottleneck. Play 3 (6-DOF surrogate DE upstream) OR phi-sweep patch are the only remaining escalations. m103 Step 4 (Geo) timed out at 480s on seed 91 — multi_phi pool used. | #post-fix-generality |
| [[m144_nm_pool_rerank_diagnostic]] | **NEW 2026-04-30** — Surrogate-MSE rerank on m103's **NM-prededup 300-pool** (post-NM-polish, pre-multi_phi truncation) on seeds 6 and 91. **Headline**: jointly truth-near candidates exist in the NM pool — seed 91 has 5 (best: q0=58°, w=8.4° at surr_mse rank 10), seed 6 has 3 (best: q0=55°, w=8.1° at surr_mse rank 22). Both seeds: alignment-cost BURIES them at ranks 31–156, but surrogate-MSE rerank surfaces them within K=22–30. **m103's `MULTI_PHI_TOP=2` truncation is the actual bottleneck** — it drops everything below the alignment-cost top-2 ω's. Concrete cheap m103 patch: increase MULTI_PHI_TOP to ~10–30 + surrogate-MSE rerank at Step 3.5 + expand m115 K to 10–30. ~1–10 line code change. **Play 3 is no longer the only escalation; m103 is salvageable.** Strategic-reframe Play 1 (m133 3-cost union) is also salvageable provided the rerank is applied PRE-truncation, not POST. m115 wall would 2–3× to consume larger pool. | #m103-patch-tractable |
| [[m145_seed91_postfix_full_patch_chain]] | **NEW 2026-04-30** — End-to-end execution of the m144 patch chain on seed 91 under correct truth. Patches landed in m103_hybrid.py (`M103_NM_RERANK_BY=surr_mse` + `M103_MULTI_PHI_TOP=10` + new Step 4.5 `M103_GEO_RERANK_BY=surr_mse`) and m115_surrogate_pipeline.py (`M115_SORT_BY=surr_mse` consumer). **Patch chain stages succeed exactly as designed** — surfaces the m144-predicted joint-truth-near (q0=52°/w=11.32°) at surr_rank 5/50 in the geo pool; m115 input has 5 of 10 ω inside marginal bridging, 2 inside reliable. **But m115's q-from-ω 3-DOF DE — the Roberto-Step-4 solver — fails to bridge q0 on every truth-near input. 0/100 DE runs land q0_err < 10°.** Lowest surrogate-MSE basin at q0_err=135° at truth-near ω (deceptive global minimum); m126 6-DOF polish then drifts ω from 3.46° to 41.32° and ω-mag to ~0 (-98.69%). Final: q0=141°/w=41°/hifi=1.20/**ρ=4.91, Band C, FAIL**. **m144 hypothesis partially refuted**: MULTI_PHI_TOP=2 truncation IS a bottleneck but not the only one — q-from-ω solver itself broken under correct truth. The Roberto Step 4 "DE+L-BFGS-B converges reliably" claim is a buggy-truth artefact. Three deferred escalation paths: q0-polish-from-input, 6-DOF surrogate DE upstream, surrogate-landscape-direct-probe. | #q-from-w-solver-fails-post-fix |

## Concepts

| Page | Summary |
|------|---------|
| [[quaternion-convention]] | 🚨 **NEW 2026-04-30** — convention (a) `R(q) = R_i→b` (LEFT-mult kinematic) vs convention (b) `R(q) = R_b→i` (RIGHT-mult kinematic). This codebase uses convention (a) everywhere; the propagator pre-fix was producing convention (b). Read before touching any q→matrix or kinematic code. |
| [[twin-degeneracy]] | 180° ±X symmetry: q_twin = q_180x * q0 (LEFT multiply), SAME omega; ±Y/±Z broken by shadows |
| [[attitude-level-set-disconnection]] | Per-epoch L(t) is disjoint components — competing physical hypotheses; min-over-anchor-set collapses them |
| [[symmetry-degeneracies]] | **OPEN:** Full symmetry analysis — exact ±X twin, near-symmetries (±Y/±Z), lo-fi vs hi-fi symmetry groups |
| [[alignment-cost]] | PAB alignment — 1 constraint/epoch, effective for 13% of seeds |
| [[brdf-cost]] | Ashikhmin-Shirley BRDF — 3 constraints/epoch, foreshortening discrimination |
| [[grid-search]] | HEALPix omega direction grid, N_DIRS × N_MAGS, delta-q factorization |
| [[nm-refinement]] | Nelder-Mead polish of top grid candidates, NM_TOP=300 current |
| [[phi-sweep]] | 1-DOF rotation about PAB circle, 360 bins, ±X normals only |
| [[glint-physics]] | Specular glint identification: mag<6 = 100% specular, PAB constraint |
| [[lo-fi-mse]] | No-shadow brightness — universal discriminator but can't replace grid cost |
| [[hi-fi-scoring]] | Ray-traced shadows — 272× slower, needed for final selection |
| [[candidate-selection]] | Bottleneck shifted from omega finding to selection; full-MSE best |
| [[basin-of-attraction]] | Classical (alignment): ~5°/~2°/±10%. Surrogate (residual, post-[[m121_basin_width_metric]]): q0 ≳5°, ω-dir <0.1°, ω-mag <0.25%. Hessian-tight (post-[[m122_hessian_curvature]]): q0 0.3–0.9°, ω-dir 0.003–0.05°, ω-mag ~0.007%; cohort-universal |
| [[surrogate-truth-offset]] | **CORRECTED 2026-04-16:** retracted via [[m123_lbfgs_polish]] — m122's gradient was parameter-space, not physical; truth IS the surrogate minimum at machine precision. Kept as historical record; `confidence: low`. |
| [[omega-magnitude-estimation]] | Peak count regression: rho=0.954, 13.2% median error |
| [[l-conservation]] | L_inertial conserved — 9 OoM gap, robust to 10° error |
| [[polhode-prior]] | **NEW 2026-05-07** — body-frame ω lives on a 1-D polhode (intersection of energy & momentum ellipsoids); reframe ω prior as `(|L|, polhode_label, polhode_phase)` instead of raw 3-vector. `|ω|(t)` wobble amplitude IS the polhode size. Plausible mechanism for s042 basin-width / |ω| inverse correlation. medium-high confidence pending cohort-tail rendering (seeds 14/17/68/81). |
| [[bridge-solver]] | Euler dynamics bridge — generates correct omega, selection fails |
| [[scipy-sensitivity]] | Seeds 12, 27, 33 consistently hard — SG7 smoothing helps |
| [[multi-phi]] | Multiple phi values — fixes 14/24 but 4x geo cost, abandoned |
| [[brightness_surface_path_matching]] | Body-frame PAB crossing geometry at peaks — kinematic error, refuted |
| [[pab-contour-isoshell]] | PAB-contour + isoshell framework — IPL centroids as attitude candidates |
| [[ipl-census]] | 100-seed IPL population statistics — 83% have >=2 tight constraints |
| [[att-fail-diagnosis]] | Dominant failure mode: omega OK but attitude wrong — lo-fi can't discriminate phi |
| [[plateau-constraint]] | Constant-magnitude phases = PAB circling a lobe — 125× discrimination but lo-fi biased |
| [[isoshell-phi-limits]] | Isoshell framework cannot discriminate phi — zero-phase surface too symmetric; shadows needed |
| [[shadow-asymmetry]] | Shadow effects (up to 4.6 mag at ±Y) are the ONLY phi discriminator; lo-fi symmetry is fundamental |
| [[anchor-alignment-error]] | **NEW:** The real phi bottleneck — cos^250 amplifies 2° anchor error to 100-1200× noise; fix by choosing best-aligned anchor epoch |
| [[global-optimizers-for-lc]] | Literature survey: PSO (Burton), DE, CMA-ES, MCMC for LC inversion |
| [[differentiable-inversion]] | **HORIZON:** Surrogate is differentiable → gradient-based opt / HMC replaces classical pipeline |
| [[multi-solution-philosophy]] | **CORE:** Pipeline should find ALL valid (q0,w) below hi-fi threshold, not single best |
| [[surrogate-model]] | 239k-param MLP replaces hi-fi (50,000× speedup); enables multi-start DE + 6-DOF joint search. **2026-04-16 caveat:** MAE 0.03 mag holds AT TRUTH only; off-truth on far-q0 attractors the surrogate cost-ratio can disagree with hi-fi by 10-15× (per [[m124_hifi_validate]] seed 27) |
| [[pab-contour-phase-angle-limitation]] | **NEW:** pab-contour assumes k1=k2; truth PAB is median 25° off IPL centroid at real IS-901 geometries; caps IPL-cost usefulness |
| [[kernel-factorization]] | **NEW:** decouple propagation from cost evaluation; any alignment-cost variant re-scorable in ~20 s on saved kernel |
| [[dark-mag-saturation]] | **NEW:** surrogate predicts ~23 mag for "all-dark" candidate attitudes → cost saturates at ~10 → locally flat landscape in close_omega region; matters for gradient-based inversion init |
| [[omega-sign-degeneracy]] | **UPGRADED 2026-04-16 (low → medium):** [[m127_flipped_omega_search]] confirmed a second flipped-ω attractor — seed 12 at (q0_err 140.36°, −ω_true), hi-fi **0.171** (PARTIAL), WIDE basin (visible at 3° grid spacing). Seed 12 compensation axis is closest to body +X (NOT +Z like seed 33) — flipped-ω compensation is seed-specific in both angle AND axis, NOT a universal body-+Z panel-symmetry. Seed 33's narrow ~0.082 basin was MISSED by the 60k SO(3) grid (q0-width <0.001° vs ~3° grid spacing), so the 10 FAIL seeds are inconclusive w.r.t. narrow basins. Population frequency of WIDE flipped-ω attractors: 2/11 (seed 12, seed 33's known basin qualitatively different in width). |
| [[observation-geometry-sources]] | **NEW 2026-04-17:** m046 (single fixed 10–11 UTC window) vs m048 (per-seed random starts, phase 9°–95°); `lib.traj_source.load_truth(seed, source)` abstraction; invariant on `observation_times`. Underpins [[m048-migration]]. |
| [[constraint-poor-regime]] | **NEW 2026-04-17:** when a seed has ≤2 spec peaks, the alignment cost has ≤1 constraint after anchor and becomes trivially satisfiable → `geo_cost ~1e-20`, truth findable but unselectable. Seed 23 example. Independent of phase angle. |
| [[phase-angle-operating-range]] | **NEW 2026-04-17:** synthesis of m046 (30-60° shared window) + m048 Phase-B cohort (15-87° per-seed). Reliable band ≈ 38-55° phase with ≥4 spec peaks. m046 cohort success rate was biased by 50-60° constraint-rich sub-band. |

## Branches

| Page | Status | Summary |
|------|--------|---------|
| [[global-optimizers]] | #dead-end | 6D joint space too large — basin fraction ~10^-8 |
| [[cascade-filtering]] | #dead-end | Combinatorial explosion: 100^K pairings |
| [[peak-graph-pipeline]] | #dead-end | Lo-fi scores uniform, no discrimination |
| [[multi-epoch-winding]] | #dead-end | All staircase omegas have 13-25° direction error |
| [[bridge-lc-selection]] | #dead-end | Chicken-and-egg: wrong q1 → wrong brightness → bad score |
| [[l-parameterization]] | #dead-end | No wider basin than omega-parameterization |
| [[pa-mode-bridge]] | #dead-end | IS-901 too triaxial (asymmetry=0.556) |
| [[pab-circle-seeding]] | #dead-end | L-BFGS-B leaves PAB circle — phi-sweep is correct approach |
| [[nm-expansion]] | #validated | NM_TOP=300 works — m098-99 confirmed |
| [[full-mse-selection]] | #validated | Full-window MSE > multi-window vote — m102 |
| [[multi-phi-approach]] | #dead-end | Fixes some seeds but too risky/expensive |
| [[lo-fi-grid-replacement]] | #dead-end | Lo-fi MSE can't replace alignment at grid level |
| [[hybrid-selection]] | #dead-end | Window consensus + full-MSE fallback — helps omega not ATT_FAIL |
| [[crossing-geometry-scoring]] | #dead-end | Pairwise peak alignment — too sensitive to omega precision at grid level |
| [[ipl-candidate-generation]] | #dead-end | IPL centroids as attitude candidates — all next steps exhausted |
| [[sparse-hifi-phi]] | #dead-end | Sparse hi-fi phi sweep — superseded by anchor-alignment-error |
| [[best-anchor-selection]] | #dead-end | Post-NM best-anchor phi re-sweep — omega error too large |
| [[de-attitude-search]] | **#validated** | **3-DOF attitude search with DE — replaces phi sweep (m113→m115 confirmed)** |
| [[surrogate-de-search]] | **#validated** | **m115: 10/10 seeds valid via surrogate multi-start DE. Replaces phi sweep.** |
| [[surrogate-omega-selection]] | **#validated** | **Inline test (4/4 seeds): surrogate MSE ranks best omega top-2; alignment cost buries it at rank 4-5** |
| [[harvester-optimization]] | #open (deprioritised) | Deprioritised 2026-04-15: alignment-cost has ~25° noise floor — harvester won't fix it. Kept open as a fallback tool. |
| [[surrogate-attitude-isoshell]] | **#validated (multi-seed, ATT_FAIL cohort)** | [[m120_tumbling_competitors]] tumbling-competitor test: truth rank 0/10004 under all 13 variants for all 6 seeds {0, 14, 27, 46, 58, 75}. Multi-seed gate cleared; next phase is search integration. |
| [[gradient-based-inversion]] | **#validated** (promoted 2026-04-16) | Wrapped pipeline `DE → polish → hi-fi(before, after) → keep_min` validated on full 11-seed baseline cohort per [[m126_wrapped_pipeline]]: 6/11 seeds improved ≥10%, 5/11 break-even (wrapper catches catastrophes), 0/11 regressed. Strict Pareto improvement over plain [[m115_surrogate_pipeline]]. Recommended as default pipeline. |
| [[m048-migration]] | **#open-active** | Phase 1 plumbing complete 2026-04-17; Phase 2 pilot effectively complete on 8 seeds (4 OK + 1 PARTIAL + 1 downstream-FAIL + 2 upstream-FAILs); Phase 3 blocked on upstream-design decision. See [[phase_B_m048_cohort]]. |
| [[upstream-redesign-6dof-surrogate-de]] | **NEW #proposed 2026-04-17** | Replace m103 grid+NM+multi-phi+geo with a single-stage 6-DOF surrogate DE over `(q0, ω)`. Motivated by Phase-B's two alignment-cost failure modes. Gated on m046 11-seed validation + Phase-B FAIL-seed rescue. |
