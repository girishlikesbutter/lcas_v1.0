# CURRENT_STATE — Inversion research

**Last updated:** 2026-04-30 (FIFTH sitting of the day — m048 cohort regenerated under correct propagator + buggy-vs-postfix comparison (m146)). The m048 truth database is now **uniformly correct across all 100 seeds**; buggy versions preserved alongside as `*_buggy.npz`. m146 cohort comparison shows the bug effect is uniformly catastrophic: **all 100 seeds Band D**, ρ median=48.27 (range 28.3–81.5), max-LC-delta median 8.92 mag (max 14.4), quaternion geodesic p99 median 178.7° (confirming q→q* attitude flip per epoch as the conv-(b)→conv-(a) inversion-form claim predicts), bright-peak displacement median 19.2 epochs (well outside m103's ±5 epoch lofi-match window). **Strong systematic correlation: ρ vs median phase angle** — worst 5 seeds (ρ 65–82) all have PA_med 80–88°; best 5 (ρ 28–34) all have PA_med 24–63°. High-PA glint geometry is more sensitive to attitude scrambling; low-PA LCs are dominated by Lambertian smoothness. Aligns with Roberto-2026-04-17 reports of seed 69 (PA=67.7°) Step-1 hang and seed 36 (high PA) Band-D outcome — the same correlate breaking inversion now under correct truth was being amplified by the bug then. The "buggy m103 wins on baseline cohort were ranking-luck" hypothesis (`feedback_bug_was_helping_m103.md`) sharpens to: **phase-angle-modulated bug effect** — low-PA seeds had a small bug perturbation that left m103's noisy ranking landing on truth-near ω more often; high-PA seeds had a large perturbation that scrambled the bright-peak geometry into either a hang or a non-truth basin.

**Previous sitting (fourth, m145):** implemented and end-to-end-validated the m144 patch chain on seed 91 under correct truth: **REFUTED at Band C, ρ=4.91**. The patch chain works through every m103 stage as designed (NM-pool surr_mse rerank surfaces joint-truth-near at top-10, MULTI_PHI_TOP=10 preserves it, geo refinement keeps 2 joint-truth-near in pool, surrogate-MSE rerank on geo pool puts the joint-truth-near at surr_rk 5/50, m115 K=10 with M115_SORT_BY=surr_mse consumes it). **But m115's q-from-ω 3-DOF DE — the Roberto-Step-4 solver — fails to bridge q0 on every single one of the 10 input ω's, including 5 that are inside the empirical marginal bridging radius (w_err ≤ 15°) and 2 that are inside reliable bridging (w_err ≤ 5°). 0 / 100 DE runs landed q0_err < 10°.** Lowest surrogate-MSE basin lands at q0=135° at truth-near ω; m126 6-DOF L-BFGS polish then drifts ω to 41° (wrong) and ω-mag to ~0 (-98.69%); final wrapped winner is q0=141°/w=41°/hifi=1.20/ρ=4.91. **The m144 hypothesis ("MULTI_PHI_TOP=2 truncation is the structural bottleneck") is partially refuted: it IS a bottleneck (the patch correctly opens it) but is NOT the only one — the q-from-ω solver itself is broken under correct truth on this seed.** The Roberto Step 4 claim "DE over q0 followed by L-BFGS-B polish converges reliably" is now characterised as a buggy-truth artefact alongside the m103 ranking-luck claim. Three escalation paths sketched (q0-polish-from-input, 6-DOF surrogate DE upstream, surrogate-landscape-direct-probe). Cohort regen still deferred — we don't have a patch chain that solves a single previously-Band-A seed under correct truth. Seed 6 end-to-end NOT run (low value given seed 91's failure mode).

**Previous sitting summary (third, m142+m143+m144 chain):** m103's `MULTI_PHI_TOP=2` truncation identified as a bottleneck. Both alignment-cost AND surrogate-MSE place jointly-truth-near candidates at NM-pool rank 10–32 on seed 91 (5 joint candidates, best q0=58°/w=8.4°) and seed 6 (3 joint candidates, best q0=55°/w=8.1°). m103 was throwing everything below align-rank 2 away in Step 3.5. Patch path proposed: increase MULTI_PHI_TOP to 10–30, surrogate-MSE rerank at Step 3.5, expand m115 K. m145 implemented; the patch chain produces the predicted geo pool but the solver after it fails.

---

## What this session accomplished (2026-04-30, second sitting)

1. **m140 — quantified the LC delta.** Compared the buggy `traj_seed006.npz.mag_hifi` against the post-fix correct LC from m139 Step 3. Result: RMS 2.18 mag, max 8.46 mag, ρ=43.6 (Band D). Bright epochs took 5–8 mag hits. **Bug is geometrically scrambled, not a small perturbation.** Wiki: [[m140_post_fix_lc_delta]].

2. **Inversion-code audit.** Searched all live (non-archive) inversion code for two dangerous post-fix patterns: `R.T @ v_J2000 → v_body` and `R @ v_body → v_J2000`. **One production file confirmed dangerous:** `notebooks/inversion/lib/attitude_viz.py:361,363` (visualisation bug, no inversion-outcome impact). Patched in this session: dropped the `.T`, added comment naming convention (a). Misleading variable name (no production impact): `notebooks/inversion/twin_render_lambertian.py:92` (debug script, kept as documentation). All other live inversion code is **self-healing under the fix** — the codebase overwhelmingly uses the conv-(a) renderer pattern (`R = as_matrix(q); v_body = R @ v_J2000`) which was silently wrong pre-fix and is silently correct post-fix.

3. **Surrogate-not-affected clarification (user-driven, important).** The surrogate maps `(k1_body, k2_body) → mag` (or similar attitude-frame inputs), NOT `(q0, ω, t) → mag`. There is no propagator in its training pipeline. The bug was at the *bridge* from `(q0, ω, t)` to `(k1_body, k2_body)`, not in the surrogate. **No surrogate retraining is required.** This collapses one of the original "scheduled follow-ons" from m139.

4. **Seed-6 truth regenerated** in canonical form. `MICRO48_WORKER_SEED=6 python3 m048_generate_trajectories_v2.py`. 53 sec wall. New `mag_hifi` matches m140 `mag_fixed` at RMS 4.7e-15 mag. Buggy NPZ preserved at `traj_seed006_buggy.npz`. Hi-fi peak count: 32 (buggy) → 30 (fixed). Phase 56–70° unchanged.

5. **m141 — first post-fix pipeline run on seed 6.** `invert.py --seed 6 --traj-source m048`. Killed mid-m115 once upstream verdict was clear. **Headline result: seed 6 is upstream-FAIL on corrected truth.** m103's 26-candidate pool contains a truth-near ω at rank 9 (w_err=16.86°), but m103's geo_cost ranking buries it under 8 candidates with w_err=53–74°. Top-3 ω candidates fed to m115 are 53–57° from truth — far outside m115's reliable bridging radius (3–5°). m115 DE confirms: 9/30 starts done, all q0_err 88–162°, MSE 3.4–4.2, n_below_10deg=0/10. Wiki: [[m141_seed6_postfix_pipeline]].

6. **Strategic implications** (also landed today after the m141 verdict):
   - **The convention bug may have been masking m103 ranking failures.** Seed 6's "OK" status under the buggy model was likely a ranking-luck accident on the buggy LC's bright-peak displacement. Under correct truth, m103's ranking pathology is exposed.
   - **m135 alignment-cost-anti-truth survives the bug fix.** The pathology isn't a buggy-truth artefact; it's structural to the alignment-cost surface.
   - **Surrogate-MSE re-rank (m133/m134) becomes the immediate next lever** — and it's now testable on a corrected forward model for the first time.
   - **The "previously-solved seed cohort" is no longer a stable benchmark.** Every result in m098/m115/m126/m131/m132 was buggy-truth-classified.

7. **All buggy-state artefacts preserved** as `_buggy` siblings (truth NPZ + 5 result directories). All wiki pages, log entries, and EXPERIMENTS.md banners updated to reflect both the m139 fix and the m141 finding.

8. **m142 — surrogate-MSE rerank on m141's saved seed-6 pool: REFUTED.** Adapter `score_geo_surrogate.py` (130-line clone of `score_lofi_surrogate.py`) reading `geo_ckpt.npz` instead of `lofi_ckpt.npz`. Truth-near ω at m103-rank 9 (w_err=16.86°) **demoted to surr_mse rank 15/26**. Pool min surr_mse=3.94 → ρ≈8.9 (Band D); no Band-A/B candidate exists. Bright-MSE and full-MSE ranking favour disjoint subsets. **Structural finding from forensics:** the NM-300 pool DOES contain 7 jointly truth-near (q0<60° AND w<30°) candidates at cost ranks 127–267/300 — alignment cost buries them. Multi-phi only re-explores top-2 ω's (4 phi anchors each); the truth-near ω at rank 9 got only 1 q0 anchor (q0_err=129.58°). Best joint NM candidate: q0_err=54.92°, w_err=8.08° at NM cost rank 156/300. **The fix is not metric-substitution at the geo_ckpt stage; it's earlier-stage candidate selection (NM-pool re-rank by surrogate, OR replace m103 with Play 3).** m115 K=1 rerun NOT triggered. New wiki: [[m142_seed6_postfix_surrogate_rerank]].

9. **m143 — seed-91 generality check + lofi-pool diagnostic on BOTH seeds.** m103 Step 4 (Geo) timed out at 480s on seed 91; ranking analysis used `multi_phi_ckpt.npz` instead. Adapter `score_geo_surrogate.py` modified to fall back to multi_phi when geo missing. **Seed 91's m103 ranks the truth-near ω at rank 1 (w_err=4.65°)** — completely different from seed 6. Surrogate-MSE rerank agrees with alignment cost on the ω ranking. **But BOTH seeds share the q0-anchor pathology**: m103's 4-phi-per-ω parameterisation reaches only q0_err 120–162° around the truth-ω. Lofi-300 pool diagnostic (the m135 rescue lever): **seed 91 has exactly 1 jointly-truth-near candidate (q0_err=25°, w_err=29°) at align rank 2 / surr_mse rank 26** — surrogate-MSE rerank DEMOTES it. **Seed 6 has 0 jointly-truth-near candidates** in lofi-300 by even the loosest criterion. Two distinct failure modes: **pool-deficient (seed 6)** vs **q0-anchor-deficient (seed 91)**. The m135 buggy-truth seed-91 surrogate-rerank rescue **does NOT generalise to correct truth** — under correct truth, alignment cost places jointly-truth-near at rank 2 (good!) and surrogate-rerank harms it. **Surrogate-rerank as a standalone patch is dead under correct truth.** New wiki: [[m143_seed91_postfix_generality]].

10. **Strategic implication: Play 1 is in question.** The strategic-reframe Play 1 (m133 3-cost union ω-ranking + K=7 to m115) depends on surrogate-MSE being a reliable signal across seeds. m143 shows it's NOT under correct truth on seed 91. Cohort-scale validation needed before declaring Play 1 dead, but two single-seed negative datapoints (6 + 91 lofi-rerank) are enough to suspend production work on Play 1 until we understand the buggy-vs-correct gap.

11. **Seed-47 buggy artefacts backed up** (`traj_seed047.npz` and 5 result dirs renamed to `_buggy` siblings) but **not regenerated this session** — 2-seed cap reached.

12. **m144 — NM-pool surrogate rerank diagnostic (THE HEADLINE FINDING of this sitting).** New script `score_nm_surrogate.py` scores the 300 NM-prededup candidates against post-fix observed truth (~30s/seed). **Both seeds: jointly-truth-near candidates EXIST in the NM pool at top-K reachable ranks.** Seed 91: 5 joint candidates, best q0_err=57.79°/w_err=8.38° at **surr_mse rank 10** / align rank 31. Seed 6: 3 joint candidates, best q0_err=54.92°/w_err=8.08° at surr_mse rank 26 / align rank 156, AND q0_err=55.59°/w_err=9.38° at **surr_mse rank 22** / align rank 132. **The m103 bug-in-design is `MULTI_PHI_TOP=2` truncation** — Step 3.5 takes only top-2 ω by alignment cost; everything else dropped to single-phi. Under buggy truth that occasionally coincided with truth-ω; under correct truth it strands the truth basin. Both rerank metrics surface jointly-truth-near at rank 10–32 on the NM pool; the patch is a tiny code change. **Play 3 demoted from "only structural fix" to "follow-on to consider after m103 patch validates".** New wiki: [[m144_nm_pool_rerank_diagnostic]].

## Next-session priorities (in order, post-m145)

The m103 patch chain (M103_NM_RERANK_BY=surr_mse, M103_MULTI_PHI_TOP=10, M103_GEO_RERANK_BY=surr_mse) and the m115 consumer side (M115_SORT_BY=surr_mse, M115_NUM_OMEGA_CANDIDATES=10) are landed and validated to produce the predicted geo pool. The new bottleneck is the q-from-ω solver itself.

1. **q-from-ω surrogate landscape diagnostic (HIGHEST priority — cheap, ~5 min wall).** At seed 91 truth-ω, plot surrogate full-LC MSE as a function of q0 over a Sobol grid in SO(3) (e.g. 1000-2000 points). Question: does the surrogate genuinely have its global minimum at q0_err≈135° rather than at truth-q0=0°? If yes, the surrogate is **wrong** for this seed and surrogate-retraining (or correction) is now on the table — `feedback_surrogate_is_bridge_independent.md`'s "no retraining needed" applies to the bug fix itself but does not protect against a separately-broken surrogate at this geometry. If no (truth-q0 is the global minimum but DE doesn't find it), the issue is DE escapes — try tighter bounds, more starts, or a different optimiser.

2. **Polish-from-input-q0 patch (likely Play 1 escalation).** m115 currently discards each candidate's input q0 and runs DE from scratch over all SO(3). Add a small NM/L-BFGS polish on surrogate full-LC MSE starting FROM each candidate's `(q0_input, ω_input)` BEFORE the global DE. Take min(polish_result, DE_result) per ω. The seed-91 joint-truth-near input idx=1 has q0_input=52.16° / w_input=11.32° — q0 within m115's empirical reliable bridging radius from truth. Polish from there should produce a Band-A or Band-B basin if (1) is a "DE escape" rather than "surrogate is wrong" failure. ~50 lines of m115 patch.

3. **6-DOF surrogate DE upstream (Play 3 proper).** `[[upstream-redesign-6dof-surrogate-de]]`. Replace m103 entirely with a 6-DOF DE on surrogate full-LC MSE over (q0, ω). May still hit the same surrogate-landscape pathology, but at least the search has 6 d.o.f. to escape the q0=135° basin. Larger code change.

4. **m126 polish guards.** Under correct truth, m126 basin 1's 6-DOF polish drove ω to 41° and ω-mag to ~0. Add polish-bounds (ω_change ≤ 30°, ω_mag stays within 50% of input?) to prevent these catastrophic drifts. ~10-line patch.

5. **Investigate the lofi `surr_mse` NaN issue (re-classified from "bug" to "intentional gating").** The lofi step's surr_mse is NaN by design when `M103_LOFI_SORT=align` (default) — the surrogate scoring is gated by the `M103_LOFI_SORT='surr'` env var. Not a bug. If we want lofi surr_mse always available for analysis, ungate the computation. Decision deferred.

6. **Cohort-scale regeneration plan.** Still deferred. We do not yet have a patch chain that solves a single previously-Band-A seed under correct truth.

7. **Seed 47 generality check** (deferred from m143 session). Buggy artefacts already backed up. Run only after item 1 or 2 above resolves the q-from-ω solver question.

8. **Seed 6 end-to-end with full patch chain** (low value until q-from-ω solver is fixed). Seed 6's joint NM candidate has worse w_err (8.08° vs seed 91's 6.17°), so likely Band C or worse on seed 6 with current solver.

9. **Convention runtime gates** (further deferred from m139): propagator output gate, renderer input gate, import-time finite-difference self-test. Defer until at least item 1 above resolves.

10. **Old-ideas re-explore** (further deferred). The m135 anti-truth finding survives the fix; cost-shape engineering on alignment-cost remains correctly suspended (`feedback_stop_cost_shape_engineering.md`). Surrogate-MSE-anti-truth is now joining alignment-cost-anti-truth as a known pathology under correct truth on seed 91.


## What is NOT to be carried forward

- Seed 6 (and by extension every "previously-solved" baseline cohort seed) should NOT be cited as "solved" without the qualifier "under buggy truth, status under corrected truth pending re-test".
- m103's geo_cost ranking should NOT be assumed reliable on any seed without surrogate-rerank validation. m141 confirms the pathology is structural.
- The m139-era proposal to "retrain the surrogate" is RETRACTED. Surrogate is bridge-independent; no retraining needed.

## ρ-band convention (UNCHANGED — still load-bearing)

ρ = √hifi_MSE / 0.05. Bands A (<2), B (2–4), C (4–8), D (≥8). Acceptance bar is ρ < 4 (A∪B). The convention is metric-only and survives the forward-model fix; only the *numerator* (hifi_MSE on a regenerated truth LC) changes.

## Documents updated this session (m145, fourth sitting)

- `notebooks/inversion/11_casadi_formulation/m103_hybrid.py` — patched: `M103_NM_RERANK_BY` env var (default 'align'), `M103_MULTI_PHI_TOP` env var (default 2), `M103_GEO_RERANK_BY` env var (default 'align'), `M103_GEO_TOP` env var (default 20). Step 3.25 (NM-pool surrogate-MSE rerank, gated) added between NM polish and dedup. Step 4.5 (geo-pool surrogate-MSE rerank, gated) added after Step 4 (Geo). nm_prededup_ckpt and geo_ckpt schemas widened with surr_mse / surr_bright_mse / *_rerank_by fields (additive).
- `notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py` — patched: `M115_SORT_BY=surr_mse` option added, reads `surr_mse` field from geo_ckpt, fails loudly if field missing or all NaN.
- `notebooks/inversion/inspect_m103_patched.py` — NEW (~140 lines), post-stage joint-truth-near auditor for nm_prededup / multi_phi / geo checkpoints.
- `notebooks/inversion/append_geo_surr_mse.py` — NEW (~110 lines), backfills geo_ckpt with surr_mse/surr_bright_mse without re-running m103.
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/` — full patched-m103 outputs (50-candidate multi_phi+geo pool with joint-truth-near at idx=1 q0=52.16°/w=11.32°, idx=25 q0=50.20°/w=6.17°).
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091.prepatch.bak/` — pre-patch artefacts preserved.
- `data/results/inversion_diagnostics/m115_surrogate_pipeline_m048/seed_091/` — patched-m115 outputs (sort_by=surr_mse, K=10).
- `data/results/inversion_diagnostics/m126_wrapped_m048/seed_091/` — m126 polish outputs (basin 1 ω drifted from 3.46° to 41.32°).
- `data/results/inversion_diagnostics/wrappedbest_m048_seed091/result.json` — final winner (FAIL, ρ=4.91, Band C).
- `data/results/inversion_diagnostics/wrappedbest_m048_seed091_lc_compare.png` — LC overlay.
- `data/results/inversion_diagnostics/invert_m048_seed091.prepatch.bak/` — preserved.
- `notebooks/inversion/wiki/wiki/experiments/m145_seed91_postfix_full_patch_chain.md` — NEW.
- `notebooks/inversion/wiki/wiki/log.md` — m145 ingest entry at top.
- `notebooks/inversion/wiki/wiki/index.md` — m145 row added.
- `notebooks/inversion/CURRENT_STATE.md` — this file (m145 update).

## Documents updated this session (m142 + m143 + m144)

- `notebooks/inversion/score_geo_surrogate.py` — NEW adapter (130 lines), reads geo_ckpt.npz with multi_phi_ckpt.npz fallback for geo-timeout cases.
- `notebooks/inversion/score_nm_surrogate.py` — NEW adapter (130 lines), reads nm_prededup_ckpt.npz; the m144 lever.
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/geo_surr_ckpt.npz` and `nm_surr_ckpt.npz` — m142+m144 rerank artefacts.
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/lofi_surr_ckpt.npz` — m143 lofi rerank for seed 6.
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/` — full post-fix m103 outputs (Step 4 timed out; multi_phi available, geo not).
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/{geo_surr_ckpt, nm_surr_ckpt, lofi_surr_ckpt}.npz` — m143+m144 rerank artefacts for seed 91.
- `data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed091.npz` (post-fix) and `traj_seed091_buggy.npz` (preserved). Same `_buggy` pattern for seed 47 NPZ (no regen).
- `data/results/inversion_diagnostics/.../seed_091_buggy/` (5 dirs preserved); same for seed_047_buggy/ (5 dirs preserved, no regen).
- `notebooks/inversion/wiki/wiki/experiments/m142_seed6_postfix_surrogate_rerank.md` — NEW.
- `notebooks/inversion/wiki/wiki/experiments/m143_seed91_postfix_generality.md` — NEW.
- `notebooks/inversion/wiki/wiki/experiments/m144_nm_pool_rerank_diagnostic.md` — NEW.
- `notebooks/inversion/wiki/wiki/index.md` — m142, m143, m144 rows added.
- `notebooks/inversion/wiki/wiki/log.md` — m142, m143, m144 ingest entries at top.
- `notebooks/inversion/CURRENT_STATE.md` — this file (m142+m143+m144 update).
- `notebooks/inversion/EXPERIMENTS.md` — Section 0 banner updated with m142 result.

## Risks / open items on disk

- **The m048 truth NPZ database is mixed.** Seeds 6 and 91 are now in canonical form (post-fix). Seeds 0–5, 7–46, 48–90, 92–99 are still buggy. **Inversion runs on those seeds are blocked until cohort regeneration is complete.**
- The m115 surrogate pipeline checkpoints from m141 are partial (killed mid-run). Safe to delete and rerun if needed.
- The buggy `_buggy` artefacts on disk are now ~12 directories + 2 NPZs. Disk cost ~few MB total; preservation cost negligible. Keep until cohort regeneration is complete.
- m046 truth NPZ is entirely unfixed. Out of scope until cohort regeneration is decided.
- `lofi_ckpt.npz` `surr_mse` field is all NaN on seed 6 — the m103 lofi step's surrogate evaluation appears to be silently failing. Verify on seed 91 once that data is in. If it's a pipeline bug, fix before NM-pool re-rank diagnostic.

## How to use this file

- **Next session start:** read this file. Read the m142 wiki page and the eventual seed-91/47 m143 page. The next concrete experiment is the NM-pool rerank diagnostic (priority 2 above) — cheap, ~10 min.
- **If "Last updated" is behind git HEAD touching `notebooks/inversion/`:** regenerate the top section from `git log -10` + log.md tail.
