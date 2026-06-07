# LCAS Inversion — Experiment Map

> **Purpose:** Read Section 1 to know where you are. Skim Section 2 for what works. Check the series index in Section 3 for detail. This document is the single source of truth for the inversion experiment series.

---

## 0. CRITICAL CONVENTION-BUG BANNER (2026-04-30) — UPDATED post-m140/m141

🚨 **Quaternion convention bug fixed (m139, commit f7fabbe). Subsequently:**

- **m140 quantified the LC delta** on m048 seed 6: RMS 2.18 mag, max 8.46 mag, ρ=43.6 (Band D). Bug is geometrically scrambled, not a small perturbation. Bright peaks took 5–8 mag hits; entire LC range shifted (buggy [7.53, 15.91] vs fixed [6.52, 16.43]).
- **Inversion-code audit complete.** One production file needed editing: `notebooks/inversion/lib/attitude_viz.py:361,363` (`.T` dropped, conv-(a) renderer pattern). All other live inversion code self-heals under the fix.
- **Surrogate model is bridge-independent.** It maps `(k1_body, k2_body) → mag`, not `(q0, ω, t) → mag`. The bug was at the propagator-renderer bridge, not in the surrogate's training data. **No retraining required** (correction to m139's "scheduled follow-on").
- **m141 first post-fix pipeline run on seed 6 — UPSTREAM-FAIL.** m103's geo_cost ranking placed all top-3 ω candidates 53–57° from truth; m115 cannot bridge that gap. Pool inspection: truth-near ω (w_err 16.86°) lives at rank 9/26. **m103's ranking is anti-correlated with truth on this seed under corrected truth** — m135 alignment-cost-anti-truth pathology survives the convention fix.
- **The convention bug may have been masking m103 ranking failures.** Seed 6's "previously-solved" status (m098, m126, m131 wins) was likely a buggy-LC ranking-luck accident. Whether the entire baseline cohort's ~50% Band-A rate persists under corrected truth is now an open empirical question.
- **m142 surrogate-MSE re-rank on seed-6 m103 pool — REFUTED.** Truth-near ω at m103-rank 9 (w_err 16.86°) **demoted to surr_mse rank 15/26** under post-fix propagator + observed truth. Pool min surr_mse=3.94 → ρ≈8.9 (Band D); no Band-A/B candidate exists in the pool. Bright-LC and full-LC MSE rank disjoint subsets. **Structural finding: m103's phi-sweep failed to find a truth-q0 anchor for the rank-9 ω (its q0_err is 129.58°)** — re-ranking is an ordering operation and cannot manufacture a candidate the upstream stage didn't produce.
- **m143 seed-91 generality check — surrogate-rerank lever inverts under correct truth.** Seed 91's m103 ranks truth-ω at rank 1 (w_err 4.65°) — different qualitative pattern from seed 6 — but BOTH share the q0-anchor pathology (multi-phi q0_err 120–162° around truth-ω). Lofi-300 diagnostic: seed 91 has 1 jointly-truth-near at align rank 2 (correctly!) but surr_mse rank 26 (DEMOTED); seed 6 has 0 jointly-truth-near in lofi-300 (pool-deficient). The m135 buggy-truth surrogate-rerank rescue does NOT generalise to correct truth. Two distinct failure modes: pool-deficient (seed 6) vs q0-anchor-deficient (seed 91).
- **m144 NM-pool rerank diagnostic — THE FIX IS CONCRETE.** The truth basin DOES exist in m103's NM-prededup 300-pool (seed 91: 5 jointly-truth-near, best at surr_mse rank 10; seed 6: 3 joint, best at surr_mse rank 22). m103's `MULTI_PHI_TOP=2` truncation (Step 3.5) is the actual structural bottleneck — it drops everything below align-rank 2. **Minimal patch: `MULTI_PHI_TOP=10–30` + surrogate-MSE rerank at Step 3.5 + expand m115 K to consume the larger pool.** 1–10 line code change. **Play 3 demoted from "only structural fix" to "follow-on";** m103 is salvageable. Strategic-reframe Play 1 is also salvageable IF rerank applied PRE-truncation, not POST. Next-session: implement and end-to-end validate on seed 91.

**Read first:** `CURRENT_STATE.md`, `wiki/wiki/experiments/{m139_convention_bug_fix, m140_post_fix_lc_delta, m141_seed6_postfix_pipeline, m142_seed6_postfix_surrogate_rerank, m143_seed91_postfix_generality, m144_nm_pool_rerank_diagnostic}.md`, `wiki/wiki/concepts/quaternion-convention.md`.

**All m046/m048 truth NPZs, all `pred_lc.npy` caches, and all inversion verdicts m070 → m138 below were produced under the bug.** Section 1 narrative preserved as historical record; do NOT re-cite numbers without a "buggy truth" footnote.

**Next-session priority:** **m103 patch + end-to-end validation on seed 91**. Patch `MULTI_PHI_TOP` from 2 to 10–30 in `notebooks/inversion/11_casadi_formulation/m103_hybrid.py` Step 3.5; add surrogate-MSE rerank at Step 3.5; expand m115 K=10–30. Run on seed 91 (truth NPZ canonical), expect Band-A/B from the q0_err=58° / w_err=8.4° NM candidate. Then seed 6 with K=22+. Wall ~30 min/seed. See `CURRENT_STATE.md`.

---

## 0a. Strategic Reframe Banner (2026-04-29)

**Read `CURRENT_STATE.md` and the three starred memory entries (`project_strategic_reframe_2026_04_29.md`, `feedback_stop_cost_shape_engineering.md`, `feedback_rho_band_yield_metric.md`) before you do anything with the chronology below.**

The chronology in Section 1 is preserved as-written, but two things in it are now retracted as production directions:

1. **The m136/m137/m138 cost-shape engineering thread is suspended.** Don't design new aggregator costs over alignment-cost / IPL centroids / attitude level sets / kernel-factorised consistency. The substrate is anti-correlated with truth (m135). The fix is surrogate full-LC MSE on m103's existing pool, not a new cost shape.
2. **The "two-part estimator" you remember (m103 → m115) is still load-bearing.** The break point is m103's *ranking* of its own pool, not the m115 q-finder.

The next session priority is **Play 1 (consolidate)**: re-rank m103's existing m048 pools using the m133 3-cost union, run m115 + m126 + hi-fi on top-K=7 ω per seed, report ρ-band yield over candidate sets. Plays 2 (constrained-anchor on sampling-failure seeds) and 3 (6-DOF surrogate DE upstream) follow if Play 1 doesn't close the gap. Full plan in `CURRENT_STATE.md`.

The success metric is **ρ-band candidate-set yield**, not single-pointer q0_err. C-band overlays are qualitatively beautiful and count.

---

## 1. Resume Point

- **🎯 2026-04-22 (late afternoon) — m134 first end-to-end pipeline test of m133 offline finding.** Patched `m115_surrogate_pipeline.py::load_omega_candidates` with `M115_SORT_BY=surr_q0polish_mse`; ran invert.py (skip-m103) on 3 Band-D seeds (59, 64, 67) where `pick_target_seeds.py` predicted the single-cost swap would uniquely add a truth-close (<20°) ω to m103's top-3.

  **ρ-band results** (ρ = √hifi/0.05, Band A: ρ<2):
  | seed | baseline | new | mechanism |
  |:---:|:---:|:---:|---|
  | 59 | ρ=27.6 D | **ρ=1.02 A** | ω=3.65° → 3/10 DE starts at q0_err=8.81° → m126 polish → q0=0.21°, w_dir=0.05°, w_mag=−0.00% |
  | 64 | ρ=30.2 D | ρ=26.9 D | ω=14.63° at slot 1, but 0/10 DE starts bridged it; winner came from wrong ω=50.9° |
  | 67 | ρ=33.1 D | **ρ=3.56 B** | ω recovered perfectly (w_dir=0.3°, w_mag=−0.08%) but q0 snapped to twin at 179.6° |

  **Key finding — m115 ω-bridging radius is ~3-5°, not 15°.** The ranking fix is necessary but not sufficient: if m103 ω is >5° from truth, m115's 10-start DE can't locate the truth-adjacent q0 basin.

  **Interpretation**: single-cost `surr_q0polish_mse` gained +1 clean Band-A rescue over baseline (matches the m133 offline prediction +1 over 17 seeds). The K=3 triple's 15/17 prediction is still end-to-end untested.

  **Artefacts**: `notebooks/inversion/14_rerank_experiment/{pick_target_seeds.py, run_q0polish_rescue.sh, compare_q0polish_rescue.py}`, `data/results/inversion_diagnostics/rerank_experiment/pipeline_test_2026_04_22/{REPORT.md, comparison.json, baseline_snapshots/, logs/}`, `data/results/inversion_diagnostics/invert_m048_seed{059,064,067}/result.json`.

  **Wiki**: new page `notebooks/inversion/wiki/wiki/experiments/m134_pipeline_test_q0polish.md`; m133 page updated with end-to-end section; log entry appended.

- **🎯 2026-04-22 (afternoon) — rerank-experiment session. Problem: m103's top-3 ω pool by `geo_cost` misses the truth-closest candidate on 12 of 17 rescuable seeds (Band-D failure dominant). Approach: score all 26 saved `geo_ckpt` candidates per seed with a surrogate-based catalog of LC costs + per-candidate q0 polish, measure top-3 hit rate on pool_min_w_err<20° seeds.**

  **Result on m048 (17 rescuable of 22):**
  - Best single cost `surr_q0polish_mse`: 10/17 top-3, 9/17 top-1 (baseline geo_cost 9/17 top-3, 5/17 top-1).
  - Best K=3 triple union `{surr_q0polish_mse, surr_peak_time, surr_autocorr}`: **15/17 top-3** (88% — matches pre-q0_polish oracle ceiling).
  - Best K=3 quadruple: **16/17 top-3** (adding `surr_q0marg_bright_mse`).
  - Best K=5 quintuple: **17/17 top-3** (oracle ceiling, avg union ~15 candidates).

  **m046 cross-validation (13 seeds with geo_ckpt, all rescuable):** every surrogate cost beats `geo_cost` (7/13) with `surr_autocorr`, `surr_spectrum`, `surr_detrend_mse`, `surr_mse`, `surr_mae`, `surr_envelope` all hitting 11/13 top-3. `surr_peak_time` was m048-overfit (7/13 on m046). Robust single cost = **`surr_autocorr`** (10/17 on m048, 11/13 on m046).

  **Cross-validated recommended triple**: `{surr_autocorr, surr_q0polish_mse, surr_spectrum}` — each a top-3-hit winner on at least one cohort and non-embarrassing on the other.

  **Irreducibility finding**: the pre-q0_polish ceiling of 15/17 was limited by seeds 64, 78, 99 whose truth-closest candidate had `q0_err > 145°`. Random q0 perturbations (σ=15°) couldn't bridge it; Nelder-Mead polish with ±X-twin restart base did. This is the key insight for rescuing the q0-distorted candidates in the pool.

  **Scope**: solves the RANKING half of m103's pool handoff to m115. Does NOT address the 5 seeds (47, 51, 79, 84, 89) whose pool never contained an ω within 20° — those need m103 Step-2/3 sampling changes (next problem).

  **Artefacts**: `notebooks/inversion/14_rerank_experiment/{rerank.py, q0_polish_cost.py, rerank_m046.py, costs_extras.py, diagnose_v2.py, cost_set_cover.py, investigate_irreducible.py}`, `data/results/inversion_diagnostics/rerank_experiment/{FINDINGS.md, rerank_results.json, seed_XXX_*.json, seed_XXX_lcmatrix.npz}`, `data/results/inversion_diagnostics/rerank_experiment_m046/`, `data/results/inversion_diagnostics/m048_25seed_dissection_2026_04_22.md`.

  **Wiki**: new page `notebooks/inversion/wiki/wiki/experiments/m133_rerank_geo_ckpt_pool.md`.

- **🎯 2026-04-22 — first honest RANDOM m048 25-seed cohort. Noise realisation across m103/m115/m126 unified (canonical `default_rng(42)` via `lib.traj_source.canonical_observed_lc`). Graceful 8-min m103 Step 4 timeout landed. Phase-B 67% "at noise floor" narrative does not generalise on random sample.**

  **Seeds**: `default_rng(42).choice(100, 25)` = [6, 7, 8, 11, 16, 17, 34, 35, 42, 45, 47, 48, 51, 57, 59, 64, 67, 69, 71, 78, 79, 84, 89, 91, 99].

  **Population by ρ-band** (ρ = √hifi_MSE / 0.05, σ_noise = 0.05 mag):

  | Band | n | % | seeds |
  |------|---|---|-------|
  | A (ρ<2) | 4 | 16% | 6, 17, 45, 91 |
  | B (2-4) | 0 | 0%  | — |
  | C (4-8) | 1 | 4%  | 79 |
  | D (ρ≥8) | 17 | 68% | 7, 8, 11, 16, 34, 47, 48, 51, 57, 59, 64, 67, 71, 78, 84, 89, 99 |
  | geo_timeout | 2 | 8% | 35, 69 |
  | m103 crash | 1 | 4% | 42 (empty `spec_peaks`) |

  **Key diagnostic**: every Band D winner has `ω_dir_err > 19°` (most 50-170°). m103's honest top-3 ω pool does not contain a near-truth ω. Bottleneck is m103 upstream ω selection — surrogate-first front-end rewrite (Roberto path-forward #1) now strongly motivated.

  **Noise-fix bonus**: seed 49 rerun under canonical noise dropped ρ 1.67 → 1.03, ω_dir err 0.89° → 0.094°, q₀ err 1.15° → 0.21°. m126 was using legacy `np.random.seed(42)` (Mersenne Twister) producing different noise than m115's `default_rng(42)` (PCG64). Correctness fix, not cosmetic.

  **Infra delivered tonight**:
  - `notebooks/inversion/lib/traj_source.py` — new `canonical_observed_lc` single source of truth, load_truth returns `observed_lc` in its dict, m103/m115/m126 all consume `truth['observed_lc']`.
  - `notebooks/inversion/11_casadi_formulation/m103_hybrid.py` — Step 4 `pool.map_async().get(timeout=MICRO103_GEO_TIMEOUT_S=480)` with graceful `geo_timeout.flag` + minimal result.json, exits 0.
  - `notebooks/inversion/invert.py` — detects the flag, skips m115/m126/wrappedbest, writes invert summary tagged `status=geo_timeout`.
  - `notebooks/inversion/12_brightness_surface/m126_wrapped_pipeline.py` — `setup_experiment(skip_true_lc=True)` (saves ~50 s/seed) + attaches `truth['observed_lc']` to ctx.
  - `notebooks/inversion/run_m048_batch.py` (new) — serial batch driver with 25-min safety-net wall cap, resume marker `.noise_fix_v1.done`, batch_log.jsonl, ρ-band summary.

  **Open blockers**: (1) m103 ω-pool miss-rate on random m048 = 68%+ Band D; (2) m103 Step 1 crash on empty spec_peaks (seed 42) — needs graceful-skip path; (3) m115 Step 1 is serial (`workers=1` inside a nested `for ic × for si` loop) — 5-7× speedup trivially available by Pool-wrapping.

  **Commits**: `36e724c` (pipeline code), `db46756` (data artifacts).

  **Artefacts**: `data/results/inversion_diagnostics/batch_m048_v1/batch_summary.json`, `batch_log.jsonl`, per-seed `wrappedbest_m048_seed{NNN}/`, `invert_m048_seed{NNN}/`, `m126_wrapped_m048/seed_{NNN}/`, `m115_surrogate_pipeline_m048/seed_{NNN}/`, and `m103_hybrid_m048/seed_{028,035,069}/geo_timeout.flag`.

  **Wiki**: new page `notebooks/inversion/wiki/wiki/branches/m048-random-cohort-baseline.md`. Writeup in `notebooks/inversion/12_brightness_surface/FINDINGS.md` "random m048 25-seed baseline" section.

- **🎯 2026-04-21 — m115 oracle-ω bug FIXED. Phase-B m048 6-seed cohort reran end-to-end under honest `geo_cost` sort. Cohort tally {4 OK, 1 PARTIAL, 1 FAIL} unchanged from the 2026-04-17 oracle-tainted numbers — pipeline's cost-based downstream (m115 DE + m126 L-BFGS polish) is robust to oracle contamination in the m103 candidate pool. *(Phase-B cohort now known to be cherry-picked — see 2026-04-22 entry for honest random-sample numbers.)***

  **Bug**: `m115_surrogate_pipeline.load_omega_candidates` sorted m103's ω candidate pool by `w0_ref_errs` (ω-direction error vs truth — ground-truth field) instead of `geo_costs` (alignment-cost, the honest inversion-available signal). Sibling scripts `m113_de_attitude_search.py` and `m114_surrogate_multistart.py` had the same antipattern (standalone, not in live pipeline). All three fixed; `M{113,114,115}_SORT_BY=oracle` env var reproduces old behaviour for diagnostics.

  **Per-seed Phase-B verdict after honest rerun** (from `data/results/inversion_diagnostics/m115_oracle_bug_audit_phaseB_wrappedbest.json`):

  | seed | phase_med | oracle cls | oracle hifi | honest cls | honest hifi | change |
  |---:|---:|:---:|---:|:---:|---:|:---|
  | 023 | 30.4° | FAIL | 1.597 | FAIL | 2.296 | honest worse (lost truth-close ω); still constraint-poor FAIL |
  | 024 | 15.3° | PARTIAL | 0.0338 | PARTIAL | 0.0338 | IDENTICAL |
  | 049 | 38.9° | OK | 0.0070 | OK | 0.0070 | IDENTICAL |
  | 081 | 45.6° | OK | 0.0026 | OK | 0.0043 | m126 polish heroically bridged m115 hifi 0.70→0.004 |
  | 090 | 50.8° | OK | 0.0024 | OK | 0.0024 | IDENTICAL |
  | 091 | 51.7° | OK | 0.0025 | OK | 0.0025 | IDENTICAL (only seed where both sorts gave same top-3 set) |

  **Divergence audit** (`audit_m103_sort_divergence.py`): oracle vs geo_cost top-3 differ on every m046 seed with geo_ckpt (0/13 identical) and on 5 of 6 Phase-B m048 seeds. But as shown above, cost-based downstream selection hides the pool difference at the verdict level in nearly all cases. The bug's danger was cohort-level claims about WHICH seed SET passes; the per-seed verdicts turn out to be robust.

  **Historical scope** (pre-2026-04-21 outputs): For m046 Phase-A, 5 of 10 baseline seeds (000, 014, 024, 027, 046) consumed geo_ckpt and were bug-exposed; the other 5 used `m102 result_npz` fallback (oracle-free). No cohort-level m046 rerun scheduled — low priority until m046 numbers become load-bearing again.

  **Artefacts**: full pre-fix snapshots at `m115_surrogate_pipeline_m048_oracle_baseline/`, `m126_wrapped_m048_oracle_baseline/`, `wrappedbest_m048_oracle_baseline/`. Divergence / audit JSONs at `m103_sort_divergence_audit.json`, `m115_oracle_bug_audit_m048.json`, `m115_oracle_bug_audit_phaseB_wrappedbest.json`.

  **Scripts touched this session**: `notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py` (fix), `m113_de_attitude_search.py` (fix), `m114_surrogate_multistart.py` (fix), `notebooks/inversion/invert.py` (fallback for pre-existing `hifi_mags_before` key-missing bug), `notebooks/inversion/12_brightness_surface/audit_m103_sort_divergence.py` (new), `notebooks/inversion/12_brightness_surface/compare_m115_oracle_vs_geocost.py` (new).

  **Wiki ingest**: no new pages this session — the existing `m115-oracle-bug` note (if present) should be updated to fixed; otherwise the FINDINGS.md entry in `notebooks/inversion/12_brightness_surface/FINDINGS.md` is the primary record.

- **🎯 2026-04-17 (night) — PHASE 2 PILOT EFFECTIVELY COMPLETE ON 8 m048 SEEDS. Two distinct upstream failure modes disentangled. Phase 3 blocked on design decision: patch m103 vs replace with 6-DOF surrogate-DE. No further m048 compute recommended until user picks a path. *(Numbers in the subsections below are now proven honest by the 2026-04-21 rerun — verdicts unchanged.)***

  **8-seed Phase-B cohort** (all numbers from on-disk `invert_m048_seedNNN/result.json` or `m103_hybrid_m048/seed_NNN/pipeline.log`):

  | seed | phase_med | cls | hifi | q0_err | w_dir | w_mag | notes |
  |---:|---:|:---:|---:|---:|---:|---:|:---|
  | 024 | 15.3° | PARTIAL | 0.034 | 5.03° | 0.99° | -0.16% | borderline; inherited from prior session |
  | 091 | 51.7° | OK | 0.0025 | 0.05° | 0.01° | -0.00% | inherited from prior session |
  | 028 | 87.3° | upstream-FAIL | — | — | — | — | m103 Pool(24) geo hang + NM truth missing; inherited |
  | 081 | 45.6° | OK | 0.0026 | 179.93° | 0.03° | -0.01% | ±X twin; this session |
  | 069 | 67.7° | killed | — | 28.8° (NM best) | — | — | killed pre-downstream — NM pool doesn't contain truth |
  | 090 | 50.8° | OK | 0.0024 | 0.15° | 0.06° | +0.00% | this session |
  | 049 | 38.9° | OK | 0.0070 | 1.15° | 0.89° | +0.03% | this session |
  | 023 | 30.4° | **FAIL** | 1.597 | 68.29° | 175.29° | +35.37% | constraint-poor (2 spec peaks); this session |

  **Tally:** 4 OK + 1 PARTIAL + 1 downstream-FAIL + 2 upstream-FAILs on 8 seeds.

  **Two failure modes** (disentangled, not confounded):
  1. **High-phase (> 65°)**: m103's alignment cost surface flattens. NM's top-20 can't localise truth. Seeds 28 + 69 both had 10-11 spec peaks (NOT constraint-poor) but NM best ω-error was 30.8°/28.8° respectively — way outside m126's ~1-5° polish basin. Seed 28's Pool(24) geo hang is a secondary symptom (L-BFGS-B burns unbounded function evals on flat cost).
  2. **Constraint-poor (seed 23 @ 30° with 2 spec peaks)**: Alignment cost degenerate when only 1 constraint remains after anchor pick. NM top-20 all have `geo_cost ~ 1e-20`. Truth is at rank #8 (q0=172.8°, ω_err=5.1°) but unselectable by geo_cost.

  **Reliable operating band** (synthesising with m046): ~**38-55° phase with ≥ 4 spec peaks**. m046's cohort of 11 seeds (5 OK / 4 PARTIAL / 2 FAIL) all sat in the 30-60° shared window; the "success rate" measurement was biased upward by the 50-60° constraint-rich sub-band (median 10 spec peaks per `scan_m048_constraints_vs_phase.py` output).

  **New/updated wiki pages** (this session):
  - NEW `[[phase_B_m048_cohort]]` — central 8-seed results page with all 3 errors per seed and failure-mode diagnosis
  - NEW `[[constraint-poor-regime]]` — concept page for seed-23-type failure
  - NEW `[[phase-angle-operating-range]]` — synthesis concept
  - NEW `[[upstream-redesign-6dof-surrogate-de]]` — `#proposed` branch page for the patient-path option
  - NEW `[[m103_seed028_geo_hang]]` — investigation of the specific hang
  - UPDATED `[[m048-migration]]` — Phase 2 outcomes + Phase 3 decision-point framing
  - UPDATED `[[alignment-cost]]` — high-phase flatness subsection

  **Phase 3 decision point (user action required).**
  - **Quick path**: patch m103 (robust geo timeout + 8 checkpointing enrichments per audit) + launch 100-seed batch. ~3 hr work + ~24 hr compute. Expect 30-40% upstream-FAIL based on cohort evidence.
  - **Patient path**: design + validate a single-stage 6-DOF surrogate DE replacement for m103. Gate on m046 11-seed cohort match + Phase-B FAIL-seed rescue (especially seed 23 which is surrogate-DE's best case — constraint-poor alignment failure + full LC available for surrogate scoring). ~1-2 sessions work + 70 min validation compute + 15-20 hr batch. See [[upstream-redesign-6dof-surrogate-de]] for gate criteria.
  - **Current recommendation:** patient path. Both exposed failure modes are alignment-cost pathologies; the surrogate-LC cost that already drives m115's inner DE would sidestep both by construction.

  **Checkpoint enrichment audit** (done this session, not yet implemented — see `project_checkpointing_audit_2026_04_17.md` memory):
  8 gaps identified — m103 grid-stage top-500 not saved, step 2b lo-fi MSE not saved, step 3 pre-dedup NM results not saved, step 3.5 phi-sweep cost curves not saved, step 4 L-BFGS-B stopping metadata not saved, m115 DE solutions stored as JSON string (not arrays), m115 hi-fi LCs not saved, m126 hi-fi LCs BEFORE polish not saved. Plus trivial: observed_lc + spec peak epochs/mags + anchor epoch + |ω|_est in structured JSON. Total extra storage ~150 KB/seed. Deferred to whichever upstream design path is chosen.

  **Script provenance.** All numerical results in this session came from the committed `invert.py` chain running `m103_hybrid.py` + `m115_surrogate_pipeline.py` + `m126_wrapped_pipeline.py`. No inline computations produced any quoted hi-fi/q0/ω numbers. The three NEW uncommitted scripts this session (`notebooks/inversion/select_phase_seeds_v2.py`, `notebooks/inversion/select_phase_descending.py`, `notebooks/inversion/scan_m048_constraints_vs_phase.py`) were seed-selection and population-characterisation utilities only — none of them produced results that feed the cohort table.

  **Pre-existing uncommitted items** (leave alone, unrelated): `.claude/skills/research-loop/SKILL.md`, `data/results/inversion_diagnostics/m115_surrogate_pipeline/batch_summary.json`, `notebooks/inversion/11_casadi_formulation/run_micro103_*.sh` wrappers, `notebooks/inversion/lib/attitude_anim.py`, `reports/2026-04-10-roberto/`, `.claude/*`.

  **This session's uncommitted items** (to commit with handoff): `notebooks/inversion/select_phase_seeds_v2.py`, `notebooks/inversion/select_phase_descending.py`, `notebooks/inversion/scan_m048_constraints_vs_phase.py`, `notebooks/inversion/run_phase_B_descending.sh`, `notebooks/inversion/11_casadi_formulation/retry_geo_serial.py`, `notebooks/inversion/run_phase_B_pilot.sh`, plus all the per-seed artefacts under `data/results/inversion_diagnostics/invert_m048_seed*/`, `m103_hybrid_m048/seed_*/`, `m115_surrogate_pipeline_m048/seed_*/`, `m126_wrapped_m048/seed_*/`, `wrappedbest_m048_seed*/`, `phase_B_*_seeds.json`, `phase_B_cohort_summary.json`, `phase_B_gap_logs/`, plus wiki and memory updates.

- **🎯 2026-04-17 (late evening) — PHASE B PHASE 1 COMPLETE. `lib.traj_source`, `setup_experiment(start_et=...)`, single-entry `invert.py` driver, TRAJ_SOURCE env var threaded through m115 / m126 / lc_compare. Back-compat on m046 verified bitwise. Zero compute ran. Next session = Phase 2 pilot on 3 m048 seeds, but first resolve upstream `m115.load_omega_candidates` gap for m048.**

  **What landed this session** (commit TBD — uncommitted at handoff start).
  1. `notebooks/inversion/lib/experiment_setup.py` — `setup_experiment()` now accepts `start_et: Optional[float]` and `duration_s: float = 3600.0`. When `start_et` is given, epochs span `[start_et, start_et + duration_s]`; else legacy `end_time_utc` path. m046 call sites that pass `end_time_utc='2020-02-05T11:00:00'` are unchanged.
  2. `notebooks/inversion/lib/traj_source.py` (NEW) — `load_truth(seed, source)` returns uniform dict `{q0_wxyz, omega0_rad, mag_hifi, observation_times, inertia_tensor, start_et, end_time_utc, duration_s, source, seed}`. Valid sources: `'m046'`, `'m048'`. Also `seeds_for_source(source)` helper.
  3. `notebooks/inversion/lib/lc_compare.py` — new `--traj-source` CLI arg; truth + ctx now loaded per-seed via `load_truth`.
  4. `notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py` — reads `TRAJ_SOURCE` env var; `OUT_BASE` tagged (`m115_surrogate_pipeline_m048/` for m048); `run_seed` takes a `truth` dict instead of `master + seed-index` lookup; m046 keeps the shared-ctx fast path, m048 builds per-seed ctx (~50 s/seed overhead).
  5. `notebooks/inversion/12_brightness_surface/m126_wrapped_pipeline.py` — reads `TRAJ_SOURCE` and `MICRO126_SEEDS` env vars; `MICRO115_BASE` and `OUT_BASE` source-tagged; `build_seed_context` threads `start_et`/`end_time_utc` into `setup_experiment`.
  6. `notebooks/inversion/invert.py` (NEW) — `python3 invert.py --seed N --traj-source {m046|m048}`. Chains m115 → m126 → wrappedbest refresh → lc_compare as subprocesses (env inheritance). Flags: `--skip-m115`, `--skip-m126`, `--skip-lc-compare`. Summary JSON at `data/results/inversion_diagnostics/invert_{source}_seed{NNN}/result.json`.

  **Verified.**
  - Back-compat: `setup_experiment(end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)` produces `observation_times` bitwise-identical to `m046_trajectories.npz['observation_times']`.
  - m048 path: `setup_experiment(start_et=634173431.91116, duration_s=3600.0)` matches `m048/per_trajectory/traj_seed000.npz` `dt_sampling` to <1e-9 s.
  - End-to-end: `python3 notebooks/inversion/invert.py --seed 0 --traj-source m046 --skip-m115 --skip-m126 --skip-lc-compare` reconstructed seed 0's wrappedbest payload from m126's `hifi_ckpt.npz` in 5 ms, reporting `q0=0.54° w_dir=0.02° w_mag=+0.01% hifi=0.00367 [OK]` — bitwise-identical winner values to the 2026-04-17 cohort audit. Dry-run outputs were reverted (wrappedbest_seed000/result.json restored to m131's "postfix 2026-04-17" label; invert_m046_seed000/ dir removed) — m131's file stays authoritative.

  **Known Phase 2 blocker.** `m115.load_omega_candidates` looks for `m103_hybrid/seed_NNN/geo_ckpt.npz` (primary) or `m102_fullmse/seed_NNN/result.npz` (fallback). Both of those upstream experiments ran on m046 only. For any m048 seed lacking these, m115 will exit with `error: 'no_omega_data'`. Options before pilot: (a) harvest m103-style geo_ckpts for the 3 pilot seeds, (b) retrofit a surrogate-DE-over-omega stage upstream of m115's existing surrogate-DE-over-q0, or (c) stub an oracle-omega path strictly for pilot triage (cleanly labelled so it never contaminates cohort claims).

  **Wiki ingest this session.**
  - NEW [[m048-migration]] branch page — 3-phase plan, Phase 1 complete, Phase 2 next.
  - NEW [[observation-geometry-sources]] concept page — m046 vs m048 dichotomy + `load_truth` contract + `observation_times` invariant.
  - [[gradient-based-inversion]] branch: added a pointer to [[m048-migration]] and scoped `confidence: high` to "on m046 cohort."
  - `index.md`: Bug-1 banner updated; added new concept + branch rows.
  - `log.md`: 3 entries (infra + branch + concept).

  **Context at wind-down.** Uses `/loop` + wind-down sequence. Deliberately stopped before Phase 2 pilot — the omega-candidate gap deserves dedicated design thinking, not a quick hack in a tail-end session.

  **Pre-existing uncommitted items** (leave alone, unrelated to Phase 1): `.claude/skills/research-loop/SKILL.md` (M, pre-existing tooling edit), `data/results/inversion_diagnostics/m115_surrogate_pipeline/batch_summary.json` (M, pre-existing seed-46 overwrite), untracked `.claude/*`, untracked `run_micro103_*.sh` wrappers, untracked `lib/attitude_anim.py` + related helpers, untracked `reports/2026-04-10-roberto/`.

- **🎯 2026-04-17 (evening) — COHORT AUDIT + PHASE B PLAN. 11-seed post-fix cohort verified; 11 lc_compare plots regenerated with cached hi-fi LCs; solution-count-by-band analysis landed; Phase B ("bow + m048 migration") scoped but not started. Next session = Phase 1 of Phase B.**

  **Cohort audit (this session).** Verified the EXPERIMENTS.md claim of 5 OK / 4 PARTIAL / 2 FAIL against actual m126_wrapped/seed_NNN/result.json and m125_keep_better/summary.json. Classification holds with one caveat: seed 74's hi-fi 0.01091 is 0.00091 above the strict OK threshold (0.01) — the claim is slightly generous. Honest yield under multi-solution philosophy: **16/33 basins below hi-fi 0.1** (48%), **5/33 below 0.01** (15%), **9/33 FAIL ≥ 0.3** (27%). See `data/results/inversion_diagnostics/m132_solution_count_by_band.png` for stratified distribution.

  **LC plots regenerated (all 11).** Old `m126_wrapped/wrappedbest_seed*` dirs were from 2026-04-16 (pre-fix) and pointed at stale winners for 7/11 seeds (e.g. seed 12 wrappedbest was `m126_basin0_before` at hi-fi 0.327; post-fix winner is `m126_basin0_after` at 0.00265). Wrote `m131_refresh_wrappedbest.py` to rebuild 11 fresh dirs at `data/results/inversion_diagnostics/wrappedbest_seed{NNN}/` using post-fix winners + cached hi-fi LCs from m126's `hifi_ckpt.npz['hifi_mags_after']` and m124's `hifi_results.npz['hifi_mags']`. **Zero hi-fi regeneration** — checkpoints already had the LCs. `lib/lc_compare.py` edited to plot truth as blue dots (markersize=3) instead of solid line. 11 PNGs at `data/results/inversion_diagnostics/wrappedbest_seed{NNN}_lc_compare.png`.

  **Correction.** My earlier cohort table said seed 6 winner was basin_1 (truth-adjacent, q0_err 0.40°). Actual winner is basin_0 (±X twin at 179.6°); basin_1 is basin_0 + 3e-4 in hi-fi. Twin and truth give identical LCs by physics so the wrapper's pick is essentially coin-flip.

  **Phase B is the next action.** m048 is already generated (100 per-trajectory .npz files with per-seed start_et, mag_hifi, omega0_rad, phase_angle_3d, full geometry). **No fresh trajectory generation needed.** m048 phase angle range: **9.2°–95.2° across seeds** (m046 was 30–60° on the one window). Full plan in `project_phase_B_bow_plan.md` memory file. Three phases:

  - **Phase 1 (refactor, zero compute, ~1 session):** Refactor `lib/experiment_setup.py` to accept `start_et` (keep `end_time_utc` backwards-compat kwarg). Add `lib/traj_source.py` loader abstraction. Audit ~10 consumer scripts. Write `notebooks/inversion/invert.py --seed N --traj-source {m046|m048}` single-entry driver that chains m115 → m126 → wrappedbest refresh → lc_compare.
  - **Phase 2 (pilot, ~30 min compute):** Run `invert.py` on 3 m048 seeds at low/mid/high phase angles. Go/no-go gate for Phase 3.
  - **Phase 3 (batch, ~5–15 hr):** Apply two cheap perf fixes first: m115 `differential_evolution(..., workers=-1)` (needs closure → callable class conversion, saves ~3 min/seed) + m115 hi-fi `Pool(3)` over 3 basins (saves ~100 s/seed). Brings per-seed wall from ~9 min to ~5 min. Then run 100 m048 seeds.

  **Per-seed cost (measured from m126 batch_summary + m115 timing notes):** current code ~9 min/seed (ctx 53s + m115 DE/polish 180s + m115 hi-fi 150s + m126 polish 36s + m126 hi-fi 90s). With perf fixes ~5 min/seed.

  **Phase-angle risks I flagged** (honest, not exhaustive): (1) high-phase >80° seeds hit [[dark-mag-saturation]] plateau harder — more LC spent near-terminator — expect more FAIL cases; (2) low-phase <20° opposition regime not characterised on surrogate vs hi-fi — worth an audit pilot; (3) bright ±X peak availability (micro96 said 87% lacked them on m046) unknown on m048; (4) ω-dir basins may be even narrower at high phase, 50k DE population could miss them.

  **Session artifacts** (committed together): `notebooks/inversion/12_brightness_surface/m131_refresh_wrappedbest.py`, `m132_solution_count.py`, edit to `lib/lc_compare.py`, 11 post-fix `wrappedbest_seed{NNN}/` dirs + PNGs, `m132_solution_count_by_band.png` + `.json`.

  **Context at wind-down: 20.1%** (ctx.sh verified, not vibe). Deliberately stopped before starting Phase 1 — refactor touches ~10 files and deserves a fresh session with full headroom for debugging + commits.

  **Pre-existing uncommitted items** (leave alone, unrelated to this session): `.claude/skills/research-loop/SKILL.md` (M), `data/results/inversion_diagnostics/m115_surrogate_pipeline/batch_summary.json` (M, pre-existing seed-46 overwrite), untracked `.claude/*`, `run_micro103_*.sh` wrappers, `lib/attitude_anim.py`/`anim_template.html`/`brightness_surface.html`, `reports/2026-04-10-roberto/`.

- **✅ 2026-04-17 — OPTION A COMPLETE. Bug 2 fixed; m122/m123/m124/m125/m126 re-run on correct 1-hour window. The bug hid BETTER results. [[gradient-based-inversion]] `#validated` REINSTATED.**

  **What happened this session.** The user picked Option A from the data-integrity bug doc. Chain executed:

  1. Added `end_time_utc='2020-02-05T11:00:00'` to m122/m124/m126's `setup_experiment` calls (commit `5d5938f`).
  2. Renamed `m046_trajectories/micro46_trajectories.npz` → `m046_trajectories.npz` (and m048 equivalent) — leftover from Phase A cleanup (commit `372ebec`).
  3. Quarantined `m119v2/seed_NNN/setup.npz` and `m122/seed_{074,093}/setup.npz` as `setup_6hr_WRONG_WINDOW_retracted.npz` — they were 6-hour caches that m122 was inheriting. **m119v2 turned out to have the same Bug 2 as m122/m124/m126** — the bug doc's "m119v2 CORRECT" claim was wrong (it lacks `end_time_utc` in its `setup_experiment` call).
  4. Re-ran m122 (with assertion fixed: was checking `>20000 s`, now `3500–3700 s`), m123, m124, m125, m126 on correct window (commit `b906691`).

  **Correct-window 11-seed cohort (was 6/11 ≥10% improved, 18/33 basins helped, 5/11 break-even):**

  | cohort | new count |
  |--------|----------:|
  | improved ≥10% | **10/11** |
  | break-even | 1/11 (seed 33 flipped-ω only) |
  | regressed | 0/11 |
  | basins helped | **33/33** |

  Classifications (hi-fi MSE): **5 OK** (0, 6, 12, 74, 93), **4 PARTIAL** (14, 24, 27, 33), **2 FAIL** (36, 46). The wrong-window reading inflated MSE values 2–60× and made the polish look partially catastrophic; on correct window polish is genuinely global in q0 + ω for most basins, and seed 33's flipped-ω is the only real break-even case (retrograde attractor persists at hi-fi 0.0796 — physics, not bug).

  **What changed qualitatively on correct window:**
  - m122 HYP3 (basin-geometry cohort universality) REFUTED (was CONFIRMED). ATT_FAIL and OK cohorts have basin-widths within 1.20× — **basin geometry does not explain ATT_FAIL**. The driver is upstream ω-direction quality.
  - m123 HYP3 REFUTED for all 5 seeds (n_attractors=4 > n_basins=3). L-BFGS from truth finds a separate attractor from the DE basins, consistent with DE basins not centred on truth.
  - m124 verdict flipped REFUTED (3/15) → PARTIAL (9/15).
  - Seeds 36 and 46 remain FAIL (hi-fi 0.39 and 0.14) — wrapped pipeline cannot fix bad upstream ω-direction candidates. Next lever: harvester for multi-candidate ω ([[harvester-optimization]] — currently #open, low priority).

  **Files updated in the wiki sweep commit (`TBD`):** `branches/gradient-based-inversion.md` (#validated reinstated, new post-fix block, new combined-cohort table), `experiments/m122_hessian_curvature.md` / `m123_lbfgs_polish.md` / `m124_hifi_validate.md` / `m125_keep_better_inline.md` / `m126_wrapped_pipeline.md` (post-fix banners with fresh numbers; m126 has new full per-basin polish-motion table), `concepts/omega-sign-degeneracy.md` / `basin-of-attraction.md` / `surrogate-model.md` (banners updated; surrogate-truth-offset formally retracted), `index.md` top banner.

  **Bug 1 still open.** All of the above still holds only on the m046 single-geometry window (10:00–11:00 UTC). Every "cohort" claim remains a single-geometry claim. Option B (migrate to m048 per-seed random windows) is the honest fix; still deferred. The surrogate is geometry-agnostic (per-epoch (k1,k2,articulation) inputs, uniform random training), so Option B's main cost is plumbing + re-run time, not retraining.

  **DO NOT run new experiments beyond this Option A re-run without a fresh decision.** Next action items, in order of strategic value: (1) Option B sizing — audit how much of the pipeline actually needs threading per-seed `start_et` through `setup_experiment`; (2) `geo_ckpt` harvest for seeds 12/33/36/46 to give them multi-candidate upstream ω (would likely rescue seed 36 and maybe 46); (3) seed 33 flipped-ω population frequency on the remaining 89 seeds (m130-style DE-over-q0 at `ω=−ω_true`).

- **🏁 2026-04-16 — POST-CLEANUP RESUME POINT. All organizational work complete. The action gate is the data-integrity bug remediation (Option A / B / C, below) — DO NOT run new experiments until the user chooses.**

  **Narrative companion.** See [[wiki/journey.md]] for the chronological m001 → m129 research arc. Use this file to figure out *what to do next*; use journey.md to figure out *what already happened and why*.

  **Cleanup complete (commits `5f4108c` → `49f2c5c`).** Seven commits landed in sequence:
  - `5f4108c` — foundation docs: `/DATA_INVARIANTS.md`, `notebooks/inversion/CONTRIBUTING.md`, `notebooks/inversion/RENAMES.md` (old → new map).
  - `026d85d` — 272 scripts renamed `microNN_*.py` → `m{NNN}_{semantic_name}.py`. Series-07 collisions consolidated. `00_pipeline_reference/` moved to `notebooks/tutorials/`. `06_*` series rationalised.
  - `813cbf0` — 39 wiki experiment pages renamed. 531 `[[microNN]]` → `[[m{NNN}_semantic]]` rewrites. `EXPERIMENTS.md` + `DATA_INTEGRITY_BUG.md` links updated.
  - `494f513` — prepended the cleanup handoff block to Section 1 (now replaced by this block).
  - `1c0000d` — ~286 result dirs under `data/results/inversion_diagnostics/` renamed; `shared/`, `analyses/`, `archive/` subdirs created; `wrappedbest_seed*/` folded into `m126_wrapped_pipeline/`. 1765 `\bmicro(\d+)` occurrences in 209 `.py` files rewritten.
  - `8e01066` — wiki + reports + FINDINGS mechanical sweep: 1268 prose tokens, 141 script paths, 3 residual links; `updated:` bumped on 66 pages.
  - `04f8d36` — bug-invalidated hi-fi numbers flagged (wrong-window banners on `m122`–`m125` experiment pages + 3 concept pages).
  - `7ca7afc` — `branches/gradient-based-inversion.md` `## Status:` line updated to `#validated-RETRACTED`.
  - `49f2c5c` — `wiki/journey.md` added; chronological narrative from m001 → m129 organised by era with revivable-dead-end flags and cross-era idea connections; linked from `wiki/index.md`.

  **Cleanup tools preserved.** `scripts/cleanup_rename.py`, `scripts/cleanup_wiki_rename.py`, `scripts/cleanup_data_rename.py`, `scripts/cleanup_wiki_content.py`, `scripts/cleanup_experiments_md.py`. Reuse the patterns for any future large-scale rewrites — they already handle `git mv` blame preservation, untracked-file `shutil` fallback, and the `\bmicro(\d+)([a-z]\d*)?` token regex.

  **Action gate — choose A, B, or C.** The bug discovered on 2026-04-16 post-midnight (next block) blocks all further experiments. `DATA_INTEGRITY_BUG.md` has the full audit. Short form:

  - **Option A (short-term, ~1 session):** fix Bug 2 only. Add `end_time_utc='2020-02-05T11:00:00'` to three scripts — `m122_hessian_curvature.py:328`, `m124_hifi_validate.py:219`, `m126_wrapped_pipeline.py:347` — then re-run `m126` (~15 min) and re-score `m122`/`m123`/`m124`/`m125` from stored states (~20 min). Update wiki claims. Restores honesty within the single-geometry m046 architecture.
  - **Option B (medium-term, ~5–15 sessions):** fix Bug 1 too. Refactor `lib/experiment_setup.py` to accept per-seed `start_et`, migrate every pipeline from `m046_trajectories.npz` to `m048_trajectories.npz`, regenerate the research arc on diverse per-seed geometries. This is probably what the user actually wants — but it's expensive.
  - **Option C (institutional, already landed):** `DATA_INVARIANTS.md`, reviewer-checklist extensions, the `setup_experiment` window invariant. In place since `5f4108c`.

  **DO NOT regenerate experiment data until A/B is chosen.** Running any new pipeline on the wrong-window architecture only adds more invalid data.

  **Working-tree state at resume.** Clean of cleanup work. Pre-existing non-cleanup items to leave alone: `.claude/skills/research-loop/SKILL.md` (M, unrelated tooling edit); `data/results/inversion_diagnostics/m115_surrogate_pipeline/batch_summary.json` (M, pre-existing seed-46 overwrite flagged in the 2026-04-16 evening block below); `notebooks/inversion/lib/{anim_template.html,attitude_anim.py,brightness_surface.html}` (untracked lib helpers); `notebooks/inversion/11_casadi_formulation/run_m103_*.sh` (broken shell wrappers since `026d85d`); `.claude/*` (untracked scratch).

---

- **🚨 2026-04-16 (post-midnight) — CRITICAL DATA-INTEGRITY BUG DISCOVERED. Session stopped. All further research-loop iterations BLOCKED until bug fix is chosen.**

  **See `notebooks/inversion/DATA_INTEGRITY_BUG.md` for the full finding.** Short version:

  - **Bug 1 (architectural):** Every pipeline uses `m046_trajectories.npz` — 100 seeds on a **single fixed observation window** (10:00–11:00 UTC, 2020-02-05). The intended dataset is `m048_trajectories.npz` with **per-seed random start times** in `[08:35, 15:35 UTC]` + 1-hour duration each. **Zero current pipelines use m048.** All "100-seed population" / "cohort-universal" claims in the wiki reduce to "100 trajectories on ONE observation night."
  - **Bug 2 (compounding, script-level):** `m122`, `m124`, `m126` (and `m123`/`m125` by inheritance) call `setup_experiment(...)` **without** `end_time_utc='2020-02-05T11:00:00'` — the config default `16:00:00` silently activated a **6-hour window** vs m046's 1-hour truth. All hi-fi MSE values from these 5 experiments are scored against a DIFFERENT observed LC than `m46['mag_hifi']`.
  - **Evidence (seed 0 m126):** reported `hifi_after=0.1119`; stored `hifi_mags_after[0]` vs m46 truth gives **MSE 6.36** (RMS 2.52 mag, 57× worse than reported); regenerated on correct 1-hour window from same state gives **MSE 0.325** (RMS 0.57 mag, still 2.9× worse than claimed).
  - **Corrupted:** all hi-fi MSE numbers from [[m122_hessian_curvature]], [[m123_lbfgs_polish]], [[m124_hifi_validate]], [[m125_keep_better_inline]], [[m126_wrapped_pipeline]]. Therefore the [[gradient-based-inversion]] `#validated` status, the "wrapped pipeline is the best method" claim, the "6/11 improved ≥10%" table, and every [inline] entry in log.md dated 2026-04-16 that cites one of these hi-fi numbers.
  - **NOT corrupted:** `q0_err`, `w_dir_err`, `w_mag_err_pct` (independent of window); [[m115_surrogate_pipeline]] state discovery (correct window); [[m127_flipped_omega_search]]/[[m128_warmstart_polish]]/[[m129_dense_grid_eval]] flipped-ω thread (correct window); all structural/symmetry observations (valid within the single-geometry caveat of Bug 1).

  **Full audit table of which scripts pass `end_time_utc` is in DATA_INTEGRITY_BUG.md.** The m115 and m127+128+129 scripts are correctly windowed; m122/124/126 are not.

  ### What the next loop MUST decide first

  - **Option A (short-term):** fix Bug 2 only. Add `end_time_utc='2020-02-05T11:00:00'` to the 3 broken scripts, re-run m126 (~15 min), re-score m122/123/124/125 from stored states (~20 min), update all wiki claims. ~1 session.
  - **Option B (medium-term):** fix Bug 1 too. Refactor `lib/experiment_setup.py` to accept per-seed `start_et`, migrate every pipeline from `m046` to `m048`, regenerate the full research arc (surrogate DE, wrapped pipeline, flipped-ω if still interesting) on diverse per-seed geometries. ~5–15 sessions.
  - **Option C (institutional, parallel):** create `DATA_INVARIANTS.md`, extend reviewer checklist for data-plumbing items, add a mandatory 1-seed dry-run + LC-compare sanity step before every full batch launch.

  **DO NOT run any new experiments (including the queued `m130`) until Bug 2 is fixed at minimum.** Running m130 on the corrupted m046+6-hr architecture would just add more invalid data. The `m127-129` flipped-ω thread findings are preserved (correct window) but the larger "wrapped pipeline is the best" narrative they were being compared against is in ruins.

  **Discovered by:** lc_compare of seed 0 wrapped winner (m126 basin 0 after) showing RMS 2.52 mag on the displayed plot despite m126 reporting hi-fi MSE 0.11 — the user flagged the visual contradiction. Traced via stored `hifi_mags_after` vs m46 truth comparison, then via comparison of `observation_times` span in a fresh `setup_experiment(...)` ctx (no `end_time_utc`) showing 21600 s = 6 hours vs m46's 3600 s = 1 hour.

- **2026-04-16 (late-night session) — m129 COMPLETE, REFUTED; flipped-ω grid-enumeration thread closed; m130 queued (unspawned).**
  - **Script:** `notebooks/inversion/12_brightness_surface/m129_dense_grid_eval.py` (writer-authored + reviewer PASS this session; built verbatim from `m127_flipped_omega_search.py` with `N_SO3: 60000→600000`, `TOP_K: 20→50`, `SEEDS=[33]`, `HIFI_POOL_CAP: 4→2`, `OUT_BASE=m129_densegrid`; env vars `MICRO129_*`). Single-seed run, Pool(8) grid, Pool(2) hi-fi, 7.3 min wall.
  - **Results:** `data/results/inversion_diagnostics/m129_densegrid/{batch_summary.json, seed_033/{stage_a_grid.npz, stage_b_polish.npz, stage_c_hifi.npz, result.json, run.log}}`.
  - **Headline: REFUTED.** Winner hi-fi MSE **2.6186** at q0_err 110° (vs <0.15 threshold for CONFIRMED). 10× grid density bought 0.3% improvement in best Stage A surrogate score vs [[m127_flipped_omega_search]] (1.1800 vs 1.1833). Classification: FLIPPED_FAIL.
  - **5 polished basins (all on saturated plateau):** basin 0 surr 0.980 / hi-fi 2.63 / q0_err 159°; basin 1 surr 0.999 / hi-fi 2.70 / q0_err 91.6° (closest to known basin at 98.5°, 7° off in q0); basin 2 (winner) surr 1.000 / hi-fi 2.62 / q0_err 110°; basin 3 surr 1.000 / hi-fi 3.04 / q0_err 141°; basin 4 surr 1.001 / hi-fi 2.78 / q0_err 145°. All cluster at surr ~1.0 / hi-fi ~2.6–3.0 — a shallow plateau, no deep basin found.
  - **Mechanism of failure (load-bearing, non-lazy):** Seed 33's known basin is sub-0.001° wide (per [[m126_wrapped_pipeline]] polish `Δq0 ≤ 0.0004°`). At 600k super-Fibonacci spacing ~1.4° median, the basin is still ~1000× narrower than grid resolution (down from ~10000× at 60k — arithmetically the same regime). More critically, the surrogate cost on the `−ω_true` slice is **saturated at ~1.0 almost everywhere** ([[dark-mag-saturation]] plateau); polish from basin 1 at q0_err 91.55° (only 7° from the known 98.53° basin) cannot walk across the saturation wall — both points sit on the plateau at surr ~0.998, no continuous gradient connects them to the 0.086 pinprick basin. Grid density cannot fix this: would need ~600M points (60 GB surrogate forward) to put a vertex inside the basin, which is infeasible.
  - **Strategic implication.** Three consecutive REFUTED results close the flipped-ω **search-from-nothing** thread: [[m127_flipped_omega_search]] (coarse SO(3) grid finds wide basins only), [[m128_warmstart_polish]] (warm-start polish from m115 basins on the wrong cost manifold), [[m129_dense_grid_eval]] (dense SO(3) grid is density-bounded by saturation plateau). The only remaining enumerator for narrow flipped-ω attractors is **m130: DE-over-q0 at `ω = −ω_true`**. Pre-run prior is NEGATIVE: DE's mutation-selection needs a gradient signal, which the saturation plateau largely erases. Expect m130 REFUTED with high probability; if so, the flipped-ω enumeration thread is definitively closed.
  - **Wiki pages touched by analyst:** `experiments/m129.md` (NEW — full mechanism analysis + next-experiment queue), `index.md` (m129 row added), `log.md` (ingest line prepended), `concepts/omega-sign-degeneracy.md` (Open-Questions section updated with REFUTED verdict), `concepts/basin-of-attraction.md` (quantitative grid-density bound added), `branches/gradient-based-inversion.md` (third `NOTE:` block — option (b) "densify 10–100×" now REFUTED, only option (c) DE remains). No changes to `branches/surrogate-attitude-isoshell.md` (does not reference flipped-ω).
  - **NEXT EXPERIMENT (fully specified, NOT spawned):**

    **m130 — DE-over-q0 on `ω = −ω_true` slice, targeted (not full 11-seed).** The original m130 spec (11 seeds, pop 200, 10 restarts) is ~30–45 min and has LOW strategic weight given the 3-REFUTED history. Recommended slimmed-down version:
    - **Hypothesis (falsifiable):** at least 1/10 independent DE restarts on seed 33 alone (pop 200, maxiter 100, 3-DOF over q0, ω fixed at `−ω_true`, bounds `rotvec ∈ [−π, +π]³`) converges to hi-fi MSE < 0.15 (seed 33's known basin target 0.082).
      - CONFIRMED → DE IS a viable narrow-basin enumerator; queue m130-full (11 seeds) in the NEXT session.
      - REFUTED → all three blind-search strategies (grid, warm-start, DE) fail to enumerate narrow flipped-ω basins on this surrogate; close the thread with a "narrow basins are unfindable by any tested blind-search" summary.
    - **Method:** single seed (33). For each of 10 DE restarts, random q0 init (uniform SO(3) via super-Fibonacci offset). DE budget: pop=200, maxiter=100, F=0.5, CR=0.7, strategy=`best1bin`. Polish winner with 3-DOF L-BFGS-B. Hi-fi validate unique basins (cluster by q0 geodesic <5°). Classify per m127 thresholds.
    - **Output:** `data/results/inversion_diagnostics/m130_de_flipped/seed_033/{de_runs.npz, polish.npz, hifi.npz, result.json, run.log}` + `batch_summary.json`.
    - **Compute budget:** DE ~200k evals × 10 restarts × ~2ms/eval / Pool(8) parallel restarts = ~250s. Polish 10 × ~1s = 10s. Hi-fi ~5 basins × 60s / Pool(2) = 150s. Setup ~50s. **Total ~7–8 min wall.**
    - **Kill criteria:** per-restart 5 min, total 15 min. If Stage A `de_runs` exceeds 5 min, abort and inspect convergence traces.
    - **Writer prompt:** build from `m127_flipped_omega_search.py` by replacing Stage A `super_fibonacci_quats(N_SO3)` grid scan with `scipy.optimize.differential_evolution(...)` over 3-DOF `rotvec`, keep Stage B polish (optional since DE output is already locally polished) and Stage C hi-fi verbatim. Vendor `hifi_validate`, `make_polish_cost` from m127 unchanged.
    - **Alternative if session time is available next round:** run m130-full (11 seeds) instead — tests population frequency of narrow flipped-ω basins simultaneously with the DE feasibility question. Cost: ~30–45 min. Preferred IF the next session has budget.
    - **Deferred higher-value experiment** (NOT queued): **adversarial-seed generalization test.** Run the `#validated` wrapped pipeline on 10 untested seeds chosen from the 87%-constraint-poor cohort ([[m096_exp1_oracle_grid]] Stage 1 census). Hypothesis: wrapped pipeline's 6/11-improved / 5/11-break-even / 0/11-regressed baseline generalizes to untested seeds. This is the higher-impact next question for closing the generalization gap. Needs design work (seed sampling strategy, baseline computation) before running — suggested for the session AFTER m130 is resolved.

  - **Uncommitted state:** m129 script, results, wiki updates, EXPERIMENTS.md update all on disk but NOT committed at handoff start. Will be committed in wind-down. Inherited from prior session: `.claude/skills/research-loop/SKILL.md`, pre-existing wiki edits on `branches/gradient-based-inversion.md`, `branches/surrogate-attitude-isoshell.md`, `concepts/basin-of-attraction.md`, etc. Also includes `m115_surrogate_pipeline/batch_summary.json` overwrite flag (pre-existing since 2026-04-16 afternoon, see below).
  - **Context at wind-down:** `ctx.sh` at 12.4% pre-commit, well under the 24% end-of-loop threshold. Session stayed disciplined; one experiment start-to-finish with analyst ingest and wiki lint.

- **2026-04-16 (night, recovery session) — m128 COMPLETE after OOM recovery; ω-sign census done; m129 queued (unspawned).**
  - **Recovery context.** Prior session (same day, evening) spawned m128 with `HIFI_POOL=4` and OOM'd on seed 12's Stage B hi-fi. Seeds 0 and 6 had completed; seed 12 had `stage_a_polish.npz` but no hi-fi. No wind-down was conducted. This session resumed with `MICRO128_POOL=2` on all 11 seeds — completed seeds no-op'd via checkpoints (~50 s ctx rebuild each, stages cached), seed 12 reused cached Stage A and ran fresh Stage B. Total wall 2322.7 s (38.7 min), no OOM.
  - **Script:** `notebooks/inversion/12_brightness_surface/m128_warmstart_polish.py` (writer-authored in the prior session, reviewer-approved; reused as-is).
  - **Results:** `data/results/inversion_diagnostics/m128_warmstart_polish/{batch_summary.json, run.log, seed_NNN/{stage_a_polish.npz, stage_b_hifi.npz, result.json, run.log}}` for all 11 seeds.
  - **Headline:** **ALL 11 seeds FLIPPED_FAIL (best hi-fi 1.42–4.73).** Script-emitted verdict `INVALID` because seed 33's control best_hifi = 2.2434 >> 0.15.
  - **Verdict reinterpretation (load-bearing).** The `INVALID` label reflects a **spec bug, not a polish-mechanics bug.** Pre-run [inline] diagnostic from log.md line 1 (same session, before re-launch) had already flagged seed 33 as ATYPICAL: its m115 basin ω is **already retrograde** (signed 161.54° from truth — axis-angle convention `acos(|·|)` folded this to 18.46°, hiding the sign). Consequently the warm-start `(q0_basin_33, −basin_ω_33)` puts the polish on the **forward** side of ω-sign, not at the known flipped-ω attractor from [[m126_wrapped_pipeline]]. ω-sign census across all 11 seeds (analyst-verified from `step2_hifi.npz`): **10/11 seeds have forward-ω basins** (signed 0.34°–10.73°) and ARE valid flipped-ω tests; **seed 33 is the sole retrograde case** and is a structurally invalid control. Correct reading: **REFUTED on 10 valid-test seeds** — warm-start + component-negate-ω + 3-DOF L-BFGS polish does NOT find flipped-ω attractors on any seed.
  - **Polish mechanics VERIFIED [inline spot-check, this session].** Re-evaluated `(q0_basin_0, +basin_ω)` for seed 33 directly through m128's `hifi_validate` path → hi-fi 0.082471; m115 recorded hifi_mse = 0.081680; **relative diff 0.97%** (within 1% tolerance). The L-BFGS machinery is fine. Seed 33's FLIPPED_FAIL is correct behaviour of the spec-as-written.
  - **Key surprise — seed 12 sanity control FAILED.** Sanity target was hi-fi < 0.3 (reproducing [[m127_flipped_omega_search]]'s wide flipped-ω attractor at hi-fi 0.171, q0_err 140.36°). Actual best 1.4161 — 4.7× short. Mechanism: m127's winner is ~62° in quaternion space from the nearest seed-12 m115 basin; L-BFGS-B's local convergence radius on this surrogate is much smaller than 62°. Warm-start **cannot cross to an independent flipped-ω attractor**. The two attractor families (forward-ω basins and flipped-ω attractors) live in different parts of SO(3), connected by no gradient-descent path on the surrogate.
  - **Non-lazy mechanism.** L-BFGS-B with FD Jacobian on a 3-DOF q0 polish (ω frozen at `−basin_ω`) follows the cost gradient at the warm-start. That gradient points toward the nearest local minimum on the `−basin_ω` slice of the surrogate — NOT toward any flipped-ω truth attractor. For a flipped-ω recovery to work, the warm-start q0 would need to lie **within** a flipped-ω attractor's basin of attraction on that slice. There is no evidence this condition holds for any of the 11 seeds' m115 basins.
  - **[[omega-sign-degeneracy]] confidence: UNCHANGED (medium).** m128 neither adds basins nor refutes them; it only rules out one search strategy. Seed 33's narrow basin still exists (re-confirmed by the 0.082 spot-check). Seed 12's wide basin from m127 is untouched. Population frequency stands at 2/11 confirmed + unknown count unsearched.
  - **Wiki pages touched by analyst:** `experiments/m128.md` (NEW — full ω-sign census table, polish-mechanics sanity, verdict reinterpretation, next-experiment queue), `index.md` (m128 row added), `concepts/omega-sign-degeneracy.md` (ω-sign census table + Open-Questions updated), `branches/gradient-based-inversion.md` (warm-start-across-ω-slices caveat added), `log.md` (ingest line appended).
  - **NEXT EXPERIMENT (fully specified, NOT spawned):**

    **m129 — Densified SO(3) grid on seed 33 alone.** Binary test of whether seed 33's known narrow flipped-ω basin is recoverable by grid density alone (orthogonal to m128's warm-start failure).
    - **Hypothesis (falsifiable):** a 600k super-Fibonacci SO(3) grid (~1° median spacing, 10× denser than [[m127_flipped_omega_search]]'s 60k) + top-50 L-BFGS-B polish with ω fixed at `−ω_true` recovers seed 33's known basin at hi-fi ≈ 0.082 (winner hi-fi < 0.15).
      - CONFIRMED: grid-density is the fix; [[m127_flipped_omega_search]]'s REFUTED is retractable on wider grids.
      - REFUTED: no coarse-then-polish method works on this landscape; pivot to population-based (DE) search.
    - **Method:** Single-seed (seed 33). Stage A: 600k super-Fibonacci quaternions, surrogate-score each at `ω = −ω_true`, keep top-50. Stage B: L-BFGS-B polish each top-50 (3-DOF q0, ω frozen). Stage C: hi-fi validate top-5 polished basins. Identical ftol/gtol/maxiter to m127.
    - **Output:** `data/results/inversion_diagnostics/m129_densegrid/seed_033/{stage_a_grid.npz, stage_b_polish.npz, stage_c_hifi.npz, result.json, run.log}`.
    - **Compute budget:** single-seed, ~10× surrogate evals vs m127 seed 33 = ~90 s grid; polish 50 × ~0.5 s = ~25 s; hi-fi 5 × ~65 s = ~325 s (Pool(2) → ~165 s); ctx build ~50 s. **Total ~5–10 min wall.** Genuinely cheap.
    - **Kill criteria:** per-seed 15 min; if Stage A exceeds 5 min something is wrong with the grid.
    - **Writer prompt:** build from `m127_flipped_omega.py` by bumping `N_GRID = 60000 → 600000` and `TOP_K = 20 → 50`, restrict `SEEDS = [33]`, otherwise identical. Vendor `hifi_validate` + `make_polish_cost` from m127. Single-seed removes Pool overhead concerns.
    - **Alternative if m129 fails:** **m130 — DE-over-q0 with ω fixed at `−ω_true`, 11 seeds, pop 200, 10 restarts/seed.** Cost ~30–45 min. Population-based search is width-agnostic and would enumerate narrow AND wide flipped-ω basins across the cohort. Higher info, 2–3× cost.

  - **Context note (fixed mid-session).** After a period of unreliable `ctx.sh` readings in the prior session (12% and 25.8% reported within minutes), this session's ctx hook is returning consistent values: ~4% at start, ~12% after analyst subagent. No action needed.
  - **Uncommitted state:** all m128 results (+ prior session's unpushed m127 state + wiki updates) on disk but NOT committed. Run `/commit` to stage.
  - **Pre-existing repo cleanup flag (found during lint, NOT fixed):** `data/results/inversion_diagnostics/m115_surrogate_pipeline/batch_summary.json` was overwritten earlier in the day by a seed-46-only rerun. Its content on disk covers `[46]` only; the git-index version covers 9 seeds but is missing seed 46. Since per-seed `seed_NNN/result.json` files are intact, the rollup is derivable. If the user cares about the batch summary, regenerate it from the per-seed files (5-min script); otherwise restore from git and accept the seed-46 gap.

- **2026-04-16 (late evening) — m127 COMPLETE + wiki bug fix + m128 spec queued (unspawned).**
  - **Script:** `notebooks/inversion/12_brightness_surface/m127_flipped_omega_search.py` (writer+reviewer approved; docstring patched for reviewer items; 29:05 wall on 11 baseline seeds with Pool(8) SO(3) grid + Pool(4) hi-fi).
  - **Results:** `data/results/inversion_diagnostics/m127_flipped_omega/{batch_summary.json, seed_NNN/{stage_a_grid.npz, stage_b_polish.npz, stage_c_hifi.npz, result.json, run.log}}`.
  - **Verdict: MIXED / UNDERPOWERED.** Script-emitted `REFUTED` is face-value correct (1 non-control PARTIAL < 3-seed threshold) but hides search-density bias. Counts: VALID=0, PARTIAL=1 (seed 12), FAIL=10 (includes seed 33 positive control).
  - **NEW DISCOVERY — seed 12 is a second flipped-ω attractor.** hi-fi MSE 0.171 at q0_err 140.36°, ω=−ω_true. VERIFIED (re-ran analyst's numbers from result.json + m115): **genuinely independent attractor**, NOT a twin-relative of any m115 basin (nearest m115 basin is 61.68° away in quaternion space). Compensation body-frame axis `[−0.848, −0.485, +0.215]` — 32° off body +X, 77.6° off body +Z. **Refutes the seed-33-derived "body +Z is the shared compensation axis" claim** — flipped-ω compensation is seed-specific in BOTH angle and body axis.
  - **Positive control seed 33 FAILED** (known by-now pre-run prediction). Its known basin from [[m126_wrapped_pipeline]] has q0-width <0.001° (per m126 polish observations Δq0 ≤ 0.0004°); the 60k super-Fibonacci SO(3) grid has ~3° resolution → ~10000× mismatch. Nearest top-20 grid quaternion to seed 33's known target is 32.7° away — far outside L-BFGS-B's reach. The method IS underpowered for narrow basins; it can only discover wide ones.
  - **Key wiki bug fixed this session.** The [[omega-sign-degeneracy]] page and [[m126_wrapped_pipeline]] page claimed `|ω_cand| = 0.30 × |ω_true|, −70.19%` for seed 33's flipped basin. Actual is `0.993 × |ω_true|, −0.70%` — someone misread `w_mag_err_pct_before = −0.70187456` as `−70%`. ω magnitude is essentially PRESERVED; only direction is retrograde. The "70% ω-magnitude shrinkage compensates" mechanism theory is retracted. True mechanism: q0 alone compensates by rotating ~98° about body +Z (verified inline) combined with time-reversal of tumble. `multi-solution-philosophy.md`, `m126.md`, `omega-sign-degeneracy.md` all patched.
  - **[[omega-sign-degeneracy]] confidence upgraded low → medium** — two confirmed cases (seed 33 narrow + seed 12 wide), mechanism axis-dependent per seed.
  - **[[gradient-based-inversion]] branch** updated with a note: L-BFGS-B polish from coarse SO(3) grid is underpowered for narrow basins; future approaches need DE-before-polish, warm-start from DE basins, or 10–100× denser grid.
  - **Wiki pages touched:** `experiments/m127.md` (NEW), `concepts/omega-sign-degeneracy.md` (mechanism rewrite + body-Z verification + seed 12 confirmation + confidence upgrade), `concepts/multi-solution-philosophy.md` (seed 12 section added), `concepts/basin-of-attraction.md` (flipped-ω basin width spread), `branches/gradient-based-inversion.md` (polish-from-coarse-grid caveat), `index.md` (m127 row added, omega-sign-degeneracy row rewritten), `log.md` (m127 ingest + bug-fix lint + m128 queue).
  - **NEXT EXPERIMENT (fully specified, NOT spawned — API overload killed 2 writer agent attempts):**

    **m128 — Warm-start flipped-ω polish from m115 DE basins.** Resolves m127's search-density bias by warm-starting L-BFGS-B at each seed's known m115 DE basins with ω component-wise negated.
    - **Hypothesis (falsifiable):** at least 2 of seeds `{0, 6, 14, 24, 27, 36, 46, 74, 93}` yield polished `(q0', −ω_basin)` with hi-fi < 0.5. Seed 33 positive control MUST reproduce hi-fi ≈ 0.082 — failure to do so marks the verdict INVALID (polish mechanics broken), not REFUTED.
    - **Method:** for each of 11 seeds × 3 m115 DE basins: load `step2_hifi.npz`, warm-start at `(q0_basin, −ω_basin)`, 3-DOF L-BFGS-B polish (`ftol=1e-7, gtol=1e-4, maxiter=200, maxfun=1000`), hi-fi validate. Classification FLIPPED_VALID<0.1, PARTIAL<0.5, FAIL≥0.5. Batch verdict CONFIRMED/REFUTED/INVALID.
    - **Output:** `data/results/inversion_diagnostics/m128_warmstart_polish/seed_{NNN}/{stage_a_polish.npz, stage_b_hifi.npz, result.json, run.log}` + `batch_summary.json`.
    - **Compute budget:** 11 seeds × 3 basins × (1–3 s polish + 50 s hi-fi) / Pool(4) ≈ 45 s/seed ≈ 10 min batch wall.
    - **Kill criteria:** per-seed 10 min, batch 30 min. If seed 33 control doesn't hit hi-fi < 0.15, verdict INVALID — flag polish mechanics bug.
    - **Writer prompt already drafted in log.md and in this Section 1.** Fresh session can build the script directly from the m127 template (vendor `hifi_validate` + `make_polish_cost` verbatim; drop Stage A grid; replace with warm-start loader; keep Stage B polish + Stage C hi-fi).
    - **Why cheaper than m127 and higher-info:** it uses known basins as warm-starts, so it tests the narrow-basin hypothesis DIRECTLY (seed 33 control proves/fails polish-mechanics; other 9 seeds tell us whether DE enumerated wrong-ω attractors that hide flipped-ω basins at the same q0 attractor).

  - **CONTEXT NOTE:** `ctx.sh` was showing 12% at one point mid-session; re-running it at the end of the session gave 25.8%/26%. The 12% reading appears to have been stale (possibly wrong session transcript) — do NOT trust the 12% number. User flagged this.
  - **Uncommitted state:** as of wind-down, all m127 results, wiki pages, and EXPERIMENTS.md updates are on-disk but NOT committed. Run `/commit` to stage.

- **2026-04-16 (evening) — m126 COMPLETE: gradient-based-inversion promoted to #validated.**
  - **Script:** `notebooks/inversion/12_brightness_surface/m126_wrapped_pipeline.py` (writer-authored, reviewer PASS, background run 1469s wall via Pool(8) on seeds {0, 6, 12, 24, 33, 36}).
  - **Results:** `data/results/inversion_diagnostics/m126_wrapped/{batch_summary.json, seed_NNN/{polish_ckpt.npz,hifi_ckpt.npz,result.json}, run.log}`.
  - **Headline:** 3/6 seeds improved ≥10% via the `DE → polish → hi-fi(before,after) → keep_min` wrapper (seed 0: 20%, seed 6: 87%, seed 24: 45%). Remaining 3/6 break-even (12, 33, 36 — wrapper correctly rejected catastrophic polish). 0/6 regressed. Hypothesis CONFIRMED.
  - **Combined population across all 11 baseline seeds (5 prior from m125 + 6 here + seed 46):** 6/11 improved ≥10%, 5/11 break-even, 0/11 regressed. Wrapped pipeline Pareto-dominates plain m115. Branch `gradient-based-inversion` promoted from `#open` to **`#validated`**.
  - **Per-seed classification after wrapping (hi-fi MSE bounds):** 0 PARTIAL+OK outright — the wrapper tightens existing PARTIAL seeds (6: 0.0173, 24: 0.0245) and was unable to rescue wrong-attractor failures. Seeds remaining FAIL post-wrap: 0 (0.11), 12 (0.33), 36 (0.60), 74 (0.25). Seeds remaining PARTIAL: 6 (0.017), 14 (0.016), 24 (0.025), 27 (0.31), 33 (0.08), 46 (0.65), 93 (0.019).
  - **[inline] upstream-ω audit:** 6 of 11 baseline seeds (6, 12, 33, 36, 74, 93) had only 1 ω candidate in m115 (`omega_source: result_npz`); 5 had 3 from `geo_ckpt`. No `geo_ckpt.npz` exists for the 6 single-ω seeds anywhere in `data/results/inversion_diagnostics/`. This is the root cause of break-even seeds 12/33/36 — upstream ω is 8-18° off truth, and polish cannot move ω-direction across attractor boundaries.
  - **NEW DISCOVERY — [[omega-sign-degeneracy]]:** Seed 33's DE basin has ω **rotating backwards** relative to truth (signed angle 161.5°, axis-angle 18.5°) plus q0=98°, and produces hi-fi MSE 0.082 (near PARTIAL threshold). This is either a genuine observational symmetry or an approximate wrong-attractor — NEXT EXPERIMENT proposed but not run.
  - **Wiki pages created/updated:** `experiments/m125.md` (was missing, inline-labeled), `experiments/m126.md`, `concepts/omega-sign-degeneracy.md` (NEW), `branches/gradient-based-inversion.md` (promoted to #validated), `index.md`, `log.md`, `concepts/dark-mag-saturation.md`, `concepts/basin-of-attraction.md`, `concepts/surrogate-model.md`, `concepts/multi-solution-philosophy.md`.
  - **Display convention caveat:** `m126_wrapped_pipeline.py` reports `w_dir_err` as the SIGNED angle between ω vectors (0-180°); all prior m115/122/123/124 scripts use AXIS-ANGLE (abs of dot product, 0-90°). They agree for ω-aligned basins but differ by `180°-x` for retrograde basins. Seed 33 basin_0 shows w_dir=161.5° in m126 vs 18.5° in m115 — same physical ω, different display. Surrogate cost and hi-fi MSE use actual ω vectors correctly; only the `w_dir_err` display number is affected. Flagged on the wiki pages and log.
  - **NEXT EXPERIMENT (proposed, not run) — m127:** Signed-ω degeneracy check. For seeds 33, 14, 93: compute hi-fi LC at `(q0_truth, -ω_truth)` (pure ω-sign flip, no q0 compensation). Hypothesis: if MSE ≈ σ²=0.0025 for all, ω-sign IS a genuine observational symmetry like ±X twin. If MSE >> σ² for most seeds, seed 33's backwards-ω basin at MSE 0.082 is an approximate wrong-attractor, not a physical symmetry. Compute: 3 hi-fi evals × Pool(3) ≈ 1 min wall + 50s setup per seed. Cheap, high-information. Would classify `omega-sign-degeneracy` concept as either `confidence: high` (symmetry) or `confidence: low` (artefact).
  - **FOLLOW-UP AFTER m127:** Harvest a `geo_ckpt.npz` for seeds {6, 12, 33, 36, 46, 74} (via a trimmed m103-style grid+NM+geo run on these 6 seeds only), then re-run wrapped pipeline on them. Expected to rescue seeds 12 and 36 from FAIL by providing 3 diverse ω candidates. Compute: ~30 min/seed for harvest + ~10 min/seed for wrapped run = ~4 hours serial, ~45 min with Pool(8) across seeds. Gate for closing the 11-seed cohort to 0 FAIL. (Deferred because m127 is cheaper and tells us whether the remaining `wrong-attractor` seeds are fundamentally unrecoverable or fixable.)
  - **PROCESS LEARNINGS (2026-04-16 evening):**
    - Strategist prediction rate: 5/6 per-seed predictions matched outcome (seeds 0, 6, 12, 24, 33, 36). Only miss: seed 0 predicted "IMPROVES to PARTIAL/OK" actually improved 20% but stayed FAIL (still 0.11 hi-fi). Upstream ω at 2.9° was better than 33/36 but worse than 24 (0.6°).
    - Reviewer PASS caught only a latent `TypeError` on unreachable code path — good signal that the writer-reviewer handoff pattern is working.
    - Analyst subagent token cost: ~127K (but only ~3-5K added to main context via summary). Subagent delegation keeps strategist context low.
    - Background `Bash` + scheduled wakeups work cleanly for compute jobs in the 15-25 min range.

- **CORRECTNESS GAP + PARALLELIZATION DEBT (2026-04-16, future-me action list).**

  **[CORRECTNESS ISSUE — priority elevated after checking]** `m115_surrogate_pipeline.py` line 77: `N_HIFI_BASINS = 3`. The clustering step can produce 8–10 distinct basins per seed, but only the top 3 by **surrogate** MSE get hi-fi validated. Basins ranked #4+ by surrogate are silently discarded without ever being scored on hi-fi. Violates the multi-solution philosophy; [[m124_hifi_validate]] proved surrogate MSE ranking disagrees with hi-fi MSE ranking off-truth.

  **Checked 2026-04-16 on seeds 27/46/74** (the "unsolved" seeds in this session): the cap is NOT currently hiding truth-adjacent basins on these three seeds. Seed 27 has zero truth-adjacent solutions at any rank (30/30 DE solutions have q0_err ≥ 14°). Seed 46 same (30/30 with q0_err ≥ 21°). Seed 74 does have truth-adjacent solutions in ranks 0–6 but they all cluster into the top-3 basins so the cap doesn't hide them — seed 74's real failure mode is different (all 10 solutions share ω-dir error 4.88°, suggesting upstream ω-candidate problem, not DE-basin-missing problem). **Implication:** removing the cap is still correct (violates philosophy, could matter on untested seeds) but it's NOT the fix for the three seeds we currently classify as unsolved. Their real failure modes are (a) DE genuinely missing truth attractors for 27, 46, and (b) upstream ω candidates being ~5° off for 74. **Action (still):** set `N_HIFI_BASINS = None`, hi-fi validate all distinct basins. Performance concern solved by perf fix #2 below.

  The full pipeline currently achieves ~10 min/seed wall-clock at ~5% CPU utilisation because the hot loops are serial. Four specific perf wins ranked by effort vs. speedup:
  1. **[HIGHEST VALUE, SMALL-MEDIUM EFFORT]** `m115_surrogate_pipeline.py` line ~513: `differential_evolution(...)` is called without `workers=` → runs 100% serial. Adding `workers=-1` would give ~6–8× speedup on the DE step (the single biggest cost in m115), BUT the current objective is a closure from `make_surrogate_3dof_objective(...)` at line ~506 which pickle can't serialise. Required refactor: convert `make_surrogate_3dof_objective` to return a **callable class instance** whose `__init__` captures the state (`delta_qs`, `sun_dirs`, `obs_dirs`, `obs_dist_km`, `observed_lc`, `surr_model`) as instance attributes and whose `__call__(x)` runs the eval. Alternatively promote the state dict to module scope and use a module-level `_de_objective(x)` that reads from it. Either approach is ~20–40 lines of edit; the callable-class approach is cleaner. Verify: DE result should be bitwise-identical vs. serial given the same `seed=`; if not, the closure had hidden mutable state. **Expected per-seed saving: ~3 min → ~30s on the DE step.**
  2. **[LOW EFFORT, MEDIUM VALUE]** `m115_surrogate_pipeline.py` — the hi-fi validation loop at line ~619 calls `hifi_validate(...)` serially per basin. Each basin is ~50s serial. Refactor to a `mp.Pool(N_basins)` with fork-inherited ctx, same pattern as `m124_hifi_validate.py:272-328` (see that file for the working template). **Expected per-seed saving: ~100s → ~50s on m115 hi-fi when there are 3 basins.**
  3. **[LOW EFFORT, LOW-MEDIUM VALUE]** `m123_lbfgs_polish.py` loops over starts (up to 4 per seed) serially. Each start is independent. Wrap the start-loop in `mp.Pool(N_starts)` where each worker calls `run_one_start(start, ctx, ...)` with a fork-inherited ctx. Same fork pattern as m124. **Expected per-seed saving: ~200s → ~60s on m123 when there are 4 starts.**
  4. **[MEDIUM EFFORT, CONDITIONAL VALUE]** Add an outer per-seed `mp.Pool` in whichever script you run across multiple seeds (m122, m123, m124, or a future "full pipeline" driver). Each seed's `setup_experiment` takes ~50s of SPICE + STL work; parallelising across seeds gives ~N× speedup on that overhead AND on any stage that's still serial within a seed after fixes 1–3. Only worthwhile for ≥3-seed runs; for single-seed debugging the outer serial loop is fine. **Expected saving: ~5× on 5-seed batch runs.**

  **Net impact if all four are done:** full pipeline drops from ~10 min/seed to ~2–3 min/seed on an 8-core machine. The first fix alone (#1) gets you from 10 to ~6 min with the smallest code change.

  **Why none of this was done this session:** (a) m115 was written when DE was considered cheap, and the closure was never refactored for pickling; (b) m122/m123 were prototypes where I asked the writer for "correctness first, speed later"; (c) m124 DID get `mp.Pool(8)` because hi-fi evals at 50s each would have made serial unacceptable. The `mp.set_start_method('fork', force=True)` boilerplate is in every new script specifically so adding `Pool()` later is drop-in.

- **2026-04-16 — m122/123/124/125 COMPLETE: Gradient-based inversion thread resolved.**
  Four experiments in one session characterised the gradient landscape and tested L-BFGS polish:
  - **m122** — 6-DOF FD Hessian at truth (5 seeds: 14, 27, 46, 74, 93). Eigenvalue spread 10⁵–10⁶; q0 soft (0.3-0.9°), ω-dir stiff (0.003-0.05°), ω-mag stiffest (0.007%). Basin geometry cohort-universal (ATT_FAIL/OK within 2.1×). Hyp2 (m121 anisotropy axis match) REFUTED 2/3 — local curvature ≠ finite-scale saturation ordering.
  - **m123** — L-BFGS from truth + m115 DE basins (5 seeds). Truth IS surrogate minimum (corrects m122's parameter-space artifact). DE basins: q0 LOCKED at attractor (<0.001° change in 12/12 basins). In INTERNAL parameter units the polish reports 0.03-0.14° ω-dir tangent motion and 0.1-0.3% |ω| motion, BUT when converted back to physical error-vs-truth coordinates, ω-dir error is UNCHANGED in 11/12 basins (only seed 14 basin_1 moves 0.34°→0.14°). Only |ω|-magnitude error meaningfully polishes (5-10× tightening, e.g. 0.12%→0.01% on seed 14). Cost drops 4-7× mostly from the |ω|-mag tightening compounding as attitude-drift over the 3600s window.
  - **m124** — Hi-fi validation of polished candidates (17 cands + 5 truth-refs, Pool(8)). REFUTED 2/12 agree within ±30% log-ratio. Seed 27 CATASTROPHIC (hi-fi WORSENS 15× after polish). Truth-polish SAFE.
  - **m125 [inline]** — Re-scoring m124 data with a `keep_min(hifi_before, hifi_after)` wrapper. PRODUCT-LEVEL VERDICT: wrapped polish improves seed-level best hi-fi by 90%/33%/70% on seeds 14/74/93; seed 27 break-even (wrapper rejects catastrophe). 9/12 basins helped. [[gradient-based-inversion]] REINSTATED to `#open-active`.
  - **Architecture (honest):** DE enumerates q0 attractors (polish cannot change q0). Within an attractor, L-BFGS polishes **|ω|-magnitude only** — ω-direction is also locked in 11/12 basins. The hi-fi(before,after) wrapper catches cases where even the |ω|-mag polish misleads (e.g. seed 27 where surrogate drove polish in the wrong direction).
  - **Per-seed honest verdict (solutions found with hi-fi MSE < 0.1, truth excluded):** seed 14 = 2/3 basins valid (truth-adjacent + ±X twin). Seed 27 = 0/3 (all wrong-attitude). Seed 74 = 0/3 (best is 0.25 ~100× noise). Seed 93 = 3/3 (truth-adjacent + two twin-like). Seed 46 = data gap (m115 never ran DE). Two seeds genuinely solved, two still broken by wrong-attractor selection in DE, one gap. Full-pipeline wall-clock 9-13 min/seed (m115 + m123 + m124 on Pool(8)).
  - **Scripts:** `m122_hessian_curvature.py`, `m123_lbfgs_polish.py`, `m124_hifi_validate.py`, `m125_keep_better_inline.py`.
  - **Results:** `data/results/inversion_diagnostics/micro{122,123,124,125_keep_better}/`.
  - **Wiki pages created/updated:** m122.md, m123.md, m124.md, surrogate-truth-offset (CORRECTED), gradient-based-inversion (#open-active), basin-of-attraction, dark-mag-saturation, surrogate-model.
  - **NEXT EXPERIMENT (proposed, not run):** m126 — apply the full wrapped pipeline (DE → polish → hi-fi(before,after) → keep_min) to ALL 10 m115 baseline seeds (not just the 5 tested here). Particularly seed 0 (borderline), 6, 12, 33, 36 which have diverse ω/q0 error profiles. This is the gate to `#validated`. Compute: ~80 min wall (10 seeds × 30s setup + ~15s L-BFGS + ~60s × 2 hi-fi per basin × 3-17 basins). Pool(8) → ~15 min.
  - **ADDITIONAL TODO (next session):** seed 74 has a suspicious failure mode worth dedicated diagnostic. All 10 DE solutions across all basins share ω-direction error 4.88° to identical precision, suggesting DE never varies ω (it's 3-DOF on q0 only) and the 3 upstream ω candidates handed to DE were all ~5° off truth. Look at the `omega_info` entries that m115 loads for seed 74 (from `m103_hybrid/seed_074/geo_ckpt.npz` or `m102_fullmse/seed_074/result.npz`), compute truth-vs-candidate angular distance for each. If all 3 upstream ω candidates are 4-5° off, the fix belongs upstream (geo refinement in m102 or a DE-on-ω stage before the DE-on-q0 stage). This diagnostic is analytical (no new compute), ~10 lines of Python.
  - **No commits made this session.** All scripts, results, and wiki pages are on-disk but uncommitted. Run `/commit` to stage.
- **2026-04-15 (late afternoon) — m121 LANDED, BASIN-WIDTH CHARACTERISED, HYPOTHESIS SPLIT-VERDICT:** Script: `notebooks/inversion/12_brightness_surface/m121_basin_width_metric.py`. 3 seeds (14, 27, 46) × 6751 candidates in ~3 min/seed (Pool(8)). Pool replaces m120 buckets with 1-D slices: q0-only / ω-direction-only / ω-magnitude-only / joint at 8/7/4 log-spaced scales × 250 samples. Scored only `mean_L1` and `mean_L2` (scope-narrowed from m120's 13 variants — deviation called out in script docstring and REPORT; NOT a methodological defect). Results: `data/results/inversion_diagnostics/m121/seed_{014,027,046}/` (candidates.npz / residuals.npz / cost_variants.npz / summary.json / plots/).
  - **Headlines (truth cost / truth rank under mean_L1, N=6751):** seed 14: 0.05545 / 9; seed 27: 0.05394 / 62; seed 46: 0.05283 / 43. High ranks = ~250 sub-0.1° perturbations sometimes beat truth by chance at noise-floor σ≈6×10⁻⁴.
  - **Basin shape (per-axis, seed 14 representative, confirmed pattern on 27 + 46):**
    - q0 axis: 0.5° → +26% cost, 2° → +188%, 5° → +534%, 20° → +2048%. Smooth monotonic. **Gradient-bearing to ~20°.**
    - ω-direction: 0.1° → +1596% (already saturated). Min at 0.1° is 0.058 (near truth 0.055), spread [0.058, 1.290]. **Basin < 0.1°, anisotropic.**
    - ω-magnitude: 0.25% → +2279% (already saturated at smallest scale). **Basin < 0.25%.**
    - joint: 0.25° → +2508% (dominated by ω-dir component).
  - **ω-direction anisotropy [inline, seed 14]:** At scale=0.1°, 3 best rotation axes cluster near inertial ±[0.60,-0.06,-0.80] → body [0.39,0.19,0.90] (dominantly body +Z); 3 worst cluster near inertial [0.05,1.00,0.02] → body [0.15,-0.98,0.09] (dominantly body -Y). 25× cost ratio between best and worst at same scale. Seed 27 and 46 also show 15-25× anisotropy but with DIFFERENT preferred body-frame axes (seed 27: +X; seed 46: -X/+Z). Common thread per analyst: worst axis always has dominant ±Y component (IS-901 panel spin axis).
  - **Mechanism (quantitative):** q0 error = constant rotational offset (bounded LC perturbation). ω errors compound linearly over 3600 s obs window: 0.1° ω-direction → 7.8° drift by end; 0.25% ω-magnitude → 11° drift. Both enough to miss peak timing on IS-901's ~114-peak LC. Cost saturates at [[dark-mag-saturation]] floor (~2.0 mean_L1) because wrong-ω predictions hit the surrogate's ~23-mag dark-side ceiling.
  - **IMPACT ON [[gradient-based-inversion]]:** ω-direction basin < 0.1° is a HARD init prerequisite. Classical pipeline ([[m102_fullmse]]) delivers ω_dir error 0.3-3° on OK seeds, 5°+ on ATT_FAIL — short by 3-30×. Pure gradient descent from random or classical-NM start will hit the saturated dark-side plateau. Warm-start from m115 DE basins (ω_dir 0.3-3°) is still outside basin. The cheap feasibility test added to the branch page: polish m115 basins with pure-numpy finite-diff gradient descent on surrogate MSE before investing in JAX port.
  - **Inline dark-mag-saturation analysis (2026-04-15, this session, [inline]):** loaded m120 residuals across 4 seeds (0, 14, 27, 46), discovered the `close_omega` bucket's "rank ~8000" in [[m120_tumbling_competitors]] is primarily artifact of surrogate's dark-mag ceiling (~23-27 mag) — close_omega predictions are >20 mag in 92-100% of epochs, residuals saturate at ~10, cost is *locally flat* in that region (no gradient signal). New concept page: `notebooks/inversion/wiki/wiki/concepts/dark-mag-saturation.md`. Refines but does not contradict m120's joint-(q0,ω)-sensitivity claim. Relevant for gradient-based-inversion init strategy.
  - **Wiki work:** created `experiments/m121.md`, `concepts/dark-mag-saturation.md`; rewrote `concepts/basin-of-attraction.md` (alignment-cost basin preserved, surrogate-residual basin added with m121 numbers); updated `branches/gradient-based-inversion.md` (un-retracted earlier + new basin-width prerequisite section + restated decision tree); updated `branches/surrogate-attitude-isoshell.md` (next-experiment section); updated `index.md` + appended `log.md`.
- **2026-04-15 (afternoon, PRIOR) — m120 LANDED, ATTITUDE-ISOSHELL BRANCH VALIDATED (single-seed):** The v2 tautology ("static ω=0 grid vs tumbling truth") is now resolved. Script: `notebooks/inversion/12_brightness_surface/m120_tumbling_competitors.py`. Seed 14, 10004 candidates across 5 buckets (truth + 3 m115 basins + 3000 uniform 6-DOF + 2000 close_ω + 2000 close_q0 + 3000 near-truth perturbations), each propagated through 500 epochs + surrogate-scored. **Truth rank = 0/10004 under ALL 13 cost variants.** Runtime 279s on Pool(8) (37 cand/s steady state, ODE-dominated). Results: `data/results/inversion_diagnostics/m120/seed_014/` (candidates.npz, residuals.npz, cost_variants.npz, summary.json, run.log, 2 plots). Key structural findings:
  - near_truth best rank 1, uniform best 13-69, close_q0 best 137-208, m115 basins rank 17-571 — all systematic;
  - **close_omega (tight ω±2°, random q0) best rank 8004/10004** — knowing ω exactly without q0 is WORSE than random 6-DOF tumbling. Cost is jointly sensitive to (q0, ω), not separable;
  - m115 wrong basins (b1 twin, b2 alt) ranked 17-571 but truth beats them in every variant.
  - See [[m120_tumbling_competitors]] wiki page for the full analysis.
- **2026-04-15 (late evening, PRIOR) — m119v2 LANDED, TIME-MISMATCH BUG FIXED:** `notebooks/inversion/12_brightness_surface/m119v2_attitude_isoshell.py` + lib patch (`notebooks/inversion/lib/experiment_setup.py` now accepts optional `true_q0_wxyz` / `true_omega0_rad` kwargs, backwards-compatible). Seed-14 rerun at honest 6-hr geometry: truth rank 0/60000 under all 13 variants, residual median 0.043 / p90 0.112 / max 0.549 mag. POC verdict "PASS" but the grid-of-static-rotations vs tumbling-truth test was near-tautological — **resolved by m120 (see above)**.
  - Results: `data/results/inversion_diagnostics/m119v2/seed_014/` (summary.json, residual_kernel.npz, cost_variants.npz, target_scores.npz, 4 plots).
  - Original `m119_attitude_isoshell.py` left in place with RETRACTED banner for provenance.
- **AUDIT RETRACTION (2026-04-15 afternoon, this session):** The earlier audit entry was WRONG. Empirical check:
  - `m115_surrogate_pipeline.py` line 719 passes `end_time_utc='2020-02-05T11:00:00'` → 1-hour window matching m046.
  - `archive/inline_omega_selection_test.py` line 109 passes the same `end_time_utc='2020-02-05T11:00:00'`.
  - `setup_experiment(n_observations=500, end_time_utc='2020-02-05T11:00:00')` produces `observation_times` that match `master['observation_times']` to machine precision (max |diff|=0.0). Reconstructed k1/k2 J2000 vectors from master quaternions match `ctx.sun_dirs/obs_dirs` to 1e-6° on seed 14.
  - **Both scripts ARE internally consistent.** The "10/10 surrogate multi-start DE breakthrough" and "4/4 surrogate omega-selection validation" results STAND.
  - Only `m119_attitude_isoshell.py` v1 had the bug (it used the config default 6-hour end_time). Fixed by `m119v2`.
  - The previous strategist pattern-matched on "script loads master AND calls setup_experiment" without verifying whether `end_time_utc` was forced to the 1-hour window. See `m119_BUG.md` § AUDIT RESOLVED for the retraction. **No further fix needed on m115 or inline_omega_selection_test.**
- **PRE-BUG ACTIVE THREAD (STILL VALID):** [[surrogate-attitude-isoshell]] was motivated by [[m118_cost_comparison]]'s finding that IPL-centroid alignment cost has a ~25° noise floor. That finding IS trustworthy (m118 didn't mix master data with setup_experiment geometry). The original question — "can the surrogate replace the pab-contour as a brightness function" — remains open. A clean reimplementation (Option A in `m119_BUG.md`: drop m046 dependency) would actually test it.
  Previous validated threads still standing: surrogate multi-start DE replaces phi sweep ([[m115_surrogate_pipeline]], 10/10); surrogate MSE replaces alignment cost for omega selection (inline 4/4, 2026-04-15). Harvester deprioritised: alignment-cost noise floor is an upstream problem the harvester can't solve.
- **HANDOFF DOCUMENT:** Key wiki pages (read in this order for a fast orient):
  1. [[m119_attitude_isoshell]] — latest experiment (this session), attitude-isoshell POC + kernel factorisation on surrogate residuals.
  2. [[m118_cost_comparison]] — preceding experiment, introduces the kernel-factorisation infrastructure and the pab-contour diagnosis.
  3. [[pab-contour-phase-angle-limitation]] — why IPL-centroid cost has a ~25° noise floor.
  4. [[surrogate-attitude-isoshell]] — current `#open` research direction, re-scoped after m119.
  5. [[kernel-factorization]] — the reusable infrastructure pattern.
  6. [[surrogate-de-search]] (#validated), [[surrogate-omega-selection]] (#validated, inline 4/4) — still-standing successes.
  7. [[multi-solution-philosophy]], [[twin-degeneracy]], [[symmetry-degeneracies]] — framing concepts.
- **2026-04-15 INLINE OMEGA-SELECTION TEST [inline]:**
  Strategist ran `notebooks/inversion/12_brightness_surface/archive/inline_omega_selection_test.py` on 4 seeds with existing `geo_ckpt.npz` (from m103_hybrid): seeds 0, 14, 24, 27.
  Method: per omega in the geo pool (26 omegas/seed), 1 surrogate-DE start (3-DOF, maxiter=200, popsize=15), rank by surrogate MSE.
  **HYPOTHESIS CONFIRMED 4/4:** surrogate MSE places a valid omega in top-2 every time; alignment cost (geo) ranks the same omegas at 4-5 for seeds 0 and 14.
  - Seed 14: surr#1 (best ω, w_dir=0.34°), q0=2.29° → **OK** (vs m102 FAIL at q0=90.7°)
  - Seed 24: surr#1 (alt valid, q0=2.16°) / surr#2 (best ω, twin, q0=178.5°) → **OK** (vs m102 FAIL at q0=152.3°)
  - Seed 0: surr#1 (best ω), q0=6.68° → PARTIAL (no regression vs m102 PARTIAL at q0=8.7°)
  - Seed 27: surr#1 (best ω), single-start q0=14.26° → still PARTIAL/FAIL on this one start; multi-start would enumerate basins per m115 pattern. **NOT a refutation** — selection mechanism works, attitude basin enumeration is a separate concern.
  Mechanism: alignment cost scores only the phi-sweep attitude per omega — when phi-sweep is wrong (ATT_FAIL seeds), alignment score reflects bad attitude, not omega quality. Surrogate-DE optimises attitude per omega before scoring, so the ranking reflects best-achievable LC fit. They measure different things.
  Results JSON: `data/results/inversion_diagnostics/inline_omega_selection/results.json`. Raw log: `inline_omega_selection/run.log`.
  **PROVENANCE:** This is INLINE strategist testing, NOT a numbered micro experiment. The drafted `m116_unified_formulation.py` (1405 lines) was **shelved** — bloated with redundant grid+NM+geo work that already existed in m103_hybrid checkpoints.
- **MICRO116 STATUS — SHELVED:**
  - Script `notebooks/inversion/12_brightness_surface/m116_unified_formulation.py` exists in repo but **was never successfully run**. It re-ran grid+NM+geo from scratch (~12 min/seed of waste — that work existed). Original estimate ~85 min/seed × 10 seeds = ~14 hours. Two latent bugs (SyntaxError on `global` redeclaration, numpy `int64` JSON serialization) were fixed in commit `e57fe14` for the historical record, but the script itself is shelved.
  - DO NOT run `m116_unified_formulation.py` as-is. If the next session needs the unified pipeline, write a slim version that loads existing geo_ckpts where available and only re-runs grid+NM+geo for the 6 missing baseline seeds (6, 12, 33, 36, 74, 93).
- **COST-PICTURE REFRAME (2026-04-15, end of session):**
  The "30 min/seed" estimate I gave was wrong-shaped — it bundled a one-time geo_ckpt harvest cost with the recurring iteration cost. Honest decomposition:
  - **Recurring iteration cost:** ~6 min/seed of pure surrogate work, IF a `geo_ckpt.npz` already exists for that seed. Proven by today's inline test (4 seeds in 26 min wall-clock, ~6.5 min/seed).
  - **One-time harvest cost:** for the 6 baseline seeds without a `geo_ckpt.npz` (6, 12, 33, 36, 74, 93), generate one. After that, all future omega-selection iterations on these seeds are 6 min/seed forever.
  - The 4 seeds with existing `geo_ckpt.npz` (0, 14, 24, 27) are from `data/results/inversion_diagnostics/m103_hybrid/seed_NNN/geo_ckpt.npz`.
- **WHY SURROGATE CAN'T REPLACE ALIGNMENT COST UPSTREAM (decided 2026-04-15):**
  Fundamental asymmetry: alignment cost is **geometric** (one PAB constraint per epoch, vectorised over 360 phi values, sub-millisecond per direction). Surrogate is **photometric** (full LC integration over 500 epochs, requires a full attitude trajectory; ~5 ms per eval but needs ~14 s of surrogate-DE per omega to extract a meaningful attitude).
  At grid scale (40000 points × 14 s = 7.7 hr/seed), surrogate-as-grid-cost is infeasible.
  **Tested dead ends:** m095 (lo-fi rerank), m097 (lo-fi as grid cost), m114 (6-DOF cold-start surrogate-DE) — all failed.
  **Untested but priors weak:** surrogate-MSE re-rank of grid top-K with phi-sweep attitude (essentially m097 with shadows; phi-sweep attitude is wrong for ATT_FAIL seeds, so surrogate-on-wrong-attitude doesn't help).
  **The alignment-cost grid is here to stay as the omega-finder.** What CAN attack the 12 min/seed cost: shrink grid resolution, skip geo refinement, parallelise across seeds.
- **NEW SURROGATE USAGE PATTERN (2026-04-15):**
  Surrogate-DE-MSE is now the **gold-standard cheap omega-pool validator**. ±0.005 of hi-fi MSE; 50,000× faster than hi-fi. Whenever ANY pipeline produces omega candidates with stored vectors, surrogate ranking is the right downstream check. Use it for:
  - Validating omega pools immediately after grid+NM+geo (catch ATT_FAIL seeds before phi-sweep)
  - Re-scoring historical experiment data that saved candidate omega vectors
  - Post-hoc oracle when debugging
- **m117 SANITY — LEVER 2 REFUTED (2026-04-15):**
  Ran `notebooks/inversion/12_brightness_surface/m117_result_harvester.py` on seed 14 (grid 2000×20 + NM_TOP=300, skip geo, save top-26 by NM cost). Grid 82.5s + NM 106.6s = 189s total on Pool(24). Output: `data/results/inversion_diagnostics/harvester_sanity/seed_014/{geo_ckpt.npz, result.json, grid_ckpt.npz}`.
  Validator `m117_validate_pipeline.py` ran surrogate-DE over the 26 NM-only candidates (405s, 15.6s/cand). Output: `harvester_sanity/seed_014/validate.json`.
  **Result: NM-only pool cannot reach the truth basin.** Best ω in NM-only pool has w_dir=48.81° (vs 0.34° in geo-refined pool from m103_hybrid). Pool minimum cost 0.0461 vs geo-refined 0.00055 (~80× gap). Best-ω surrogate rank 7/26 (vs #1 in the 2026-04-15 inline test on geo-refined data). Top-5 by surrogate MSE are exact duplicates at w_dir=85° / q0_err=103°.
  **[inline pool inspection]:** NM-top-26 is dominated by duplicate basins (5× at 85.5°, 3× each at 87.2°/83.5°/77.7°/85.3°). NM_TOP=300 is exploring far fewer than 300 distinct basins. Truth-adjacent cluster (w_dir<2°) entirely absent.
  **Mechanism:** alignment-cost landscape has wide shallow basins around wrong omegas and narrow deep basins around truth-adjacent ones. Nelder-Mead (derivative-free simplex) finds the shallow basins; L-BFGS-B (geo step) follows gradients into the deep ones. Geo refinement is a basin-class change, not a polish. See [[m117_result_harvester]] and updated [[nm-refinement]] wiki pages.
  **Branch status:** [[harvester-optimization]] stays #open. Lever 2 closed. Levers 1 (coarser grid) and 3 (outer-parallel) still viable.
- **m118 — KERNEL-FACTORED IPL COST DIAGNOSTIC (2026-04-15, seed 14 only):**
  Full spec + results in [[m118_cost_comparison]] wiki page. Scripts: `notebooks/inversion/12_brightness_surface/m118_kernel_computation.py`, `m118_cost_comparison.py`, `m118_diagnostic_mode.py`. Outputs: `data/results/inversion_diagnostics/m118/seed_014/`.
  - Built a propagation kernel (`q_delta[2000, 20, 255_epochs, 4]`, 156 MB, 94 s on Pool(24)) that decouples kinematics from scoring. See [[kernel-factorization]].
  - Upgraded anchor selection to cascade by loop count (1-loop > 2-loop > ...) with q75 length filter. Seed 14 chose epoch 119, loop_count=2, 2 centroids. Full trace saved in `kernel.npz`.
  - Constraint epoch set expanded from ~14 spec peaks to 255 (spec peaks ∪ tight-IPL ∪ exclude anchor).
  - Scored 6 cost variants on this kernel: facet_normal, ipl_centroid_uniform, ipl_centroid_weighted, ipl_active_centroid (ORACLE), ipl_centroid_weighted_ext, ipl_active_ring (ORACLE ring cost). Each variant ~20 s at full 360 phi × 2 anchor centroids, parallelised via Pool(16) over direction chunks.
  - **No variant puts truth at rank 1.** Best result: oracle active_centroid at rank 575K / 14.4M candidates; Q0_ERR_AT_RANK_1 = 110°. See m118 wiki page for per-variant table.
  - **Key diagnostic:** at seed 14's 255 constraint epochs, median ang_dist from truth PAB to its nearest IPL centroid is **25.7°** (not ~0). The pab-contour's zero-phase / lo-fi assumptions systematically offset the IPL from where truth actually is. Full analysis in [[pab-contour-phase-angle-limitation]].
  - **Infrastructure lives on:** `m118_diagnostic_mode.py` is parameterised (MICRO118_SEEDS, MICRO118_VARIANTS, MICRO118_FORCE). Any new cost variant is a ~20 s re-score on the saved kernel. See [[kernel-factorization]].
- **INFRA: `.claude/hooks/ctx.sh`** — agent-visible context-usage helper. Reads the most-recent session transcript JSONL and prints `used/total (pct%)`. Use before wind-down decisions; see memory `feedback_check_context.md`.
- **m119 — SURROGATE ATTITUDE ISOSHELL POC (2026-04-15 evening, seed 14 only):**
  Full spec + results in [[m119_attitude_isoshell]] wiki page. Script: `notebooks/inversion/12_brightness_surface/m119_attitude_isoshell.py`. Output: `data/results/inversion_diagnostics/m119/seed_014/`.
  - Built the surrogate residual kernel `residual[60000 SO(3) grid × 255 constraint_epochs]` float32 (~122 MB) in 47 s on Pool(8). Stage A+B setup <1 s. Total runtime 49.5 s — way under the 10-min kill budget.
  - SO(3) grid = 60,000 super-Fibonacci quaternion samples (~3° median geodesic resolution). Constraint epochs reused from `m118/seed_014/kernel.npz` (255: union of spec_peaks + tight-IPL epochs).
  - Scored 13 cost variants on the saved residual tensor (~0.4 s total): `mean_L2, mean_L1, max_abs, count_pass_σ` for σ∈{0.05,0.10,0.15,0.20,0.30,0.50,1.00}, `soft_pass_σ` for σ∈{0.10,0.20,0.50}. Full `score[60000]` AND `topK[1000]` saved per variant.
  - **Hypothesis split-verdict:** threshold-level-set framing REFUTED (truth residual median 0.49, p90 4.21, max 8.14 mag — far from the <0.1/<0.3 target). Score-based discrimination VALIDATED (truth rank 0/60000 under mean_L1 and soft_pass_050; beats m118 rank-1 competitor by 4000-13000 ranks under every variant).
  - **Target table (out of 60000 under mean_L1):** truth=0, m115_basin0 (near-truth, q0=2.3°)=160, m115_basin1 (twin, q0=178°)=63, m115_basin2 (twin, q0=179°)=0 (ties truth), m118_facet_rank1 (q0=168°, w_dir=52°)=6094, m118_ipl_weighted_ext_rank1 (q0=171°, w_dir=7.6°)=10600. Truth + its twin dominate mean_L1 as expected per [[multi-solution-philosophy]].
  - **Why truth residual is not near zero:** the 241 tight-IPL constraint epochs correlate with dim / near-shadow configurations where the surrogate's MAE is likely worse than the advertised 0.03 (bright-regime). The `surrogate-model` wiki concept was updated with a dim-regime caveat — the audit against noiseless hi-fi at truth is still pending.
  - **Infrastructure lives on:** `m119_attitude_isoshell.py` is parameterised (MICRO119_SEED, MICRO119_POOL, MICRO119_FORCE, MICRO119_N_SO3) and skip-if-exists on Stage C; any new cost variant or target is a ~1 s re-score on the saved residual kernel.
- **NEXT EXPERIMENT (proposed by analyst, queued for next session):**
  **m122 — Hessian at truth (seeds 14, 27, 46 + OK-cohort 74, 93).** Compute the 7×7 finite-difference Hessian of mean_L1 surrogate cost w.r.t. (q0_xyz, ω_xyz, |ω|) at truth with step ~0.05°. ~150 surrogate evals per seed. Eigenvalues give definitive basin widths below m121's sampling resolution. Eigenvectors identify SO(3) slab principal-axis orientation (testable against m121's empirical best-axis). Gives a ready-to-use mass-matrix for any future HMC / preconditioned-gradient-descent attempt. Pairing with OK-cohort seeds tests whether the narrow-ω story is ATT_FAIL-specific or universal — load-bearing for search-integration feasibility. Runtime: seconds per seed. Infrastructure reuse: same `setup.npz` / `propagate_attitude` / surrogate path as m120/121; new seeds (74, 93) need a ~45 s setup.npz pre-step each via the m119v2 pattern.
- **NEXT STEPS (updated 2026-04-15 afternoon, post-m120):**
  1. **[DONE this session, inline] Surrogate fidelity audit at truth.** 4 seeds (14, 27, 0, 93): MAE ~0.03 mag, dim MAE (~0.029) LOWER than bright MAE (~0.070). Dim-regime caveat was an artefact of the corrupted m119 v1, not the surrogate. See surrogate-model wiki concept page.
  2. **[SUPERSEDED] Spec-peak-only m119 rerun.** Not a priority now that m120 has validated the branch on a meaningful competitor set.
  3. **Multi-seed m120 on ATT_FAIL cohort [highest value, ~50 min].** Seeds 0, 27, 46, 58, 75. Each needs its own `m119v2 setup.npz` first (~45 s of pre-computation per seed reusing existing setup pattern). Then m120 is 5 min/seed. This is the GATE to multi-seed validation of the attitude-isoshell branch. Pattern: generalise `MICRO120_SEED` — it already reads the v2 setup.npz by seed; just need the v2 setup to exist first.
  4. **Extend inline_omega_selection_test to all 10 baseline seeds** — needs `geo_ckpt.npz` for seeds 6, 12, 33, 36, 74, 93 (the harvester run). Lower priority now that attitude-isoshell is validated and could obsolete the grid+NM+geo pipeline.
  5. **Re-score historical omega-candidate pools** with surrogate-MSE where vectors exist (m103_hybrid covers ~13 seeds). Free analytical work.
  6. **Search integration [medium value].** m120 scores a pool given truth is IN the pool. To *find* truth from scratch, combine with [[surrogate-de-search]]: DE to point-estimate, m120-scoring to characterise basin width.
  7. **Visualise seed 14 pab-contour phase-angle offsets.** Pick one epoch with large ang_dist, plot the IPL centroid vs actual truth PAB on the body sphere, overlay with what surrogate says would be allowed at observed mag.
  8. **100-seed population study** — deferred until multi-seed m120 completes.
  9. **Symmetry degeneracy census** — see wiki `symmetry-degeneracies`.
  10. **Fix ±Y phi range bug** in main pipeline — still pending.
  11. **Analytic propagation** (Jacobi elliptic functions) — would speed warm-start 6-DOF further.
- **PROCESS LEARNINGS (2026-04-15):**
  - Reviewer subagents check mechanical script correctness, NOT scientific compute budget. Today's reviewer PASSed `m116_unified_formulation.py` (1405 lines, ~14 hr batch) on conventions but missed that the budget was ~5× the closest analogous experiment (m115 at 5 min/seed). **New strategist-time gate:** before launching any experiment, compare per-seed cost to nearest analogous experiment and flag >2× outliers BEFORE writing the script.
  - Reviewer's "stylistic" notes can be fatal. m116's redundant `global` declaration was flagged as a style nit but caused immediate `SyntaxError`. Don't downgrade reviewer findings without verifying.
  - **Strategist-time data-lifecycle gate:** before any experiment runs, ask "would we ever want to query candidate-level state (not just winner) from this run later?" If yes, the script MUST save full pools per stage. The `feedback_checkpoint_design_first.md` rule covers this for new experiments; the strategist must enforce it at design time, not leave it to the writer alone.
  - **Checkpointing rule history:** `feedback_checkpoint_design_first.md` was created AFTER m102 ran. m103+ scripts follow it (which is why we have geo_ckpts for 4 seeds). Old experiments only saved winners. Not an ongoing failure — but legacy data gap costs us the harvester re-run.
- **SURROGATE MODEL (NEW, 2026-04-13):**
  User provided a trained MLP surrogate at `~/surrogate_model/` that replaces the hi-fi forward model.
  - **Validated:** 0.03 mag MAE, 0.999+ correlation with hi-fi across 3 seeds (27, 93, 0)
  - **Speed:** ~5ms per 500-epoch LC → 28× faster than lo-fi, 50,000× faster than hi-fi
  - **Key insight [inline]:** Surrogate MSE landscape has the SAME false minima as lo-fi when omega is wrong. Shadows don't fix the omega-error problem. For seeds 27, 0, 46, 58, 75 — truth q0 has WORSE MSE than DE-found q0 with BOTH surrogate and lo-fi cost.
  - **Root cause:** omega error (~3°) shifts the predicted LC systematically; no cost function can compensate.
  - **What the surrogate DOES enable:** (1) multi-start 3-DOF DE in ~15s each, (2) 6-DOF joint search, (3) basin enumeration
- **CRITICAL PHILOSOPHY CHANGE (2026-04-13, previous session):**
  Light curve inversion is ill-posed. Multiple (q0, omega) pairs produce observationally indistinguishable light curves. **The pipeline goal is to find ALL valid solutions (hi-fi MSE below threshold), NOT a single best match to truth.** Evaluate by candidate SET quality, not proximity to one truth.
  - **Evidence:** m103 seed 14's "FAIL" (q0_err=178°) has BETTER hi-fi MSE (0.249) than the truth-adjacent solution (0.489). It found a valid degeneracy.
  - See wiki `multi-solution-philosophy` and memory `feedback_multi_solution.md`.
- **m115 COMPLETE — BREAKTHROUGH** (`notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py`):
  - Surrogate multi-start 3-DOF DE across ALL 10 m102 baseline seeds.
  - **10/10 seeds have valid solutions (hi-fi MSE < 1.0).** m102 had 0 OK / 2 PARTIAL / 8 FAIL by strict criteria.
  - **6/10 seeds have truth-adjacent basin (q0 < 10°).**
  - Surrogate tracks hi-fi to ±0.005 — reliable for ranking and search.
  - Total compute: 50 min for 10 seeds (5 min/seed avg). m102 took 12 min/seed for worse results.
  - **Key per-seed results:**

    | Seed | m102 q0 | m115 hifi_MSE | m115 q0 | m115 w_dir | n_basins |
    |------|---------|---------------|---------|------------|----------|
    | 0    | 8.7°    | 0.140         | 6.7°    | 2.9°       | 4        |
    | 6    | 176.8°  | 0.130         | 176.8°  | 3.1°       | 4        |
    | 12   | 169.6°  | 0.327         | 10.2°   | 8.0°       | 4        |
    | 14   | 90.7°   | 0.161         | 2.3°    | 0.3°       | 6        |
    | 24   | 152.3°  | 0.044         | 2.2°    | 1.9°       | 6        |
    | 27   | 175.4°  | 0.310         | 169.2°  | 3.1°       | 17       |
    | 33   | 134.7°  | 0.082         | 98.5°   | 18.5°      | 6        |
    | 36   | 177.3°  | 0.602         | 14.1°   | 10.7°      | 5        |
    | 74   | 6.7°    | 0.376         | 172.7°  | 4.9°       | 3        |
    | 93   | 178.4°  | 0.063         | 179.9°  | 0.5°       | 3        |

  - **Seed 14, 24:** m102 selected WRONG omega (56°, 35° error); m115 used best omega from geo_ckpt (0.3°, 1.9°) → truth basin found at 2.3° and 2.2°.
  - **Seed 33:** Predicted failure (best omega 18.5°) found valid alternative (MSE=0.08, q0=98.5°) — genuine LC degeneracy, not a twin.
  - **Seed 12:** Truth basin recovered at q0=10.2° (m102 had 169.6°).
  - Results: `data/results/inversion_diagnostics/m115_surrogate_pipeline/`
  - Per-seed: `seed_NNN/result.json`, `step1_de.npz`, `step2_hifi.npz`
  - Population: `batch_summary.json`
- **TWIN QUATERNION CONVENTION FIX (2026-04-13, this session):**
  - Correct twin: `q0_twin = q_180x * q0_true` (LEFT multiply = body-frame rotation), SAME omega.
  - RIGHT multiply (`q0_true * q_180x`) is WRONG — applies R_180x in J2000 space, gives MSE ≈ 5.
  - Both give `attitude_error_deg = 180°` — the error metric doesn't distinguish them.
  - m114's `check_twin_degeneracy` uses left-multiply (correct).
  - See wiki `twin-degeneracy` and `symmetry-degeneracies`.
- **m114 PARTIAL** (`notebooks/inversion/12_brightness_surface/m114_surrogate_multistart.py`):
  - Surrogate-powered multi-start 3-DOF DE + 6-DOF joint search on seed 27
  - **Step 1 (validation):** Surrogate MAE 0.031-0.034 mag, r=0.999, 9ms per 500-epoch eval.
  - **Step 2 (3-DOF multi-start, COMPLETE, 11.1 min):** 5 omegas × 10 starts = 50 solutions.
    - omega[0] (w_err=3.1°): 2 basins found — truth (14.3°, MSE=0.316) and twin (169°, MSE=0.314). Twin has BETTER MSE = valid degeneracy.
    - omega[1-4] (w_err=17-29°): all MSE > 2.4. Bad omegas → useless. Only omega[0] matters.
    - Confirms: multi-start basin enumeration works, but omega quality is the bottleneck.
  - **Step 3 (6-DOF, KILLED after 3/5 starts, each ~55 min):** 
    - **NEGATIVE RESULT.** All 3 completed starts: q0_err 125-163°, w_dir 77-87°. DE cannot find omega from scratch in [-0.03, 0.03]³.
    - Root cause: 6D search space too large. Each ODE eval ~40ms → 72k evals × 40ms = 48 min per start. The search budget is exhausted exploring the wrong omega regions.
    - **This vindicates the existing pipeline architecture:** grid+NM for omega, then DE/phi-sweep for attitude. The grid's alignment cost IS the right tool for omega; the surrogate's value is in the attitude step.
  - Steps 4-5 (clustering, hi-fi validation) never ran due to kill.
  - Results: `data/results/inversion_diagnostics/m114_surrogate/seed_027/pipeline.log`
  - **IMPORTANT:** Set `OPENBLAS_NUM_THREADS=1` before importing numpy — prevents BLAS multi-threading from saturating CPU in single-threaded DE.
- **m113 COMPLETED** (`notebooks/inversion/12_brightness_surface/m113_de_attitude_search.py`):
  - Ceiling test (truth omega, seed 27): q0_err = 0.6° — DE finds truth when omega is perfect.
  - Estimated omega: mixed results. Seed 27: 165°→17°. Seed 0: 90°→7°. Seeds 46, 58: lo-fi false minima persist.
  - Full results: `data/results/inversion_diagnostics/m113_de_attitude/seed_027/result.json`
- **NEXT STEPS (post-m115):** SUPERSEDED by the new top-level "NEXT STEPS" block above. Item 3 ("Omega selection improvement") was answered by the 2026-04-15 inline test.
- **IPL census (2026-04-12, COMPLETED):**
  - Script: `notebooks/inversion/12_brightness_surface/archive/ipl_census.py`
  - Full per-epoch data: `data/results/inversion_diagnostics/isoshell_viewer/ipl_all_epochs.npz` (all 500 epochs × 100 seeds)
  - Census JSON: `data/results/inversion_diagnostics/isoshell_viewer/ipl_census.json` (minima only)
  - 83% of seeds have ≥2 tight minima (centroid < 5° from truth PAB)
  - Median best centroid distance: 1.2°. Dominant loop counts at minima: 4 (55%), 8 (16%), 6 (11%)
  - **Discrimination gap confirmed:** At 57% of tight minima, the truth PAB is >20° from the nearest standard normal but only 2.3° from the nearest IPL centroid. 92/100 seeds have at least one such epoch. The standard alignment cost is blind where IPL centroids see clearly.
  - **For IS-901: 2-loop = ±X, always.** No 2-loop epochs exist at dim magnitudes. At dim magnitudes, 4-loop (35%) and 6-loop (40%) dominate. The 4-loop centroids are ±X + ±Z (or ±YZ-combo). 945 four-loop epochs at dim magnitudes have centroid distance < 10° across the population.
- **m107 — IPL centroid grid cost (FAILED, 2026-04-12):**
  - Script: `notebooks/inversion/12_brightness_surface/m107_ipl_cost_function.py`
  - Used 8 tight IPL minima as constraint epochs with centroid alignment cost
  - Seed 27: truth ranked 754/2000 in grid (vs ~100 with standard cost)
  - **Root cause: delta-q amplification.** 0.67° grid omega error → 30° body-frame PAB error at 40 min from anchor. IPL centroids (1-3° targets) can't be hit with 30° error.
- **m108 — Central-anchor IPL cost (FAILED, 2026-04-13):**
  - Script: `notebooks/inversion/12_brightness_surface/m108_ipl_central_difference.py`
  - Two strategies tested: (A) minimize max |dt| to cluster, (B) minimize max(centroid_dist + PAB_error)
  - Strategy B gives better clusters but WORSE grid ranking (seed 27: 1470/2000)
  - **Root cause: too many centroids → false positives.** Epochs with 8-12 IPL loops have centroids covering most of the sphere. `max_j(centroid_j · pab)` is high for ANY direction → no discrimination. The tighter the centroid distances (better cluster), the more loops (worse discrimination). Fundamental tension.
  - 100-seed population analysis [inline] confirmed: with strategy B (K=4), all 81 viable seeds have estimated grid cost < 0.05 (better than m102's best). But ACTUAL grid runs fail because the cost ESTIMATES assumed worst-case PAB error, while the real false-positive rate from many-centroid epochs is much higher.
- **KEY FINDING — Failure mode reclassification (2026-04-13):**
  - Analysed all 15 m103 seeds by failure mode:
    - **3 GRID failures** (seeds 1, 28, 44): omega not in candidate pool — need wider search
    - **3 TWIN** (seeds 14, 19, 24): valid ±X optical degeneracy — inherent, not a failure
    - **6 ATT_FAIL** (seeds 0, 11, 27, 46, 58, 75): **omega is correct (<5°) but attitude is wrong (47-165°)**
    - **ATT_FAIL is the dominant failure mode.** The grid finds the right omega, the right anchor direction (< 2.5° from truth), but the PHI SELECTION produces the wrong twist angle.
  - Standard alignment cost at truth omega at bright peaks: already ~0.001 (near-zero). The grid cost function is NOT the bottleneck.
- **Phi discrimination tests (2026-04-13, all [inline]):**
  - **Multi-peak lo-fi (3 peaks, 11 epochs each):** For OK seeds (93, 73), truth phi found exactly (51x discrimination). For ATT_FAIL seeds, wrong phis produce 1.3-5.1× LOWER lo-fi MSE than truth. Lo-fi can't resolve phi for constraint-poor seeds.
  - **Shadow analysis:** At many peaks, lo-fi = hi-fi (zero shadow effect). At others, shadows add 0.1-0.5 mag asymmetry on peak flanks. Some zero-shadow peaks also fail → **shadows are not the only issue**.
  - **Plateau lo-fi (15 dim epochs, constant magnitude):** Massive 125× MSE discrimination range — much more discriminating than peaks (3.7×). But truth STILL ranks 8.9× worse than best phi. The lo-fi model is systematically biased at dim magnitudes for these seeds.
  - **Centroid tracking over dim epochs:** Tested continuous PAB tracking through 4/6-loop structure. Average centroid distance at dim epochs is 30-40° → quadrant assignment is meaningless. Tracking doesn't discriminate phi.
  - **Near-centroid epoch tracking (5 epochs, centroid_dist < 15°):** All phis give identical scores (avg_dist ~6.9°). Runs too short, distance varies too much within run.
  - **Conclusion: lo-fi brightness evaluation cannot distinguish truth phi from false positives for ATT_FAIL seeds**, regardless of which epochs are used. The brightness surface is too symmetric at the lobes these seeds peak at.
- **m109 — IPL Phi Discrimination Diagnostic (2026-04-13, NEGATIVE):**
  - Script: `notebooks/inversion/12_brightness_surface/m109_ipl_phi_diagnostic.py`
  - Results: `data/results/inversion_diagnostics/m109_ipl_phi/diagnostic.npz`
  - Tested 8 metrics for phi discrimination on 6 seeds (3 ATT_FAIL + 2 borderline + 1 OK)
  - **All IPL centroid metrics FAIL:** centroid proximity disc=0.997, weighted disc=0.997, membership 2/5, stability 0/5
  - **Derivative matching marginally better:** 5/5 win rate but min disc=1.008 (seed 58 barely passes)
  - **Zero-phase lo-fi MSE:** 4/5 win rate, disc=1.76 mean (BUT this is for twist-at-t0 parameterization, not pipeline-equivalent)
  - **Root cause:** isoshell = zero-phase brightness surface, which is too symmetric about lobe normals. IPL centroids ≈ standard normals at tight epochs. All IPL-derived metrics reduce to alignment cost or lo-fi MSE.
  - **Wiki:** `isoshell-phi-limits` concept page documents the analysis comprehensively
- **m110 — Sparse Hi-Fi Phi Sweep (2026-04-13, MIXED):**
  - Script: `notebooks/inversion/12_brightness_surface/m110_hifi_phi_study.py`
  - Results: `data/results/inversion_diagnostics/m110_hifi_phi/seed_*.npz`
  - Hi-fi + lo-fi evaluation at 10 brightest peaks, 72 phi values (5° step), truth omega
  - **Seed 27: WORKS** — hi-fi rank 1/72 (disc=1.038), lo-fi rank 1/72 (disc=1.067)
  - **Seeds 46, 58, 0, 75: FAIL** — hi-fi ranks 5-20/72, disc 0.46-0.90
  - **Even OK control (seed 93): FAIL** — hi-fi rank 4/72 (disc=0.881)
  - **Root cause:** 10 peaks insufficient signal. The 10 brightest peaks are at ±X lobes where shadows are WEAK. Shadow-rich epochs (±Y lobes) are at dim magnitudes, not in the top 10.
  - **Key finding:** exact truth attitude gives MSE=0.0015 (noise floor), phi-parameterized truth gives MSE=0.18. The 1-2° phi match error already dominates. At 5° grid spacing, noise and parameterization error overwhelm the discriminating signal.
- **Shadow asymmetry quantification (2026-04-13):**
  - Shadow effects reach 4.6 mag at ±Y lobes (solar panel occlusion)
  - 40-44% of epochs have |shadow| > 0.1 mag
  - Shadows concentrated at ±Y lobes — these are NOT among the 10 brightest peaks
  - **Wiki:** `shadow-asymmetry` concept page
- **OPEN QUESTIONS for next session:**
  1. **Surrogate multi-start DE on 10 baseline seeds:** Replace phi sweep with 10-start surrogate 3-DOF DE per omega candidate. Load omega from m103 geo_ckpt (4 seeds) or m102 result.npz (6 seeds). Hi-fi validate top basins. Direct A/B vs m102.
  2. **Symmetry degeneracy census (wiki: `symmetry-degeneracies`):** For R in {R_180x, R_180y, R_180z, R_90x, small-angle-near-X, ...}, compute surrogate+hi-fi MSE of (R*q0_true, omega_true) vs truth LC across 5-10 seeds. Map exact symmetries, near-symmetries, and symmetry-breakers. Quantify lo-fi vs hi-fi symmetry groups — hypothesis: lo-fi has LARGER symmetry group, explaining ATT_FAIL. Check near-twin basin shape (sweep 0°→180° about X at 1° steps).
  3. **Twin convention fix audit:** The correct twin is `q_180x * q0` (LEFT multiply, body-frame rotation), SAME omega. RIGHT multiply (`q0 * q_180x`) is wrong — applies R_180x in J2000. Verify all scripts and wiki pages use the correct convention.
  4. **Warm-started 6-DOF:** Seed DE population near grid+NM omega (±5° bounds). Keeps grid's omega-finding while jointly refining omega + attitude with the surrogate.
  5. **Replace alignment cost with lo-fi MSE for pipeline phi selection:** The pipeline uses alignment cost for phi, which is KNOWN to fail. Simply switching to lo-fi MSE might fix some ATT_FAIL seeds. Zero cost, zero risk. Testable by re-scoring existing m103 data.
- **Precomputed data assets (reusable):**
  - `ipl_census.json`: per-seed IPL minima with centroid distances, loop counts (100 seeds)
  - `ipl_all_epochs.npz`: full per-epoch IPL data — centroids, loop counts, angular distances, crossing arcs at all peaks (100 seeds × 500 epochs, 2.3 MB)
  - Both generated by scripts in `notebooks/inversion/12_brightness_surface/`
- **m104 crossing geometry diagnostic (PARTIALLY refuted):** Direct extraction via peak shapes doesn't work (Ω_L≠ω_body + noisy FWHM). But the CONSTRAINT SATISFACTION reformulation (m105) works.
- **m103 expanded (13 seeds):** 2 OK + 2 PARTIAL + 9 FAIL
  - FAIL breakdown: 6 selection failures (truth in pool), 3 grid failures (truth not in pool)
  - m105 addresses the 3 grid failures; selection failures need separate treatment
- **m103 design:** Two changes from m102: (1) top-2 multi-phi after NM dedup (4 phis, 20° sep → 26 candidates for geo), (2) hybrid selection (window consensus with full-MSE fallback).
  - **Re-scoring analysis (2026-04-09):** Comprehensive analysis of all 10 m102 seeds showed 3/4 FAIL seeds are SELECTION failures (truth omega in pool but wrong candidate selected). Only seed 33 is a genuine grid failure. Hybrid selection alone fixes seed 27; multi-phi fixes seeds 14, 24 by providing better attitude starting points.
- **PRIOR:** m102 full-MSE selection — 10 seeds complete (2 OK + 2 PARTIAL + 6 FAIL by strict criteria).
  - **m100 (multi-phi diagnostic):** 4 phis per omega + best-by-geo + vote. Seeds 14, 24: FAIL→OK (+X twin, 62-1314% MSE margin). Seed 0: FAIL (vote noise + wrong phi selection). Confirmed multi-phi fixes attitude basin problem.
  - **m101 (multi-phi v2):** NM phi + best-geo phi + full-MSE. Seed 6 geo hung >50 min (L-BFGS-B on flat landscape with 80 candidates). maxfun=1500 cap fixed hang but degraded seed 12 (truncated geo convergence). Multi-phi approach shelved — adds complexity and runtime (25 min/seed) for mixed results.
  - **m102 (conservative fix):** NM_TOP=300 + full-window MSE selection, NO multi-phi. Re-scoring m099 data proves: no regressions, seed 74 FAIL→PARTIAL. Same ~12 min/seed as m099. **This is the current best pipeline.**
  - **Three findings from m100-101-102:**
    1. **Full-MSE > multi-window vote** — vote is noisy, full-MSE is safe. Free improvement.
    2. **Multi-phi fixes attitude basin problems** (seeds 14, 24) but risks regression via wrong-phi selection (seed 0) and adds 4× geo cost.
    3. **LC MSE doesn't perfectly correlate with actual errors** — fundamental limitation. Seed 0: wrong phi has MSE 0.38 vs correct phi's 0.48.
  - **Population analysis:** Our 10 test seeds are biased toward high-spec, fast-tumbling cases (seeds 0, 14, 74, 93 all in top 4% for specular peaks). Seed 6 (5 spec, 0.71 dps) is the most representative.
- **PRIOR (m099):** NM_TOP=300 with 2000-dir grid on 10 seeds. 3 OK + 2 PARTIAL + 5 FAIL. Net regression from m090.
  - **m096 Stage 1:** 100-seed constraint census. 87% lack bright ±X constraints. Pipeline designed for atypical 13%.
  - **m096 Exp 1-5:** Oracle grid (alignment rank median 105/500). Lo-fi MSE discriminates truth 100/100 vs random. |w| estimation: 12.8% median error. Medium-band peaks richest constraints.
  - **m097a (oracle q0):** Lo-fi MSE as grid ranker — median rank 178 vs alignment 105. WORSE overall but improves 34/90 seeds.
  - **m097b (phi-sweep q0):** Lo-fi MSE — median rank 124. Better than oracle q0 but still worse than alignment.
  - **m097c (combined):** Weighted sum (alpha=0.85) median 82. Two-stage (align top-K → lo-fi): K=300 captures 81/90 seeds.
  - **m097 conclusion:** Lo-fi MSE is a GLOBAL discriminator (truth vs random) but POOR LOCAL discriminator (truth vs nearby grid omegas). Cannot replace alignment cost at grid level.
  - **m098 (seed 0):** NM_TOP=300 pipeline with 500-dir grid — FAIL. 3.6° grid spacing exceeds 0.5° NM basin. Need 2000-dir grid.
  - **m099 (10 seeds, 2000-dir grid, NM_TOP=300):** 3 OK + 2 PARTIAL + 5 FAIL. **Net regression from m090** (4 OK + 3 PARTIAL + 3 FAIL). NM expansion rescued seed 6 (PARTIAL→OK) and seed 12 (FAIL→OK) but regressed seeds 14, 24, 36 (OK/PARTIAL→FAIL). Root cause: hi-fi multi-window selection picks wrong candidates from the larger deduped pool (20 vs ~10 in m090). The NM expansion FINDS truth (seed 24 had 0.6° at geo#1) but hi-fi SELECTS wrong. **The bottleneck has shifted from omega finding to candidate selection.**
- **PRIOR:** m094 — adaptive cost (hybrid BRDF for ≥6 bright constraints, pure alignment otherwise) + multi-phi (top-4 per omega) + 8000 Fibonacci dirs + alignment geo refinement.
  - **m094 seed 93:** OK — q0=179.1° (+X twin), w_dir=0.16°, w_mag=-0.09%. Multi-phi fixed the attitude basin problem. Hi-fi correctly selected the right phi.
  - **m094 seed 0:** OK — q0=3.6° (direct), w_dir=0.13°, w_mag=+0.20%.
  - **m094 seed 6:** Omega FOUND at 1.5° (w#4 after geo) but geo rank picked wrong candidate. Hi-fi would fix this. Full candidate pool NOT saved due to checkpointing bug.
  - **m094 seed 12:** FAIL — grid failure, all top-5 at 86-89°.
  - **m094 seed 27:** FAIL — grid failure at both 2000 and 8000 dirs. Only 3 bright + 5 dim constraints; bright ones geometrically redundant.
  - **Key discovery: 17,000× cost scale mismatch** between BRDF (magnitude²) and alignment ((1-dot)²) costs. Pure BRDF cost at all peaks drowns dim alignment constraints. Adaptive strategy fixes this.
  - **Key discovery: multi-phi essential** — single phi selection converges to wrong attitude basin. Top-4 separated phis give geo refinement the correct starting point.
  - **Remaining: seeds 14, 24, 33, 36, 74 not yet run with final architecture.**
  - **m090:** m077 + Savitzky-Golay smoothed anchor selection. Fixes noise-sensitive anchor flipping (seed 93 anchor was flipping between ep 27 and ep 189 across noise realizations). 5-line change. Results on 10-seed validation (noise seed 42):

    | Seed | q0 err | w_dir err | w_mag err | Status | Notes |
    |------|--------|-----------|-----------|--------|-------|
    | 0 | 3.63° | 0.13° | +0.20% | OK | +X twin, full attitude |
    | 6 | 178.09° | 3.01° | -0.14% | PARTIAL | NOT +X twin (axis≈Y) |
    | 12 | 111.17° | 89.17° | +0.09% | FAIL | scipy version issue |
    | 14 | 178.95° | 1.65° | -0.00% | OK | +X twin confirmed |
    | 24 | 179.94° | 0.25° | +0.04% | OK | NOT +X twin (axis≈Y+Z) |
    | 27 | 176.43° | 36.38° | +0.33% | FAIL | scipy version issue |
    | 33 | 134.03° | 17.87° | -0.68% | FAIL | scipy version issue |
    | 36 | 173.56° | 3.70° | +0.17% | PARTIAL | NOT +X twin (axis≈Y) |
    | 74 | 5.74° | 4.41° | -0.07% | PARTIAL | |
    | 93 | 179.72° | 0.12° | -0.03% | OK | +X twin confirmed |

  - **m091:** Hi-fi twin test — 180° rotation about each of 10 normal vectors, identity q0, same omega, with shadows. **Only ±X produces identical LCs (RMS=0.000000).** ±Y/±Z give RMS=0.542 (shadows break the symmetry). ±WD/±ED give RMS≈1.6 (inertia tensor not preserved). Data saved to `m091_twin_test/twin_lcs.npz`.
  - **Noise seed bug found and fixed:** m077 was committed with `default_rng(42 + TRAJ_SEED)` but batch results used `default_rng(42)`. Now reverted to `default_rng(42)`.
  - **3 failing seeds (12, 27, 33):** Same anchor, same noise, same code — different grid results due to scipy/ODE version change between batch run (Apr 1-4) and now. Alignment cost basins are <0.5° wide, so tiny ODE differences flip which grid directions score well.
- **m092 twin-axis diagnosis:** Seeds 6, 24, 36 show q0 err near 180° but error rotation axes are near -Y (dot with +X < 0.04), NOT valid +X twins. Omega recovery is fine (3-4° dir err), attitude is wrong. Visualisation tool: `python3 notebooks/inversion/lib/attitude_viz.py 6 24 36` or `/attitude-viz` skill.
- **NEXT STEPS:**
  1. **m094 — BRDF-based cost function (PRIMARY).** Replace alignment cost with direct Ashikhmin-Shirley BRDF brightness prediction. Uses n·k1, n·k2, n·h (3 constraints per epoch vs 1). Eliminates calibration table. See `MICRO94_BRDF_COST_PLAN.md` for full plan, rationale, implementation details, and critical questions to evaluate.
  2. Consider combining BRDF cost with N_DIRS=8000 (m086 showed 8000 needed for ~1° spacing).
  3. Re-run on all 25+ seeds once m094 is validated.
- **Prior findings from m087-88 still valid:** SLERP unsafe with NM, calibration caching works, relaxed ODE tolerances negligible error but untested end-to-end.
- **WHAT WORKS (use these as building blocks):**
  1. ~~Attitude recovery given known omega~~ — HISTORICAL. 9/10 with oracle omega (m051b), but q0 and omega_dir are inseparable: knowing omega implicitly constrains the attitude search. Not a standalone capability.
  2. Omega magnitude from peak count (ALL peaks, not just bright) — 13% median error, rho=0.92 (m052)
  3. Specular glint identification — mag < 6.0 = 100% specular, always ±X (104/104, m047)
  4. PAB-circle 1-DOF attitude constraint at glint epochs (m034-36)
  5. **Delta-q factorization** — propagation is q-independent for torque-free dynamics. One propagation per (dir, mag) serves all phi candidates. (m067)
  6. **Full-curve hi-fi scoring** — hi-fi correctly selects truth at rank #1. (m069)
  7. **180° optical twin — ±X ONLY.** Rotating q0 by 180° about +X produces identical hi-fi LCs (RMS=0.000000 mag). ±Y/±Z preserve inertia (same dynamics) but produce different shadow patterns (RMS=0.542). ±WD/±ED break inertia entirely (RMS≈1.6). Only ±X is a true twin. (m071e, m091)
  8. **Hi-fi brightness calibration table** — for each of 10 normals, precompute magnitude vs alignment angle (0°-60°) averaged over phi twist angles. 84s, 1200 single-epoch hi-fi evals. (m082-86)
  9. **Expected-dot cost** — replaces (1-dot)^2 with (dot - expected_dot)^2 where expected_dot is inverted from calibration table given observed magnitude. Naturally weights bright peaks tightly, eliminates X/WD/ED co-alignment degeneracy. (m084-86)
  10. **Fine grid (8000 dirs)** — ~1° spacing. No omega refinement needed — grid does the work. Fine phi sweep (720 bins) for attitude. Straight to hi-fi. 17.5 min/seed. (m086)
- **REMAINING WORK:**
  - **Replace alignment cost with expected-dot cost in NM (m093).** Root cause identified: alignment cost max-over-normals picks the wrong normal when candidate attitude is off by a few degrees. Expected-dot cost constrains the correct normal via brightness lookup. Never cleanly tested with NM. See NEXT STEPS #1.
  - ±Z degeneracy: +Z and -Z are interchangeable (mirror symmetry about xy plane). Not yet applied — would halve Z-related search space.
  - Break the twin degeneracy (may require multi-epoch polarimetry or accept as inherent)
  - Report preparation
- **GRID SPEED OPTIMISATIONS (m087-88 results):**
  1. **Magnitude SLERP interpolation — UNSAFE with NM.** Gives ~4× grid speedup, but changes NM candidate pool enough to cause 4/10 regressions in m088. SLERP quaternion error is O(delta_mag²) ≈ 0.1° — negligible for the grid cost, but the downstream NM convergence is chaotic and amplifies this. **Do not use SLERP with NM-based pipelines.** May be safe in pipelines without NM (m086-style), where it was validated on seed 35.
  2. **Relaxed ODE tolerances — SAFE, ~3× speedup.** rtol=1e-6 for grid search (rtol=1e-10 kept for NM/geo/final propagation). No accuracy loss detected on any seed tested. This is the recommended optimisation. NOT YET TESTED in isolation on m077 pipeline — needs m089.
  3. **Pre-sorted dt_constraints — minor.** Sorts fwd/bwd times once instead of per-call.
  4. **Calibration caching — VALIDATED for m086 pipeline.** N/A for m077 (no calibration step).
  Still untried: drop weak constraints (~1.5×), cache rotation matrices (~1.3×).
- **KEY FINDINGS (m087-88, 2026-04-07):**
  - **SLERP magnitude interpolation is unsafe with NM.** m088 (m077 + SLERP) regresses 4/10 seeds (12, 27, 33, 36). The SLERP error is tiny (~0.1° quaternion) but NM convergence is chaotic — different grid candidates → different NM starting points → different (wrong) convergence basins. The pipeline's NM step acts as a sensitivity amplifier.
  - **Relaxed ODE tolerances appear safe.** rtol=1e-6 tested in grid search across many seeds. No regressions attributable to tolerance alone. Gives ~3× grid speedup. Should be tested in isolation (m089).
  - **m086 pipeline (expected-dot cost, no NM) is not competitive.** m087 got 3/10 OK, 4/10 PARTIAL on m077's working seeds. NM refinement is what makes m077 work.
  - **NM with expected-dot cost converges to wrong minima.** Tested on seed 27: most candidates moved away from truth.
  - **NM with alignment cost + safety cap also fails.** Starting points from the optimised grid are too far from NM basin.
  - **Noise seed: ALWAYS use `default_rng(42)`.** The m077 batch used fixed seed 42. The code was mistakenly changed to 42+seed and has been reverted. Different noise seeds produce different anchor selections and break reproducibility.
  - **Lo-fi peak matching is critical.** For seed 93, grid top-5 had >42° error for both costs, but lo-fi found truth at rank #1 (24/28 peaks matched).
  - **m088 full results (m077 + SLERP + relaxed tol + N_MAGS=40):**

    | Seed | m088 w_dir / q0 / w_mag / time | m077 w_dir / q0 / w_mag / time | |
    |------|-----------------------------------|----------------------------------|-|
    | 0 | 1.1° / 179.1° / +0.2% / 187s | 1.9° / 175.7° / +0.1% / 483s | OK |
    | 6 | 3.1° / 5.4° / -0.2% / 498s | 2.6° / 163.1° / -1.1% / 806s | OK |
    | 12 | 89.0° / 172.2° / +0.0% / 102s | 3.3° / 175.0° / -0.0% / 248s | FAIL |
    | 14 | 1.7° / 179.0° / -0.0% / 155s | 1.7° / 178.9° / -0.0% / 309s | OK |
    | 24 | 2.1° / 178.9° / -0.1% / 303s | 0.7° / 179.8° / +0.0% / 302s | OK |
    | 27 | 64.2° / 122.1° / +23.5% / 486s | 2.6° / 4.5° / +0.1% / 363s | FAIL |
    | 33 | 84.8° / 97.9° / -2.0% / 151s | 0.2° / 179.9° / -0.1% / 403s | FAIL |
    | 36 | 26.2° / 78.9° / +0.5% / 140s | 4.0° / 7.1° / +0.1% / 413s | FAIL |
    | 74 | 1.9° / 149.8° / +0.1% / 144s | 0.5° / 145.7° / +0.2% / 429s | OK |
    | 93 | 1.1° / 1.2° / -0.0% / 155s | 1.1° / 1.2° / -0.0% / 424s | OK |
- **KEY FINDINGS (2026-04-07):**
  - **Co-alignment degeneracy:** +X, +WD, +ED normals are 15° apart (dot=0.966). Alignment cost (1-dot)^2 can't distinguish them. Brightness calibration breaks the degeneracy because ±X produces mag 4.93 at perfect alignment while ±WD/±ED produce mag 7.43. (m080)
  - **NM refinement destroys good candidates:** step 2b found truth at rank #2 (w_dir=0.3°) for seed 35, but NM moved it to 30.2° using the degenerate alignment cost. Same pattern in seeds 17, 8. Fix: bypass NM entirely. (m081)
  - **Geometric refinement (L-BFGS-B) also corrupts:** same alignment cost degeneracy. Lo-fi LC refinement also fails — lo-fi is systematically biased (no shadows), so its MSE minimum differs from truth. (m082-85)
  - **Coarse magnitude grid caused regressions:** widening from ±20% to ±30% with same 20 points made grid 50% coarser, breaking seed 93 entirely. Fix: N_MAGS=40 to maintain density. (m081)
  - **Grid resolution is the bottleneck, not refinement:** 2000 dirs gives ~3° omega error which compounds to ~130° q0 error over 1450s back-propagation. 8000 dirs gives ~1° → solvable. No refinement step can reliably improve beyond grid accuracy. (m084-86)
- **KEY BUGS FIXED (2026-04-04):**
  - Beta pipeline banner said "Alpha" (cosmetic)
  - Timing JSON omitted step 2b
  - Duplicate peak detection in step 2b (recomputed peaks_idx)
  - Step 2b recomputed phi/normal for 200 candidates instead of using cached grid data
  - Noise seed was fixed at 42 for all trajectories (should be 42+seed for independent noise, but reverted to 42 for comparability with old results)
- **KEY BUGS FIXED (2026-03-25):**
  - m068 phi sweep scored against all 10 normals instead of ±X only (inconsistent with grid search)
  - m068 phi sweep used 36 bins (10° spacing) instead of 360 (1°) — introduced unnecessary 30° attitude error
  - Windowed evaluation time-offset bug: ObjectiveFunction places initial state at times[0], not at the anchor
  - Lo-fi scoring on full curve is unreliable: phantom glints from unshadowed facets corrupt candidate selection
  - Anti-glint constraint is invalid: "alignment → glint" is wrong because shadows suppress glints
- **KEY PHYSICS CORRECTIONS (2026-03-20):**
  - R(q) = J2000→body (the code comment was correct all along)
  - Peak-count calibration must use ALL peaks (not just bright) — 13% vs 83% error

---

## 2. Validated Building Blocks

These are established results with quantitative backing. Use them; don't re-derive.

### Basin of Attraction (Series 04, 06)
- Attitude-only basin: ~5° (all 48 trials converge, consistent 0.15° final error)
- Omega direction basin: ~2° (collapses by 10°)
- Omega magnitude basin: ±10% hi-fi, ±5% lo-fi
- Attitude error > 5° collapses the omega basin
- **Target for candidate generation:** attitude within ~5°, omega within ~2° direction

### Timing Benchmarks (exp00)
| Operation | Time |
|-----------|------|
| propagate_attitude (500 epochs) | 77 ms |
| Lo-fi full LC eval | 221 ms |
| Hi-fi full LC eval | 60 s (272x slower) |
| Single-epoch lo-fi | 13 ms |
| Single-epoch hi-fi | 47 ms |
| L-BFGS-B iter (6p lo-fi) | 4,222 ms |
| Omega bridge (single pair) | 162 ms (dt=50s), ~11s (dt=500s) |

### Glint Physics (Series 09, 09b)
- **mag < 6.0 = 100% specular** — zero exceptions across ~6000 peaks, all phase angles
- At specular glint: attitude constrained to 1-DOF circle (rotation about PAB)
- 14 unique normals on IS-901, 10 glint-producing (4 dish-edge faces too small)
- Brightness band → normal group: ±X at 5.5 mag, ±Y/±Z at 6.7-7.1, dishes at 7.6-8.0
- GB classifier F1=0.90 (peak_mag 79% importance, max_slope 15%)
- Geometric glint filter: 78ms/eval, precision 0.94 (true) vs 0.50 (random)
- Anti-glint: at dim epochs (mag>11), cos(8°) threshold on 10 normals, 0 violations at truth

### Attitude Recovery Given Known Omega (Series 10, m051b)
- 9/10 trajectories correct to within 180° ambiguity at [0.2, 1.5] deg/s
- 5/10 fully resolve (median 2.15°), 4/10 return antiparallel (fixable post-hoc), 1/10 wrong
- Pipeline: 10 hyp × 36 phi → PAB alignment score → top 4 lo-fi → top 2 hi-fi → winner
- Runtime: ~120s/trajectory (dominated by 2 hi-fi evaluations)
- ±180° ambiguity is inherent for box-shaped satellites; hi-fi shadows resolve 5/9 cases

### L-Conservation Winding Filter (Series 07b)
- L = R(q) @ (I @ omega_body) is conserved in inertial frame
- At shared peak node: ||DL|| ranks true pair #1/64, gap = 113 kg·m²/s (9 orders of magnitude)
- Shared-node attitude error is **mathematically invariant** (R cancels in the norm)
- Robust to 10° endpoint error (100% correct, gap = 95)
- T (kinetic energy) adds nothing beyond L

### Omega Magnitude from Peak Count (m052)
- |omega| = 0.040 × n_peaks + 0.042 (deg/s), rho=0.954
- Median error 13.2%, within ±20% for 75/100 trajectories
- Lomb-Scargle is weaker (rho=0.740, median 31.1%)

### Bridge Omega Generation (Series 03, 06, 10)
- Bridge solver: L-BFGS-B on quaternion distance, parallelizes well (5x on 8 cores)
- Band-sweep enumeration (0.5 dps bands, 10 random starts) finds all winding families
- Bridge GENERATES correct omega (0.3° direction error with oracle attitudes)
- Bridge SELECTION fails: LC scoring ranks correct omega at #34K/90K (chicken-and-egg)
- **Status: generation works, selection is the unsolved problem**

### Key Physics
- R(q) = body→inertial (code comment "J2000 to body" is WRONG, but used consistently)
- L_inertial = R(q) @ (I @ omega_body) — constant for torque-free motion
- omega errors compound: δq ~ δω × t (2° omega error → 126° attitude drift at 3600s)
- L errors don't compound (L is constant) — **motivation for L-parameterization**
- IS-901 asymmetry = 0.556 — triaxial, analytic propagation fails, Euler dynamics required
- Constant-omega approximation valid: drift < 0.001 dps over 3600s (m007a)

---

## 3. Series Index

Each series has its own directory with scripts and FINDINGS.md. Read the FINDINGS.md for detail.

| Series | Directory | Status | One-Line Summary |
|--------|-----------|--------|-----------------|
| 00 | `notebooks/inversion/` (01-08_*.py) | DONE | Forward model and pipeline reference |
| 01 | `notebooks/inversion/` (exp_*.py) | CLOSED | Global optimizers all fail on 6D joint space |
| 02 | `notebooks/inversion/` (exp_*filter*.py) | CLOSED | Iso-brightness cascade: combinatorial explosion |
| 03 | `notebooks/inversion/` (exp_omega_*.py) | CLOSED | Chain bridging: aggressive culling kills truth |
| 04 | `notebooks/inversion/` (exp0[0-3]_*.py) | DONE | Basin characterization (5°/2°/±10%) |
| 05 | `notebooks/inversion/` (m001-14) | CLOSED | Peak-graph pipeline: scoring cannot discriminate |
| 06 | `notebooks/inversion/` (bench_*, m001[a-d]) | DONE | Bridge solver benchmarks: 162ms/pair, linear noise |
| 07 | `07_multi_epoch_scoring/` | CLOSED | Multi-epoch winding score: omega direction is wrong |
| 07b | `07_L_conservation/` | DONE | L-conservation winding filter: validated (9 OoM gap) |
| 07c | `07_winding_enumeration/` | DONE | Band-sweep fixes staircase gap |
| 08 | `08_integration/` | CLOSED | Bridge coverage bottleneck on long legs |
| 09 | `09_glint_analysis/` (m034-48) | DONE | Glint physics: mag<6 = specular, PAB circles, classifier |
| 09c | `09_glint_analysis/` (m041-43) | DONE | Two-phase phi sweep: 1.77° att, 1.15° omega (oracle ω) |
| 10 | `10_glint_inversion/` (m049-51b) | DONE | Attitude recovery given known omega: 9/10 |
| 10b | `10_glint_inversion/` (m052-59) | CLOSED | Bridge-LC omega selection: chicken-and-egg, all variants fail |
| 11 | `11_casadi_formulation/` (m060-64) | CLOSED | Windowed estimation: basin widens but lo-fi can't discriminate |
| 11b | `11_casadi_formulation/` (m065-68) | CLOSED | Glint grid search: omega funnel works, lo-fi scoring fails |
| 11c | `11_casadi_formulation/` (m069-70) | **DONE** | **Full pipeline: q0=1.94° ω=0.14° in 7.8 min (seed 93)** |
| 11d | `11_casadi_formulation/` (m087-88) | **DONE** | **Speed opts: relaxed tol safe (~3×), SLERP unsafe with NM (4/10 regress)** |
| 11e | `11_casadi_formulation/` (m089-99) | DONE | BRDF cost + multi-phi + NM_TOP=300 sweet spot (rescues seed 6) |
| 11f | `11_casadi_formulation/` (m100-03) | DONE | Multi-phi + full-MSE selection; m102 is pre-surrogate best (3 OK + 3 PARTIAL + 4 FAIL) |
| 12 | `12_brightness_surface/` (m105-12) | CLOSED | IPL surface era: pab-contour approximation capped usefulness; multiple cost variants refuted |
| 12b | `12_brightness_surface/` (m113-21) | **DONE** | **Surrogate + 3-DOF DE breakthrough: m115 gives 10/10 valid solutions; attitude-isoshell validated (m120)** |
| 12c | `12_brightness_surface/` (m122-29) | PARTIAL | Wrapped pipeline + flipped-ω search; m122–m126 hi-fi numbers CORRUPTED (see DATA_INTEGRITY_BUG.md) |

**CLOSED** = approach exhausted, fundamental limitation identified. See `DEAD_ENDS.md` for detail.
**DONE** = validated result, reusable building block.
**PARTIAL** = partial result or bug-corrupted; see linked doc.

---

## 4. Decision Log (Key Pivots Only)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-09 | Abandon global optimizers | Joint basin (~5° × 0.02 dps) impossibly narrow for blind 6D search |
| 2026-02-19 | Abandon cascade filtering | Combinatorial explosion: 100^K pairings |
| 2026-02-20 | Anchor on brightness peaks | dL/dt≈0 provides extra constraint; 3 peaks, not all epochs |
| 2026-02-27 | Scoring is the bottleneck | Graph pipeline generates paths but LC residual can't discriminate (m013/14) |
| 2026-03-10 | L-conservation is the winding filter | 9 OoM gap, robust to 10° error, T adds nothing (m023-25) |
| 2026-03-11 | Band-sweep replaces staircase | Staircase misses 5/12 winding families on longer legs (m019/20) |
| 2026-03-12 | Bridge coverage is THE bottleneck | 10 starts/band insufficient at dt=721s; true omega absent from leg 1 (m026c) |
| 2026-03-12 | Pivot to glint physics | PAB alignment constrains attitude to 1-DOF circles (m034) |
| 2026-03-18 | Two-phase phi sweep validated | 1.77° att, 1.15° omega from oracle omega start (m042b) |
| 2026-03-18 | Attitude given omega demonstrated | 9/10 trajectories, known omega, no attitude oracle (m051b) |
| 2026-03-19 | Bridge-LC omega selection is dead | Chicken-and-egg confirmed across 9 experiments (m059) |
| 2026-03-19 | **Pivot to CasADi + IPOPT** | Multiple shooting + exact gradients + glint constraints. Also: retest L-parameterization (buggy exp invalidated prior conclusion) |
| 2026-03-25 | **Full pipeline SUCCESS** | Fix ±X-only phi scoring + 360 phi bins + full hi-fi scoring + geometric refinement. CasADi not needed — glint geometry sufficient. (m069-70) |

---

## 5. Test Case (All Experiments)

- **Satellite:** Intelsat 901
- **True q0:** axis=[0.6, 0.3, 0.8]/norm, angle=45°
- **True omega0:** [0.5, -0.3, 2.0] deg/s ("fast tumbler")
- **N_OBS:** 500, dt~7.2s, window=3600s
- **Lo-fi:** no shadows. **Hi-fi:** ray-traced shadows
- **Noise:** Gaussian, σ=0.05 mag
- **Realistic omega range:** [0.1, 1.5] deg/s (literature: retired GEO tumbles well below 1°/s)
- **Peaks:** epochs [183, 260, 360] (standard test case)
- **Datasets:** m046 (100 fixed-time), m048 (100 varied-time)

---

## 6. Conventions

- **LC convention:** `observed_lc` / `true_lc` are **magnitudes** (higher = dimmer). Brightness peaks = local minima. Dips = local maxima.
- **Quaternion:** scalar-first (w, x, y, z)
- **Angular velocity:** rad/s in body frame (deg/s in experiment descriptions)
- **R(q):** body→inertial. Code comment says "J2000 to body" — WRONG but consistent.
- **L_inertial:** R(q) @ (I @ omega_body). Conserved.
- **Normal groups:** +X(G13), -X(G0), +Y(G8), -Y(G5), +Z(G7), -Z(G6), +WD(G11), -WD(G2), +ED(G12), -ED(G1). G3/G4/G9/G10 excluded (1.1 m² dish edges, never produce glints).
