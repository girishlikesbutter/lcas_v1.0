---
title: "m141 — Seed 6 inversion pipeline rerun under corrected forward model: m103 ω-selection FAILS"
type: experiment
sources:
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/geo_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/result.json"
  - "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed006.npz"
  - "data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed006_buggy.npz"
related:
  - "[[m139_convention_bug_fix]]"
  - "[[m140_post_fix_lc_delta]]"
  - "[[m115_surrogate_pipeline]]"
  - "[[m115_de_bridging_radius]]"
  - "[[m134_pipeline_test_q0polish]]"
  - "[[m135_alignment_cost_forensics_constrained_anchor]]"
  - "[[surrogate-rerank]]"
  - "[[alignment-cost]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# m141 — Seed 6 inversion pipeline rerun under corrected forward model

#first-post-fix-pipeline-test #m103-omega-selection-fail

## TL;DR

After regenerating m048 seed 6 truth with the fixed propagator (53 sec wall via `MICRO48_WORKER_SEED=6 m048_generate_trajectories_v2.py`), ran the canonical inversion pipeline (`invert.py --seed 6 --traj-source m048` = m103 → m115 → m126 → wrappedbest). Killed mid-m115 once the upstream verdict was clear: **m103's geo_cost ranking placed all top-4 ω candidates 52–57° from truth; m115 cannot bridge that gap.** Pool inspection revealed the truth-near ω (w_err 16.86°) DOES exist in m103's 26-candidate pool, but at rank 9/26 — m103's ranking is **anti-correlated with truth on this seed** (m135-style failure mode now confirmed against physical truth, not just buggy truth). No candidate in the pool is jointly truth-near in both q0 and ω, so even surrogate-MSE re-rank may not produce a Band-A solution without further upstream work. Seed 6 was previously classified "OK" / "previously-solved" under the buggy forward model (m098, m126, m131 cohort wins). **Under corrected truth, seed 6 is upstream-FAIL.** This is the first seed-level evidence that the convention bug may have been *helping* m103's noisy ranking land on truth-near ω by accident, and that the m103 → m115 architecture inherits the m135 alignment-cost-anti-truth pathology even after the bug fix.

## What

This is the first end-to-end test of the inversion pipeline under physically correct truth, on a seed previously classified as "OK" under the buggy forward model. Goal: see whether the post-m139 fix automatically restores inversion success, OR whether the m135 alignment-cost-anti-truth thread (and the m133/m134/m135 surrogate-rerank fix) is genuinely the bottleneck independent of the convention bug.

Seed 6 chosen because:
- It's the canonical Step-3 validation seed in [[m139_convention_bug_fix]] and the subject of [[m140_post_fix_lc_delta]] (LC delta already quantified at ρ=43.6).
- Per `MEMORY.md → project_micro98_99_findings.md`: previously rescued from PARTIAL → OK by NM_TOP=300 + 2000-dir grid (m098-099).
- Per the m126 wrapped-pipeline log: 87% hi-fi MSE improvement under keep_better — top-tier baseline cohort win.
- Phase angle 56–70° (mid-band, not constraint-poor): not a known failure mode.

## How

1. **Backup buggy artefacts** as `_buggy` siblings (preserves audit trail per CURRENT_STATE.md guidance):
   - `traj_seed006.npz` → `traj_seed006_buggy.npz`
   - `m103_hybrid_m048/seed_006/` → `seed_006_buggy/`
   - `m115_surrogate_pipeline_m048/seed_006/` → `seed_006_buggy/`
   - `m126_wrapped_m048/seed_006/` → `seed_006_buggy/`
   - `wrappedbest_m048_seed006/` → `wrappedbest_m048_seed006_buggy/`
   - `invert_m048_seed006/` → `invert_m048_seed006_buggy/`
2. **Regenerate seed-6 truth** via existing single-seed worker mode: `MICRO48_WORKER_SEED=6 python3 notebooks/inversion/09_glint_analysis/m048_generate_trajectories_v2.py`. 53 sec wall. Verified mag_hifi matches m140's `mag_fixed` at machine precision (RMS 4.7e-15 mag).
3. **Run pipeline:** `python3 notebooks/inversion/invert.py --seed 6 --traj-source m048`. Background dispatch.
4. **Kill mid-m115** once the upstream verdict was unambiguous (~15 min into the run, m115 was on DE start 9 of 30).
5. **Pool analysis:** loaded `m103_hybrid_m048/seed_006/geo_ckpt.npz` (saved by m103 Step 4 before m115 began), inspected the full 26-candidate ω pool's truth-distance distribution against m103's geo_cost ranking.

## m103 results

**m103 Step 1–4 completed.** Setup looked healthy (21 peaks, 8 spec, 7 constraints, mid-band phase, anchor mag 7.46). The grid + NM + multi-phi + geo chain produced a 26-candidate ω pool with the geo_ckpt structure documented in [[m103_hybrid]].

**m115 received the top-3 candidates** (ranking determined by m115's `load_omega_candidates`, which historically reads from the geo_ckpt by geo_cost ascending):

| candidate | w_dir_err | true_w_mag |
|:---:|:---:|:---:|
| 0 | 56.99° | 72.4% (28% under) |
| 1 | 53.56° | 65.4% (35% under) |
| 2 | 54.57° | 65.9% (34% under) |

All three are 3.5× outside m115's reliable bridging radius (~3–5° per [[m134_pipeline_test_q0polish]]) and outside even its marginal radius (~15°). Across DE starts 1–9 (out of 30), every start gave q0_err 88–162°, MSE 3.4–4.2, n_below_10deg=0/10. The pipeline was killed at this point.

## The full m103 pool — anti-correlation with truth

Sorted by m103's geo_cost ranking:

| m103 rank | geo_cost | w_dir_err | q0_ref_err |
|:---:|:---:|:---:|:---:|
| 1 | 4.87e-03 | 56.99° | 171.58° |
| 2 | 7.61e-03 | 53.56° | 174.87° |
| 3 | 8.44e-03 | 54.57° | 145.06° |
| 4 | 9.96e-03 | 52.85° | 161.40° |
| 5 | 3.17e-02 | 67.28° | 153.85° |
| 6 | 3.42e-02 | 63.97° | 64.78° |
| 7 | 5.88e-02 | 67.74° | 129.29° |
| 8 | 7.75e-02 | 74.13° | 126.50° |
| **9** | **8.39e-02** | **16.86°** | **129.58°** |
| 10 | 9.12e-02 | 81.31° | 137.19° |
| ... | ... | ... | ... |
| 21 | 5.27e-01 | 85.65° | 31.20° |
| 25 | 1.33e+00 | 63.19° | 39.47° |

**The truth-near ω (rank 9, w_err 16.86°) is buried by m103's geo_cost ranking under 8 candidates with w_err 53–74°.** Rank 1's geo_cost is 17× lower than rank 9's, but rank 9 is 3.4× closer to truth in ω-direction. **m103's ranking is anti-correlated with ω truth-distance on this seed.**

q0_ref_err follows a different pattern: candidates at ranks 6, 11, 21, 25 have q0_ref_err 30–65° — *closer* in q0 but *further* in ω than rank 9. **No candidate in the 26-pool is jointly truth-near in both q0 and ω** (closest joint candidate would be rank 6: w_err 63.97°, q0_err 64.78°). The phi-sweep / multi-phi step that explores q0 around each ω did not surface a truth-q0 anchor for the truth-near ω at rank 9.

## Why this matters

### 1. The convention bug may have been masking m103 ranking failures

Pre-fix, seed 6 was reliably "OK" — m126 reported 87% improvement from baseline; m098-099 rescued it from PARTIAL. Under post-fix correct truth, seed 6 is upstream-FAIL. The simplest explanation: **m103's geo_cost ranking on the buggy LC happened to place a buggy-truth-near ω at the top of the pool by accident**. The buggy LC's bright-peak displacement (m140: 5–8 mag on bright epochs) shifted the alignment-cost surface in a way that inadvertently aligned m103's noisy ranking with buggy-truth's ω. On the correct LC, that coincidence vanishes.

This is a candidate explanation for why the m115/m126 cohort had ~50% Band-A on baseline seeds — the bug's noise-floor shift may have boosted m103's ω-selection accidentally on a fraction of seeds. If true, the corrected forward model is **harder for m103, not easier**, even though it's physically right.

### 2. m135 alignment-cost-anti-truth pattern survives the convention fix

[[m135_alignment_cost_forensics_constrained_anchor]] documented that on 5 random-cohort failure seeds, alignment-cost favoured candidates 8–82° from truth over the truth itself. That finding was made against buggy truth. m141 confirms it on corrected truth: rank-9 truth-near candidate has 17× *higher* geo_cost than rank-1. The pathology is independent of the convention bug.

### 3. Surrogate-MSE re-rank (m133/m134) becomes the immediate next lever

The m133 thread (re-rank m103's pool by surrogate full-LC MSE instead of geo_cost) is now the unblocked next step. With seed 6's pool min ω-error at 16.86° (rank 9), surrogate re-rank could plausibly:
- **Best case:** rank 9 promoted to rank 1, m115 DE bridges 16.86° → BAND-B/C result (per m134 seed 67 precedent).
- **Realistic case:** rank 9 promoted but m115 cannot bridge from 16.86° (right at the edge of m134's empirical 15° marginal radius), no rescue.
- **Worst case:** even surrogate re-rank doesn't promote rank 9 to top-K, no rescue.

**This unlocks Play 1 from the strategic reframe** ([[upstream-redesign-6dof-surrogate-de]] / m133 surrogate-rerank thread), now with a corrected forward model.

### 4. The "previously-solved seed cohort" is not a stable benchmark

Every result in m098, m115, m126, m131, m132 cited seed 6 as a baseline win. Those wins were against buggy truth. Whether seed 6 is solvable under correct truth depends entirely on whether m103-pool re-ranking + m115-DE bridging can recover the rank-9 ω OR whether a wholly-replaced upstream stage (Play 3: 6-DOF surrogate DE over (q0, ω)) is required.

## Numbers

| metric | value |
|---|---|
| Truth regen wall | 53 sec (single-seed `MICRO48_WORKER_SEED=6` mode) |
| Mag_hifi vs m140 mag_fixed RMS | 4.70e-15 mag (machine precision) |
| Hi-fi peak count (buggy) | 32 |
| Hi-fi peak count (fixed) | 30 |
| Phase angle range | 56.0° – 70.8° |
| m103 NM-300 wall | ~5 min (estimated from output timing) |
| m103 Step 4 (Geo) wall | ≤ 8 min (timeout 480s, completed before m115 launch) |
| m115 wall (killed) | ~5 min (9/30 DE starts done) |
| Total pipeline wall (killed) | ~15 min |

| pool stat | value |
|---|---|
| n_candidates | 26 |
| m103 rank-1 w_dir_err | 56.99° |
| m103 rank-1 q0_ref_err | 171.58° |
| Pool min w_dir_err | 16.86° (at m103-rank 9) |
| Pool min q0_ref_err | 31.20° (at m103-rank 21) |
| Pool joint min (q0_err + w_err) | 64.78° + 63.97° = 128.75° (rank 6) |

## Artefacts

- `data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed006.npz` (post-fix correct truth)
- `data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed006_buggy.npz` (preserved for audit)
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/{geo_ckpt.npz, result.json, pipeline.log, grid_top500_ckpt.npz, lofi_ckpt.npz, nm_prededup_ckpt.npz, multi_phi_ckpt.npz, phi_sweeps_ckpt.npz, result.npz}` (post-fix m103 outputs)
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_006_buggy/` (pre-fix m103 outputs, preserved)
- Same `_buggy` preservation pattern for m115_surrogate_pipeline_m048, m126_wrapped_m048, wrappedbest_m048_seed006, invert_m048_seed006.

## Out of scope here (next-session priorities)

1. **Surrogate-MSE re-rank on the saved m103 pool.** Run the m133 3-cost-union or m138 surrogate-rerank script over `geo_ckpt.npz`'s 26 candidates. ~5–10 min wall, no fresh m103 rerun needed. Headline: does re-rank promote the rank-9 truth-near ω to top-K?
2. **If re-rank promotes the truth-near ω:** rerun m115 from that ω only (3-DOF DE), see whether 16.86° is bridgeable. m134 precedent says 15° is marginal.
3. **If re-rank fails:** decide whether to invest in Play 3 (6-DOF surrogate DE upstream, replacing m103 entirely) or to expand the m103 pool size and re-test.
4. **Validate generality:** repeat on a second seed (e.g. seed 91 from the strategic-reframe Play-1 cohort, or seed 47 from the m138 failure-cohort) to confirm whether m141's "ranking-anti-truth survives the bug fix" finding is universal.

## Cross-references

- [[m139_convention_bug_fix]] — the bug fix that made this experiment possible.
- [[m140_post_fix_lc_delta]] — established that the LC delta is large (ρ=43.6 on this seed), motivating the regeneration.
- [[m135_alignment_cost_forensics_constrained_anchor]] — pre-fix evidence of alignment-cost-anti-truth; m141 confirms it survives the fix.
- [[m134_pipeline_test_q0polish]] — established m115 ω-bridging radius of ~3-5° reliable, ~15° marginal.
- [[m115_de_bridging_radius]] — concept page summarising the empirical radius.
- [[surrogate-rerank]] — the unblocked next lever.
- [[alignment-cost]] — the substrate now confirmed anti-truth post-fix.
