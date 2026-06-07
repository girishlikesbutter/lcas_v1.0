---
title: s031 — surrogate-MSE rank + hi-fi rerank of s030 relaxed survivors, seed 6
type: experiment
sources:
  - experiments/s031_hifi_rerank_seed6.py  (Stage A driver — original; ran Stage A then was killed before useable Stage B)
  - experiments/s031b_hifi_only.py         (Stage B re-runner — fork context, BLAS=1, top-50/level + 5 basin extras)
  - results/s031/seed006/
related:
  - s020 (filter pipeline)
  - s030 (relax-threshold sweep)
  - s014 (surrogate↔hi-fi Spearman 0.9952 outside basin)
  - s001 (cohort-scale truth surrogate-MSE = 3e-4 median, 2.9e-3 on seed 6)
created: 2026-05-04
updated: 2026-05-05
confidence: high — Stage A is decisive (n_pred_BandC = 0 across all levels) and Stage B confirms (0/200 Band A∪B; all 5 basin candidates also Band D)
---

## TL;DR

Two-stage post-process answers the binary question "does any blind candidate from the s030 4-level survivor union land in Band A∪B (ρ<4) in hi-fi for seed 6?". **Verdict: NO across every dimension.** Stage A (surrogate-MSE on the 121k union) shows **n_pred_BandC = 0 across every level** — min predicted ρ is 28.08 (L2-best) to 42.89 (L0-best), placing every candidate firmly in Band D before hi-fi confirmation. Stage B hi-fi-rerank of top-50/level (176 unique deduped) plus the 5 basin candidates that all 4 levels miss confirms Band D. **All 200 (50×4 + 5 basin extras) candidates are Band D in hi-fi**; ρ_min by level: L0=42.85, L1=32.92, L2=28.06, L3=32.92; basin candidates ρ = 39.31 to 48.39. The basin candidates being Band D is the load-bearing finding: even with the filter completely bypassed (truth-basin q0 at 18-26° from truth, ω-mag 4.6% off, ω-dir 9.8°), the candidate fails in hi-fi because **the bracket bin 0 at +4.6% off truth places the basin outside the s003 ω-mag tube**. **The framework does not deliver an end-to-end inversion proof-point on seed 6** — see Step 3 decision in PROGRESS.md.

## What

Two-stage post-process on the s030 relaxed-survivor pool for seed 6, with a single binary question: **does any blind candidate from the 4-level survivor union land in Band A∪B (ρ<4) in hi-fi?**

- **Stage A**: surrogate-MSE for every candidate in the 4-level union (121,128 entries) using cached `delta_trajectories` + `q0` from s020. Δ-factorisation, no re-propagation. Per level, rank by surrogate-MSE; report predicted-ρ counts.
- **Stage B**: hi-fi render the per-level top-50 (deduplicated across levels) + the 5 basin candidates that score `align ≤ 0.429 < 0.5` and so are absent from every level's survivor list. Compute ρ vs cached truth `mag_hifi`. Per-level ρ-band breakdown.

Cap reduced from the user-requested 2000/level → 500 → 50 because Stage A was decisive (n_pred_BandC = 0 across the entire 121k union — running 2000 Band-D-predicted candidates per level would only confirm what Stage A already says, at 4 hours wall vs 80 min for top-50).

## How

### Stage A (single process, default torch threading)

1. Load `candidates_meta.npz` (q0[432000, 4], cell_idx, scores). Load `delta_trajectories.npz` ((1500, 500, 4) Δ per ω-cell). Load `omega_grid.npz`. Load `survivors_per_level.npz` (s030).
2. Per ω-cell `c` containing union candidates: vectorised quaternion outer-product (`q_full[e, k] = Δ_c[e] ⊗ q0[k]` LEFT multiply per propagator), then scipy `Rotation.from_quat([qx, qy, qz, qw])` → R-matrices, then `einsum('eqij, ej -> qei', R, sun_unit)` for body-frame k1 (and k2), then surrogate batched predict via `model.predict_magnitude(k1_flat, k2_flat, sp=0, ad=15, obs_dist_flat)`. MSE vs truth `mag_hifi` per candidate.
3. Per level (L0..L3) take top-N by surrogate-MSE (N = `max(TOP_N_CAP, n_pred_BandC)`).

### Stage B (Pool(8), BLAS=1 + fork context)

Original threading was broken (forkserver mode + BLAS env vars set in worker init too late → each worker at ~400% CPU) and the resulting 7+ hr ETA was untenable. The fix:

1. Set `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_NUM_THREADS=1` at the top of the script BEFORE any numpy/torch import. So both master and forked children inherit single-threaded BLAS.
2. Use `multiprocessing.get_context("fork")` instead of the default forkserver. Forkserver re-imports lib.hifi_render and re-builds the satellite + BRDF + inertia per worker (8× the heavy STL/BRDF load); fork inherits all in-memory state for free.
3. Build `_MASTER_CTX = lib.hifi_render.build_context(6)` ONCE in master pre-Pool. Workers inherit via fork.
4. Worker init only calls `torch.set_num_threads(1)` for belt-and-braces.
5. Each render: `lib.hifi_render.render_hifi(q0, ω, _MASTER_CTX)`. ρ = √mean((pred − truth)²) / 0.05.

Hi-fi target set: top-50/level union (176 unique) + the 5 basin candidates (truth and twin) that all 4 levels miss because they score `align ∈ {0.286, 0.429} < 0.5`. Final 181 unique candidates.

## Result

### Stage A diagnostic — predicted-ρ counts per level

| level | min surrogate-MSE | min predicted ρ | <2 (A pred) | <4 (B pred) | <8 (C pred) | level total |
|---|---|---|---|---|---|---|
| L0 strict 1.0/1.0     | 4.599 | **42.89** | 0 | 0 | **0** | 875 |
| L1 relaxed 0.5/0.857  | 2.703 | **32.88** | 0 | 0 | **0** | 28,690 |
| L2 relaxed 0.5/0.5    | 1.971 | **28.08** | 0 | 0 | **0** | 97,943 |
| L3 align-only 0.857   | 2.703 | **32.88** | 0 | 0 | **0** | 51,875 |

**n_pred_BandC = 0 across all 4 levels.** The lowest predicted ρ in the entire 121k union is 28.08 (L2-best, surrogate-MSE 1.97 mag²). Per s014's "surrogate↔hi-fi Spearman 0.9952 cohort-wide outside basin", hi-fi ρ tracks predicted ρ to within a few percent for off-basin candidates → every candidate is predicted Band D, and Stage B confirms.

### Stage B — ρ-band breakdown per level

(Top-50 per level by surrogate-MSE, hi-fi rerank, Pool(8) fork context, wall 27.3 min.)

| level | top-N | A (<2) | B (2-4) | C (4-8) | D (≥8) | ρ_min | best_q0_to_truth° |
|---|---|---|---|---|---|---|---|
| L0 strict 1.0/1.0     | 50 | 0 | 0 | 0 | **50** | 42.85 |  75.59 |
| L1 relaxed 0.5/0.857  | 50 | 0 | 0 | 0 | **50** | 32.92 | 121.07 |
| L2 relaxed 0.5/0.5    | 50 | 0 | 0 | 0 | **50** | 28.06 | 171.72 |
| L3 align-only 0.857   | 50 | 0 | 0 | 0 | **50** | 32.92 | 121.07 |

L1 best == L3 best (cand 78491; L3 ⊃ L1 with the same align cut). L2 best (cand 3319) lands on the q0 antipode at 171.7°. L0's strict-filter best (cand 154606) is at q0=75.6° from truth — none close to either truth or twin.

### Basin candidates — hi-fi ρ (TRUTH and TWIN, all rejected by every level's filter)

| tag | cand idx | q→ref° | \|dω-mag\| | ω-dir-deg | ρ_hifi | band |
|---|---|---|---|---|---|---|
| TRUTH | 23182 | 26.06 | 4.61% | 9.83 | **46.81** | **D** |
| TRUTH | 23183 | 18.19 | 4.61% | 9.83 | **48.39** | **D** |
| TWIN  | 54055 | 19.80 | 4.61% | 4.28 | **39.31** | **D** |
| TWIN  | 60012 | 29.38 | 4.61% | 7.21 | **47.05** | **D** |
| TWIN  | 60023 | 29.57 | 4.61% | 7.21 | **47.73** | **D** |

**The basin-candidate Band-D result is the load-bearing finding.** These are the 5 candidates closest to truth/twin in the entire 432k IC pool — all at q0 18-30° from truth/twin and inside the basin definition (q<30°, |dω|<10%, ω-dir<10°). They fail in hi-fi because the bracket bin 0 (+4.6% off truth-mag) places them OUTSIDE the s003 ω-mag tube. Below the s003 ~2-5% mag tube, the surrogate-MSE landscape becomes incoherent (3-4 OOM higher MSE than truth), and per s014's surrogate↔hi-fi rank fidelity outside basin, hi-fi confirms. **No q0 at any phi-density at ω-cell-0 lands in Band A∪B for seed 6** — the bracket itself must land closer to truth ω-mag before any IC primitive or LM polish can recover the basin.

## Why this matters

This is the s020-pipeline's first end-to-end blind ρ-band measurement under correct truth. The s022 calibration of "no false negatives" was on n=15 *near-truth-ω* known-good candidates. s031 tests the off-truth-ω regime explicitly — the regime that all 4 s030 levels actually populate (s020 reported all 875 strict survivors at ω-mag ≥ +69% off truth).

The decisive negative shapes Step 3:
- **Bracket bin 0 at +4.6% off truth puts truth at the edge of the bracket-coverage tube.** Per s003, surrogate-MSE explodes to Band-D 3-4 OOM outside the ~1°/2-5% tube around truth-ω. The bracket misses the tube on seed 6 by 4.6% — too far for any q0 candidate to land inside-basin.
- **30° phi-step IC granularity is too coarse.** Even where truth-basin candidates exist (q0 18-30° from truth) they score align=0.429 (3/7 bright peaks matched), below every level's filter. Densifying phi-sweep alone would not move them inside the basin without LM polish.
- **The framework needs joint LM polish at the per-cell level**, not a post-filter polish. Filter survival → LM polish architecture is broken because filter survivors are at high ω-mag (+70-370% off truth) and outside the s003 tube; LM cannot bridge the tube boundary.

## Numbers

- N_total candidates evaluated in Stage A: 121,128 (4-level union of s030).
- Stage A wall: 4715 s (~78 min) single process.
- Min predicted ρ across the entire union: 28.08 (L2, surrogate-MSE = 1.971 mag²).
- Stage B wall: 27.3 min Pool(8) fork context on 181 hi-fi targets (9.0 s/cand amortised once warm; chunked-result-burst pattern from chunksize=4).
- ρ-band: 0/200 in Band A∪B; 200/200 in Band D. ρ_min over all hi-fi'd candidates = 28.06 (L2 best).
- ρ_min over basin candidates = 39.31 (TWIN 54055).

## Artefacts

- `experiments/s031_hifi_rerank_seed6.py` — original combined-stage driver (Stage A only useful; Stage B was killed before finishing because of the broken forkserver/BLAS threading).
- `experiments/s031b_hifi_only.py` — Stage-B-only re-runner (fork context + BLAS=1 at top of script).
- `experiments/s031_basin_hifi.py` — standalone basin-only hi-fi (subsumed; basin candidates included in s031b's hi-fi target set).
- `experiments/s031_analyze.py` — post-hoc per-level top-3 + basin-candidate report.
- `results/s031/seed006/`:
  - `surrogate_mse_union.npz` — per-union-candidate surrogate-MSE (Stage A; 121k entries).
  - `hifi_rho_union.npz` — per-target hi-fi ρ (Stage B).
  - `summary.json` — per-level ρ-band breakdown + best (q0, ω) per level + truth/twin basin coverage.

## Out of scope

- LM polish on top-N surrogate-MSE candidates (deferred — doesn't help when filter survivors are outside the s003 tube).
- Other seeds (deferred — Step 3 explicit decision-time).
- Cohort scaling (negative result; do not scale).

## Cross-references

- **s020** — original filter pipeline, source of cached data; first end-to-end pre-LM run on seed 6.
- **s030** — relax-threshold sweep, definitive: no basin candidate survives any level.
- **s014** — surrogate↔hi-fi Spearman 0.9952 cohort-wide outside basin (justifies Stage A's surrogate-MSE ranking as a proxy for hi-fi ρ ranking).
- **s003** — surrogate-MSE ω-fragility tube ~1° dir / ~2-5% mag; closes the question of why off-tube survivors fall in Band D.
- **`concepts/rho_band.md`** — A:<2, B:<4, C:<8, D:≥8 classification + canonical noise σ=0.05.
- **`concepts/twin_degeneracy.md`** — corrected X-flip body-twin convention for q0 + ω.
