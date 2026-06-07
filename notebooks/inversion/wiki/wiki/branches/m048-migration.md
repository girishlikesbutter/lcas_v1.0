---
title: "m048 Migration — from single-window to per-seed random-start observation geometries"
type: branch
sources:
  - "raw/inversion_diagnostics/m048_trajectories/m048_trajectories.npz"
  - "raw/inversion_diagnostics/invert_m046_seed000/result.json"
related:
  - "[[observation-geometry-sources]]"
  - "[[gradient-based-inversion]]"
  - "[[surrogate-model]]"
  - "[[m126_wrapped_pipeline]]"
  - "[[multi-solution-philosophy]]"
created: 2026-04-17
updated: 2026-04-17
confidence: medium
---

# Branch: m048 Migration

## Status: #open-active (Phase 1 complete, Phase 2 pilot effectively complete on 8 seeds, Phase 3 blocked on upstream-design decision)

## The question

Every [[gradient-based-inversion]] cohort claim to date has been "100 trajectories on ONE observation night" ([[observation-geometry-sources]] — m046 is a single fixed 2020-02-05 10:00–11:00 UTC window). The research arc cannot honestly claim generalization until the pipeline runs on per-seed random geometries. `m048_trajectories.npz` was generated precisely for this: 100 trajectories, per-seed `start_et ∈ [08:35, 15:35 UTC]`, 1-hour window each, **phase angle 9.2°–95.2°** (vs m046's 30°–60°). This is Option B from [[DATA_INTEGRITY_BUG]].

## The 3-phase plan (from `project_phase_B_bow_plan.md`)

| Phase | Scope | Compute | Status |
|-------|-------|--------:|--------|
| 1 — Refactor | Thread `start_et` through `setup_experiment`; add `traj_source.load_truth`; retrofit m115, m126, lc_compare; write `invert.py` single-entry driver | 0 | **✅ complete 2026-04-17** |
| 2 — Pilot | Run `invert.py --seed N --traj-source m048` on 3 seeds (low/mid/high phase) to verify pipeline runs clean | ~30 min | queued |
| 3 — Batch | Apply two m115 perf fixes (DE `workers=-1` closure→callable-class; hi-fi `Pool(3)`); run 100 m048 seeds | ~5–15 hr | blocked on Phase 2 |

## What landed in Phase 1

**Library changes (backwards-compatible):**
- `notebooks/inversion/lib/experiment_setup.py` — `setup_experiment()` now accepts `start_et: Optional[float]` and `duration_s: float = 3600.0`. When `start_et` is given, epochs span `[start_et, start_et + duration_s]`; otherwise the legacy `end_time_utc` / config path is used. m046 call sites that pass `end_time_utc='2020-02-05T11:00:00'` are unchanged.
- `notebooks/inversion/lib/traj_source.py` (NEW) — `load_truth(seed, source)` returns a uniform dict `{q0_wxyz, omega0_rad, mag_hifi, observation_times, inertia_tensor, start_et, end_time_utc, duration_s, source, seed}`. `seeds_for_source(source)` returns available seed indices. Valid sources: `'m046'`, `'m048'`.
- `notebooks/inversion/lib/lc_compare.py` — new `--traj-source` CLI arg; truth/ctx now loaded per-seed via `load_truth`.

**Pipeline scripts retrofitted:**
- `m115_surrogate_pipeline.py` — reads `TRAJ_SOURCE` env var; `OUT_BASE` tagged per source (`m115_surrogate_pipeline_m048/` for m048). `run_seed` takes a `truth` dict from `load_truth` instead of `master + seed-index` lookup. m046 keeps the shared-ctx fast path (one `setup_experiment` for all seeds); m048 builds ctx per-seed (~50 s/seed overhead).
- `m126_wrapped_pipeline.py` — reads `TRAJ_SOURCE` and `MICRO126_SEEDS` env vars; `MICRO115_BASE` and `OUT_BASE` are source-tagged. `build_seed_context` threads `start_et`/`end_time_utc` through `setup_experiment`.

**Single-entry driver (NEW):**
- `notebooks/inversion/invert.py` — `python3 invert.py --seed N --traj-source {m046|m048}`. Chains m115 → m126 → wrappedbest refresh → lc_compare. Subprocesses inherit `TRAJ_SOURCE + MICRO{115,126}_SEEDS` from the driver's environment. Flags: `--skip-m115`, `--skip-m126`, `--skip-lc-compare`. Summary JSON at `data/results/inversion_diagnostics/invert_{source}_seed{NNN}/result.json`.

## Verified

- Back-compat: `setup_experiment(end_time_utc='2020-02-05T11:00:00', skip_true_lc=True)` produces `observation_times` identical to `m046_trajectories.npz['observation_times']` (bitwise).
- m048 path: `setup_experiment(start_et=634173431.91116, duration_s=3600.0)` matches `m048/per_trajectory/traj_seed000.npz` `dt_sampling` to <1e-9 s.
- End-to-end: `python3 notebooks/inversion/invert.py --seed 0 --traj-source m046 --skip-m115 --skip-m126 --skip-lc-compare` reconstructs seed 0's wrappedbest dir from `m126_wrapped/seed_000/hifi_ckpt.npz` in 5 ms and reports `q0=0.54° w_dir=0.02° w_mag=+0.01% hifi=0.00367 [OK]` — bitwise-identical to the 2026-04-17 cohort audit.

## Known blockers for Phase 2

- **`m115.load_omega_candidates` has no m048 fallback.** It looks in `m103_hybrid/seed_NNN/geo_ckpt.npz` (primary) or `m102_fullmse/seed_NNN/result.npz` (fallback). Both of those upstream experiments ran on m046 only. For any m048 seed lacking these, m115 exits with `error: 'no_omega_data'`. Phase 2 needs to either (a) harvest m103-style geo_ckpts for the 3 pilot seeds, (b) stub an oracle-omega path for pilot triage, or (c) retrofit a surrogate-DE-over-omega stage upstream of m115's existing surrogate-DE-over-q0 stage.

## Phase-angle risks (honest, not exhaustive)

From `project_phase_B_bow_plan.md`:
1. **High-phase (>80°)** seeds hit the [[dark-mag-saturation]] plateau harder — more of the LC spent near-terminator means less gradient for DE/polish. Expect more FAIL cases.
2. **Low-phase (<20°)** opposition regime not characterised on surrogate vs hi-fi. Worth a surrogate-fidelity audit on 5–10 m048 seeds.
3. **Bright ±X peak availability** shifts. [[m096_exp1_oracle_grid]] Stage 1 census said 87% of m046 seeds lacked bright ±X peaks; distribution on m048 unknown.
4. **Narrow-basin seeds may become unfindable.** [[m121_basin_width_metric]] measured ω-dir basin width <0.1° on m046. At higher phase with stronger shadowing, basins might be narrower — current 50k DE population could miss them.

## Phase 2 outcome (2026-04-17)

The omega-candidate gap was resolved by adding a `MICRO103_SKIP_HIFI=1` env var to `m103_hybrid.py` (emits `geo_ckpt.npz` after Step 4 without running Step 5). `invert.py` now chains m103 harvest → m115 → m126 → lc_compare when no prior geo_ckpt exists for a seed.

8 m048 seeds have been run across three sub-runs spanning 15°–87° median phase. Full per-seed table in [[phase_B_m048_cohort]].

| sub-run | seeds | phase range | tally |
|---|---|---|---|
| Original pilot (prior session) | 024, 091, 028 | 15, 52, 87° | 1 OK + 1 PARTIAL + 1 upstream-FAIL (geo hang) |
| Gap probe (this session) | 081, 069 | 46, 68° | 1 OK + 1 killed-before-downstream (NM truth missing) |
| Descending-phase (this session) | 090, 049, 023 | 51, 39, 30° | 2 OK + 1 FAIL (constraint-poor) |

**Cumulative 8-seed tally:** 4 OK + 1 PARTIAL + 1 downstream-FAIL + 2 upstream-FAILs.

### Two failure modes disentangled

1. **High-phase failure (seeds 69 @ 68°, 28 @ 87°)**: m103's alignment cost surface flattens. NM cannot localise truth's basin — best ω-error in the NM top-20 was 28.8° (seed 69) and 30.8° (seed 28) respectively. This is NOT a constraint shortage — both seeds had ≥10 spec peaks. The failure mechanism is specific to alignment-cost geometry at high phase. Seed 28's Pool(24) geo hang is a secondary symptom of the flat cost surface (L-BFGS-B burns unbounded function evals per iter). See `[[alignment-cost]]#high-phase-flatness`.

2. **Constraint-poor failure (seed 23 @ 30°)**: Only 2 specular peaks → 1 alignment constraint after anchor. With a single constraint, alignment cost is trivially satisfiable — NM top-20 has all `geo_cost` values in the 1e-23 to 1e-20 range, truth lives at rank #8 (q0=172.8°, ω_err=5.1°) but isn't selected. Truth is findable, but unselectable. See [[constraint-poor-regime]].

### Phase-angle operating range

Synthesising m046 (5 OK / 4 PARTIAL / 2 FAIL on 30-60° shared window) with m048 Phase-B (8 new seeds on 15-87°): **reliable operating band is roughly 38-55° phase with ≥ 4 spec peaks**. The m046 cohort's implicit success rate was biased upward by the 50-60° constraint-rich sub-band (median 10 spec peaks per [[phase-angle-operating-range]] cross-tabulation).

## Phase 3 decision point

Before launching a 100-seed m048 batch, a design decision is required:

- **Quick path**: patch m103 (robust geo timeout + enrich checkpoints per the 2026-04-17 checkpoint audit) and accept 30-40% upstream-FAIL rate at the edges of phase-angle distribution. ~3 hr infrastructure work + ~24 hr batch compute.
- **Patient path**: replace m103's alignment-cost grid + NM + multi-phi + geo with a single-stage 6-DOF surrogate DE over `(q0, ω)`. Validate against the m046 11-seed cohort and the 3 Phase-B FAIL seeds (23, 28, 69) before the 100-seed batch. See [[upstream-redesign-6dof-surrogate-de]].

Current recommendation (not yet actioned): patient path. The two failure modes exposed by the Phase-B cohort are both alignment-cost pathologies, and the surrogate-LC cost (already validated in m115's inner DE) would sidestep both by construction.

## Next action

User decision on Phase 3 strategy. Pending that, no further m048 experiments should run — the 8-seed cohort has adequately exposed the failure modes, and running more seeds on the current pipeline would just accumulate more of the same upstream failures.
