---
title: s020 — bracket-augmented filter pipeline (NO LM polish), seed-6 diagnostic
type: experiment
sources:
  - experiments/s020_seed_pipeline.py
  - experiments/s020_seed_animate.py
  - results/s020/seed006/
related:
  - s019 (bracket prior)
  - s019b (ω-dir basin width)
  - s018a + s018b (tier classifier)
  - s018c (phi-sweep IC)
  - s021..s027 (filter framework)
created: 2026-05-04
updated: 2026-05-04
confidence: high — first end-to-end run on 432k candidates; clear positive (architecture works) AND clear negative (truth-basin filtered out by strict thresholds)
---

## TL;DR

End-to-end filter pipeline run on seed 6: LS-bracket → Fibonacci ω-dir grid → phi-sweep q0 IC pool → Δ-factorised propagation → geo cost → alignment cost → categorisation. **No LM polish.** 432,000 candidates × 1500 ω-cells. Wall: 61 min Pool(8). **Headline finding: filter as configured does NOT recover truth or twin for seed 6** — 875 survivors at strict 1.0/1.0 thresholds are all multi-solution (closest q0 to truth = 35.84°, all at ω-mag ≥ +69% off truth). HOWEVER, **2 true truth-basin and 3 true twin-basin candidates exist in the pool** (q<30°, |ω-mag|<10%, ω-dir<10°) at q-distances 18-26° from truth — they score geo=0/align=0.43 and are filtered out by the strict thresholds. **Diagnosis: 5° geo tolerance + binary scoring (only 2 spec events) is too strict for the 30°-phi-step IC granularity; truth-basin candidates miss spec events even at small q0 deviation.** Pipeline mechanics validated; filter calibration needs work.

## What

s020 is the first session in which all four prerequisites are simultaneously available:

1. **ω-mag prior** (s019): top-5 LS peaks include truth ω-mag for seed 6 within 4.61%.
2. **ω-dir basin** (s019b): 3-5° wide at pinned truth-mag → 300 Fibonacci cells (~5° spacing) cover the basin. Nearest cell to truth-dir on seed 6: 3.72°.
3. **Filter framework** (s021–s027): alignment + geo cost reject 99%+ of structurally wrong candidates with no false negatives against known-good. Truth-calibrated thresholds.
4. **Phi-sweep q0 IC primitive** (s018c) anchored on the s018a/s018b tier classifier: 4 classifiable peaks × ~6 face shortlist × 12 phi = 288 q0 candidates per ω-cell.

The pipeline is NOT meant to fully solve inversion — it stops before LM polish. The question it answers: how rich is the survivor set produced by filter alone, and where does truth land in it?

## How

Per seed, in cell-major order:

1. **Bracket** (`run_bracket`): Lomb-Scargle on cached `mag_hifi`. Take top-5 LS peaks by power as ω-mag candidates. Truth is most often within 5% of one of these peaks (s019 cohort coverage: 98/100 within 5%).

2. **ω-dir grid**: 300 Fibonacci sphere points at ~5° spacing on S². Total ω-cells: 5 mag × 300 dir = 1500.

3. **q_target pool**: For each tier-classifiable bright peak, take the tier-shortlisted face normals and phi-sweep 12 steps over 360°. ω-independent — built once.

4. **Per ω-cell** (`_process_cell`, run via Pool(8)):
   a. Δ propagation: `propagate_attitude(identity, ω, t)` — one ODE solve per cell.
   b. q0 derivation: `q0[j] = Φ(t_peak)⁻¹ ⊗ q_target[j]` (LEFT multiply per propagator).
   c. Body-frame transform per (ω-cell, candidate, epoch) via factorised quaternion outer product `q_full[e, q] = Δ(e) ⊗ q0[q]`.
   d. **Geo cost** (vectorised over candidates × spec epochs): rotate PAB_inertial into body frame using R(q_full); fraction of spec events where a tier-allowed face is within 5° of body-PAB.
   e. **Alignment cost** on EVERY candidate (split into geo-PASS and geo-FAIL groups for timing): batched surrogate predict per group → for each candidate, count truth bright peaks where the candidate has a local-min mag below threshold within ±3 epochs.
   f. Survivor LCs cached (passes both filters); reject LCs discarded (regenerable from stored q0/ω).

5. **Truth-calibrated thresholds** (per s023 framework): truth's own scores set the per-seed thresholds. For seed 6: truth_geo = 1.0, truth_align = 1.0.

6. **Categorisation**:
   - `cat_both` — passed geo AND align (filter survivors)
   - `cat_geo_only` — passed geo, failed align
   - `cat_align_only` — passed align, failed geo
   - `cat_neither` — rejected by both

7. **Survivor diagnostics**: q0-geodesic to truth, q0-geodesic to body-twin (X-flip + ω-transform per corrected concept page).

## Result

### Filter survival rate

| category | count | % of total |
|---|---|---|
| passed_both | **875** | 0.20% |
| passed_geo_only | 5,770 | 1.34% |
| passed_align_only | 18,569 | 4.30% |
| rejected_both | 406,786 | **94.16%** |

Filter intersection rejects **94.16%** of candidates — broadly consistent with s023's 99% rejection rate on random panels (geometrically anchored ICs survive at higher rate).

### Survivor diagnostics (q-only metric)

| metric | value | note |
|---|---|---|
| n_survivors | 875 | passed both filters at 1.0/1.0 threshold |
| min q0_geo to truth | 35.84° | not within 30° basin |
| median q0_geo to truth | 132.39° | bulk far from truth |
| min q0_geo to twin (q-only, no ω-transform) | 6.22° | misleading — see proper twin below |
| ω-mag bin distribution | bin 0: 0; bin 1: 16; bin 2: 104; bin 3: 335; bin 4: 420 | NO survivors at truth ω-mag bin |

### Survivor diagnostics (proper twin convention: q + ω-transform)

| class | definition | count |
|---|---|---|
| TRUE truth-basin | q<30°, ∣ω-mag∣<10%, ω-dir<10° from truth | **0** |
| TRUE twin-basin  | q<30° from twin AND ω matches reflected truth | **0** |
| Multi-solution | everything else | 875 |

### IC pool truth/twin coverage (independent of thresholds)

| metric | value |
|---|---|
| min q0 to truth (any ω, all 432k cands) | **2.26°** |
| min q0 to twin  (any ω, all 432k cands) | **2.20°** |
| candidates with q<30° of truth | 2,610 |
| candidates with q<30° of twin | 2,546 |
| TRUE truth-basin candidates in IC pool | 2 |
| TRUE twin-basin candidates in IC pool | 3 |

The 2 truth-basin candidates score geo=0.000, align=0.429 — REJECTED by strict thresholds despite being inside the 30°/10%/10° truth basin.

### Wall budget breakdown

| stage | wall (s) | wall (min) |
|---|---|---|
| bracket | 0.08 | — |
| loop (Δ + geo + align over 1500 cells × 288 q0) | 3,684 | 61.4 |
| geo total (vectorised) | 0.4 | — |
| align total (surrogate, both PASS+FAIL) | 29,166 | 486.1 |
| align on geo-PASS (6,645 cands) | 466 | 7.8 |
| align on geo-FAIL (425,355 cands) | 28,700 | 478.3 |
| **WOULD-HAVE-SAVED in production-mode** | **28,700** | **98.4%** of align time |
| total | 3,684 | **61.4** |

Production-mode (skip surrogate on geo-rejects) wall ≈ 466 s align + 0.4 s geo + 70 s Δ ≈ **9 min per seed instead of 61 min**. **6.7× speedup** with the filter-first architecture.

### Pre-LM band yield

Truth basin (q<30°/ω<10%/dir<10°) candidates exist but are rejected by filter — pre-LM ρ-band on filter survivors would be uninformative for truth recovery. Survivor LCs are stored (`survivor_lcs.npz`) for hi-fi rerank if desired (would characterise the multi-solution attractor pool).

### Diagnostic verdict

The filter as configured (geo_threshold=1.0 at 5° face tolerance + align_threshold=1.0 over 7 bright peaks) is **too strict for the IC pool granularity**:

- 30° phi-step IC generation lands q0 candidates 2-15° from optimal at the closest case.
- Each q0 deviation cascades into spec-event timing/face mismatches.
- With only 2 spec events for seed 6, geo is binary 0/0.5/1.0; basin candidates land in 0.5 bin, not 1.0.
- Result: TRUE truth-basin candidates exist in IC pool (2 of 432k) but are filtered out.

The pipeline ARCHITECTURE is sound. The threshold/IC-granularity COMBINATION needs revision before truth recovery is feasible from filter alone.

## Why this matters

This is the first pipeline that combines all post-fix discoveries into a single inversion artefact. If filter survival is dominated by truth-class candidates, we may not need LM polish to deliver the cohort architecture. If filter survival is dominated by multi-solution attractors with truth absent, we know the geometric IC primitive needs strengthening (denser phi, tighter face shortlist) before LM polish can help.

The diagnostic mode (alignment cost on geo-rejects too) gives us the empirical "would-have-saved" measurement that calibrates production wall budget — necessary input before scaling to the 100-seed cohort.

## Numbers

- ω-mag bracket (top-5 LS peaks): `[0.013025, 0.021159, 0.029185, 0.042199, 0.058685]` rad/s. Truth at 0.012451 → nearest cell 4.61% off.
- ω-dir grid (300 Fibonacci): nearest cell to truth-dir at **3.72°**.
- q_target pool: 288 candidates from 4 classifiable peaks × tier shortlist × 12 phi.
- Total candidates: 1500 cells × 288 = **432,000**.
- Truth scores: geo = 1.000, align = 1.000.

## Artefacts

- `experiments/s020_seed_pipeline.py` — pipeline driver.
- `experiments/s020_seed_animate.py` — Plotly funnel HTML builder.
- `results/s020/seed006/`:
  - `bracket.npz` — LS spectrum + bracket cells + truth-distance metadata.
  - `omega_grid.npz` — full ω vectors + truth-direction distance per cell.
  - `delta_trajectories.npz` — Δ(t) per ω-cell (1500 × 500 × 4).
  - `q_target_pool.npz` — phi-sweep q_target list + per-candidate metadata.
  - `candidates_meta.npz` — per-candidate (q0, ω-cell, geo_score, align_score, timing, category flags).
  - `thresholds.npz` — truth-calibrated thresholds.
  - `spec_geometry.npz` — face normals, tier table, truth_pab_body.
  - `survivor_lcs.npz` — stored LCs for filter survivors only (reject LCs regenerable).
  - `survivor_diagnostics.npz` — survivor q0/ω + geodesics to truth/twin.
  - `summary.json` — top-level stats.
  - `s020_funnel_seed006.html` — interactive 4-panel funnel.

## Out of scope (deferred)

- **LM polish on filter survivors.** Explicit user decision. Survivors form the candidate pool that LM would polish; this run characterises that pool's quality before paying the polish cost.
- **5-seed pilot.** Wait for seed-6 verdict.
- **Hi-fi ρ-band of survivors.** Cheap (~minutes) follow-up; only worth doing if survivors look promising.
- **Cohort scan.** Wait for pilot.

## Cross-references

- s019 cohort coverage: top-N LS peaks vs full-bracket performance.
- s019b cohort basin variation: 3-5° at 3 seeds; cohort distribution unmeasured.
- s023: filter intersection rejects 99% on random panels; this run measures rejection on geometrically-anchored candidates.
- s014: cohort selector (lowest surrogate-MSE per seed) trustworthy at fixed truth-ω. s020 measures the OFF-truth-ω regime.
- Concept pages: `twin_degeneracy.md` (corrected X-flip), `surrogate_model.md`, `quaternion_convention.md`.
