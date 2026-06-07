---
title: s059j — cloud-data ω-grid search single-seed pilot on seed 28
type: experiment
sources:
  - experiments/s059j_cloud_data_omega_grid.py
  - experiments/s059j_design.md
  - experiments/s059i_validator.md
  - experiments/s059i_validator_perturbed.md
  - experiments/s059i_cohort_density_scan.md
related:
  - project_omega_grid_architecture.md
  - feedback_oracle_injection_taints_yield.md
  - feedback_use_existing_lib_forward.md
  - feedback_lm_cost_use_surrogate.md
created: 2026-05-08
updated: 2026-05-08
confidence: high (pipeline correct; bottleneck identified post-hoc by W-sweep + axis-decoupled probe)
---

# TL;DR

The s059j architecture executes end-to-end on seed 28 with NO truth-cluster
injection. **Headline: Band A∪B yield = 0/20 polished candidates.** The
GO/NO-GO gate fails. Wall 18.2 min (within budget).

**Bottleneck (post-hoc decoupled probe): ω-direction GRID SPACING, not
clustering, not local-window length.** A W-sweep on the cached score grid
that scored truth-EXACT, q_a-only-noise, ω-only-noise, and joint-noise
cells at W ∈ {5, 10, 25, 50, 100, 200} reveals two opposite behaviors:

- **q_a noise is BENIGN** — wider W IMPROVES it (3° q_a alone: ρ=13.8 at
  W=5 → ρ=5.9 at W=100; the noise gets amortized).
- **ω-direction noise is MALIGNANT** — wider W WORSENS it (5° ω alone:
  ρ=25 at W=5 → ρ=53 at W=100; the angular drift accumulates).

At truth-EXACT (q_a=truth, ω=truth) the cost is ρ=0.33–0.51 across every
W tested — the architecture IS correct, the cost surface IS sharp. But
at the closest grid cell (q_a noise 3.06° from cloud + ω-direction noise
5.33° from N_DIRS=200 Fibonacci spacing), ρ = 41.5 at W=10. The 5°
ω-direction quantization alone already pushes ρ into Band D.

**Forward path for the next agent**: densify N_DIRS from 200 to 800–1600
(7° → 3.5° → 1.5° avg sphere spacing) so the closest grid ω is within
1–2° of truth. At ~1° ω noise, ρ should land in 1–3 (Band A∪B) by
interpolation between the truth-EXACT and 5° data points. Cost:
4–8× the current 14.9-min score-grid wall, ~60–120 min Pool(24) per
seed. The user's "polish more clusters" intuition does NOT help here —
the truth-equivalent joint cell sits at rank 733,925 / 1,840,800 in the
full grid (bottom 60%), so no practical top-K covers it without a denser
ω-grid. The reviewer's "wider window" intuition does NOT help either —
W-sweep keeps truth-nearest at rank 7/7 across every W tested.

# What

Single-seed pilot of the s059j architecture per `experiments/s059j_design.md`.
Tests whether the cloud-data ω-grid + local-window surrogate-MSE + LM
polish pipeline lands ≥1 polished candidate in Band A∪B on seed 28
without injecting the truth cluster into the polish set. Followed by a
post-hoc decoupled W-sweep on the cached score grid that diagnoses why
the headline yield is zero.

# How

`experiments/s059j_cloud_data_omega_grid.py`. Nine stages per design doc:

1. **Anchor select** — coarse 50k Haar pool, project+survive on epochs
   [3, 30), pick T_A = argmin |C_t|. Wall 100s.
2. **Dense at anchor** — dense 400k Haar pool, project+survive at T_A
   only. Wall 28s.
3. **|C_a|-cap** — cap at 3000 (not triggered for seed 28; |C_a|=1534).
4. **ω-grid** — Fibonacci(200) × linspace(0.7, 1.3, 6) × |ω_a_truth|.
   ORACLE flag: |ω_a_body|_truth at T_A used for grid centering (matches
   s059i validators).
5. **Joint score grid** — for every (q_a, ω) ∈ C_a × ω_grid (1.84M
   cells), compute surrogate-MSE on local window [T_A−10, T_A+10] via
   `s059e.score_local_window`. Pool(24) over q_a chunks. Wall 14.9 min.
6. **Top-K + canon cluster** — top-K=5000, `lib.twin.canonical_batch`,
   greedy-cluster (q_radius=8°, ω_radius=15°, mag_pct=25%). 809 clusters.
7. **LM polish (NO truth injection)** — `s059e.lm_polish_local` on top-20
   cluster reps. Surrogate v2 only.
8. **Hi-fi gate + classify** — render iff surrogate_rho_local_polished
   < 4.0; ρ-band classify against `mag_hifi_truth`.
9. **Headline** — count Band A∪B; report yield.

**Post-hoc W-sweep diagnostic** (after pilot returned 0 yield):
loaded `score_grid.npz` + `lib.hifi_render.build_context(28)`, scored
8 specific cells at W ∈ {5, 10, 25, 50, 100, 200} via
`score_local_window`. Cells include truth-EXACT (q_a=truth_q_a,
ω=truth_ω_a — NOT on the grid, not in C_a), three single-axis noise
variants, and current-W=10 ranks {1, 10, 100, 1000}.

# Result

```
Pipeline:
  T_A=25, |C_{T_A}|=208 (coarse), |C_a|=1534 (dense), cap not triggered.
  closest q_a in C_a to truth = 3.062°.
  oracle |ω_a_body|_truth at T_A = 0.024909 rad/s (1.4272 dps).

Score grid:
  1.84M cells, Pool(24), wall 14.9 min.
  min MSE = 8.27e-02, min ρ_local = 5.753 (Band C).

Top-K + cluster:
  top-K=5000 by MSE; 5000 → 809 clusters.
  diagnostic truth-equivalent in top-K: idx=4574, score-rank 4575/5000.
  truth-cluster diagnostic rank = 564/809.

LM polish (top-20 cluster reps, NO truth injection):
  All 20 polishes converged to local minima with q0_err 75°-179°,
  |ω|err in [-40.5%, +79.5%], ω_dir_err 7.8°-89°.
  19/20 above the surrogate-ρ < 4 hi-fi gate.

Hi-fi:
  Rank 1 (only one passing gate) hi-fi ρ=66.40, band=D.

Bands: A=0  B=0  C=0  D=1  GATED=19  ERR=0
Band A∪B yield: 0/20 polished candidates.
Wall total: 18.2 min (1094s).
```

**Post-hoc W-sweep (decoupled axis probe)** at T_A=25:

| Cell                         | qa_d  | om_d  | ρ@W=5 | ρ@W=10 | ρ@W=50 | ρ@W=100 | ρ@W=200 |
|------------------------------|-------|-------|-------|--------|--------|---------|---------|
| **truth_EXACT**              | 0.00° | 0.00° | 0.51  | 0.39   | 0.33   | 0.42    | 0.47    |
| q_a 3° + ω EXACT             | 3.06° | 0.00° | 13.80 | 10.10  | 6.92   | **5.91**| 7.59    |
| q_a EXACT + ω 5° (grid)      | 0.00° | 5.33° | 24.99 | 38.06  | 54.49  | 53.09   | 61.40   |
| both grid (joint nearest)    | 3.06° | 5.33° | 36.12 | 41.54  | 56.01  | 54.09   | 61.85   |
| current rank 1 (W=10)        | 165°  | 30°   | 4.62  | 5.75   | 41.27  | 44.64   | 51.09   |
| current rank 100 (W=10)      | 147°  | 84°   | 6.53  | 8.38   | 39.82  | 45.75   | 60.35   |

Three readings:

1. **truth_EXACT is rank 1/8 at every W**. ρ stays in [0.33, 0.51].
   The architecture is mechanically correct; the cost surface is sharp.
2. **Adding 3° q_a noise alone**: ρ ∈ [6, 14], minimised at W=100
   (ρ=5.91). Wider W AMORTIZES q_a noise.
3. **Adding 5° ω-direction noise alone**: ρ ∈ [25, 61], grows
   monotonically with W. Wider W AMPLIFIES ω noise.

The N_DIRS=200 Fibonacci sphere has ~7° avg angular spacing; truth-ω
falls 5.33° from its nearest grid cell. That single 5° ω-quantization
gap suffices to push the truth-near grid cell into Band D (ρ=38).
Compounding with 3° q_a noise yields ρ=41.5 (the joint-nearest cell).

# Why this matters

This is the cleanest decoupled axis probe so far in the s059 series. It
identifies the architectural bottleneck precisely:

- **Cost function is correct.** Truth-EXACT (q_a, ω) → Band A at every
  W tested. The s059e local-window surrogate-MSE is doing its job.
- **q_a-axis is well-handled.** 3° q_a noise admits a polish back to
  Band C with W=100. The cloud-pool density (s059i_density_scan
  closest-survive 2.32° median at N=400k for seed 28) is sufficient.
- **ω-direction-axis is the bottleneck.** A 5° ω noise alone pushes
  ρ to ~38–53 across all W. The s059j default N_DIRS=200 leaves
  truth at this regime by construction.
- **Wider W does not help.** W-sweep keeps truth-nearest at rank 7/7
  across W ∈ {5, …, 200}. Wider W penalises ω-axis noise faster than
  it amortizes q_a-axis noise, so it CAN'T rescue grid-quantized
  truth-nearest cells.
- **More clustering / wider polish does not help.** The truth-equivalent
  joint cell sits at rank 733,925 / 1,840,800 of the full grid (bottom
  60%), so no top-K of any practical size covers it.

The s059i validators (rank 1/1407 at q_a noise ≤ 7.5°) implicitly
sidestepped this issue by INSERTING truth-ω into the grid at idx 0
(s059i_validator.py line 162-181). Without that insertion, the closest
discrete cell carries Fibonacci-quantization ω noise, and the cost
explodes. **This is a load-bearing methodology lesson**: validators
must use the same cost-surface evaluation as the production
architecture, including the same discrete grid resolution. A validator
with truth-ω inserted at idx 0 measures something different from a
production search with a fixed grid.

# Forward path

Priority order, all gated on cohort decision:

1. **A) Densify N_DIRS** — 200 → 800 (3.5° spacing) or 1600 (1.7°
   spacing). Cost: 4× or 8× the score-grid wall (~60 min or ~120 min
   Pool(24) per seed). Confirms whether ω-grid spacing alone is the
   bottleneck. If the closest grid ω drops to 1.5° from truth, ρ at
   the truth-near joint cell should land in 1–3 by interpolation;
   polish from there should converge to Band A∪B. **Highest-leverage
   next step.** Cohort 8-seed wall 8–16 hours, acceptable if
   single-seed test confirms.
2. **B) Adaptive ω-grid around top-density** — keep N_DIRS=200 globally,
   but locally densify around the M most-survived directions per seed.
   Cheaper than A; cost depends on whether truth ω falls in a high-density
   region (often yes by construction of the cost surface — truth-ω
   neighborhood would have many cells with low MSE, attracting density).
   Defer until A confirms the densification mechanism works.
3. **C) Surrogate gradient on ω** — bypass the discrete grid entirely
   by running scipy LM on (q_a, ω) jointly from many cluster reps.
   Polish budget would need to grow from 20 to 200+. Risk: LM at W=10
   from 5° ω noise + 3° q_a noise basin radius is unknown — need a
   smoke test.
4. **D) Multi-solution acceptance audit** — the W-sweep shows
   `current rank 1 W=10` (q_a=165°, ω=30°) at ρ=5.75 at W=10; this is
   either a body-twin alternate or a multi-solution attractor (per
   s014b's n_rotations<2 cohort tail prediction for seed 28). LM
   polishing rank 1 with full-LC LM and hi-fi gate gives one data
   point on whether seed 28 has a Band A∪B alternate. If yes, the
   architecture is partially saved by multi-solution acceptance even
   without finding truth.

A is the cleanest next move. It tests a specific hypothesis with a
well-defined fix and a quantitative budget. Run A on seed 28 first
(N_DIRS=800, ~60 min) before deciding whether to scale to cohort.

# Numbers worth remembering

- Truth-EXACT ρ (the cost surface's noise floor): **0.33–0.51 at every
  W from 5 to 200**.
- ρ at q_a noise 3° ALONE (truth ω): 5.91 at W=100, 13.80 at W=5.
  Optimal W for q_a-only-noise scoring: ~50–100.
- ρ at ω-direction noise 5° ALONE (truth q_a): 25–61 across W. No
  optimal W; monotonically grows.
- Truth-equivalent joint cell rank in full 1.84M-cell grid: **733,925**
  (bottom 60%). Out of any practical top-K.
- Truth-equivalent joint cell rank in top-K=5000 (the diagnostic
  reported in run.log): 4575/5000 (bottom 9%) — but this is misleading
  without the full-grid context above.
- N_DIRS=200 Fibonacci sphere → 7° avg spacing; truth-ω-direction noise
  5.33° at this density.
- Pipeline wall on seed 28 with N_DENSE=400k Pool(24): 18.2 min.
- Score-grid wall: 14.9 min for 1.84M cells (=2050 cells/sec aggregate
  at Pool(24)). Pool startup overhead is ~30s; first rate reading
  underestimated steady-state by 15× — a discipline note: don't trust
  early rates during pool warmup.

# Artefacts

- `experiments/s059j_cloud_data_omega_grid.py` — single-seed pilot
  script.
- `results/s059j_cloud_data_omega_grid/seed028/run.log` — full
  pipeline log.
- `results/s059j_cloud_data_omega_grid/seed028/anchor_summary.json` —
  T_A pick + per-epoch |C_t|.
- `results/s059j_cloud_data_omega_grid/seed028/score_grid.npz` — full
  1.84M-cell score matrix + C_a + ω-grid + diagnostic truth state.
- `results/s059j_cloud_data_omega_grid/seed028/clusters.npz` — top-K
  joint cells + cluster IDs + diagnostic truth-cluster rank.
- `results/s059j_cloud_data_omega_grid/seed028/polished_states.npz` —
  20 polished (q_0, ω_0) + per-candidate q_a/ω_a/ρ/error metrics.
- `results/s059j_cloud_data_omega_grid/seed028/summary.json` — full
  config + per-stage stats + headline yield + diagnostic truth.
- `results/s059j_cloud_data_omega_grid/seed028/score_grid.png` —
  log10(ρ_local) heatmap, q_a rows sorted by row-min ρ.
- The W-sweep diagnostic was an interactive python script not
  persisted as a standalone — the next agent re-runs the same probe
  by loading `score_grid.npz`, calling `build_context(28)`, and
  invoking `score_local_window` from `s059i_validator` on the
  truth-EXACT and noise-decoupled cells listed in the table above.

# Out of scope

- Cohort run on the 8-seed set. Per the design's verification step,
  cohort run is gated on seed-28 single-seed Band A∪B yield ≥ 1.
  That gate failed; cohort run is deferred until a densified-grid
  variant lands Band A∪B on seed 28.
- Estimator-|ω| (s055a pol_diam) variant. The pilot used oracle
  truth |ω| ±30%; an estimator pilot would only further degrade the
  cost surface (24% MAPE on |ω| widens the bracket effectively to
  ±54%). Densify N_DIRS first, THEN test estimator |ω|.
- Sobol-Shoemake refactor of `sample_so3_pool`. Per s059i_density_scan
  cohort scan, Haar-uniform at N=400k is in the robust regime; sampler
  refactor is cleanup, not blocking.
- A standalone reproducible W-sweep script. The interactive probe was
  enough to surface the diagnosis; productionising it is left to the
  next agent if the densification path fails to confirm the
  ω-quantization hypothesis.

# Cross-references

- `experiments/s059j_design.md` — the architecture handoff this pilot
  implements.
- `experiments/s059i_validator.md` — cost-surface validator at TRUTH
  q_a; measured truth-ω rank 1/1407 with truth-ω INSERTED in grid.
  This pilot exposes that "truth inserted at idx 0" hides the
  ω-direction quantization sensitivity.
- `experiments/s059i_validator_perturbed.md` — q_a-noise sweep with
  truth-ω still INSERTED; the gap between this regime and a discrete
  grid without truth-insertion is what s059j surfaces.
- `experiments/s059i_cohort_density_scan.md` — confirms |C_a| at
  N=400k is cohort-viable (closest-survive 2.32° median on seed 28);
  this pilot confirms 3.06° on the actual run, well within robust regime.
- `experiments/s059e_local_window.py` — the LM polish substrate
  reused as-is.
- `experiments/s059_pilot.py` — constants and `stage_cluster` reused
  via a synthetic fp dict with -MSE scores for HIGHER-IS-BETTER
  semantics.
- `feedback_oracle_injection_taints_yield.md` — the no-injection
  rule this pilot enforces.
