---
title: "s082 — joint (q0, ω) grid pivot: Branch v2 is dead; v3 is the path, conditional on |ω|-bracket densification"
type: experiment
sources:
  - experiments/s082_joint_grid_pivot.py
  - results/s082/holdout_stratification.json
  - results/s082/summary.json
  - results/s082/seed_116/summary.json
  - results/s082/seed_119/summary.json
  - results/s082/seed_103/summary.json
  - results/s082/seed_{116,119,103}/candidates.npz
  - results/s082/seed_{116,119,103}/pivot_diagnostics.png
related:
  - report/blind_inversion_15min_plan_2026-05-20.md (§3.4 grid-density math, §4 branch specs, §5 pivot spec, §10.7 surrogate cost risk)
  - experiments/s019_ls_bracket_omega_mag.md  (LS-bracket method; pre-fix ~50-cell variant 98/100 within 5%)
  - experiments/s005_joint_local_descent.md   (LM grab radius q0 5-15°, ω-dir 1-2°, |ω| 3-5%)
  - experiments/s072_path2_closed_form_q.md   (Path 2 closed-form q(t))
  - experiments/s074_elliprj_path2.md         (elliprj φ(t) integration — landed alongside s082, default Path 2 method now)
  - experiments/s068_replicate_s011.md        (Sobol N=64 + LM @ truth-ω: 10/10 PA-stratified)
  - experiments/s043_twin_hifi_verify.md      (body-X twin halving free 2×)
  - experiments/s003_landscape_vs_omega.md    (truth-ω tube width: ~1° dir, ~2-5% mag; outside is incoherent)
  - experiments/s081_hifi_rho_bands_twin.md   (surrogate↔hi-fi 145/145 band agreement)
created: 2026-05-20
updated: 2026-05-20
confidence: high (3-seed pivot with stable per-axis decomposition; results consistent across LAM-slow / LAM-fast / SAM)
---

## TL;DR

Joint (q0, ω) grid pivot on 3 fresh holdout seeds (116 LAM-slow, 119 LAM-fast, 103 SAM) measured the rank of the truth-nearest cell under full-LC Jacobi+surrogate scoring on Sobol-Shoemake N=64 × Fibonacci(canonical 1000 from N=2000) × LS-bracket N=5 = 320k joint candidates per seed. **Median truth-nearest rank 70823/320000** — far outside both top-200 (Branch v2 trigger) and top-10000 (Branch v3 trigger). Plan §5.4 auto-decision: `reconsider`. Per-axis diagnostic breakdown identifies the bottleneck as **|ω|-bracket coarseness, not N_dir or N_q**: truth-nearest cells were 0.94–2.69° in ω-direction (well inside LM grab radius) and 16–33° in q (at Sobol-N=64 quantisation) but **20–43% off in |ω|** because the 5-cell geomspace LS-bracket has √2 ≈ 41% spacing. Worse, **top-50 candidates are random LC fitters, not near truth**: top-50 median q-geo 139°, ω-dir 29–82°, |ω|-err 22–83%. Even the top-1 candidate on every seed lands Band C/D (ρ_surr 6.55 / 26.70 / 25.04 for seeds 116/119/103). **Branch v2 (dense grid + top-K=200 polish) is dead** — no top-K of any size catches truth-near candidates from this grid. **Branch v3 (multi-anchor consistency filter via Path 2 + L_J2000) is the path**, conditional on (a) increasing N_mag from 5 to ~10–15 to make truth-|ω| reachable, and (b) using the elliprj Path 2 just landed in s074 to fit the resulting denser candidate count inside the 15-min budget. Compute wall 100 min Pool(24), 33 min/seed; pre-launch surrogate timing revealed risk §10.7 as realised (~50 ms/candidate single-process vs the plan's 3–5 ms estimate). Source: `results/s082/summary.json`.

## What

Per plan §5.1: measure where truth (and any Band A∪B candidate) ranks in the full-LC Jacobi+surrogate-MSE ordering on a dense joint `(q0, ω)` grid. The median rank across 3 fresh holdout seeds picks Branch v2 vs Branch v3 for the Day-2 inversion pipeline.

## How

### Stratification (plan §5.2)

For each holdout seed in 100..119, computed `disc = 2T·I_2 − |L|²` (LAM if > 0, SAM otherwise), `|ω|` in dps, and mean phase angle across the LC from cached `k1_body, k2_body`. Truth metadata used ONLY for picking seeds and final error reporting; the pipeline itself runs blind on `(observation_times, mag_hifi, sun_pos, obs_pos, sat_pos, inertia)`.

Holdout regime split: **17 LAM / 3 SAM out of 20** (source: `results/s082/holdout_stratification.json`).

**Picks (1 LAM-slow / 1 LAM-fast / 1 SAM):**

| Label | Seed | Regime | \|ω\| (dps) | mean PA (deg) |
|---|---:|---|---:|---:|
| LAM-slow | 116 | LAM | 0.134 | 29.9 |
| LAM-fast | 119 | LAM | 1.482 | 10.7 |
| SAM | 103 | SAM | 0.530 | 12.3 |

Selection ranks LAMs by |ω| (116 is slowest of 17 holdout LAMs, 119 is fastest); SAM picks the median of the 3 holdout SAMs. Seed 116 is slower than any m048-cohort seed used in s081 (slowest there was seed 89 at 0.240 dps); seed 119 is comparable to m048 seed 28 (1.438 dps, the near-separatrix tight-basin case).

### Per-seed pivot (plan §5.3)

Each picked seed runs:

```
ctx = lib.hifi_render.build_context(seed)           # blind inputs
om_mag_grid = lib.lc_features.ls_bracket(times, mag_hifi, n_cells=5)
q_cands, om_cands = build_joint_candidates(         # Sobol × Fibonacci-canonical × |ω|
    n_q=64, n_dir=2000, n_mag=5, om_mag_grid=om_mag_grid
)   # → 320k pairs (Fibonacci filtered to ω_y > 0 canonical hemisphere)

# Pool(24) — Path 2 forward → q → R(q) → k1/k2_body → surrogate predict → MSE
for (q0, ω0) in candidates:
    q_hist, _ = propagate_jacobi_path2(q0, ω0, inertia, times)
    R(q_hist) → k1_body, k2_body
    mag_pred = surrogate.predict_magnitude(...)
    mse = mean((mag_pred - mag_observed)²)

# Truth loaded ONLY at end for error reporting (plan §1.3, §9 rule 1)
truth_nearest_idx = argmin( q_geo + om_dir_err + 100·om_mag_err_rel )
rank_by_mse = 1 + sum(mses < mses[truth_nearest_idx])
```

Per-seed diagnostic axes saved alongside rank — letting any bad rank be attributed to a specific grid axis:
- q-geodesic to truth (Sobol pool quantisation cap),
- ω-direction offset to truth (Fibonacci angular spacing),
- |ω|-offset to truth (LS-bracket grid quantisation).

Twin canonicalisation (`lib/twin.py`, s043 bit-exact) applied so the truth comparison is done in canonical space (ω_y > 0).

### Per-candidate cost — measured (NOT the plan's estimate)

Pre-launch benchmark on seed 100 (single-process, 50 iterations after warmup, pre-elliprj Path 2 since s074 was still building):

| Stage | Per-candidate ms | Notes |
|---|---:|---|
| Path 2 (500 epochs, solve_ivp φ-ODE) | ~4 ms | Seed-dependent — s073b's 16.4 ms was on seed 89 specifically; seed 100 has an easier ODE |
| q → R + project | ~0.03 ms | Vectorised `Rotation.from_quat` + `einsum` |
| Surrogate full-LC | ~48 ms | **Dominant cost** — batched test shows 38–50 ms saturating regardless of batch size |
| **Total single-thread** | **~52 ms** | 320k @ Pool(24) actual = 6.22 ms amortised/candidate ≈ 33 min/seed |

Plan §3.2 estimated surrogate at 3–5 ms/candidate (sourced "from `concepts/surrogate_model.md` claims ~50000× hi-fi speedup"). The realised cost is **10× higher**. Risk §10.7 (`Risk: surrogate is the per-candidate bottleneck`) **was realised**. The surrogate is a pure-numpy residual-ensemble MLP whose forward pass is BLAS-bound on the `(N×candidates, n_features)` GEMM; batching does not amortise per-call Python overhead because that overhead is already small relative to the matrix multiplies. No GPU port investigated (out of scope for the pivot).

Note for Day 2: s074 (landed during this pivot, commit `f9eff5b`) makes Path 2 ~5 ms (Pool(24) median) regardless of seed via elliprj. The s082 candidate-scoring workers were forked **before** the s074 commit, so the pivot itself used the pre-elliprj `solve_ivp` Path 2. The Day-2 pipeline picks up elliprj automatically as the new default in `lib/jacobi_propagator.py` (source: `experiments/s074_elliprj_path2.md`).

## Result

### Per-seed truth-nearest rank

(source: `results/s082/seed_*/summary.json` for all numbers below)

| Seed | Label | Rank / 320k | q-geo (°) | ω-dir (°) | \|ω\|-err (%) | ρ_surr truth-nearest | Wall (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 116 | LAM-slow | 70823 (top 22.1%) | 33.13 | **0.94** | **20.02** | 29.85 | 1989 |
| 119 | LAM-fast | 259195 (top 81.0%) | 16.55 | 2.35 | **42.74** | 48.12 | 2084 |
| 103 | SAM | 36568 (top 11.4%) | 29.34 | 2.69 | **21.65** | 38.31 | 1956 |
| **median** | | **70823** | | | | | |

**Plan §5.4 auto-decision: `reconsider`** (median rank > 10000). But the per-axis decomposition tells a sharper story than the rank threshold alone — see § Why this matters below.

### Top-K characteristics — top-50 are random fitters, NOT Band A∪B multi-sols

| Seed | top-1 ρ_surr | top-10 ρ_surr | top-50 ρ_surr | top-50 median q-geo (°) | top-50 median ω-dir (°) | top-50 median \|ω\|-err (%) |
|---|---:|---:|---:|---:|---:|---:|
| 116 | 6.55 (Band C) | 8.82 (Band D) | 11.58 (Band D) | 138.9 | 28.9 | 36.9 |
| 119 | 26.70 (Band D) | 27.31 | 28.14 | 143.5 | 81.7 | 83.4 |
| 103 | 25.04 (Band D) | 26.78 | 28.62 | 139.7 | 61.3 | 21.6 |

**Key result: top-K does NOT contain truth-near candidates on any seed.** Top-50 median q-geodesic from truth ≈ 140° on every seed (essentially antipodal in SO(3) — random with respect to truth). Even top-1 is Band C or D on every seed; no top-K of any size contains a Band A∪B multi-sol because none exists in this 320k-candidate grid.

### Minimum-axis errors (best-cell-on-each-axis, NOT the same cell)

(source: `min_axis_errors_unconditional` in each `seed_*/summary.json`)

| Seed | min q-geo (°) | min ω-dir (°) | min \|ω\|-err (%) |
|---:|---:|---:|---:|
| 116 | 33.13 | 0.94 | 20.02 |
| 119 | 16.55 | 2.35 | 42.74 |
| 103 | 29.34 | 2.69 | 21.65 |

The truth-nearest cell happens to be the best-on-each-axis cell on each seed — i.e., the combined-metric minimum is dominated by the |ω|-magnitude axis here. With the 5-cell LS-bracket, no cell ever lands closer than 20% in |ω| on any of the 3 seeds; the truth-|ω| simply isn't representable.

## Why this matters — Branch v2 is dead, v3 is the path

### 1. Branch v2 (dense grid + top-K=200 polhode-basis polish) cannot work on this substrate

The plan §4.1 premise: full-LC Jacobi+surrogate scoring puts truth-near cells in a polishable top-K (≤200). **Refuted on all 3 seeds.** Top-K=200 contains random LC fitters wandering 140° from truth in q-space; no amount of polishing the top-200 reaches a Band A∪B basin. The LM polish grab radius (s005: q0 5–15°, ω-dir 1–2°, |ω| 3–5%) is far smaller than the top-K spread.

The plan's framing — "truth-nearest cell ranks well in surrogate-MSE on the joint grid" — was an architectural hypothesis. **The pivot falsifies it.** The s003 finding ("truth-ω tube is ~1° dir × 2–5% mag; outside is incoherent") was already pointing in this direction; s082 confirms it at the joint-grid scale.

### 2. The bottleneck axis is |ω| — not N_dir, not N_q

Per-axis story across the 3 seeds (numbers from § Per-seed truth-nearest rank above):

- **ω-direction grid is good.** Truth-direction within 0.94–2.69° at N_dir = 2000 full-sphere = 1000 canonical-hemisphere. This is right at or inside the LM grab radius (1–2°). N_dir doubling/tripling would help marginally but isn't the load-bearing issue.
- **q-coarseness is borderline.** Sobol N=64 caps closest q-geodesic at 16.55°–33.13° on these seeds, which is just outside LM q grab radius (5–15°). Needs N_q ≥ 256 (Shoemake quantisation ~12° at N=256) to be safely inside.
- **|ω| is the killer.** 5-cell LS-bracket has √2 ≈ 41% spacing over a 4× span [0.5·min_peak, 2·max_peak]; worst-case nearest-cell offset is ~17%, realised offset on these seeds is **20–43%**. s019's pre-fix bracket gets 98/100 within 5% **at ~50 cells (5% step)**; condensing to 5 cells trades that down to never-within-5%. **This is the single design choice that kills v2.**

### 3. The s074 elliprj landing changes the budget arithmetic

s074 (`experiments/s074_elliprj_path2.md`) replaced the φ-ODE in Path 2 with a closed-form Carlson R_J evaluation. Path 2 per-candidate wall dropped from 16.4 ms (s073b on seed 89) to **4.76 ms Pool(24) median** — a 3.4× speedup, up to 47× on Gate-3 regime B (181 ms → 3.9 ms, source: `s074_elliprj_path2.md`).

Post-elliprj per-candidate cost: Path 2 ~5 ms + surrogate ~48 ms ≈ 53 ms (Path 2 no longer the bottleneck, surrogate is). Pool(24) amortised ~6 ms/candidate (same as observed in s082 — surrogate was already the dominant cost in the pivot). **Wall budget per seed at 15 min Pool(24) ≈ 6 ms × 24 workers × 900s / 6 ms = 144k candidate-evaluations** (rounded for the surrogate-amortised case). Subject to the surrogate cost.

### 4. Branch v3 (multi-anchor consistency via Path 2 + L_J2000) is the only viable path

Plan §4.2: pre-prune candidates using cheap multi-anchor consistency checks BEFORE the expensive full-LC surrogate scoring. With elliprj Path 2 at ~5 ms per full-LC forward, plus per-candidate per-anchor surrogate single-epoch predict + nearest-q lookup (~1–3 ms total per candidate per anchor), the architecture handles a much larger pre-prune candidate count.

**Conditional fix for the pivot's |ω| issue: increase N_mag from 5 to 10–15** (cells at 15–20% spacing, worst-case nearest-cell offset ≈ 7–10%). With elliprj+pre-pruning, this fits the budget. Concretely:

| Branch | N_q | N_dir (canonical) | N_mag | Total | Stage 5 wall (15 min budget?) |
|---|---:|---:|---:|---:|---|
| v2 (plan) | 64 | 1000 | 5 | 320k | ~33 min (surrogate cost; pivot measurement) |
| v2 retuned for |ω| | 64 | 1000 | 15 | 960k | ~99 min — over budget |
| **v3 elliprj** | 64 | 2500 | 10 | **1.6M** | ~3 min for multi-anchor pre-prune at K=3 anchors, then ~30 sec for top-5k full-LC + ~2 min polish ≈ **6–8 min/seed**; FITS |

[hypothesis — multi-anchor unit costs not measured. The above v3 budget assumes ~1.5 ms/candidate for one anchor check (Path 2 propagation to one secondary anchor + 1-epoch surrogate predict + KD-tree nearest-q lookup). Path 2 to a single epoch should be much cheaper than the full 500-epoch trajectory since the underlying elliprj evaluation cost scales with the number of t_eval points; the unit cost needs benchmarking on Day 2 morning before locking the design.]

### 5. What this DOES NOT establish

- Doesn't refute Branch v3 — only Branch v2. v3 was always the harder architecture; the pivot does not test it.
- Doesn't tell us where the truth-near LM grab basin actually sits relative to a denser grid — that requires running the pivot again at N_mag=15+.
- N=3, all from holdout. Cohort-scale behavior may differ on the m048 main seeds. But the failure mode (|ω| coarseness) is structural, not seed-specific.
- The "top-1 Band C on seed 116" result (ρ_surr 6.55) is interesting — it's a non-truth candidate that fits the LC moderately well. Worth investigating in a follow-up whether it's a known multi-sol class (e.g. s081 Group A ω-scaling) or a random fitter.

## Numbers

Per-seed compute walls and scoring summary:

- seed 116: 1989 s scoring (33.2 min), 6.22 ms/candidate amortised, MSE min 0.107 ⇒ ρ_surr 6.55, p1 1.016 ⇒ ρ_surr 20.16, p50 3.358. (source: `results/s082/seed_116/summary.json:mse_distribution`)
- seed 119: 2084 s scoring (34.7 min), 6.51 ms/candidate amortised, MSE min 1.782 ⇒ ρ_surr 26.70, p1 2.668, p50 4.561. (source: `results/s082/seed_119/summary.json:mse_distribution`)
- seed 103: 1956 s scoring (32.6 min), 6.11 ms/candidate amortised, MSE min 1.568 ⇒ ρ_surr 25.04, p1 2.783, p50 4.663. (source: `results/s082/seed_103/summary.json:mse_distribution`)
- Total wall: 6030 s (100.5 min) including ~10 s stratification + per-seed candidate generation overhead.

## Artefacts

- `experiments/s082_joint_grid_pivot.py` — pivot driver
- `experiments/s082_joint_grid_pivot.md` — this file
- `lib/lc_features.py` — modified, `ls_bracket(times, mag, n_cells=5)` added
- `results/s082/holdout_stratification.json` — 20-seed regime + |ω| + PA metadata
- `results/s082/summary.json` — cohort-level rank + branch decision
- `results/s082/seed_{116,119,103}/summary.json` — per-seed full per-axis errors and top-K stats
- `results/s082/seed_{116,119,103}/candidates.npz` — all 320k `(q, ω, mse, q_geo_deg, om_dir_err_deg, om_mag_err_rel)` per seed; raw material for Day 2 v3 design + N_mag sensitivity follow-up
- `results/s082/seed_{116,119,103}/pivot_diagnostics.png` — MSE histogram + top-1000 scatter (q-geo × ω-dir-err coloured by log10(MSE))

## Out of scope

- Day-2 v3 pipeline implementation (Branch v3 with N_mag=10–15 is the recommendation; design + driver land in s083+).
- N_mag sensitivity sweep — should be a quick analytical check before launching Day-2 production (re-run on the same 3 pivot seeds at N_mag=11/15/25 with N_dir=1000, look at where truth-nearest rank moves to).
- Multi-anchor unit-cost benchmark for v3 — needs to be done Day-2 morning before locking the architecture.
- Surrogate GPU port — surrogate is the dominant cost but a separate axis; in scope only if v3 wall doesn't fit.

## Branch decision (post-pivot)

**Decision: Branch v3 (multi-anchor consistency filter via Path 2 + L_J2000), with N_mag densified to 10–15.**

**Rationale:** Branch v2 falsified — top-K=200 polish doesn't catch truth-near candidates because (a) truth-nearest is rank 36k–260k, far outside top-200; (b) top-50 candidates are 140°-from-truth random LC fitters with no Band A∪B in the entire 320k grid. Branch v3's multi-anchor pre-prune is the only mechanism that can identify truth-near candidates from a denser candidate pool inside the 15-min wall budget. The s074 elliprj landing (commit `f9eff5b`) makes the denser pool feasible.

**Plan's auto-decision of `reconsider` is preserved in the cohort summary.json** — the rank-threshold framing was honest about saying "this doesn't fit either branch as specified". The richer per-axis story argues for v3 with a specific grid retuning. Both readings agree on "don't do Branch v2".

## Cross-references

- Plan: `report/blind_inversion_15min_plan_2026-05-20.md` §3.4 (grid-density arithmetic), §4 (branch v2/v3 specs), §5 (pivot spec), §10.7 (surrogate cost risk — realised).
- Parallel-track landing same day: `experiments/s074_elliprj_path2.md` (elliprj Path 2 production default).
- Prior closed approaches: s003 (decoupled ω-outer/q0-inner closed — predicted this pivot result), s055b/c (LC-only ω-direction priors closed), s059j (local-window scoring closed), s063c (constant-ω invalid).
- Production-ready tooling now stacked: s019 (LS-bracket method), s072 + s074 (Path 2 elliprj), s064 (polhode-basis LM), s043 (twin), s081 (surrogate↔hi-fi 145/145).
