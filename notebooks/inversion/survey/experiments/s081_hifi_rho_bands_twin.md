---
title: "s081 — hi-fi ρ-band + twin classification of s011/s068 competing basins; Group-A 'searchable form' is ω-scaling, not same-polhode"
type: experiment
sources:
  - experiments/s081_hifi_rho_bands_twin.py
  - results/s081/basin_hifi.npz
  - results/s081/summary.json
  - results/s081/s081_rho_bands.png
  - results/s081/s081_hifi_vs_surrogate.png
  - scratch/s081_error_table.py
  - scratch/s081_form_check.py
related:
  - experiments/s077_l_vector_basin_sweep.md
  - experiments/s079_regime_stratified_l_basins.md
  - experiments/s068_replicate_s011.md
  - experiments/s073d_polhode_match_cluster457.md
  - experiments/s073f_cluster457_local_geometry.md
created: 2026-05-20
updated: 2026-05-20
confidence: high (41/41 pipeline control PASS; 145/145 surrogate↔hi-fi exact band agreement; polhode invariants read directly from cached s077 metrics)
---

# TL;DR

Rendered hi-fi the 145 truth + competing-low-MSE basins from the s077 substrate (10-seed s011/s068 pilot). **41/41 truth-basin control returned hi-fi Band A** (pipeline gate PASS) and **145/145 basins agreed in band exactly between surrogate-v2 and hi-fi**, retiring s077/s079's loose-surrogate caveat. Of 104 competing basins, **30 land hi-fi Band A∪B** (15 A, 15 B); all 30 labelled `distinct` (no body-X twin hits at the 8°/8% tolerance). Splitting ω into direction/magnitude on the 30 reveals two regimes: **Group A** (19 endpoints, ω-direction < 15° from truth, q0 rotated 134–180°, ω-magnitude scaled 0.5–6%) and **Group B** (11 endpoints, all seed 10, ω-direction 52–89° from truth, polhode invariants off up to 38%). **Group A is not the s077 "same-polhode, free-L-direction" form** — d_2T_rel and d_L2_rel mismatch truth by median 3.3%, max 12.5%, with `d_2T_rel ≈ d_L2_rel ≈ 2·L_mag_rel` in every row (the algebraic signature of `ω → c·ω` magnitude scaling). The searchable form is *fix ω-direction, sweep (|ω|, q0-inertial-offset)*, recoverable *given* an ω-direction estimate near truth — cold-start remains open. Source: `results/s081/summary.json`, `scratch/s081_form_check.py`.

# What

s077 (re-score of 640 s011/s068 polish endpoints) found that among 104 surrogate-competing basins, |L| was pinned to truth (median 0.73% off) while L-direction varied freely (median 91°). s079 stratified this by tumbling regime and found 96/104 of the competing basins concentrated on the 6 LAM seeds. Both used the loose surrogate gate `final_mse < 0.5` — and `project_local_window_polish_phantom_basins.md` puts the surrogate↔hi-fi gap at 30–70 ρ in pathological cases. This experiment renders the 145 truth + competing basins hi-fi to:

1. Retire the loose-surrogate caveat — does any "competing" basin survive at hi-fi ρ < 4?
2. Apply the body-X twin classification so the "distinct" subset is unambiguous (per `lib/twin.py` — the only verified IS-901 LC-preserving symmetry, s043).
3. Re-check the s079 LAM/SAM split on the hi-fi Band A∪B *distinct* subset specifically.
4. Read off the actual structure of the multi-solution set: are the Band A basins instances of a low-dimensional searchable form?

# How

**Substrate.** `results/s077/basin_l_metrics.npz` (640 basins = 10 seeds × 64 ICs over post-fix s011/s068 polish endpoints). Render mask = truth ∪ twin ∪ competing_low_mse = 145 basins. The 495 high-MSE basins are skipped: surrogate ρ ≥ 14 there and the surrogate failure mode is over-optimism (high score, low quality), not pessimism, so a hi-fi A∪B hiding in surrogate-D is not a credible risk on this substrate. The 145/145 surrogate↔hi-fi band agreement (see Result) supports that scoping post-hoc.

**Per-basin render.**

- `render_hifi(q0_final, omega_final, ctx)` → 500-epoch hi-fi LC via `lib/hifi_render.py`.
- `rho_from_hifi(pred, truth)` = √MSE / 0.05.
- `rho_band` per `concepts/rho_band.md` (A: ρ<2, B: 2≤ρ<4, C: 4≤ρ<8, D: ρ≥8).
- Twin classification: q-geodesic to `twin(q0_truth, ω_truth)` < 8° AND relative ω distance < 8%. Body-X twin only; no body-Y/Z symmetry is verified for IS-901.

**Methodology note — OOM bug fix.** The first-pass s081 (Pool(24), no `gc.collect()` between renders) OOM-killed on the 30 GB box. Diagnosis (probe-driven):

- Single hi-fi render peak RSS = **1.121 GB** (one render, one process).
- Four consecutive renders without `gc.collect()`: peak climbs to **1.554 GB** — `compute_shadows` (trimesh) leaves cyclic garbage that CPython's refcounter cannot free, so back-to-back renders accumulate scratch.
- Fourteen consecutive renders WITH `gc.collect()` between: peak stays flat at **1.131 GB** (verified over 14-render probe).
- `Pool(24)` × 1.55+ GB ≈ 38+ GB on a 30 GB box → OOM kill.

Fix: `gc.collect()` at the end of `_render_one`, `Pool(24) → Pool(16)`. 16 × 1.13 GB ≈ 18 GB, comfortable headroom. Rerun completed in 844.3 s (~14 min) on Pool(16) with `summary.json` written cleanly. Methodology lesson lives in `memory/feedback_pool_size_with_trimesh_gc.md`.

**Post-render analysis (cached, no re-render).**

- `scratch/s081_error_table.py` — splits ω into direction error (deg) and magnitude error (relative). s081 itself saved only the combined relative ω distance; per the project rule (`feedback_report_all_three_errors.md`) both components are reported here.
- `scratch/s081_form_check.py` — joins to `results/s077/basin_l_metrics.npz` for the polhode invariants 2T, |L|², k², regime per basin (s077 already computed `d_twoT_rel`, `d_L2_rel` vs truth). Splits the 30 distinct Band A∪B basins by ω-direction error at 15° (a clean gap in the data: Group A 0–8°, Group B 52–89°).

# Result

## Hi-fi band distribution (145 rendered)

| Class | n | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| Overall | 145 | 56 | 15 | 48 | 26 |
| Truth basins | 41 | 41 | 0 | 0 | 0 |
| Competing low-MSE | 104 | 15 | 15 | 48 | 26 |

Truth control: **41/41 → Band A** (source: `results/s081/summary.json:band_distribution_hifi.by_basin_class.truth`). Surrogate vs hi-fi: **145/145 exact band agreement**, 0 surrogate-A∪B-but-hi-fi-worse, 0 surrogate-worse-but-hi-fi-A∪B (source: `summary.json:surrogate_vs_hifi`).

## s079 LAM/SAM split — re-checked on the hi-fi Band A∪B distinct subset

| Regime | n_seeds | n_competing_hifi_AB | n_distinct_hifi_AB | per-seed distribution |
|---|---:|---:|---:|---|
| LAM | 6 | 28 | 28 | seed 6: 5, seed 10: 19, seed 44: 3, seed 48: 1, seed 60: 0, seed 84: 0 |
| SAM | 4 | 2 | 2 | seed 21: 0, seed 28: 0, seed 41: 2, seed 91: 0 |

s079's LAM-concentration finding **holds at hi-fi**: 28/30 distinct Band A∪B basins live on LAM seeds; SAM contribution is 2/30, both on seed 41. (Source: `summary.json:s079_split_on_hifi_band_AB`.)

## Group A / Group B split — the searchable-form structure

Three-error breakdown on the 30 distinct Band A∪B endpoints, split at ω-direction error = 15° (clean data gap):

| | n endpoints | n unique (per-seed dedup) | ω-dir vs truth | ω-mag vs truth | q0 vs truth | d_2T_rel | d_L2_rel |
|---|---:|---:|---:|---:|---:|---:|---:|
| Group A | 19 | ~10 | 0.1–8° | −0.6% to +6.4% | 134–180° | 0.9–12.5% (med 3.3%) | 0.9–12.6% (med 3.3%) |
| Group B | 11 | ~6 | 52–89° | −15% to +5% | 77–180° | 1.1–37.8% | 1.4–40.3% |

**Mechanical signature in Group A**: `d_2T_rel ≈ d_L2_rel ≈ 2·L_mag_rel` in every row. Examples:

| seed | ρ_hifi | L_mag_rel | d_L2_rel | 2·L_mag_rel | d_2T_rel | d_L2_rel - d_2T_rel |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 3.63 | +0.74% | +1.475% | +1.48% | +1.425% | +0.05% |
| 10 | 1.29 | +5.054% | +10.364% | +10.11% | +10.404% | −0.04% |
| 10 | 1.46 | +6.093% | +12.557% | +12.19% | +12.461% | +0.10% |
| 44 | 2.35 | +0.512% | +1.021% | +1.02% | +0.922% | +0.10% |
| 48 | 1.24 | +2.710% | +5.347% | +5.42% | +4.055% | +1.29% |

(Source: `scratch/s081_form_check.py` stdout; raw data `results/s077/basin_l_metrics.npz` keys `L_mag_rel_diff`, `d_L2_rel`, `d_twoT_rel`.)

Under uniform ω-magnitude rescaling `ω → c·ω`: `2T = ω·I·ω → c²·2T` and `|L|² = (Iω)·(Iω) → c²·|L|²`, so both move together at rate ≈ 2·(c−1) for small (c−1). The columns confirm this is what's happening: |ω| has been scaled, ω-direction held within 15°, q0 rotated 134–180°.

**Group A is NOT the s077 "same-polhode, free L-direction" form.** A same-polhode family requires d_2T_rel and d_L2_rel ≈ 0 (within the propagator's 1e-12 floor). Here d_2T_rel mismatches truth by 0.9% (seed 44, the tightest) up to 12.5% (seed 10, ρ1.46). At the worst end this exceeds the s073d "nearby polhodes" picture (cluster_457 at 1.1%) by an order of magnitude — these are *different polhodes of approximately similar regime* (the discriminant sign and k² are roughly preserved by ω-scaling), not the same polhode.

# Why this matters

1. **The s077/s079 loose-surrogate caveat is retired.** 145/145 surrogate↔hi-fi band agreement means the s077 cohort finding (|L|-pinned, L-direction-free among competing basins) and the s079 LAM/SAM split survive the hi-fi gate without revision. The PROGRESS.md Next #1 item ("s080 — hi-fi ρ-band validation") is closed.

2. **Multi-solution structure exists at hi-fi but is more constrained than the s077 framing suggested.** 30 distinct hi-fi Band A∪B endpoints (~16 deduped unique states) is a measured, non-trivial multi-solution census. But the s077 framing ("|L| pinned, L-direction free") was at surrogate level and described an *approximate* invariance; the hi-fi-grade structure has 2T and |L|² visibly drifting together at 0.9–12.5%, not pinned.

3. **The actually-searchable form (Group A): fix ω-direction near a guess, sweep (|ω|, q0-inertial-offset).** This is a ~4-parameter family (1 ω-magnitude × 3 q0-offset) instead of blind SO(3) × ω³ (6 parameters). It is essentially s011/s068's actual search relaxed by ±6% in |ω| and with q0 covering all of SO(3). It is *not* a continuous family — s077 already found discrete clumps — but it is the right substructure to search.

4. **Group B (ω moved 50–90°) is the unexplained residue.** 11 endpoints, all on seed 10, with d_2T_rel up to 37.8%. Genuinely different dynamical states producing the same LC. None of these fit any low-dimensional searchable form yet identified. Seed 10 was already flagged in s068 as the post-fix "rich competing-attractor" seed (3/64 in-basin landings at N=64, vs the pilot median 5/64); Group B is the residue of that richness at hi-fi.

5. **The s073-cat-4 framing weakens further at cohort scale.** s073d found cluster_457 (seed 89) at 1.1% Casimir mismatch — *nearby* polhode, not same. s073f's strict-reading verdict (cluster_457 is locally isolated, not a member of a continuous family) is consistent with s081: 19 Group A basins across 5 LAM seeds, none on the same polhode as truth. The "free L-direction at constant Casimir" cat-4 claim has no support in this 30-basin hi-fi sample.

6. **Search provenance caveat.** Every basin here was discovered by a 6-DOF LM polish *seeded at truth-ω*. The fact that Group A's targeted form works *given* an ω-direction estimate near truth does not establish that such a search would work *cold*. Cold-start ω recovery remains open and is the natural next architectural question.

# Numbers

- Render set: 145 basins (41 truth + 104 competing_low_mse), 10 seeds (6, 10, 21, 28, 41, 44, 48, 60, 84, 91). Source: `results/s081/summary.json:n_rendered`.
- Wall: **844.3 s** on Pool(16). Source: `summary.json:render_wall_s`.
- Truth control: **41/41** hi-fi Band A. Source: `summary.json:band_distribution_hifi.by_basin_class.truth`.
- Surrogate↔hi-fi exact band agreement: **145/145** (zero in either off-diagonal). Source: `summary.json:surrogate_vs_hifi`.
- Distinct Band A∪B competing basins: 30 endpoints; per-seed greedy 5°/5% dedup gives 15 unique clusters (seed 6: 1, seed 10: 11, seed 41: 1, seed 44: 1, seed 48: 1). Source: `summary.json:per_seed[*].n_distinct_hifi_band_AB_dedup`.
- LAM/SAM split on Band A∪B distinct: LAM 28, SAM 2 (seed 41 only). Source: `summary.json:s079_split_on_hifi_band_AB`.
- Group A polhode invariants: |d_2T_rel| min 0.92% / median 3.33% / max 12.46%; |d_L2_rel| min 0.90% / median 3.33% / max 12.56%. Source: `scratch/s081_form_check.py` stdout, derived from `results/s077/basin_l_metrics.npz`.
- Group A error ranges (on the 30, before dedup): q0_err 76.6°–179.7° (median 168°), ω-dir 0.1°–88.9° (median 3.1°), ω-mag −14.85% to +6.35% (median −0.35%). Source: `scratch/s081_error_table.py` stdout.
- OOM diagnostic peaks: 1 render = 1.121 GB; 4 renders no gc = 1.554 GB; 14 renders WITH `gc.collect()` = flat 1.131 GB. Source: transcript-only probes `be74txd3v` / `blpbddb2b` (this session).

# Artefacts

- `experiments/s081_hifi_rho_bands_twin.py` — render driver with the gc.collect / Pool(16) fix.
- `results/s081/basin_hifi.npz` — per-basin q0_final, omega_final, rho_hifi, band_hifi, twin_label, q_to_truth_deg, q_to_twin_deg, om_to_truth_rel, om_to_twin_rel, L_dir_angle_deg, L_mag_rel_diff, render_mask.
- `results/s081/summary.json` — overall + by-class + by-twin + s079 split + per-seed dedup.
- `results/s081/s081_rho_bands.png` — hi-fi band stacked bars by twin label and by regime.
- `results/s081/s081_hifi_vs_surrogate.png` — surrogate-ρ vs hi-fi-ρ scatter, coloured by twin label.
- `scratch/s081_error_table.py` — q0 / ω-dir / ω-mag breakdown on the 30 distinct Band A∪B.
- `scratch/s081_form_check.py` — polhode-invariant (2T, |L|²) cross-read against s077.

# Out of scope

- Hi-fi rendering of the 495 high-MSE basins. Surrogate ρ ≥ 14 there; on this substrate the 145/145 band agreement supports the "skip high-MSE" scoping.
- Cold-start ω recovery. All 30 Band A∪B basins were found by LM seeded at truth-ω. Whether a targeted "fix ω-direction guess, sweep (|ω|, q0)" search recovers Group A from a non-truth ω-direction guess is open and is the natural next architectural question.
- Body-Y / body-Z twin enumeration. `lib/twin.py` validates only the body-X 180° twin for IS-901 (s043). No other exact LC-preserving symmetry is verified; the polhode-invariant mismatch (Group A is not same-polhode as truth) already shows the q0≈180° Group-A rows are not explained by a same-polhode symmetry, but additional discrete symmetries are not ruled out.
- Cohort scaling. 10-seed pilot only; the 100-seed cohort census remains queued.
- Group B mechanism. The 11 seed-10 endpoints with ω-direction moved 50–89° and polhode invariants off up to 38% have no proposed structural explanation yet.

# Cross-references

- `experiments/s077_l_vector_basin_sweep.md` — substrate; s081 retires its surrogate caveat.
- `experiments/s079_regime_stratified_l_basins.md` — LAM/SAM split, re-validated at hi-fi here.
- `experiments/s068_replicate_s011.md` — post-fix s011 re-run that produced the 640-basin substrate.
- `experiments/s073d_polhode_match_cluster457.md` — seed-89 cluster_457 1.1% Casimir mismatch; s081 generalises to 0.9–12.5% across 19 basins.
- `experiments/s073f_cluster457_local_geometry.md` — local-geometry stiffness probe whose STRICT-reading verdict s081 supports cohort-wide.
- `experiments/s043_twin_hifi_verify.md` — body-X twin LC equivalence verification (the only IS-901 exact symmetry).
- `concepts/rho_band.md` — ρ-band thresholds.
- `concepts/twin_degeneracy.md` — body-X twin algebra.
- `memory/project_local_window_polish_phantom_basins.md` — surrogate↔hi-fi divergence motivation for the hi-fi gate.
- `memory/feedback_pool_size_with_trimesh_gc.md` — new this session, the gc/pool-size lesson.
