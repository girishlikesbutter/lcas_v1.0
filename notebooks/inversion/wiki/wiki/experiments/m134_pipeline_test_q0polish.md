---
title: "m134 — Pipeline test of surr_q0polish_mse sort on Band-D rescues"
type: experiment
sources: ["data/results/inversion_diagnostics/rerank_experiment/pipeline_test_2026_04_22/REPORT.md",
          "data/results/inversion_diagnostics/rerank_experiment/pipeline_test_2026_04_22/comparison.json",
          "data/results/inversion_diagnostics/invert_m048_seed059/result.json",
          "data/results/inversion_diagnostics/invert_m048_seed064/result.json",
          "data/results/inversion_diagnostics/invert_m048_seed067/result.json"]
related: ["[[m133_rerank_geo_ckpt_pool]]", "[[m115_surrogate_pipeline]]",
          "[[m048-random-cohort-baseline]]", "[[surrogate-model]]",
          "[[observational-indistinguishability]]"]
created: 2026-04-22
updated: 2026-04-22
confidence: medium
---

# m134 — First end-to-end pipeline test of `surr_q0polish_mse` sort

Validates m133's offline rerank finding by actually rerunning m115 + m126 + wrappedbest under the new ω-ranking. Tests whether replacing `geo_cost` with `surr_q0polish_mse` as m103's top-3 ranker translates into real hi-fi MSE improvement on Band-D failed seeds.

## Setup

- **Trajectory source:** m048, per-seed random start_et (canonical noise realisation per the 2026-04-21 fix).
- **Observation window:** per-seed 500×dt=7.21 s, standard invert.py plumbing.
- **Target seeds:** 59, 64, 67 — the Band-D subset identified by `pick_target_seeds.py` where `surr_q0polish_mse` top-3 uniquely adds a truth-close (<20°) ω that `geo_cost` top-3 had missed. The 14 other Band-D failed seeds either already had truth in `geo_cost` top-3 (downstream m115/m126 failures) or had no truth-close ω in the 26-candidate pool at all.
- **Patch:** `m115_surrogate_pipeline.py::load_omega_candidates` accepts `M115_SORT_BY=surr_q0polish_mse`, loads per-candidate polish MSE from `rerank_experiment/seed_XXX_q0polish.json`, sorts ascending.
- **Protocol:** backed up baseline m115/m126 seed dirs → cleared checkpoints → ran `invert.py --skip-m103` with the new env var.
- **Noise metric:** ρ = √hifi/0.05, Bands A (ρ<2) / B (2-4) / C (4-8) / D (≥8).
- **Kill criterion:** 40 min/seed. Actual wall was 9.6 / 11.5 / 8.8 min.

## Results

| seed | base ρ | base band | base q0 | base w_dir | base w_mag | new ρ | new band | new q0 | new w_dir | new w_mag | Δρ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 59 | 27.63 | D | 177.6° | 118.1° | −3.50% | **1.02** | **A** | 0.21° | 0.05° | −0.00% | −26.61 |
| 64 | 30.15 | D | 156.9° | 83.7° | −5.81% | 26.92 | D | 174.1° | 50.9° | −63.90% | −3.23 |
| 67 | 33.09 | D | 171.5° | 163.4° | +0.60% | **3.56** | **B** | 179.6° | 0.3° | −0.08% | −29.53 |

**Tally:** 2/3 band-improved, 3/3 ρ-reduced, 1/3 reached Band A (noise-level).

## Per-seed mechanism

### Seed 59 — clean rescue to Band A

New top-3 ω: w_dir_err 3.65°, 34.62°, 56.19° (baseline had 27.7°, 72.3°, 60.6° — no truth-close). m115 DE from ω=3.65° found a truth-adjacent q0 basin in **3/10 starts** at q0_err=8.81°. Surrogate MSE ranked this basin #3 (two non-twin spurious basins at q0_err≈174° had slightly lower surrogate MSE), but m115's top-3 hi-fi validation included it, and m126 wrapped polish took it from q0_err=8.81° to q0_err=**0.21°**, w_dir=0.05°, w_mag=−0.00%. Final ρ=1.02 — comfortably inside Band A.

### Seed 64 — ranking fix worked, downstream DE didn't

New top-3 ω: w_dir_err 50.93°, 14.63°, 60.91°. The 14.63° truth-close ω WAS correctly placed at slot 1 by the new ranker. But m115's 10-start 3-DOF DE from ω=14.63° produced **0/10 starts below q0_err=10°**; all basins from that ω sat at hifi ~1.88-1.90. Winner chosen by surrogate MSE was basin 0 from the wrong ω=50.9° (hifi=1.81, w_mag=−64%). Marginal improvement: ρ 30.15→26.92, still Band D. **Ranking fix put truth-close ω in the hand-off; downstream m115 DE couldn't exploit a 14.63° ω offset.**

### Seed 67 — twin-type recovery, not inside Band A

New top-3 ω: w_dir_err 2.97°, 67.85°, 75.58° (very truth-close at slot 0). m115 DE found a basin with w_dir=0.3° and w_mag=−0.08% — ω direction and magnitude essentially perfect. But q0 landed at q0_err=179.6°, likely the IS-901 geometric ±X-twin direction. ρ=3.56 means the LC is distinguishable from truth at ~3.6σ; not inside Band A, but Band B is a big improvement over Band D. See [[observational-indistinguishability]] — Band B solutions ARE LC-distinguishable from truth, so this isn't a valid (q₀, ω) per the ρ<2 convention, just a big reduction in error.

## Finding: m115's ω-bridging radius is ~3-5°, not 15°

Seed 59 bridged ω_err=3.65° → truth-adjacent q0 basin in 3/10 DE starts.
Seed 64 failed to bridge ω_err=14.63° → 0/10 DE starts below q0_err=10°.
Seed 67 bridged ω_err=2.97° → ω recovered perfectly (0.3°) but q0 snapped to the twin.

So **the ranking-fix-alone strategy works only when m103 delivers truth-close ω inside ~5°.** For the Band-D seeds where m103 produces ω at 10-20° error, the pipeline still fails even with correct ranking. This motivates either (a) increasing m115 N_STARTS from 10 to 30+, (b) densifying m103 candidate generation so surviving ω candidates sit closer to truth, or (c) adding a dedicated ω-refinement stage between m103 and m115.

## Caveats

- n=3 only. Single-cost `surr_q0polish_mse` was predicted to gain ~+1 top-3 hit vs baseline across 17 seeds (9→10); observing 1 clean Band-A rescue on 3 pre-selected seeds is consistent with that.
- Seed 67's Band B verdict is honest (LC distinguishable from truth), even though the ω recovery is exact. Twin-type degeneracies are not free passes under the ρ<2 convention; they only become valid if their hi-fi LC is within noise of truth, which here it isn't.
- The K=3 triple union `{surr_q0polish_mse, surr_peak_time, surr_autocorr}` (m133 predicted 15/17 Band-D rescuable) remains end-to-end untested. That would be the next natural experiment.

## Artefacts

- Patch: `notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py` (new `M115_SORT_BY=surr_q0polish_mse` branch in `load_omega_candidates`)
- Target picker: `notebooks/inversion/14_rerank_experiment/pick_target_seeds.py`
- Driver: `notebooks/inversion/14_rerank_experiment/run_q0polish_rescue.sh`
- Comparison: `notebooks/inversion/14_rerank_experiment/compare_q0polish_rescue.py`
- Data: `data/results/inversion_diagnostics/rerank_experiment/pipeline_test_2026_04_22/` (baseline snapshots, logs, comparison.json, REPORT.md)
- Final results: `data/results/inversion_diagnostics/invert_m048_seed{059,064,067}/result.json`
