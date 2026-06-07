# RENAMES.md — Organizational Cleanup Mapping (2026-04-16)

**Purpose:** Rename 272 Python scripts (~286 result directories, 39 wiki pages) from cryptic `microNN_*` convention to semantic `m{NNN}_{semantic_name}` for clarity and maintainability.

---

## Open-decision resolutions (2026-04-16 — strategist call)

These override the "Open Decisions" section at the bottom of this file where they conflict.

1. **Three colliding `07_*` dirs** → Consolidate into a single `07_omega_winding/`. All seven scripts (m019 winding landscape, m020 multistart staircase, m021 multi-epoch winding score, m022a winding-score nudged qA, m023 L oracle test, m024 L nudge sensitivity, m025 L three leg) belong to one coherent research thread.

2. **`00_pipeline_reference/` placement** → Move to `notebooks/tutorials/` (out of `notebooks/inversion/`). These are LCAS light-curve-pipeline tutorials, not inversion experiments. Scripts stay as-is (they don't need the `m{NNN}` convention since they're not experiments).

3. **Series-06 resolution** → Delete empty `06_omega_basin/`. Rename `06_omega_bridging_benchmarks/` → `06_omega_bridging/`. Archive all seven of its scripts under `06_omega_bridging/archive/` (pre-systematic benchmark variants of m001 concepts; keep FINDINGS.md at series level).

4. **Duplicate experiment numbers in series 11 & 12** → Letter-suffix convention:
   - Canonical experiment keeps the bare number (e.g., `m089_production_pipeline.py`).
   - Variants get lowercase letter suffix (`m089d_diagnosis_pipeline.py`).
   - Existing `b`/`c` suffixes preserved where they already encode chronology (`m093b`, `m093c`, `m095b`, etc.).
   - Resolutions:
     - `micro89_diagnosis.py` → `m089d_diagnostic_pipeline.py` (diagnosis variant)
     - `micro89_production.py` → `m089_production_pipeline.py` (canonical)
     - `micro93b_extreme_peaks.py` → `m093b_extreme_peaks.py` (keep)
     - `micro93b_peak_census.py` → `m093c_peak_census.py` (renumber)
     - `micro117_harvester.py` → `m117_harvester.py` (canonical)
     - `micro117_validate.py` → `m117v_validate.py`
     - `micro118_kernel.py` → `m118_kernel.py` (canonical — the kernel-factored diagnostic is the load-bearing one)
     - `micro118_costs.py` → `m118c_costs.py`
     - `micro118_diag.py` → `m118d_diag.py`
     - `micro119_attitude_isoshell.py` → `m119_attitude_isoshell.py` (canonical, retracted per DATA_INTEGRITY_BUG.md)
     - `micro119_multiseed_aggregate.py` → `m119a_multiseed_aggregate.py` (a = aggregator)

5. **Orphan result dirs** →
   - `harvester_lever1/`, `harvester_sanity/` → `analyses/harvester_lever1/`, `analyses/harvester_sanity/`
   - `inline_omega_selection/` → `analyses/inline_omega_selection/`
   - `isoshell_viewer/` → `shared/isoshell_viewer/`
   - `wrappedbest_seed{NNN}/` (several) → folded into `m126_wrapped_pipeline/wrappedbest_seed{NNN}/`
   - `plots/` → `archive/plots/` pending inspection
   - `unsorted/` → `archive/unsorted/` pending inspection

6. **Loose files in results root** → `shared/` subdir structure:
   - `is901_brightness_table.npz` → `shared/brightness_tables/is901_brightness_table.npz`
   - `micro46_trajectories.npz` (if loose) → already under `micro46_trajectories/` dir; renamed to `shared/m046_trajectories/m046_trajectories.npz`
   - `micro48_trajectories.npz` → `shared/m048_trajectories/m048_trajectories.npz`
   - `micro104_crossing_diagnostic.npz` → folded into `m104_crossing_diagnostic/` if dir exists, else `shared/m104_crossing_diagnostic.npz`
   - `micro50_omega_landscape.{npz,json,png}` → `m050_omega_landscape/` (collate)
   - `micro59_step*_*.npz` (18 files) → fold into `m059_full_blind_pipeline/steps/`
   - `micro69b_*.npz`, `micro69c_*.npz` → fold into `m069_b_geometric_refinement/` and `m069_c_lc_comparison/`
   - Other loose `.npz` → fold into producing experiment's dir

**Result:** after Phase 2–5 complete, `data/results/inversion_diagnostics/` will have exactly 4 top-level entries beyond the `m{NNN}_*` experiment dirs: `shared/`, `analyses/`, `archive/`, and a single `inversion_diagnostics` symlink-to-self (legacy — will be removed if found safe).

---

---

## Conventions

- **Scripts**: `m{NNN}_{semantic_name}.py` (e.g., `m127_flipped_omega_search.py`)
- **Result directories**: `m{NNN}_{semantic_name}/` matching producing script
- **Wiki pages**: `m{NNN}_{semantic_name}.md` matching experiment
- **Legacy (pre-microNN or helpers)**: `archive/{original_filename}` (non-experiment scripts, workers, demos)
- **Shared datasets**: `shared/{filename}` (standalone result files, brightness tables, crossing diagnostics)
- **One-off analyses**: `analyses/{dirname}/` (orphan result dirs without a clear producing script)

---

## Scripts by Series

### 00_pipeline_reference (8 files, ALL ARCHIVE)
Tutorial notebooks demonstrating the LC pipeline. All lack experiment numbers. Flag for decision: keep in place, rename to `tutorials/`, or move outside `inversion/`.

| Old | New | Notes |
|-----|-----|-------|
| `01_inertia_calculation.py` | `archive/01_inertia_calculation.py` | Tutorial |
| `02_attitude_propagation.py` | `archive/02_attitude_propagation.py` | Tutorial |
| `03_lightcurve_inversion.py` | `archive/03_lightcurve_inversion.py` | Tutorial |
| `04_landscape_analysis.py` | `archive/04_landscape_analysis.py` | Tutorial |
| `05_basin_size_study.py` | `archive/05_basin_size_study.py` | Tutorial |
| `06_optimizer_comparison.py` | `archive/06_optimizer_comparison.py` | Tutorial |
| `07_robustness_analysis.py` | `archive/07_robustness_analysis.py` | Tutorial |
| `08_mixed_fidelity_inversion.py` | `archive/08_mixed_fidelity_inversion.py` | Tutorial |

### 01_global_search (24 files, ALL ARCHIVE)
Early exp3 basin convergence studies + helper scripts (inversion demos, workers, plotting). No microNN experiments.

| Old | New | Notes |
|-----|-----|-------|
| `exp3_basin_resume.py` | `archive/exp3_basin_resume.py` | Legacy exp3 |
| `exp3_convergence_basin.py` | `archive/exp3_convergence_basin.py` | Legacy exp3 |
| `exp3_convergence_basin_lofi.py` | `archive/exp3_convergence_basin_lofi.py` | Legacy exp3 |
| `exp3_large_budget.py` | `archive/exp3_large_budget.py` | Legacy exp3 |
| `exp3_parallel_strategies.py` | `archive/exp3_parallel_strategies.py` | Legacy exp3 |
| `exp3_single_strategy.py` | `archive/exp3_single_strategy.py` | Legacy exp3 |
| `exp_alternating.py` | `archive/exp_alternating.py` | Legacy exp |
| `exp_bruteforce.py` | `archive/exp_bruteforce.py` | Legacy exp |
| `exp_decoupled_fast.py` | `archive/exp_decoupled_fast.py` | Legacy exp |
| `exp_decoupled_grid.py` | `archive/exp_decoupled_grid.py` | Legacy exp |
| `exp_dual_annealing_6d.py` | `archive/exp_dual_annealing_6d.py` | Legacy exp |
| `exp_multistart_tight.py` | `archive/exp_multistart_tight.py` | Legacy exp |
| `exp_omega_first.py` | `archive/exp_omega_first.py` | Legacy exp |
| `exp_tight_start_proof.py` | `archive/exp_tight_start_proof.py` | Legacy exp |
| `generate_basin_plots.py` | `archive/generate_basin_plots.py` | Plotting helper |
| `inversion_demo.py` | `archive/inversion_demo.py` | Demo/reference |
| `inversion_demo_v2.py` | `archive/inversion_demo_v2.py` | Demo/reference |
| `inversion_demo_v3.py` | `archive/inversion_demo_v3.py` | Demo/reference |
| `lofi_worker.py` | `archive/lofi_worker.py` | Worker/helper |
| `run_exp3_dynamics_fidelity.py` | `archive/run_exp3_dynamics_fidelity.py` | Runner |
| `run_exp3_only.py` | `archive/run_exp3_only.py` | Runner |
| `run_exp3_timed.py` | `archive/run_exp3_timed.py` | Runner |
| `run_exp3_v3_bounded.py` | `archive/run_exp3_v3_bounded.py` | Runner |
| `test_exp3_only.py` | `archive/test_exp3_only.py` | Test |

### 02_isobrightness_filtering (14 files, ALL ARCHIVE)
Legacy exp_* experiments on brightness filtering & residual robustness. No microNN.

| Old | New | Notes |
|-----|-----|-------|
| `check_residual_correlation.py` | `archive/check_residual_correlation.py` | Legacy exp |
| `exp_brightness_filter.py` | `archive/exp_brightness_filter.py` | Legacy exp |
| `exp_explore_filtering.py` | `archive/exp_explore_filtering.py` | Legacy exp |
| `exp_isobrightness.py` | `archive/exp_isobrightness.py` | Legacy exp |
| `exp_mixed_fidelity_search.py` | `archive/exp_mixed_fidelity_search.py` | Legacy exp |
| `exp_mixed_fidelity_search_v2.py` | `archive/exp_mixed_fidelity_search_v2.py` | Legacy exp v2 |
| `exp_parallel_lofi.py` | `archive/exp_parallel_lofi.py` | Legacy exp |
| `exp_residual_robustness.py` | `archive/exp_residual_robustness.py` | Legacy exp |
| `exp_residual_vs_error.py` | `archive/exp_residual_vs_error.py` | Legacy exp |
| `exp_sequential_filter.py` | `archive/exp_sequential_filter.py` | Legacy exp |
| `exp_sequential_filter_v2.py` | `archive/exp_sequential_filter_v2.py` | Legacy exp v2 |
| `exp_sequential_filter_v3.py` | `archive/exp_sequential_filter_v3.py` | Legacy exp v3 |
| `exp_triplet_matching.py` | `archive/exp_triplet_matching.py` | Legacy exp |
| `exp_true_in_set.py` | `archive/exp_true_in_set.py` | Legacy exp |

### 03_omega_bridging (12 files, ALL ARCHIVE)
Legacy exp_* on omega bridging, chain matching, q0/omega inference. No microNN.

| Old | New | Notes |
|-----|-----|-------|
| `exp_chain7_tight.py` | `archive/exp_chain7_tight.py` | Legacy exp |
| `exp_conservation_and_L_param.py` | `archive/exp_conservation_and_L_param.py` | Legacy exp |
| `exp_culling_vs_dt.py` | `archive/exp_culling_vs_dt.py` | Legacy exp |
| `exp_forward_prop.py` | `archive/exp_forward_prop.py` | Legacy exp |
| `exp_omega_bridge.py` | `archive/exp_omega_bridge.py` | Legacy exp |
| `exp_omega_bridge_v2.py` | `archive/exp_omega_bridge_v2.py` | Legacy exp v2 |
| `exp_omega_chain4.py` | `archive/exp_omega_chain4.py` | Legacy exp |
| `exp_omega_chain4_v2.py` | `archive/exp_omega_chain4_v2.py` | Legacy exp v2 |
| `exp_q0_omega_analytic.py` | `archive/exp_q0_omega_analytic.py` | Legacy exp |
| `exp_q0_omega_nm.py` | `archive/exp_q0_omega_nm.py` | Legacy exp |
| `exp_q0_omega_opt.py` | `archive/exp_q0_omega_opt.py` | Legacy exp |
| `exp_two_epoch_matching.py` | `archive/exp_two_epoch_matching.py` | Legacy exp |

### 04_basin_characterization (4 files, ALL ARCHIVE)
Early basin sanity checks. No microNN.

| Old | New | Notes |
|-----|-----|-------|
| `exp00_timing.py` | `archive/exp00_timing.py` | Legacy exp |
| `exp01_sanity.py` | `archive/exp01_sanity.py` | Legacy exp |
| `exp02_attitude_basin.py` | `archive/exp02_attitude_basin.py` | Legacy exp |
| `exp03_omega_basin.py` | `archive/exp03_omega_basin.py` | Legacy exp |

### 05_peak_graph_pipeline (27 micro + 1 archive)
First wave of microNN experiments: brightness degeneracy, peak constraint validation, bridge pipeline, joint q0-omega inference, dense sampling, peak graph methods.

| Old | New | Notes |
|-----|-----|-------|
| `micro01_brightness_degeneracy.py` | `m001_brightness_degeneracy.py` | Brightness ambiguity quantization |
| `micro02_peak_constraint.py` | `m002_peak_constraint.py` | Peak anchor constraints |
| `micro03_bridge_validation.py` | `m003_bridge_validation.py` | Omega bridge validation |
| `micro04_bridge_pipeline.py` | `m004_bridge_pipeline.py` | Complete bridge pipeline |
| `micro05_hifi_candidates.py` | `m005_hifi_candidates.py` | Hi-fi candidate filtering |
| `micro06_large_sampling.py` | `m006_large_sampling.py` | Large SO(3) grid |
| `micro07_propagate_and_match.py` | `m007_propagate_and_match.py` | Forward propagation + LC matching |
| `micro07a_omega_drift.py` | `m007a_omega_drift.py` | Omega drift during propagation |
| `micro08_joint_pipeline.py` | `m008_joint_pipeline.py` | Joint q0-omega optimization |
| `micro09_isobrightness_optim.py` | `m009_isobrightness_optim.py` | Isoline constraint optimization |
| `micro10_scaled_isobrightness.py` | `m010_scaled_isobrightness.py` | Scaled isoline filtering |
| `micro10b_dense_sampling.py` | `m010b_dense_sampling.py` | Dense peak sampling variant |
| `micro11_dense_joint_pipeline.py` | `m011_dense_joint_pipeline.py` | Dense joint inference |
| `micro12_interpolated_peaks.py` | `m012_interpolated_peaks.py` | Interpolated peak extraction |
| `micro13_graph_pipeline.py` | `m013_graph_pipeline.py` | Peak connectivity graph |
| `micro13b_nudge_test.py` | `m013b_nudge_test.py` | Discrete nudge perturbation test |
| `micro13c_lc_compare.py` | `m013c_lc_compare.py` | Light curve comparison viz |
| `micro14_hifi_rescore.py` | `m014_hifi_rescore.py` | Hi-fi candidate rescoring |
| `micro14_parallel_test.py` | `m014_parallel_test.py` | Parallel execution test (duplicate num) |
| `micro15_min_omega.py` | `m015_min_omega.py` | Minimum omega estimation |
| `micro15b_alpha10.py` | `m015b_alpha10.py` | Alpha=10 deg constraint variant |
| `micro15b_alpha_sweep.py` | `m015b_alpha_sweep.py` | Alpha sweep parameter study |
| `micro16_dip_constraint.py` | `m016_dip_constraint.py` | Dip angle constraint |
| `micro16b_dip_leg1.py` | `m016b_dip_leg1.py` | Leg 1 dip constraint |
| `micro16c_trough_constrained_bridge.py` | `m016c_trough_constrained_bridge.py` | Trough constraint for bridge |
| `micro17_staircase_omega.py` | `m017_staircase_omega.py` | Staircase omega profile |
| `micro18_L_consistency.py` | `m018_L_consistency.py` | Angular momentum conservation check |
| `omega_basin_characterisation.py` | `archive/omega_basin_characterisation.py` | Helper analysis |

### 06_omega_basin (4 files, ALL ARCHIVE)
Basin magnitude & direction studies (early, pre-microNN naming).

| Old | New | Notes |
|-----|-----|-------|
| `basin_01_sanity.py` | `archive/basin_01_sanity.py` | Basin sanity checks |
| `basin_02_magnitude.py` | `archive/basin_02_magnitude.py` | Basin magnitude analysis |
| `basin_03_direction.py` | `archive/basin_03_direction.py` | Basin direction robustness |
| `basin_04_q_degradation.py` | `archive/basin_04_q_degradation.py` | Q attitude degradation |

### 06_omega_bridging_benchmarks (4 micro + 3 archive)
Omega BVP benchmarks + peak recovery micro-experiments.

| Old | New | Notes |
|-----|-----|-------|
| `micro01_peak_omega_recovery.py` | `m001_peak_omega_recovery.py` | Peak-based omega estimation |
| `micro01b_peak_q_nudge.py` | `m001b_peak_q_nudge.py` | Q nudge perturbation |
| `micro01c_peak_q_nudge_lofi.py` | `m001c_peak_q_nudge_lofi.py` | Lo-fi nudge variant |
| `micro01d_animations.py` | `m001d_animations.py` | Animation generation |
| `bench_omega_bvp.py` | `archive/bench_omega_bvp.py` | Benchmark runner |
| `bench_omega_bvp_noisy.py` | `archive/bench_omega_bvp_noisy.py` | Noisy benchmark |
| `bench_omega_bvp_parallel.py` | `archive/bench_omega_bvp_parallel.py` | Parallel benchmark |

**NOTE:** Series 06 has two similarly-numbered directories. Recommend renaming this to `06b_omega_bridging_benchmarks` to disambiguate from `06_omega_basin`.

### 07_L_conservation (3 micro)
Angular momentum conservation tests.

| Old | New | Notes |
|-----|-----|-------|
| `micro23_L_oracle_test.py` | `m023_L_oracle_test.py` | L conservation with oracle |
| `micro24_L_nudge_sensitivity.py` | `m024_L_nudge_sensitivity.py` | L sensitivity to q0 nudges |
| `micro25_L_three_leg.py` | `m025_L_three_leg.py` | L conservation over 3-epoch chain |

### 07_multi_epoch_scoring (2 micro)
Multi-epoch winding number scoring for disambiguation.

| Old | New | Notes |
|-----|-----|-------|
| `micro21_multi_epoch_winding_score.py` | `m021_multi_epoch_winding_score.py` | Winding number across epochs |
| `micro22a_winding_score_nudged_qA.py` | `m022a_winding_score_nudged_qA.py` | Nudged Q variant |

### 07_winding_enumeration (2 micro)
Winding number landscape and multistart exploration.

| Old | New | Notes |
|-----|-----|-------|
| `micro19_winding_landscape.py` | `m019_winding_landscape.py` | Winding landscape mapping |
| `micro20_multistart_staircase.py` | `m020_multistart_staircase.py` | Multistart from staircase seeds |

**COLLISION FLAG:** Three `07_*` directories share different sub-topics (L_conservation, multi_epoch_scoring, winding_enumeration). Recommend consolidating into one `07_multi_parameter_inference/` or keeping separate with disambiguation in naming (e.g. `07_L_conservation`, `07b_multi_epoch_scoring`, `07c_winding_enumeration`).

### 08_integration (8 micro)
Integration of multi-epoch, peak shape, and hi-fi pruning filters into unified pipeline.

| Old | New | Notes |
|-----|-----|-------|
| `micro26_integration_oracle.py` | `m026_integration_oracle.py` | Oracle integration baseline |
| `micro26b_direction_aware_dedup.py` | `m026b_direction_aware_dedup.py` | Direction-aware deduplication |
| `micro26c_oracle_direction_dedup.py` | `m026c_oracle_direction_dedup.py` | Oracle direction dedup |
| `micro27_bridge_screening.py` | `m027_bridge_screening.py` | Bridge candidate screening |
| `micro28_integration_nudged.py` | `m028_integration_nudged.py` | Integration with q0 nudges |
| `micro29_peak_shape_filter.py` | `m029_peak_shape_filter.py` | Peak shape-based filtering |
| `micro30_hifi_pruning.py` | `m030_hifi_pruning.py` | Hi-fi validation pruning |
| `micro40_bridge_nstarts_sweep.py` | `m040_bridge_nstarts_sweep.py` | Bridge multistart sweep |

### 09_glint_analysis (15 micro + 1 archive)
Specular glint identification, PAB alignment, facet normal seeding, and trajectory generation.

| Old | New | Notes |
|-----|-----|-------|
| `micro34_pab_alignment.py` | `m034_pab_alignment.py` | Phase angle bisector alignment |
| `micro35_multi_trajectory_pab.py` | `m035_multi_trajectory_pab.py` | PAB multi-trajectory analysis |
| `micro36_pab_candidate_filter.py` | `m036_pab_candidate_filter.py` | PAB-based filtering |
| `micro37_brdf_glint_profile.py` | `m037_brdf_glint_profile.py` | BRDF glint profiling |
| `micro38_pab_seeded_candidates.py` | `m038_pab_seeded_candidates.py` | PAB-seeded candidate generation |
| `micro39_glint_identification.py` | `m039_glint_identification.py` | Specular glint detection |
| `micro41_glint_classification.py` | `m041_glint_classification.py` | Glint magnitude classification |
| `micro42_glint_anchored_nlp.py` | `m042_glint_anchored_nlp.py` | Glint anchor NLP solver |
| `micro42b_anchor_hypothesis_sweep.py` | `m042b_anchor_hypothesis_sweep.py` | Phi-sweep hypothesis test |
| `micro43_focused_pab_seeding.py` | `m043_focused_pab_seeding.py` | Focused PAB seeding |
| `micro44_normal_sphere.py` | `m044_normal_sphere.py` | Normal vector sphere parameterization |
| `micro45_glint_filter_basin.py` | `m045_glint_filter_basin.py` | Glint basin width analysis |
| `micro46_generate_trajectories.py` | `m046_generate_trajectories.py` | Synthetic trajectory set generation |
| `micro47_glint_statistics.py` | `m047_glint_statistics.py` | Glint frequency statistics |
| `micro48_generate_trajectories_v2.py` | `m048_generate_trajectories_v2.py` | Trajectory generation v2 |
| `plot_micro34_pab.py` | `archive/plot_micro34_pab.py` | Plotting helper |

### 10_glint_inversion (29 micro)
End-to-end glint-based inversion pipeline variants: omega landscape, phi-sweep scoring, bridge omega inference, bodyframe solutions, L-conserving tests, staged pipelines.

| Old | New | Notes |
|-----|-----|-------|
| `micro49_generalization_and_census.py` | `m049_generalization_and_census.py` | Anchor generalization + census |
| `micro50_omega_landscape.py` | `m050_omega_landscape.py` | Omega magnitude grid |
| `micro50b_full_epoch_omega.py` | `m050b_full_epoch_omega.py` | Full epoch omega grid |
| `micro50c_omega_multistart.py` | `m050c_omega_multistart.py` | Omega multistart optimizer |
| `micro50d_omega_de.py` | `m050d_omega_de.py` | Differential evolution omega search |
| `micro51_end_to_end.py` | `m051_end_to_end.py` | Full end-to-end pipeline |
| `micro51b_hifi_disambiguation.py` | `m051b_hifi_disambiguation.py` | Hi-fi candidate disambiguation |
| `micro52_spectral_omega.py` | `m052_spectral_omega.py` | Spectral peak-count omega |
| `micro52b_phisweep_omega_scorer.py` | `m052b_phisweep_omega_scorer.py` | Phi-sweep scoring method |
| `micro52c_lc_omega_scorer.py` | `m052c_lc_omega_scorer.py` | Light curve omega scorer |
| `micro53_bridge_omega_from_circles.py` | `m053_bridge_omega_from_circles.py` | Circle-fit omega bridge |
| `micro53b_bridge_lc_scored.py` | `m053b_bridge_lc_scored.py` | Bridge with LC scoring |
| `micro53c_bridge_all_lc.py` | `m053c_bridge_all_lc.py` | Bridge candidates all-LC eval |
| `micro54_full_blind_pipeline.py` | `m054_full_blind_pipeline.py` | Fully blind end-to-end |
| `micro55_bridge_coverage_diagnostic.py` | `m055_bridge_coverage_diagnostic.py` | Bridge coverage analysis |
| `micro55b_alignment_bridge.py` | `m055b_alignment_bridge.py` | Alignment-constrained bridge |
| `micro56_alignment_nm_pipeline.py` | `m056_alignment_nm_pipeline.py` | Alignment NM refinement |
| `micro56b_antiglint_nm.py` | `m056b_antiglint_nm.py` | Anti-glint NM polish |
| `micro57_bodyframe_lc_nm.py` | `m057_bodyframe_lc_nm.py` | Bodyframe LC NM solver |
| `micro58_bodyframe_phisweep.py` | `m058_bodyframe_phisweep.py` | Bodyframe phi-sweep |
| `micro58b_anchor_centered.py` | `m058b_anchor_centered.py` | Anchor-centered bodyframe |
| `micro59_L_conserving_test.py` | `m059_L_conserving_test.py` | L-conservation validation |
| `micro59_selection_diagnostic.py` | `m059_selection_diagnostic.py` | Candidate selection metrics |
| `micro59_step1_arrival_filter.py` | `m059_step1_arrival_filter.py` | Stage 1: arrival filtering |
| `micro59_step1_brightness_filter.py` | `m059_step1_brightness_filter.py` | Stage 1: brightness filtering |
| `micro59_step2_stratified_lc.py` | `m059_step2_stratified_lc.py` | Stage 2: stratified LC eval |
| `micro59_step3_cheap_phisweep.py` | `m059_step3_cheap_phisweep.py` | Stage 3: fast phi-sweep |
| `micro59_step4_omega_grid.py` | `m059_step4_omega_grid.py` | Stage 4: omega grid search |
| `micro59a_checkpoint_stage2.py` | `m059a_checkpoint_stage2.py` | Stage 2 checkpoint/resume |

### 11_casadi_formulation (80 micro + 1 archive)
**Largest series.** CasADi NLP formulations for full geometric + dynamic inversion. Covers L-param basins, windowed scoring, multistart, full pipelines, geometric refinement, glint-based optimization, twin disambiguation, normal identification, basin width studies, BRDF integration, confidence estimation, final production pipelines.

| Old | New | Sample purpose |
|-----|-----|---------|
| `micro60_L_param_basin.py` | `m060_L_param_basin.py` | L-parameter sensitivity basin |
| `micro61_basin_vs_window.py` | `m061_basin_vs_window.py` | Basin size vs LC window |
| `micro62_windowed_joint_basin.py` | `m062_windowed_joint_basin.py` | Joint q0-omega windowed basin |
| `micro63_grid_progressive.py` | `m063_grid_progressive.py` | Progressive grid refinement |
| `micro63b_multistart_medium.py` | `m063b_multistart_medium.py` | Medium-scale multistart |
| `micro64_full_pipeline.py` | `m064_full_pipeline.py` | Complete inversion pipeline |
| `micro65_real_pipeline.py` | `m065_real_pipeline.py` | Real-data pipeline |
| `micro66_alignment_minima.py` | `m066_alignment_minima.py` | Sun-alignment minima |
| `micro67_glint_omega_search.py` | `m067_glint_omega_search.py` | Glint-constrained omega |
| `micro68_full_pipeline.py` | `m068_full_pipeline.py` | Full NLP pipeline |
| `micro69_fixed_scoring.py` | `m069_fixed_scoring.py` | Scoring table fix |
| `micro69_full_pipeline.py` | `m069_full_pipeline.py` | Full pipeline variant |
| `micro69b_geometric_refinement.py` | `m069b_geometric_refinement.py` | Geometric constraint refinement |
| `micro69c_lc_comparison.py` | `m069c_lc_comparison.py` | LC comparison visualization |
| `micro70_full_pipeline.py` | `m070_full_pipeline.py` | End-to-end with parallel NM |
| `micro71_geometric_selection.py` | `m071_geometric_selection.py` | Geometric feature selection |
| `micro71b_wrong_winner_lc.py` | `m071b_wrong_winner_lc.py` | Diagnostic: incorrect ranking |
| `micro71c_symmetry_check.py` | `m071c_symmetry_check.py` | Twin symmetry validation |
| `micro71d_symmetry_bodyframe.py` | `m071d_symmetry_bodyframe.py` | Bodyframe symmetry check |
| `micro71e_correct_twin.py` | `m071e_correct_twin.py` | Correct twin identification |
| `micro71f_twin_lc_plot.py` | `m071f_twin_lc_plot.py` | Twin LC comparison plot |
| `micro72_mvp_pipeline.py` | `m072_mvp_pipeline.py` | MVP (minimum viable product) |
| `micro72b_twin_animation.py` | `m072b_twin_animation.py` | Twin trajectory animation |
| `micro72c_peak_filter_test.py` | `m072c_peak_filter_test.py` | Peak filtering validation |
| `micro73_alpha_pipeline.py` | `m073_alpha_pipeline.py` | Alpha-parameterized pipeline |
| `micro74_seed27_diagnosis.py` | `m074_seed27_diagnosis.py` | Seed 27 pathology analysis |
| `micro74b_peak_normal_table.py` | `m074b_peak_normal_table.py` | Peak-normal correspondence |
| `micro74c_open_normal_test.py` | `m074c_open_normal_test.py` | Open normal hypothesis test |
| `micro75_beta_pipeline.py` | `m075_beta_pipeline.py` | Beta-parameterized pipeline |
| `micro76_scoring_table.py` | `m076_scoring_table.py` | Scoring metric table |
| `micro76b_phi_resolution.py` | `m076b_phi_resolution.py` | Phi angle resolution study |
| `micro76c_anti_alignment.py` | `m076c_anti_alignment.py` | Anti-alignment test |
| `micro77_beta_pipeline.py` | `m077_beta_pipeline.py` | Beta pipeline v2 |
| `micro78_normal_sequence.py` | `m078_normal_sequence.py` | Normal identification sequence |
| `micro79_hypothesis_seed018.py` | `m079_hypothesis_seed018.py` | Seed 18 hypothesis test |
| `micro80_brightness_cost.py` | `m080_brightness_cost.py` | Brightness-based cost function |
| `micro81_skip_nm.py` | `m081_skip_nm_nelder_mead.py` | Skip NM refinement test |
| `micro82_brightness_pipeline.py` | `m082_brightness_pipeline.py` | Brightness constraint pipeline |
| `micro83_filtered_alignment.py` | `m083_filtered_alignment.py` | Alignment with filtering |
| `micro84_expected_dot.py` | `m084_expected_dot_product.py` | Expected alignment dot |
| `micro85_local_lofi_refine.py` | `m085_local_lofi_refine.py` | Local lo-fi refinement |
| `micro86_fine_grid.py` | `m086_fine_grid_search.py` | Fine-resolution grid |
| `micro87_fast_grid.py` | `m087_fast_grid_search.py` | Fast grid evaluation |
| `micro88_fast_beta.py` | `m088_fast_beta_estimation.py` | Quick beta solve |
| `micro89_diagnosis.py` | `m089_diagnostic_pipeline.py` | Diagnostic mode |
| `micro89_production.py` | `m089_production_pipeline.py` | Production mode (duplicate num) |
| `micro90_robust_peaks.py` | `m090_robust_peak_selection.py` | Peak robustness study |
| `micro91_twin_test.py` | `m091_twin_state_test.py` | Twin state identification |
| `micro92_twin_axis_viz.py` | `m092_twin_axis_visualization.py` | Axis flip visualization |
| `micro93_expected_dot_nm.py` | `m093_expected_dot_nelder_mead.py` | NM refinement on alignment |
| `micro93b_extreme_peaks.py` | `m093b_extreme_peak_analysis.py` | Extreme peak behavior |
| `micro93b_peak_census.py` | `m093b_peak_census.py` | Peak statistics census (duplicate) |
| `micro93c_feature_analysis.py` | `m093c_feature_analysis.py` | Feature importance |
| `micro94_brdf_cost.py` | `m094_brdf_cost_function.py` | BRDF-integrated cost |
| `micro95_grid_cost_diagnosis.py` | `m095_grid_cost_diagnostic.py` | Cost landscape diagnosis |
| `micro95b_expanded_nm_pool.py` | `m095b_expanded_nm_pool.py` | Larger NM candidate pool |
| `micro95c_geo_hifi_selection.py` | `m095c_geometric_hifi_selection.py` | Geometric hi-fi picking |
| `micro95d_short_window_hifi.py` | `m095d_short_window_hifi.py` | Window-size sensitivity |
| `micro95e_integrated.py` | `m095e_integrated_pipeline.py` | Full integrated version |
| `micro96_exp1_oracle_grid.py` | `m096_exp1_oracle_grid.py` | Exp 1: Oracle grid baseline |
| `micro96_exp2_basin_width.py` | `m096_exp2_basin_width.py` | Exp 2: Basin width study |
| `micro96_exp3_normal_id.py` | `m096_exp3_normal_identification.py` | Exp 3: Normal finding |
| `micro96_exp4_lofi_discrim.py` | `m096_exp4_lofi_discrimination.py` | Exp 4: Lo-fi ranking |
| `micro96_exp5_omega_est.py` | `m096_exp5_omega_estimation.py` | Exp 5: Omega recovery |
| `micro96_stage1_analysis.py` | `m096_stage1_analysis.py` | Stage 1 metrics |
| `micro96_stage1_constraints.py` | `m096_stage1_constraints.py` | Stage 1 constraint tuning |
| `micro97a_lofi_rerank.py` | `m097a_lofi_candidate_reranking.py` | Lo-fi re-ranking |
| `micro97b_lofi_phisweep.py` | `m097b_lofi_phi_sweep.py` | Lo-fi phi parametrization |
| `micro97c_combined_ranking.py` | `m097c_combined_ranking.py` | Fused ranking |
| `micro98_nm300_pipeline.py` | `m098_nm300_polish_pipeline.py` | 300-start NM refinement |
| `micro99_batch.py` | `m099_batch_evaluation.py` | Batch mode |
| `micro99_nm300_2kgrid.py` | `m099_nm300_2k_grid.py` | 2k grid + 300 NM |
| `compare_experiments.py` | `archive/compare_experiments.py` | Comparison helper |

**NOTE:** Duplicate file numbers: `micro89` (diagnosis & production), `micro93b` (2 files). Recommend resolving: `m089_diagnostic_pipeline.py`, `m089_production_pipeline.py` or `m089p_production_pipeline.py`; `m093b_extreme_peak_analysis.py`, `m093b_peak_census.py` or `m093b_census.py`, `m093d_peak_census.py`.

### 12_brightness_surface (31 micro + 3 archive)
IPL (interpolated brightness) surface, surrogate model training, unified pipelines, flipped-omega tests, warmstart polish, dense grids.

| Old | New | Purpose |
|-----|-----|---------|
| `micro105_pairwise.py` | `m105_pairwise_ipl.py` | Pairwise IPL computation |
| `micro106_pairwise_vec.py` | `m106_pairwise_vec_ipl.py` | Vectorized IPL |
| `micro107_ipl_cost.py` | `m107_ipl_cost_function.py` | IPL cost landscape |
| `micro108_ipl_central.py` | `m108_ipl_central_difference.py` | Central diff IPL |
| `micro109_ipl_phi_diagnostic.py` | `m109_ipl_phi_diagnostic.py` | Phi parameter diagnosis |
| `micro110_hifi_phi.py` | `m110_hifi_phi_study.py` | Hi-fi phi validation |
| `micro111_shadow_isoshell.py` | `m111_shadow_isoshell.py` | Shadow + isoshell integration |
| `micro111b_shadow_epoch_hifi.py` | `m111b_shadow_epoch_hifi.py` | Multi-epoch shadow hi-fi |
| `micro112_bestanchor.py` | `m112_bestanchor_selection.py` | Best anchor picking |
| `micro113_de_attitude.py` | `m113_de_attitude_search.py` | Differential evolution attitude |
| `micro114_surrogate_multistart.py` | `m114_surrogate_multistart.py` | Surrogate + multistart |
| `micro115_surrogate_pipeline.py` | `m115_surrogate_pipeline.py` | Full surrogate inversion |
| `micro116_unified.py` | `m116_unified_formulation.py` | Unified framework |
| `micro117_harvester.py` | `m117_result_harvester.py` | Result extraction tool |
| `micro117_validate.py` | `m117_validate_pipeline.py` | Validation harness (duplicate) |
| `micro118_costs.py` | `m118_cost_comparison.py` | Cost function comparison |
| `micro118_diag.py` | `m118_diagnostic_mode.py` | Diagnostic analysis (duplicate) |
| `micro118_kernel.py` | `m118_kernel_computation.py` | Kernel matrix (duplicate) |
| `micro119_attitude_isoshell.py` | `m119_attitude_isoshell.py` | Attitude isoshell surface |
| `micro119_multiseed_aggregate.py` | `m119_multiseed_aggregate.py` | Cross-seed aggregation (duplicate) |
| `micro119v2_attitude_isoshell.py` | `m119v2_attitude_isoshell.py` | v2 of m119 |
| `micro120_tumbling_competitors.py` | `m120_tumbling_competitors.py` | Tumbling state exploration |
| `micro121_basin_width.py` | `m121_basin_width_metric.py` | Basin extent measurement |
| `micro122_hessian_at_truth.py` | `m122_hessian_curvature.py` | Hessian at ground truth |
| `micro123_lbfgs_polish.py` | `m123_lbfgs_polish.py` | L-BFGS-B final refinement |
| `micro124_hifi_validate.py` | `m124_hifi_validate.py` | Hi-fi solution validation |
| `micro125_keep_better_inline.py` | `m125_keep_better_inline.py` | Inline improvement logic |
| `micro126_wrapped_pipeline.py` | `m126_wrapped_pipeline.py` | Complete end-to-end wrapper |
| `micro127_flipped_omega_search.py` | `m127_flipped_omega_search.py` | Negated omega basin test |
| `micro128_warmstart_polish.py` | `m128_warmstart_polish.py` | Warm-start refinement |
| `micro129_densegrid.py` | `m129_dense_grid_eval.py` | Fine-grained grid evaluation |
| `extract_all_epoch_ipl.py` | `archive/extract_all_epoch_ipl.py` | IPL extraction helper |
| `inline_omega_selection_test.py` | `archive/inline_omega_selection_test.py` | Inline omega test |
| `ipl_census.py` | `archive/ipl_census.py` | IPL statistics |

**DUPLICATES FLAG:** `micro118` (3 files), `micro119` (2 files), `micro117` (2 files). Recommend: `m118_cost_comparison.py`, `m118_diagnostic_mode.py` → `m118d_diagnostic_mode.py`, `m118_kernel_computation.py` → `m118k_kernel_computation.py`; `m119_attitude_isoshell.py`, `m119_multiseed_aggregate.py` → `m119m_multiseed_aggregate.py`; `m117_result_harvester.py`, `m117_validate_pipeline.py` → `m117v_validate_pipeline.py`.

---

## Result Directories Summary

**Total:** ~286 directories under `data/results/inversion_diagnostics/` (mostly named `microNNN_*` or `microNNN_seed_NNN`).

### Auto-mapped directories (produce script matched)
The majority follow the pattern `microNNN_*` or `microNNN_seed_NNN/`, and map directly to the producing script's `m{NNN}_{semantic}` name.

**Examples:**
- `micro01d_animations/` → `m001d_animations/`
- `micro70_full_pipeline/` → `m070_full_pipeline/`
- `micro127_flipped_omega/` → `m127_flipped_omega_search/`
- `micro90_seed006/` → `m090_robust_peak_selection/seed006/` (subdirs within exp result)
- `wrappedbest_seed000/` → `m126_wrapped_pipeline/wrappedbest_seed000/` (output subdir)

### Orphan/shared directories (flagged for manual review)
| Dir | Proposed location | Notes |
|-----|-------------------|-------|
| `harvester_lever1/` | `analyses/harvester_lever1/` | One-off analysis output |
| `harvester_sanity/` | `analyses/harvester_sanity/` | Test/validation output |
| `inline_omega_selection/` | `analyses/inline_omega_selection/` | Standalone study |
| `isoshell_viewer/` | `shared/isoshell_viewer/` | Visualization tool output |
| `micro49_generalization_and_census.json` (loose file) | `shared/micro049_generalization_census.json` | Standalone result |
| `micro50_omega_landscape.json`, `.npz`, `.png` (loose files) | `shared/micro050_omega_landscape/` | Collate related files |
| `plots/` | `analyses/plots/` or `shared/plots/` | Visualization collection |
| `unsorted/` | `archive/unsorted/` or flag for cleanup | Temp/uncategorized |
| `is901_brightness_table.npz` (loose file) | `shared/is901_brightness_table.npz` | Satellite-specific asset |
| `micro104_crossing_diagnostic.npz` (loose file) | `shared/micro104_crossing_diagnostic.npz` | Standalone diagnostic |

---

## Wiki Experiment Pages

All 39 pages under `notebooks/inversion/wiki/wiki/experiments/` should be renamed to match the script naming scheme.

| Old | New | Covers |
|-----|-----|--------|
| `micro70.md` | `m070_full_pipeline.md` | Parallel NM pipeline |
| `micro73.md` | `m073_alpha_pipeline.md` | Alpha-parameterized study |
| `micro87-88.md` | `m087_m088_fast_grid.md` | Fast grid variants |
| `micro90.md` | `m090_robust_peak_selection.md` | Peak robustness |
| `micro91.md` | `m091_twin_state_test.md` | Twin identification |
| `micro92.md` | `m092_twin_axis_visualization.md` | Axis flip visualization |
| `micro93.md` | `m093_expected_dot_nelder_mead.md` | NM alignment refinement |
| `micro94.md` | `m094_brdf_cost_function.md` | BRDF cost |
| `micro95.md` | `m095_grid_cost_diagnostic.md` | Cost landscape |
| `micro96.md` | `m096_exp1_oracle_grid.md` | Oracle baseline + 5-exp suite |
| `micro97.md` | `m097_candidate_ranking.md` | Ranking strategies |
| `micro98-99.md` | `m098_m099_nm_grid_pipeline.md` | NM + grid search |
| `micro100-101.md` | `m100_m101_batch_multi_phi.md` | Batch + multi-phi |
| `micro102.md` | `m102_fullmse.md` | Full MSE cost |
| `micro103.md` | `m103_hybrid.md` | Hybrid cost |
| `micro104.md` | `m104_crossing_diagnostic.md` | Crossing event analysis |
| `micro106.md` | `m106_pairwise_vec_ipl.md` | Vectorized IPL |
| `micro107-108.md` | `m107_m108_ipl_cost.md` | IPL cost variants |
| `micro109.md` | `m109_ipl_phi_diagnostic.md` | Phi parameter study |
| `micro110.md` | `m110_hifi_phi_study.md` | Hi-fi phi validation |
| `micro111.md` | `m111_shadow_isoshell.md` | Shadow integration |
| `micro112.md` | `m112_bestanchor_selection.md` | Anchor selection |
| `micro113.md` | `m113_de_attitude_search.md` | DE attitude optimization |
| `micro114.md` | `m114_surrogate_multistart.md` | Surrogate + multistart |
| `micro115.md` | `m115_surrogate_pipeline.md` | Full surrogate inversion |
| `micro117.md` | `m117_result_harvester.md` | Result extraction |
| `micro118.md` | `m118_cost_comparison.md` | Cost function comparison |
| `micro119.md` | `m119_attitude_isoshell.md` | Attitude surface exploration |
| `micro119v2.md` | `m119v2_attitude_isoshell.md` | v2 improvements |
| `micro120.md` | `m120_tumbling_competitors.md` | Competing tumbling states |
| `micro121.md` | `m121_basin_width_metric.md` | Basin measurement |
| `micro122.md` | `m122_hessian_curvature.md` | Hessian analysis |
| `micro123.md` | `m123_lbfgs_polish.md` | L-BFGS-B refinement |
| `micro124.md` | `m124_hifi_validate.md` | Solution validation |
| `micro125.md` | `m125_keep_better_inline.md` | Improvement logic |
| `micro126.md` | `m126_wrapped_pipeline.md` | End-to-end wrapper |
| `micro127.md` | `m127_flipped_omega_search.md` | Negated omega test |
| `micro128.md` | `m128_warmstart_polish.md` | Warm-start refinement |
| `micro129.md` | `m129_dense_grid_eval.md` | Dense grid evaluation |

---

## Open Decisions Requiring Escalation

### 1. Three colliding `07_*` directories
   - `07_L_conservation/` (3 micro experiments on angular momentum conservation)
   - `07_multi_epoch_scoring/` (2 micro experiments on multi-epoch winding scoring)
   - `07_winding_enumeration/` (2 micro experiments on winding landscapes)
   
   **Decision needed:** 
   - Keep separate with directory numbering disambiguation? (e.g. `07_L_conservation`, `07b_multi_epoch_scoring`, `07c_winding_enumeration`)
   - Or consolidate into one `07_multi_parameter_inference/` superdir with subdirs?

### 2. `00_pipeline_reference/` placement
   - Currently 8 tutorial notebooks (no experiment numbers).
   
   **Decision needed:**
   - Archive as `archive/*` (keeps them out of active research)?
   - Move to `tutorials/` subdir (outside `inversion/`)?
   - Rename to `00_tutorials/` and keep in place?
   - Delete if superseded by external documentation?

### 3. Duplicate micro numbers in two series
   - **06_omega_bridging_benchmarks** has `micro01*` which collide with **05_peak_graph_pipeline** `micro01*`.
   - **06_omega_basin** has `basin_*` which is pre-microNN naming.
   
   **Decision needed:**
   - Merge `06_omega_bridging_benchmarks` into `06_omega_basin` and rename to `06b_omega_basin_benchmarks`?
   - Or renumber `micro01*` in 06_omega_bridging_benchmarks to avoid collision (e.g., `micro601*`)?

### 4. Duplicate numeric IDs within series 11 & 12
   - `micro89` (diagnosis + production)
   - `micro93b` (2 files: extreme_peaks, peak_census)
   - `micro118` (3 files: costs, diag, kernel)
   - `micro119` (2 files: attitude_isoshell, multiseed_aggregate)
   - `micro117` (2 files: harvester, validate)
   
   **Decision needed:** Rename one file per collision (e.g., `m089_diagnostic_pipeline.py`, `m089p_production_pipeline.py`).

### 5. Result directory orphans
   Directories without a clear producing script:
   - `harvester_lever1/`, `harvester_sanity/`, `inline_omega_selection/`, `isoshell_viewer/`, `plots/`, `unsorted/`
   
   **Decision needed:** Classify as `analyses/`, `shared/`, or `archive/` as appropriate, then manually move (or record moves in migration script).

### 6. Loose result files (non-directory)
   Several `.npz`, `.json`, `.png` files in result root (not in subdirs):
   - `is901_brightness_table.npz`
   - `micro104_crossing_diagnostic.npz`
   - Various `microNNN_*.json` and `.png` files
   
   **Decision needed:** Consolidate into experiment directories (`m{NNN}_*/`) or create `shared/datasets/` and `shared/analysis_outputs/`.

---

## Summary Statistics

| Series | Total scripts | Micro experiments | Archive/legacy | Result dirs |
|--------|---------------|--------------------|----------------|-------------|
| 00_pipeline_reference | 8 | 0 | 8 | 0 |
| 01_global_search | 24 | 0 | 24 | 0 |
| 02_isobrightness_filtering | 14 | 0 | 14 | 0 |
| 03_omega_bridging | 12 | 0 | 12 | 0 |
| 04_basin_characterization | 4 | 0 | 4 | 0 |
| 05_peak_graph_pipeline | 28 | 27 | 1 | ~30 |
| 06_omega_basin | 4 | 0 | 4 | 0 |
| 06_omega_bridging_benchmarks | 7 | 4 | 3 | ~8 |
| 07_L_conservation | 3 | 3 | 0 | ~5 |
| 07_multi_epoch_scoring | 2 | 2 | 0 | ~2 |
| 07_winding_enumeration | 2 | 2 | 0 | ~3 |
| 08_integration | 8 | 8 | 0 | ~10 |
| 09_glint_analysis | 16 | 15 | 1 | ~20 |
| 10_glint_inversion | 29 | 29 | 0 | ~50 |
| 11_casadi_formulation | 81 | 80 | 1 | ~100 |
| 12_brightness_surface | 34 | 31 | 3 | ~50 |
| **TOTAL** | **272** | **201** | **70** | **~286** |

**Wiki pages:** 39 (mostly m070–m129)

**Key findings:**
- **201 out of 272** scripts are micro-experiments (73.9%)
- **70 scripts** are legacy/helper/demo files (25.7%)
- **First 4 series** (00–03) are entirely legacy or tutorials
- **Series 05–12** are the active research area with micro-experiments
- **Series 11 & 12** dominate by count (80 + 31 = 111 micro-experiments out of 201)
- **Result directory count** aligns roughly with script count, plus loose files and orphan analyses

