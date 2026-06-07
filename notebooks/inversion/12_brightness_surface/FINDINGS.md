# Series 12: Brightness Surface / IPL Framework

## Key Findings

### random m048 25-seed baseline (2026-04-22)

First representative sample of pipeline performance on the m048 trajectory database under the fully honest pipeline (canonical noise realisation + 8-min m103 Step 4 graceful timeout). Seeds picked via `np.random.default_rng(42).choice(100, 25)`. **Wall: 4.79 h.**

**Population by ρ-band** (ρ = √hifi_MSE / 0.05, σ_noise = 0.05 mag, Roberto convention):

| Band | n | % | seeds |
|------|---|---|-------|
| A (ρ<2) | 4 | 16% | 6, 17, 45, 91 |
| B (2-4) | 0 | 0% | — |
| C (4-8) | 1 | 4% | 79 |
| D (ρ≥8) | 17 | 68% | 7, 8, 11, 16, 34, 47, 48, 51, 57, 59, 64, 67, 71, 78, 84, 89, 99 |
| geo_timeout | 2 | 8% | 35, 69 (m103 Step 4 8-min cap) |
| m103 crash | 1 | 4% | 42 (empty `spec_peaks` — no bright peaks to anchor) |

**Band A winners** — near-perfect recoveries (twin is expected ±X degeneracy, valid solution):

| seed | ρ | q₀ err | ω dir err | ω mag err |
|------|---|--------|-----------|-----------|
| 6  | 0.96 | 179.98° (twin) | 0.07° | +0.01% |
| 91 | 0.98 | 0.05°          | 0.01° | −0.00% |
| 17 | 0.98 | 179.99° (twin) | 0.06° | +0.00% |
| 45 | 1.14 | 179.80° (twin) | 0.40° | −0.00% |

**Band D diagnostic**: every single Band D winner has ω direction error **> 19°** (most 50-170°). Clean upstream-ω-pool-miss signature: m103's honest top-3 ω candidates don't contain a near-truth ω, and the m115 DE + m126 polish combination cannot recover without one. Matches the Phase-B seed 23 failure mode but at a cohort-wide rate.

**Contrast with prior Phase-B 8-seed cohort** (4A + 1B + 0C + 1D + 2 upstream-fails = 67% at/near noise floor): that cohort was cherry-picked from prior rounds because the pipeline did well on them. Random sample is 16% A + ~20% ≤ Band C + 76% upstream/downstream fails. **Phase-B numbers do not generalise. Roberto's 2026-04-17 "12 of 16 at noise floor" headline is a biased subset, not a pipeline-yield estimate.**

**Noise-realisation fix bonus**: seed 49 rerun under unified `default_rng(42)` (was: m126 used legacy `np.random.seed(42)` Mersenne Twister producing different noise than m115's PCG64) dropped ρ 1.67 → 1.03, ω_dir err 0.89° → 0.094°, q₀ err 1.15° → 0.21°, hifi MSE 0.00697 → 0.00266. Not cosmetic — the inconsistency was leaving polish quality on the table for well-behaved seeds.

**Infra added**:
- `lib/traj_source.canonical_observed_lc` — single source of truth for noisy LC. `load_truth()` returns `observed_lc`.
- m103 Step 4 `pool.map_async().get(timeout=MICRO103_GEO_TIMEOUT_S=480)` with graceful `geo_timeout.flag` handoff.
- `invert.py` detects the flag and skips downstream with `status=geo_timeout`.
- m126 switched to `setup_experiment(skip_true_lc=True)` — saves ~50 s/seed.
- `run_m048_batch.py` sequential driver: resume marker `.noise_fix_v1.done`, batch_log.jsonl, 25-min safety-net wall cap.

**Open work**:
1. m103 upstream ω selection is the dominant bottleneck — surrogate-first front-end rewrite (Roberto path-forward #1) now strongly motivated by evidence.
2. m103 Step 1 crash on empty `spec_peaks` (seed 42) — needs graceful-skip path similar to geo_timeout.
3. m115 Step 1 parallelisation — nested `for ic × for si` loop with `workers=1`; 5-7× speedup trivially available.

**Commits**: `36e724c` (code), `db46756` (data).
**Artefacts**: `data/results/inversion_diagnostics/batch_m048_v1/`, per-seed `wrappedbest_m048_seed{NNN}/`, `invert_m048_seed{NNN}/`, `m103_hybrid_m048/seed_{028,035,069}/geo_timeout.flag`.
**Wiki**: `notebooks/inversion/wiki/wiki/branches/m048-random-cohort-baseline.md`.

---

### IPL Census (100 seeds)
- 83% of seeds have ≥2 tight IPL minima (centroid < 5° from truth PAB)
- Median best centroid distance: 1.2°
- Discrimination gap: at 57% of tight minima, truth PAB is >20° from nearest standard normal but only 2.3° from nearest IPL centroid. 92/100 seeds benefit.
- For IS-901: 2-loop epochs = ±X only (always bright, mag < 7). No 2-loop at dim magnitudes.
- 4-loop at dim magnitudes: 14,318 epochs (35% of dim), centroids at ±X + ±Z/±YZ-combo. 945 have centroid distance < 10°.

### m107 — IPL Centroid Grid Cost (FAILED)
- Replaced standard alignment cost with IPL centroid cost at grid level
- Seed 27 truth ranked 754/2000 (should be top ~100)
- Root cause: delta-q amplification at distant constraint epochs

### m108 — Central-Anchor IPL Cost (FAILED)
- Strategy A (minimize max |dt|): picked epochs with 9-10° centroid distances → truth ranked 1299/2000
- Strategy B (minimize combined error): better clusters but MORE centroids per epoch → truth ranked 1470/2000
- Root cause: max-over-centroids cost is non-discriminating when epochs have 8-12 loops. Any direction aligns with some centroid.
- Fundamental tension: tight centroids (small angular distance) ↔ many loops (poor discrimination)

### Failure Mode Reclassification
- 15 m103 seeds: 3 GRID, 3 TWIN, 6 ATT_FAIL, 1 OK, 2 PARTIAL
- **ATT_FAIL is the dominant failure mode**: omega correct (<5°), attitude wrong (47-165°)
- Standard anchor direction already within 1.5-2.5° of truth — anchor quality NOT the bottleneck
- Problem is PHI SELECTION (twist angle around anchor normal)

### Phi Discrimination Tests (all lo-fi)
- Multi-peak (3 peaks): OK seeds find exact truth phi (52× range). ATT_FAIL seeds: wrong phi wins by 1.3-5.1×
- Plateau (15 dim epochs): 125× discrimination range but truth still 8.9× worse than best phi
- Lo-fi brightness model is systematically biased for ATT_FAIL seeds — brightness surface too symmetric at their lobes

### Shadow Analysis
- Many peaks have zero shadow effect (lo-fi = hi-fi)
- Some peaks have 0.1-0.5 mag asymmetric shadow effects on flanks
- Zero-shadow peaks also fail → shadows are NOT the sole cause of lo-fi phi degeneracy

### m130 — Surrogate v1 vs v2 head-to-head (2026-04-18)

A new residual-ensemble surrogate (v2) dropped in `~/surrogate_model/surrogate_model/`. Advertised headline: 5.5× better bright-MAE (0.008 vs 0.044 mag), 82% error reduction over v1, same 32D feature vector, 3-member ensemble, pure-NumPy inference. Four experiments run to decide v1→v2 migration:

**E1 — Population LC fidelity at truth** (`m130b_population_fidelity.py`, 100 m048 seeds × 500 epochs):
- Pooled 50k predictions. v1 MAE 0.0355 → v2 MAE 0.0098 (**3.6× better overall**). Bright MAE 0.071 → 0.011 (**6.6× better**). Max error 2.08 mag (v1) → 0.41 mag (v2) — 5× tighter tail. **v2 wins on every one of 100 seeds** (no rank inversions). Improvement ratio is roughly seed-independent — no "v1 already adequate here, v2 helps there" split.

**E2a — Re-score m115 DE basin candidates** (`m130c_rescore_m115.py`, 17 seeds × ~30 cand):
- Spearman vs hi-fi on validated subset: v1 0.996 → v2 0.9993. Both already tight. **Both surrogates pick the hi-fi-best basin on similar fractions of seeds** (v1 10/17, v2 13/17). Neither shows catastrophic >1-dex mis-ranking on m115 basin endpoints (expected: m115's clustering preserves only good candidates).
- Sanity check: v1 re-scored MSE matches v1's stored `surr_mse` to 2e-16 (numerical noise), confirming the re-scoring pipeline is faithful.

**E2b — Re-score m124 polished off-truth candidates** (`m130d_rescore_m124.py`, 25 cand = 5 truth + 20 polished across 5 seeds):
- At **truth-reference points** (where hi-fi MSE ≈ 0.0024 = observation noise floor): **v1 reports 0.0042–0.0082 (1.7–3.4× overestimate); v2 reports 0.0025–0.0026 (3–8% overestimate)**. This matters for calibration — v1 cannot distinguish observation noise from real modelling error, v2 can.
- At **off-truth polished points**: v1 p90 log-ratio = +0.26 (up to 80% MSE overestimation on 10% of candidates); v2 p90 = +0.026 (max 6% on 10%). Spearman polished: v1 0.974 → v2 0.989. **No catastrophic (>10×) ratios on either surrogate on this set** — the seed-27 "surrogate anti-correlated with hi-fi" story from the original m124 run was retracted after the b906691 re-run (data-integrity bug, not a surrogate property).

**E3 — Latency benchmark** (`m130e_speed_benchmark.py`):

| call | v1 | v2 | v2-fast (single member) | v2 slowdown |
|---|---:|---:|---:|---:|
| single epoch | 0.21 ms | 0.72 ms | 0.38 ms | 3.4× |
| 500-epoch LC | 3.5 ms | 53 ms | 46 ms | 15× |
| 1000-epoch LC | 6.4 ms | 84 ms | 72 ms | 13× |
| 10k-dir manifold call | 87 ms | 804 ms | 640 ms | 9.2× |
| 50k batch (100 × 500) | 536 ms | 4.3 s | 3.3 s | 8.1× |

The overhead is dominated by NoShadowEvaluator (analytical BRDF over 3840 facets, chunk 50). Amortises with batch size. Fast-mode v2 (single ensemble member) saves only 14–17% — not a middle-ground worth the accuracy hit.

**Decision matrix (E4 synthesis, `m130_replot.py` + the above) — use v2 where accuracy matters and call count is bounded; keep v1 where call count is in the tens-of-thousands and search-basin correctness is what matters, not fine ranking:**

| call site | recommendation | rationale |
|---|---|---|
| `hifi_isoshell_viewer.py` (PAB manifold) | **v2** | 500 epochs × 800 ms = ~7 min per viewer rebuild. Accuracy gain fixes backlit/high-phase manifold noise. One-off rendering — latency is a rounding error. |
| Hi-fi *validation* path (m115 step 2, m124-type) | **v2** | Already low call count (≤30 per seed). v2 catches v1's truth-MSE noise-floor overestimate, preventing false FAIL classifications. |
| DE inner loop (m115 step 1, m123 polish, m126, 3-DOF/6-DOF DE) | **v1 for search, re-rank top-K with v2** | 15× slowdown turns an 18s DE into 4.4 min. At 10 starts × 3 omegas × 10 seeds = 300 DE runs, the cost goes from ~90 min to ~22 hours. v1's 0.996 Spearman on basin endpoints says v1 is fine for finding basins; v2 only matters for picking between them. Re-rank the top candidates with v2 before hi-fi validation. |
| OK/PARTIAL/FAIL classification threshold (hi-fi MSE 0.1 gate) | **use v2 as oracle for the threshold** | v1's truth-MSE overestimate (1.7–3.4×) biases what "0.1 is close to truth" even means. v2 is faithful at truth, so thresholds calibrated on v2 are physically meaningful. |
| 6-DOF joint search (future) | **v1 exploration + v2 polish** | Same rule as DE: v1 drives exploration cheaply, v2 validates and polishes. |

**Things that DID NOT show up empirically** (and therefore should not motivate additional experiments):
- A "plateau catastrophe" where v1 surrogate gradient anti-correlates with hi-fi gradient off-truth. Both m130c and m130d show both surrogates inside ±0.3 dex on every single candidate. The original seed-27 m124 claim was a data-integrity bug, not a surrogate failure mode.
- Any seed or trajectory regime where v2 is WORSE than v1. On 100 m048 seeds, v2 is at least as good as v1 on every seed (scatter all above diagonal).
- Any accuracy win from v2-fast-mode over v2 full ensemble. The 14–17% speed gain costs 23% accuracy (0.0048 vs 0.0039 log10 MAE) — not worth it.

### m115 oracle-ω bug — fixed 2026-04-21

**Bug**: `m115_surrogate_pipeline.load_omega_candidates` sorted m103's ω candidate pool by `w0_ref_errs` (truth-directed ω direction error — ground truth), not by `geo_costs` (alignment-cost, honest). All Phase-A (m046) and Phase-B (m048) numbers that came through m115's `omega_source=geo_ckpt` branch had oracle-assisted candidate selection. Sibling scripts `m113_de_attitude_search.py` and `m114_surrogate_multistart.py` had the same antipattern (standalone, not in live pipeline).

**Fix**: `m115_surrogate_pipeline.py:295-342` now defaults to `np.argsort(geo_costs)`. Env var `M{113,114,115}_SORT_BY=oracle` reproduces the pre-fix behaviour for A/B diagnostics only. `result.json` now carries per-candidate `geo_cost`, `cand_idx`, `sort_by` and a top-level `omega_sort_by` key.

Also fixed in passing: `invert.py:131` tried to read `hifi_mags_before` from m126's `hifi_ckpt.npz` when polish worsened hifi; that key isn't saved by m126 (only `hifi_mags_after` is). Added fallback. Seed 23 under honest sort tripped this because all 3 basins had worse hifi after polish — pre-existing bug, masked when polish always improved.

**Divergence audit** (`audit_m103_sort_divergence.py` → `m103_sort_divergence_audit.json`): oracle-top-3 vs geo_cost-top-3 differ on every m046 seed with geo_ckpt (0/13 identical) and on 5/6 m048 Phase-B seeds (seed 091 is the only identical-set case).

**Phase-B honest rerun (m048, 6 seeds)** — identical cohort tally despite different ω pools:

| seed | oracle cls | oracle hifi | honest cls | honest hifi | delta |
|-----:|:----------:|:-----------:|:----------:|:-----------:|:------|
|  23  | FAIL       | 1.597       | FAIL       | 2.296       | same verdict; honest worse because lost the truth-adjacent ω (5° w_err → 46°); seed is constraint-poor, was doomed either way |
|  24  | PARTIAL    | 0.0338      | PARTIAL    | 0.0338      | IDENTICAL |
|  49  | OK         | 0.0070      | OK         | 0.0070      | IDENTICAL |
|  81  | OK         | 0.0026      | OK         | 0.0043      | same verdict; m115 flipped PARTIAL→FAIL (lost the 0.4° ω, best honest top-3 ω was 1.6°), but m126 L-BFGS polish bridged 0.70→0.004 and still landed under OK threshold |
|  90  | OK         | 0.0024      | OK         | 0.0024      | IDENTICAL |
|  91  | OK         | 0.0025      | OK         | 0.0025      | IDENTICAL (only seed where geo_cost and oracle had identical top-3 set) |

**Cohort tally unchanged**: 4 OK + 1 PARTIAL + 1 FAIL both before and after. The pipeline's cost-based downstream (m115 DE + m115 hi-fi basin selection + m126 L-BFGS polish + m126 hi-fi wrapped selection) is robust to oracle contamination in the m103 candidate POOL as long as ≥1 truth-close ω stays in the geo_cost top-3. The bug was dangerous, but the downstream hid its effect on verdicts.

**Historical scope**: For m046, 5 of 10 baseline seeds (000, 014, 024, 027, 046) consumed geo_ckpt and were bug-exposed. The other 5 fell through to `m102 result_npz` (oracle-free). Deferred a cohort-level m046 rerun — low priority given the Phase-B null result.

**Artefacts saved (in `data/results/inversion_diagnostics/`)**:
- `m115_surrogate_pipeline_m048_oracle_baseline/` — full pre-fix seed outputs (step1/step2/result.json) for 6 Phase-B seeds
- `m126_wrapped_m048_oracle_baseline/` — full pre-fix m126 outputs
- `wrappedbest_m048_oracle_baseline/` — full pre-fix final Phase-B verdicts + LC pngs
- `m103_sort_divergence_audit.json` — per-seed top-3 divergence report
- `m115_oracle_bug_audit_m048.json` — m115-level honest vs oracle per-seed
- `m115_oracle_bug_audit_phaseB_wrappedbest.json` — final Phase-B honest vs oracle per-seed

**Scripts**: `m115_surrogate_pipeline.py` (fixed), `m113_de_attitude_search.py` (fixed), `m114_surrogate_multistart.py` (fixed), `audit_m103_sort_divergence.py` (new), `compare_m115_oracle_vs_geocost.py` (new), `invert.py:131` (fallback patch).

## Dead Ends (This Series)
- IPL centroids as drop-in grid cost replacement
- Centroid tracking at dim epochs (centroid distances 30-40°, no discrimination)
- Near-centroid epoch tracking (runs too short, all phis identical)

## Open Directions
1. Hi-fi phi discrimination at peaks and plateaus (shadows may break the symmetry lo-fi can't)
2. Lobe asymmetry map from pab-c surface (predict which seeds are phi-degenerate)
3. Combined peak + plateau scoring (different false-positive phis → intersection eliminates them)
4. Multiple-anchor strategy (two independent body-frame directions → discrete candidate set)
5. Plateau tracking for slow tumblers (continuous constraints from PAB circling a lobe)

## Scripts
- `archive/ipl_census.py` — Extract IPL minima statistics from 100 HTML viewers
- `archive/extract_all_epoch_ipl.py` — Extract full per-epoch IPL data (centroids, loops, angular distances) for all 500 epochs × 100 seeds
- `m107_ipl_cost_function.py` — IPL centroid grid cost (failed)
- `m108_ipl_central_difference.py` — Central-anchor IPL cost (failed)
- `m130_v1v2_plots.py` — Hi-fi vs v1/v2 single-seed comparison at 1000 epochs (vectorised hi-fi; truth + residual plots for chaotic seed 19)
- `m130b_population_fidelity.py` — v1 vs v2 population LC fidelity across all 100 m048 seeds × 500 epochs
- `m130c_rescore_m115.py` — Re-score m115 DE candidates with v2; ranking fidelity vs hi-fi on 17 seeds
- `m130d_rescore_m124.py` — Re-score m124 polished off-truth candidates with v2; off-truth bias test
- `m130e_speed_benchmark.py` — Latency benchmark across single-epoch, 500/1000-epoch LC, 10k-dir manifold, 50k batch
- `m130_replot.py` — Regenerate all m130/m130b plots at 300 dpi from cached NPZs (no recomputation)
