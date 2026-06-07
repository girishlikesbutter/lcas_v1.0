---
title: "s006 — seed 28 surrogate landscape probe at fixed truth-ω"
type: experiment
sources:
  - data/results/inversion_diagnostics/m048_trajectories/per_trajectory/traj_seed028.npz (post-fix; commit ac1fdf4)
  - data/results/inversion_diagnostics/m048_trajectories/m048_trajectories.npz (inertia tensor)
  - ~/surrogate_model (v2 residual ensemble; bridge-independent)
  - lib/forward.py (post-fix propagator + scipy Rotation)
  - results/s002/per_seed_landscape.npz (8-seed baseline at fixed truth-ω)
  - results/s005/runs.npz (joint LM convergence; seed 28 = 3/10 ICs)
related:
  - experiments/s002_surrogate_landscape_probe.md (Q2 — argmin at truth on 8/8)
  - experiments/s005_joint_local_descent.md (Q4a/Q4b — joint LM works; seed 28 has tight ~2° basin)
  - experiments/s004_alignment_landscape_vs_omega.md (alignment cost on seed 28: argmin 176.8° from truth at baseline ω; 2 Sobol below truth)
  - concepts/q_omega_coupling.md (basin geometry)
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# s006 — seed 28 surrogate landscape probe at fixed truth-ω

## TL;DR

s005 found seed 28 has the tightest joint-LM basin in the 5-seed cohort (~2° in q0_geodesic; 3/10 ICs converge, 7 escape to MSE 5–9 mag² with q0_err 20°–113°). Q4d asks: are those LM escapes landing at competing surrogate-MSE basins **visible at Sobol resolution**, or are they LM-stall artefacts in a single-basin landscape that's just very narrow?

On seed 28, with ω = ω_truth, **2046 Sobol-Shoemake quaternions plus truth + twin → 2048 candidates**:

- **argmin = truth** (full_mse = 5.85e-4 mag², matches s001 / s005 truth-MSE reference exactly).
- **n_sobol_below_truth = 0**, and even at relaxed thresholds **n_sobol < 100 × truth_mse = 0**. The entire Sobol cloud is at least 1456× above truth_mse.
- **sobol-min-geo-to-truth = 10.06°.** The closest Sobol candidate (12.24° from truth) has MSE = 0.85 mag² — i.e. the basin is so narrow that a 10° geodesic miss costs 3 OOM in MSE.
- **twin_mse = 4.87 mag²** — twin is not a competing minimum at fixed truth-ω (consistent with s002's 8/8 finding).
- The Sobol MSE distribution is unimodal, concentrated around the **high-MSE plateau** (median 11.1 mag², p10 7.5 mag², p90 15.6 mag²). No bimodality; no secondary local-minimum signature.

Decision: **case (A)** — seed 28's landscape is structurally honest at truth-ω; the basin is sub-Sobol-resolution. The s005 LM escapes (final MSE 5–9 mag²) sit inside this Sobol-visible high-MSE plateau and are LM-stall points, **not competing minima**. Implication for Q4c: Sobol density alone cannot seed seed-28-class basins — every Sobol candidate must be followed by an LM polish, AND seed 28's ~2° basin is below the 10° Sobol coverage radius even at 2046 points, so a polish from the nearest Sobol won't reliably enter the basin without (a) much higher Sobol density (10²×–10⁴×) or (b) ω-mag prior from peak-spacing to collapse one search dimension first.

## What

Score the surrogate full-LC MSE at 2046 Sobol-Shoemake quaternions on SO(3) for seed 28 alone, with ω fixed at truth-ω. Insert truth-q0 at index 0 and twin (q_180x · q0_truth) at index 1 so both are exactly evaluated. Use SOBOL_SEED = 42 (matching s002) so the Sobol point set is identical to a "what would s002 have shown if it had included seed 28?" experiment — direct comparability with the existing 8-seed baseline.

Beyond s002's per-seed argmin/n_below_truth reporting, add a **competing-basin diagnostic**: count of Sobol candidates with full_mse < K · truth_mse for K ∈ {2, 5, 10, 100}, and the geodesic-to-truth distribution of those candidates. This catches case (B) — Sobol-visible competing basins — even when truth remains the global argmin.

## How

- **Seed:** 28 only. PA from s001 = 88.1° (PA-high). PA-stratified pick from s005's failure analysis.
- **Grid:** `scipy.stats.qmc.Sobol(d=3, scramble=True, seed=42)`, 2046 points, mapped through Shoemake's uniform-S^3 (`shoemake_to_quat` in script). Plus truth + twin at indices 0 / 1.
- **Propagation:** `lib/forward.py:propagate_to_body_frame` — same wrapper as s002 / s005 (post-fix `propagate_attitude(mode='tumbling', inertia_tensor=...)` plus per-epoch `Rotation.from_quat([qx,qy,qz,qw]).as_matrix()` projection of sun/obs into body frame).
- **Scoring:** `surrogate_eval.full_lc_mse(predicted, mag_hifi)` over all 500 cached observation epochs. Bright-MSE also recorded but the population-scale `mag_hifi` is bright-dominated for IS-901 so the two correlate.
- **Diagnostics:** `diagnose_competing_basins` — for K ∈ {2, 5, 10, 100}:
    - `n_sobol_below_K_truth`: count of Sobol points with full_mse < K · truth_mse.
    - `n_sobol_below_K_truth_far_from_truth`: subset of the above with geodesic-to-truth > 30°.
    - `geo_min/max/median_deg`: geodesic-to-truth statistics of the K-eligible Sobol candidates.
- **Round-trip validation:** truth-q0 (index 0) gets full_mse = 5.847e-4, exactly matching s001's cached `surr_full_mse` and s005's `truth_mse_ref` for seed 28 → propagation is bit-identical to the cache-generating run.
- **Wall:** 28.7 s for 2048 candidates with Pool(8), BLAS=1 in workers.

## Result

| metric | value | comparison |
|---|---|---|
| truth_full_mse | 5.85e-4 mag² | matches s001 cache exactly |
| twin_full_mse | 4.87 mag² | not a competing basin |
| argmin index | 0 (truth) | s002 8/8 baseline preserved |
| argmin_full_mse | 5.85e-4 | = truth_full_mse |
| n_sobol_below_truth | 0 / 2046 | s002 had 0/2046 on every seed |
| n_sobol_below_2×truth | 0 | no near-truth competing basins |
| n_sobol_below_5×truth | 0 | no shallow competing basins |
| n_sobol_below_10×truth | 0 | no moderate competing basins |
| n_sobol_below_100×truth | 0 | even at K=100, zero |
| sobol_min_geo_to_truth | 10.06° | s002 range was 4.2°–12.4° (seed 28 at upper end) |
| best_sobol_mse | 0.851 mag² @ 12.24° geo | 1456× above truth_mse |
| sobol mse distribution | median 11.07, p10 7.52, p25 9.56, p90 15.55 mag² | unimodal, concentrated on plateau |

**The Sobol cloud's MSE distribution is concentrated tightly around 10–15 mag²** (median 11.1, IQR 9.6–13.0). The best Sobol candidate (0.85 mag²) is a 12° outlier; the next-best is at MSE 1.27 mag². There is **no second mode in the histogram** — seed 28's surrogate landscape under fixed truth-ω is **single-basin**, with the basin being narrower than the Sobol coverage radius (10°).

**Comparison with s005 LM escapes on the same seed:**

| s005 IC label | initial q0 offset | final q0_err | final MSE | s006 Sobol equivalent? |
|---|---|---|---|---|
| T1_inside (basin) | 2.0° | 0.03° | 5.78e-4 | matches truth (off-grid) |
| T2_inside (escape) | 5.0° | 60.6° | 6.45 | inside Sobol p10–p90 plateau |
| T3_edge (escape) | 8.0° | 38.6° | 8.86 | inside Sobol p25–p75 plateau |
| T4_outside (escape) | 15.0° | 32.9° | 9.15 | inside Sobol p25–p75 plateau |
| T5_outside (escape) | 30.0° | 69.1° | 7.87 | inside Sobol p10–p75 plateau |
| T6_outside (escape) | 60.0° | 113.1° | 7.15 | inside Sobol p10–p75 plateau |
| R1_random (escape) | 4.1° | 20.5° | 7.96 | inside Sobol p10–p75 plateau |
| R4_random (escape) | 2.0° | 36.2° | 5.12 | inside Sobol bottom 10% |

LM escapes land at MSE values squarely inside the Sobol high-MSE plateau (5–9 mag²) — **not below it.** This is the smoking gun: the LM-escape final states are not at deeper-than-truth competing minima; they are at LM-stall points within the broad high-MSE region the Sobol cloud already maps. The basin around truth is so narrow (~2° in q0) that LM stepping from outside the basin doesn't find the basin gradient — it finds whatever flat-ish direction reduces residuals locally and stalls.

## Why this matters

1. **Q4d is decisive: case (A).** Seed 28 has a narrow but genuine single-basin surrogate-MSE landscape. The s004 alignment-cost multi-basin pathology (argmin 176.8° from truth at baseline ω, 2 Sobol below truth) does **not** carry over to surrogate-MSE on seed 28. The s005 narrow-basin observation is consistent with a single global basin that's just narrower than s005's IC offsets — there is no competing minimum that a global searcher could mistake for truth.
2. **Q4c-i basin-volume estimate becomes harder for seed 28.** s005 measured the basin radius in q0 at ~2°; s006 confirms the basin is also unresolved at 10° Sobol spacing. Naive uniform Sobol on SO(3) at 2046 points has ~5° expected nearest-neighbour spacing (the realised seed-28 nearest-neighbour was 10° — Sobol scrambling variance), but seed 28's basin is tighter than that. So 2046-Sobol cannot seed seed 28 even with LM polish from every Sobol point — the polish entry-point is outside the basin. Estimated lower bound on Sobol density to land at least one inside-basin candidate per seed: with q0-axis basin ~2°, the basin volume on SO(3) is ~`(2°/π)³ × (4π/3)` ≈ 1.6e-5 of SO(3), so ~6e4 uniform candidates needed for one in-basin hit *on average*. Seed 28 alone ≳ 60k Sobol candidates; cohort scale would need ≳ 10⁷ with naive sampling. Confirms the Q4c-iii **ω-mag-from-peak-spacing prior** is the most likely path forward (collapses one dim of the search).
3. **The s005 LM escapes are bias-free w.r.t. truth direction.** The escape final-q0 errors (20°, 33°, 36°, 60°, 69°, 113°) span 20°–113° fairly uniformly — they're not preferentially landing near twin (180°) or any specific direction. This is consistent with the Sobol cloud's geodesic distribution (uniform on SO(3)). LM escapes are diffusion in the high-MSE plateau, **not attraction to a specific competing minimum**.
4. **Comparison with s002's other seeds:** seed 28's `sobol_min_geo_to_truth = 10.06°` is at the upper end of s002's 4.2°–12.4° range. Combined with s005's 2° basin radius (tightest of the 5-seed cohort), seed 28 is **the worst-case among the seeds we've measured** for global-search-then-polish architectures. If seed 28 is representative of the bottom 10–20% of the cohort, **architectural decisions for Q4c should be calibrated to seed 28, not to median seeds.**
5. **Twin is still not a fixed-ω basin.** twin_full_mse = 4.87 mag² on seed 28 — sits inside the Sobol high-MSE plateau (around the median 11.07 but at a lower spot). Confirms the s002 finding extends to the seed s002 didn't sample.

## Numbers

- N candidates: **2048** (= 1 truth + 1 twin + 2046 Sobol-Shoemake, SOBOL_SEED=42)
- argmin = truth on **1/1** seeds tested (consistent with s002's 8/8)
- n_sobol_below_K × truth_mse, K ∈ {1, 2, 5, 10, 100}: all **0 / 2046**
- best_sobol / truth_mse ratio: **1456×** (geo 12.24°)
- sobol_min_geo_to_truth: **10.06°** (s002 range 4.2°–12.4°; seed 28 at upper end)
- truth_mse: **5.85e-4 mag²** (ρ-equiv 0.48; Band A)
- twin_mse: **4.87 mag²**
- Sobol MSE distribution: median **11.07**, p10 **7.52**, p90 **15.55** mag²
- s005 LM-escape final MSEs (7 ICs): **5.12 to 9.15 mag²** — squarely inside Sobol p10–p90 band
- wall: **28.7 s** (Pool(8), BLAS=1)

## Artefacts

- `experiments/s006_seed28_landscape_at_truth_omega.py` — script.
- `experiments/s006_seed28_landscape_at_truth_omega.md` — this writeup.
- `results/s006/landscape.npz` — q_grid + kind + full_mse + bright_mse + geo_to_truth_deg + geo_to_twin_deg + geo_to_truth_or_twin_deg.
- `results/s006/summary.json` — argmin / n_sobol_below_truth / competing-basin counts at K∈{2,5,10,100} / Sobol-min-geo-to-truth / wall.
- `results/s006/landscape.png` — single-panel MSE-vs-geodesic-to-truth scatter, truth/twin/argmin annotated.
- `results/s006/competing_basins.png` — two-panel: stacked histogram of geodesic-to-truth coloured by MSE band, plus scatter coloured by MSE band.
- `results/s006_run.log` — run log.

## Out of scope

- **ω-misspecified landscape on seed 28.** This experiment fixes ω = ω_truth. s003 sweep was on seeds 6/41/91; not seed 28. If Q4d had returned case (B), that gap would matter; with case (A) confirmed, it's lower priority. Seed 28's known multi-basin behaviour at baseline ω in **alignment cost** (s004) does not extend to surrogate-MSE here — separate cost surfaces, separate landscape geometries.
- **Sub-10° structure.** A competing basin sized between Sobol's 10° resolution and s005's 2° measurement would be invisible to both. Possible but not motivated — the s005 LM escapes don't show evidence of being attracted into deeper basins (their final MSEs are inside the Sobol high-MSE band, not below it).
- **Higher Sobol density** (e.g. 16k–65k candidates) — would refine the basin coverage measurement but the structural finding (single-basin, narrow-basin, sub-2046-Sobol-resolution) is already settled.
- **Other tight-basin seeds.** Seed 28 was chosen because s005 identified it as the tight-basin outlier. If 28 isn't representative — e.g. if 5–10% of the cohort has even tighter basins — the Q4c density estimate will need revision. Cohort-scale basin-radius probe is on the Q4c roadmap.

## Cross-references

- **Q2 (s002):** found argmin = truth on 8/8 PA-stratified seeds (excluding 28); n_sobol_below_truth = 0 on 8/8. s006 extends this to 9/9 by adding seed 28, and adds the competing-basin diagnostic at K ∈ {2, 5, 10, 100} which strengthens the argmin claim by ruling out near-truth Sobol-visible competitors.
- **Q4 (s005):** measured seed 28's joint-LM basin radius at ~2° in q0_geodesic; 3/10 ICs converge, 7 escape to MSE 5–9 mag². s006 explains those escapes as LM-stall in the high-MSE plateau, not landings in competing minima.
- **Q5 (s004):** alignment cost on seed 28 has 2 Sobol candidates below truth at baseline ω = ω_truth (multi-basin pathology). s006 shows surrogate-MSE on seed 28 has 0 Sobol candidates below truth. **Surrogate-MSE on seed 28 is structurally cleaner than alignment cost on the same seed.** Further evidence the alignment-cost / decoupled-architecture path is dead.
- **concepts/q_omega_coupling.md:** the "basin shape in (q0, ω) is jointly determined" thesis is reinforced — the q0-only landscape with ω fixed at truth is single-basin and well-behaved; the LM-escape failures observed in s005 happen because the IC was outside the joint basin, and the q0-axis projection of the joint basin was below the IC's effective step radius.
