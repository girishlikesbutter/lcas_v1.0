---
title: "s083 — |ω| basin sharpness probe (the s082 Day-2 #1 N_mag sensitivity check)"
type: experiment
sources:
  - experiments/s083_nmag_sensitivity.py
  - scratch/s083_bugcheck.py
  - results/s083/summary.json
  - results/s083/seed_{116,119,103}.json
related:
  - experiments/s082_joint_grid_pivot.md
  - experiments/s084_omega_refine_viability.md
  - experiments/s019_ls_bracket_omega_mag.md
  - project_omega_mag_basin_scales_with_omega.md
created: 2026-05-21
updated: 2026-05-21
confidence: high (1-D measurement, cross-validated against DOP853 propagator; basin width scales as constant accumulated-phase across 3 seeds)
---

# TL;DR
The |ω| good-fit basin is **razor-thin — ±2% (slow seed 116) down to <±0.5% (fast 119, SAM 103)** when q + ω-direction are held at truth (source: `results/s083/seed_*.json` `layer_A_truth_q_dir.bandB_halfwidth_pct`). So the s082 prescription "densify N_mag 5→10–15" is **10–100× too coarse** — covering the 8.6–142× LS-bracket at <2% spacing needs ~100–1000 cells, not 15. Brute-force |ω| gridding is dead; |ω| must be **refined in 1-D**. Independently, with the *best* grid q (16–33°) + best ω-dir + an oracle |ω| sweep, ρ_surr never reaches Band B (min **11.85 / 45.65 / 30.50**, all Band D) — so **q-coarseness at Sobol-64 is independently fatal**, and the s082 "densify N_mag" fix would not have rescued v2. The surrogate-MSE landscape itself is sound: truth-exact scores ρ 0.20/0.36/0.33 (deep Band A), rank 1 of the cached 320k.

# What
s082 found truth-nearest rank 70823/320000 (median), with the truth-nearest cell 20–43% off in |ω| (because the 5-cell `ls_bracket` spans 8.6–142×, not the "4× / √2-spacing" the s082 writeup assumed). s082's stated next step (Day-2 #1) was "re-run at N_mag=15+". This experiment answers the underlying question without a 5-hour full re-grid: **how fine must the |ω| grid be — i.e. how wide is the |ω| basin?**

# How
Entrypoint `experiments/s083_nmag_sensitivity.py`, 3 s082 seeds, elliprj Path 2 + surrogate-v2 full-LC residual (cost = ρ_surr = √MSE/0.05). Two 1-D sweeps of |ω| over ±60% of truth in 0.5% steps:
- **Layer A** (basin sharpness): q0 + ω-direction held at **truth**, sweep |ω|. Forward-model sensitivity probe (à la s073f) — truth used only to centre the sweep, NOT a search-yield claim.
- **Layer B** (best achievable grid): q0 + ω-dir held at the **best cells the s082 grid actually achieved** (min q_geo, min dir_err, pulled from cached `candidates.npz`), sweep |ω| (incl. truth). Generous upper bound — pairs best-q from one grid cell with best-dir from another, plus oracle |ω|.
Cross-check `scratch/s083_bugcheck.py`: same |ω| sweep through both `propagate_jacobi_path2` (Jacobi closed-form, used by s082/s083) and `propagate_to_body_frame` (DOP853, the propagator that *generated* the cached truth LCs).

# Result

| Seed | \|ω\| (dps) | truth-exact ρ | **Band B \|ω\| half-width** | Layer B min ρ (best q/dir + oracle \|ω\|) | best grid q_geo |
|---|---:|---:|---|---:|---:|
| 116 LAM-slow | 0.134 | 0.20 (A) | **±2%** (Band A ±1%) | **11.85 (D)** | 33.1° |
| 119 LAM-fast | 1.482 | 0.36 (A) | **<±0.5%** | **45.65 (D)** | 16.5° |
| 103 SAM | 0.530 | 0.33 (A) | **<±0.5%** | **30.50 (D)** | 29.3° |

Cross-propagator check: Jacobi vs DOP853 k1 vectors agree to **~1e-6°** across the whole sweep, on- and off-truth; the two ρ(|ω|) curves are identical to 3 decimals. The narrow basin is real physics, not a Jacobi-propagator artefact. The Band B edge sits at a **constant ~9.6° of accumulated orientation phase** on all three seeds (basin% × total-rotation: 116 ±2%×481°=9.6°; 103 ~0.5%×1908°=9.5°; 119 ~0.18%×5335°=9.6°) — a clean phase-accumulation signature, confirming `project_omega_mag_basin_scales_with_omega`.

# Why this matters
- **Brute-force |ω| gridding is dead.** The s082 "densify N_mag 5→10–15" prescription is struck — it would need ~100–1000 cells. |ω| must be a **1-D refinement** (validated in s084), not a joint-grid axis.
- **q must be anchored, not coarse-gridded.** Layer B shows even the best Sobol-64 q (16–33°) + perfect |ω| can't reach Band B. So both axes (q, |ω|) are independently fatal at the s082 grid resolution — confirming the Branch-v3 (anchor-based) direction with a clean mechanism rather than s082's half-confounded one.
- **Plan refinement.** The blind-inversion plan (`report/blind_inversion_15min_plan_2026-05-20.md:144`) asserts "|ω| is essentially free", citing s019 "98/100 within 5%". But s019's within-5% was the **~50–100-cell** variant; its **5-cell** result is only the `in_grid` metric (truth lies inside [lo,hi] *range*). The plan conflated the two. The plan's |ω| handling (5-cell bracket + `±6%` multi-mag-start polish) cannot bridge a 20–43% miss into a <2% basin — this is where the plan needs fixing before the v3 build.

# Numbers
- truth-exact ρ_surr: 0.204 / 0.361 / 0.331 (source: `results/s083/seed_{116,119,103}.json:truth_exact.rho_surr`).
- Layer A Band B half-width: [−2.0,+2.0]% (116), [0,0]% i.e. <0.5% (119, 103); Band A [−1,+1]% (116) (source: same files, `layer_A_truth_q_dir.band{A,B}_halfwidth_pct`).
- Layer B min ρ_surr: 11.853 (116, q_geo 33.13°) / 45.654 (119, q_geo 16.55°) / 30.504 (103, q_geo 29.34°) (source: same, `layer_B_bestgrid_q_dir.min_rho_surr`).
- Cross-propagator max k1 angle diff: 1.71e-6 / 2.09e-6 deg (source: `scratch/s083_bugcheck.py` stdout).
- Compute wall: 64.3 s (single process, s083); cross-check seconds (Pool not used).

# Artefacts
- `experiments/s083_nmag_sensitivity.{py,md}`
- `scratch/s083_bugcheck.py` (cross-propagator validation; not committed-worthy as a result)
- `results/s083/summary.json`, `results/s083/seed_{116,119,103}.json`
- `results/s083/seed_{116,119,103}_omega_basin.png` (ρ_surr vs |ω|-offset, Layer A + Layer B)

# Out of scope
- Exact <0.5% basin widths on 119/103 (limited by the 0.5% sweep step — upper bounds only). A finer zoom would pin them but doesn't change the conclusion.
- Whether the 1-D refine actually converges over the full bracket — answered in **s084**.
- Hi-fi confirmation of the basin width (surrogate-only here; s081 145/145 band agreement makes this low-risk).

# Cross-references
- `experiments/s082_joint_grid_pivot.md` — the pivot whose |ω| diagnosis this corrects.
- `experiments/s084_omega_refine_viability.md` — tests the 1-D refine this motivates.
- `project_omega_mag_basin_scales_with_omega.md` — basin ∝ 1/(|ω|·T_obs) confirmed (constant ~9.6° phase edge).
- `report/blind_inversion_15min_plan_2026-05-20.md` §3.2/§4 — the |ω| premise that needs fixing.
