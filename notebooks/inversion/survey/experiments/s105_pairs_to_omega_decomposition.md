---
title: "s105 — pairs→ω decomposition on seed 119: the hard-shoot trap is the blocker; return-map and LS-peaks refuted"
type: experiment
sources:
  - experiments/s105a_dir_basin_width.py
  - experiments/s105b_third_anchor_collapse.py
  - experiments/s105c_connect_null.py
  - experiments/s105d_truthpair_fulllc_discrim.py
  - experiments/s105e_truthpair_multistart_fulllc.py
  - experiments/s105f_realpair_multistart_fulllc.py
  - experiments/s105_verify_backprop.py
  - experiments/s105g_windowed_score.py
  - experiments/s101_ls_peak_truth_proximity.py
  - experiments/s101b_subharm_null.py
  - experiments/s103_return_map_scan.py
  - experiments/s104_returnmap_cost.py
related:
  - experiments/s100_5step_coverage_proto.md
  - experiments/s106_hybrid_loss_polish.md
created: 2026-05-28
updated: 2026-05-28
confidence: high (each leg has a result file or null control; N=1 seed 119, one cross-check on 116)
---

# TL;DR
Given the s100 finding that coverage is solved but the single-shoot cross still aliases, this chain pins down exactly why and what does/doesn't recover truth on seed 119. The binding blocker is the **hard-shoot trap**: a `shoot` hard-constrains ω to *exactly* connect two cloud endpoints that are each ~1° off truth, and over the 3.1-turn A→B baseline that bakes in a **4.25° ω-direction error → full-LC Band D (RMSE 0.687)** (source: `results/s105/realpair_multistart.json`). The LM multistart generator is NOT the problem — on the EXACT truth pair it returns truth at Band A (RMSE 0.0180, dir-err ~0°; source: `results/s105/truthpair_multistart.json`). Two clever shortcuts were refuted along the way: the **propagator return-map** ("return depth picks the true direction") — false, every direction connects (source: `results/s105/dir_basin_width.json`); and **LS-peak |ω| priors** — band-tiling, not signal ([[feedback_lc_spectral_omega_prior_dead]]). Windowing the *score* (not the objective) also fails to rescue (source: `results/s105/windowed_score.json`). Fix is in [[s106_hybrid_loss_polish]].

# What
After s100 localized the blocker to single-shoot ω-selection on a slightly-off pair, the open questions were: (a) is there a cheap way to pick the right ω *direction*/winding without truth? (b) does any generator recover truth from a realistic pair? (c) is the full-LC RMSE the villain (would windowing fix the score)? This chain answers all three, and closes two tempting dead-ends (LS peaks, return-map) so the next agent doesn't re-walk them.

# How
All on seed 119, anchors A=ep69/B=ep172/C=ep377, dt_ab=743 s (3.10 turns), full LC 14.82 turns, truth |ω_a|=1.5023 deg/s, surrogate floor 0.0180. Physical |ω| bracket [0.1, 1.6] deg/s. Pool(24) where parallel.
- **s101 / s101b** — LS-peak proximity to truth across 120 seeds + null control.
- **s103 / s104** — propagator return-map demo (sweep |ω| per direction, geodesic-to-q_b minima = "returns"; depth claimed to discriminate direction) + its projected cost.
- **s105a** — direction-basin width: offset the guessed direction off truth and Fibonacci-sweep N_dir; record return depth.
- **s105b / s105c** — third-anchor collapse: how many ω connect q_b (<3°), do they also thread q_c, + a 20k-random null for enrichment.
- **s105d** — dir-grid return-map enumeration (N_dir=4000) of the connecting family + full-LC scoring.
- **s105e** — `multistart_shoot` on the EXACT truth pair (170 and 1300 starts) + full-LC score.
- **s105f** — `multistart_shoot` on the realistic nearest cloud pair (qa 1.02°, qb 0.68°) + full-LC, plus a KNN=5 yield scan.
- **s105_verify** — three independent checks the anchor→full-LC back-propagation is correct.
- **s105g** — windowed vs full RMSE on the near-truth pair's roots.

# Result

| leg | question | result | verdict | source |
|---|---|---|---|---|
| s101/b | LS peaks → |ω| prior? | true subharmonic-hit 88.3% vs random 77.5% @5% | band-tiling, dead | results/s101, [[feedback_lc_spectral_omega_prior_dead]] |
| s105a | does return depth pick true dir? | offset-0 depth 0.0003° but n_under_5° = ALL dirs (20/20…1000/1000); deepest at 18–50° off truth | return-map REFUTED | results/s105/dir_basin_width.json |
| s105b/c | do 3 anchors pin truth? | 16 ω connect B, all 16 thread C; best q_c dir 51° off, |ω| 1.02 dps (truth 1.50) | 16 discrete multi-sols, not truth | results/s105/third_anchor.json |
| s105d | dir-grid generator finds truth needle? | 76-member family, best 1.21 (C/D) at 148.6° off; truth-near (10.4°) ranks 67 | dir-grid MISSES truth | results/s105/truthpair_fulllc.json |
| s105e | LM multistart, EXACT pair? | best RMSE **0.0180 Band A**, dir-err 0.0° (170 & 1300 starts) | generator is sound | results/s105/truthpair_multistart.json |
| s105f | LM multistart, REAL ~1° pair? | best RMSE **0.687 Band D**, dir-off **4.25°**; yield 25 pairs 0 Band A/B | the hard-shoot trap | results/s105/realpair_multistart.json |
| s105_verify | backprop correct? | reconstruction ~2e-6°, two paths identical, control 0.018 | result is real, not a bug | experiments/s105_verify_backprop.py |
| s105g | does windowing the SCORE help? | window 8.19 turns; window-best root 0.704, still 4.25° off | windowing score FAILS | results/s105/windowed_score.json |

# Why this matters
The pair→ω step is the whole ballgame, and it has a clean structure:
- **Connection is degenerate.** A single pair admits a 2-parameter family of connecting ω (every direction connects); 3 anchors collapse it only to ~16 discrete winding multi-solutions, none at truth. So no purely-geometric connection rule recovers truth — the LC must discriminate.
- **The generator must refine BOTH direction and magnitude.** Fixed-direction dir-grid enumeration (s105d) misses truth (needle on the connecting manifold); free-direction LM multistart (s105e) nails it given the exact pair.
- **The hard constraint is fatal on a slightly-off pair.** Forcing ω to thread ~1°-off endpoints across 3.1 turns amplifies into 4.25° → Band D. This is the high-|ω| narrow-basin tail made concrete.
- **Windowing the score is not the lever** (it re-ranks the same phantoms). This is what made s106's *window-as-objective* finding non-obvious.

# Numbers
- s105a seed 119: offset-0 best_depth 2.65e-4°; n_under_5deg 20/20 … 1000/1000; winner_offset_deg 18.2–50.1° (source: results/s105/dir_basin_width.json)
- s105b: n_connect_b 16, gate {1:16,3:16,5:16,10:16,20:16}, best_qc_dir_offset 51.02°, best_qc_wmag 1.0215 dps (source: results/s105/third_anchor.json)
- s105d: n_family 76, best_rmse 1.2069 (C/D) dir_off 148.63°, truth_near 10.43° rank 67 RMSE 2.301 (source: results/s105/truthpair_fulllc.json)
- s105e: coarse best 0.018039 Band A dir 1.0e-5°; dense best 0.018038 dir 0.0° (source: results/s105/truthpair_multistart.json)
- s105f: nearest pair qa 1.0167°/qb 0.6838° → best 0.6870 Band D, dir-off 4.2486°, |ω| 1.5018 dps; yield 25 pairs min 0.4345 median 1.263 nA 0 nB 0 (source: results/s105/realpair_multistart.json)
- s105g: window [0,273] 8.19 turns; truth window RMSE 0.01926 / full 0.01804; window-best 0.7038 dir 4.25° (source: results/s105/windowed_score.json)
- 116 cross-check (s105a): turns 0.268 (slow); even there all dirs connect — return-map dead in both regimes (source: results/s105/dir_basin_width.json seed 116)

# Artefacts
- `results/s105/{dir_basin_width,third_anchor,truthpair_fulllc,truthpair_multistart,realpair_multistart,windowed_score}.json`
- `results/s105/{third_anchor,truthpair_fulllc}.npz`
- `results/s103/{return_map.png,return_map_seed116.png}`
- `results/s101/peak_proximity.json`

# Out of scope
- The fix (relax hard connection → photometry-objective polish): [[s106_hybrid_loss_polish]].
- Generalization beyond seed 119 (only 116 cross-checked, for the return-map leg).
- 1M-cloud reps (s105f used s100's 12M-cloud repA/repB).

# Cross-references
- [[s100_5step_coverage_proto]] — coverage solved, single-shoot blocker localized.
- [[s106_hybrid_loss_polish]] — the windowed-photometry LM polish that clears Band A.
- [[feedback_lc_spectral_omega_prior_dead]] — updated with s101/s101b corroboration.
- [[feedback_omega_prior_physical_bracket]] — the [0.1,1.5] (solver 1.6) deg/s clamp used throughout.
