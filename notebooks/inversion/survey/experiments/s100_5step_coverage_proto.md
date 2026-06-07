---
title: "s100 — 5-step coverage-fix proto on seed 119: coverage closes, single-shoot cross still aliases (Band D)"
type: experiment
sources:
  - experiments/s100_5step_proto.py
  - experiments/s100_hifi.py
  - results/s100/seed119/invert.json
  - results/s100/seed119/hifi_rho.json
related:
  - experiments/s099_blind_fast_invert_116.md
  - experiments/s105_pairs_to_omega_decomposition.md
  - experiments/s106_hybrid_loss_polish.md
created: 2026-05-28
updated: 2026-05-28
confidence: high (end-to-end blind run completed; hi-fi ρ-band confirmed; N=1 seed 119)
---

# TL;DR
The s099 pipeline failed on seed 119 for two suspected reasons — anchor starvation and single-shoot ω-aliasing. s100 attacks starvation by COVERAGE (dense isophote fill → decimate to 2° reps → cross), and it **works on coverage**: anchor B's nearest-truth rep dropped from a starved 10.5° (480k smoke) to **0.68°** at a 12M adaptive fill, anchor A to 1.02° (source: `results/s100/seed119/invert.json` `nt_b_dense`/`nt_a_dense`). But the end-to-end inversion still lands **top blind ρ 48.17, Band D, 0/15 Band A** (source: `results/s100/seed119/hifi_rho.json`). Cause is isolated by the built-in probe: the single finite-diff-init shoot on the truth-near pair connects to an ω **104° off** truth at |ω| below the bracket → rejected; a bounded multi-start recovers it to **4.58°** (source: `invert.json` `probe`). Coverage is not the binding blocker — single-shoot ω-selection is. Decomposed further in [[s105_pairs_to_omega_decomposition]]; fixed in [[s106_hybrid_loss_polish]].

# What
Seed 119 is the fast-tumbler wall (|ω·dt_ab| = 19.22 rad ≈ 3.1 turns across the A→B baseline; full LC ≈ 14.8 turns). s099's blind+fast pipeline closed seed 116 but failed 119 (top hi-fi ρ 33, Band D). The two hypotheses were (1) cloud STARVATION (sharpest anchor admitted too few pool members → no truth-near candidate), and (2) winding ALIASING (a truth-near pair connects to the wrong ω winding under a single finite-diff-init shoot). s100 is the user's 5-step redesign to attack (1) by coverage and *measure* whether (2) is the real wall.

# How
Script: `experiments/s100_5step_proto.py` (`S100_SEED=119`, Pool(24), v2 surrogate, fork CoW). Five steps, all blind on |ω| (s019 LS bracket, no truth, no T_pol):
1. **Blind all-epoch sharpness scan** — survivor count per epoch over a 30k SO(3) pool.
2. **Pick anchors A/B/C** — the 3 globally-sharpest epochs, greedily spaced ≥ GAP_MIN=20 epochs (blind index spacing). Got A=ep69(17 surv)/B=ep172(20)/C=ep377(25).
3. **Adaptive dense isophote fill at A,B** — Haar SO(3) batches, keep |pred−obs|<0.10 mag; sample until ≥5000 survivors/anchor or 24M cap. Used 12M.
4. **Decimate** each cloud to one rep per 2° geodesic cell (cap 2000 reps/anchor).
5. **Cross reps** → finite-diff-init `shoot` → connectability + bracket + C-photometry filters → coarse-K=50 RMSE → full-500 surrogate RMSE rank.
Built-in **probe (b)**: on the nearest-dense truth pair, compare single-shoot vs `multistart_shoot` (170-start) dir-err vs truth.
Hi-fi ρ-band via `experiments/s100_hifi.py` (back-prop each candidate to t=0, serial trimesh render, ρ = √(MSE)/0.05... reported as RMSE/band).

OOM note (this session): the 6M→24M adaptive fill peaks ~1 GB itself, but concurrent ComfyUI RAM use OOM-killed the run twice at STEP 3; re-ran clean to completion (989 s / 16.5 min).

# Result

| stage | metric | value | source |
|---|---|---|---|
| coverage A | nearest-truth dense / rep | 1.02° / 1.02° | invert.json `nt_a_dense`,`nt_a_rep` |
| coverage B | nearest-truth dense / rep | **0.68°** / 0.68° | invert.json `nt_b_dense`,`nt_b_rep` |
| probe single-shoot | dir-err / |ω| / in-bracket | 104.07° / 0.00415 rad/s / **false** | invert.json `probe` |
| probe multi-start | distinct roots / best dir-err | 94 / **4.58°** | invert.json `probe` |
| cross | C-pass survivors | 721 | invert.json `n_cross_survivors` |
| end-to-end | best surrogate full-RMSE | 2.41 (truth floor 0.018) | invert.json `best_full_rmse` |
| hi-fi | top blind ρ / band / n_bandA | **48.17 / D / 0** | hifi_rho.json |

The truth-near pair (A 1.02°, B 0.68°) is REJECTED by the cross because its single-shoot |ω|=0.00415 rad/s falls below the bracket floor (0.004289); the best *surviving* pair is 38.8°/68.8° off in attitude. Coverage delivered a truth-near pair; the single-shoot cross threw it away.

# Why this matters
Closes the starvation hypothesis: at 12M fill, BOTH anchors are sub-2° to truth — coverage is solved, and 1M/cloud is enough (the 12M was wasteful; see handoff). The binding blocker is now precisely localized to **single-shoot ω-selection on a slightly-off pair**, not candidate availability. This redirected the entire session away from densification/bracket/speed toward the pair→ω step.

# Numbers
- nt_a_dense 1.0167°, nt_b_dense 0.6838° (source: results/s100/seed119/invert.json:17-18)
- single_dir_err 104.07°, single_wmag 0.004150 rad/s, single_in_bracket false (source: invert.json `probe`)
- multistart 94 roots, best 4.58° (source: invert.json `probe`)
- best_full_rmse 2.408, truth_floor 0.018038 (source: invert.json:26)
- hi-fi top blind ρ 48.17 Band D, 0/3 rendered Band A (source: results/s100/seed119/hifi_rho.json)
- |ω·dt_ab| 19.22 rad ≈ 3.06 turns; dt_ab 743 s; truth |ω_a| 1.5023 deg/s (source: invert.json, s100 run log)
- wall 989 s / 16.5 min, Pool(24), 12M fill (source: invert.json `wall_s`)

# Artefacts
- `results/s100/seed119/invert.npz` (top-15 candidates + repA/repB caches), `invert.json`
- `results/s100/seed119/hifi_rho.json`
- `results/s100/seed119_run.log`, `seed119_hifi.log`

# Out of scope
- 1M-cloud rerun (argued sufficient, not run this session).
- The actual ω-selection fix (single-shoot → multi-start/return-map/polish) — see s105/s106.
- Other seeds; this is N=1 (119).

# Cross-references
- [[s099_blind_fast_invert_116]] — predecessor; seed 116 closed, 119 failed.
- [[s105_pairs_to_omega_decomposition]] — why single-shoot fails and what generators recover truth.
- [[s106_hybrid_loss_polish]] — the windowed-photometry polish that reaches Band A on 119.
