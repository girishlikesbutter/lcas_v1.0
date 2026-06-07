---
title: "s098 — localized densification CLOSES a blind Band-A inversion on seed 116; truth is one of several multi-solutions (full-LC RMSE non-monotonic, all top hi-fi rho<2)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s098_rescore_steer.py
  - notebooks/inversion/survey/experiments/s098_local_densify_116.py
  - notebooks/inversion/survey/experiments/s098b_hifi_rho.py
  - notebooks/inversion/survey/results/s098/rescore.json
  - notebooks/inversion/survey/results/s098/densify.json
  - notebooks/inversion/survey/results/s098/hifi_rho.json
related:
  - experiments/s097_times0_gauge_bug_audit.md
  - experiments/s096_density_usefulness_116.md
  - experiments/s093_fullc_score_survivors.md
  - experiments/s092_cross_cloud_116.md
created: 2026-05-25
updated: 2026-05-25
confidence: high on the closure + multi-sol verdict (hi-fi rho-band, renderer validated rho_truth=0.00); N=1 seed 116 (easiest); |w| prior band reuses truth (s092 stand-in)
---

# TL;DR
One blind round of **localized densification** (re-sample the brightness isophote only inside the surviving cloud region, steered by the s093 full-LC-RMSE rank) pulls a candidate from the coarse 5.65/3.23 deg nearest-truth gap down to **qa 0.91 / qb 1.22 deg, omega-dir 0.45 deg** — into the s096 <1 deg regime. The end-to-end pipeline **closes a blind inversion on seed 116**: the top candidate by surrogate full-LC RMSE is **hi-fi rho 0.34 (Band A)**. The twist: **truth is NOT the unique or best-fitting solution.** Full-LC surrogate RMSE is **non-monotonic in orientation error** — a ~15.5 deg q-offset / near-perfect-omega-dir basin (rho 0.34) and the body-twins (rho 0.37-0.51) fit the hi-fi LC *better* than a ~1 deg perturbation of truth (rho 1.09). All top-10-by-RMSE + the truth-near candidate are **hi-fi Band A** — the alternate minima are GENUINE multi-solutions, not surrogate artifacts.

# What
s097 left the loop-closing test unrun: does rank-by-full-LC-RMSE + densify/polish close into a blind Band-A inversion on seed 116? s096 proved that at <1 deg the discriminator works but only against *random junk* (floor 0.109) — it never probed whether a denser pool produces a truth-near candidate blind, nor what sits *below* the junk floor. This runs the densification the user prioritized (rank-4-at-18.6deg is not a polish seed; need a candidate actually near truth) and gates the survivors hi-fi.

# How
Three stages, Pool(24), v2 surrogate for search + the validated hi-fi forward model (`lib/hifi_render.py`) for the rho-band gate.

1. **`s098_rescore_steer.py` (steering, wall 289 s).** Re-score all 79,443 s092 survivors by full-LC RMSE (reusing s093's s097-fixed `_propagate_full`), saving every per-survivor array. Infra check: truth-LC floor **0.0102 mag**, `_propagate_full` reproduces truth to **1.71e-6 deg** — scoring path validated. Steering fact: the near-truth-q survivors (qa<8 & qb<5) rank at RMSE positions **85 / 224 / 420 / 435 / 1266**, so a blind top-N (N>=100) cut selects the best one (rank 85: qa 5.65, qb 3.37, dir 1.08). The rank-4 most-truth-like-OMEGA candidate is 18.6 deg off in orientation (useless to densify); the closest-qa survivor (5.65 deg) has dir 64.6 deg and ranks 58,759 — only the JOINT near-truth-q survivors are good seeds.
2. **`s098_local_densify_116.py` (densify + cross, wall 6,704 s — see Cost).** Top-N=250 RMSE survivors (blind). For each (q_a, q_b): densify each endpoint with M_PERT=800 isotropic rotvec perturbations (|rotvec| uniform in [0, R_DENSE=8] deg), keep those still on the brightness isophote (|pred-obs|<0.10 mag), cap K_KEEP=100, always include the parent. Neighborhood-restricted cross Da x Db: finite-diff init -> shoot -> connect (geo<1e-3) -> |w| in +-30% band (truth-|w| stand-in, s092) -> C-pass brightness check -> full-LC surrogate RMSE.
3. **`s098b_hifi_rho.py` (hi-fi gate, wall 528 s, 12 serial renders).** Back-propagate each candidate's (q_a, w_a) to its t=0 state (times[0]==0 gauge), `render_hifi`, `rho_from_hifi`. Candidates: exact truth (control) + top-10 by surrogate RMSE + the best-joint truth-near.

# Result

**Densification closes the geometric gap.** Nearest-truth orientation: A **5.65 -> 0.73 deg**, B **3.23 -> 0.67 deg**. Best joint candidate **qa 0.91 / qb 1.22 deg, omega-dir 0.45 deg, surrogate RMSE 0.0530** (source: `results/s098/densify.json`). Only 2 candidates reached joint<1.5 deg (truth's basin is sharp + thinly sampled even after densifying).

**Full-LC surrogate RMSE is non-monotonic in q-error** (non-twin, dir<10 deg; source: `densify.npz`):

| joint q-offset | min full-LC RMSE |
|---|---|
| 0-1.5 deg (truth basin) | 0.0530 |
| 1.5-3 deg | 0.0335 |
| 3-6 deg | 0.0320 |
| 10-20 deg | **0.0182** (global non-twin min) |

Truth floor 0.0102; the alternate minima sit at ~1.8x floor — within surrogate noise of truth. Top-100 by RMSE: 80 body-twins (dir>90), 20 with dir<10.

**Hi-fi rho-band gate — ALL Band A** (source: `results/s098/hifi_rho.json`):

| label | hi-fi rho | band | qa | qb | dir | surr-RMSE |
|---|---|---|---|---|---|---|
| truth (control) | 0.00 | A | 0 | 0 | 0 | 0 |
| rmse#1 | **0.34** | A | 15.55 | 15.56 | 0.14 | 0.0182 |
| rmse#2-9 (twins) | 0.37-0.51 | A | ~169-173 | ~129-175 | ~174-177 | 0.020-0.026 |
| rmse#10 | 0.52 | A | 56.62 | 69.33 | 6.16 | 0.0264 |
| best-joint (truth-near) | **1.09** | A | 0.91 | 1.22 | 0.45 | 0.0530 |

The renderer is validated (truth -> rho 0.00). The lower-RMSE-than-truth basins are **genuine hi-fi multi-solutions** (rho 0.34-0.52 < 2), NOT surrogate artifacts. The truth-near candidate is also Band A but fits *worse* (rho 1.09) than the alternates.

# Why this matters
- **The blind inversion CLOSES on seed 116.** Densify (s098) on top of cross (s092) + full-LC RMSE rank (s093) + hi-fi gate returns a **Band A (rho 0.34)** state blind. This is the loop-closing test s097 flagged unrun. Under the survey's stated metric (find >=1 Band A∪B basin; "fit the LC well, not recover truth"; rho<2 valid regardless of twin status), seed 116 is a **PASS**.
- **The "discriminator doesn't rank truth #1" worry dissolves into the multi-solution structure.** The non-monotonic RMSE is NOT a failure — full-LC surrogate RMSE faithfully ranks the best hi-fi fits to the top, and they are all Band A. Truth is one of several sharp Band-A basins; a 15.5 deg q-offset / same-omega basin and the body-twins fit at least as well. s096's "100% below junk" was true but blind to the sub-junk multi-solution family.
- **For *unique truth recovery* specifically, densify+RMSE is insufficient by construction** — truth's basin (rho 1.09 at 0.91 deg) is not the global min; densification finds valid alternates that out-score it. Disambiguating truth from its multi-solution family would need extra over-determination (3rd anchor, s089) or a tighter |w| / geometry prior, NOT more density.
- The 15.5 deg / same-omega quasi-degeneracy is distinct from the standard 180-deg body-twin (dir~0, qa~180) — a same-spin orientation offset that leaves the full-arc LC invariant to rho 0.34. Likely an IS-901 geometry x illumination near-symmetry; uncharacterized.

# Cost / process note
The densify cross ran **112 min vs a ~5 min estimate (22x overrun)**. Cause: densified local pairs are tightly connectable, so ~70% (1,750,658 / ~2.5M) passed connectability + |w| + C-pass and each got a full-LC score. Per "kill at 2x" discipline this should have been caught at ~10 min; it was backgrounded and only checked at completion. **Fix for any re-run / cohort scale:** cap survivors per seed, or full-LC-score only a capped top-K per neighborhood (connectability + C-pass is the cheap prefilter; full-LC is the expensive step and does not need all 1.75M).

# Numbers
- steering ranks of near-truth-q survivors: 85/224/420/435/1266 (source: `results/s098/rescore.json`).
- densify nearest-truth: A 5.65->0.73, B 3.23->0.67 deg; best joint qa 0.91/qb 1.22, dir 0.45, RMSE 0.0530, RMSE-rank 792 (source: `results/s098/densify.json`).
- non-monotonic min-RMSE by q-bucket: 0.0530 / 0.0335 / 0.0320 / 0.0182 (0-1.5 / 1.5-3 / 3-6 / 10-20 deg) (source: `densify.npz`).
- hi-fi rho: truth 0.00, rmse#1 0.34, twins 0.37-0.51, rmse#10 0.52, best-joint 1.09 — all Band A (source: `results/s098/hifi_rho.json`).
- walls: rescore 289 s, densify 6,704 s, hi-fi 528 s (12 renders ~44 s each).

# Caveats
- N=1, seed 116 — the deliberately-easiest seed (LAM-slow, unique winding, near%~100, s095). The aliased half of the cohort (near%=0, e.g. 119) has no truth-near-omega candidate to densify toward — orthogonal blocker.
- The |w| prior band (0.7-1.3 x |w|_true) reuses truth as the s092 stand-in for the s019 LS-bracket; the inversion is "blind" modulo this loose +-30% band.
- "Blind" pipeline = top-surrogate-RMSE -> hi-fi gate (returns rmse#1, rho 0.34). The truth-near candidate was located via oracle (argmin joint-to-truth) only to LABEL where truth sits; it is not part of the blind selection.

# Artefacts
- `results/s098/rescore.{npz,json}` — full per-survivor RMSE arrays + steering ranks.
- `results/s098/densify.{npz,json}` — densified survivors (rmse, dir, qa/qb-geo, parent, omega, q_a, q_b).
- `results/s098/hifi_rho.json` — hi-fi rho-bands of the gated candidates.

# Out of scope / next
- Efficiency re-architecture (cap full-LC scoring) before any cohort run.
- Aliased / fast seeds (119) — does the cross even admit a truth-near candidate (s095 near%=0)?
- Characterize the 15.5 deg same-omega quasi-degeneracy (IS-901 geometry symmetry?).
- Whether unique truth recovery needs 3rd-anchor over-determination (s089) on top of densification.
