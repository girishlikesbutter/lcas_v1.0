---
title: "s097 — times[0]==0 gauge-bug audit: re-validation REVERSES s093 and s094"
type: audit
sources:
  - notebooks/inversion/survey/lib/jacobi_propagator.py
  - notebooks/inversion/survey/experiments/s093_fullc_score_survivors.py
  - notebooks/inversion/survey/experiments/s094_window_score.py
  - notebooks/inversion/survey/experiments/s095_calibration_threshold_transfer.py
  - notebooks/inversion/survey/experiments/s096_density_usefulness_116.py
related:
  - experiments/s093_fullc_score_survivors.md
  - experiments/s094_window_score.md
  - experiments/s096_density_usefulness_116.md
created: 2026-05-22
updated: 2026-05-22
confidence: high (fix validated two independent ways: exact-truth reproduction 1.7e-6deg vs 108deg buggy, and s093/s094-W_full agree bit-for-bit)
---

# TL;DR
A `times[0]==0` precondition violation in every wrapper that propagated a candidate **from a non-zero anchor epoch** (`_TREL = times0 - times0[ep_a]`, first element negative) silently mis-referenced the trajectory by up to **107.9°** — `propagate_jacobi_path2` pins q0 at the first time sample under a `phi(0)=0` gauge (`lib/jacobi_propagator.py:669`), so it is only self-consistent when `times[0]==0`. The bug was caught by an infra check in s096 (exact truth scored **1.67 mag** against its own LC; correct floor is **0.0077 mag**). Fixing it **REVERSES s093 and s094**: full-LC RMSE is in fact a **strong** discriminator (most-truth-like survivor rank **31,515 → 4** of 79,443), and the *full* LC is the **best** window — windowing **hurts** (the entire "between-anchor residual" thread was a bug artifact). The connectability/ω-dir/`shoot` results (s088/s089/s092, and s095's winding-aliasing) are **untouched** — `shoot` always passed `times=[0,dt]`.

# What
The discriminator thread (s093 → s094 → between-anchor windowing → s095 filter) was built on windowed/full-LC surrogate RMSE. s096's infra block (built to validate the density-vs-discriminator basin probe) scored *exact truth* over the W_ACb window and got 1.67 mag — a floor that should be the ~0.008 mag surrogate noise floor. That is a bug signature, not physics. This audit isolates it, fixes it everywhere, and re-runs the affected experiments.

# How
**Root cause.** `propagate_jacobi_path2(q0, omega0, inertia, times)` reconstructs R(t) by pinning `R_J→L` from `(q0, theta_hist[0], psi_hist[0])` under the gauge `phi(0)=0` (`lib/jacobi_propagator.py:669-674`). `theta_hist[0]`/`psi_hist[0]` are the values at `times[0]`; the phi gauge is at the time-axis zero. These coincide only when `times[0]==0` — the docstring states it (`:600`). The scoring wrappers passed `_TREL = times0 - times0[ep_a]` (so `times[0] = -times0[ep_a] != 0`), pinning the candidate at the wrong epoch.

**Fix.** Propagate with the anchor at `times[0]==0`:
- forward-only windows (s095/s096, window ⊆ [ep_a, end]): slice geometry from `ep_a`, pass `times0[ep_a:] - times0[ep_a]`.
- full-LC windows (s093/s094): `_propagate_full` = forward `[ep_a:]` + a **reversed** backward call `(times0[:ep_a+1]-times0[ep_a])[::-1]` (each with `times[0]==0`), stitched into a full (N,4) history.

**Validation.** `_verify_propfull_stitch.py` (now deleted): the stitch reproduces the truth trajectory to **1.71e-6°** over all 500 epochs vs **107.9°** for the old way. s096 infra: exact-truth floor **0.0077 mag** (correct) vs **1.6738 mag** (buggy). Independent cross-check: post-fix s093 and s094's `W_full` agree bit-for-bit (top-100 = 54, top-500 = 235).

# Result

| experiment | conclusion BEFORE (buggy) | conclusion AFTER (fixed) | verdict |
|---|---|---|---|
| **s093** full-LC RMSE rank | "fails to rank truth; most-truth-like rank **31,515**/79,443; top-100 **0** within 10°" | most-truth-like (ω-dir 0.11°) rank **4**/79,443; top-100 **54%** within 10° (baseline 2.5%) | **REVERSED — strong discriminator** |
| **s094** window comparison | "**W_ACb best** (top-100 19→was-12); W_AB worst; full-LC washes out" | **W_full best** (top-100 **54**) > W_ACb (19) > W_AC (13) ≈ W_AB (12) | **REVERSED — windowing HURTS** |
| **s095** truth-near RMSE / filter | "filter vacuous; truth-near RMSE > LC dyn-range (norm>1)" | clean-seed truth-near RMSE small (seed 116 raw_p99 **0.30**, norm 0.32) | **REVERSED on RMSE side** |
| s095 winding-aliasing | `near%`=0 for 8/16 seeds | unchanged | **STANDS** (from `shoot`) |
| s092 / s088 / s089 connectability | — | unchanged | **STANDS** (`times=[0,dt]`) |
| **s096** density usefulness | (was bug-corrupted) | δ<1° → RMSE ~0.03 mag, 100% below junk, ω-dir<1.25° | **density lever CONFIRMED** |

Note the post-fix s093 top is a mix of truth-near (ω-dir 0.11°, 6°) AND 180° body-twins (ω-dir 174-178°, rank 1) — expected (observational indistinguishability), not a defect.

# Why this matters
- **The discriminator we spent four experiments working around was never broken.** Full-LC pointwise RMSE ranks the truth-near survivor to rank 4 on seed 116, blind. The windowing (s094) and cohort-calibration-filter (s095) detours were rationalisations of a false negative.
- **The "*short window* beats the full LC" claim is REFUTED on correct geometry** — more light curve is better. NOTE the underlying *insight* (the interior between anchors is free, unconstrained signal that discriminates by breaking twin-degeneracy) STANDS and is in fact *confirmed* — even W_AB beats baseline, and W_full is best because more of that free signal helps. Only the "divergence-room vs phase-fragility → window it down" operationalization was the artifact ([[project_cross_cloud_interior_is_free_signal]]).
- **NOT yet established:** that ranking + polish closes into a blind *inversion* (polish-top-K → Band A → hi-fi) — untested. And seed 116 is the deliberately-easiest seed (s092); the `near%`=0 aliased half of the cohort has no truth-near-ω candidate to rank.
- **Methodology win:** the infra check (score exact truth, demand it floors at the surrogate noise level) is what caught this. Make it standard for any new q→LC scoring wrapper.

# Numbers
- buggy vs correct: exact-truth reproduction 107.9° vs 1.71e-6°; floor 1.6738 vs 0.0077 mag (source: `results/s096/basin.json`).
- s093 post-fix: most-truth-like ω-dir 0.11° → RMSE rank 4/79443; top-100 54/100 within 10° (source: `results/s093/score.json`).
- s094 post-fix top-100 within 10°: W_full 54, W_ACb 19, W_AC 13, W_AB 12 (source: `results/s094/windows.json`).
- s095 post-fix: seed-116 truth-near raw_p99 0.298 / norm_p99 0.322; per-seed p99 spread 12.4× (source: `results/s095/transfer.json`).
- s096 post-fix: δ=1° rmse_p50 0.033 (100% < junk_p1=0.109, 100% ω-dir<1.25°); δ=2° 95%/71%; δ=3° 63%/35% (source: `results/s096/basin.json`).

# Artefacts
- Fixed scripts: `experiments/s093_fullc_score_survivors.py`, `s094_window_score.py`, `s095_calibration_threshold_transfer.py`, `s096_density_usefulness_116.py` (all carry the `times[0]==0` rationale in comments / `_propagate_full`).
- Re-run outputs: `results/s093/score.json`, `results/s094/windows.json`, `results/s095/{transfer.json,transfer.png}`, `results/s096/{basin.json,basin.png}`.

# Out of scope
- Polish-top-K → Band A → hi-fi (the loop-closing test). Not run.
- Whether s076/s080/s085-s089 (other prior windowed-RMSE-style scorers, if any) used the same `_TREL` pattern — only s093/s094/s095/s096 audited here.
- Re-run on aliased / fast seeds (119 etc.); seed 116 only.

# Cross-references
- `s096_density_usefulness_116.md` — the density confirmation + where the bug surfaced.
- `s093_fullc_score_survivors.md`, `s094_window_score.md` — carry RETRACTION banners pointing here.
- Memory: `feedback_propagator_times0_gauge` (the gotcha), `project_full_lc_rmse_discriminator_works` (the reversed state).
