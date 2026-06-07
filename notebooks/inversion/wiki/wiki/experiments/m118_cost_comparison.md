---
title: "m118 — Kernel-factored IPL cost landscape diagnostic (seed 14)"
type: experiment
sources:
  - "raw/inversion_diagnostics/m118/seed_014/kernel.npz"
  - "raw/inversion_diagnostics/m118/seed_014/summary.json"
  - "raw/inversion_diagnostics/m118/seed_014/cost_facet_normal.npz"
  - "raw/inversion_diagnostics/m118/seed_014/cost_ipl_centroid_uniform.npz"
  - "raw/inversion_diagnostics/m118/seed_014/cost_ipl_centroid_weighted.npz"
  - "raw/inversion_diagnostics/m118/seed_014/cost_ipl_active_centroid.npz"
  - "raw/inversion_diagnostics/m118/seed_014/cost_ipl_centroid_weighted_ext.npz"
  - "raw/inversion_diagnostics/m118/seed_014/cost_ipl_active_ring.npz"
related:
  - "[[m115_surrogate_pipeline]]"
  - "[[m117_result_harvester]]"
  - "[[pab-contour-isoshell]]"
  - "[[alignment-cost]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[pab-contour-phase-angle-limitation]]"
  - "[[kernel-factorization]]"
  - "[[att-fail-diagnosis]]"
created: 2026-04-15
updated: 2026-04-16
confidence: high
---

> See also: [[m119_attitude_isoshell]] — tested the proposed surrogate-attitude-isoshell successor on seed 14. Threshold-based isoshell framing REFUTED (truth residual median 0.49 ≫ 0.1), but score-based mean_L1 ranks truth at 0/60000 and beats all m118 rank-1 competitors by 4000–13000-rank gaps.

# m118 — Kernel-factored IPL cost landscape diagnostic

## Hypothesis

IPL centroids are a more accurate generalisation of magnitude-band facet normals as targets for the alignment cost. Swapping facet normals for IPL centroids should tighten the cost basin around truth and reduce false minima from ATT_FAIL phi ambiguity.

## Method

**Kernel factorisation (see [[kernel-factorization]]):** decouple the expensive propagation step from the cost evaluation so multiple cost variants can be scored on the same grid without re-running the propagation.

- Grid: 2000 directions × 20 magnitudes (same as m102/103).
- Anchor selection upgraded to cascade by loop count: prefer 1-loop epochs, fall through to 2-loop, 3-loop, etc. Within a tier take the minimum IPL length among epochs whose length is below the 75th percentile. Trace saved in `kernel.npz`.
- Constraint epochs: union(spec peaks, IPL-tight epochs) − anchor. For seed 14 this is **255 epochs** (14 spec peaks + 250 tight-IPL − 9 overlap/anchor) vs the ~14 spec peaks in m102/103's cost.
- Kernel: `q_delta[2000, 20, 255, 4]` float32 = 156 MB. Propagated in 94 s on Pool(24).

**Cost variants scored** (each ~20 s at full 360 phi resolution, 2 anchor centroids; direction-chunk parallel Pool(16) inside the scorer):

1. `facet_normal` — baseline, allowed ±X/±Y/±Z per epoch from magnitude band
2. `ipl_centroid_uniform` — all centroids at each epoch, uniform weight
3. `ipl_centroid_weighted` — all centroids, weighted by `1/IPL_length`
4. `ipl_active_centroid` — truth-containing centroid only (ORACLE, upper bound of IPL info)
5. `ipl_centroid_weighted_ext` — variant 3 on the full 255-epoch extended set
6. `ipl_active_ring` — oracle with ring cost `(PAB·c − cos(ang_dist))²` instead of bullseye `(1 − PAB·c)²`

## Anchor chosen

Seed 14: epoch 119 (t=858.5 s), loop_count=**2**, IPL length 0.4674 rad, 2 centroids. No 1-loop epochs existed below the length-q75 cutoff so the cascade stepped to the 2-loop tier.

## Results (seed 14)

| Variant | Rank 1 (q0_err, w_dir) | Truth rank / 14.4M | Best ω rank in top-K | q0_err at best ω |
|---------|-------------------------|----------------------|-------------------------|--------------------|
| facet_normal | 168°, 52° | 3.83M | 4199 | 95.9° |
| ipl_centroid_uniform | 170°, 52° | 3.81M | 3038 | 94.9° |
| ipl_centroid_weighted | 169°, 52° | 4.56M | 2562 | 174.1° (twin) |
| **ipl_active_centroid (oracle bullseye)** | 110°, 35° | 575K | **310** | 81.5° |
| ipl_centroid_weighted_ext (255 eps) | 171°, **7.6°** | 2.06M | 9609 | 126.4° |
| ipl_active_ring (oracle ring) | 110°, 35° | 599K | 310 | 81.5° |

**Truth basin recovered = True in all 6 variants.** The correct answer sits somewhere in the top-1000 basins for every cost. But no variant puts it at rank 1 or anywhere near it; all variants assign lower cost to non-truth basins.

## Key findings

### 1. No IPL variant beats facet_normal decisively

The oracle variants (active_centroid, active_ring) give the best "truth ω in top-K" metric (rank ~310), but all five non-extended variants produce rank 1 at the *same wrong basin* (w_dir ≈ 52°, q0 ≈ 168°). The extended variant finds a rank-1 ω direction near truth (7.6° off) but still with wrong attitude (q0 171°).

### 2. Ring cost ≈ bullseye cost on this seed

`ipl_active_ring` differs from `ipl_active_centroid` only in the per-epoch form: `(PAB·c − cos(ang_dist))²` vs `(1 − PAB·c)²`. For small `ang_dist`, `cos(ang_dist) ≈ 1` and the two costs coincide. But at this seed's constraint epochs the ang_dists are *not* small: **median 25.7°, max 81.6°, only 4% below 5°.** So ring and bullseye *should* have produced meaningfully different rankings — and they didn't. Rank 1 is identical, truth rank is ~4% different. That is a signal in itself.

### 3. The real culprit: PAB-contour approximations

The pab-contour — from which all IPL centroids are extracted — assumes `k1 = k2 = h` (zero phase angle) and is computed with lo-fi (no shadows). Real observations have phase angles of 30-60° for IS-901 and major shadow effects (up to 4.6 mag at ±Y per [[shadow-asymmetry]]).

At seed 14 this mismatch manifests as **median ang_dist = 25.7°** between truth PAB and its nearest centroid [inline]. Truth does not pass through the centroid. The IPL level set is systematically offset from where the truth trajectory actually is.

**[inline]** The ang_dist distribution at seed 14's 255 constraint epochs was measured by a one-off inline Bash `python3 -c` call on `data/results/inversion_diagnostics/isoshell_viewer/ipl_all_epochs.npz` + `m118/seed_014/kernel.npz`. It is NOT an output of the m118 scripts. To reproduce: load `s014_ang_dists` and filter by `kernel['constraint_epochs']`.

See [[pab-contour-phase-angle-limitation]] for the full diagnosis.

### 4. The "extended epochs" variant is interesting

`ipl_centroid_weighted_ext` scores over 255 epochs rather than 14 spec peaks. It finds rank 1 with correct ω direction to within 7.6° (all other variants are at 52°). More constraint epochs help ω discrimination, even with bad target directions. But phi/attitude is still wrong. **Conclusion: adding data helps; fixing the targets is what we actually need.**

## Infrastructure that survives this experiment

Regardless of the negative cost-variant results, m118 produced reusable infrastructure:

- **Kernel factorisation pattern ([[kernel-factorization]]).** Propagation (expensive, depends only on grid + anchor + constraint epochs) is now decoupled from cost evaluation (cheap, depends on target directions + weights). Cost-variant experiments are ~20 s on 16 cores per variant. The `kernel.npz` for a seed is good forever.
- **Parameterised diag tool.** `MICRO118_SEEDS`, `MICRO118_VARIANTS`, `MICRO118_FORCE` env vars. Skip-if-exists by default — only re-scores variants whose cost NPZ doesn't exist. Enables "test one new cost variant" iteration without touching previously-computed ones.
- **Cascaded anchor selection.** The 1-loop > 2-loop > 3-loop preference with q75 length filter is likely the right anchor picker going forward. For seed 14 it chose a tight 2-loop epoch.

## Files

- Scripts: `notebooks/inversion/12_brightness_surface/m118_kernel_computation.py`, `m118_cost_comparison.py`, `m118_diagnostic_mode.py`
- Output: `data/results/inversion_diagnostics/m118/seed_014/`
  - `kernel.npz` (144 MB, **committed**) — propagated q_delta tensor + anchor metadata + truth pose
  - `cost_<variant>.npz` (~95 MB each × 6 variants = ~570 MB, **NOT committed** — regenerate with `MICRO118_FORCE=1 MICRO118_SEEDS=14 python3 m118_diagnostic_mode.py`, ~2 min)
  - `topK_<variant>.npz` (~280 KB each, **committed**) — top-10k candidates per variant with q0, ω, errors
  - `summary.json` (**committed**) — per-variant stats
  - `plots/` (4 MB, **committed**) — Mollweide + top-K scatter per variant + truth-rank summary

## What we learned (meta)

1. **Kernel factorisation is a paradigm win.** We tested 6 cost variants in ~15 min total instead of running full grid+NM pipelines.
2. **Ring vs bullseye isn't the bottleneck.** The cost form is a secondary concern relative to the target directions themselves being wrong.
3. **The pab-contour's physics approximations matter at the ~25° level.** This is larger than the precision required to put truth at rank 1 on a sub-degree grid.
4. **Surrogate-DE's success on omega selection** (validated [[surrogate-omega-selection]] 4/4) makes sense in this light: it uses full physics (phase + shadows via the surrogate), not the pab-contour's lie.

## Next experiment (proposed)

See [[surrogate-attitude-isoshell]]. Use the surrogate as `B(k1_body, k2_body, panel, dish, dist)` to compute per-epoch attitude isoshells in SO(3), then intersect across epochs. This generalises IPL to correct physics and maintains the constraint-set/geometric worldview.
