---
title: "s018a — Bright-band PAB-alignment calibration on post-fix m048 cohort"
type: experiment
sources:
  - "results/s018a/calibration.npz"
  - "results/s018a/summary.json"
  - "results/s018a_run.log"
related:
  - "[[s018b_face_identity_tiers]]"
  - "[[s001_cost_at_truth_cohort]]"
  - "[[concepts/known_pathologies_to_revalidate]]"
created: 2026-05-02
updated: 2026-05-02
confidence: high
---

## TL;DR

**Buggy-era `bright_threshold = 11` (inherited from m103 spec-peak detection) is too loose under correct truth: at `mag < 11` only 22% of "bright" epochs are within 5° of PAB. Calibrated post-fix threshold is `mag < 8` — P(min_ang_dist < 5° | mag < 8) = 0.60, rising to 0.68 at mag < 7 and 1.00 at mag < 6.** The cached `min_ang_dist`, `best_group`, `group_flux`, and `hifi_peak_epochs` fields in every `traj_seedXXX.npz` are exactly the calibration ground truth — no STL load, no PAB compute, no rendering needed. Cohort-wall 0.6 s.

**m096 buggy-era claim "87% of seeds lack bright ±X spec constraints" softens to 56% under correct truth.** Still a majority, so any inversion architecture that relies on ±X anchors is structurally limited to ≤44/100 seeds — but multi-face tier classification (s018b) opens broader coverage.

## What

Cohort aggregation that re-validates the buggy-era "bright peak ⇒ some body normal is PAB-aligned" claim under correct truth, and produces a calibrated bright-magnitude threshold for downstream consumers (s018b face-identity tier classifier; future phi-sweep IC generator and bright-peak ω-mag prior).

The buggy-era pipelines used `mag < 11` as a spec-peak filter throughout m103/m115. `lib/surrogate_eval.py:DEFAULT_BRIGHT_THRESHOLD = 11.0` was inherited as-is into the survey. This experiment tests whether that threshold is appropriate under the post-fix forward model, and how cleanly bright magnitude maps to PAB alignment of a body-frame face normal.

## How

Pure cohort aggregation on cached NPZ fields:

1. For each of 100 m048 seeds, load `mag_hifi`, `min_ang_dist`, `best_group`, `hifi_peak_epochs` (all pre-computed at truth in the post-fix generation pass `m048_generate_trajectories_v2.py:243-251`).
2. Pool 100 × 500 = 50,000 epochs into flat arrays.
3. Compute conditional probabilities `P(min_ang_dist < {1°, 3°, 5°, 10°} | mag < B)` for B ∈ {6, 7, 8, 9, 10, 11, 12, 13}.
4. Pick `BRIGHT_THRESHOLD_POSTFIX` as the largest mag bin where `P(min_ang_dist < 5°) ≥ 0.5`.
5. Per-group spec dominance: at the geometric ground-truth criterion (`min_ang_dist < 5°`), tally `best_group` frequencies.
6. m096 re-test: count seeds where any epoch has `min_ang_dist < 5°` AND `best_group ∈ {0, 1}` (±X).
7. Per-seed bright-peak count distribution at the calibrated threshold.

No propagation. No surrogate eval. No rendering. Just `np.load`.

## Result

**Calibrated bright threshold: `mag < 8`**.

| mag bin B | n epochs (pooled) | P(<1°) | P(<3°) | P(<5°) | P(<10°) |
|---|---|---|---|---|---|
| 6 | 161 | 0.043 | 0.354 | **1.000** | 1.000 |
| 7 | 548 | 0.036 | 0.339 | **0.681** | 1.000 |
| 8 | 1269 | 0.025 | 0.232 | **0.597** | 1.000 |
| 9 | 2101 | 0.016 | 0.142 | 0.416 | 1.000 |
| 10 | 2908 | 0.011 | 0.103 | 0.302 | 0.961 |
| 11 | 3970 | 0.008 | 0.076 | **0.221** | 0.843 |
| 12 | 6564 | 0.005 | 0.046 | 0.134 | 0.523 |
| 13 | 17377 | 0.002 | 0.017 | 0.051 | 0.198 |

**Per-group spec frequencies (geometric criterion `min_ang_dist < 5°`, n=885):**

Spec contributions are FLAT across the 10 face groups — face area does not predict spec frequency. ±X (97.3 m², largest area) contribute only ~9% each; ±Z (22.8 m²) contribute ~12%; even the 9.8 m² dishes contribute ~9% each. What determines spec frequency is how often the face normal sweeps through the PAB locus during the 1-hour observation, which depends on tumbling dynamics — not face area.

**m096 re-test:** **56/100 seeds LACK any ±X spec event under correct truth** (was 87% under buggy truth). Buggy-era's "±X anchors are rare" framing survives but softens by 31 pp.

**Per-seed bright-peak count (mag < 8):** median 12, p10 = 0, p90 = 27. **19/100 seeds have zero bright peaks** — fallback to Sobol regardless.

## Why this matters

This is a methodological reset for any inversion architecture that anchors on bright peaks (m103 phi-sweep, the unbuilt phi-sweep IC generator, the bright-peak ω-mag prior).

- The buggy-era m103 was using `mag < 11` as its spec-peak filter, which under correct truth admits 78% non-spec contamination.
- The pre-existing bright_threshold=11 inherited by the survey via `lib/surrogate_eval.py` is similarly contaminated.
- Spec-event identification under correct truth requires `mag < 8` AND ideally `mag < 6` for high-purity (T1) classification.
- The face-identity classifier (s018b) builds directly on this threshold to give a *brightness-tier → candidate face set* lookup — the structured IC generator the survey was missing.

The cached `ang_dist[g, t] = angle(n_g_body, pab_body[t])` saves us from ever needing to recompute PAB or STL face normals during the survey. This is load-bearing infrastructure.

## Numbers (canonical)

- Cohort wall: 0.6 s on cached arrays.
- 50,000 pooled epochs.
- 885 spec events (geometric ground truth, `min_ang_dist < 5°`).
- Calibrated bright threshold: `mag < 8`.
- m096 re-test: 56% lack ±X, 44% have ≥1 ±X spec event (n=100 seeds).
- Per-seed bright (mag<8): median 12, p10=0, p90=27, 19 with zero.

## Artefacts

- Script: `experiments/s018a_bright_band_calibration.py` (220 lines).
- NPZ: `results/s018a/calibration.npz` — pooled `(mag, min_ang, best_group, seed, at_peak)` arrays + per-seed counts at multiple thresholds.
- JSON: `results/s018a/summary.json` — full P(align | mag) table, per-group counts, m096 re-test result.
- Plots: `results/s018a/p_align_vs_mag.png`, `results/s018a/group_dominance.png`, `results/s018a/per_seed_bright_count.png`.
- Run log: `results/s018a_run.log`.

## Out of scope

- Phase-angle dependence of the bright-mag-to-PAB-alignment relationship. Spec events at different phase angles have different specular-lobe geometries; the cohort aggregation marginalises over phase angle. Per-phase-angle calibration could be added if it matters downstream.
- Face-identity classifier (covered in s018b — built on this calibration).
- Distance-normalisation of the magnitude scale. Within m048 the cohort obs_dist range is 38580–38657 km (slant range to GEO from a fixed ground station); the apparent-vs-absolute correction is empirically <0.004 mag. s018b applies the formal correction (`mag_abs = mag_apparent − 5 log10(obs_dist / D_REF_KM)`) for portability across observer geometries.

## Cross-references

- [[s018b_face_identity_tiers]] — formal tiered classifier built on this calibration.
- [[s001_cost_at_truth_cohort]] — cohort-wide cost-at-truth, established that the surrogate is honest under correct truth on all 100 seeds.
- [[concepts/known_pathologies_to_revalidate]] — m103 spec-peak filter `mag < 11` is now formally re-calibrated to `mag < 8`.
- m103/m115/m126 (parent project, contraband) — the buggy-era pipelines that used `mag < 11`.
- m096 (parent project wiki, frozen) — buggy-era 87%-lack-±X claim that this experiment re-tests.
