---
title: "s095 — cohort-calibrated truth-near RMSE threshold: transfers, but the load-bearing finding is winding-aliasing (near%=0 for half the cohort)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s095_calibration_threshold_transfer.py
  - notebooks/inversion/survey/results/s095/transfer.json
related:
  - experiments/s097_times0_gauge_bug_audit.md
  - experiments/s096_density_usefulness_116.md
  - experiments/s088_bvp_shoot_conditioning.md
created: 2026-05-22
updated: 2026-05-22
confidence: medium (RMSE-threshold side re-run post-fix; the winding-aliasing finding is high-confidence and bug-independent)
---

# TL;DR
Tests whether a truth-near windowed-RMSE retention threshold, calibrated on cohort seeds where truth is known, **transfers** to held-out seeds (the operational requirement, since we can't perturb the real truth at inversion time). Across 16 seeds (8 cal / 8 holdout, spanning |ω| 0.106-1.482 dps), held-out retention is 100%, but the per-seed p99 threshold spans **12.4×** raw — too loose to be a tight filter. **The load-bearing, bug-independent finding: for 8/16 seeds a pool-scale (2-7°) perturbation of truth's own anchors aliases to the wrong winding (`near%=0`) — `shoot` never lands a near-truth-ω candidate** (s088/s089 winding ladder). The RMSE-magnitude side of this experiment was originally corrupted by the `times[0]==0` bug (s097) and is re-run here post-fix; on correct geometry clean-seed truth-near RMSE is small (seed 116 raw_p99 0.30).

# What
The filter framing: rather than rank truth to #1, draw an RMSE threshold that retains truth while discarding junk. The threshold must be set without knowing the real truth → calibrate on a synthetic cohort split and test transfer to held-out seeds. This probes (a) does the threshold transfer, (b) does a normalized (÷ LC dynamic range) threshold transfer better, (c) does a truth-near-ω candidate even exist per seed.

# How
`experiments/s095_calibration_threshold_transfer.py`, Pool(24), v2 surrogate. 16 seeds picked evenly across the cohort |ω| range, interleaved into cal/holdout. Per seed: P=300 perturbations of truth at the two frac-of-span anchors (2-7°, the 30k-pool gap), `shoot` → ω, propagate from the anchor (`times[0]==0`, s097 fix), W_ACb windowed RMSE (raw + ÷dyn-range). Threshold = 99th pct on cal seeds; retention = fraction of held-out truth-near below it. `near%` = fraction of connecting perturbations with ω-dir < 15°.

# Result
- **Winding-aliasing (the headline):** `near%`=0 for 8/16 seeds (109, 67, 114, 53, 75, 85, 36, 119) — not |ω|-monotonic; a winding property of the A→B baseline. For these, the cross-cloud architecture has no truth-near-ω candidate to retain *regardless of the filter* (source: `results/s095/transfer.json`).
- **Transfer:** held-out retention 100% (raw, norm, near-ω), but per-seed p99 spread 12.4× raw / 9.9× norm — the threshold that retains all truth-near is too loose to be a tight filter. Normalization did **not** materially reduce spread.
- **Post-fix RMSE magnitudes:** clean (`near%`=100) seeds now have small truth-near RMSE (seed 116 raw_p99 0.298 / norm 0.322; seed 10 0.409) — the pre-fix "filter vacuous / truth-near worse than the mean" conclusion was a `times[0]` artifact (s097).

# Why this matters
- **Reframes the bottleneck for half the cohort:** it's not the discriminator/filter, it's *candidate existence* — winding-aliasing means a 30k-pool truth-near pair at this anchor baseline can't be connected to a near-truth ω. Levers: denser pool, or anchor baselines chosen to be in the unique-winding regime (s092 chose 116 *because* it has one).
- **The filter framing is the wrong frame** given s097: full-LC RMSE ranks truth directly (s093 post-fix rank 4), so a hard retention threshold is not needed — rank + polish is the path.

# Numbers
- `near%`=0 for 8/16 seeds; clean seeds `near%`≈100 (source: `results/s095/transfer.json` `per_seed`).
- held-out retention raw/norm 100%; per-seed p99 spread raw 12.4×, norm 9.9× (source: same).
- seed-116 truth-near raw_p99 0.298, norm_p99 0.322 (source: same).

# Artefacts
- `results/s095/transfer.json`, `results/s095/transfer.png`, `results/s095/truthnear_rmse.npz` (per-candidate rows: seed, pidx, rmse, rmse_n, ddir, geo).

# Out of scope
- Choosing anchor baselines to avoid aliasing (the `near%`=0 fix).
- Whether the threshold tightens if calibrated on clean seeds only.

# Cross-references
- `s097_times0_gauge_bug_audit.md` — the bug that corrupted the RMSE side here.
- `s088_bvp_shoot_conditioning.md` / `s089` — the winding-ladder this `near%`=0 finding instantiates cohort-wide.
- `s096_density_usefulness_116.md` — density lever for the clean seeds.
