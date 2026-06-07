---
title: "m127 — Flipped-ω compensating-q0 search on 11 baseline seeds"
type: experiment
sources:
  - "raw/inversion_diagnostics/m127_flipped_omega/batch_summary.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_000/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_006/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_014/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_024/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_027/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_033/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_036/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_046/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_074/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_093/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/stage_a_grid.npz"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/stage_b_polish.npz"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/stage_c_hifi.npz"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_033/stage_a_grid.npz"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_033/stage_b_polish.npz"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_033/stage_c_hifi.npz"
related:
  - "[[m126_wrapped_pipeline]]"
  - "[[omega-sign-degeneracy]]"
  - "[[multi-solution-philosophy]]"
  - "[[basin-of-attraction]]"
  - "[[symmetry-degeneracies]]"
  - "[[gradient-based-inversion]]"
  - "[[surrogate-model]]"
created: 2026-04-16
updated: 2026-04-16
confidence: medium
---

# m127 — Flipped-ω compensating-q0 search on 11 baseline seeds

## Status: MIXED — one WIDE-basin confirmation (seed 12), one FAILED positive control (seed 33) that reveals search-density bias

## One-line

Searching for flipped-ω compensating-q0 attractors on the 11-seed baseline cohort with a 60k super-Fibonacci SO(3) grid + top-20 L-BFGS-B polish finds **one new PARTIAL attractor (seed 12, hi-fi 0.171 at q0_err=140.36°)** but **FAILS to recover the known seed-33 basin** used as positive control. The method is UNDERPOWERED for narrow basins: [[m126_wrapped_pipeline]]'s seed-33 basin has q0-width sub-0.001°, far below the ~3° grid spacing.

## Hypothesis

From [[m126_wrapped_pipeline]]: seed 33 has a flipped-ω DE basin (`ω_cand ≈ -ω_true`, q0 ≈ 98.5° from truth) with hi-fi MSE 0.082. Inline probe [this session] confirmed pure ω-sign flip at `(q0_truth, -ω_truth)` gives surrogate MSE 3–10 on ALL 11 baseline seeds — no universal symmetry. Hypothesis: for ≥3 of the 11 baseline seeds OTHER than seed 33, a 3-DOF q0 search with ω fixed at `-ω_truth` finds a `(q0', -ω_truth)` state whose hi-fi MSE is < 0.5 (PARTIAL or better).

- **CONFIRMED (≥3 non-control seeds PARTIAL+):** flipped-ω-with-q0-compensation is population-wide; enumerate as a class in the multi-solution pipeline.
- **REFUTED (0–1):** seed-33-specific coincidence; keep [[omega-sign-degeneracy]] at `confidence: low`.
- **MIXED (1–2):** investigate the specific geometric regimes where it succeeds.

Seed 33 is run as a positive control (the known basin should re-appear if the search is well-calibrated).

## Method (5-stage per-seed pipeline)

Script: `notebooks/inversion/12_brightness_surface/m127_flipped_omega.py`. Run per-seed in a `Pool(8)` outer worker pool (with an inner `Pool(4)` cap for hi-fi validation).

1. **Load truth + setup.** `experiment_setup` at `2020-02-05T10:00:00 → 11:00:00`, 500 constraint epochs, `σ_obs = 0.05 mag`. Negate truth ω → `ω_search = -ω_truth`, preserving magnitude exactly.
2. **Stage A — SO(3) grid.** 60000 super-Fibonacci quaternions on S³ (median nearest-neighbour spacing ~3°). Score each as `surr_MSE(q0, ω_search)` over all 500 epochs via surrogate forward. Keep top-20 by surrogate cost.
3. **Stage B — L-BFGS-B polish over q0.** From each top-20 seed, run L-BFGS-B with FD Jacobian on 3-parameter `q0_tangent` space (ω held at `ω_search`). `ftol=1e-7, gtol=1e-4, maxiter=200, maxfun=1000`. Cluster polished endpoints by q0 distance (`5°` threshold); keep up to `MAX_BASINS=5` unique basins.
4. **Stage C — hi-fi validation.** Propagate each basin through the true BRDF+shadow hi-fi forward model; record per-epoch magnitudes and hi-fi MSE vs observed CE curve. Hi-fi pool capped at 4 to stay inside the Pool(8) outer budget.
5. **Classify.** `VALID` if winner hi-fi < 0.1, `PARTIAL` if < 0.5, else `FAIL`. Script emits `REFUTED` verdict if non-control VALID+PARTIAL count < 2.

Wall: **29:05 total for 11 seeds**, ~150–170 s/seed in Pool(8).

## Results (verified from NPZ + result.json on disk)

| seed | stage_a_best_surr | stage_b_best_surr | winner hi-fi MSE | winner q0_err (°) | class |
|-----:|------------------:|------------------:|-----------------:|-----------------:|:-----:|
|   0  | 1.239 | 0.914 | 2.313 | 168.83 | FLIPPED_FAIL |
|   6  | 1.006 | 0.904 | 2.545 | 170.92 | FLIPPED_FAIL |
|  12  | **0.615** | **0.230** | **0.171** | **140.36** | **FLIPPED_PARTIAL** |
|  14  | 1.079 | 1.432 | 4.338 |  62.30 | FLIPPED_FAIL |
|  24  | 1.145 | 0.999 | 2.943 |  86.29 | FLIPPED_FAIL |
|  27  | 1.132 | 1.142 | 3.071 | 138.28 | FLIPPED_FAIL |
|  33* | 1.183 | 0.978 | 2.578 |  91.43 | FLIPPED_FAIL |
|  36  | 1.163 | 0.987 | 2.472 | 129.95 | FLIPPED_FAIL |
|  46  | 1.083 | 0.638 | 1.247 | 125.89 | FLIPPED_FAIL |
|  74  | 1.106 | 0.359 | 0.738 |  38.11 | FLIPPED_FAIL |
|  93  | 1.191 | 1.142 | 3.727 |  83.86 | FLIPPED_FAIL |

(*) seed 33 = positive control.

**Counts:** VALID=0, PARTIAL=1 (seed 12), FAIL=10 (incl. control).
**Non-control PARTIAL+ count:** 1. Script-emitted `hypothesis_verdict = REFUTED`.

**Note on strategist-pasted table:** the pre-handoff summary listed seed 33 winner q0_err as "(control)" without a number. The NPZ/result.json show seed 33's winner (basin 0) at q0_err = 91.43° (not the m126 basin's 98.5°). Minor discrepancy, harmless for interpretation.

## Positive-control miss (seed 33) — the load-bearing finding

The m126 seed-33 flipped basin is known at `q0 ≈ 98.5°` from truth about body +Z (2.5° off), `ω = -ω_true`, surrogate MSE ≈ 0.086, hi-fi MSE ≈ 0.082. m127 Stage A on seed 33 produced:

- **Best grid surrogate MSE: 1.183** (top-20 range 1.183–1.190)
- **Nearest top-20 grid quaternion to the known-basin target:** 32.7° away (angular gap)
- **Stage B polish** from the top-20 converged to 5 basins at surr ~0.98–1.00, hi-fi 2.58–3.04

The known ~0.086-cost basin is **not on the grid at all** — at 60k super-Fibonacci spacing (~3° median), the q0-width of the basin (sub-0.001° per [[m126_wrapped_pipeline]]'s Δq0 ≤ 0.0004° polish move) is roughly 10000× narrower than grid resolution. L-BFGS-B cannot jump from a ~3° grid miss into a ~0.001° basin. **The search method is biased toward wide basins.**

This matters because it invalidates the naive `REFUTED` reading of the hypothesis: failing to find a narrow basin tells us nothing about whether such basins exist for the other 10 seeds. The experiment can only rule out WIDE flipped-ω attractors.

## Seed 12 — genuine discovery (WIDE flipped-ω basin)

Seed 12's winner is a **new** attractor never previously observed.

**Verified numbers** (from `seed_012/result.json` and NPZ):

- `q0_wxyz_winner = [−0.2934, +0.7807, +0.1345, −0.5351]`
- `q0_err = 140.36°` (from truth `[−0.8915, +0.2479, −0.3572, +0.1271]`)
- `ω_search = −ω_true` magnitude preserved to machine precision
- Stage A best surr **0.6147** — a full 0.5 below every other seed's best, indicating the basin is visible even at 3° grid spacing
- Stage B polish drops it to surr **0.2297** (3× cost reduction)
- Stage C hi-fi **0.171** — PARTIAL class (< 0.5), but not VALID (> 0.1). Above σ² noise floor (0.0025) by ~70×, so it's an observational degeneracy with a real but measurable LC mismatch.

### Is this a twin-relative of seed 12's m115 basins?

Compared winner q0 against the 3 m115 hi-fi basins (all at `+ω_true`, ω_dir-err 8.01°):

| m115 basin (ω = +ω_true, ω-dir err 8°) | raw quaternion angular distance to m127 winner |
|---|---|
| basin 0 (q0_err 10.2°, hi-fi 0.327) | 149.63° |
| basin 1 (q0_err 170.8°, hi-fi 0.501) | 61.68° |
| basin 2 (q0_err 172.8°, hi-fi 0.650) | 108.98° |

Tested all six `{±180X, ±180Y, ±180Z} × {left, right}` multiplications on each m115 basin — NO match within 0.05 quaternion distance. The m127 winner is **NOT a simple left/right-multiply twin** of any m115 basin. It is a **genuinely independent attractor** in the flipped-ω half of parameter space.

### Body-frame compensation axis (seed 12) — NOT body +Z

`q_comp_body_truth = q0_truth⁻¹ · q0_cand`, expressed in seed-12 truth body frame:

- angle = **140.36°**
- axis = `[−0.848, −0.485, +0.215]` in truth body frame
- angle to body +Z = **77.6°** (orthogonal-ish — definitely not aligned)
- angle to body +X = **32.1°** (closest)
- angle to body +Y = **61.0°**

**Contrast with seed 33** (m126's known basin, inline this session): compensation 98° about axis `[0.041, −0.013, +0.999]` — 2.5° off body +Z.

**Conclusion:** the "body +Z is the structural axis for flipped-ω compensation" hypothesis from the seed-33 analysis **does NOT hold for seed 12**. Seed 12's compensation is closest to body +X (panels-perpendicular direction) with a large component along body −Y. The compensation geometry is less constrained than the seed-33 panel-deployment-axis picture suggested. Flipped-ω compensation is **seed-specific in both rotation angle AND body-frame axis**, not a single shared structural symmetry.

## Why does seed 12 have a wider basin than seed 33?

Three candidate mechanisms, listed with concrete falsification tests (none run here):

1. **ω-magnitude effect.** |ω_true| for seed 12 = 0.01524 rad/s, for seed 33 = 0.02525 rad/s (seed 33 is 1.66× faster). A slower tumble samples less of SO(3) over the 3600 s window, which may leave more attitudes observationally equivalent. Falsifiable by: re-running m127 on seed 12 with an artificial |ω| rescale to match seed 33's magnitude and checking whether the basin narrows.

2. **Observation-window phase-angle coverage.** Seed 12 happens to have a phase-angle regime where the BRDF's dominant contribution is from diffuse/panel geometry (less sensitive to body-Z orientation), while seed 33 samples a specular-glint regime (more sensitive). Falsifiable by: computing the per-epoch `k1·k2` distribution for both seeds and checking whether seed 12's window has systematically higher phase angles.

3. **Initial orientation near a principal-axis alignment.** If seed 12's truth q0 happens to put a major symmetry plane of the satellite near the sun-observer bisector, the compensating rotation has more "room" to land. Falsifiable by: computing the angle between body +X (shortest bus dimension) and the sun-observer bisector at t=0 for each seed and correlating with basin width.

**Preferred test (cheapest):** (1). Takes ~150 s to re-run seed 12 with a scaled ω. If basin width shrinks with larger |ω|, we have a mechanism. This is NOT run in this experiment.

## Lazy-explanation audit (on the above)

Re-read of the "why wider" section — each candidate mechanism is explicitly speculative with a named falsification. None is asserted as confirmed. The seed-33 comparison ("seed 33 is 1.66× faster") is a verified number (`np.linalg.norm(truth_omega)` on both seeds). The connection from |ω| to basin width is physics speculation marked as such.

Body-frame axis claims in the seed-12 section ARE verified: quaternion math done in-session on loaded NPZ q0 arrays, results reported to 3 significant figures.

## Honest verdict

**MIXED, with the search-density caveat front and centre.** The script's automatic `REFUTED` verdict is technically correct at face value (only 1 non-control PARTIAL+, below the 3-seed threshold) but hides two facts:

1. **Seed 12 is a genuine new observational degeneracy** — hi-fi 0.171 at `(q0_err=140°, -ω_true)`, independent of any m115 basin or simple twin. This is exactly what [[multi-solution-philosophy]] says the pipeline should surface as a valid labelled solution.
2. **The positive control failed** — seed 33's known 0.082-MSE basin was invisible to the 60k grid because the q0-width is ~10000× narrower than grid spacing. The search cannot reject the existence of narrow flipped-ω basins on the other 9 FAIL seeds; it can only confirm the existence of WIDE ones.

More precise category: **UNDERPOWERED-NEGATIVE with one WIDE-basin confirmation.** Re-running with warm-starts from m115 DE basins' `ω = -ω_upstream` (cheap, reuses existing basins) OR with a 10–100× denser grid (expensive) could still overturn this.

## What we learned

1. **Flipped-ω attractors can be WIDE.** Seed 12's basin is visible at 3° grid spacing (Stage A best 0.61). Not all flipped-ω basins are m126-seed-33-narrow. **This raises [[omega-sign-degeneracy]] confidence from `low` to `medium`** — we now have 2/11 seeds with confirmed flipped-ω solutions (1 narrow + 1 wide).

2. **Body +Z is NOT a universal compensation axis.** Seed 33's 98° about body +Z is seed-specific. Seed 12's 140° is closest to body +X with a large −Y component. The geometric structure of flipped-ω compensation varies by seed, consistent with "mixed spatial-temporal symmetry with seed-specific geometry" in [[symmetry-degeneracies]] rather than a clean universal group.

3. **L-BFGS-B polish from coarse SO(3) grid is underpowered for narrow basins.** 60k super-Fibonacci (~3° median) misses seed 33's known <0.001° basin by ~10000×. Future gradient-based attacks on this landscape must warm-start from DE output, densify the grid 10–100×, or accept that ONLY wide basins will be found. See note added to [[gradient-based-inversion]].

4. **Surrogate-to-hi-fi correlation holds for seed 12** (surr 0.230 → hi-fi 0.171, ratio 0.74×). The 10–15× modelling-error regime from [[m124_hifi_validate]] seed 27 doesn't appear here. That's reassuring but 1 data point.

## Next-experiment queue

1. **m128 (recommended).** Warm-start Stage A with the 11 seeds' **m115 DE basins at `ω = -ω_upstream`**. For each seed, take its 3–5 m115 basins, negate their ω, run L-BFGS polish over q0 from each. Cost: ~20 s/seed × 11 = ~4 min (NO grid stage). If seed 33's known basin appears from warm-start, the grid-density hypothesis is confirmed. If other seeds produce new PARTIAL+ solutions, flipped-ω becomes population-wide.
2. **m129 (medium-cost).** Densify seed 33's SO(3) grid 10× (600k points, ~15 s surrogate per seed) to check whether grid-density alone recovers the known basin.
3. **m130 (expensive, deferred).** DE-over-q0 with ω fixed at `-ω_true`, population 200, 10 restarts per seed. ~150 s/seed × 11 = ~27 min. More expensive than m127 but should find narrow basins the grid misses.

## Plots + checkpoints

Per-seed outputs preserved in `data/results/inversion_diagnostics/m127_flipped_omega/seed_{NNN}/`:
- `stage_a_grid.npz` — 60k surrogate scores, top-20 quaternions
- `stage_b_polish.npz` — up to 5 polished basins per seed, iteration counts, polish metadata
- `stage_c_hifi.npz` — hi-fi MSE + 500-epoch hi-fi magnitudes per basin
- `result.json` — structured per-seed summary (4 classes: VALID, PARTIAL, FAIL, ERROR)
- `run.log` — per-seed pipeline log

Batch-level:
- `data/results/inversion_diagnostics/m127_flipped_omega/batch_summary.json` — 11 seeds × 6 headline columns + config snapshot
