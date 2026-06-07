---
title: s059i_validator_perturbed — q_a noise sensitivity of the ω-grid cost surface
type: validator
sources:
  - experiments/s059i_validator_perturbed.py
  - experiments/s059i_validator.md
  - experiments/s059_thread.md
related:
  - project_omega_grid_architecture.md
created: 2026-05-08
updated: 2026-05-08
confidence: high (3 trials × 6 perturbation magnitudes; bimodal regime at the realistic cloud-quantization scale identified clearly)
---

# TL;DR

The ω-grid + local-window surrogate-MSE cost surface (s059i premise) is
sharply discriminative for ω **up to ~7.5° q_a geodesic noise** on seed 28
at T_A=25 — truth-ω stays in top-16 across all trials. At 10° noise the
architecture is **bimodal**: 1/3 trials lands truth at rank 1, 2/3 trials
lands it at rank ~165. At 15° it fails (rank 128-422). The realistic SO(3)
survival cloud quantization at T_A=25 puts the closest-to-truth member at
**10.96°** (per s059h log) — exactly at the bimodal cliff. The s059j ω-grid
architecture as proposed is **borderline operational** at current cloud
density: it works ~33% of the time at this q_a noise, fails the rest.

# What

Sweep q_a perturbation magnitude over {0, 2.5, 5, 7.5, 10, 15}° (geodesic
on SO(3)) with 3 deterministic-RNG-axis trials per magnitude. At each
perturbed q_a, score the same Fibonacci 200-dir × 6-mag (±30%) ω-grid as
s059i_validator (1407 grid points with truth (dir, mag) at flat idx 0).
Report rank of truth-ω + top-1 ρ + top-1 ω error, and a 3-trial median.

# How

`experiments/s059i_validator_perturbed.py` imports `score_local_window`
and `fibonacci_sphere` from `s059i_validator`. Perturbation is implemented
as `Rotation.from_rotvec(np.radians(angle_deg) * unit_axis)` left-applied
to truth_q_a, with `unit_axis ~ N(0,I) / norm` (deterministic per trial).
Verified `quat_geodesic_deg(q_perturbed, q_truth) == angle_deg` to 1e-6.

# Result

| q_a perturb | rank median (range) | truth_ρ median | top-1_ρ median | top-1 = truth |
|-------------|---------------------|----------------|----------------|---------------|
|  0.00°      | 1   (1   – 1)       |  0.39          |  0.39          | 3 / 3         |
|  2.50°      | 1   (1   – 1)       |  5.56          |  5.56          | 3 / 3         |
|  5.00°      | 6   (2   – 16)      | 17.32          | 16.58          | 0 / 3         |
|  7.50°      | 2   (2   – 3)       | 11.84          |  9.96          | 0 / 3         |
| 10.00°      | 163 (1   – 166)     | 31.66          | 20.31          | 1 / 3         |
| 15.00°      | 346 (128 – 422)     | 34.08          | 22.85          | 0 / 3         |

Three regimes:

- **Robust (≤ 7.5° q_a noise)**: truth-ω stays in top-K (top-3 to top-16).
  Even though truth_ρ_local jumps from 0.39 (noise floor) to 5-17 (Band D
  by absolute scale), the cost surface preserves the *relative* ranking
  because every grid point is degraded by roughly the same q_a-noise floor
  lift.
- **Bimodal (~10° q_a noise)**: at the same nominal noise magnitude,
  outcome depends on the perturbation direction. One trial landed truth
  at rank 1 (ρ=4.09, Band B); the other two trials had truth at rank
  163, 166 (ρ=33). The top-1 grid point in the bad trials is at large
  ω-direction error (32-64°) and large |ω| error (-18 to -30%) — a
  spurious basin that local-window MSE prefers under that specific q_a
  drift direction.
- **Failing (≥ 15° q_a noise)**: truth-ω drops out of practical top-K
  (rank 128-422 / 1407 ≈ top 9-30%). The cost surface has been
  re-arranged enough by the q_a noise that truth is no longer
  distinguishable from spurious basins.

The realistic SO(3) cloud quantization at T_A=25 on seed 28 is **~7°
average spacing** with **closest-to-truth member at 10.96°** (per s059h
log). That places real-world s059j searches **right at the bimodal
cliff**.

# Why this matters

The s059i_validator (zero-noise) already demonstrated the ω-grid premise
is sound when q_a is correct. This perturbed validator localises the
**operating envelope** of that premise as a function of q_a noise:

- The architecture **works straight-ahead at q_a noise ≤ 5°**.
- It **degrades gracefully at 5-10°** (truth still top-K).
- It **breaks abruptly at 10-15°** (truth drops to rank 100-400+).

For s059j to be reliably operational at 100k-Sobol cloud density, one of
the following is required:

1. **Densify q_a sampling.** 200k or 400k Sobol gives ~5-6° average
   spacing, putting every seed comfortably inside the robust regime.
   Cost: 2-4× the cloud-generation step (currently ~5 min cold), and
   2-4× the (q_a × ω) inner loop in s059j. Still well within a few-minute
   wall budget per seed.
2. **Joint (q_a, ω) LM polish on top-K candidates.** The s059e LM
   converges to Band B from oracle-seeded truth cluster on seed 28 with
   `(rotvec, ω)` 6-DOF parameterisation. If we polish top-K = 100-200
   candidates jointly, basins within ~10° of truth-q_a should converge
   even when their seed-time score-rank is poor. Cost: K × ~3-50s per
   polish. K=200 × 25 s = 80 min — too slow as currently parameterised;
   needs a tighter LM budget or a Pool over polishes.
3. **Anchor selection.** Pick T_A where the closest-to-truth survivor is
   already < 7°. This depends on the per-epoch |C_t| breathing pattern
   and is seed-dependent. Pre-computable from cloud cache (~1s).

# Numbers worth remembering

- **Operating envelope**: q_a noise ≤ 7.5° → truth-ω at rank ≤ 16
  (deterministic). 10° q_a noise → bimodal (rank 1 or 165). 15° → fails.
- **truth_ρ_local at q_a-noise 10°**: 31.66 median (was 0.39 at 0°).
  Most of the cost-surface lift comes from k1/k2 drift over the local
  window; window length matters.
- **Realistic q_a noise at seed 28 T_A=25**: closest cloud member 10.96°
  (s059h), average spacing ~7°. Right at the cliff.
- **Cliff sharpness**: between 7.5° (robust) and 15° (failed), the rank
  jumps by ~100×.

Wall: 184 s for 6 perturbations × 3 trials × 1407 grid evals × 21-epoch
window = 25,326 evals at 137/s single-thread.

# Artefacts

- `experiments/s059i_validator_perturbed.py`
- `results/s059i_validator/seed028_T025_W10_perturbed/{run.log, summary.json,
  rank_vs_perturbation.png}`

# Out of scope

- Seed-class generality. Seed 28 has high |ω| (1.43 dps) which makes
  q_a noise compound rapidly over the 21-epoch window. Lower-|ω| seeds
  (e.g. seed 89, 0.24 dps; seed 14, 1.97 dps polhode-binding) will have
  different envelopes. Each needs validation before claiming the
  architecture works cohort-wide.
- Window-size sensitivity. W=10 was chosen to match s059e. Smaller W
  (e.g. W=5, 11-epoch window) reduces |ω|·dt drift and may widen the
  q_a-noise envelope at the cost of less ω-discrimination per pair.
- Non-random perturbations. Cloud quantization noise is structured
  (lattice-like), not Gaussian. The closest-to-truth cloud member at
  T_A=25 is at 10.96°, not Gaussian-sampled. The actual rank under
  cloud-quantized q_a is the next direct test.
- Joint LM polish from rank-100 candidates. Not yet measured.

# Cross-references

- `experiments/s059i_validator.md` — zero-noise baseline (rank 1/1407,
  ρ=0.389).
- `experiments/s059_thread.md` — s059h log for the 10.96° closest-cloud
  measurement.
- `project_omega_grid_architecture.md` — the architecture this validates.
