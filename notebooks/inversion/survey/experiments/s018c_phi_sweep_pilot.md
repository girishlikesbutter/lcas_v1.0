---
title: "s018c — phi-sweep IC generator + ω-grab diagnostic (pilot CANCELLED at the grab gate)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s018b_face_identity_tiers.md
  - notebooks/inversion/survey/experiments/s018a_bright_band_calibration.md
  - notebooks/inversion/survey/experiments/s011_q4cii_sobol_so3_polish_pilot.md
  - notebooks/inversion/survey/experiments/s015_joint_q0_omega_pilot.md
  - notebooks/inversion/survey/experiments/s016c_prime_fresh_sobol_fixed_omag.md
  - notebooks/inversion/survey/experiments/s017_hifi_rho_band_s016c_prime.md
  - notebooks/inversion/survey/experiments/s003_landscape_vs_omega.md
related:
  - notebooks/inversion/survey/concepts/quaternion_convention.md
  - notebooks/inversion/survey/concepts/q_omega_coupling.md
  - notebooks/inversion/survey/concepts/known_pathologies_to_revalidate.md
created: 2026-05-02
updated: 2026-05-02
confidence: high
---

## TL;DR

s018c packages the s018b face-identity tier classifier into a *structured*
q0 IC generator: for each bright peak, the body-frame face normal `n_g`
of every candidate face in the tier shortlist is rotated to align with
PAB at the peak epoch, then phi-swept around the PAB axis (1-DOF residual
freedom). Each phi value, combined with a candidate ω from a coarse grid,
back-propagates through `Φ⁻¹(t_peak; ω)` to produce a `q0` initial
condition that satisfies "face g aligned with PAB at peak t" by
construction.

**Decision: full 5-seed pilot CANCELLED at the design-gate.** A pre-pilot
ω-perturbation diagnostic (`scratch/s018c_omega_grab.py`) on seed 6
measured the joint LM grab around truth-ω as **|Δω-dir| ≤ 1°** and
**Δω-mag ∈ [-10%, +5%]** — precisely the s003-style ω-fragility tube
that killed joint Sobol architectures. The phi-sweep q0-IC primitive is
sound (at truth-ω it lands seed 6 at q0_err = 0.287°, surr_MSE 2.79e-3
≈ s001 truth_mse_ref to 4.5%), but s018c does not solve ω-search; any
feasible ω-grid is too coarse to put a cell inside the tube on most
seeds. **Pivot to S016-A** as the principled forward path; phi-sweep is
retained as a **q0-IC sub-component** for any S016-A grid cell that
lands inside the tube.

Three pre-pilot diagnostics drove the decision:

1. **Smoke validation (`scratch/s018c_smoke.py`).** PAB-alignment
   geometry holds at machine precision under post-fix truth (alignment =
   the cached `min_ang_dist`). Back-propagation convention is **LEFT**:
   `q0 = Φ⁻¹ ⊗ q_target` where `Φ` is the propagation of (identity, ω0)
   to t_peak. Closest-IC q0 error matches `min_ang_dist` exactly (3.15° vs
   3.12° on the +Y-face spec5 peak of seed 6) — the IC generator is
   geometrically tight.
2. **Pre-launch tier audit.** Two of the originally-planned pilot seeds
   (41 and 91) have **zero classifiable peaks** under the s018a-calibrated
   `mag_abs < 8` threshold — they fall in s018b's 19/100 zero-classifiable
   cohort. The phi-sweep IC generator produces zero ICs for them. Pilot
   composition was revised to {6, 28, 79, 44, 84} preserving role coverage.
3. **ω-grab diagnostic.** 15 cells × 32 LM-polished ICs each on seed 6,
   sweeping ω-dir at {0,1,5,10,20,30,60}° and ω-mag at
   {-50,-20,-10,-5,+5,+10,+20,+50}% from truth. Decisive negative on
   the joint-search ω-grid architecture: only |Δdir| ≤ 1° AND
   Δmag ∈ [-10%, +5%] keep the LM-best landing in basin. Outside this
   tube the LM-best q0_err jumps to 100°-180° and final_surr_MSE to
   1.9-3.4 mag² (Band D). Wall: 36 min on Pool(8) with N_PHI=12, single
   cell at a time.

## What

Run a 5-seed pilot of the joint (q0, ω) inversion architecture with:
- q0 ICs from the **s018c phi-sweep generator** (anchored on bright peaks)
- ω from a **coarse (ω-dir × ω-mag) grid**
- Joint 6-DOF LM polish on surrogate full-LC MSE (s011/s015 idiom)
- Cohort selector = lowest LM-polished surrogate full-LC MSE per seed
- Top-3 → hi-fi ρ-band rerank

**Decisive question:** does s018c hit ≥4/5 Band A∪B where s017's
joint-Sobol-q0 architecture hit 3/7?

Pilot seeds (revised post-tier-audit):

| seed | role | tier coverage | s011/s014/s017 verdict | predicted Band |
|------|------|---------------|-------------------------|----------------|
| 6  | s011-recoverable, s017-D — joint-search architectural failure | 2 tiers (T2:2, T3:2) | s017 D | A (control)        |
| 28 | sub-Sobol-narrow basin (s006 ~2°)                              | 3 tiers (T1:2, T2:1, T3:4) | s017 D | A or B (uncertain) |
| 79 | s017 boundary multi-solution                                   | 1 tier (T2:3) | s017 B | B+ |
| 44 | sub-Sobol-narrow + s010 competing-basin; best case             | 4 tiers, 15 peaks | s011 in-basin; s010 found competing | A (best case) |
| 84 | s014 dish-heavy multi-solution (9 candidates)                  | 2 tiers (T3:1, T4:3) | s011 + s014 multi-solution | A∪B (uncertain) |

Originally-planned 41 and 91 dropped (0 classifiable peaks; in s018b's
zero-classifiable 19/100 cohort).

## How

### Phi-sweep IC math

For each peak epoch `t_peak` with phase-angle bisector `pab(t_peak)` in
J2000, and for each candidate face `g` from the s018b tier shortlist
(face index in IS901 10-group set), the body-frame normal `n_g` must
satisfy the alignment constraint `R_b2i(t_peak) @ n_g = pab(t_peak)`.

Decompose:
```
R_b2i(t_peak) = R_phi(pab) @ R_align(n_g, pab)
```
where `R_align(n_g, pab)` is the shortest-path Rodrigues rotation taking
`n_g` to `pab`, and `R_phi(pab)` is rotation by `phi ∈ [0, 2π)` about the
PAB axis (the 1-DOF residual freedom).

For a candidate ω, propagate `(identity, ω)` to `t_peak` to obtain
`Φ(t_peak; ω)`. Then back-derive:
```
q_target_at_peak = rot_to_quat_wxyz(R_phi(pab) @ R_align(n_g, pab))
q0_ic            = Φ⁻¹ ⊗ q_target_at_peak
```
The forward-propagated `(q0_ic, ω)` reproduces the alignment to machine
precision (verified in `scratch/s018c_smoke.py`).

### Back-prop convention

`scratch/s018c_smoke.py` compared `Φ⁻¹ ⊗ q_target` (LEFT) vs
`q_target ⊗ Φ⁻¹` (RIGHT) on a known peak. LEFT recovers truth-q0 to
~2×10⁻⁶ deg; RIGHT is 124° off. Convention: **LEFT**.

This matches the propagator's documented kinematics `dq/dt = 0.5 ω ⊗ q`
(left-multiply Hamilton, body-frame ω) — solution `q(t) = Φ(t) ⊗ q0`,
inverse `q0 = Φ⁻¹ ⊗ q(t)`.

### Tier shortlists (from s018b summary.json)

| tier  | mag_abs band | candidate face indices            | shortlist | purity |
|-------|--------------|----------------------------------|-----------|--------|
| T1_X  | < 6          | 0 (+X), 1 (-X)                   | 2         | 99.4%  |
| T2_YZ | 6 – 7        | 2 (+Y), 3 (-Y), 4 (+Z), 5 (-Z)   | 4         | 97.7%  |
| T3_any| 7 – 8        | 2..9 (any non-±X)                | 8         | 100%   |
| T4_D  | 8 – 9        | 6 (+WD), 7 (-WD), 8 (+ED), 9 (-ED)| 4        | 97.4%  |

(Distance-normalised at D_REF_KM=38649.2; correction <0.004 mag within m048.)

### IS-901 body-frame face normals

STL-derived geometric invariants; copied from `notebooks/inversion/lib/
attitude_anim.py:32` (admissible under workspace contract — STL-derived
geometry, not buggy-era inversion code):

```
+X:  ( 1.0000,  0.0000, 0.0)
-X:  (-1.0000,  0.0000, 0.0)
+Y:  ( 0.0000,  1.0000, 0.0)
-Y:  ( 0.0000, -1.0000, 0.0)
+Z:  ( 0.0000,  0.0000,  1.0)
-Z:  ( 0.0000,  0.0000, -1.0)
+WD: ( 0.9659, -0.2588, 0.0)   # west dish
-WD: (-0.9659,  0.2588, 0.0)
+ED: ( 0.9659,  0.2588, 0.0)   # east dish
-ED: (-0.9659, -0.2588, 0.0)
```

### Pipeline

Phase 1: per-seed IC generation (single process; no surrogate calls).
Phase 2: surrogate full-LC MSE scoring of all (q0, ω) ICs — Pool(8),
BLAS=1, torch_threads=1.
Phase 3: LM polish of top-K=128 by surrogate-MSE — Pool(8), max_nfev=60,
6-DOF parametrisation matching s015 (`δθ` tangent space relative to
seed q0).
Phase 4: cohort selector = lowest LM-polished surrogate full-LC MSE per
seed; top-3 → hi-fi ρ-band rerank (separate downstream script).

### Architecture parameters

```
N_PHI       = 6      phi values per (peak, face)
N_OMEGA_DIR = TBD    Fibonacci sphere directions (set after omega-grab diag)
N_OMEGA_MAG = TBD    log-/lin-spaced mag bins over [0.1, 1.5] dps
TOP_K_LM    = 128    rank by surrogate full-LC MSE → polish top-K
MAX_NFEV    = 60
```

The ω-grid density is set by an upstream diagnostic (`scratch/
s018c_omega_grab.py`) that perturbs ω in dir / mag and measures where
LM joint-grab loses the basin.

## Result

### Smoke (truth-ω diagnostic, sound architecture at truth)

`scratch/s018c_smoke_truth_omega.py` — single-cell ω-grid at truth-ω on
seed 6:

- **best_q0_err = 0.287°**  (Band A)
- **best_ω_dir_err = 0.014°**
- **best_ω_mag_err = -0.002%**
- **best_final_surr_MSE = 2.79e-3 mag²** (matches s001 cached truth_mse_ref
  2.92e-3 within 4.5% — classic s005-style "LM finds slightly lower local
  min" of the surrogate noise floor; ρ ≈ 1.05 → Band A)

288 ICs from the 4 bright peaks of seed 6 (1 ω-cell × {2 T2 × 4 faces +
2 T3 × 8 faces} × 12 phi); top-64 LM polished. The phi-sweep IC + LM
machinery is correct at truth-ω.

### Coarse-grid smoke (12 ω cells)

`scratch/s018c_smoke_pipeline.py` at N_ω_dir=6, N_ω_mag=2 (truth ω-mag
0.713 dps; nearest grid mag 0.1 or 1.5 — 7× or 2× off):

- **best_q0_err = 140°, best_surr_MSE = 2.39 (Band D)**

No grid cell is within LM grab of truth-ω; phi-sweep ICs anchor on
falsely-propagated dynamics; LM polishes to off-truth local minima.

### ω-grab perturbation (decisive, 15-cell sweep)

| d_dir (°) | d_mag (%) | best_q0_err (°) | min_q0_err (°) | best_surr_MSE | Band |
|-----------|-----------|-----------------|-----------------|----------------|------|
| **0**     | **0**     | **0.287**       | **0.287**       | **2.79e-3**    | **A** |
| **1**     | **0**     | **0.287**       | **0.287**       | **2.79e-3**    | **A** |
| 5         | 0         | 161.114         | 12.633          | 1.85e+0        | D    |
| 10        | 0         | 172.024         | 60.856          | 2.58e+0        | D    |
| 20        | 0         | 105.281         | 67.697          | 2.10e+0        | D    |
| 30        | 0         | 130.361         | 86.255          | 2.82e+0        | D    |
| 60        | 0         | 121.432         | 26.823          | 2.11e+0        | D    |
| 0         | -50       | 149.019         | 97.754          | 2.00e+0        | D    |
| 0         | -20       | 145.797         | 53.382          | 2.48e+0        | D    |
| **0**     | **-10**   | **0.287**       | **0.287**       | **2.79e-3**    | **A** |
| **0**     | **-5**    | **0.287**       | **0.287**       | **2.79e-3**    | **A** |
| **0**     | **+5**    | **0.287**       | **0.287**       | **2.79e-3**    | **A** |
| 0         | +10       | 179.038         | 37.380          | 3.33e+0        | D    |
| 0         | +20       | 23.444          | 23.444          | 2.61e+0        | D    |
| 0         | +50       | 141.499         | 50.308          | 3.42e+0        | D    |

**LM grab in ω (Band-A boundary):**

- **ω-direction: |Δdir| ≤ 1°** (5° already collapses; 1° still in basin).
- **ω-magnitude: Δmag ∈ [-10%, +5%]** — asymmetric. -10% in basin; -20%
  collapsed. +5% in basin; +10% collapsed. The asymmetry is real (not
  noise) — the +20% case has all 32 LMs converge to one off-truth basin
  (q0_err = 23.4°, mse = 2.61), suggesting a stable but wrong attractor
  on the +Δmag side; -10%/-50%/-20% cases all converge to noisy
  multi-attractor patterns.

### Decisive architectural finding

The phi-sweep q0-IC primitive is **geometrically anchored on observed
peak alignment** but only valid when the *propagated dynamics* (which
depend on ω) reproduces that alignment at the assumed peak epoch. ω
outside the tube produces a propagated trajectory where the IC's
"face-aligned-with-PAB at t_peak" condition is broken: face-PAB
alignment shifts by tens of degrees away from the peak epoch, so the IC
is no longer geometrically informative.

**Implication:** s018c does not solve the ω-search problem. The
ω-fragility tube identified in s003 (~1° dir / ~5-10% mag) carries
through to the s018c architecture. To put one ω-cell inside the tube
across the cohort:

- ω-dir: ~1° spacing on S² → ~13000 cells (4π / π × (1°)²).
- ω-mag: ~5-7% spacing over [0.1, 1.5] dps → ~14 mag bins.
- Total joint cells: ~180000 per seed.

At ~50 ms surrogate-eval per IC + 144 phi-sweep ICs per cell + 5 seeds,
this is multi-day per-seed compute. Infeasible.

### Cost-benefit gate: pilot CANCELLED

Per the workspace methodology rule "cost-benefit gate before batches",
the 5-seed × 3-8 hr pilot at any feasible ω-grid (12-48 dir × 3-6 mag)
is now known to predict 0/5 Band A∪B by architecture (the truth-ω cell
is not in the grid for any pilot seed at ω-mag ranging 0.106-1.476 dps
× ω-dir uniformly distributed). Running it would consume compute to
re-confirm a result already settled at the diagnostic.

The 1-seed budget consumed for the diagnostic + smokes (~50 min total,
mostly the 36 min ω-grab sweep on seed 6) is the entire planned cost
of the s018c experiment line.

## Why this matters

1. **Settles the principal s018c question.** Phi-sweep alone does not
   bridge the ω-fragility tube. The s018b face-identity primitive is
   a sound q0-IC generator, but it inherits the ω-search bottleneck
   from prior architectures — the bottleneck moved one layer, it did
   not vanish.
2. **Validates the phi-sweep primitive as a sub-component for S016-A.**
   When ω is inside the tube (e.g., from a denser S016-A cell), the
   phi-sweep IC delivers q0_err = 0.287° on seed 6 — orders of
   magnitude tighter than the s011 Sobol-Shoemake N=64 baseline (which
   recovered seed 6 at q0_err < 5° on 8/64 ICs but with ω given). For
   seed-28-class ~2° basins (s006), the phi-sweep IC primitive is
   uniquely valuable: it's the only known IC source that lands inside
   the basin without needing q0-Sobol density to do it.
3. **Saves ~3-8 hr of compute** that the proposed full pilot would have
   spent re-confirming an architecturally-settled result.
4. **Recasts S016-A as the load-bearing follow-up.** The pre-s017
   PROGRESS document had S016-A queued as primary fallback; that's now
   the active question.

## Numbers

| metric | value |
|--------|-------|
| s018b cohort coverage ≥1 classifiable peak | 81/100 |
| s018b zero-classifiable seeds | 19/100 |
| s018b cohort with ≥2 distinct tiers | 73/100 |
| Smoke truth-ω q0_err (seed 6) | 0.287° |
| Smoke truth-ω final_surr_MSE (seed 6) | 2.79e-3 mag² |
| s001 cached truth_mse_ref (seed 6) | 2.92e-3 mag² |
| Coarse-grid smoke (12 ω cells) q0_err | 140° |
| ω-grab in dir (Band-A boundary) | ≤ 1° |
| ω-grab in mag negative (Band-A boundary) | ≥ 10% |
| ω-grab in mag positive (Band-A boundary) | < 10% (between +5% and +10%) |
| Cohort ω-mag range (truth) | 0.106-1.476 dps |
| Naive grid for full cohort ω-tube coverage | ~180k cells/seed (infeasible) |
| Diagnostic wall (15 cells × 32 LMs each, Pool(8)) | 36 min |

## Artefacts

- `experiments/s018c_phi_sweep_pilot.{py,md}` — pilot driver + writeup
  (driver kept for future reuse inside an S016-A cell loop).
- `experiments/s018c_hifi_rerank.py` — hi-fi ρ-band rerank (kept; not
  exercised since the pilot was cancelled).
- `scratch/s018c_smoke.py` — geometric / back-prop / phi-sweep
  round-trip; established LEFT back-prop convention.
- `scratch/s018c_smoke_pipeline.py` — coarse-grid 1-seed smoke
  (12 ω cells, Band D verdict).
- `scratch/s018c_smoke_truth_omega.py` — truth-ω injection diagnostic
  (Band A; primitive is sound).
- `scratch/s018c_omega_grab.py` — 15-cell ω-perturbation grab sweep.
- `scratch/s018c_grab_analysis.py` — table + LM-grab plot from the diagnostic JSON.
- `results/s018c_truth_omega_diag/{summary.json, omega_grid.npz, seed006/}`
- `results/s018c_coarse_grid_smoke_diag/{summary.json, omega_grid.npz, seed006/}`
  (renamed from `s018c/` — was the cancelled-pilot's leftover coarse-grid smoke).
- `results/s018c_omega_grab_diag/{summary.json, lm_grab_in_omega.png}`

## Out of scope

- Sobol fallback for the 19/100 zero-classifiable seeds (separate
  follow-up question; not addressed by s018c).
- LC-dim-envelope IC primitive (orthogonal to phi-sweep; would extend
  coverage for low-rotation seeds without bright peaks).
- Hi-fi ρ-band rerank: handled by a separate downstream script that
  consumes `results/s018c/seed{XXX}/top3_for_hifi.npz`.

## Cross-references

- `s018b_face_identity_tiers.md` — the tier table this experiment consumes.
- `s018a_bright_band_calibration.md` — the `mag_abs < 8` threshold.
- `s011_q4cii_sobol_so3_polish_pilot.md` — the Pool / LM idiom this builds on.
- `s015_joint_q0_omega_pilot.md` — the 6-DOF LM parametrisation.
- `s017_hifi_rho_band_s016c_prime.md` — the failure mode this aims to fix.
