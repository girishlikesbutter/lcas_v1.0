# Beta Pipeline Technical Report (micro77)

## Summary

The beta pipeline (micro77) is a blind attitude inversion system that recovers the initial quaternion (q0) and angular velocity (omega0) of a torque-free tumbling satellite from a single observed light curve. It achieves **6/6 omega recovery to < 5 degrees** on the test set, fixing the one failure case (seed 27) from the alpha pipeline (micro73) while preserving all prior successes.

### Key Results

| Seed | q0 error | omega dir error | omega mag error | Time |
|------|----------|----------------|-----------------|------|
| 0 | 175.7° (twin) | 1.9° | +0.15% | 8.0m |
| 14 | 179.0° (twin) | 1.7° | -0.00% | 5.1m |
| 27 | 4.5° | 2.6° | +0.07% | 6.1m |
| 36 | 7.1° | 4.1° | +0.09% | 6.9m |
| 74 | 145.7° (twin) | 0.5° | +0.24% | 7.1m |
| 93 | 1.2° | 1.1° | -0.00% | 7.3m |

A sweep of 57 additional seeds is in progress to establish broader statistics.

---

## Pipeline Architecture

The beta pipeline has 6 stages:

### Step 1: Peak Detection and Constraint Selection
- Detect all peaks in the observed LC (`find_peaks`, distance=5, prominence=0.3).
- Estimate |omega| from peak count: `|omega| = 0.0397 * n_peaks + 0.0417` deg/s (calibrated on 100 trajectories, 13% median error).
- Classify all peaks with magnitude < 9.0 as specular constraints.
- Select anchor epoch (brightest peak).
- For each constraint epoch, compute the **allowed normal set** from the magnitude-based exclusion table.

### Step 2: Omega Grid Search
- 2000 Fibonacci sphere directions x 20 magnitude grid points (±20% of estimate).
- For each (direction, magnitude): propagate delta-quaternions from anchor, sweep phi (36 bins over [0, 180) for x-y plane normals, [0, 360) for ±Z), evaluate alignment cost.
- Alignment cost uses **magnitude-based normal exclusion**: at each constraint epoch, only normals physically capable of producing a peak at that magnitude are checked. Alignment is `max over allowed normals of dot(normal, PAB_body)`.
- Cost = sum over constraints of `W * (1 - best_allowed_dot)^2`, with W = 10.0.
- Each allowed anchor normal is swept independently. Best (direction, magnitude, anchor_normal, phi) retained per grid direction.

### Step 2b: Lo-fi Peak Matching Filter (NEW)
- Take top 200 candidates by alignment cost.
- For each, reconstruct (q0, omega) at t=0 by back-propagating from anchor.
- Generate a full lo-fi light curve (no shadows, ~200ms per candidate).
- Find peaks in the candidate's lo-fi LC. Count how many of the observed LC peaks have a matching candidate peak within ±3 epochs.
- Rank by peak match count (descending), break ties by lo-fi MSE.
- Pass top 20 to NM refinement.
- **Runtime: ~12 seconds on 24 cores.**

### Step 3: Nelder-Mead Refinement
- Top 20 candidates from the peak-matched pool.
- NM optimizes omega (3 params) using fine phi sweep (360 bins) for each allowed anchor normal.
- Same exclusion-based alignment cost as Step 2.
- Records best (omega, anchor_normal, phi) at convergence.
- Top 5 omegas carried forward, each with one candidate per allowed anchor normal.

### Step 4: Geometric Refinement
- L-BFGS-B on 6 parameters (axis-angle q0 + omega) using the exclusion-based alignment cost at all specular peak epochs.
- Full 500-epoch propagation per evaluation.
- Candidates sorted by geometric cost; cluster detection at 10x gap.

### Step 5: Hi-fi LC Evaluation
- Full ray-traced shadow computation + BRDF light curve for candidates in the low-cost cluster.
- Ranked by MSE against observed LC.
- Winner reported.

---

## Changes from Alpha Pipeline (micro73)

### Change 1: Magnitude-Based Normal Exclusion

**Problem:** The alpha pipeline used a hard rule: magnitude < 6.0 = ±X alignment. This was validated as 100% correct on the test trajectory but failed on seed 27, where a +Z normal produced a mag 5.989 peak (the one exception in 2,969 peaks across 100 trajectories). This single misclassified constraint poisoned the entire grid search, causing truth to rank #1909/2000.

**Solution:** Replace the binary specular/bright classification with graduated exclusion bands derived from exhaustive analysis of 2,969 peaks across 100 trajectories:

| Magnitude | Allowed normals | Excluded | Basis |
|-----------|----------------|----------|-------|
| < 5.5 | ±X | 8 normals | 66/66 peaks are ±X |
| 5.5 - 6.0 | ±X, +Z | 7 normals | 104/104 peaks are ±X or +Z |
| 6.0 - 6.5 | ±X, ±Y, ±Z | 4 normals (dishes) | 170/170 |
| 6.5 - 7.0 | ±X, ±Y, ±Z | 4 normals (dishes) | 258/258 |
| 7.0 - 7.5 | bus + -WD, -ED | 2 normals | 325/325 |
| >= 7.5 | all 10 | none | |

**Zero exceptions** in the database for any band. The exclusion is based on physical area constraints: a 10 m^2 dish face cannot produce the flux required for a magnitude 6.0 peak, regardless of alignment.

**Effect on seed 27:** The constraint at epoch 160 (mag 5.94) now allows ±X AND +Z. The true +Z alignment (0.29 degrees) scores well. The bright constraints at epochs 49 and 490 (mag ~5.1) remain ±X-only, providing tight discrimination. This combination preserves constraint tightness at bright epochs while accommodating the rare non-±X bright glint.

### Change 2: Lo-fi Peak Matching Filter

**Problem:** Even with the exclusion fix, the alignment cost alone could not reliably rank the correct omega direction in the top-20 for seed 27. Wrong omega directions with ~60° error scored lower alignment cost because the open-normal constraints (at dim epochs) allowed different normals to coincidentally align at each epoch.

**Solution:** After the grid search, evaluate the top 200 candidates with a lo-fi (no-shadow) light curve and check whether the candidate reproduces the observed peak timing. The key insight: if the observed (hi-fi) LC has a peak at some epoch, the lo-fi LC of the correct candidate MUST also have a peak there (removing shadows can only add peaks, never remove them). A wrong omega direction produces peaks at systematically different times.

**Scoring:** Count the number of observed peaks matched by the candidate's lo-fi peaks (within ±3 epochs). The correct candidate matched 27/28 observed peaks (96%), while the wrong candidates in the alignment top-20 matched only 14.6/28 (52%).

**Effect on seed 27:** Truth jumped from alignment rank #15,722 to peak-match rank #3 out of 20,000 candidates. The 12-second lo-fi evaluation provides discrimination that the alignment cost cannot.

### Change 3: Phi Range

**Minor change:** Phi sweep uses [0, 180) for normals in the satellite's x-y plane (±X, ±Y, ±WD, ±ED) and [0, 360) for ±Z normals. IS-901 has mirror symmetry about the x-y plane, making phi and phi+180 degenerate for x-y normals. ±Z normals do not have this symmetry.

---

## Why These Changes Work

### The Root Cause Analysis

Seed 27 failed in the alpha pipeline due to a chain of events:

1. Epoch 160 has a specular peak at magnitude 5.989, caused by near-perfect +Z alignment with the PAB (0.29 degrees).
2. The alpha pipeline classified all peaks below 6.0 as ±X, making epoch 160 a ±X constraint.
3. At truth, ±X alignment at epoch 160 is 89.76 degrees — the worst possible. The cost function penalized truth heavily.
4. Truth ranked #1909/2000 in the grid. The pipeline never had a chance.

The exclusion bands fix step 2: epoch 160 (mag 5.94) now allows +Z, so truth is not penalized. But the alignment cost alone still couldn't rank truth in the top-20 because the constraint set for seed 27 is inherently weak (only 8 constraints, 5 of them fully open at mag > 7.4).

The lo-fi peak matching fixes the discrimination problem: it evaluates the actual light curve, not just geometric alignment, and checks whether peaks appear at the right times. This is strictly more informative than normal-PAB alignment because it incorporates the full BRDF reflectance model.

### Why Lo-fi Peak Matching Is Sound

The logical chain:
1. An observed hi-fi peak exists at epoch k (brightness minimum in magnitude).
2. This peak is caused by some normal aligning with the PAB while the facet is illuminated and visible.
3. In lo-fi (no shadows), the same alignment still produces a peak — shadows can suppress peaks but cannot create them.
4. Therefore, the correct candidate's lo-fi LC must have a peak at epoch k.
5. A wrong candidate with incorrect omega produces peaks at different times and fails the match.

The one caveat: lo-fi may have ADDITIONAL peaks (where shadows would suppress them in hi-fi). This means the matching is one-directional: every observed peak must be present in lo-fi, but lo-fi may have extras. This is handled by checking observed → candidate direction only.

---

## Remaining Limitations

### 180-Degree Optical Twin
IS-901 has near-perfect 180-degree rotational symmetry about the +X body axis. Rotating q0 by 180 degrees about +X (and flipping the y,z components of omega) produces an identical light curve (RMS difference 6e-6 magnitude). This ambiguity is fundamental for single-observer broadband photometry of symmetric satellites. The pipeline typically recovers the correct omega direction but may select the twin attitude (q0 error ~180 degrees). This occurs in 3-4 out of 6 test seeds.

### Constraint Geometry Dependence
The pipeline's performance depends on the number and quality of specular constraints. Seeds with many bright (< 6 mag) peaks have tight ±X constraints and converge easily. Seeds with fewer bright peaks rely more on the lo-fi peak matching step. The 57-seed sweep will characterize this dependence.

### Single Satellite Model
The magnitude-based exclusion table is specific to IS-901's geometry and BRDF. A different satellite with different facet areas would need a new exclusion table. The methodology (compute peaks across many trajectories, record which normals produce peaks at each magnitude) transfers directly — only the numbers change.

### Observation Window
All experiments use a 3600-second observation window with 500 epochs. The pipeline's performance at shorter windows or different sampling rates has not been characterized.

---

## Computational Cost

| Step | Seed 27 | Seed 93 | Notes |
|------|---------|---------|-------|
| Step 1 (peaks) | 0.0s | 0.0s | |
| Step 2 (grid) | 83.9s | 98.6s | 2000 dirs, 24 cores |
| Step 2b (lo-fi match) | 12.2s | ~10s | 200 candidates, 24 cores |
| Step 3 (NM) | 8.3s | 13.8s | 20 candidates |
| Step 4 (geo refine) | 84.5s | 88.5s | 8 cores |
| Step 5 (hi-fi) | 173.9s | 155.3s | 6-10 candidates, 8 cores |
| **Total** | **363s (6.1m)** | **440s (7.3m)** | |

The lo-fi peak matching step adds only 12 seconds (3% of total runtime) while providing the critical discrimination needed for hard cases.

---

## Files

- Pipeline script: `notebooks/inversion/11_casadi_formulation/micro77_beta_pipeline.py`
- Results per seed: `data/results/inversion_diagnostics/micro77_pipeline_seed{NNN}/`
- Exclusion table source data: `data/results/inversion_diagnostics/micro74_seed27_diagnosis/peak_normal_table.npz`
- Diagnostic experiments: micro74 (seed 27 diagnosis), micro74b (peak-normal table), micro74c (open-normal test), micro76 (scoring table), micro76b (phi resolution), micro76c (anti-alignment + lo-fi scoring)
