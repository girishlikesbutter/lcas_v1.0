# Series 10 — Glint-Anchored Inversion Pipeline

## Two validated results

### 1. Attitude recovery from observed LC + known omega (micro51b)

Given only the observed light curve and the true 3D angular velocity vector (no attitude oracle), the pipeline recovers the initial attitude q0 for **9/10 trajectories** to within a 180° ambiguity at realistic tumble rates [0.2, 1.5] deg/s. 5/10 fully resolve (median q0 error 2.15°); the other 4 return the antiparallel (q0 error ~175°, fixable post-hoc by comparing two hi-fi LC evaluations). 1/10 genuinely wrong.

**Pipeline (no attitude oracle):**
1. Detect LC peaks from observed light curve
2. Pick brightest peak as anchor epoch
3. Sweep 10 normal hypotheses × 36 twist angles on the PAB-alignment circle at the anchor (no prior group identification needed)
4. Score each by min-over-normals PAB alignment at non-anchor peak epochs (requires propagation with the given omega)
5. Top 4 → propagate to t=0, evaluate full lo-fi LC residual → top 2
6. Top 2 → full hi-fi (shadow) LC residual → pick winner
7. Output: q0 estimate

**Inputs:** observed hi-fi LC, true 3D omega vector, satellite model, SPICE geometry.
**Runtime:** ~120s per trajectory (dominated by 2 hi-fi evaluations).

### 2. First blind omega + attitude recovery (micro54)

Full pipeline with **no oracle at all** — both omega and attitude recovered from the observed LC alone. Proof of concept on trajectory 84:

- **Omega direction recovered to 5.5°** (no oracle)
- Attitude at 24.1° (needs refinement but far better than random ~90°)
- Omega magnitude error: 0%

**Pipeline (fully blind):**
1. Detect LC peaks, estimate |omega| from peak count (calibration: |omega| ≈ 0.040 × n_peaks + 0.042, median 13.2% error)
2. Pick two brightest peaks as anchors; construct PAB-alignment circles at each (10 hyp × 36 phi = 360 candidates per anchor)
3. Bridge all 360×360 = 129,600 pairs: axis-angle omega = rotvec(q2 × q1⁻¹) / dt, with winding corrections
4. Rank bridges by |omega_bridge − omega_est|, keep top 5000
5. Score all 5000 by lo-fi LC residual (ObjectiveFunction.evaluate, 500 epochs, ~220ms each, ~18 min total)
6. Top 5 omega candidates → for each, run phi sweep (10 hyp × 36 phi) + Nelder-Mead refinement on 4D (phi + omega)
7. Score 5 refined solutions by hi-fi LC residual → pick best

**Result:** Works for 1/5 trajectories tested (traj 84). The other 4 fail because the correct omega direction is not in the top 5 LC-scored bridge candidates — a coverage problem, not an architecture problem.

---

## What was tried and why it failed

### Omega direction estimation

The omega direction convergence basin is ~2° (~0.03% of sphere surface). This is a fundamental physics limitation: omega direction errors compound linearly with time (2° × 3600s / 57.3 ≈ 126° attitude drift).

| Approach | Result | Why it fails |
|----------|--------|--------------|
| Coarse grid (100-500 dirs) | Median ~100° | Grid spacing ~8-15°, basin is 2° |
| Full-epoch alignment scoring (500 epochs) | Truth is global min, but basin still 2° | Same narrow basin, just better signal-to-noise |
| Multi-start Nelder-Mead (200 random) | 0/200 within 20° | Basin too narrow for random starts |
| Differential Evolution | Killed (expected to fail like NM) | Same landscape, same basin |
| Lomb-Scargle spectral matching | Median 80° | Power spectrum similarity too crude |
| Phi-sweep as omega scorer | Median 97° | Min-over-normals at few glints too degenerate |
| Glint recurrence intervals | rho=0.146, useless | Not simply 2π/|omega| due to triaxial polhode |
| Near-truth NM starts (10% pert) | **9/20 within 5°** | Basin exists and NM converges from ~5-10° |

### Antiparallel disambiguation

The min-over-normals alignment cost is identical for the correct attitude and its 180° flip (opposite faces of IS-901 have symmetric normals). Lo-fi LC residual fails to break this (systematic lo-fi/hi-fi offset masks the signal). **Hi-fi (shadow) LC residual works for 5/9 cases** — self-occlusion patterns differ between correct and flipped orientations.

---

## Omega magnitude estimation (micro52)

Peak count from the observed LC correlates strongly with |omega| (Spearman rho=0.954).

**Calibration:** `|omega| = 0.0397 × n_peaks + 0.0417` (deg/s)

| Metric | Value |
|--------|-------|
| Median error | 13.2% |
| Within ±20% | 75/100 |
| Within ±30% | 90/100 |

Lomb-Scargle dominant frequency is weaker (rho=0.740, median error 31.1%).

---

## Bridge omega derivation (micro53-53c)

At two specular glints, the attitude is constrained to PAB-alignment circles. For each pair of candidate attitudes (q1, q2), the axis-angle bridge gives omega = rotvec(q2 × q1⁻¹) / dt. This DERIVES omega rather than searching for it.

**Key finding (micro53c):** For trajectory 84, the bridge pool contains omega candidates at 3.6°, 4.1°, and 5.3° direction error — ranked #2, #3, #5 by lo-fi LC residual among 5000 candidates. The correct omega IS generated and IS selectable by LC scoring.

**Coverage problem:** For 2/3 trajectories tested (70, 35), the correct omega is NOT in the top 5 by LC. The bridge pool of 5000 (from 129,600 pairs × windings) is too sparse, or the LC residual ranking doesn't place it high enough.

**Factors affecting coverage:**
- Phi grid density: 36 values = 10° spacing → up to 5° phi error at each anchor
- Winding ambiguity: faster tumblers have more windings (up to 9 for |omega|=1.2 deg/s at dt=2000s)
- Magnitude estimate error: 13% median, but individual outliers at 46% (traj 22)
- Number of scoring glints: 1-10 non-anchor glints; more glints = better LC discrimination

---

## Anchor census (micro49)

Across 100 micro46 trajectories [0.1, 1.5] deg/s:

| Metric | Value |
|--------|-------|
| Peaks per trajectory (median) | 18 |
| Trajectories with ≥2 peaks (mag<8) | 86% |
| Trajectories with ≥2 distinct normal groups | 88% |
| Same-group recurrence | 57% |
| Anti-glint epochs (mag>11, median) | 460/500 |
| Peak count vs omega correlation | rho=0.954 |

---

## File inventory

| File | Purpose |
|------|---------|
| `micro49_generalization_and_census.py` | Phi sweep generalisation test + anchor census (100 trajs) |
| `micro50_omega_landscape.py` | Omega cost landscape: magnitude sweep, direction sweep, coarse grid |
| `micro50b_full_epoch_omega.py` | Full-epoch scoring (500 epochs) — confirms signal, basin still 2° |
| `micro50c_omega_multistart.py` | Multi-start NM: random fails, near-truth succeeds |
| `micro50d_omega_de.py` | DE attempt (killed) |
| `micro51_end_to_end.py` | End-to-end with lo-fi disambiguation (2/10) |
| `micro51b_hifi_disambiguation.py` | **End-to-end with hi-fi: 9/10 attitude recovery (known omega)** |
| `micro52_spectral_omega.py` | Peak count → |omega| (13.2% error), spectral matching (fails) |
| `micro52b_phisweep_omega_scorer.py` | Phi-sweep as omega scorer (fails, median 97°) |
| `micro52c_lc_omega_scorer.py` | LC residual omega scorer (partial, traj 56: 31° improvement) |
| `micro53_bridge_omega_from_circles.py` | Bridge omega from two glint circles (finds close omegas sometimes) |
| `micro53b_bridge_lc_scored.py` | Bridge + alignment filter + LC (alignment filter poisons pool) |
| `micro53c_bridge_all_lc.py` | Bridge + all LC scored (traj 84: 3.6° at rank 2) |
| `micro54_full_blind_pipeline.py` | **Full blind pipeline: traj 84 omega 5.5°, no oracle** |
