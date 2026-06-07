# Series 10 — Glint-Anchored Inversion Pipeline

> **Status:** Series 10 (m049-51b) = DONE — attitude recovery given known omega is validated.
> Series 10b (m052-59) = CLOSED — bridge-LC omega selection fails due to fundamental chicken-and-egg problem. See `DEAD_ENDS.md`.

## Two validated results

### 1. Attitude recovery from observed LC + known omega (m051b)

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

### 2. First blind omega + attitude recovery (m054)

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

## Omega magnitude estimation (m052)

Peak count from the observed LC correlates strongly with |omega| (Spearman rho=0.954).

**Calibration:** `|omega| = 0.0397 × n_peaks + 0.0417` (deg/s)

| Metric | Value |
|--------|-------|
| Median error | 13.2% |
| Within ±20% | 75/100 |
| Within ±30% | 90/100 |

Lomb-Scargle dominant frequency is weaker (rho=0.740, median error 31.1%).

---

## Bridge omega derivation (m053-53c)

At two specular glints, the attitude is constrained to PAB-alignment circles. For each pair of candidate attitudes (q1, q2), the axis-angle bridge gives omega = rotvec(q2 × q1⁻¹) / dt. This DERIVES omega rather than searching for it.

**Key finding (m053c):** For trajectory 84, the bridge pool contains omega candidates at 3.6°, 4.1°, and 5.3° direction error — ranked #2, #3, #5 by lo-fi LC residual among 5000 candidates. The correct omega IS generated and IS selectable by LC scoring.

**Coverage problem:** For 2/3 trajectories tested (70, 35), the correct omega is NOT in the top 5 by LC. The bridge pool of 5000 (from 129,600 pairs × windings) is too sparse, or the LC residual ranking doesn't place it high enough.

**Factors affecting coverage:**
- Phi grid density: 36 values = 10° spacing → up to 5° phi error at each anchor
- Winding ambiguity: faster tumblers have more windings (up to 9 for |omega|=1.2 deg/s at dt=2000s)
- Magnitude estimate error: 13% median, but individual outliers at 46% (traj 22)
- Number of scoring glints: 1-10 non-anchor glints; more glints = better LC discrimination

---

## Anchor census (m049)

Across 100 m046 trajectories [0.1, 1.5] deg/s:

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
| `m049_generalization_and_census.py` | Phi sweep generalisation test + anchor census (100 trajs) |
| `m050_omega_landscape.py` | Omega cost landscape: magnitude sweep, direction sweep, coarse grid |
| `m050b_full_epoch_omega.py` | Full-epoch scoring (500 epochs) — confirms signal, basin still 2° |
| `m050c_omega_multistart.py` | Multi-start NM: random fails, near-truth succeeds |
| `m050d_omega_de.py` | DE attempt (killed) |
| `m051_end_to_end.py` | End-to-end with lo-fi disambiguation (2/10) |
| `m051b_hifi_disambiguation.py` | **End-to-end with hi-fi: 9/10 attitude recovery (known omega)** |
| `m052_spectral_omega.py` | Peak count → |omega| (13.2% error), spectral matching (fails) |
| `m052b_phisweep_omega_scorer.py` | Phi-sweep as omega scorer (fails, median 97°) |
| `m052c_lc_omega_scorer.py` | LC residual omega scorer (partial, traj 56: 31° improvement) |
| `m053_bridge_omega_from_circles.py` | Bridge omega from two glint circles (finds close omegas sometimes) |
| `m053b_bridge_lc_scored.py` | Bridge + alignment filter + LC (alignment filter poisons pool) |
| `m053c_bridge_all_lc.py` | Bridge + all LC scored (traj 84: 3.6° at rank 2) |
| `m054_full_blind_pipeline.py` | **Full blind pipeline: traj 84 omega 5.5°, no oracle** |
| `m055_bridge_coverage_diagnostic.py` | Bridge coverage: full pool has <5° omega for 9/10, magnitude filter kills it |
| `m055b_alignment_bridge.py` | Alignment filter + body-frame fix: 9/10 coverage but LC ranking fails |
| `m056_alignment_nm_pipeline.py` | Alignment + NM on alignment cost: degenerate (many false zeros) |
| `m056b_antiglint_nm.py` | Anti-glint NM + ODE re-rank + magnitude penalty: NM collapses or finds wrong minima |
| `m057_bodyframe_lc_nm.py` | **Body-frame bridge + LC scoring: correct omega at LC#1 for traj 19 (0.9° error!)** |

---

## Body-frame omega conversion (m055-57)

### Critical bug discovered (m055)

The bridge computes omega in the **inertial frame** (rotvec of R2 × R1⁻¹ / dt), but the propagator expects **body-frame** omega. Without correction, this causes ~60-120° omega error at t=0.

**Fix:** `omega_body = R_anchor^T @ omega_inertial`, where R_anchor is the body-to-inertial rotation from the anchor quaternion. With the fix:
- Perfect bridge + perfect anchor → 0.0° error at t=0
- Correct normal + 5° phi grid → 2.0° error at t=0
- The error is dominated by phi discretization, not the frame conversion

### Magnitude filter is the bottleneck, not phi density (m055)

For 10 trajectories × 2 phi densities (36 vs 72):
- Full bridge pool has correct omega within 5° for **9-10/10 trajectories** regardless of phi density
- After top-10K magnitude filter, only **3-4/10** have correct omega within 5°
- The magnitude filter kills correct bridges when the peak-count estimate is wrong (up to 46% for traj 22)
- Denser phi (72 vs 36) does NOT help because the bottleneck is the magnitude filter

### Alignment cost is too degenerate (m056/56b)

The min-over-normals PAB alignment cost has many degenerate solutions:
- IS-901 has 10 face normals covering ~50% of the unit sphere
- Many wrong omega vectors produce zero alignment cost at glint epochs by accidentally aligning different normals
- NM refinement converges to these degenerate solutions instead of the correct omega
- Adding anti-glint penalties (m056b) helps marginally but doesn't break the fundamental degeneracy
- Adding omega magnitude constraints prevents NM collapse to omega ≈ 0 but doesn't eliminate wrong-direction convergence

### Body-frame conversion enables direct LC scoring (m057)

With body-frame conversion, the lo-fi LC residual becomes a powerful omega discriminator:
- **Traj 19 (1.476 deg/s):** correct omega at **LC rank #1** with 0.9° direction error
- The LC ranking BEFORE NM refinement is better than AFTER (NM on LC converges to false lo-fi minima)
- This suggests: use LC scoring for omega SELECTION (not NM refinement), then phi sweep for attitude recovery

### Recommended next pipeline (m058) — SUPERSEDED by m059 findings

~~Based on these findings, the correct architecture is bridge → magnitude filter → LC scoring → phi sweep → hi-fi.~~

**m059 (2026-03-19) showed this pipeline fundamentally cannot work.** LC scoring of bridge candidates ranks the correct omega at #34K/90K because the bridge's anchor attitude (q1) is always far from truth. See Series 10b findings below.

---

## Omega selection failure analysis (m059, 2026-03-19)

### The chicken-and-egg problem

The bridge produces correct omega at 0.3° direction error — the GENERATION is fine. The problem is SELECTION: no scoring method can identify the correct omega from the pool because:

1. LC scoring requires propagating from a starting attitude (q1). The bridge's q1 is 14-76° from truth.
2. Wrong q1 → wrong predicted brightness → bad LC score, regardless of omega quality.
3. Wrong omegas with coincidentally better q1 attitudes get better LC scores.
4. Finding a good q1 requires the correct omega (the phi sweep only works with exact omega, per m051b).

This was confirmed across 9 experiments: stratified winding selection, anchor-centered scoring, phi-sweep-improved q1, alignment filtering, grid search, combined specular+anti-glint scoring, focused bridging — all fail for the same root cause.

### Top-10K magnitude ranking kills correct omega

The top 10K candidates by magnitude error all come from ONE winding (the one closest to the peak-count estimate). The correct omega at a different winding (8.6% magnitude error) is excluded. Even stratified selection (2K/winding) doesn't help because LC scoring still can't find it.

### Alignment scoring contaminated by diffuse peaks

The phi sweep's PAB alignment scoring included peaks with mag 6-9, many of which are diffuse (NOT specular). At diffuse peaks, no face normal aligns with PAB, so the alignment cost is uninformative. Only mag < 6.5 gives reliable specular alignment. Traj 19 has only 1 specular non-anchor peak — insufficient for discrimination.

### Brightness filter at anchors (useful tool)

Single-epoch hi-fi brightness evaluation of all 720 PAB-circle candidates at each anchor epoch: 720 → 144-288 survivors at |Δmag| < 0.5. The correct attitude survives. This is a cheap (~8s per anchor) and effective pre-filter that should be standard in the pipeline.

### Brightness profile shape scoring (explored, insufficient)

**Approach:** For a candidate q1 at the anchor and a candidate omega direction, propagate to ±5 epochs around each brightness peak and compare predicted vs observed profile shape. The shape of a specular spike depends on which direction omega sweeps the face through the mirror geometry.

**Results (traj 19, 400 omega directions on Fibonacci grid):**

| Scoring | True omega rank |
|---------|----------------|
| Anchor peak only (hi-fi) | #142/400 |
| + close peak (ep451, dt=267s) | #87/400 |
| + medium peak (ep337, dt=1089s) | #64/400 |
| All 13 peaks | #124/400 (worse — distant peaks add noise) |

**Key insight:** Close peaks help, distant peaks hurt. At 3.8° omega error × dt seconds / 57.3 = attitude drift. Beyond ~500s from anchor, drift > 30° and predictions are meaningless. The useful scoring region is limited to nearby peaks.

### Previously recommended next direction — LOW CONFIDENCE

> **2026-03-19 reassessment:** The numbers below don't support optimism. Rank #64/400 at 10° spacing means ~64 NM starts needed. m050c showed 0/200 random NM starts converge. Even at 5° spacing (2000 dirs), expected rank ~200/2000 — still far from the top-5 needed for reliable NM convergence. **The CasADi + IPOPT constrained NLP formulation (multiple shooting + exact gradients + glint constraints) is the more promising direction.** See EXPERIMENTS.md Section 1.

~~1. **Denser omega grid** (2000 dirs, ~5° spacing) + shape scoring at nearby peaks only → aim for top 20-30 ranking~~
~~2. **Per-direction phi optimization** — find the best q1 for each omega direction before shape scoring~~
~~3. **NM refinement** from top-20 shape candidates (m050c showed 45% convergence from ~10° starts)~~
~~4. **Test on slower tumblers** — may have more useful nearby peaks and less attitude drift~~
