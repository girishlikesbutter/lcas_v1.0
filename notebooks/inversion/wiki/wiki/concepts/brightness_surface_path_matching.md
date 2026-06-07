---
title: "Brightness Surface Path Matching"
type: concept
sources:
  - "notebooks/inversion/lib/brightness_surface.py"
  - "data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz"
related:
  - "[[m104_crossing_diagnostic]]"
  - "[[crossing-geometry-scoring]]"
  - "[[alignment-cost]]"
  - "[[grid-search]]"
  - "[[glint-physics]]"
  - "[[omega-magnitude-estimation]]"
created: 2026-04-12
updated: 2026-04-16
confidence: low
---

# Brightness Surface Path Matching — Analysis & Feasibility

> **Date:** 2026-04-12
> **Status:** Core claim refuted by [[m104_crossing_diagnostic]]; kinematic error corrected
> **Context:** Emerged from building the interactive brightness surface tool (`notebooks/inversion/lib/brightness_surface.py`)

---

## 1. The Core Idea

The satellite's brightness can be represented as a 3D radial surface on the unit sphere in the body frame — the **brightness surface** S(d). Each direction d on the sphere encodes the lo-fi flux when the Phase Angle Bisector (PAB) points in that body-frame direction (at zero phase angle).

During an observation pass:
- The **inertial-frame PAB path** p_I(t) is known exactly from sun/observer ephemerides
- The **body-frame PAB path** p_B(t) = R(t; q₀, ω₀) · p_I(t) depends on the unknown attitude
- The **light curve** is the brightness sampled along the body-frame path: L(t) ≈ S(p_B(t))

Peaks occur when the body-frame PAB path crosses a bright lobe. The peak's shape (height, width, asymmetry) encodes the **crossing geometry** — which lobe was crossed, at what angle, and at what speed.

This means we can potentially **derive candidate ω directions from peak shapes** instead of brute-force grid searching.

---

## 2. IS-901 Brightness Surface Topology (SP=0°, AD=15°)

### Lobe Properties

| Lobe | Total Area | n_phong | HWHM | Peak flux | Relative brightness |
|------|-----------|---------|------|-----------|-------------------|
| ±X | 116.9 m² | 250–300 | ~4–5° | 374 | 1.0 (dominant) |
| ±Z | 22.8 m² | 300 | ~4° | 73 | 0.19 |
| ±Y | 19.2 m² | 300 | ~4° | 61 | 0.16 |
| WD/ED | ~10 m² | 200 | ~5° | ~32 | 0.09 |

At SP=0° the surface is **strongly biaxial** — the ±X lobes are 5× brighter than everything else because the solar panel normals (87.5 m² combined) align with ±X.

### Dual-Scale Structure

- **Specular spikes**: HWHM ≈ 4–5°, extremely bright, barely resolved temporally
- **Diffuse hills**: HWHM ≈ 37°, ~12–17 samples per hill, well-resolved but low contrast

The areas and BRDF parameters per component at SP=0°:

```
Bus:     ±X: 9.8 m²,  ±Y: 15.7 m², ±Z: 19.6 m²  | r_d=0.020, r_s=0.500, n=300
SP (×2): ±X: 43.8 m², ±Y: 0.6 m²,  ±Z: 0.2 m²   | r_d=0.026, r_s=0.300, n=250
AD (×2): ±X: 9.8 m²,  ±Y: 1.1 m²,  ±Z: 1.4 m²   | r_d=0.010, r_s=0.400, n=200
```

---

## 3. Peak Crossing Geometry — Quantitative Analysis

### Timescales

| Quantity | Value | Source |
|----------|-------|--------|
| Sampling interval | 7.2 s | m046 dataset |
| Body rotation rate | ~1.28 °/s | Test case \|ω\| |
| Angular sweep per sample | **9.2°** | 1.28 × 7.2 |
| Inertial PAB drift | 0.002 °/s | GEO orbital mechanics |
| Body/PAB ratio | **609×** | Tumbling completely dominates |
| Specular lobe FWHM crossing time | 6–15 s | n_phong dependent |
| Diffuse hill FWHM crossing time | 86–121 s | cos² profile |

### Actual Peak Resolution (Seed 93, 29 peaks)

Despite the narrow specular spike, **peaks have 3–5 data points** significantly brighter than background because the diffuse component extends the measurable envelope:

```
Peak 27 (+X, mag 5.03): -7s:9.3, 0s:5.0, +7s:7.6, +14s:11.6  → 4 points
Peak 91 (+X, mag 5.10): -14s:8.4, -7s:6.9, 0s:5.1, +7s:8.5   → 4 points
Peak  7 (-X, mag 6.88): -14s:8.3, -7s:7.8, 0s:6.9, +7s:8.0   → 5 points (broad: 6.8° approach)
```

22/29 peaks have ≥2 samples above background−3 mag.

### Body-Frame PAB Path at Peaks

At peak 27 (closest approach 1.5° to +X):
```
idx 25: PAB = [0.943, -0.218, 0.251]  (19.4° from +X)
idx 26: PAB = [0.984, -0.107, 0.143]  (10.3° from +X)
idx 27: PAB = [1.000,  0.000, 0.026]  ( 1.5° from +X)  ← PEAK
idx 28: PAB = [0.990,  0.101, -0.095] ( 8.0° from +X)
idx 29: PAB = [0.957,  0.191, -0.219] (16.9° from +X)
```

The path sweeps through the lobe on a near-great-circle arc at ~8–9° per sample step.

---

## 4. What Crossing Geometry Encodes

Near a peak at lobe n_j, the body-frame PAB velocity is:

```
ṗ_B ≈ ω_body × p_B ≈ ω_body × n_j
```

For +X peaks (n_j = [1,0,0]): **ṗ_B ≈ [0, ω_z, −ω_y]**

This directly gives **two components of ω_body at the peak time**.

### Verified on Seed 93

| Peak | Time | Inferred ω_y | Inferred ω_z | \|ω_perp_X\| |
|------|------|-------------|-------------|--------------|
| 27 | 195 s | 0.94 °/s | 0.83 °/s | 1.26 °/s |
| 91 | 657 s | 0.45 °/s | 1.15 °/s | 1.23 °/s |
| True ω₀ | 0 s | −0.30 °/s | 2.00 °/s | 2.02 °/s |

Key observations:
- The **crossing direction rotated ~27°** between the two +X peaks (462 s apart) — this is the polhode precession
- **\|ω_perp_X\| is approximately conserved** (1.26 vs 1.23 °/s) — kinetic energy invariant
- The individual components differ from ω₀ due to polhode precession

### Per-Peak Constraints

| Constraint | Source | Precision estimate |
|-----------|--------|-------------------|
| 2 components of ω_body ⊥ n_j | Crossing direction (central difference) | ~15–30° direction, ~20% speed |
| \|ω_perp\| | Crossing speed | ~10–20% |
| Third component ω ∥ n_j | Need \|ω\| from peak count + subtraction | Poor (~30%+ error) |
| Polhode precession rate | Direction change between same-lobe peaks | Measurable if ≥2 peaks at same lobe |

---

## 5. Dimensionality Comparison

### Current Pipeline (Grid Search)

- 2000 Fibonacci directions × 20 magnitudes = **40,000 candidates**
- Each: propagate attitude, compute alignment at ~20 peaks
- Grid spacing: ~3° → barely covers the ~2° NM basin
- Total grid time: ~135 s

### Path-Derived Approach

From N specular peaks at K distinct lobe normals, extract 2K constraints on ω_body:
- 2 peaks at +X → (ω_y, ω_z) at two times → polhode arc
- 1 peak at ±Z → (ω_x, ω_y) at one time
- Altogether: overdetermined system for ω₀ (3 unknowns)

**Estimated search region: ~5–15° cone around derived candidate** (due to discrete-sampling precision and polhode complication). This contains ~10–100 grid points — a **10–100× reduction**.

### Where This Helps vs Hurts

**Helps:** FAIL seeds where the true ω falls outside the grid's top-N due to alignment cost degeneracies (e.g., +X/+WD/+ED co-alignment at 15° separation). The path-derived direction avoids the alignment cost entirely.

**Hurts:** Seeds with few specular peaks (87% of the population per m096 census) or very fast tumblers (>3 °/s → peaks become single-point spikes).

---

## 6. Relationship to Existing Pipeline Stages

| Pipeline stage | Path-matching interpretation | What path-matching adds |
|---------------|---------------------------|----------------------|
| Alignment cost at peaks | Binary lobe-crossing check: is p_B near n_j? | **Crossing direction** (which way the path approaches the lobe) |
| Lo-fi MSE | Full path matching on unshadowed surface | Nothing new — lo-fi MSE already does this |
| Expected-dot cost | Brightness-weighted lobe proximity | Peak height → closest approach distance (same info) |
| NM refinement | Local search near grid candidate | Could provide better NM starting points |
| Hi-fi MSE | Full path matching with shadows | Shadow effects not captured by brightness surface |
| Geometric refinement | Local alignment optimizer | Crossing geometry could seed this better |

**The KEY new information** is the crossing direction at each peak — the current alignment cost reduces this to a scalar (dot product), discarding the vector direction of approach.

---

## 7. Complications

### Polhode Precession
ω_body is NOT constant for triaxial bodies. For IS-901 (asymmetry 0.556), the polhode period is ~500 s. Between two +X peaks (~400–500 s apart), ω_body components change significantly while \|ω_perp\| is approximately conserved. Relating ω_body(t_peak) back to ω₀ requires Euler propagation — the same dynamics the current pipeline already computes.

### Phase Angle Approximation
The brightness surface is computed at zero phase angle (k₁ = k₂ = PAB). At non-zero phase (typical GEO: 5–30°), the effective lobe shifts by ~α/2 = 2.5–15°. This is **comparable to the lobe width** — a significant systematic for precise matching. A parameterized surface S(d, α) could handle this but adds complexity.

### Shadow Effects
The lo-fi brightness surface has no shadows. In hi-fi, shadows suppress or modify peaks (validated: anti-glint constraints are invalid because shadows suppress expected glints). Peak-based matching would need to handle "missing" peaks.

### Sampling Precision
With 3–5 points per peak and σ=0.05 mag noise, the central-difference velocity estimate has ~15–30° direction uncertainty. This gives a search region, not a point estimate.

### 87% Problem
Per m096 census, 87% of trajectories lack bright ±X constraints. These trajectories produce mostly dim peaks from ±Y/±Z/WD/ED lobes, where the magnitude classification is less reliable and the lobe overlap (WD/ED within 15° of X) creates ambiguity.

---

## 8. Peak Catalogue Concept

For each of the ~10 glint-producing lobes, precompute a family of peak templates:

**Parameters:** (crossing_angle θ ∈ [0°, 360°), crossing_speed v_perp ∈ [0.5, 3.0] °/s)

**Template:** predicted magnitude profile over ±30 s centered on closest approach, computed from the brightness surface S(n_j + offset(t; θ, v_perp)).

**Size:** ~20 angles × 10 speeds × 10 lobes = **2000 templates**. Template matching per peak is trivially fast.

**Output per peak:** matched (θ, v_perp) → ω_body × n_j → two components of ω_body.

---

## 9. Proposed Implementation Path

### Step 1: Diagnostic (NO pipeline change)
For each of 100 m046 seeds, compute body-frame PAB crossing geometry at each peak using the KNOWN true attitude. Report:
- Extracted (crossing_angle, crossing_speed) vs truth
- Precision statistics across the population
- How many seeds have ≥2 peaks at different lobes (needed for full ω recovery)

**This validates whether crossing geometry is extractable with useful precision.**

### Step 2: Peak Catalogue Prototype
Pre-compute ±X lobe templates at IS-901 BRDF parameters. Match against observed peaks from 10 seeds. Report recovery precision for (θ, v_perp).

### Step 3: ω-Direction from Crossings
Using peaks from 2+ different lobes, derive candidate ω₀ values. Compare the derived search region against the grid's top-N. **Key question:** does the crossing-derived region contain the truth for seeds where the grid fails?

### Step 4: Augment Grid Search (if Step 3 positive)
Replace or supplement the 2000-direction Fibonacci grid with crossing-derived candidates + neighborhoods. Rest of pipeline unchanged.

### Step 5: Full Path-Matching (deferred)
Phase-angle correction, shadow handling, polhode-aware multi-peak consistency. Only if Steps 1–3 show clear value.

---

## 10. Key Files

- Brightness surface generator: `notebooks/inversion/lib/brightness_surface.py`
- HTML template: `notebooks/inversion/lib/brightness_surface_template.html`
- Generated surface: `notebooks/inversion/lib/brightness_surface.html`
- Trajectory data (100 seeds): `data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz`
  - `pab_body[seed, epoch, 3]`: body-frame PAB directions
  - `pab_j2000[epoch, 3]`: inertial PAB directions
  - `mag_hifi[seed, epoch]`: hi-fi light curves
  - `peak_epochs`, `peak_seeds`: peak locations per seed
- Current pipeline: `notebooks/inversion/11_casadi_formulation/m103_*.py`
- Experiment map: `notebooks/inversion/EXPERIMENTS.md`

---

## 11. Bottom Line

~~The brightness surface reveals that each LC peak encodes a vector crossing direction -- two components of omega_body perpendicular to the lobe normal. Extracting this information could yield a 10-100x search space reduction for omega direction.~~

**REFUTED by [[m104_crossing_diagnostic]].** See Section 12 below.

---

## 12. Validation Results (m104)

[[m104_crossing_diagnostic]] tested the full concept on 100 seeds (2969 peaks) and **refuted the core claim**.

### Kinematic Error Correction

Section 4 claimed the body-frame PAB velocity is:

```
dp_B/dt = omega_body x p_B
```

This is **incorrect**. The correct kinematic equation is:

```
dp_B/dt = Omega_L x p_B    where Omega_L = R(t) * omega_body(t)
```

Here Omega_L is the "left" angular velocity (Rdot * R^T projected into the body frame), which depends on BOTH the attitude R(t) AND the angular velocity omega_body(t). Because the attitude is unknown (it is one of the two quantities we are trying to recover), we CANNOT extract omega_0 from crossing geometry alone.

The table in Section 4 showing "inferred omega_y, omega_z" was computing Omega_L components, not omega_body components as claimed.

### Phase A: Kinematic Extraction is Exact

With the known attitude, extracting Omega_L_perp at peaks gives: median direction error 0.13 deg, median magnitude error 0.48%. The kinematic extraction itself is NOT a limiting factor.

### Phase B: Peak Shape Measurement is Too Noisy

Estimating crossing speed from peak FWHM in the discrete 7.2s-sampled LC: 27% median error, only 36% of peaks within 20% error. The 3-5 data points per peak are insufficient for precise width measurement.

### Phase C: No Scoring Value

The FWHM x crossing_speed product has CV=50% (median 25% per seed), far too noisy to discriminate correct candidates.

### Verdict (m104 — direct extraction)

The direct extraction approach (measuring omega from peak shapes) is not viable: kinematic error (Omega_L != omega_body) plus measurement noise (FWHM 27% error).

---

## 13. Revised Approach — Constraint Satisfaction (m105)

m104 tested the WRONG formulation. The correct approach is **per-peak constraint satisfaction + dynamics linking**, not direct omega extraction.

### Reformulation

Each peak constrains a manifold of (q0, omega0):
1. At peak time t_k: R(t_k) must map p_I(t_k) to lobe n_j → constrains R(t_k) to 1-DOF (twist psi about n_j)
2. For a given omega0 candidate: delta-q factorization determines q0 from psi
3. At a SECOND peak: the propagated attitude must ALSO align → hard constraint
4. Two peaks with lobe assignments give 3 constraints for 3 remaining unknowns → exactly determined

### Why this works where direct extraction fails

- Direct extraction tries to INVERT: peak shape → omega. This fails because Omega_L = R * omega_body entangles the unknowns.
- Constraint satisfaction CHECKS CONSISTENCY: for each candidate omega0, does the dynamics produce alignment at multiple peaks simultaneously? This doesn't need to know Omega_L.

### POC Results [inline] (2026-04-12)

Tested on grid-failure seeds 28 and 44 with 2500 candidates via inline strategist bash testing (NOT from m105_pairwise_ipl.py — that script only implements 3-pair intersection, not the full-observation scoring that produced the peak-count numbers):

| Seed | 3-pair intersection | Truth: peaks aligned | Best FP: peaks aligned |
|------|---------------------|---------------------|----------------------|
| 28   | 2501 -> 18 (139x)   | 11/14               | 6/14                 |
| 44   | 2501 -> 20 (125x)   | 11/11               | 6/11                 |

Truth survives with near-perfect alignment; false positives max at 6 peaks. **Crossing speed was NOT used** — pure alignment + dynamics suffices.

See [[crossing-geometry-scoring]] for the branch status; full m105 results are in `notebooks/inversion/12_brightness_surface/m105_REPORT.md`.
