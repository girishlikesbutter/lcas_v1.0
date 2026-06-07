# m094 — BRDF-Based Cost Function

> **STATUS:** Plan ready for implementation. Written by a prior context session on 2026-04-08.
> **INSTRUCTION TO FRESH CONTEXT:** Read this document fully. Then critically evaluate the reasoning — look for flaws, unstated assumptions, or missed alternatives. Refine the plan if needed, then implement m094. Do NOT skip the critique step.

---

## 1. The Problem (What's Broken)

The inversion pipeline (m090/m093) uses geometric cost functions that extract ONE constraint per epoch from the light curve. This is provably insufficient.

### Current cost functions

**Alignment cost** (used in grid search, Steps 2-3):
```
cost = Σ_epochs w × (1 - max_over_normals(dot(n_i, PAB_body)))²
```
Only uses n·h (normal vs Phase Angle Bisector). Asks: "is SOME normal pointing at the PAB?"

**Expected-dot cost** (tested in m093 NM step):
```
cost = Σ_epochs w × min_over_normals((dot(n_i, PAB) - expected_dot(n_i, obs_mag))²)
```
Uses n·h AND observed magnitude (via a calibration table that averages over phi). An improvement, but still collapses to a 1D constraint per epoch.

### Why they fail

The Ashikhmin-Shirley BRDF equation (see `src/computation/brdf.py:92-130`) computes flux from THREE independent dot products:

```
flux = [ρ_diff(n·k1, n·k2) + ρ_spec(n·h, n·k1, n·k2, h·k1)] × A × (n·k1) × (n·k2)
```

Where:
- **n·k1** = normal · sun direction (in body frame)
- **n·k2** = normal · observer direction (in body frame)
- **n·h** = normal · halfway vector (= PAB direction)
- **h·k1** = halfway · sun ≈ cos(phase_angle/2), nearly constant for GEO

The alignment cost uses only n·h. The expected-dot cost uses n·h + a calibration-table approximation. Neither uses n·k1 or n·k2 directly. **We are throwing away 2 of 3 independent geometric constraints.**

### Concrete demonstration of the gap

Two candidate attitudes with the same n·h ≈ 0.99 (identical alignment cost):

| | Candidate A | Candidate B |
|---|---|---|
| n·k1 (sun angle) | 0.8 | 0.3 |
| n·k2 (observer angle) | 0.7 | 0.9 |
| Foreshortening (n·k1 × n·k2) | 0.56 | 0.27 |
| **Flux ratio** | **2.1×** | **1.0×** |
| **Magnitude difference** | **~0.8 mag** | (baseline) |

The alignment cost rates them identically. The BRDF predicts 0.8 magnitude difference. This is a massive discriminating signal.

Additionally, the diffuse Fresnel terms provide further discrimination:
- At n·k1 = 0.8: `(1 - (1 - 0.4)^5)` = 0.922
- At n·k1 = 0.3: `(1 - (1 - 0.15)^5)` = 0.556

Another 1.7× factor in the diffuse contribution.

### Root cause of narrow basins (<0.5°)

The alignment cost uses `max_over_normals(n_i · PAB)`. When the candidate attitude is a few degrees off truth, the correct normal (e.g., +X) drifts away from PAB, but a wrong normal (e.g., +WD, 15° away in body frame) may happen to align better. The max operation picks the wrong normal, creating a discontinuous cost landscape with <0.5° basins.

The expected-dot cost partially fixes this by weighting by brightness (bright peaks → only ±X can produce them). But it still uses the calibration table approximation (averaged over phi, at a single reference epoch) rather than the exact BRDF at each epoch's actual geometry.

The BRDF cost fixes this completely: given the actual k1 and k2 at each epoch, the predicted brightness for +WD at perfect alignment is mag 7.4 — it CANNOT produce the observed mag 5.0. The BRDF naturally selects the correct normal without any calibration table or exclusion bands.

---

## 2. The Solution (BRDF Cost Function)

### Core idea

Replace the geometric cost with a **direct BRDF brightness prediction**. At each constraint epoch, for each allowed normal n_i:

1. Compute n_i·k1_body, n_i·k2_body, n_i·h_body from the candidate attitude
2. Compute h·k1 (= cos(phase_angle/2), nearly constant)
3. Evaluate the Ashikhmin-Shirley BRDF → predicted flux
4. Convert to predicted magnitude
5. Cost = (predicted_mag - observed_mag)²
6. Take min over allowed normals

### The BRDF formula (from `src/computation/brdf.py`)

```python
# Diffuse component
alpha = 1.0 - n_dot_k1 / 2.0
beta = 1.0 - n_dot_k2 / 2.0
rho_diff = (28.0 * r_d / (23.0 * np.pi)) * (1.0 - r_s) * (1.0 - alpha**5) * (1.0 - beta**5)

# Specular component
fresnel = r_s + (1.0 - r_s) * (1.0 - h_dot_k1)**5
denominator = h_dot_k1 * max(n_dot_k1, n_dot_k2)
rho_spec = ((n_phong + 1.0) / (8.0 * np.pi)) * (n_dot_h**n_phong / denominator) * fresnel

# Total flux from one facet
flux = (rho_diff + rho_spec) * area * n_dot_k1 * n_dot_k2
```

### BRDF parameters per component (from IS-901 config)

| Component | r_d | r_s | n_phong | Area (m²) |
|-----------|-----|-----|---------|-----------|
| Bus (±X, ±Y, ±Z faces) | 0.02 | 0.5 | 300 | varies by face |
| Solar panels | 0.026 | 0.3 | 200 | large |
| Antenna dishes | 0.01 | 0.4 | 200 | small |

### Magnitude conversion

```python
magnitude = -26.74 + 5*log10(distance_m) - 2.5*log10(flux)
```

### What this replaces

| Current | Proposed | Difference |
|---------|----------|------------|
| `(1 - max(n_i·h))²` | `min((pred_mag_i - obs_mag)²)` | Uses full BRDF physics |
| 1 constraint per epoch (n·h) | 3 constraints per epoch (n·k1, n·k2, n·h) | 3× more information |
| Calibration table needed | No calibration needed | BRDF IS the model |
| <0.5° basins | Expected: wider basins | n·k1/n·k2 disambiguate normals |

### Computational cost

The BRDF formula is ~20 FLOPs per (normal, epoch) pair. The alignment cost is ~5 (one dot product + power). Roughly 4× more per evaluation, but this is negligible compared to ODE propagation which dominates runtime.

The key additional computation is: n·k1 and n·k2 require knowing k1 and k2 in body frame at each constraint epoch. We already compute PAB in body frame (h_body = R^T @ PAB_j2000). Computing k1_body = R^T @ k1_j2000 and k2_body = R^T @ k2_j2000 is the same operation — two more matrix-vector products per epoch, fully vectorizable.

---

## 3. What This Doesn't Fix

**Shadows.** The BRDF prediction assumes no self-shadowing. At bright specular peaks (the constraint epochs), this is probably acceptable — the dominant facet is face-on to both sun and observer, making self-shadowing unlikely. But at dim peaks, shadow effects are significant.

**The ±X twin degeneracy.** This is a true optical degeneracy (m091 confirmed RMS=0.000 for ±X rotation). No cost function can resolve it from single-observer broadband photometry.

**Grid resolution.** m086 showed 8000 directions is needed for ~1° omega spacing. The current pipeline uses 2000. The BRDF cost should help within the grid's resolution, but won't compensate for the grid being too coarse. Consider combining BRDF cost with N_DIRS=8000.

---

## 4. Evidence from This Session

### m093 results (expected-dot NM, 2000-dir grid)

Same tally as m090 — 4 OK, 3 PARTIAL, 3 FAIL. Expected-dot NM didn't fix failing seeds because:
- Seeds 12, 33: grid failure (correct omega never in top-200) — upstream of NM
- Seed 27: NM found correct candidate (w_dir=6.0° at rank #2) but hi-fi scoring selected wrong one (w_dir=36.9°, MSE gap only 0.03)

### Seed 27 deep analysis

The winner and rank #2 produce nearly identical light curves (superimposed plot at `m093_seed027_winner_vs_rank2.png`) despite being:
- 105.5° apart in attitude
- 31° apart in omega direction
- Related by rotation about -X axis (dot=0.998 with -X)

The integrated MSE over 500 epochs can't distinguish them. The discriminating signal is in the bright peaks, but MSE dilutes it across ~480 dim epochs where both candidates look identical.

**Key finding:** The geometric cost (L-BFGS-B on alignment at specular peaks) correctly ranked the 6° candidate #1 and the 37° candidate #2. But lo-fi re-ranking and hi-fi scoring both flipped the order. The BRDF cost would be used WHERE the geometric cost currently works (Steps 2-3) and could potentially replace the lo-fi/hi-fi scoring stages with something more targeted.

### Per-epoch geometry analysis (m093c)

Aggregate LC features (peak counts, spacing, spectral) had NO correlation above 0.3 with omega direction features (dot products with body axes). But per-epoch analysis showed strong signals:

- **dot(X_normal, k2_body) at each peak → is_X_peak: ρ=0.383, p=4.8e-66**
- **|(ω×X)·k2_body| at each peak → peak_magnitude: ρ=0.287, p=1.5e-36**
- Mean dot(X, k2) at X-peaks: **0.992** vs 0.578 for dim peaks

The geometry-LC relationship is strong per-epoch but washes out in aggregates. The BRDF cost operates per-epoch, so it captures this signal directly.

### Lo-fi vs hi-fi peak census

- 86% of seeds have more lo-fi peaks than hi-fi (shadows kill peaks)
- 14% have more hi-fi peaks (shadows create peaks by reshaping local topology)
- 43% of seeds have at least one hi-fi-only peak
- Extreme cases: seed 4 (5 extra lo-fi peaks, shadows killed), seed 28 (4 extra hi-fi peaks, shadows created)
- Plots at `m093b_extreme_peak_comparison.png`

### Feature analysis (m093c)

57 features × 100 seeds. Full Spearman correlation matrix, partial correlations controlling for |ω|, scatter plots. The cross-product features `(ω × axis) · observer` showed the strongest correlations with LC features (ρ=0.307 for n_X peaks → |(ω×Y)·PAB|). Feature table saved at `m093c_feature_analysis/feature_table.npz`.

---

## 5. Implementation Plan for m094

### Architecture

Take m093 (m090 + expected-dot NM) as the base. Replace the cost function in BOTH the grid search (Step 2) AND the NM step (Step 3) with the BRDF cost.

### New function: `vectorized_phi_cost_brdf`

```python
def vectorized_phi_cost_brdf(q_anchors_xyzw, delta_qs, 
                              k1_j2000_arr, k2_j2000_arr,
                              obs_mags_arr,
                              allowed_per_constraint, normals,
                              brdf_params, observer_dist, w):
    """
    BRDF-based cost: predicted_mag vs observed_mag.
    
    For each constraint epoch, for each allowed normal:
      1. Compute n·k1, n·k2, n·h in body frame
      2. Evaluate Ashikhmin-Shirley BRDF → predicted flux → predicted mag
      3. Cost = (pred_mag - obs_mag)²
    Take min over allowed normals.
    
    New inputs vs alignment cost:
      k1_j2000_arr: (N_constraints, 3) sun direction at each constraint
      k2_j2000_arr: (N_constraints, 3) observer direction at each constraint  
      obs_mags_arr: (N_constraints,) observed magnitudes
      brdf_params: dict mapping normal_index → (r_d, r_s, n_phong, area)
      observer_dist: float, observer distance in km
    """
```

### What needs to change in the pipeline

1. **Step 1:** Also extract k1_j2000 and k2_j2000 at constraint epochs (from the trajectory database or compute from sun/obs/sat positions). Currently only PAB is extracted.

2. **Step 2 (grid search):** Replace `vectorized_phi_cost_excl` with `vectorized_phi_cost_brdf`. The grid search evaluates directions × magnitudes × phis — the BRDF cost replaces the inner cost evaluation.

3. **Step 3 (NM):** Same replacement in `refine_one_nm`.

4. **Steps 4-5 (geo refinement, hi-fi):** Unchanged. These use the actual forward model, not the geometric cost.

5. **BRDF parameters:** Need to extract (r_d, r_s, n_phong, area) per normal group from the satellite model at pipeline startup. One-time cost.

### Key implementation details

- k1_j2000 and k2_j2000 are available from the trajectory database (`sun_pos`, `obs_pos`, `sat_pos`) or from `k1_body`/`k2_body` in the master NPZ.
- The body-frame vectors at each epoch are: k1_body = R(q)^T @ k1_j2000, k2_body = R(q)^T @ k2_j2000. The rotation R(q) is already computed for the PAB transformation.
- h_body = normalize(k1_body + k2_body). This is the PAB, which we already have.
- The observer distance is approximately constant over the observation window for GEO. Use a single value.
- h·k1 = cos(phase_angle/2) is approximately constant. Precompute once.

### Validation

Run on all 10 validation seeds: 0, 6, 12, 14, 24, 27, 33, 36, 74, 93.

**Success criteria:**
1. No regressions on OK seeds (0, 14, 93 must stay OK)
2. Failing seeds improve (especially seed 27 — does the correct candidate separate from the wrong one?)
3. Basin width: run a diagnostic on seed 93 with perturbed omega to measure how wide the BRDF cost basin is vs alignment cost

**Comparison:** For each seed, report the cost landscape width (at what omega perturbation does the cost double?) for both the old alignment cost and the new BRDF cost. This directly measures whether basins widened.

---

## 6. Questions for the Fresh Context to Consider

1. **Is the single-facet approximation valid at dim peaks (mag 7-9)?** Multiple facets may contribute comparable flux. Should the BRDF cost sum over allowed normals rather than taking the min? Or restrict to bright peaks only (mag < 7)?

2. **Should we use this for scoring too (Steps 4b/5)?** Currently lo-fi MSE and hi-fi MSE are used. A BRDF-based score at constraint epochs only (ignoring dim epochs) might discriminate better than integrated MSE. The seed 27 data supports this — the geometric cost got the ranking right but was overridden by MSE.

3. **Interaction with N_DIRS=8000:** m086 showed 8000 dirs is needed. Should m094 also increase N_DIRS? Or test BRDF cost at 2000 first, then at 8000? Testing at 2000 first isolates the effect of the cost function change.

4. **The `(n·k1) × (n·k2)` foreshortening factor:** This enters the flux formula multiplicatively but isn't in the BRDF itself. Make sure the predicted magnitude includes this factor. It's line 177 in brdf.py: `return rho * facet.area * n_dot_k1 * n_dot_k2`.

5. **Are there any numerical issues?** When n·k1 or n·k2 → 0 (grazing angle), the flux goes to zero and the magnitude goes to infinity. Need to handle this gracefully in the cost function (cap at mag 15 or skip facets with n·k1 < 0.01).

6. **Could we combine BRDF cost (for bright peaks) with alignment cost (for dim peaks)?** Bright peaks provide tight BRDF constraints (specular lobe is narrow); dim peaks provide loose geometric constraints (many normals could produce the brightness). A hybrid might be optimal.

---

## 7. Files Created/Modified in This Session

### New scripts
- `m093_expected_dot_nelder_mead.py` — m090 + expected-dot cost in NM step
- `m093b_extreme_peak_analysis.py` — lo-fi vs hi-fi peak census + extreme case plots
- `m093c_feature_analysis.py` — 57-feature table + Spearman correlation analysis

### New results
- `m093_seed{000-093}/` — 10 seeds, all with result.json (full candidate state for seeds run after save-block patch: 0, 6, 12, 14, 24, 33, 36, 74, 93 have full state; seed 27 has full state from re-run)
- `m093b_extreme_peak_comparison.png` — lo-fi vs hi-fi stacked plot
- `m093c_feature_analysis/` — feature_table.npz, correlations.npz, scatter_top12.png

### Modified tools
- `notebooks/inversion/lib/attitude_viz.py` — generalised to compare any two states with custom labels
- `notebooks/inversion/lib/lc_compare.py` — new tool, supports --candidate flag for non-winner candidates
- `.claude/skills/attitude-viz/SKILL.md` — updated for generalised interface
- `.claude/skills/lc-compare/SKILL.md` — new skill
- `.claude/skills/resume/SKILL.md` — added Step 7b script checklist
- `requirements.txt` — added astropy

### Comparison plots
- `m093_seed027_lc_compare.png` — truth vs winner LC (w_dir=36.9°)
- `m093_seed027_lc_compare_cand1.png` — truth vs rank#2 LC (w_dir=6.0°)
- `m093_seed027_winner_vs_rank2.png` — winner vs rank#2 superimposed (nearly identical despite 105° attitude / 31° omega separation)
- `m093_seed027_attitude_viz.png` — winner vs rank#2 attitude/omega 3D comparison

---

## 8. Memory Updates Needed

The following memory files should be updated after implementing m094:
- `project_micro93_plan.md` → superseded by this document
- `project_beta_pipeline_status.md` → update to reflect BRDF cost as current direction
- New memory: `project_brdf_cost_rationale.md` — captures WHY we moved from alignment/expected-dot to full BRDF

The resume skill's EXPERIMENTS.md Section 1 should be updated to point to m094 as the active thread with this document as reference.
