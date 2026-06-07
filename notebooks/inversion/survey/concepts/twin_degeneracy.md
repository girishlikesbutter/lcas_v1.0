---
title: "Twin degeneracy — IS-901 ±X-axis symmetry produces LC-equivalent twin"
type: concept
created: 2026-04-30
updated: 2026-05-04
confidence: high
---

# Twin degeneracy (IS-901, X-axis flip)

## The structural claim — CORRECTED 2026-05-04 (s021 / filter-costs validation)

IS-901 has 2-fold geometric symmetry about the **body X-axis** (NOT Y as previously documented). Under the body-X 180° rotation R_180x = diag(1, -1, -1):

- ±X faces stay (97.3 m² panel front/back).
- SP_North at (0, 0, +9.2) ↔ SP_South at (0, 0, -9.2) (Z flips, identical material/articulation).
- AD_East at (-1.5, -4, 0) ↔ AD_West at (-1.5, +4, 0) (Y flips, identical material/articulation).
- Bus offset along -X (-1.5) is invariant under X-rotation.

**Y or Z rotations do NOT preserve component positions** (e.g. Y-flip of SP at (0, 0, +9.2) gives (0, 0, -9.2) — but the bus offset along -X also flips to +X, breaking the geometry).

For a candidate state `(q0, ω)`, the LC-equivalent twin is:

```
q0_twin     = q_180x · q0                       # LEFT-multiply
omega_twin  = R_180x · omega                    # ALSO transform ω
```

**Both transformations are required.** Just `q_180x · q0` with same ω does NOT produce a twin (median surrogate-LC diff ~1.4 mag); the ω components must be expressed in the new (relabeled) body frame.

Empirical validation (s021_smoke, seed 6):
- Twin LC vs truth LC: median |Δmag| = 0.004, max |Δmag| = 0.722.
- Tier-classified geo cost: 1.0 on truth = 1.0 on twin.
- Alignment cost (peaks at right times): 1.0 on twin.
- Cohort-scale (s021, 100 seeds): twin scored 1.0 / 1.0 on every recoverable seed where truth scored 1.0 / 1.0.

## Why LEFT-multiply q AND why ω must transform

In conv-(a) where R(q) = R_{i→b}: `q_180x · q` corresponds to applying the 180x rotation in the **body frame** (post-multiplied). The new R becomes R_180x · R_truth, meaning the "body frame" has been relabeled. Body-frame quantities (face normals — fixed in body; ω — body-frame-defined) must be re-expressed under the relabeling.

Face normals are static, so the test "does any face line up with PAB?" is invariant under tier-symmetric face renaming (T1 = {±X} stays in itself; T2 = {±Y, ±Z} — the Y-flip and Z-flip pairs swap inside the same set; etc.). Hence geo cost on truth = geo cost on twin = 1.0.

ω is body-frame components. Relabel ω: ω_new = R_180x · ω_old. Without this, the "twin trajectory" has the wrong angular velocity in the new body frame and is not actually equivalent to truth.

## What was wrong in the old version of this page

The page previously claimed Y-axis with same ω. That claim was untested. When tested at session 2026-05-04 (s021_smoke, all 6 combinations of axis × side × ω-transform):

```
  X q_180 · q w_R: median=0.004 max=0.722  ← THE TRUE TWIN
  Y q_180 · q w_R: median=0.025 max=4.388
  Z q_180 · q w_R: median=0.025 max=4.455
  Y q · q_180 w_same: median=0.650 max=6.051  ← old page claim
  X q_180 · q w_same: median=1.352 max=9.120
  ...
```

X-flip with ω-transformed is essentially perfect (~surrogate noise floor). Y/Z-flip with ω-transformed are also LC-similar but not as tight.

## Practical implications

- A "FAIL" candidate at `q0_err ≈ 178°-179°` is often the X-twin attractor.
  Twin distance metric: `min(quat_geodesic(q_cand, q_truth), quat_geodesic(q_cand, q_180x · q_truth))`.
- The twin attractor is observationally indistinguishable from truth at the surrogate-noise floor — both are valid Band A solutions.
- Filter costs (alignment, geo) preserve twins by construction (necessary condition holds for both). This is good — filters do NOT spuriously reject the twin attractor.
- A cost surface that places truth at rank 2 with twin at rank 1 is honestly reporting both attractors; not a failure.

## Practical implications

- A "FAIL" candidate at `q0_err ≈ 178°-179°` is often the twin attractor, not random noise. Distance-to-twin metric (`q0_err vs q0_twin_truth`) clarifies.
- Per `concepts/observational_indistinguishability.md`, ρ < 2 at the twin is a valid solution observationally — the twin state is a real LC attractor, just not THE truth. Whether the inversion should "claim" it or only the truth basin is a downstream question.
- Twin-near-truth states should NOT be used as positive-control "solved" cases without acknowledging the degeneracy.

## What the survey should check post-fix

- For each seed, locate the twin state `(q_180y · q0_truth, ω_truth)`. Score surrogate full-LC MSE at the twin. If the twin scores at the surrogate's intrinsic noise floor (similar to truth), the twin is a valid attractor on that seed. If it scores well above, the satellite's BRDF asymmetry breaks the symmetry on that seed.
- Population: what fraction of seeds have a strong twin attractor under correct truth?
- If a cost surface places truth at rank 2 with twin at rank 1, that's NOT a failure — it's the surface honestly reporting both attractors. Don't penalise this in the survey's cost-at-truth metric.

## Cross-references

- Auto-memory: `feedback_twin_degeneracy.md`
- Frozen-reference concept page: parent wiki `concepts/twin-degeneracy.md`
- Survey concept: `observational_indistinguishability.md`
