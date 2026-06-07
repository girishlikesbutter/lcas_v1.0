---
title: "Surrogate Forward Model"
type: concept
sources: []
related: ["[[hi-fi-scoring]]", "[[lo-fi-mse]]", "[[de-attitude-search]]", "[[multi-solution-philosophy]]", "[[surrogate-de-search]]", "[[m114_surrogate_multistart]]", "[[m115_surrogate_pipeline]]", "[[m119_attitude_isoshell]]", "[[m124_hifi_validate]]", "[[m126_wrapped_pipeline]]", "[[dark-mag-saturation]]", "[[gradient-based-inversion]]", "[[omega-sign-degeneracy]]"]
created: 2026-04-13
updated: 2026-04-18
confidence: high (at-truth) / low (off-truth)
---

# Surrogate Forward Model

## Overview

Replaces the hi-fi forward model (ray-traced shadows + Ashikhmin-Shirley BRDF) for IS-901. Pure NumPy, no dependencies. Location: `~/surrogate_model/surrogate_model/` (v2, current) and `~/surrogate_model/` (v1, legacy).

## v2 vs v1 — the new residual ensemble is live (2026-04-18)

**v2 is a drop-in physical-accuracy upgrade**, not an API swap. It uses **residual learning + 3-model ensemble**: an analytical Ashikhmin-Shirley BRDF evaluator (vectorised NumPy over all 3840 IS-901 facets, no ray tracing) produces `phi_noshadow`, and the MLP ensemble predicts only the shadow correction `delta = log10(phi_shadow / phi_noshadow)`. Because `delta` has a 12× smaller dynamic range (std 0.116) than the full brightness target (std 1.4), the network has a far easier learning task.

| Property | v1 (s10_5M, direct MLP) | v2 (s12_residual_5M, residual ensemble) | Change |
|----------|-------------------------|------------------------------------------|--------|
| Approach | Direct `log10(phi)` regression | Analytical BRDF + MLP learns `delta` | — |
| Ensemble members | 1 | 3 (seeds 42, 123, 777), averaged | — |
| Architecture | 32D → 256×4 → 128 → 1 (same) | 32D → 256×4 → 128 → 1 per member | — |
| Training data | 5M ray-traced samples | 5M (same generator), filter `phi > 1e-8` | — |
| **Overall log10 MAE** | 0.024 | **0.0039** | **6.2× better** |
| **Bright (mag<10) MAE** | 0.044 mag (advertised) | **0.008 mag** | **5.5× better** |
| <0.05 mag accuracy | ~85% | **97.6%** | — |
| <0.10 mag accuracy | ~92% | **99.2%** | — |
| Max error (empirical, seed 0) | 0.75 mag | 0.11 mag | 6.8× tighter tail |
| Speed per 500-epoch LC | ~4 ms | ~130 ms (ensemble + NoShadowEvaluator) | v2 is ~30× slower but still ~460× vs hi-fi |
| Speedup vs hi-fi (234 ms/sample) | ~50,000× | ~460× | — |
| Files | `s10_5M_weights.npz` + `s10_5M_normalization.npz` | `s12_residual_5M_s{42,123,777}_weights.npz` + `s12_residual_5M_normalization.npz` + `s11_geometry.npz` | — |
| Entry point | `SurrogateModel(weights, norm)` | `SurrogateModel.load_default()` | API changed |

Measured on real m048 seed-0 trajectory (500 epochs, `panel=0°, dish=15°`, noiseless `mag_hifi` comparison): v1 MAE = 0.030, v2 MAE = 0.009 → **3.4× improvement overall, 4.7× in bright regime**. Max error collapses from 0.75 mag (v1) to 0.11 mag (v2) — the extreme-geometry outliers that motivated the "Limitations" section below are largely gone in v2.

## API — v2 (current)

```python
import sys; sys.path.insert(0, '/home/girish/surrogate_model/surrogate_model')
from surrogate import SurrogateModel
model = SurrogateModel.load_default()  # loads s12 ensemble + geometry automatically
mag = model.predict_magnitude(k1_body, k2_body,
                              panel_deg=0.0, dish_deg=15.0,
                              observer_distance_km=dist_km)
# also: predict_log_phi(...), predict_phi(...), predict_single(...)
```

Key change vs v1: `SurrogateModel` constructor now takes three paths (weights list + normalization + geometry), and `load_default()` is the usual entry point. The `predict_magnitude` / `predict_log_phi` signatures are unchanged, so any code that only calls those methods can be swapped by re-binding the model object.

The **single-member fast mode** (still 3× faster than the ensemble, only slightly less accurate — 0.0048 vs 0.0039 log10 MAE) is available by passing a single-element `weights_paths` list.

## Key Properties (v2)

| Property | Value |
|----------|-------|
| Architecture | 32D → 256 → 256 → 256 → 256 → 128 → 1 (ReLU) per ensemble member |
| Parameters | 238,849 per model × 3 = 716,547 total |
| Ensemble members | 3 (seeds 42, 123, 777), arithmetic mean of predicted `delta` in normalised space |
| Training data | 5M ray-traced samples (85/7.5/7.5 train/val/test) with variable articulation |
| Training loss | MSE on z-normalised `delta` |
| Training time | 34 hours total (≈11h per seed on 1 GPU) |
| MAE (bright, mag<10) | **0.008 mag** |
| MAE (all) | **0.0039 log10 = ~0.010 mag** |
| Correlation with hi-fi | 0.999+ |
| Speed (batch, 500 epochs) | ~130 ms = ~260 μs/sample |
| Speedup vs hi-fi | ~460× |
| Speedup vs lo-fi | still >>1× (lo-fi is ~100× slower than v2) |

## How v2 works (residual + ensemble)

```
input (k1, k2, panel_deg, dish_deg)
    │
    ├──→ Analytical Ashikhmin-Shirley BRDF (NoShadowEvaluator)     →  phi_noshadow
    │    (vectorised NumPy over 3840 IS-901 facets, no ray tracing)       │
    │                                                             log10(phi_noshadow)
    │                                                                      │
    └──→ 32D features → 3 × MLP (seeds 42/123/777) → avg(delta)           │
         (learned shadow correction in normalised space)                   │
                                                                         + ─┘
                                                                           │
                                                                  log10(phi_with_shadows)
```

**Why residual learning helps**: the full brightness function spans ~10 orders of magnitude (BRDF + specular glints + Fresnel + shadowing, all tangled). v1 learns this entire surface end-to-end. v2 removes the analytical part (BRDF + geometric visibility, which we can compute exactly) and asks the network to learn only the shadow attenuation delta. That signal has:

- Mean = -0.048, std = 0.116 (vs full target std = 1.4 → **12× smaller dynamic range**)
- Bounded above at 0 (shadows can only reduce flux)
- 25.8% of samples have delta ≈ 0 (no shadow at all — network just has to learn "no attenuation")

Decomposition gives **82% error reduction** over the v1 direct approach on the same training data.

**NoShadowEvaluator internals** (in `surrogate.py`):
- Satellite = IS-901 with 5 components (Bus 768 facets, SP_North+SP_South 1536, AD_East 768, AD_West 768 — total 3840 facets after subdivision level 3).
- All articulation is z-axis rotation. Instead of rotating thousands of facet normals per sample, the evaluator inverse-rotates the light/observer vectors and dots them against fixed reference normals — a single matrix multiply per sample.
- Ashikhmin-Shirley diffuse + specular with Fresnel term. BRDF params baked into `s11_geometry.npz` (Bus: `r_d=0.02, r_s=0.5, n=300`; panels: `0.026/0.3/200`; dishes: `0.01/0.4/200`).
- Chunks samples at 50 at a time for cache efficiency — tune `chunk_size` up for large batches.

## Inputs/Outputs

**Inputs** (all in body frame):
- k1: (N, 3) — sun direction unit vectors
- k2: (N, 3) — observer direction unit vectors
- panel_deg: scalar or (N,) — solar panel angle (0° for IS-901 inversion)
- dish_deg: scalar or (N,) — antenna dish angle (15° for IS-901 inversion)

**Output**: apparent magnitude (via `predict_magnitude()`)

## What It Replaces

The surrogate replaces shadow computation + BRDF evaluation. It does NOT replace:
- SPICE geometry computation
- Attitude propagation (ODE solve)
- Quaternion → body-frame vector conversion

## Key Finding: Shadows Don't Fix Omega Error

Validated in m114 inline diagnostics: the surrogate MSE landscape has the SAME false-minimum pattern as lo-fi when omega has even ~3° error. Truth q0 has WORSE MSE than false minima for ATT_FAIL seeds with both surrogate and lo-fi cost.

**Root cause**: omega error shifts the entire predicted LC, creating a systematic bias that neither shadows nor lack thereof can fix. The only way to eliminate omega error is to search omega jointly (6-DOF).

## What the Surrogate Enables

1. **Multi-start DE**: at ~15s per 3-DOF DE run (vs 8 min lo-fi), 50+ restarts become feasible
2. **6-DOF joint search**: at ~50ms per 6-DOF eval (40ms ODE + 10ms surrogate), full q0+omega search takes ~10 min
3. **Multi-solution enumeration**: cheap enough to find ALL basins, not just one
4. **Large-scale population studies**: 100-seed validation in hours, not weeks

## Gold-Standard Cheap Validator Pattern (2026-04-15)

Surrogate-DE-MSE is now the **gold-standard cheap omega-pool validator**. ±0.005 of hi-fi MSE; 50,000× faster than hi-fi. It tracks hi-fi closely enough that its ranking is reliable for downstream selection (validated m114 → m115 → inline omega-selection 2026-04-15).

**When to apply it:**
- **Immediately after any pipeline produces omega candidates with stored vectors** — surrogate-DE-per-omega + rank by MSE is the right first downstream check. Catches ATT_FAIL seeds before the legacy phi-sweep would mislabel them.
- **Re-scoring historical data** — any experiment that saved candidate omega vectors (e.g. `m103_hybrid/seed_NNN/geo_ckpt.npz` covers ~13 seeds with 26 omegas each) can be re-ranked at zero compute cost beyond ~6 min/seed of surrogate-DE.
- **Post-hoc oracle for debugging** — when a pipeline output disagrees with intuition, surrogate-MSE is the cheap second opinion before paying for hi-fi.

**Cost: ~14 s per omega for surrogate-DE; ~6 min for a 26-omega pool**. See [[surrogate-omega-selection]] for the validated mechanism.

## What the Surrogate CANNOT Replace (2026-04-15 decision)

Surrogate is **photometric** — needs a full attitude trajectory to evaluate, ~5 ms per LC. Alignment cost is **geometric** — one PAB constraint per epoch, vectorised over 360 phi values, sub-millisecond per direction.

At grid scale (40000 omega evaluations × 14 s of surrogate-DE-per-omega = 7.7 hr/seed), surrogate cannot replace alignment cost as the grid cost function. Tested dead ends: m095 lo-fi rerank, m097 lo-fi as grid cost, m114 6-DOF cold-start surrogate-DE.

The alignment-cost grid stays as the omega-finder. Surrogate's role is downstream **ranking + attitude opt**, not upstream search.

## Usage

### v2 (current — residual ensemble, recommended)

```python
import sys; sys.path.insert(0, '/home/girish/surrogate_model/surrogate_model')
from surrogate import SurrogateModel
model = SurrogateModel.load_default()
mag = model.predict_magnitude(k1_body, k2_body,
                              panel_deg=0.0, dish_deg=15.0,
                              observer_distance_km=obs_dist_km)
```

### v1 (legacy — still loadable if speed matters more than accuracy)

```python
import sys; sys.path.insert(0, '/home/girish/surrogate_model')
from surrogate import SurrogateModel    # the legacy module
model = SurrogateModel('s10_5M_weights.npz', 's10_5M_normalization.npz')
mag = model.predict_magnitude(k1_body, k2_body, panel_deg=0.0, dish_deg=15.0,
                              observer_distance_km=obs_dist_km)
```

Note v1 is also shipped inside the v2 package as `surrogate_v1.py` for convenience.

## Empirical v2 validation (2026-04-18, series m130)

Four back-to-back re-scoring experiments (`12_brightness_surface/m130*`) pinned down where v2 actually helps:

**Population LC fidelity (100 m048 seeds × 500 epochs at truth)**: pooled MAE 0.0355 → 0.0098 (3.6× better), bright MAE 0.071 → 0.011 (6.6×), max 2.08 → 0.41 mag (5×). v2 beats v1 on every single seed.

**Chaotic single seed (seed 19, 1000-epoch hi-fi)**: v1 MAE 0.033, bright 0.055, max 0.505 → v2 MAE 0.010, bright 0.009, max 0.189.

**m115 DE basin endpoints (17 seeds)**: Spearman vs hi-fi 0.996 (v1) → 0.9993 (v2). Both already tight — m115's clustering preserves only good candidates.

**m124 polished off-truth candidates (25 incl. 5 truth-refs)**:
- At truth, v1 reports hifi_mse overestimated **1.7–3.4×**; v2 reports **1.03–1.08×**. This means v2 can distinguish observation noise floor from real modelling error, v1 cannot.
- Off-truth (polished-from-basins): v1 p90 log-ratio +0.26; v2 p90 +0.026. Spearman 0.974 → 0.989.
- **No "catastrophic" >10× anti-correlation on either surrogate** — the 2026-04-16 seed-27 claim was retracted on b906691 re-run and does not reproduce here.

**Latency (median over 10+ runs, OPENBLAS_NUM_THREADS=1)**:

| config | v1 | v2 | v2-fast | v2 slowdown |
|---|---:|---:|---:|---:|
| single epoch | 0.21 ms | 0.72 ms | 0.38 ms | 3.4× |
| 500-epoch LC | 3.5 ms | 53 ms | 46 ms | 15× |
| 1000-epoch LC | 6.4 ms | 84 ms | 72 ms | 13× |
| 10k-dir manifold | 87 ms | 804 ms | 640 ms | 9.2× |
| 50k batch | 536 ms | 4.3 s | 3.3 s | 8.1× |

NoShadowEvaluator (3840-facet BRDF) dominates the overhead; it amortises with larger batches. v2-fast-mode (single ensemble member) gives 14–17% speed-up for a 23% accuracy hit — **not a useful middle ground**.

**Decision matrix — use v2 where accuracy matters and call count is bounded; keep v1 where call count is in the tens-of-thousands and finding the basin matters more than ranking inside it:**

| call site | swap? | rationale |
|---|---|---|
| `hifi_isoshell_viewer.py` PAB manifold | **v2** | one-off rendering, 500 epochs × 800 ms = ~7 min per rebuild, fixes backlit noise |
| Hi-fi validation (m115 step 2, m124) | **v2** | ≤30 calls per seed, must reflect the noise floor faithfully |
| OK/PARTIAL/FAIL classification threshold | **v2** | v1's truth-MSE overestimate (1.7–3.4×) makes the 0.1-mag² gate physically incoherent |
| DE inner loop (3-DOF, 6-DOF, L-BFGS polish) | **v1 for search, rescore top-K with v2** | 15× slowdown turns 18s DE → 4.4 min. v1 Spearman on basin endpoints 0.996 — good enough to find the right basins, v2 only needed to rank inside them |
| 6-DOF global search (future) | **v1 exploration, v2 polish** | same rule — cheap exploration, accurate polish |

## Migration notes: swapping v1 → v2 in existing code (2026-04-18)

Files that currently call v1 and need to be audited for swap:

- `notebooks/inversion/lib/hifi_isoshell_viewer.py` (lines 37–55): `SURROGATE_WEIGHTS = 's10_5M_weights.npz'`; constructor call at line 431. **Swap plan**: change `_SURROGATE_DIR` to the `surrogate_model/surrogate_model` subfolder, replace the two-arg constructor with `SurrogateModel.load_default()`. `predict_magnitude` signature is unchanged, so `compute_manifold_hifi` needs no downstream changes.
- `notebooks/inversion/12_brightness_surface/` — 15+ files reference v1 (m114, m115, m116, m119, m120, m121, m122, m123, m124, m126, m127, m128, m129, plus one archived script). These either construct `SurrogateModel(weights, norm)` directly or share a loader helper.
- `notebooks/inversion/EXPERIMENTS.md`, `journey.md`, reports — prose-level references; update only if numerics are quoted.

**Open question before bulk migration**: v2 is ~30× slower than v1 per LC (130 ms vs 4 ms). For pipelines that run millions of LC evaluations (DE with tens of thousands of trials × multi-start), that's a real budget hit. The right answer is likely "v2 everywhere except the inner loop of DE/grid, where v1 stays as a fast pre-filter and v2 re-scores the top-K". Keep v1 usable via `surrogate_v1.py`.

**What the v2 accuracy gain buys us**: the "off-truth surrogate unreliable" failure modes documented below (plateau catastrophe on seeds 12/27/33/36/46, surrogate gradient anti-correlated with hi-fi gradient) were flagged on v1. Whether they persist on v2 is an empirical question — the 5.5× lower bright-MAE and 6.8× tighter max-error tail should shrink the "coverage holes" that caused the anti-correlation, but this needs a targeted re-run of (a subset of) m124 / m126 on v2 to confirm.

## Limitations

- **IS-901 specific.** Geometry + BRDF parameters baked into `s11_geometry.npz`. Retraining required for any other satellite.
- **Panel range**: [-180°, 180°] (periodic, handled via sin/cos encoding). **Dish range: [0°, 90°] — do NOT exceed.**
- **v2 tail** (empirical): max error ~0.1–0.2 mag on real trajectories, versus ~0.6–0.75 mag on v1. Degrades gracefully in the very-faint regime (log10(phi) < -4 → MAE ~0.03 mag, still fine for observational SNR).
- **Zero-flux edge cases**: when `log10(phi) < -15`, `predict_magnitude` returns `np.inf` by design.
- **Does not capture articulation changes beyond the training range** (panel uniform on [-180, 180], dish uniform on [0, 90]; IS-901 inversion uses fixed `panel=0, dish=15` so this is not a concern in practice).

## Dim-regime caveat — RETRACTED (2026-04-15 afternoon)

A previous revision of this page (early 2026-04-15) claimed a "dim-regime caveat": that tight-IPL / dim constraint epochs produce surrogate residual median 0.491 / p90 4.21 / max 8.14 mag at truth. **That claim was based on corrupted geometry from m119 v1** (1-hour m046 data mixed with 6-hour setup_experiment geometry, k2 off by up to 74° at the end of the run). Not a surrogate property.

## Fidelity at truth — measured honestly (2026-04-15 afternoon, [inline])

Surrogate evaluated at the exact truth trajectory across all 500 epochs, consistent geometry (`end_time_utc='2020-02-05T11:00:00'` → 1-hour window matching m046). Compared against the **noiseless** `master.mag_hifi[seed]`:

| seed | MAE | p50 | p90 | p99 | max | bright (mag<10) MAE | dim (mag>=10) MAE |
|------|-----|-----|-----|-----|-----|---------------------|-------------------|
| 14 | 0.032 | 0.022 | 0.076 | 0.166 | 0.250 | 0.067 | 0.027 |
| 27 | 0.031 | 0.021 | 0.064 | 0.201 | 0.582 | 0.065 | 0.029 |
| 0  | 0.034 | 0.021 | 0.070 | 0.239 | 0.589 | 0.077 | 0.030 |
| 93 | 0.035 | 0.024 | 0.080 | 0.173 | 0.202 | 0.073 | 0.030 |

Key observations:
- Overall MAE (~0.03 mag) matches the advertised 0.061-all / 0.045-bright headline.
- **Dim epochs are NOT worse than bright.** Across all 4 seeds, dim MAE (~0.029) is ~2.5× LOWER than bright MAE (~0.070). Bright epochs (glint / ±X peaks) are where the surrogate struggles most, not dim ones. The "dim-regime caveat" above was backwards.
- Max residuals ~0.2-0.6 mag at a handful of epochs, consistent with the "0.6 mag extreme geometry" note in Limitations.

The [[m119v2_attitude_isoshell]] residual medians (0.043 at truth, p90 0.112 against noisy observed) now reconcile cleanly: surrogate noise floor (~0.03 mag) + observation noise sigma (0.05 mag) ≈ 0.06 mag RMS, so residual p90 ~0.11 is in line.

## Surrogate unreliable OFF truth (2026-04-16, [[m124_hifi_validate]])

> ## ✅ 2026-04-17 — m124 RE-SCORED ON CORRECT 1-HOUR WINDOW; SURROGATE REHABILITATED
>
> m124 re-run (commit `b906691`) on the correct 1-hour window flips the agreement ratio from **2/12 (REFUTED)** to **9/15 (PARTIAL)**. The seed-27 "12–16× WORSE" catastrophe also disappears — seed 27 basin 1 polishes from surrogate 1.0 to hi-fi 0.0222 (PARTIAL class); the wrong-window result was a mismatched-LC artefact, not a surrogate failure mode. The qualitative "surrogate is reliable at truth" claim is fully confirmed (truth-polish gives hi-fi 0.0024 within 1σ noise floor across all 5 seeds).
>
> The broader "surrogate gradient anti-correlated with hi-fi gradient on the dark-mag plateau" story from April-16 does not replicate on the correct window — polish actually moves toward truth on most basins ([[m126_wrapped_pipeline]]'s 33/33 basins helped). The remaining disagreement (6 of 15 basins outside ±30%) is consistent with surrogate MAE widening off-truth, not anti-correlation. This removes "surrogate unreliable off-truth" as an architectural obstacle — the wrapped-pipeline `keep_min` wrapper still catches any basin-level polish miss, but such misses are now rare.

The 0.03 mag MAE / r=0.999 fidelity numbers above are measured **at the truth trajectory**. [[m124_hifi_validate]] hi-fi-validated 12 L-BFGS-B-polished candidates (from [[m123_lbfgs_polish]]) that started at m115 DE basins (q0 errors 100–180° from truth) and asked: does the surrogate-cost reduction match the hi-fi MSE reduction?

**Answer: NO. Only 2/12 within ±30% (log-ratio).** Three regimes:

| regime | seeds | surrogate ratio | hi-fi ratio | interpretation |
|---|---|---|---|---|
| catastrophic | 27 (3/3 basins) | ~1.15× improvement | **0.06–0.08× (12–16× WORSE)** | surrogate gradient anti-correlated with hi-fi |
| over-statement | 74, 93 (most), 14 (some) | 3–13× improvement | 1.2–6.6× | surrogate over-states gain by 2–3× |
| agreement | 14 basin_1 (twin-of-truth), 93 basin_2 | 6–13× | 7–10× | surrogate and hi-fi agree |

The polished `(q0, ω)` candidates that produced post-polish hi-fi MSE ≈ 4.5 mag² on seed 27 had RMS predicted-vs-observed magnitude error of ~2 mag, while the *surrogate* still reported `mean_L1` cost of ~1.34 (well within the dark-mag-saturated regime; see [[dark-mag-saturation]]). The surrogate cannot see the ~2 mag hi-fi errors at these geometries because they fall outside its training-distribution coverage.

**Implication for downstream pipelines:**

- **At truth** (within the [[basin-of-attraction]] ω-dir basin <0.1°): surrogate is faithful. Truth-polish is safe. Hessian / sensitivity work is valid.
- **Near truth + valid attractors** (twins, near-symmetries): surrogate gradient appears physical (e.g. seed 14 basin_1).
- **Far-from-truth attractors** (seed 27 basins): surrogate "gradient" descends a modelling-error landscape. Following it can move candidates to a *worse* hi-fi state.

Practical consequence: any pipeline using the surrogate as an objective for off-truth optimisation (DE+L-BFGS hybrid, HMC, gradient descent) MUST hi-fi-validate the result. The surrogate is a search heuristic, not a ground-truth oracle, off-truth.

This caveat does NOT invalidate:
- [[m120_tumbling_competitors]]'s truth rank 0/10004 result (that's an at-truth claim, scoring KNOWN truth against competitors).
- [[m115_surrogate_pipeline]]'s 10/10 valid solutions (that uses hi-fi scoring to rank DE outputs, exactly the safety wrapper that m124's failure mode requires).
- The "gold-standard cheap omega-pool validator" pattern above (uses surrogate-DE as an internal validator for ranking, paired with downstream hi-fi).

## Off-truth extension and seed 33 flipped-ω pocket (2026-04-16, [[m126_wrapped_pipeline]])

[[m126_wrapped_pipeline]] further characterised the off-truth surrogate behaviour on 6 additional seeds. Two new observations:

1. **Plateau catastrophe is predictable and common.** On 5 of 11 baseline seeds (12, 27, 33, 36, 46), polish driven by surrogate cost produces hi-fi MSE regressions of 5–15×. Root cause: surrogate's `(k1, k2)` training distribution has coverage holes at the body-frame geometries traced by these seeds' wrong-q0 attractors. The predictor is free at DE time: upstream ω-dir ≥ 5° + single upstream candidate + all basins q0_err ≥ 10° → expect plateau catastrophe. Wrapped pipeline catches it without extra cost.
2. **Flipped-ω attractor is a separate surrogate local minimum.** Seed 33 has a basin at `(q0 ≈ 98°, ω_retrograde, |ω| = 30% of truth)` where the surrogate reports `mean_L1 ≈ 0.70` and hi-fi MSE is 0.082 (sub-OK-threshold 0.1). This is NOT a training-distribution coverage hole — it is an honest surrogate local minimum at a non-truth physical state that happens to produce a similar LC. See [[omega-sign-degeneracy]] for the mechanism hypothesis. Practical implication: L-BFGS polish from this attractor moves the state slightly DEEPER into the pocket (surrogate drops 60–68%, hi-fi moves 0.082 → 0.10–0.15). The wrapper catches the hi-fi regression but the `(q0, ω)` state is still a valid observational solution worth reporting under [[multi-solution-philosophy]].
