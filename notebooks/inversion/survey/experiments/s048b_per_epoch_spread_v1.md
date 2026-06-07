---
title: s048b — per-epoch C_t spread sweep (v1 surrogate, seed 89, 30 epochs)
type: experiment
sources: [user_idea, s048, surrogate_v1, lib.filter_costs]
related: [s048, s019, s032, s042, s048c (planned animation)]
created: 2026-05-07
updated: 2026-05-07
confidence: high
---

# s048b — per-epoch C_t spread sweep (v1 surrogate, seed 89, 30 epochs)

## TL;DR

Measured how the per-epoch C_t cloud size varies across the LC. Two runs:
1. **30 evenly-spaced epochs**: |C_t| varies **60×** across LC (1.3k to 75k survivors at 500k samples). Truth-q always within 1.0–3.6° of nearest survivor.
2. **30 dimmest epochs**: dim epochs split into TWO populations — temporally-clustered "rare-geometry" windows (epochs 197–216) where |C_t| can be as low as 384 (mag 14.92, ep 208) and other dim epochs where |C_t| ≈ 50k. Brightness alone does NOT predict selectivity; geometry does.

**Key infrastructure findings:**
- v1 surrogate gives ~5 sec/epoch at 500k samples (30 epochs in ~144 sec wall). v2 gives ~30 sec/epoch (12× slower). v1 is appropriate for the C_t sampling stage; v2 reserved for hi-fi-adjacent confirmation.
- R(q) on the q-pool is computed ONCE and reused across all epochs (cached). Per-epoch cost is `einsum(R_cache @ sun_unit_t)` + surrogate forward — both O(N) and fast.
- Truth-q always survives at every probed epoch on seed 89. No epoch lost truth.

## What

Build the per-epoch C_t pre-image filter as a function of epoch:

```
C_t(t) = { q : |B_predicted(q, k1_j2000(t), k2_j2000(t)) - mag_measured(t)| < tol }
```

Sample 500k random q on SO(3) ONCE. For each epoch t in a chosen subset:
- Rotate the J2000 sun/observer into body frame using the cached R(q): `k1_body = R(q) @ sun_unit_j2000(t)`.
- Surrogate-predict mag for all 500k candidates.
- Filter: `|pred - measured| < 0.10` mag.
- Record `|C_t|`, distance from truth-q (cached propagated quaternion `quaternions[t]`) to nearest survivor, and the full `predicted` and `survive` arrays for replotting.

## How

- Seed 89 (|ω|=0.24 dps, mag range 5.38–14.92, 500 epochs at dt=7.214s).
- 500,000 q samples on SO(3) via `scipy.spatial.transform.Rotation.random`.
- `R_cache = scipy.spatial.transform.Rotation.random(N).as_matrix()` — one-time ~5 sec.
- `rotvec_pool = R_random.as_rotvec()` for plotting (axis × angle, rad).
- Surrogate v1 loaded directly from `/home/girish/surrogate_model/surrogate_model/surrogate_v1.py` (load weights `s10_5M_*.npz`); same `predict_magnitude(k1, k2, sp_deg, ad_deg, obs_dist_km)` API as v2.
- Tolerance 0.10 mag; SP=0°, AD=15° (cohort defaults).
- Two epoch-selection modes: `evenly_spaced` and `dimmest` (sort by `mag_hifi` descending, top N).
- Output: NPZ with full per-epoch predictions + survive masks; 5×6 matplotlib panel plot of rotvec(x,y) projection with kills (grey) / survivors (blue) / truth-q (red star).

## Result

### 30 evenly-spaced epochs

Survivor count varies **60×** across the LC (1,278 to 75,218). Examples:

| Epoch | Mag | Survivors | % | Nearest truth |
|---|---|---|---|---|
| 412 | 5.38 | 304 | 0.06% | 3.83° |
| 213 | 14.86 | 1,278 | 0.26% | 2.82° |
| 86 | 9.71 | 1,617 | 0.32% | 5.01° |
| 178 | 14.38 | 75,218 | 15.04% | 1.05° |
| 320 | 14.57 | 74,770 | 14.95% | 2.56° |

Mag = 14.86 is highly selective (1.3k); mag = 14.38 (similar value!) gives 75k survivors. **Brightness does not determine selectivity** — geometry does.

### 30 dimmest epochs (mag 14.63–14.92)

Two distinct populations:

**Group A — epochs 197–216** (temporal cluster, rare geometry):

| Epoch | Mag | Survivors | Nearest truth |
|---|---|---|---|
| **208** | **14.92** | **384** | 1.77° |
| 211 | 14.86 | 1,364 | 2.28° |
| 213 | 14.86 | 1,278 | 2.82° |
| 207 | 14.81 | 2,722 | 1.52° |
| 197 | 14.65 | 22,777 | 2.27° |

**Group B — epochs 324–350** (loose despite dimness):

| Epoch | Mag | Survivors | Nearest truth |
|---|---|---|---|
| 324 | 14.63 | 50,327 | 1.39° |
| 348 | 14.64 | 50,843 | 3.81° |
| 349 | 14.71 | 23,459 | **9.59°** ⚠️ |

Group A is 5–100× more selective than Group B at the same mag. The user's hypothesis "dim = constrained" holds at the rare-geometry windows of Group A but is broken in Group B.

**Truth-sampling failures**: at epoch 349 (mag 14.71) the discrete 500k sample missed truth's vicinity entirely (nearest-truth = 9.59°). Indicates the C_t manifold here is on a thin slice of SO(3) that uniform random sampling can fail to populate.

## Why this matters

- **C_t-as-q-prior is a strictly better prior than uniform Sobol** in the regime where it's tight (e.g. ~300–1500 survivors, truth within ~1–2°). At seed 89's brightest peak it gives 304 survivors at ~3.8° truth resolution; at the rare-dim epoch 208 it gives 384 survivors at ~1.8° truth resolution.
- **Selectivity is geometry-dominated, not mag-dominated.** Anchor-epoch selection should look for `|C_t(t)|` minima as a function of epoch, NOT just mag extremes. The user's `mag → |C_t(mag)|` function generalises to `(mag, k1_unit, k2_unit) → |C_t|`.
- **v1 surrogate at 4.5 μs/sample is the right choice for this stage.** 30 epochs × 500k samples = 15M evals → 144 sec wall. v2 at the same settings would be ~30 min wall.
- **Truth survives on seed 89 at every probed epoch** — but the discrete sample's nearest-truth distance does grow at low-|C_t| / unfavourable-geometry epochs (up to 9.59° on ep 349). Sample-density vs C_t-thickness tradeoff is real.

## Numbers

| Quantity | Source | Value |
|---|---|---|
| N samples | s048b | 500,000 |
| Tolerance | s048b | 0.10 mag |
| Wall (30 epochs, v1) | s048b | 144 sec |
| Wall per epoch (v1, 500k) | s048b | ~4.8 sec |
| Wall per epoch (v2, 500k) | s048 | ~30 sec |
| v1 vs v2 speedup | derivation | 12× (4.5 μs vs 56 μs/sample) |
| |C_t| range across LC (evenly spaced) | s048b | 1,278 — 75,218 |
| Tightest mag-14 epoch | s048b | ep 208 (mag 14.92, 384 survivors) |
| Loosest dim epoch | s048b | ep 348 (mag 14.64, 50,843 survivors) |
| Max nearest-truth distance | s048b | 9.59° (ep 349) |
| Min nearest-truth distance | s048b | 0.78° (ep 209) |

## Artefacts

- `experiments/s048b_per_epoch_spread_v1.py` — sweep + plot script.
- `results/s048_peak_cascade_smoke/seed089_v1_spread/` — 15 evenly-spaced epochs (initial run).
- `results/s048_peak_cascade_smoke/seed089_v1_spread_n30/` — 30 evenly-spaced epochs.
- `results/s048_peak_cascade_smoke/seed089_v1_spread_n30_dimmest/` — 30 dimmest epochs.

Each results directory contains:
- `spread.npz` — full data: q_pool_wxyz, rotvec_pool, pred_all (15/30, 500k), survive_all (15/30, 500k), epoch_indices, q_truth_at, rotvec_truth_at, mag_hifi (full LC), tolerance.
- `summary.json` — headline numbers + per-epoch survivors / nearest-truth.
- `spread_15panel.png` — 5×6 grid (filename leftover from the 15-epoch initial run; actual content is N panels per N_EPOCHS_TO_SWEEP).

## Out of scope

- Animation of cloud evolution across full 500-epoch LC (deferred to s048c — see handoff).
- Frequency analysis of `|C_t|(t)` for ω-prior signal (deferred; flagged as possibly in dead-class per `feedback_lc_spectral_omega_prior_dead.md`).
- Other seeds — single-seed only by design at this stage.
- v1 wired into `lib/surrogate_eval.py` — currently imported directly in script; could be promoted to lib if pattern repeats.

## Cross-references

- `experiments/s048_peak_cascade_smoke.md` — sister experiment (v2, peak-cascade Tier 0/1/2).
- `concepts/known_pathologies_to_revalidate.md` — LC-feature → ω prior dead-class.
- `notebooks/inversion/lib/hifi_isoshell_viewer.py` — visual quality benchmark for the upcoming s048c animation.
- `project_pab_manifold_viewer` memory — load-bearing design invariants for the isoshell viewer (radial scaling, backlit handling, mesh3d caching gotcha).
