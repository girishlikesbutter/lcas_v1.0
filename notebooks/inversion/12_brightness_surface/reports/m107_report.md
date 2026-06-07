# m107 -- IPL Centroid Grid Cost Function

## Hypothesis

Replacing the brightness-threshold alignment cost with epoch-specific IPL centroid alignment at the grid search level improves omega discrimination, especially for the 87% of seeds that lack bright +/-X constraints (m096 census).

## Rationale

The standard pipeline (m102) uses `get_allowed_normals(mag)` to determine which body-frame face normals are "allowed" at each spectral peak, based on brightness magnitude bands. This has two weaknesses:

1. **Magnitude bands are coarse.** The 4 bands (mag < 5.9, 6.3, 7.3, else) admit too many normals for faint peaks, making the constraint uninformative.
2. **Bright peaks are rare.** 87% of seeds lack epochs where only +/-X normals are allowed, so the strongest constraints are never activated.

IPL (Isophote-PAB Loop) minima provide a geometry-derived alternative: at each epoch, the brightness surface has isolated minima where isophote loops pinch close to the PAB direction. The centroid directions of these loops are body-frame directions that the PAB *must* align with (within the loop's angular extent). Tight minima (small angular distance) give highly discriminative constraints.

## Method

Base: `m102_fullmse.py` (499 lines). Changes are surgical -- only the constraint mechanism differs.

### What changed

| Component | m102 | m107 |
|-----------|----------|----------|
| Anchor selection | Brightest spectral peak | Tightest IPL minimum (smallest `ang_dist_deg`) |
| Anchor normals | `get_allowed_normals(anchor_mag)` -> face normal indices | IPL centroid directions (arbitrary unit vectors) |
| Constraint epochs | Other spectral peaks (mag < 9) | Other tight IPL minima (`ang_dist < 15 deg`, K=8 max) |
| Constraint normals | `get_allowed_normals(mag)` per epoch | Epoch-specific centroid arrays |
| Cost function | `vectorized_phi_cost_excl` (indexed into shared normals) | `vectorized_phi_cost_ipl` (epoch-specific centroid arrays) |
| Phi sweep range | `[0, pi)` for XY normals, `[0, 2pi)` for Z normals | `[0, 2pi)` for all centroids (arbitrary directions) |
| Geo refinement | All spec peaks + `get_allowed_normals` | All tight IPL epochs + centroid arrays |

### What is unchanged

- Steps 2b (lo-fi peak matching), 5 (multi-window hi-fi): identical to m102
- Grid structure: 2000 Fibonacci directions x 20 magnitude bins
- NM refinement: same optimizer, same dedup logic
- Winner selection: full-window MSE
- All pipeline parameters: NM_TOP=300, GEO_TOP=20, LOFI_TOP=300

### IPL data loading

Reads `data/results/inversion_diagnostics/isoshell_viewer/ipl_census.json`. Per seed:
- Sort `minima_detail` by `ang_dist_deg` ascending
- Take those with `ang_dist_deg < 15`, cap at K=8
- Fallback: if fewer than 2 tight minima, use top-K by angular distance regardless of threshold

### New cost function

```python
def vectorized_phi_cost_ipl(q_anchors_xyzw, delta_qs, pab_arr, constraint_centroid_arrays, w):
    """
    For each constraint epoch ci:
      pbs = R_all.apply(pab_arr[ci])           # PAB in body frame
      centroids = constraint_centroid_arrays[ci] # (n_centroids, 3)
      bds = (pbs @ centroids.T).max(axis=1)     # best dot per phi
      costs += w * (1 - bds)^2
    """
```

Key difference: `constraint_centroid_arrays[ci]` is a variable-size array specific to each constraint epoch, rather than indices into the shared 10-normal array.

### Diagnostics

1. **Truth omega rank:** Prints where the truth omega direction falls in the IPL-cost grid ranking (for comparison with m102's standard-cost ranking).
2. **Truth PAB-centroid alignment:** For each tight IPL epoch, prints how well the truth body-frame PAB aligns with the available centroids.

## Checkpoints

| File | Contents |
|------|----------|
| `grid_checkpoint.npz` | `grid_costs`, `grid_omegas`, `grid_best_centroid_idx`, `grid_best_phi_idx` |
| `lofi_checkpoint.npz` | `lofi_q0s`, `lofi_w0s`, `lofi_n_matched`, `lofi_mse`, `lofi_align_cost`, `lofi_anchor_ci` |
| `nm_checkpoint.npz` | `refined_costs`, `refined_omegas`, `refined_phi_idx`, `refined_centroid_idx`, `candidates_q0`, `candidates_w0` |
| `geo_checkpoint.npz` | `geo_q0_ref`, `geo_w0_ref`, `geo_cost` |
| `result.npz` | `q0_refined`, `w0_refined`, `true_q0`, `true_omega0` |
| `result.json` | Full result with classification, timing, IPL constraint details, truth grid rank |

## Usage

```bash
MICRO107_SEED=0 python3 notebooks/inversion/12_brightness_surface/m107_ipl_cost_function.py
```

Results saved to: `data/results/inversion_diagnostics/m107_ipl_cost/seed_000/`

## Expected outcomes

- Seeds with tight IPL minima (n_tight_15 >= 3): IPL cost should rank truth omega higher than standard alignment cost
- Seeds with only loose minima: fallback to top-K may not improve, but should not regress (same candidate pool enters lo-fi/hi-fi)
- Overall: if IPL centroids are more discriminative than magnitude bands, grid rank improvement should translate to fewer FAIL seeds after hi-fi selection

## Classification

- **OK**: q0 < 5 AND w_dir < 5 AND |w_mag| < 5%
- **PARTIAL**: any metric 5-10
- **FAIL**: any metric > 10

## Status

NOT YET RUN. Script written, awaiting execution.
