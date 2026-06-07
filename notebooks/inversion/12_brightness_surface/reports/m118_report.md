# m118 — IPL-cost landscape diagnostic (seed 14)

## Hypothesis

Alignment cost using IPL centroids as targets **generalizes** the m103 facet-normal cost:
instead of coarse-banded allowed normals driven by magnitude thresholds,
each epoch gets a set of candidate body-frame PAB directions coming directly
from shadow-isoshell intersections. Further, **weighting by IPL tightness**
(inverse set length) should **sharpen the truth basin** because short IPL sets
are geometrically high-confidence.

The active-centroid scorer is an **oracle ceiling**: it tells us the best
discrimination the IPL representation can provide if the right sub-set were
selected. The extended scorer probes whether using *all* tight-IPL epochs
(not only spec peaks) buys resolution.

## Method

### Anchor selection
- Over all 500 epochs with `loop_count >= 1`, pick the epoch with the smallest
  IPL length. The IPL at this epoch is the tightest constraint we have on the
  body-frame PAB, so the anchor-twist manifold is most compact.
- If the anchor epoch has multiple centroids, each is used as an independent
  anchor choice — the cost tensor carries a `C` axis enumerating them.

### Constraint epoch set
- `spec_peaks` — `find_peaks(-observed_lc, distance=5, prominence=0.3)`,
  then filter `observed_lc[p] < 9.0`, minus the anchor epoch. Same rule as
  m103.
- `tight_ipl_epochs` — `loop_count >= 1 AND length < median(length)`.
- **Kernel epochs = union(spec_peaks, tight_ipl_epochs) \ {anchor}**.
  Variants that only want spec peaks mask down to that subset at scoring time.

### Omega grid
- 2000 fibonacci-sphere directions × 20 magnitudes spanning
  `omega_est_rad × np.linspace(0.70, 1.30, 20)`.

### Cost variants (5)

| # | Name | Targets per epoch | Weights |
|---|------|---|---|
| 1 | `facet_normal` | `unique_normals[get_allowed_normals(mag)]` | uniform |
| 2 | `ipl_centroid_uniform` | all centroids at that epoch | uniform |
| 3 | `ipl_centroid_weighted` | all centroids at that epoch | `1/(length+eps)` normalized so Σw = N |
| 4 | `ipl_active_centroid` (**oracle**) | single active (truth-containing) centroid | uniform |
| 5 | `ipl_centroid_weighted_ext` | all centroids | same as #3 but on the extended epoch set |

### Kernel representation
`q_delta[dir, mag, epoch]` propagated from `q_identity` under `ω = dir·mag`
across `Δt = obs_time[epoch] − anchor_time`. Stored as float32 wxyz.
Built once with `Pool(POOL_SIZE=24)` over the 2000 directions.

### Anchor attitude grid
`q_anchor[c, phi] = anchor_q_from_phi(phi, centroid_c, pab_j2000[anchor])`
with `N_PHI = 360` (uniform over [0, 2π)). Built in-process in the diag script.

### Cost formula (per variant)
```
q_total = q_anchor[c, phi] ⊗ q_delta[dir, mag, epoch]     # Hamilton product, wxyz
pab_body = R(q_total) @ pab_j2000[epoch]
cost[dir, mag, c, phi] = Σ_ep w_ep · (1 - max_{target ∈ targets(ep)} pab_body · target)²
```

## Expected runtime (honest guesses)

- **Script 1 (kernel build):** ~300 constraint epochs expected (~17 spec peaks + ~250 tight-IPL epochs).
  2000 dirs × 20 mags × ~300 epochs `propagate_attitude` calls
  ≈ 40k calls, parallelized 24-wide. My best guess: **20–40 min** on 24 cores.
- **Script 2:** library only, no runtime.
- **Script 3 (diag):** each variant's cost is a
  `(2000, 20, C, 360)` tensor. Per variant dominant cost is ~`D·M·C·P·E` einsum
  ≈ 2k·20·C·360·~50 ≈ 7×10⁸ ops for C=1, ~5–30 s per variant.
  Plus top-K q0-reconstruction: 10 000 × 2 `propagate_attitude` calls,
  **likely the bottleneck: ~1–3 min per variant, serial.**
  5 variants × ~2 min ≈ **10–15 min total**.

## Output paths and schemas

Base directory: `data/results/inversion_diagnostics/m118/seed_014/`

### Kernel (Script 1)
- `kernel.npz` — full kernel (see spec for keys; includes `pab_anchor_j2000`).
- `kernel_summary.json` — metadata (anchor epoch, epoch counts, timing).
- `run.log`

### Diagnostic (Script 3)
- `cost_<variant>.npz` — `{cost: (D, M, C, P) float32}` for each of 5 variants.
- `topK_<variant>.npz` — per-variant top-10 000 sorted by cost with reconstructed
  `q0, omega, q0_err, w_dir_err, w_mag_err`.
- `summary.json` — per-variant analytics: rank-1 errors, truth-neighbor rank +
  cost gap, best-omega-in-pool errors, basin count in top-1000 (5° threshold),
  plus `BEST_OMEGA_RANK`, `TRUTH_OMEGA_BASIN_RECOVERED`, `Q0_ERR_AT_RANK_1`.
- `plots/<variant>_mollweide.png` — Mollweide projection of per-direction
  minimum cost (one panel per variant).
- `plots/<variant>_topK_scatter.png` — `(w_dir_err, q0_err)` scatter of the
  top-10 000, colored by cost.
- `plots/truth_rank_summary.png` — cross-variant bar chart of truth rank and
  cost gap.
- `diag_run.log`

## The 4 deliverables (diag script)

1. **Per-variant cost tensor** — `cost_<variant>.npz`, shape `(D, M, C, P)`,
   ~115 MB each × 5 ≈ 575 MB.
2. **Per-variant top-K** — `topK_<variant>.npz`, 10 000 candidates with full
   kinematic reconstruction and error metrics.
3. **Diagnostic plots** — Mollweide landscape + error scatter per variant,
   plus a cross-variant truth-rank summary.
4. **Summary JSON** — single-file rollup enabling post-hoc comparison across
   variants.
