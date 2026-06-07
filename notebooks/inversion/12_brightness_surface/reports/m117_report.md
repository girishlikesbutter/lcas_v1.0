# m117 — Grid + NM harvester (skip geo refinement)

## Hypothesis

The geo-refinement / phi-sweep / hi-fi stages of m103 are not needed to
produce a useful omega-vector pool for downstream surrogate-DE omega
ranking (`archive/inline_omega_selection_test.py`). The 26 NM-refined omega
candidates from grid + NM alone should be diverse enough to cover the
truth omega across most seeds, and downstream DE re-solves q0 anyway.

## Method

Replicates m103_hybrid stages 2 + 3 only:
1. **Grid:** 2000 fibonacci omega directions × 20 magnitudes (HEALPix-ish
   sampling), N_PHI_COARSE=36 anchor twist evaluations per allowed normal.
   Same `eval_one_direction` body as m103.
2. **NM refinement:** Top-300 grid winners refined with Nelder-Mead
   (200 iter, xatol=1e-6, fatol=1e-10), same `refine_one_nm` body.
3. **Pool selection:** Top-26 by NM cost, no dedup (pool diversity
   preserved on purpose; downstream surrogate-DE handles the ranking).

Geo refinement is **skipped on purpose** — its purpose in m103 is to
sharpen q0/omega for the chosen winner, which the downstream DE step
re-solves from scratch.

## Schema notes (key reuse)

The output NPZ is named `geo_ckpt.npz` and uses the exact key set the
downstream loader expects (`n_candidates, omega_ranks, phi_ranks, q0_refs,
w0_refs, geo_costs, q0_ref_errs, w0_ref_errs`). Semantic remapping:

- `geo_costs` holds **NM cost** (since geo wasn't run). Downstream uses
  this purely as a ranking proxy column, never absolute scale.
- `q0_refs` is identity quaternion rows — phi-sweep wasn't run, and
  downstream DE solves q0.
- `phi_ranks`, `q0_ref_errs`, `w0_ref_errs` are zeros (unused downstream).
- `omega_ranks` carries the original grid rank for traceability.

## Parameters

| Env var | Default | Meaning |
|---|---|---|
| `MICRO117_SEED` | required | Trajectory seed |
| `MICRO117_POOL_SIZE` | 24 | Inner Pool workers (set to 3 under outer parallel harness) |
| `MICRO117_OUTPUT_DIR` | `data/results/inversion_diagnostics/harvester` | Output root |

## Output

```
{output_dir}/seed_{NNN}/
├── grid_ckpt.npz   # top-300 grid (intermediate insurance)
├── geo_ckpt.npz    # top-26 NM pool (downstream input)
├── result.json     # timing + nm_cost_stats + metadata
└── run.log         # full stdout
```

## Expected runtime

Per seed, on 24 workers: ~grid 60–90 s, NM ~30–60 s, total ~2–3 min.
On 3 workers (outer-parallel harness, e.g. 8 seeds × 3 workers): ~10–15 min/seed.

## Downstream use

`notebooks/inversion/12_brightness_surface/archive/inline_omega_selection_test.py`
loads `w0_refs` and `geo_costs` from `geo_ckpt.npz`, runs one
surrogate-DE start per omega, and ranks candidates by surrogate MSE.
This harvester lets that test run on any seed without first executing
the full m103 pipeline (saving ~10 min/seed of geo + hi-fi work
that the test discards anyway).
