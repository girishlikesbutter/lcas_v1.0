# 13_clean_slate_omega — shared lib contract

This `lib/` is the single source of truth for data loading, forward simulation,
and scoring across the three sub-experiments:

- `a_spectral/` — Idea 1: recover |ω| from LC spectrum, then sweep ω_direction on S² under residual threshold.
- `b_differentiable/` — Idea 2: differentiable end-to-end inversion (PyTorch v2 surrogate, Adam + many starts).
- `c_learned_inverse/` — Idea 3: learned LC → ω regressor (mixture-density head so we get a distribution natively).

All three use:

```python
from lib.data import load_seed, all_seeds
from lib.forward import predict_lc, get_surrogate, body_vectors_from_attitude
from lib.scoring import lc_mse, omega_errors, quat_geodesic_deg, classify
from lib.scoring import RESIDUAL_MSE_GATE, RESIDUAL_MSE_TIGHT, V2_TRUTH_MSE_FLOOR
```

## Truth bundle per seed (`load_seed(n)`)
See `lib/data.py` docstring — returns everything the three ideas need.
Notably includes `mag_hifi` (the observed LC), inertial sun/obs vectors,
the inertia tensor (shared across seeds), and the ground-truth (q0, ω).

## Forward model contract
`predict_lc(q0, omega0, inertia_tensor, observation_times, sun_j2k, obs_j2k, sat_j2k, obs_dist)`
→ predicted magnitude vector (500,). Uses v2 surrogate by default. Matches
the convention `propagate_attitude(..., mode="tumbling")` used to generate m048.

## Residual thresholds
Set once in `lib/scoring.py`:
- `RESIDUAL_MSE_TIGHT = 0.005` mag² → "OK"
- `RESIDUAL_MSE_GATE  = 0.02`  mag² → "PARTIAL"
- above = "FAIL"

## Checkpointing rule
Per `feedback_checkpoint_design_first.md` and `feedback_save_results.md`:
**every** script saves to NPZ/JSON, every stage saves candidates, never just
"best". Directory layout:

```
data/results/inversion_diagnostics/13_clean_slate_omega/<subseries>/seed{NNN}/
```

Minimum per-seed payload:
- `candidates.npz` — all (q0, ω, mse, meta) under GATE threshold
- `result.json` — summary (truth, best candidate, classification, runtime, notes)
- `lc_fit.npz` — predicted LC for top-K candidates for later visual inspection

## Wiki
Session findings go to `notebooks/inversion/13_clean_slate_omega/FINDINGS.md`.
Cross-reference into the wiki (`notebooks/inversion/wiki/`) follows the
`feedback_doc_structure.md` pattern — lean top-level FINDINGS.md + wiki
pages for each sub-idea as `branch/clean-slate-omega-<sub>.md`.
