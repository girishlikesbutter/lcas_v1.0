# m127 — Flipped-omega compensating-q0 search

## Hypothesis

Seed 33 (per m126) has a DE basin with omega **retrograde** relative to
truth (signed w_dir 161.5 deg, magnitude essentially unchanged at -0.7%)
plus q0 ~ 98 deg that produces hi-fi MSE 0.082 (near-OK class). An inline
probe confirmed: pure omega-sign flip at q0_truth gives surrogate MSE 3-10
for ALL 11 baseline seeds — so there's no universal (q0=truth,
omega=-omega_truth) symmetry. The question is whether a COMPENSATING q0
exists for each seed that makes (q0', -omega_truth) a near-valid LC.

**Falsifiable claim**: for at least 3 of the 11 baseline seeds (other than
seed 33), a 3-DOF q0 search with omega fixed at -omega_truth finds a
(q0', -omega_truth) state whose hi-fi MSE is < 0.5 (PARTIAL class or
better).

## Method (5-stage pipeline, per seed)

- **A. SO(3) grid scan** — super-Fibonacci N=60000 quaternions, score each
  by `mean_L1` residual between surrogate magnitude and observed LC
  (propagated under `omega_search = -omega_true`). Parallelised across
  POOL_SIZE workers (chunked over q0 indices; delta-quaternion transport
  precomputed once).
- **B. L-BFGS-B polish** — 3-DOF rotvec tangent around each of the top-20
  grid q0s, omega held fixed at `omega_search`. Options
  `ftol=1e-7, gtol=1e-4, maxiter=200, maxfun=1000`.
- **C. Clustering** — sort polished points by final surrogate cost, merge
  by q0 geodesic distance < 5 deg; cap at 5 basins.
- **D. Hi-fi validation** — vendored `hifi_validate` (copied from
  m115) on each basin, parallelised in `Pool(min(n_basins, 4))` with
  fork start.
- **E. Classification (custom labels)** —
  `FLIPPED_VALID` (hi-fi MSE < 0.1),
  `FLIPPED_PARTIAL` (0.1 ≤ MSE < 0.5),
  `FLIPPED_FAIL` (MSE ≥ 0.5).

Batch aggregation: counts + `hypothesis_verdict` in
`batch_summary.json`. Verdict excludes seed 33 (positive control):
CONFIRMED iff ≥3 seeds ≠ 33 are VALID or PARTIAL, REFUTED iff ≤1, else
MIXED.

## Target seeds

`[0, 6, 12, 14, 24, 27, 33, 36, 46, 74, 93]` — all 11 baseline seeds.
Seed 33 is the positive control: if the search doesn't find its q0 ~ 98 deg
basin, the search is broken, not the hypothesis.

## Expected outcomes + decision tree

- **≥3 non-control seeds VALID or PARTIAL** → flipped-omega-with-q0-
  compensation is a POPULATION-WIDE degeneracy; enumerate in the
  multi-solution pipeline. Verdict: `CONFIRMED`.
- **Only seed 33** (excl count = 0 and control is VALID/PARTIAL) →
  seed-33-specific coincidence; keep `omega-sign-degeneracy` at
  `confidence: low`. Verdict: `REFUTED`.
- **0 seeds including a control miss** → search method failed; the
  hypothesis is untestable with this pipeline. Verdict: `REFUTED`, follow
  up with wider L-BFGS options or DE for compensating q0.
- **1–2 non-control seeds** → Verdict: `MIXED`; investigate which seeds
  succeed and whether they share a geometric regime.

## Pre-run compute budget

Per-seed, serial:
- Stage A (SO(3) grid): ~60000 quaternions × 500 epochs, surrogate
  batched per worker. Analogue: m119v2 Stage C on full 60000×500 is
  ~50-80 s with POOL=8. Expect **~50-80 s**.
- Stage B (L-BFGS-B polish): 20 starts × ~50-200 cost evals each;
  each surrogate cost is ~10-30 ms on 500 epochs. Serial. Expect
  **~30-60 s**.
- Stage C (hi-fi validation): up to 5 basins × ~50 s hi-fi each, 4-way
  parallel. Expect **~60-80 s** (ceiling = 5 × 50 / 4 ≈ 65 s).
- **Per-seed total: ~140-220 s (2.3–3.7 min)**.

Batch: 11 seeds × ~3 min = **~30-40 min wall** serial-over-seeds.

Resource footprint:
- Stage A Pool(8): 8 worker processes, each loading surrogate model (~a
  few hundred MB); fits easily in 16-core / 64 GB host.
- Stage C Pool(4): 4 workers doing full hi-fi ray tracing — each worker
  holds its own `ExperimentContext` via fork-inheritance.

## Kill criteria

- **Per-seed hard cap**: 10 min wall. If any seed exceeds 10 min (≈ 2.7×
  expected ceiling), abort and investigate. Common causes: surrogate
  model load stalling, hi-fi Pool deadlock from fork+threading
  interaction, propagator numerical issues (giant omega in L-BFGS step).
- **Stage A pool watch**: if `pool done` takes > 3 min for any seed,
  kill — likely indicates a worker crash without error propagation.
- **Batch hard cap**: 60 min total. Beyond this, stop and re-scope.
- **No sleep-wait**: monitor `run.log` from the per-seed dir directly.
  Checkpoints (stage_a_grid.npz / stage_b_polish.npz / stage_c_hifi.npz)
  make resumption cheap.

## Files

- Script: `notebooks/inversion/12_brightness_surface/m127_flipped_omega_search.py`
- Output base: `data/results/inversion_diagnostics/m127_flipped_omega/`
- Per seed: `seed_NNN/stage_a_grid.npz`, `stage_b_polish.npz`,
  `stage_c_hifi.npz`, `result.json`, `run.log`
- Batch: `batch_summary.json`
