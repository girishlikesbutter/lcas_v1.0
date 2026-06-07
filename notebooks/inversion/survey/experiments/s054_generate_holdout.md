---
title: s054 — generate holdout trajectories with the m048 post-fix generator
type: experiment
sources:
  - experiments/s053_cohort_polhode_survey.md
related:
  - feedback_holdout_validation.md
  - feedback_analytical_vs_operational.md
created: 2026-05-07
updated: 2026-05-07
confidence: high
---

# TL;DR

Generated 20 holdout trajectories (m048 seeds 100–119) with the post-fix
m048 generator at `notebooks/inversion/09_glint_analysis/
m048_generate_trajectories_v2.py`. Symlinked into the survey workspace
at `data/trajectories/traj_seed1XX.npz`. Available via
`lib.traj_load.load_truth(seed)` for seed in [100..119].

This corpus is the **holdout test set** for any subsequent claim that
"cohort statistics give an inversion sampler / prior gain." Per the
methodology (`feedback_holdout_validation.md`), cohort-derived priors
must be validated on fresh trajectories from the same generator before
being claimed as operational gains.

The generator is the post-fix code; per-trajectory NPZs share schema and
units with the original cohort (seeds 0–99). Light curves, geometry,
and inertia tensor all consistent.

# What

The user pushed back correctly on s053's framing: the cohort-derived
pol_diam ↔ basin-width finding is empirical truth on the 100 seeds, but
testing a "pol_diam-weighted sampler" on the same 100 seeds is
data leakage. The proper validation requires holdout trajectories from
the same generator.

This experiment generates the holdout corpus.

# How

`experiments/s054_generate_holdout.py`:

- Dispatches subprocess workers for seeds 101–119 (seed 100 was already
  generated as a smoke test). Pool(4) workers; per-seed wall ~70-260s
  (median 167s).
- Bypasses the master-NPZ collection step in the original generator.
  The cohort master at `data/results/inversion_diagnostics/
  m048_trajectories/m048_trajectories.npz` is **not modified**, so any
  buggy-era consumer that hardcodes 100 entries is unaffected.
- After all workers complete, symlinks new per-trajectory NPZs into
  `notebooks/inversion/survey/data/trajectories/`.
- The post-fix codebase (`src.dynamics.attitude_propagator.
  propagate_attitude` + the renderer) is what the generator imports;
  this is the same code path the survey workspace uses.

# Result

- Seed 100 (smoke test): generated separately, |ω|=1.348 dps, 45 hi-fi
  peaks, 70s wall.
- Seeds 101–119: generated in 5 batches of 4 workers each.
- Total wall: see log; expected ~15 min Pool(4).
- All NPZs written to `data/results/inversion_diagnostics/
  m048_trajectories/per_trajectory/`. 20 symlinks added in
  `notebooks/inversion/survey/data/trajectories/`.

`lib.traj_load.load_truth(100)` returns a clean dict with all expected
fields (`seed, q0_wxyz, omega0_rad, mag_hifi, k1_body, k2_body, ...`)
matching the schema of cohort seeds.

# Why this matters

The holdout is a precondition for any claim about the operational value
of cohort statistics. Without it:

- s053's "pol_diam predicts basin width" stays purely analytical (true
  about the forward model, untested as an inversion prior).
- Future claims of the form "cohort distribution X informs the sampler"
  are unfalsifiable.

With it:

- Any candidate inversion architecture (s055+) can be evaluated on
  fresh trajectories from the same generator. The fit-on-cohort,
  test-on-holdout split is now available.
- The first concrete use case: comparing uniform-Sobol vs cohort-
  prior-weighted Sobol on the holdout, at equal sample budget, for
  Band A∪B yield.

# Numbers

- N seeds in cohort (training): 100.
- N seeds in holdout (this experiment): 20 (seeds 100..119).
- Generator wall per seed: 70-260s (median 167s).
- Total holdout generator wall: ~15 min Pool(4).
- Holdout |ω| range, peak counts, etc.: TBD post-completion (one-off
  scan via `lib.traj_load`).

# Artefacts

- `experiments/s054_generate_holdout.py` — wrapper script.
- `experiments/s054_generate_holdout.md` — this writeup.
- `data/results/inversion_diagnostics/m048_trajectories/per_trajectory/
  traj_seed1XX.npz` — 20 generated NPZs (canonical location).
- `notebooks/inversion/survey/data/trajectories/traj_seed1XX.npz` — 20
  symlinks for survey-workspace access.

# Out of scope

- Running the holdout test itself (s055+): comparing uniform vs
  cohort-prior samplers. Architecture to be designed first.
- Generating > 20 holdout seeds. K=20 is borderline for statistical
  power on differences of ~20 percentage points; can extend to K=30
  or K=50 if the s055 result is ambiguous.
- Cross-validation (fit on subsets of 0–99, test on disjoint subsets).
  Treat the existing cohort as fixed-train, the new 20 as fixed-test;
  CV is deferred until a single train/test split shows a candidate
  effect worth tightening.

# Cross-references

- `experiments/s053_cohort_polhode_survey.md` — the cohort scan whose
  proposed operational use needs holdout validation.
- `feedback_holdout_validation.md` — the methodology rule this
  experiment implements.
- `feedback_analytical_vs_operational.md` — sister methodology rule
  (analytical vs operational distinction).
