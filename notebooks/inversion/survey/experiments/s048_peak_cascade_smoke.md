---
title: s048 — peak cascade smoke test (v2 surrogate, single seed)
type: experiment
sources: [user_idea, lib.surrogate_eval, lib.twin, lib.filter_costs, src.dynamics.attitude_propagator]
related: [s019, s032, s042, s043, s044, s047, s048b]
created: 2026-05-07
updated: 2026-05-07
confidence: medium
---

# s048 — peak cascade smoke test (v2 surrogate, single seed)

## TL;DR

User idea: at each LC local minimum (peak in brightness), `(q, ω)` is constrained because the brightness function has a critical point there. Smoke test on seed 89:

- **Tier 0 PASS** — single-epoch C_t (peak 412, mag 5.38) survivors = 396/500k = 0.08%; tight as expected for very-bright peak.
- **Tier 1 FAIL** — joint `(q_peak, ω)` 3-epoch consistency yielded 6 triples but truth NOT among them (nearest 25° in q, 450% off in ω); structural failure mode = finite-difference ω inference noise-amplified by discrete-q sampling.
- **Tier 2 trivial** — only 6 candidates entering, 0 survived the next-peak Euler propagation filter (i.e. the 6 were noise).

The architecture is mechanically sound but the specific finite-diff ω construction breaks at low rotation rates because per-`dt` rotation (1.7° on seed 89) is comparable to discrete-sample resolution (~1.5°), so `ω = (q_b ⊗ q_a^{-1}) / dt` accumulates O(100%) noise. Re-test on a higher-|ω| seed before declaring the architecture dead.

## What

Run the user's peak-cascade construction:
1. Sample random q on SO(3); filter against measured mag at one peak (Tier 0).
2. Joint `(q_peak, ω)` sample with constant-ω propagation to ±dt; filter against 3-epoch mag signature (Tier 1).
3. For each Tier-1 survivor, propagate under true Euler dynamics (IS-901 inertia) to the next peak; filter against that peak's 3-epoch mag (Tier 2).
4. Iterate Tier 2 across all peaks (Tier 3 — not run since Tier 1 failed).

Conventions: scalar-first quaternions (w, x, y, z); LEFT-multiply for `q(t) = quat_exp(ω·dt) ⊗ q_0` per `src.dynamics.attitude_propagator`. Body-twin canonicalisation applied at Tier 1 via `lib.twin.canonical_batch`.

## How

- Seed 89 (forgiving, |ω|=0.24 dps, validated end-to-end in s035).
- Surrogate v2 throughout (was the wrong choice — see s048b for v1 retest).
- 500k random q on SO(3) via `scipy.spatial.transform.Rotation.random`.
- Tolerance 0.10 mag (2× noise floor).
- ω agreement threshold 5% relative.
- Propagator: `src.dynamics.attitude_propagator.propagate_attitude(mode='tumbling', inertia_tensor=lib.filter_costs.load_static_geometry()['inertia_tensor'])`.

Pass/fail thresholds written before run:
- Tier 0: `|C_peak|/N < 0.1` (single-epoch tight enough).
- Tier 1: per-peak selectivity `<10⁻³` AND truth survives.
- Tier 2: decimation `<0.5` AND truth survives.

## Result

| Tier | Outcome | Headline numbers |
|---|---|---|
| 0 | **PASS** | 396/500k = 0.08% at mag=5.38 peak |
| 1 | **FAIL** (truth) | 6 triples accepted; nearest survivor q-dist 25.41°, |Δω|/|ω| 450% |
| 2 | trivial | 0/6 survivors (all 6 were noise; truth not in input set) |

Per-epoch C_t (independent runs): C_a=343, C_peak=343, C_b=339 survivors. Truth-q nearest distance: a=1.43°, peak=0.53°, b=2.14° — truth WAS in each per-epoch C_t set.

Failure mechanism (post-hoc analysis):
- Discrete sampling resolution at 500k q on SO(3): ~1.5° nearest-neighbour at the C_t manifold.
- Per-dt rotation at seed 89's |ω|=0.24 dps: 1.7°.
- Finite-diff ω from `(q_a, q_peak)` and `(q_peak, q_b)` accumulates noise of order `(δ_a + δ_peak)/dt` ≈ `(1.43° + 0.53°) / 7.2s` ≈ 0.27°/s = **0.0047 rad/s** ≈ ω_truth (0.0042 rad/s).
- The 5% agreement threshold rejects all near-truth triples; only random-coincidence triples (with random ω) pass.

This is a **low-|ω| structural failure**, not a bug. At ω=1.0 dps the per-dt rotation is ~7°, dwarfing the sampling noise, and the ω-noise drops to ~20% of ω_truth — workable.

## Why this matters

- **Per-epoch C_t pre-image filtering works cleanly** — confirmed across 3 epochs that truth-q lies within the discrete sample at workable resolution. This is the load-bearing primitive that makes the user's reformulation possible (see s048b).
- **The cascade architecture is geometrically tight at high |ω|** — Tier 1 truth-survival should be revisited on seeds with |ω| ≥ 0.5 dps before declaring the cascade dead.
- **Surrogate version matters for budgeting** — this run used v2 (~56 μs/sample); v1 is ~12× faster (4.5 μs/sample) and is appropriate for the C_t sampling stage. See s048b for v1 timings.

## Numbers

| Quantity | Source | Value |
|---|---|---|
| N q samples | s048 | 500,000 |
| Tolerance | s048 | 0.10 mag |
| ω agreement | s048 | 5% relative |
| C_a survivors (ep 411, mag 5.40) | s048 | 343 / 500k |
| C_peak survivors (ep 412, mag 5.38) | s048 | 343 / 500k |
| C_b survivors (ep 413, mag 5.46) | s048 | 339 / 500k |
| Truth-q nearest in C_a | s048 | 1.43° |
| Truth-q nearest in C_peak | s048 | 0.53° |
| Truth-q nearest in C_b | s048 | 2.14° |
| Tier 1 accepted triples | s048 | 6 |
| Tier 1 nearest survivor to truth (q) | s048 | 25.41° |
| Tier 1 nearest survivor to truth (Δω/ω) | s048 | 450% |
| Tier 2 survivors | s048 | 0/6 |
| Total wall (v2) | s048 | 170 sec |

## Artefacts

- `experiments/s048_peak_cascade_smoke.py`
- `results/s048_peak_cascade_smoke/seed089/{tier0.npz, tier1.npz, tier2.npz, summary.json}`
- `results/s048_peak_cascade_smoke/run.log`

## Out of scope

- Re-test on higher-|ω| seed (deferred).
- Stationarity-constraint algorithm (∇B(q_peak)·ω = 0 as direct 2D ω constraint, no finite-diff needed).
- Multi-peak triple intersection (Tier 3) — never reached.

## Cross-references

- `experiments/s048b_per_epoch_spread_v1.md` — characterisation of per-epoch C_t across 30 epochs of seed 89, with v1 surrogate.
- `lib/twin.py` (s044) — canonical-hemisphere primitive used at Tier 1.
- `lib/filter_costs.py` `load_static_geometry()` — IS-901 inertia tensor source.
- `concepts/known_pathologies_to_revalidate.md` — the dead-class flag on LC-feature → ω priors (s007/s008).
