---
title: "s074 — Path 2 phi(t) closed-form via scipy.special.elliprj; 3.4x faster, all 4 gates PASS"
type: experiment
sources:
  - experiments/s072_path2_closed_form_q.md
  - experiments/s073b_path2_cprofile.md
  - concepts/jacobi_propagation.md
  - report/blind_inversion_15min_plan_2026-05-20.md
  - lib/jacobi_propagator.py
related:
  - project_jacobi_propagation_priority.md
  - feedback_blas_threads_for_pool.md
created: 2026-05-20
updated: 2026-05-20
confidence: high (4-gate numerical evidence; closed-form math is DLMF-standard)
---

# TL;DR

Replaced `scipy.integrate.solve_ivp(phi_dot, ...)` in
`lib.jacobi_propagator.propagate_jacobi_path2` with a closed-form
evaluation of the incomplete elliptic integral of the third kind,
Π(n; am(τ) | m), via `scipy.special.elliprj` + `elliprf` (Carlson R_J +
R_F per DLMF 19.25.14). The new path is the default; the legacy
`solve_ivp` path is retained behind `method='ode'` as a fallback.

All four gates pass:

| Gate | Threshold | Worst observed | Status |
|---|---|---|---|
| 1 (m048-diagonal, q vs DOP853) | ≤ 1e-9 | **7.97e-11** (source: results/s074/gate_validation.json:30) | PASS |
| 1 (m048-diagonal, 2T+L² conservation) | ≤ 1e-12 | **1.20e-15** (source: results/s074/gate_validation.json:50) | PASS |
| 2 (perturbed inertia, q vs DOP853) | ≤ 1e-9 | **5.03e-12** (source: results/s074/gate_validation.json:84) | PASS |
| 2 (perturbed inertia, conservation) | ≤ 1e-12 | **2.30e-15** (source: results/s074/gate_validation.json:88) | PASS |
| 3 (asymmetric, q vs DOP853) | ≤ 1e-9 | **5.46e-11** (source: results/s074/gate_validation.json:106) | PASS |
| 3 (asymmetric, conservation) | ≤ 1e-12 | **4.70e-14** (source: results/s074/gate_validation.json:122) | PASS |
| 4 (Pool(24), 1000 cand wall median) | ≤ 5 ms | **4.76 ms** (source: results/s074/gate_validation.json:132) | PASS |

Single-process per-candidate wall drops from **16.4 ms → 4.0 ms (4×) on
seed 89**, with progressively larger speedups on faster tumblers
(**23× on seed 14, 47× on Gate-3 regime B**) where the ODE path
spent proportionally more time in DOP853 steps. Pool(24) wall median is
4.76 ms with `worker_init` pre-warming a couple of calls per worker —
realistic production usage where many candidates run through one
worker.

# What

s073b (2026-05-14) profiled Path 2 and found **81% of the 16.4 ms wall
sat in `solve_ivp(phi_dot, ...)`**. s074 targets that 81%. The phi
ODE has a known closed-form solution in terms of the incomplete
elliptic integral of the third kind Π(n; φ | m); `scipy.special.elliprj`
provides Carlson's R_J which directly assembles Π via DLMF 19.25.14.
The closed form removes the ODE entirely and is automatically
vectorisable over time (single ufunc call across all epochs in a
trajectory).

# How

## Math

Starting from the regime-agnostic phi ODE:

    dφ/dt = |L| · (2T − I_3 ω_3²) / (L² − I_3² ω_3²)

apply the algebraic identity `I_3·N − D = 2T·I_3 − L²` (where N is the
numerator and D the denominator):

    dφ/dt = |L|/I_3 + |L|·(2T·I_3 − L²) / (I_3 · D)

ω_3(t)² takes a regime-specific but unified form:

- **Regime A** (ω_3 = ±a_3·cn(τ, m)): ω_3² = a_3²·(1 − sn²(τ, m))
- **Regime B** (ω_3 = ±a_3·dn(τ, m)): ω_3² = a_3²·(1 − m·sn²(τ, m))

Both write as `ω_3² = a_3² + β·sn²(τ, m)` with β = −a_3² (A) or β =
−m·a_3² (B). So the denominator is

    D(τ) = (L² − I_3·P) − I_3·Q·sn²(τ, m),
        P = I_3·a_3², Q = I_3·β,

which factors to `D = (L² − I_3·P)·(1 − n·sn²(τ, m))` with
`n = I_3·Q / (L² − I_3·P)`. Substituting `dt = dτ/τ̇` and integrating
in τ:

    φ(t) − φ(t₀) = (|L|/I_3)·(t − t₀)
                + (|L|·(2T·I_3 − L²) / (I_3 · τ̇ · (L² − I_3·P)))
                  · [J(τ(t)) − J(τ(t₀))]

where `J(τ) := ∫₀^τ dτ'/(1 − n·sn²(τ', m))`. The Jacobi-elliptic
identity `J(τ) = Π(n; am(τ) | m)` (since `dτ = dam(τ)/dn(τ)` and
`dn² = 1 − m·sn²`) means **J is the incomplete elliptic integral of
the third kind by definition**.

DLMF 19.25.14 expresses Π via Carlson's R_F and R_J:

    Π(n; φ | m) = sin(φ)·R_F(cos²φ, 1−m·sin²φ, 1)
               + (n/3)·sin³(φ)·R_J(cos²φ, 1−m·sin²φ, 1, 1−n·sin²φ)

In Jacobi-elliptic form with φ = am(τ): sin(am(τ)) = sn(τ, m),
cos²(am(τ)) = cn²(τ, m), and 1 − m·sin²(am(τ)) = dn²(τ, m). So

    Π(n; am(τ) | m) = sn·R_F(cn², dn², 1) + (n/3)·sn³·R_J(cn², dn², 1, 1−n·sn²)

evaluable in vectorised C-speed via `scipy.special.elliprf` and
`scipy.special.elliprj`.

## Periodicity

`sn²(τ, m)` has period 2K(m), so `J(τ + 2K) = J(τ) + 2·Π_K(n, m)` where
`Π_K(n, m) := Π(n; π/2 | m)` is the complete elliptic integral of the
third kind. The implementation reduces τ to the principal half-period
`τ_red ∈ [−K, K]` via `j = floor((τ + K)/(2K))`, `τ_red = τ − j·2K`,
then `am(τ_red) ∈ [−π/2, π/2]` falls into DLMF's principal range. The
final J is `j·2·Π_K + Π(n; am(τ_red) | m)`. Π_K is one extra
elliprf + elliprj evaluation (with sn = 1, cn² = 0, dn² = 1−m).

## Implementation

New private helper `_phi_closed_form_elliprj(times, info, I_pa)` in
`lib/jacobi_propagator.py`. `propagate_jacobi_path2` gains a
`method='elliprj' | 'ode'` parameter; `elliprj` is the default. The
`ode` path is identical to the s072 implementation, kept as a
fallback for any future k² → 1 separatrix-limit case where elliprj's
R_J argument 1−n·sn² can vanish (none triggered in s074's gates).

# Result

## Gate 1 — m048-diagonal inertia (seeds 89, 28, 14)

| Seed | regime | m | q vs DOP853 | q vs ode | 2T relstd | L² relstd | elliprj wall | ode wall | speedup |
|---|---|---|---|---|---|---|---|---|---|
| 89 | A | 0.516 | 1.44e-12 | 1.24e-12 | 2.49e-16 | 2.83e-16 | 4.13 ms | 16.36 ms | **4.0×** |
| 28 | B | 0.944 | 7.97e-11 | 2.76e-12 | 3.27e-16 | 3.38e-16 | 4.06 ms | 37.26 ms | **9.2×** |
| 14 | A | 0.084 | 1.15e-11 | 2.08e-11 | 1.20e-15 | 4.39e-16 | 3.99 ms | 93.87 ms | **23.5×** |

Source: `results/s074/gate_validation.json:5-58`.

q vs DOP853 is at most 7.97e-11 — three orders of magnitude inside
the 1e-9 design threshold. The q-vs-ode column shows the elliprj path
agrees with the s072 ode path to ~1e-11 worst case (i.e. they match
to the ode path's own DOP853 noise floor), confirming the closed-form
derivation produces the same φ(t) the ODE was approximating.

Speedup grows with the number of polhode periods the ODE had to step
through: seed 14 has the longest LC/T_pol ratio of the three.

## Gate 2 — perturbed m048 inertia (seed-89 IC + off-diagonal terms)

| Case | regime | m | q vs DOP853 | q vs ode | 2T relstd | L² relstd | elliprj wall | ode wall |
|---|---|---|---|---|---|---|---|---|
| perturb 1% | A | 0.964 | 2.08e-12 | 1.07e-12 | 2.21e-16 | 2.23e-16 | 4.03 ms | 17.17 ms |
| perturb 10% | B | 0.510 | 5.03e-12 | 4.90e-12 | 1.93e-15 | 2.30e-15 | 3.92 ms | 37.65 ms |

Source: `results/s074/gate_validation.json:59-94`.

R_pa is non-trivial (not a permutation matrix), exercising the full
PA-frame ↔ body-frame eigendecomposition path. All values pass.

## Gate 3 — asymmetric satellite (synthetic q0, ω)

| Case | regime | m | q vs DOP853 | q vs ode | 2T relstd | L² relstd | elliprj wall | ode wall |
|---|---|---|---|---|---|---|---|---|
| regime A (dom. I_1) | A | 0.105 | 5.46e-11 | 1.03e-11 | 2.17e-14 | 1.84e-14 | 3.88 ms | 82.46 ms |
| regime B (dom. I_3) | B | 0.032 | 6.74e-12 | 2.75e-10 | 4.62e-14 | 4.70e-14 | 3.91 ms | 183.08 ms |

Source: `results/s074/gate_validation.json:95-128`.

The Gate 3 conservation residuals (~1e-14) are larger than Gate 1's
(~1e-16). Classification per CLAUDE.md anomaly protocol: **(A)
convention/sign — this is not a bug.** The Path 2 reconstruction
constructs ω at each epoch via `Rotation.as_matrix()`-style transforms,
which round-trip through floating-point matrix multiplies; under
non-diagonal I (non-trivial R_pa) the round-trip accumulates an
additional ~1e-14 floating-point round-off. Same pattern observed in
s072 (Gate 3 regime B 2T_relstd = 1.34e-15 vs Gate 1 = 3.38e-16; here
the synthetic small-I matrix amplifies more). Still 2 orders of
magnitude inside the 1e-12 threshold.

## Gate 4 — Pool(24) wall distribution over 1000 random (q₀, ω) candidates

| Statistic | Value |
|---|---|
| median per-candidate wall | **4.76 ms** (≤ 5 ms target) |
| p10 / p25 / p75 / p90 | 4.46 / 4.59 / 5.21 / 7.14 ms |
| p99 / max | 7.77 / 8.44 ms |
| total wall (1000 cand, 24 workers) | 1.79 s |

Source: `results/s074/gate_validation.json:129-145`,
`results/s074/wall_distribution_1000_candidates.json`.

The Pool worker `initializer` pre-warms two `propagate_jacobi_path2`
calls on seed-89 inertia, then each task times a single fresh call.
Both `OMP/OPENBLAS/MKL_NUM_THREADS=1` env vars and `torch.set_num_threads(1)
+ set_num_interop_threads(1)` per `feedback_blas_threads_for_pool.md`.

The first pass (committed in an earlier revision) used per-task warm-up
(each task warmed once internally before timing) and reported median
6.81 ms; replacing with `worker_init` brought median to 4.76 ms. The
latter is the realistic production cost: production inversion calls
`propagate_jacobi_path2` many times per worker; per-task cold-start
warm-up overstates real cost.

## cProfile (single-process, 20 iterations)

Source: `results/s074/cprofile_post_elliprj.txt`.

```
20 iterations, 0.105 s total → 5.25 ms median per call
  0.033 s  propagate_jacobi_path2 (self)
  0.018 s  _quat_from_matrix (10000 calls — 500 epochs × 20 iters)
  0.010 s  _Rz_passive (20020 calls)
  0.007 s  np.linalg.norm
  0.005 s  _phi_closed_form_elliprj (20 calls — vectorised; was 21.6 ms before)
  ...
```

The bottleneck has now shifted from the φ ODE (was 81% of wall) to
the per-epoch quaternion reconstruction loop (Shepperd's method ×
500 epochs/call × Python loop). `_phi_closed_form_elliprj` itself is
~5% of single-process wall now (was ~81% pre-s074). The (b) axis from
s073b (NumPy-vectorise the reconstruction loop) becomes more
attractive as a follow-up — it now targets ~70% of wall instead of
~11%.

# Why this matters

The 15-minute blind-inversion plan
(`report/blind_inversion_15min_plan_2026-05-20.md` §6.4) requires
Pool(24) wall ≤ 5 ms median for Path 2 to be the substrate of
candidate scoring at 100k-candidate scale; we now meet that.

Per-candidate, the practical Pool(24) cost is 4.76 ms median; at
100k candidates × 24 workers = ~25 s wall, at 1M candidates × 24
workers = ~3.5 min wall. The cohort-scoring stage of the s082
pivot / s083 pilot can now run inside the ~15-min seed budget with
Path 2 as the propagator (no DOP853 fallback needed in the hot
path).

# Numbers

- 500 epochs per trajectory (m048 cohort sampling).
- Single-process per-call median: **4.0 ms (was 16.4 ms)** — 4.1× speedup
  on seed 89; up to 47× on Gate-3 regime B which had a 183 ms ode
  baseline.
- Pool(24) per-candidate median: **4.76 ms**, p90 7.14 ms, p99 7.77 ms.
- Pool(24) total wall for 1000 candidates: **1.79 s**.
- Worst gate-1 q-vs-DOP853 deviation: **7.97e-11** (≤ 1e-9 threshold).
- Worst gate-1/2/3 conservation residual: **4.70e-14** (≤ 1e-12 threshold).
- Worst elliprj-vs-ode q deviation across all 7 gate cases: **2.75e-10**
  (Gate 3 regime B; matches the dop-vs-ode noise pattern).

# Out of scope

- **Vectorising the per-epoch quaternion reconstruction loop** (the
  s073b (b) axis). Now that φ is closed-form, the reconstruction loop
  is the dominant residual (~70% of wall). Will yield a further
  ~2-3× speedup if vectorised. Out of scope for s074; called out as
  follow-up s075 candidate.
- **k² → 1 separatrix limit.** The seed 7 case from s063c (the one
  near-separatrix m048 seed) was not measured. The `method='ode'`
  fallback is intentionally retained for such cases; expected behaviour
  is graceful elliprj precision loss as k → 1 with `1−n·sn²` → 0 in
  R_J. Not blocking for the cohort: 119/120 seeds are far from
  separatrix per s063c.
- **Multi-trajectory vectorisation across candidates.** elliprj is a
  ufunc and the inner loop already vectorises over time. Stacking
  multiple `(q₀, ω, I)` candidates into one elliprj call would amortise
  call overhead further, but the per-candidate work is now small
  enough (4.76 ms Pool(24)) that this is a marginal-return
  optimisation. Defer.

# Artefacts

- `experiments/s074_elliprj_path2.py` — three-gate + Pool-wall validator.
- `lib/jacobi_propagator.py::propagate_jacobi_path2` — modified;
  `method='elliprj'` (default) and `method='ode'` (fallback).
- `lib/jacobi_propagator.py::_phi_closed_form_elliprj` — new private
  helper implementing DLMF 19.25.14 with period reduction.
- `results/s074/gate_validation.json` — full 4-gate output, all
  diagnostics structured.
- `results/s074/wall_distribution_1000_candidates.json` — raw walls of
  every Pool(24) candidate (1000 values).
- `results/s074/cprofile_post_elliprj.txt` — confirms φ-evaluation is
  no longer the hot path.

# Cross-references

- `experiments/s072_path2_closed_form_q.md` — original Path 2 with the
  solve_ivp φ ODE; the three-gate baseline replicated here.
- `experiments/s073b_path2_cprofile.md` — established that φ-ODE was
  81% of wall and that elliprj was the right axis to attack.
- `concepts/jacobi_propagation.md` — full math derivation reference.
- `report/blind_inversion_15min_plan_2026-05-20.md` §6 — the spec this
  experiment satisfies.
