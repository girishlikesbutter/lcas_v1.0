---
title: "s073b — Path 2 cProfile: solve_ivp(phi_dot) is 81% of wall; elliprj replacement is the right axis"
type: experiment
sources:
  - experiments/s072_path2_closed_form_q.md
  - lib/jacobi_propagator.py
related:
  - project_jacobi_propagation_priority.md
created: 2026-05-14
updated: 2026-05-14
confidence: high (single-candidate profile, clear hot-path)
---

# TL;DR

Single-candidate × 500-epoch run of `propagate_jacobi_path2` on seed 89
truth, profiled with cProfile (5 warmed runs).
**Median wall: 16.4 ms.** Cumulative-time split:

| Component | Cumulative time | Share of wall |
|---|---:|---:|
| `propagate_jacobi_path2` (total) | 26.7 ms | 100% |
| `solve_ivp(phi_dot, ...)` (the 1-D scalar phi ODE) | **21.6 ms** | **81%** |
| `omega_func_pa(t)` called inside `phi_dot` (`omega_at_t`) | 6.2 ms self | 23% self |
| Per-epoch quaternion reconstruction loop | < 3 ms | < 11% |
| Setup (eigendecomp, omega_func build, etc.) | ~2 ms | ~8% |

**Verdict: elliprj replacement is the right next axis.** Replacing
`solve_ivp(phi_dot, ...)` with `scipy.special.elliprj` (closed-form Π
via Carlson's R_J) targets the 81% that's actually expensive and
removes the ODE entirely — automatically vectorisable across both
candidates and times. NumPy-vectorising the per-epoch reconstruction
loop would target <11% of wall; ceiling is too low and it doesn't
remove the ODE cost. Don't do (b).

# What

Post-s072 framing correction landed an open question: which Path 2
optimisation axis is the real bottleneck? Either
(a) `scipy.special.elliprj` replacing the φ-ODE (more durable,
estimated half-day write-time) or (b) NumPy vectorisation of the
per-epoch reconstruction loop (easier, lower ceiling).

s073b runs cProfile to pick objectively rather than guess.

# How

```python
propagate_jacobi_path2(q0_truth_seed89, omega0_truth_seed89,
                      I_m048, observation_times)  # N=500
```

5 warmed iterations, single process, BLAS pinned to 1 thread.
cProfile output sorted by cumulative time (where total time was
spent including children) and by self/tottime (where time was
spent in the function body alone). Both tables saved to
`results/s073b/profile.txt`.

# Result

```
=== s073b cProfile ===  seed=89, N=500
wall_s median over 5 runs: 0.0164  (min 0.0163, max 0.0171)

Top 5 by self-time:
  0.0062s  ncalls=  1548  jacobi_propagator.py:136 (omega_at_t)
  0.0028s  ncalls=  1548  _shape_base_impl.py:625 (column_stack)
  0.0025s  ncalls=   107  rk.py:14 (rk_step)
  0.0017s  ncalls=     1  jacobi_propagator.py:455 (propagate_jacobi_path2)
  0.0014s  ncalls=  6647  ~:0 (<built-in method numpy.array>)

Top 5 by cumulative time:
  0.0267s  ncalls=     1  jacobi_propagator.py:455 (propagate_jacobi_path2)
  0.0216s  ncalls=     1  ivp.py:160 (solve_ivp)
  0.0163s  ncalls=    97  base.py:179 (step)
  0.0162s  ncalls=    97  rk.py:111 (_step_impl)
  0.0149s  ncalls=   107  rk.py:14 (rk_step)
```

Reading the cumulative-time table top-down: `propagate_jacobi_path2`
itself is 26.7 ms total. Of that, `solve_ivp` is 21.6 ms (81%). Of
`solve_ivp`'s time, `rk_step` (the DOP853 step kernel) is 14.9 ms
(56% of total). Each rk_step costs ~0.14 ms. The 1548 calls to
`omega_at_t` are the closure being evaluated by `phi_dot` at every
RK stage (with DOP853 + dense output that's ~14 calls per accepted
step over 107 steps).

The per-epoch reconstruction loop (lines 545-556 of
`jacobi_propagator.py`) is not in the top 5 by self-time or cumtime.
Estimated cost from total − solve_ivp − setup = 26.7 − 21.6 − ~2 ≈ 3 ms,
about 11% of wall. The Shepperd quat-from-matrix runs 500 times and
each call is ~5 μs in pure Python; that's ~2.5 ms, consistent.

# Why this matters

Two optimisation axes were on the table per the s072 framing-correction
commit (`af8fcbe`):

  (a) Replace `solve_ivp(phi_dot, ...)` with `scipy.special.elliprj`
      (incomplete elliptic integral of the third kind via Carlson's
      R_J). Removes the ODE entirely. **Targets 81% of wall.**
      Automatically vectorisable across candidates and times because
      `elliprj` is a vectorised ufunc. Estimated write-time half-day.

  (b) NumPy-vectorise the per-epoch quaternion reconstruction loop
      (the `for i in range(N)` over R_z @ R_x @ R_z @ R_J→L plus
      Shepperd's `_quat_from_matrix`). Easier. **Targets ≤ 11% of wall.**
      Doesn't remove ODE cost; doesn't help across candidates.

(a) is the clear winner on cost-benefit and durability. (b) might be
worth doing later as a cleanup, but only after (a) lands — at which
point the reconstruction loop becomes a larger relative fraction and
the optimisation criterion may flip.

**Scaling estimate (rough):** at single-process × 16.4 ms/candidate,
100k candidates take ~27 minutes; 1M candidates take ~4.5 hours. With
Pool(8) parallelism: ~3.5 min and ~33 min respectively. If `elliprj`
replacement collapses Path 2 wall to ~3 ms (the non-ODE residual),
100k single-process drops to ~5 min and 1M to ~50 min. Cross-candidate
vectorisation (only available without the ODE) could compress further
by 10-100× depending on memory bandwidth. These are projections, not
measurements — must remeasure post-implementation.

# Numbers

- 500 epochs over 60 minutes (m048 cohort sampling).
- 5 warmed iterations, single-thread BLAS, no Pool.
- 16.4 ms median wall, 0.4% relative spread across 5 runs.
- 107 DOP853 accepted steps for `phi_dot` (rtol=1e-13, atol=1e-15).
- 1548 omega_at_t calls / 107 steps ≈ 14.5 evaluations per step
  (DOP853 has 12 stages + dense output, consistent).

# Out of scope

- Implementing the `elliprj` replacement. That's the recommended
  follow-up (call it s074 if pursued).
- Per-call overhead measurement at the multi-candidate scale. The
  setup costs (~2 ms × candidates) become visible at 1M scale and
  may motivate vectorising the candidate axis even before `elliprj`
  lands. Out of scope here; deferred to s074 design.
- Cohort representativity. We profile on seed 89; other seeds with
  different |ω| and polhode regime could have different ODE-step
  counts. The 14 RK calls/step factor is structural to DOP853 and
  unlikely to vary, but |ω| variations could change accepted-step
  count by ~2×. Not load-bearing for the (a)-vs-(b) decision.

# Artefacts

- `experiments/s073b_path2_cprofile.py` — script.
- `results/s073b/profile.txt` — full cProfile tables (cumulative + self).
- `results/s073b/summary.json` — top-5 hot lines structured.

# Cross-references

- `experiments/s072_path2_closed_form_q.md` — Path 2 derivation + 3-gate validation.
- `concepts/jacobi_propagation.md` — math; Π(n, φ | m) is the closed form.
- `lib/jacobi_propagator.py::propagate_jacobi_path2` — the function profiled.
