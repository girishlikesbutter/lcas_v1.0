---
title: "m103 Seed 028 Geo Hang — high-phase Pool(24) deadlock investigation"
type: experiment
sources:
  - "raw/inversion_diagnostics/m103_hybrid_m048/seed_028/pipeline.log"
  - "raw/inversion_diagnostics/m103_hybrid_m048/seed_028/multi_phi_ckpt.npz"
  - "raw/inversion_diagnostics/m103_hybrid_m048/seed_028/retry_geo_serial.log"
related:
  - "[[m103_hybrid]]"
  - "[[alignment-cost]]"
  - "[[phase_B_m048_cohort]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
created: 2026-04-17
updated: 2026-04-17
confidence: high
---

# m103 Seed 028 Geo Hang Investigation

## Context

Seed 028 of the m048 trajectory set (median phase 87.3°, range 80.2°–94.4°, 10 spec peaks, ω = 1.438 dps) was the high-phase pick in the original [[phase_B_m048_cohort|Phase 2 pilot]]. Both attempts to run `invert.py --seed 28 --traj-source m048` resulted in [[m103_hybrid]] hanging in Step 4 (Geo refinement) for 5+ hours until user intervention. This page documents the investigation.

## Hang pattern (observed)

```
T=0–5 min:   m103 Steps 1-3.5 complete normally
             Grid 112 s, Lo-fi 15 s, NM 158 s, Multi-phi 0.3 s
             multi_phi_ckpt.npz written at T=5min with 26 candidates
T=5–10 min:  Pool(24) Step 4 launches — all 24 workers at ~80% CPU
T=10-15 min: Most candidates finish, CPU drops; 1-3 stragglers remain
T=15+ min:   All 24 workers drop to 0% CPU, blocked in `futex_do_wait`
             Parent process stuck in pool.map() — never returns
```

The parent never emits `Geo done in Xs`. The hang persisted for 5+ hours on both attempts before manual kill.

## Inferred mechanism

L-BFGS-B on m103's `geometric_cost` at seed 028's alignment-cost surface enters a pathological state:

- **Cost surface is near-flat at high phase** (see [[alignment-cost]] "High-phase flatness" subsection). The `geometric_cost` function is `Σ (1 − max(n·PAB))²` over 9 constraint epochs. At 87° phase with 10 spec peaks rotating through body frame during a 1-hour window, many distinct (q0, ω) pairs produce near-equivalent costs.
- **Line search burns unbounded function evaluations.** Each L-BFGS-B iteration spawns 6 finite-difference evaluations for the gradient plus variable-length line-search evaluations. On a flat surface the line search can request 20+ evals per iteration. The `maxfun=1500` cap in `refine_one_geo` should eventually terminate, but before it fires the worker may hit a pathological state (NaN propagation through quaternion renormalisation, failed pickle when result crosses the Pool boundary, or a Fortran-backend deadlock).
- **Pool deadlock follows worker death.** `multiprocessing.Pool.map` has no timeout. A single worker that fails to return its result puts the parent's `map` call into permanent wait.

## Serial rescue attempts

Two serial rescues attempted, both failed for different reasons:

### Attempt 1: `retry_geo_serial.py --maxfun 500 --maxiter 50` (2026-04-17 morning)

Buffered stdout (through `| tee` pipe) meant zero output for 25 minutes despite 100% CPU activity (1579 CPU-seconds burned). Killed at 25 min wall after the user flagged the unacceptable delay.

Lesson: **always run with `python3 -u`** for pipeline scripts when stdout is piped. Added to future-script conventions.

### Attempt 2: `retry_geo_serial.py --maxfun 100 --maxiter 20` (2026-04-17 afternoon)

With `python3 -u`:
```
cand 0:  30.4s  geo=1.482959  q0=133.42  w=66.42  [ok]
cand 1:  36.3s  geo=1.982788  q0=163.90  w=73.62  [ok]
cand 2:  20.0s  geo=1.421444  q0=167.10  w=38.91  [ok]
(killed at cand 2 — candidates taking too long, all with q0 > 130°)
```

Three candidates completed in ~87 s with ALL having q0_err > 130° and ω_err > 38°. Rate was ~29 s/candidate, projecting 11 minutes for all 26.

**The real diagnosis came from the upstream pipeline.log**, not from more geo attempts. See below.

## Root-cause diagnosis from pipeline.log

NM top-20 for seed 028 (from `m103_hybrid_m048/seed_028/pipeline.log`):

```
--- Step 3: NM (300) ---
NM done in 157.9s
  Deduped: 20 (from 89 unique, capped at 20)
  w#1  gcost=3.05e-02 | q0=133.4  w=68.2
  w#2  gcost=3.35e-02 | q0=163.9  w=79.8
  w#3  gcost=3.70e-02 | q0=167.1  w=41.0
  ...
  w#19 gcost=6.57e-02 | q0=167.2  w=30.8  <-- best ω-error in pool
  w#20 gcost=6.67e-02 | q0=179.4  w=84.8
```

**The best ω-direction error in the entire NM top-20 is 30.8°**, at rank #19 (excluded by geo_cost ranking). Truth is NOT in the pool. Running geo refinement on this pool can only produce garbage — even successful L-BFGS-B convergence would refine around candidates that are already ~30° from truth.

The Pool(24) hang is a **symptom** of trying to refine a search-failure pool. The real failure is upstream at grid+NM, which is a property of the alignment cost function at high phase, not a property of the Pool orchestration.

## Implication: fixing the hang ≠ fixing the seed

A SERIAL_GEO fallback (per-candidate L-BFGS-B with hard timeouts) would make m103 return successfully on seed 028 instead of hanging. But the returned geo_ckpt.npz would contain 26 refined candidates all with ω_err > 30°. Downstream:
- m115 would take the top-3 ω by `w0_ref_errs` (ranked by distance to truth — oracle info m115 doesn't actually have; it ranks by `geo_cost` instead, which is even less reliable)
- m115's 3-DOF q0 DE would converge to whatever basin sits near the wrong ω
- Hi-fi MSE would be in the FAIL range regardless

For seed 028 to actually invert correctly, the upstream cost function has to change. See [[upstream-redesign-6dof-surrogate-de]].

## Comparison with seed 069 (67.7° phase)

Seed 069 is the intermediate-phase analogue. Its NM top-20 has best ω-error = 28.8° (rank #8) — same upstream failure, slightly less severe. Seed 069's geo stage had started executing when killed pre-completion based on this diagnosis; the Pool(24) hang may or may not have manifested at 68° phase. Either way the downstream outcome would have been similar to seed 028: FAIL.

## Verdict

Seed 028 is not a special-case bug to patch. It's an instance of the **high-phase flatness** failure mode, which also manifests at seed 069 (68°) and likely every m048 seed with median phase ≥ ~65°. The Pool(24) hang is the most alarming symptom but not the root cause.

Implications:
- Retrying with SERIAL_GEO + timeouts would unblock m103 but produce FAIL results.
- The 2+ hours of wall time spent on the two rescue attempts were a wash — the diagnosis came from the NM log dump that was already on disk.
- **Lesson (logged for future sessions):** when a pipeline stage hangs, check the UPSTREAM log output BEFORE retrying with a more defensive execution strategy. The hang may be downstream of an upstream search failure.
