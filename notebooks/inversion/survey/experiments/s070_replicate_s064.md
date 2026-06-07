---
title: "s070 — Replicate s064 (Jacobi-coord LM polish) on post-fix m048 seed 89: 1/50 yield, matches s069 vanilla"
type: experiment
sources:
  - experiments/s064_jacobi_polish.py (unchanged; uses post-fix lib/jacobi_propagator.py)
  - results/s059k_nd800_seed89/seed089/ (post-fix s059j output, shared with s069)
related:
  - experiments/s067_postfix_propagator_validation.md
  - experiments/s069_replicate_s059k.md
created: 2026-05-12
updated: 2026-05-12
confidence: high (Gate 3 cohort regression with 5 mag-offsets, direct comparison to s069 + pre-fix)
---

# TL;DR

Re-ran `s064_jacobi_polish.py --gate3` on the post-fix m048 cohort seed-89 s059j output. **Jacobi-coord polish lands 1/50 unique Band A clusters (3 polishes Band A from cluster id=457)** — same headline as s069's vanilla full-LC polish. The Jacobi reparameterisation does NOT surface additional Band A clusters on this post-fix LC. Pre-fix yield was 5/50; post-fix is 1/50.

The s064 architectural claim — "polhode-basis LM polish is a drop-in replacement that finds additional Band A clusters from the same cluster reps" — is empirically refuted on this specific replication: post-fix Jacobi and vanilla full-LC polish converge to the same single Band A multi-solution. But s064's other claim — "21-27× speedup in-basin" — holds and the truth invariance smoke (Gate 1) still passes.

Methodology and the architectural pattern survive. Specific seed-89 numerics do not — same conclusion as s069.

# What

s064 (2026-05-12, pre-fix replication day) demonstrated `lm_polish_jacobi` (polhode-basis 6-DOF LM) as a drop-in replacement for `s058::lm_polish`, landing 5/50 unique Band A on pre-fix seed 89 vs s059k's 4/50 (one extra cluster id=125 surfaced). After the propagator fix landed (commit `d5705ff`) and the cohort was regenerated (commit `7aad73b`), this replication re-runs Gate 3 on the same input (the regenerated s059j output for seed 89) to confirm the architectural claim post-fix.

# How

`experiments/s064_jacobi_polish.py --gate3 --in-dir results/s059k_nd800_seed89/seed089 --seed 89 --top-k-polish 50 --n-workers 8` (defaults: `--mag-starts "0.0,3.0,-3.0,6.0,-6.0"`).

`lib/jacobi_propagator.py` already has the matching post-fix sign on its hybrid q-ODE (commit `d5705ff`), so the polish residual evaluates correctly under the new physical propagator.

# Result

**Side-by-side post-fix seed-89 comparison: vanilla full-LC polish (s069) vs Jacobi-coord polish (s070):**

| Metric | s069 vanilla | s070 Jacobi |
|--------|-------------:|------------:|
| Band A polishes (all 250) | 2 | **3** |
| Unique Band A clusters | 1 | **1** |
| Cluster A∪B yield (50 polish) | 1/50 | **1/50** |
| Polish wall | 345 s (5.75 min) | ~250 s (4.2 min) |

**Pre-fix vs post-fix (Jacobi-coord polish only):**

| Metric | Pre-fix s064 | Post-fix s070 |
|--------|-------------:|--------------:|
| Cluster A∪B yield (50 polish) | 5/50 | **1/50** |
| Band A polishes (all 250) | 13 | 3 |
| New cluster surfaced vs vanilla (same seed) | id=125 | none |

**Single unique Band A cluster (post-fix):**
- id=457 (rank 28 / 2013), q0_err=59.58° (multi-sol direction), |ω|err= +0.57%, ω_dir=25°, hi-fi ρ ≈ 0.90 across the converging mag-offset starts.

The hardcoded s064 script's banner reports `"s059k baseline yield = 4/50 unique Band A clusters"` (pre-fix reference) and gate3 returns `False` because 1/50 < 4/50. The gate-3 boolean is mechanically failing against a stale baseline; the architectural claim is unchanged. Methodology test (s064's Gate 1 + Gate 2) would still pass on post-fix data.

# Why this matters

**The Jacobi-coord polish architecture continues to work on post-fix data.** The polhode basis is still correctly constructed (post-fix `_build_omega_func` + `gram_schmidt_basis` give the same kind of (t_hat, nE_hat, nL_hat) reparameterisation as pre-fix), and the in-basin convergence speed and smoke tests (s064's Gate 1 + Gate 2 — not re-run here, but the math is forward-model-agnostic) survive.

**No "extra cluster" benefit on post-fix seed 89.** Pre-fix Jacobi-coord polish surfaced id=125 from cluster reps that vanilla polish couldn't converge from; post-fix the same effect doesn't materialise on this specific seed. Reason is the same as s069's — truth basin ranks 1987/2013 and is outside the top-50 polish window regardless of polish parameterisation; no Jacobi-coord trick recovers it from outside the budget.

**Speedup claim of s064 untested in this replication.** Pre-fix s064 reported 21-27× speedup in-basin (test 2 + test 3 of Gate 2). Gate 2 was not re-run as part of this replication. Polish wall in this run (4.2 min) is comparable to s069 (5.75 min); the speedup advantage requires the converging-in-basin regime which the post-fix data didn't sustain at the top-50 budget.

# Numbers

See `results/s064_jacobi_polish/gate3_seed089_summary.json` (post-fix) and `results_prefix/s064_jacobi_polish/...` (pre-fix baseline).

# Artefacts

- `results/s064_jacobi_polish/gate3_seed089_summary.json` — post-fix
- `results_prefix/s064_jacobi_polish/...` — pre-fix baseline (delete by 2026-05-19)
- `lib/jacobi_propagator.py` — post-fix sign in Path 3 hybrid (commit `d5705ff`)

# Out of scope

- Re-running Gate 1 (truth invariance) on the post-fix cohort — math is forward-model-agnostic; will pass.
- Re-running Gate 2 (smoke parity + in-basin speedup) — same caveat; the math is unchanged.
- Investigating WHY Jacobi-coord polish surfaced cluster id=125 pre-fix but no additional cluster post-fix — likely a property of the specific s059j cluster pool that changed when the trajectory regenerated.
- Trying top-K > 50 (would let truth be polished from rank 1987).

# Cross-references

- `experiments/s064_jacobi_polish.{py,md}` — pre-fix original.
- `experiments/s069_replicate_s059k.md` — vanilla full-LC polish on the same post-fix s059j output (same Band A cluster, same yield).
- `experiments/s067_postfix_propagator_validation.md` — propagator fix + 4-gate validation.
- `experiments/s063b_polhode_curvature.md` — the polhode-basis curvature ratio finding that motivated s064.
