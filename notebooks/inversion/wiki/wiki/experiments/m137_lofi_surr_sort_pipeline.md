---
title: "m137 — M2 sort patch: surrogate-LC re-rank at m103 lofi stage (sort-only patch INSUFFICIENT)"
type: experiment
sources:
  - "notebooks/inversion/11_casadi_formulation/m103_hybrid.py"
  - "notebooks/inversion/audit_failure_seed_battery.py"
  - "data/results/inversion_diagnostics/failure_seed_battery_2026_04_28/audit_summary.json"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_047/{lofi,nm_prededup,multi_phi,geo}_ckpt.npz"
related:
  - "[[m135_alignment_cost_forensics_constrained_anchor]]"
  - "[[m136_kernel_consistency_failure]]"
  - "[[alignment-cost]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[attitude-level-set-disconnection]]"
created: 2026-04-28
updated: 2026-04-28
confidence: high
---

# m137 — M2 sort patch: surrogate full-LC MSE re-rank at m103 lofi stage

#dead-end (sort-only patch). The patch fires correctly and preserves truth at lofi rank 1, but downstream alignment-cost stages (NM polish + Geo L-BFGS-B refinement) destroy it. **Localised cause: Step 4 Geo, not Step 3 NM.**

## Hypothesis

m135 Finding 2 — that surrogate full-LC MSE re-ranks m103's lofi-300 pool to rank-1 at 2.76° from truth on seed 91, vs alignment cost's 174.83° — generalises across the random m048 failure cohort (47, 51, 79, 84, 89) and, when wired into m103 as the lofi sort key, rescues those seeds end-to-end through m115/m126.

## Phase 1 — offline audit on existing lofi_surr_ckpt.npz

Built `audit_failure_seed_battery.py`. Read out (a) `pool_min` ω-direction error (truth-closest in lofi-300), (b) rank-1 by alignment cost, (c) rank-1 by `lofi_mse` (full hi-fi LC MSE, no shadows), (d) rank-1 by surrogate full-LC MSE. (Path bug `parents[1]→parents[2]` fixed during the audit.)

| seed | pool_min | align_t1 | lofimse_t1 | surr_t1 | regime |
|---:|---:|---:|---:|---:|---|
| 47 | 1.60° | 32.02° | 1.60° | **1.60°** | rescuable by sort |
| 51 | 14.73° | 145.30° | 53.32° | 53.32° | **pool-FAIL** (truth not within 5°) |
| 79 | 16.30° | 62.13° | 70.40° | 70.40° | **pool-FAIL** |
| 84 | 4.10° | 97.61° | 44.91° | 44.91° | **ranking-FAIL** (pool OK, surr-MSE picks wrong cand) |
| 89 | 4.78° | 134.41° | 56.26° | 56.26° | **ranking-FAIL** |
| 91 | 0.96° | 174.83° | 174.00° | **2.76°** | reference: rescue confirms m135 Finding 2 |

**Within m115's 5° bridging radius: 1/5 by surrogate sort, 1/5 by lofi_mse sort, 0/5 by alignment cost.** Finding 2 generalises only marginally; on 4/5 failure seeds, surrogate-MSE has the same anti-truth basin signature as alignment cost (driven by the same dim-saturation / shadow-degeneracy LC ambiguities). For 51/79, alignment-cost upstream filter killed truth-near candidates — sort change can't fix.

## Phase 2 — pipeline test on seed 47 with `M103_LOFI_SORT=surr`

Patched `m103_hybrid.py` with env-var-gated surrogate full-LC MSE scoring + sort (default `align` preserves legacy behaviour). Re-ran `invert.py --seed 47 --traj-source m048 --force-m103`.

Trace through the pipeline:

| stage | rank-0 ω-direction error | mechanism |
|---|---:|---|
| Step 2b' (surrogate sort, NEW) | **1.60°** | truth at lofi rank 1; min `surr_mse=1.018` |
| Step 3 (NM polish, alignment cost) | **1.31°** | NM 200-iter from good init can't escape local basin → ω preserved (q0_err=119.79°) |
| Step 4 (Geo L-BFGS-B, alignment cost) | **152.62°** | L-BFGS-B with gradients descends to deeper spurious alignment-cost basin → ω destroyed |
| handed to m115 (top-3 by geo_cost) | 68° / 93° / 133° | pool_min in entire geo_ckpt is 35° — far above 5° bridging radius |
| m115 DE (30 starts) + m126 polish | hifi=1.4725, q0_err=93.56°, w_dir_err=62.73° | **identical to baseline** — same wrong-basin attractor |

**Sharper finding from final result**: the M2 patch's final pipeline output on seed 47 (hifi=1.47254, q0=93.56°, w_dir=62.73°, w_mag=-3.85%) is **bit-identical to the alignment-cost-sorted baseline**. The pipeline downstream of m103 is so deterministic in its wrong-basin convergence that varying the lofi sort changes nothing about the m115 DE attractor. m115 finds the same 16 basins from the same geo_ckpt-top-3 ω handoff regardless of how m103 originally selected those candidates.

## What this localises

m103's failure mode is now precisely localised to **Step 4 Geo refinement under alignment cost**, NOT the lofi-stage selection. Two pieces of evidence:

1. NM polish (Step 3) at 200 Nelder-Mead iters can't bridge from a good init to a wrong-basin minimum — too few iters, no gradient. So a truth-near init *survives* NM under alignment cost.
2. Geo (Step 4) at L-BFGS-B with proper gradients DOES bridge — it walks from truth-near (1.31°) to anti-truth (152°) along the alignment-cost gradient.

This refines [[alignment-cost]]'s "anti-correlated with truth" claim from [[m135_alignment_cost_forensics_constrained_anchor]]: the cost is anti-correlated **wherever a gradient-descender can find the spurious basin**. Cost-at-truth probe in m135 measured static cost values; m137 demonstrates the dynamic of L-BFGS-B finding those spurious basins from truth-near inits.

## What this rules out

- **"Single-line m103 patch" cannot fix the failure cohort** — even if it preserves truth-near ω at the lofi stage, downstream geo refinement under alignment cost destroys it. Sort-only patches at any single stage are architecturally insufficient.
- **"Re-rank with surrogate-MSE" as a generic fix** — surrogate full-LC MSE is anti-correlated with truth in the same way as alignment cost on 4/5 failure seeds. Different cost shape, same family of failure.

## What this does NOT rule out

- **Replacing geo's cost with surrogate** (M2.5, untested): would let truth-near ω survive Step 4. Per the seed 47 trace, NM is fine; only Geo is the killer. Estimated patch: ~30 lines in m103. Worth piloting on seeds 47, 84, 89 before going heavy.
- **[[upstream-redesign-6dof-surrogate-de]]** with [[attitude-level-set-disconnection]]-aware cost (H1: surrogate-attitude-isoshell with per-epoch L(t) clustering + trajectory-membership cost) — the architectural fix that sidesteps NM/Geo entirely. Pre-empted by m136's lesson, motivated empirically by m137's pipeline trace.

## Bugs caught

1. **`audit_failure_seed_battery.py` path bug** — `Path(__file__).resolve().parents[1]` resolved to `notebooks/` not project root. Fixed to `parents[2]`. The script never ran successfully prior to this session.
2. **invert.py m103 auto-skip** — by default skips m103 if `geo_ckpt.npz` exists, ignoring env-var changes. Required `--force-m103` to actually re-run with new sort. m115/m126 caches also required manual deletion.

## Files

- `notebooks/inversion/11_casadi_formulation/m103_hybrid.py` — env-gated `M103_LOFI_SORT={align|surr}` patch (lines ~370-400, ~415).
- `notebooks/inversion/audit_failure_seed_battery.py` — path fix (parents[1]→[2]).
- `data/results/inversion_diagnostics/failure_seed_battery_2026_04_28/audit_summary.json` — 5-seed audit table.
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_047/{lofi,nm_prededup,multi_phi,geo}_ckpt.npz` — seed 47 stage-by-stage trace under surrogate sort.
- `data/results/inversion_diagnostics/invert_m048_seed047/result.json` — final pipeline outcome under M2 patch.

## Compute budget consumed

- Audit run: ~2 sec.
- Seed 47 pipeline (`--force-m103`): m103 ~95s (incl. 36s surrogate scoring) + m115 ~7 min + m126 ~3-5 min = ~12 min total.
- Seeds 84, 89 SKIPPED — cost-benefit gate fired after seed 47 result revealed the architectural issue (sort-patch insufficient regardless of which seed).

## Next move

H1 (surrogate-attitude-isoshell with per-epoch L(t) clustering + trajectory-membership cost). Skips m103 entirely; new upstream stage feeding m115. See [[surrogate-attitude-isoshell]] and [[attitude-level-set-disconnection]] for the cost-shape spec.
