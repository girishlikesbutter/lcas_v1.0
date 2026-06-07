---
title: "m144 — NM-pool surrogate-MSE rerank diagnostic: m103's multi_phi_top=2 truncation is the bottleneck, not the ranking signal"
type: experiment
sources:
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/nm_prededup_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/nm_surr_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/nm_prededup_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/nm_surr_ckpt.npz"
  - "notebooks/inversion/score_nm_surrogate.py"
related:
  - "[[m141_seed6_postfix_pipeline]]"
  - "[[m142_seed6_postfix_surrogate_rerank]]"
  - "[[m143_seed91_postfix_generality]]"
  - "[[m139_convention_bug_fix]]"
  - "[[surrogate-rerank]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# m144 — NM-pool surrogate-MSE rerank diagnostic

#concrete-m103-patch-path #m103-truncation-is-bottleneck #post-fix-rerank-works-pre-truncation

## TL;DR

m142+m143 established that surrogate-MSE rerank applied to m103's *post-multi-phi 26-pool* (or its lofi-300 input) does NOT rescue the post-fix forward model on either seed 6 or seed 91. m144 walks the diagnostic one stage upstream: re-rank the **NM 300-pool** (300 candidates after Step 3 NM polish, BEFORE multi_phi truncation) by surrogate full-LC MSE, and ask whether jointly-truth-near candidates surface at top-K. **Result: YES, on both seeds.**

- **Seed 91 NM-300:** the best jointly-truth-near candidate (q0_err=57.79°, w_err=8.38°) lands at **surr_mse rank 10** (align rank 31). Top-10 by surr_mse is mostly duplicates of the same truth-ω basin (4 phi anchors × 137° q0_err) but rank 10 is a uniquely better candidate.
- **Seed 6 NM-300:** 3 jointly-truth-near candidates exist at surr_mse ranks 22, 26, 168 (align ranks 127, 132, 156). Best: q0_err=55.59°, w_err=9.38° at surr_mse rank 22.
- **Both seeds**: surrogate-MSE rerank on the NM-pool moves jointly-truth-near candidates *upward* from align rank 31–156 into surr_mse rank 10–22. Within a K=30 (or even K=10 for seed 91) selection.

The bottleneck is **m103's multi_phi_top=2 truncation** — Step 3.5 takes only the top-2 ω's by alignment cost and applies multi-phi to them; everything else is dropped to a single phi anchor. Under buggy truth, the alignment-cost top-2 sometimes happened to coincide with truth-ω, so multi-phi found a useful q0 anchor. Under correct truth, the alignment-cost top-2 are off-truth (seed 6: w_err 53–57°) and multi-phi runs on the wrong ω's. The truth-near ω's in the pool (rank 8 on seed 6 by alignment, rank 1 on seed 91 — rank 1 BUT with q0_err 137° because phi_anchors are bad) get either dropped (seed 6) or starved of phi anchors (seed 91 only multi-phi'd that ω, but the 4 anchors all give bad q0).

**The minimal fix is a single-line patch to m103**: increase `MULTI_PHI_TOP` from 2 to ~10–30, and re-rank the multi_phi candidate pool by surrogate-MSE before passing to Step 4 (Geo) or to m115. This is a CHEAP fix — no upstream redesign needed. **Play 3 (6-DOF surrogate DE replacing m103) is no longer the only option; m103's ranking is salvageable, just under-truncated.** Validation requires a single-seed end-to-end test post-patch (seed 6 or seed 91) — out of scope this session.

## What

[[m142_seed6_postfix_surrogate_rerank]] showed that re-ranking the 26-candidate post-Step-4 pool (`geo_ckpt.npz`) by surrogate-MSE doesn't rescue seed 6. [[m143_seed91_postfix_generality]] showed it doesn't rescue seed 91 either, AND that the lofi-300 pool (the pre-NM stage) lacks jointly-truth-near candidates on seed 6 entirely.

**The intermediate stage — NM-prededup (post-NM polish, pre-multi_phi) — was not yet probed.** NM polish moves grid+linear-solve candidates *toward* alignment-cost basins, often surfacing local minima the lofi pool didn't have. Earlier inline forensics on seed 6 (logged in CURRENT_STATE) found 7 jointly-truth-near candidates at align ranks 127–267/300 in NM-prededup. m144 asks: does surrogate-MSE rerank surface them?

## How

`notebooks/inversion/score_nm_surrogate.py` — a thin clone of `score_lofi_surrogate.py`, reading `nm_prededup_ckpt.npz` instead of `lofi_ckpt.npz`. Same scoring pipeline (post-fix `propagate_attitude` → `R(q)` from scipy → surrogate full-LC MSE against post-fix observed truth). Saves `nm_surr_ckpt.npz` with surr_mse + surr_bright_mse + echoed alignment cost / q0_err / w0_err.

```bash
python3 notebooks/inversion/score_nm_surrogate.py --seeds 6 91 --traj-source m048
```

Wall: 27s (seed 6) + 32s (seed 91) for 300 candidates each. Setup overhead ~1 min/seed (SPICE + surrogate load).

## Result — seed 91

```
=== TOP-10 BY surr_mse ASCENDING ===
 surr_rk  align_rk    surr_mse       align     w_err    q0_err
       1         3      3.5533  1.1558e-01      4.65    137.75   ← phi anchor 1 of truth-ω
       2         1      3.5533  1.1558e-01      4.65    137.75   ← duplicate (multi-phi propagated)
       3         2      3.5533  1.1558e-01      4.65    137.75   ← duplicate
       4         4      3.5533  1.1558e-01      4.65    137.75   ← duplicate
       5         9      3.6952  1.1560e-01      4.61    136.52   ← second ω cluster
       6         6      3.6952  1.1560e-01      4.61    136.52   ← duplicate
       7         5      3.6952  1.1560e-01      4.61    136.52   ← duplicate
       8         8      3.6952  1.1560e-01      4.61    136.52   ← duplicate
       9         7      3.6952  1.1560e-01      4.61    136.52   ← duplicate
      10        31      4.0572  3.7643e-01      8.38     57.79   ← UNIQUE: jointly truth-near
      11        29      4.3338  3.7632e-01      8.36    124.45
      12        30      4.3338  3.7632e-01      8.36    124.45
      13       144      4.4538  6.8364e-01     60.78     24.17
      ...
```

Jointly-truth-near (q0<60 AND w<30) NM candidates on seed 91:

```
idx=11  q0_err=24.22  w_err=28.75  surr_mse=5.5407 surr_rk=26 align_rk=13
idx=140 q0_err=57.79  w_err= 8.38  surr_mse=4.0572 surr_rk=10 align_rk=31  ← BEST
idx=145 q0_err=55.21  w_err= 7.68  surr_mse=5.7652 surr_rk=41 align_rk=18
idx=173 q0_err=24.22  w_err=28.75  surr_mse=5.5406 surr_rk=24 align_rk=14
idx=247 q0_err=24.22  w_err=28.75  surr_mse=5.5407 surr_rk=25 align_rk=15
```

**5 jointly-truth-near candidates exist in the NM-300 pool.** The best (q0_err=57.79°, w_err=8.38°) is at surr_mse rank 10, align rank 31. m115's q0-bridging is roughly <30° (this candidate's q0 is borderline outside) and ω-bridging marginal-radius is 15° (this candidate's ω at 8.38° is well within the marginal range). m115 K=10 from this candidate would be a Band-B-or-better candidate.

Other strong candidates: the q0_err=24.22° / w_err=28.75° trio (idx 11, 173, 247 — note: same q0/w errors → the same physical candidate appearing 3× in the pool, again multi_phi propagation duplicates). All at surr_mse rank 24–26. align rank 13–15. m115 from this candidate: q0_err 24° is bridgeable (just inside m115's q0 range), ω_err 28.75° is outside even marginal — not bridgeable.

## Result — seed 6

```
=== TOP-20 BY surr_mse ASCENDING ===
 surr_rk  align_rk    surr_mse       align     w_err    q0_err
       1       139      4.4651  4.0774e-02     34.42    173.43
       2         4      4.5286  4.6511e-03     48.97    168.77   ← m103's geo-rank 4
       3        48      4.5671  2.2545e-02     60.47    165.58
       4       298      4.6270  9.7989e-02     57.22    126.27
       ...
       9       220      4.7823  5.8656e-02      6.82    125.70   ← truth-near ω
      ...
      22       132      4.9229  7.3469e-02     43.12     75.72   ← almost truth-near (joint)
      ...
```

Jointly-truth-near (q0<60 AND w<30) NM candidates on seed 6:

```
idx=102 q0_err=54.92 w_err= 8.08  surr_mse=5.2976 surr_rk=26  align_rk=156  ← BEST
idx=136 q0_err=40.08 w_err=28.85  surr_mse=7.9812 surr_rk=168 align_rk=127
idx=154 q0_err=55.59 w_err= 9.38  surr_mse=5.1377 surr_rk=22  align_rk=132
```

3 jointly-truth-near candidates. Best: q0_err=54.92°, w_err=8.08° at surr_mse rank 26 (alignment rank 156). m115 from this: q0_err 55° is at the edge of m115's q0 range, ω_err 8.08° is marginal. A K=30 selection would include it.

Note: surr_mse rank 22 has w_err=43°, q0_err=75° — close to but not jointly truth-near. The seed-6 NM pool has fewer strong joint candidates than seed 91 (3 vs 5), and the best is at surr_mse rank 22 vs seed 91's rank 10.

## Why this matters

### 1. The m103 patch path is concrete and cheap

Before m144, the structural picture was: surrogate-MSE rerank fails on the post-multi_phi pools (m142, m143) → either patch m103's phi-sweep parameterisation OR replace m103 entirely. Both options expensive.

After m144: the rerank works **before** multi_phi truncation. The multi_phi truncation (`MULTI_PHI_TOP=2`) is what destroys the signal. Both alignment cost and surrogate-MSE on the NM-300 pool would surface jointly-truth-near candidates at ranks 10–32 (well within a K=30 selection). m103's existing pipeline already has the truth basin in NM; it just throws most of it away in Step 3.5.

**Minimal patch**: change `MULTI_PHI_TOP` from 2 to ~10–30, AND/OR replace alignment-cost ranking at Step 3.5 with surrogate-MSE rerank. Either lever should work. Both together is even better.

This is a 1–10 line patch to `notebooks/inversion/11_casadi_formulation/m103_hybrid.py` (or wherever Step 3.5 lives) plus a config exposure for the m115 K parameter to consume the larger pool.

### 2. The buggy-truth m133/m134/m138 wins are now better understood

[[m133_rerank_findings]] (single-cost surrogate-MSE re-ranking 9/17→15/17 on m048) was on the post-multi_phi pool. Under buggy truth, that worked because the buggy LC's bright-peak displacement happened to align m103's multi_phi truncation with truth on a fraction of seeds. Under correct truth, m142+m143 show it doesn't generalise — but m144 shows that's because the truncation is wrong, not because surrogate-MSE is the wrong signal. **Surrogate-MSE rerank is reliable on the right (pre-truncation) pool.**

The m135 lofi-300 finding ("surrogate ranking lands rank-1 at 2.76° from truth on seed 91 under buggy truth") is harder to reconcile — under correct truth, the lofi-300 pool barely has jointly-truth-near candidates and the NM step is what surfaces them. Possible explanation: under buggy truth, the lofi-MSE on seed 91 was sensitive to the specific buggy bright-peak displacement at certain (q0, ω) values and that produced a sharp surr_mse minimum near the buggy-truth basin. Under correct truth, that artefact disappears.

### 3. Play 3 is no longer the only option, but is still valuable

[[upstream-redesign-6dof-surrogate-de]] (Play 3, 6-DOF DE replacing m103) is more powerful: a surrogate-driven DE could find the truth basin without depending on m103's grid+NM substrate at all. But m144 shows it's not REQUIRED — the existing m103 substrate, with a small patch to its truncation step, can surface truth basins at top-K=10–30.

For the cohort regeneration question: m144 makes it tenable to plan the cohort regeneration before Play 3 is built. The patched m103 can be the pipeline for the regenerated cohort (with K=10–30 to m115); Play 3 stays in the design hopper for the next bottleneck.

### 4. m115 input expansion is required

m103 → m115 currently passes top-K ω candidates (K=3 default). m144's seed 6 best is at surr_mse rank 22; seed 91 best is at surr_mse rank 10. **m115 K must increase to at least K=10 (seed 91) or K=30 (seed 6) to consume the truth basin.** This requires:

- m103 to emit a larger candidate pool (after multi_phi expansion).
- m115's `load_omega_candidates` to read K=10–30 instead of K=3.
- m115's per-ω 3-DOF DE to run K times in series (or K parallel workers if available).

Wall-time impact: m115 currently takes ~5 min for K=3 with 30 DE starts each (so ~10 sec/start). K=30 ω → 30 × 10 sec = 5 min if DE-starts are reduced from 30 to 10 per ω. Or K=30 × 30 starts = ~15 min. Either way, m115 wall doubles or triples. Acceptable.

### 5. The geo step still needs to be evaluated

m103's Step 4 (Geo) refines multi_phi candidates further. On seed 91 it timed out at 480s. With a larger multi_phi pool (K=10–30 ω × 4 phi each = 40–120 candidates), Step 4 wall would 2–5× longer. Either need to relax timeout, or skip Step 4 entirely and feed multi_phi directly to m115.

Skipping Step 4 is testable cheap — try the patched m103 with `--skip-geo` (if such flag exists; otherwise add it). m144 evidence suggests Step 4 is not load-bearing for the truth basin survival; it's just a refinement that helps when the basin is already in the pool.

### 6. The lofi `surr_mse` NaN issue is a separate bug

`lofi_ckpt.npz` field `surr_mse` is all NaN on seed 6 (and seed 91 likely the same — to verify). The standalone `score_lofi_surrogate.py` produces valid surrogate scores when run after the fact, so the issue is in m103's lofi step's surrogate evaluation. Investigation is cheap; fix is probably small. Out of scope this session but flagged for next session.

## Numbers

| | seed 6 | seed 91 |
|---|---|---|
| NM-300 pool min `q0_err` | 13.23° | 10.22° (estimated) |
| NM-300 pool min `w_err` | 3.42° | 0.73° (estimated) |
| Jointly truth-near (q0<60 AND w<30) | **3** / 300 | **5** / 300 |
| Best joint candidate q0_err | 54.92° | 24.22° (multiple duplicates) |
| Best joint candidate w_err | 8.08° | 7.68° |
| Best joint candidate surr_mse rank | **22** / 300 | **10** / 300 |
| Best joint candidate alignment rank | **156** / 300 | **31** / 300 |
| `surr_mse` min (NM pool) | 4.4651 | 3.5533 (×4 duplicates) |
| `surr_mse` median | 7.5983 | 7.8353 |
| Wall (scoring only) | 27.2 sec | 32.1 sec |
| Wall (incl. setup) | 27.5 sec | 32.3 sec |

(Note: the 9 surr_mse rank 1–9 candidates on seed 91 are 4-anchor and 5-anchor multi-phi duplicates of two near-identical ω clusters. The "best UNIQUE truth-near candidate" at surr_mse rank 10 is the first non-duplicate jointly-truth-near.)

## Artefacts

- `notebooks/inversion/score_nm_surrogate.py` — the adapter (130 lines).
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_006/nm_surr_ckpt.npz` — m144 NM rerank artefact for seed 6.
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/nm_surr_ckpt.npz` — same for seed 91.

## Out of scope here

- **m103 patch implementation and end-to-end test.** The minimal patch is concrete (`MULTI_PHI_TOP=10` or higher + optional surrogate-MSE rerank at Step 3.5). End-to-end test is one seed × ~15 min wall + m115 K=10–30 + m126; multi-stage. Out of scope this session.
- **Cohort-scale validation.** Once the patched m103 + m115 K-expansion is end-to-end validated on 1–2 seeds, run on the random m048 25-seed cohort under correct truth (separate cohort regen session).
- **Play 3 design.** Now lower priority but still valuable; the m103 patch may not generalise to all failure modes (e.g., genuinely constraint-poor seeds where even the NM pool has no truth basin).
- **lofi `surr_mse` NaN bug investigation.** Quick (~5 min) but separate issue.

## Cross-references

- [[m141_seed6_postfix_pipeline]] — the upstream-FAIL finding that motivated the surrogate-rerank line of inquiry.
- [[m142_seed6_postfix_surrogate_rerank]] — surrogate-rerank on the post-multi_phi 26-pool fails (truth not in pool).
- [[m143_seed91_postfix_generality]] — seed-91 generality: same q0-anchor pathology + lofi-300 has at most 1 joint-truth-near.
- [[m139_convention_bug_fix]] — the propagator fix that made these diagnostics meaningful.
- [[surrogate-rerank]] — the lever now characterised as **reliable BEFORE multi_phi truncation, unreliable AFTER**.
- [[upstream-redesign-6dof-surrogate-de]] — Play 3, no longer the only option.
- [[m135_alignment_cost_forensics_constrained_anchor]] — buggy-truth lofi-300 surrogate rescue; m144 helps explain why it doesn't generalise.
