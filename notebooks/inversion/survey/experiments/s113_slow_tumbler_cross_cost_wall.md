---
title: "s113 — slow-tumbler generality BLOCKED by a weak-anchor cross-cost wall; the s111 adaptive coarsening was load-bearing for tractability"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s100_5step_proto.py
  - notebooks/inversion/survey/experiments/run_slow_sweep.sh
  - notebooks/inversion/survey/experiments/s113_slow_tumbler_generality.py
  - research_os/contracts/contract_slow-tumbler-generality.json
related:
  - experiments/s111_anchor_cap_116_invert_and_decimation_bug.md
  - experiments/s112_slow_tumbler_pipeline_report.md
created: 2026-06-01
updated: 2026-06-01
confidence: high (cell-count + cross-rate are deterministic; verdict from cell counts, not a completed grind)
---

# TL;DR
The contract (`contract_slow-tumbler-generality`) aimed to fix the three s111 bugs and re-run the slow sweep on 116/10/42/31. The **116 method-canary** killed it twice — not on science, on **cross cost**. The s111 decimation fix is **correct** (fixed-2° preserves resolution: B nt **2.51°**, not the bug's 6.61°), but it removed the only thing bounding the A×B cross. On 116, anchor **B=ep52 is pathologically weak** (46469 dense survivors = **4.6%** of 1M, vs A=ep7's 0.6%) → **40941 cells at 2° → 211M cross pairs (~8.2 hr)**. The v1 amendment (coarse-6°-then-fine-2°) cut it only to **16.4M pairs (~38 min)** — still over budget — because B has 14502 cells even at 6°. **Root cause:** the anchor-cap (`dt_ab<π/ω_hi=336s`, the s111 aliasing cure) FORCES a close-and-weak B; close ⟹ weak ⟹ huge cross, and no decimation granularity that keeps the truth basin resolves it. **The s111 116 win used the BUGGED adaptive coarsening, which was load-bearing for cross-tractability.** Per user call: stop & record.

# What
Pre-registered as `contract_slow-tumbler-generality` (v0 frozen → v1 amended). v0: fix #1 (fixed-2° decimation), #3 (run_slow_sweep.sh padding), #4 (PAIR_BUDGET 6M→50M); run 116(canary)/10/42/31. v1 (post-canary, user-approved): replace the single-pass cross with coarse-then-fine.

# How
- s100 edits: fix #1 `decimate_adaptive → decimate_2deg(max_reps=0)`; fix #4 PAIR_BUDGET→50M; v1 coarse-then-fine in the anchor-cap branch (`run_cross` helper run twice: coarse `COARSE_DEG=6°` w/ relaxed `COARSE_TOL_MAG=0.30`, then fine 2° within surviving coarse cells via `cell_codes`/`np.isin`).
- run_slow_sweep.sh: fix #3 padded `seed$(printf '%03d')`; seed list `116 10 42 31` (116 first as canary).
- s113_slow_tumbler_generality.py: the artefact aggregator (surr-ρ table + 4 plots + scoring) — **never run** (no candidates produced).

# Result
**v0 fixed-2° canary (116), killed at `[5]`:** A 6024 dense → 5159 cells; **B 46469 dense → 40941 cells**; cross wanted 5159×40941 = **211M pairs**; PAIR_BUDGET=50M random-culled B → 7071 → still 36.5M pairs (~85 min). The random cull is exactly the truth-drop fix #4 meant to remove.

**v1 coarse-6° canary (116), killed at `[4c]`:** A 6024→**1132 cells**, B 46469→**14502 cells** → coarse cross **16.4M pairs (~38 min)** — over the canary 2× kill before the fine pass.

Cross rate **7140 pairs/s** (source: `results/s100/seed42_capped_run.log`, 2.48M→348s).

| pass | A cells | B cells | cross pairs | est wall |
|---|---|---|---|---|
| v0 fixed-2° | 5159 | 40941 | 211M | ~8.2 hr |
| v0 capped (PAIR_BUDGET 50M) | 5159 | 7071 | 36.5M | ~85 min |
| v1 coarse 6° | 1132 | 14502 | 16.4M | ~38 min |

# Why this matters
The decimation "bug" was **load-bearing**: adaptive coarsening bounded the cross by destroying resolution; the fix preserves resolution but unbounds the cross. The anchor-cap (close B, cures aliasing) and cross-tractability (sharp/few-cell B) are in **direct tension** on weak-anchor seeds. Generality of the as-built pipeline is NOT cheaply achievable — the cross architecture needs a weak-anchor-aware redesign. Candidate directions (NOT run): (a) C-anchor pre-prune of B via A→B→C connectability (C=ep433 already computed); (b) connectability-tube cross (per-A forward-propagate + kd-tree match nearby B, the s061 thread idea); (c) coarser coarse cells (10-12°) at the risk of dropping truth.

# Numbers
- A=ep7 admission 6024/1M=0.6%; B=ep52 admission 46469/1M=4.6% (source: `results/s100/seed116_cf_canary.log [3]`).
- cells: A 2°→5159 / 6°→1132; B 2°→40941 / 6°→14502 (source: `seed116_capped_run.log [4]`, `seed116_cf_canary.log [4c]`).
- dt_cap=336s (π/ω_hi); A→B `|ω|·dt_ab`=0.76 rad < π (source: canary `[2]`).
- decimation fix: B nt 2.51° (fixed-2°) vs 6.61° (bugged adaptive on seed 42) (source: `[4c]`, s111 writeup).
- 116 s111 reference intact: truth q0 0.19°, ω-dir 0.03° (`results/s110/polish_116_s111ref.json`).

# Artefacts
- `research_os/records/s113_slow_tumbler_cross_cost_wall.json` — this run record.
- `results/s100/seed116_capped_run.log` (v0 canary), `seed116_cf_canary.log` (v1 canary).
- `results/s100/seed116_s111ref/`, `results/s110/polish_116_s111ref.json` — the s111 116 reference (preserved).
- Code: s100 (fix #1/#4 + coarse-then-fine), run_slow_sweep.sh (fix #3), s113_slow_tumbler_generality.py (aggregator, unrun).

# Out of scope
Did not score 10/42/31 (the canary gated them and never released). Did not run the candidate redesigns (a/b/c above).

# Cross-references
- Contract: `research_os/contracts/contract_slow-tumbler-generality.json` (v0 frozen + v1 amendment).
- Parents: s111 (116 inverts + decimation bug), s112 (slow-tumbler report).
