---
title: "s069 — Replicate s059k (Band A on cohort, no oracle) on post-fix m048: architecture works, seed 89 yield shifts 4→1"
type: experiment
sources:
  - experiments/s059j_cloud_data_omega_grid.py (unchanged)
  - experiments/s059k_full_lc_from_seeds.py (unchanged)
related:
  - experiments/s067_postfix_propagator_validation.md
  - results/s059k_nd800_seed89/seed089/ (post-fix s059j output)
  - results/s059k_nd800_seed89/seed089/full_lc_seeds/summary.json (post-fix s059k full_lc)
created: 2026-05-12
updated: 2026-05-12
confidence: high (full pipeline + 5 mag-offsets, direct pre/post comparison)
---

# TL;DR

Re-ran the s059k pipeline on seed 89 under the post-fix m048 cohort. The architecture functions correctly — surrogate-score → cluster → full-LC LM polish with multi-mag-start delivers Band A solutions on the post-fix data — but yield on this specific seed shifted from 4/50 unique clusters (pre-fix) to **1/50** (post-fix). The single post-fix Band A cluster (id=457, q0_err=59.58°, ρ_hifi=0.90) is a multi-solution attractor, not the truth basin. Truth cluster ranks 1987/2013 by score (pre-fix it was 797/2412) and stays outside the top-50 polish window. Methodology survives; specific seed-89 numerics did not.

# What

s059k (2026-05-08) was the first end-to-end Band A inversion on a cohort seed without oracle injection. On pre-fix seed 89: 4/50 unique Band A clusters (truth at id=338 → ρ=0.18; body-twin id=32 → ρ=0.17; multi-sol id=125; one more). After the propagator fix landed (commit `d5705ff`) and the cohort was regenerated (commit `7aad73b`), this replication re-runs the s059j+s059k_full_lc pipeline on the regenerated seed 89 to confirm the architecture still works.

# How

Two-stage pipeline (both scripts unchanged):
1. `experiments/s059j_cloud_data_omega_grid.py --seed 89 --n-dirs 800 --n-workers 16 --out-root results/s059k_nd800_seed89`
2. `experiments/s059k_full_lc_from_seeds.py --in-dir results/s059k_nd800_seed89/seed089 --seed 89 --top-k-polish 50 --mag-starts "0.0,3.0,-3.0,6.0,-6.0" --n-workers 8`

`n-workers 16` (down from pre-fix 24) due to memory pressure on the host post-regen.

# Result

**s059j stage (post-fix)** — score grid + cluster + 20-rank local-window polish:
- Anchor T_A=22, |C_a|=18939, dense pool capped to 3000 q_a candidates.
- 2013 clusters from 5000 top-K canonicalised. **Truth cluster rank: 1987/2013** (vs pre-fix 797/2412).
- Local-window polish: 20/20 land surrogate ρ_pol = 0.09–0.10, but hi-fi gating reveals all 20 are phantom basins (Band D, hi-fi ρ = 27–48). Identical s059k pre-fix phantom-basin phenomenon — this is why s059k_full_lc exists.
- Score stage wall: 5071 s (~85 min). Pre-fix was 4212 s (~70 min) — slower due to system swap pressure during the post-fix replication run; not algorithmic.

**s059k_full_lc stage (post-fix)** — full-LC LM polish, 50 clusters × 5 mag-offsets = 250 polishes:

| Metric | Pre-fix | Post-fix |
|--------|--------:|---------:|
| Band A polishes (all 250) | 11 | **2** |
| Unique Band A clusters | 4 | **1** |
| Cluster A∪B yield (50 polish) | 4/50 | **1/50** |
| Truth cluster rank | 797 / 2412 | 1987 / 2013 |
| Polish wall | 1202 s | 345 s |

**Post-fix unique Band A cluster:**
- cluster id=457 (rank 28 / 2013), q0_err=59.58°, |ω|err=+0.57%, ω_dir_err=25.05°, surrogate ρ_pol=0.98 → **hi-fi ρ=0.904 = Band A**.
- Multi-solution attractor (q0_err=60° is far from any twin direction), NOT truth basin.

**Pre-fix unique Band A clusters (reference):**
- id=338 (truth): q0_err=0.86°, ρ=0.177
- id=32 (body-twin): q0_err=179.5°, ρ=0.17
- id=125: q0_err=?° (added by Jacobi polish — see pre-fix s064)
- one more

# Why this matters

**Architecture is sound on post-fix data.** The s059j → s059k_full_lc pipeline still produces Band A hi-fi solutions without oracle injection. Phantom-basin phenomenon under local-window polish is preserved (s059k's full-LC polish remains necessary).

**Specific seed-89 numerics shifted dramatically.** Pre-fix's headline "4/50 unique Band A on seed 89" is NOT preserved. The truth basin now ranks at the very bottom of the score grid (1987/2013) and is not surfaced by top-50 polish. This is consistent with the cleanup plan's stated assumption ("specific seed numerics don't survive; methodology does").

**Multi-solution acceptance pays off.** Cluster 457 at q0_err=59° hi-fi ρ=0.90 IS a legitimate Band A LC match — under the multi-solution philosophy this counts as a successful inversion of this LC. Truth recovery on this specific post-fix LC remains an open question; the architecture would need either deeper polish (top-K > 50) or different anchor selection to surface truth at rank 1987.

**Possible reasons for the yield shift** (not investigated here):
1. Post-fix LC for seed 89 may have different anchor/cluster structure that ranks truth lower.
2. The score function's W=10 local-window cost surface may have a different argmin geometry on the new physical data.
3. Multi-solution geometry may have shifted — fewer near-truth attractors and more distant-but-LC-matching ones.

All three are open empirical questions; not on the cleanup critical path.

# Numbers

See `results/s059k_nd800_seed89/seed089/full_lc_seeds/summary.json` (post-fix) and `results_prefix/s059k_full_lc_seed89_nd800/summary.json` (pre-fix baseline).

# Artefacts

- `results/s059k_nd800_seed89/seed089/{anchor_summary.json, score_grid.npz, clusters.npz, polished_states.npz, summary.json, run.log}` — post-fix s059j output
- `results/s059k_nd800_seed89/seed089/full_lc_seeds/summary.json` — post-fix s059k_full_lc output
- `results_prefix/s059k_nd800_seed89/`, `results_prefix/s059k_full_lc_seed89_nd800/` — pre-fix baseline (delete by 2026-05-19)

# Out of scope

- Re-running with top-K > 50 to see if extending polish budget surfaces truth at rank 1987.
- Trying alternative anchor selection (sharpness map / multi-anchor a la s060).
- Running s059k on seed 10 (the pre-fix companion seed).
- Investigating WHY truth ranks 1987/2013 post-fix vs 797/2412 pre-fix — possible diagnostic targets: anchor T_A position, |C_a| pool quality, ω-grid centring around `|ω_a_body|`.

# Cross-references

- `experiments/s059k_full_lc_from_seeds.py` — full-LC polish architecture, unchanged.
- `experiments/s059j_cloud_data_omega_grid.py` — score-grid + cluster pipeline, unchanged.
- `experiments/s067_postfix_propagator_validation.md` — propagator fix + 4-gate validation.
- `experiments/s070_replicate_s064.md` — same pipeline polished via Jacobi-coord LM (also lands 1/50 cluster on the same id=457).
