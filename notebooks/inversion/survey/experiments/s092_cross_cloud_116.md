---
title: "s092 — first end-to-end cross of two REAL anchor clouds (seed 116)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s092_cross_cloud_116.py
  - notebooks/inversion/survey/results/s092/cross.json
related:
  - experiments/s088_bvp_shoot_conditioning.md
  - experiments/s089_third_anchor_overdetermination.md
  - experiments/s093_fullc_score_survivors.md
created: 2026-05-22
updated: 2026-05-22
confidence: high (full deterministic cross, 1.87M pairs; one reporting index-bug found and fixed mid-session — headline counts unaffected, truth-pair boolean was the only corrupted field)
---

# TL;DR
First cross of two *real* brightness-matched anchor clouds with NO handed-in correspondence (s088/s089 used perturbed-truth pairs). On seed 116 (LAM-slow, 30k Haar pool), crossing all **1,866,770** (q_a, q_b) pairs: connectability keeps **62.9%**, the cloud-free 3rd-anchor brightness check then keeps **4.26% (79,443)** — a **23.5× prefilter cut that retains the truth pair** (`truth_in_cpass=True`, source: `results/s092/cross.json`). But connectability is NOT a strong discriminator on a slow seed, and the 79k survivors are scattered (median 91.8° ω-dir; only 584 within 5°). **Architectural consequence: connectability + one brightness check is a cheap *prefilter*, not the discriminator — truth is retained but not isolated.**

# What
Tests the cross-cloud architecture end-to-end for the first time on real clouds: does crossing two brightness-matched clouds and screening by connectability (+ a cloud-free brightness check at a third anchor) isolate the truth-consistent pair from the combinatorial junk? Seed 116 chosen because it has the tightest measured anchor (s085: q to 1.47°) and a unique in-prior winding (s088/s089) — so the plumbing is tested without also fighting disambiguation.

# How
`experiments/s092_cross_cloud_116.py`, Pool(24), BLAS pinned, v2 surrogate.
1. Haar SO(3) pool (N=30k) → parallel sharpness search → 3 sharp anchors A=ep112, B=ep244, C=ep295 (greedy, min 40-epoch gap).
2. Cloud at each = pool members with `|surrogate_mag − obs_mag| < 0.10` (`survive_at_epoch`, `lib.c_t_pipeline`). No truth injection.
3. Positive control: nearest-truth member of A × of B → BVP solve.
4. Cross all pairs: connectability = one finite-diff-init `shoot` (geo < 1e-3° AND |ω| in ±30% band — a stand-in for the s019 LS-bracket prior).
5. Cloud-free 3rd-anchor check: propagate (q_a, ω) to C, one surrogate eval, keep if brightness matches within 0.10. No cloud built at C.

# Result
| stage | count | % of pairs |
|---|---:|---:|
| pairs crossed | 1,866,770 | 100% |
| connectable in-prior | 1,174,874 | 62.94% |
| + cloud-free C-brightness pass | 79,443 | 4.26% |
| truth pair survives both | yes ✓ | — |

- Positive control: real nearest-truth pair (q_a 5.65°, q_b 3.23°) connects exactly (geo=0), in-prior, **ω-dir 3.27°** — re-confirms s088/s089 on real members.
- Survivors scattered: ω-dir vs truth min 0.11°, **median 91.8°**, max 179.8°; **584 within 5°, 1,956 within 10°** of truth ω-dir.
- Why connectability is weak here: on a slow seed the connecting |ω| ≈ geodesic-angle/Δt, and a large fraction of random pairs land in the wide ±30% band — so connectability ≈ "is the rotation-rate plausible," broadly satisfied. Sharper in the winding regime (fast seeds, untested).

# Why this matters
- **The plumbing works:** real cloud cross at scale (1.87M pairs, 325s), truth survives with no correspondence handed in — the thing never tested before.
- **Connectability + one brightness check is a PREFILTER, not the discriminator.** 23.5× cut keeping truth, but truth not isolated (79k survivors). The discriminator must come from more of the light curve (→ s093 full-LC, → s094 windowed).
- **Slow seeds are the weak case for connectability.** 119 (winding regime) is the proper next test of connectability selectivity.

# Numbers
- npairs 1,866,770; connectable 1,174,874 (62.94%); C-pass 79,443 (4.2556%) (source: `results/s092/cross.json`).
- positive control ω-dir 3.27°, geo 0.0 (source: same, `positive_control`).
- survivors within 5°/10° ω-dir: 584 / 1,956; median 91.79° (source: same, `cpass_omega_dir_deg`).
- anchors A/B/C ep 112/244/295; clouds 1390/1343; nearest-truth A 5.65° B 3.23° (30k pool); wall 325s.

# Artefacts
- `results/s092/cross.json` (includes `cpass_a_pool`/`cpass_b_pool` raw survivor pool indices for re-scoring).

# Out of scope
- Fast-seed (119) connectability selectivity in the winding regime.
- Denser pool (best survivor < ~1° for the coherent-tube regime).
- The actual discriminator on survivors — see s093 (full-LC) and s094 (windowed).

# Cross-references
- `s088`/`s089` — the BVP primitive + over-determination, on perturbed-truth pairs.
- `s093_fullc_score_survivors.md` — full-LC RMSE scoring of these 79k survivors (negative).
- `s019` LS-bracket |ω| prior (the real source of the ±30% band).
