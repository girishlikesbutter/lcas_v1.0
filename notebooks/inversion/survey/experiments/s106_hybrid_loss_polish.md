---
title: "s106 — windowed-photometry LM polish clears Band A on seed 119 (0.687→0.030); the window is the lever, soft-B is inert"
type: experiment
sources:
  - experiments/s106_hybrid_loss_polish.py
  - experiments/s106b_polish_timing.py
  - results/s106/hybrid_polish_seed119_abc.json
  - results/s106/hybrid_polish_seed119_full.json
  - results/s106/hybrid_polish_seed119_c.json
  - results/s106/polish_timing.json
related:
  - experiments/s105_pairs_to_omega_decomposition.md
  - experiments/s100_5step_coverage_proto.md
created: 2026-05-28
updated: 2026-05-28
confidence: medium (decisive N=1 on seed 119's oracle-nearest pair; discrimination + generalization UNTESTED)
---

# TL;DR
The s105 hard-shoot trap is broken by replacing the hard A→B connection with a **photometry-objective LM polish**: free ω, minimize predicted−observed magnitude over a forward window from the anchor, seeded from the pair-shoot ω. On seed 119's near-truth cloud pair (qa 1.02°, qb 0.68° off) this pulls the seed ω from **0.687 / Band D / 4.25° off** to **0.0300 / Band A / 0.57° off, |ω| 1.5025 deg/s** (source: `results/s106/hybrid_polish_seed119_abc.json`) — the first Band-A result on this fast-tumbler seed. Two non-obvious findings: (1) the **soft B-attitude term is inert** — w_B = 0/0.1/1.0 give identical answers, the photometry window does everything; (2) the **window choice is the lever** — a forward A→C span (369 ep) → Band A, but full-LC (500 ep) → Band D (32° off) and a narrow window around C (41 ep) → Band D (151° off). Distinct from s105g (windowing the *score* fails): here the window is the *optimization objective*.

# What
s105 proved the binding blocker on 119 is the hard-shoot trap (connecting ~1°-off endpoints over 3.1 turns bakes a 4.25° ω error → Band D) and that windowing the *score* doesn't rescue it. The user's fix: stop hard-connecting. Make the polish objective the *light-curve fit* over a span of epochs, with the B-attitude match demoted to a soft side-residual (or dropped). This experiment builds and runs that polish, and isolates which window makes it work.

# How
Script: `experiments/s106_hybrid_loss_polish.py` (seed 119, anchors A=ep69/B=ep172/C=ep377). Hold q_a at the nearest-truth cloud rep (1.02° off, NOT corrected — realistic), free ω, seed from the best `multistart_shoot` root, then `least_squares(method='lm')` on:

```
resid(ω) = [ surrogate_mag(propagate(q_a, ω, t_k)) − obs_mag_k   for k in window ]
        ++ [ sqrt(w_B) · rotvec( propagate(q_a, ω, dt_AB), q_b ) ]     # soft B, 3 comps
```

Window modes: `c` (±PAD around C, 41 ep), `abc` (A→C forward span, 369 ep), `full` (all 500 ep). w_B swept {0, 0.1, 1.0}. Propagation via `propagate_jacobi_path2` from the anchor (forward+backward stitched for full-LC scoring; back-prop verified in s105_verify). |ω| solver bracket [0.1, 1.6] deg/s. Diagnostics: truth floor 0.0180, and `ceil_qa` 0.1189 = the best full-LC RMSE achievable while holding q_a at the 1.02°-off rep with truth ω (the realistic ceiling this pair can reach).
Timing isolated in `experiments/s106b_polish_timing.py` (median of 7 solves, single core, reports nfev + ms/eval).

# Result

| objective window | photo ep | best full-RMSE | band | dir-off | |ω| dps | source |
|---|---|---|---|---|---|---|
| seed (pair-shoot) | — | 0.6870 | D | 4.25° | 1.5018 | realpair_multistart.json |
| **abc (A→C, 369 ep)** | 369 | **0.0300** | **A** | **0.57°** | 1.5025 | hybrid_polish_seed119_abc.json |
| full (500 ep) | 500 | 1.9532 | D | 32.11° | 1.5542 | hybrid_polish_seed119_full.json |
| c (±PAD, 41 ep) | 41 | 1.7295 | D | 150.91° | 1.2225 | hybrid_polish_seed119_c.json |

w_B sweep (abc): 0 → 0.029957, 0.1 → 0.029957, 1.0 → 0.029958 (identical to 4 dp; source: hybrid_polish_seed119_abc.json `polishes`).

Timing (source: `results/s106/polish_timing.json`):

| objective | n_ep | wall/solve | nfev | ms/eval |
|---|---|---|---|---|
| attitude-only (the shoot) | 2 | 0.94 ms | 2 | 0.47 |
| photo c | 41 | 1.67 s | 102 | 16.4 |
| **photo abc** | 369 | **1.45 s** | **9** | 161 |
| photo full | 500 | 14.08 s | 70 | 201 |

The working abc window converges in just **9 LM iterations** — it is both correct AND ~10× cheaper than full-LC, which thrashes at 70 nfev and lands Band D.

# Why this matters
- **First Band A on seed 119**, the high-|ω| narrow-basin tail that resisted s099/s100. Validates the user's "relax hard connection → soft photometry fit" insight.
- **The lever is the photometry window, not the soft-B term.** Pure photometry over the right window does everything. This reframes the production step from "shoot then score" to "seed-then-photometry-polish."
- **Window-as-objective is a real optimization-basin effect** [my read, mechanism UNCONFIRMED]: the near-anchor, low-turn span gives LM a smooth gradient to truth; the full-LC's high-turn tails are too rugged and trap it (32° off); a narrow C window has no smooth foothold and diverges out of basin (151° off). Confirming this (residual-landscape slice + nfev/convergence logging) is the cheapest open follow-up.

# Numbers
- seed→polished: 0.6870 D (4.25°) → 0.029957 A (ρ 0.5991, dir-off 0.5694°, |ω| 1.5025 dps, qB_geo 0.924°), q_a held 1.0167° off (source: results/s106/hybrid_polish_seed119_abc.json `seed_full_rmse`,`best`)
- realistic ceiling ceil_qa 0.11885; truth floor 0.018038 (source: same)
- w_B {0,0.1,1.0} → {0.029957,0.029957,0.029958} (source: same `polishes`)
- full mode 1.9532 D dir 32.11°; c mode 1.7295 D dir 150.91° (source: hybrid_polish_seed119_{full,c}.json)
- timing attitude-only 0.94 ms (2 nfev) vs abc 1.45 s (9 nfev) vs full 14.08 s (70 nfev) (source: results/s106/polish_timing.json)
- pipeline estimate: ~721 s100 survivors × ~1.45 s / 24 cores ≈ 44 s (abc window); far pairs that hit max_nfev=400 could cost ~58 s worst-case [estimate, unverified at scale]

# Out of scope — THE OPEN CRUX
- **Discrimination (make-or-break):** this is the *oracle-nearest* pair. Does the windowed polish pull only near-truth pairs to Band A while far pairs stay Band D, or does it pull everything to Band A (phantoms)? Until this is run, "reaches Band A" is not "inverts."
- **Window sweet-spot:** PAD/extent sensitivity between the failing 41-ep and 500-ep and the working 369-ep.
- **Generalization:** seed 116 and other 119 pairs untested.
- **Integration:** fold the polish into the s100 cross (replace single-shoot ω-selection).

# Cross-references
- [[s105_pairs_to_omega_decomposition]] — the hard-shoot trap this fixes; windowing-the-score (s105g) vs windowing-the-objective (here).
- [[s100_5step_coverage_proto]] — coverage proto that surfaces the cross candidates this polishes.
- [[feedback_omega_prior_physical_bracket]] — [0.1,1.5] prior, solver 1.6 deg/s margin.
- [[feedback_dont_overclaim_from_one_data_point]] — N=1; discrimination/generalization pending.
