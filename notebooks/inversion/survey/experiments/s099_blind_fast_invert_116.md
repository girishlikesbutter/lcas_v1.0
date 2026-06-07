---
title: "s099 — BLIND + FAST inversion of seed 116 (truth-free |w| bracket, 11.7 min)"
type: experiment
created: 2026-05-26
updated: 2026-05-26
confidence: high
sources:
  - notebooks/inversion/survey/experiments/s098_local_densify_116.md
  - notebooks/inversion/survey/experiments/s019_ls_bracket_omega_mag.md
  - notebooks/inversion/survey/experiments/s092_cross_cloud_116.py
---

## TL;DR

s098 "closed a blind inversion" on seed 116 but had two holes flagged by the
user: (1) the |w|-magnitude acceptance band was `[0.70,1.30]*|w|_TRUE` — centered
on the answer, NOT blind; (2) it ran 112 min. s099 fixes both:

- **GOAL 1 (truly blind |w|):** replaced the truth-centered band with the s019
  Lomb-Scargle bracket computed PURELY from the observed LC. Validated held-out
  20/20 on seeds 100-119 (s019 only ever ran 0-99), so the policy was never tuned
  on the test seeds. The 116 bracket `[0.00109,0.00935]` rad/s is a *superset* of
  the old cheating band — strictly more honest, can only admit more candidates.
- **GOAL 2 (<15 min):** profiling showed 90.8% of the 112 min was full-500-epoch
  surrogate RMSE on 1.75M survivors, and that connectability + the |w| band reject
  *nothing* (100% pass — the prior agent's "prefilter" plan was a dead end). A
  K=50 uniform-decimated RMSE preserves the full ranking (Spearman 0.997). Two-
  stage scoring (coarse-K -> full-500 top-4000) + K_KEEP 100->50 -> **705 s
  (11.7 min)**.

**Result: the top BLIND candidate renders hi-fi rho 0.52 (Band A); ALL 15 saved
blind candidates are Band A (rho 0.52-0.75)** — a mix of near-truth-omega basins
(omega-dir <1.5 deg, q 14-22 deg) and body-twins (dir ~173-177 deg, q 145-176 deg).
Multi-solution with truth's basin in the set, now with a truth-free |w| prior and
9.6x faster.

## Blindness audit (what touches truth)

| ingredient | blind? | source |
|---|---|---|
| SO(3) pool, isophote clouds A/B | yes | observed mag only |
| anchor selection (sharpest |C_t|) | yes | reproduces s092 A=112,B=244,C=295 |
| connectability (geo<1e-3) | yes | geometric |
| **|w| acceptance window** | **yes (s019 bracket)** | observed LC Lomb-Scargle; was `0.7-1.3*|w|_true` in s092/s098 |
| C-pass brightness @ anchor C | yes | observed mag |
| full-500 surrogate RMSE rank | yes | observed LC |
| oracle qa/qb/dir labels | label-only | never used to rank |

The ONLY remaining truth use is infra (truth-LC floor must be ~0.01) and labeling.

## Pipeline + wall (Pool 24, v2 surrogate)

| stage | what | count | wall |
|---|---|---|---|
| A cross | 1390x1343 pairs -> connect -> blind |w| -> C-pass -> coarse-K RMSE | 97,938 survivors | 365 s |
| (rank) | top-250 parents by coarse RMSE (best 0.0794) | 250 | — |
| B densify | M_PERT=800, K_KEEP=50 local clouds -> local cross -> coarse-K RMSE | 446,613 survivors | 289 s |
| C full-500 | real 500-epoch RMSE on coarse-top-4000 | 4000 | 15 s |
| **total** | | | **705 s (11.7 min)** |

(blind window admitted 97,938 cross survivors vs s092's 79,443 with the cheating
band — only ~23% more, because connectable pairs' |w| is already concentrated.)

## Blind ranking (full-500 surrogate RMSE; floor 0.0102)

```
rank | full-RMSE | dir    | qa     qb    | hi-fi rho (band)
   1 |   0.0274  |  0.86  | 21.85  19.61 | 0.52 (A)   <- top BLIND candidate
   2 |   0.0313  | 174.51 | 175.52 131.71| 0.63 (A)   body-twin
   3 |   0.0326  |  1.45  | 14.67  10.65 | 0.67 (A)
   4 |   0.0327  |  0.31  | 17.09  16.77 | 0.63 (A)
   5 |   0.0328  | 173.87 | 175.99 133.13| 0.67 (A)
   6 |   0.0329  |  0.47  | 18.24  16.77 | 0.63 (A)
   7 |   0.0329  |  0.17  | 17.09  16.46 | 0.62 (A)
  ...  (ranks 8-15 all Band A: rho 0.62-0.75; near-truth-omega + body-twins)
```
All 15 saved blind candidates render Band A (rho<1). Full 16-render table (incl
truth control rho 0.00) in results/s099/hifi_rho.json.

## Gotcha fixed

The coarse scorer first violated the `times[0]==0` propagator gauge (it propagated
directly to decimated epochs whose first relative time != 0), giving a truth
coarse floor of 0.149 vs full-500 0.0102. Fixed by 0-anchoring both fwd and bwd
coarse time arrays and dropping the anchor sample; coarse floor -> 0.0106. The
s099c rank check was unaffected (it propagated full then subsampled the residual).

## Artefacts

- `experiments/s099_invert_116.py` — blind+fast pipeline (env-knob overrides for smoke)
- `experiments/s099_hifi.py` — hi-fi rho-band of blind winners
- `experiments/s099a_omega_bracket_holdout.py` — held-out bracket validity (20/20)
- `experiments/s099b_profile_densify.py` — the 90.8%-full-LC profile
- `experiments/s099c_coarse_rank_check.py` — coarse-K rank fidelity (Spearman 0.997)
- `results/s099/{invert,hifi_rho,bracket_holdout,profile,coarse_rank}.json`

## Seed 119 — held-out generalization (aliased-class, diagnostic FAIL)

Ran the identical blind+fast pipeline on seed 119 (S099_SEED env; outputs in
`results/s099/seed119/`). 119 is the WORST held-out bracket (factor 142x vs 116's
8.58x; truth |w|=0.0259 = 11x faster tumbler, 9 LS peaks) AND a known near%=0
*aliased* seed (s095). Two predictions tested:

- **Prediction 1 (wide bracket -> slow): REFUTED.** 119 ran in **288 s (4.8 min)** —
  *faster* than 116. The wide |w| window was NOT the bottleneck; the cross was
  tiny because the sharp anchor B admitted only |B|=20 pool members (vs 116's 1343).
- **Prediction 2 (aliasing -> no truth-near candidate): CONFIRMED, and stronger —
  no Band-A candidate at all.** Cloud B nearest-truth was 37.1 deg (|A| nearest
  15.1 deg); best blind full-500 surrogate RMSE **1.65 mag** (~90x the 0.018 floor,
  ~50x 116's Band-A candidates). Top blind candidate hi-fi **rho 33.0 (Band D)**;
  0/3 Band A (`results/s099/seed119/hifi_rho.json`).

**Conclusion: the blind+fast machinery is seed-agnostic in MECHANISM (ran clean
and fast on the worst-bracket seed), but multi-sol yield is gated UPSTREAM by
anchor quality.** 119 fails not because of the |w| bracket or speed (s099's scope)
but because the 30k pool does not represent truth at the auto-selected sharp
anchors — the s095 near%=0 aliasing. The fix is architectural (anchor-baseline
selection in the unique-winding regime, s095 next-step), NOT density/bracket/speed.

| seed | bracket factor | wall | cross |B| (nt) | best surr-RMSE | top hi-fi rho | multi-sol |
|---|---|---|---|---|---|---|
| 116 | 8.58x | 11.7 min | 1343 (3.2 deg) | 0.027 | 0.52 (A) | PASS (15/15 A) |
| 119 | 142x | 4.8 min | 20 (37.1 deg) | 1.65 | 33.0 (D) | FAIL (0/3 A) |

## Open / next

- **Anchor-baseline aliasing is now the binding blocker (not bracket/speed).** 119
  proved the blind+fast pipeline is fast and correct but yields garbage when the
  pool doesn't represent truth at the auto-selected anchors. Next high-leverage
  move = the s095 anchor-baseline fix: pick anchors in the unique-winding regime
  (and/or avoid over-sharp anchors that starve the cloud, e.g. 119's |B|=20) so a
  truth-near-omega candidate exists. This gates cohort viability more than density.
- Bracket width did NOT cost wall on 119 (the cloud-size constraint dominated), so
  the per-harmonic-tightened alternative (s019 harm-div bases) is lower priority
  than feared — revisit only if a seed appears where the wide window genuinely
  floods the candidate set.
- Still only 2 seeds (1 clean PASS, 1 aliased FAIL). A clean near%>0 seed beyond
  116, and a high-|w| narrow-basin seed, remain untested.
