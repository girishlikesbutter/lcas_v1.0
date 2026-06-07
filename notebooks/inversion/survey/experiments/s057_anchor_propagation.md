---
title: "s057 — anchor-propagation pilot on seed 89 (single anchor, single Δt)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s057_anchor_propagation.py
  - notebooks/inversion/survey/results/s048c_cloud_viewer/seed089/8bb9b81f1602/spread.npz
  - notebooks/inversion/survey/data/trajectories/traj_seed089.npz
related:
  - s048c — cloud viewer (substrate)
  - s057b — anchor scan
  - s057c — truth-pair ω diagnostic
  - s057d — Δt sweep
  - s057f — per-q_a consistency
  - s057g — forward-propagation discrimination (the major outcome of this series)
created: 2026-05-07
updated: 2026-05-07
confidence: medium (Δt=10 chosen analytically, not empirically — see s057d)
status: First proof of architecture concept. Concentration weak (1.79× over uniform baseline at ±25% prior). Closest-pair (q_a*, q_b*) implies ω 20.4° from truth direction — pool-discretisation noise floor at this Δt. Sets up the family of experiments s057a-i.
---

## TL;DR

User-proposed architecture: anchor at deepest |C_t| constriction (t_a=411, |C_a|=44), generate (q_a, ω) hypotheses by all-pair finite-diff with C_{t_a+Δ} at Δ=10 epochs (~72s), filter by |ω|-prior at three bracket widths (oracle ±0.5%, tight ±5%, loose ±25%). Question: do surviving ω-vectors concentrate near truth-ω-direction? Answer: weakly. Loose ±25% prior gave 294 surviving pairs out of 14872, with 2.7% within 10° of truth (vs 1.5% baseline = 1.79× elevated). The closest-pair (q_a*, q_b*) — both pool-survivors near truth — gave ω at 20.4° from truth direction with |ω| +28% high. Single-anchor signal is real but weak.

## What

First pilot of the cloud-data architecture. User proposed: pick small-cloud anchor t_a, use s055a-style |ω|-prior to filter cross-epoch q_b candidates, each (q_a, q_b) pair implies an ω-vector, look for concentration near truth.

## How

1. Smoke-test quaternion convention by finite-diff of `q_truth_t[1]`/`q_truth_t[2]` against cached `omega0_rad`. Test active vs passive vs sign-flipped variants. **Result: passive convention (unsigned), recovery err 0.755%.**
2. Anchor: t_a = argmin(n_survivors) = 411 (|C_a|=44).
3. Forward epoch: t_b = t_a + 10 (Δt = 72.1s); |C_b|=338.
4. Truth ω at t_a from finite-diff over [t_a, t_b] window: |ω|=0.241 dps.
5. All-pair finite-diff: 44 × 338 = 14872 pairs, each yielding an ω-vector via `(2/Δt)·log(q_b·q_a⁻¹)_vec` (passive).
6. Three |ω|-prior bracket settings; per setting compute angular distance of each surviving ω-vector to truth-ω-direction, plot histogram, compute concentration vs uniform-on-sphere baseline (1−cos(X°)/2 for axis-distance).

Wall: ~5s.

## Result

| Prior | n_pass | f<10° | f<30° | min ang | median ang |
|---|---|---|---|---|---|
| oracle (±0.5%) | 4 | 0.0% | 25% | 24.8° | 67.3° |
| tight (±5%) | 49 | 2.0% | 16.3% | 9.4° | 64.4° |
| loose (±25%) | 294 | 2.7% | 14.3% | 1.2° | 63.7° |
| baseline (uniform) | – | 1.52% | 13.4% | – | – |

- **truth_q_a in cloud at t_a**: TRUE (closest pool point at 2.43° from truth survived)
- **truth_q_b in cloud at t_b**: FALSE (closest pool 4.73° from truth did not pass surrogate filter; closest survivor in cloud is 6.30° away — see s057c)
- Loose ±25% concentration: 1.79× over baseline at 10° — modest signal
- The all-pair |ω| histogram is **bimodal**, with truth |ω|=0.241 sitting at a DIP between geometric noise modes (driven by random-pair-on-SO(3) statistics, not ω)

## Why this matters

This pilot established the architecture's structural shape and the convention smoke test, but the modest concentration (1.79× over baseline) showed that single-anchor + single-Δt is too noisy. Set up the question chain that s057b–i answered: where does the noise come from? (s057c, s057d), can multi-anchor help? (s057e), can per-q_a consistency rank truth? (s057f), can forward-propagation discriminate? (s057g — yes, 4400× over null).

## Numbers

- t_a = 411, t_b = 421, Δt = 72.1s
- |C_a| = 44, |C_b| = 338, total pairs = 14872
- truth |ω| at t_a = 0.241 dps
- Best pair ω-direction error: 1.2° (loose prior); 9.4° (tight); 24.8° (oracle)
- Convention: passive, recovery err 0.76%

## Artefacts

- `experiments/s057_anchor_propagation.py`
- `results/s057_anchor_propagation/anchor_propagation_overview.png` (4-panel: |ω| hist; ang-dist hist; sphere scatter; summary text)
- `results/s057_anchor_propagation/summary.json`

## Out of scope (handled by follow-ons)

- Δt choice was analytic (SNR estimate), not empirical — addressed by s057d
- Single anchor only — multi-anchor in s057e
- Single-pair finite-diff per (q_a, q_b) — per-q_a aggregation in s057f, forward-propagation aggregation in s057g

## Cross-references

- `experiments/s057b_anchor_scan.md` — find anchor pairs where truth is in cloud at both sides
- `experiments/s057c_truth_pair_omega.md` — diagnose what ω the truth-representative pair implies
- `experiments/s057d_dt_sweep.md` — empirical Δt sweep
- `experiments/s057g_forward_propagation.md` — major outcome
