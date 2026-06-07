---
title: "s107 — Discrimination test on seed 119: NO PHANTOMS across 1107 cross pairs, but s106's Band A also doesn't survive the production pipeline (oracle pair dropped by cross filter; polish basin ~5° in omega-direction)"
type: experiment
sources:
  - experiments/s107_discrimination_test.py
  - experiments/s107_hifi.py
  - results/s107/seed119/cross.npz
  - results/s107/seed119/polish.npz
  - results/s107/seed119/summary.json
  - results/s107/seed119/hifi_rho.json
  - results/s107/seed119/seed_quality_probe.json
  - results/s107/seed119/discrimination_plot.png
related:
  - experiments/s106_hybrid_loss_polish.md
  - experiments/s100_5step_coverage_proto.md
  - experiments/s105_pairs_to_omega_decomposition.md
created: 2026-05-28
updated: 2026-05-28
confidence: high on the no-phantoms finding (N=1107 cross pairs, 3 hi-fi confirms); high on the basin-of-convergence finding (control: same oracle pair, two seeds, deterministic); medium on generalization (N=1 seed)
---

# TL;DR
The s106 windowed-photometry polish does **not phantom** — across 1107 stratified cross-survivor pairs on seed 119, **0 reach Band A** (surrogate RMSE<0.10 mag), 0 phantoms (Band A AND polished-ω-dir > 30° off truth), and the 3 highest-promise candidates all confirm Band D in hi-fi (ρ 29.4, 42.9, 41.1 — `results/s107/seed119/hifi_rho.json`). **But the production pipeline as-is also doesn't reach Band A on this seed**, because (a) the s106 oracle-nearest pair (qa=1.02°, qb=0.68°) is **dropped by the s100 cross filter** under the [0.1, 1.6] dps bracket (the single-shoot omega is 104° off truth direction and fails the C-pass brightness check), and (b) a control run on the s106 pair from the single-shoot seed lands Band D (1.84), while only the multistart-best seed (4.25° off direction) reaches Band A (0.0300, bit-identical to the s106 published value) — so the polish basin of convergence in ω-direction is **~5°, narrow**. s106 is therefore real and reproducible, but not a robust production step under the current cross filter (`results/s107/seed119/seed_quality_probe.json`).

# What
s106 (commit 728332e) landed the first Band A on seed 119 via a windowed-photometry LM polish on the oracle-nearest cross pair. PROGRESS.md's top open crux: *does the polish discriminate (Band A only when close to truth) or does it phantom (Band A from any cross-survivor)?* s107 answers this by polishing a stratified sample of the s100 production cross survivors and asking the question across input-distance bins.

# How
Script: `experiments/s107_discrimination_test.py`.

1. **Re-cross** s100's cached `repA × repB` (2000×2000=4M attempts) through the exact `s100._cross_one_qa` filter (single finite-diff-init shoot + connectability tol 1e-3° + |ω| bracket **[0.1, 1.6] deg/s** [the physical bracket from `feedback_omega_prior_physical_bracket`, not the wider s019 bracket s100 originally used] + C-pass brightness within 0.10 mag). Pool(24). Cache `cross.npz`.
2. **Stratified subsample** by ω_seed_dir_off (bins [0,5,10,20,40,90,180) deg, ≤500 per bin) + always-include any pair with qa_off + qb_off < 6° (the s106-territory pairs). Original 27,946 survivors → 1107 polished — sized for ~15 min Pool(24) wall.
3. **Polish** each subsample pair with the exact s106 winning config: `least_squares(method='lm', max_nfev=400)` over the abc forward window (ep_a → ep_c+60 = 369 epochs), w_B=0 (s106 showed inert), q_a held at the cloud rep (FREEQA=0). Pool(24).
4. **Per-pair record** (saved to `polish.npz`): input distances (qa_off, qb_off, ω_seed_dir_off vs truth-ω at A, ω_seed_mag_dps), seed full-LC surrogate RMSE, polished full-LC surrogate RMSE + surrogate band, polished ω, polished ω direction error vs truth, polished |ω|, in-bracket flag, nfev.
5. **Hi-fi confirm** (`s107_hifi.py`, serial trimesh) the surrogate blind winner + the truth-near surviving pair + the best state-recovery (lowest polished-ω-dir off) — three candidates plus truth control.
6. **Seed-quality probe**: on the s106 oracle pair (a_idx=31, b_idx=1471), polish from both single-shoot and multistart-best seeds and compare. Tests whether s106's Band A requires multistart or merely the right pair.

Pre-launch design predictions (frozen, recorded for honesty):
- **H1 (good)**: Band A rate decreases monotonically with ω_seed_dir_off; the s106 pair lands Band A; no phantoms.
- **H2 (phantoms, fatal)**: >10% Band A rate even in [90, 180°) ω_seed_dir_off bin.

Actual result: a **third pattern** the framing didn't anticipate (see Result).

Past-error self-check (CLAUDE.md): times[0]==0 gauge satisfied in `_propagate_full` (both forward and backward legs start at t=0) and verified by `assert t_sel[0] == 0.0` for the abc window; no truth injection (distances are post-hoc labels, polish input is whatever the production cross filter supplies); BLAS thread limit 1 enforced before Pool fork; killed the first attempt at 12 min wall (projected 6.5 hr) and switched to stratified subsample, per the 2×-expected-time rule.

# Result

## Cross-step output (4M attempts, 27,946 survivors)
- **All survivors have ω_seed_dir_off > 10°** vs truth-ω direction: bin counts [0,5)=0, [5,10)=0, [10,20)=45, [20,40)=61, [40,90)=14125, [90,180)=13715 (source: `cross.npz`, computed in deep-dive). The single finite-diff-init shoot does not produce a truth-near omega from random cross-survivor q-pairs — consistent with [[s105_pairs_to_omega_decomposition]]'s hard-shoot trap.
- **The s106 oracle-nearest pair (a_idx=31, qa=1.02°; b_idx=1471, qb=0.68°) is ABSENT from the cross survivors** (verified directly). For a_idx=31 (truth-near q_a) only 17 pairs survive, with q_b ∈ [1.73°, 179°]; for b_idx=1471 (truth-near q_b) only 7 pairs survive, all with q_a ≥ 177° (antipodal). The pair (31, 1471) itself fails the C-pass brightness filter because its single-shoot ω is 104° off truth → propagated q_c is wildly wrong → predicted brightness at C misses the observed value by more than 0.10 mag.

## Polish-step output (1107 stratified pairs)

| metric | value | source |
|---|---|---|
| polishes total | 1107 | `summary.json` |
| Band A | **0 (0.0%)** | `summary.json` `bands.A` |
| Band B | 0 | `summary.json` `bands.B` |
| Band C | 0 | `summary.json` `bands.C` |
| Band D | 1107 (100%) | `summary.json` `bands.D` |
| phantoms (Band A AND pol_dir_off > 30°) | **0** | `summary.json` `n_phantom` |
| best polished full-LC surrogate RMSE | 1.4680 | `summary.json` `best_polish.pol_full_rmse` |
| best polished band | D | `summary.json` `best_polish.pol_band` |
| best polished ω-dir off truth | 58.01° | `summary.json` `best_polish.pol_dir_off_deg` |
| pairs polished within 5° of truth ω-direction | 6 | deep-dive |
| pairs polished within 10° of truth ω-direction | 12 | deep-dive |
| Wall (polish only, Pool(24), resumed from cache) | 959 s (16 min) | `summary.json` `wall_polish_s` |

Bin breakdown (ω_seed_dir_off, source: `summary.json` `bin_stats`):

| bin (deg) | N | Band A rate | min pol_rmse | min pol_dir_off |
|---|---|---|---|---|
| [10, 20) | 45 | 0.0% | 1.558 | 5.91° |
| [20, 40) | 61 | 0.0% | 1.638 | 14.84° |
| [40, 90) | 500 | 0.0% | 1.477 | 1.93° |
| [90, 180) | 501 | 0.0% | 1.468 | 2.35° |

## Hi-fi confirm (3 candidates + truth control, source: `hifi_rho.json`)

| label | surr-RMSE | hi-fi ρ | hi-fi band | pol_dir_off | qa+qb |
|---|---|---|---|---|---|
| truth | 0.000 | 0.00 | A | 0.00° | 0.00° |
| winner (min pol_rmse) | 1.468 | 29.36 | D | 58.01° | 346.45° |
| near_truth_pair (idx 18: qa=1.02, qb=1.73) | 2.145 | 42.90 | D | 94.99° | 2.75° |
| best_state (min pol_dir_off=1.93°) | 2.055 | 41.10 | D | 1.93° | 281.35° |

Surrogate D-band predictions hold up in hi-fi. No surprises in the surrogate→hi-fi mapping at this RMSE range.

## Seed-quality control on the s106 oracle pair (source: `seed_quality_probe.json`)

Same pair (qa=1.02°, qb=0.68°), same polish (abc window, w_B=0, FREEQA=0), two different seeds:

| seed source | seed dir-off | seed |ω| dps | seed full-RMSE | polished full-RMSE | polished band | polished dir-off | polished |ω| dps |
|---|---|---|---|---|---|---|---|---|
| single-shoot (what cross supplies) | 104.07° | 0.2378 | 3.7572 | 1.8372 | D | 98.12° | 0.2417 |
| multistart-best | 4.25° | 1.5018 | 0.6870 | **0.0300** | **A** | **0.57°** | **1.5025** |
| s106 reported (commit 728332e) | (multistart) | — | — | 0.0300 | A | 0.5694 | 1.5025 |

The multistart-from-s106-pair number reproduces bit-identically to the value reported in `results/s106/hybrid_polish_seed119_abc.json` (rmse 0.029957). The polish basin of convergence in ω-direction is therefore **~5°**, **narrow**: a 104°-off seed cannot escape into the Band A basin; a 4.25°-off seed can.

# Why this matters
- **No phantoms = the s106 polish is well-behaved.** The fear that a windowed-photometry objective might manufacture Band A fits from arbitrary seeds is refuted by N=1107 cross pairs and N=3 hi-fi confirmations. This is the central question PROGRESS.md asked, and the answer is clean.
- **But s106 is not a production-ready inversion step on seed 119 as-is.** Under the s100 cross filter (single-shoot + bracket + C-pass), 0 pairs polish to Band A. The architectural gap is *not* in the windowed polish (the basin works when seeded correctly, exactly reproducing s106 0.0300) — the gap is in **delivering a multistart-best seed within ~5° of truth to the polish**. Two layers cause the failure:
  1. The cross filter drops the s106 oracle pair because its single-shoot ω is 104° off → C-pass brightness fails. The architecturally-best pair is architecturally inaccessible.
  2. Even on cross-admitted pairs, single-shoot seeds are uniformly 10°+ off truth ω-direction → outside the polish basin.
- **The "discrimination" reframing.** "Does the polish discriminate?" splits into two questions:
  - Q1: *Does the polish manufacture false positives?* — **NO** (s107 answers this).
  - Q2: *Does the polish lock onto truth from any reasonable cross-survivor seed?* — **NO, basin is ~5° wide; production seeds are 10°+ off** (s107 also answers this).
- **The reproducibility check on the s106 pair (0.0300 ↔ 0.0300)** removes any residual doubt that s106's Band A was a numerical artifact. It is real; it is just behind an architectural delivery problem.

# Numbers
- 4,000,000 cross attempts → 27,946 survivors (655 s, Pool(24), `summary.json` `wall_cross_s`)
- Stratified subsample: 1107 polished (45 + 61 + 500 + 500 + 1 near-truth, source: log + design)
- Polish wall 959 s (Pool(24), median nfev 34, max 400, 32/1107 out-of-bracket, source: `polish.npz` `nfev` array)
- Band A rate: 0/1107 = 0.0% (`summary.json`); phantoms: 0/0 (`summary.json`)
- Seed full-LC RMSE distribution (surrogate, all 1107): median 5.95, max 47 (source: `polish.npz` `seed_full_rmse`)
- Polished full-LC RMSE: min 1.468, median 2.001, max 5.547 (source: `polish.npz` `pol_full_rmse`)
- Polished ω-direction off truth: 6/1107 within 5°, 12/1107 within 10°, 50/1107 within 20° (source: deep-dive on `polish.npz` `pol_dir_off`)
- Hi-fi: truth ρ 0.00, winner ρ 29.36, near_truth_pair ρ 42.90, best_state ρ 41.10 (source: `hifi_rho.json` `rows`)
- Seed-quality probe: single-shoot 104°-off → Band D (1.84); multistart 4.25°-off → Band A (0.0300); s106 reported 0.0300 (sources: `seed_quality_probe.json`, `results/s106/hybrid_polish_seed119_abc.json`)
- a_idx=31, b_idx=1471 (the s106 oracle pair indices): not in `cross.npz` `a_idx`/`b_idx`; for a_idx=31 alone, 17 b_idx survive; for b_idx=1471 alone, 7 a_idx survive (verified directly against `cross.npz`)

# Out of scope
- **Multistart-seeded polish at scale** (the natural s108): run `multistart_shoot` on each cross survivor, polish from the best root by surrogate full-LC. Each multistart ~0.27 s + ~30 root-scores ≈ 1.2 s/survivor → ~28 min on 27,946 Pool(24). Tests whether the basin-of-convergence problem dissolves under proper seeding.
- **Loosen-the-cross-filter follow-up**: what filter modification admits the s106 oracle pair? The C-pass brightness check is rejecting it. Replacing the single-shoot ω with multistart-best in the C-pass evaluation would likely admit it (since the multistart-best ω of 1.5018 dps propagates to a brightness-matching q_c).
- **Other seeds**: N=1. 116 (the easy one) and other 119 pairs untested.
- **Window sweep**: abc only; PAD sensitivity untested at scale.

# Cross-references
- [[s106_hybrid_loss_polish]] — the polish under test; this writeup confirms its Band A is real and reproducible from a multistart seed, but not from a single-shoot seed.
- [[s100_5step_coverage_proto]] — the cross filter; this writeup shows it drops the s106 oracle pair.
- [[s105_pairs_to_omega_decomposition]] — established the hard-shoot trap; s107 extends to: the trap also defines an architectural seed-quality floor that the polish basin can't escape.
- [[feedback_omega_prior_physical_bracket]] — [0.1, 1.6] dps bracket used here.
- [[feedback_dont_overclaim_from_one_data_point]] — N=1 SEED caveat (though N=1107 PAIRS).
- [[feedback_oracle_injection_taints_yield]] — guarded by: no truth in residual, distances are post-hoc labels.
- [[feedback_kill_stuck_early]] — initial 6.5 hr projection killed at 12 min; switched to stratified subsample.
