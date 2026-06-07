---
title: "s017 — hi-fi ρ-band of s016c_prime cohort-selector winners (multi-solution acceptance test)"
type: experiment
sources:
  - "results/s017/rho_top3.npz"
  - "results/s017/summary.json"
  - "results/s017/run.log"
related:
  - "[[s013_rho_band_validation]]"
  - "[[s014_cohort_rho_band]]"
  - "[[s014b_n_rotations_analysis]]"
  - "[[s015_joint_q0_omega_pilot]]"
  - "[[s016c_prime_fresh_sobol_fixed_omag]]"
created: 2026-05-01
updated: 2026-05-01
confidence: high
---

## TL;DR

**3/7 cohort-selector winners hi-fi-validate as Band A∪B (seeds 13, 41,
79); 4/7 fall in Band C∪D (seeds 6, 28, 49 → D; seed 91 → C). Threshold
was ≥4/7. DECISION: C_PRIME_KILLED — multi-solution reframe rejected on
the s015/s016c_prime architecture.** Top-3 per seed adds zero coverage:
no rank-2/3 candidate achieves Band A∪B on a seed where rank-1 misses.

Pre-run predictions held tightly (predicted 3/7; got 3/7; per-seed bands
matched on all 7 seeds).

**Notable secondary finding:** hi-fi/surrogate MSE ratio in this off-
truth-ω regime is **3-5× wider** than s014's truth-ω non-basin reference
(p10-p90 [0.983, 1.027] vs s014's [0.998, 1.006]). Per-seed argmin order
is still preserved (cohort-selector trustworthy at the rank level), but
absolute MSE has more spread when ω is far from truth. Worth flagging
for future experiments that lean on the surrogate-as-hi-fi-proxy claim.

## What

Hi-fi ρ-band classification of the top-3 lowest-surrogate-MSE candidates
per seed in s016c_prime (7 seeds × 3 candidates = 21 hi-fi renders).

The s016c_prime experiment closed Option C **structurally for truth
recovery** (0/7 seeds in loose basin even with PERFECT ω-mag on seed 28).
Its cohort-selector winners landed at surrogate MSE 0.005-0.083 mag² on
seeds 13 / 41 / 79 / 91 — Band-A territory by s014's surrogate ≈ hi-fi
non-basin proxy. If hi-fi confirmed, s016c_prime's "failure" reframed to
"valid multi-solution candidates found" under the survey's user-confirmed
ρ < 4 acceptance bar. This was the decisive test of that reframe.

## How

1. Load `results/s016c_prime/runs.npz`, group by seed.
2. Per seed, sort rows by `final_mse` ascending and take top-3.
3. For each candidate, render hi-fi LC via `lib.hifi_render.render_hifi`
   from converged (q0_final_wxyz, omega_final_rad).
4. Compute MSE vs cached `mag_hifi_truth`, ρ = √MSE / 0.05, band per
   `rho_band` helper.
5. Pool(8) workers; per-worker context cache; `torch.set_num_threads(1)`
   per the s011 fix.

## Predictions (pre-run)

Using s014's "non-basin: surrogate ≈ hi-fi to <1%" proxy:

| seed | min surr MSE | predicted ρ | predicted band |
|------|--------------|-------------|----------------|
| 6    | 1.250        | 22.4        | D              |
| 13   | 0.0045       | 1.34        | A              |
| 28   | 5.388        | 46.4        | D              |
| 41   | 0.00587      | 1.53        | A              |
| 49   | 3.153        | 35.5        | D              |
| 79   | 0.0107       | 2.07        | B (boundary)   |
| 91   | 0.0825       | 5.74        | C              |

**Pre-run yield prediction: 3/7 in Band A∪B (seeds 13, 41, 79).** Right
below the ≥4/7 threshold for C' viability. Pre-registered: a single seed
flipping from C → B (e.g. seed 91 if surrogate is mildly pessimistic
here) flips the decision.

## Result

**Wall: 195.8 s (9.3 s/render avg, Pool(8)).** Faster than s013/s014's
~73 s/render — consistent with the smaller batch using fewer per-worker
context-build cycles (only 7 ctx builds for 21 renders).

### Per-seed cohort-selector winner (rank-1)

| seed | surr MSE | hi-fi MSE | ratio | ρ | band | q0_err | twin_err | ωd_err | ωm_err |
|------|----------|-----------|-------|---|------|--------|----------|--------|--------|
| 6    | 1.250    | 1.254    | 1.003 | **22.39** | D | 87.04° | 163.5° | 131.87° | -2.68% |
| 13   | 0.00447  | 0.00462  | 1.035 |  **1.36** | **A** | 152.56° | 63.4° | 96.01° | -2.25% |
| 28   | 5.388    | 5.388    | 1.000 | **46.43** | D | 132.35° | 141.6° | 99.08° | -0.01% |
| 41   | 0.00587  | 0.00581  | 0.990 |  **1.52** | **A** | 172.85° | 14.8° | 122.13° | -1.36% |
| 49   | 3.153    | 3.155    | 1.001 | **35.53** | D | 100.04° | 115.8° | 50.34° | -8.15% |
| 79   | 0.0107   | 0.0104   | 0.975 |  **2.04** | **B** | 83.42° | 97.1° | 79.29° | -4.67% |
| 91   | 0.0825   | 0.0848   | 1.027 |  **5.82** | C | 127.10° | 179.5° | 73.17° | +0.20% |

### Decisive counts

- **WINNERS Band A: 2/7** (seeds 13, 41).
- **WINNERS Band A∪B: 3/7** (seeds 13, 41, 79). **Threshold was ≥4/7.**
- **SEEDS with ANY top-3 candidate Band A∪B: 3/7** (top-3 doesn't help —
  no rank-2/3 candidate flips a seed's verdict).
- Per-seed best-rank-band-in-topK: only seed 91 has a rank-2 better than
  rank-1 (ρ 5.80 vs 5.82, both Band C).
- Predictions vs measurement: matched on all 7 seeds.

### Decision

**C_PRIME_KILLED.** <4/7 winners hi-fi-validate as Band A∪B. The
multi-solution reframe is rejected for the s015/s016c_prime joint-search
architecture as configured (random ω-init → ω-mag harvest → fresh
Sobol(q0×ω-dir) at fixed ω-mag, N=64).

### Hi-fi / surrogate MSE ratio

| stat | s017 (off-truth-ω, n=21) | s014 reference (truth-ω non-basin, n=540) |
|------|--------------------------|--------------------------------------------|
| median | 1.000 | 1.000 |
| p10    | 0.983 | 0.998 |
| p90    | 1.027 | 1.006 |
| min    | 0.955 | — |
| max    | 1.037 | — |

**Ratio band ~3-5× wider in this regime.** Seeds with widest ratios:
13 (1.035), 41 #2 (0.955), 79 (0.975), 91 (1.027). Per-seed argmin order
preserved (Spearman effectively 1.0 within seed-3-candidate windows),
so the cohort-selector still picks the right candidate by surrogate MSE.
Implication: when reporting absolute hi-fi MSE from surrogate values
in off-truth-ω regimes, treat the prediction as ±3-5% rather than
s014's ±1%.

### Per-seed top-3 detail (relevant items)

- **seed 6**: rank-1 surr 1.250 → hi-fi 1.254 (D). All 3 top
  candidates Band D (ρ 22-29). ω-mag harvest -2.7% — narrow miss but
  surrogate plateau at ~1.2-2.1 mag² for the entire seed-6 N=64 run.
- **seed 13**: **rank-1 hits Band A** (ρ 1.36) at q0_err=152.6°,
  ω_dir=96°, |ω_mag|=2.2%. Class_2 multi-solution attractor (q0 in
  [30°, 150°] generalised — 152.6° at the boundary). Rank-2 is
  Band C (ρ 4.5), so seed 13 has only one near-Band-A multi-solution
  basin in top-3.
- **seed 28**: rank-1 surr 5.388 → hi-fi 5.388 (D, ρ 46.4) AT PERFECT
  ω-mag (-0.01%). Joint Sobol(q0)+Sobol(ω-dir)+LM at N=64 misses both
  truth basin AND any low-MSE multi-solution. Confirms s006's
  prediction: seed 28's truth basin (~2°) is unreachable from generic
  joint-search ICs even with perfect ω-mag.
- **seed 41**: TWO Band A candidates in top-3 (ρ 1.52 + 1.94). Rank-1
  q0_err=172.85° with twin_err=14.83° — **near-twin** (180°-flip
  q0 about a non-X axis, since s011/s005 found 0/640+50 strict twin
  recoveries). The same multi-solution attractor s014b classified as
  class_3 (near-180° flip).
- **seed 49**: rank-1 surr 3.153 → hi-fi 3.155 (D, ρ 35.5). All 3
  candidates Band D. Worst harvest in the set (ω_mag -8 to -9%);
  unsurprisingly does not bridge.
- **seed 79**: rank-1 hits Band B (ρ 2.04, just above Band A
  threshold). q0_err=83°, ω_dir=79°, |ω_mag|=4.7%. The seed-79
  candidate sits 0.04 above the A↔B boundary — would have been Band A
  if the surrogate ratio had been ~1% lower (it was 0.975, hi-fi mse =
  0.0104 vs surr 0.0107). Rank-2 is Band C — single Band-B basin.
- **seed 91**: rank-1 ρ=5.82 (C), rank-2 ρ=5.80 (C). Two clusters
  near (q0 127-138°, ω_dir 73°, |ω_mag|<1%). Class_2 multi-solution
  attractor BUT at high n_rot (14.3) — broadens s014b's "low-rot
  causes class_2" narrative. Seed 91 was Band A under truth-ω in
  s011/s014; under joint search with imperfect ω-mag, the best
  attainable is Band C.

## Why this matters

**Closes the s015/s016/s016c_prime chain.** The joint-search architecture
delivers production-quality candidates on 3/7 (43%) of the pilot. That
is not enough for cohort production. Three forward paths:

1. **S016-A — ω-grid stratification.** Replace random ω-init / ω-mag
   harvest with structured ω-grid + per-cell q0-Sobol. The s003-measured
   truth-ω tube (~1° dir × ~5% mag) is sub-cell at any feasible grid
   density, but coarse adaptive (30° dir × 0.2 dps mag pass-1, refine on
   low-surr-MSE cells) was already sketched in PROGRESS as the natural
   pivot. **Estimated cost: ~4 hr/seed pass-1 at Pool(8); 30-100k LM runs
   per seed for full refine.**

2. **S017-density — re-test C' at higher Sobol density (N=512+).** Three
   of the four Band C∪D seeds may be density-recoverable (seeds 6, 91
   have surrogate MSE 1.25 / 0.083 best at N=64; seed 49 has bad ω-mag
   harvest; seed 28 is structurally unreachable per s006). 5x compute
   over s016c_prime per density step (~6 hr total Pool(8)). Riskier than
   S016-A; doesn't address seed 28.

3. **Hierarchical / basin-hopping.** Use the s016c_prime cohort-selector
   winners as starts for basin-hopping or simulated annealing in (q0, ω)
   joint space. Compute uncharacterised; conceptually plausible.

**Most informative next experiment is the S016-A coarse adaptive design.**
S017-density is a cheap consolation prize that doesn't add architectural
information. Hierarchical search is overdue but premature — first show
ω-grid is needed.

**Second-order finding load-bearing for future scoring:** the surrogate ↔
hi-fi MSE ratio is regime-dependent. s014 measured it on truth-ω LM
landings; s017 measures it on off-truth-ω LM landings; they differ by
3-5× at the p10/p90 quantiles. Future cohort scans that report Band
yield from surrogate MSE alone (without hi-fi rerank) need to budget for
this widening. The cohort-selector argmin is preserved either way (rank
correlation within seed × top-K is effectively 1.0).

## Numbers

- 21 hi-fi renders × 9.3 s/render Pool(8) wall = 195.8 s total.
- Yield: 3/7 winners Band A∪B; 2/7 Band A; 4/7 Band C∪D.
- Per-seed Band: 6=D, 13=A, 28=D, 41=A, 49=D, 79=B, 91=C.
- Hi-fi/surrogate ratio top-K stats: median 1.000, p10 0.983, p90 1.027,
  min 0.955, max 1.037 (s014 truth-ω non-basin reference: median 1.000,
  p10-p90 [0.998, 1.006]).
- Predicted-vs-measured: 7/7 bands matched.
- Best multi-solution candidate per seed: seed 6 ρ=22.4 (D), seed 13
  ρ=1.36 (A), seed 28 ρ=46.4 (D), seed 41 ρ=1.52 (A), seed 49 ρ=35.5
  (D), seed 79 ρ=2.04 (B), seed 91 ρ=5.80 (C).

## Artefacts

- `experiments/s017_hifi_rho_band_s016c_prime.{py,md}`
- `results/s017/{rho_top3.npz, summary.json, rho_top3_per_seed.png,
  surrogate_vs_hifi_mse.png}`
- `results/s017_run.log` (gitignored — too verbose for git, but mirror
  of stdout)

## Out of scope

- Hi-fi ρ-band of all 960 s016c_prime LM landings. The cohort-selector
  outputs one per seed; top-3 was enough to decide the multi-solution
  reframe.
- S016-A architecture design. Pending; this experiment is the gate that
  makes it the priority.
- Re-test C' at higher density. Optional consolation pre-S016-A; not
  load-bearing.
- Production cohort scan (seeds beyond pilot 7).

## What s017 does NOT decide

- Whether the surrogate ↔ hi-fi ratio widening is permanent at off-
  truth-ω or specific to this batch (n=21 is small). Cohort-scale
  measurement would clarify.
- Whether seeds 6 / 91 are density-recoverable (could be tested with
  s017-density; seed 28 already known unreachable per s006).
- Whether class_2 multi-solution attractors at high n_rot (seed 91)
  are systematic or seed-91-specific. Would broaden s014b's mechanism
  story; needs more high-rot seeds with imperfect ω-mag harvest.
- Whether the 3 Band-A∪B winners (seeds 13, 41, 79) are at structurally
  distinct multi-solution attractors or share a class. Seed 41 rank-1
  has twin_err 14.8° (near-twin); seed 79 has twin_err 97° (not twin);
  seed 13 has twin_err 63° (not twin). Mixed.

## Cross-references

- `s014_cohort_rho_band.md` — surrogate↔hi-fi ratio reference (truth-ω).
- `s014b_n_rotations_analysis.md` — multi-solution attractor classification.
- `s016c_prime_fresh_sobol_fixed_omag.md` — source experiment.
- `concepts/observational_indistinguishability.md` — multi-solution
  acceptance philosophy.
- `concepts/rho_band.md` — band definitions.
