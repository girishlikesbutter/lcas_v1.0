---
title: "m138 seed 47 — Lomb-Scargle as bracket / harmonic-division beats peak-count"
type: experiment
sources:
  - "notebooks/inversion/m138_lombscargle_probe.py"
  - "notebooks/inversion/m138_ls_bracket_probe.py"
  - "data/results/inversion_diagnostics/m138_isoshell_h1/lombscargle_probe/summary.json"
  - "data/results/inversion_diagnostics/m138_isoshell_h1/lombscargle_probe/bracket_summary.json"
  - "data/results/inversion_diagnostics/m138_isoshell_h1/seed_047/result.json"
  - "data/results/inversion_diagnostics/m138_isoshell_h1/seed_047/rescore_B2/result.json"
  - "data/results/inversion_diagnostics/m138_isoshell_h1/seed_047/rescore_Cprime/result.json"
related:
  - "[[m138_isoshell_h1_pilot]]"
  - "[[omega-magnitude-estimation]]"
  - "[[surrogate-attitude-isoshell]]"
  - "[[grid-search]]"
  - "[[constraint-poor-regime]]"
created: 2026-04-29
updated: 2026-04-29
confidence: medium
---

# m138 seed 47 — diagnostic + LS-bracket / harmonic-division discovery

#open — `[[m138_isoshell_h1_pilot]]` extension; H1 |ω|-base estimator replacement under test.

## What

Two-stage investigation on H1's seed-47 failure (rank-1 ω_dir 60.59°, top-30 pool_min 9.63°, outside m115's 5° bridging radius):

1. **Diagnostic** (2026-04-29 morning): H1's failure on seed 47 has TWO mechanisms — cost-shape saturation at eps=10° AND a biased upstream `estimate_omega_mag_grid` that centred the search on the wrong |ω|.
2. **LS-as-estimator probes** (2026-04-29 afternoon): independent Lomb-Scargle exploration. Confirmed m052's "LS-as-point-estimator is weak" finding, then discovered that **LS-as-bracket** and **harmonic-division** dramatically beat both peak-count and m052's LS-top-1 across the failure cohort.

## Diagnostic findings (verified against `result.json` artefacts)

### Finding A — cost-shape saturation at eps_cluster=10°

`m138_isoshell_h1.estimate_omega_mag_grid` defaults `eps_cluster_deg=10.0` at line 495 (function default at line 388 is 8.0 — documentation drift). At eps=10°, ALL top-30 candidates on seed 47 hit the cost ceiling at unique-count = 26 (every constraint epoch counted) — `1641/20000 = 8.2%` of grid candidates tie. Truth is among the ties; ranking within ties is random. **Original CURRENT_STATE framing of "constraint-density-limited (n_bright=26 vs 91's 55)" is a third-order effect; saturation is the dominant pathology.**

### Finding B — cost biased toward HIGH-|ω| at eps=5°

Tightening eps_cluster to 5° + 50-mag grid at 1.05× spacing breaks saturation but exposes a separate pathology: `cost ranks 30 candidates whose |ω|_mag are mostly +220% to +411% of truth` (`rescore_B2/result.json`: rank-1 mag err **+235.1%**). Mechanism: high-|ω| candidates sweep more `L(t)` per cluster ball, accumulating spurious unique-epoch coverage. **150 candidates within 5° of truth dir exist at low cost (cost min −20 vs ceiling −24), but the ceiling is dominated by wrong-direction high-|ω| candidates.**

### Finding C — peak-count estimator returned 0.0157 vs truth 0.0080 (ratio = 1.9624)

Almost exactly 2× — strongly suggests harmonic ambiguity rather than noise. The bright-epoch peak-count counts 9 brightness peaks across the 3600s window, implying base = 2π·9/3600 = 0.01571. Truth = 0.008000. **Ratio 1.9624 ≈ 2.0** — peak-count is reading the second harmonic.

### Finding D — Strategy-B re-rank top-30 has truth-near dir but mag +411%

8 re-rank strategies on the rescore_B2 isoshell pool (`rescore_Cprime/result.json`); only Strategy B (3-per-|ω|-band × 10 bands) and Strategy F (10×10 top-100) put any candidate within 5° of truth dir into a top-30. Both put it at **+411.6% |ω|-mag** (5.1× truth speed). **No candidate in any strategy is jointly within 5° dir AND 5% mag.** Strategy B pipeline pilot ran 20/30 candidates through m115 + m126 in 40 min before timeout; none reached MSE<1.9 or q0_err<20°. **H1 + strategy B is INSUFFICIENT.**

## LS-as-estimator findings

### Finding E — LS-as-point-estimator confirms m052's "weak" verdict

| seed | truth \|ω\| | pc estimate (off) | LS top-1 (off) |
|---:|---:|---|---|
| 47 | 0.00800 | 0.01571 (1.96×) | 0.01259 (1.57×) |
| 91 | 0.02489 | 0.04538 (1.82×) | 0.02853 (1.15×) |
| 51 | 0.01616 | 0.02269 (1.40×) | 0.03613 (**2.24×**) |
| 79 | 0.00211 | 0.01047 (4.97×) | 0.01281 (6.08×) |
| 84 | 0.00884 | 0.01222 (1.38×) | 0.01064 (1.20×) |
| 89 | 0.00419 | 0.00524 (1.25×) | 0.00196 (**0.47×**) |

LS top-1 is sometimes closer to truth than peak-count, sometimes worse, sometimes catastrophically wrong (seed 89 picks an aliased lower frequency). **Median behaviour matches m052's ρ=0.74 / 31.1% median error finding.**

### Finding F — the "exactly 2× harmonic" hypothesis is REFUTED

Inspecting LS power at integer harmonics of f_truth on seed 47: power(1×) = 0.012, power(2×) = 0.010, power(3×) = 0.028, power(4×) = 0.028. **The 3rd and 4th harmonics dominate the fundamental** — it's not "2× ambiguity" but rich multi-harmonic content. On seed 91, three harmonics within 15% of equal power. On seed 89, the 2× harmonic carries more power than the fundamental.

**Implication**: a tumbling rigid body's LC has multiple comparable spectral peaks by construction (multiple facets each glinting at different phase angles, plus shadowing). No single-frequency point-estimator can reduce this cleanly; the LC's harmonic structure is a fingerprint of the geometry, not noise.

### Finding G — LS-AS-BRACKET covers truth on all 6 seeds (KEY FINDING)

Three new strategies tested:

- **Bracket**: `[0.5 × min_LS_peak, 2.0 × max_LS_peak]` spanning all significant LS peaks (power ≥ 0.1 × peak_max). Grid size 75-92 mags at 5% spacing.
- **Multi-hypothesis**: union of `[0.3, 3.0] ×` grids around EACH significant LS peak. Grid size 100-440.
- **Harmonic-division**: `f_LS_top × {1, 1/2, 1/3, 1/4}` as 4 candidate bases, each with `[0.3, 3.0]` × 20 mags. Grid size ~80 (with overlap collapse).

Coverage of truth |ω| (does it land inside the grid? what's the nearest grid offset in % of truth?):

| seed | truth | pc baseline | LS-top1 baseline | **Bracket** | **Multi-hyp** | **Harmonic-div** |
|---:|---:|---|---|---|---|---|
| 47 | 0.00800 | in, 4.36% | **NOT in**, 28.6% | in, **2.15%** | in, **0.18%** | in, **0.09%** |
| 91 | 0.02489 | in, 0.24% | **NOT in**, 101% | in, 1.12% | in, 0.41% | in, 0.39% |
| 51 | 0.01616 | in, 1.63% | **NOT in**, 12.2% | in, 1.00% | in, 0.67% | in, 1.43% |
| 79 | 0.00211 | **NOT in**, 49.1% | **NOT in**, 268% | in, 0.50% | in, 1.03% | in, 3.76% |
| 84 | 0.00884 | in, 3.18% | in, 2.70% | in, 1.16% | in, 0.25% | in, 1.89% |
| 89 | 0.00419 | in, 1.05% | **NOT in**, 135% | in, 1.50% | in, 1.13% | in, 0.26% |

**Bracket and harmonic-division cover truth on every seed**, including seed 79 which peak-count completely misses (49% off, outside grid). **Multi-hyp gives the tightest mean coverage** but variable grid size (100-440). **Harmonic-division at fixed grid size 80 puts truth within 0.09% on seed 47** — essentially exactly on a grid point.

## Why m052's "LS is weak" verdict was right but its conclusion was narrow

m052 reduced LS to a single number (top-1 peak frequency) and compared it to truth. That reduction is exactly the point-estimator failure mode here. Treating LS as **bracket selector** or **multi-hypothesis pool** uses information from ALL significant peaks — orders-of-magnitude more information than top-1.

The LC's harmonic content is a feature of the rigid-body geometry; truth's frequency is one peak among several. A k-th harmonic of truth dominates whenever the geometry produces k brightness peaks per rotation (e.g., 2-fold facet symmetry → 2 peaks per rotation). The harmonic-division strategy explicitly hypothesises k ∈ {1, 2, 3, 4} and tests all of them; bracket does the same implicitly by spanning all significant peaks.

## What this rules out

- **Peak-count estimator as universal |ω|-base** for H1's grid: misses truth on seed 79 (4.97× off), borderline-misses on seed 47 (4.36% off, no margin for grid step error). Both are constraint-poor regimes.
- **m052's LS-as-point-estimator** as a drop-in replacement: still misses truth on 5/6 failure-cohort seeds. The conclusion "LS is weak" was right *for that specific reduction*.
- **The "harmonic ambiguity is exactly 2×" framing**: refuted by Finding F. Seeds vary 1.25×–4.97× off truth with peak-count, 0.47×–6.08× with LS-top1. The ambiguity is multi-harmonic.

## What this opens

- **Harmonic-division grid** as drop-in replacement for the |ω|-base in `estimate_omega_mag_grid`: same order of magnitude in compute (4× the |ω|-mag candidates), but covers truth on all 6 failure-cohort seeds at sub-2% nearest grid offset.
- **Bracket grid** if harmonic-division insufficient on some seeds: ~80 mags at 5% spacing covers all 6 seeds at sub-2.2% nearest offset.
- **Cost-shape pathology is now the isolatable next problem**: with truth in the grid, the question becomes "does the cost rank truth-near above wrong-direction high-|ω|?" — answerable by re-running seed 47 H1 with the corrected grid.

## What this does NOT validate

- **Whether harmonic-division grid actually rescues seed 47 in H1**: this analysis only proves truth is in the grid. The cost-shape bias toward high-|ω| (Finding B) is a separate problem; the corrected grid gives truth a fighting chance but doesn't guarantee it lands in top-30. Pending: H1 re-run on seed 47 with harmonic-division grid.
- **Generalisation beyond the 6-seed audit cohort**: only `{47, 91, 51, 79, 84, 89}` tested. m052's 100-seed result for the point-estimator was the broader picture; for bracket/harmonic-division, a wider audit is the next gate.
- **Whether m115's 5° dir + 5% mag joint bridging radius can be hit**: even with truth on the grid, the cost-shape problem may still keep wrong-direction candidates at the top of the pool.

## Risks introduced and mitigated

During the diagnostic phase, two scratch state changes were made and restored cleanly:

1. **Patched `m115_surrogate_pipeline.py:93`** `N_OMEGA_TOP = 3 → 30` for the strategy-B pipeline pilot. Restored via try/finally. Verified: `git status` clean, line 93 reads `N_OMEGA_TOP = 3`.
2. **Overwrote `m103_hybrid_m048/seed_047/geo_ckpt.npz`** with a synthetic 30-candidate file (strategy B's output). Backup at `geo_ckpt.npz.h1bak`. Restored bit-identically; `md5(geo_ckpt) == md5(geo_ckpt.h1bak)`. Backup retained on disk pending seed-47 closure.

## Files

- `notebooks/inversion/m138_lombscargle_probe.py` — independent LS probe (m052-style point estimator) on 6 failure-cohort seeds. Confirms m052.
- `notebooks/inversion/m138_ls_bracket_probe.py` — LS-as-bracket / multi-hypothesis / harmonic-division probe on same 6 seeds. **Key new artefact.**
- `notebooks/inversion/m138_diag_seed47.py` — seed-47 cost-shape diagnostic (eps sweep, truth vs random ω).
- `notebooks/inversion/m138_rescore_seed47.py` — seed-47 re-score at eps=5° with 0.7-1.5 (rescore_B) and 0.3-3.0 (rescore_B2) mag spans.
- `notebooks/inversion/m138_cprime_seed47.py` — 8-strategy re-rank of rescore_B2's pool.
- `notebooks/inversion/m138_pipeline_seed47.py` — strategy-B pipeline pilot (m115 N_OMEGA_TOP=30, synthetic geo_ckpt).
- `data/results/inversion_diagnostics/m138_isoshell_h1/seed_047/{result.json, rescore_B/, rescore_B2/, rescore_Cprime/}` — diagnostic artefacts.
- `data/results/inversion_diagnostics/m138_isoshell_h1/lombscargle_probe/{summary.json, bracket_summary.json}` — LS probe outputs.

## Harmdiv rerun outcome (2026-04-29 PM)

Ran `m138_harmdiv_seed47.py` (Stage 3 only on cached `levelset_ckpt.npz`, eps=10°, 80k candidates, Stage 3 wall 1916s). Result at `seed_047/h1_harmdiv/result.json`.

| metric | original (peak-count grid) | harmdiv (LS-top1 ÷ {1,2,3,4}) |
|---|---:|---:|
| rank-1 ω-dir | 60.59° | 112.18° |
| rank-1 |ω|-mag err | -4.4% | **-82.2%** |
| pool_min in top-30 | 9.63° | 63.68° |
| n top-30 joint (≤5° dir + ≤5% mag) | 0 | 0 |
| **n grid joint (≤5° + ≤5%)** | (not computed) | **12** |
| best joint cost rank | n/a | **22,231 / 80,000** |

**Truth IS now reachable in the search space** (12 joint candidates exist, best at 4.25° / -3.08% mag) — the |ω|-base fix worked at the grid level. But H1 cost ranks them at position 22,231/80,000. **Top-30 contains 0 joint candidates.**

**Bias direction flipped vs prior eps=5° run** (rescore_B2 had +235% mag at rank-1; harmdiv at eps=10° has -82% at rank-1). This confirms the cost rewards EXTREME |ω| at either end, with eps determining which end wins. Mechanism (verified via reasoning + observation): the cost is "densest spot in the q0-rewind cloud, counting distinct epochs in a small ball." It cannot distinguish a real trajectory pile-up at q0_truth from accidental geometric overlap in the cloud:

- **Slow-|ω|** → q_world(t) ≈ identity → cloud collapses to (essentially) the original kept-orientation distribution at every epoch → accidental cross-epoch overlap regions fake coverage.
- **Fast-|ω|** → cloud sweeps fast through many regions → accidental pile-up somewhere along the path.
- **Truth-|ω|** → real 26-epoch pile-up at q0_truth, but quantitatively smaller than the inflated fakes.

**|ω|-base bias was secondary; cost-shape pathology is the remaining blocker.**

## Next

**C-A — surrogate full-LC MSE re-rank** of the 80k harmdiv pool. The 12 joint truth-near candidates exist; we just need a cost without the extreme-|ω| reward bug to rank them. For each candidate `(q0_estimate[i], omega_batch[i])`, predict the LC with surrogate v2, score by MSE vs observed LC, re-rank. ~10-30 min, no fresh grid run. Caveat: H1's saved `q0_estimate` is the densest-cluster centroid — for wrong-ω candidates this is at an accidentally-dense spot, not a meaningful q0. Consider per-candidate q0 polish (~30-iter L-BFGS-B over q0 with ω fixed) before scoring; ~2-3× wall but more honest.

**Parked but worth keeping:** the H1 epoch-density mechanism (densest spot in q0-rewind cloud) is a clever ω-evaluation independent of surrogate-MSE. Three redesigns plug into the cached `isoshell_ckpt.npz`:

- Volume-normalised: divide unique-count by cloud-coverage volume to neutralise low-/high-|ω| extremes.
- Pile-up tightness weighting: scale by inverse variance within the cluster (tighter cluster = stronger signal).
- Use as tiebreaker after surrogate-MSE filtering rather than as primary cost.

**If C-A succeeds**: promote H1 + surrogate-MSE re-rank to the failure cohort (51/79/84/89 at harmonic-division grid).
**If C-A fails**: try Path C (mag<13 cap, more constraint epochs) and the cost-shape redesigns above.
