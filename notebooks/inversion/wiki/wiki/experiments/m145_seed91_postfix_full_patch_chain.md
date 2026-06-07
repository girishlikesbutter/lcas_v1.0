---
title: "m145 — Seed 91 full patch chain end-to-end: NM-pool surrogate-rerank + multi_phi_top=10 + geo surrogate-rerank + m115 K=10 surr_mse → Band C, REFUTED"
type: experiment
sources:
  - "notebooks/inversion/11_casadi_formulation/m103_hybrid.py"
  - "notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py"
  - "notebooks/inversion/inspect_m103_patched.py"
  - "notebooks/inversion/append_geo_surr_mse.py"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/nm_prededup_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/multi_phi_ckpt.npz"
  - "data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/geo_ckpt.npz"
  - "data/results/inversion_diagnostics/m115_surrogate_pipeline_m048/seed_091/result.json"
  - "data/results/inversion_diagnostics/m126_wrapped_m048/seed_091/result.json"
  - "data/results/inversion_diagnostics/wrappedbest_m048_seed091/result.json"
  - "data/results/inversion_diagnostics/wrappedbest_m048_seed091_lc_compare.png"
related:
  - "[[m144_nm_pool_rerank_diagnostic]]"
  - "[[m143_seed91_postfix_generality]]"
  - "[[m141_seed6_postfix_pipeline]]"
  - "[[m139_convention_bug_fix]]"
  - "[[surrogate-rerank]]"
  - "[[upstream-redesign-6dof-surrogate-de]]"
  - "[[m115_de_bridging_radius]]"
created: 2026-04-30
updated: 2026-04-30
confidence: high
---

# m145 — Seed 91 full patch chain end-to-end: REFUTED at Band C

#m103-patch-built #q-from-w-solver-fails-post-fix #seed91-band-c #new-bottleneck-revealed

## TL;DR

Implemented the m144-prescribed m103 patch chain (`M103_NM_RERANK_BY=surr_mse`,
`M103_MULTI_PHI_TOP=10`, `M103_GEO_RERANK_BY=surr_mse`) plus the m115
consumer side (`M115_SORT_BY=surr_mse`, `M115_NUM_OMEGA_CANDIDATES=10`) and
ran end-to-end on m048 seed 91 under post-m139-fix correct truth.

**The patch chain works as designed for what m144 predicted:** the
NM-prededup 300-pool surrogate rerank surfaces 5 jointly-truth-near
candidates at top-10 surr_mse; multi-phi expansion preserves them into the
50-candidate pool; geo refinement preserves 2 jointly-truth-near
(idx=1 q0=52.16°/w=11.32°, idx=25 q0=50.20°/w=6.17°); m115 is fed 10
geo-refined ω candidates ranked by surrogate full-LC MSE, of which **5 are
within m115's empirical marginal bridging radius (w_err≤15°)** and 2 are
within reliable bridging (w_err≤5°).

**But m115's per-ω 3-DOF DE on the surrogate full-LC MSE — the
"q-from-good-ω" solver of the [[2026-04-17 Roberto report]] Step 4 — fails
to bridge q0 on every single one of those 10 ω inputs under correct
truth.** All 10 omegas × 10 starts (100 DE runs) produce 0/10 with
q0_err < 10° on each ω. The lowest surrogate MSE basin lands at q0=135°
(MSE=1.36) at the truth-near ω; the m126 6-DOF L-BFGS polish then drifts
further (winning basin 1 polish: q0=141.43°, w_dir 3.46° → 41.32°,
w_mag -98.69%, hifi_MSE=1.20). **Final ρ=4.91 → Band C, FAIL.**

The m144 hypothesis ("m103's MULTI_PHI_TOP=2 truncation is the only
structural bottleneck") is **partially refuted**: the truncation IS a
bottleneck (the patch correctly opens it), but it is not the **only**
bottleneck. Under correct truth on this seed, the surrogate-MSE landscape
at the truth-ω has a deceptive global minimum at q0_err≈135° that the DE
(rotvec ∈ [-π,π]³, 10 starts × 200 maxiter × 15 popsize) consistently
finds in preference to the truth-q0 basin. **The "DE over q0 followed by
L-BFGS-B polish converges reliably" claim from the Roberto report is a
buggy-truth artefact** — the solver itself is the new bottleneck.

## What

[[m144_nm_pool_rerank_diagnostic]] established that m103's NM-prededup
pool contains jointly-truth-near candidates at top-K reachable surr_mse
ranks on both seed 6 and seed 91, and that the structural fix is a 1-10
line patch chain. m145 is the first end-to-end execution of that patch
chain on a seed with canonical post-fix truth.

The headline question: under correct truth, does the patched pipeline
deliver a Band-A or Band-B basin on seed 91? Per the user's session
prompt, acceptance bar is ρ < 4 (Band A∪B).

## How

### Code changes

`notebooks/inversion/11_casadi_formulation/m103_hybrid.py`:

- Line 70: `GEO_TOP` exposed via `M103_GEO_TOP` env var (default 20). Caps the
  number of deduped NM candidates fed into multi-phi/geo. Must be ≥
  MULTI_PHI_TOP.
- Line 80–82: new `M103_NM_RERANK_BY` env var with values `'align'`
  (default) or `'surr_mse'`. Validates at startup.
- Line 95–98: `MULTI_PHI_TOP` exposed via `M103_MULTI_PHI_TOP` env var
  (default 2 preserves legacy behaviour).
- Line 487–522: new Step 3.25 — when `M103_NM_RERANK_BY=surr_mse`, score
  the 300 NM-prededup candidates by surrogate full-LC MSE (mirrors the
  scoring chain in `score_nm_surrogate.py`). Always populated to
  `nm_prededup_ckpt.npz` as new fields `surr_mse`, `surr_bright_mse`
  (NaN-filled when rerank is off — additive schema, does not break
  legacy readers).
- Line 537–544: dedup sort source picked by `M103_NM_RERANK_BY`. When
  `surr_mse`, sort key is `_nm_surr_mse` (NaN→+inf); when `align`, the
  legacy `refined_costs`. Multi-phi truncation `MULTI_PHI_TOP` operates on
  the resulting sort.
- Line 754–805: new Step 4.5 — when `M103_GEO_RERANK_BY=surr_mse`, score
  the geo-refined candidates by surrogate full-LC MSE. Saved to
  `geo_ckpt.npz` as new fields `surr_mse`, `surr_bright_mse`,
  `geo_rerank_by`.

`notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py`:

- Line 324: added `'surr_mse'` to `_VALID_SORTS`.
- Line 339–353: new sort branch — when `M115_SORT_BY=surr_mse`, read the
  `surr_mse` field from `geo_ckpt.npz` and rank by ascending surrogate
  full-LC MSE. Fails loudly if the field is missing or all NaN (caller
  asked for a signal that wasn't pre-computed).

`notebooks/inversion/inspect_m103_patched.py`: new helper (~140 lines) for
post-stage joint-truth-near auditing of nm_prededup, multi_phi, geo
checkpoints with surr_mse vs align ranking diff.

`notebooks/inversion/append_geo_surr_mse.py`: new ad-hoc adapter (~110
lines) that appends `surr_mse`/`surr_bright_mse` to an existing
`geo_ckpt.npz` without re-running m103 — used to bootstrap the seed-91
end-to-end test (saved 13 min of m103 wall).

### Pipeline runs

1. **Smoke test (m103-only)**:
   ```
   M103_MULTI_PHI_TOP=10 M103_NM_RERANK_BY=surr_mse \
       MICRO103_GEO_TIMEOUT_S=1200 \
       python3 notebooks/inversion/invert.py --seed 91 --traj-source m048 \
           --skip-m115 --skip-m126 --skip-lc-compare
   ```
   Wall: 793s (13.2 min). Step 3 NM 148s; Step 3.25 NM-rerank 42.5s; Step
   3.5 multi-phi 0.8s; Step 4 Geo 468.6s.

2. **Append geo surr_mse**:
   ```
   python3 notebooks/inversion/append_geo_surr_mse.py --seeds 91 --traj-source m048
   ```
   7.9s scoring; 50 geo candidates. Min surr_mse=4.38, median=6.20.

3. **End-to-end m115 + m126 + lc_compare**:
   ```
   M115_NUM_OMEGA_CANDIDATES=10 M115_SORT_BY=surr_mse \
       python3 notebooks/inversion/invert.py --seed 91 --traj-source m048 --skip-m103
   ```
   Wall: 1509.7s (25.2 min). m115 Step 1 DE: ~16 min (100 runs × ~10s);
   m115 Step 2 hi-fi: 193.6s; m126 polish: 47.1s; m126 hi-fi: 106.6s;
   lc_compare: 0.8s.

Total session wall on seed 91 (incl. one prepatch + one fresh m103 +
end-to-end): ~52 min.

## Result — patch chain stages succeeded as designed

### NM-prededup pool (n=300)

```
nm_rerank_by=surr_mse   surr_mse min=3.5533, median=7.8353  (matches m144 numbers)
jointly-truth-near (q0<60 AND w<30): 5/300
  idx=11   q0=24.22  w=28.75  surr_mse=5.5407  align_rk=13   surr_rk=26
  idx=140  q0=57.79  w= 8.38  surr_mse=4.0572  align_rk=31   surr_rk=10  ← BEST
  idx=145  q0=55.21  w= 7.68  surr_mse=5.7652  align_rk=18   surr_rk=41
  idx=173  q0=24.22  w=28.75  surr_mse=5.5406  align_rk=14   surr_rk=24
  idx=247  q0=24.22  w=28.75  surr_mse=5.5407  align_rk=15   surr_rk=25
```

Surrogate-MSE rerank surfaces idx=140 (the best joint-truth-near per
m144) at top-10 surr_mse. **Patch behaves exactly as predicted.**

### Multi-phi pool (n=50, with MULTI_PHI_TOP=10)

```
n_omega_clusters=20, joint-truth-near=1/50
  idx=1  w_rank=1  phi_rank=-1  q0=57.79  w=8.38  glint=3.7643e-01  align_rk=5/50
```

idx=140's NM candidate became multi_phi idx=1 (the original NM phi anchor
preserved + 4 new phi anchors per ω). Joint-truth-near survived at
glint_cost rank 5/50.

### Geo pool (n=50, post Step 4 L-BFGS-B refinement)

```
geo_cost min=3.24e-01, median=1.63e+00
joint-truth-near=2/50
  idx=1   q0=52.16  w=11.32  geo=1.7619  geo_rk=27/50  surr_mse=5.0568  surr_rk=5/50
  idx=25  q0=50.20  w= 6.17  geo=2.0776  geo_rk=32/50  surr_mse=6.5???  surr_rk=20/50

surr_mse min=4.38, median=6.20

Top-10 by surr_mse (M115_SORT_BY=surr_mse picks these):
  surr_rk=1   idx=23   w_rank=1  q0=109.70  w=14.55  surr_mse=4.3803
  surr_rk=2   idx=48   w_rank=9  q0=173.87  w=27.37  surr_mse=4.8610
  surr_rk=3   idx=0    w_rank=0  q0=147.02  w= 3.38  surr_mse=4.9249
  surr_rk=4   idx=21   w_rank=0  q0=128.86  w= 3.46  surr_mse=5.0257
  surr_rk=5   idx=1    w_rank=1  q0= 52.16  w=11.32  surr_mse=5.0568   ← JOINT-TRUTH-NEAR
  surr_rk=6   idx=47   w_rank=9  q0=175.88  w= 8.76  surr_mse=5.1878
  surr_rk=7   idx=30   w_rank=3  q0= 91.92  w=37.64  surr_mse=5.1981
  surr_rk=8   idx=9    w_rank=9  q0=170.05  w=10.46  surr_mse=5.2748
  surr_rk=9   idx=27   w_rank=2  q0= 50.66  w=61.67  surr_mse=5.3405
  surr_rk=10  idx=2    w_rank=2  q0= 19.75  w=62.00  surr_mse=5.3463
```

Geo step's L-BFGS-B on alignment cost demoted joint-truth-near idx=1 from
glint_rank 5/50 → geo_rank 27/50 (same pathology m144 found at the
post-multi_phi pool — alignment cost buries truth-near after refinement).
**Surrogate-MSE rerank rescues:** idx=1 lands at surr_rank 5/50, well
within m115's K=10 input. **Patch chain works through to m115's input.**

### m115 — q-from-ω solver fails on every truth-near input

Surr_mse-ranked top-10 ω inputs to m115 Step 1, with **5 inside the
empirical marginal bridging radius (w_err ≤ 15°)** and **2 inside the
reliable bridging radius (w_err ≤ 5°)**:

| omega_idx | w_dir_err | best q0_err (over 10 starts) | best surr_mse | n_below_10° |
|-----------|-----------|------------------------------|---------------|-------------|
| 0 | 14.55 | 42.58 | 1.824 | 0/10 |
| 1 | 27.37 | 177.06 | 1.844 | 0/10 |
| **2** | **3.38** | **134.82** | **1.384** | **0/10** |
| **3** | **3.46** | **135.06** | **1.362** ← min | **0/10** |
| **4** | **11.32** | **47.29** ← best q0 | **2.364** | **0/10** |
| **5** | **8.76** | **97.85** | **2.733** | **0/10** |
| 6 | 37.64 | 137.28 | 1.754 | 0/10 |
| **7** | **10.46** | **175.52** | **2.770** | **0/10** |
| 8 | 61.67 | 165.55 | 1.602 | 0/10 |
| 9 | 62.00 | 89.20 | 1.602 | 0/10 |

**0/100 DE runs landed q0_err < 10°.** The minimum surrogate MSE is at
omega 3 (truth-near, w=3.46°) with q0=135° — a stable wrong-q0 attractor
in the surrogate-MSE landscape. The closest q0 (47°) is at omega 4 with
MSE 2.36 — higher than the wrong-q0 winner.

The DE search domain is `rotvec ∈ [-π, π]³` (all of SO(3)) with sobol or
latin-hypercube initialisation, maxiter=200, popsize=15. The surrogate
landscape at this seed's truth-ω has a deceptive global minimum at
q0_err≈135° that wins the DE every time.

### m115 → m126 → wrappedbest

```
m115 best_hifi_mse  = 1.362465  (winner basin: q0=135.06°, w_dir=3.46°)
m115 omega_sort_by  = surr_mse  (the patch's M115_SORT_BY)
m115 hi-fi top 3 wall: 193.6s

m126 polish (3 basins, L-BFGS 6-DOF on surrogate cost):
  basin 0: q0  135.06° → 134.47°   w_dir  3.46° →  6.26°   surr 1.21 → 1.16  (slight drift)
  basin 1: q0  153.23° → 141.43°   w_dir  3.46° → 41.32°   surr 1.40 → 0.90  (massive ω drift!)  ← winner
  basin 2: q0  129.78° → 138.73°   w_dir  3.46° →  1.15°   surr 1.38 → 1.23
  m126 hi-fi top 1: hifi_after = 1.20273

wrappedbest selection:
  classification: FAIL
  q0_err     = 141.43°
  w_dir_err  =  41.32°
  w_mag_err  = -98.69%   (ω magnitude essentially zeroed)
  hifi_MSE   =  1.20273
  ρ          =  √(1.20/0.05) = 4.906   →  Band C (4 ≤ ρ < 8)
```

The "winner" basin is m126 basin 1 — its 6-DOF polish drove ω from
3.46° (truth-near) to 41.32° (way off truth) and its magnitude to ≈ 0
(-98.69% error). That polished state has a *better* surrogate hi-fi MSE
than the unpolished q0=135° state, but only because the surrogate at this
degenerate (q0, ω≈0) configuration coincidentally produces an LC closer to
the observed than the unpolished state did. **The hi-fi model arbitrates
when surrogate and hi-fi disagree** (per Roberto Step 5), so the final
hifi_MSE=1.20 is taken at face value — but the *state* is nonsense.

This is also a known polish pathology — m126's L-BFGS in 6-DOF can drive
ω toward zero when the surrogate landscape near q0=141° has a strong
attractor at small ω. The wrapped-best logic correctly picks the lowest
hi-fi MSE among the polished candidates, but on this seed the lowest is
the ω-zeroed nonsense.

## Why this matters

### 1. The m144 patch is genuine but insufficient

m144 said: "the m103 MULTI_PHI_TOP=2 truncation is the structural
bottleneck." m145 confirms the truncation **is** a bottleneck (the patch
correctly preserves the truth basin through the m103 stages and feeds it
to m115), but it is **not** the only bottleneck. Under correct truth on
seed 91, m115's q-from-ω DE solver is **also** broken: even given truth-
near ω, it cannot find truth-q0 because the surrogate-MSE landscape has a
deceptive global minimum elsewhere.

[[upstream-redesign-6dof-surrogate-de]] (Play 3) was demoted by m144 to
"follow-on if m103 patch fails." m145 puts it back on the table: but the
fix isn't necessarily 6-DOF DE; it could equivalently be a different
**q0**-search strategy than rotvec-bounded DE.

### 2. The Roberto Step 4 claim is a buggy-truth artefact

Per [[2026-04-17-roberto report]] §"Surrogate Integration":

> With ω held approximately fixed, q0 → LĈ(q0;ω) is smooth and cheap.
> DE over q0 followed by L-BFGS-B polish converges reliably.

Under post-m139-fix correct truth on seed 91, this is **false**. With
Omega 2 / Omega 3 inputs at w_err = 3.38°, 3.46° (textbook "good ω"),
all 20 DE runs converged to q0 = 135° instead of truth-q0 = 0°. The
"converges reliably" claim is a property of the buggy forward model, not
the algorithm. Same root cause as `feedback_bug_was_helping_m103.md` —
the bug's bright-peak displacement was creating a surrogate landscape
where DE coincidentally found truth-q0; under correct truth, that
landscape has a different (wrong) global minimum.

### 3. New q-from-ω lever candidates

Several recovery directions, all out of scope this session:

(a) **Polish from input q0** (not just from DE winner). m115 currently
    discards the input q0 (only the input ω is used). For each input
    candidate (q0_input, ω_input), run a small NM/L-BFGS polish on
    surrogate-MSE starting from q0_input *before* the DE. The
    joint-truth-near input idx=1 has q0_input = 52.16° — within m115's
    empirical reliable bridging radius from truth.
    
(b) **6-DOF surrogate DE upstream** (Play 3 proper). Use surrogate full-
    LC MSE as the global objective from grid-stage onward, no m103
    alignment-cost substrate. The surrogate landscape may still have
    deceptive minima, but a 6-DOF search has more degrees of freedom to
    escape them.
    
(c) **DE bounds tighter around input q0**. Currently rotvec ∈ [-π, π]³
    (all SO(3)). Constraining to a ball around q0_input would force DE
    to find the local minimum near q0_input rather than the global
    minimum elsewhere. Would defeat global search but might reliably
    bridge from truth-near inputs.
    
(d) **Investigate the surrogate landscape directly**. At seed 91 truth-ω,
    plot surrogate-MSE as a function of q0 over a Sobol grid in SO(3).
    If the landscape genuinely has its global minimum at q0=135°, then
    the surrogate is **wrong** for this seed, and surrogate-retraining
    becomes a real option (despite `feedback_surrogate_is_bridge_independent.md`'s
    correct claim that no retraining is needed for the *bug fix*).

### 4. m126 polish can corrupt ω under correct truth

m126 basin 1's 6-DOF polish drove ω from 3.46° (truth-near) to 41.32°
(severely wrong) AND ω-magnitude from 1.4 dps (truth) to ~0.02 dps
(-98.7% error). m126 was previously trusted under buggy truth; under
correct truth it can produce nonsense. This is a separate issue from the
DE failure but compounds it.

### 5. ρ = 4.91 is "Band C" by convention but the state is wrong

Per `feedback_observational_indistinguishability.md`, "ρ < 2 is a valid
solution regardless of twin status." But ρ = 4.91 with a state that has
q0 off by 141° AND ω magnitude essentially zero is **not** a publishable
LC fit — it's two compounding pathologies that happen to land within
Band C by hi-fi MSE. The "Band C as publishable LC fit" caveat from
`feedback_rho_band_convention.md` doesn't apply when both q0 AND ω are
catastrophically wrong.

## Numbers

| metric | value |
|---|---|
| m103 wall (patched, MULTI_PHI_TOP=10, NM rerank on, GEO timeout 1200s) | 792.7s (13.2 min) |
| m103 Step 3.25 NM-rerank wall | 42.5s |
| m103 Step 4 (Geo) wall | 468.6s |
| append_geo_surr_mse.py wall | 7.9s + ~1 min setup |
| m115 wall (K=10, sort_by=surr_mse, 100 DE runs + hi-fi top 3) | ~1100s (~18 min) |
| m126 wall (polish + hi-fi top 1) | 153.7s |
| Total invert.py wall (end-to-end m115+m126) | 1509.7s (25.2 min) |
| **m115 best_hifi_mse** | **1.362** |
| **m126 winner hifi_after** | **1.20273** |
| **ρ** | **4.906** (√(1.20/0.05)) |
| **Band** | **C** |
| **Verdict** | **FAIL** (ρ ≥ 4) |
| q0_err winner | 141.43° |
| w_dir_err winner | 41.32° |
| w_mag_err winner | -98.69% |
| n_below_10° (q0_err) across all 100 DE runs | 0/100 |
| Best q0_err across all 100 DE runs | 47.29° (Omega 4) |
| Lowest surr_mse across all 100 DE runs | 1.36 (Omega 3, q0=135°) |

## Artefacts

- `notebooks/inversion/11_casadi_formulation/m103_hybrid.py` — patched.
- `notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py` — patched.
- `notebooks/inversion/inspect_m103_patched.py` — NEW post-stage inspector.
- `notebooks/inversion/append_geo_surr_mse.py` — NEW geo-rerank backfill adapter.
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091/` — full patched-m103 outputs (geo_ckpt.npz now has surr_mse field).
- `data/results/inversion_diagnostics/m103_hybrid_m048/seed_091.prepatch.bak/` — pre-patch m103 outputs preserved.
- `data/results/inversion_diagnostics/m115_surrogate_pipeline_m048/seed_091/` — patched-m115 outputs (surr_mse-sorted ω input, K=10).
- `data/results/inversion_diagnostics/m126_wrapped_m048/seed_091/` — m126 polish outputs.
- `data/results/inversion_diagnostics/wrappedbest_m048_seed091/result.json` — final winner.
- `data/results/inversion_diagnostics/wrappedbest_m048_seed091_lc_compare.png` — LC overlay (winner has q0_err=141°, ω≈0 — visually obvious nonsense).
- `data/results/inversion_diagnostics/invert_m048_seed091.prepatch.bak/` — pre-patch invert.py outputs.

## Out of scope here

- **Seed 6 end-to-end (was deferred per session plan).** With seed 91 in
  Band C, the value of running seed 6 with the same patch chain is low —
  the q-from-ω solver fails on seed 91, and seed 6's best NM joint
  candidate has worse w_err (8.08° vs seed 91's 6.17°). Likely Band C or
  worse on seed 6. Defer to next session, after the q-from-ω solver
  question is resolved.
- **q-from-ω diagnostic (option d above).** Plot surrogate-MSE vs q0 at
  truth-ω on a Sobol grid of SO(3). Confirms / refutes "deceptive global
  minimum" hypothesis. Cheap (~1 min wall, ~500 surrogate evaluations).
  Out of scope this session.
- **Polish-from-input-q0 patch (option a above).** Add a 1-iteration
  L-BFGS polish on surrogate-MSE starting from each candidate's
  (q0_input, ω_input) BEFORE m115's global DE. If the surrogate has a
  truth-q0 basin with a non-trivial radius (q0_err < 30° empirical),
  polish from joint-truth-near inputs (idx=1, q0=52° / idx=25, q0=50°)
  should bridge. ~50 lines of m115 patch. Defer to next session.
- **Cohort regeneration**. Definitely deferred; we don't even have a
  patch chain that solves a single previously-Band-A seed under correct
  truth.

## Cross-references

- [[m144_nm_pool_rerank_diagnostic]] — the prediction that m103's
  truncation was the bottleneck. m145 confirms it is **a** bottleneck
  but not the only one.
- [[m143_seed91_postfix_generality]] — established the q0-anchor
  pathology in m103 multi-phi. m145's geo step continues that pattern.
- [[m141_seed6_postfix_pipeline]] — first post-fix pipeline run, also
  REFUTED but for a different reason (seed 6 was upstream-FAIL — m103's
  ω ranking bad). Seed 91 is downstream-FAIL — ω ranking good, q-from-ω
  solver bad.
- [[m139_convention_bug_fix]] — the propagator fix that exposed all of
  these failures.
- [[surrogate-rerank]] — the lever; m145 shows it is reliable through
  m103 stages but does not save m115's solver from a different
  pathology.
- [[upstream-redesign-6dof-surrogate-de]] — Play 3, now back on the
  table.
- [[m115_de_bridging_radius]] — the empirical bridging numbers it cites
  (q0 < 30° reliable; ω < 15° marginal) survive the bug fix in form but
  the **q-bridging at fixed ω** behaviour does not — m145 shows DE
  cannot bridge q0 even from truth-near ω under correct truth.
