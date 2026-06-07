---
title: "Omega sign (flipped-ω) degeneracy"
type: concept
sources:
  - "raw/inversion_diagnostics/m126_wrapped/seed_033/result.json"
  - "raw/inversion_diagnostics/m126_wrapped/seed_033/polish_ckpt.npz"
  - "raw/inversion_diagnostics/m115_surrogate_pipeline/seed_033/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/batch_summary.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/result.json"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/stage_a_grid.npz"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/stage_b_polish.npz"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_012/stage_c_hifi.npz"
  - "raw/inversion_diagnostics/m127_flipped_omega/seed_033/stage_a_grid.npz"
  - "raw/inversion_diagnostics/m128_warmstart_polish/batch_summary.json"
  - "raw/inversion_diagnostics/m129_densegrid/batch_summary.json"
  - "raw/inversion_diagnostics/m129_densegrid/seed_033/result.json"
related:
  - "[[multi-solution-philosophy]]"
  - "[[twin-degeneracy]]"
  - "[[symmetry-degeneracies]]"
  - "[[basin-of-attraction]]"
  - "[[surrogate-model]]"
  - "[[m126_wrapped_pipeline]]"
  - "[[m127_flipped_omega_search]]"
  - "[[m128_warmstart_polish]]"
  - "[[m129_dense_grid_eval]]"
  - "[[m115_surrogate_pipeline]]"
created: 2026-04-16
updated: 2026-04-16
confidence: medium
---

> ## ✅ 2026-04-17 — SEED 33 HI-FI CONFIRMED ON CORRECT 1-HOUR WINDOW
>
> Option A re-run of [[m126_wrapped_pipeline]] on the correct 1-hour window (commit `b906691`) gives seed 33's flipped-ω basin `hifi_before = 0.0817` and `hifi_after = 0.0796` (essentially unchanged from the wrong-window 0.0817). **The flipped-ω degeneracy is a real physical observation-geometry symmetry on the 1-hour window, not a wrong-window numerical artefact.** Geometric findings (`q0 ≈ 98° from truth`, `ω 161.5° retrograde`, `|ω| within 0.7%`) stand. The [[m127_flipped_omega_search]] / [[m128_warmstart_polish]] / [[m129_dense_grid_eval]] thread was always correctly windowed and its REFUTED verdicts stand. Seed 33 is now the **only** break-even seed in the 11-seed wrapped-pipeline cohort — strengthening the case that flipped-ω is a distinct failure mode, not a polish-mechanics artefact.

> **2026-04-16 CORRECTION.** The original page claimed `|ω_cand| = 0.30 × |ω_true|` and `|ω| error = −70.19%`. Both are wrong — the `w_mag_err_pct_before` value in `polish_ckpt.npz` is `-0.70187456`, which is **−0.70%** (|ω_cand|/|ω_true| = 0.993, essentially unchanged), not −70%. The mechanism section that leaned on "70% ω-magnitude shrinkage compensates for a shorter cone-loop" is therefore unsupported and rewritten below.
>
> **What seed 33 actually shows:** ω_cand has the same magnitude as ω_true (within 0.7%), pointing 161.5° away (retrograde), with q0 ≈ 98° from truth. The compensating mechanism is q0 alone, not magnitude.
>
> **Inline probe [inline, this session]:** surrogate MSE at `(q0_truth, -ω_truth)` for all 11 baseline seeds is 3-10 (far from noise-floor 0.0025). Pure ω-sign flip is NOT a universal symmetry. Seed 33's valid fit at hi-fi 0.082 is entirely due to the compensating q0. See `/tmp/flipped_omega_probe_results.json`.


# Omega-sign (flipped-ω) degeneracy

## Status: OPEN — TWO confirmed cases (seed 33 narrow + seed 12 wide), mechanism axis-dependent per seed

**Confidence: medium** (upgraded from `low` after [[m127_flipped_omega_search]] confirmed seed 12).

## One-line

For at least one baseline seed (seed 33), a body-frame angular-velocity vector rotating **backwards** (antipodal to truth ω) combined with a specific compensating q0 produces a light curve whose hi-fi MSE is ≈ 0.08 (near-OK class) — observationally hard to distinguish from the truth fit.

## The finding (m126 seed 33)

Pre-polish state across all 3 DE basins (verified 2026-04-16 against `polish_ckpt.npz`):

| quantity | value | note |
|---------|-------|------|
| signed angle `(ω_cand · ω_true) / (|ω||ω_true|)` | `cos⁻¹ ≈ 161.5°` | ω rotates BACKWARDS |
| axis-angle `acos(|ω̂_cand · ω̂_true|)` | 18.46° | sign-folded report in [[m115_surrogate_pipeline]] |
| |ω| error | **−0.70%** | \|ω_cand\|/\|ω_true\| ≈ 0.993 — magnitude essentially preserved |
| q0 errors across 3 basins | 98.5°, 136.3°, 178.6° | no near-truth attractor |
| hi-fi MSE pre-polish | 0.082, 0.082, 0.258 | near-OK on 2/3 basins |

So: ω points backwards with essentially the same magnitude, no basin has a near-truth q0, yet the LC fit is near-OK. This is not noise — it is a genuine observational degeneracy for this seed's geometry/window. The compensation is carried entirely by q0 (rotated ~98° from truth).

## Why axis-angle hid this for 12 seeds of history

[[m115_surrogate_pipeline]], [[m090_robust_peak_selection]], [[m102_fullmse]] all report `w_dir_err = acos(|ω̂_cand · ω̂_true|) ∈ [0°, 90°]`. By construction this folds `θ` and `180° − θ` together. A retrograde ω shows up as `180° − θ_signed`. Seed 33's 161.5° signed becomes 18.5° axis-angle — reported as "moderate ω-direction error", masking the qualitative sign-flip.

m126 uses the signed convention (`acos((ω̂_cand · ω̂_true))`), which unmasks the flip. This is recommended as the new default for any cost-distance or basin-of-attraction analysis where the sign of ω matters (always).

## Hypothesised mechanism (revised 2026-04-16 after magnitude correction)

Over the 3600 s window with |ω_true| ≈ 0.025 rad/s, the truth attitude trajectory is `R_true(t) = R(q0_true) · exp([ω_true]×·t)` (principal-axis form; full Euler tumbling for triaxial IS-901 deviates but preserves the same structure). The body-frame sun/observer vectors at time t are:

```
k1_body(t) = R_true(t)^T · k1_J2000(t)
k2_body(t) = R_true(t)^T · k2_J2000(t)
```

For seed 33's flipped basin, `ω_cand = −ω_true · (1 − 0.007)`, so the kinematics are:

```
R_cand(t) = R(q0_cand) · exp([−ω_true · 0.993]×·t)
          ≈ R(q0_cand) · exp([ω_true]× · (−0.993 t))
```

Under the same q0, this is **time-reversed** at essentially the original rate (the 0.7% magnitude change is negligible over 3600 s — gives ≤ 0.5° drift). For this to produce the same LC, we need a q0_cand that makes:

```
R(q0_cand) · exp([ω_true]×·(−t))^T · k_J2000(t) ≈ R(q0_true) · exp([ω_true]×·t)^T · k_J2000(t)
```

i.e., the body-frame trajectories `(k1_body, k2_body)(t)` sweep an approximately LC-equivalent multiset over the window. The surrogate is a scalar function `B(k1, k2, panel, dish)` that depends only on body-frame directions — the LC can match if the time-reversed trajectory visits approximately the same `(k1, k2)` regions at the same epochs (or produces the same magnitude multiset over the window in the right temporal order).

**Candidate interpretation for seed 33:** The q0_cand ≈ 98° from truth combined with time-reversal of the tumble happens to map the body-frame observation geometry onto an approximately LC-equivalent trajectory. This is NOT a clean global symmetry — it is a geometry-specific coincidence.

**Geometric structure [inline 2026-04-16]:** The relative rotation `q_cand · q_true^-1` is 98.5° about an axis that, expressed in the truth **body frame**, is `[0.041, -0.013, +0.999]` — **2.5° off the body +Z axis**. The compensating q0 rotates the satellite by ~98° about its own body Z (up to 2.5°). This suggests IS-901's body structure has an approximate LC-symmetry under rotations about body Z combined with time-reversal — a mixed symmetry involving both spatial (body-Z) and temporal (ω sign) inversion.

**Body-Z is exactly the panel-deployment axis for IS-901** (verified from `data/models/intelsat_901/intelsat_901_config.yaml`):

- `SP_North` position `[0, 0, +9.2]` — solar panel 9.2 m along +Z
- `SP_South` position `[0, 0, −9.2]` — solar panel 9.2 m along −Z
- `AD_West` position `[−1.5, +4, 0]`, `AD_East` position `[−1.5, −4, 0]` — antennas in the Y-Z plane, perpendicular to Z
- Panels rotate about Z (rotation_axis `[0, 0, 1]`) under sun tracking

A rotation by 98° about body +Z rotates the bus-XY-plane (antennas + bus sides) but leaves the solar-panel axial positions fixed (panels stay on ±Z). Since the panels are sun-tracked (articulation removes a rotational DOF in the BRDF), their contribution to the LC is invariant under body-Z rotation at first order. The body-bus + antenna-dish combined shape has approximate 4-fold symmetry about Z (bus is roughly a box, antennas are on the ±Y faces). 90° steps approximately map the bus + dish profile onto itself; 98° is the specific angle DE converged to given the 3600 s tumble-phase alignment.

This combined with time-reversal of the tumble (ω → −ω) lets the flipped trajectory retrace an LC-equivalent sequence of bus/antenna visibility patterns. It's a mixed spatial-temporal symmetry — the kind predicted qualitatively in [[symmetry-degeneracies]].

**Why only seed 33?** A coarse body-Z angle sweep (10° resolution, -ω fixed) on seeds 0, 14, 33, 93 finds:

| seed | min MSE | best angle | mean MSE | note |
|-----:|--------:|-----------:|---------:|------|
| 0  | 2.86 | 0° | 5.06 | no compensating rotation on this axis |
| 14 | 5.41 | 80° | 7.00 | no compensating rotation on this axis |
| 33 | 2.99 | 90° | 4.52 | close to DE's 98.5° — coarse grid misses the tight basin |
| 93 | 4.76 | 160° | 8.09 | no compensating rotation on this axis |

For seed 33 the coarse sweep lands at 90° with MSE 2.99 (the actual basin at 98.5° has MSE 0.08 — the 10° grid is too coarse). For other seeds, no body-Z angle gets below MSE ~3. This suggests seed 33's compensation along body Z is specific to its tumble/observation geometry, NOT a universal body-Z symmetry. [[m127_flipped_omega_search]] (queued) will run SO(3) grid + polish on all 11 seeds to test more thoroughly.

**Crucially, this is NOT the ±X twin** ([[twin-degeneracy]]): the twin is `q_twin = q_180x · q_true` with `ω_twin = ω_true` (SAME ω). The flipped-ω degeneracy has antipodal ω (essentially exact magnitude) AND a geometrically unrelated q0.

## Seed 12 confirmation ([[m127_flipped_omega_search]], 2026-04-16) — WIDE basin, different body axis

[[m127_flipped_omega_search]] ran a 60k SO(3) grid + L-BFGS-B polish with `ω = −ω_true` across all 11 baseline seeds. Headline: 1/11 non-control seed produced a flipped-ω PARTIAL solution — seed 12.

**Verified from `seed_012/result.json`:**

- `q0_wxyz_winner = [−0.2934, +0.7807, +0.1345, −0.5351]`
- `q0_err = 140.36°` (from truth `[−0.8915, +0.2479, −0.3572, +0.1271]`)
- `ω_winner = −ω_true` exactly (magnitude preserved to machine precision)
- surrogate MSE = **0.2297** (post-polish; Stage A grid already found 0.6147 → basin is visible at 3° grid spacing → WIDE basin)
- hi-fi MSE = **0.171** — PARTIAL class (< 0.5), ~70× above σ² noise floor (0.0025)

**Body-frame compensation axis (seed 12):** `q_comp_body_truth = q0_truth⁻¹ · q0_cand` expressed in seed-12 truth body frame:

- angle = 140.36°
- axis = `[−0.848, −0.485, +0.215]`
- angle to body +Z = **77.6°** (NOT aligned)
- angle to body +X = **32.1°** (closest)
- angle to body +Y = **61.0°**

**Contrast with seed 33:** 98.5° about body ≈ +Z (2.5° off). Seed 12 is 140° about an axis closest to body +X. The two confirmed flipped-ω compensation geometries share NO common body axis. The earlier "body +Z is the shared structural axis" hypothesis (based on seed 33 alone) does NOT generalise.

**Is seed 12's flipped basin a twin-relative of its m115 basins?** No. Tested all `{±180X, ±180Y, ±180Z} × {left, right}` multiplications on each of the 3 m115 hi-fi basins — no match within 0.05 quaternion distance. Raw angular distance from m127 winner to nearest m115 basin is 61.68°. The flipped-ω winner is a **genuinely independent attractor**, not a twin.

### Seed 33 positive-control failed (search-density bias)

m127 also ran seed 33 as a positive control expecting to re-discover the m126 basin (q0 ≈ 98.5°, hi-fi 0.082). Result: **FLIPPED_FAIL** — winner hi-fi 2.578, q0_err 91.43° (a DIFFERENT basin at surr ~0.98). The m126 basin's q0-width is sub-0.001° per [[m126_wrapped_pipeline]] (`Δq0 ≤ 0.0004°` in polish), whereas the 60k super-Fibonacci grid has median spacing ~3°. The nearest top-20 grid quaternion was **32.7°** away from the known basin target; L-BFGS-B cannot jump from that gap into a 0.001°-wide basin.

**Implication:** m127's search method finds WIDE flipped-ω basins but is blind to NARROW ones. Seed 33's known basin exists — it just isn't on this grid. The 10 FAIL seeds are **inconclusive** w.r.t. whether they too might harbour narrow flipped-ω basins.

### Conclusion after m127

- **Population frequency of WIDE flipped-ω attractors: at least 1/11 seeds** (seed 12). Plus seed 33's narrow one.
- **The compensation geometry is seed-specific** — no shared body-frame axis across confirmed cases.
- The phenomenon is real and population-wide-ish (≥2/11 confirmed), but the mechanism is NOT a clean universal symmetry. It is likely a collection of geometry-specific near-coincidences, consistent with [[symmetry-degeneracies]]'s "mixed spatial-temporal near-symmetry" framing.

## Seed 46 candidate connection (check pending)

[[m126_wrapped_pipeline]] inline note: seed 46 has w_dir 6.39° (axis-angle) with 3 wrong-q0 basins. If the signed w_dir is ~173° (not 6.39°), seed 46 is another flipped-ω case. Need to inspect m115 seed_046 polish_ckpt signed w_dir (or compute it from `(ω_cand · ω_true)` directly). Tracked as an open question.

## Implications

### For cost-function reporting

Axis-angle ω-direction error is a *signed quantity masqueraded as unsigned*. Any basin-of-attraction analysis, any rank-by-ω-error plot, any "this seed has ω wrong by X°" statement should use the signed convention OR explicitly report `θ_signed ∈ [0°, 180°]` alongside. See the m126 page convention discussion.

### For multi-solution philosophy

[[multi-solution-philosophy]] argues that the algorithm should return ALL valid `(q0, ω)` below a hi-fi threshold, not a single winner. Seed 33's flipped-ω attractor hi-fi-validates at 0.082 (< 0.1 threshold). It IS a valid solution under the multi-solution framing — just not in the expected slice of parameter space. The pipeline's output for seed 33 should include this solution labelled as "flipped-ω" rather than discarded as "failure to recover truth".

### For DE search

[[surrogate-de-search]]'s DE over q0 per ω candidate cannot find the flipped ω because ω is fixed at the upstream candidate. The flipped-ω attractor exists in a different ω-slice. To find it via DE, you would need:

- Initial ω guess at `−ω_upstream` (or any retrograde candidate), OR
- 6-DOF DE (q0 + ω joint), which was shown costly in [[m114_surrogate_multistart]] but is exactly what finds these alternative attractors.

### For polish

As documented in [[m126_wrapped_pipeline]], L-BFGS polish at seed 33's flipped-ω basin moves surrogate cost down 60–68% while hi-fi goes up slightly (0.08 → 0.10–0.15). The flipped-ω attractor IS a surrogate local minimum with its own gradient-bearing neighbourhood — but the neighbourhood's gradient direction slides along a surrogate modelling-error manifold, taking the state marginally worse on hi-fi. Wrapper (`keep_min`) rescues this correctly.

## ω-sign census across all 11 baseline seeds (2026-04-16, post-m128)

Computed directly from `data/results/inversion_diagnostics/m115_surrogate_pipeline/seed_NNN/step2_hifi.npz` and `data/results/inversion_diagnostics/m046_trajectories/m046_trajectories.npz` truth ω, using `signed_w_dir = acos((ω̂_basin · ω̂_truth))`:

| seed | basin 0 signed | basin 1 signed | basin 2 signed | class |
|-----:|---------------:|---------------:|---------------:|:-----:|
| 0  | 2.90°  | 2.90°  | 2.90°  | forward |
| 6  | 3.08°  | 3.08°  | 3.08°  | forward |
| 12 | 8.01°  | 8.01°  | 8.01°  | forward |
| 14 | 0.34°  | 0.34°  | 0.34°  | forward |
| 24 | 1.93°  | 1.93°  | 0.64°  | forward |
| 27 | 3.08°  | 3.08°  | 3.08°  | forward |
| 33 | **161.54°** | **161.54°** | **161.54°** | **RETROGRADE** |
| 36 | 10.73° | 10.73° | 10.73° | forward |
| 46 | 6.39°  | 6.39°  | 6.39°  | forward |
| 74 | 4.88°  | 4.88°  | 4.88°  | forward |
| 93 | 0.49°  | 0.49°  | 0.49°  | forward |

**10/11 seeds have forward-ω m115 basins. Seed 33 is the sole retrograde case.** All 3 of seed 33's DE basins share the same retrograde ω — DE enumerates q0 attractors on the `(ω_upstream)`-slice only, it cannot change sign. The axis-angle convention in [[m115_surrogate_pipeline]]'s JSON output folds 161.54° → 18.46° — this is what concealed the retrograde ω at report time.

**Practical implication for any "flipped-ω" recovery experiment:** the warm-start geometry of `(q0_m115basin, −basin_ω)` tests "flipped-ω relative to m115's basin" — which equals "retrograde relative to truth" ONLY for seeds with forward m115 ω. For seed 33 the `−basin_ω` is forward relative to truth. See [[m128_warmstart_polish]] for the consequence on an experiment that ignored this distinction in its positive-control design.

## Open questions

- **~~Population frequency.~~** [[m127_flipped_omega_search]] attempted a 60k-SO(3) grid + L-BFGS polish with ω=−ω_true on all 11 seeds. Outcome: **MIXED / UNDERPOWERED**. Only 1/10 non-control seed (seed 12) PARTIAL'd (hi-fi 0.171). Positive control (seed 33) FAILED because its known basin's q0-width (<0.001°) is ~10000× narrower than the 3° grid spacing. So the 3-seed threshold was NOT met, but the experiment can only rule out WIDE flipped-ω basins on the FAIL seeds.
- **~~m128 test of warm-start polish.~~** [[m128_warmstart_polish]] (2026-04-16, 38.7 min with Pool(2)) ran 3-DOF L-BFGS-B at `(q0_m115basin, −basin_ω)` on 3 basins × 11 seeds. Outcome: **REFUTED on 10 valid-test seeds, INVALID spec'd for seed 33**. All 10 valid-test seeds hit FLIPPED_FAIL (best_hifi 1.42–4.73); seed 12 missed its 0.3 sanity target (actual 1.42). Seed 33's spec positive-control is structurally impossible — per the census above, its warm-start `−basin_ω` is forward-ω, not a flipped-ω test. Polish mechanics verified OK via spot-check at `+basin_ω` (re-eval 0.0825 vs recorded 0.0817 = 0.97% match). The `(q0_m115basin, −basin_ω)` warm-start is too far from any flipped-ω attractor (known or hypothesised) for L-BFGS-B to reach, confirming that gradient-based polish from forward-ω basins CANNOT enumerate flipped-ω solutions.
- **Search-density gap (partially closed by [[m129_dense_grid_eval]]; DE leg still open).** Both [[m127_flipped_omega_search]] (coarse SO(3) grid + polish) and [[m128_warmstart_polish]] (warm-start polish) fail to find narrow flipped-ω basins. Neither rules them out. Follow-up status:
  - **~~m129 [medium, ~15 min].~~** **DONE 2026-04-16 — REFUTED.** 600k super-Fibonacci SO(3) grid (~1.4° median spacing, 10× denser than [[m127_flipped_omega_search]]) on seed 33 alone. Stage A best grid surrogate 1.1800 vs m127's 1.1833 (0.3% improvement). Winner hi-fi 2.6186, classification FLIPPED_FAIL. Seed 33's known basin (q0-width <0.001°) is still ~1000× narrower than 1.4° spacing — density would need to increase by a further 1000× to put a grid vertex inside the basin. Grid-then-polish is **narrow-basin-blind regardless of density**. Confidence on this concept unchanged (medium): m129 does NOT confirm or refute basin existence, it only closes grid+polish as an enumerator.
  - **m130 [expensive ~30 min].** DE-over-q0 with ω=−ω_true, pop 200, 10 restarts/seed, all 11 seeds. Population-based search is width-agnostic. **Now required, not optional**, because grid-based enumeration has been exhausted.
- **Body-frame geometry prerequisite.** What structure in IS-901 makes flipped-ω degeneracies possible for this seed? Candidate: the solar-panel bilateralism + bus box-symmetry combine to make many reflected trajectories approximately equivalent over the BRDF. Test: re-run m127 on a less symmetric body (torus_plate test model). Deferred behind m127.
- **Observation-window duration.** Does the degeneracy persist if the window grows? Longer observations should distinguish tumble direction (because the full trajectory depends on Euler triaxial dynamics). Test: re-score seed 33 with the 6-hr extended window.
- **[inline 2026-04-16] Seed 46 check DONE — NOT a flipped-ω case.** Signed w_dir = 6.39° (same as axis-angle); ω is NOT retrograde. Census above across all 11 baseline seeds: only seed 33 has a signed w_dir > 90°. The "suspicion" from the original wiki note (based on the 70% magnitude misreading) is refuted. Data: `/tmp/flipped_omega_census.json`.
