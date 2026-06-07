# 13_clean_slate_omega — FINDINGS

> Rolling findings for the clean-slate ω-recovery series. Three sub-experiments,
> one shared lib. Each sub-agent appends its own block below. Main-context
> Claude synthesises cross-idea comparisons at session close.

## Framing

Given hi-fi `mag_hifi(t)` and inertial observation geometry (sun/obs/sat J2000
positions over 3600 s, 500 epochs), recover angular velocity ω. The second-half
"q given ω" pipeline is considered solid and is NOT in scope here. Dataset:
100 m048 seeds, |ω| ∈ [~0.5, 15] dps, phase angles 25–75°.

v2 surrogate (`~/surrogate_model/surrogate_model/`) is the forward model for
both candidate scoring and (in Idea 3) training-data generation. Shared lib
at `lib/` is the single source of truth for data loading, forward, scoring.

Residual thresholds (shared): RESIDUAL_MSE_TIGHT = 0.005 mag² → OK;
RESIDUAL_MSE_GATE = 0.02 mag² → PARTIAL. All candidates under GATE are kept
(`feedback_multi_solution` philosophy).

---

## Idea 1 — Spectral recovery (a_spectral/)

### 2026-04-18 — oracle-q0 S² sweep (a2), 2562-dir Fibonacci grid, 5 pilot seeds

Premise: assume |ω| known from the LC spectrum (a1 passed) and ω_direction
unknown. Score each of 2562 directions with the ORACLE q0_true. If the
direction axis is the only unknown, we'd expect a sharp minimum at the truth
direction.

**Result — unambiguous failure of the grid premise**: every pilot seed has
**0 / 2562** directions under the GATE (0.02 mag²).

| seed | |ω| dps | phase mean | best MSE | dir-err @ best | n_under_GATE |
|------|---------|-----------|----------|----------------|--------------|
| 000  | 1.449   | 27.7°     | 2.25     | 9.6°           | 0            |
| 023  | 0.965   | 30.4°     | 1.33     | 10.3°          | 0            |
| 049  | 1.110   | 38.9°     | 1.69     | 4.7°           | 0            |
| 069  | 1.126   | 67.7°     | 2.26     | 5.0°           | 0            |
| 081  | 1.176   | 45.6°     | 1.36     | 2.0°           | 0            |

Direction resolution: even the grid direction *closest* to truth (1–2°) has
MSE 1.3–4.0 — 65–200× the GATE. Reason: over T=3600 s with |ω|~1 dps the
total swept angle is ~3600°; a 2° direction perturbation compounds into
~72° of accumulated attitude error by the last observation.

**Landscape**: ~50 local minima per seed (within 15° neighbourhood on S²).
Among the top-10 lowest-MSE directions per seed, many are 100–175° from truth
(e.g. seed 000: 9.6°, 1.9°, 8.7°, 19.9°, 31.4°, **171.1°**, 35.3°, **144.5°**,
42.1°, **110.2°**). This confirms the landscape genuinely is multi-solution
even under oracle q0 + oracle |ω|.

**Consequence for Idea 1 as planned**: the 2562-point Fibonacci grid is
under-resolved by roughly 10–100× for the GATE threshold. Densifying by 100×
(→ ~250k directions) is feasible (~1 CPU-hour per seed at 150 ms/eval,
Pool 4), but the fundamental conclusion stands: pure S²-grid-then-select
is compute-inefficient vs a gradient method traversing the basins directly
(Idea 2).

**Joint-q0 diagnostic (seed 000, 642-dir × 24-q0 grid, 414 s wall)**: also
**0/642 under GATE**. Joint best: idx 188, MSE 3.19, dir-err 24.9°. The
same-grid oracle-q0 re-run for apples-to-apples got MSE 3.13 at dir-err 1.3°.
Conclusion: varying q0 does NOT rescue a bad ω direction — the 24-point SO(3)
q0 grid is too coarse and/or the q0–ω coupling at wrong direction has no
descent direction on this grid. Grid-only search, even joint, is stuck.

### Status verdict (pending joint-q0): Idea 1 as a grid sweep is dead in its
current form. The right downstream experiment is either (a) a very dense grid
around spectrally-constrained rings, or (b) use the S² MSE landscape as a
warm-start library for Idea 2. Both follow-ups cost << than scaling Idea 1
to the ~250k-direction density the GATE requires.

---

## Idea 2 — Differentiable inversion (b_differentiable/)

### 2026-04-18 — parity checks (b1, b2) ✓, Adam multi-start (b3) in progress

**b1 — torch v2 surrogate matches numpy**: log(φ) parity max |Δ| = 1.02e-14,
mag parity max |Δ| = 2.66e-14 on seed 000, float64. Gradients w.r.t. (k1, k2,
panel, dish) all finite. Verified single-forward time ~90 ms vs numpy 155 ms.

**b2 — torch RK4 propagator matches scipy DOP853**: on seed 000 truth (q0,
omega0), torch propagate_euler_torch with substeps_per_obs=16 gives k1/k2 body
vectors matching the saved m048 truth arrays to 7e-9 (vs 5.5e-16 for the scipy
reference — the 7e-9 is from RK4 truncation, not a convention bug). Gradients
w.r.t. q0 and omega0 finite.

**b3 — Adam multi-start joint (q0, ω)**: 16 random starts + 1 truth-init start
per seed, batched in 2×8 Adam runs, lr=3e-2, 150 iter, substeps_per_obs=4.
Pilot seeds: 0, 49, 81.

Smoke-test (4 starts, 10 iter) on seed 000 already reproduced the multi-solution
landscape: truth-init landed MSE 2.6e-4 (under TIGHT) after 1 step; the other
3 random starts diverged to MSE 4.6+ in 10 iter. Multi-solution philosophy
validated — cold-start Adam converges to the basin nearest its initialization,
so many-start coverage matters.

*(Full results filled in when b3 run completes.)*

---

## Idea 3 — Learned inverse MDN (c_learned_inverse/)

*Sub-agent appends findings here.*

---

## Cross-idea synthesis

### 2026-04-18 — Three-idea framework abandoned, anchor-seeded search adopted

All three ideas were either empirically dead (Idea 1 grid sweep: 0/2562 under GATE with oracle q0 across all pilot seeds) or superseded before completing at scale (Idea 2 random-start Adam on 6-D with ~50+ basins — statistically ill-posed with 16 starts; Idea 3 MDN — deferred). The session pivoted mid-run to a cleaner formulation.

### 2026-04-18 — The actual working frame: per-epoch 3-DOF search + ω factorization

See new series `d_per_epoch_search/`. Core observation: at each epoch t, specifying `(k1_body, k2_body)` consistent with the epoch's known phase angle fully determines the attitude q(t) — so the search at ANY single epoch is 3-DOF on SO(3), not 6-D.

**Stage 1 on seed 0 validated.** Running the dense 3-DOF search at the 30 brightest epochs (batched v2 over 3.24M samples, 228 s wall, tol 0.15 mag):

- The 2 brightest glint peaks (t=400, t=313) give tight candidate sets (|C|~130-150) containing truth (1-4° from some candidate).
- Mid-brightness epochs (mag 7-10) are ~50/50 — tight |C| does NOT guarantee truth; gradient of the manifold is steep so the grid can miss truth's neighbor while accidental brightness-matches pass.
- Dim epochs (mag > 11) almost always contain truth but with |C| of thousands — useless as anchors.

**Key design note on multi-anchor "intersection"**: does not work as a pure attitude-set operation. Each `C_t` is attitudes AT time t; different epochs have different truth attitudes (satellite rotates). Multi-anchor consistency is necessarily trajectory-level: ∃ (q₀, ω) such that `propagate(q₀, ω, t_i) ∈ C_{t_i}` for all anchors. That requires ω — belongs in stage 2.

### 2026-04-18 — Phase-invariant manifold: lofi ≡ hi-fi

Ported the original lofi `isoshell_viewer.py` to use v2 with k1=k2=PAB (`lib/isoshell_viewer_v2.py`). The output manifold topology is visually identical to lofi. Reason: when k1=k2, every shadowed facet is also back-face-culled (same ray test), so shadow-awareness collapses into visibility. v2 can produce a phase-invariant manifold, but that manifold contains no information that the lofi one didn't already have.

**Conclusion**: if you need shadow-aware hi-fi information, you need phase-awareness, which needs azimuth-around-PAB, which is the unknown we're solving for — circular at anchor-extraction time. The d_per_epoch_search approach (which does NOT use a manifold at all — it just queries v2 at every candidate (k1, k2) pair directly at the epoch's true phase angle) sidesteps this entirely.

## Stage 2 (deferred to next session)

For candidates at the tightest anchor(s): search ω grid (|ω| from spectral + S² direction), back-propagate to q₀, score full LC residual with v2, GATE-filter. Design in `CURRENT_STATE.md`.

---

## e_rescore — 2026-04-18 (session reorient)

Session goal: use reliable "stage 2" (m115 q₀ DE given ω) and push forward. Horizontal characterisation across 8 Phase-B m048 seeds {23, 24, 28, 49, 69, 81, 90, 91}.

**e1 — m103 pool re-ranking** (`e1_rescore_m103_pools.py`): For 8 seeds, compute surrogate-LC MSE for every candidate in m103's final pool (geo_ckpt: 023/024/049/081/090/091; multi_phi_ckpt: 028/069). Re-rank by MSE. Truth-pool containment:
- **ω-IN-POOL** (minW<5°): 024, 049, 081, 091
- **ω-FRINGE** (5-15°): 023, 090
- **ω-OUT-OF-POOL** (>15°): 028 (30.8°), 069 (28.8°)
Surrogate-MSE ranking is competitive (top-3 in 4/8) but not a universal fix; it doesn't rescue out-of-pool seeds.

**e2 — decision table** (`e2_summary_table.py`): formalises the containment buckets above.

**e3/e3b — cost-surface basin width** (`e3_truth_and_s2_grid.py` + `e3b_basin_width.py`): 300-dir Fibonacci S² at truth-|ω|, q₀ fixed at truth. All 8 seeds show surrogate-MSE at truth is tiny (0.0002–0.0009 mag²); no grid dir beats truth. Basin width θ₅₀ ≈ 4.7–7.7° for most seeds; **seed 028 anomalous**: θ₅₀=176° (basin collapses to a needle).

**e5 — joint 7-DOF NM polish of m103 pool** (`e5_polish_m103_pool.py`): For each seed's 26 candidates, polish (q₀, ω) via Nelder-Mead on surrogate-LC cost. Outcomes per seed:
- 049 (easy): best cost 0.16, w_dir 0.75° — truth landed. ✓
- 023, 024, 081, 091: NM *finds* truth-close polished candidate (w_dir<1.5°) but **ranks a worse-ω candidate top** — joint 7-DOF NM gets q₀ stuck even when ω converges.
- 023 specifically: all 26 polishes *drift away* from the 5° init to 24° (basin θ₅₀ narrower than init).
- 028, 069: no polish reaches truth basin.

**e7 — q₀-only NM polish with truth ω held** (`e7_q0_only_truth_omega.py`): For 4 seeds {23, 28, 69, 81}, 10 random q₀ starts × maxiter=40. Finds truth/twin:
- 23: 0/10 (constraint-poor → q₀ weakly determined even with right ω)
- 28: 3/10 (1 near-truth + 2 near-twin)
- 69: 1/10 (near-twin)
- 81: 4/10 (all near-twin)

**Conclusion**: downstream q₀ search is sufficient for 3/4 of the tested seeds *given truth ω*. **The entire bottleneck for 028/069/081 is upstream ω recovery.** For 023, q₀ is also weakly determined.

**e8 — 3000-dir dense S² grid for 028, 069** (`e8_dense_s2_seed028.py`), q₀ oracle:
- seed 069: grid point 3° from truth has MSE=1.05 (vs truth=0.00035). Usable as NM init.
- seed 028: grid minimum is at **145° from truth**, MSE=6.79. Cost surface has no basin signal at tractable resolutions.

**e9 — polish from dense-grid near-truth inits for seed 069** (`e9_polish_from_grid.py`): 3 dirs × 5 q₀ starts = 15 joint polishes. All 15 kept w_dir<5° (ω stays in basin); 3/15 hit q_err<10° or near-twin. Proper m115 DE on these ω candidates should push to ~99% success.

**m115 oracle bug**: `load_omega_candidates` sorts by `w0_ref_errs`, i.e., the *truth-directed ω error in degrees*. Verified at runtime for seed 49 — m115 picked geo_ckpt idx [0, 11, 22] = oracle-top-3; geo_cost top-3 would be [0, 1, 21]. For 049/081/091 truth-close ω is in BOTH top-3 lists so verdict is likely unchanged under fix. For seed 023 it wouldn't change verdict either (already failing). Fix needed for honest reporting regardless.

**Spectral |ω|** (`a_spectral/spectrum_census.json` re-read): per-seed-best rule gives ≤5% err for 028 (`omega_eq_2pi_acf`), 069 (`omega_eq_pi_f`), 023 (`omega_eq_pi_acf`). No single rule dominates (22/100 best-rule within 5%), but multi-rule union is usable for scoping dense-grid |ω|.

**Recipe for 069** (`RECIPE_069.md`): spectral |ω| × multi-rule → 3000-dir dense S² per |ω| → q₀ inner-search (3 NM starts × maxiter=20) → top-K joint polish → m115 q₀ DE → hi-fi validate. ~35 min/seed. Not run this session; handoff artifact.

**Seeds taxonomy after this session**:
| bucket | seeds | path forward |
|--------|-------|-------------|
| solvable today | 49, 81, 90, 91 | m115+m126 already OK (modulo oracle bug cleanup) |
| solvable with dense grid | 069 | implement RECIPE_069; ~35 min/seed |
| constraint-poor (q₀ ambiguous) | 023 | even truth-ω doesn't rescue q₀ — needs fundamentally different cost |
| cost-surface pathology | 028 | grid search futile; needs feature-matching or external prior |
| PARTIAL (surrogate/hi-fi gap) | 024 | m115 finds basin, hi-fi 0.034 — may be surrogate error at low phase |



---

## d_per_epoch_search — 2026-04-18 (stage 2 pilot; ω factorization from pairwise anchors)

**Thread**: after stage 1 validated per-epoch 3-DOF q-search on seed 0 (t=313 and t=400 give tight |C|=129, 144 containing truth at 3.7° and 1.2°), stage 2 factorizes ω via pairwise matching on the two tightest anchors.

**Algorithm**: for each (q_a, q_b) ∈ C_a × C_b, compute q_rel = q_a⁻¹⊗q_b, extract axis n̂ + principal angle θ_p, enumerate wrap count k ∈ {0,1,2} × sign {±1} to get ω at t_a, back-prop q_a to q_0 (constant-ω approximation), then score full-LC via v2 tumbling forward. Pool(16), 111,456 tasks, 740 s. See `stage2_omega_factorization.py`.

### Sanity check on TRUTH-closest pair (before pilot)

For qa[48] (truth_dist=3.70°) × qb[133] (truth_dist=1.20°):
- k=0: |ω_est| = 0.186 dps vs truth 1.449 dps — **8× undershoot** (wrap-ambiguity: truth |ω|Δt = 5.05π)
- k=2 (right wrap): |ω_est| = 1.333 dps — magnitude within 8% of truth
- All wraps: ω direction 37° (sign=+1) or 142° (sign=-1) off from truth — **tumbling precession bias** (I₁≈I₂>>I₃ symmetric top, 748 s precession period)

So pairwise is an INITIALIZATION, not a solution. The approximation assumes constant ω_body but the body precesses.

### Pilot results (seed 0, 105,006 scored candidates)

- **Best MSE = 2.72** (rank 0 by MSE: q0_err=84°, ω_dir=134°). Noise floor ≈ 0.0002. No candidate is near truth by MSE.
- **Truth-closest (by q0_err)**: q0_err=10.55°, ω_dir=23.8°, ω_mag_err=0.19, MSE=4.28. Rank by MSE is ~100-500+.
- **MSE-ranking does NOT surface truth-adjacent candidates**: top-100 by MSE all have q0_err > 49° (wrong basin, spurious LC match). Truth-close ones sit in MSE=4-5 band.

### Polish results (NM maxiter=120, 7-DOF: q0_xyz + ω)

**Mechanical polish of top-100 by MSE**: MSE 2.72-3.48 → 2.39-2.70 (marginal). All 100 hit maxiter; q0_err after polish still 49-177°. NM cannot bridge from wrong basins.

**ORACLE polish of top-20 by q0_err (truth-selected)** — diagnostic for upper bound:
- Pre-polish q0_err 10.55–16.95° → POST-polish q0_err **10.4–37.3°** (mostly WORSE; NM drifts away from truth).
- Pre MSE 4.28–7.25 → post MSE 3.00–5.50 (reduced but still far from GATE 0.02).
- **ORACLE FAIL: even from truth-close init, NM on v2-LC MSE cannot converge to truth.**

### Implication

The v2-LC cost surface in 7-DOF at q0_err ~10–15° is multimodal with spurious minima AWAY from truth. Plain Nelder-Mead gets caught in them. This matches the prior lesson from the m115 pipeline, where DE (global optimizer with larger effective basin) is used for q₀-given-ω search. The stage-2 pairwise-ω approach as currently designed cannot stand alone.

### Paths forward (for next session)

1. **Replace NM with DE** on the same top-K (or a larger pool). DE has mutation + crossover that can escape local minima. Budget: ~5 min per candidate on Pool(16) → top-100 DE polish in ~30 min.
2. **Tighten stage 1 init**: 108k samples per anchor gave |C_a|=129 with min truth_dist=3.70°. A 10× finer grid would shrink truth_dist to ~1°, shrinking the stage-2 init error. Cost: ~10× stage 1 compute = ~40 min per seed.
3. **Chain 3+ anchors**: cross-anchor consistency on (q₀, ω) reduces wrap ambiguity and spurious matches. Requires stage 2 rewrite to enumerate triplets instead of pairs.
4. **Use m115's q₀ DE directly on pairwise ω estimates**: stage 2 produces (q₀_init, ω_init) candidates; hand them to m115 as initial guesses rather than polishing ourselves. Bypass the NM brittleness.

Current verdict: pairwise ω factorization + NM polish is **not viable for seed 0**. One of paths 1–4 must land before d_per_epoch_search can be evaluated on the Phase-B cohort.
