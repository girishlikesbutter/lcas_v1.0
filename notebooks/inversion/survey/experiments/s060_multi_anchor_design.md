---
title: "s060 — multi-anchor C_t architecture design (post-s059k)"
type: design
sources:
  - notebooks/inversion/survey/experiments/s060_sharpness_map.py
  - notebooks/inversion/survey/results/s060_sharpness_map/seed028/summary.json
  - notebooks/inversion/survey/experiments/s059j_design.md
  - notebooks/inversion/survey/experiments/s059k_full_lc_from_seeds.py
related:
  - s059j — single-anchor cloud-data ω-grid (failed go/no-go on seed 28)
  - s059k — densified s059j + multi-mag-start (Band A on seeds 89+10)
  - s057f — per-pair finite-diff failure (small Δt geometric noise)
  - s057g — forward-propagation discrimination (4400× over null, all-cloud weighting)
  - s018b — face-identity tier classifier (buggy-era predecessor)
  - project_sharpness_not_brightness.md — sharp anchors include dim moments
created: 2026-05-09
updated: 2026-05-10
confidence: medium (untested architecture; sharpness + topology + cohort substrate confirmed)
status: DESIGN — diagnostics landed (s060_sharpness_map, s060_anchor_topology, s060_cohort_anchor_topology); implementation queued behind seed 28 closeout via s059k
---

## TL;DR

Replace s059k's **single-anchor + ω-grid** architecture with **multi-anchor + ω-from-shoot**, with **per-seed routing between two anchor roles**:

- **BRIGHT passage validators** (universal — 11/11 s011 cohort): bright-extreme anchors have 7-30 small clusters separated by ~170°; not enumerable as discrete hypotheses but useful as "does the propagated trajectory pass through ANY of these specular configurations at this epoch?" pass/fail checks.
- **DIM clean enumerators** (opportunistic — ~5/11 s011 cohort): when an ultra-dim moment has |C_t| ≤ 100 with top-3 cluster mass ≥ 0.7, the survivors collapse into 3-5 distinct geometric attitude classes. Enumerable as discrete hypothesis sources; tuple-search across K=4 dim anchors with 4 clusters each gives 4⁴ = 256 hypothesis-class tuples.
- **Routing**: per-seed gate on `|C_t|(dim_extreme) < 100 AND top-3 mass ≥ 0.7` decides whether to use the clean-enumerator path (when available) or fall back to bright-only passage validation.

For seeds with the clean-dim path: enumerate cluster-tuples ∏ (dim_clusters_T_i), Newton-shoot ω from first pair, polhode-thread by passing through remaining bright validators. The ω-grid disappears; quantization noise — the s059j seed-28 bottleneck — is replaced by Newton convergence noise.

s060 cohort topology diagnostic (2026-05-10) confirms bright fragmentation is universal but the clean-dim pattern from the user's seed-89 observation generalizes only on ~5/11 seeds. Architectural design must split per-seed.

## What

The s059j/k pipeline picks ONE sharp anchor T_A and grid-searches ω at every q_a ∈ C_{T_A}. Two failure modes:

1. **ω-grid quantization noise** (s059j seed 28): N_DIRS=200 Fibonacci → 7° spacing → 5° truth offset → ρ=38 at W=10 with EXACT q_a. s059k fixed this by densifying to N_DIRS=800 + multi-mag-start [0,±3,±6]%. Worked on seeds 89+10.
2. **Single-anchor weak constraint**: forward-propagating from one (q_a, ω_a) hypothesis to validate against the rest of the LC depends entirely on whether the LM polish basin lands on truth. Multi-solution seeds (where multiple polhodes match around a single anchor) are not narrowed by single-anchor structure.

s060 attacks both by using **multiple sharp anchors as joint constraints**:

- K anchor epochs T_1 < T_2 < ... < T_K with small |C_{T_i}|.
- A trajectory (q_0, ω_0) is consistent with the LC IFF q(T_i) ∈ C_{T_i} for all i (within tolerance), AND the dynamics integrates correctly (Euler + kinematic eqns).
- The first two anchors fix ω₀ via Newton shooting; the remaining K-2 anchors validate the polhode threading.
- No ω-grid → no quantization gap.

## How

### Stage 1: Sharp anchor identification + per-seed routing classification

```
1.1 Build coarse Sobol pool, N_COARSE = 50_000 (s060 used 25k diagnostic).
1.2 For each epoch t in 0..N-1 (full LC, NOT just early window):
        |C_t|(t) = sum( |surrogate_pred(pool@t) - measured(t)| < TOL_MAG )
1.3 Identify the BRIGHT-extreme sharp anchor:
        bright_idx = epochs in bottom 5% of mag values (the brightest)
        T_B^* = argmin |C_t|(t) for t in bright_idx
1.4 Identify the DIM-extreme sharp anchor:
        dim_idx = epochs in top 5% of mag values (the dimmest)
        T_D^* = argmin |C_t|(t) for t in dim_idx
1.5 Measure cluster topology at each (40° greedy cluster). Per-seed routing:
        if |C_t|(T_D^*) < 100 AND top-3 mass at T_D^* ≥ 0.7:
            ROUTE = "clean-dim-enumerator"
        else:
            ROUTE = "bright-validator-only"
1.6 Pick additional bright validator anchors:
        cluster_independent_brights = greedy K=3-4 anchors from full-LC bright
        regions (mag_pct ≤ 15%) with Δt ≥ 30 epochs.
1.7 If clean-dim ROUTE: pick K_dim=2-3 dim enumerator anchors (cluster-
    independent if possible). If bright-only ROUTE: skip.
```

### Stage 2: Dense resample at top-K anchors

```
2.1 Build dense Sobol pool, N_DENSE = 400_000 (matches s059j density).
2.2 For each T_i in chosen anchors:
        project + survive → C_{T_i}^dense
        cap to |C| ≤ 1000 via random subsample if needed
2.3 Canonicalise via lib.twin.canonical_batch (body-twin halving, 2× speedup).
```

### Stage 3a: Clean-dim ROUTE — polhode threading via cluster enumeration

```
3a.1 At each dim enumerator anchor T_D_i, get cluster representatives
     {q^c_{T_D_i}} for c in top-3 clusters (after canonical_batch halving).
3a.2 For each cluster-tuple (c_1, c_2, ..., c_{K_dim}) ∈ ∏ {1..3 clusters}:
        # Pick representative quaternions (cluster centroid or member nearest centroid)
        q_a = rep(C^{c_1}_{T_D_1})
        q_b = rep(C^{c_2}_{T_D_2})
        # First pair fixes ω_a via Newton
        ω_a^(0) = finite_diff_omega(q_a, q_b, T_D_2 - T_D_1)
        ω_a     = newton_shoot(q_a, q_b, T_D_1, T_D_2, I, ω_a^(0))
        # Validate downstream dim anchors (passage check)
        for each remaining dim anchor T_D_i (i ≥ 3):
            q_pred = propagate(q_a, ω_a, T_D_i - T_D_1, I)
            if not any cluster_rep_close(q_pred, C^*_{T_D_i}, δ=20°):
                REJECT this tuple
        # Validate bright passage anchors
        for each bright validator T_B_j:
            q_pred = propagate(q_a, ω_a, T_B_j - T_D_1, I)
            if not any survivor_close(q_pred, C_{T_B_j}, δ=15°):
                REJECT this tuple
        if all checks pass:
            ACCEPT (q_a, ω_a) as candidate.
3a.3 Cluster surviving (q_a, ω_a) via s057h canonical_batch + greedy.
```

**Tuple count for clean-dim:** K_dim=3 anchors × 3 clusters each = 27 tuples. Trivial. Most rejected at first downstream validator. Surviving candidates enter Stage 4.

### Stage 3b: Bright-only ROUTE — pairwise passage check

```
3b.1 At each bright validator anchor T_B_j (full survivor set, no enumeration).
3b.2 For each (q_a, q_b) pair from the two sharpest bright validators:
        ω_a^(0) = finite_diff_omega(q_a, q_b, T_B_2 - T_B_1)
        ω_a     = newton_shoot(...)
        # Validate against remaining bright validators
        ...
3b.3 Tuple count: |C_{T_B_1}| × |C_{T_B_2}| typically 100×100 = 10⁴ pairs.
     Newton-shoot ~10ms each → ~100 sec single-thread.
```

**Newton shoot inner loop:** 3D root-find on `propagate(q_1, ω, Δt, I) ⊖ q_2 = 0`. Jacobian via finite differences (3 forward propagations per step). Typical convergence in 5-8 iterations from finite-diff init. Per-tuple cost ≈ 10ms.

### Stage 4: Full-LC LM polish (s059k Phase-4)

```
4.1 For each surviving (q_a, ω_a) cluster representative:
        back_propagate (q_a, ω_a) → (q_0, ω_0) at t=0
        lm_polish(q_0, ω_0, ctx, target)  # full-LC, surrogate-v2 residual
        + multi-mag-start at offsets [0, ±3, ±6]%
4.2 Hi-fi gate: render iff surrogate_rho < 4. Classify into ρ-bands.
```

### Stage 5: Headline yield

```
5.1 Count A∪B basins per seed (no truth injection).
5.2 Compare to s059k baseline on same seeds.
```

## Why this matters

**The ω-grid is the s059j seed-28 bottleneck and the s059k partial fix.** Densifying the ω-grid (N_DIRS=200→800) was sufficient on seeds 89+10 but expensive (75-127 min wall per seed). The grid is fundamentally a workaround for the absence of dynamics-derived ω. With multi-anchor, dynamics gives ω directly per (q_a, q_b) pair via the propagator's own integration — exactly the right tool.

**Multi-solution discrimination.** s059k's seed 89 result returned 3 distinct (q0, ω) basins (truth, body-twin-direction, q0=60° alternate) because all three independently satisfy the single-anchor LC residual. Multi-anchor's K-2 validator clouds DIRECTLY test whether each candidate's polhode threads through the rest of the LC. Two distinct polhodes might thread the same single anchor; threading 4 sharp anchors is much rarer. Multi-solution seeds may collapse to fewer candidates here — a genuine reduction in degeneracy, not a loss of information.

**No oracle |ω| anchor.** s059j/k's grid is centered on truth |ω_a| (oracle). Multi-anchor's Newton shoot derives ω from (q_a, q_b) directly — no |ω| prior needed. This is the long-promised path to **fully oracle-free inversion**, not just oracle-cluster-id-free as in s059k.

**Connection to s057g forward-propagation.** s057g already used ALL clouds with weighted hits and got 4400× over null. The dilution by wide clouds was a known issue; multi-anchor's hard threshold on K narrow clouds is the sharpened version of the same idea.

## Numbers

### s060 sharpness map (seed 28, N=25k Sobol, TOL=0.10)

- |C_t| min = **5** at t=312 (mag=20.62, dimmest in LC)
- |C_t| median = 719
- |C_t| max = 1880
- 32 epochs with |C_t| < 100; 89 with |C_t| < 250 (1% of pool); 5 at the floor (|C_t| < 25)
- Top-10 sharp anchors: 4 in dim regime (mag > 18), 6 in bright regime (mag < 7), 0 in mid-band

### s060 vs s059k cost projection (seed 28)

| stage              | s059k         | s060 (projected) |
|--------------------|---------------|------------------|
| sharp anchor scan  | n/a (early window only) | ~5 min Pool(8) full LC |
| dense resample     | 1× anchor     | K×anchors        |
| ω-grid scoring     | 75 min N_DIRS=800 Pool(24) | **eliminated** |
| polhode threading  | n/a           | ~1 min Pool(8) for ~10⁴ tuples |
| full-LC polish     | 28 min Pool(8) | similar (smaller candidate count) |
| **total wall**     | ~105 min       | ~35 min projected |

### Sharp anchor cluster structure on seed 28 (top-10)

- Dim cluster 1: t ∈ {123, 224, 312} — single sharp anchor each
- Dim cluster 2: t = 440 — isolated
- Bright cluster 1: t ∈ {368, 389, 390, 391} — 4 epochs in 25-window (one polhode period)
- Bright cluster 2: t ∈ {491, 492} — 2 epochs adjacent

After cluster-independence enforcement (Δt_min = 30): {224, 312, 390, 492} or {123, 312, 390, 492} as the 4 chosen anchors. Span ~270 epochs ≈ 1+ polhode period at |ω|=1.43 dps.

## Open questions to validate before implementing

1. **Newton shooting convergence** on the chosen-anchor pair (T_1 → T_2). Is finite-diff ω a good enough init, or does the 17% magnitude error blow up the Jacobian? Test on truth pair from seed 28.
2. **Tolerance δ_q for cloud-membership** at validator anchors. Too tight → no surviving tuples; too loose → too many. Initial guess: 20° at dim cluster reps (post-40°-clustering, members within ~20° of centroid), 15° at bright passage validators (smaller intra-cluster scale).
3. **Polhode-period ambiguity**: at long Δt, multiple ω solutions thread the same (q_a, q_b). Newton finds the local one but might miss the global. May need polhode-prior (s055a) to break the ambiguity.
4. **Cohort generality**: clean-dim path validated on ~5/11 s011 seeds (10, 28, 44, 89, 91 borderline). Bright-only path needed on ~6/11 seeds. Architecture must support both.
5. **Holdout validation**: clean-dim availability rate on m048 100..119 (s054 holdout corpus). Tests whether 5/11 cohort fraction generalizes.

## Artefacts

- `notebooks/inversion/survey/experiments/s060_sharpness_map.py` + `.md` — full-LC |C_t|(t) diagnostic on 3 seeds.
- `notebooks/inversion/survey/experiments/s060_anchor_topology.py` + `.md` — per-anchor cluster structure.
- `notebooks/inversion/survey/experiments/s060_cohort_anchor_topology.py` + `.md` — 11-seed cohort generalization.
- `notebooks/inversion/survey/results/s060_sharpness_map/seed{028,089,010}/` — per-seed sharpness + topology data.
- `notebooks/inversion/survey/results/s060_cohort_topology/seed*.json` — 11-seed topology summaries.

To be built (this design):
- `lib/multi_anchor.py` — sharp_anchor_select, route_classify, newton_shoot, polhode_thread.
- `experiments/s060a_newton_smoke.py` — sanity test on truth pair (seed 28).
- `experiments/s060b_seed28_pilot.py` — full pipeline, single-seed clean-dim path.
- `experiments/s060b2_bright_only_pilot.py` — bright-only path on a fragmented-dim seed (e.g. seed 21 or 48).
- `experiments/s060c_cohort_run.py` — multi-seed run (gated on s060b yield).

## Out of scope

- ω-grid backup as a "if shooting fails" fallback. Defer until shooting is shown to fail.
- Polhode-prior (s055a) integration as init or ω-disambiguator. Layer on once the basic architecture works.
- NN-based ω predictor (the previous-turn spitball). Orthogonal architecture; pursue separately if multi-anchor saturates.
- Multi-anchor on seeds without ≥4 sharp anchors. Document as out-of-class; route to s059k path.
- Full-LC sharpness scan at N=400k pool. Diagnostic at N=25k is sufficient for top-K selection; the dense resample is per-anchor anyway.

## Cross-references

- `experiments/s059j_design.md` line 239-241: "multi-anchor fallback" was listed as bonus extension; this is the concrete realization.
- `concepts/known_pathologies_to_revalidate.md` — s018b face-identity tiers were buggy-era predecessor; geometric reasoning survives the bug fix.
- `feedback_finite_diff_small_geodesic_noise.md` — s057f's failure mode at small Δt; multi-anchor uses LARGE Δt to exploit polhode integration over long span.
- `project_sharpness_not_brightness.md` — anchor selection across full LC, not bright-peak filtered.
- `project_omega_grid_architecture.md` (post-s059) — the architecture this design is replacing.
