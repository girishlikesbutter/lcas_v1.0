# LCAS Inversion — Experiment Dependency Map

> **Purpose:** After days away, read Section 1 to know where you are. Read Section 2 to understand the full research arc. This document is the single source of truth for the inversion experiment series.

---

## 1. Resume Point

- **ACTIVE THREAD:** Series 07b — L-conservation winding filter (2026-03-10)
- **LAST COMPLETED:** micro25 — L-conservation validated as winding filter. **Works perfectly with oracle attitudes AND with up to 10 deg endpoint error.** The blocking problem is staircase coverage, not L-consistency.
- **COMPLETED TODAY (2026-03-10):**
  - micro23 (oracle L-test): **L-conservation identifies true pair rank 1/64, gap = 113 kg*m^2/s (9 orders of magnitude)**
  - micro24 (nudge sensitivity): **100% correct at all nudge levels 0.5-10 deg.** Shared-node nudge is mathematically invariant (R cancels). Endpoint nudge degrades gap linearly but remains robust.
  - micro25 (3-leg test): True triple ranks 1/512 with 4 peaks. 2 legs sufficient for this test case.
- **RUNNING:** micro15 (α=0.1, 101 pairs), micro15b_alpha10 (α=10), micro16c (trough-constrained bridge)
- **BLOCKING:** Leg 1 staircase misses true winding (jumps from 0.25 to 3.28 dps, skipping 2.08 dps). Need denser staircase or bidirectional search.
- **OPEN QUESTION:** How to fix the staircase so it reliably includes the true winding on all legs?

---

## 2. The Map

### Test Case (all experiments)
- Satellite: Intelsat 901
- True q0: axis=[0.6, 0.3, 0.8]/norm, angle=45 deg
- True omega0: [0.5, -0.3, 2.0] deg/s ("fast tumbler")
- N_OBS=500, dt~7.2s, window=3600s
- Lo-fi = no shadows; Hi-fi = ray-traced shadows

---

### Series 00 — Pipeline Reference (2026-01-21 → 2026-02-06)

Foundational notebooks establishing the forward model and inversion infrastructure. Built incrementally via Ralph (automated PRD agent). These are **reference implementations**, not experiments.

| Script | Purpose | Status |
|--------|---------|--------|
| `01_inertia_calculation.py` | Compute inertia tensor from STL mesh | DONE |
| `02_attitude_propagation.py` | Demonstrate principal-axis and tumbling propagation | DONE |
| `03_lightcurve_inversion.py` | End-to-end inversion pipeline (multi-start L-BFGS-B) | DONE |
| `04_landscape_analysis.py` | 1D/2D slices of objective, gradient/Hessian at truth | DONE |
| `05_basin_size_study.py` | Basin of attraction vs initialisation distance | DONE |
| `06_optimizer_comparison.py` | Multi-start vs DE vs Basin-hopping (evaluation-matched) | DONE |
| `07_robustness_analysis.py` | Noise level and observation density sweeps | DONE |
| `08_mixed_fidelity_inversion.py` | Two-stage lo-fi DE → hi-fi L-BFGS-B pipeline | DONE |

**Key finding:** Local optimisers work near truth but global optimisers (DE, basin-hopping) fail on the 6D joint space within practical budgets. Mixed-fidelity two-stage pipeline is viable *if* a good initial guess exists.

---

### Series 01 — Global Search Attempts (2026-02-06 → 2026-02-09)

Tried every standard global optimisation strategy on the 6D problem. **All failed.** This series established that naive global search cannot solve the inversion problem, motivating the pivot to structured approaches.

```
inversion_demo.py          ─── Roberto demo (multi-start lo-fi → hi-fi)
├── inversion_demo_v2.py   ─── Decoupled sequential + multi-optimizer
└── inversion_demo_v3.py   ─── Grid + CMA-ES + omega recovery + joint hi-fi
```

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `inversion_demo.py` | Can multi-start lo-fi find truth? | No — best candidates far from truth | SUPERSEDED |
| `inversion_demo_v2.py` | Does decoupled sequential (att→omega→joint) work? | Partial — attitude grid too coarse | SUPERSEDED |
| `inversion_demo_v3.py` | Does grid+CMA-ES+omega recovery work? | No — 5832-point grid insufficient | SUPERSEDED |
| `exp_tight_start_proof.py` | Does the pipeline work given a *good* start? | **Yes** — 0.5-3 deg init → converges | DONE |
| `exp_bruteforce.py` | Can 8³ att × 3³ omega grid + refinement work? | No — 13,824 points too sparse | SUPERSEDED |
| `exp_decoupled_grid.py` | Decoupled 30° grid (1728 att) → per-candidate omega? | Too slow (~35 min), no convergence | SUPERSEDED |
| `exp_decoupled_fast.py` | Parallel version of decoupled grid | Faster but still fails | SUPERSEDED |
| `exp_dual_annealing_6d.py` | Dual annealing on full 6D? | No convergence in 8 min | SUPERSEDED |
| `exp_multistart_tight.py` | 500 multi-start L-BFGS-B with tight omega bounds? | No — even ±0.02 deg/s too wide | SUPERSEDED |
| `exp_omega_first.py` | Estimate omega magnitude from Lomb-Scargle, grid direction? | Magnitude extraction unreliable | SUPERSEDED |
| `exp_alternating.py` | Alternating att/omega optimisation? | Stuck in local minima | SUPERSEDED |
| `exp3_convergence_basin.py` | Hi-fi convergence basin radius? | Att ~5 deg, omega ~0.02 dps | DONE |
| `exp3_convergence_basin_lofi.py` | Lo-fi convergence basin? | Similar shape, noisier | DONE |
| `exp3_basin_resume.py` | Complete and refine basin mapping | Extended basin data | DONE |
| `exp3_large_budget.py` | Can 20-50k lo-fi DE evals find solution? | No — budget insufficient | SUPERSEDED |
| `exp3_single_strategy.py` | Subprocess wrapper for parallel strategy runs | Utility | DONE |
| `exp3_parallel_strategies.py` | DE vs CMA-ES vs multi-start in parallel? | None converged | SUPERSEDED |
| `run_exp3_dynamics_fidelity.py` | Principal-axis ultra-fast → tumbling hi-fi? | Dynamics fidelity matters more than shadows | DONE |
| `run_exp3_v3_bounded.py` | Data-driven omega bounds from Lomb-Scargle + ACF? | Bounds too wide to help | SUPERSEDED |
| `run_exp3_timed.py` | Timed pipeline validation (30 min budget) | Pipeline works, budget insufficient | DONE |
| `run_exp3_only.py` | Mixed-fidelity hierarchical (DE lo-fi → L-BFGS-B hi-fi) | Pipeline validates, global search fails | DONE |
| `test_exp3_only.py` | Quick validation of mixed-fidelity pipeline | Test passes | DONE |
| `generate_basin_plots.py` | Basin heatmaps for Roberto slides | Plots generated | DONE |
| `lofi_worker.py` | Parallel lo-fi evaluation worker | Utility | DONE |

> ⚡ **PIVOT (2026-02-09):** Global search exhaustively failed. Attitude basin is ~5 deg, omega basin ~0.02 dps — the 6D joint basin is impossibly narrow for blind search. *Decision:* abandon global optimisers, exploit lightcurve structure instead.

---

### Series 02 — Iso-brightness Filtering (2026-02-19 → 2026-02-20)

Explored the idea that individual brightness measurements constrain attitude. If brightness is highly selective (~0.05% match rate at 1% tolerance), can we filter candidates across multiple epochs?

```
exp_isobrightness.py
├── exp_brightness_filter.py     ─── Single-epoch brightness screening
├── exp_explore_filtering.py     ─── Multi-step filtering cascade
├── exp_true_in_set.py           ─── Verify truth appears in candidate set
├── exp_residual_vs_error.py     ─── Residual-error correlation
├── exp_residual_robustness.py   ─── Robustness across 10 random truths
├── exp_mixed_fidelity_search.py ─── Lo-fi candidates → hi-fi refinement
├── exp_mixed_fidelity_search_v2.py ─── Dense grid (100k) variant
├── exp_parallel_lofi.py         ─── Parallel lo-fi iso-brightness optim
├── exp_sequential_filter.py     ─── Multi-epoch cascade (6 stages)
├── exp_sequential_filter_v2.py  ─── Tighter thresholds, better spacing
├── exp_sequential_filter_v3.py  ─── FFT-based omega bounding
├── exp_triplet_matching.py      ─── 3-epoch triplet omega consistency
└── check_residual_correlation.py ─── Lo-fi residual vs attitude error
```

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `exp_isobrightness.py` | How many attitudes match a given brightness? | Hundreds per epoch (L-BFGS-B finds ~16 clusters) | DONE |
| `exp_brightness_filter.py` | Single-epoch brightness as attitude constraint? | Strong constraint (~0.05% pass) but degenerate | DONE |
| `exp_explore_filtering.py` | Multi-step filtering power? | Each epoch filters ~100x but doesn't converge | DONE |
| `exp_true_in_set.py` | Does truth appear in candidate set (200 seeds)? | Yes — within ~1 deg in most cases | DONE |
| `exp_residual_vs_error.py` | Does low residual imply low attitude error? | Weak correlation — residual ranking unreliable | DONE |
| `exp_residual_robustness.py` | Do bottom-200 by residual contain truth? | Not reliably across random truths | DONE |
| `exp_mixed_fidelity_search.py` | Lo-fi iso-brightness → hi-fi refinement? | Works in principle, hi-fi expensive | DONE |
| `exp_mixed_fidelity_search_v2.py` | 100k dense grid (no optimiser)? | Picks 200 closest, feeds hi-fi L-BFGS-B | DONE |
| `exp_parallel_lofi.py` | Parallel lo-fi iso-brightness (2000 seeds)? | Scales well with 8 workers | DONE |
| `exp_sequential_filter.py` | 6-stage cascade (brightness→omega→pairing)? | Filter works but combinatorial explosion | SUPERSEDED |
| `exp_sequential_filter_v2.py` | Tighter thresholds, better epoch spacing? | Improved but still combinatorial | SUPERSEDED |
| `exp_sequential_filter_v3.py` | FFT-based omega bounding? | Bound too loose to help | SUPERSEDED |
| `exp_triplet_matching.py` | 3-epoch omega consistency filtering? | ‖ω₁₂ − ω₂₃‖ filter works in principle | DONE |
| `check_residual_correlation.py` | Lo-fi residual correlates with attitude error? | Incomplete/diagnostic | ABANDONED |

> ⚡ **PIVOT (2026-02-19):** Sequential cascade filtering suffers combinatorial explosion: at each epoch, ~100 candidates pass, and multi-epoch pairing creates N^k combinations. *Decision:* need physics-based constraints to prune — enter omega bridging.

---

### Series 03 — Omega Bridging (2026-02-19 → 2026-02-20)

Tested whether angular velocity can *bridge* between attitude candidates at different epochs: given q_A at epoch 1 and q_B at epoch 2, solve for the omega that connects them under torque-free dynamics.

```
exp_two_epoch_matching.py
├── exp_omega_bridge.py          ─── 3-epoch omega bridge + triplet consistency
├── exp_omega_bridge_v2.py       ─── (duplicate of above)
├── exp_omega_chain4.py          ─── 4-epoch adjacent chain
├── exp_omega_chain4_v2.py       ─── Memory-safe streaming version
├── exp_chain7_tight.py          ─── 7-epoch chain with 99% culling
├── exp_culling_vs_dt.py         ─── Culling power vs epoch spacing
├── exp_forward_prop.py          ─── Forward-prop filter with omega grid
├── exp_conservation_and_L_param.py ─── L-param vs omega-param basins ⚠️ BUG
├── exp_q0_omega_analytic.py     ─── Analytic (axisymmetric) propagation
├── exp_q0_omega_nm.py           ─── Nelder-Mead q0-fixed omega optimisation
└── exp_q0_omega_opt.py          ─── L-BFGS-B q0-fixed omega optimisation (10k candidates)
```

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `exp_two_epoch_matching.py` | 6-phase pipeline: iso-brightness → pairs → ω → filter → rank? | Pipeline works end-to-end but ranking noisy | DONE |
| `exp_omega_bridge.py` | 3-epoch bridges + triplet consistency? | Triplet consistency check is discriminating | DONE |
| `exp_omega_bridge_v2.py` | (Duplicate of above) | Same | SUPERSEDED |
| `exp_omega_chain4.py` | 4-epoch chain consistency? | Chain works but O(N^4) is prohibitive | DONE |
| `exp_omega_chain4_v2.py` | Memory-safe streaming version? | Reduces memory, still slow | DONE |
| `exp_chain7_tight.py` | 7-epoch chain + 99% culling? | Extreme culling kills truth path too | ABANDONED |
| `exp_culling_vs_dt.py` | How does culling power scale with Δt? | Larger Δt = better culling (rotation grows linearly) | DONE |
| `exp_forward_prop.py` | Forward-prop with omega grid + early termination? | Works but O(N_cand × N_omega × N_epochs) expensive | DONE |
| `exp_conservation_and_L_param.py` | L-param basin wider than omega-param? | ⚠️ BUG: doesn't actually freeze attitude. L basin NOT wider. | DONE (buggy) |
| `exp_q0_omega_analytic.py` | Analytic propagation for near-axisymmetric IS-901? | IS-901 asymmetry=0.556, not axisymmetric enough | ABANDONED |
| `exp_q0_omega_nm.py` | Nelder-Mead on q0-fixed omega? | Concept validation — converges but slow | DONE |
| `exp_q0_omega_opt.py` | L-BFGS-B on q0-fixed omega (10k candidates)? | Rankings by residual don't recover truth reliably | DONE |

> ⚡ **PIVOT (2026-02-20):** Chain-based filtering (bridge all epoch pairs) is combinatorially explosive and aggressive culling kills truth. *Decision:* anchor on brightness **peaks** (where dL/dt≈0 provides extra constraint) and build a graph across a small number of peaks.

---

### Series 04 — Basin Characterisation (2026-02-23 → 2026-03-01)

Quantitative measurement of convergence basin widths. These numbers define the accuracy targets for candidate generation.

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `exp00_timing.py` | Baseline timing (10 reps) | Lo-fi: 221ms, hi-fi: 60s (272x), single-epoch lo-fi: 13ms, hi-fi: 47ms | DONE |
| `exp01_sanity.py` | Forward model round-trip? | All pass. Hi-fi residual at truth: 0.0024 ≈ σ². Conservation <1e-10. | DONE |
| `exp02_attitude_basin.py` | Attitude-only basin (omega truly fixed)? | Basin: all 48 trials converge (1-20 deg), consistent 0.15 deg error | DONE |
| `exp03_omega_basin.py` | Omega basin: L-space vs omega-space? | Partial run (.tmp file). Direction basin ~2 deg, magnitude basin ~10-20% | ACTIVE |

**Basin results from series 06 (omega-specific):**

| Study | Fidelity | Key Finding |
|-------|----------|-------------|
| basin_01 (sanity) | Hi-fi | MSE at truth: 0.0025, converges in 13 iters, omega err 0.004 dps |
| basin_01 (sanity) | Lo-fi | MSE at truth: 0.0084, converges but omega err 0.032 dps (6x worse) |
| basin_02 (magnitude) | Hi-fi | Converges for magnitude fractions 0.9–1.1 (±10% of true |ω|) |
| basin_02 (magnitude) | Lo-fi | Converges for 0.9–1.05 only (narrower basin) |
| basin_03 (direction) | Hi-fi | Converges up to ~2 deg offset; collapses by 10 deg |
| basin_03 (direction) | Lo-fi | Similar ~2 deg limit but higher final error |
| basin_04 (q degradation) | Hi-fi | Attitude error 0.5 deg → omega recoverable; 10 deg → fails |
| basin_04 (q degradation) | Lo-fi | Collapses by 5 deg attitude error |

> **Key numbers for pipeline design:** Need attitude candidates within ~5 deg and omega within ~0.02 dps (or ~2 deg direction, ±10% magnitude) for local optimiser to converge.

---

### Series 05 — Peak-Graph Pipeline (2026-02-26 → 2026-02-27)

**The main line of work.** Exploits lightcurve peaks (brightness maxima where dL/dt≈0) as anchor points. At each peak, brightness + derivative constraints narrow attitude candidates. Then omega bridging connects candidates across peaks via a graph, and the shortest path = best (q, omega) estimate.

```
micro01 ─── Brightness degeneracy at peaks
├── micro02 ─── Peak derivative constraint (g·ω≈0)
├── micro03 ─── Bridge solver validation (ω error → attitude error)
│   └── micro04 ─── Full bridge pipeline (oracle ω, 2 peaks)
├── micro05 ─── Hi-fi candidate counts at peaks
├── micro06 ─── 1M random sampling at peaks
│   ├── micro07a ─── ⚡ Omega drift measurement (constant-ω invalid!)
│   ├── micro07 ─── Propagate-and-match (peak bridging with Euler dynamics)
│   │   └── micro08 ─── Joint constraint pipeline (brightness + deriv + bridge)
│   ├── micro09 ─── L-BFGS-B iso-brightness optimisation (3.6x closer than random)
│   │   └── micro10 ─── Scale to 10K seeds (5643 basins)
│   └── micro10b ─── Dense 10M random sampling (13,848 candidates)
│       └── micro11 ─── Dense joint pipeline (13,848 cands, 3.1M propagations)
│           └── micro12 ─── ⚡ Peak time interpolation (lo-fi vs hi-fi mismatch)
│               └── micro13 ─── Graph pipeline (50 cands × 3 peaks, shortest path)
│                   ├── micro13b ─── Nudge sensitivity test
│                   ├── micro13c ─── LC visual comparison (truth vs nudged)
│                   ├── micro14 ─── Hi-fi rescore of graph pipeline
│                   └── micro14_parallel_test ─── Hi-fi parallel eval benchmark
```

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `micro01` | How degenerate is brightness at peaks? | 0.05-1.7% of random SO(3) match within 1-10% tol | DONE |
| `micro02` | Does peak derivative (g·ω≈0) help filter? | 79→5 candidates (15x reduction with known ω) | DONE |
| `micro03` | How does ω error propagate to arrival attitude? | 0.01 dps → 3 deg error, 0.1 dps → 31 deg (over 462s) | DONE |
| `micro04` | Can peak-anchored bridging find truth? | True pair ranks #5/18 with oracle ω, 1.1 deg error | DONE |
| `micro05` | Does hi-fi reduce candidate count vs lo-fi? | Marginal improvement — lo-fi candidates mostly survive hi-fi | DONE |
| `micro06` | 1M random sampling: nearest candidate to truth? | ~2 deg nearest (at peaks), derivative filter powerless without known ω | DONE |
| `micro07a` | Is constant-ω valid for bridging? | **Yes** — max drift 0.0007 dps over 3600s (IS-901 near-axisymmetric) | DONE |
| `micro07` | 1M attitudes × ω-plane scan → propagate → match? | Pipeline works but truth 5.95 deg away in initial candidates | DONE |
| `micro08` | Joint filter (brightness + derivative + bridge)? | Truth not recovered — initial candidates too sparse | DONE |
| `micro09` | L-BFGS-B iso-brightness (100 seeds)? | Nearest: 1.64 deg (3.6x closer than 1M random's 5.95 deg) | DONE |
| `micro10` | Scale to 10K seeds? | 5,643 unique basins, nearest ~1 deg from truth | DONE |
| `micro10b` | 10M random sampling? | 13,848 candidates, nearest 2.14 deg from truth | DONE |
| `micro11` | Dense joint pipeline (13,848 cands)? | 3.1M propagations; some candidates pass all filters | DONE |
| `micro12` | Why does scoring fail? Peak time mismatch? | Lo-fi peak time ≠ hi-fi peak time. Fine-scan gives 85x better g·ω | DONE |
| `micro13` | Graph pipeline (50×3 peaks, shortest path)? | 98.5% feasible bridges, but **lo-fi scoring cannot discriminate truth** | DONE |
| `micro13b` | How does small δq at peak 1 affect full-LC residual? | Small nudges propagate as full-trajectory errors | DONE |
| `micro13c` | Visual: truth vs 2-deg-nudged lightcurve? | Compound divergence visible over observation window | DONE |
| `micro14` | Does hi-fi rescoring fix discrimination? | Truth rank ~13k/121k — **still poor** | DONE |
| `micro14_parallel_test.py` | Multiprocessing speedup for hi-fi single-epoch eval? | Benchmark for fork-based Pool parallelism | DONE |
| `omega_basin_characterisation.py` | Fixed q_true, varying ω: basin shape? | Magnitude errors easier than direction offsets | DONE |

> ⚡ **PIVOT (2026-02-27, micro07a):** Constant-omega approximation *is* valid (drift <0.001 dps) — this simplifies bridging by eliminating need for full Euler propagation between adjacent peaks. However, Euler dynamics still needed for long-range bridging.

> ⚡ **PIVOT (2026-02-27, micro12):** Lo-fi and hi-fi peak times differ. The g·ω≈0 constraint at discrete epoch indices is noisy; interpolating to the true lo-fi peak time gives 85x better derivative constraint.

> ⚡ **CRITICAL FINDING (2026-02-27, micro13+14):** The graph pipeline generates ~2% feasible paths (2,500/125,000) but **intermediate brightness scoring cannot discriminate the true path**. Lo-fi intermediate scores are nearly uniform across all feasible paths. Hi-fi rescoring doesn't fix this. The scoring function is the bottleneck — not candidate generation, not bridging, not the graph structure.

---

### Series 06 — Omega Bridging Benchmarks (2026-03-01)

Isolated benchmarks for the omega bridging subroutine (given two quaternion endpoints, recover ω). Also includes peak-anchored omega recovery with quaternion nudge error analysis.

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `bench_omega_bvp.py` | Single-pair omega recovery cost? | L-BFGS-B: 162ms, 44 evals, sub-microdeg/s accuracy | DONE |
| `bench_omega_bvp_parallel.py` | Parallel scaling (n² pairings)? | 2500 pairs in 177s (4.9x speedup on 8 cores) | DONE |
| `bench_omega_bvp_noisy.py` | Noise sensitivity? | ~0.025 dps per degree of q error (linear) | DONE |
| `micro01_peak_omega_recovery.py` | Recover ω from consecutive q-pairs near peak? | Reconstructed LCs match truth for 2/4 trajectories | DONE |
| `micro01b_peak_q_nudge.py` | Hi-fi: ω recovery with 1-5 deg q nudge? | Anisotropic sensitivity — some directions 10x worse | DONE |
| `micro01c_peak_q_nudge_lofi.py` | Lo-fi: same nudge experiment? | Lo-fi fails for trajectories with strong self-shadowing | DONE |
| `micro01d_animations.py` | 3D shadow-pattern animations for nudge experiment | Visually confirms self-shadowing explains lo-fi failure | DONE |

**Key numbers:** Omega bridging is cheap (~160ms/pair), parallelises well (5x on 8 cores), and noise propagation is linear (0.025 dps/deg). This confirms bridging is not the bottleneck.

---

### Shared Infrastructure

| File | Purpose |
|------|---------|
| `lib/experiment_setup.py` | `setup_experiment()` → `ExperimentContext` (satellite, geometry, true LC) |
| `experiment_roadmap.md` | Original 5-phase plan (partially outdated — covers series 04 mainly) |
| `progress_tracker.md` | Early phase tracking (Phase 0-1 only; last updated 2026-02-23) |

---

### Series 07 — Minimum-Magnitude ω and Dip Constraint (2026-03-10)

Roberto's Feb 27 direction: address the scoring discrimination failure by changing both the ω search strategy and the scoring signal.

**Context:** micro-13/14 used an axis-angle ω estimate (zeroth-order constant-rotation approximation) as the initial guess for bridging, and scored paths by LC residual at intermediate epochs. Both failed. Roberto's prescription: (1) search for the *smallest* ω that achieves the bridge (avoids landing on a wrong branch of the multiplicity), (2) score by hi-fi brightness at troughs rather than intermediate epochs.

**LC convention (critical):** `CTX.observed_lc` / `CTX.true_lc` are stored as **magnitudes** (higher value = dimmer). Brightness peaks (glints) are **local minima** of the magnitude array. Brightness troughs (dips) are **local maxima**. Always use `argmax` to find dip epochs between two peaks.

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `micro15_min_omega.py` | Does zero-initial-guess + min-\|ω\| find a different ω than axis-angle? Does true pair rank low by \|ω_min\|? (101 pairs, leg 0, α=0.1) | TBD | RUNNING |
| `micro15b_alpha10.py` | Does α=10 (100× larger) improve true-pair rank vs α=0.1? | TBD | RUNNING |
| `micro15b_alpha_sweep.py` | Full α sweep [0, 0.0001…100] on true pair + 5 random pairs. What α gives best arrival + min-\|ω\|? | **α=1.0 sweet spot**: true pair \|ω\|=0.326 dps, err=0.002°. But 0.326 dps IS the lowest-winding solution (staircase step 0 ≈ 0.316 dps) — NOT the true ω (2.083 dps). **Min-\|ω\| always finds wrong winding.** Random pairs also land at ~0.46 dps. No discrimination possible. **Superseded by micro17 staircase.** | DONE |
| `micro16_dip_constraint.py` | Do brightness dips between peaks (hi-fi) discriminate true path? Leg 0 (183→260), 101 pairs | Trough at ep 211 (202s). True rank 65/101, score 1.99 vs mean 2.04 — **no signal**. Axis-angle bridge gives wrong attitude at trough for ALL pairs (incl. truth). Predicted ~12.5 mag, observed 14.5 mag. | DONE |
| `micro16b_dip_leg1.py` | Same dip constraint on leg 1 (260→360) | Trough at ep 338 (563s). True rank 36/101, score 1.53 vs mean 2.49 — **weak signal**. Same systematic brightness gap. | DONE |
| `micro16c_trough_constrained_bridge.py` | Embed trough brightness inside bridge objective: `arrival_err + α_trough×(lofi_trough−ref)²`. Only true pair has ω satisfying both endpoints AND trough simultaneously. | TBD | RUNNING |
| `micro17_staircase_omega.py` | Roberto's staircase: find ω_min, set as lower bound, find next winding, repeat. Does trough score identify correct winding? (true pair only, 8 steps) | Staircase works — 8 distinct winding solutions (0.316 → 4.824 deg/s), all with arrival_err~0. **Single trough insufficient**: step 2 (1.601 deg/s) scores best (err=0.021) but true ω is ~2.083 deg/s (step 3, 2.230 deg/s, scores 0.792). Mechanism validated; need multiple dips or hi-fi to discriminate. | DONE |

---

### Series 07a — Multi-Epoch LC Shape Scoring (2026-03-10)

Branch: `exp/multi-epoch-winding-score`. Tests whether evaluating brightness at MULTIPLE intermediate epochs between peaks can discriminate the correct winding from micro17's staircase family.

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `micro21_multi_epoch_winding_score.py` | Does multi-epoch lo-fi MSE identify the correct winding? | **NO — correct step ranks #3/8 (vs observed) and #5/8 (vs lo-fi ref).** All staircase omegas have 13-25° direction error. Removing shadow mismatch makes ranking worse, not better. Subsampling never recovers correct winding. | DONE |
| `micro22a_winding_score_nudged_qA.py` | Is ranking stable under q_A perturbation? | **Stable but wrong.** Rank stays at 3/8 for 0-3° nudge, slightly degrades at 5°. Attitude precision is not the bottleneck. | DONE |

**Root cause:** The staircase explores ω along one fixed rotation axis (rotvec(q_A⁻¹·q_B)/dt) which is 13-25° off from the true ω direction at peak A. All 8 winding solutions have the wrong direction, so all intermediate trajectories are wrong. Multi-epoch scoring IS discriminating between windings (MSE range 2.0-4.0) but NONE match the observed LC because the rotation axis is wrong.

**Conclusion:** Multi-epoch scoring would work if the omega direction were correct. The staircase implementation is the bottleneck, not the scoring concept. This motivates the L-conservation approach (Series 07b) which avoids intermediate brightness evaluation entirely.

See `07_multi_epoch_scoring/FINDINGS.md` for detailed analysis, ranking tables, and LC overlay plot.

---

### Series 07b — L-Conservation Winding Filter (2026-03-10)

Branch: `exp/L-conservation-winding-filter`. Tests whether angular momentum conservation L = R(q)·(I·ω) can discriminate the correct winding pair at shared peak nodes. micro18 failed because leg 1's staircase missed the true ω — this series injects truth to test the principle.

| Script | Question | Key Result | Status |
|--------|----------|------------|--------|
| `micro23_L_oracle_test.py` | With truth injected, does L-conservation identify the correct pair? | **YES — rank 1/64, gap = 113 kg·m²/s (9 orders of magnitude).** Both ||ΔL|| and |ΔT| rank truth #1. T is redundant — L alone suffices. | DONE |
| `micro24_L_nudge_sensitivity.py` | How much attitude error can L-consistency tolerate? | **100% correct at all nudges 0.5-10 deg.** Part A: q_mid nudge is mathematically invariant (R cancels). Part B: endpoint nudge grows true-pair L_err linearly (~1.2/deg) but gap stays >95. | DONE |
| `micro25_L_three_leg.py` | Does a 3rd leg (4 peaks) improve discrimination? | True triple ranks 1/512. 2 legs already sufficient for this test case. 3rd leg provides redundancy. | DONE |

**Key insight:** L-consistency at a shared node is equivalent to body-frame omega matching weighted by I: `||ΔL|| = ||I @ Δω||`. The shared-node quaternion cancels out entirely. The signal (winding gap ~113) vs noise (attitude error ~1.2/deg) gives SNR ~ 2π/δq — independent of leg duration.

**Conclusion:** L-conservation is a viable, robust winding filter. The blocking problem is ensuring the staircase on each leg includes the true winding.

See `07_L_conservation/FINDINGS.md` for detailed analysis and plots.

---

## 3. Superseded Work

### Global Optimisers on 6D Joint Space (Series 01)
- **Tried:** Differential Evolution, CMA-ES, dual annealing, basin-hopping, multi-start L-BFGS-B, alternating att/omega, brute-force grid, decoupled grid — every standard method.
- **Why abandoned:** Joint convergence basin is ~5 deg × 0.02 dps. With 6 DOF, the basin volume is negligibly small relative to the search space. No global optimiser found it within practical budgets (up to 50k evaluations, 45 min wall time).
- **Replaced by:** Structured approaches exploiting lightcurve physics (iso-brightness filtering → peak-graph pipeline).

### Sequential Cascade Filtering (Series 02, `exp_sequential_filter_v1-v3`)
- **Tried:** Multi-epoch brightness screening with progressive tightening. Each epoch filters ~100x, so K epochs should give 100^K reduction.
- **Why abandoned:** Combinatorial explosion: 100 survivors at each of K epochs = 100^K pairs to check. Even with FFT-based omega bounding (v3), the pairing step was prohibitive.
- **Replaced by:** Peak-anchored approach (only use ~3 peaks, not all epochs).

### Chain-Based Omega Bridging (Series 03, `exp_chain7_tight`)
- **Tried:** 7-epoch chain with 99% culling at each step: keep only the 1% of candidates whose propagated attitude best matches next epoch's brightness.
- **Why abandoned:** Aggressive culling (99%) frequently kills the true path. With 7 chained culling steps, P(truth survives) = 0.01^7 ≈ 0.
- **Replaced by:** Graph-based approach with softer ranking (keep top-50 per peak, score paths rather than cull).

### Analytic (Axisymmetric) Propagation (Series 03, `exp_q0_omega_analytic`)
- **Tried:** Closed-form attitude propagation assuming near-axisymmetric body (IS-901 has I2/I3 ≈ 0.992).
- **Why abandoned:** Asymmetry parameter = 0.556 — triaxial enough that analytic solution diverges from numerical over 3600s.
- **Replaced by:** Full Euler dynamics propagation (DOP853 ODE solver, 77ms for 500 epochs).

### Lo-fi Intermediate Brightness Scoring (Series 05, micro-13)
- **Tried:** Score graph paths by lo-fi brightness at intermediate epochs between peaks.
- **Why abandoned:** Lo-fi scores are nearly uniform across all feasible paths — no discrimination. Truth path ranked ~10th percentile.
- **Status:** Hi-fi rescoring (micro-14) also failed to clearly discriminate (truth rank ~13k/121k). **This remains the key open problem.**

### Multi-Epoch LC Shape Scoring (Series 07a, micro21+22a)
- **Tried:** Score micro17 staircase windings by lo-fi MSE at all ~78 intermediate epochs between peaks. Also tested with lo-fi reference (no shadow mismatch) and subsampling.
- **Why abandoned:** All staircase omegas have 13-25° direction error — the bridge only constrains endpoints, not the rotation axis. Every winding's intermediate trajectory is wrong, so multi-epoch scoring can't identify the correct one. Correct step ranks #3/8 at best.
- **Key insight:** The problem is omega *direction*, not winding *number*. Scoring is sound but requires correct omega direction first.
- **Replaced by:** L-conservation winding filter (Series 07b), which avoids intermediate brightness entirely.

### Residual-Based Candidate Ranking (Series 02, `exp_residual_vs_error`)
- **Tried:** Rank candidates by full-LC residual, hope truth is near the top.
- **Why abandoned:** Weak correlation between residual and attitude error. Truth not reliably in top 200 across random test cases.
- **Replaced by:** Peak-anchored constraints (brightness + derivative at peaks).

---

## 4. Decision Log

### 2026-01-21 → 2026-01-26: Foundation built
- **Tried:** Built dynamics module (attitude propagation), inertia computation, inversion infrastructure
- **Found:** Working forward model with principal-axis and tumbling modes
- **Decision:** Proceed to systematic investigation of inversion difficulty

### 2026-02-06: Pipeline reference complete
- **Tried:** Series 00 — landscape analysis, basin size, optimizer comparison, robustness, mixed-fidelity
- **Found:** Basin radius ~5 deg (attitude), local optimisers converge near truth, global optimisers fail on 6D
- **Decision:** Need structured candidate generation, not brute-force global search

### 2026-02-06 → 2026-02-09: Global search exhausted
- **Tried:** Every standard global optimiser (DE, CMA-ES, dual annealing, basin-hopping, multi-start, alternating, grid search, decoupled approaches)
- **Found:** None converge within practical budgets. Joint basin is impossibly narrow for blind search.
- **Decision:** Abandon global optimisers. Exploit lightcurve structure (iso-brightness, peak anchoring) for candidate generation.

### 2026-02-19: Iso-brightness filtering explored
- **Tried:** Single-epoch brightness matching, multi-epoch cascade, triplet consistency
- **Found:** Brightness is highly selective (~0.05% match at 1% tolerance) but sequential cascade has combinatorial explosion
- **Decision:** Need physics-based pruning. Pivot to omega bridging.

### 2026-02-20: Omega bridging explored
- **Tried:** 2-epoch pairs, 4-epoch chains, 7-epoch chains with aggressive culling
- **Found:** Chain-based culling kills truth path. Need softer ranking, fewer anchor points.
- **Decision:** Anchor on brightness **peaks** (dL/dt≈0 gives extra DOF constraint). Build graph over small number of peaks.

### 2026-02-23: Basin characterisation formalised
- **Tried:** exp00 (timing), exp01 (sanity checks) — systematic measurement
- **Found:** Lo-fi: 221ms, hi-fi: 60s (272x). Hi-fi residual at truth = 0.0024 (matches noise). Conservation laws hold to 1e-10.
- **Decision:** Numbers established. Proceed to peak-graph pipeline.

### 2026-02-26: Peak-graph pipeline built (micro01→micro11)
- **Tried:** Progressive micro-experiments: degeneracy → constraints → bridging → sampling → joint pipeline
- **Found:** 1M random: nearest ~2 deg at peaks. L-BFGS-B iso-brightness: 3.6x closer (1.64 deg). 10M: 13,848 candidates. Joint pipeline works but needs better seeds.
- **Decision:** Candidate generation is tractable. Focus shifts to scoring/ranking.

### 2026-02-27: Scoring problem identified + Roberto direction (micro12→micro14)
- **Tried:** Graph pipeline with lo-fi intermediate scoring (micro-13), hi-fi rescoring (micro-14)
- **Found:** Lo-fi scoring cannot discriminate true path. Hi-fi rescoring doesn't fix it (truth rank ~13k/121k). Peak time mismatch between lo-fi and hi-fi confirmed. Edge cost (LC residual at intermediate epochs) is insufficient regardless of fidelity — arbitrary ω can produce plausible-looking LC segments between peaks.
- **Decision (Roberto):** Scoring function is the bottleneck. New direction: (1) use minimum-magnitude ω with zero initial guess, then explore solution family by magnitude; (2) test dip constraint — check brightness troughs between peaks at hi-fi; (3) frame as sequence of optimisation problems, each step reducing the candidate space; (4) talk to Jack about mixed-integer programming formulation (integer vars for quaternion selection, continuous for ω).

### 2026-03-10: Winding ambiguity identified + staircase validated (micro16, micro17)
- **Tried:** Dip constraint scoring (micro16/16b), staircase ω winding recovery (micro17)
- **Found:** The axis-angle bridge is always the lowest-winding solution — wrong for a fast tumbler (~3 revolutions over 554s). This explains why micro16 had no signal: the bridged trajectory uses the wrong winding, so mid-interval attitude is completely wrong for ALL pairs. The staircase (micro17) confirms: 8 distinct winding solutions exist, all valid bridges (arrival_err~0). Each step adds ~one full revolution. True ω (~2.083 deg/s) corresponds to step 3. A single brightness dip cannot discriminate the correct winding — step 2 (1.601 deg/s) scored better by coincidence.
- **Decision:** Multiple intermediate scoring points (or hi-fi) needed to resolve winding ambiguity. micro16c tests embedding the trough inside the bridge objective. Next direction: use multiple dips across the LC as winding discriminators.

### 2026-03-10: L-conservation validated as winding filter (micro23-25)
- **Tried:** Injected true ω into both legs' staircase candidates, tested L = R(q)·(I·ω) conservation at shared peak nodes
- **Found:** L-conservation uniquely identifies correct winding pair: rank 1/64, gap = 113 kg·m²/s (9 orders of magnitude). Shared-node attitude error is mathematically irrelevant (R cancels). Endpoint error up to 10 deg → still 100% correct. Three legs provide redundancy but 2 are sufficient.
- **Decision:** L-conservation is the winding selector. The remaining problem is staircase coverage: leg 1 misses the true winding (jumps from 0.25 to 3.28 dps). Fix: denser staircase stepping, adaptive barrier gaps, or bidirectional search.

### 2026-03-01: Omega basin characterisation
- **Tried:** Systematic omega basin study (magnitude, direction, q-degradation) at lo-fi and hi-fi fidelity
- **Found:** Direction basin ~2 deg, magnitude basin ±10% (hi-fi) or ±5% (lo-fi). Attitude error >5 deg collapses omega basin.
- **Decision:** Quantitative targets confirmed. Pipeline must deliver candidates within ~5 deg (attitude) and ~2 deg omega direction for local refinement to succeed.
