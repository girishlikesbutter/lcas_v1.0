---
title: "Blind inversion in 15 min/seed — strategic plan + 3-day execution"
type: plan
sources:
  - experiments/s001..s081 (full survey workspace, post-fix)
  - PROGRESS.md (post-s081 banner)
  - concepts/*.md (forward-model invariants)
  - MEMORY.md (load-bearing methodology rules)
related:
  - experiments/s019_ls_bracket_omega_mag.md  # |ω| prior, 98/100 within 5%
  - experiments/s005_joint_local_descent.md   # joint LM grab radius
  - experiments/s068_replicate_s011.md        # 10/10 at truth-ω, 3.8 min/seed Pool(8)
  - experiments/s072_path2_closed_form_q.md   # Jacobi Path 2 closed-form q(t)
  - experiments/s073_cluster457_l_vector_check.md  # L_J2000 cross-anchor gate
  - experiments/s073b_path2_cprofile.md       # 81% of Path 2 wall = φ-ODE
  - experiments/s064_jacobi_polish.md         # polhode-basis LM, 21-27× speedup
  - experiments/s081_hifi_rho_bands_twin.md   # multi-sol structure at hi-fi
  - experiments/s059j_cloud_data_omega_grid.md  # local-window failure
  - experiments/s059k_densify_ndirs.md        # full-LC polish + multi-mag-start
  - experiments/s055c_aux_priors.md           # LC-only ω-direction priors are dead
  - experiments/s003_landscape_vs_omega.md    # truth-ω tube width
created: 2026-05-20
updated: 2026-05-20
confidence: high (architectural synthesis; pivot experiment is what validates the branch decision)
---

# Blind LC inversion in 15 min/seed — strategic plan + 3-day execution

> **Audience.** A fresh agent picking this up cold. You will have auto-loaded `MEMORY.md` + the project root `CLAUDE.md` + the survey `CLAUDE.md`. This document is the single-pane handoff for the inversion push covering Wed/Thu/Fri 2026-05-21..23.

> **Read order before doing anything.**
> 1. This document, end-to-end.
> 2. `notebooks/inversion/survey/README.md` — workspace contract.
> 3. `notebooks/inversion/survey/CLAUDE.md` — agent rules in this workspace.
> 4. `notebooks/inversion/survey/PROGRESS.md` — post-s081 banner.
> 5. The cited experiment `.md` files in this doc as you need them. Do NOT try to read all 81 experiments first; pull them by citation.

---

## TL;DR

**The hard goal**: blind inversion of an IS-901 LC to recover `(q0, ω)` candidates in Band A∪B (ρ < 4) in ≤15 minutes wall per seed, on 10 stratified seeds (5 LAM + 5 SAM, span slow→fast |ω|, span low→high mean PA), with no use of cached truth and no oracle regime labelling.

**Status**: viable. We have the tooling and the cost surface is honest under post-fix truth (multiple sources). What we don't yet know is the load-bearing empirical question — at full-LC Jacobi+surrogate scoring on a dense joint `(q0, ω)` grid, where does truth (or a Band-A multi-sol) rank? That single number picks between a simple two-step pipeline (Branch v2) and a multi-anchor-pruned variant (Branch v3).

**Plan**: Day 1 runs the gating experiment + parallel `elliprj` track. Day 2 implements the chosen branch end-to-end. Day 3 runs the stratified 10-seed cohort.

**The two architectures share Stages 1-3 and 7-8**:
1. **|ω| LS-bracket** (`s019`, 98/100 within 5%, p90 2.26% — source: `results/s019/summary.json`).
2. **K=2-3 sharp `|C_t|`-min anchors** (s060 pattern — source: `experiments/s060_sharpness_map.md`).
3. **Dense q-clouds at each anchor** via 400k Haar pool projection (s059i density scan — source: `experiments/s059i_cohort_density_scan.md`).
7. **Polhode-basis 6-DOF LM polish** (`s064`, 21-27× in-basin speedup — source: `results/s064_jacobi_polish/gate2_smoke_parity.json`) with multi-mag-start `[0, ±3, ±6]%` (`s059k` — source: `experiments/s059k_densify_ndirs.md`).
8. **Hi-fi gate** (surrogate-ρ < 4) → ρ-band classify + body-X twin tagging (`s081` pattern — source: `experiments/s081_hifi_rho_bands_twin.md`).

**They differ in the middle (the candidate-pruning machinery): see § Branch v2 vs v3 below.**

---

## 1. Goal and scope

### 1.1 Hard goal (user-stated, 2026-05-20)

> Wed + Thu + Fri morning: blind inversion ≤15 min/seed on at least 10 seeds spanning tumbling mode (LAM vs SAM) and viewing geometry (mean PA), demonstrating the pipeline is not biased to either.

### 1.2 Inputs that ARE given (the "blind" scope, user-confirmed 2026-05-20)

- IS-901 STL geometry + articulation defaults `SP=0°, AD=15°`. Source: `data/models/intelsat_901/intelsat_901_config.yaml`; verified by `STLLoader` in `lib/hifi_render.py`.
- Inertia tensor (m048 diagonal: `I = diag(37985.16, 38305.71, 7749.01) kg·m²`, source: `data/results/inversion_diagnostics/m048_trajectories/m048_trajectories.npz` and `experiments/s073_cluster457_l_vector_check.md:124`).
- SPICE ephemeris for sun + observer + satellite positions per epoch.
- The post-fix propagator (`src/dynamics/attitude_propagator.py`, commit `d5705ff`).
- The v2 residual-ensemble surrogate (`~/surrogate_model`, bridge-independent — concept `concepts/surrogate_model.md`).
- The Jacobi Path 2 closed-form q(t) (`lib/jacobi_propagator.py::propagate_jacobi_path2`).
- Apparent-magnitude noise floor `σ = 0.05 mag` (the ρ-band convention, `concepts/rho_band.md`).

### 1.3 Inputs that are NOT given (the inversion's actual unknowns)

- The truth `(q0_wxyz, omega0_rad)` for any seed. The pipeline MUST NOT load `traj_seedXXX.npz`'s `q0_wxyz` or `omega0_rad` or cached `quaternions` arrays for any seed during the search. Truth is loaded ONLY at the END for error reporting.
- The tumbling regime (LAM/SAM). If a regime classification helps the pipeline, it must be a per-candidate diagnostic computed from `(q_a, ω_a)` (e.g., `disc = 2T·I_2 − |L|²`), NOT an oracle look-up.
- The truth ω-direction.

### 1.4 Success metric (user-confirmed 2026-05-20)

**Hybrid.** Primary gate: per seed, return ≥1 candidate at `ρ_hifi < 4` (Band A∪B). Multi-solutions count: Group-A ω-scaling alternates, body-X twins, and SAM-class competing basins are all valid. Secondary metric (reported, not gated): truth in the candidate list.

Rationale (user, paraphrased): "in the real world, ρ<4 solutions are indistinguishable from truth at the noise floor. If we find truth it'll vastly out-compete; if we don't, we won't know we haven't. So the operational metric is `did we fit the LC?`, not `did we find truth?`."

### 1.5 Validation cohort (user-confirmed 2026-05-20)

**5 LAM + 5 SAM**, span `|ω|` slow→fast, span mean PA low→high. Mix m048 cohort (seeds 0-99) + holdout (seeds 100-119) — see § 7 below for the specific selection procedure.

### 1.6 Compute budget (user-confirmed 2026-05-20)

**Pool(24) on the 30 GB dev box**, dropping to Pool(16) for hi-fi rendering stages per the `s081` trimesh-gc fix (source: `feedback_pool_size_with_trimesh_gc.md`).

---

## 2. The capability landscape — what we have, what we know works, what we know doesn't

### 2.1 What works (load-bearing positives)

| Capability | Established by | What it gives | Status |
|---|---|---|---|
| **|ω| LS-bracket prior** | `s019` (source: `results/s019/summary.json`) | 98/100 seeds within 5% of truth-|ω| via 5-cell geomspace grid; median offset 1.25%, p90 2.26%. 7.5s wall for all 100 seeds. | Production-ready |
| **Sobol-Shoemake(SO(3)) N=64 + joint 6-DOF LM @ truth-ω** | `s068` post-fix replication of `s011` (source: `results/s011/summary.json`) | 10/10 PA-stratified pilot seeds in-basin; 38 min Pool(8) ≈ 3.8 min/seed *at known ω*. | Reference baseline (truth-ω assumed) |
| **Joint 6-DOF LM grab radius** | `s005` (source: `results/s005/summary.json`) | q0 ~5-15° (seed-dependent), ω-dir 1-2°, |ω| 3-5%. Bimodal convergence — strict = loose. Twin not a near-truth attractor (0/50). Seed 28 tightest at ~2°/1°/2%. | Empirical fact |
| **Jacobi Path 2 closed-form q(t)** | `s072` (source: `results/s072/summary.json`) | 16.4 ms / 500-epoch trajectory; q vs DOP853 ≤ 2.81e-10 worst-case; conservation 1e-15. Bottleneck = scipy `solve_ivp(phi_dot)` at 81% of wall (`s073b`). | Production-ready |
| **L_J2000 cross-anchor gate** | `s073` (source: `results/s073/summary.json`) | Propagation-free 3-DOF gate on (q_A, ω_A, q_B, ω_B) pair-space. Truth vs Band-A multi-sol on seed 89: |L| matches 0.55%, **direction differs 128.6°** — direction is the load-bearing axis. | Validated N=1; cohort generalisation untested |
| **Polhode-basis 6-DOF LM polish** | `s064` (source: `results/s064_jacobi_polish/gate2_smoke_parity.json`) | 21-27× in-basin convergence speedup vs raw-ω-component LM. Drop-in replacement for `s058::lm_polish`. | Production-ready |
| **Body-X twin halving** | `s043` (source: `experiments/s043_twin_hifi_verify.md`) | Bit-exact LC equivalence under `q_180x · q0`. Free 2× speedup at every search stage via `lib/twin.py::canonical_batch`. | Production-ready |
| **Full-LC surrogate-MSE as cost surface** | `s002`, `s014`, `s081` (sources: `results/s002/summary.json`, `results/s014/summary.json`, `results/s081/summary.json`) | argmin = truth-q0 at fixed truth-ω on 8/8 PA-stratified; Spearman 0.9952 vs hi-fi on 9-seed pilot; **145/145 surrogate↔hi-fi exact band agreement** on s081 rendered set. | Production-ready |
| **Multi-mag-start `[0, ±3, ±6]%`** | `s059k` (source: `experiments/s059k_densify_ndirs.md`) | Surfaces Group-A multi-sols. Necessary for landing the |ω|-scaling-family Band A endpoints. | Production-ready |
| **Holdout trajectories 100-119** | `s054` (source: `experiments/s054_generate_holdout.md`) | 20 fresh post-fix trajectories from the same m048 generator. Available via `lib.traj_load.load_truth(seed)` for seed ∈ [100, 119]. | Available |

### 2.2 What does NOT work (decisive negatives — do not relitigate)

| Approach | Closed by | Why it fails |
|---|---|---|
| **Decoupled ω-outer / q0-inner search** | `s003` (source: `results/s003/summary.json`) | Outside the ~1° dir × 2-5% mag truth-ω tube, the surrogate landscape is incoherent multi-basin noise. The per-ω best-MSE signal has no gradient pointing toward truth-ω. **Closes m103/m115/m126-style nested decoupling.** |
| **LC-only ω-direction priors** | `s055b`, `s055c` (sources: `results/s055b_lhat_lc_regression/...`, `results/s055c_aux_priors/summary.json`) | L̂ regression: 74° holdout median, worse than 90° random baseline by only 15°. |cos(L̂, PAB)| regression: holdout RMSE 0.299 vs cohort-mean baseline 0.285 (model worse than constant). Polhode-binary class: holdout acc 0.80 vs majority baseline 0.85. **LC features carry ω-magnitude info but no usable ω-direction info at the s008-feature granularity.** Don't propose LC-feature→ω-direction approaches again without new feature engineering. |
| **Constant-ω propagation over a full 60-min LC** | `s063c` (source: `results/s063c/summary.json`) | Median polhode period 1594s vs 3600s LC. ω cycles 2-3× in body frame across the LC for 119/120 cohort seeds. Constant-ω represents a non-physical trajectory over the full LC. (Constant-ω is fine over short windows ≲ 200s — that's what `s059j`'s local-window scoring uses, which has its own problem below.) |
| **Local-window scoring (W=10)** | `s059j` (source: `results/s059j_cloud_data_omega_grid/seed028/summary.json`) | Wider W *amortises* q_a noise but *amplifies* ω-direction noise. At 5° ω-quantisation, even the cell nearest truth has ρ=38-61 across all W ∈ {5..200}. **Score on the full LC, not a window.** |
| **Local-window LM polish** | `project_local_window_polish_phantom_basins.md` (source: `experiments/s059k_densify_ndirs.md`) | W=10 polish converges to phantom basins satisfying the window but disagreeing with full-LC hi-fi by ρ=30-70. **Polish on the full LC.** |
| **Horizon-density anchor score (L3)** | `s076` (source: `results/s076/summary.json`) | IS-901's cuboid bus reverses the L3 premise: high horizon-density correlates with **bright**, not dim. The sharp-anchor mechanism is BRDF-specular, not aspect-boundary-density. Stick with empirical `|C_t|`-minimum anchor selection. |
| **Validators that insert truth at idx 0** | `feedback_validators_must_match_production_cost.md` (source: `experiments/s059j_cloud_data_omega_grid.md`) | A validator that pre-inserts truth into the grid measures basin sharpness, not production discrimination. The `s059i` validators looked positive (rank 1/1407 at q_a noise ≤ 7.5°) because they inserted truth-ω at idx 0; the production search at fixed grid quantisation lands ρ=38 at the closest cell. **Validators must use the same discretisation as production.** |

### 2.3 Open / ambiguous tooling

| Tool | Status | Decision |
|---|---|---|
| **elliprj Path 2** (`s074`, projected) | Half-day work. Replaces `scipy.special.elliprj` for the φ-ODE, removing 81% of Path 2 wall (`s073b`) and making Path 2 fully vectorisable across candidates. Expected ~5× per-candidate speedup. | Run **in parallel** on Day 1 (per user direction 2026-05-20). |
| **NLL residual** (`s078`) | A/B'd negative on seed 89; stays opt-in in `lib/surrogate_eval.nll_cost` for a future real-noise thread (source: `experiments/s078_nll_residual_ab.md`). | **Do not adopt** into the production cost path. |
| **Cluster-457 cohort generalisation** | s073 is N=1. The "L_J2000 cross-anchor matching discriminates Band A multi-sols" claim is hypothesised, not cohort-validated. | The pivot experiment will incidentally probe this. |

---

## 3. The 15-minute math — where the wall budget goes

### 3.1 The fundamental constraint

s003 said: **truth-q0 is the surrogate argmin only inside a ~1° ω-direction × ~2-5% ω-magnitude tube around truth-ω.** Outside the tube, the surface is multi-basin incoherent noise.

s005 said: **joint 6-DOF LM converges to truth basin only from inside that tube.** Grab radius is bimodal — strict success or 20°+ failure, no intermediate.

s019 said: **|ω| is essentially free** — LS bracket gets us within 2-5% on 98/100 cohort seeds.

**Therefore the inversion problem reduces to: get an ω-direction estimate within 1-2° of truth, then joint 6-DOF LM finishes the job.** s068 gives the post-LM wall: 3.8 min/seed Pool(8) ≈ 1.3 min/seed Pool(24).

That leaves ~13 min/seed for ω-direction acquisition.

### 3.2 Per-stage per-candidate cost (single thread)

| Operation | Cost | Source |
|---|---|---|
| Jacobi Path 2 forward (500 epochs) | 16.4 ms | `s073b` profile (median over 5 runs) |
| Surrogate full-LC eval (500 epochs, batched) | ~3-5 ms (estimate; verify Day 1 morning) | `concepts/surrogate_model.md` claims ~50000× hi-fi speedup |
| Polhode-basis LM polish, in-basin (4 s, 50-200 nfev) | 4 s | `s064` Gate 2 |
| Polhode-basis LM polish, out-of-basin | 25-100 s | `s064` Gate 2 |
| Hi-fi render (single LC) | ~1 s | `s081` infers from 145 renders in 844 s Pool(16) |

### 3.3 Joint candidate density vs Pool(24) budget

| N_candidates | Path 2 (16 ms each) wall Pool(24) | Path 2 + surrogate (~20 ms each) wall Pool(24) |
|---:|---:|---:|
| 100k | 67 sec | 83 sec |
| 500k | 5.6 min | 7 min |
| 1M | 11 min | 14 min |
| 2M | 22 min | 28 min |

**Budget ceiling ~1M candidates per seed at current Path 2 speed.** With elliprj Path 2 (~3 ms), ceiling rises to ~3-5M.

### 3.4 Required ω-direction grid density vs LM grab radius

For Fibonacci sphere with N points, average angular spacing ≈ √(41,253 / N) degrees.

| N_dir (full sphere) | avg spacing | post-twin-halving N_q | covers LM grab radius? |
|---:|---:|---:|---|
| 200 | 14.4° | n/a | No — `s059j` failure regime |
| 800 | 7.2° | n/a | Marginal — `s059k` borderline regime |
| 2000 | 4.5° | n/a | Most cells outside basin; truth-nearest probably outside |
| 5000 | 2.9° | n/a | Most cells outside basin; truth-nearest probably inside |
| 10000 | 2.0° | n/a | Truth-nearest near basin edge |
| 20000 | 1.4° | n/a | Truth-nearest inside basin for most seeds |
| 41000 | 1.0° | n/a | Truth-nearest reliably inside basin |

**With N_q = 64 Sobol-Shoemake (32 after twin-halving) × N_mag = 5 LS-bracket cells × N_dir candidates**, total = 160 × N_dir:

| N_dir | Total candidates |
|---:|---:|
| 800 | 128k |
| 2000 | 320k |
| 5000 | 800k |
| 10000 | 1.6M |
| 20000 | 3.2M |

**Sweet spot for Path 2 today: N_dir ≤ 5000 (800k candidates ≈ 11 min Pool(24)).** That puts the closest-to-truth cell at ~2-3° in ω-direction — outside or at the edge of the joint LM basin (~1-2° per `s005`).

**This is the crux.** At N_dir = 5000, will the closest grid cell to truth rank high enough in surrogate-MSE that a top-K polish reaches truth? Open empirically.

With elliprj Path 2: N_dir = 10000-20000 (1.6M-3.2M candidates ≈ 4-8 min Pool(24) after the speedup). Closest cell ~1-1.5° — confidently inside basin.

### 3.5 The wall-budget conclusion

There are two viable architectures, separated by how cheap the per-candidate evaluation is:

- **If we accept N_dir ≤ 5000 and current Path 2**: we need a *pre-propagation filter* to prune candidates before evaluation. → **Branch v3** (multi-anchor + L_J2000).
- **If elliprj Path 2 lands AND closest-cell rank is good**: simple dense joint grid + top-K polish works. → **Branch v2**.

The pivot experiment in § 5 decides between them.

---

## 4. Architecture — two branches

Both branches share the same outer-loop sequence:

```
INPUTS: observation_times[N], mag_observed[N], sun_pos[N,3], obs_pos[N,3],
        sat_pos[N,3], inertia_tensor[3,3], stl_geometry, articulation_defaults
        (NO truth, NO regime label, NO cached arrays)

STAGE 1: |ω| LS-bracket            →  5-cell |ω| grid covering ±5%
STAGE 2: Anchor selection          →  K=2-3 sharp |C_t|-min anchors
STAGE 3: Dense q-clouds at anchors →  |C_a_i| ≈ 1-5k each (post-LC-tol)
STAGE 4: Candidate generation      →  joint (q, ω) pool
STAGE 5: Candidate scoring/pruning →  top-K survivors
STAGE 6: Cluster                   →  K=50-200 cluster reps
STAGE 7: Polhode-basis LM polish   →  multi-mag-start, full-LC residual
STAGE 8: Hi-fi gate + ρ-band       →  per-candidate (band, twin_label)

OUTPUTS: result.json with per-candidate (q0, ω0, ρ_surrogate, ρ_hifi,
         band, twin_label, q0_err_vs_truth, ω_dir_err, ω_mag_err, walls)
```

Stages 4 and 5 are where v2 and v3 differ.

### 4.1 Branch v2 — Dense joint grid + full-LC surrogate rank + top-K polish

**Premise**: full-LC Jacobi+surrogate puts truth-near cells in a polishable top-K (≤200) on a joint grid of ≤2M candidates.

**Stage 4 (v2)**: Sobol-Shoemake(SO(3)) N=64 (→ 32 after twin canonicalisation) × Fibonacci(ω-dir) N=2000-5000 × |ω|-bracket N=5 = **320k-800k joint candidates**.

**Stage 5 (v2)**:
- For each candidate: `propagate_jacobi_path2(q0, ω0, I, observation_times) → q(t), ω(t)` (500 epochs).
- Compute `k1_body(t), k2_body(t)` from `R(q(t))^T @ (sun_pos[t] - sat_pos[t])/|...|` and similar for observer.
- Surrogate full-LC predict: `mag_pred[N] = surrogate.predict_magnitude(k1, k2, SP=0°, AD=15°, obs_dist)`.
- `mse = mean((mag_pred - mag_observed)²)`; `ρ_surr = √mse / 0.05`.
- Pool(24) across candidates.

**Stage 6 (v2)**: Top-K=200 by `ρ_surr` → greedy cluster (q-geodesic 5°, ω-direction 5°, |ω| 5%) → cluster representatives.

**Stage 7 (v2)**: Polhode-basis 6-DOF LM polish (`s064`) on each cluster representative × 5 mag-offsets `[0, +3%, -3%, +6%, -6%]` (`s059k`). Polish residual = surrogate full-LC. Max nfev = 200.

**Branch v2 will work IFF the pivot experiment shows truth-near cells in top-200.** Pivot decides.

### 4.2 Branch v3 — Multi-anchor consistency filter + L_J2000 + Jacobi + polish

**Premise**: pivot shows full-LC scoring alone doesn't selectively rank truth-near candidates, OR we want a more defensible architecture for the cohort sweep.

**Stage 4 (v3)**: At the PRIMARY anchor (sharpest `|C_t|`-min): generate joint (q_a, ω_a) candidates from `|C_a_0| × N_dir × N_mag`. With `|C_a_0| ≈ 5000, N_dir = 2000, N_mag = 5` → 50M candidates. (For Stage 5, we'll prune this dramatically.)

**Stage 5 (v3) — pre-propagation prune**:
1. For each candidate, compute `L_J2000_0 = R(q_a)^T @ I @ ω_a` (3 matmuls, ~100ns/candidate, 50M × 100ns = 5 sec total).
2. **Optional but recommended**: |L| filter via LS-bracket on |L| ≈ I_avg · |ω| (further prune by ~30%).
3. Build KD-tree on L_J2000 vectors per anchor.

**Stage 5b (v3) — multi-anchor consistency filter (the key novel step)**: For each (q_a_0, ω_a_0) candidate at the primary anchor:
- Jacobi Path 2 forward to all secondary anchors A_1, ..., A_K (cheap — 1 Path 2 call per candidate gives all q(t) values at once).
- For each secondary anchor A_j:
  - Find the nearest q in `C_a_j` to the predicted `q(t_A_j)`.
  - Compute q-geodesic and surrogate-predicted-mag-at-anchor.
  - **Hit** if `q-geo < tol_q (e.g. 5°)` AND `|mag_pred - mag_observed| < tol_mag (e.g. 0.1 mag)`.
- Score per candidate = anchor_hit_count + full_LC_surrogate_MSE (weighted).

**Why this is cheaper than v2**: by gating on multi-anchor consistency BEFORE full-LC scoring, we throw away candidates that disagree with the observed LC at known sharp epochs. The L_J2000 conservation is implicit in this filter (a true trajectory has consistent L_J2000 at all anchors; if Path 2 propagates correctly from A_0 to A_j, that's already enforced). The explicit L_J2000 KD-tree match is a debug/diagnostic tool, not a primary filter.

**Wall budget for v3 Stage 5b**:
- Path 2 per candidate (full-trajectory): 16 ms (current) or ~3 ms (elliprj).
- Per-candidate cost: Path 2 + K anchor checks (each anchor check is ~1 μs nearest-neighbour query + ~5 μs surrogate-at-one-epoch).
- At 16 ms × 5M filtered candidates = 80,000 sec single-thread = 56 min Pool(24). Too slow without elliprj.
- With elliprj at 3 ms × 5M = 15,000 sec single = 10 min Pool(24). Feasible.

**Branch v3 is "expensive enough that elliprj is recommended; expensive enough that it must be the right answer when v2 fails."**

**Stages 6-8 same as v2.**

### 4.3 Where v3's "L_J2000" lever actually sits

To be precise about what `s073` showed and what v3 leverages:

- `L_J2000` is conserved across time by torque-free dynamics (post-fix, machine precision per `s067`).
- Two candidates `(q_A_i, ω_A_i)` at anchor `A_i` and `(q_A_j, ω_A_j)` at anchor `A_j` are on the same trajectory IFF `L_J2000_i = L_J2000_j` (necessary, not sufficient).
- The trajectory consistency check — does Path 2 from `(q_A_i, ω_A_i)` over `(t_A_j - t_A_i)` land at `(q_A_j, ω_A_j)`? — is the SUFFICIENT version.
- **v3 uses the sufficient check directly** via the Path 2 forward + nearest-q in the secondary cloud. L_J2000 as an explicit KD-tree filter is a backup/diagnostic — useful for understanding why v3 prunes what it does, but not the load-bearing mechanism.

### 4.4 Why not other architectures

- **Branch v1 (decoupled ω-grid + q0-DE)**: closed by s003 (incoherent landscape outside truth-ω tube).
- **Cascade-style multi-epoch q-cloud intersection** (s048-s050): repeatedly failed; sensitive to per-epoch noise + per-pair finite-diff noise; structural problems in v1 surrogate (not relevant under v2 surrogate but architectural problems remain).
- **Per-pair finite-diff ω from q-cloud** (s057c-f): finite-diff at small geodesic distance is geometrically noisier (`feedback_finite_diff_small_geodesic_noise.md`).
- **Forward-propagation consistency at constant-ω** (s057g): 4400× over null, but constant-ω is invalid over full LC (s063c). Branch v3 IS this idea with Jacobi Path 2 substituted for constant-ω.
- **LC-feature→ω regression**: closed by s055b/c.

---

## 5. The pivot experiment (Day 1) — `s082_joint_grid_pivot`

### 5.1 Goal

Measure where truth (and Band A∪B candidates) rank in the **full-LC Jacobi+surrogate-MSE ordering** on a dense joint `(q0, ω)` grid, on 3 fresh holdout seeds. The answer picks Branch v2 vs v3.

### 5.2 Seeds

**3 fresh holdout seeds from 100-119** (no prior-stare bias, user directive 2026-05-20).

Picking procedure (Day 1 morning, ~5 min wall):
1. For each seed in 100..119, load `(q0, ω0, inertia)` via `lib.traj_load.load_truth(seed)` and compute polhode invariants via `lib.jacobi_propagator._build_omega_func` (used at truth-q0, truth-ω here purely for STRATIFICATION metadata — NOT in the pipeline itself).
2. Compute regime label `disc = 2T·I_2 − |L|²` (Case A = LAM, Case B = SAM).
3. Compute |ω| in dps and mean PA across the LC (from cached `k1_body`, `k2_body`).
4. Bin into a 2×2 grid: {LAM, SAM} × {slow, fast |ω|}.
5. Pick 3 seeds covering different bins. Preference: 1 LAM-slow, 1 LAM-fast, 1 SAM (any |ω|).

Save the stratification metadata at `results/s082/holdout_stratification.json`. This metadata IS NOT used in the pipeline; it's used to PICK the test seeds and to interpret outcomes.

### 5.3 Method

For each of the 3 chosen seeds:

```
# All inputs from the seed's NPZ EXCEPT q0_wxyz, omega0_rad, quaternions
ctx = lib.hifi_render.build_context(seed)  # gives sun_pos, obs_pos, sat_pos, observation_times, mag_hifi (=mag_observed), inertia_tensor
mag_observed = ctx['mag_hifi']
N = mag_observed.shape[0]  # 500 epochs

# Stage 1: |ω| LS-bracket
omega_mag_grid = lib.lc_features.ls_bracket(observation_times, mag_observed)  # 5 cells
# Implementation reference: experiments/s019_ls_bracket_omega_mag.py

# Stage 4 (v2-style joint grid)
q0_pool = lib.c_t_pipeline.sample_so3_pool(n_samples=64, sample_seed=42)['q_pool_wxyz']
q0_pool = lib.twin.canonical_batch(q0_pool)  # halve to ~32 unique
dir_grid = fibonacci_sphere(N_DIR=2000)  # full sphere, N_DIR_TOTAL points

candidates = []
for q0 in q0_pool:
    for dir_hat in dir_grid:
        for om_mag in omega_mag_grid:
            candidates.append((q0, dir_hat * om_mag))
# 32 × 2000 × 5 = 320k candidates per seed

# Stage 5: Jacobi Path 2 + full-LC surrogate scoring (Pool(24))
def score_candidate(q0, om):
    q_hist, om_hist = lib.jacobi_propagator.propagate_jacobi_path2(q0, om, ctx['inertia_tensor'], ctx['observation_times'])
    # Compute k1_body, k2_body via R(q_hist) and ctx['sun_pos'/'obs_pos'/'sat_pos']
    # surrogate predict
    # mse vs mag_observed
    return mse

# Output per seed
result = {
    'seed': seed,
    'regime': 'LAM' or 'SAM',
    'n_candidates': 320000,
    'walls': {...},
    'mse_distribution': {'min': ..., 'p1': ..., 'p5': ..., 'p50': ..., 'median': ...},
    'truth_metrics': {  # CRITICAL: loaded ONLY here, for reporting
        'q0_truth_wxyz': ctx['q0_wxyz'],
        'om_truth_rad': ctx['omega0_rad'],
        'closest_grid_cell_to_truth': {idx, q0_err_deg, om_dir_err_deg, om_mag_err_pct, mse, rank_by_mse},
        'top_50_q0_err_distribution': [...],
        'top_50_om_dir_err_distribution': [...],
        'rank_of_truth_nearest_cell': int,  # IF truth-nearest cell in top-K, what K?
    },
}
```

### 5.4 Decision tree based on `rank_of_truth_nearest_cell`

| Outcome (rank of truth-nearest cell, median over 3 seeds) | Branch |
|---:|---|
| ≤ 200 | v2. Simple top-K polish works. |
| 200 - 10,000 | v3. Multi-anchor consistency filter needed to selectively rank. |
| > 10,000 | architectural concern. Probably means densify N_dir → run elliprj first. Reconsider Day 2 plan. |

### 5.5 Anti-patterns to avoid

- **DO NOT** sort candidates and report only top-50 without saving the full per-candidate `(q0, ω, mse, q0_err, om_dir_err, om_mag_err)` array. Save as NPZ — it's the raw material for the Day 2 design.
- **DO NOT** load `q0_wxyz` or `omega0_rad` from the NPZ during the scoring stage. Truth loads only in the final reporting block.
- **DO NOT** insert truth into the grid at idx 0 (the s059i validator trap, `feedback_validators_must_match_production_cost.md`).
- **DO** set `OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1` in the env AND `torch.set_num_threads(1), torch.set_num_interop_threads(1)` in the worker init (`feedback_blas_threads_for_pool.md`).

### 5.6 Expected wall

3 seeds × 320k candidates × ~20 ms / 24 workers ≈ 13 min total (1 min anchor setup + 12 min joint eval). Add ~5 min for stratification + write-up = ~20 min Day 1 morning.

### 5.7 Artefacts to produce

```
notebooks/inversion/survey/experiments/s082_joint_grid_pivot.py
notebooks/inversion/survey/experiments/s082_joint_grid_pivot.md
notebooks/inversion/survey/results/s082/
    holdout_stratification.json
    seed{xxx}/
        candidates.npz       (all 320k (q0, ω, mse, errors))
        summary.json
        rank_distribution.png
        top_100_scatter.png
notebooks/inversion/survey/results/s082/branch_decision.md  (the explicit decision + rationale)
```

---

## 6. Parallel track — `s074_elliprj_path2` (Day 1 afternoon, separate agent)

### 6.1 Goal

Replace `solve_ivp(phi_dot, ...)` in `lib/jacobi_propagator.py::propagate_jacobi_path2` with `scipy.special.elliprj`-based closed-form Π(n, φ | m) evaluation. Expected per-candidate Path 2 wall drops from 16.4 ms → ~3 ms.

### 6.2 Method

Math (closed-form for the φ-integral):

`dφ/dt = |L| (2T − I₃ ω₃²) / (L² − I₃² ω₃²)`

This integrates to a Π-function form by parameter substitution. Carlson's R_J relates to Π as documented in DLMF 19.16. See `concepts/jacobi_propagation.md` for derivation references.

`scipy.special.elliprj(x, y, z, p)` computes Carlson's R_J. The transformation from (n, φ, m) to (x, y, z, p) is a standard form (DLMF 19.25.14).

### 6.3 Gates (three-gate validation, same as s072)

1. `||q_path2_new − q_path2_current||_∞ < 1e-9` (or ≤ DOP853 noise floor — same accuracy on the Gates 1/2/3 seeds of s072).
2. `2T, L²` rel-std < 1e-12 (conservation).
3. Wall reduction confirmed via cProfile: per-candidate ~3-5 ms.

### 6.4 Cohort regression test

After landing the elliprj path, **re-run `s072` three-gate** + add a 4th gate: a Pool(24) timing run on 1000 random `(q0, ω)` candidates and report median per-candidate wall. Required: ≤ 5 ms median; ideal: ≤ 3 ms.

### 6.5 Integration

`propagate_jacobi_path2(q0, ω0, I, times, method='elliprj' | 'ode')` with `elliprj` as default once gates pass. The `ode` path stays as fallback for `k² → 1` seeds (only seed 7 of 120, per `s063c`).

### 6.6 Failure mode

If elliprj doesn't land cleanly in Day 1 afternoon (e.g., the Π transformation requires a parameter-region split for stability), revert to `ode` and proceed with current Path 2. Day 2/3 pipeline still works, just at lower N_dir.

### 6.7 Artefacts

```
notebooks/inversion/survey/experiments/s074_elliprj_path2.py
notebooks/inversion/survey/experiments/s074_elliprj_path2.md
notebooks/inversion/survey/lib/jacobi_propagator.py  # MODIFIED, method='elliprj' added
notebooks/inversion/survey/results/s074/
    gate_validation.json
    cprofile_post_elliprj.txt
    wall_distribution_1000_candidates.json
```

---

## 7. Day 3 validation cohort — seed selection

### 7.1 Stratification target

5 LAM + 5 SAM, span slow→fast |ω|, span low→high mean PA. User-confirmed 2026-05-20.

### 7.2 Selection procedure (Day 3 morning)

The pipeline must run **blind** on each seed. To stratify the seed selection itself, we use truth metadata ONCE, ONLY for picking — same pattern as the pivot.

```python
candidates = list(range(120))  # m048 0-99 + holdout 100-119
metadata = []
for s in candidates:
    truth = lib.traj_load.load_truth(s)
    inv = lib.jacobi_propagator._build_omega_func(I_pa, omega0_pa)[1]
    metadata.append({
        'seed': s,
        'regime': 'LAM' if disc(...) > 0 else 'SAM',
        'omega_mag_dps': np.degrees(np.linalg.norm(truth['omega0_rad'])),
        'mean_pa_deg': compute_mean_pa(truth['k1_body'], truth['k2_body']),  # from cached
    })
# Sort by regime, then bin |ω| (3 bins: slow ≤ 0.4 dps, mid 0.4-0.9, fast ≥ 0.9)
# and PA (3 bins: low ≤ 40°, mid 40-70°, high ≥ 70°)
# Pick 5 LAM + 5 SAM with maximum coverage across the (|ω|-bin, PA-bin) joint grid.
```

Save selection at `results/s083/cohort_selection.json` with seeds + their (regime, |ω|, mean PA).

### 7.3 Expected outcome

From PROGRESS history (cohort distribution per `s063c`): 104 LAM / 16 SAM in m048. Stratifying 5 SAM is going to use most of the available SAM seeds.

Known seeds (label from `s081`/PROGRESS):
- **LAM**: 6 (wide basin), 10 (multi-sol rich), 44, 48, 60, 84
- **SAM**: 21, 28 (near-separatrix, tight basin), 41 (clean), 91 (m115 anchor)

Plus 20 holdout seeds (100-119) of unknown regime distribution — likely ~17 LAM / 3 SAM by m048 base rate.

A workable picks:
- LAM (5): 6, 10, 60, + 2 holdout LAMs (slow + fast)
- SAM (5): 28, 41, 91, + 2 holdout SAMs (or use 21 + 1 holdout SAM if only 1 SAM available in holdout)

### 7.4 If holdout SAM availability is too low

Fall back: 5 LAM + 4 SAM (all 4 known SAM from m048: 21, 28, 41, 91) + 1 LAM repeat for the 10th = 6 LAM + 4 SAM. Document the deviation in the writeup.

---

## 8. Day-by-day execution plan

### Day 1 — Wednesday 2026-05-21

**Morning (Track A — pivot)**:
- Implement `s082_joint_grid_pivot.py`.
  - `lib.lc_features.ls_bracket(times, mag)` — implement if not present (port from `experiments/s019_ls_bracket_omega_mag.py`).
  - `lib.fibonacci_sphere(N)` — implement if not present (standard formula).
  - The candidate-scoring main loop: Pool(24), Path 2 + surrogate full-LC.
- Pick the 3 holdout seeds via stratification.
- Run the pivot. ~15-20 min wall.

**Afternoon (Track A continued)**:
- Read s082 results. Decide Branch v2 vs v3 vs reconsider.
- Write `s082_joint_grid_pivot.md` with the decision rationale.
- Commit on `forward_survey` (autocommit, no Co-Authored-By per `feedback_no_coauthor.md`).
- Begin Day 2's pipeline scaffold.

**Afternoon (Track B — separate agent, parallel)**:
- Implement `s074_elliprj_path2`. Three-gate validation.
- If gates pass: commit, integrate into `lib/jacobi_propagator.py` as default with `method` flag.
- If gates fail: document the obstacle and leave the existing Path 2 as the production path.

### Day 2 — Thursday 2026-05-22

**Morning**:
- Implement the chosen branch end-to-end as `lib/inversion_v0.py` (or appropriately-named module).
- Components:
  - `stage_ls_bracket(times, mag) → omega_mag_grid`
  - `stage_anchor_select(times, mag, sun_pos, obs_pos, sat_pos, k=3) → anchor_indices`
  - `stage_dense_q_cloud_at_anchor(anchor_idx, n_haar=400000) → C_a (q_pool, k1, k2, ...)`
  - `stage_generate_candidates(q_clouds, anchors, omega_mag_grid, n_dir, branch='v2'|'v3') → candidates`
  - `stage_score_candidates(candidates, surrogate, branch) → top_K`  (Path 2 + surrogate full-LC + optional anchor-hit count)
  - `stage_cluster(top_K) → cluster_reps`
  - `stage_polish(cluster_reps, multi_mag_offsets=[0, 3, -3, 6, -6]) → polished_states`
  - `stage_hifi_gate(polished_states, surrogate_rho_threshold=4) → hifi_states`
  - `stage_classify(hifi_states) → (band, twin_label) per state`

- Create a driver: `experiments/s083_blind_pilot.py`.
- Validate on the SAME 3 holdout seeds from the pivot (apples-to-apples).

**Afternoon**:
- Iterate on Day 2's outputs. Time each stage. Fix the slowest stage if budget overshoots 15 min.
- Multi-sol acceptance check: report not just top-1 but ALL Band A∪B candidates per seed (truth in candidate list as a sub-metric).
- Commit + writeup `s083_blind_pilot.md`.

### Day 3 — Friday 2026-05-23 (morning only)

**Morning**:
- Run cohort selection (§ 7.2).
- Run the pipeline on the 10-seed cohort. Sequential (each seed in 15 min Pool(24)) → ~2.5 hours total wall.
  - OR parallelise across machines if available; user noted "I will probably get two agents to work at the same time" — could be two-agent cohort split.
- Aggregate results into `experiments/s083_cohort_results.md` per ρ-band yield distribution.
- Verify: ≥ 10 seeds with ≥1 Band A∪B candidate. Report truth-in-candidate-list as secondary.
- Commit.

**Late morning**:
- Wind-down ritual via `/wind-down` skill: PROGRESS.md update, log.md entry, handoff prompt.

---

## 9. Hard rules and gotchas (numbered for citation)

Every rule below has caused real problems in this workspace. Skip at peril.

1. **No use of cached truth in the pipeline.** Load `q0_wxyz`/`omega0_rad`/`quaternions` ONLY at the end for error reporting. Do not use truth-derived |ω| as a centre for the ω-grid. Do not insert truth-q_a or truth-ω into any candidate set. (`feedback_validators_must_match_production_cost.md`)

2. **No oracle regime labelling.** LAM/SAM may be classified per-candidate from `(q_a, ω_a)` polhode invariants as a diagnostic; it must not be loaded from a truth-derived per-seed annotation. (User directive 2026-05-20.)

3. **Body-X twin canonicalisation at every stage** via `lib/twin.py::canonical_batch`. Apply at the q0 Sobol sample, the q-cloud generation, and the polished pool. Free 2× speedup. (`s043`)

4. **BLAS threads = 1 in Pool workers**, including the torch lines:
   ```python
   os.environ['OMP_NUM_THREADS'] = '1'
   os.environ['OPENBLAS_NUM_THREADS'] = '1'
   os.environ['MKL_NUM_THREADS'] = '1'
   torch.set_num_threads(1)
   torch.set_num_interop_threads(1)
   ```
   Missing the torch lines causes 8× slowdown that `s011` identified. (`feedback_blas_threads_for_pool.md`)

5. **`gc.collect()` after every hi-fi render inside Pool workers** to free trimesh cyclic garbage. Pool(24) with hi-fi rendering OOM-kills on the 30 GB box without this. Use Pool(16) for hi-fi stages. (`feedback_pool_size_with_trimesh_gc.md`, established in `s081`.)

6. **Full-LC residual in scoring AND polishing.** Never use local-window scoring (W=10) — `s059j` showed it amplifies ω-direction noise. Never use local-window polish — `project_local_window_polish_phantom_basins.md` showed it lands at phantom basins disagreeing with full-LC hi-fi by ρ=30-70.

7. **Polhode-basis 6-DOF LM polish** (`s064`) over raw ω-components. 21-27× in-basin convergence speedup is load-bearing for the wall budget.

8. **Multi-mag-start polish** with offsets `[0, +3%, -3%, +6%, -6%]` (`s059k`). Necessary to land Group-A |ω|-scaling-family multi-sols, which constitute most Band A∪B alternates on LAM seeds (`s081`).

9. **Jacobi Path 2 for all propagation** (`s072`). Constant-ω over a full LC is invalid for 119/120 cohort seeds (`s063c`). DOP853 is correct but ~108× slower than Path 2's ω closed-form portion.

10. **Cite the file for every number** in writeups, per CLAUDE.md (commit `d284569`). Quantitative claims need `(source: path/to/file[:line])` or explicit `[unverified]`. Memory recall is not a source.

11. **No commits with `Co-Authored-By` lines** (`feedback_no_coauthor.md`).

12. **Save intermediate results in NPZ checkpoints** after every expensive stage (`feedback_save_results.md`, `feedback_checkpoint_always.md`). For the pivot: save `candidates.npz` with all 320k rows. For the pipeline: save anchor selection, q-clouds, post-Path-2 surrogate scores, polished pool, and hi-fi results separately.

13. **No parallel CPU-heavy runs** (`feedback_no_parallel_cpu.md`). Don't stack Pool(24) jobs. The exception is the two-agent Wed afternoon parallel work (s082 pivot is mostly waiting on a single Pool run; s074 elliprj is small-process implementation work).

14. **Kill stuck batches at 2× expected wall** (`feedback_kill_stuck_early.md`). Investigate immediately. Use the time for analysis, never sleep-wait.

15. **Classify anomalies before alarming** per CLAUDE.md (`d284569`). (A) convention/sign/coordinate choice vs (B) implementation bug vs (C) unknown. State classification first, then evidence.

---

## 10. Risks and contingencies

### 10.1 Risk: pivot shows truth ranks outside top-1000

**Likelihood**: medium. The truth-ω tube is 1° and N_dir=2000 has 4.5° spacing — closest cell to truth is ~2-3° ω-direction.

**Contingency**: Run Branch v3 on Day 2. If v3 doesn't converge truth-near in budget either, escalate to elliprj-Path-2 + denser grid (N_dir = 10k+).

### 10.2 Risk: elliprj implementation hits a parameter-region issue

**Likelihood**: low-medium. The Π-via-R_J transformation is standard but has piecewise stability concerns at `m → 1` and `n → m`.

**Contingency**: Fall back to current Path 2. Day 2/3 pipeline still works with `N_dir ≤ 5000`.

### 10.3 Risk: SAM seed availability in holdout 100-119 is < 2

**Likelihood**: medium (m048 base rate is 16/120 SAM = 13%; in 20 holdouts, expected 2-3 SAM).

**Contingency**: Use all 4 known m048 SAM seeds (21, 28, 41, 91) + 1 holdout = 5 SAM. Document the deviation.

### 10.4 Risk: seed 28's tight basin (~2° q0, ~1° ω-dir) blocks recovery even at N_dir=10k

**Likelihood**: medium. `s005`/`s011` both flagged seed 28 as the worst-case basin. Multi-axis Sobol on SO(3) at N=64 lands 7/64 in-basin at truth-ω, but blind ω-direction grid at 4.5° spacing probably misses.

**Contingency**: Multi-solution acceptance (`s081` showed 0 distinct Band A∪B on SAM seeds 21, 28, 60, 84) — seed 28's truth basin recovery is hard but a Band-A multi-sol may exist on the LC fitting metric. The hybrid success metric (§ 1.4) is the right gate.

### 10.5 Risk: anchor-hit filter in v3 prunes truth out

**Likelihood**: low. The filter is `q-geo < 5° AND mag-fit < 0.1 mag`. Both tolerances are wider than the LM grab radius (q ~5-15°, mag-noise floor 0.05). But seed 28 might lose truth if all 3 anchors happen to be in the tight-basin regime.

**Contingency**: Make the v3 filter THRESHOLDS configurable. On Day 3 seed-by-seed iteration, relax the tolerance for failing seeds.

### 10.6 Risk: 15-min wall ceiling overshoots on some seeds

**Likelihood**: medium-high. Seed-by-seed variation in anchor sharpness, q-cloud size, and Path 2 step count is real.

**Contingency**: Per-stage wall accounting. If Stage 5 (scoring) exceeds 10 min on a seed, drop N_dir × N_mag by 2× for the next seed. If Stage 7 (polish) exceeds 5 min, reduce top-K. Document the per-seed parameter choices.

### 10.7 Risk: surrogate is the per-candidate bottleneck (not Path 2)

**Likelihood**: low-medium. I estimated surrogate full-LC at 3-5 ms/candidate but did not measure precisely.

**Contingency**: Day 1 morning, before launching s082, run a quick benchmark: 1000 candidates × Path 2 → surrogate → mse, single-thread + Pool(24). Measure per-call wall and the Path 2 vs surrogate split. If surrogate dominates, batched/GPU surrogate paths might be worth exploring (but I'd defer that to Day 2 if it doesn't bite us).

---

## 11. What to commit, when

The `forward_survey` branch convention is autocommit, no Co-Authored-By, hard size gate.

Per `/wind-down` skill defaults:
- Wed: Commit `s082` + `s074` (if landed) at end-of-day with PROGRESS.md updated.
- Thu: Commit pipeline scaffold + `s083_blind_pilot` at end-of-day.
- Fri: Commit cohort run + cohort writeup + final PROGRESS.md update.

Each commit must include:
- Experiment writeup (`.md` + frontmatter per the s055-s081 pattern: title, type, sources, related, created, updated, confidence).
- Result NPZs / JSONs under `results/sXXX/`.
- Plots if generated.
- `log.md` ingest line: `## [YYYY-MM-DD] ingest | sXXX | one-line summary`.
- PROGRESS.md update at the top.

Do NOT include in commits:
- Files > 100 MB (`project_push_blocked_large_files.md`).
- Any cached truth NPZs accidentally regenerated.

---

## 12. Appendix A — experiment cross-reference (what each numbered experiment established)

| ID | One-line | Source |
|---|---|---|
| s001 | Cost-at-truth cohort-scale: surrogate-MSE Band A on 100/100; alignment cost inapplicable on 24/100 | `experiments/s001_cost_at_truth_cohort.md` |
| s002 | Surrogate argmin = truth at fixed truth-ω on 8/8 PA-stratified seeds | `experiments/s002_surrogate_landscape_probe.md` |
| s003 | Truth-ω tube width: ~1° dir, ~2-5% mag; outside is incoherent. **Closes decoupled ω-outer / q0-inner.** | `experiments/s003_landscape_vs_omega.md` |
| s005 | Joint 6-DOF LM grab radius: q0 5-15°, ω-dir 1-2°, |ω| 3-5%. Bimodal convergence. | `experiments/s005_joint_local_descent.md` |
| s007, s008 | LC-only |ω| point-estimators dead (16% MAPE) | `experiments/s007_omega_mag_peak_spacing_pilot.md`, `experiments/s008_lc_feature_regression_omega.md` |
| s011 | Sobol-Shoemake(SO(3)) N=64 + LM @ truth-ω: 9/10 PA-stratified pilot in-basin | `experiments/s011_q4cii_sobol_so3_polish_pilot.md` |
| s014 | Surrogate ↔ hi-fi Spearman 0.9952 cohort-wide | `experiments/s014_cohort_rho_band.md` |
| s019 | **LS-bracket gives 98/100 within 5% of truth-|ω|, median 1.25%.** | `experiments/s019_ls_bracket_omega_mag.md` |
| s043 | Body-X twin LC equivalence bit-exact (max |Δ|=2.29e-8 mag) | `experiments/s043_twin_hifi_verify.md` |
| s054 | Holdout 100-119 generated; available via `lib.traj_load` | `experiments/s054_generate_holdout.md` |
| s055a-c | LC-only ω-direction priors closed dead-end | `experiments/s055{a,b,c}_*.md` |
| s057g | Forward-prop discrimination 4400× over null at constant-ω; truth rank 55/548 | `experiments/s057g_forward_propagation.md` |
| s059j | **Local-window scoring amplifies ω-direction noise.** ω-grid at 7° spacing fails. | `experiments/s059j_cloud_data_omega_grid.md` |
| s059k | Multi-mag-start `[0, ±3, ±6]%` + full-LC polish. Architectural baseline post-fix. | `experiments/s059k_densify_ndirs.md` |
| s063c | Cohort polhode census: 119/120 seeds non-near-separatrix; median T_pol 1594s | `experiments/s063c_polhode_census.md` |
| s064 | **Polhode-basis LM polish: 21-27× in-basin speedup.** Drop-in for `s058::lm_polish`. | `experiments/s064_jacobi_polish.md` |
| s067 | Propagator sign fix validated; L_J2000 conservation 1e-12 cohort-wide | `experiments/s067_postfix_propagator_validation.md` |
| s068 | Post-fix replication of s011: **10/10 yield (improved from 9/10)** in 38 min Pool(8) | `experiments/s068_replicate_s011.md` |
| s072 | **Jacobi Path 2 closed-form q(t)**: 16.4 ms / 500-epoch, 1e-9 vs DOP853 | `experiments/s072_path2_closed_form_q.md` |
| s073 | **L_J2000 cross-anchor gate**: truth vs Band-A multi-sol differ 128.6° in L direction | `experiments/s073_cluster457_l_vector_check.md` |
| s073b | Path 2 cProfile: 81% of wall = φ-ODE; elliprj is the right next axis | `experiments/s073b_path2_cprofile.md` |
| s074 (planned) | **elliprj replaces φ-ODE in Path 2**, projected ~5× speedup | n/a (Day 1 deliverable) |
| s076 | Aspect-graph horizon-density anchor score fails on IS-901 cuboid bus | `experiments/s076_horizon_density.md` |
| s077 | Cohort L-vector basin sweep: |L| pinned 0.7%, L-direction free; surrogate-gate caveat retired by s081 | `experiments/s077_l_vector_basin_sweep.md` |
| s078 | RF25 NLL residual A/B negative on seed 89; stays opt-in | `experiments/s078_nll_residual_ab.md` |
| s079 | LAM/SAM split: 28/30 Band A∪B distinct competing basins on LAM | `experiments/s079_regime_stratified_l_basins.md` |
| s081 | **Hi-fi validates surrogate↔hi-fi 145/145 band agreement**; Group A multi-sols are |ω|-scaling family | `experiments/s081_hifi_rho_bands_twin.md` |
| s082 (planned) | **Joint-grid pivot experiment** — picks Branch v2 vs v3 | this document, Day 1 |
| s083 (planned) | **Blind pilot pipeline + cohort validation** | this document, Days 2-3 |

---

## 13. Appendix B — file inventory the agent will need

### 13.1 Library modules

- `lib/traj_load.py::load_truth(seed)` → returns `(q0_wxyz, omega0_rad, mag_hifi, k1_body, k2_body, ...)`. **Only call this for stratification metadata and end-of-pipeline truth reporting. Never inside the search loop.**
- `lib/forward.py::propagate_to_body_frame(q0, ω0, times, sun_pos, obs_pos, sat_pos, inertia)` → `(k1_body, k2_body, quaternions)`. Uses DOP853 (`src.dynamics.attitude_propagator`). Reference path; in production use `propagate_jacobi_path2`.
- `lib/jacobi_propagator.py::propagate_jacobi_path2(q0, ω0, I, times)` → `(q_hist, ω_hist)`. The fast path. 16 ms / 500 epochs.
- `lib/c_t_pipeline.py` → `sample_so3_pool`, `compute_j2000_units`, `project_directions`, `survive_at_epoch`, `nearest_in_pool_to_truth(s)`.
- `lib/surrogate_eval.py` → `predict(k1, k2, obs_dist, sp_deg, ad_deg)`, `full_lc_mse`, `bright_mse`, `rho`. (Also `nll_residual` / `nll_cost` — DO NOT use for production cost path.)
- `lib/twin.py::canonical_batch(q_array)` → maps each q to the canonical hemisphere under body-X twin.
- `lib/hifi_render.py::build_context(seed)` → returns dict with `observation_times`, `sun_pos`, `obs_pos`, `sat_pos`, `inertia_tensor`, `mag_hifi` (= mag_observed). **No q0 / ω0 returned.** This is the right entry point for blind-inversion inputs.
- `lib/hifi_render.py::render_hifi(q0, ω0, ctx)` → hi-fi LC. Use only at Stage 8.
- `lib/lc_features.py` → LC feature extractors (peak times, etc.). May need to add `ls_bracket(times, mag)` on Day 1, porting from `experiments/s019_ls_bracket_omega_mag.py`.
- `lib/lc_compare.py` → comparison plotters. Use for diagnostic figures.

### 13.2 Helpers to add (Day 1-2)

- `lib/lc_features.py::ls_bracket(times, mag) → omega_mag_grid (np.ndarray)`. Port from `experiments/s019_ls_bracket_omega_mag.py:64-95`.
- `lib/inversion_v0.py` (new): houses the staged pipeline. Stages 1-8 as functions.
- (Optional) `lib/fibonacci_sphere.py::fibonacci_sphere(N) → unit_vectors (N, 3)`. Or inline into `lib/inversion_v0.py`.

### 13.3 Configs

- IS-901 STL + articulation config: `data/models/intelsat_901/intelsat_901_config.yaml`. Articulation defaults SP=0°, AD=15° per `concepts/surrogate_model.md`.
- Trajectory NPZs: `data/trajectories/traj_seed{seed:03d}.npz` (symlinks to canonical post-fix per-trajectory cache).
- Inertia tensor: `data/results/inversion_diagnostics/m048_trajectories/m048_trajectories.npz['inertia_tensor']` (m048-diagonal `diag(37985.16, 38305.71, 7749.01) kg·m²`).

### 13.4 Existing scripts to read but NOT import

- `experiments/s019_ls_bracket_omega_mag.py` — port `ls_bracket` from here.
- `experiments/s059k_full_lc_from_seeds.py` — reference for full-LC LM polish with multi-mag-start.
- `experiments/s064_jacobi_polish.py` — reference for polhode-basis LM polish (the `gram_schmidt_basis` function + the LM closure).
- `experiments/s068_replicate_s011.py` — reference for joint-LM-at-truth-ω baseline (DO NOT use truth-ω in production).
- `experiments/s072_path2_closed_form_q.py` — reference for Path 2 usage.
- `experiments/s073_cluster457_l_vector_check.py` — reference for L_J2000 computation.
- `experiments/s081_hifi_rho_bands_twin.py` — reference for hi-fi gate + Pool(16) with `gc.collect()`.

---

## 14. Appendix C — example numerical pipeline call sketches

### 14.1 Stratification (s082 morning)

```python
import json
import numpy as np
from lib import traj_load
from lib.jacobi_propagator import _eigendecompose_inertia

records = []
for seed in range(100, 120):
    truth = traj_load.load_truth(seed)
    I = truth['inertia_tensor']  # 3x3
    I_pa, R_pa = _eigendecompose_inertia(I)
    om = truth['omega0_rad']
    om_pa = R_pa.T @ om
    two_T = float(np.sum(I_pa * om_pa**2))
    L_pa = I_pa * om_pa
    L_sq = float(np.sum(L_pa**2))
    # Case A (LAM) if 2T·I_2 > L²; Case B (SAM) otherwise. I_2 = middle eigenvalue.
    disc = two_T * I_pa[1] - L_sq
    regime = 'LAM' if disc > 0 else 'SAM'
    om_mag_dps = float(np.degrees(np.linalg.norm(om)))
    # Mean PA across the LC
    k1 = truth['k1_body']  # (N, 3)
    k2 = truth['k2_body']
    pa = np.degrees(np.arccos(np.clip(np.sum(k1 * k2, axis=1), -1, 1)))
    mean_pa = float(np.mean(pa))
    records.append({'seed': seed, 'regime': regime, 'omega_mag_dps': om_mag_dps, 'mean_pa_deg': mean_pa})

with open('results/s082/holdout_stratification.json', 'w') as f:
    json.dump(records, f, indent=2)

# Pick 3 seeds: 1 LAM-slow, 1 LAM-fast, 1 SAM (any |ω|)
# Save the selection at the top of the file.
```

### 14.2 LS-bracket (s019 port)

```python
# Per-seed |ω| LS bracket. Pure spectral. Does NOT use truth.
from astropy.timeseries import LombScargle

def ls_bracket(times, mag):
    """5-cell |ω| grid covering ±5% of truth on 98/100 cohort seeds (s019)."""
    t = np.asarray(times, dtype=np.float64)
    m = np.asarray(mag, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(m)
    t, m = t[mask], m[mask]
    m -= np.mean(m)
    # Frequency grid: [1/window, 0.5/dt]
    window = t[-1] - t[0]
    dt = np.median(np.diff(t))
    freqs = np.linspace(1 / window, 0.5 / dt, 4000)
    pgram = LombScargle(t, m).power(freqs)
    # Significant peaks: power ≥ 0.1 × max
    peak_thresh = 0.1 * np.max(pgram)
    peak_mask = pgram >= peak_thresh
    peak_freqs = freqs[peak_mask]
    omega_peaks = 2 * np.pi * peak_freqs  # rad/s
    omega_grid = np.geomspace(0.5 * omega_peaks.min(), 2.0 * omega_peaks.max(), num=5)
    return omega_grid
```

### 14.3 Joint candidate scoring (Pool(24)) — sketch

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from multiprocessing import Pool
import numpy as np
from lib.jacobi_propagator import propagate_jacobi_path2
from lib.surrogate_eval import get_model, predict

def init_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _ = get_model()  # warm surrogate

def score_one(args):
    q0, om, ctx_pickled = args
    ctx = ctx_pickled  # passed once via initargs ideally; simplified here
    q_hist, om_hist = propagate_jacobi_path2(q0, om, ctx['I'], ctx['times'])
    # Build k1, k2 from q_hist + sun/obs/sat
    k1, k2 = build_body_vectors(q_hist, ctx)
    mag_pred = predict(k1, k2, ctx['obs_dist'], sp_angle_deg=0.0, ad_angle_deg=15.0)
    mse = float(np.mean((mag_pred - ctx['mag_observed'])**2))
    return mse

# Driver
candidates = [(q0, om, ctx_for_seed) for q0 in q0_pool for om in om_pool]
with Pool(24, initializer=init_worker) as p:
    mses = p.map(score_one, candidates, chunksize=200)
```

(In production this would be vectorised better — pass ctx once via initargs, batch surrogate calls, etc. The sketch above is illustrative.)

---

## 15. What "done" looks like by Friday noon

Cohort `experiments/s083_cohort_results.md` containing:
- 10 seeds × `(regime, |ω|, mean_PA, wall_total, n_Band_A_candidates, n_Band_B_candidates, n_distinct, truth_in_top_3, q0_err_of_best_Band_A_to_truth, …)`.
- ≥ 10/10 seeds with ≥ 1 Band A∪B candidate (primary success).
- Truth-in-candidate-list rate reported (secondary metric).
- Per-seed walls; ≥ 9/10 ≤ 15 min wall.
- A 4-panel plot: ρ distribution, q0_err CDF, ω-error decomposition, per-seed wall histogram.

A 2-paragraph PROGRESS.md banner explaining: which branch we took, why, what the cohort-scale yield is, and what's queued for the next session.

---

## 16. Cross-references summary

- This document: `notebooks/inversion/survey/report/blind_inversion_15min_plan_2026-05-20.md`
- Prior wind-down + analytical summary: `notebooks/inversion/survey/report/aspect_graph_synthesis.{tex,pdf}` (not load-bearing here; aspect-graph framing is body-dependent per s076)
- PROGRESS.md current banner: post-s081 (commit `12b8d6e`)
- Workspace contract: `README.md`, `CLAUDE.md`
- Memory autoload: `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/MEMORY.md`
- Project rules: `CLAUDE.md` (root) — cite-file-for-numbers + classify-anomalies-before-alarming + analytical-vs-operational

---

**End of plan. Hand off to the next agent.**
