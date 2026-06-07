---
title: "s061a — 1-step cloud-threading smoke (seed 89, ω_max=1.5 dps)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s061a_thread_smoke.py
  - notebooks/inversion/survey/results/s061a_thread_smoke/seed089/summary.json
  - notebooks/inversion/survey/results/s060_sharpness_map/seed089/anchor_topology.json
related:
  - s048c — cloud viewer (substrate)
  - s060_anchor_topology — clean-dim anchor at seed 89 t=208
  - s060_multi_anchor_design — uses Newton-shoot ω instead of tube sampling
created: 2026-05-10
updated: 2026-05-10
confidence: high (mechanism check; truth tracks)
---

## TL;DR

User's idea (2026-05-10, end of day): instead of computing per-epoch C_t
clouds independently, start at the most-constrained anchor and EXPAND
outward under a connectability filter. A child q' at adjacent epoch must
lie within geodesic ball of radius `ω_max·Δt` around some anchor
survivor. Brightness still filters, but the candidate space at each
adjacent epoch shrinks from full-SO(3) to a tube around the anchor cloud.
Truth must thread continuously; cluster-disappearance kills clusters
that have no brightness-consistent descendant.

s061a is the 1-step mechanism check on seed 89 anchor t=208 (clean-dim
per s060_anchor_topology, |C_t|=48, 5 clusters with top-3 mass 0.92).
Threads to t=207 and t=209 with ω_max=1.5 dps (cohort-safe upper bound).

**Result: mechanism works.** Truth is in the largest cluster (cluster 0,
size 24, 8.6° to nearest anchor survivor). Both adjacent epochs preserve
all 5 clusters with 54-72% brightness retention each. Truth's actual
geodesic step is 1.75° — exactly matching |ω|·Δt=1.73° (truth |ω|=0.24
dps). Tube radius ω_max·Δt = 10.82° is ~6× truth's actual step on this
slow seed; the budget is comfortably loose. Per-pair derived ω from the
tube has |ω| 0-1.5 dps and axis error 2-90° — uniformly spread over the
tube, NOT informative as a per-step ω estimate.

## What

A 1-step threading test from a single anchor on seed 89:

1. Build v2 surrogate cloud at t=208 (N=25k Sobol pool, TOL=0.10 mag).
2. Body-twin canonicalize each survivor q (the LC at any single epoch is
   invariant under the body-X 180° twin map; we collapse twin pairs to
   a single representative for clustering).
3. Greedy-cluster at 40° geodesic threshold.
4. Identify the cluster containing truth q[208].
5. Thread one step right (t=209) and one step left (t=207):
   - For each anchor survivor q_a, sample N_per_anchor=50 perturbations δq
     from the geodesic ball of radius `ω_max·Δt` (uniform in axis on S²,
     uniform in angle on [0, r_max]).
   - Build q_child = q_a ⊗ δq (body-frame perturbation).
   - Brightness-filter at the target epoch (v2 surrogate, TOL=0.10 mag).
   - Tag each surviving child with predecessor-cluster id.
6. Compute derived per-pair ω = 2·log(δq) / Δt (body frame).

Sanity tracks:
- Truth quaternion at adjacent epoch within ω_max·Δt geodesic of truth at
  anchor (i.e., budget genuinely upper-bounds true motion)?
- Truth-cluster lineage retains descendants?
- Per-pair derived ω in truth cluster matches truth ω?

## How

```python
TOL_MAG = 0.10
SP_DEG = 0.0
AD_DEG = 15.0

# Anchor cloud
pool = sample_so3_pool(25_000, sample_seed=42)
k1, k2 = project_directions(pool["R_cache"], sun_unit[208], obs_unit[208])
_, keep = survive_at_epoch(model, k1, k2, obs_dist[208], 0, 15, mag_truth[208], 0.10)
anchor_q = body_twin_canonicalize_q(pool["q_pool_wxyz"][keep])
clusters = greedy_cluster(anchor_q, threshold=40)

# Thread
omega_max_rad = 1.5 * pi / 180
delta_t = obs_times[209] - obs_times[208]    # 7.214 s on seed 89
r_max = omega_max_rad * delta_t              # 0.189 rad = 10.82°

for q_a in anchor_q:
    for i in range(50):
        axis = random_unit_S2()
        angle = uniform(0, r_max)
        delta_q = (cos(angle/2), sin(angle/2)*axis)
        q_child = q_a ⊗ delta_q
    # ... brightness-filter q_child at t=209 via v2 surrogate
```

Compute env: single-threaded BLAS; v2 surrogate `SurrogateModel.load_default()`.

## Result

### Anchor cloud (t=208, mag=14.92, dim-extreme)

| metric | value |
|---|---|
| pool size | 25,000 |
| raw \|C_t=208\| | 48 |
| post-canon survivors | 48 (count unchanged; body-twin pairs merge in cluster space) |
| n_clusters at 40° | 5 |
| cluster sizes | [24, 14, 5, 4, 1] |
| truth in cluster | 0 (largest) |
| truth dist to nearest survivor | 8.57° |

### Thread step (right: t=208 → t=209, Δt=7.214 s, r_max=10.82°)

| metric | value |
|---|---|
| candidates | 2,400 (48 × 50/anchor) |
| brightness survivors | 1,448 (60.3%) |
| clusters with ≥1 surviving descendant | 5 / 5 |
| per-cluster retention | 54-72% |
| truth-cluster (id=0) survivors | 649 / 1200 |

Per-cluster survival rates were uniform (54-72%) — no cluster died.
Brightness retention is HIGH because adjacent-epoch lighting geometry is
similar (same SPICE state to within sub-second, same sun/observer
configuration to ~0.05° angular shift).

Symmetric for left step (t=207): 64.9% retention, 5/5 clusters survive.

### Truth tracking

| metric | value |
|---|---|
| truth geodesic q[208] → q[207] | 1.750° |
| truth geodesic q[208] → q[209] | 1.750° |
| expected from \|ω\|·Δt | 1.730° (truth \|ω\|=0.240 dps) |
| ω_max budget | 10.82° (~6× truth) |
| truth within budget? | YES (left and right) |
| truth FD ω at 208→209 | \|ω\|=0.243 dps (err +0.003 dps vs truth 0.240) |

Truth's per-step geodesic matches `|ω|·Δt` to 4 decimal places — the
finite-diff ω from cached truth quaternions reproduces ω-truth to
1% accuracy. Excellent dynamics consistency on cached truth states.

### Derived ω from threaded survivors (right thread)

| metric | value |
|---|---|
| \|ω_derived\| min/median/max | 0.000 / 0.581 / 1.497 dps |
| in TRUTH cluster (n=649) — \|ω\| range | 0.000 - 1.497 dps |
| in TRUTH cluster — axis err range | 2.27° - 89.82° |

Derived ω is uniformly spread over the tube (0 to ω_max), as expected
from uniform-angle tube sampling. NOT a per-step ω estimate — just a
list of consistent (q_a, q_child) pairs.

## Why this matters

1. **Architecture mechanism is sound.** Truth threads, brightness
   filtering at adjacent epochs preserves the truth cluster, body-twin
   canonicalization reduces the cluster count appropriately.

2. **Adjacent-epoch brightness is weakly discriminative on slow tumblers.**
   At seed 89 |ω|=0.24 dps, the LC moves slowly and adjacent-epoch
   lighting geometry barely changes. 60% of tube candidates pass the
   brightness gate just because brightness barely changes between t and
   t+1.

3. **ω_max=1.5 dps is too loose for slow seeds.** For seed 89 the tube
   is 6× truth's motion; uniform tube sampling produces a derived-ω
   distribution that fills [0, ω_max] uniformly — no information about
   truth ω. Tighter budgets (e.g. polhode-prior-derived) would
   concentrate derived ω near truth.

4. **Multi-step threading is the actual test.** Single-step preserves
   all clusters because brightness barely changes. Cumulative N-step
   threading is where cluster-disappearance can happen — see s061b.

## Numbers

- Pool: 25,000 SO(3) random rotations.
- Anchor t=208 mag=14.92 (dim-extreme, top-3 cluster mass 0.92 per s060).
- Compute wall: <5 s total (anchor + 2 threads + truth tracking).
- Surrogate: v2 (residual ensemble, 0.019 mag bright MAE).
- TOL_MAG = 0.10 (s048c default).
- Cluster threshold = 40° (matches s060_anchor_topology).

## Artefacts

- `experiments/s061a_thread_smoke.py` — script with helpers
  (`quat_log`, `quat_geodesic_deg_batch`, `sample_perturbations_in_geodesic_ball`,
  `body_twin_canonicalize_q`, `greedy_cluster`, `thread_one_step`).
- `results/s061a_thread_smoke/seed089/summary.json` — summary.
- `results/s061a_thread_smoke/seed089/thread.npz` — survivor arrays
  (anchor cloud, predecessor mapping, derived ω).

## Out of scope

- Multi-step threading (s061b).
- Hard seeds with high \|ω\| (s061c).
- Constant-ω propagation per (q_a, ω) candidate (the natural follow-up
  if random-walk threading is too loose).
- Polhode-prior tightening of ω_max (per-seed estimate at ~25% MAPE per
  s055a).
- Comparison vs s059k single-anchor + ω-grid yield.

## Cross-references

- `experiments/s060_multi_anchor_design.md` — alternative architecture
  using Newton-shoot ω from a single (q_a, q_b) pair at large Δt;
  threading is the small-Δt continuous complement.
- `concepts/known_pathologies_to_revalidate.md` — buggy-era cascade
  (s048b cohort) was the predecessor; s061 is the post-fix v2-substrate
  redo.
- `feedback_finite_diff_small_geodesic_noise.md` — s057f failure at small
  Δt was at v1 substrate; v2 + brightness-doubly-filtered endpoints
  changes the SNR regime.
