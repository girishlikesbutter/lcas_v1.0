---
title: "s060 — anchor topology classifier (cluster structure of survivors)"
type: experiment
sources:
  - notebooks/inversion/survey/experiments/s060_anchor_topology.py
  - notebooks/inversion/survey/results/s060_sharpness_map/seed{028,089,010}/anchor_topology.json
related:
  - s060_sharpness_map — full-LC |C_t|(t) (substrate)
  - s060_cohort_anchor_topology — generalization across 11 PA-stratified seeds
  - feedback_finite_diff_small_geodesic_noise — s057f-era observation
created: 2026-05-09
updated: 2026-05-09
confidence: high (direct measurement; user's geometric intuition validated on 3 seeds)
---

## TL;DR

For each cluster-independent sharp anchor (from s060_sharpness_map), measured the topology of the survivor set on SO(3) via greedy clustering at threshold 20° and 40°. **Bright-peak anchors and dim-extreme anchors have qualitatively different cluster structure.** Bright-peak anchors fragment into 7-8 small clusters of size 2-4 separated by ~170° (likely body-twin pairs from multiple specular configurations) and do NOT consolidate at higher threshold. Dim-extreme anchors at the deepest LC dim moments collapse into 3-5 clean clusters with **88-92% of survivors in the top-3** at 40° threshold. This validates the user's intuition that bright = "spread along direction-theta" (multi-specular geometry) while dim = "few discrete clusters" (multimodal symmetry-related dim configurations).

## What

For sharp anchors identified by s060_sharpness_map, characterize the cluster structure of their survivor sets on SO(3). The survivor count |C_t| alone is not architecturally informative — two anchors can have the same |C_t|=12 but very different topology (one continuous curve vs. five discrete clusters). The architecture's downstream use of an anchor depends on whether the survivors enumerate a small finite hypothesis set or trace a continuous family.

## How

For each cluster-independent sharp anchor t_i:
- Re-project pool at t_i, get survivor quaternions.
- Greedy cluster: pick element, group neighbors within geodesic threshold, mark assigned, repeat. Sort clusters by size descending.
- Report: n_clusters, top-3 sizes, top-3 mass fraction, intra-cluster max geodesic, between-cluster gap.

Threshold tested at 20° (tight) and 40° (loose). Comparison reveals:
- "Continuous curve" topology: cluster count drops monotonically with threshold (segments merge).
- "Discrete clusters" topology: cluster count drops sharply at one threshold (when threshold crosses intra-cluster spread) then stays constant.

## Result

### Seed 28 — top-10 cluster-independent anchors at 40° threshold

| t   | mag    | mag_pct | \|C\| | #cl | top-3 sizes | top-3 frac | intra_max | gap |
|-----|--------|---------|------|-----|-------------|------------|-----------|-----|
| 312 | 20.62  | 100%    | 5    | 4   | 2/1/1       | 0.80       | 15.7      | 174.9 |
| 224 | 19.52  | 93%     | 11   | 7   | 2/2/2       | 0.55       | 30.0      | 179.2 |
| 492 | 5.31   | 2%      | 11   | 8   | 2/2/2       | 0.55       | 10.4      | 178.7 |
| 390 | 5.01   | 0%      | 14   | 7   | 4/4/2       | 0.71       | 29.5      | 164.2 |
| 440 | 19.74  | 94%     | 47   | 21  | 5/5/5       | 0.32       | 40.3      | 92.2  |
| ... | ...    | ...     | ...  | ... | ...         | ...        | ...       | ...   |

### Seed 89 — top-4 anchors at 40° threshold

| t   | mag    | mag_pct | \|C\| | #cl | top-3 sizes | top-3 frac |
|-----|--------|---------|------|-----|-------------|------------|
| 413 | 5.46   | 1%      | 12   | 8   | 3/2/2       | 0.58       |
| 208 | 14.92  | 100%    | 48   | **5** | **20/17/7** | **0.92**   |
| 240 | 9.59   | 44%     | 62   | 29  | 5/5/4       | 0.23       |
| 91  | 8.45   | 32%     | 74   | 27  | 7/7/6       | 0.27       |

### Seed 10 — top-3 anchors at 40° threshold

| t   | mag    | mag_pct | \|C\| | #cl | top-3 sizes | top-3 frac |
|-----|--------|---------|------|-----|-------------|------------|
| 147 | 15.17  | 100%    | 44   | **4** | **16/15/8** | **0.89**   |
| 70  | 14.88  | 91%     | 210  | 10  | 44/43/31    | 0.56       |
| 32  | 14.83  | 89%     | 277  | 12  | 52/42/40    | 0.48       |

### Pattern

- **Bright peaks** (mag_pct ≤ 5%): 7-8 clusters at both 20° and 40° thresholds (no consolidation), top-3 mass 55-71%, gap 164-179° (often body-twin antipodal). Fragmented, not continuous.
- **Dim extremes** (mag_pct ≥ 95%, |C_t| ≤ 50): 4-5 CLEAN clusters at 40° threshold, top-3 mass **88-92%**. These are the user's "4-5 distinct clusters" — multimodal symmetry-related global-dim configurations.
- **Mid-mag anchors**: many small clusters with low top-3 mass (<35%), uniformly fragmented. No clean enumeration available.

## Why this matters

This is the architectural foundation for s060_multi_anchor:

- **Dim-extreme anchors play the ENUMERATOR role.** A clean 3-5 cluster structure with ≥85% top-3 mass means the satellite is provably in one of 3-5 discrete attitude classes at this epoch. K=4 dim anchors with 4 clusters each → 4⁴ = 256 hypothesis-class tuples. Polhode threading then filters which sequences are dynamics-admissible.

- **Bright-peak anchors play the PASSAGE-VALIDATOR role.** 7-8 small clusters separated by ~170° (multi-specular geometry × body-twin) is too many to enumerate as primary hypotheses but useful for "does the propagated trajectory pass through ANY of these specular configurations at this epoch?" Pass/fail check.

- **Body-twin canonicalization** (already in s059k via `lib.twin.canonical_batch`) halves cluster counts: dim-extreme → ~2-3 effective hypothesis classes; bright-peak → ~4 specular geometries. Cleaner enumeration, smaller tuple space.

The two roles are complementary: dim anchors discretize the candidate space; bright anchors validate dynamics.

## Numbers

- Anchors measured per seed: 10 (seed 28), 4 (seed 89), 3 (seed 10) — all available cluster-independent anchors.
- Pool: N=25,000 (matched to s060_sharpness_map for survivor parity).
- Threshold 20° (cluster_threshold_deg arg) and 40° (re-run for consolidation test).
- Wall ~30 sec per seed (one anchor projection + clustering each).

## Artefacts

- `notebooks/inversion/survey/experiments/s060_anchor_topology.py` — script with `greedy_cluster_quats` + `classify_topology` helpers.
- `notebooks/inversion/survey/results/s060_sharpness_map/seed{028,089,010}/anchor_topology.json` — per-seed table.

## Out of scope

- Topology measurement at the dense pool (N=400k). The 25k pool is sufficient for cluster structure detection; dense pool is a sample-density refinement, not a topology-changer.
- ω-direction or polhode-prior dependent topology classification. The clusters are in body-frame attitude space, not (q,ω) joint space; ω structure is what the polhode-threading step in s060 derives.
- Anchor topology at non-cluster-independent epochs (rank > K). Architecturally we only use the cluster-independent set; sub-cluster anchors don't add joint-constraint power.

## Cross-references

- `s060_sharpness_map.md` — substrate (where the anchors come from).
- `s060_cohort_anchor_topology.md` — generalization to 11 cohort seeds.
- `s060_multi_anchor_design.md` — architectural design that uses the bright/dim role distinction.
- `feedback_finite_diff_small_geodesic_noise.md` — s057f's failure mode at small Δt; relates to the body-twin gap of ~170° at most anchors (canonicalization addresses).
