# m106 -- Vectorized Pairwise Peak Alignment at Scale

## Hypothesis

Vectorized pairwise peak alignment will achieve ~100x+ reduction on 10K omega candidates in under 10 min per seed, and will work on all 5 grid-failure seeds (1, 11, 28, 44, 46).

## Background

m105 POC validated a two-stage approach on seeds 28 and 44 with 2500 candidates (40K grid, but the code was only ever tested inline, never via the script at scale):
- **Stage 1 -- 3-pair intersection:** Select top 3 bright cross-family peak pairs, check if each candidate omega0 produces alignment at BOTH peaks simultaneously (twist psi sweep about lobe normal). Intersect survivors across all 3 pairs. Result: ~18-20 survivors from 2500 (100-140x reduction).
- **Stage 2 -- Full-observation scoring:** For each survivor, count how many of ALL bright peaks it can align at. Truth scored 11/14 and 11/11; best false positive only 6. This is the real discriminator.

m105 script limitations addressed by m106:
1. Only implements Stage 1 (3-pair intersection), NOT Stage 2
2. Has a slow per-candidate Python loop in `check_alignment_batch`
3. Only propagates at pair-specific epochs, not all bright peak epochs

## Method

### Grid
- 2000 Fibonacci sphere directions x 5 magnitude bins = 10,000 candidates
- Magnitude range: true |omega| +/- 30%

### Stage 0 -- Propagation
- Propagate all 10K candidates from identity quaternion at ALL bright peak epochs (not just 6 pair epochs)
- Pool(24), chunksize=50
- Checkpoint: `stage0_delta_qs.npz` with (n_cand, n_times, 4) wxyz quaternions

### Stage 1 -- 3-pair fast filter (VECTORIZED)
- Select top 3 bright cross-family peak pairs (same logic as m105)
- For each pair, check alignment for ALL candidates simultaneously using pure numpy:
  - Batch quaternion-vector rotation: `v' = v + 2w(q_xyz x v) + 2(q_xyz x (q_xyz x v))`
  - Batch Rodrigues alignment rotation
  - Batch psi twist sweep via Rodrigues with broadcasting: `(n_cand, n_psi, 3)` tensor
  - Batch angular distance with sign ambiguity
- Intersect survivors across 3 pairs
- Checkpoint: `stage1_results.npz`

### Stage 2 -- Full-observation scoring
- For each survivor from Stage 1:
  - Use the brightest peak as reference
  - Sweep 72 psi values at the reference peak
  - For each psi, count how many of ALL other bright peaks align (angular distance to assigned lobe < 10 deg)
  - Score = max count across psi values
- **Oracle mode:** Known lobe assignments per peak
- **Blind mode:** Try all lobes as reference assignment; for each other peak, assign to whichever lobe gives best angular distance
- Checkpoint: `stage2_results.npz`

### Seeds
[1, 11, 28, 44, 46] -- the 5 grid-failure seeds from m103

## Vectorization detail

The core bottleneck in m105 was `check_alignment_batch`, which looped over candidates in Python calling scipy `Rotation` objects. m106 replaces this entirely with numpy broadcasting:

1. **Quaternion rotation of a vector** (all candidates at once):
   ```
   t = 2 * cross(q_xyz, v)         # (n_cand, 3)
   v' = v + w*t + cross(q_xyz, t)  # (n_cand, 3)
   ```

2. **Rodrigues alignment** (v1 -> n1, applied to v2, all candidates at once):
   ```
   axis = cross(v1, n1) / |cross(v1, n1)|
   v2_aligned = v2*cos(a) + (axis x v2)*sin(a) + axis*(axis.v2)*(1-cos(a))
   ```
   With edge-case handling for parallel and anti-parallel v1/n1.

3. **Psi twist sweep** (n_cand x n_psi broadcast):
   ```
   v_twisted[i,j] = v2_aligned[i]*cos(psi[j]) + (n1 x v2_aligned[i])*sin(psi[j])
                   + n1*(n1.v2_aligned[i])*(1-cos(psi[j]))
   ```
   Shape: (n_cand, n_psi, 3)

4. **Angular distance**: `arccos(|dot(v_twisted, n2)|)` with (n_cand, n_psi) output.

Zero Python loops over candidates. The only loops are over peak pairs (3) and lobe combos (~72 for blind mode).

## Expected outcomes

### Stage 1
- Oracle single-pair reduction: 5-20x per pair (consistent with m105 POC)
- Oracle 3-pair intersection: 50-200x reduction (100-200 survivors from 10K)
- Blind: 2-5x more survivors than oracle
- Truth should survive all pairs for all 5 seeds

### Stage 2
- Truth oracle score: ~10-14 out of n_bright peaks
- Best false positive oracle score: ~5-7 (clear separation)
- Truth rank: top 1-3 among oracle survivors

### Timing
- Stage 0 (propagation): ~3-5 min per seed (10K candidates, 10-20 epochs)
- Stage 1 (vectorized): ~10-30s per seed (pure numpy)
- Stage 2 (oracle): ~5-30s per seed (only ~100-300 survivors)
- Stage 2 (blind): ~1-5 min per seed (10 lobe choices x n_peaks peaks x n_lobes lobes)
- Total: well under 10 min per seed

### Key success criteria
1. Truth survives Stage 1 intersection for all 5 seeds
2. Stage 1 reduces 10K to O(100-300) -- sufficient for NM refinement
3. Stage 2 oracle: truth scores significantly higher than best false positive
4. Total wall time < 10 min per seed

## Output structure

```
data/results/inversion_diagnostics/m106_pairwise_vec/
  seed_001/
    stage0_delta_qs.npz    # propagated quaternions checkpoint
    stage1_results.npz     # pair survivor masks checkpoint
    stage2_results.npz     # full-observation scores checkpoint
    result.json            # complete results
    pipeline.log           # stdout log
  seed_011/
    ...
  seed_028/
    ...
  seed_044/
    ...
  seed_046/
    ...
  results_combined.json    # all seeds combined
```

## Script

`notebooks/inversion/12_brightness_surface/m106_pairwise_vec_ipl.py`

Run: `python3 notebooks/inversion/12_brightness_surface/m106_pairwise_vec_ipl.py`

## What to look for in results

1. **Truth survival through Stage 1:** If truth fails to survive a pair, check the threshold (10 deg may be too tight for that seed's grid resolution).
2. **Reduction factor:** Target 50-200x from Stage 1 intersection. If only 10x, the pairwise constraint is too weak for that seed.
3. **Stage 2 separation:** The gap between truth's score and the best false positive's score. A gap of 3+ peaks is strong discrimination. A gap of 1-2 is marginal.
4. **Blind vs oracle overhead:** How much does blind mode cost in time and survivor count? If blind is within 3x of oracle on both metrics, it's viable for the real pipeline.
5. **Timing:** If any seed exceeds 10 min total, investigate whether propagation or scoring dominates.
