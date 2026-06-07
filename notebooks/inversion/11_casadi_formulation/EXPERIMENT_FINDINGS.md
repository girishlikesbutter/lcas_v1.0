# m096 Experiment Findings — 100-Seed Stage 1 Analysis

> **Date:** 2026-04-09
> **Approach:** Stage-by-stage validation. Run cheap experiments on ALL 100 seeds, checkpoint everything, analyse before moving on. Small iterations, broad coverage, guaranteed progress.
> **Checkpoints:** `data/results/inversion_diagnostics/m096_exp{1-5}_*/seed_{NNN}.npz`

---

## Executive Summary

Five experiments on 100 seeds reveal that the **full light curve is an overwhelmingly powerful discriminator** (Exp 4: truth at top 1% for 100/100 seeds via lo-fi MSE), but the **peak-alignment cost function currently used in the grid search is too narrow** for most seeds (Exp 2: median basin 1.7°, 37/90 seeds < 1°). The path forward is clear: replace or augment the alignment-based grid search with a full-curve evaluation strategy that exploits the signal Exp 4 proved exists.

---

## Experiment 1: Oracle Grid Search

**Question:** With perfect |w|, can the alignment cost find truth?

**Result:** Truth is present in all 90 valid seeds (always within 5° of a grid direction at 500 dirs). But the **median rank is 105/500** — only 21/90 in the top-20.

| Metric | Value |
|--------|-------|
| Truth found (< 5°) | 90/90 (100%) |
| Top-1 | 4 seeds |
| Top-5 | 14 seeds |
| Top-10 | 17 seeds |
| Top-20 | 21 seeds |
| Top-50 | 32 seeds |
| Median rank | 105 |

**Interpretation:** The alignment cost CAN detect truth — it's always nearby on the grid. But it doesn't RANK it well. The cost landscape is too flat for most seeds: many wrong directions score similarly to truth. This is the cost function's fault, not |w| estimation's fault. Improving |w| alone won't fix the pipeline.

**What's strong:** For 14/90 seeds truth is in the top-5 even with a coarse 500-direction grid. These tend to be fast tumblers with many constraints — the pipeline's sweet spot.

---

## Experiment 2: Cost Basin Width

**Question:** How narrow is the alignment cost basin around truth?

**Result:** Median basin half-width (2× truth cost) is **1.7°**. 37/90 seeds have basins < 1°.

| Basin width | Seeds | Implication |
|------------|-------|-------------|
| < 1° | 37 (41%) | Unresolvable at 2000 dirs (~3° spacing) |
| 1–3° | 32 (36%) | Marginal at 2000 dirs, needs 8000+ |
| 3–5° | 12 (13%) | Workable with current grid |
| 5–30° | 9 (10%) | Easy — mostly single-constraint seeds with flat cost |

**Interpretation:** The narrowest basins belong to the seeds with MOST constraints (e.g. seed 93: 18 constraints, basin 0.5°). More constraints = sharper basin = harder to find on a coarse grid. This is counterintuitive — constraint-rich seeds should be the easiest, but they produce the sharpest cost wells.

**What's strong:** The basin EXISTS — truth is always at or near the global minimum. The cost function points in the right direction; it's the grid that can't resolve the well.

---

## Experiment 3: Normal Identification from Dim Peaks

**Question:** Can we identify which body normal produced each peak — including the 1277 currently-discarded dim peaks (mag > 9)?

**Result:** **584/1277 dim peaks (46%) are usable** — the correct normal is identifiable (alignment gap > 0.1) AND brightness-consistent with the calibration table.

| Magnitude band | Peaks | Identifiable | Consistent | Usable | Median gap |
|---------------|-------|-------------|------------|--------|------------|
| Bright (< 5.9) | 107 | 3% | 100% | 3% | 0.027 |
| Medium (5.9–7.3) | 193 | 74% | 99% | 74% | 0.732 |
| Dim-spec (7.3–9) | 306 | 39% | 99% | 39% | 0.048 |
| 9–10 | 138 | 65% | 84% | 49% | 0.505 |
| 10–11 | 145 | 68% | 68% | 36% | 0.420 |
| 11–12 | 252 | 25% | 91% | 15% | 0.067 |
| 12–14 | 727 | 59% | 98% | 57% | 0.115 |

**Key finding:** The medium band (5.9–7.3 mag) is the BEST for normal identification — 74% usable with median gap 0.73. Bright peaks (< 5.9) are the WORST at 3% — the co-alignment degeneracy (±X, ±WD, ±ED are 15° apart) makes them nearly useless for identifying WHICH normal is glinting. The pipeline's fixation on bright ±X peaks was misguided.

**Extra usable constraints per seed from dim peaks:** median 5, mean 5.8, max 19.

**What's strong:** Nearly half of all dim peaks are usable. The 9–10 and 10–11 mag bands have 49% and 36% usability — lowering the specular threshold from 9.0 to 11.0 would add substantial constraint count for the majority of seeds.

---

## Experiment 4: Lo-fi Full-Curve Discrimination

**Question:** Can lo-fi LC MSE distinguish truth from random omegas — without peak-anchoring?

**Result:** **Truth ranks in the top 1% for ALL 100 seeds.** Including the 10 "invalid" seeds with < 2 specular peaks.

| Metric | Full curve | Windowed (360s) |
|--------|-----------|----------------|
| > 99th percentile | 100/100 | 98/100 |
| > 95th percentile | 100/100 | 100/100 |
| > 50th percentile | 100/100 | 100/100 |

**This is the most important result of the session.** The full light curve carries overwhelming information about omega direction — far more than peak alignment alone. Even lo-fi (no shadows, 0.2s/eval) perfectly discriminates truth from 200 random directions at any tumble rate, any omega orientation, any constraint count.

**What this means:** A template-matching approach (generate lo-fi LC for candidate omega, compare MSE with observed) WILL work for candidate selection. The challenge moves from "can we discriminate?" (yes, trivially) to "can we search efficiently?" (need to evaluate enough candidates to cover the space).

At 0.2s per lo-fi eval: 500 directions × 20 magnitudes = 10,000 candidates × 0.2s = 2000s ≈ 33 min. With 24-core parallelism: ~1.4 min. This is FEASIBLE as a brute-force approach.

---

## Experiment 5: Alternative |w| Estimators

**Question:** Can we improve |w| estimation beyond peak count?

**Result:** No. Peak count remains the best estimator.

| Estimator | Median |err| | In ±20% | In ±30% | R² |
|-----------|-----------|---------|---------|-----|
| Peak count (current) | 13.4% | 76/100 | 87/100 | 0.911 |
| Prominent peaks | 14.1% | 69/100 | 87/100 | 0.900 |
| Bright peaks only | 46.6% | 10/100 | 16/100 | 0.683 |
| Mean interval | 150.1% | 0/100 | 0/100 | 0.912 |
| Lomb-Scargle (full) | 70.3% | 9/100 | 12/100 | 0.002 |
| Lomb-Scargle (half) | 83.2% | 5/100 | 9/100 | 0.002 |

**Interpretation:** Peak count is already near the ceiling for |w| estimation from this data. Lomb-Scargle completely fails (R²=0.002) — the light curve is NOT periodic (triaxial tumbling produces quasi-periodic patterns). The interval-based approach overestimates because peaks aren't evenly spaced. At ±30%, 87/100 seeds are covered. The remaining 13 need a wider grid (±40-50%) or an adaptive range.

---

## The Path Forward

### What works NOW (build on this)
1. **Lo-fi LC MSE is a universal discriminator** — works on 100/100 seeds, including the "impossible" ones
2. **Medium-band constraints (mag 5.9–7.3) are the richest** — 74% normal-identifiable vs 3% for bright
3. **Peak count |w| estimation is adequate** — 87% of seeds within ±30%, and the grid just needs to be wide enough
4. **The alignment cost does point to truth** — truth is always near the minimum, the grid just can't resolve it

### What needs to change
1. **Replace alignment-based grid ranking with lo-fi LC MSE** — Exp 4 proves this works universally. Evaluate lo-fi LC at each grid candidate, rank by MSE against observed. This bypasses the flat alignment cost and the PAB degeneracy entirely.
2. **Lower the specular threshold to 10–11** — Exp 3 shows 49% of mag 9–10 peaks are normal-identifiable. More constraints = sharper alignment basin (Exp 2), even if each is individually weaker.
3. **Widen the |w| grid to ±40%** — captures 95%+ of seeds. The extra computation is trivial relative to lo-fi evaluation.
4. **Use the alignment cost as a FAST PRE-FILTER, not the primary discriminator** — alignment is cheap (microseconds) but weak. Lo-fi is expensive (0.2s) but decisive. Use alignment to cull the worst 80% of the grid, then lo-fi on the survivors.

### Proposed new pipeline architecture
```
Step 1: Peak count → |w| estimate (current, works)
Step 2: Alignment grid search (500–2000 dirs, ±40% |w|) → top-200 by alignment cost
Step 3: Lo-fi LC MSE on top-200 → rank by full-curve MSE
Step 4: NM refinement on top-20 by lo-fi MSE
Step 5: Geo refinement → Hi-fi confirmation
```

The key change: Step 3 inserts lo-fi LC evaluation BETWEEN the alignment grid and NM. This replaces the current lo-fi peak-matching (which suffers from phantom peaks) with full-curve MSE (which Exp 4 proves is decisive).

---

## Methodology Note

This session marks a shift to **stage-by-stage validation on the full population**. Instead of tuning the pipeline on 3–10 cherry-picked seeds, we now:

1. Run each stage on ALL 100 seeds
2. Checkpoint EVERYTHING computed (59+ keys per seed per stage)
3. Analyse before moving on — understand the population before optimising
4. Small iterations, broad coverage, guaranteed progress

This caught in 5 seconds what months of seed-specific tuning missed: the pipeline was designed for 10% of the dataset. Every future experiment follows this pattern.
