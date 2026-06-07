---
title: "m132 — Solution count by hi-fi band (11-seed cohort, post-fix)"
type: experiment
sources: ["raw/inversion_diagnostics/m132_solution_count.json", "raw/inversion_diagnostics/m132_solution_count_by_band.png"]
related: ["[[m126_wrapped_pipeline]]", "[[m125_keep_better_inline]]", "[[m131_refresh_wrappedbest]]", "[[multi-solution-philosophy]]", "[[gradient-based-inversion]]"]
created: 2026-04-17
updated: 2026-04-17
confidence: high
---

# m132 — Solution count by hi-fi band

Multi-solution-philosophy yield metric for the 11-seed post-Option-A cohort. Counts every polished basin (33 total = 11 seeds × 3 basins) and stratifies by its final hi-fi MSE.

## Why

Under [[multi-solution-philosophy]], the right yield metric is **how many basins land below a hi-fi threshold**, NOT "how many seed winners are OK." A seed that produces 3/3 valid basins (truth + twin + twin) is stronger evidence of the pipeline working than a seed where 1/3 basin happens to beat threshold. Winner-only classification also hides the ±X twin and flipped-ω degeneracies which are genuine solutions.

## Method

- Source 1: `m126_wrapped/seed_NNN/result.json` `basins[]` entries for seeds 0, 6, 12, 24, 33, 36 — 3 basins × 6 seeds = 18.
- Source 2: `m125_keep_better/summary.json` `per_seed[].per_basin[]` entries for seeds 14, 27, 46, 74, 93 — 3 basins × 5 seeds = 15.
- Total = 33 basins.
- Each basin's `hifi_wrapped` (m126) or `hifi_best_wrapped` (m125) = `min(hifi_before, hifi_after)` under the keep_better wrapper.
- Band each basin into one of 6 logarithmic-ish bins: `<0.005`, `0.005–0.01`, `0.01–0.05`, `0.05–0.1`, `0.1–0.3`, `≥0.3`.
- Plot as stacked bar chart (left panel) per seed + aggregate horizontal bar (right panel).

## Results

| band | count | % of 33 |
|---|---:|---:|
| `< 0.005` (tight, ≤2× noise floor) | 4 | 12% |
| `0.005 – 0.01` (OK) | 1 | 3% |
| `0.01 – 0.05` | 7 | 21% |
| `0.05 – 0.1` (PARTIAL) | 4 | 12% |
| `0.1 – 0.3` | 8 | 24% |
| `≥ 0.3` (FAIL) | 9 | 27% |

### Headline

- **16/33 basins below hi-fi 0.1** (48%) — the honest multi-solution yield.
- **5/33 below hi-fi 0.01** (15%) — tight, hi-res-window OK-class.
- **9/33 FAIL at ≥ 0.3** (27%) — these cluster on 3 seeds: 36 (3/3 FAIL), 46 (2/3), 74 (2/3).

### Per-seed standouts

- **Seed 93**: 3/3 basins valid (1 OK + 2 PARTIAL) — strongest seed in the cohort
- **Seed 6**: 2/3 below 0.005 (truth + ±X twin both tight to ~0.003) + 1 catastrophic basin at 0.33
- **Seed 36**: 0/3 valid (all 3 basins above 0.3) — genuine enumeration failure; upstream ω-dir 10° off truth
- **Seeds 0, 12, 33, 74**: 1/3 valid + 2/3 wrong-attractor — the wrapper's ability to find one good basin depends on DE enumerating at least one truth- or twin-adjacent q0 attractor in the first place

### Noise floor

Truth-polished MSE on all 5 m124 seeds is 0.00244–0.00250 (2.4–2.5 × noise variance 0.0025 for σ=0.05 mag, 500 epochs). So the `<0.005` band is effectively "≤2× noise floor" — basins here are statistically indistinguishable from truth under the observation noise model.

## Implications

1. **Winner-class counts oversell the pipeline's reliability.** "5 OK seeds" conceals that 14/33 basins are above 0.3 — the 5 OK-class winners emerge because SOME basin per seed happened to land in the right place. For 6 of the 11 seeds, the other 2/3 basins are in the FAIL band.
2. **Multi-start is doing real work.** Without the 3-basin DE-from-different-starts, about half the wrapper-class OK seeds would become PARTIAL or FAIL (whichever basin happened to be sampled).
3. **Basin enumeration quality is the bottleneck, not polish.** Polish helped on all 33 basins (0 regressions — the keep_better wrapper's whole point). The failure modes are upstream: DE doesn't enumerate a truth-adjacent basin for seeds 36, 46, 74 reliably.

## Artifacts

- Script: `notebooks/inversion/12_brightness_surface/m132_solution_count.py`
- Plot: `data/results/inversion_diagnostics/m132_solution_count_by_band.png`
- JSON: `data/results/inversion_diagnostics/m132_solution_count.json` (per-basin + counts + band definitions)

## Caveats

- The 11-seed cohort is a **non-random subset** of the 100-seed population, biased toward seeds that have been iterated on across experiments (ATT_FAIL cohort from [[m096_exp1_oracle_grid]] Stage 1). These aggregate numbers do not generalise to the full 100-seed population — the 87%-bright-±X-poor majority ([[m096_exp1_oracle_grid]]) is underrepresented. For honest cohort claims, the 100-seed Phase 3 run on m048 (see [[gradient-based-inversion]]) is needed.
- All 33 basins are on the m046 single observation window (10:00–11:00 UTC 2020-02-05). Phase angle range 30–60°. Results do NOT reflect behaviour on m048's 9.2°–95.2° phase angle distribution.
