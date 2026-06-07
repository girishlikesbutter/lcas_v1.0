# NEXT_SESSION_PROMPT — Play 1: consolidate the random m048 cohort under ρ-band yield

Use this verbatim as the `/research-loop` arguments at start of next session.

---

The user has aligned on a strategic reframe (logged in CURRENT_STATE.md and `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/project_strategic_reframe_2026_04_29.md`). The headline:

- **m115 q-finder works.** Don't redesign it. Don't continue cost-shape engineering on alignment-cost descendents (m136 kernel-consistency / m138 H1 surrogate-attitude-isoshell / any new IPL-centroid variant). That direction is suspended.
- **m103 alignment-cost ranking is the bottleneck.** It's anti-correlated with truth on failure seeds.
- **Surrogate full-LC MSE re-ranking of m103's existing pool is the validated fix.** Three independent validations exist (m133, m135, m138 surr-rerank).
- **Success metric is ρ-band candidate-set yield, not single-pointer q0_err.** Acceptance bar is ρ < 4 (Band A or B). Band C is publishable as an LC fit with a partial-state caveat. Headline cohort number is **% seeds with ≥1 (A∪B) basin**.

Your job in this session is **Play 1: consolidate**. Build a clean cohort report.

## What Play 1 produces

A single markdown report at `data/results/inversion_diagnostics/play1_random_cohort_yield/REPORT.md` with:

1. **Per-seed candidate sets** — for each of the 22 random m048 seeds (the random_25 cohort excluding the 3 m103-failures 35/42/69), every basin with hi-fi MSE < 1.0:
   - hi-fi MSE, ρ, ρ-band (A/B/C/D)
   - q0_err, w_dir, w_mag
   - source ω rank under each of the 3 cost re-rankers + alignment-cost baseline (provenance)
2. **Cohort yield table** (headline acceptance is the ≥1 (A∪B) row):
   ```
   | metric                          | baseline (top-3 alignment) | top-K=7 (3-cost union) |
   | seeds with ≥1 Band A basin      | 4 (16%)                     | ?                       |
   | seeds with ≥1 (A∪B) basin ★     | 4 (16%)                     | ?                       |
   | seeds with ≥1 (A∪B∪C) basin     | 5 (20%)                     | ?                       |
   ```
3. **Three example overlay plots** — one each from a Band-A, Band-B, and Band-C seed. The A and B examples are the acceptance-rate exhibits; the C example is the "still publishable as a fit even if state is partial" exhibit.

## How Play 1 works (no new physics, no new costs)

1. **Re-rank the m103 pool by surrogate-LC.** For each seed with `data/results/inversion_diagnostics/m103_hybrid_m048/seed_NNN/geo_ckpt.npz`:
   - Score the 26 candidates with the m133 3-cost union: `surr_q0polish_mse`, `surr_autocorr`, `surr_spectrum`.
   - Take the union of top-3 from each → expected K ≈ 5–7.
   - Infrastructure: `notebooks/inversion/14_rerank_experiment/{rerank.py, q0_polish_cost.py, costs_extras.py}`.

2. **Run m115 on each top-K ω.** Existing pipeline. Either:
   - Loop the existing `m115_surrogate_pipeline.py` per ω with `M115_SORT_BY=oracle` style override pointing to the re-ranked list, OR
   - Wrap `invert.py --seed N --traj-source m048` with the new ranker. Add `--omega-source rerank` flag if needed.
   - Verify whether m115 already supports passing K candidates or whether you need to wrap it. (Check `m115_surrogate_pipeline.py::run_seed` and `load_omega_candidates`.)

3. **Collect all basins per seed**, not just the winner.
   - m115 already produces multi-basin output via DE multi-start (n_basins per seed in result.json).
   - m126 polish runs per basin via `keep_better` wrapper.
   - Hi-fi validate every polished basin under ρ < 4.5 (i.e. hi-fi MSE < 0.5, generous bound — the band classification will sort it).
   - Save each basin's predicted hi-fi LC for the overlay plots (`feedback_save_hifi_lcs.md`).

4. **Classify by ρ-band.** ρ = √(hifi_MSE / 0.0025), bands A/B/C/D per `feedback_rho_band_convention.md`.

5. **Write the cohort report.** Markdown table with the comparison to baseline.

## What you might find / report honestly

- m133 predicted +6 seeds gain top-3 truth-close ω (9/17 → 15/17). Under multi-basin counting at K=7, expected lift is similar but not directly comparable — you're counting Band-A∪B∪C yield, not pool top-3 hits.
- High-phase / constraint-poor seeds (28, 69, possibly 23 if it's in this cohort) may still be Band D — that's the genuine envelope edge, not a Play-1 failure.
- Seed 91 is a Band-A reference; seed 6 is +X-twin-resolved; seed 79 should move from C up to B if the rerank works as predicted.

## What NOT to do

- Don't propose new cost variants. The brief is "use what we already have."
- Don't run anything on m046. Play 1 is m048-only.
- Don't run new m103 grids. Play 1 only consumes existing `geo_ckpt.npz`.
- Don't write a 6-DOF DE script (that's Play 3, gated behind Plays 1 and 2).
- Don't twin-filter. ρ < 2 is valid regardless.

## Files / paths to read first

1. `notebooks/inversion/CURRENT_STATE.md` — the reframe in detail
2. `~/.claude/projects/-home-girish-projects-lcas-v1-0/memory/MEMORY.md` and the three new memory files referenced there
3. `notebooks/inversion/14_rerank_experiment/rerank.py` and `costs_extras.py` — the m133 scorers
4. `notebooks/inversion/12_brightness_surface/m115_surrogate_pipeline.py` — to see the current ω-loading and decide wrap vs patch
5. `data/results/inversion_diagnostics/batch_m048_v1/batch_summary.json` — the baseline number you'll be comparing against
6. The Play 1 harness skeleton at `notebooks/inversion/15_play1_consolidate/` if that has been started in the previous session (check git log)

## Wind-down rule

When Play 1 is done OR you hit ~70% context: rewrite the top of CURRENT_STATE.md with the new state, commit, and stop. Don't try to start Play 2 in the same session.

The user wants production yield numbers, not another experimental thread.
