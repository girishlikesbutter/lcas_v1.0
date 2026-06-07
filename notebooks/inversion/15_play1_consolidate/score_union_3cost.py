#!/usr/bin/env python3
"""Phase 1a — score the random m048 cohort by the m133 3-cost union (top-K=7).

For each seed in the random_25 cohort with a usable m103 geo_ckpt + cached
surrogate-cost scores under rerank_experiment/, compute the union of top-3
candidates from each of {surr_q0polish_mse, surr_autocorr, surr_spectrum}
and write the union-rank ω indices to disk.

This is read by m115's `M115_SORT_BY=union_3cost` mode (next session adds the
mode; this script ships the data it consumes).

Outputs:
  data/results/inversion_diagnostics/play1_random_cohort_yield/
    seed_NNN_union3_topK.json   — per seed, the union top-K ω indices
    union3_diagnostic.json      — cohort-level diagnostic
    union3_summary.md           — human-readable summary

Usage:
  python3 notebooks/inversion/15_play1_consolidate/score_union_3cost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Random m048 25-seed cohort (from batch_m048_v1/batch_summary.json).
RANDOM_25 = [6, 7, 8, 11, 16, 17, 34, 35, 42, 45, 47, 48, 51, 57, 59,
             64, 67, 69, 71, 78, 79, 84, 89, 91, 99]

# Excluded by upstream failure (geo_timeout / m103 crash). m115 cannot consume
# their pools because they don't have geo_ckpt.npz with 26 candidates.
EXCLUDED = {35, 42, 69}

# m133 cross-validated 3-cost union. q0polish was the m048-strong cost; autocorr
# was the cross-cohort robust cost; spectrum was m046-strong and m048-acceptable.
UNION_COSTS = ['surr_q0polish_mse', 'surr_autocorr', 'surr_spectrum']
TOP_PER_COST = 3

DIAG = PROJECT_ROOT / "data/results/inversion_diagnostics"
RERANK_DIR = DIAG / "rerank_experiment"
M103_DIR = DIAG / "m103_hybrid_m048"
OUT_DIR = DIAG / "play1_random_cohort_yield"


def load_seed_costs(seed: int) -> Dict | None:
    """Load all surrogate costs + q0polish + geo_ckpt metadata for one seed.

    Returns None if any required file is missing.
    """
    scores_path = RERANK_DIR / f"seed_{seed:03d}_scores.json"
    q0p_path = RERANK_DIR / f"seed_{seed:03d}_q0polish.json"
    geo_path = M103_DIR / f"seed_{seed:03d}" / "geo_ckpt.npz"

    if not scores_path.exists():
        return None
    if not q0p_path.exists():
        return None
    if not geo_path.exists():
        return None

    with open(scores_path) as f:
        scores = json.load(f)
    with open(q0p_path) as f:
        q0p = json.load(f)

    geo = np.load(str(geo_path), allow_pickle=True)
    n_cand = len(geo['geo_costs'])

    out = {
        'seed': seed,
        'n_candidates': n_cand,
        'geo_costs': geo['geo_costs'].tolist(),
        'w0_ref_errs': geo['w0_ref_errs'].tolist(),  # ORACLE — for diagnostic only
        'q0_ref_errs': geo['q0_ref_errs'].tolist() if 'q0_ref_errs' in geo else None,
        'costs': scores['costs'],
        'surr_q0polish_mse': q0p['surr_q0polish_mse'],
    }

    # Sanity: every cost array should have length n_cand.
    bad = [k for k in out['costs'] if len(out['costs'][k]) != n_cand]
    if bad:
        print(f"  seed {seed}: WARN cost length mismatch on {bad}")
    if len(out['surr_q0polish_mse']) != n_cand:
        print(f"  seed {seed}: WARN q0polish length {len(out['surr_q0polish_mse'])} != n_cand {n_cand}")

    return out


def cost_array(seed_data: Dict, cost_name: str) -> np.ndarray:
    """Pull a named cost array from a loaded seed_data dict."""
    if cost_name == 'surr_q0polish_mse':
        return np.asarray(seed_data['surr_q0polish_mse'], dtype=float)
    return np.asarray(seed_data['costs'][cost_name], dtype=float)


def union_top_k(seed_data: Dict, costs: List[str] = UNION_COSTS,
                top_per_cost: int = TOP_PER_COST) -> Dict:
    """Compute the union of top-N indices across the given costs.

    Returns a dict with the union order, per-cost top-N, and oracle diagnostics.
    """
    n_cand = seed_data['n_candidates']
    w_errs = np.asarray(seed_data['w0_ref_errs'])

    # NaNs / infs go to the bottom of any sort
    per_cost_top = {}
    for c in costs:
        arr = cost_array(seed_data, c)
        order = np.argsort(np.where(np.isfinite(arr), arr, np.inf))
        per_cost_top[c] = order[:top_per_cost].tolist()

    # Union, preserving first-occurrence order across costs (then within-cost rank)
    seen = set()
    union: List[int] = []
    for c in costs:
        for idx in per_cost_top[c]:
            if idx not in seen:
                seen.add(idx)
                union.append(int(idx))

    # Oracle diagnostics
    oracle_min_in_pool = float(np.min(w_errs))
    oracle_argmin_in_pool = int(np.argmin(w_errs))
    oracle_min_in_union = float(np.min(w_errs[union])) if union else float('nan')
    oracle_in_union = bool(oracle_argmin_in_pool in union)

    # Per-cost top-1 oracle distance (for diagnostic table)
    per_cost_top1_w_err = {
        c: float(w_errs[per_cost_top[c][0]]) for c in costs
    }

    return {
        'seed': seed_data['seed'],
        'n_candidates': n_cand,
        'union_costs': costs,
        'top_per_cost': top_per_cost,
        'per_cost_top_idx': per_cost_top,
        'per_cost_top1_w_err': per_cost_top1_w_err,
        'union_idx': union,
        'union_size': len(union),
        # ORACLE (do NOT use for ranking; reporting only)
        'oracle_min_w_err_in_pool': oracle_min_in_pool,
        'oracle_argmin_in_pool': oracle_argmin_in_pool,
        'oracle_min_w_err_in_union': oracle_min_in_union,
        'oracle_truth_close_in_union': oracle_in_union,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = [s for s in RANDOM_25 if s not in EXCLUDED]
    print(f"Scoring {len(cohort)} seeds (random_25 minus {sorted(EXCLUDED)} = upstream errors)")
    print(f"Costs: {UNION_COSTS}, top-{TOP_PER_COST} per cost")
    print()

    per_seed: List[Dict] = []
    union_sizes: List[int] = []
    truth_close_in_union: List[bool] = []
    truth_close_in_pool: List[float] = []

    for seed in cohort:
        seed_data = load_seed_costs(seed)
        if seed_data is None:
            print(f"  seed {seed:3d}: SKIP (missing input files)")
            continue
        result = union_top_k(seed_data)
        per_seed.append(result)
        union_sizes.append(result['union_size'])
        truth_close_in_union.append(result['oracle_truth_close_in_union'])
        truth_close_in_pool.append(result['oracle_min_w_err_in_pool'])

        # Per-seed JSON
        out_path = OUT_DIR / f"seed_{seed:03d}_union3_topK.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        marker = '✓' if result['oracle_truth_close_in_union'] else ' '
        print(f"  seed {seed:3d}: union K={result['union_size']:2d}  "
              f"oracle min_w_err pool={result['oracle_min_w_err_in_pool']:6.2f}°  "
              f"union={result['oracle_min_w_err_in_union']:6.2f}°  {marker}")

    # Cohort diagnostic
    n_seeds = len(per_seed)
    n_truth_in_union = sum(truth_close_in_union)
    pool_lt_5 = sum(1 for w in truth_close_in_pool if w < 5)
    pool_lt_10 = sum(1 for w in truth_close_in_pool if w < 10)
    pool_lt_20 = sum(1 for w in truth_close_in_pool if w < 20)

    diagnostic = {
        'cohort_size': n_seeds,
        'cohort_seeds': [r['seed'] for r in per_seed],
        'excluded_upstream': sorted(EXCLUDED),
        'union_costs': UNION_COSTS,
        'top_per_cost': TOP_PER_COST,
        'union_size_mean': float(np.mean(union_sizes)) if union_sizes else 0.0,
        'union_size_min': int(np.min(union_sizes)) if union_sizes else 0,
        'union_size_max': int(np.max(union_sizes)) if union_sizes else 0,
        'oracle_truth_close_in_pool_lt_5deg': pool_lt_5,
        'oracle_truth_close_in_pool_lt_10deg': pool_lt_10,
        'oracle_truth_close_in_pool_lt_20deg': pool_lt_20,
        'oracle_truth_close_in_union': n_truth_in_union,
        'note': (
            'oracle_truth_close_in_pool_lt_5deg = how many seeds have ANY ω in the 26-candidate '
            'pool within 5° of truth (upper bound — what m115 could find under perfect ranking). '
            'oracle_truth_close_in_union = how many of those the union top-K=7 ranking '
            'actually surfaces. Difference = ranking-quality gap.'
        ),
    }

    with open(OUT_DIR / 'union3_diagnostic.json', 'w') as f:
        json.dump(diagnostic, f, indent=2)

    # Markdown summary
    md = [
        '# Phase 1a — m133 3-cost union ranking on random m048 cohort',
        '',
        f'**Cohort:** {n_seeds} seeds (random_25 minus upstream errors {sorted(EXCLUDED)}).',
        f'**Costs:** `{", ".join(UNION_COSTS)}` (top-{TOP_PER_COST} per cost, union taken).',
        f'**Union size:** mean = {diagnostic["union_size_mean"]:.1f}, '
        f'range = [{diagnostic["union_size_min"]}, {diagnostic["union_size_max"]}].',
        '',
        '## Pool-quality vs ranking-quality',
        '',
        f'- {pool_lt_5} of {n_seeds} seeds have ANY ω in the 26-candidate pool within  5° of truth (m115\'s bridging radius).',
        f'- {pool_lt_10} of {n_seeds} seeds have ANY ω within 10°.',
        f'- {pool_lt_20} of {n_seeds} seeds have ANY ω within 20°.',
        f'- **{n_truth_in_union} of {n_seeds} seeds have the truth-closest ω in the union top-K.**',
        '',
        f'Ranking-quality ceiling = pool-quality (≤5°). Gap = '
        f'{pool_lt_5 - n_truth_in_union} seeds where truth-close ω is in the pool but not in the union.',
        '',
        '## Per-seed table',
        '',
        '| seed | K | oracle ω-err pool | oracle ω-err union | truth-close in union? | per-cost top-1 ω-err (q0polish / autocorr / spectrum) |',
        '|---:|---:|---:|---:|:---:|:---|',
    ]
    for r in per_seed:
        in_union = '✓' if r['oracle_truth_close_in_union'] else '✗'
        per_cost_top1_str = ' / '.join(
            f'{r["per_cost_top1_w_err"][c]:.2f}°' for c in UNION_COSTS
        )
        md.append(
            f'| {r["seed"]:3d} | {r["union_size"]} | '
            f'{r["oracle_min_w_err_in_pool"]:.2f}° | '
            f'{r["oracle_min_w_err_in_union"]:.2f}° | {in_union} | '
            f'{per_cost_top1_str} |'
        )
    md.append('')
    md.append('## Next: Phase 1b')
    md.append('')
    md.append(
        'Patch `m115_surrogate_pipeline.py::load_omega_candidates` to support '
        '`M115_SORT_BY=union_3cost` reading the per-seed `seed_NNN_union3_topK.json` '
        'files in this directory. Then run the batch under '
        '`M115_NUM_OMEGA_CANDIDATES=7 M115_SORT_BY=union_3cost TRAJ_SOURCE=m048`.'
    )

    summary_path = OUT_DIR / 'union3_summary.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(md))

    print()
    print(f'Wrote per-seed JSONs: {OUT_DIR}/seed_NNN_union3_topK.json')
    print(f'Wrote diagnostic:     {OUT_DIR / "union3_diagnostic.json"}')
    print(f'Wrote summary:        {summary_path}')
    print()
    print('Cohort summary:')
    print(f'  union size mean:   {diagnostic["union_size_mean"]:.1f}')
    print(f'  pool has ω<5°:     {pool_lt_5}/{n_seeds} seeds (ranking-quality ceiling)')
    print(f'  union has ω<min:   {n_truth_in_union}/{n_seeds} seeds (current ranking achieves)')
    print(f'  ranking gap:       {pool_lt_5 - n_truth_in_union} seeds')


if __name__ == '__main__':
    main()
