#!/usr/bin/env python3
"""
m125 [INLINE] -- keep_better wrapper analysis.

Provenance: this is a strategist-time INLINE re-scoring, NOT a full writer-authored
micro experiment. Re-processes m124/summary.json + m115 results with zero
new compute. Emitted by the research-loop strategist 2026-04-16 after the analyst
flagged it as the highest-information next check.

Question answered
-----------------
m124 REFUTED the hypothesis "L-BFGS polish surrogate-cost ratio matches hi-fi
MSE ratio to within 30%" (2/12 basins passed). But the PRODUCT-level question is
different: does a polish-with-safety-wrapper (min(hifi_before, hifi_after) per
basin) IMPROVE the best hi-fi MSE per seed compared to plain m115 (DE + hi-fi
rank)?

Result
------
YES, on 3/4 seeds with DE basins:
  seed 14:  m115 best 0.1613  ->  wrapped 0.0162   (90% improvement)
  seed 27:  m115 best 0.3105  ->  wrapped 0.3105   (break-even, wrapper correctly
                                                    rejected seed-27 catastrophe)
  seed 74:  m115 best 0.3760  ->  wrapped 0.2534   (33% improvement)
  seed 93:  m115 best 0.0626  ->  wrapped 0.0187   (70% improvement)
  seed 46:  no DE basins in m115 -- n/a

Per-basin: 9/12 basins HELPED by polish, 3/12 HURT (all seed 27, caught by wrapper).

Interpretation
--------------
m124's REFUTED verdict was correct AS STATED (ratio-agreement, not improvement
prevalence). But the wrapper-protected L-BFGS polish IS a real improvement tool:
paired with hi-fi safety check, it catches the seed-27 catastrophe AND produces
33-90% hi-fi MSE reduction on the other 3 seeds.

Recommended architecture (revision to [[gradient-based-inversion]]):
  1. DE: find q0 attractors (m115 unchanged)
  2. L-BFGS polish each attractor (m123 method)
  3. Hi-fi validate before AND after polish
  4. KEEP min per basin
  5. Rank across basins by hi-fi MSE

This restores #open-active status; it is a genuine improvement on the m115
pipeline for 3/4 seeds tested.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / 'data' / 'results' / 'inversion_diagnostics'


def main() -> None:
    m124 = json.load(open(ROOT / 'm124' / 'summary.json'))

    per_seed: dict[int, list[dict]] = {}
    for c in m124['per_candidate']:
        if c['is_truth_start']:
            continue
        seed = c['seed']
        per_seed.setdefault(seed, []).append({
            'label': c['start_label'],
            'hifi_before': c['hifi_mse_before'],
            'hifi_after': c['hifi_mse_after'],
            'hifi_best_wrapped': min(c['hifi_mse_before'], c['hifi_mse_after']),
            'polish_helped': c['hifi_mse_after'] < c['hifi_mse_before'],
        })

    rows: list[dict] = []
    print("\n=== Per-seed best: wrapped vs plain m115 ===")
    print(f"{'seed':>5}  {'m115_best_hifi':>15}  {'wrapped_best':>13}  "
          f"{'naive_polish':>13}  {'improved?':>10}")
    for seed in sorted(per_seed.keys()):
        m115 = json.load(
            open(ROOT / 'm115_surrogate_pipeline' / f'seed_{seed:03d}' / 'result.json'))
        m115_best = m115['best_hifi_mse']
        cands = per_seed[seed]
        naive_best = min(c['hifi_after'] for c in cands)
        wrapped_best = min(c['hifi_best_wrapped'] for c in cands)
        improved = wrapped_best < m115_best - 1e-6
        print(f"{seed:>5}  {m115_best:>15.6f}  {wrapped_best:>13.6f}  "
              f"{naive_best:>13.6f}  {str(improved):>10}")
        rows.append({
            'seed': seed,
            'm115_best_hifi_mse': float(m115_best),
            'wrapped_best_hifi_mse': float(wrapped_best),
            'naive_polish_best_hifi_mse': float(naive_best),
            'improvement_pct_vs_m115': (
                (m115_best - wrapped_best) / m115_best * 100 if improved else 0.0),
            'per_basin': cands,
        })

    n_helped = sum(1 for s in per_seed.values() for c in s if c['polish_helped'])
    n_hurt = sum(1 for s in per_seed.values() for c in s if not c['polish_helped'])

    # Save result next to the other diagnostics directories.
    out = ROOT / 'm125_keep_better' / 'summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'provenance': 'INLINE strategist re-scoring 2026-04-16 after m124',
        'source_files': [
            'm124/summary.json',
            'm115_surrogate_pipeline/seed_{014,027,074,093}/result.json',
        ],
        'seeds': sorted(per_seed.keys()),
        'per_seed': rows,
        'n_basins_helped': n_helped,
        'n_basins_hurt': n_hurt,
        'n_basins_total': n_helped + n_hurt,
        'verdict': 'WRAPPER_IMPROVES_ON_MICRO115 on 3/4 seeds; seed 27 breaks even',
    }
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out}")
    print(f"\nSummary: {n_helped}/12 basins HELPED; {n_hurt}/12 HURT (all seed 27, wrapper rejects)")


if __name__ == '__main__':
    main()
