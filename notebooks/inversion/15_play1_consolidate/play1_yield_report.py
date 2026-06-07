#!/usr/bin/env python3
"""Phase 1d — aggregate m115/m126 outputs into a ρ-band yield report.

Reads existing per-seed result.json files from invert_m048_seed*/, m126_wrapped_m048/,
and m115_surrogate_pipeline_m048/ and produces a candidate-set yield report classified
by ρ-band (A/B/C/D).

Headline acceptance metric is **% seeds with ≥1 (A∪B) basin** per the 2026-04-29
strategic reframe. ρ = √hifi_MSE / 0.05 mag.

Works on whatever data exists right now. After Phase 1c batch lands, re-run for the
post-rerank numbers. The script is idempotent + non-destructive.

Usage:
  python3 notebooks/inversion/15_play1_consolidate/play1_yield_report.py
  python3 notebooks/inversion/15_play1_consolidate/play1_yield_report.py --tag baseline
  python3 notebooks/inversion/15_play1_consolidate/play1_yield_report.py --tag union3_K7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DIAG = PROJECT_ROOT / "data/results/inversion_diagnostics"

RANDOM_25 = [6, 7, 8, 11, 16, 17, 34, 35, 42, 45, 47, 48, 51, 57, 59,
             64, 67, 69, 71, 78, 79, 84, 89, 91, 99]

NOISE_SIGMA = 0.05  # mag — load-bearing for ρ definition


def rho_from_mse(mse: float) -> float:
    """ρ = √hifi_MSE / σ_noise. Returns inf for non-finite MSE."""
    if not np.isfinite(mse) or mse < 0:
        return float('inf')
    return float(np.sqrt(mse) / NOISE_SIGMA)


def band_from_rho(rho: float) -> str:
    """ρ < 2: A; 2≤ρ<4: B; 4≤ρ<8: C; ≥8: D."""
    if not np.isfinite(rho):
        return 'D'
    if rho < 2:
        return 'A'
    if rho < 4:
        return 'B'
    if rho < 8:
        return 'C'
    return 'D'


def load_invert_summary(seed: int) -> Optional[Dict]:
    """Load the invert.py top-level summary (winner only)."""
    path = DIAG / f"invert_m048_seed{seed:03d}" / "result.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_m126_basins(seed: int) -> List[Dict]:
    """Load all m126-polished basins for a seed.

    Primary source: m126_wrapped_m048/seed_NNN/result.json's `basins` list,
    which has per-basin hifi_after + q0_err_after + w_dir_err_after + w_mag_err_pct_after.

    Fallback: invert_m048_seedNNN/result.json's `winner` (single basin only).
    """
    seed_dir = DIAG / "m126_wrapped_m048" / f"seed_{seed:03d}"
    summary_path = seed_dir / "result.json"

    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            summary = None

        if summary is not None and isinstance(summary.get('basins'), list):
            basins: List[Dict] = []
            for i, b in enumerate(summary['basins']):
                # Use hifi_wrapped (= min(before, after)) as the canonical residual
                # — this is what the wrapper's keep_better promotes.
                mse = b.get('hifi_wrapped', b.get('hifi_after', b.get('hifi_before')))
                if mse is None:
                    continue
                mse = float(mse)
                rho = rho_from_mse(mse)
                basin = {
                    'basin_idx': int(b.get('basin_idx', i)),
                    'hifi_mse': mse,
                    'hifi_after': float(b['hifi_after']) if 'hifi_after' in b else None,
                    'hifi_before': float(b['hifi_before']) if 'hifi_before' in b else None,
                    'rho': rho,
                    'band': band_from_rho(rho),
                    'is_twin': bool(b.get('is_twin', False)),
                }
                if 'q0_err_after' in b:
                    basin['q0_err'] = float(b['q0_err_after'])
                elif 'q0_err' in b:
                    basin['q0_err'] = float(b['q0_err'])
                if 'w_dir_err_after' in b:
                    basin['w_dir_err'] = float(b['w_dir_err_after'])
                elif 'w_dir_err' in b:
                    basin['w_dir_err'] = float(b['w_dir_err'])
                if 'w_mag_err_pct_after' in b:
                    basin['w_mag_err_pct'] = float(b['w_mag_err_pct_after'])
                elif 'w_mag_err_pct' in b:
                    basin['w_mag_err_pct'] = float(b['w_mag_err_pct'])
                basins.append(basin)
            if basins:
                return basins

    # Fallback: invert summary winner only
    inv = load_invert_summary(seed)
    if inv is not None:
        winner = inv.get('winner') or inv
        if isinstance(winner, dict) and ('hifi' in winner or 'hifi_mse' in winner):
            mse = float(winner.get('hifi', winner.get('hifi_mse')))
            rho = rho_from_mse(mse)
            return [{
                'basin_idx': 0,
                'hifi_mse': mse,
                'rho': rho,
                'band': band_from_rho(rho),
                'q0_err': winner.get('q0_err'),
                'w_dir_err': winner.get('w0_err') or winner.get('w_dir_err'),
                'w_mag_err_pct': winner.get('w_mag_err_pct'),
                'source': 'invert_summary_winner_only',
            }]

    return []


def cohort_classification(seeds: List[int]) -> Dict:
    """Build the cohort-level ρ-band yield table."""
    per_seed: List[Dict] = []
    has_a = []
    has_ab = []
    has_abc = []
    upstream_failed: List[int] = []

    for seed in seeds:
        basins = load_m126_basins(seed)
        if not basins:
            upstream_failed.append(seed)
            per_seed.append({
                'seed': seed,
                'basins': [],
                'has_band_A': False,
                'has_band_AB': False,
                'has_band_ABC': False,
                'best_rho': None,
                'best_band': None,
                'note': 'no m126 output (upstream failure or not yet run)',
            })
            continue

        bands = [b['band'] for b in basins]
        ha = 'A' in bands
        hab = ha or ('B' in bands)
        habc = hab or ('C' in bands)
        best = min(basins, key=lambda b: b['rho'])

        per_seed.append({
            'seed': seed,
            'n_basins': len(basins),
            'basins': basins,
            'has_band_A': ha,
            'has_band_AB': hab,
            'has_band_ABC': habc,
            'best_rho': best['rho'],
            'best_band': best['band'],
            'best_q0_err': best.get('q0_err'),
            'best_w_dir_err': best.get('w_dir_err'),
            'best_w_mag_err_pct': best.get('w_mag_err_pct'),
        })
        has_a.append(ha)
        has_ab.append(hab)
        has_abc.append(habc)

    n_total = len(seeds)
    n_run = n_total - len(upstream_failed)

    return {
        'cohort_size_total': n_total,
        'cohort_size_with_output': n_run,
        'upstream_failed_seeds': upstream_failed,
        'n_seeds_with_band_A': sum(has_a),
        'n_seeds_with_band_AB': sum(has_ab),  # ★ headline
        'n_seeds_with_band_ABC': sum(has_abc),
        'pct_seeds_with_band_A': 100.0 * sum(has_a) / n_total if n_total else 0.0,
        'pct_seeds_with_band_AB': 100.0 * sum(has_ab) / n_total if n_total else 0.0,
        'pct_seeds_with_band_ABC': 100.0 * sum(has_abc) / n_total if n_total else 0.0,
        'per_seed': per_seed,
    }


def write_report(tag: str, classification: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    md = []
    n_total = classification['cohort_size_total']
    md.append(f'# Random m048 cohort — ρ-band yield ({tag})')
    md.append('')
    md.append(f'Cohort: {n_total} seeds (random_25). Upstream-failed: '
              f'{classification["upstream_failed_seeds"]}.')
    md.append('')
    md.append(f'ρ = √hifi_MSE / σ_noise where σ_noise = {NOISE_SIGMA} mag. '
              'A: ρ<2; B: 2-4; C: 4-8; D: ≥8. Bands classify the LC fit residual; '
              'state correctness is the (q0_err, w_dir, w_mag) triple alongside.')
    md.append('')
    md.append('## Cohort headline')
    md.append('')
    md.append('| Metric | Count | % |')
    md.append('|---|---:|---:|')
    md.append(f'| seeds with ≥1 Band A basin | {classification["n_seeds_with_band_A"]} | '
              f'{classification["pct_seeds_with_band_A"]:.0f}% |')
    md.append(f'| **seeds with ≥1 (A∪B) basin ★ acceptance** | '
              f'**{classification["n_seeds_with_band_AB"]}** | '
              f'**{classification["pct_seeds_with_band_AB"]:.0f}%** |')
    md.append(f'| seeds with ≥1 (A∪B∪C) basin (presentable LC) | '
              f'{classification["n_seeds_with_band_ABC"]} | '
              f'{classification["pct_seeds_with_band_ABC"]:.0f}% |')
    md.append('')

    md.append('## Per-seed candidate sets')
    md.append('')
    md.append('| seed | n basins | best ρ | best band | bands | best q0_err | best w_dir | best w_mag |')
    md.append('|---:|---:|---:|:---:|:---|---:|---:|---:|')
    for ps in classification['per_seed']:
        if not ps.get('n_basins'):
            md.append(f'| {ps["seed"]:3d} | — | — | — | (upstream failure) | — | — | — |')
            continue
        bands_compact = ''.join(b['band'] for b in ps['basins'])
        bb = ps['best_rho']
        rho_str = f'{bb:.2f}' if bb is not None and np.isfinite(bb) else '∞'
        q0_str = f'{ps["best_q0_err"]:.1f}°' if ps.get('best_q0_err') is not None else '?'
        wdir_str = f'{ps["best_w_dir_err"]:.2f}°' if ps.get('best_w_dir_err') is not None else '?'
        wmag_str = f'{ps["best_w_mag_err_pct"]:+.2f}%' if ps.get('best_w_mag_err_pct') is not None else '?'
        md.append(
            f'| {ps["seed"]:3d} | {ps["n_basins"]} | {rho_str} | '
            f'{ps["best_band"]} | {bands_compact} | {q0_str} | {wdir_str} | {wmag_str} |'
        )
    md.append('')

    md.append('## Per-basin detail (for seeds with at least one A or B basin)')
    md.append('')
    md.append('| seed | basin | ρ | band | hifi_MSE | q0_err | w_dir | w_mag |')
    md.append('|---:|---:|---:|:---:|---:|---:|---:|---:|')
    for ps in classification['per_seed']:
        if not ps.get('has_band_AB'):
            continue
        for b in ps['basins']:
            rho_str = f'{b["rho"]:.2f}' if np.isfinite(b['rho']) else '∞'
            q0_str = f'{b["q0_err"]:.1f}°' if b.get('q0_err') is not None else '?'
            wdir_str = f'{b["w_dir_err"]:.2f}°' if b.get('w_dir_err') is not None else '?'
            wmag_str = f'{b["w_mag_err_pct"]:+.2f}%' if b.get('w_mag_err_pct') is not None else '?'
            md.append(
                f'| {ps["seed"]:3d} | {b["basin_idx"]} | {rho_str} | {b["band"]} | '
                f'{b["hifi_mse"]:.4f} | {q0_str} | {wdir_str} | {wmag_str} |'
            )

    md.append('')
    md.append('## Notes')
    md.append('')
    md.append(
        '- Acceptance bar is ρ < 4 (Band A or B) per `feedback_rho_band_yield_metric.md`. '
        'Band C is publishable as an LC fit but flag the partial-state caveat. '
        'Band D is failure.'
    )
    md.append(
        '- Per-basin classification follows the 2026-04-17 Roberto report Table 2 format. '
        'A Band-A basin can be the truth, the +X twin, or another LC-degenerate state — '
        'the band classifies the LC, not the state.'
    )

    report_path = out_dir / f'REPORT_{tag}.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(md))

    json_path = out_dir / f'classification_{tag}.json'
    with open(json_path, 'w') as f:
        json.dump(classification, f, indent=2)

    print(f'Wrote {report_path}')
    print(f'Wrote {json_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='current',
                        help='Tag for the output report (e.g. "baseline", "union3_K7")')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: data/results/inversion_diagnostics/play1_random_cohort_yield/)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (DIAG / 'play1_random_cohort_yield')
    out_dir.mkdir(parents=True, exist_ok=True)

    classification = cohort_classification(RANDOM_25)
    write_report(args.tag, classification, out_dir)

    print()
    print('Cohort headline:')
    print(f'  ≥1 Band A basin:        {classification["n_seeds_with_band_A"]}/'
          f'{classification["cohort_size_total"]} '
          f'({classification["pct_seeds_with_band_A"]:.0f}%)')
    print(f'  ≥1 (A∪B) basin (★):     {classification["n_seeds_with_band_AB"]}/'
          f'{classification["cohort_size_total"]} '
          f'({classification["pct_seeds_with_band_AB"]:.0f}%)')
    print(f'  ≥1 (A∪B∪C) basin:       {classification["n_seeds_with_band_ABC"]}/'
          f'{classification["cohort_size_total"]} '
          f'({classification["pct_seeds_with_band_ABC"]:.0f}%)')


if __name__ == '__main__':
    main()
