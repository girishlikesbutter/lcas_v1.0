"""s018c hi-fi rerank — read top-3 LM-polished candidates per pilot seed,
render hi-fi LCs, classify by ρ-band, save per-seed verdicts.

Consumes:  results/s018c/seed{XXX}/top3_for_hifi.npz
Produces:  results/s018c/hifi_rerank.json + per-seed plots in results/s018c/

Wall budget: ~75 s/render × 5 seeds × 3 candidates / 8 workers = ~140 s.

This is the gating step for the s018c pre-registered question — does
≥4/5 of the pilot land in Band A∪B (ρ < 4) by hi-fi?
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import json
from pathlib import Path
from multiprocessing import Pool

import numpy as np

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"

sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from lib import hifi_render  # noqa: E402

OUT_DIR = SURVEY_DIR / "results" / "s018c"
PILOT_SEEDS = [6, 28, 79, 44, 84]


def render_one(args):
    seed, q0_wxyz, omega_rad = args
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    ctx = hifi_render.build_context(seed)
    pred = hifi_render.render_hifi(q0_wxyz, omega_rad, ctx)
    rho = hifi_render.rho_from_hifi(pred, ctx['mag_hifi_truth'])
    band = hifi_render.rho_band(rho)
    return {
        'seed': int(seed),
        'q0_wxyz': q0_wxyz.tolist(),
        'omega_rad': omega_rad.tolist(),
        'rho': float(rho),
        'band': band,
        'pred_mag_hifi': pred.tolist(),
    }


def main():
    print(f"=== s018c hi-fi rerank ===", flush=True)
    print(f"Pilot seeds: {PILOT_SEEDS}", flush=True)

    # Collect args: (seed, q0_wxyz, omega_rad) for each top-3 candidate.
    args_list = []
    seed_to_meta = {}
    for seed in PILOT_SEEDS:
        topk_path = OUT_DIR / f"seed{seed:03d}" / "top3_for_hifi.npz"
        if not topk_path.exists():
            print(f"  seed {seed}: no top3 (zero classifiable peaks?)", flush=True)
            seed_to_meta[seed] = {'no_topk': True}
            continue
        d = np.load(topk_path)
        for i in range(d['q0_final_wxyz'].shape[0]):
            args_list.append((seed,
                              d['q0_final_wxyz'][i],
                              d['omega_final_rad'][i]))

    print(f"Total hi-fi renders: {len(args_list)}", flush=True)

    t0 = time.time()
    with Pool(processes=8) as pool:
        results = pool.map(render_one, args_list, chunksize=1)
    print(f"Render wall: {time.time()-t0:.1f}s", flush=True)

    # Bundle per seed
    by_seed = {s: [] for s in PILOT_SEEDS}
    for r in results:
        by_seed[r['seed']].append(r)

    print(f"\n=== per-seed verdict ===", flush=True)
    print(f"  seed | rank | rho      | band | classification",
          flush=True)
    print(f"  -----+------+----------+------+----------------",
          flush=True)
    summary = {'pilot_seeds': PILOT_SEEDS, 'per_seed': {}}
    n_ab = 0  # seeds that landed in Band A or B (rank-1)
    for seed in PILOT_SEEDS:
        seed_results = by_seed[seed]
        if not seed_results:
            print(f"  {seed:4d} | --   | --       | --   | NO_ICS_GENERATED",
                  flush=True)
            summary['per_seed'][seed] = {'note': 'zero classifiable peaks; no ICs'}
            continue
        # rank-1 by lowest ρ (since the LM order may be diff)
        seed_results.sort(key=lambda x: x['rho'])
        for rank, r in enumerate(seed_results, start=1):
            print(f"  {seed:4d} |   {rank:1d}  | {r['rho']:7.3f}  | {r['band']:4s} | "
                  f"{'BAND A∪B' if r['band'] in ('A', 'B') else '----'}",
                  flush=True)
        rank1 = seed_results[0]
        if rank1['band'] in ('A', 'B'):
            n_ab += 1
        summary['per_seed'][seed] = {
            'rank1_rho': rank1['rho'],
            'rank1_band': rank1['band'],
            'all_rhos': [r['rho'] for r in seed_results],
            'all_bands': [r['band'] for r in seed_results],
        }
    summary['n_seeds_band_AB'] = int(n_ab)
    summary['threshold'] = '≥4/5 → cohort scan; <4/5 → S016-A fallback'
    summary['decision'] = 'PASS (cohort scan)' if n_ab >= 4 else 'FALL_BACK_S016A'

    out_path = OUT_DIR / "hifi_rerank.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== summary ===", flush=True)
    print(f"  Band A∪B count: {n_ab}/{len(PILOT_SEEDS)}", flush=True)
    print(f"  Decision: {summary['decision']}", flush=True)
    print(f"  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
