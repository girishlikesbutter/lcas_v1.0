#!/usr/bin/env python3
"""
m124 -- Hi-fi validation of m123 L-BFGS polished candidates.

Hypothesis (one falsifiable claim)
----------------------------------
Hi-fi MSE reduction factor tracks surrogate MSE reduction factor to within
±30% across the 12 DE-basin polished points.

    surr_ratio = surrogate_cost_before / surrogate_cost_after
    hifi_ratio = hifi_mse_before        / hifi_mse_after

Agreement metric: |log(surr_ratio/hifi_ratio)| < 0.3  <=> within ±30%.

    CONFIRMED  iff  >= 75% (9/12) basins within threshold
    REFUTED    iff  <  25% (3/12) basins within threshold
    PARTIAL    otherwise

Additional sub-check (truth-start polished, 5 seeds): truth_hifi_mse should
be ~noise^2 = 0.0025. Significantly larger => surrogate fidelity at truth is
worse than advertised.

Method
------
  Stage A: build candidate list.
    - Load m123/seed_NNN/polish.npz -> records_json (JSON array).
    - Split into 'truth' and 'basin_k' starts.
    - For each 'basin_k' record, match to m115/seed_NNN/result.json
      de_results.basins[k] and read hifi_mse (pre-polish hi-fi).
    - Persist candidates.json.

  Stage B: run hi-fi evals in parallel.
    - Build per-seed ExperimentContext via setup_experiment (5 ctx total).
    - Fork Pool(POOL); module-global stores ctx_by_seed + obs data.
    - Dispatch 22 jobs: 5 truth (one hi-fi call per seed at exact truth:
      provides noise-floor reference) + 17 polished-point evals (5 truth-
      polished + 12 basin-polished).
    - Each worker calls m115.hifi_validate(q0, w, obs_times, I_tensor,
      observed_lc, ctx).
    - Persist hifi_results.npz.

  Stage C: assemble summary.json with per-candidate agreement + verdict.

Env
---
  MICRO124_SEEDS  default "14,27,46,74,93"
  MICRO124_POOL   default 8
  MICRO124_FORCE  default 0 (set to 1 to override skip-if-exists)

Outputs under data/results/inversion_diagnostics/m124/:
  candidates.json   Stage A
  hifi_results.npz  Stage B
  summary.json      Stage C
  run.log           tee'd stdout

TODOs (deferred)
----------------
- Per-candidate hi-fi LC plot overlays (data preserved in hifi_results.npz).
- Seed-level attractor clustering cross-ref to m123 summary.
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
import time
import json
import math
from pathlib import Path
import multiprocessing as mp

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion" / "12_brightness_surface"))
sys.path.insert(0, '/home/girish/surrogate_model')
os.chdir(PROJECT_ROOT)

from lib.experiment_setup import setup_experiment
# Reuse hifi_validate from m115 (do NOT reimplement).
from m115_surrogate_pipeline import hifi_validate  # noqa: E402


# ---- Config --------------------------------------------------------------
SEEDS = [int(s) for s in os.environ.get(
    'MICRO124_SEEDS', '14,27,46,74,93').split(',') if s.strip()]
POOL_SIZE = int(os.environ.get('MICRO124_POOL', '8'))
FORCE = os.environ.get('MICRO124_FORCE', '0') == '1'

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
MICRO46_NPZ = RESULTS_DIR / "m046_trajectories" / "m046_trajectories.npz"
MICRO115_BASE = RESULTS_DIR / "m115_surrogate_pipeline"
MICRO123_BASE = RESULTS_DIR / "m123"

OUT_DIR = RESULTS_DIR / "m124"

NOISE_SEED = 42
NOISE_SIGMA = 0.05
N_OBS = 500

# Agreement threshold on |log(surr_ratio) - log(hifi_ratio)|.
LOG_RATIO_THRESHOLD = 0.3  # ~ ±30%
CONFIRM_FRAC = 0.75
REFUTE_FRAC = 0.25


# ---- Logging / IO helpers ------------------------------------------------
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data); f.flush()
            except (ValueError, OSError):
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except (ValueError, OSError):
                pass


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode('utf-8', errors='replace')
    return str(o)


def atomic_json_save(filepath, data):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, filepath)


# ---- Stage A: build candidate list --------------------------------------
def load_micro115_basins(seed):
    """Return list of basins (with 'hifi_mse' field) from m115 result.json."""
    path = MICRO115_BASE / f"seed_{seed:03d}" / "result.json"
    if not path.exists():
        return []
    with open(path) as f:
        d = json.load(f)
    return d.get('de_results', {}).get('basins', [])


def load_micro123_records(seed):
    """Return list of polish records from m123 polish.npz records_json."""
    path = MICRO123_BASE / f"seed_{seed:03d}" / "polish.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing m123 polish.npz for seed {seed}: {path}")
    d = dict(np.load(path, allow_pickle=True))
    records = json.loads(str(d['records_json']))
    return records


def build_candidates():
    """Return list of candidate dicts covering all seeds + labels."""
    cands = []
    for seed in SEEDS:
        records = load_micro123_records(seed)
        m115_basins = load_micro115_basins(seed)

        for rec in records:
            label = rec['start_label']
            cand = {
                'seed': int(seed),
                'start_label': label,
                'start_q0_wxyz': list(rec['start_q0_wxyz']),
                'start_omega_rad': list(rec['start_omega_rad']),
                'final_q0_wxyz': list(rec['final_q0_wxyz']),
                'final_omega_rad': list(rec['final_omega_rad']),
                'surr_cost_before': float(rec['start_cost']),
                'surr_cost_after': float(rec['final_cost']),
                'hifi_mse_before': None,
                'is_truth_start': (label == 'truth'),
            }
            if label.startswith('basin_'):
                try:
                    idx = int(label.split('_', 1)[1])
                except ValueError:
                    idx = -1
                if 0 <= idx < len(m115_basins):
                    hifi_before = m115_basins[idx].get('hifi_mse', None)
                    cand['hifi_mse_before'] = (
                        float(hifi_before) if hifi_before is not None else None)
            cands.append(cand)
    return cands


# ---- Stage B: hi-fi evaluation in a worker pool -------------------------
# Per-seed shared state (built in parent, inherited via fork).
_WORKER_STATE = {}  # seed -> {ctx, obs_times, I_tensor, observed_lc}


def build_seed_state(seed):
    """Build ExperimentContext + obs data for one seed."""
    m46 = np.load(MICRO46_NPZ)
    truth_q0 = m46['q0s'][seed].astype(np.float64)
    truth_q0 /= np.linalg.norm(truth_q0)
    truth_omega0 = m46['omega0s'][seed].astype(np.float64)

    ctx = setup_experiment(
        n_observations=N_OBS,
        end_time_utc='2020-02-05T11:00:00',  # DATA_INVARIANTS.md — must match m046 1-hour window
        noise_sigma=NOISE_SIGMA,
        random_seed=NOISE_SEED,
        skip_true_lc=False,
        true_q0_wxyz=truth_q0,
        true_omega0_rad=truth_omega0,
    )

    return {
        'ctx': ctx,
        'obs_times': ctx.observation_times.astype(np.float64),
        'I_tensor': ctx.inertia_tensor.astype(np.float64),
        'observed_lc': ctx.observed_lc.astype(np.float64),
        'truth_q0': truth_q0,
        'truth_omega0': truth_omega0,
    }


def worker_eval(job):
    """
    Worker: evaluate hi-fi MSE for (seed, q0_wxyz, omega_rad, tag).

    'tag' is a unique string id so caller can match result to request.
    Returns dict including hifi_mse, hifi_mags, wall_seconds.
    """
    seed = job['seed']
    tag = job['tag']
    q0 = np.asarray(job['q0_wxyz'], dtype=np.float64)
    w = np.asarray(job['omega_rad'], dtype=np.float64)

    state = _WORKER_STATE[seed]
    t0 = time.time()
    hifi_mse, hifi_mags = hifi_validate(
        q0, w,
        state['obs_times'], state['I_tensor'],
        state['observed_lc'], state['ctx'])
    wall = time.time() - t0

    return {
        'tag': tag,
        'seed': seed,
        'hifi_mse': float(hifi_mse),
        'hifi_mags': np.asarray(hifi_mags, dtype=np.float64),
        'wall_seconds': float(wall),
    }


def _worker_init(state_dict):
    # Not used (fork-based inheritance); kept for API clarity if spawn fallback needed.
    global _WORKER_STATE
    _WORKER_STATE.update(state_dict)


def stage_B_hifi(candidates):
    """Run hi-fi evals for all candidates + truth-reference points."""
    t0 = time.time()
    out_npz = OUT_DIR / "hifi_results.npz"

    if out_npz.exists() and not FORCE:
        print(f"  loading cached {out_npz.name}")
        d = dict(np.load(out_npz, allow_pickle=True))
        results = json.loads(str(d['results_json']))
        print(f"  Stage B (cached) done in {time.time()-t0:.1f}s")
        return results

    # 1. Build per-seed state in parent process (fork inherits).
    print(f"\n[Stage B] building ExperimentContext for {len(SEEDS)} seeds "
          f"(each ~30-60s, serial)")
    global _WORKER_STATE
    for seed in SEEDS:
        t_s = time.time()
        print(f"  building ctx for seed {seed}...")
        _WORKER_STATE[seed] = build_seed_state(seed)
        print(f"    done in {time.time()-t_s:.1f}s")

    # 2. Assemble jobs: one per polished candidate + one per seed for truth ref.
    jobs = []
    # Truth-reference jobs (one per seed, unique tag).
    for seed in SEEDS:
        st = _WORKER_STATE[seed]
        jobs.append({
            'tag': f"truthref_seed_{seed:03d}",
            'kind': 'truth_reference',
            'seed': seed,
            'q0_wxyz': st['truth_q0'].tolist(),
            'omega_rad': st['truth_omega0'].tolist(),
        })
    # Polished candidate jobs.
    for i, c in enumerate(candidates):
        jobs.append({
            'tag': f"cand_{i:03d}_seed_{c['seed']:03d}_{c['start_label']}",
            'kind': 'polished',
            'cand_idx': i,
            'seed': c['seed'],
            'q0_wxyz': c['final_q0_wxyz'],
            'omega_rad': c['final_omega_rad'],
        })

    n_jobs = len(jobs)
    print(f"\n[Stage B] running {n_jobs} hi-fi evals on Pool({POOL_SIZE})...")
    t_pool = time.time()

    # fork-based multiprocessing: workers inherit _WORKER_STATE from parent.
    results_by_tag = {}
    with mp.Pool(POOL_SIZE) as pool:
        for k, res in enumerate(pool.imap_unordered(worker_eval, jobs)):
            results_by_tag[res['tag']] = res
            print(f"  [{k+1}/{n_jobs}] {res['tag']}  "
                  f"hifi_mse={res['hifi_mse']:.6f}  "
                  f"wall={res['wall_seconds']:.1f}s")

    print(f"  pool done in {time.time()-t_pool:.1f}s")

    # 3. Materialise ordered results list matching job order.
    results = []
    for j in jobs:
        r = results_by_tag[j['tag']]
        entry = {
            'tag': j['tag'],
            'kind': j['kind'],
            'seed': j['seed'],
            'q0_wxyz': list(j['q0_wxyz']),
            'omega_rad': list(j['omega_rad']),
            'hifi_mse': r['hifi_mse'],
            'wall_seconds': r['wall_seconds'],
            'hifi_mags': r['hifi_mags'].tolist(),
        }
        if j['kind'] == 'polished':
            entry['cand_idx'] = j['cand_idx']
        results.append(entry)

    # 4. Checkpoint: arrays-of-arrays + raw JSON blob.
    hifi_mse_arr = np.array([r['hifi_mse'] for r in results], dtype=np.float64)
    hifi_mags_arr = np.stack([np.asarray(r['hifi_mags']) for r in results])
    tags_arr = np.array([r['tag'] for r in results], dtype='S64')
    kinds_arr = np.array([r['kind'] for r in results], dtype='S32')
    seeds_arr = np.array([r['seed'] for r in results], dtype=np.int64)
    wall_arr = np.array([r['wall_seconds'] for r in results], dtype=np.float64)

    # Strip hifi_mags from JSON blob to keep it lean; keep in array form.
    results_for_json = []
    for r in results:
        rj = {k: v for k, v in r.items() if k != 'hifi_mags'}
        results_for_json.append(rj)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        hifi_mse=hifi_mse_arr,
        hifi_mags=hifi_mags_arr,
        tags=tags_arr,
        kinds=kinds_arr,
        seeds=seeds_arr,
        wall_seconds=wall_arr,
        results_json=np.array(json.dumps(results_for_json, default=_json_default)),
    )
    print(f"  saved {out_npz}")
    print(f"  Stage B done in {time.time()-t0:.1f}s")
    return results


# ---- Stage C: assemble summary ------------------------------------------
def stage_C_summary(candidates, hifi_results):
    print(f"\n[Stage C] assembling summary")
    t0 = time.time()

    # Index results by tag/kind.
    truth_ref_by_seed = {}
    cand_hifi_by_idx = {}
    for r in hifi_results:
        if r['kind'] == 'truth_reference':
            truth_ref_by_seed[r['seed']] = r['hifi_mse']
        elif r['kind'] == 'polished':
            cand_hifi_by_idx[r['cand_idx']] = r

    # Per-candidate agreement.
    per_cand = []
    basin_within = []
    basin_surr_ratios = []
    basin_hifi_ratios = []

    for i, c in enumerate(candidates):
        r = cand_hifi_by_idx[i]
        hifi_after = float(r['hifi_mse'])
        surr_before = float(c['surr_cost_before'])
        surr_after = float(c['surr_cost_after'])
        hifi_before = c['hifi_mse_before']

        # surr_ratio
        if surr_after > 0:
            surr_ratio = surr_before / surr_after
        else:
            surr_ratio = None

        hifi_ratio = None
        ratio_log_diff = None
        within = None
        if hifi_before is not None and hifi_after > 0 and hifi_before > 0 \
                and surr_ratio is not None and surr_ratio > 0:
            hifi_ratio = float(hifi_before) / hifi_after
            if hifi_ratio > 0:
                ratio_log_diff = float(math.log(surr_ratio) - math.log(hifi_ratio))
                within = abs(ratio_log_diff) < LOG_RATIO_THRESHOLD

        entry = {
            'seed': c['seed'],
            'start_label': c['start_label'],
            'is_truth_start': c['is_truth_start'],
            'start_q0_wxyz': c['start_q0_wxyz'],
            'start_omega_rad': c['start_omega_rad'],
            'final_q0_wxyz': c['final_q0_wxyz'],
            'final_omega_rad': c['final_omega_rad'],
            'surr_cost_before': surr_before,
            'surr_cost_after': surr_after,
            'surr_ratio': surr_ratio,
            'hifi_mse_before': hifi_before,
            'hifi_mse_after': hifi_after,
            'hifi_ratio': hifi_ratio,
            'ratio_agreement_log': ratio_log_diff,
            'within_30pct': within,
            'hifi_at_truth': (truth_ref_by_seed.get(c['seed'])
                              if c['is_truth_start'] else None),
            'wall_seconds': r['wall_seconds'],
        }
        per_cand.append(entry)

        if c['start_label'].startswith('basin_'):
            if within is not None:
                basin_within.append(within)
                basin_surr_ratios.append(surr_ratio)
                basin_hifi_ratios.append(hifi_ratio)

    # Aggregate verdict over basin-start candidates.
    n_basin = len(basin_within)
    n_within = int(sum(basin_within))
    frac_within = (n_within / n_basin) if n_basin > 0 else None
    if n_basin == 0:
        verdict = 'INCONCLUSIVE'
        reason = 'no basin candidates had both surr and hifi ratios'
    elif frac_within >= CONFIRM_FRAC:
        verdict = 'CONFIRMED'
        reason = (f'{n_within}/{n_basin} basins within ±30% log-ratio '
                  f'(frac={frac_within:.2f} >= {CONFIRM_FRAC})')
    elif frac_within < REFUTE_FRAC:
        verdict = 'REFUTED'
        reason = (f'{n_within}/{n_basin} basins within ±30% log-ratio '
                  f'(frac={frac_within:.2f} < {REFUTE_FRAC})')
    else:
        verdict = 'PARTIAL'
        reason = (f'{n_within}/{n_basin} basins within ±30% log-ratio '
                  f'(frac={frac_within:.2f} in [{REFUTE_FRAC},{CONFIRM_FRAC}))')

    # Truth-start hi-fi summary.
    truth_cands = [p for p in per_cand if p['is_truth_start']]
    truth_ref_summary = {
        f"seed_{s:03d}": truth_ref_by_seed.get(s) for s in SEEDS}
    truth_polish_summary = {
        f"seed_{p['seed']:03d}": p['hifi_mse_after'] for p in truth_cands}

    # Console report.
    print()
    print(f"m124 -- hi-fi validation of m123 polished candidates")
    n_truth = sum(1 for c in candidates if c['is_truth_start'])
    n_basins = sum(1 for c in candidates
                   if c['start_label'].startswith('basin_'))
    print(f"  loaded {len(candidates)} candidates "
          f"({n_truth} truth + {n_basins} basin) across {len(SEEDS)} seeds")
    print(f"  [Stage B] running {len(hifi_results)} hi-fi evals on "
          f"Pool({POOL_SIZE})...")
    total_wall = sum(r['wall_seconds'] for r in hifi_results)
    print(f"  [{len(hifi_results)}/{len(hifi_results)}] done in {total_wall:.1f}s "
          f"(sum of worker walls)")
    print()
    print(f"  TRUTH-start hi-fi MSE (noise floor ~{NOISE_SIGMA**2:.4f} expected):")
    parts = []
    for s in SEEDS:
        v = truth_ref_by_seed.get(s)
        parts.append(f"seed {s}: {v:.4f}" if v is not None else f"seed {s}: --")
    print(f"    (at exact truth) {'  '.join(parts)}")
    parts2 = []
    for s in SEEDS:
        tc = next((p for p in truth_cands if p['seed'] == s), None)
        v = tc['hifi_mse_after'] if tc else None
        parts2.append(f"seed {s}: {v:.4f}" if v is not None else f"seed {s}: --")
    print(f"    (polished truth) {'  '.join(parts2)}")
    print()
    print(f"  DE-basin polish agreement:")
    print(f"    within ±30%: {n_within}/{n_basin}")
    if basin_surr_ratios:
        srs = [x for x in basin_surr_ratios if x is not None]
        hrs = [x for x in basin_hifi_ratios if x is not None]
        print(f"    surr_ratio range: [{min(srs):.3f}, {max(srs):.3f}]"
              f"  hifi_ratio range: [{min(hrs):.3f}, {max(hrs):.3f}]")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"  reason: {reason}")

    summary = {
        'seeds_run': SEEDS,
        'n_candidates': len(candidates),
        'n_hifi_evals': len(hifi_results),
        'pool_size': POOL_SIZE,
        'log_ratio_threshold': LOG_RATIO_THRESHOLD,
        'confirm_frac': CONFIRM_FRAC,
        'refute_frac': REFUTE_FRAC,
        'truth_reference_hifi_mse_by_seed': truth_ref_summary,
        'truth_polished_hifi_mse_by_seed': truth_polish_summary,
        'aggregate': {
            'n_basin_with_agreement': n_basin,
            'n_within_30pct': n_within,
            'fraction_within': frac_within,
            'verdict': verdict,
            'reason': reason,
        },
        'per_candidate': per_cand,
    }
    atomic_json_save(OUT_DIR / "summary.json", summary)
    print(f"\n  saved {OUT_DIR / 'summary.json'}")
    print(f"  Stage C done in {time.time()-t0:.1f}s")
    return summary


# ---- Main ---------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "run.log"
    log_f = open(log_path, 'w')
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, log_f)

    t_all = time.time()
    try:
        print("=" * 72)
        print(f"m124 -- hi-fi validation of m123 polished candidates")
        print("=" * 72)
        print(f"  seeds={SEEDS}  POOL={POOL_SIZE}  FORCE={FORCE}")
        print(f"  out_dir={OUT_DIR}")

        # Stage A -------------------------------------------------------
        cand_path = OUT_DIR / "candidates.json"
        if cand_path.exists() and not FORCE:
            print(f"\n[Stage A] loading cached {cand_path.name}")
            with open(cand_path) as f:
                candidates = json.load(f)['candidates']
        else:
            print(f"\n[Stage A] building candidate list from m123 + m115")
            candidates = build_candidates()
            n_truth = sum(1 for c in candidates if c['is_truth_start'])
            n_basin = sum(1 for c in candidates
                          if c['start_label'].startswith('basin_'))
            atomic_json_save(cand_path, {
                'seeds': SEEDS,
                'n_total': len(candidates),
                'n_truth': n_truth,
                'n_basin': n_basin,
                'candidates': candidates,
            })
            print(f"  {len(candidates)} candidates ({n_truth} truth + "
                  f"{n_basin} basin) saved -> {cand_path}")

        # Stage B -------------------------------------------------------
        hifi_results = stage_B_hifi(candidates)

        # Stage C -------------------------------------------------------
        stage_C_summary(candidates, hifi_results)

        print(f"\n  Total wall: {time.time()-t_all:.1f}s")
    finally:
        sys.stdout = orig_stdout
        log_f.close()


if __name__ == '__main__':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    main()
