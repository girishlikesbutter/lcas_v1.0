#!/usr/bin/env python3
"""
m099: Batch runner for NM_TOP=300 pipeline with 2000-dir grid.

Runs m095e (modified with NM_TOP=300) on multiple seeds sequentially.
Each seed saves results independently. Produces batch summary.

Usage:
  python3 m099_batch.py           # default 10 seeds
  python3 m099_batch.py 0 27 93   # specific seeds
"""

import sys, os, time, json, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
BATCH_DIR = RESULTS_DIR / "m099_nm300"
BATCH_DIR.mkdir(exist_ok=True)

DEFAULT_SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]
SEEDS = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_SEEDS

SCRIPT = str(PROJECT_ROOT / "notebooks" / "inversion" / "11_casadi_formulation" / "m095e_integrated.py")

print("=" * 70)
print(f"m099: NM_TOP=300 batch runner ({len(SEEDS)} seeds)")
print(f"  Seeds: {SEEDS}")
print("=" * 70, flush=True)

batch_results = []
t_total = time.time()

for seed in SEEDS:
    # Check for existing result from a prior run
    ckpt = BATCH_DIR / f"seed_{seed:03d}"
    result_path = ckpt / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            r = json.load(f)
        w = r['winner']
        is_twin = w['q0_err'] > 170
        st = "OK" if w['w0_err'] < 5 and (w['q0_err'] < 5 or is_twin) else \
             "PARTIAL" if w['w0_err'] < 10 else "FAIL"
        print(f"\n  seed {seed:3d}: CACHED q0={w['q0_err']:.1f}° w_dir={w['w0_err']:.1f}° "
              f"[{st}]", flush=True)
        batch_results.append(r)
        continue

    print(f"\n  seed {seed:3d}: RUNNING...", flush=True)
    t0 = time.time()

    env = os.environ.copy()
    env['MICRO95E_SEED'] = str(seed)
    env['MICRO95E_NM_TOP'] = '300'
    env['MICRO95E_LOFI_TOP'] = '300'
    env['MICRO95E_CKPT_DIR'] = str(ckpt)
    env['PYTHONUNBUFFERED'] = '1'

    proc = subprocess.run(
        [sys.executable, SCRIPT],
        env=env, capture_output=True, text=True, timeout=1800)

    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    FAILED ({elapsed:.0f}s) — {proc.stderr[-200:]}", flush=True)
        batch_results.append({'traj_seed': seed, 'error': 'subprocess_failed'})
        continue

    # Read result — m095e saves to CKPT_DIR set via env var
    r_path = result_path  # ckpt / "result.json"
    if r_path.exists():
        with open(r_path) as f:
            r = json.load(f)
        batch_results.append(r)
        w = r['winner']
        is_twin = w['q0_err'] > 170
        st = "OK" if w['w0_err'] < 5 and (w['q0_err'] < 5 or is_twin) else \
             "PARTIAL" if w['w0_err'] < 10 else "FAIL"
        tw = " (+X twin)" if is_twin else ""
        print(f"    q0={w['q0_err']:.1f}° w_dir={w['w0_err']:.1f}° "
              f"w_mag={w['w_mag_err_pct']:+.1f}% [{st}]{tw} {elapsed:.0f}s", flush=True)
    else:
        print(f"    NO RESULT FILE ({elapsed:.0f}s)", flush=True)
        batch_results.append({'traj_seed': seed, 'error': 'no_result_file'})

total_time = time.time() - t_total

# Summary
print(f"\n{'='*70}")
print(f"BATCH SUMMARY ({total_time:.0f}s = {total_time/60:.1f} min)")
print(f"{'='*70}")
print(f"\n  {'seed':>4s} {'q0':>7s} {'w_dir':>6s} {'w_mag':>7s} {'status':>8s}")
print(f"  {'----':>4s} {'----':>7s} {'-----':>6s} {'-----':>7s} {'------':>8s}")
for r in batch_results:
    if 'error' in r:
        print(f"  {r['traj_seed']:4d}   ERROR: {r['error']}")
        continue
    w = r['winner']
    q0e, wde, wme = w['q0_err'], w['w0_err'], w['w_mag_err_pct']
    is_tw = q0e > 170
    st = "OK" if wde < 5 and (q0e < 5 or is_tw) else "PARTIAL" if wde < 10 else "FAIL"
    print(f"  {r['traj_seed']:4d} {q0e:7.1f} {wde:6.1f} {wme:+7.2f}% {st:>8s}")

ok = sum(1 for r in batch_results if 'winner' in r and r['winner']['w0_err'] < 5
         and (r['winner']['q0_err'] < 5 or r['winner']['q0_err'] > 170))
print(f"\n  OK: {ok}/{len(batch_results)}")

with open(str(BATCH_DIR / "batch_summary.json"), 'w') as f:
    json.dump({'seeds': SEEDS, 'nm_top': 300, 'results': batch_results,
               'total_time_s': total_time}, f, indent=2)
print(f"\nSaved: {BATCH_DIR}/")
