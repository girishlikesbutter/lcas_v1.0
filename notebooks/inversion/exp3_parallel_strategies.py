"""
Exp 3 - Parallel strategy comparison.

Runs multiple optimization strategies simultaneously as separate processes,
each using 1 core. With 32 cores available, we can run many trials in parallel.

Strategies:
1. DE with large population (popsize=50, 20k budget)
2. CMA-ES (naturally good at 6D continuous)
3. Multi-start L-BFGS-B (20 random starts on lo-fi, top 5 → hi-fi)

Each strategy runs as a separate subprocess → true parallelism, no pickle issues.
"""
import sys
from pathlib import Path
import time
import json
import os
import subprocess
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")

# We'll run individual strategy scripts as subprocesses
STRATEGY_SCRIPT = PROJECT_ROOT / "notebooks" / "inversion" / "exp3_single_strategy.py"

strategies = [
    {"name": "de_large", "budget": 20000, "popsize": 50, "seed": 42},
    {"name": "de_large_2", "budget": 20000, "popsize": 50, "seed": 123},
    {"name": "de_large_3", "budget": 20000, "popsize": 50, "seed": 777},
    {"name": "cmaes", "budget": 20000, "seed": 42},
    {"name": "cmaes_2", "budget": 20000, "seed": 123},
    {"name": "cmaes_3", "budget": 20000, "seed": 777},
    {"name": "multistart", "n_starts": 50, "seed": 42},
    {"name": "multistart_2", "n_starts": 50, "seed": 123},
]

print(f"Launching {len(strategies)} strategy runs in parallel...")
print(f"Each uses 1 core, {len(strategies)}/32 cores utilized")

procs = []
for s in strategies:
    args_json = json.dumps(s)
    p = subprocess.Popen(
        [VENV_PYTHON, str(STRATEGY_SCRIPT), args_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    procs.append((s["name"], p))
    print(f"  Launched {s['name']} (PID {p.pid})")

print(f"\nAll launched. Waiting for completion...")
print(f"Timeout: 60 min")

t0 = time.time()
TIMEOUT = 3600  # 60 min

results = {}
for name, p in procs:
    remaining = max(1, TIMEOUT - (time.time() - t0))
    try:
        stdout, _ = p.communicate(timeout=remaining)
        output = stdout.decode('utf-8', errors='replace')
        
        # Parse result from last line (JSON)
        lines = output.strip().split('\n')
        result_line = None
        for line in reversed(lines):
            if line.startswith('RESULT_JSON:'):
                result_line = line[len('RESULT_JSON:'):]
                break
        
        if result_line:
            results[name] = json.loads(result_line)
            r = results[name]
            status = "✓" if r.get('success') else "✗"
            print(f"  {status} {name}: ω_err={r.get('omega_err_deg_s', '?'):.4f}°/s | f={r.get('f_best', '?'):.4f} | {r.get('time_s', '?'):.0f}s")
        else:
            print(f"  ✗ {name}: No result found in output")
            # Print last 5 lines for debugging
            for line in lines[-5:]:
                print(f"    {line}")
    except subprocess.TimeoutExpired:
        p.kill()
        print(f"  ✗ {name}: TIMEOUT")

total_time = time.time() - t0
print(f"\nAll done in {total_time:.0f}s")

# Summary
successes = {k: v for k, v in results.items() if v.get('success')}
if successes:
    best_name = min(successes, key=lambda k: successes[k]['omega_err_deg_s'])
    best = successes[best_name]
    print(f"\n🏆 Best: {best_name}")
    print(f"   ω_err = {best['omega_err_deg_s']:.4f}°/s")
    print(f"   f_best = {best['f_best']:.6f}")
    print(f"   x_best = {best.get('x_best', '?')}")
else:
    print("\n❌ No strategy succeeded.")
    if results:
        best_name = min(results, key=lambda k: results[k].get('omega_err_deg_s', 999))
        best = results[best_name]
        print(f"   Closest: {best_name} with ω_err={best.get('omega_err_deg_s', '?'):.4f}°/s")

# Save all results
all_results = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'total_time_s': total_time,
    'strategies': results,
}
results_path = RESULTS_DIR / 'exp3_parallel_results.json'
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved to {results_path}")
