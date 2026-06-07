#!/usr/bin/env python3
"""
m100 re-scoring: Compare selection methods on m100 results.

Methods:
  1. Majority vote (current m100/m099 method)
  2. Full-window MSE only
  3. Shortest-window (180s) MSE only
  4. Geometric mean across windows

Reads result.json from all available m100 seeds.
No computation — just re-ranks existing hi-fi results.
"""

import json, sys
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path(__file__).resolve().parents[3] / "data" / "results" / "inversion_diagnostics"
M100_DIR = RESULTS_DIR / "m100_multi_phi"
M101_DIR = RESULTS_DIR / "m101_multi_phi_v2"
M99_DIR = RESULTS_DIR / "m099_nm300"
M90_DIR = RESULTS_DIR  # m090 results are in per-seed m077_beta_seed* dirs

SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]


def classify(q0_err, w0_err):
    is_twin = q0_err > 170
    if w0_err < 5 and (q0_err < 5 or is_twin):
        return "OK"
    elif w0_err < 10 and (q0_err < 10 or is_twin):
        return "PARTIAL"
    return "FAIL"


def select_by_method(candidates, method):
    """Return the best candidate according to the given method."""
    if method == 'vote':
        votes = Counter()
        for wk in [180, 360, 720, 'full']:
            wk_str = str(wk)
            ranked = sorted(candidates, key=lambda c: c.get(wk_str, c.get(wk, 999)))
            votes[ranked[0]['omega_rank']] += 1
        best_rank = votes.most_common(1)[0][0]
        # Tie-break
        top_two = votes.most_common(2)
        if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
            ranked = sorted(candidates, key=lambda c: c.get('full', 999))
            return ranked[0]
        return [c for c in candidates if c['omega_rank'] == best_rank][0]
    elif method == 'full':
        return min(candidates, key=lambda c: c.get('full', 999))
    elif method == '180s':
        return min(candidates, key=lambda c: c.get('180', c.get(180, 999)))
    elif method == 'geomean':
        import math
        def gm(c):
            vals = [c.get(str(w), c.get(w, 999)) for w in [180, 360, 720]]
            vals.append(c.get('full', 999))
            return math.exp(sum(math.log(max(v, 1e-10)) for v in vals) / len(vals))
        return min(candidates, key=gm)


print(f"{'':>4s} | {'--- Vote ---':>23s} | {'--- Full MSE ---':>23s} | {'--- 180s MSE ---':>23s} | {'--- GeoMean ---':>23s}")
print(f"{'seed':>4s} | {'q0':>6s} {'w':>5s} {'st':>4s} | {'q0':>6s} {'w':>5s} {'st':>4s} | {'q0':>6s} {'w':>5s} {'st':>4s} | {'q0':>6s} {'w':>5s} {'st':>4s}")
print("-" * 110)

tallies = {m: {'OK': 0, 'PARTIAL': 0, 'FAIL': 0} for m in ['vote', 'full', '180s', 'geomean']}
available = 0

for seed in SEEDS:
    rpath = M100_DIR / f"seed_{seed:03d}" / "result.json"
    if not rpath.exists():
        print(f"{seed:4d} | {'':>23s} | {'':>23s} | {'':>23s} | (not available)")
        continue
    available += 1

    r = json.load(open(rpath))
    cands = r['all_candidates']

    row = f"{seed:4d} |"
    for method in ['vote', 'full', '180s', 'geomean']:
        winner = select_by_method(cands, method)
        st = classify(winner['q0_err'], winner['w0_err'])
        tallies[method][st] += 1
        tw = "tw" if winner['q0_err'] > 170 else "  "
        row += f" {winner['q0_err']:6.1f} {winner['w0_err']:5.1f} {st:>4s}{tw} |"
    print(row)

print("-" * 110)
print(f"{'':>4s} |", end="")
for method in ['vote', 'full', '180s', 'geomean']:
    t = tallies[method]
    print(f" {t['OK']:>2d}OK {t['PARTIAL']:>1d}P {t['FAIL']:>1d}F {'':>7s} |", end="")
print()

# Also show m099 for comparison
print(f"\n--- m099 (NM_TOP=300, single phi, vote) for comparison ---")
for seed in SEEDS:
    rpath = M99_DIR / f"seed_{seed:03d}" / "result.json"
    if rpath.exists():
        r = json.load(open(rpath))
        w = r['winner']
        st = classify(w['q0_err'], w['w0_err'])
        tw = "+Xtw" if w['q0_err'] > 170 else ""
        print(f"  seed {seed:3d}: q0={w['q0_err']:6.1f}° w={w['w0_err']:5.1f}° [{st}] {tw}")
