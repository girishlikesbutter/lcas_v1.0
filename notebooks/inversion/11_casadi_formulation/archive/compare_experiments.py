#!/usr/bin/env python3
"""
Compare m090, m099, m100, m101 results across all 10 seeds.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[3] / "data" / "results" / "inversion_diagnostics"
SEEDS = [0, 6, 12, 14, 24, 27, 33, 36, 74, 93]

# m090 results are hardcoded from EXPERIMENTS.md (different result format)
MICRO90 = {
    0:  {'q0_err': 3.63, 'w0_err': 0.13, 'w_mag_err_pct': 0.20},
    6:  {'q0_err': 178.09, 'w0_err': 3.01, 'w_mag_err_pct': -0.14},
    12: {'q0_err': 111.17, 'w0_err': 89.17, 'w_mag_err_pct': 0.09},
    14: {'q0_err': 178.95, 'w0_err': 1.65, 'w_mag_err_pct': -0.00},
    24: {'q0_err': 179.94, 'w0_err': 0.25, 'w_mag_err_pct': 0.04},
    27: {'q0_err': 176.43, 'w0_err': 36.38, 'w_mag_err_pct': 0.33},
    33: {'q0_err': 134.03, 'w0_err': 17.87, 'w_mag_err_pct': -0.68},
    36: {'q0_err': 173.56, 'w0_err': 3.70, 'w_mag_err_pct': 0.17},
    74: {'q0_err': 5.74, 'w0_err': 4.41, 'w_mag_err_pct': -0.07},
    93: {'q0_err': 179.72, 'w0_err': 0.12, 'w_mag_err_pct': -0.03},
}

EXPERIMENTS = {
    'm090': None,  # hardcoded above
    'm099': RESULTS_DIR / "m099_nm300",
    'm101': RESULTS_DIR / "m101_multi_phi_v2",
}


def classify(q0_err, w0_err):
    is_twin = q0_err > 170
    if w0_err < 5 and (q0_err < 5 or is_twin):
        return "OK"
    elif w0_err < 10 and (q0_err < 10 or is_twin):
        return "PARTIAL"
    return "FAIL"


def load_result(exp_name, seed):
    if exp_name == 'm090':
        return MICRO90.get(seed)
    d = EXPERIMENTS[exp_name]
    rpath = d / f"seed_{seed:03d}" / "result.json"
    if rpath.exists():
        r = json.load(open(rpath))
        return r['winner']
    return None


# Header
exps = list(EXPERIMENTS.keys())
print(f"{'seed':>4}", end="")
for exp in exps:
    print(f" | {'q0':>6} {'w':>5} {'st':>4}", end="")
print(f" | change")
print("-" * (8 + 20 * len(exps) + 10))

tallies = {exp: {'OK': 0, 'PARTIAL': 0, 'FAIL': 0, 'N': 0} for exp in exps}

for seed in SEEDS:
    print(f"{seed:4d}", end="")
    results = {}
    for exp in exps:
        r = load_result(exp, seed)
        if r:
            st = classify(r['q0_err'], r['w0_err'])
            tw = "tw" if r['q0_err'] > 170 else "  "
            print(f" | {r['q0_err']:6.1f} {r['w0_err']:5.1f} {st:>4}{tw}", end="")
            tallies[exp][st] += 1
            tallies[exp]['N'] += 1
            results[exp] = st
        else:
            print(f" | {'':>6} {'':>5} {'':>4}  ", end="")

    # Change column: compare m101 vs m090
    m90 = results.get('m090', '?')
    m101 = results.get('m101', '?')
    if m90 != '?' and m101 != '?':
        if m101 == m90:
            print(f" | =", end="")
        elif m101 == 'OK' and m90 != 'OK':
            print(f" | +FIXED", end="")
        elif m90 == 'OK' and m101 != 'OK':
            print(f" | -REGRESS", end="")
        elif m101 == 'PARTIAL' and m90 == 'FAIL':
            print(f" | +improved", end="")
        elif m90 == 'PARTIAL' and m101 == 'FAIL':
            print(f" | -worse", end="")
        else:
            print(f" | ~", end="")
    else:
        print(f" |", end="")
    print()

print("-" * (8 + 20 * len(exps) + 10))
print(f"{'TALLY':>4}", end="")
for exp in exps:
    t = tallies[exp]
    print(f" | {t['OK']:>2}OK {t['PARTIAL']:>1}P {t['FAIL']:>1}F {'':>4}  ", end="")
print()
