#!/usr/bin/env python3
"""
m096 — Deep analysis of Stage 1 checkpoints across 100 seeds.

Load every seed's checkpoint and look for patterns, correlations,
and the actual handholds we can grab onto.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("data/results/inversion_diagnostics")
STAGE1_DIR = RESULTS_DIR / "m096_stage1"

summary = json.load(open(STAGE1_DIR / "stage1_summary.json"))
valid = [r for r in summary if 'error' not in r]
invalid = [r for r in summary if 'error' in r]

print("=" * 70)
print(f"DEEP ANALYSIS — 100 seeds ({len(valid)} valid, {len(invalid)} invalid)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# 1. THE INVALID SEEDS — what kills them?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("1. INVALID SEEDS (< 2 specular peaks)")
print(f"{'='*70}")

for r in invalid:
    seed = r['seed']
    d = np.load(str(STAGE1_DIR / f"seed_{seed:03d}.npz"), allow_pickle=True)
    peaks = d['peaks_idx']
    peak_mags = d['peak_mags']
    spec = d['spec_peaks']
    wmag = r['omega_true_dps']
    wdir = d['omega_body_dots']
    dom = ['X', 'Y', 'Z'][np.argmax(wdir)]

    # What peaks DO exist?
    mag_str = ', '.join(f'{m:.1f}' for m in sorted(peak_mags)) if len(peak_mags) > 0 else 'none'
    print(f"  seed {seed:3d}: |w|={wmag:.3f} dps, {len(peaks)} peaks (mags: {mag_str}), "
          f"{len(spec)} spec, dom={dom}")

print(f"\n  All invalid seeds have |w| < 0.40 dps (slow tumblers).")
print(f"  Slow tumblers produce fewer peaks → fewer chances for specular alignment.")


# ══════════════════════════════════════════════════════════════════════
# 2. OMEGA MAGNITUDE vs CONSTRAINT COUNT
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("2. OMEGA MAGNITUDE vs CONSTRAINT COUNT")
print(f"{'='*70}")

# Bin by omega magnitude
mag_bins = [(0, 0.3, 'very slow'), (0.3, 0.6, 'slow'),
            (0.6, 0.9, 'medium'), (0.9, 1.2, 'fast'), (1.2, 1.6, 'very fast')]

for lo, hi, label in mag_bins:
    seeds_in_bin = [r for r in summary if lo <= r['omega_true_dps'] < hi]
    n_total = len(seeds_in_bin)
    n_invalid = sum(1 for r in seeds_in_bin if 'error' in r)
    valid_in_bin = [r for r in seeds_in_bin if 'error' not in r]

    if valid_in_bin:
        specs = [r['n_spec'] for r in valid_in_bin]
        csts = [r['n_constraints'] for r in valid_in_bin]
        brights = [r['n_bright'] for r in valid_in_bin]
        print(f"  {label:10s} ({lo:.1f}-{hi:.1f} dps): {n_total:2d} seeds, "
              f"{n_invalid} invalid | "
              f"spec: {np.median(specs):.0f} [{min(specs)}-{max(specs)}] | "
              f"cstr: {np.median(csts):.0f} [{min(csts)}-{max(csts)}] | "
              f"bright: {np.median(brights):.0f} [{min(brights)}-{max(brights)}]")
    else:
        print(f"  {label:10s} ({lo:.1f}-{hi:.1f} dps): {n_total:2d} seeds, "
              f"ALL invalid")


# ══════════════════════════════════════════════════════════════════════
# 3. DOMINANT OMEGA AXIS vs CONSTRAINT QUALITY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("3. DOMINANT OMEGA AXIS vs CONSTRAINT QUALITY")
print(f"{'='*70}")

for axis in ['X', 'Y', 'Z']:
    seeds_ax = [r for r in valid if r['dominant_omega_axis'] == axis]
    if not seeds_ax:
        continue
    specs = [r['n_spec'] for r in seeds_ax]
    brights = [r['n_bright'] for r in seeds_ax]
    mags_est_err = [abs(r['omega_est_err_pct']) for r in seeds_ax]

    print(f"  {axis}-dominant: {len(seeds_ax)} seeds | "
          f"spec: {np.median(specs):.0f} [{min(specs)}-{max(specs)}] | "
          f"bright: {np.median(brights):.0f} [{min(brights)}-{max(brights)}] | "
          f"|w| err: {np.median(mags_est_err):.1f}%")


# ══════════════════════════════════════════════════════════════════════
# 4. PEAK MAGNITUDE DISTRIBUTION — where is the information?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("4. PEAK MAGNITUDE DISTRIBUTION ACROSS ALL SEEDS")
print(f"{'='*70}")

all_peak_mags = []
all_spec_mags = []
for seed in range(100):
    d = np.load(str(STAGE1_DIR / f"seed_{seed:03d}.npz"), allow_pickle=True)
    pm = d['peak_mags']
    all_peak_mags.extend(pm.tolist())
    spec = d['spec_peaks']
    obs = d['observed_lc']
    if len(spec) > 0:
        all_spec_mags.extend(obs[spec].tolist())

all_peak_mags = np.array(all_peak_mags)
all_spec_mags = np.array(all_spec_mags)

mag_edges = [4, 5, 5.9, 6.3, 7.3, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0]
print(f"\n  All peaks (N={len(all_peak_mags)}):")
for i in range(len(mag_edges)-1):
    n = np.sum((all_peak_mags >= mag_edges[i]) & (all_peak_mags < mag_edges[i+1]))
    pct = n / len(all_peak_mags) * 100
    label = ""
    if mag_edges[i+1] <= 5.9: label = " [±X only]"
    elif mag_edges[i+1] <= 6.3: label = " [±X, ±Z]"
    elif mag_edges[i+1] <= 7.3: label = " [bus faces]"
    elif mag_edges[i+1] <= 9.0: label = " [all 10 — CURRENT CUTOFF]"
    else: label = " [currently UNUSED]"
    bar = "#" * int(pct)
    print(f"    {mag_edges[i]:5.1f}-{mag_edges[i+1]:5.1f}: {n:5d} ({pct:5.1f}%) {bar}{label}")

print(f"\n  Specular peaks only (mag < 9, N={len(all_spec_mags)}):")
for i in range(len(mag_edges)-1):
    if mag_edges[i] >= 9.0:
        break
    n = np.sum((all_spec_mags >= mag_edges[i]) & (all_spec_mags < mag_edges[i+1]))
    pct = n / len(all_spec_mags) * 100
    print(f"    {mag_edges[i]:5.1f}-{mag_edges[i+1]:5.1f}: {n:5d} ({pct:5.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# 5. CONSTRAINT ALLOWED NORMALS — what do we actually have?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("5. CONSTRAINT NORMAL AMBIGUITY DISTRIBUTION")
print(f"{'='*70}")

total_constraints = 0
constraints_by_n_allowed = Counter()
for r in valid:
    seed = r['seed']
    d = np.load(str(STAGE1_DIR / f"seed_{seed:03d}.npz"), allow_pickle=True)
    counts = d['constraint_allowed_counts']
    for c in counts:
        constraints_by_n_allowed[int(c)] += 1
        total_constraints += 1

print(f"  Total constraints across {len(valid)} seeds: {total_constraints}")
print(f"  By number of allowed normals:")
for n_allowed in sorted(constraints_by_n_allowed.keys()):
    count = constraints_by_n_allowed[n_allowed]
    pct = count / total_constraints * 100
    bar = "#" * int(pct)
    if n_allowed == 2: label = "±X only — strongest constraint"
    elif n_allowed == 4: label = "±X, ±Z"
    elif n_allowed == 6: label = "bus faces"
    elif n_allowed == 10: label = "all normals — weakest"
    else: label = ""
    print(f"    {n_allowed:2d} normals: {count:4d} ({pct:5.1f}%) {bar}  {label}")


# ══════════════════════════════════════════════════════════════════════
# 6. ANCHOR MAGNITUDE — what are we anchoring on?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("6. ANCHOR MAGNITUDE DISTRIBUTION")
print(f"{'='*70}")

anchor_mags = [r['anchor_mag'] for r in valid]
print(f"  min={min(anchor_mags):.2f}, median={np.median(anchor_mags):.2f}, max={max(anchor_mags):.2f}")
print(f"  Bright (< 5.9, ±X only): {sum(1 for m in anchor_mags if m < 5.9)}/{len(valid)}")
print(f"  Medium (5.9-7.3, bus):    {sum(1 for m in anchor_mags if 5.9 <= m < 7.3)}/{len(valid)}")
print(f"  Dim (≥ 7.3, all normal):  {sum(1 for m in anchor_mags if m >= 7.3)}/{len(valid)}")

# Anchor tie-breaking
n_tied = sum(1 for r in valid if r.get('anchor_tie_broken', False))
print(f"  Anchor ties (top-2 within 0.05 mag): {n_tied}/{len(valid)}")


# ══════════════════════════════════════════════════════════════════════
# 7. OMEGA ESTIMATE ERROR — what's driving the big errors?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("7. OMEGA ESTIMATE ERROR ANALYSIS")
print(f"{'='*70}")

for r in sorted(valid, key=lambda x: abs(x['omega_est_err_pct']), reverse=True)[:15]:
    seed = r['seed']
    est = r['omega_est_dps']
    true = r['omega_true_dps']
    err = r['omega_est_err_pct']
    n_peaks = r['n_peaks']
    dom = r['dominant_omega_axis']

    # Check if true |w| is in the grid range (±20% of estimate)
    lo = est * 0.7
    hi = est * 1.3
    in_range = "YES" if lo <= true <= hi else "NO"

    print(f"  seed {seed:3d}: est={est:.3f} true={true:.3f} err={err:+.1f}% "
          f"peaks={n_peaks:2d} dom={dom} in_grid={in_range}")

in_range_count = sum(1 for r in valid
                     if r['omega_est_dps'] * 0.7 <= r['omega_true_dps'] <= r['omega_est_dps'] * 1.3)
print(f"\n  True |w| within ±30% grid range: {in_range_count}/{len(valid)}")

in_range_20 = sum(1 for r in valid
                  if r['omega_est_dps'] * 0.8 <= r['omega_true_dps'] <= r['omega_est_dps'] * 1.2)
print(f"  True |w| within ±20% grid range: {in_range_20}/{len(valid)}")


# ══════════════════════════════════════════════════════════════════════
# 8. TIME SPREAD — how spread out are constraints?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("8. CONSTRAINT TIME SPREAD")
print(f"{'='*70}")

dt_ranges = [r['dt_range_s'] for r in valid]
print(f"  dt_range: min={min(dt_ranges):.0f}s, median={np.median(dt_ranges):.0f}s, "
      f"max={max(dt_ranges):.0f}s")

# Seeds with very narrow time spread (all constraints near anchor)
narrow = [r for r in valid if r['dt_range_s'] < 1000]
if narrow:
    print(f"  Narrow spread (<1000s): {len(narrow)} seeds")
    for r in narrow:
        print(f"    seed {r['seed']:3d}: dt_range={r['dt_range_s']:.0f}s, "
              f"cstr={r['n_constraints']}")


# ══════════════════════════════════════════════════════════════════════
# 9. NON-SPECULAR PEAKS — untapped information?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("9. NON-SPECULAR PEAKS (mag 9-14) — UNUSED INFORMATION")
print(f"{'='*70}")

for r in valid:
    seed = r['seed']
    d = np.load(str(STAGE1_DIR / f"seed_{seed:03d}.npz"), allow_pickle=True)
    peaks = d['peaks_idx']
    peak_mags = d['peak_mags']
    n_nonspec = np.sum(peak_mags >= 9.0)
    n_dim_useful = np.sum((peak_mags >= 9.0) & (peak_mags < 12.0))
    r['n_nonspec'] = int(n_nonspec)
    r['n_dim_useful'] = int(n_dim_useful)

nonspec = [r['n_nonspec'] for r in valid]
dim_useful = [r['n_dim_useful'] for r in valid]
print(f"  Non-specular peaks (mag ≥ 9): median={int(np.median(nonspec))}, "
      f"mean={np.mean(nonspec):.1f}, max={max(nonspec)}")
print(f"  Dim-but-detectable (mag 9-12): median={int(np.median(dim_useful))}, "
      f"mean={np.mean(dim_useful):.1f}, max={max(dim_useful)}")

# Seeds where non-spec peaks vastly outnumber spec peaks
print(f"\n  Seeds with many more non-spec than spec peaks:")
for r in sorted(valid, key=lambda x: x['n_nonspec'] - x['n_spec'], reverse=True)[:10]:
    print(f"    seed {r['seed']:3d}: {r['n_spec']} spec, {r['n_nonspec']} non-spec, "
          f"{r['n_dim_useful']} dim-useful")


# ══════════════════════════════════════════════════════════════════════
# 10. CORRELATIONS — what predicts success?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("10. FEATURE CORRELATIONS")
print(f"{'='*70}")

# Build feature matrix
features = []
for r in valid:
    features.append([
        r['omega_true_dps'],
        r['n_spec'],
        r['n_constraints'],
        r['n_bright'],
        abs(r['omega_est_err_pct']),
        r['pab_min_dot'],
        r['dt_range_s'],
        r.get('n_nonspec', 0),
    ])
features = np.array(features)
names = ['|w|', 'n_spec', 'n_cstr', 'n_bright', '|w|_err%', 'PAB_min', 'dt_range', 'n_nonspec']

# Correlation with n_spec (most important feature for pipeline viability)
print(f"\n  Correlation with n_spec (specular peak count):")
for i, name in enumerate(names):
    if name == 'n_spec':
        continue
    corr = np.corrcoef(features[:, i], features[:, 1])[0, 1]
    print(f"    {name:12s}: r = {corr:+.3f}")

# Correlation with n_bright
print(f"\n  Correlation with n_bright:")
for i, name in enumerate(names):
    if name == 'n_bright':
        continue
    corr = np.corrcoef(features[:, i], features[:, 3])[0, 1]
    print(f"    {name:12s}: r = {corr:+.3f}")


# ══════════════════════════════════════════════════════════════════════
# 11. POPULATION SEGMENTATION — natural groups?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("11. POPULATION SEGMENTATION")
print(f"{'='*70}")

# Segment by what the pipeline can actually work with
tiers = {
    'A: Rich (≥6 cstr, ≥2 bright)': [],
    'B: Moderate (≥4 cstr, ≥1 bright)': [],
    'C: Sparse (≥2 cstr, 0 bright)': [],
    'D: Minimal (1 cstr)': [],
    'X: Invalid (<2 spec)': [],
}

for r in summary:
    if 'error' in r:
        tiers['X: Invalid (<2 spec)'].append(r['seed'])
    elif r['n_constraints'] >= 6 and r['n_bright'] >= 2:
        tiers['A: Rich (≥6 cstr, ≥2 bright)'].append(r['seed'])
    elif r['n_constraints'] >= 4 and r['n_bright'] >= 1:
        tiers['B: Moderate (≥4 cstr, ≥1 bright)'].append(r['seed'])
    elif r['n_constraints'] >= 2:
        tiers['C: Sparse (≥2 cstr, 0 bright)'].append(r['seed'])
    else:
        tiers['D: Minimal (1 cstr)'].append(r['seed'])

for tier, seeds in tiers.items():
    print(f"  {tier}: {len(seeds)} seeds")
    if len(seeds) <= 20:
        print(f"    Seeds: {seeds}")

# For each tier, what's the |w| distribution?
print(f"\n  |w| by tier:")
for tier, seeds in tiers.items():
    if not seeds:
        continue
    wmags = [r['omega_true_dps'] for r in summary if r['seed'] in seeds]
    print(f"    {tier[:20]:20s}: {np.median(wmags):.3f} dps "
          f"[{min(wmags):.3f} - {max(wmags):.3f}]")


# ══════════════════════════════════════════════════════════════════════
# 12. KEY QUESTION: What if we lower the specular threshold?
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("12. EFFECT OF SPECULAR THRESHOLD ON CONSTRAINT COUNT")
print(f"{'='*70}")

for threshold in [9.0, 10.0, 11.0, 12.0]:
    counts = []
    n_viable = 0
    for seed in range(100):
        d = np.load(str(STAGE1_DIR / f"seed_{seed:03d}.npz"), allow_pickle=True)
        peaks = d['peaks_idx']
        peak_mags = d['peak_mags']
        n_under = int(np.sum(peak_mags < threshold))
        counts.append(n_under)
        if n_under >= 2:
            n_viable += 1
    counts = np.array(counts)
    print(f"  Threshold {threshold:5.1f}: viable={n_viable}/100, "
          f"median peaks={int(np.median(counts))}, "
          f"mean={np.mean(counts):.1f}")


print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
