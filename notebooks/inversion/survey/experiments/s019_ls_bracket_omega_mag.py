"""s019 — LS-bracket / multi-hyp / harmonic-division ω-mag coverage on the
100 post-fix m048 trajectories.

Replicates the m138 (buggy-era) bracket / harmonic-division strategies
under correct truth, using cached `traj_seedXXX.npz['mag_hifi']` and
`['observation_times']`. Pure spectral analysis — no rendering, no
propagation, no surrogate. Per workspace contract, the math is
**copy-adapted** from `notebooks/inversion/m138_ls_bracket_probe.py`
(NOT imported).

Strategies (each yields a candidate |ω|-grid):

  • peakcount (m052 baseline) — bright-mask transitions → n_peaks → linear
    base → grid_around(base, [0.3, 3.0], N=20).
  • LS-top1 (m138 control) — top LS peak → grid_around(base, [0.3, 3.0], N=20).
  • bracket — span ALL significant LS peaks at 5% step.
  • multi-hyp — union of grid_around(peak, [0.3, 3.0], N=20) for each significant peak.
  • harmonic-division — top peak f1 + {f1/2, f1/3, f1/4}, each with grid_around([0.3, 3.0], N=20).

Per seed × strategy: in_grid (Y/N), nearest_offset_pct, n_grid_within_5pct.

Pre-registered question:
  Does bracket / harmonic-division cover truth-|ω| at the ±5% bar
  (i.e., at least 1 grid point within 5% of truth) on ≥80/100 seeds?

Pre-registered prediction:
  Yes — m138's 6/6 buggy-era cohort coverage should generalise. The
  zero-classifiable cohort (s018b 19/100) is exactly the constraint-poor
  regime where bracket beat point estimators.

Wall budget: ~30 sec for 100 seeds, single process (no Pool needed).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import lombscargle, find_peaks

PROJECT_ROOT = Path("/home/girish/projects/lcas_v1.0")
SURVEY_DIR = PROJECT_ROOT / "notebooks" / "inversion" / "survey"
sys.path.insert(0, str(SURVEY_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = SURVEY_DIR / "results" / "s019"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FREQS = 4000
THRESHOLD_FRAC = 0.1
MIN_PEAK_DIST = 5
N_MAGS_PER_BASIS = 20
COVERAGE_BAR_PCT = 5.0


def peakcount_estimator(obs_lc, obs_times):
    """m052 baseline: bright-mask transitions → estimate of |ω| (rad/s)."""
    valid = np.isfinite(obs_lc)
    if valid.sum() == 0:
        return np.nan
    bright_mask = valid & (obs_lc < obs_lc[valid].mean() - 1.0)
    transitions = np.diff(bright_mask.astype(int))
    n_peaks = max(1, int((transitions > 0).sum()))
    window = float(obs_times[-1] - obs_times[0])
    if window <= 0:
        return np.nan
    return 2 * np.pi * n_peaks / window


def run_ls(times, signal):
    """Return (freqs_Hz, power) on a uniform grid [1/window, 0.5/dt]."""
    s = -signal
    s = s - np.mean(s)
    dt = float(np.median(np.diff(times)))
    f_min = 1.0 / (times[-1] - times[0])
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, N_FREQS)
    ang = 2 * np.pi * freqs
    power = lombscargle(times, s, ang, normalize=True)
    return freqs, power


def significant_peaks(freqs, power):
    """Peaks with power ≥ 0.1 × max(power); sorted by descending power."""
    pmax = float(power.max())
    if pmax <= 0:
        return [], []
    idx, _ = find_peaks(power, distance=MIN_PEAK_DIST,
                        height=THRESHOLD_FRAC * pmax)
    if len(idx) == 0:
        return [], []
    order = idx[np.argsort(power[idx])[::-1]]
    return freqs[order].tolist(), power[order].tolist()


def grid_around(base_omega, span_low=0.3, span_high=3.0, n_mags=N_MAGS_PER_BASIS):
    return np.geomspace(span_low * base_omega, span_high * base_omega, n_mags)


def coverage_metric(grid, truth_mag):
    """Returns (in_grid, nearest_offset_pct, n_within_5pct)."""
    if len(grid) == 0:
        return False, np.inf, 0
    grid = np.sort(np.asarray(grid))
    in_grid = bool(grid[0] <= truth_mag <= grid[-1])
    pct = float(np.min(np.abs(grid - truth_mag)) / truth_mag * 100)
    n_within = int((np.abs(grid - truth_mag) / truth_mag <= COVERAGE_BAR_PCT / 100).sum())
    return in_grid, pct, n_within


def run_seed(seed):
    traj_path = SURVEY_DIR / "data" / "trajectories" / f"traj_seed{seed:03d}.npz"
    d = np.load(traj_path)
    obs_lc = d['mag_hifi'].astype(float)
    obs_times = d['observation_times'].astype(float)
    truth_omega = d['omega0_rad'].astype(float)
    truth_mag = float(np.linalg.norm(truth_omega))     # rad/s

    finite = np.isfinite(obs_lc)
    valid_lc = obs_lc[finite]
    valid_t = obs_times[finite]
    if len(valid_lc) < 50:
        return None

    base_pc = peakcount_estimator(obs_lc, obs_times)
    freqs, power = run_ls(valid_t, valid_lc)
    pf, pw = significant_peaks(freqs, power)
    if not pf:
        return {
            'seed': seed, 'truth_mag': truth_mag,
            'n_significant_peaks': 0,
            'no_significant_peaks': True,
        }
    peak_omegas = sorted(2 * np.pi * np.array(pf))
    f1 = peak_omegas[-1]            # highest-power peak (sorted asc by ω)
    # Note: pf is sorted by descending POWER; peak_omegas is sorted asc by ω.
    # m138 used f1 = peak_omegas[-1] (highest-ω) — keep for parity.

    # Strategies
    grid_pc = grid_around(base_pc) if np.isfinite(base_pc) and base_pc > 0 else np.array([])
    cov_pc = coverage_metric(grid_pc, truth_mag)

    base_ls = f1
    grid_ls = grid_around(base_ls)
    cov_ls = coverage_metric(grid_ls, truth_mag)

    lo = 0.5 * peak_omegas[0]
    hi = 2.0 * peak_omegas[-1]
    n_bracket = max(N_MAGS_PER_BASIS,
                    int(np.ceil(np.log(hi / max(lo, 1e-12)) / np.log(1.05))))
    grid_bracket = np.geomspace(lo, hi, n_bracket)
    cov_bracket = coverage_metric(grid_bracket, truth_mag)

    grids_multi = [grid_around(po) for po in peak_omegas]
    grid_multi = np.unique(np.concatenate(grids_multi))
    cov_multi = coverage_metric(grid_multi, truth_mag)

    bases_harm = [f1, f1 / 2.0, f1 / 3.0, f1 / 4.0]
    grids_harm = [grid_around(b) for b in bases_harm]
    grid_harm = np.unique(np.concatenate(grids_harm))
    cov_harm = coverage_metric(grid_harm, truth_mag)

    return {
        'seed': seed,
        'truth_mag_rad_s': truth_mag,
        'truth_mag_dps': float(np.degrees(truth_mag)),
        'n_significant_peaks': len(peak_omegas),
        'pc_base': float(base_pc),
        'pc_in_grid': cov_pc[0], 'pc_nearest_pct': cov_pc[1], 'pc_n_within_5pct': cov_pc[2],
        'ls_top1_base': float(base_ls),
        'ls_in_grid': cov_ls[0], 'ls_nearest_pct': cov_ls[1], 'ls_n_within_5pct': cov_ls[2],
        'bracket_lo': float(lo), 'bracket_hi': float(hi), 'bracket_grid_size': int(n_bracket),
        'bracket_in_grid': cov_bracket[0], 'bracket_nearest_pct': cov_bracket[1],
        'bracket_n_within_5pct': cov_bracket[2],
        'multi_grid_size': int(len(grid_multi)),
        'multi_in_grid': cov_multi[0], 'multi_nearest_pct': cov_multi[1],
        'multi_n_within_5pct': cov_multi[2],
        'harm_grid_size': int(len(grid_harm)),
        'harm_in_grid': cov_harm[0], 'harm_nearest_pct': cov_harm[1],
        'harm_n_within_5pct': cov_harm[2],
    }


def main():
    print("=== s019 — LS-bracket / harmonic-division ω-mag coverage on 100 seeds ===",
          flush=True)
    t0 = time.time()
    results = {}
    no_sig_peaks = []
    for seed in range(100):
        r = run_seed(seed)
        if r is None:
            continue
        if r.get('no_significant_peaks'):
            no_sig_peaks.append(seed)
        results[seed] = r
    wall = time.time() - t0
    print(f"Wall: {wall:.1f}s for {len(results)} seeds", flush=True)
    if no_sig_peaks:
        print(f"Seeds with no significant LS peaks: {no_sig_peaks}", flush=True)

    # Aggregate coverage
    strats = ['pc', 'ls', 'bracket', 'multi', 'harm']
    cohort_cov = {s: {'in_grid': 0, 'within_5pct': 0,
                      'nearest_offsets': []} for s in strats}
    for r in results.values():
        if r.get('no_significant_peaks'):
            continue
        for s in strats:
            if r[f'{s}_in_grid']:
                cohort_cov[s]['in_grid'] += 1
            if r[f'{s}_n_within_5pct'] >= 1:
                cohort_cov[s]['within_5pct'] += 1
            cohort_cov[s]['nearest_offsets'].append(r[f'{s}_nearest_pct'])

    # Print summary
    print(f"\n=== Cohort coverage (n={len(results) - len(no_sig_peaks)} seeds with significant peaks) ===",
          flush=True)
    print(f"  strategy   | in_grid | within_5pct | offset_p10 | offset_med | offset_p90 | grid_size",
          flush=True)
    print(f"  -----------+---------+-------------+------------+------------+------------+----------",
          flush=True)
    sizes = {
        'pc':       N_MAGS_PER_BASIS,
        'ls':       N_MAGS_PER_BASIS,
        'bracket':  '~50-100',
        'multi':    'variable (depends on K peaks)',
        'harm':     '~70 (4 bases × 20 with overlap collapse)',
    }
    for s in strats:
        c = cohort_cov[s]
        offs = np.array(c['nearest_offsets'])
        p10, med, p90 = np.percentile(offs, [10, 50, 90])
        print(f"  {s:<10s} | {c['in_grid']:>4d}/{len(offs)}  | "
              f"{c['within_5pct']:>4d}/{len(offs)}    | "
              f"{p10:>9.2f}% | {med:>9.2f}% | {p90:>9.2f}% | {sizes[s]}",
              flush=True)

    # Save
    with open(OUT_DIR / "summary.json", 'w') as f:
        json.dump({
            'config': {
                'N_FREQS': N_FREQS, 'THRESHOLD_FRAC': THRESHOLD_FRAC,
                'MIN_PEAK_DIST': MIN_PEAK_DIST,
                'N_MAGS_PER_BASIS': N_MAGS_PER_BASIS,
                'COVERAGE_BAR_PCT': COVERAGE_BAR_PCT,
            },
            'cohort_coverage': {
                s: {
                    'in_grid': cohort_cov[s]['in_grid'],
                    'within_5pct': cohort_cov[s]['within_5pct'],
                    'n_eval': len(cohort_cov[s]['nearest_offsets']),
                    'nearest_offset_p10': float(np.percentile(
                        cohort_cov[s]['nearest_offsets'], 10)),
                    'nearest_offset_p50': float(np.percentile(
                        cohort_cov[s]['nearest_offsets'], 50)),
                    'nearest_offset_p90': float(np.percentile(
                        cohort_cov[s]['nearest_offsets'], 90)),
                } for s in strats
            },
            'no_significant_peaks_seeds': no_sig_peaks,
            'per_seed': results,
        }, f, indent=2, default=float)
    print(f"\nSaved: {OUT_DIR/'summary.json'}", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = []
    for s in strats:
        bars.append((s, cohort_cov[s]['within_5pct'],
                    len(cohort_cov[s]['nearest_offsets'])))
    x = range(len(bars))
    cnt = [b[1] for b in bars]
    pct = [100 * b[1] / max(b[2], 1) for b in bars]
    ax.bar(x, pct, color=['tab:gray', 'tab:gray', 'tab:blue',
                          'tab:orange', 'tab:green'])
    for i, p in enumerate(pct):
        ax.text(i, p + 1, f"{cnt[i]}/{bars[i][2]}\n({p:.0f}%)",
                ha='center', va='bottom', fontsize=9)
    ax.axhline(80.0, color='tab:red', linestyle='--', alpha=0.5,
               label='pre-registered bar 80%')
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("% seeds with at least one grid point within 5% of truth |ω|")
    ax.set_ylim(0, 110)
    ax.set_title("s019 ω-mag coverage on 100 post-fix m048 truth NPZs")
    ax.legend()
    fig.tight_layout()
    out_png = OUT_DIR / "coverage_within_5pct.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_png}", flush=True)


if __name__ == "__main__":
    main()
