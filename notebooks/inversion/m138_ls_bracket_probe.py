"""m138 LS-as-bracket / LS-as-multi-hypothesis probe.

Two new strategies (different from m052's point-estimator):

  STRATEGY 1 (BRACKET): take all LS peaks above power threshold P*. Build
    |ω|-grid spanning [0.5 × min_peak_omega, 2.0 × max_peak_omega]. Tests
    whether truth lies inside the bracket and how many grid points/density
    needed to hit truth within 5%.

  STRATEGY 2 (MULTI-HYP): treat each significant LS peak as a candidate
    base |ω|. Build the SAME [0.3, 3.0] × base grid around EACH peak.
    Total grid = union of K such grids. Tests whether truth gets the
    [0.3-3.0] coverage when one of the K bases is correct. The K is
    typically 2-5, so total grid size is K × n_mags.

  STRATEGY 3 (HARMONIC-DIVISION): for the dominant LS peak f1, also test
    f1/2, f1/3, f1/4 as candidate base frequencies. Captures the "k-th
    harmonic dominates" regime explicitly.

For each seed × strategy:
  - bracket: does truth lie in [f_lo, f_hi]? what fraction-of-truth offset?
  - multi-hyp: smallest |truth - base| / truth across all K bases?
  - harmonic-division: same, including f1/2, f1/3, f1/4?
"""
import sys
import json
from pathlib import Path
import numpy as np
from scipy.signal import lombscargle, find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion"))

from lib.traj_source import load_truth, CANONICAL_NOISE_SIGMA
from lib.experiment_setup import setup_experiment

OUT = PROJECT_ROOT / "data" / "results" / "inversion_diagnostics" / "m138_isoshell_h1" / "lombscargle_probe"
OUT.mkdir(parents=True, exist_ok=True)


def peakcount_estimator(obs_lc, obs_times):
    valid = np.isfinite(obs_lc)
    bright_mask = valid & (obs_lc < obs_lc[valid].mean() - 1.0)
    transitions = np.diff(bright_mask.astype(int))
    n_peaks = max(1, (transitions > 0).sum())
    window = obs_times[-1] - obs_times[0]
    return 2 * np.pi * n_peaks / window


def run_ls(times, signal, n_freqs=4000):
    s = -signal
    s = s - np.mean(s)
    dt = np.median(np.diff(times))
    f_min = 1.0 / (times[-1] - times[0])
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, n_freqs)
    ang = 2 * np.pi * freqs
    power = lombscargle(times, s, ang, normalize=True)
    return freqs, power


def significant_peaks(freqs, power, threshold_frac=0.1, min_distance=5):
    """Peaks with power >= threshold_frac × max(power)."""
    pmax = power.max()
    idx, _ = find_peaks(power, distance=min_distance,
                        height=threshold_frac * pmax)
    if len(idx) == 0:
        return [], []
    order = idx[np.argsort(power[idx])[::-1]]
    return freqs[order].tolist(), power[order].tolist()


def grid_around(base_omega, span_low=0.3, span_high=3.0, n_mags=20):
    return np.geomspace(span_low * base_omega, span_high * base_omega, n_mags)


def coverage_metric(grid, truth_mag):
    """Returns (in_grid, nearest_pct_off, n_mags_within_5pct)."""
    if len(grid) == 0:
        return False, np.inf, 0
    grid = np.sort(np.asarray(grid))
    in_grid = grid[0] <= truth_mag <= grid[-1]
    pct = np.min(np.abs(grid - truth_mag)) / truth_mag * 100
    n_within_5 = int((np.abs(grid - truth_mag) / truth_mag <= 0.05).sum())
    return bool(in_grid), float(pct), n_within_5


def run_seed(seed, source="m048", power_thr=0.1, n_mags_per_basis=20):
    truth = load_truth(seed, source)
    obs_lc = truth["observed_lc"]
    truth_mag = float(np.linalg.norm(truth["omega0_rad"]))

    if source == "m048":
        ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                               random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                               start_et=truth["start_et"], skip_true_lc=True)
    else:
        ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                               random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                               end_time_utc=truth["end_time_utc"], skip_true_lc=True)
    obs_times = ctx.observation_times
    finite = np.isfinite(obs_lc)
    valid_lc = obs_lc[finite]
    valid_t = obs_times[finite]

    base_pc = peakcount_estimator(obs_lc, obs_times)
    freqs, power = run_ls(valid_t, valid_lc)

    pf, pw = significant_peaks(freqs, power, threshold_frac=power_thr)
    if not pf:
        return None
    peak_omegas = sorted(2 * np.pi * np.array(pf))

    # ── Strategy 0: peak-count baseline ──
    grid_pc = grid_around(base_pc, n_mags=n_mags_per_basis)
    cov_pc = coverage_metric(grid_pc, truth_mag)

    # ── Strategy 1: BRACKET — span all LS peaks ──
    # Conservative bracket: [0.5 × lowest, 2.0 × highest]
    lo = 0.5 * peak_omegas[0]
    hi = 2.0 * peak_omegas[-1]
    n_mags_bracket = max(n_mags_per_basis,
                         int(np.ceil(np.log(hi / lo) / np.log(1.05))))
    grid_bracket = np.geomspace(lo, hi, n_mags_bracket)
    cov_bracket = coverage_metric(grid_bracket, truth_mag)

    # ── Strategy 2: MULTI-HYP — union of [0.3, 3.0] × peak grids ──
    grids_multi = [grid_around(po, 0.3, 3.0, n_mags_per_basis) for po in peak_omegas]
    grid_multi = np.unique(np.concatenate(grids_multi))
    cov_multi = coverage_metric(grid_multi, truth_mag)

    # ── Strategy 3: HARMONIC-DIVISION — top peak + f1/2, f1/3, f1/4 ──
    f1 = peak_omegas[-1]  # highest-power peak
    bases_harm = [f1, f1 / 2, f1 / 3, f1 / 4]
    grids_harm = [grid_around(b, 0.3, 3.0, n_mags_per_basis) for b in bases_harm]
    grid_harm = np.unique(np.concatenate(grids_harm))
    cov_harm = coverage_metric(grid_harm, truth_mag)

    # ── Strategy 4: dominant-LS-peak only (m052 baseline, for control) ──
    # Highest-power peak as |ω|_estimate
    base_ls = peak_omegas[-1]  # highest-power peak
    grid_ls = grid_around(base_ls, 0.3, 3.0, n_mags_per_basis)
    cov_ls = coverage_metric(grid_ls, truth_mag)

    return {
        "seed": seed,
        "truth_mag": truth_mag,
        "n_significant_peaks": len(peak_omegas),
        "peak_omegas": peak_omegas,
        "peak_powers": pw,
        "pc_base": base_pc,
        "pc_grid_size": len(grid_pc),
        "pc_in_grid": cov_pc[0],
        "pc_nearest_pct": cov_pc[1],
        "pc_n_within_5pct": cov_pc[2],
        "ls_top1_base": base_ls,
        "ls_grid_size": len(grid_ls),
        "ls_in_grid": cov_ls[0],
        "ls_nearest_pct": cov_ls[1],
        "ls_n_within_5pct": cov_ls[2],
        "bracket_lo": lo,
        "bracket_hi": hi,
        "bracket_grid_size": len(grid_bracket),
        "bracket_in_grid": cov_bracket[0],
        "bracket_nearest_pct": cov_bracket[1],
        "bracket_n_within_5pct": cov_bracket[2],
        "multi_grid_size": len(grid_multi),
        "multi_in_grid": cov_multi[0],
        "multi_nearest_pct": cov_multi[1],
        "multi_n_within_5pct": cov_multi[2],
        "harm_grid_size": len(grid_harm),
        "harm_in_grid": cov_harm[0],
        "harm_nearest_pct": cov_harm[1],
        "harm_n_within_5pct": cov_harm[2],
    }


if __name__ == "__main__":
    seeds = [47, 91, 51, 79, 84, 89]
    results = {}
    for s in seeds:
        try:
            results[s] = run_seed(s)
        except Exception as e:
            results[s] = {"error": str(e)}

    (OUT / "bracket_summary.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved: {OUT}/bracket_summary.json")

    print(f"\n{'='*100}")
    print(f"COVERAGE: does truth |ω| land inside the grid? what's the nearest grid offset?")
    print(f"{'='*100}")
    print(f"{'seed':>4}  {'truth':>9}  {'n_pks':>5}  | "
          f"{'pc_in':>5} {'pc_off%':>7} {'pc_N':>4} | "
          f"{'ls_in':>5} {'ls_off%':>7} {'ls_N':>4} | "
          f"{'br_in':>5} {'br_off%':>7} {'br_N':>4} {'br_size':>7} | "
          f"{'mul_in':>6} {'mul_off%':>8} {'mul_N':>5} {'mul_size':>8} | "
          f"{'hrm_in':>6} {'hrm_off%':>8} {'hrm_N':>5}")
    for s, r in results.items():
        if "error" in r:
            print(f"  {s}  ERROR: {r['error']}")
            continue
        if r is None:
            print(f"  {s}  no significant peaks")
            continue
        def Y(b): return "Y" if b else "N"
        print(f"  {s:>2}  {r['truth_mag']:.5f}  {r['n_significant_peaks']:>5}  | "
              f"{Y(r['pc_in_grid']):>5} {r['pc_nearest_pct']:>7.2f} {r['pc_n_within_5pct']:>4} | "
              f"{Y(r['ls_in_grid']):>5} {r['ls_nearest_pct']:>7.2f} {r['ls_n_within_5pct']:>4} | "
              f"{Y(r['bracket_in_grid']):>5} {r['bracket_nearest_pct']:>7.2f} {r['bracket_n_within_5pct']:>4} {r['bracket_grid_size']:>7} | "
              f"{Y(r['multi_in_grid']):>6} {r['multi_nearest_pct']:>8.2f} {r['multi_n_within_5pct']:>5} {r['multi_grid_size']:>8} | "
              f"{Y(r['harm_in_grid']):>6} {r['harm_nearest_pct']:>8.2f} {r['harm_n_within_5pct']:>5}")

    print(f"\nLegend: in='truth in grid Y/N', off%='nearest grid pt distance to truth', "
          f"N='# grid points within 5% of truth', size='grid size'")
    print(f"\nNote: pc_grid_size = ls_grid_size = harm individual basis = 20 (per strategy_0/1).")
