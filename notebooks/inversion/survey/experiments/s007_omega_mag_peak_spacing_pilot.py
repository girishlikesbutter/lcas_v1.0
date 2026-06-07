"""s007 — Q4c-iii pilot: omega-magnitude estimate from LC peak spacing.

Hypothesis: the truth LC's dominant period (estimated via Lomb-Scargle on
the hi-fi truth magnitudes) maps to truth omega_mag within ~3% on
well-sampled seeds (>=3 rotations per 60-min LC). If so, this collapses
the omega-magnitude axis from a 25-point grid to ~3 candidates (the top
periodogram peaks), making Q4c (global-search -> local-polish) tractable
at cohort scale.

Method:
  10 PA/omega-stratified pilot seeds (6/10/18/21/28/41/48/60/84/91).
  For each:
    1. Lomb-Scargle on the truth hi-fi LC (also lofi for cross-check).
    2. Extract top-5 spectral peaks.
    3. Convert each peak to omega-mag at 1x and 2x harmonic hypotheses
       (LC may repeat per half-rotation if the satellite has 2-fold
       symmetric appearance from the observer).
    4. Pick the peak/harmonic pair whose omega is closest to truth -
       this is the oracle test of "does the prior have any peak that
       matches truth?".
    5. Pick the dominant peak's best-harmonic - this is the "blind"
       estimator that the actual inversion would use.
    6. Compute simple ACF as a cross-check.

Reports:
  - per-seed table: omega_truth, omega_dom_1x, omega_dom_2x,
    omega_dom_best, omega_oracle (best of top-5 x 2 harmonics),
    omega_acf, errors.
  - well-sampled subcohort (rot_per_lc >= 3): median error and worst case.
  - slow-rotator subcohort (rot_per_lc < 3): expected to be worse.
  - decision: is the dominant-peak blind estimator within 3% on the
    well-sampled subcohort?

Outputs:
  results/s007/peakspacing.npz        - per-seed arrays
  results/s007/summary.json           - decision-grade scalars
  results/s007/periodograms.png       - LS power vs freq, all seeds
  results/s007/truth_vs_estimate.png  - omega_truth vs blind/oracle estimates
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.traj_load import truth_state  # noqa: E402

PILOT_SEEDS = [6, 10, 18, 21, 28, 41, 48, 60, 84, 91]
OUT_DIR = ROOT / "results" / "s007"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lomb-Scargle frequency grid: f_min = 1 / window, f_max = Nyquist
# 60-min window, 7.2 s sampling -> Nyquist = 1/14.4 Hz
# We are interested in periods ~50-3500 s (omega 0.1-7 dps in 1x harmonic),
# so freq range 1/3600 to 1/14.4 Hz. Use a dense oversampled grid.
N_FREQ = 8000
N_TOP = 5  # top-N spectral peaks to keep per seed


def lomb_scargle_topk(t: np.ndarray, y: np.ndarray, n_top: int = N_TOP):
    """Return top-N (freq, power) peaks of the LS periodogram."""
    f_min = 1.0 / (t[-1] - t[0])  # one cycle in the full window
    f_max = 0.5 / np.median(np.diff(t))  # Nyquist
    freqs = np.linspace(f_min, f_max, N_FREQ)
    power = LombScargle(t, y).power(freqs)
    # find local maxima
    peaks, _ = find_peaks(power)
    if len(peaks) == 0:
        return freqs, power, np.array([]), np.array([])
    order = np.argsort(-power[peaks])
    top_idx = peaks[order[:n_top]]
    return freqs, power, freqs[top_idx], power[top_idx]


def acf_first_peak_period(t: np.ndarray, y: np.ndarray) -> float:
    """Estimate dominant period from autocorrelation first non-zero peak.

    Returns period in seconds, or NaN if no peak found.
    """
    dt = float(np.median(np.diff(t)))
    yc = y - y.mean()
    acf = np.correlate(yc, yc, mode="full")[len(yc) - 1 :]
    if acf[0] <= 0:
        return float("nan")
    acf = acf / acf[0]
    # exclude very short lags (below 2*dt) and require >=0.05 normalised acf
    min_lag = max(int(2.0 / dt), 2)
    candidates, _ = find_peaks(acf[min_lag:], height=0.05)
    if len(candidates) == 0:
        return float("nan")
    lag_samples = candidates[0] + min_lag
    return float(lag_samples * dt)


def omega_estimates_from_freq(f_hz: float) -> tuple[float, float]:
    """Convert a periodogram frequency to omega-mag (deg/s) at 1x and 2x harmonics.

    1x: assume LC period equals rotation period -> omega = 360 * f.
    2x: assume LC period equals half rotation (2-fold symmetric appearance)
        -> omega = 180 * f.
    """
    return 360.0 * f_hz, 180.0 * f_hz


def per_seed_analyze(seed: int) -> dict:
    s = truth_state(seed)
    t = s["observation_times"]
    mag_hifi = s["mag_hifi"]
    mag_lofi = s["mag_lofi"]
    omega_truth_dps = float(s["omega_mag_dps"])
    period_truth = 360.0 / omega_truth_dps
    rot_per_lc = (t[-1] - t[0]) / period_truth

    # Lomb-Scargle on hifi
    freqs_hi, power_hi, top_f_hi, top_p_hi = lomb_scargle_topk(t, mag_hifi, N_TOP)
    # Lomb-Scargle on lofi (cross-check)
    freqs_lo, power_lo, top_f_lo, top_p_lo = lomb_scargle_topk(t, mag_lofi, N_TOP)

    # Dominant blind estimate (hifi)
    if len(top_f_hi) >= 1:
        f_dom = float(top_f_hi[0])
        om_1x, om_2x = omega_estimates_from_freq(f_dom)
        # blind: pick whichever harmonic minimises a sanity prior. We have NO
        # truth at inversion time, so we report BOTH and let downstream sample
        # both (low cost). The "oracle" diagnostic below tests if either matched.
        blind_1x = om_1x
        blind_2x = om_2x
    else:
        blind_1x = float("nan")
        blind_2x = float("nan")

    # Oracle: cheapest of top-5 peaks x {1x, 2x harmonics}, picked vs truth
    if len(top_f_hi) >= 1:
        candidates = []
        for f in top_f_hi:
            for om in omega_estimates_from_freq(float(f)):
                candidates.append(om)
        candidates = np.array(candidates)
        rel_errs = np.abs(candidates - omega_truth_dps) / omega_truth_dps
        best_i = int(np.argmin(rel_errs))
        omega_oracle = float(candidates[best_i])
        oracle_err_pct = float(100.0 * rel_errs[best_i])
        oracle_kind = "1x" if best_i % 2 == 0 else "2x"
        oracle_peak_rank = best_i // 2  # 0 = top peak, 1 = 2nd peak, ...
    else:
        omega_oracle = float("nan")
        oracle_err_pct = float("nan")
        oracle_kind = ""
        oracle_peak_rank = -1

    # ACF
    T_acf = acf_first_peak_period(t, mag_hifi)
    if np.isfinite(T_acf):
        om_acf_1x = 360.0 / T_acf
        om_acf_2x = 720.0 / T_acf
        err_acf_1x = abs(om_acf_1x - omega_truth_dps) / omega_truth_dps
        err_acf_2x = abs(om_acf_2x - omega_truth_dps) / omega_truth_dps
        if err_acf_1x <= err_acf_2x:
            om_acf_best = om_acf_1x
            acf_best_kind = "1x"
            err_acf_best = err_acf_1x
        else:
            om_acf_best = om_acf_2x
            acf_best_kind = "2x"
            err_acf_best = err_acf_2x
    else:
        om_acf_1x = om_acf_2x = om_acf_best = float("nan")
        err_acf_best = float("nan")
        acf_best_kind = ""

    return {
        "seed": seed,
        "omega_truth_dps": omega_truth_dps,
        "period_truth_s": period_truth,
        "rot_per_lc": float(rot_per_lc),
        # hifi LS
        "freqs_hi": freqs_hi,
        "power_hi": power_hi,
        "top_f_hi": top_f_hi,
        "top_p_hi": top_p_hi,
        # lofi LS
        "top_f_lo": top_f_lo,
        "top_p_lo": top_p_lo,
        # blind (dominant peak hifi)
        "blind_1x_dps": blind_1x,
        "blind_2x_dps": blind_2x,
        "blind_1x_err_pct": 100.0 * abs(blind_1x - omega_truth_dps) / omega_truth_dps if np.isfinite(blind_1x) else float("nan"),
        "blind_2x_err_pct": 100.0 * abs(blind_2x - omega_truth_dps) / omega_truth_dps if np.isfinite(blind_2x) else float("nan"),
        "blind_best_err_pct": min(
            100.0 * abs(blind_1x - omega_truth_dps) / omega_truth_dps if np.isfinite(blind_1x) else np.inf,
            100.0 * abs(blind_2x - omega_truth_dps) / omega_truth_dps if np.isfinite(blind_2x) else np.inf,
        ),
        # oracle
        "omega_oracle_dps": omega_oracle,
        "oracle_err_pct": oracle_err_pct,
        "oracle_kind": oracle_kind,
        "oracle_peak_rank": oracle_peak_rank,
        # acf
        "T_acf_s": T_acf,
        "om_acf_1x": om_acf_1x,
        "om_acf_2x": om_acf_2x,
        "om_acf_best": om_acf_best,
        "acf_best_kind": acf_best_kind,
        "acf_best_err_pct": 100.0 * err_acf_best if np.isfinite(err_acf_best) else float("nan"),
    }


def make_periodogram_plot(results: list[dict], outpath: Path):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.6 * n), sharex=True)
    for ax, r in zip(axes, results):
        ax.plot(r["freqs_hi"], r["power_hi"], "k-", lw=0.7, label="hifi")
        # mark truth-omega 1x and 2x lines
        f_truth_1x = r["omega_truth_dps"] / 360.0
        f_truth_2x = r["omega_truth_dps"] / 180.0
        ax.axvline(f_truth_1x, color="tab:green", ls="--", alpha=0.6, label=f"truth 1x = {f_truth_1x:.4f} Hz")
        ax.axvline(f_truth_2x, color="tab:blue", ls=":", alpha=0.6, label=f"truth 2x = {f_truth_2x:.4f} Hz")
        # mark top-5 hifi peaks
        ax.scatter(r["top_f_hi"], r["top_p_hi"], color="tab:red", s=15, zorder=5, label="top-5 LS peaks")
        ax.set_ylabel(f"s{r['seed']:03d}\n({r['rot_per_lc']:.1f} rot)")
        ax.legend(fontsize=6, loc="upper right", ncol=4)
        ax.set_xlim(0, max(r["freqs_hi"]))
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("s007 — Lomb-Scargle periodograms (hifi truth LC)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def make_truth_vs_estimate_plot(results: list[dict], outpath: Path):
    truth = np.array([r["omega_truth_dps"] for r in results])
    blind1 = np.array([r["blind_1x_dps"] for r in results])
    blind2 = np.array([r["blind_2x_dps"] for r in results])
    oracle = np.array([r["omega_oracle_dps"] for r in results])
    rot = np.array([r["rot_per_lc"] for r in results])
    seeds = [r["seed"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    lo = 0.05
    hi = 2.0
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x")
    ax.plot([lo, hi], [2 * lo, 2 * hi], "k:", alpha=0.3, label="y=2x")
    ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], "k:", alpha=0.3, label="y=x/2")
    ax.scatter(truth, blind1, marker="o", color="tab:red", s=70, label="blind 1x", zorder=4)
    ax.scatter(truth, blind2, marker="s", color="tab:blue", s=70, label="blind 2x", zorder=4)
    ax.scatter(truth, oracle, marker="*", color="tab:green", s=120, label="oracle (best of top-5 x harmonics)", zorder=5)
    for s, t, o, r_ in zip(seeds, truth, oracle, rot):
        ax.annotate(f"{s}\n({r_:.1f}r)", (t, o), fontsize=7, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("truth omega_mag (deg/s)")
    ax.set_ylabel("estimated omega_mag (deg/s)")
    ax.set_title("s007 — peak-spacing omega-mag estimate vs truth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    print(f"s007 pilot — {len(PILOT_SEEDS)} seeds")
    results = []
    for seed in PILOT_SEEDS:
        r = per_seed_analyze(seed)
        print(
            f"  seed {seed:3d}: omega_truth={r['omega_truth_dps']:.4f} dps "
            f"({r['rot_per_lc']:.2f} rot)  "
            f"blind_1x={r['blind_1x_dps']:.4f} ({r['blind_1x_err_pct']:.1f}%)  "
            f"blind_2x={r['blind_2x_dps']:.4f} ({r['blind_2x_err_pct']:.1f}%)  "
            f"oracle={r['omega_oracle_dps']:.4f} ({r['oracle_err_pct']:.2f}%, {r['oracle_kind']}, peak#{r['oracle_peak_rank']})  "
            f"acf_best={r['om_acf_best']:.4f} ({r['acf_best_err_pct']:.1f}%, {r['acf_best_kind']})"
        )
        results.append(r)

    # Save NPZ — strip variable-length / array fields into a structured dump
    npz_payload = {}
    npz_payload["seeds"] = np.array([r["seed"] for r in results])
    npz_payload["omega_truth_dps"] = np.array([r["omega_truth_dps"] for r in results])
    npz_payload["rot_per_lc"] = np.array([r["rot_per_lc"] for r in results])
    npz_payload["blind_1x_dps"] = np.array([r["blind_1x_dps"] for r in results])
    npz_payload["blind_2x_dps"] = np.array([r["blind_2x_dps"] for r in results])
    npz_payload["blind_1x_err_pct"] = np.array([r["blind_1x_err_pct"] for r in results])
    npz_payload["blind_2x_err_pct"] = np.array([r["blind_2x_err_pct"] for r in results])
    npz_payload["blind_best_err_pct"] = np.array([r["blind_best_err_pct"] for r in results])
    npz_payload["omega_oracle_dps"] = np.array([r["omega_oracle_dps"] for r in results])
    npz_payload["oracle_err_pct"] = np.array([r["oracle_err_pct"] for r in results])
    npz_payload["oracle_kind"] = np.array([r["oracle_kind"] for r in results])
    npz_payload["oracle_peak_rank"] = np.array([r["oracle_peak_rank"] for r in results])
    npz_payload["T_acf_s"] = np.array([r["T_acf_s"] for r in results])
    npz_payload["acf_best_err_pct"] = np.array([r["acf_best_err_pct"] for r in results])
    # store per-seed top-5 peak data
    npz_payload["top_f_hi"] = np.array([np.pad(r["top_f_hi"], (0, max(0, N_TOP - len(r["top_f_hi"]))), constant_values=np.nan) for r in results])
    npz_payload["top_p_hi"] = np.array([np.pad(r["top_p_hi"], (0, max(0, N_TOP - len(r["top_p_hi"]))), constant_values=np.nan) for r in results])
    npz_payload["top_f_lo"] = np.array([np.pad(r["top_f_lo"], (0, max(0, N_TOP - len(r["top_f_lo"]))), constant_values=np.nan) for r in results])
    npz_payload["top_p_lo"] = np.array([np.pad(r["top_p_lo"], (0, max(0, N_TOP - len(r["top_p_lo"]))), constant_values=np.nan) for r in results])
    np.savez(OUT_DIR / "peakspacing.npz", **npz_payload)

    # Subcohort split
    well_sampled_mask = npz_payload["rot_per_lc"] >= 3.0
    slow_mask = ~well_sampled_mask
    blind_best_well = npz_payload["blind_best_err_pct"][well_sampled_mask]
    oracle_well = npz_payload["oracle_err_pct"][well_sampled_mask]
    blind_best_slow = npz_payload["blind_best_err_pct"][slow_mask]
    oracle_slow = npz_payload["oracle_err_pct"][slow_mask]

    summary = {
        "n_seeds": len(results),
        "n_well_sampled": int(well_sampled_mask.sum()),
        "n_slow": int(slow_mask.sum()),
        "well_sampled_seeds": [int(s) for s in npz_payload["seeds"][well_sampled_mask]],
        "slow_seeds": [int(s) for s in npz_payload["seeds"][slow_mask]],
        "blind_best_err_pct": {
            "all_median": float(np.median(npz_payload["blind_best_err_pct"])),
            "all_max": float(np.max(npz_payload["blind_best_err_pct"])),
            "well_median": float(np.median(blind_best_well)) if len(blind_best_well) else None,
            "well_max": float(np.max(blind_best_well)) if len(blind_best_well) else None,
            "slow_median": float(np.median(blind_best_slow)) if len(blind_best_slow) else None,
            "slow_max": float(np.max(blind_best_slow)) if len(blind_best_slow) else None,
        },
        "oracle_err_pct": {
            "all_median": float(np.median(npz_payload["oracle_err_pct"])),
            "all_max": float(np.max(npz_payload["oracle_err_pct"])),
            "well_median": float(np.median(oracle_well)) if len(oracle_well) else None,
            "well_max": float(np.max(oracle_well)) if len(oracle_well) else None,
            "slow_median": float(np.median(oracle_slow)) if len(oracle_slow) else None,
            "slow_max": float(np.max(oracle_slow)) if len(oracle_slow) else None,
        },
        "decision_threshold_pct": 3.0,
        "blind_best_passes_well": bool(
            len(blind_best_well) > 0 and np.median(blind_best_well) <= 3.0
        ),
        "oracle_passes_well": bool(len(oracle_well) > 0 and np.median(oracle_well) <= 3.0),
        "wall_seconds": float(time.perf_counter() - t0),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    make_periodogram_plot(results, OUT_DIR / "periodograms.png")
    make_truth_vs_estimate_plot(results, OUT_DIR / "truth_vs_estimate.png")

    # Console summary
    print("")
    print(f"=== s007 pilot summary (wall {summary['wall_seconds']:.2f} s) ===")
    print(f"well-sampled seeds (>=3 rot): {summary['well_sampled_seeds']}")
    print(f"slow seeds (<3 rot):          {summary['slow_seeds']}")
    print("blind-best error pct:")
    print(f"  well-sampled: median={summary['blind_best_err_pct']['well_median']:.2f}  max={summary['blind_best_err_pct']['well_max']:.2f}")
    if summary["blind_best_err_pct"]["slow_median"] is not None:
        print(f"  slow:         median={summary['blind_best_err_pct']['slow_median']:.2f}  max={summary['blind_best_err_pct']['slow_max']:.2f}")
    print("oracle error pct (best of top-5 x 2 harmonics):")
    print(f"  well-sampled: median={summary['oracle_err_pct']['well_median']:.2f}  max={summary['oracle_err_pct']['well_max']:.2f}")
    if summary["oracle_err_pct"]["slow_median"] is not None:
        print(f"  slow:         median={summary['oracle_err_pct']['slow_median']:.2f}  max={summary['oracle_err_pct']['slow_max']:.2f}")
    print(f"decision (3% threshold on well-sampled median):")
    print(f"  blind_best passes: {summary['blind_best_passes_well']}")
    print(f"  oracle passes:     {summary['oracle_passes_well']}")
    print("Saved:")
    print(f"  {OUT_DIR / 'peakspacing.npz'}")
    print(f"  {OUT_DIR / 'summary.json'}")
    print(f"  {OUT_DIR / 'periodograms.png'}")
    print(f"  {OUT_DIR / 'truth_vs_estimate.png'}")


if __name__ == "__main__":
    main()
