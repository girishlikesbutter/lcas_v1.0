"""Stage 1: LC spectrum census across all 100 m048 seeds.

Goal: given hi-fi light curve mag_hifi(t), try to estimate |ω| from
FFT peak frequencies and autocorrelation first-side-peak, under several
candidate mapping rules:

    rule "omega_eq_2pi_f"    : |ω| = 2π × f_peak   (full-revolution period)
    rule "omega_eq_pi_f"     : |ω| = π × f_peak    (half-period — LC is often
                                                    symmetric-lobe so one rev
                                                    gives 2 peaks)
    rule "omega_eq_pi_f_h2"  : |ω| = π × (f_peak/2) (second harmonic)
    rule "omega_eq_2pi_acf"  : |ω| = 2π / T_acf_peak (autocorrelation period)
    rule "omega_eq_pi_acf"   : |ω| = π  / T_acf_peak (half-period acf)

Body-frame |ω| evolves on the inertia ellipsoid, so "the" period is
ambiguous for non-trivial inertia. We report all rules and let the
downstream stages decide.

Outputs (paths printed):
- spectrum_census.npz  (per-seed arrays)
- spectrum_census.json (summary table)
- spectrum_census.png  (|ω_truth| vs |ω_est| panels)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "inversion" / "13_clean_slate_omega"))

from lib.data import all_seeds, load_seed  # noqa: E402


OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "inversion_diagnostics"
    / "13_clean_slate_omega"
    / "a_spectral"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------- spectrum primitives --------------------------------------------------


def fft_power(y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided FFT power spectrum (demeaned, Hann-windowed)."""
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    w = np.hanning(len(y))
    Y = np.fft.rfft(y * w)
    freqs = np.fft.rfftfreq(len(y), d=dt)
    power = (np.abs(Y) ** 2) / np.sum(w ** 2)
    return freqs, power


def top_peaks(
    freqs: np.ndarray,
    power: np.ndarray,
    k: int = 5,
    min_freq: float = 1e-4,
    min_separation: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, powers) of the top-k local maxima above min_freq,
    enforcing `min_separation` bin spacing."""
    mask = freqs >= min_freq
    idx = np.where(mask)[0]
    order = np.argsort(power[idx])[::-1]
    chosen: list[int] = []
    for j in order:
        gi = idx[j]
        # local max check
        lo = max(0, gi - 1)
        hi = min(len(power) - 1, gi + 1)
        if power[gi] < power[lo] or power[gi] < power[hi]:
            continue
        if any(abs(gi - c) < min_separation for c in chosen):
            continue
        chosen.append(gi)
        if len(chosen) >= k:
            break
    chosen = np.array(chosen, dtype=int)
    return freqs[chosen], power[chosen]


def autocorr(y: np.ndarray) -> np.ndarray:
    """One-sided normalised autocorrelation of y (demeaned)."""
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    n = len(y)
    Y = np.fft.rfft(y, n=2 * n)
    acf = np.fft.irfft(Y * np.conj(Y))[:n]
    acf /= acf[0] + 1e-30
    return acf


def first_strong_acf_peak(
    acf: np.ndarray,
    dt: float,
    min_lag_s: float = 20.0,
) -> tuple[float, float]:
    """Return (T_seconds, value) of the first local max above min_lag_s.

    Returns (nan, nan) if none found above the 0.05 threshold.
    """
    lags = np.arange(len(acf)) * dt
    start = int(min_lag_s / dt) + 1
    for i in range(start, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.05:
            return float(lags[i]), float(acf[i])
    return float("nan"), float("nan")


def top_acf_peaks(
    acf: np.ndarray,
    dt: float,
    k: int = 3,
    min_lag_s: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    lags = np.arange(len(acf)) * dt
    start = int(min_lag_s / dt) + 1
    peaks = []
    values = []
    for i in range(start, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
            peaks.append(lags[i])
            values.append(acf[i])
    if not peaks:
        return np.array([np.nan] * k), np.array([np.nan] * k)
    peaks = np.array(peaks)
    values = np.array(values)
    # top-k by value
    order = np.argsort(values)[::-1][:k]
    out_t = peaks[order]
    out_v = values[order]
    # pad
    if len(out_t) < k:
        pad = k - len(out_t)
        out_t = np.concatenate([out_t, np.full(pad, np.nan)])
        out_v = np.concatenate([out_v, np.full(pad, np.nan)])
    return out_t, out_v


# -------- mapping rules --------------------------------------------------------


def mapping_estimates(f_peak_hz: float, t_acf_s: float) -> dict[str, float]:
    """Apply candidate mapping rules. Returns |ω| estimate in deg/s."""
    out: dict[str, float] = {}
    if f_peak_hz > 0 and np.isfinite(f_peak_hz):
        omega_rad_2pi = 2.0 * np.pi * f_peak_hz
        omega_rad_pi = np.pi * f_peak_hz
        omega_rad_pi_half = np.pi * (f_peak_hz / 2.0)
        out["omega_eq_2pi_f"] = float(np.degrees(omega_rad_2pi))
        out["omega_eq_pi_f"] = float(np.degrees(omega_rad_pi))
        out["omega_eq_pi_f_h2"] = float(np.degrees(omega_rad_pi_half))
    else:
        out["omega_eq_2pi_f"] = float("nan")
        out["omega_eq_pi_f"] = float("nan")
        out["omega_eq_pi_f_h2"] = float("nan")

    if t_acf_s > 0 and np.isfinite(t_acf_s):
        omega_rad_full = 2.0 * np.pi / t_acf_s
        omega_rad_half = np.pi / t_acf_s
        out["omega_eq_2pi_acf"] = float(np.degrees(omega_rad_full))
        out["omega_eq_pi_acf"] = float(np.degrees(omega_rad_half))
    else:
        out["omega_eq_2pi_acf"] = float("nan")
        out["omega_eq_pi_acf"] = float("nan")
    return out


# -------- main -----------------------------------------------------------------


def main() -> None:
    seeds = all_seeds()
    print(f"Census over {len(seeds)} seeds.")

    # per-seed storage
    per_seed: list[dict] = []
    # FFT/ACF arrays need uniform shape; store lists of arrays and pack at end.
    all_freqs: list[np.ndarray] = []
    all_powers: list[np.ndarray] = []
    all_acfs: list[np.ndarray] = []
    all_top5_freq: list[np.ndarray] = []
    all_top5_power: list[np.ndarray] = []
    all_top3_acf_t: list[np.ndarray] = []
    all_top3_acf_v: list[np.ndarray] = []

    rules = [
        "omega_eq_2pi_f",
        "omega_eq_pi_f",
        "omega_eq_pi_f_h2",
        "omega_eq_2pi_acf",
        "omega_eq_pi_acf",
    ]

    t0 = time.perf_counter()
    for si, seed in enumerate(seeds):
        b = load_seed(seed)
        y = b["mag_hifi"]
        dt = b["dt_sampling"]
        truth_dps = b["omega_mag_dps"]

        f, p = fft_power(y, dt)
        acf = autocorr(y)
        top_f, top_fp = top_peaks(f, p, k=5)
        top_at, top_av = top_acf_peaks(acf, dt, k=3)

        # dominant = highest-power FFT peak; dominant ACF = first strong peak
        f_dom = float(top_f[0]) if len(top_f) else float("nan")
        t_acf_first, _ = first_strong_acf_peak(acf, dt)

        est = mapping_estimates(f_dom, t_acf_first)

        row = {
            "seed": int(seed),
            "omega_mag_dps_true": float(truth_dps),
            "phase_mean_deg": float(np.mean(b["phase_angle_3d"])),
            "f_dominant_hz": f_dom,
            "t_acf_first_s": float(t_acf_first),
            "top5_freqs_hz": top_f.tolist() + [float("nan")] * (5 - len(top_f)),
            "top5_powers": top_fp.tolist() + [float("nan")] * (5 - len(top_fp)),
            "top3_acf_t_s": top_at.tolist(),
            "top3_acf_v": top_av.tolist(),
            "estimates_dps": est,
        }
        # per-rule rel errors
        row["rel_err_pct"] = {
            r: (
                (est[r] - truth_dps) / truth_dps * 100.0
                if (truth_dps > 0 and np.isfinite(est[r]))
                else float("nan")
            )
            for r in rules
        }
        per_seed.append(row)

        all_freqs.append(f)
        all_powers.append(p)
        all_acfs.append(acf)
        # pad top5 arrays to length 5
        tf = np.concatenate([top_f, np.full(5 - len(top_f), np.nan)]) if len(top_f) < 5 else top_f
        tp = np.concatenate([top_fp, np.full(5 - len(top_fp), np.nan)]) if len(top_fp) < 5 else top_fp
        all_top5_freq.append(tf)
        all_top5_power.append(tp)
        all_top3_acf_t.append(top_at)
        all_top3_acf_v.append(top_av)

        if (si + 1) % 25 == 0:
            print(f"  [{si + 1:3d}/{len(seeds)}] seed {seed:03d}  truth|ω|={truth_dps:6.3f} dps  "
                  f"f_dom={f_dom:.5f} Hz  T_acf={t_acf_first:7.2f} s")

    t_total = time.perf_counter() - t0
    print(f"Spectra done in {t_total:.1f} s.")

    # pack arrays
    freqs_arr = np.stack(all_freqs)       # (Nseed, Nfreq)
    powers_arr = np.stack(all_powers)
    acf_arr = np.stack(all_acfs)
    top5_freq_arr = np.stack(all_top5_freq)
    top5_power_arr = np.stack(all_top5_power)
    top3_acf_t_arr = np.stack(all_top3_acf_t)
    top3_acf_v_arr = np.stack(all_top3_acf_v)

    truth_dps = np.array([r["omega_mag_dps_true"] for r in per_seed])
    phase = np.array([r["phase_mean_deg"] for r in per_seed])

    # per-rule vectors
    est_by_rule = {r: np.array([row["estimates_dps"][r] for row in per_seed]) for r in rules}
    relerr_by_rule = {r: np.array([row["rel_err_pct"][r] for row in per_seed]) for r in rules}

    # NPZ dump
    npz_path = OUT_DIR / "spectrum_census.npz"
    np.savez_compressed(
        npz_path,
        seeds=np.array([r["seed"] for r in per_seed]),
        truth_dps=truth_dps,
        phase_mean_deg=phase,
        freqs_hz=freqs_arr,
        fft_power=powers_arr,
        acf=acf_arr,
        top5_freqs_hz=top5_freq_arr,
        top5_powers=top5_power_arr,
        top3_acf_lag_s=top3_acf_t_arr,
        top3_acf_value=top3_acf_v_arr,
        f_dominant_hz=np.array([r["f_dominant_hz"] for r in per_seed]),
        t_acf_first_s=np.array([r["t_acf_first_s"] for r in per_seed]),
        **{f"est_{r}": est_by_rule[r] for r in rules},
        **{f"relerr_{r}": relerr_by_rule[r] for r in rules},
    )
    print(f"Saved: {npz_path}")

    # JSON summary table
    summary = {
        "n_seeds": len(seeds),
        "dt_sampling_s": float(load_seed(seeds[0])["dt_sampling"]),
        "duration_s": float(load_seed(seeds[0])["duration_s"]),
        "nyquist_hz": 0.5 / float(load_seed(seeds[0])["dt_sampling"]),
        "rules": rules,
        "rule_summary": {},
        "per_seed": per_seed,
    }
    # per-rule how many |est - truth|/truth < 10%, 5%, 25%
    for r in rules:
        re = relerr_by_rule[r]
        abs_re = np.abs(re)
        summary["rule_summary"][r] = {
            "median_abs_err_pct": float(np.nanmedian(abs_re)),
            "mean_abs_err_pct": float(np.nanmean(abs_re)),
            "within_5pct": int(np.nansum(abs_re < 5.0)),
            "within_10pct": int(np.nansum(abs_re < 10.0)),
            "within_25pct": int(np.nansum(abs_re < 25.0)),
        }
    # find best rule per seed
    rule_matrix = np.stack([np.abs(relerr_by_rule[r]) for r in rules], axis=1)  # (Nseed, Nrule)
    best_rule_idx = np.argmin(np.where(np.isnan(rule_matrix), np.inf, rule_matrix), axis=1)
    best_rules = [rules[i] for i in best_rule_idx]
    from collections import Counter
    summary["best_rule_counts"] = dict(Counter(best_rules))

    json_path = OUT_DIR / "spectrum_census.json"
    with open(json_path, "w") as f_:
        json.dump(summary, f_, indent=2, default=str)
    print(f"Saved: {json_path}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for ai, r in enumerate(rules):
            ax = axes[ai]
            est = est_by_rule[r]
            ok = np.isfinite(est) & np.isfinite(truth_dps)
            ax.scatter(truth_dps[ok], est[ok], c=phase[ok], cmap="viridis", s=25)
            lim = max(np.nanmax(truth_dps), np.nanmax(est[ok])) * 1.1
            ax.plot([0, lim], [0, lim], "k--", lw=0.8)
            # label outliers
            abs_rel = np.abs(relerr_by_rule[r])
            bad = np.where(abs_rel > 50.0)[0]
            for bi in bad[:15]:
                ax.text(truth_dps[bi], est[bi], str(int(per_seed[bi]["seed"])),
                        fontsize=7, color="red")
            ax.set_title(f"{r}\nwithin10%: {summary['rule_summary'][r]['within_10pct']}/100")
            ax.set_xlabel("truth |ω| (dps)")
            ax.set_ylabel("estimated |ω| (dps)")
            ax.grid(alpha=0.3)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
        # last pane: best-rule histogram
        ax = axes[-1]
        from collections import Counter
        counts = Counter(best_rules)
        keys = list(counts.keys())
        ax.bar(range(len(keys)), [counts[k] for k in keys])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_title("Best rule per seed (argmin |rel err|)")
        fig.suptitle("Stage 1 spectral |ω| census — 100 m048 seeds", fontsize=14)
        fig.tight_layout()
        png_path = OUT_DIR / "spectrum_census.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"Saved: {png_path}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
