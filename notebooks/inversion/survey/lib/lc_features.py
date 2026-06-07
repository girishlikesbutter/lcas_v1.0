"""LC feature extraction for the survey workspace.

Lifted verbatim from `experiments/s008_lc_feature_regression_omega.py`
(lines 84-179) so that downstream regression experiments (s055a, ...)
can reuse the same 28-feature set without copy-paste drift.

The features are intentionally LC-only — input is `(t, mag)` and nothing
else. No truth ω, no geometry. The same features that closed Q4c-iv on
|ω| (s008: LOO MAPE 16.4%) feed s055a's polhode-diameter regression.

Module-level `FEATURE_NAMES` gives a stable column ordering for
`lc_features(t, mag)[1]`.
"""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

K_LS_PEAKS = 5
N_FREQ = 4000  # LS frequency grid

FEATURE_NAMES: list[str] = (
    [f"ls_top{i}_f" for i in range(K_LS_PEAKS)]
    + [f"ls_top{i}_p" for i in range(K_LS_PEAKS)]
    + ["pwr_low", "pwr_high", "pwr_ratio"]
    + ["mag_mean", "mag_std", "mag_min", "mag_max", "mag_skew", "mag_kurt"]
    + ["dmag_mean_abs", "dmag_std"]
    + ["acf_lag1", "acf_lag5", "acf_lag20", "acf_first_peak_lag"]
    + ["glint_count", "glint_spacing_mean", "glint_spacing_std"]
)


def lc_features(t: np.ndarray, mag: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract a fixed-length feature vector from a single LC.

    Parameters
    ----------
    t   : (N,) observation times in ET seconds.
    mag : (N,) magnitude time series (lower = brighter).

    Returns
    -------
    feats : (28,) feature vector aligned with `FEATURE_NAMES`.
    names : list[str] of length 28 — same ordering as `FEATURE_NAMES`.
    """
    dt = float(np.median(np.diff(t)))
    span = float(t[-1] - t[0])

    f_min = 1.0 / span
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, N_FREQ)
    power = LombScargle(t, mag).power(freqs)
    peaks, _ = find_peaks(power)
    if len(peaks) >= K_LS_PEAKS:
        order = np.argsort(-power[peaks])
        top = peaks[order[:K_LS_PEAKS]]
        top_f = freqs[top]
        top_p = power[top]
    else:
        top_f = np.zeros(K_LS_PEAKS)
        top_p = np.zeros(K_LS_PEAKS)
        if len(peaks) > 0:
            order = np.argsort(-power[peaks])
            top_f[: len(peaks)] = freqs[peaks[order]]
            top_p[: len(peaks)] = power[peaks[order]]

    f_split = 0.01
    pwr_low = float(power[freqs < f_split].sum())
    pwr_high = float(power[freqs >= f_split].sum())
    pwr_ratio = pwr_high / max(pwr_low, 1e-12)

    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    mag_min = float(np.min(mag))
    mag_max = float(np.max(mag))
    mag_skew = float(skew(mag))
    mag_kurt = float(kurtosis(mag))

    dmag = np.diff(mag) / dt
    dmag_mean_abs = float(np.mean(np.abs(dmag)))
    dmag_std = float(np.std(dmag))

    yc = mag - mag.mean()
    acf_full = np.correlate(yc, yc, mode="full")[len(yc) - 1 :]
    acf = acf_full / max(acf_full[0], 1e-12)
    acf_lag1 = float(acf[1]) if len(acf) > 1 else 0.0
    acf_lag5 = float(acf[5]) if len(acf) > 5 else 0.0
    acf_lag20 = float(acf[20]) if len(acf) > 20 else 0.0

    min_lag = max(int(2.0 / dt), 2)
    cand_peaks, _ = find_peaks(acf[min_lag:], height=0.05)
    if len(cand_peaks) > 0:
        acf_first_peak_lag = float((cand_peaks[0] + min_lag) * dt)
    else:
        acf_first_peak_lag = 0.0

    bright_thresh = float(np.percentile(mag, 10))
    glint_idx, _ = find_peaks(-mag, height=-bright_thresh)
    glint_count = float(len(glint_idx))
    if len(glint_idx) >= 2:
        glint_spacings = np.diff(t[glint_idx])
        glint_spacing_mean = float(np.mean(glint_spacings))
        glint_spacing_std = float(np.std(glint_spacings))
    else:
        glint_spacing_mean = 0.0
        glint_spacing_std = 0.0

    feats = np.concatenate([
        top_f,
        top_p,
        [pwr_low, pwr_high, pwr_ratio],
        [mag_mean, mag_std, mag_min, mag_max, mag_skew, mag_kurt],
        [dmag_mean_abs, dmag_std],
        [acf_lag1, acf_lag5, acf_lag20, acf_first_peak_lag],
        [glint_count, glint_spacing_mean, glint_spacing_std],
    ])

    return feats, list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# LS-bracket |ω| prior (port of s019 bracket strategy, condensed to N cells).
#
# s019 (pre-fix substrate) measured the full ~50-cell bracket variant on 100
# m048 seeds: 98/100 in-grid, 98/100 within 5%, p50 1.25%, p90 2.26%
# (source: results_prefix/s019/summary.json — pre-fix cohort; method survives
# the propagator fix because it's pure spectral analysis of mag_hifi). The
# 5-cell `geomspace([0.5*min_peak_omega, 2.0*max_peak_omega], 5)` form here
# preserves coverage (truth lies in [min,max] in 98/100 pre-fix) but trades
# nearest-cell offset for a smaller grid — worst-case offset ≈ 17% at 5 cells
# vs 5% at the full ~50-cell bracket. Pivot consumers should record the
# nearest-cell |ω|-offset as a diagnostic.
# ---------------------------------------------------------------------------


def ls_bracket(
    times: np.ndarray,
    mag: np.ndarray,
    n_cells: int = 5,
    threshold_frac: float = 0.1,
) -> np.ndarray:
    """5-cell |ω| geomspace bracket from LS-peak [min, max] (rad/s).

    Lomb-Scargle peak detection on the LC, then geomspace from half of the
    lowest-frequency significant peak to twice the highest, with `n_cells`
    cells. Used as the |ω| prior in the s082 joint-grid pivot and the
    Day-2 blind pipeline. See `experiments/s019_ls_bracket_omega_mag.py`
    for the wider ~50-cell variant.

    Parameters
    ----------
    times : (N,) observation times in ET seconds.
    mag : (N,) magnitude time series (lower = brighter).
    n_cells : int, number of |ω| grid cells. Default 5.
    threshold_frac : float, peak power threshold as a fraction of max power.

    Returns
    -------
    omega_grid : (n_cells,) geomspaced |ω| values in rad/s.
                 Empty array if no significant LS peaks.
    """
    t = np.asarray(times, dtype=np.float64)
    m = np.asarray(mag, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(m)
    if mask.sum() < 50:
        return np.array([])
    t = t[mask]
    m = m[mask]

    dt = float(np.median(np.diff(t)))
    span = float(t[-1] - t[0])
    if span <= 0 or dt <= 0:
        return np.array([])
    f_min = 1.0 / span
    f_max = 0.5 / dt
    freqs = np.linspace(f_min, f_max, N_FREQ)
    power = LombScargle(t, m).power(freqs)

    p_max = float(power.max())
    if p_max <= 0:
        return np.array([])
    peaks, _ = find_peaks(power, distance=5, height=threshold_frac * p_max)
    if len(peaks) == 0:
        return np.array([])

    peak_freqs = freqs[peaks]
    omega_peaks = 2.0 * np.pi * peak_freqs  # rad/s
    lo = 0.5 * float(omega_peaks.min())
    hi = 2.0 * float(omega_peaks.max())
    if lo <= 0 or hi <= lo:
        return np.array([])
    return np.geomspace(lo, hi, int(n_cells))
