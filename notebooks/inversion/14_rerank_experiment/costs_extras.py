"""Additional cost ideas to test if q0_marginal isn't strong enough.

These isolate ω from q0 by:
1. spectrum_mse — FFT both LCs, compare power spectra (q0 mostly phase, ω is freq).
2. envelope_mse — segment-wise (min/max) envelope; same shape with different q0 → same envelope.
3. autocorr_mse — autocorrelation profile; ω determines the autocorrelation period.
4. peak_signature — predicted peaks' brightness distribution vs observed — q0-invariant.
"""
import numpy as np


def cost_spectrum_mse(lc_pred, observed, **_):
    finite = np.isfinite(lc_pred)
    if finite.sum() < 100:
        return np.inf
    # Zero-fill missing to avoid FFT leakage nightmares
    p = lc_pred.copy()
    if (~finite).any():
        p[~finite] = np.mean(lc_pred[finite])
    # Detrend
    p = p - p.mean()
    o = observed - observed.mean()
    P = np.abs(np.fft.rfft(p))
    O = np.abs(np.fft.rfft(o))
    # Compare normalized spectra (bounded)
    P = P / (np.linalg.norm(P) + 1e-12)
    O = O / (np.linalg.norm(O) + 1e-12)
    return float(np.mean((P - O) ** 2))


def cost_envelope_mse(lc_pred, observed, n_segments=20, **_):
    finite = np.isfinite(lc_pred)
    if finite.sum() < 100:
        return np.inf
    p = lc_pred.copy()
    p[~finite] = np.nanmedian(lc_pred[finite])
    N = len(p)
    seg = np.array_split(np.arange(N), n_segments)
    env_p = np.array([[p[s].min(), p[s].max()] for s in seg])
    env_o = np.array([[observed[s].min(), observed[s].max()] for s in seg])
    return float(np.mean((env_p - env_o) ** 2))


def cost_autocorr_mse(lc_pred, observed, max_lag=100, **_):
    finite = np.isfinite(lc_pred)
    if finite.sum() < 100:
        return np.inf
    p = lc_pred.copy()
    p[~finite] = np.mean(lc_pred[finite])
    p = p - p.mean(); o = observed - observed.mean()
    ap = np.correlate(p, p, mode='full')[len(p) - 1 : len(p) - 1 + max_lag]
    ao = np.correlate(o, o, mode='full')[len(o) - 1 : len(o) - 1 + max_lag]
    ap = ap / (ap[0] + 1e-12); ao = ao / (ao[0] + 1e-12)
    return float(np.mean((ap - ao) ** 2))


def cost_peak_signature(lc_pred, observed, **_):
    """Compare distribution of peak brightnesses (sorted). q0 changes peak
    timing, ω changes the set of brightnesses achievable."""
    from scipy.signal import find_peaks
    finite = np.isfinite(lc_pred)
    if finite.sum() < 100:
        return np.inf
    pk_p, _ = find_peaks(-lc_pred, distance=3, prominence=0.2)
    pk_o, _ = find_peaks(-observed, distance=3, prominence=0.2)
    if len(pk_p) == 0 or len(pk_o) == 0:
        return float(len(observed))
    vp = np.sort(lc_pred[pk_p])[:len(pk_o)]  # take top-N matching observed count
    vo = np.sort(observed[pk_o])[:len(vp)]
    n = min(len(vp), len(vo))
    if n == 0:
        return np.inf
    return float(np.mean((vp[:n] - vo[:n]) ** 2))


EXTRAS = {
    "surr_spectrum":  cost_spectrum_mse,
    "surr_envelope":  cost_envelope_mse,
    "surr_autocorr":  cost_autocorr_mse,
    "surr_peak_sig":  cost_peak_signature,
}
