"""m138 Lomb-Scargle independent probe — seed 47 (FAIL) + seed 91 (PASS).

Question: does Lomb-Scargle on the LC find truth |ω| = ω_rot/(2π) cleanly,
or does it pick up the same 2× harmonic that the peak-count estimator does?

Two LC inputs:
  full   — entire 500-epoch noisy obs_lc (m052's approach)
  bright — only mag<11 epochs (matches m138's peak-count input)

Reports:
  - top-5 LS peaks (frequency Hz, normalised power)
  - implied |ω| for each peak (= 2π·f if 1 peak per rotation, = π·f if 2 per rotation)
  - ratio peak/truth and peak/peak-count-estimate
  - harmonic ladder check: does power at f, 2f, 3f line up?
  - peak-count estimator value (m138's exact rule) for comparison
"""
import sys
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
    """Exact replica of m138_isoshell_h1.estimate_omega_mag_grid base."""
    valid = np.isfinite(obs_lc)
    bright_mask = valid & (obs_lc < obs_lc[valid].mean() - 1.0)
    transitions = np.diff(bright_mask.astype(int))
    n_peaks = max(1, (transitions > 0).sum())
    window = obs_times[-1] - obs_times[0]
    base = 2 * np.pi * n_peaks / window
    return base, n_peaks, int(bright_mask.sum())


def run_ls(times, signal, n_freqs=4000, f_min=None, f_max=None):
    """LS on demeaned brightness (-mag, demeaned). Returns freqs (Hz), power."""
    s = -signal
    s = s - np.mean(s)
    dt = np.median(np.diff(times))
    if f_min is None:
        f_min = 1.0 / (times[-1] - times[0])
    if f_max is None:
        f_max = 0.5 / dt  # Nyquist
    freqs = np.linspace(f_min, f_max, n_freqs)
    ang = 2 * np.pi * freqs
    power = lombscargle(times, s, ang, normalize=True)
    return freqs, power


def top_peaks(freqs, power, k=5):
    idx, _ = find_peaks(power, distance=5)
    if len(idx) == 0:
        return [], []
    order = idx[np.argsort(power[idx])[::-1]][:k]
    return freqs[order].tolist(), power[order].tolist()


def harmonic_ladder_score(freqs, power, f_base, n_harmonics=4, tol_rel=0.05):
    """Power summed at f_base, 2*f_base, 3*f_base, ... within ±tol_rel."""
    out = []
    for h in range(1, n_harmonics + 1):
        f_h = h * f_base
        if f_h > freqs.max():
            break
        mask = np.abs(freqs - f_h) <= tol_rel * f_h
        if mask.any():
            out.append((h, f_h, float(power[mask].max())))
    return out


def run_seed(seed, source="m048"):
    print(f"\n========== seed {seed:03d} ({source}) ==========")
    truth = load_truth(seed, source)
    obs_lc = truth["observed_lc"]
    true_omega = truth["omega0_rad"]
    truth_mag = float(np.linalg.norm(true_omega))
    truth_f = truth_mag / (2 * np.pi)
    truth_period = 1.0 / truth_f
    print(f"  truth |ω|       = {truth_mag:.5f} rad/s  ({np.degrees(truth_mag):.4f} deg/s)")
    print(f"  truth f_rot     = {truth_f:.6f} Hz       (period {truth_period:.1f} s)")

    if source == "m048":
        ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                               random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                               start_et=truth["start_et"], skip_true_lc=True)
    else:
        ctx = setup_experiment(n_observations=500, noise_sigma=CANONICAL_NOISE_SIGMA,
                               random_seed=42, true_omega_deg=(0.5, -0.3, 2.0),
                               end_time_utc=truth["end_time_utc"], skip_true_lc=True)
    obs_times = ctx.observation_times
    print(f"  window          = {obs_times[-1]-obs_times[0]:.1f} s, dt={np.median(np.diff(obs_times)):.2f} s, n={len(obs_times)}")

    base_pc, n_pk, n_bright = peakcount_estimator(obs_lc, obs_times)
    pc_f = base_pc / (2 * np.pi)
    print(f"  peak-count      n_peaks={n_pk}, n_bright={n_bright}, |ω|_pc={base_pc:.5f} rad/s, f_pc={pc_f:.6f} Hz")
    print(f"  peak-count/truth ratio: {base_pc/truth_mag:.4f}")

    finite = np.isfinite(obs_lc)
    valid_lc = obs_lc[finite]
    valid_t = obs_times[finite]

    # Variant 1: full LC, demeaned brightness
    f_full, p_full = run_ls(valid_t, valid_lc, n_freqs=4000)
    pf_full, pw_full = top_peaks(f_full, p_full, k=5)
    print(f"\n  --- LS on FULL LC ({len(valid_t)} epochs) ---")
    for i, (f, p) in enumerate(zip(pf_full, pw_full)):
        omega_imp = 2 * np.pi * f
        print(f"    peak {i+1}: f={f:.6f} Hz, power={p:.4f}, |ω|_imp={omega_imp:.5f}, "
              f"ratio_to_truth={omega_imp/truth_mag:.3f}, ratio_to_pc={f/pc_f:.3f}")

    if pf_full:
        for h_data in [harmonic_ladder_score(f_full, p_full, truth_f),
                       harmonic_ladder_score(f_full, p_full, pf_full[0])]:
            pass
        print(f"    harmonic ladder at f_truth={truth_f:.6f} Hz:")
        for h, fh, ph in harmonic_ladder_score(f_full, p_full, truth_f):
            print(f"      {h}× → f={fh:.6f}, max_power={ph:.4f}")
        print(f"    harmonic ladder at f_pc={pc_f:.6f} Hz:")
        for h, fh, ph in harmonic_ladder_score(f_full, p_full, pc_f):
            print(f"      {h}× → f={fh:.6f}, max_power={ph:.4f}")

    # Variant 2: bright-only LC
    bright_mask = finite & (obs_lc < obs_lc[finite].mean() - 1.0)
    bt = obs_times[bright_mask]
    bl = obs_lc[bright_mask]
    if len(bt) > 5:
        f_min_b = 1.0 / (bt[-1] - bt[0])
        f_max_b = 0.5 / np.median(np.diff(bt))
        f_br, p_br = run_ls(bt, bl, n_freqs=4000, f_min=f_min_b, f_max=f_max_b)
        pf_br, pw_br = top_peaks(f_br, p_br, k=5)
        print(f"\n  --- LS on BRIGHT-only LC ({len(bt)} epochs, dt_med={np.median(np.diff(bt)):.1f}s) ---")
        for i, (f, p) in enumerate(zip(pf_br, pw_br)):
            omega_imp = 2 * np.pi * f
            print(f"    peak {i+1}: f={f:.6f} Hz, power={p:.4f}, |ω|_imp={omega_imp:.5f}, "
                  f"ratio_to_truth={omega_imp/truth_mag:.3f}, ratio_to_pc={f/pc_f:.3f}")

    # Verdict
    print(f"\n  --- verdict ---")
    if pf_full:
        f_top1 = pf_full[0]
        omega_top1 = 2 * np.pi * f_top1
        ratio_truth = omega_top1 / truth_mag
        ratio_pc = omega_top1 / base_pc
        print(f"    LS_top1 / truth = {ratio_truth:.4f}  (target ≈ 1.00)")
        print(f"    LS_top1 / pc    = {ratio_pc:.4f}    (pc was 1.96 off on seed 47)")
        # Where is f_truth in the LS power array?
        ix_truth = int(np.argmin(np.abs(f_full - truth_f)))
        rank = int((p_full > p_full[ix_truth]).sum())
        print(f"    power@f_truth   = {p_full[ix_truth]:.4f}, rank {rank+1}/{len(f_full)} (lower is better)")
        if pf_br:
            f_top1b = pf_br[0]
            omega_top1b = 2 * np.pi * f_top1b
            print(f"    LS_top1_BR / truth = {omega_top1b/truth_mag:.4f}")

    return {
        "seed": seed,
        "truth_mag": truth_mag,
        "truth_f": truth_f,
        "pc_base": base_pc,
        "pc_n_peaks": n_pk,
        "pc_n_bright": n_bright,
        "ls_full_top_freqs": pf_full,
        "ls_full_top_powers": pw_full,
        "ls_full_top1_omega": 2 * np.pi * pf_full[0] if pf_full else None,
    }


if __name__ == "__main__":
    seeds = [47, 91, 51, 79, 84, 89]
    results = {}
    for s in seeds:
        try:
            results[s] = run_seed(s)
        except Exception as e:
            print(f"  seed {s} ERROR: {e}")
            results[s] = {"error": str(e)}

    # Save summary
    import json
    summary = {}
    for s, r in results.items():
        clean = {k: v for k, v in r.items() if k not in ()}
        summary[s] = clean
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved: {OUT}/summary.json")

    # Final consolidated table
    print(f"\n{'='*72}")
    print(f"{'seed':>5}  {'truth_|ω|':>10}  {'pc_|ω|':>10}  {'pc/tru':>7}  {'LS_full':>10}  {'LS/tru':>7}  {'pc_n_pk':>7}")
    for s, r in results.items():
        if "error" in r:
            print(f"  {s}  ERROR")
            continue
        ls = r.get("ls_full_top1_omega")
        ls_str = f"{ls:.5f}" if ls else "n/a"
        ls_ratio = f"{ls/r['truth_mag']:.3f}" if ls else "n/a"
        print(f"  {s:>3}  {r['truth_mag']:.5f}  {r['pc_base']:.5f}  "
              f"{r['pc_base']/r['truth_mag']:>7.3f}  {ls_str:>10}  {ls_ratio:>7}  {r['pc_n_peaks']:>7}")
