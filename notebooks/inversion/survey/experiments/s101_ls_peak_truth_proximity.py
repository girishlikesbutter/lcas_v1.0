"""s101 — how close is truth |ω| to the RAW significant Lomb-Scargle peaks?

Forget the padded bracket. For each cohort seed, run the SAME LS pipeline as
s019 (imported, not copied), get the significant peak angular frequencies, and
measure how close the nearest one is to truth |ω| = |omega0_rad|. Also test the
harmonic hypothesis: is truth near peak/k for a small integer k (i.e. is a
fundamental hiding under harmonic peaks)?

All rates reported in deg/s. Pure spectral analysis, no propagation/render.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import json
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
sys.path.insert(0, str(SURVEY / "experiments"))
import importlib
s019 = importlib.import_module("s019_ls_bracket_omega_mag")

R2D = 180.0 / np.pi
TRAJ = SURVEY / "data" / "trajectories"
OUT = SURVEY / "results" / "s101"
OUT.mkdir(parents=True, exist_ok=True)


def peaks_for_seed(seed):
    p = TRAJ / f"traj_seed{seed:03d}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    lc = d["mag_hifi"].astype(float)
    t = d["observation_times"].astype(float)
    truth = float(np.linalg.norm(d["omega0_rad"].astype(float)))  # rad/s
    fin = np.isfinite(lc)
    if fin.sum() < 50:
        return None
    freqs, power = s019.run_ls(t[fin], lc[fin])
    pf, _ = s019.significant_peaks(freqs, power)
    if not pf:
        return dict(seed=seed, truth_dps=truth * R2D, n_peaks=0)
    peaks = np.sort(2 * np.pi * np.array(pf))          # rad/s, ascending
    # nearest raw peak
    rel = np.abs(peaks - truth) / truth
    j = int(np.argmin(rel))
    # harmonic test: is truth ~ peak/k for any peak and k in 1..16?
    ks = np.arange(1, 17)
    sub = peaks[:, None] / ks[None, :]                 # (n_peaks, 16) candidate fundamentals
    relsub = np.abs(sub - truth) / truth
    jh = np.unravel_index(np.argmin(relsub), relsub.shape)
    return dict(
        seed=seed, truth_dps=truth * R2D, n_peaks=len(peaks),
        peaks_dps=(peaks * R2D).tolist(),
        nearest_peak_dps=float(peaks[j] * R2D),
        nearest_peak_pct=float(rel[j] * 100),
        nearest_peak_rank_lo=j,                         # 0 = lowest-freq peak
        nearest_is_lowest=bool(j == 0),
        truth_below_all=bool(truth < peaks[0]),
        truth_above_all=bool(truth > peaks[-1]),
        best_subharm_pct=float(relsub[jh] * 100),
        best_subharm_k=int(ks[jh[1]]),
        best_subharm_peak_dps=float(peaks[jh[0]] * R2D),
    )


def main():
    rows = []
    for s in range(120):
        r = peaks_for_seed(s)
        if r is not None:
            rows.append(r)
    have = [r for r in rows if r.get("n_peaks", 0) > 0]
    npk = np.array([r["nearest_peak_pct"] for r in have])
    sub = np.array([r["best_subharm_pct"] for r in have])
    print(f"=== s101 raw-LS-peak proximity to truth |w| | {len(have)} seeds w/ peaks (of {len(rows)}) ===")
    print(f"{'seed':>4} {'truth':>7} {'#pk':>3} {'nearest_pk':>10} {'near%':>7} {'rank':>4} {'lowest?':>7} | {'subharm%':>8} {'k':>2}")
    for r in have:
        print(f"{r['seed']:>4} {r['truth_dps']:>7.3f} {r['n_peaks']:>3} "
              f"{r['nearest_peak_dps']:>10.3f} {r['nearest_peak_pct']:>7.1f} "
              f"{r['nearest_peak_rank_lo']:>4} {str(r['nearest_is_lowest']):>7} | "
              f"{r['best_subharm_pct']:>8.1f} {r['best_subharm_k']:>2}")
    def pct(a): return np.percentile(a, [10, 50, 90])
    print(f"\nNEAREST RAW PEAK to truth (% of truth): p10/med/p90 = {pct(npk)[0]:.1f} / {pct(npk)[1]:.1f} / {pct(npk)[2]:.1f}")
    for bar in (5, 10, 20, 30):
        print(f"  seeds with a raw peak within {bar:>2}% of truth: {int((npk<=bar).sum())}/{len(have)}")
    print(f"\nBEST SUBHARMONIC peak/k (k<=16) to truth: p10/med/p90 = {pct(sub)[0]:.1f} / {pct(sub)[1]:.1f} / {pct(sub)[2]:.1f}")
    for bar in (5, 10, 20):
        print(f"  seeds with peak/k within {bar:>2}% of truth: {int((sub<=bar).sum())}/{len(have)}")
    nlow = sum(r["nearest_is_lowest"] for r in have)
    nbelow = sum(r["truth_below_all"] for r in have)
    print(f"\nnearest peak is the LOWEST-freq peak: {nlow}/{len(have)}")
    print(f"truth BELOW all significant peaks: {nbelow}/{len(have)}")
    with open(OUT / "peak_proximity.json", "w") as f:
        json.dump(dict(n_seeds=len(have), rows=have), f, indent=2, default=float)
    print(f"\nSaved: {OUT/'peak_proximity.json'}")


if __name__ == "__main__":
    main()
