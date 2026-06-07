"""s101b — null control for the s101 subharmonic finding.

s101 found truth |ω| ≈ peak/k for small k on ~all seeds. Risk: with ~140
candidate subharmonics (peaks × k≤16) clustering densely in the physical band,
peak/k may land near ANY target by chance. Control: for each seed, compare the
TRUE |ω| subharmonic-hit to RANDOM targets drawn uniformly in [0.1,1.5] deg/s
against that SAME seed's peaks. If random hits ≈ true hits, the match is
band-tiling, not signal. All deg/s.
"""
import json
from pathlib import Path
import numpy as np

SURVEY = Path(__file__).resolve().parents[1]
J = SURVEY / "results" / "s101" / "peak_proximity.json"
PRIOR_LO, PRIOR_HI = 0.1, 1.5
KS = np.arange(1, 17)
RNG = np.random.default_rng(0)
N_RAND = 200


def best_subharm_pct(peaks_dps, target):
    sub = np.asarray(peaks_dps)[:, None] / KS[None, :]
    return float(np.min(np.abs(sub - target) / target) * 100)


def main():
    rows = [r for r in json.load(open(J))["rows"] if r.get("n_peaks", 0) > 0]
    bars = (5, 10, 20)
    true_hits = {b: 0 for b in bars}
    rand_hits = {b: 0.0 for b in bars}
    n = len(rows)
    for r in rows:
        peaks = r["peaks_dps"]
        tp = best_subharm_pct(peaks, r["truth_dps"])
        for b in bars:
            true_hits[b] += int(tp <= b)
        # random targets in the physical band, same peaks
        rt = RNG.uniform(PRIOR_LO, PRIOR_HI, N_RAND)
        rp = np.array([best_subharm_pct(peaks, t) for t in rt])
        for b in bars:
            rand_hits[b] += float((rp <= b).mean())
    print(f"=== s101b subharmonic null control | {n} seeds | random targets U[0.1,1.5] deg/s, k<=16 ===")
    print(f"{'bar':>5} | {'TRUE |w| hit':>14} | {'RANDOM target hit (exp.)':>26}")
    for b in bars:
        print(f"{b:>4}% | {true_hits[b]:>6}/{n} ({100*true_hits[b]/n:>5.1f}%) | "
              f"{rand_hits[b]:>8.1f}/{n} ({100*rand_hits[b]/n:>5.1f}%)")
    print("\nIf TRUE >> RANDOM, the subharmonic match is real signal; if ~equal, it's band-tiling.")


if __name__ == "__main__":
    main()
