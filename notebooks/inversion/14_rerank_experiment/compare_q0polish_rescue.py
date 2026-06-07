#!/usr/bin/env python3
"""Compare baseline vs q0polish-sorted pipeline on seeds 59, 64, 67.

Reports in ρ-band convention: ρ = √hifi / 0.05, Bands A/B/C/D at 2/4/8.

Reads:
- Baseline snapshot:
    rerank_experiment/pipeline_test_2026_04_22/baseline_snapshots/invert_{seed}.json
- Current invert driver output:
    invert_m048_seed{seed}/result.json

Writes:
- rerank_experiment/pipeline_test_2026_04_22/comparison.json
- rerank_experiment/pipeline_test_2026_04_22/REPORT.md
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIAG = ROOT / "data/results/inversion_diagnostics"
OUT = DIAG / "rerank_experiment/pipeline_test_2026_04_22"
SNAP = OUT / "baseline_snapshots"
TARGETS = [59, 64, 67]

NOISE_SIGMA = 0.05


def rho(hifi):
    if hifi is None or hifi < 0:
        return None
    return math.sqrt(hifi) / NOISE_SIGMA


def band(r):
    if r is None:
        return "n/a"
    if r < 2:  return "A"
    if r < 4:  return "B"
    if r < 8:  return "C"
    return "D"


def extract(doc):
    if doc is None:
        return None
    w = doc.get("winner") or {}
    h = w.get("hifi")
    r = rho(h)
    return {
        "hifi": h,
        "rho": r,
        "band": band(r),
        "q0_err": w.get("q0_err"),
        "w_dir_err": w.get("w0_err"),
        "w_mag_err_pct": w.get("w_mag_err_pct"),
        "source_label": w.get("source_label"),
    }


def load(path):
    if not path.exists():
        return None
    return json.load(open(path))


def main():
    rows = []
    for s in TARGETS:
        base = extract(load(SNAP / f"invert_{s:03d}.json"))
        new  = extract(load(DIAG / f"invert_m048_seed{s:03d}/result.json"))
        rows.append({"seed": s, "baseline": base, "q0polish": new})

    # Console table
    hdr = (f"{'seed':>4} | {'base ρ':>7} {'bd':>2} {'base q0':>8} "
           f"{'base w_dir':>10} {'base w_mag%':>11}  |  "
           f"{'new ρ':>7} {'bd':>2} {'new q0':>8} {'new w_dir':>10} {'new w_mag%':>11}  | "
           f"{'Δρ':>8}")
    print()
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        b, n = r["baseline"], r["q0polish"]
        if b is None or n is None:
            print(f"{r['seed']:>4} | missing data")
            continue
        dr = n["rho"] - b["rho"]
        print(f"{r['seed']:>4} | "
              f"{b['rho']:>7.2f} {b['band']:>2} {b['q0_err']:>8.2f} "
              f"{b['w_dir_err']:>10.2f} {b['w_mag_err_pct']:>+11.2f}  |  "
              f"{n['rho']:>7.2f} {n['band']:>2} {n['q0_err']:>8.2f} "
              f"{n['w_dir_err']:>10.2f} {n['w_mag_err_pct']:>+11.2f}  | "
              f"{dr:>+8.2f}")

    # Tally
    def band_rank(bd):
        return {'A': 0, 'B': 1, 'C': 2, 'D': 3}.get(bd, 4)
    band_improved = sum(1 for r in rows
                        if r["baseline"] and r["q0polish"]
                        and band_rank(r["q0polish"]["band"]) < band_rank(r["baseline"]["band"]))
    rho_reduced = sum(1 for r in rows
                      if r["baseline"] and r["q0polish"]
                      and r["q0polish"]["rho"] < r["baseline"]["rho"])
    reached_A = sum(1 for r in rows
                    if r["q0polish"] and r["q0polish"]["band"] == "A")
    print(f"\nTally: {band_improved}/{len(rows)} band-improved, "
          f"{rho_reduced}/{len(rows)} ρ-reduced, "
          f"{reached_A}/{len(rows)} reached Band A.")

    # Save machine-readable
    with open(OUT / "comparison.json", "w") as f:
        json.dump({
            "target_seeds": TARGETS,
            "rows": rows,
            "tally": {
                "band_improved": band_improved,
                "rho_reduced": rho_reduced,
                "reached_band_A": reached_A,
                "total": len(rows),
            },
            "sort_modes": {"baseline": "geo_cost", "new": "surr_q0polish_mse"},
            "noise_sigma": NOISE_SIGMA,
        }, f, indent=2)
    print(f"\nSaved: {OUT / 'comparison.json'}")

    # Markdown report
    md = ["# Pipeline test — geo_cost vs surr_q0polish_mse on Band-D seeds",
          "",
          f"Seeds: {TARGETS}. Trajectory source: m048. Change: "
          "`M115_SORT_BY=geo_cost` (baseline) → `M115_SORT_BY=surr_q0polish_mse`. "
          "All other pipeline stages identical. m103 outputs unchanged (only the top-3 ω ranking differs).",
          "",
          "ρ = √hifi / 0.05 (noise σ=0.05). Band A: ρ<2 (observationally indistinguishable from noise). "
          "Band B: 2≤ρ<4. Band C: 4≤ρ<8. Band D: ρ≥8.",
          "",
          "| seed | base ρ | base band | base q0 | base w_dir | base w_mag | new ρ | new band | new q0 | new w_dir | new w_mag | Δρ |",
          "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    for r in rows:
        b, n = r["baseline"], r["q0polish"]
        if b is None or n is None:
            md.append(f"| {r['seed']} | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        dr = n["rho"] - b["rho"]
        md.append(
            f"| {r['seed']} | {b['rho']:.2f} | {b['band']} | "
            f"{b['q0_err']:.1f}° | {b['w_dir_err']:.1f}° | {b['w_mag_err_pct']:+.2f}% | "
            f"{n['rho']:.2f} | **{n['band']}** | "
            f"{n['q0_err']:.1f}° | {n['w_dir_err']:.1f}° | {n['w_mag_err_pct']:+.2f}% | "
            f"{dr:+.2f} |")
    md += ["",
           f"**Tally:** {band_improved}/{len(rows)} band-improved, "
           f"{rho_reduced}/{len(rows)} ρ-reduced, "
           f"{reached_A}/{len(rows)} reached Band A.",
           "",
           "## Per-seed mechanism",
           "",
           "**Seed 59 (D → A):** New top-3 ω had w_dir_err 3.65°, 34.62°, 56.19° (baseline: 27.7°, 72.3°, 60.6°). "
           "m115 DE from ω=3.65° found a truth-adjacent q0 basin in 3/10 starts (q0_err=8.81°). "
           "Surrogate MSE ranked this basin #3 (two ~174° spurious basins had lower surrogate MSE), but "
           "m115's top-3 hi-fi validation included basin 2, and m126 polish took it to q0_err=0.21°. ρ=1.02 "
           "— comfortably inside noise-level Band A.",
           "",
           "**Seed 64 (D → D, marginal):** New top-3 ω had w_dir_err 50.93°, 14.63°, 60.91° — the 14.63° "
           "truth-close ω was at slot 1. But m115's 10-start DE from ω=14.63° landed 0/10 near-truth; all "
           "basins hifi ~1.8-1.9. Surrogate MSE picked the wrong-ω basin (ω=50.93°, w_dir=50.9°, "
           "w_mag=-64%) as winner. ρ dropped 30.15→26.92 but still deep in Band D. **Ranking fix put "
           "truth-close ω in the hand-off; downstream m115 DE couldn't exploit it.**",
           "",
           "**Seed 67 (D → B):** New top-3 ω had w_dir_err 2.97°, 67.85°, 75.58° (very truth-close at slot 0). "
           "m115 DE found a basin with w_dir=0.3° and w_mag=-0.08% — ω essentially perfect. But q0_err=179.6°, "
           "likely the IS-901 ±X-twin direction. ρ=3.56 means the LC is still distinguishable from truth "
           "at ~3.6σ — not inside Band A, so observationally it's flagged as different, but within-4× noise.",
           "",
           "## Scope",
           "",
           "These 3 seeds were the ones where `pick_target_seeds.py` predicted the single-cost swap would "
           "*specifically* introduce a truth-close ω (<20°) into the top-3 that `geo_cost` had buried. Of "
           "the 17 Band-D failed seeds in the random m048 cohort, the other 14 either already had truth "
           "in geo_cost's top-3 (downstream m115/m126 failures, not ranking failures — seeds 7, 57, 71) or "
           "had no truth-close ω anywhere in the 26-candidate pool (sampling-miss — seeds 47, 51, 84, 89, "
           "plus some others). **Single-cost `surr_q0polish_mse` is therefore a strict subset of what the "
           "m133 finding promised; the 15/17 prediction was for the K=3 triple, not a single cost.**",
           "",
           "## Takeaways",
           "",
           "1. Ranking fix is necessary but not sufficient. Two of three rescuable-seeded runs had their "
           "top-3 populated correctly with truth-close ω but still FAILed downstream because m115's "
           "10-start DE from the truth-close ω didn't locate the correct q0 basin. The ω-error budget "
           "that m115 can bridge seems to be roughly 3-5°, not 15°.",
           "2. Seed 59 is a clean Band-A rescue — the single-cost swap genuinely fixed it.",
           "3. Seed 67 exposes a residual observability limit: ω is recovered perfectly but q0 ends up at "
           "the geometric twin. ρ=3.56 means the LC is still distinguishable (not inside Band A).",
           "4. Natural next experiments: (a) the K=3 triple union for more seeds; (b) increase m115 "
           "N_STARTS from 10 to 30+ to widen the ω-bridging radius; (c) densify m103 candidate pool so "
           "truth-close ω arrives at smaller than 14° error.",
           ""]
    with open(OUT / "REPORT.md", "w") as f:
        f.write("\n".join(md))
    print(f"Saved: {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
