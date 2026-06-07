#!/usr/bin/env python3
"""
m048 random 25-seed batch — solution-count-by-band dissection (m132-style).

Per basin, not per seed winner. Every polished basin is placed into a hi-fi
MSE bin, and we also tag each basin with its w_dir_err / q0_err state so we
can attribute failures.
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np

COHORT = [6, 7, 8, 11, 16, 17, 34, 35, 42, 45, 47, 48, 51, 57, 59, 64, 67,
          69, 71, 78, 79, 84, 89, 91, 99]

ROOT = Path(__file__).parent.parent.parent
DIAG = ROOT / "data/results/inversion_diagnostics"

# m132 bands (hi-fi MSE):
# noise floor is ~0.0025 (σ=0.05 mag, 500 epochs). 2× noise floor = 0.005.
BANDS = [
    ("<0.005",      lambda x: x < 0.005),           # ≤2× noise floor — "tight"
    ("0.005–0.01",  lambda x: 0.005 <= x < 0.01),   # OK
    ("0.01–0.05",   lambda x: 0.01 <= x < 0.05),
    ("0.05–0.1",    lambda x: 0.05 <= x < 0.1),     # PARTIAL
    ("0.1–0.3",     lambda x: 0.1 <= x < 0.3),
    (">=0.3",       lambda x: x >= 0.3),            # FAIL
]


def band_of(x):
    if x is None or not np.isfinite(x):
        return "missing"
    for name, fn in BANDS:
        if fn(x):
            return name
    return "missing"


def upstream_status(seed):
    """Return upstream failure tag if m126 absent."""
    m126 = DIAG / "m126_wrapped_m048" / f"seed_{seed:03d}" / "result.json"
    if m126.exists():
        return None
    m103_dir = DIAG / "m103_hybrid_m048" / f"seed_{seed:03d}"
    if (m103_dir / "geo_timeout.flag").exists():
        return "geo_timeout"
    if not (m103_dir / "result.json").exists():
        return "m103_crash"
    return "no_m126"


def load_basins(seed):
    m126 = DIAG / "m126_wrapped_m048" / f"seed_{seed:03d}" / "result.json"
    if not m126.exists():
        return []
    r = json.load(open(m126))
    out = []
    for b in r.get("basins", []):
        out.append({
            "hifi": b["hifi_wrapped"],
            "q0_err": b.get("q0_err_after"),
            "w_dir_err": b.get("w_dir_err_after"),
            "w_mag_err_pct": b.get("w_mag_err_pct_after"),
            "is_twin": b.get("is_twin"),
        })
    return out


def attribute(basin):
    """Tag each basin with its state classifier (what 'kind' of solution is it?)."""
    if basin["hifi"] < 0.005:
        return "tight"
    if basin["hifi"] < 0.01:
        return "OK"
    if basin["hifi"] < 0.05:
        return "near-OK"
    if basin["hifi"] < 0.1:
        return "PARTIAL"
    # FAIL domain — attribute the cause
    w_err = basin["w_dir_err"] or 0
    q_err = basin["q0_err"] or 0
    if w_err > 20:
        return "FAIL_ω"
    if q_err > 20 and abs(q_err - 180) > 20:  # not a twin
        return "FAIL_q0"
    if abs(q_err - 180) <= 20:
        return "FAIL_twin_unreduced"  # twin where truth isn't ±X compatible
    return "FAIL_other"


def main():
    band_names = [b[0] for b in BANDS]
    # Headline: aggregate count per band
    totals = Counter()
    # Per-seed row: counts per band
    per_seed = {}
    upstream_tags = {}

    for seed in COHORT:
        up = upstream_status(seed)
        if up is not None:
            upstream_tags[seed] = up
            continue
        basins = load_basins(seed)
        band_counts = Counter(band_of(b["hifi"]) for b in basins)
        per_seed[seed] = {"basins": basins, "bands": band_counts}
        for name in band_counts:
            totals[name] += band_counts[name]

    n_analysed_seeds = len(per_seed)
    n_basins = sum(sum(s["bands"].values()) for s in per_seed.values())

    # --- Headline table ---
    print(f"# m048 random 25-seed cohort — solution count by hi-fi band")
    print()
    print(f"Analysed: {n_analysed_seeds} seeds × 3 basins = {n_basins} polished basins.")
    if upstream_tags:
        print(f"Upstream-failed (no m126): {len(upstream_tags)} seeds — ", end="")
        print(", ".join(f"{s} ({t})" for s, t in upstream_tags.items()))
    print()
    print("| band | count | % of basins |")
    print("|------|------:|------------:|")
    for name in band_names:
        c = totals[name]
        pct = 100 * c / n_basins if n_basins else 0
        print(f"| `{name}` | {c} | {pct:.0f}% |")
    print()

    # Cumulative: basins below 0.01 (OK-tight), below 0.1 (≤ PARTIAL).
    below_005 = totals["<0.005"]
    below_01 = below_005 + totals["0.005–0.01"]
    below_05 = below_01 + totals["0.01–0.05"]
    below_1 = below_05 + totals["0.05–0.1"]
    print(f"**Cumulative**: "
          f"{below_005}/{n_basins} ({100*below_005/n_basins:.0f}%) at ≤2× noise floor; "
          f"{below_01}/{n_basins} ({100*below_01/n_basins:.0f}%) OK-tight (<0.01); "
          f"{below_1}/{n_basins} ({100*below_1/n_basins:.0f}%) ≤PARTIAL (<0.1).")
    print()

    # --- Per-seed row: basin-band histogram + attribution ---
    print("## Per-seed basin distribution and failure attribution")
    print()
    print("Each row = 3 polished basins for that seed. Cells list basin hi-fi MSE "
          "(bold = best). Final column = attribution based on the **best** basin.")
    print()
    print("| seed | basin0 | basin1 | basin2 | best_band | attribution |")
    print("|------|--------|--------|--------|-----------|-------------|")
    for seed in COHORT:
        if seed in upstream_tags:
            print(f"| {seed} | — | — | — | — | {upstream_tags[seed]} |")
            continue
        basins = per_seed[seed]["basins"]
        hifis = [b["hifi"] for b in basins]
        best_idx = int(np.argmin(hifis))
        cells = []
        for i, b in enumerate(basins):
            s = f"{b['hifi']:.4f}"
            if i == best_idx:
                s = f"**{s}**"
            cells.append(s)
        best = basins[best_idx]
        attribution = attribute(best)
        print(f"| {seed} | {cells[0]} | {cells[1]} | {cells[2]} | "
              f"`{band_of(best['hifi'])}` | {attribution} "
              f"(q0={best['q0_err']:.1f}° ω={best['w_dir_err']:.1f}°) |")
    print()

    # --- Attribution breakdown ---
    attr_count = Counter()
    for seed in per_seed:
        b = per_seed[seed]["basins"]
        best = min(b, key=lambda x: x["hifi"])
        attr_count[attribute(best)] += 1
    for s, t in upstream_tags.items():
        attr_count[t] += 1

    print("## Winner-basin attribution (best of 3 per seed)")
    print()
    for k, v in attr_count.most_common():
        print(f"- {k}: {v}")
    print()

    # --- All-basins attribution (multi-solution view) ---
    all_attr = Counter()
    for seed in per_seed:
        for b in per_seed[seed]["basins"]:
            all_attr[attribute(b)] += 1

    print("## All-basins attribution (multi-solution view, 3 × 22 = 66 basins)")
    print()
    print("| attribution | count |")
    print("|-------------|------:|")
    for k, v in sorted(all_attr.items(), key=lambda x: -x[1]):
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
