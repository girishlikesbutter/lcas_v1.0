#!/usr/bin/env python3
"""Demo plot: the s113 A×B cross cost-wall, as a glance instead of a metrics block.

Every number is read live from the real run record — no hardcoding.
Source: research_os/records/s113_slow_tumbler_cross_cost_wall.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "research_os/records/s113_slow_tumbler_cross_cost_wall.json"
m = json.load(open(REC))["metrics"]

rate = m["cross_rate_pairs_per_s"]


def wall(p):
    s = p / rate
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
fig.suptitle(
    "s113 · seed 116 · A×B cross cost wall   (contract budget: 3600 s total wall over 4 seeds)",
    fontsize=12, fontweight="bold",
)

# --- left: pair counts per cross strategy ---
strategies = ["v0 single-pass\n2° cross",
              "v0 + PAIR_BUDGET\ncap",
              "coarse-6° → fine-2°\n(v1 amendment)"]
pairs = [m["v0_cross_pairs_116"], m["v0_cross_capped_pairs_116"], m["coarse6_cross_pairs_116"]]
colors = ["#d9534f", "#e8a33d", "#5cb85c"]
y = np.arange(len(strategies))
ax1.barh(y, pairs, color=colors)
ax1.set_yticks(y)
ax1.set_yticklabels(strategies, fontsize=10)
ax1.set_xscale("log")
ax1.set_xlabel("A×B candidate pairs to score  (log scale)")
ax1.invert_yaxis()
for yi, p in zip(y, pairs):
    ax1.text(p * 1.15, yi, f"{p / 1e6:.1f}M  ·  {wall(p)}", va="center", fontsize=10)
ax1.set_xlim(right=pairs[0] * 5)
ax1.set_title("Even the v1 cut is ~38 min/seed → over budget ×4 seeds", fontsize=10)

# --- right: root cause, anchor cell counts ---
labels = [f"A (ep{m['anchor_A_ep']})", f"B (ep{m['anchor_B_ep']})"]
cells_2 = [m["A_cells_2deg_116"], m["B_cells_2deg_116"]]
cells_6 = [m["A_cells_6deg_116"], m["B_cells_6deg_116"]]
x = np.arange(len(labels))
w = 0.35
ax2.bar(x - w / 2, cells_2, w, label="2° cells", color="#4c78a8")
ax2.bar(x + w / 2, cells_6, w, label="6° cells", color="#9ecae9")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("decimated cells")
af_a = m["A_dense_admission_frac_116"] * 100
af_b = m["B_dense_admission_frac_116"] * 100
ax2.set_title(
    f"Root cause: anchor B is weak (admission A {af_a:.1f}% vs B {af_b:.1f}%)\n"
    f"close anchor ⟹ many survivors ⟹ huge cross",
    fontsize=10,
)
ax2.legend()
for xi, c in zip(x - w / 2, cells_2):
    ax2.text(xi, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
for xi, c in zip(x + w / 2, cells_6):
    ax2.text(xi, c, f"{c:,}", ha="center", va="bottom", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
OUT = Path(__file__).resolve().parent / "stream" / "s113_costwall.png"
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"Saved: {OUT}")
