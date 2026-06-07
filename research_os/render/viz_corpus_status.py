#!/usr/bin/env python3
"""The run-record corpus by outcome — the binding constraint, measured from the store.

Scans every run record's `status`. Source: research_os/records/*.json
"""
import collections
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

status = collections.Counter()
for f in glob.glob(str(ROOT / "research_os/records/*.json")):
    c = json.load(open(f))
    status[c.get("status", "unknown")] += 1
total = sum(status.values())

dead = status.get("refuted", 0) + status.get("inconclusive", 0)
dead_pct = 100 * dead / total if total else 0

ORDER = ["confirmed", "inconclusive", "refuted", "blocked", "unknown"]
COLOR = {"confirmed": "#3fb950", "inconclusive": "#e8a33d", "refuted": "#d9534f",
         "blocked": "#8b949e", "unknown": "#6e7681"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle(f"Research OS · run-record corpus by outcome  (N={total})",
             fontsize=12, fontweight="bold")

# left: status bars
labels = [s for s in ORDER if status.get(s)] + [s for s in status if s not in ORDER]
counts = [status[s] for s in labels]
y = range(len(labels))
ax1.barh(list(y), counts, color=[COLOR.get(s, "#6e7681") for s in labels])
ax1.set_yticks(list(y))
ax1.set_yticklabels(labels, fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel("run records")
for yi, c in zip(y, counts):
    ax1.text(c + total * 0.005, yi, f"{c}  ({100*c/total:.0f}%)", va="center", fontsize=10)
ax1.set_xlim(0, max(counts) * 1.25)
ax1.set_title("most runs do not confirm — by design, that's the cost being attacked", fontsize=10)

# right: the headline
ax2.axis("off")
ax2.set_title("The binding constraint: rework, not research", fontsize=10)
ax2.text(0.5, 0.62, f"{dead_pct:.0f}%", ha="center", va="center",
         fontsize=58, fontweight="bold", color="#e8a33d")
ax2.text(0.5, 0.36, "of runs are refuted or inconclusive", ha="center", fontsize=11)
ax2.text(0.5, 0.14,
         "The trust machine exists to make this cheap:\n"
         "fail fast, fail cited, never re-chase a dead branch.",
         ha="center", fontsize=9.5, style="italic", color="#8b949e")

plt.tight_layout(rect=[0, 0, 1, 0.93])
OUT = Path(__file__).resolve().parent / "stream" / "corpus_status.png"
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"status counts: {dict(status)}  | dead={dead}/{total} = {dead_pct:.1f}%")
