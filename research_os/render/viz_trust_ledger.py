#!/usr/bin/env python3
"""Trust ledger of the research claims deck, as a glance — status mix + blast-radius.

All numbers read live from the files. Source:
  research_os/claims/research/*.json   (claim status + depends_on)
  research_os/substrate/*.json         (current substrate head versions)
"""
import collections
import glob
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# --- claims ---
status = collections.Counter()
needs_rep = []
for f in glob.glob(str(ROOT / "research_os/claims/research/*.json")):
    c = json.load(open(f))
    status[c.get("status", "?")] += 1
    if c.get("status") == "needs_replication":
        needs_rep.append((c["id"], c.get("depends_on", [])))
total = sum(status.values())

# --- substrate head (propagator) ---
head = {}
for f in glob.glob(str(ROOT / "research_os/substrate/*.json")):
    s = json.load(open(f))
    name = s.get("id") or s.get("name") or Path(f).stem
    ver = s.get("version") or s.get("head") or s.get("current")
    head[str(name).replace("substrate_", "")] = ver
prop_head = next((v for k, v in head.items() if "propagator" in k.lower()), "?")

ORDER = ["live", "needs_replication", "superseded", "retracted", "draft"]
COLOR = {"live": "#3fb950", "needs_replication": "#e8a33d", "superseded": "#8b949e",
         "retracted": "#d9534f", "draft": "#58a6ff"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
fig.suptitle(f"Research OS · trust ledger — research claims deck (N={total})",
             fontsize=12, fontweight="bold")

# left: status mix
labels = [s for s in ORDER if status.get(s)]
counts = [status[s] for s in labels]
y = range(len(labels))
ax1.barh(list(y), counts, color=[COLOR[s] for s in labels])
ax1.set_yticks(list(y))
ax1.set_yticklabels([s.replace("_", " ") for s in labels], fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel("claims")
for yi, c in zip(y, counts):
    ax1.text(c + 0.05, yi, str(c), va="center", fontsize=11, fontweight="bold")
ax1.set_xlim(0, max(counts) + 1.2)
ax1.set_title(f"{status.get('live',0)} live · "
              f"{status.get('needs_replication',0)} await re-replication", fontsize=10)

# right: blast-radius
ax2.axis("off")
ax2.set_title("Blast-radius: one substrate bump, frozen claims", fontsize=10)
ax2.text(0.5, 0.86, f"propagator  v1.0.0   ──►   v{prop_head}",
         ha="center", fontsize=13, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", fc="#161b22", ec="#e8a33d"))
ax2.text(0.5, 0.70, "(substrate head moved; evidence below ran on the old version)",
         ha="center", fontsize=9, color="#8b949e")
ax2.text(0.04, 0.55, f"{len(needs_rep)} claims stranded on propagator@1.0.0 → needs_replication:",
         fontsize=10, fontweight="bold")
for i, (cid, _) in enumerate(needs_rep):
    ax2.text(0.07, 0.45 - i * 0.10, "• " + cid.replace("claim_", ""),
             fontsize=10, color="#e8a33d", family="monospace")
ax2.text(0.04, 0.05, "This query is the line that used to cost s001–s066.",
         fontsize=9, style="italic", color="#8b949e")

plt.tight_layout(rect=[0, 0, 1, 0.93])
OUT = Path(__file__).resolve().parent / "stream" / "trust_ledger.png"
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"Saved: {OUT}")
