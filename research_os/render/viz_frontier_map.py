#!/usr/bin/env python3
"""The open/revivable frontier as a glance — what's on the table and how stale.

orient/strategize render the frontier as text; this is the visual twin. Every
number reads live from the derived live-head. Source: live_head.py --json
"""
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

import ro_viz

ro_viz.apply_dark_style()
H = ro_viz.head_json()
fr = H["frontier"]
trunk = H["trunk"]
trust = H["trust"]

now = datetime.now(timezone.utc)


def days_stale(at):
    try:
        d = datetime.fromisoformat(at.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return (now - d).days
    except Exception:
        return 0


rows = sorted(fr, key=lambda f: days_stale(f.get("last_at", "")), reverse=True)
labels = [f["id"].replace("goal_", "") for f in rows]
ages = [days_stale(f.get("last_at", "")) for f in rows]
colors = [ro_viz.BLUE if f["state"] == "open" else ro_viz.AMBER for f in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                               gridspec_kw={"width_ratios": [1.35, 1]})
fig.suptitle(f"Research OS · frontier — {len(fr)} branches on the table",
             fontsize=12, fontweight="bold")

# left: staleness bars
y = range(len(rows))
ax1.barh(list(y), ages, color=colors)
ax1.set_yticks(list(y))
ax1.set_yticklabels(labels, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("days since last touched")
for yi, a, f in zip(y, ages, rows):
    ax1.text(a + max(ages) * 0.01, yi, f"{a}d · {f['state']}", va="center", fontsize=8.5,
             color=ro_viz.MUTED)
ax1.set_xlim(0, max(ages) * 1.32 if ages else 1)
ax1.set_title("blue = open · amber = revivable", fontsize=10)

# right: the trunk + trust glance
ax2.axis("off")
ax2.set_title("The question the frontier serves", fontsize=10)
import textwrap
q = textwrap.fill(trunk["title"], 34)
ax2.text(0.5, 0.80, q, ha="center", va="top", fontsize=10.5, color=ro_viz.FG)
n_open = sum(1 for f in fr if f["state"] == "open")
n_rev = sum(1 for f in fr if f["state"] == "revivable")
ax2.text(0.5, 0.44, f"{n_open} open · {n_rev} revivable", ha="center",
         fontsize=13, fontweight="bold", color=ro_viz.BLUE)
nr = len(trust.get("needs_replication", []))
cb = H.get("current_branch") or {}
cb_id = cb.get("id", "?").replace("goal_", "") if isinstance(cb, dict) else str(cb)
cb_state = cb.get("state", "") if isinstance(cb, dict) else ""
ax2.text(0.5, 0.27, f"current: {cb_id}  ({cb_state})",
         ha="center", fontsize=9.5,
         color=ro_viz.RED if cb_state == "blocked" else ro_viz.MUTED)
ax2.text(0.5, 0.10,
         f"{nr} live findings flipped to needs_replication\n(blast-radius — pick these up under trust-repair)",
         ha="center", fontsize=9, style="italic", color=ro_viz.AMBER)

plt.tight_layout(rect=[0, 0, 1, 0.93])
ro_viz.emit(fig, "frontier_map",
            f"Frontier: {n_open} open / {n_rev} revivable branches, by staleness; "
            f"trunk = recover (q0, ω) from one LC. Source: live_head.py --json")
