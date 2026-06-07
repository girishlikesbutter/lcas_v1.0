#!/usr/bin/env python3
"""The Tool bench — the laboratory's shelved, labelled boxes (ADR-0007 / W-D scour).

Reads LIVE from research_os/substrate/*.json + pipelines/*.json. Draws every canon Tool
as a box on one of two shelves (LIVE LABORATORY vs DORMANT / PARALLEL), with a variant
badge (+N) and a glance headline. in_use is parsed from the card interface tag the scour
stamped ('[in_use=True/False]').
"""
import glob
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import ro_viz

ro_viz.apply_dark_style()
ROOT = Path(__file__).resolve().parents[2]

cards = [json.load(open(f)) for f in glob.glob(str(ROOT / "research_os/substrate/*.json"))]
def is_live(c):
    m = re.search(r"\[in_use=(\w+)\]", c.get("interface", ""))
    return (m.group(1) == "True") if m else False

canon = [c for c in cards if c.get("canon", True) and not c.get("variant_of")]
variants = [c for c in cards if c.get("variant_of")]
vcount = {}
for v in variants:
    vcount[v["variant_of"]] = vcount.get(v["variant_of"], 0) + 1

pipes = [json.load(open(f)) for f in glob.glob(str(ROOT / "research_os/pipelines/*.json"))]
terms = len(glob.glob(str(ROOT / "research_os/glossary/*.json")))

live = sorted([c for c in canon if is_live(c)], key=lambda c: -vcount.get(c["id"], 0))
dorm = sorted([c for c in canon if not is_live(c)], key=lambda c: -vcount.get(c["id"], 0))

# --- layout: two shelves of boxes ---
COLS = 5
BW, BH, GX, GY = 1.0, 0.62, 0.12, 0.42

def draw_shelf(ax, tools, y_top, title, color):
    ax.text(0, y_top + 0.30, title, fontsize=11, fontweight="bold", color=color)
    for i, c in enumerate(tools):
        r, col = divmod(i, COLS)
        x = col * (BW + GX)
        y = y_top - r * (BH + GY)
        box = FancyBboxPatch((x, y - BH), BW, BH, boxstyle="round,pad=0.012,rounding_size=0.04",
                             linewidth=1.1, edgecolor=color, facecolor=ro_viz.PANEL)
        ax.add_patch(box)
        name = c["id"].replace("_", "\n", 1) if len(c["id"]) > 13 else c["id"]
        ax.text(x + BW / 2, y - BH / 2 + 0.04, name, ha="center", va="center",
                fontsize=7.6, color=ro_viz.FG)
        n = vcount.get(c["id"], 0)
        if n:
            ax.text(x + BW - 0.06, y - 0.10, f"+{n}", ha="right", va="top",
                    fontsize=7.5, color=ro_viz.AMBER, fontweight="bold")
    rows = (len(tools) + COLS - 1) // COLS
    return y_top - rows * (BH + GY)

fig = plt.figure(figsize=(13, 9.2))
fig.suptitle("Research OS · Tool bench  —  the laboratory, shelved & labelled (W-D scour, ADR-0007)",
             fontsize=13, fontweight="bold")

ax = fig.add_axes([0.04, 0.06, 0.92, 0.80])
ax.axis("off")
ax.set_xlim(-0.1, COLS * (BW + GX))
y_after_live = draw_shelf(ax, live, 6.0, f"LIVE LABORATORY  ·  {len(live)} canon Tools in use", ro_viz.GREEN)
draw_shelf(ax, dorm, y_after_live - 0.45,
           f"DORMANT / PARALLEL  ·  {len(dorm)} canon Tools (registered, in_use:false)", ro_viz.GREY)
ax.set_ylim(y_after_live - 0.45 - ((len(dorm) + COLS - 1) // COLS) * (BH + GY) - 0.3, 6.6)

# headline strip
hl = fig.add_axes([0.04, 0.875, 0.92, 0.085]); hl.axis("off")
total = len(canon) + len(variants)
bits = [(f"{total}", "Tool cards", ro_viz.BLUE),
        (f"{len(canon)}+{len(variants)}", "canon + variant", ro_viz.FG),
        (f"{len(live)}", "live", ro_viz.GREEN),
        (f"{len(dorm)}", "dormant", ro_viz.GREY),
        (f"{terms}", "artifact types", ro_viz.AMBER),
        (f"{len(pipes)}", "pipelines", ro_viz.BLUE)]
for i, (num, lab, col) in enumerate(bits):
    x = 0.02 + i * 0.165
    hl.text(x, 0.7, num, fontsize=22, fontweight="bold", color=col, transform=hl.transAxes)
    hl.text(x, 0.15, lab, fontsize=9, color=ro_viz.MUTED, transform=hl.transAxes)

ro_viz.emit(fig, "tool_bench",
            f"Tool bench (W-D scour): {total} Tools ({len(canon)} canon + {len(variants)} variant), "
            f"{len(live)} live / {len(dorm)} dormant, {terms} artifact types, {len(pipes)} pipelines",
            run="W-D")
print(f"canon={len(canon)} (live={len(live)} dorm={len(dorm)}) variants={len(variants)} pipelines={len(pipes)} terms={terms}")
