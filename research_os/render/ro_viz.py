#!/usr/bin/env python3
"""Shared style + emit helpers for Research OS plot-stream visualisations.

The `dynamic-viz` skill writes short scripts that import this module so every
plot shares one dark palette (matching stream/index.html) and reaches the stream
through a single `emit()` call. Harvested from the three demo viz scripts
(viz_corpus_status / viz_trust_ledger / viz_s113_costwall) — the palette + the
save+stream_add dance were duplicated in each; this factors them out.

Conventions every generated viz keeps:
  - numbers read LIVE from the store (records/claims/goals/substrate) — never hardcoded
  - one panel is a "glance" headline (the single thing the artifact wants you to see),
    not just a data table
  - save to stream/<name>.png, then emit to the manifest

Usage:
    import ro_viz
    ro_viz.apply_dark_style()
    fig, ax = plt.subplots(...)
    ...
    ro_viz.emit(fig, "my_plot", "one-line caption", run="s123")
"""
import glob
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

import stream_add  # noqa: E402  (same dir)

ROOT = Path(__file__).resolve().parents[2]
RENDER = Path(__file__).resolve().parent
STREAM = RENDER / "stream"

# Palette — matches stream/index.html so plots sit in the page seamlessly.
BG = "#0d1117"        # page background
PANEL = "#161b22"     # card background
BORDER = "#30363d"
FG = "#c9d1d9"        # primary text
MUTED = "#8b949e"     # secondary text
BLUE = "#58a6ff"
GREEN = "#3fb950"
AMBER = "#e8a33d"
RED = "#d9534f"
GREY = "#6e7681"

# Status → colour. One map for both run records and claim cards (shared vocab).
STATUS_COLORS = {
    # run / corpus statuses
    "confirmed": GREEN, "inconclusive": AMBER, "refuted": RED,
    "blocked": MUTED, "unknown": GREY,
    # claim statuses
    "live": GREEN, "needs_replication": AMBER, "superseded": MUTED,
    "retracted": RED, "draft": BLUE,
}


def apply_dark_style():
    """Set rcParams so a plain matplotlib figure matches the stream theme."""
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": PANEL, "axes.edgecolor": BORDER,
        "axes.labelcolor": FG, "axes.titlecolor": FG,
        "text.color": FG, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": BORDER, "legend.facecolor": PANEL,
        "legend.edgecolor": BORDER, "legend.labelcolor": FG,
        "font.size": 10,
    })


def emit(fig, name, caption, run=""):
    """Save fig to stream/<name>.png and (re-)emit it to the plot-stream manifest.

    Upsert by filename: re-running a viz with the same `name` replaces its card and
    moves it to newest, rather than stacking a duplicate that points at the now-
    overwritten PNG. (The raw stream_add primitive is append-only by design; this
    policy lives in the helper that skill-generated plots flow through.)

    `name` is a slug (no extension). Returns the saved Path.
    """
    STREAM.mkdir(exist_ok=True)
    out = STREAM / f"{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    # drop any prior manifest entry for this filename so the re-emit is an update
    if stream_add.MANIFEST.exists():
        items = json.loads(stream_add.MANIFEST.read_text()).get("items", [])
        kept = [it for it in items if it.get("file") != out.name]
        if len(kept) != len(items):
            stream_add.MANIFEST.write_text(json.dumps({"items": kept}, indent=2))
    stream_add.add(str(out), caption, run)   # prints "stream += ..." and "Saved: ..."
    return out


# --- live store readers (so generated scripts don't re-glob boilerplate) ---

def records():
    """All run records as dicts. Source: research_os/records/*.json"""
    return [json.load(open(f)) for f in glob.glob(str(ROOT / "research_os/records/*.json"))]


def claims(area="research"):
    """Claim cards as dicts. Source: research_os/claims/<area>/*.json"""
    return [json.load(open(f)) for f in glob.glob(str(ROOT / f"research_os/claims/{area}/*.json"))]


def head_json():
    """The derived live-head (trunk/frontier/trust/recent) as a dict.

    Runs the canonical renderer rather than re-deriving — single source of truth.
    """
    out = subprocess.run(
        ["python3", str(RENDER / "live_head.py"), "--json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)
