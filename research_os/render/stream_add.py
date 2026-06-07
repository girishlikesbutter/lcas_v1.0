#!/usr/bin/env python3
"""Append a plot to the Research OS browser plot-stream manifest.

The minimal 'emit to stream' primitive — the dynamic-viz skill will call this
after it draws something. Usage:
    stream_add.py <png_path> "caption" [run_id]
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

STREAM = Path(__file__).resolve().parent / "stream"
MANIFEST = STREAM / "manifest.json"


def add(png_path, caption, run=""):
    STREAM.mkdir(exist_ok=True)
    src = Path(png_path)
    dst = STREAM / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    items = []
    if MANIFEST.exists():
        items = json.loads(MANIFEST.read_text()).get("items", [])
    items.append({
        "file": dst.name,
        "caption": caption,
        "run": run,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    MANIFEST.write_text(json.dumps({"items": items}, indent=2))
    print(f"stream += {dst.name}  ({len(items)} total)")
    print(f"Saved: {dst}")


if __name__ == "__main__":
    add(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
