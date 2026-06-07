#!/usr/bin/env python3
"""Re-materialize the DERIVED goal-node fields from the run records.

`child_runs`, `spent.runs`, `spent.wall_s`, and `goals/_index.json:n_experiments`
are caches computed from the records that point at each node — never authored by
hand. After `close` writes a new run_record (and optionally edits the AUTHORED fields
`last_measured` / `state` / `budget`), this script recomputes the derived caches so
the live-head's "N runs spent" stays honest.

This is `backfill/reconcile.py` steps (b)+(d) WITHOUT its backfill baggage: it stamps
the real date (not the frozen 2026-06-01), touches only nodes that actually changed,
and never rewrites the backfill README. Authored fields are preserved verbatim.

Usage:
    python research_os/loop/refresh.py --date 2026-06-01   # explicit (deterministic)
    python research_os/loop/refresh.py                       # today (system clock)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/


def _records(root):
    out = {}
    for f in glob.glob(os.path.join(root, "records/*.json")):
        d = json.load(open(f))
        out[d["id"]] = d
    return out


def _wall_of(rec: dict) -> float:
    m = rec.get("metrics") or {}
    for k in ("wall_s", "wall"):
        if isinstance(m.get(k), (int, float)):
            return float(m[k])
    return 0.0


def refresh(root: str = ROOT, stamp: str | None = None) -> list[str]:
    """Recompute derived caches. Returns the list of node ids that changed."""
    if stamp is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    rec = _records(root)

    by_node_runs = defaultdict(list)
    by_node_wall = defaultdict(float)
    for rid, d in rec.items():
        gid = d.get("goal_node")
        by_node_runs[gid].append(rid)
        by_node_wall[gid] += _wall_of(d)

    changed: list[str] = []
    node_count: dict[str, int] = {}
    for f in glob.glob(os.path.join(root, "goals/*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        d = json.load(open(f))
        gid = d["id"]
        kids = sorted(by_node_runs.get(gid, []))
        wall = round(by_node_wall.get(gid, 0.0), 3)
        node_count[gid] = len(kids)
        sp = dict(d.get("spent") or {})
        new_sp = {"runs": len(kids), "wall_s": wall}
        # preserve any extra spent.* keys an author added
        merged = {**sp, **new_sp}
        if d.get("child_runs") != kids or sp.get("runs") != len(kids) or sp.get("wall_s") != wall:
            d["child_runs"] = kids
            d["spent"] = merged
            d["updated_at"] = stamp
            with open(f, "w") as fh:
                fh.write(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
            changed.append(gid)

    # refresh _index.json n_experiments (does not bump a timestamp; it's a pure cache)
    idx_path = os.path.join(root, "goals", "_index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        dirty = False
        for e in idx:
            n = node_count.get(e["id"], 0)
            if e.get("n_experiments") != n:
                e["n_experiments"] = n
                dirty = True
        if dirty:
            with open(idx_path, "w") as fh:
                fh.write(json.dumps(idx, indent=2, ensure_ascii=False) + "\n")
            changed.append("_index")

    return changed


def main():
    ap = argparse.ArgumentParser(description="Re-materialize derived goal-node fields.")
    ap.add_argument("--date", help="YYYY-MM-DD stamp for changed nodes (default: today UTC)")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    stamp = f"{args.date}T00:00:00Z" if args.date else None
    changed = refresh(args.root, stamp)
    if changed:
        print(f"refreshed {len(changed)} node(s): {', '.join(changed)}")
    else:
        print("derived caches already current — no changes.")


if __name__ == "__main__":
    main()
