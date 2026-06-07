#!/usr/bin/env python3
"""Live head — the ~30-line derived session-start view (PLAN §2).

Reads the canonical trust-machine store (run records, goal nodes, claim cards,
substrate components) and renders the minimum a session needs to orient:
trunk + last_measured, current branch, the open/revivable frontier, the trust
state (claim status + blast-radius flags), and the last-N runs.

DERIVED, NEVER AUTHORED. This script owns no state — delete it and lose nothing
but the view. It is pure-read over `research_os/{records,goals,claims,substrate}`
and deterministic (no clock, no RNG), so the same store always renders the same
head. The `orient` skill wraps this; the web app will index the same JSON.

Usage:
    python research_os/render/live_head.py            # human render (~30 lines)
    python research_os/render/live_head.py --json      # machine view (dashboard/cache)
    python research_os/render/live_head.py --frontier 8 # show N frontier nodes (default 6)

Phase-1 note: "next-ranked" is NOT ranked here — without `/strategize` (Phase 2
governor verb) the frontier is the set of open/revivable branch+question nodes,
ordered by most-recent activity. The render labels it so no one mistakes recency
for a strategist's ranking.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/

# Frontier = actionable leaf-ish nodes, not structural containers.
FRONTIER_KINDS = {"branch", "question"}
FRONTIER_STATES = {"open", "revivable"}
_ID_RE = re.compile(r"^s(\d+)([a-z]*)")


def _id_sortkey(run_id: str):
    """Order run ids by experiment sequence: s108 < s111 < s112 < s112a."""
    m = _ID_RE.match(run_id)
    if not m:
        return (0, run_id)
    return (int(m.group(1)), m.group(2))


def _run_sortkey(rec: dict):
    """Newest-last ordering: by created_at date, then experiment sequence."""
    return (rec.get("created_at", ""), _id_sortkey(rec["id"]))


def load_store(root: str = ROOT) -> dict:
    """Load every canonical object. Returns dicts keyed by id (+ raw lists)."""

    def _load(pattern):
        out = {}
        for f in glob.glob(os.path.join(root, pattern)):
            if os.path.basename(f).startswith("_"):  # skip _index.json etc.
                continue
            d = json.load(open(f))
            out[d["id"]] = d
        return out

    return {
        "runs": _load("records/*.json"),
        "goals": _load("goals/*.json"),
        "pipelines": _load("pipelines/*.json"),
        "claims": _load("claims/*/*.json"),
        "substrate": _load("substrate/*.json"),
    }


def compute_head(store: dict, frontier_n: int = 6) -> dict:
    """Derive the live-head view object from the store. Pure; no I/O."""
    runs, goals, claims, substrate = (
        store["runs"], store["goals"], store["claims"], store["substrate"],
    )
    pipelines = store.get("pipelines", {})

    # --- "what we've tried": pipelines + derived tested_by count (ADR-0005) ---
    tested_by: dict[str, int] = {}
    for rec in runs.values():
        for pid in rec.get("tests", []) or []:
            tested_by[pid] = tested_by.get(pid, 0) + 1
    _PIPE_ACTIVE = {"open", "revivable", "blocked"}

    def _pipe_key(p):  # active first, then most-recently measured
        lm = (p.get("last_measured") or {}).get("at", "") or ""
        return (p.get("state") in _PIPE_ACTIVE, lm)

    pipe_rows = []
    for p in sorted(pipelines.values(), key=_pipe_key, reverse=True):
        lm = p.get("last_measured") or {}
        pipe_rows.append({
            "id": p["id"],
            "state": p.get("state"),
            "title": p.get("title", ""),
            "serves": p.get("serves", []),
            "tested_by": tested_by.get(p["id"], 0),
            "summary": lm.get("summary"),
        })

    # --- last activity per goal node (max run date, sequence tiebreak) ---
    last_run_of: dict[str, dict] = {}
    for rec in runs.values():
        gid = rec.get("goal_node")
        cur = last_run_of.get(gid)
        if cur is None or _run_sortkey(rec) > _run_sortkey(cur):
            last_run_of[gid] = rec

    # --- trunk (the single thesis root) ---
    trunk = next(
        (g for g in goals.values() if g.get("node_kind") == "thesis" and g.get("parent") is None),
        None,
    )

    # --- newest run overall -> current branch ---
    newest = max(runs.values(), key=_run_sortkey) if runs else None
    cur_branch = goals.get(newest["goal_node"]) if newest else None

    def _chapter_of(node):
        """Walk up to the enclosing chapter for context."""
        seen = set()
        while node and node.get("parent") and node["id"] not in seen:
            seen.add(node["id"])
            parent = goals.get(node["parent"])
            if parent is None or parent.get("node_kind") in ("chapter", "thesis"):
                return parent
            node = parent
        return goals.get(node["parent"]) if node else None

    # --- frontier: open/revivable branch+question nodes, recency-ordered ---
    frontier = [
        g for g in goals.values()
        if g.get("node_kind") in FRONTIER_KINDS and g.get("state") in FRONTIER_STATES
    ]

    def _front_key(g):
        lr = last_run_of.get(g["id"])
        return (lr.get("created_at", "") if lr else "", _id_sortkey(lr["id"]) if lr else (0, ""))

    frontier.sort(key=_front_key, reverse=True)

    def _front_row(g):
        lr = last_run_of.get(g["id"])
        ch = _chapter_of(g)
        return {
            "id": g["id"],
            "state": g["state"],
            "title": g.get("title", ""),
            "last_run": lr["id"] if lr else None,
            "last_at": (lr.get("created_at", "")[:10] if lr else None),
            "chapter": ch["id"] if ch else None,
        }

    # --- trust block: claim status counts + blast-radius (needs_replication) ---
    status_counts: dict[str, int] = {}
    needs_rep = []
    for c in claims.values():
        st = c.get("status", "?")
        status_counts[st] = status_counts.get(st, 0) + 1
        if st == "needs_replication":
            # surface the stale-substrate edge that flipped it
            stale = [d for d in c.get("depends_on", [])
                     if any(d == f"{sid}@{s['current_version']}" for sid, s in substrate.items()) is False
                     and d.split("@")[0] in substrate]
            needs_rep.append({"id": c["id"], "depends_on": c.get("depends_on", []), "stale": stale})

    # --- recent runs (last 5) ---
    recent = sorted(runs.values(), key=_run_sortkey)[-5:][::-1]

    return {
        "store_counts": {k: len(store[k]) for k in ("runs", "goals", "pipelines", "claims", "substrate")},
        "pipelines": pipe_rows,
        "newest_run": {"id": newest["id"], "at": newest.get("created_at", "")[:10]} if newest else None,
        "trunk": (
            {
                "id": trunk["id"],
                "state": trunk.get("state"),
                "title": trunk.get("title", ""),
                "last_measured": trunk.get("last_measured"),
            }
            if trunk else None
        ),
        "current_branch": (
            {
                "id": cur_branch["id"],
                "state": cur_branch.get("state"),
                "title": cur_branch.get("title", ""),
                "chapter": (_chapter_of(cur_branch) or {}).get("id"),
                "latest_run": newest["id"],
                "latest_status": newest.get("status"),
                "spent_runs": (cur_branch.get("spent") or {}).get("runs"),
            }
            if cur_branch else None
        ),
        "frontier": [_front_row(g) for g in frontier[:frontier_n]],
        "frontier_total": len(frontier),
        "trust": {
            "status_counts": status_counts,
            "needs_replication": needs_rep,
            "substrate_heads": {sid: s["current_version"] for sid, s in sorted(substrate.items())},
        },
        "recent": [
            {"id": r["id"], "status": r.get("status"), "goal": r.get("goal_node")}
            for r in recent
        ],
    }


_STATUS_GLYPH = {"open": "●", "revivable": "○"}
_PIPE_GLYPH = {"open": "●", "revivable": "○", "blocked": "◐", "closed": "✓", "superseded": "⊘"}
_RUN_ABBR = {"confirmed": "confirm", "refuted": "refuted", "inconclusive": "inconcl"}


def render(head: dict) -> str:
    """Format the head dict as the ~30-line terminal banner."""
    L = []
    sc = head["store_counts"]
    nr = head["newest_run"]
    L.append(
        f"RESEARCH OS — live head  ·  store: {sc['runs']} runs / {sc['pipelines']} pipelines "
        f"/ {sc['claims']} claims / {sc['goals']} goals  ·  newest: {nr['id'].split('_')[0] if nr else '—'} "
        f"({nr['at'] if nr else '—'})"
    )

    t = head["trunk"]
    if t:
        L.append("")
        L.append(f"TRUNK  {t['id']}  [{t['state']}]")
        L.append(f"  {t['title']}")
        lm = t.get("last_measured") or {}
        if lm:
            L.append(f"  last measured  {lm.get('run','—')}  ({lm.get('at','—')[:10]})")
            if lm.get("summary"):
                L.append(f"    {lm['summary']}")

    cb = head["current_branch"]
    if cb:
        L.append("")
        ch = f"  ·  chapter: {cb['chapter']}" if cb.get("chapter") else ""
        L.append(f"CURRENT BRANCH  {cb['id']}  [{cb['state']}]{ch}")
        L.append(f"  {cb['title']}")
        L.append(
            f"  latest run  {cb['latest_run'].split('_')[0]}  "
            f"{cb.get('latest_status','?')}  ·  {cb.get('spent_runs','?')} runs spent"
        )

    L.append("")
    L.append(f"FRONTIER  (open/revivable branches — UNRANKED; /strategize to rank · {head['frontier_total']} total)")
    for r in head["frontier"]:
        g = _STATUS_GLYPH.get(r["state"], "·")
        short = r["id"].replace("goal_", "")
        lr = (r["last_run"].split("_")[0] if r["last_run"] else "—")
        chap = r["chapter"].replace("goal_", "") if r["chapter"] else "—"
        L.append(
            f"  {g} {short:36s} {r['state']:9s} {lr:6s} {r['last_at'] or '—':10s} ({chap})"
        )
        if r.get("title"):
            L.append(f"      {r['title']}")

    pls = head.get("pipelines", [])
    if pls:
        L.append("")
        L.append(f"TRIED  (pipelines — how-we're-trying · {len(pls)} total)")
        for p in pls:
            g = _PIPE_GLYPH.get(p["state"], "·")
            short = p["id"].replace("pipeline_", "")
            serves = ",".join(s.replace("goal_", "") for s in p["serves"])
            L.append(f"  {g} {short:30s} {p['state']:10s} {p['tested_by']}× → {serves}")
            if p.get("summary"):
                L.append(f"      {p['summary']}")

    tr = head["trust"]
    counts = tr["status_counts"]
    L.append("")
    L.append(
        "TRUST  " + " · ".join(f"{k} {counts[k]}" for k in
        ("live", "needs_replication", "superseded", "retracted", "draft") if k in counts)
    )
    if tr["needs_replication"]:
        L.append("  ⚠ needs_replication (blast-radius — evidence on stale substrate):")
        for c in tr["needs_replication"]:
            stale = ",".join(c["stale"]) if c["stale"] else ",".join(c["depends_on"])
            L.append(f"     {c['id']:48s} {stale}")
    heads = " · ".join(f"{k}@{v}" for k, v in tr["substrate_heads"].items())
    L.append(f"  substrate heads: {heads}")

    L.append("")
    L.append(
        "RECENT  " + " · ".join(
            f"{r['id'].split('_')[0]} {_RUN_ABBR.get(r['status'], r['status'] or '?')}"
            for r in head["recent"]
        )
    )
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Render the Research OS live head.")
    ap.add_argument("--json", action="store_true", help="emit the head as JSON (dashboard/cache)")
    ap.add_argument("--frontier", type=int, default=6, help="number of frontier nodes to show")
    ap.add_argument("--root", default=ROOT, help="store root (default: research_os/)")
    args = ap.parse_args()

    store = load_store(args.root)
    head = compute_head(store, frontier_n=args.frontier)
    if args.json:
        print(json.dumps(head, indent=2))
    else:
        print(render(head))


if __name__ == "__main__":
    main()
