#!/usr/bin/env python3
"""Control plane — the write-only intent queue (PLAN §5, Q11).

The dashboard owns NO research state. The single thing it writes is *intent*:
approved contracts, branch blessings, "run this skill" requests. These land as
append-only JSON files in ``research_os/queue/`` for the (Phase-2) headless executor
to consume. Nothing here mutates a goal / run / claim — those stay canonical, authored
only by the spine skills in the terminal. Delete the queue → lose only un-consumed
requests.
"""
from __future__ import annotations

import itertools
import json
import os
import time

_seq = itertools.count()

RESEARCH_OS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUEUE = os.path.join(RESEARCH_OS, "queue")

ALLOWED_KINDS = {
    "run_skill",      # request a spine/instrument skill be run (in a terminal or headless)
    "bless_contract", # approve a frozen branch contract
    "pick_frontier",  # mark a frontier item as the chosen next action
    "confirm_claim",  # promote an auto-drafted claim card draft->live
    "note",           # freeform operator note routed to the executor
}


def enqueue(intent: dict) -> dict:
    """Append an intent to the queue. Returns the stored record (with id + ts)."""
    os.makedirs(QUEUE, exist_ok=True)
    kind = intent.get("kind")
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unknown intent kind {kind!r}; allowed: {sorted(ALLOWED_KINDS)}")
    ts = time.time()
    rec = {
        # ms timestamp + a process-local counter so two intents in the same
        # millisecond don't collide and silently overwrite each other.
        "id": f"intent_{int(ts*1000)}_{next(_seq)}",
        "kind": kind,
        "payload": intent.get("payload", {}),
        "source": intent.get("source", "webapp"),
        "status": "queued",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
    }
    path = os.path.join(QUEUE, f"{rec['id']}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def listing() -> list[dict]:
    """All queued intents, newest first."""
    if not os.path.isdir(QUEUE):
        return []
    out = []
    for fn in os.listdir(QUEUE):
        if fn.endswith(".json"):
            try:
                out.append(json.load(open(os.path.join(QUEUE, fn))))
            except Exception:
                continue
    out.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return out
