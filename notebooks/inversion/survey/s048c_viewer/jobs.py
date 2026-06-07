"""In-memory job registry for the Flask compute dispatcher.

The Flask process is single-user and single-worker: a
ThreadPoolExecutor(max_workers=1) sits at module scope in `serve.py`.
This module just holds the dict the worker writes through and the
progress-poll route reads from.

Race conditions between writes (worker thread) and reads (poll
endpoint) are tolerated — values are eventually consistent and the
poll re-reads every 500 ms. CPython dict assignment is atomic, so no
lock is required for this access pattern.

Job dict schema:
    {
      "progress":     float in [0, 1],
      "message":      str,
      "status":       'queued' | 'running' | 'done' | 'error',
      "config_id":    str (set when done or known up-front),
      "seed":         int,
      "started_at":   ISO 8601 UTC,
      "completed_at": ISO 8601 UTC | None,
    }
"""

from __future__ import annotations

import datetime as _dt

JOBS: dict[str, dict] = {}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def register(job_id: str, *, seed: int, config_id: str) -> None:
    JOBS[job_id] = {
        "progress": 0.0,
        "message": "queued",
        "status": "queued",
        "seed": int(seed),
        "config_id": config_id,
        "started_at": _now_iso(),
        "completed_at": None,
    }


def mark_running(job_id: str) -> None:
    if job_id in JOBS:
        JOBS[job_id]["status"] = "running"


def update_progress(job_id: str, fraction: float, message: str) -> None:
    if job_id in JOBS:
        JOBS[job_id]["progress"] = float(fraction)
        JOBS[job_id]["message"] = str(message)


def mark_done(job_id: str, *, config_id: str | None = None) -> None:
    if job_id in JOBS:
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress"] = 1.0
        JOBS[job_id]["completed_at"] = _now_iso()
        if config_id:
            JOBS[job_id]["config_id"] = config_id


def mark_error(job_id: str, message: str) -> None:
    if job_id in JOBS:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["message"] = message
        JOBS[job_id]["completed_at"] = _now_iso()
