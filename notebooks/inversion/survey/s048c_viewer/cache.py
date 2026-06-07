"""Config-id hashing + cached-run discovery for the C_t viewer.

A run is uniquely identified by:
    seed, surrogate, n_samples, sample_seed, tolerance_mag, epoch_indices

`make_config_id` hashes a canonical JSON of these fields and returns the
first 12 hex chars of sha256. 48 bits of entropy = collisions
effectively zero in single-user use.

Layout on disk:
    results/s048c_cloud_viewer/
    └── seed{NNN}/
        └── {config_id}/
            ├── spread.npz
            ├── meta.json
            └── animation.html
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

SURVEY_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = SURVEY_DIR / "results" / "s048c_cloud_viewer"


def make_config_id(
    seed: int,
    surrogate: str,
    n_samples: int,
    sample_seed: int,
    tolerance_mag: float,
    epoch_indices: Iterable[int],
) -> str:
    """Stable 12-hex-char id for a run config."""
    canonical = json.dumps(
        {
            "seed": int(seed),
            "surrogate": str(surrogate),
            "n_samples": int(n_samples),
            "sample_seed": int(sample_seed),
            "tolerance_mag": round(float(tolerance_mag), 6),
            "epoch_indices": sorted(int(e) for e in epoch_indices),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def run_dir(seed: int, config_id: str) -> Path:
    return RESULTS_ROOT / f"seed{int(seed):03d}" / config_id


def list_runs() -> list[dict]:
    """All cached runs, newest first. Skips malformed meta.json entries."""
    if not RESULTS_ROOT.exists():
        return []
    runs = []
    for meta_path in RESULTS_ROOT.glob("seed*/*/meta.json"):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["_run_dir"] = str(meta_path.parent)
            meta["_has_animation"] = (meta_path.parent / "animation.html").exists()
            runs.append(meta)
        except (json.JSONDecodeError, OSError):
            continue
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs


def find_run(config_id: str) -> Path | None:
    """Locate a run directory by config_id. Returns None if absent."""
    if not RESULTS_ROOT.exists():
        return None
    matches = list(RESULTS_ROOT.glob(f"seed*/{config_id}"))
    return matches[0] if matches else None
