"""Free-text epoch-spec parser.

Supported forms (no mixes):
    all                 → np.arange(n_obs)
    ::N                 → np.arange(0, n_obs, N)
    M-N                 → np.arange(M, N + 1)         (inclusive on both ends)
    M,N,P,...           → np.array([M, N, P, ...])

Anything else, or out-of-range indices, raises ValueError with a clear
message.
"""

from __future__ import annotations

import re
import numpy as np


_SLICE_RE = re.compile(r"^::\s*(\d+)$")
_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_LIST_RE = re.compile(r"^[\d,\s]+$")


def parse_epoch_spec(spec: str, n_obs: int) -> np.ndarray:
    """Parse `spec` into a sorted, unique np.ndarray of int epoch indices.

    Raises ValueError on bad input or out-of-range indices.
    """
    if spec is None:
        raise ValueError("epoch spec is empty")
    s = spec.strip().lower()
    if s == "":
        raise ValueError("epoch spec is empty")

    if s == "all":
        return np.arange(int(n_obs), dtype=np.int64)

    m = _SLICE_RE.match(s)
    if m:
        step = int(m.group(1))
        if step < 1:
            raise ValueError(f"slice step must be >=1, got {step}")
        return np.arange(0, int(n_obs), step, dtype=np.int64)

    m = _RANGE_RE.match(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            raise ValueError(f"range start {a} > end {b}")
        idx = np.arange(a, b + 1, dtype=np.int64)
        _check_in_range(idx, n_obs)
        return idx

    if _LIST_RE.match(s):
        parts = [p for p in s.replace(" ", "").split(",") if p != ""]
        if not parts:
            raise ValueError("comma list is empty")
        try:
            idx = np.array(sorted({int(p) for p in parts}), dtype=np.int64)
        except ValueError as e:
            raise ValueError(f"bad integer in comma list: {e}")
        _check_in_range(idx, n_obs)
        return idx

    raise ValueError(
        f"epoch spec {spec!r} not understood. Supported forms: "
        "'all' | '::N' | 'M-N' | 'M,N,P,...'"
    )


def _check_in_range(idx: np.ndarray, n_obs: int) -> None:
    if idx.size == 0:
        raise ValueError("epoch list is empty after parsing")
    if idx.min() < 0 or idx.max() >= n_obs:
        bad = idx[(idx < 0) | (idx >= n_obs)]
        raise ValueError(
            f"epoch indices out of range [0, {n_obs}): {bad.tolist()[:10]}"
        )
