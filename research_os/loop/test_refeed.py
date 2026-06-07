#!/usr/bin/env python3
"""Unit gate for the Slice-2 re-feed seam ($artifact.<id>) — loop/artifacts resolver +
run_pipeline integration. Store-safe: reifies toy materials into a TEMP shelf, never the
real one (analytical before computational — proves the MECHANISM without a real batch).

Covers:
  - ref recognition (string '$artifact.…' / dict {'$artifact': id}),
  - load_ref round-trips the exact array; subkey indexing into a dict blob,
  - the type-check: keyed-map port (exact), legacy-list port (membership), unconstrained,
    a deliberate mismatch REFUSES (ArtifactTypeError), unknown id RAISES (FileNotFoundError),
  - resolve_supplied (run_tool's --input side): refs load, non-refs pass through,
  - run_pipeline.resolve_ref pours a shelved material into a step, type-checked; dry-run
    validates structure without loading the blob.

    python research_os/loop/test_refeed.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "loop"))
sys.path.insert(0, REPO)

import artifacts          # noqa: E402
import run_pipeline       # noqa: E402  (exercises the pipeline resolver against the same shelf)


def _real_card(tool_id: str) -> dict:
    return json.load(open(os.path.join(ROOT, "substrate", f"{tool_id}.json")))


def _check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def _raises(name, exc, fn):
    try:
        fn()
    except exc:
        print(f"  PASS  {name} (raised {exc.__name__})")
        return
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name} (raised {type(e).__name__}, wanted {exc.__name__})")
        raise AssertionError(name)
    print(f"  FAIL  {name} (no exception, wanted {exc.__name__})")
    raise AssertionError(name)


def _reify_one(rv, atype, seq, port="x"):
    """Reify a single material of artifact_type `atype` via a minimal fabricated card; return its id."""
    card = {"id": f"_src_{atype}", "ports": {"output": {port: atype}}}
    cards, _ = artifacts.reify(rv, card, {"run": "tr_test", "step": None},
                               op="compose", seq_start=seq, now="2026-06-08T00:00:00", commit="test")
    assert len(cards) == 1, f"expected 1 card, got {len(cards)}"
    return cards[0]["id"]


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="refeed_test_")
    artifacts.INSTANCES = tmp
    artifacts.DATA = os.path.join(tmp, "data")

    # --- ref recognition --------------------------------------------------
    print("\n[is_artifact_ref]")
    _check("string form recognised", artifacts.is_artifact_ref("$artifact.ai_x"))
    _check("dict form recognised", artifacts.is_artifact_ref({"$artifact": "ai_x"}))
    _check("$steps is NOT an artifact ref", not artifacts.is_artifact_ref("$steps.sample.q"))
    _check("plain literal is not a ref", not artifacts.is_artifact_ref([1, 2, 3]))

    # --- shelve toy materials ---------------------------------------------
    cloud = np.random.default_rng(7).standard_normal((40, 4))
    lc = np.linspace(0, 1, 30)
    nested = {"a": [1.0, 2.0, 3.0], "b": 7}
    aid_ia = _reify_one(cloud, "ia-cloud", 0)
    aid_lc = _reify_one(lc, "light-curve", 1)
    aid_dict = _reify_one(nested, "ia-cloud", 2, port="result")  # rv keys != port -> whole dict is the material

    # --- load_ref round-trips ---------------------------------------------
    print("\n[load_ref]")
    _check("string form load == original", np.array_equal(artifacts.load_ref(f"$artifact.{aid_ia}"), cloud))
    _check("dict form load == original", np.array_equal(artifacts.load_ref({"$artifact": aid_ia}), cloud))
    _check("subkey indexes into a dict blob", artifacts.load_ref(f"$artifact.{aid_dict}.b") == 7)
    _check("dict-form key= subkey", artifacts.load_ref({"$artifact": aid_dict, "key": "a"}) == [1.0, 2.0, 3.0])

    # --- type-check against the consuming port ----------------------------
    print("\n[typecheck_ref]")
    twin = _real_card("twin_dedup")  # keyed map: {q0_arr: ia-cloud, omega_arr: omega-cloud}
    _check("ia-cloud into q0_arr (keyed) OK",
           artifacts.typecheck_ref(f"$artifact.{aid_ia}", twin, "q0_arr")["artifact_type"] == "ia-cloud")
    _raises("light-curve into q0_arr (keyed) REFUSED", artifacts.ArtifactTypeError,
            lambda: artifacts.typecheck_ref(f"$artifact.{aid_lc}", twin, "q0_arr"))
    _raises("ia-cloud into omega_arr (keyed, wants omega-cloud) REFUSED", artifacts.ArtifactTypeError,
            lambda: artifacts.typecheck_ref(f"$artifact.{aid_ia}", twin, "omega_arr"))
    legacy = {"id": "_legacy", "ports": {"input": ["ia-cloud"]}}  # legacy flat list -> membership gate
    _check("legacy list membership OK",
           artifacts.typecheck_ref(f"$artifact.{aid_ia}", legacy, "q0_arr")["id"] == aid_ia)
    _raises("legacy list non-member REFUSED", artifacts.ArtifactTypeError,
            lambda: artifacts.typecheck_ref(f"$artifact.{aid_lc}", legacy, "q0_arr"))
    free = {"id": "_free", "ports": {"input": []}}  # no constraint -> pour anything
    _check("unconstrained port accepts any type",
           artifacts.typecheck_ref(f"$artifact.{aid_lc}", free, "whatever")["id"] == aid_lc)
    _raises("unknown id RAISES", FileNotFoundError,
            lambda: artifacts.typecheck_ref("$artifact.ai_does_not_exist", twin, "q0_arr"))

    # --- resolve_supplied (run_tool --input side) -------------------------
    print("\n[resolve_supplied]")
    omega = np.ones((40, 3))
    out = artifacts.resolve_supplied({"q0_arr": {"$artifact": aid_ia}, "omega_arr": omega}, twin)
    _check("ref kwarg loaded to its array", np.array_equal(out["q0_arr"], cloud))
    _check("non-ref kwarg passes through unchanged", out["omega_arr"] is omega)
    _raises("resolve_supplied refuses a type mismatch", artifacts.ArtifactTypeError,
            lambda: artifacts.resolve_supplied({"q0_arr": {"$artifact": aid_lc}}, twin))

    # --- run_pipeline.resolve_ref (the pipeline seam) ---------------------
    print("\n[run_pipeline.resolve_ref]")
    ctx = {"in": {}, "steps": {}}
    _check("pipeline resolver loads a shelved material",
           np.array_equal(run_pipeline.resolve_ref(f"$artifact.{aid_ia}", ctx, card=twin, kwarg="q0_arr"), cloud))
    _check("dry-run validates without loading the blob",
           run_pipeline.resolve_ref(f"$artifact.{aid_ia}", ctx, dry=True, card=twin, kwarg="q0_arr")
           is run_pipeline._DRY)
    _raises("pipeline resolver refuses a type mismatch", artifacts.ArtifactTypeError,
            lambda: run_pipeline.resolve_ref(f"$artifact.{aid_lc}", ctx, card=twin, kwarg="q0_arr"))

    print(f"\nALL re-feed ($artifact) cases PASS  (tmp: {tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
