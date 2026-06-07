#!/usr/bin/env python3
"""The run-button — execute ONE Tool on ONE input (ADR-0007 req #15, smallest form).

"Run the IA-Cloud Tool on seed 39 and KNOW one agreed thing happens." This is the
keystone the laboratory registry (the W-D scour) was built for, in its SMALLEST form:
a manual, single-Tool invocation — NOT an autonomous queue-draining daemon (that
decision stays deferred).

Flow (Q1-safe — the store owns definitions + records, never compute):
  1. Load the Tool card (substrate/<id>.json).
  2. DRIFT-CHECK first (loop/tool_lint): if the entry_point no longer resolves or the
     source hash moved, REFUSE — write a status='drift_refused' tool_run and stop.
     This is the mechanical "we can SEE the binding is stale and won't run it." --force
     overrides (recorded honestly as hash_ok=false).
  3. Effective params = the card's default_params (minus _meta keys) overlaid with
     --params; the input artifact comes from --input. (ADR-0007: variation lives in
     params, not copies.)
  4. Import the bound entry_point, bind the accepted kwargs, invoke under a wall timer.
  5. Summarise the return into metrics, save array-ish output as a gitignored artifact,
     emit any produced plot to the browser plot-stream.
  6. Write a CANONICAL tool_run record (research_os/tool_runs/) stamped with the Tool
     version + the hash in force — first-class, citable bench provenance.

    python research_os/loop/run_tool.py sample_so3_pool --params '{"n_samples": 1000}'
    python research_os/loop/run_tool.py rho_band_classify --input '{"lc": "..."}' --reason "spot-check"
    python research_os/loop/run_tool.py <id> --dry-run     # drift-check + arg preview, no run

Exit codes: 0 ok · 2 refused/usage (drift, unknown tool, missing args) · 1 Tool raised.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
REPO = os.path.dirname(ROOT)
TOOL_RUNS = os.path.join(ROOT, "tool_runs")
ARTIFACTS = os.path.join(TOOL_RUNS, "artifacts")

sys.path.insert(0, os.path.join(ROOT, "loop"))
import tool_lint  # noqa: E402  (sibling module, drift-check)
import artifacts as artifact_store  # noqa: E402  (sibling — reify port-typed outputs to the shelf)

PLOT_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".html", ".gif")
# Tools that read oracle/truth data by definition — a run is not oracle_clean.
_ORACLE_TOOLS = {"load_truth"}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stamp():
    return time.strftime("%Y%m%dt%H%M%S", time.localtime())


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def load_card(tool_id: str) -> dict | None:
    fp = os.path.join(ROOT, "substrate", f"{tool_id}.json")
    return json.load(open(fp)) if os.path.isfile(fp) else None


def import_callable(entry_point: str):
    """Import 'path.py:Symbol[.attr]' and return the callable. Raises on failure."""
    path, _, symbol = entry_point.partition(":")
    abspath = os.path.join(REPO, path)
    for p in (os.path.dirname(abspath), REPO):
        if p not in sys.path:
            sys.path.insert(0, p)
    modname = "rotool_" + path.replace("/", "_").replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(modname, abspath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    obj = mod
    for part in symbol.split("."):
        obj = getattr(obj, part)
    return obj


def bind_kwargs(fn, supplied: dict):
    """Return (accepted_kwargs, missing_required, accepts_all)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(supplied), [], True  # builtin/uninspectable — pass everything
    accepts_all = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    accepted, missing = {}, []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name in supplied:
            accepted[name] = supplied[name]
        elif p.default is p.empty:
            missing.append(name)
    if accepts_all:  # forward any extra keys the signature didn't name
        for k, v in supplied.items():
            accepted.setdefault(k, v)
    return accepted, missing, accepts_all


def summarise_return(rv) -> tuple[dict, list[str]]:
    """(metrics, plot_paths) — a shape/key digest plus any plot file paths found."""
    metrics, plots = {}, []
    try:
        import numpy as np
    except Exception:
        np = None

    def is_arr(x):
        return np is not None and isinstance(x, np.ndarray)

    def note_plot(x):
        if isinstance(x, str) and x.lower().endswith(PLOT_EXTS) and os.path.isfile(
                x if os.path.isabs(x) else os.path.join(REPO, x)):
            plots.append(x)

    if isinstance(rv, dict):
        metrics["returned_keys"] = list(rv.keys())
        for k, v in rv.items():
            if is_arr(v):
                metrics[f"{k}_shape"] = list(v.shape)
            elif isinstance(v, (int, float, str, bool)):
                metrics[k] = v
            note_plot(v)
    elif is_arr(rv):
        metrics["return_shape"] = list(rv.shape)
        metrics["return_dtype"] = str(rv.dtype)
    elif isinstance(rv, (int, float, str, bool)):
        metrics["return"] = rv
        note_plot(rv)
    elif isinstance(rv, (list, tuple)):
        metrics["return_len"] = len(rv)
        metrics["return_type"] = type(rv).__name__
        for x in rv:
            note_plot(x)
    elif rv is not None:
        metrics["return_type"] = type(rv).__name__
        metrics["return_repr"] = repr(rv)[:200]
    return metrics, plots


def save_artifact(rv, run_id: str) -> list[dict]:
    """Persist array-ish output to a gitignored .npz; return artefact entries."""
    try:
        import numpy as np
    except Exception:
        return []
    arrays = {}
    if isinstance(rv, np.ndarray):
        arrays["return"] = rv
    elif isinstance(rv, dict):
        arrays = {k: v for k, v in rv.items() if isinstance(v, np.ndarray)}
    if not arrays:
        return []
    os.makedirs(ARTIFACTS, exist_ok=True)
    fp = os.path.join(ARTIFACTS, f"{run_id}.npz")
    np.savez_compressed(fp, **arrays)
    return [{"path": os.path.relpath(fp, REPO), "kind": "data",
             "caption": f"{run_id} output ({', '.join(arrays)})"}]


def emit_plots(plot_paths, run_id, artefacts):
    sys.path.insert(0, os.path.join(ROOT, "render"))
    try:
        import stream_add
    except Exception:
        return
    for p in plot_paths:
        ap = p if os.path.isabs(p) else os.path.join(REPO, p)
        try:
            stream_add.add(ap, f"{run_id}: {os.path.basename(p)}", run_id)
            artefacts.append({"path": os.path.relpath(ap, REPO), "kind": "plot",
                              "caption": f"{run_id} plot"})
        except Exception:
            pass


def write_record(rec: dict) -> str:
    os.makedirs(TOOL_RUNS, exist_ok=True)
    fp = os.path.join(TOOL_RUNS, f"{rec['id']}.json")
    with open(fp, "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    return fp


def run(tool_id, params=None, input_=None, reason="", force=False, dry_run=False,
        no_write=False, oracle_clean=None, source="cli"):
    card = load_card(tool_id)
    if card is None:
        print(f"ERROR unknown Tool '{tool_id}' (no substrate/{tool_id}.json)", file=sys.stderr)
        return 2, None

    verdict = tool_lint.lint_tool(card)
    tool_version = card.get("current_version")
    run_id = f"tr_{tool_id}_{_stamp()}"

    # effective params: card defaults (minus _meta keys) overlaid with --params
    eff_params = {k: v for k, v in (card.get("default_params") or {}).items()
                  if not str(k).startswith("_")}
    eff_params.update(params or {})
    input_ = input_ or {}
    if oracle_clean is None:
        oracle_clean = tool_id not in _ORACLE_TOOLS

    base = {
        "schema_version": "1.0.0", "id": run_id, "kind": "tool_run",
        "tool": tool_id, "tool_version": tool_version,
        "hash_at_run": verdict.get("live_hash"),
        "hash_ok": (not verdict["drift"]),
        "input": input_, "params": eff_params,
        "oracle_clean": bool(oracle_clean), "reason": reason,
        "commit": _git_sha(), "source": source, "created_at": _now(),
    }

    # --- drift gate -------------------------------------------------------
    if verdict["drift"] and not force:
        print(f"REFUSED [{verdict['verdict']}] {tool_id}: {verdict['reason']}", file=sys.stderr)
        rec = {**base, "status": "drift_refused", "wall_s": None,
               "error": f"{verdict['verdict']}: {verdict['reason']}", "metrics": {}, "artefacts": []}
        if not no_write:
            print("logged:", os.path.relpath(write_record(rec), REPO))
        return 2, (rec if no_write else run_id)

    # --- import + bind ----------------------------------------------------
    try:
        fn = import_callable(card["entry_point"])
    except Exception as e:
        print(f"ERROR importing {card['entry_point']}: {e}", file=sys.stderr)
        rec = {**base, "status": "error", "wall_s": None, "metrics": {}, "artefacts": [],
               "error": f"import failed: {type(e).__name__}: {e}"}
        if not no_write:
            print("logged:", os.path.relpath(write_record(rec), REPO))
        return 1, (rec if no_write else run_id)

    supplied = {**input_, **eff_params}
    # re-feed (Slice 2): any '$artifact.<id>' value loads a shelved material, type-checked
    # against this Tool's input port. The record keeps the REF in `input` (honest provenance);
    # only the call sees the loaded array. A bad ref/type is a caller wiring error -> exit 2.
    try:
        supplied = artifact_store.resolve_supplied(supplied, card)
    except (FileNotFoundError, artifact_store.ArtifactTypeError) as e:
        print(f"REFUSED {tool_id}: re-feed {type(e).__name__}: {e}", file=sys.stderr)
        return 2, None
    accepted, missing, _ = bind_kwargs(fn, supplied)
    if missing:
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(uninspectable)"
        print(f"USAGE {tool_id} needs {missing} — signature {fn.__name__}{sig}\n"
              f"      supply via --input / --params (have: {sorted(supplied)})", file=sys.stderr)
        return 2, None  # caller mistake — not a canonical event

    if dry_run:
        print(f"DRY-RUN {tool_id} [{verdict['verdict']}] would call "
              f"{card['entry_point']} with {accepted}")
        return 0, None

    # --- invoke -----------------------------------------------------------
    print(f"running {tool_id} ({card['entry_point']}) ...")
    t0 = time.perf_counter()
    try:
        rv = fn(**accepted)
        wall_s = round(time.perf_counter() - t0, 4)
    except Exception as e:
        wall_s = round(time.perf_counter() - t0, 4)
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        print(f"ERROR {tool_id} raised after {wall_s}s: {tb}", file=sys.stderr)
        rec = {**base, "status": "error", "wall_s": wall_s, "metrics": {}, "artefacts": [],
               "error": tb, "executed_at": _now()}
        if not no_write:
            print("logged:", os.path.relpath(write_record(rec), REPO))
        return 1, (rec if no_write else run_id)

    metrics, plots = summarise_return(rv)
    metrics["wall_s"] = wall_s
    # 'recognised materials only' (ADR-0007): a keyed output port map -> reify typed materials
    # to the shelf; otherwise fall back to the legacy array dump (figures-equivalent).
    omap = artifact_store.output_map(card)
    artefacts, artifacts_produced = [], []
    if omap:
        mats, _ = artifact_store.reify(rv, card, {"run": run_id, "step": None},
                                       op="compose", seq_start=0, now=_now(), commit=base["commit"])
        artifacts_produced = [m["id"] for m in mats]
    else:
        artefacts = save_artifact(rv, run_id)
    emit_plots(plots, run_id, artefacts)

    rec = {**base, "status": "ok", "wall_s": wall_s, "metrics": metrics,
           "artefacts": artefacts, "executed_at": _now()}
    if artifacts_produced:
        rec["artifacts_produced"] = artifacts_produced
    if not no_write:
        print("logged:", os.path.relpath(write_record(rec), REPO))
    suffix = f"  +{len(artifacts_produced)} material(s)" if artifacts_produced else ""
    print(f"OK  {tool_id}  {wall_s}s  metrics={ {k: metrics[k] for k in list(metrics)[:6]} }{suffix}")
    return 0, (rec if no_write else run_id)


def main():
    ap = argparse.ArgumentParser(description="Run-button — execute one Tool on one input (ADR-0007).")
    ap.add_argument("tool", help="substrate_component (Tool) id")
    ap.add_argument("--params", default="{}", help="JSON dict of param overrides (over default_params)")
    ap.add_argument("--input", default="{}", help="JSON dict of input artifact args")
    ap.add_argument("--reason", default="", help="why this Tool was run (bench-log note)")
    ap.add_argument("--force", action="store_true", help="run despite drift (recorded hash_ok=false)")
    ap.add_argument("--dry-run", action="store_true", help="drift-check + arg-bind preview, do not run")
    ap.add_argument("--no-write", action="store_true", help="do not write the tool_run record")
    ap.add_argument("--oracle-dirty", action="store_true", help="force oracle_clean=false")
    ap.add_argument("--source", default="cli", choices=["cli", "webapp", "skill"])
    args = ap.parse_args()

    code, _ = run(args.tool, params=json.loads(args.params), input_=json.loads(args.input),
                  reason=args.reason, force=args.force, dry_run=args.dry_run,
                  no_write=args.no_write, oracle_clean=(False if args.oracle_dirty else None),
                  source=args.source)
    sys.exit(code)


if __name__ == "__main__":
    main()
