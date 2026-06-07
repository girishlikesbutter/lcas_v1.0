#!/usr/bin/env python3
"""Run a Pipeline as a DAG — the compose/map executor (ADR-0007 §5.3).

Where `run_tool` runs ONE Tool on ONE input, this runs a Pipeline's `steps[]` DAG:
each step runs a Tool (drift-checked exactly like the run-button), and the ref language
threads one step's output into the next. The two combinators ADR-0007 §2 names:

  - compose : invoke the Tool once. inputs wire its kwargs from the run context.
  - map     : lift a per-epoch Tool across `over` (a collection) -> a LIST of outputs
              ("a cloud for the whole LC is the same Tool mapped over a window").

Ref language (a step `inputs` VALUE, or a map step's `over`):
  - "$in.<key>"            -> a pipeline-level input (from --input)
  - "$steps.<sid>"         -> the whole return value of an earlier step
  - "$steps.<sid>.<key>"   -> one key of an earlier step's dict return
  - anything else          -> a literal (number / string / list / dict, passed as-is)

Q1-safe (same boundary as run_tool): the store owns the Pipeline DEFINITION and the
pipeline_run RECORD; it never owns compute. Each step binds canonical code via the Tool
card's entry_point + hash; a stale binding REFUSES the whole DAG (unless --force).

    python research_os/loop/run_pipeline.py pipeline_so3-pool-dedup
    python research_os/loop/run_pipeline.py pipeline_blind-5step-invert --input '{"seed": 39}'
    python research_os/loop/run_pipeline.py <id|path.json> --dry-run   # drift-check + plan, no run
    python research_os/loop/run_pipeline.py <id> --params '{"sample": {"n_samples": 2000}}'

--params is keyed BY STEP id, layered over each step's params/default_params.
Exit codes: 0 ok · 2 refused/usage (drift, unknown pipeline, no steps) · 1 a step raised.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
REPO = os.path.dirname(ROOT)
PIPELINE_RUNS = os.path.join(ROOT, "pipeline_runs")

sys.path.insert(0, os.path.join(ROOT, "loop"))
import tool_lint   # noqa: E402  (sibling — drift-check)
import run_tool     # noqa: E402  (sibling — reuse import/bind/invoke/summarise primitives)
import artifacts as artifact_store  # noqa: E402  (sibling — reify port-typed step outputs)


def _slug(pipeline_id: str) -> str:
    """pipeline_so3-pool-dedup -> so3_pool_dedup (for the pr_ run id)."""
    s = pipeline_id[len("pipeline_"):] if pipeline_id.startswith("pipeline_") else pipeline_id
    return s.replace("-", "_")


def load_pipeline(arg: str) -> dict | None:
    """Load a pipeline node by id (pipelines/<id>.json) or by direct .json path."""
    if arg.endswith(".json") and os.path.isfile(arg):
        return json.load(open(arg))
    fp = os.path.join(ROOT, "pipelines", f"{arg}.json")
    if os.path.isfile(fp):
        return json.load(open(fp))
    # tolerate a bare slug ("so3-pool-dedup" -> "pipeline_so3-pool-dedup")
    fp2 = os.path.join(ROOT, "pipelines", f"pipeline_{arg}.json")
    return json.load(open(fp2)) if os.path.isfile(fp2) else None


_SENTINEL = object()
_DRY = object()  # dry-run placeholder: an upstream step output that wasn't materialised


def resolve_ref(val, ctx: dict, dry: bool = False, card: dict | None = None, kwarg: str | None = None):
    """Resolve a single inputs/over value: a `$...` ref against ctx, else a literal.

    In dry-run the upstream outputs don't exist, so we validate STRUCTURE only — the
    referenced step must exist and appear earlier — and return `_DRY` once we hit an
    unmaterialised value rather than drilling into keys we can't have yet.

    A `$artifact.<id>` value is the Slice-2 re-feed seam: it loads a SHELVED material (which
    exists independent of run state) and type-checks it against the consuming port (card+kwarg).
    The existence + type check run even in dry-run (caught at plan time); the blob is only
    loaded on a real run.
    """
    if artifact_store.is_artifact_ref(val):
        artifact_store.typecheck_ref(val, card, kwarg)  # FileNotFound / ArtifactTypeError -> wiring error
        return _DRY if dry else artifact_store.load_ref(val)
    if not isinstance(val, str) or not val.startswith("$"):
        return val  # literal (number/list/dict/plain string)
    parts = val[1:].split(".")
    root = parts[0]
    if root == "in":
        cur, rest = ctx["in"], parts[1:]
    elif root == "steps":
        if len(parts) < 2:
            raise KeyError(f"ref '{val}' needs a step id (e.g. $steps.<sid>)")
        sid = parts[1]
        if sid not in ctx["steps"]:
            raise KeyError(f"ref '{val}' -> step '{sid}' has no output yet (order/typo?)")
        cur, rest = ctx["steps"][sid], parts[2:]
    else:
        raise KeyError(f"ref '{val}' must start with $in or $steps")
    for key in rest:
        if cur is _DRY:
            return _DRY  # dry-run: structure is valid, value not materialised
        if isinstance(cur, dict):
            if key not in cur:
                if dry:
                    return _DRY
                raise KeyError(f"ref '{val}' -> key '{key}' not in keys {list(cur)[:8]}")
            cur = cur[key]
        else:
            got = getattr(cur, key, _SENTINEL)
            if got is _SENTINEL:
                if dry:
                    return _DRY
                raise KeyError(f"ref '{val}' -> '{key}' not resolvable on {type(cur).__name__}")
            cur = got
    return cur


def resolve_inputs(inputs: dict, ctx: dict, dry: bool = False, card: dict | None = None) -> dict:
    # card+kwarg let an `inputs` value re-feed a shelved material ($artifact.<id>), type-checked
    # against the consuming Tool's input port (Slice 2).
    return {kw: resolve_ref(ref, ctx, dry, card=card, kwarg=kw) for kw, ref in (inputs or {}).items()}


def _eff_params(card: dict, step: dict, run_params: dict) -> dict:
    """card default_params (minus _meta) <- step.params <- run --params[step_id]."""
    eff = {k: v for k, v in (card.get("default_params") or {}).items()
           if not str(k).startswith("_")}
    eff.update(step.get("params") or {})
    eff.update((run_params or {}).get(step["id"], {}))
    return eff


def _invoke(fn, supplied: dict):
    """Bind accepted kwargs and call. Returns rv; raises on missing-required or in-Tool error."""
    accepted, missing, _ = run_tool.bind_kwargs(fn, supplied)
    if missing:
        raise TypeError(f"missing required kwargs {missing}; have {sorted(supplied)}")
    return fn(**accepted)


def run_step(step: dict, ctx: dict, run_params: dict, force: bool, dry_run: bool,
             run_id: str = "", seq_start: int = 0, commit: str = "") -> tuple[dict, int]:
    """Execute one step. Returns (step-result dict, next artifact-seq); on op=map ctx output is a list.

    On success, port-typed outputs are reified to artifact_instance cards ('recognised materials
    only', ADR-0007); their ids land on res['artifacts_produced']. seq_start/next-seq thread the
    per-run artifact counter so ids stay unique across steps.
    """
    sid, op, tid = step["id"], step["op"], step["tool"]
    card = run_tool.load_card(tid)
    res = {"step": sid, "tool": tid, "op": op, "status": "error",
           "tool_version": None, "hash_at_run": None, "hash_ok": True,
           "wall_s": None, "metrics": {}, "error": None}
    if card is None:
        res["error"] = f"unknown Tool '{tid}' (no substrate/{tid}.json)"
        return res, seq_start
    res["tool_version"] = card.get("current_version")

    verdict = tool_lint.lint_tool(card)
    res["hash_at_run"] = verdict.get("live_hash")
    res["hash_ok"] = not verdict["drift"]
    if verdict["drift"] and not force:
        res["status"] = "drift_refused"
        res["error"] = f"{verdict['verdict']}: {verdict['reason']}"
        return res, seq_start

    eff_params = _eff_params(card, step, run_params)
    try:
        wired = resolve_inputs(step.get("inputs"), ctx, dry=dry_run, card=card)
    except (KeyError, FileNotFoundError, artifact_store.ArtifactTypeError) as e:
        res["error"] = f"input wiring: {e}"
        return res, seq_start

    if dry_run:
        plan = {**eff_params, **wired}
        res["status"] = "ok"
        res["metrics"] = {"would_call": card["entry_point"], "kwargs": sorted(plan)}
        if op == "map":
            res["metrics"]["over"] = step.get("over")
        ctx["steps"][sid] = _DRY
        return res, seq_start

    fn = run_tool.import_callable(card["entry_point"])
    t0 = time.perf_counter()
    try:
        if op == "compose":
            rv = _invoke(fn, {**eff_params, **wired})
            metrics, _ = run_tool.summarise_return(rv)
        elif op == "map":
            coll = resolve_ref(step.get("over"), ctx)
            elems = list(coll)
            map_as = step.get("map_as")
            rv = []
            for el in elems:
                per = dict(wired)
                if map_as:
                    per[map_as] = el
                elif isinstance(el, dict):
                    per.update(el)
                else:
                    raise TypeError(f"map step '{sid}': element {type(el).__name__} needs map_as")
                rv.append(_invoke(fn, {**eff_params, **per}))
            res["n_mapped"] = len(elems)
            metrics = {"n_mapped": len(elems)}
            if rv:
                first, _ = run_tool.summarise_return(rv[0])
                metrics["first"] = {k: first[k] for k in list(first)[:6]}
        else:
            raise ValueError(f"unknown op '{op}'")
    except Exception as e:
        res["wall_s"] = round(time.perf_counter() - t0, 4)
        res["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
        return res, seq_start

    res["wall_s"] = round(time.perf_counter() - t0, 4)
    res["metrics"] = metrics
    res["status"] = "ok"
    ctx["steps"][sid] = rv
    next_seq = seq_start
    if artifact_store.output_map(card):  # 'recognised materials only' -> reify to the shelf
        mats, next_seq = artifact_store.reify(rv, card, {"run": run_id, "step": sid},
                                              op=op, seq_start=seq_start,
                                              now=run_tool._now(), commit=commit)
        res["artifacts_produced"] = [m["id"] for m in mats]
    return res, next_seq


def write_record(rec: dict) -> str:
    os.makedirs(PIPELINE_RUNS, exist_ok=True)
    fp = os.path.join(PIPELINE_RUNS, f"{rec['id']}.json")
    with open(fp, "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    return fp


def run(arg, inputs=None, run_params=None, reason="", force=False, dry_run=False,
        no_write=False, source="cli"):
    pipe = load_pipeline(arg)
    if pipe is None:
        print(f"ERROR unknown pipeline '{arg}' (no pipelines/{arg}.json)", file=sys.stderr)
        return 2, None
    steps = pipe.get("steps") or []
    if not steps:
        print(f"ERROR pipeline '{pipe['id']}' is descriptive-only (no steps DAG) — not runnable.",
              file=sys.stderr)
        return 2, None

    inputs = inputs or {}
    run_params = run_params or {}
    ctx = {"in": inputs, "steps": {}}
    oracle_clean = not any(s["tool"] in run_tool._ORACLE_TOOLS for s in steps)
    run_id = f"pr_{_slug(pipe['id'])}_{run_tool._stamp()}"
    base = {
        "schema_version": "1.0.0", "id": run_id, "kind": "pipeline_run",
        "pipeline": pipe["id"], "input": inputs, "params": run_params,
        "oracle_clean": bool(oracle_clean), "reason": reason,
        "commit": run_tool._git_sha(), "source": source, "created_at": run_tool._now(),
    }

    print(f"running pipeline {pipe['id']} ({len(steps)} steps){' [dry-run]' if dry_run else ''} ...")
    step_results, status, top_error = [], "ok", None
    t0 = time.perf_counter()
    seq = 0  # per-run artifact-instance counter (keeps reified ids unique across steps)
    for i, step in enumerate(steps):
        r, seq = run_step(step, ctx, run_params, force, dry_run, run_id, seq, base["commit"])
        step_results.append(r)
        tag = r["status"].upper()
        print(f"  [{tag:>13}] {step['id']:<20} {step['tool']:<22} "
              f"{('' if r['wall_s'] is None else str(r['wall_s'])+'s'):>9}"
              f"{'' if not r['error'] else '  ' + r['error']}")
        if r["status"] != "ok":
            status = r["status"] if r["status"] in ("drift_refused", "error") else "error"
            top_error = f"step '{step['id']}': {r['error']}"
            for s in steps[i + 1:]:  # everything after a failure never ran
                step_results.append({"step": s["id"], "tool": s["tool"], "op": s["op"],
                                     "status": "skipped", "tool_version": None,
                                     "hash_ok": True, "wall_s": None, "metrics": {}, "error": None})
            break
    wall_s = round(time.perf_counter() - t0, 4)

    rec = {**base, "status": status, "wall_s": (None if dry_run else wall_s),
           "steps": step_results, "error": top_error,
           **({} if dry_run else {"executed_at": run_tool._now()})}

    if dry_run:
        print(f"DRY-RUN {pipe['id']} — {len(steps)} steps planned, drift {('OK' if status=='ok' else status)}.")
        return (0 if status == "ok" else 2), rec
    if not no_write:
        print("logged:", os.path.relpath(write_record(rec), REPO))
    code = {"ok": 0, "drift_refused": 2, "error": 1}[status]
    print(f"{'OK ' if status=='ok' else status.upper()}  {pipe['id']}  {wall_s}s  "
          f"({sum(1 for r in step_results if r['status']=='ok')}/{len(steps)} steps ok)")
    return code, (rec if no_write else run_id)


def main():
    ap = argparse.ArgumentParser(description="Run a Pipeline DAG — compose/map executor (ADR-0007).")
    ap.add_argument("pipeline", help="pipeline id (pipelines/<id>.json) or a direct .json path")
    ap.add_argument("--input", default="{}", help="JSON dict of pipeline-level inputs ($in.*)")
    ap.add_argument("--params", default="{}", help="JSON dict keyed BY STEP id of param overrides")
    ap.add_argument("--reason", default="", help="why this pipeline was run (bench-log note)")
    ap.add_argument("--force", action="store_true", help="run despite drift (recorded hash_ok=false)")
    ap.add_argument("--dry-run", action="store_true", help="drift-check + plan each step, do not run")
    ap.add_argument("--no-write", action="store_true", help="do not write the pipeline_run record")
    ap.add_argument("--source", default="cli", choices=["cli", "webapp", "skill"])
    args = ap.parse_args()
    code, _ = run(args.pipeline, inputs=json.loads(args.input), run_params=json.loads(args.params),
                  reason=args.reason, force=args.force, dry_run=args.dry_run,
                  no_write=args.no_write, source=args.source)
    sys.exit(code)


if __name__ == "__main__":
    main()
