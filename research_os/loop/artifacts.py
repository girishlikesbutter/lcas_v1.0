#!/usr/bin/env python3
"""Reify produced materials as first-class artifact_instance cards (ADR-0007).

The "shelf" side of the laboratory. A Tool/pipeline-step output becomes an addressable
artifact_instance IFF the Tool's `ports.output` NAMES it — the "recognised materials only"
rule (Girish, 2026-06-07). Nameless scratch returns stay in-memory.

Each reified material = a CARD (research_os/artifact_instances/<id>.json, closure-counted by
validate.py) + a BLOB (research_os/artifact_instances/data/<id>.{npz,json}, gitignored like a
figure). The card is the committed, addressable handle; the blob is the material.

ports.output forms (substrate_component.schema.json, transition):
  - keyed map {return-key/tuple-position-name: artifact_type}  -> REIFY these (the new form).
  - legacy flat list [artifact_type, ...]                      -> describes types only, NOT reified.

Return-shape dispatch (compose):
  - dict       : reify rv[key] for each map key present in rv.
  - tuple      : zip the map's entries to positions IN ORDER; a key 'name@<idx>' selects a
                 NON-leading position (e.g. 'quats@2' picks rv[2], skipping k1/k2).
  - list (1-entry map) : the WHOLE list is ONE material (e.g. multi_start_optimize's
                 [OptimizationResult, ...] is one candidate-set), not position 0.
  - single val : a 1-entry map; that value is the material (the map key is just its label).
  - dataclass / list-of-dataclass : asdict()-serialised to a json blob (InertiaResult,
                 OptimizationResult, InversionResult); the sample digest holds field names, not arrays.
map step (op="map"): rv is a LIST of per-element returns; reify ONE 'set' card per declared
  output, stacking that key across elements when each element exposes it.

Q1-safe: writes only under artifact_instances/ (definitions + a gitignored blob). The store
owns the handle; the material is on disk, reflected — never owned-as-compute.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_os/
REPO = os.path.dirname(ROOT)
INSTANCES = os.path.join(ROOT, "artifact_instances")
DATA = os.path.join(INSTANCES, "data")


def output_map(card: dict) -> dict | None:
    """The keyed output port map {name: artifact_type}, or None if legacy-list / absent.

    Only the keyed-map form binds a return value to a type, so only it is reifiable.
    """
    out = ((card or {}).get("ports") or {}).get("output")
    return out if isinstance(out, dict) and out else None


def _slug_run(run_id: str) -> str:
    """Strip the tr_/pr_ prefix so the artifact id stays terse but unique (run id is unique)."""
    for pre in ("tr_", "pr_"):
        if run_id.startswith(pre):
            return run_id[len(pre):]
    return run_id


def _jsonable(v):
    import dataclasses
    try:
        import numpy as np
    except Exception:
        np = None
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _jsonable(dataclasses.asdict(v))  # result dataclass -> dict (arrays handled below)
    if np is not None and isinstance(v, np.generic):
        return v.item()
    if np is not None and isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _save_blob(value, aid: str) -> tuple[str, dict, int]:
    """Persist one material; return (repo-relative path, sample digest, size_bytes)."""
    try:
        import numpy as np
    except Exception:
        np = None
    os.makedirs(DATA, exist_ok=True)
    import dataclasses
    def _is_dc(x):
        return dataclasses.is_dataclass(x) and not isinstance(x, type)
    is_dc = _is_dc(value)
    is_dc_list = isinstance(value, list) and bool(value) and all(_is_dc(x) for x in value)
    if is_dc or is_dc_list:  # result dataclass(es) -> asdict -> json; sample stays a digest, never the arrays
        fp = os.path.join(DATA, f"{aid}.json")
        with open(fp, "w") as f:
            json.dump(_jsonable(value), f)
        if is_dc:
            sample = {"dataclass": type(value).__name__,
                      "fields": [fld.name for fld in dataclasses.fields(value)]}
        else:
            sample = {"len": len(value), "dataclass": type(value[0]).__name__,
                      "fields": [fld.name for fld in dataclasses.fields(value[0])]}
        return os.path.relpath(fp, REPO), sample, os.path.getsize(fp)
    is_arr = np is not None and isinstance(value, np.ndarray)
    is_arr_dict = (np is not None and isinstance(value, dict) and bool(value)
                   and all(isinstance(v, np.ndarray) for v in value.values()))
    if is_arr:
        fp = os.path.join(DATA, f"{aid}.npz")
        np.savez_compressed(fp, value=value)
        sample = {"shape": list(value.shape), "dtype": str(value.dtype)}
    elif is_arr_dict:
        fp = os.path.join(DATA, f"{aid}.npz")
        np.savez_compressed(fp, **value)
        sample = {"keys": list(value), **{f"{k}_shape": list(v.shape) for k, v in value.items()}}
    else:
        fp = os.path.join(DATA, f"{aid}.json")
        with open(fp, "w") as f:
            json.dump(_jsonable(value), f)
        if isinstance(value, dict):
            sample = {"keys": list(value)}
        elif isinstance(value, (list, tuple)):
            sample = {"len": len(value)}
        else:
            sample = {"value": _jsonable(value)}
    return os.path.relpath(fp, REPO), sample, os.path.getsize(fp)


def _is_file_ref(v) -> bool:
    """True iff v is a path to a tool-written FILE — reified by REFERENCE, not copied (Slice 1.5b).

    A pathlib.Path / os.PathLike is unambiguously a file path (the kernel-gen Tools return
    Path / Tuple[Path,Path]); a bare str counts only if it points at an existing file, so a
    plain string scalar material never gets mistaken for a reference.
    """
    if isinstance(v, os.PathLike):
        return True
    return isinstance(v, str) and bool(v) and os.path.isfile(v)


def _reference_material(value) -> tuple[str, dict, int | None]:
    """A file-reference material (ADR-0007 Slice 1.5b): the producing Tool wrote a FILE in its
    own location (e.g. a SPICE kernel); the card REFERENCES it — no blob copy. Returns
    (path, sample-digest, size_bytes|None). Path is repo-relative when the file lives under the
    repo, else absolute. The file need not exist yet (sample.exists records it); a dormant Tool's
    output location is still a valid handle."""
    ap = os.path.abspath(os.fspath(value))
    rel = os.path.relpath(ap, REPO)
    path = ap if rel.startswith(os.pardir) else rel  # absolute iff outside the repo
    exists = os.path.isfile(ap)
    sample = {"file": os.path.basename(ap), "ext": os.path.splitext(ap)[1], "exists": exists}
    size = os.path.getsize(ap) if exists else None
    return path, sample, size


def _materials(rv, omap: dict, op: str):
    """Yield (port_name, artifact_type, value) for each reifiable output.

    op='map': rv is a list of per-element returns -> one stacked 'set' value per declared output.
    op='compose': dispatch on rv shape (dict / tuple-list / single).
    """
    try:
        import numpy as np
    except Exception:
        np = None

    if op == "map":
        elems = list(rv or [])
        for name, atype in omap.items():
            vals = []
            for el in elems:
                if isinstance(el, dict) and name in el:
                    vals.append(el[name])
                else:
                    vals = None
                    break
            if vals is None:
                continue  # element shape doesn't expose this named output — skip (logged by caller)
            if np is not None and vals and all(isinstance(v, np.ndarray) for v in vals):
                try:
                    yield name, atype, np.stack(vals)
                    continue
                except Exception:
                    pass
            yield name, atype, vals  # heterogeneous set -> json list
        return

    # compose
    if isinstance(rv, dict):
        matching = [n for n in omap if n in rv]
        if matching:  # dict-of-named-outputs: reify each named key (e.g. sample_so3_pool's q_pool_wxyz)
            for name in matching:
                yield name, omap[name], rv[name]
        else:  # the dict IS the single material (e.g. geometry_data, lit_status_dict, mcmc result)
            name, atype = next(iter(omap.items()))
            yield name, atype, rv
    elif isinstance(rv, list) and len(omap) == 1:
        # a homogeneous list is ONE material (e.g. multi_start_optimize's [OptimizationResult, ...]
        # is one candidate-set), NOT position 0. Python multi-returns are tuples, never lists,
        # so this never collides with positional dispatch below.
        name, atype = next(iter(omap.items()))
        yield name, atype, rv
    elif isinstance(rv, (list, tuple)):
        # positional multi-return: zip omap entries to positions in order. A key written
        # 'name@<idx>' selects a NON-leading position (e.g. 'quats@2' skips k1/k2).
        for i, (raw, atype) in enumerate(omap.items()):
            idx, name = i, raw
            if "@" in raw:
                base, _, suf = raw.rpartition("@")
                if suf.isdigit():
                    idx, name = int(suf), base
            if idx < len(rv):
                yield name, atype, rv[idx]
    else:
        for name, atype in omap.items():  # single value -> first declared output
            yield name, atype, rv
            break


def reify(rv, card: dict, produced_by: dict, op: str = "compose", seq_start: int = 0,
          now: str = "", commit: str = "") -> tuple[list[dict], int]:
    """Reify a run/step's port-typed outputs into artifact_instance cards.

    Returns (cards_written, next_seq). No-op (returns [], seq_start) when the Tool has no
    keyed output map — the 'recognised materials only' gate. Cards are written to disk here;
    the caller records their ids on the run/step record (artifacts_produced).
    """
    omap = output_map(card)
    if not omap:
        return [], seq_start
    cardinality = "set" if op == "map" else "one"
    slug = _slug_run(produced_by["run"])
    os.makedirs(INSTANCES, exist_ok=True)
    cards, seq = [], seq_start
    for name, atype, value in _materials(rv, omap, op):
        aid = f"ai_{slug}_{seq}"
        seq += 1
        storage = "blob"
        try:
            if _is_file_ref(value):  # Slice 1.5b: material is a tool-written file — reference, don't copy
                path, sample, size = _reference_material(value)
                storage = "reference"
            else:
                path, sample, size = _save_blob(value, aid)
        except Exception as e:  # never let an unserializable output break the run
            sys.stderr.write(f"[reify] skipped {atype} ('{name}') from {produced_by['run']}: "
                             f"unserializable ({type(e).__name__}: {e})\n")
            continue
        cardinality_n = None
        if cardinality == "set":
            try:
                cardinality_n = int(len(value)) if not hasattr(value, "shape") else int(value.shape[0])
            except Exception:
                cardinality_n = None
        cap = f"{atype} . from {produced_by.get('step') or card.get('id')}"
        rec = {
            "schema_version": "1.0.0", "id": aid, "kind": "artifact_instance",
            "artifact_type": atype, "cardinality": cardinality,
            "produced_by": {"run": produced_by["run"], "step": produced_by.get("step")},
            "port": name, "path": path, "cardinality_n": cardinality_n,
            "size_bytes": size, "sample": sample, "caption": cap,
            "commit": commit, "created_at": now,
        }
        if storage != "blob":  # default stays implicit so the 35 blob cards are untouched
            rec["storage"] = storage
        fp = os.path.join(INSTANCES, f"{aid}.json")
        with open(fp, "w") as f:
            json.dump(rec, f, indent=2)
            f.write("\n")
        cards.append(rec)
    return cards, seq


def instance_card(aid: str) -> dict:
    """Load an artifact_instance CARD by id. Raises FileNotFoundError on an unknown id."""
    fp = os.path.join(INSTANCES, f"{aid}.json")
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"unknown artifact_instance '{aid}' (no {os.path.relpath(fp, REPO)})")
    return json.load(open(fp))


def load(aid: str):
    """Load a stored material by artifact_instance id (the $artifact.<id> re-feed seam, Slice 2).

    Returns the in-memory value: the single array for a 1-array npz, a dict for a multi-array
    npz, or the parsed object for a json blob. For a storage='reference' material (Slice 1.5b)
    returns the file PATH instead — you furnsh a SPICE kernel, you don't parse it into memory; a
    downstream consumer takes the path. Raises FileNotFoundError on an unknown id.
    """
    card = instance_card(aid)
    if card.get("storage") == "reference":
        p = card["path"]
        return p if os.path.isabs(p) else os.path.join(REPO, p)
    blob = os.path.join(REPO, card["path"])
    if blob.endswith(".npz"):
        import numpy as np
        with np.load(blob, allow_pickle=False) as z:
            keys = list(z.keys())
            return z["value"] if keys == ["value"] else {k: z[k] for k in keys}
    return json.load(open(blob))


# --- Slice 2: re-feed ($artifact.<id>) — pour a stored material into a new run -------------
#
# A run's input value may be a REFERENCE to a shelved material instead of an inline literal:
#   string  "$artifact.<id>"            (parity with the pipeline ref language's $in/$steps)
#   string  "$artifact.<id>.<subkey>"   (index into a dict/multi-array blob)
#   dict    {"$artifact": "<id>"}        (run_tool --input JSON form; optional "key": "<subkey>")
# The resolver LOADS the blob (load() above) and TYPE-CHECKS it against the consuming input port
# — the port type-system's payoff: you can't pour an ia-cloud into a light-curve port. This is the
# "working pantry" half of the shelf (Girish, 2026-06-07): a stored material re-cooked, no recompute.


class ArtifactTypeError(Exception):
    """A re-fed material's artifact_type does not match the consuming input port (Slice 2)."""


def is_artifact_ref(val) -> bool:
    """True iff val is a $artifact re-feed reference (string '$artifact.…' or {'$artifact': id})."""
    if isinstance(val, str):
        return val.startswith("$artifact.")
    return isinstance(val, dict) and "$artifact" in val


def _parse_artifact_ref(val) -> tuple[str, list[str]]:
    """(artifact_id, subkeys) from either ref form. Instance ids carry no '.', so the dotted
    string form parses unambiguously: the id is the first token, the rest are subkeys."""
    if isinstance(val, str):
        parts = val[len("$artifact."):].split(".")
        return parts[0], parts[1:]
    aid = val["$artifact"]
    key = val.get("key")
    return aid, ([key] if key else [])


def _port_want(card: dict | None, kwarg: str):
    """The type constraint a consuming kwarg imposes, or None if unconstrained.

      keyed-map input  {kwarg: type}  -> the declared type (None if kwarg absent: unconstrained).
      legacy-list input [type, …]     -> ('__list__', [types]) — membership-only (no kwarg binding).
      absent / empty                  -> None (unconstrained).
    """
    pin = ((card or {}).get("ports") or {}).get("input")
    if isinstance(pin, dict):
        return pin.get(kwarg)
    if isinstance(pin, list) and pin:
        return ("__list__", pin)
    return None


def typecheck_ref(val, card: dict | None = None, kwarg: str | None = None) -> dict:
    """Validate a $artifact ref against the consuming input port; return its instance card.

    Raises FileNotFoundError (unknown id) or ArtifactTypeError (type mismatch). With no
    card/kwarg the existence check still runs but the type constraint is skipped (e.g. a map
    step's `over`, which has no named port).
    """
    aid, _ = _parse_artifact_ref(val)
    inst = instance_card(aid)
    if card is None or kwarg is None:
        return inst
    atype = inst.get("artifact_type")
    want = _port_want(card, kwarg)
    if want is None:
        return inst  # unconstrained port — pour anything
    if isinstance(want, tuple) and want[0] == "__list__":
        if atype not in want[1]:
            raise ArtifactTypeError(
                f"artifact '{aid}' is '{atype}' but Tool '{card.get('id')}' accepts {want[1]}")
    elif atype != want:
        raise ArtifactTypeError(
            f"artifact '{aid}' is '{atype}' but port '{kwarg}' of Tool "
            f"'{card.get('id')}' wants '{want}'")
    return inst


def load_ref(val):
    """Load the in-memory material a $artifact ref points at (no type-check). Honours subkeys."""
    aid, subkeys = _parse_artifact_ref(val)
    value = load(aid)
    for k in subkeys:
        value = value[k]
    return value


def resolve_supplied(supplied: dict, card: dict | None) -> dict:
    """Replace any $artifact ref VALUES with their loaded material, type-checked per kwarg.

    The store side of run_tool's --input re-feed: non-ref values pass through untouched. Raises
    FileNotFoundError / ArtifactTypeError (the caller turns these into a REFUSED exit).
    """
    out = {}
    for kw, v in supplied.items():
        if is_artifact_ref(v):
            typecheck_ref(v, card, kw)
            out[kw] = load_ref(v)
        else:
            out[kw] = v
    return out
