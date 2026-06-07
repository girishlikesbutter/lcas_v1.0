#!/usr/bin/env python3
"""Research OS web app — FastAPI backend (PLAN §5/§11, Phase 4).

The surface = a derived read-model + a write-only control plane (Q1/Q11):

  * GET  /api/snapshot         the whole read-model (indexer.build_snapshot)
  * GET  /api/events           SSE: {"type":"store"|"stream"|"intents","rev":...} on change
  * GET  /api/stream/manifest  plot-stream manifest (visualisation-as-we-go)
  * GET  /stream/<file>        plot-stream images
  * GET  /api/terminals        list live PTYs
  * POST /api/terminals        spawn a PTY      DELETE /api/terminals/{id}  reap it
  * WS   /ws/terminal/{id}      attach to a PTY (multiple browsers may share)
  * GET  /api/intents          queued intents   POST /api/intent  enqueue one
  * /                          the built frontend (dist/), if present

A ``watchfiles`` task watches the canonical store + the plot stream and broadcasts a
revision over SSE; the UI refetches. The backend never writes research state.
"""
from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import indexer
import intents
from terminals import TerminalManager

RESEARCH_OS = indexer.RESEARCH_OS
STREAM_DIR = os.path.join(RESEARCH_OS, "render", "stream")
DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")
DIST_DIR = os.path.abspath(DIST_DIR)

# No CORS middleware: the app is single-origin in both modes — prod serves dist/
# from this process, dev proxies /api·/stream·/ws through vite (vite.config.ts). A
# wildcard CORS would only widen the surface of a 127.0.0.1-bound control plane.
app = FastAPI(title="Research OS")

terminals = TerminalManager()

# --- SSE pub/sub ----------------------------------------------------------
_subscribers: set[asyncio.Queue] = set()


def _broadcast(event: dict):
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except Exception:
            pass


# categorise a changed path into an SSE event channel
_STORE_DIRS = ("records", "goals", "claims", "pipelines", "substrate", "contracts", "glossary")


@app.on_event("startup")
async def _start_watcher():
    app.state.rev = indexer.store_revision()
    app.state.watch_task = asyncio.create_task(_watch_loop())


@app.on_event("shutdown")
async def _stop():
    terminals.shutdown()
    t = getattr(app.state, "watch_task", None)
    if t:
        t.cancel()


async def _watch_loop():
    from watchfiles import awatch
    watch_paths = [os.path.join(RESEARCH_OS, d) for d in _STORE_DIRS]
    # render/ holds derived artifacts (the plot stream + the frontier ranking);
    # queue/ is the control plane. None of these is the canonical store.
    watch_paths += [os.path.join(RESEARCH_OS, "render"), os.path.join(RESEARCH_OS, "queue")]
    watch_paths = [p for p in watch_paths if os.path.isdir(p)]
    try:
        async for changes in awatch(*watch_paths, recursive=True, step=200):
            kinds = set()
            for _change, path in changes:
                if path.endswith("frontier_ranking.json") or path.endswith("machinery_map.json"):
                    # both are derived render artifacts the snapshot embeds; a fresh
                    # one doesn't move the store rev, so signal the dedicated refetch channel
                    kinds.add("ranking")
                elif "/render/stream" in path:
                    kinds.add("stream")
                elif "/queue" in path:
                    kinds.add("intents")
                elif any(f"/{d}/" in path or path.endswith(f"/{d}") for d in _STORE_DIRS):
                    kinds.add("store")
            if "store" in kinds:
                # store_revision globs + stats + hashes the whole store; keep that
                # off the event-loop thread so SSE/WS pumps aren't stalled on a big store.
                app.state.rev = await asyncio.to_thread(indexer.store_revision)
                _broadcast({"type": "store", "rev": app.state.rev})
            if "stream" in kinds:
                _broadcast({"type": "stream"})
            if "intents" in kinds:
                _broadcast({"type": "intents"})
            if "ranking" in kinds:
                # ranking is derived; a fresh one doesn't change the store rev, so
                # signal a dedicated channel the UI refetches the snapshot on.
                _broadcast({"type": "ranking"})
    except asyncio.CancelledError:
        pass


# --- read-model -----------------------------------------------------------
@app.get("/api/snapshot")
def snapshot():
    return JSONResponse(indexer.build_snapshot())


@app.get("/api/events")
async def events(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.add(q)

    async def gen():
        # prime with the current rev so a fresh client syncs immediately
        yield f"data: {json.dumps({'type':'store','rev':getattr(app.state,'rev','')})}\n\n"
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=20)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # comment frame
                if await request.is_disconnected():
                    break
        finally:
            _subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --- plot stream ----------------------------------------------------------
@app.get("/api/stream/manifest")
def stream_manifest():
    mpath = os.path.join(STREAM_DIR, "manifest.json")
    if os.path.exists(mpath):
        return JSONResponse(json.load(open(mpath)))
    return JSONResponse({"items": []})


@app.get("/stream/{name}")
def stream_file(name: str):
    safe = os.path.basename(name)
    path = os.path.join(STREAM_DIR, safe)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "not found"}, status_code=404)


# --- terminals ------------------------------------------------------------
@app.get("/api/terminals")
def list_terminals():
    return {"terminals": terminals.list()}


@app.post("/api/terminals")
async def create_terminal(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    t = terminals.create(title=body.get("title", ""),
                         cols=int(body.get("cols", 80)), rows=int(body.get("rows", 24)))
    return {"id": t.id, "title": t.title}


@app.delete("/api/terminals/{tid}")
async def delete_terminal(tid: str):
    # async so close() runs on the event-loop thread — its remove_reader + the
    # call_later-based child reaper must touch the loop from the loop's own thread.
    terminals.close(tid)
    return {"ok": True}


@app.websocket("/ws/terminal/{tid}")
async def terminal_ws(ws: WebSocket, tid: str):
    await ws.accept()
    if tid not in terminals.terms:
        await ws.close(code=4404)
        return
    q: asyncio.Queue = asyncio.Queue()
    scrollback = terminals.attach(tid, q)
    if scrollback:
        await ws.send_bytes(scrollback)

    async def pump_out():
        try:
            while True:
                data = await q.get()
                if data is None:  # shell exited / detached
                    await ws.close(code=4000)
                    return
                await ws.send_bytes(data)
        except Exception:
            pass

    out_task = asyncio.create_task(pump_out())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                terminals.write(tid, msg["bytes"])
            elif msg.get("text") is not None:
                try:
                    frame = json.loads(msg["text"])
                except Exception:
                    continue
                if frame.get("t") == "in":
                    terminals.write(tid, frame.get("d", "").encode("utf-8", "replace"))
                elif frame.get("t") == "size":
                    terminals.resize(tid, int(frame.get("rows", 24)), int(frame.get("cols", 80)))
    except WebSocketDisconnect:
        pass
    finally:
        terminals.detach(tid, q)
        out_task.cancel()


# --- control plane --------------------------------------------------------
@app.get("/api/intents")
def get_intents():
    return {"intents": intents.listing()}


@app.post("/api/intent")
async def post_intent(request: Request):
    body = await request.json()
    try:
        rec = intents.enqueue(body)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    _broadcast({"type": "intents"})
    return rec


@app.get("/api/health")
def health():
    return {"ok": True, "rev": getattr(app.state, "rev", indexer.store_revision())}


# --- built frontend (single-origin production serve) ----------------------
if os.path.isdir(DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

    @app.get("/")
    def _index():
        return FileResponse(os.path.join(DIST_DIR, "index.html"))

    @app.get("/{full_path:path}")
    def _spa(full_path: str):
        # SPA fallback: serve a real file if it exists, else index.html for client routing
        candidate = os.path.join(DIST_DIR, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8138, log_level="info")
