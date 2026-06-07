#!/usr/bin/env bash
# Research OS web app — build + serve (single origin).
#
# Builds the frontend once, then runs the FastAPI backend which serves the built
# dist/ AND the read-model API / SSE / terminals / plot-stream on one port. The
# backend watches the canonical research_os/ store and live-refreshes the UI.
#
# Prod mode binds the host's TAILSCALE IP by default (tailnet-only remote access —
# Girish reaches it from another machine over Tailscale), falling back to 127.0.0.1
# if Tailscale is down. NOT 0.0.0.0 (that would also expose the control plane — live
# PTY shells + the intent queue — to the LAN). Override with HOST=<addr>.
#
# Usage:  research_os/webapp/start.sh [port]      (default 8138; binds tailscale IP)
#         HOST=127.0.0.1 research_os/webapp/start.sh   (force a bind address)
#         HOST=0.0.0.0   research_os/webapp/start.sh   (all interfaces — incl. LAN)
#         research_os/webapp/start.sh --dev       (vite dev server + backend on localhost, hot reload)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="$REPO/.venv/bin/python"
PORT="${1:-8138}"

[ -x "$VENV" ] || { echo "no venv at $VENV — create it / adjust this script"; exit 1; }

# Resolve the bind address: explicit HOST wins; else the Tailscale IPv4 (tailnet-only);
# else localhost. Auto-detect so it survives a tailscale IP change.
resolve_host() {
  if [ -n "${HOST:-}" ]; then echo "$HOST"; return; fi
  local ts; ts="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [ -n "$ts" ]; then echo "$ts"; else echo "127.0.0.1"; fi
}

# ---- dev mode: vite hot-reload + backend -------------------------------------
if [ "${1:-}" = "--dev" ]; then
  echo "▸ backend (uvicorn :8138, reload)…"
  ( cd "$HERE/backend" && "$VENV" -m uvicorn app:app --host 127.0.0.1 --port 8138 --reload ) &
  BACK=$!
  echo "▸ frontend (vite dev :5180)…"
  ( cd "$HERE/frontend" && npm run dev )
  kill $BACK 2>/dev/null || true
  exit 0
fi

# ---- prod mode: build once, serve single-origin ------------------------------
echo "▸ building frontend…"
( cd "$HERE/frontend" && [ -d node_modules ] || npm install )
( cd "$HERE/frontend" && npm run build )

BIND="$(resolve_host)"
URL="http://${BIND}:${PORT}/"
echo "▸ serving Research OS at ${URL}  (bind ${BIND})"
[ "$BIND" = "127.0.0.1" ] && echo "  (tailscale IP not found — localhost only; set HOST= to override)"
( sleep 1.5; xdg-open "$URL" >/dev/null 2>&1 || true ) &
cd "$HERE/backend"
exec "$VENV" -m uvicorn app:app --host "$BIND" --port "$PORT" --log-level info
