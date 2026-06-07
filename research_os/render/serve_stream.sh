#!/usr/bin/env bash
# Serve the Research OS plot-stream and open it in a browser (idempotent).
# Per ADR-0003: PNGs in stream/, static index.html polls manifest.json.
# Usage: serve_stream.sh [port]   (default 8137)
set -euo pipefail

PORT="${1:-8137}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/stream" && pwd)"
URL="http://localhost:${PORT}/index.html"

# Already serving our index.html on this port? Don't start a second server.
if python3 - "$PORT" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(0.3)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
then
  echo "plot-stream already served on :${PORT}"
else
  ( cd "$DIR" && nohup python3 -m http.server "$PORT" >/tmp/ro_stream_${PORT}.log 2>&1 & )
  echo "started plot-stream server on :${PORT}  (log: /tmp/ro_stream_${PORT}.log)"
fi

# Open (or focus) the tab. Best-effort — harmless if no display.
( xdg-open "$URL" >/dev/null 2>&1 & ) || true
echo "Plot stream: ${URL}"
