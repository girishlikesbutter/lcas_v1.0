import { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";

// One xterm instance bound to one backend PTY over a WebSocket. The PTY persists
// server-side, so unmounting/remounting (e.g. toggling the dock) reconnects and the
// scrollback buffer replays — the session is never lost.

const THEME = {
  background: "#0d1117",
  foreground: "#c9d1d9",
  cursor: "#58a6ff",
  cursorAccent: "#0d1117",
  selectionBackground: "#1f6feb55",
  black: "#0d1117", red: "#f0594f", green: "#3fb950", yellow: "#e8a33d",
  blue: "#58a6ff", magenta: "#bc8cff", cyan: "#39c5cf", white: "#c9d1d9",
  brightBlack: "#6e7681", brightRed: "#ff7b72", brightGreen: "#56d364",
  brightYellow: "#e3b341", brightBlue: "#79c0ff", brightMagenta: "#d2a8ff",
  brightCyan: "#56d4dd", brightWhite: "#f0f6fc",
};

export function TermSession({
  termId, active, initialInput,
}: { termId: string; active: boolean; initialInput?: string }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sentInitRef = useRef(false);

  useEffect(() => {
    if (!hostRef.current) return;
    const term = new Terminal({
      fontFamily: '"JetBrains Mono", ui-monospace, monospace',
      fontSize: 13,
      lineHeight: 1.2,
      cursorBlink: true,
      theme: THEME,
      allowProposedApi: true,
      scrollback: 5000,
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.loadAddon(new WebLinksAddon());
    term.open(hostRef.current);
    termRef.current = term;
    fitRef.current = fit;
    try { fit.fit(); } catch { /* not laid out yet */ }

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/terminal/${termId}`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    // Deep-link payload (e.g. `cc "/orient"\n`) is flushed once the PTY emits its
    // first bytes — i.e. the shell prompt is actually up — rather than on a fixed
    // 350 ms timer that loses the race when the box is slow to spawn the shell. A
    // hard fallback covers the (unexpected) case of a prompt that never prints.
    let settleTimer = 0;
    let fallbackTimer = 0;
    const flushInit = () => {
      if (!initialInput || sentInitRef.current || ws.readyState !== 1) return;
      sentInitRef.current = true;
      ws.send(JSON.stringify({ t: "in", d: initialInput }));
    };

    ws.onopen = () => {
      const { cols, rows } = term;
      ws.send(JSON.stringify({ t: "size", cols, rows }));
      if (initialInput) fallbackTimer = window.setTimeout(flushInit, 1500);
    };
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) term.write(new Uint8Array(ev.data));
      else term.write(ev.data);
      // first output = prompt is rendering; debounce-settle, then type the command.
      if (initialInput && !sentInitRef.current) {
        window.clearTimeout(settleTimer);
        settleTimer = window.setTimeout(flushInit, 220);
      }
    };
    ws.onclose = () => term.write("\r\n\x1b[2m[disconnected]\x1b[0m\r\n");

    const dataSub = term.onData((d) => ws.readyState === 1 && ws.send(JSON.stringify({ t: "in", d })));
    const resizeSub = term.onResize(({ cols, rows }) =>
      ws.readyState === 1 && ws.send(JSON.stringify({ t: "size", cols, rows })),
    );

    // Debounce fit to one call per animation frame: during a tiling spawn/close
    // the container resizes many times: coalescing avoids a flood of PTY resizes
    // and the stretched-glyph flicker, while still settling on the final size.
    let fitRaf = 0;
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(fitRaf);
      fitRaf = requestAnimationFrame(() => { try { fit.fit(); } catch { /* */ } });
    });
    ro.observe(hostRef.current);

    return () => {
      window.clearTimeout(settleTimer);
      window.clearTimeout(fallbackTimer);
      cancelAnimationFrame(fitRaf);
      dataSub.dispose();
      resizeSub.dispose();
      ro.disconnect();
      ws.close();
      term.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [termId]);

  // refit + focus when this tab becomes active
  useEffect(() => {
    if (active) {
      const t = setTimeout(() => {
        try { fitRef.current?.fit(); } catch { /* */ }
        termRef.current?.focus();
      }, 40);
      return () => clearTimeout(t);
    }
  }, [active]);

  // Visibility is the dock's job now: in tiled mode every session is shown, in
  // tabbed mode the dock hides the inactive panes. This div always fills its host.
  return <div ref={hostRef} className="h-full w-full" />;
}
