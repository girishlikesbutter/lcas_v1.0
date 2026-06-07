import { useEffect, useRef } from "react";
import { useTerminals } from "../terminal/TerminalsProvider";
import { IconTerminal } from "../ui/icons";

// The dedicated terminals workspace. Rather than mount a second set of xterm
// instances (which would double-connect each PTY), this page simply expands the
// single persistent dock to full-screen and restores it on leave. One session per
// PTY, always — "multiple terminals in the browser", focused work in-terminal.

export function TerminalsView() {
  const { dock, setDock, openTerminal, sessions } = useTerminals();
  const prev = useRef(dock);

  useEffect(() => {
    prev.current = dock === "full" ? "half" : dock;
    setDock("full");
    if (sessions.length === 0) openTerminal();
    return () => setDock(prev.current === "hidden" ? "hidden" : prev.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex h-[60vh] flex-col items-center justify-center gap-3 text-[var(--color-faint)]">
      <IconTerminal width={28} height={28} />
      <div className="text-[13px]">Terminal workspace is full-screen below.</div>
      <div className="text-[11px]">Run <span className="kbd">claude</span> inside a terminal to start a session · <span className="kbd">⌃`</span> toggles the dock anywhere.</div>
    </div>
  );
}
