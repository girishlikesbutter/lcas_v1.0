import { useCallback, useState, type CSSProperties, type ReactNode } from "react";
import { useTerminals, type DockHeight } from "./TerminalsProvider";
import { TermSession } from "./TermSession";
import {
  IconPlus, IconClose, IconTerminal, IconChevron, IconTiled, IconTabbed,
} from "../ui/icons";

// The persistent embedded terminal pane (PLAN §5). Mounted ONCE at the app shell so
// PTY sessions and their xterm instances survive view switches — "focused work stays
// in-terminal". Two layouts: TILED (all PTYs visible at once, auto-tiled
// Hyprland-style — the default) and TABBED (one pane + a tab strip, kept for narrow
// docks). Height cycles bar → half → full; panels deep-link in via launchSkill().

const HEIGHTS: Record<DockHeight, string> = {
  hidden: "0px",
  bar: "38px",
  half: "46vh",
  full: "calc(100vh - 56px)",
};

/** Auto-tile geometry: a near-square grid; the last tile spans any empty trailing
 *  columns so a 3rd/5th/7th tile fills its row instead of leaving a gap. */
function tileGeometry(n: number): { style: CSSProperties; lastSpan: number } {
  const cols = Math.max(1, Math.ceil(Math.sqrt(n)));
  const rows = Math.max(1, Math.ceil(n / cols));
  const remainder = n % cols;
  const lastSpan = remainder === 0 ? 1 : cols - remainder + 1;
  return {
    style: {
      gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
      gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
    },
    lastSpan,
  };
}

export function TerminalDock() {
  const {
    sessions, activeId, dock, setDock, layout, setLayout,
    setActive, openTerminal, closeTerminal,
  } = useTerminals();
  const visible = dock !== "hidden";

  // Keep a closing tile mounted for one exit-animation beat before it's removed,
  // so the spawn/close feels alive instead of popping out.
  const [closing, setClosing] = useState<Set<string>>(new Set());
  const requestClose = useCallback((id: string) => {
    setClosing((s) => new Set(s).add(id));
    window.setTimeout(() => {
      closeTerminal(id);
      setClosing((s) => { const n = new Set(s); n.delete(id); return n; });
    }, 180);
  }, [closeTerminal]);

  return (
    <div
      className="shrink-0 overflow-hidden border-t border-[var(--color-border)] bg-[var(--color-bg-2)] transition-[height] duration-200 ease-out"
      style={{ height: visible ? HEIGHTS[dock] : "0px" }}
    >
      {/* control bar */}
      <div className="flex h-[38px] items-center gap-1 border-b border-[var(--color-border)] bg-[var(--color-panel-2)] px-2">
        <IconTerminal className="mr-1 text-[var(--color-faint)]" width={15} height={15} />

        {layout === "tabbed" ? (
          <TabStrip
            sessions={sessions}
            activeId={activeId}
            onSelect={setActive}
            onClose={requestClose}
            onNew={() => openTerminal()}
          />
        ) : (
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <button
              onClick={() => openTerminal()}
              title="New terminal"
              className="flex items-center gap-1 rounded-md px-2 py-1 text-[12px] text-[var(--color-muted)] hover:bg-[var(--color-elev)] hover:text-[var(--color-fg)]"
            >
              <IconPlus width={13} height={13} /> new
            </button>
            {sessions.length > 0 && (
              <span className="text-[11px] text-[var(--color-faint)]">
                {sessions.length} tiled
              </span>
            )}
          </div>
        )}

        {/* layout + height controls */}
        <div className="flex items-center gap-1">
          <div className="mr-1 flex items-center gap-0.5 rounded-md border border-[var(--color-border)] p-0.5">
            <LayoutBtn label="Tile all sessions" active={layout === "tiled"} onClick={() => setLayout("tiled")}>
              <IconTiled width={13} height={13} />
            </LayoutBtn>
            <LayoutBtn label="Single pane + tabs" active={layout === "tabbed"} onClick={() => setLayout("tabbed")}>
              <IconTabbed width={13} height={13} />
            </LayoutBtn>
          </div>
          <DockBtn label="half" active={dock === "half"} onClick={() => setDock("half")} />
          <DockBtn label="full" active={dock === "full"} onClick={() => setDock("full")} />
          <button
            title={dock === "bar" ? "Expand" : "Minimise"}
            onClick={() => setDock(dock === "bar" ? "half" : "bar")}
            className="flex h-6 w-6 items-center justify-center rounded text-[var(--color-muted)] hover:bg-[var(--color-elev)] hover:text-[var(--color-fg)]"
          >
            <IconChevron width={14} height={14} style={{ transform: dock === "bar" ? "rotate(-90deg)" : "rotate(90deg)" }} />
          </button>
          <button
            title="Hide dock"
            onClick={() => setDock("hidden")}
            className="flex h-6 w-6 items-center justify-center rounded text-[var(--color-muted)] hover:bg-[var(--color-red)]/20 hover:text-[var(--color-red)]"
          >
            <IconClose width={14} height={14} />
          </button>
        </div>
      </div>

      {/* terminal surfaces */}
      <div className="relative h-[calc(100%-38px)] bg-[#0d1117]">
        {sessions.length === 0 ? (
          <EmptyState onNew={() => openTerminal()} />
        ) : layout === "tabbed" ? (
          // stacked panes, only the active one shown (PTYs all stay mounted)
          sessions.map((s) => (
            <div key={s.id} className="absolute inset-0" style={{ display: activeId === s.id ? "block" : "none" }}>
              <TermSession termId={s.id} active={activeId === s.id} initialInput={s.initialInput} />
            </div>
          ))
        ) : (
          <TiledGrid
            sessions={sessions}
            activeId={activeId}
            closing={closing}
            onFocus={setActive}
            onClose={requestClose}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// tiled layout — every session visible at once
// ---------------------------------------------------------------------------

function TiledGrid({
  sessions, activeId, closing, onFocus, onClose,
}: {
  sessions: { id: string; title: string; initialInput?: string }[];
  activeId: string | null;
  closing: Set<string>;
  onFocus: (id: string) => void;
  onClose: (id: string) => void;
}) {
  const { style, lastSpan } = tileGeometry(sessions.length);
  return (
    <div className="grid h-full w-full gap-1.5 p-1.5" style={style}>
      {sessions.map((s, i) => {
        const active = activeId === s.id;
        const isLast = i === sessions.length - 1;
        return (
          <div
            key={s.id}
            onMouseDown={() => onFocus(s.id)}
            className={`flex min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border bg-[#0d1117] transition-[border-color,box-shadow] ${
              closing.has(s.id) ? "tile-exit" : "tile-enter"
            } ${active ? "border-[var(--color-blue)]" : "border-[var(--color-border)]"}`}
            style={{
              ...(isLast && lastSpan > 1 ? { gridColumn: `span ${lastSpan}` } : null),
              boxShadow: active ? "0 0 0 1px var(--color-blue), 0 8px 30px -16px #58a6ff66" : undefined,
            }}
          >
            <TileHeader s={s} active={active} onClose={() => onClose(s.id)} />
            <div className="relative min-h-0 flex-1">
              <TermSession termId={s.id} active={active} initialInput={s.initialInput} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TileHeader({
  s, active, onClose,
}: { s: { id: string; title: string }; active: boolean; onClose: () => void }) {
  return (
    <div
      className={`flex h-[26px] shrink-0 items-center gap-1.5 border-b px-2 text-[11px] ${
        active
          ? "border-[var(--color-border-2)] bg-[var(--color-elev)]"
          : "border-[var(--color-border)] bg-[var(--color-panel-2)]"
      }`}
    >
      <span className="font-mono text-[10px] text-[var(--color-blue)]">{s.id}</span>
      <span className="min-w-0 flex-1 truncate text-[var(--color-muted)]">{s.title}</span>
      <button
        onMouseDown={(e) => { e.stopPropagation(); onClose(); }}
        title="Close terminal"
        className="rounded p-0.5 text-[var(--color-faint)] hover:bg-[var(--color-red)]/20 hover:text-[var(--color-red)]"
      >
        <IconClose width={11} height={11} />
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// tabbed layout — single pane + tab strip (the previous model, kept as fallback)
// ---------------------------------------------------------------------------

function TabStrip({
  sessions, activeId, onSelect, onClose, onNew,
}: {
  sessions: { id: string; title: string }[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onClose: (id: string) => void;
  onNew: () => void;
}) {
  return (
    <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
      {sessions.map((s, i) => (
        <button
          key={s.id}
          onClick={() => onSelect(s.id)}
          className={`group flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[12px] transition-colors ${
            activeId === s.id
              ? "bg-[var(--color-elev)] text-[var(--color-fg)]"
              : "text-[var(--color-muted)] hover:bg-[var(--color-elev)]/50"
          }`}
        >
          <span className="font-mono text-[11px] text-[var(--color-blue)]">{s.id}</span>
          <span className="max-w-[120px] truncate">{s.title || `terminal ${i + 1}`}</span>
          <span
            role="button"
            onClick={(e) => { e.stopPropagation(); onClose(s.id); }}
            className="rounded p-0.5 opacity-0 transition-opacity hover:bg-[var(--color-red)]/20 group-hover:opacity-100"
          >
            <IconClose width={11} height={11} />
          </span>
        </button>
      ))}
      <button
        onClick={onNew}
        title="New terminal"
        className="flex items-center gap-1 rounded-md px-2 py-1 text-[12px] text-[var(--color-muted)] hover:bg-[var(--color-elev)] hover:text-[var(--color-fg)]"
      >
        <IconPlus width={13} height={13} /> new
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------

function EmptyState({ onNew }: { onNew: () => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-[13px] text-[var(--color-faint)]">
      <IconTerminal width={26} height={26} />
      <div>No terminals open.</div>
      <button
        onClick={onNew}
        className="rounded-md border border-[var(--color-border-2)] px-3 py-1.5 text-[12px] text-[var(--color-fg-dim)] hover:border-[var(--color-blue)] hover:text-[var(--color-blue)]"
      >
        + open a terminal
      </button>
      <div className="text-[11px]">tip: deep-link a skill, or type <span className="kbd">claude</span> to start a session</div>
    </div>
  );
}

function DockBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded px-1.5 py-0.5 text-[11px] ${
        active ? "bg-[var(--color-elev)] text-[var(--color-blue)]" : "text-[var(--color-faint)] hover:text-[var(--color-fg)]"
      }`}
    >
      {label}
    </button>
  );
}

function LayoutBtn({
  label, active, onClick, children,
}: { label: string; active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      title={label}
      onClick={onClick}
      className={`flex h-5 w-6 items-center justify-center rounded ${
        active ? "bg-[var(--color-elev)] text-[var(--color-blue)]" : "text-[var(--color-faint)] hover:text-[var(--color-fg)]"
      }`}
    >
      {children}
    </button>
  );
}
