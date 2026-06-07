import { useState } from "react";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { StoreProvider, useStore } from "./lib/store";
import { TerminalsProvider, useTerminals } from "./terminal/TerminalsProvider";
import { TerminalDock } from "./terminal/TerminalDock";
import { TrunkBar } from "./shell/TrunkBar";
import { CommandPalette } from "./shell/CommandPalette";
import { Dot } from "./ui/kit";
import {
  IconOverview, IconTree, IconFrontier, IconRuns, IconClaims, IconPipelines,
  IconSubstrate, IconMachinery, IconGlossary, IconStream, IconTerminal, IconSearch, IconQueue,
  IconMaterials,
} from "./ui/icons";

import { Overview } from "./views/Overview";
import { Stream } from "./views/Stream";
import { TerminalsView } from "./views/TerminalsView";
import { GoalTree } from "./views/GoalTree";
import { Frontier } from "./views/Frontier";
import { Runs } from "./views/Runs";
import { Claims } from "./views/Claims";
import { Pipelines } from "./views/Pipelines";
import { Materials } from "./views/Materials";
import { Substrate } from "./views/Substrate";
import { Machinery } from "./views/Machinery";
import { Glossary } from "./views/Glossary";
import { Control } from "./views/Control";

const NAV = [
  { to: "/", label: "Overview", icon: IconOverview, end: true },
  { to: "/tree", label: "Goal tree", icon: IconTree },
  { to: "/frontier", label: "Frontier", icon: IconFrontier },
  { to: "/runs", label: "Runs", icon: IconRuns },
  { to: "/claims", label: "Claims", icon: IconClaims },
  { to: "/pipelines", label: "Pipelines", icon: IconPipelines },
  { to: "/materials", label: "Materials", icon: IconMaterials },
  { to: "/substrate", label: "Substrate", icon: IconSubstrate },
  { to: "/machinery", label: "Machinery", icon: IconMachinery },
  { to: "/glossary", label: "Glossary", icon: IconGlossary },
  { to: "/stream", label: "Plot stream", icon: IconStream, stream: true },
  { to: "/control", label: "Control", icon: IconQueue, intents: true },
  { to: "/terminals", label: "Terminals", icon: IconTerminal },
];

function NavRail({ onSearch }: { onSearch: () => void }) {
  const { streamTick, pendingIntents } = useStore();
  return (
    <nav className="flex w-[208px] shrink-0 flex-col border-r border-[var(--color-border)] bg-[var(--color-bg-2)]/60 px-3 py-4">
      <div className="mb-5 flex items-center gap-2.5 px-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg" style={{ background: "linear-gradient(135deg,#1f6feb,#39c5cf)" }}>
          <span className="text-[15px] font-bold text-[#08090c]">R</span>
        </div>
        <div className="leading-tight">
          <div className="text-[13px] font-semibold text-[var(--color-fg)]">Research OS</div>
          <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--color-faint)]">trust machine</div>
        </div>
      </div>

      <button
        onClick={onSearch}
        className="mb-4 flex items-center gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-2.5 py-2 text-[12px] text-[var(--color-faint)] hover:border-[var(--color-border-2)]"
      >
        <IconSearch width={14} height={14} />
        <span>Search…</span>
        <span className="ml-auto kbd">⌘K</span>
      </button>

      <div className="flex flex-1 flex-col gap-0.5">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            className={({ isActive }) =>
              `group relative flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] transition-colors ${
                isActive
                  ? "bg-[var(--color-elev)] text-[var(--color-fg)]"
                  : "text-[var(--color-muted)] hover:bg-[var(--color-elev)]/50 hover:text-[var(--color-fg-dim)]"
              }`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full transition-opacity"
                  style={{ background: "var(--color-blue)", opacity: isActive ? 1 : 0 }}
                />
                <n.icon width={17} height={17} className={isActive ? "text-[var(--color-blue)]" : ""} />
                <span>{n.label}</span>
                {n.stream && streamTick > 0 && (
                  <span className="ml-auto"><Dot tone="teal" pulse size={6} /></span>
                )}
                {n.intents && pendingIntents > 0 && (
                  <span className="ml-auto rounded-full bg-[var(--color-blue)] px-1.5 text-[10px] font-semibold text-[#08090c]">
                    {pendingIntents}
                  </span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </div>

      <ConnStatus />
    </nav>
  );
}

function ConnStatus() {
  const { connected, snap } = useStore();
  return (
    <div className="mt-3 flex items-center gap-2 px-2 text-[11px] text-[var(--color-faint)]">
      <Dot tone={connected ? "green" : "faint"} pulse={connected} size={7} />
      <span>{connected ? "live" : "connecting…"}</span>
      {snap && <span className="ml-auto font-mono">rev {snap.rev.slice(0, 6)}</span>}
    </div>
  );
}

function TermToggle() {
  const { dock, setDock, sessions } = useTerminals();
  const open = dock !== "hidden";
  return (
    <button
      onClick={() => setDock(open ? "hidden" : "half")}
      className={`flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-[12px] transition-colors ${
        open ? "border-[var(--color-blue)] text-[var(--color-blue)]" : "border-[var(--color-border)] text-[var(--color-muted)] hover:text-[var(--color-fg)]"
      }`}
      title="Toggle terminal dock (⌃`)"
    >
      <IconTerminal width={15} height={15} />
      <span>Terminal</span>
      {sessions.length > 0 && (
        <span className="rounded bg-[var(--color-elev)] px-1.5 text-[10px]">{sessions.length}</span>
      )}
    </button>
  );
}

function Shell() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const loc = useLocation();

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <div className="flex min-h-0 flex-1">
        <NavRail onSearch={() => setPaletteOpen(true)} />
        <div className="flex min-w-0 flex-1 flex-col">
          <TrunkBar right={<TermToggle />} />
          <main className="min-h-0 flex-1 overflow-y-auto">
            <div className="mx-auto max-w-[1320px] px-7 py-6 fadein" key={loc.pathname}>
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/tree" element={<GoalTree />} />
                <Route path="/frontier" element={<Frontier />} />
                <Route path="/runs" element={<Runs />} />
                <Route path="/runs/:id" element={<Runs />} />
                <Route path="/claims" element={<Claims />} />
                <Route path="/pipelines" element={<Pipelines />} />
                <Route path="/pipelines/:id" element={<Pipelines />} />
                <Route path="/materials" element={<Materials />} />
                <Route path="/substrate" element={<Substrate />} />
                <Route path="/machinery" element={<Machinery />} />
                <Route path="/glossary" element={<Glossary />} />
                <Route path="/control" element={<Control />} />
                <Route path="/stream" element={<Stream />} />
                <Route path="/terminals" element={<TerminalsView />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>
      {/* Persistent dock — mounted once so PTYs survive navigation. The /terminals
          route simply expands it to full-screen (single xterm per PTY, no dupes). */}
      <TerminalDock />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      <KeyBindings onPalette={() => setPaletteOpen((v) => !v)} />
    </div>
  );
}

function KeyBindings({ onPalette }: { onPalette: () => void }) {
  const { dock, setDock } = useTerminals();
  // ⌘K palette · ⌃` terminal
  useKeyHandler((e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      onPalette();
    } else if (e.ctrlKey && e.key === "`") {
      e.preventDefault();
      setDock(dock === "hidden" ? "half" : "hidden");
    }
  });
  return null;
}

import { useEffect, useRef } from "react";
function useKeyHandler(fn: (e: KeyboardEvent) => void) {
  // Keep the latest handler in a ref and bind the listener exactly once, so we
  // don't add/remove a window listener on every render.
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    const h = (e: KeyboardEvent) => fnRef.current(e);
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);
}

export default function App() {
  return (
    <BrowserRouter>
      <StoreProvider>
        <TerminalsProvider>
          <Shell />
        </TerminalsProvider>
      </StoreProvider>
    </BrowserRouter>
  );
}
