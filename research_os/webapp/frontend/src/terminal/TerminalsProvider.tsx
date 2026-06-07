import {
  createContext, useContext, useState, useCallback, useRef, type ReactNode,
} from "react";
import { api } from "../lib/api";

export type DockHeight = "hidden" | "bar" | "half" | "full";
/** tiled = all PTYs visible at once (Hyprland-style); tabbed = one pane, tab strip. */
export type DockLayout = "tiled" | "tabbed";

interface Session {
  id: string;       // backend PTY id (t1, t2, …)
  title: string;
  initialInput?: string;
}

interface TermCtx {
  sessions: Session[];
  activeId: string | null;
  dock: DockHeight;
  setDock: (h: DockHeight) => void;
  layout: DockLayout;
  setLayout: (l: DockLayout) => void;
  setActive: (id: string) => void;
  /** spawn a fresh PTY; optionally pre-type a command (deep-link from a panel). */
  openTerminal: (initialInput?: string, title?: string) => Promise<string>;
  /**
   * Spawn a terminal that BOOTS Claude Code and runs a spine/instrument skill.
   * Every skill deep-link goes through here so the `cc "/skill"` convention lives
   * in exactly one place (PHASE4_REVIEW Item 2). `cc` is the user's bash function
   * (`claude --dangerously-skip-permissions "$@"`), so a slash command handed to it
   * as the initial prompt boots Claude Code AND fires the skill — typing `/orient`
   * into a bare shell only errored with "no such file or directory".
   */
  launchSkill: (name: string, arg?: string) => Promise<string>;
  closeTerminal: (id: string) => void;
}

/** Shell-quote a skill invocation for `cc "<…>"`. ids are safe chars; quoting is
 *  defensive (JSON string escaping covers " and \\). */
function ccCommand(name: string, arg?: string): string {
  const slash = arg ? `/${name} ${arg}` : `/${name}`;
  return `cc ${JSON.stringify(slash)}\n`;
}

const Ctx = createContext<TermCtx | null>(null);

export function TerminalsProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [dock, setDock] = useState<DockHeight>("hidden");
  const [layout, setLayout] = useState<DockLayout>("tiled");
  const creating = useRef(false);

  const openTerminal = useCallback(async (initialInput?: string, title?: string) => {
    // Guard against rapid double-clicks on a deep-link spawning two PTYs. The
    // previous version had an empty `if` body, so it never actually returned.
    if (creating.current) return "";
    creating.current = true;
    try {
      const { id } = await api.createTerminal(120, 32, title || "");
      const sess: Session = { id, title: title || `terminal`, initialInput };
      setSessions((s) => [...s, sess]);
      setActiveId(id);
      setDock((d) => (d === "hidden" || d === "bar" ? "half" : d));
      return id;
    } finally {
      creating.current = false;
    }
  }, []);

  const launchSkill = useCallback(
    (name: string, arg?: string) => openTerminal(ccCommand(name, arg), name),
    [openTerminal],
  );

  const closeTerminal = useCallback((id: string) => {
    api.deleteTerminal(id).catch(() => {});
    setSessions((s) => {
      const next = s.filter((x) => x.id !== id);
      setActiveId((cur) => (cur === id ? (next[next.length - 1]?.id ?? null) : cur));
      if (next.length === 0) setDock("hidden");
      return next;
    });
  }, []);

  const setActive = useCallback((id: string) => setActiveId(id), []);

  return (
    <Ctx.Provider
      value={{ sessions, activeId, dock, setDock, layout, setLayout, setActive, openTerminal, launchSkill, closeTerminal }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useTerminals() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useTerminals outside provider");
  return c;
}
