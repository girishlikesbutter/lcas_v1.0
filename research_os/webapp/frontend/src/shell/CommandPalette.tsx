import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import { IconSearch, IconArrowRight } from "../ui/icons";
import { NODE_KIND, RUN_STATUS, CLAIM_STATUS, type Tone } from "../lib/ui";
import { TONE_HEX } from "../lib/ui";

interface Item {
  id: string;
  label: string;
  sub: string;
  kind: string;
  tone: Tone;
  to?: string;
  action?: () => void;
}

// ⌘K — jump to any object, or fire a deep-link skill into a terminal. Kills
// path-hunting; "do whatever I need" starts here.
export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { snap } = useStore();
  const { launchSkill } = useTerminals();
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setQ("");
      setSel(0);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [open]);

  const items = useMemo<Item[]>(() => {
    if (!snap) return [];
    const out: Item[] = [];
    // skill deep-links first (commands). Each boots Claude Code via `cc "/skill"`
    // — `fafo` is intentionally absent (designed, not built).
    const skills = ["orient", "strategize", "align", "execute", "close", "glossary", "dynamic-viz"];
    for (const s of skills)
      out.push({ id: `cmd_${s}`, label: `/${s}`, sub: "boot Claude Code + run skill", kind: "skill", tone: "amber", action: () => launchSkill(s) });
    for (const g of snap.goals)
      out.push({ id: g.id, label: g.title, sub: g.id, kind: g.node_kind, tone: NODE_KIND[g.node_kind]?.tone ?? "muted", to: "/tree" });
    for (const r of snap.runs)
      out.push({ id: r.id, label: r.id, sub: r.question || r.goal_title || "", kind: "run", tone: RUN_STATUS[r.status]?.tone ?? "muted", to: `/runs/${r.id}` });
    for (const c of snap.claims)
      out.push({ id: c.id, label: (c.statement ?? c.id).slice(0, 80), sub: c.id, kind: "claim", tone: CLAIM_STATUS[c.status]?.tone ?? "muted", to: "/claims" });
    for (const p of snap.pipelines)
      out.push({ id: p.id, label: p.title, sub: p.id, kind: "pipeline", tone: "teal", to: "/pipelines" });
    for (const t of snap.glossary)
      out.push({ id: t.id, label: t.term, sub: "glossary · " + t.definition.slice(0, 60), kind: "term", tone: "blue", to: "/glossary" });
    return out;
  }, [snap, launchSkill]);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return items.slice(0, 40);
    const scored = items
      .map((it) => {
        const hay = (it.label + " " + it.sub + " " + it.kind).toLowerCase();
        const idx = hay.indexOf(needle);
        return idx === -1 ? null : { it, score: idx };
      })
      .filter(Boolean) as { it: Item; score: number }[];
    scored.sort((a, b) => a.score - b.score);
    return scored.slice(0, 40).map((s) => s.it);
  }, [items, q]);

  useEffect(() => setSel(0), [q]);

  if (!open) return null;

  const run = (it?: Item) => {
    if (!it) return;
    onClose();
    if (it.action) it.action();
    else if (it.to) nav(it.to);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 pt-[12vh] backdrop-blur-sm" onClick={onClose}>
      <div
        className="card w-[640px] max-w-[92vw] overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-[var(--color-border)] px-4 py-3">
          <IconSearch className="text-[var(--color-faint)]" />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, filtered.length - 1)); }
              else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
              else if (e.key === "Enter") { e.preventDefault(); run(filtered[sel]); }
              else if (e.key === "Escape") onClose();
            }}
            placeholder="Search goals, runs, claims, terms — or type a /skill…"
            className="flex-1 bg-transparent text-[14px] text-[var(--color-fg)] outline-none placeholder:text-[var(--color-faint)]"
          />
          <span className="kbd">esc</span>
        </div>
        <div className="max-h-[52vh] overflow-y-auto py-1">
          {filtered.length === 0 && <div className="px-4 py-6 text-center text-[13px] text-[var(--color-faint)]">No matches.</div>}
          {filtered.map((it, i) => (
            <button
              key={it.id}
              onMouseEnter={() => setSel(i)}
              onClick={() => run(it)}
              className={`flex w-full items-center gap-3 px-4 py-2 text-left ${i === sel ? "bg-[var(--color-elev)]" : ""}`}
            >
              <span className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase" style={{ backgroundColor: TONE_HEX[it.tone] + "22", color: TONE_HEX[it.tone] }}>
                {it.kind}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13px] text-[var(--color-fg-dim)]">{it.label}</span>
                <span className="block truncate font-mono text-[11px] text-[var(--color-faint)]">{it.sub}</span>
              </span>
              {i === sel && <IconArrowRight width={15} height={15} className="text-[var(--color-faint)]" />}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
