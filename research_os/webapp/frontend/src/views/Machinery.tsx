import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import type { Machinery as MachineryMap, MachineryNode, MachineryStage } from "../lib/types";
import { Card, Pill, Dot, SectionTitle, Spinner } from "../ui/kit";
import { TONE_HEX, toneBg, fmtDate } from "../lib/ui";
import { IconMachinery, IconRefresh, IconWarn, IconArrowRight, IconSubstrate } from "../ui/icons";

// Machinery — the "visualised pseudocode" of the inversion machinery. The data-flow
// stages of the pipeline, each holding the load-bearing functions that implement it.
// AUTO-DERIVE-THEN-CURATE: the STRUCTURE (signature, line, call edges) is AST-derived
// by render/derive_machinery.py; the MEANING (stage, role, substrate tag) is curated
// in render/machinery_overlay.json; `drift` is the computed cross-check. This is the
// code analogue of the frontier ranking: a derived render artifact the backend only
// reads, regenerated from a terminal — never the canonical store.

const DERIVE_CMD = "python research_os/render/derive_machinery.py\n";

export function Machinery() {
  const { snap, loading } = useStore();
  const { openTerminal } = useTerminals();
  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;

  const m = snap.machinery;
  const reDerive = () => openTerminal(DERIVE_CMD, "derive-machinery");

  if (!m) {
    return (
      <div className="space-y-6">
        <Explainer m={null} onDerive={reDerive} />
        <Card className="flex flex-col items-center gap-3 py-12 text-center">
          <IconMachinery width={26} height={26} className="text-[var(--color-faint)]" />
          <div className="text-[13px] text-[var(--color-muted)]">No machinery map derived yet.</div>
          <div className="max-w-md text-[12px] text-[var(--color-faint)]">
            Author <span className="font-mono">render/machinery_overlay.json</span>, then run the derive to
            ground it in the code and produce <span className="font-mono">render/machinery_map.json</span>.
          </div>
          <button
            onClick={reDerive}
            className="mt-1 inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-[12px]"
            style={toneBg("blue")}
          >
            <IconRefresh width={13} height={13} /> Run derive_machinery.py
          </button>
        </Card>
      </div>
    );
  }

  // group nodes by stage in overlay (data-flow) order; unstaged go last
  const byStage = new Map<string, MachineryNode[]>();
  for (const n of m.nodes) {
    const k = n.stage || "_unstaged";
    (byStage.get(k) ?? byStage.set(k, []).get(k)!).push(n);
  }
  const orderedStages: (MachineryStage & { _unstaged?: boolean })[] = [...m.stages];
  const stageKeys = new Set(orderedStages.map((s) => s.key));
  const extra = [...byStage.keys()].filter((k) => !stageKeys.has(k));
  for (const k of extra) orderedStages.push({ key: k, title: k === "_unstaged" ? "unstaged" : k, _unstaged: true });

  const driftCount = m.drift.missing.length + m.drift.uncovered.length;

  return (
    <div className="space-y-6">
      <Explainer m={m} onDerive={reDerive} />

      {/* macro stage-flow ribbon */}
      <Card className="overflow-x-auto p-4">
        <div className="flex min-w-max items-stretch gap-2">
          {m.stages.map((s, i) => {
            const n = (byStage.get(s.key) ?? []).length;
            return (
              <div key={s.key} className="flex items-stretch gap-2">
                <a
                  href={`#stage-${s.key}`}
                  className="flex min-w-[128px] max-w-[180px] flex-col rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-2)] px-3 py-2 transition-colors hover:border-[var(--color-border-2)]"
                >
                  <span className="text-[12px] font-medium text-[var(--color-fg)]">{s.title}</span>
                  <span className="mt-0.5 font-mono text-[10px] text-[var(--color-faint)]">{s.key}</span>
                  <span className="mt-1 text-[10px] text-[var(--color-muted)]">{n} fn{n === 1 ? "" : "s"}</span>
                </a>
                {i < m.stages.length - 1 && (
                  <div className="flex items-center text-[var(--color-faint)]">
                    <IconArrowRight width={15} height={15} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>

      {/* drift banner */}
      {driftCount > 0 && <DriftPanel m={m} />}

      {/* per-stage function cards */}
      {orderedStages.map((s) => {
        const nodes = byStage.get(s.key) ?? [];
        if (nodes.length === 0) return null;
        return (
          <div key={s.key} id={`stage-${s.key}`} className="scroll-mt-4">
            <SectionTitle
              icon={<IconMachinery width={14} height={14} className="text-[var(--color-blue)]" />}
              count={nodes.length}
            >
              {s.title}
            </SectionTitle>
            {(s.purpose || s.consumes || s.produces) && (
              <p className="-mt-1.5 mb-3 max-w-3xl text-[12px] leading-relaxed text-[var(--color-muted)]">
                {s.purpose}
                {(s.consumes || s.produces) && (
                  <span className="ml-1 font-mono text-[11px] text-[var(--color-faint)]">
                    {" "}· {s.consumes} <IconArrowRight width={10} height={10} className="inline" /> {s.produces}
                  </span>
                )}
              </p>
            )}
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              {nodes.map((n) => <NodeCard key={`${n.file}:${n.name}`} n={n} />)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Explainer({ m, onDerive }: { m: MachineryMap | null; onDerive: () => void }) {
  const driftCount = m ? m.drift.missing.length + m.drift.uncovered.length : 0;
  return (
    <Card className="flex items-start gap-3 p-5">
      <IconMachinery width={20} height={20} className="mt-0.5 shrink-0 text-[var(--color-blue)]" />
      <div className="min-w-0 flex-1">
        <h1 className="text-[15px] font-semibold text-[var(--color-fg)]">Machinery · visualised pseudocode</h1>
        <p className="mt-1 max-w-3xl text-[12px] leading-relaxed text-[var(--color-muted)]">
          The inversion machinery, stage by stage. The <span className="text-[var(--color-fg-dim)]">structure</span>
          {" "}(signature, line, call edges) is AST-derived from the real code; the <span className="text-[var(--color-fg-dim)]">meaning</span>
          {" "}(stage, role, substrate tag) is hand-curated; <span className="text-[var(--color-amber)]">drift</span> is the
          computed cross-check between the two. A derived artifact — re-run the derive after the code moves.
        </p>
        {m && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-[var(--color-faint)]">
            <span className="font-mono">{m.nodes.length} fns · {m.edges.length} edges · {m.stages.length} stages</span>
            {m.generated_at && <span>· derived {fmtDate(m.generated_at)}</span>}
            {m.generated_for_code && <span className="font-mono">· code {m.generated_for_code}</span>}
          </div>
        )}
      </div>
      <div className="flex shrink-0 flex-col items-end gap-2">
        {m && (m.stale ? (
          <Pill tone="amber" className="whitespace-nowrap"><IconWarn width={12} height={12} /> stale · code moved</Pill>
        ) : driftCount > 0 ? (
          <Pill tone="amber" className="whitespace-nowrap">{driftCount} drift</Pill>
        ) : (
          <Pill tone="green" className="whitespace-nowrap">in sync with code</Pill>
        ))}
        <button
          onClick={onDerive}
          className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-border)] px-2.5 py-1.5 text-[12px] text-[var(--color-muted)] transition-colors hover:text-[var(--color-fg)]"
          title="Run derive_machinery.py in a terminal"
        >
          <IconRefresh width={13} height={13} /> re-derive
        </button>
      </div>
    </Card>
  );
}

function NodeCard({ n }: { n: MachineryNode }) {
  const sub = n.substrate_component;
  return (
    <Card
      className="p-4"
      style={sub ? { borderColor: TONE_HEX.pink + "55", backgroundColor: TONE_HEX.pink + "08" } : undefined}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[13px] font-medium text-[var(--color-fg)]">{n.name}</span>
            {sub && (
              <Link to="/substrate" title="substrate component">
                <Pill tone="pink" className="font-mono"><IconSubstrate width={11} height={11} /> {sub}</Pill>
              </Link>
            )}
            {!n.exists && (
              <Pill tone="red"><IconWarn width={11} height={11} /> missing</Pill>
            )}
          </div>
          {n.signature && (
            <div className="mt-1 font-mono text-[11px] text-[var(--color-fg-dim)]">{n.signature}</div>
          )}
        </div>
        <span className="shrink-0 font-mono text-[10px] text-[var(--color-faint)]">
          {n.file.split("/").pop()}{n.line ? `:${n.line}` : ""}
        </span>
      </div>

      {n.role && <p className="mt-2 text-[12px] leading-relaxed text-[var(--color-muted)]">{n.role}</p>}
      {n.doc && n.doc !== n.role && (
        <p className="mt-1 text-[11px] italic leading-relaxed text-[var(--color-faint)]">{n.doc}</p>
      )}

      {n.calls && n.calls.length > 0 && (
        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">calls</span>
          {n.calls.map((c) => (
            <span key={c} className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-fg-dim)]">
              <IconArrowRight width={9} height={9} className="text-[var(--color-faint)]" />{c}
            </span>
          ))}
        </div>
      )}
    </Card>
  );
}

function DriftPanel({ m }: { m: MachineryMap }) {
  const { missing, uncovered } = m.drift;
  return (
    <div
      className="rounded-lg border-l-2 px-4 py-3.5"
      style={{ borderLeftColor: TONE_HEX.amber, backgroundColor: TONE_HEX.amber + "10" }}
    >
      <div className="mb-2 flex items-center gap-2.5">
        <IconWarn width={16} height={16} className="text-[var(--color-amber)]" />
        <span className="text-[13px] font-semibold text-[var(--color-fg)]">Drift</span>
        <span className="text-[12px] text-[var(--color-muted)]">overlay vs code is out of sync</span>
      </div>
      {missing.length > 0 && (
        <div className="mb-2">
          <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[var(--color-red)]">
            <Dot tone="red" size={7} /> missing — curated, but gone from the code ({missing.length})
          </div>
          <div className="flex flex-wrap gap-1.5">
            {missing.map((x) => (
              <span key={`${x.file}:${x.name}`} className="rounded-md border px-2 py-1 font-mono text-[11px]" style={toneBg("red")}>
                {x.name} <span className="text-[var(--color-faint)]">· {x.file.split("/").pop()}</span>
              </span>
            ))}
          </div>
        </div>
      )}
      {uncovered.length > 0 && (
        <div>
          <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
            <Dot tone="amber" size={7} /> uncovered — in scope, not yet on the map ({uncovered.length})
          </div>
          <div className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto pr-1">
            {uncovered.map((x) => (
              <span key={`${x.file}:${x.name}`} className="rounded-md border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-2 py-1 font-mono text-[11px] text-[var(--color-fg-dim)]">
                {x.name}<span className="text-[var(--color-faint)]">:{x.line}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
