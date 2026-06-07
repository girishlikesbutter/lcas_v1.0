import { useState } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import type { Substrate as SubstrateComponent, SubstrateVersion, SubstrateFunction, ToolRun, ToolDrift } from "../lib/types";
import { Card, Pill, Dot, SectionTitle, Spinner } from "../ui/kit";
import { shortId, runShort, fmtDate, TONE_HEX, toneBg, type Tone } from "../lib/ui";
import { IconSubstrate, IconBlast, IconWarn, IconRuns, IconClaims, IconArrowRight, IconChevron, IconMachinery } from "../ui/icons";

// ADR-0007 run-button: drop the canonical run_tool command into a fresh PTY. Compute
// happens in the terminal (visible, no daemon); run_tool drift-checks, invokes the
// bound entry_point, and writes the canonical tool_run — the file-watch → SSE loop
// then refreshes this view. The dashboard never writes the store (Q1 boundary intact).
function runToolCommand(id: string): string {
  return `python research_os/loop/run_tool.py ${id} --source webapp\n`;
}

const DRIFT_TONE: Record<ToolDrift["verdict"], Tone> = {
  clean: "green", missing: "red", hash_moved: "amber", no_entry_point: "faint",
};
const STATUS_TONE: Record<ToolRun["status"], Tone> = {
  ok: "green", error: "red", drift_refused: "amber",
};

// Substrate — the version registry + AUTOMATIC BLAST RADIUS. The trust-machine
// centrepiece: "the thing that cost s001–s066 becomes one line." Each component
// carries a version history; the indexer stamps every run/claim with the substrate
// version it ran against, so resting on a superseded version is computed, not
// remembered. The amber panels below are that payoff made visible.

const MAX_BLAST_RUNS = 20;

export function Substrate() {
  const { snap, loading } = useStore();
  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;

  const components = snap.substrate;
  const totalBlastRuns = components.reduce((a, c) => a + c.blast_runs.length, 0);
  const totalBlastClaims = components.reduce((a, c) => a + c.blast_claims.length, 0);
  const hot = totalBlastRuns + totalBlastClaims > 0;

  return (
    <div className="space-y-6">
      {/* page explainer */}
      <Card className="flex items-start gap-3 p-5">
        <IconSubstrate width={20} height={20} className="mt-0.5 shrink-0 text-[var(--color-pink)]" />
        <div className="min-w-0 flex-1">
          <h1 className="text-[15px] font-semibold text-[var(--color-fg)]">Substrate registry · blast radius</h1>
          <p className="mt-1 text-[12px] leading-relaxed text-[var(--color-muted)]">
            Every shared component is versioned. Each run and claim is stamped with the substrate version it
            rested on, so a <span className="text-[var(--color-amber)]">bug-fix bump</span> automatically lights up
            every downstream result still resting on the old version — the <span className="text-[var(--color-fg-dim)]">blast radius</span>.
            What once cost an s001–s066 manual audit is now one computed line.
          </p>
        </div>
        <div className="hidden shrink-0 items-center gap-2 sm:flex">
          {hot ? (
            <Pill tone="amber" className="whitespace-nowrap">
              <IconBlast width={12} height={12} />
              {totalBlastRuns} runs · {totalBlastClaims} claims exposed
            </Pill>
          ) : (
            <Pill tone="green" className="whitespace-nowrap">all dependents on head</Pill>
          )}
        </div>
      </Card>

      {/* component cards */}
      <SectionTitle icon={<IconSubstrate width={15} height={15} className="text-[var(--color-pink)]" />} count={components.length}>
        Components
      </SectionTitle>

      {components.length === 0 ? (
        <Card className="py-10 text-center text-[13px] text-[var(--color-faint)]">No substrate components registered.</Card>
      ) : (
        <div className="space-y-5">
          {components.map((c) => <ComponentCard key={c.id} c={c} />)}
        </div>
      )}
    </div>
  );
}

function ComponentCard({ c }: { c: SubstrateComponent }) {
  const hasBlast = c.blast_runs.length > 0 || c.blast_claims.length > 0;
  // newest-first for display
  const history = [...(c.versions ?? [])].reverse();

  return (
    <Card className="overflow-hidden p-0">
      {/* header */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[var(--color-border)] px-5 py-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[15px] font-semibold text-[var(--color-fg)]">{c.name}</span>
            <span className="font-mono text-[11px] text-[var(--color-faint)]">{shortId(c.id)}</span>
            <Pill tone="green" className="font-mono">@{c.current_version}</Pill>
          </div>
          <div className="mt-1 font-mono text-[11px] text-[var(--color-faint)]">{c.path}</div>
          {c.interface && (
            <p className="mt-2 max-w-3xl text-[12px] leading-relaxed text-[var(--color-muted)]">{c.interface}</p>
          )}
        </div>
        <div className="shrink-0 text-right">
          {hasBlast ? (
            <Pill tone="amber" className="whitespace-nowrap">
              <IconBlast width={12} height={12} />
              {c.blast_runs.length} · {c.blast_claims.length} exposed
            </Pill>
          ) : (
            <Pill tone="green" className="whitespace-nowrap">on head</Pill>
          )}
        </div>
      </div>

      {/* ADR-0007 run-button bar: drift status + entry_point binding + Run */}
      <RunBar c={c} />

      {/* functions — the comprehension granularity beneath the single versioned seam */}
      {c.functions && c.functions.length > 0 && (
        <FunctionsSection fns={c.functions} />
      )}

      {/* body: timeline (left) + blast radius (right) */}
      <div className="grid grid-cols-1 gap-0 lg:grid-cols-5">
        {/* VERSION HISTORY — vertical timeline, newest at top */}
        <div className="border-b border-[var(--color-border)] px-5 py-4 lg:col-span-2 lg:border-b-0 lg:border-r">
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-faint)]">
            version history
          </div>
          <div className="space-y-0">
            {history.map((v, i) => (
              <VersionRow
                key={v.version}
                v={v}
                isCurrent={v.version === c.current_version}
                isLast={i === history.length - 1}
              />
            ))}
          </div>
        </div>

        {/* BLAST RADIUS — the payoff */}
        <div className="px-5 py-4 lg:col-span-3">
          <BlastPanel c={c} />
        </div>
      </div>

      {/* bench runs — the run-button's canonical tool_run records for this Tool */}
      {(c.bench_run_count ?? 0) > 0 && <BenchRuns c={c} />}
    </Card>
  );
}

// The run-button bar: shows whether the Tool's code binding is in sync (drift-check)
// and, when clean, a Run button that fires run_tool in a terminal. The mechanical
// "we can SEE the binding is stale and refuse to run it" Girish asked for.
function RunBar({ c }: { c: SubstrateComponent }) {
  const { openTerminal } = useTerminals();
  const drift = c.drift;
  const hasEntry = !!c.entry_point;
  const tone: Tone = drift ? DRIFT_TONE[drift.verdict] : "faint";
  const runnable = !!drift?.runnable;

  const params = c.default_params
    ? Object.entries(c.default_params).filter(([k]) => !k.startsWith("_"))
    : [];

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-[var(--color-border)] bg-[var(--color-bg-2)]/40 px-5 py-3">
      {/* drift verdict */}
      {drift ? (
        <Pill tone={tone} className="whitespace-nowrap font-mono">
          {drift.verdict === "clean" ? "● in sync" : `▲ ${drift.verdict}`}
        </Pill>
      ) : (
        <Pill tone="faint">no binding</Pill>
      )}

      {/* entry_point */}
      {hasEntry ? (
        <span className="min-w-0 truncate font-mono text-[11px] text-[var(--color-fg-dim)]" title={c.entry_point!}>
          {c.entry_point}
        </span>
      ) : (
        <span className="font-mono text-[11px] text-[var(--color-faint)]">no entry_point — descriptive only</span>
      )}

      {/* default params preview */}
      {params.length > 0 && (
        <span className="hidden truncate font-mono text-[10.5px] text-[var(--color-faint)] md:inline">
          {params.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join("  ")}
        </span>
      )}

      <div className="ml-auto flex items-center gap-2">
        {(c.bench_run_count ?? 0) > 0 && (
          <span className="font-mono text-[10.5px] text-[var(--color-faint)]">{c.bench_run_count} run{c.bench_run_count === 1 ? "" : "s"}</span>
        )}
        <button
          disabled={!runnable}
          onClick={() => openTerminal(runToolCommand(c.id), `run ${shortId(c.id)}`)}
          title={
            runnable
              ? `Run ${c.id} in a terminal (run_tool.py)`
              : drift?.reason || "not runnable — no entry_point binding"
          }
          className="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40"
          style={
            runnable
              ? { borderColor: TONE_HEX.green, color: TONE_HEX.green, backgroundColor: TONE_HEX.green + "12" }
              : { borderColor: "var(--color-border)", color: "var(--color-faint)" }
          }
        >
          ▶ Run
        </button>
      </div>
    </div>
  );
}

function BenchRuns({ c }: { c: SubstrateComponent }) {
  const [open, setOpen] = useState(false);
  const runs = c.bench_runs ?? [];
  return (
    <div className="border-t border-[var(--color-border)] bg-[var(--color-bg-2)]/20 px-5 py-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-faint)] transition-colors hover:text-[var(--color-fg-dim)]"
      >
        <IconChevron width={12} height={12} style={{ transform: open ? "rotate(90deg)" : "none", transition: "transform .15s" }} />
        <IconRuns width={12} height={12} /> bench runs
        <span className="rounded bg-[var(--color-elev)] px-1.5 py-0.5 text-[var(--color-muted)]">{c.bench_run_count}</span>
      </button>
      {open && (
        <div className="mt-2.5 space-y-1.5">
          {runs.map((t) => <BenchRunRow key={t.id} t={t} />)}
        </div>
      )}
    </div>
  );
}

function BenchRunRow({ t }: { t: ToolRun }) {
  const tone = STATUS_TONE[t.status];
  const m = t.metrics || {};
  const metricKeys = Object.keys(m).filter((k) => k !== "wall_s").slice(0, 4);
  return (
    <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 border-l-2 pl-3 text-[11px]" style={{ borderLeftColor: TONE_HEX[tone] }}>
      <Pill tone={tone} className="font-mono">{t.status}</Pill>
      <span className="font-mono text-[10.5px] text-[var(--color-fg-dim)]">{t.id.replace(/^tr_/, "")}</span>
      {t.wall_s != null && <span className="font-mono text-[10px] text-[var(--color-faint)]">{t.wall_s}s</span>}
      {!t.oracle_clean && <Pill tone="amber">oracle</Pill>}
      {metricKeys.length > 0 && (
        <span className="truncate font-mono text-[10px] text-[var(--color-muted)]">
          {metricKeys.map((k) => `${k}=${JSON.stringify(m[k])}`).join("  ")}
        </span>
      )}
      {t.reason && <span className="truncate text-[10.5px] text-[var(--color-faint)] italic">{t.reason}</span>}
      <span className="ml-auto text-[10px] text-[var(--color-faint)]">{fmtDate(t.executed_at || t.created_at)}</span>
    </div>
  );
}

function FunctionsSection({ fns }: { fns: SubstrateFunction[] }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border-b border-[var(--color-border)] bg-[var(--color-bg-2)]/30 px-5 py-3">
      <div className="flex items-center justify-between">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-faint)] transition-colors hover:text-[var(--color-fg-dim)]"
        >
          <IconChevron
            width={12} height={12}
            style={{ transform: open ? "rotate(90deg)" : "none", transition: "transform .15s" }}
          />
          functions
          <span className="rounded bg-[var(--color-elev)] px-1.5 py-0.5 text-[var(--color-muted)]">{fns.length}</span>
        </button>
        <Link
          to="/machinery"
          className="flex items-center gap-1 text-[10px] text-[var(--color-faint)] transition-colors hover:text-[var(--color-blue)]"
          title="See where these sit in the machinery map"
        >
          <IconMachinery width={11} height={11} /> machinery map <IconArrowRight width={10} height={10} />
        </Link>
      </div>
      {open && (
        <div className="mt-2.5 space-y-2">
          {fns.map((f) => (
            <div key={f.name} className="flex flex-col gap-0.5 border-l-2 border-[var(--color-border-2)] pl-3">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className="font-mono text-[12px] font-medium text-[var(--color-fg)]">{f.name}</span>
                {f.line != null && <span className="font-mono text-[10px] text-[var(--color-faint)]">:{f.line}</span>}
              </div>
              {f.signature && (
                <div className="font-mono text-[10.5px] leading-snug text-[var(--color-fg-dim)]">{f.signature}</div>
              )}
              <p className="text-[11.5px] leading-relaxed text-[var(--color-muted)]">{f.role}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function VersionRow({ v, isCurrent, isLast }: { v: SubstrateVersion; isCurrent: boolean; isLast: boolean }) {
  const bugFix = !!v.is_bug_fix;
  const tone: Tone = bugFix ? "red" : isCurrent ? "green" : "faint";

  return (
    <div className="relative flex gap-3 pb-4">
      {/* rail */}
      <div className="relative flex flex-col items-center">
        <Dot tone={tone} size={9} />
        {!isLast && <span className="mt-1 w-px flex-1 bg-[var(--color-border)]" />}
      </div>
      {/* content */}
      <div
        className={`min-w-0 flex-1 rounded-lg px-3 py-2 ${
          isCurrent ? "border border-[var(--color-border-2)] bg-[var(--color-elev)]/50" : ""
        }`}
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[12px] font-medium" style={{ color: TONE_HEX[tone] }}>
            {v.version}
          </span>
          {isCurrent && <Pill tone="green">head</Pill>}
          {bugFix && (
            <Pill tone="red">
              <IconWarn width={11} height={11} />
              bug-fix
            </Pill>
          )}
          <span className="ml-auto text-[10px] text-[var(--color-faint)]">{fmtDate(v.changed_on)}</span>
        </div>
        <p className="mt-1.5 text-[11.5px] leading-relaxed text-[var(--color-muted)]">{v.change_reason}</p>
        {v.commit && (
          <div className="mt-1.5 font-mono text-[10px] text-[var(--color-faint)]">commit {v.commit}</div>
        )}
      </div>
    </div>
  );
}

function BlastPanel({ c }: { c: SubstrateComponent }) {
  const [expanded, setExpanded] = useState(false);
  const runs = c.blast_runs;
  const claims = c.blast_claims;
  const hasBlast = runs.length > 0 || claims.length > 0;

  if (!hasBlast) {
    return (
      <div
        className="flex h-full min-h-[80px] items-center gap-2.5 rounded-lg border px-4 py-3"
        style={toneBg("green")}
      >
        <Dot tone="green" size={8} />
        <span className="text-[12px] text-[var(--color-green)]">
          No blast radius — every dependent run and claim rests on <span className="font-mono">@{c.current_version}</span>.
        </span>
      </div>
    );
  }

  const shownRuns = expanded ? runs : runs.slice(0, MAX_BLAST_RUNS);
  const moreRuns = runs.length - shownRuns.length;

  return (
    <div
      className="rounded-lg border-l-2 px-4 py-3.5"
      style={{ borderLeftColor: TONE_HEX.amber, backgroundColor: TONE_HEX.amber + "10" }}
    >
      {/* headline */}
      <div className="mb-3 flex items-center gap-2.5">
        <IconBlast width={17} height={17} className="text-[var(--color-amber)]" />
        <span className="text-[13px] font-semibold text-[var(--color-fg)]">
          {runs.length} run{runs.length === 1 ? "" : "s"} · {claims.length} claim{claims.length === 1 ? "" : "s"}
        </span>
        <span className="text-[12px] text-[var(--color-muted)]">rest on a superseded version</span>
      </div>

      {/* exposed claims */}
      {claims.length > 0 && (
        <div className="mb-3">
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
            <IconClaims width={12} height={12} className="text-[var(--color-amber)]" />
            exposed claims
          </div>
          <div className="flex flex-wrap gap-1.5">
            {claims.map((id) => (
              <Link
                key={id}
                to="/claims"
                className="inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[11px] transition-colors hover:brightness-125"
                style={toneBg("amber")}
              >
                {shortId(id)}
                <IconArrowRight width={11} height={11} />
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* exposed runs */}
      {runs.length > 0 && (
        <div>
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
            <IconRuns width={12} height={12} className="text-[var(--color-amber)]" />
            exposed runs
          </div>
          <div className="flex max-h-44 flex-wrap gap-1.5 overflow-y-auto pr-1">
            {shownRuns.map((id) => (
              <Link
                key={id}
                to={`/runs/${id}`}
                className="rounded-md border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-2 py-1 font-mono text-[11px] text-[var(--color-fg-dim)] transition-colors hover:border-[var(--color-amber)] hover:text-[var(--color-amber)]"
              >
                {runShort(id)}
              </Link>
            ))}
            {moreRuns > 0 && (
              <button
                onClick={() => setExpanded(true)}
                className="rounded-md border border-dashed border-[var(--color-border-2)] px-2 py-1 text-[11px] text-[var(--color-muted)] transition-colors hover:text-[var(--color-fg)]"
              >
                +{moreRuns} more
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
