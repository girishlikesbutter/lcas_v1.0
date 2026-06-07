import { useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import type { Pipeline, PipelineRun, PipelineStep, ArtifactInstance } from "../lib/types";
import { Card, Pill, Dot, SectionTitle, Spinner, Button, Empty, MetricChip } from "../ui/kit";
import {
  GOAL_STATE, shortId, runShort, fmtDate, fmtWall, TONE_HEX, type Tone,
} from "../lib/ui";
import {
  IconPipelines, IconBolt, IconRuns, IconSubstrate, IconTree, IconArrowRight,
  IconMaterials, IconChevron, IconWarn,
} from "../ui/icons";

// ADR-0005 "how we're trying" — pipelines are METHOD nodes, distinct from goals.
// A goal is a question we want answered; a pipeline is a reusable way of trying.
// A method that serves >1 goal is a reuse hub (teal accent). Open/blocked first.
//
// One component serves both "/pipelines" (the list) and "/pipelines/:id" (the run
// detail — the static DAG + the pipeline_run feed + the materials each step shelved),
// mirroring Runs.tsx (ADR-0007 §5.3 + Slice 3).

const OPEN_STATES = new Set(["open", "blocked", "revivable"]);

function rank(p: Pipeline): number {
  // open/blocked/revivable first, then closed/superseded
  if (p.state === "open") return 0;
  if (p.state === "blocked") return 1;
  if (p.state === "revivable") return 2;
  if (p.state === "closed") return 3;
  return 4; // superseded
}

export function Pipelines() {
  const { id } = useParams<{ id: string }>();
  const { snap, loading } = useStore();
  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;
  if (id) return <PipelineDetail id={id} />;
  return <PipelineList />;
}

function PipelineList() {
  const { snap } = useStore();

  const pipelines = [...snap!.pipelines].sort((a, b) => {
    const dr = rank(a) - rank(b);
    if (dr !== 0) return dr;
    return (b.tested_by ?? 0) - (a.tested_by ?? 0);
  });

  const live = pipelines.filter((p) => OPEN_STATES.has(p.state)).length;
  const hubs = pipelines.filter((p) => (p.serves?.length ?? 0) > 1).length;

  return (
    <div className="space-y-6">
      {/* header explainer */}
      <Card className="p-5">
        <SectionTitle
          icon={<IconPipelines width={16} height={16} className="text-[var(--color-teal)]" />}
          count={pipelines.length}
        >
          Pipelines · how we're trying
        </SectionTitle>
        <p className="max-w-3xl text-[13px] leading-relaxed text-[var(--color-muted)]">
          Pipelines are <span className="text-[var(--color-fg-dim)]">method</span> nodes (ADR-0005), distinct from goals:
          a goal is a question we want answered; a pipeline is a reusable way of <span className="text-[var(--color-fg-dim)]">trying</span>.
          Each carries a one-line wager (its hypothesis), the goals it <span className="text-[var(--color-fg-dim)]">serves</span>, the
          substrate it <span className="text-[var(--color-fg-dim)]">composes</span>, and the runs that <span className="text-[var(--color-fg-dim)]">tested</span> it.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px]">
          <span className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-1 text-[var(--color-muted)]">
            <Dot tone="green" size={7} /> {live} active
          </span>
          <span
            className="inline-flex items-center gap-1.5 rounded-md border px-2 py-1"
            style={{ backgroundColor: TONE_HEX.teal + "16", color: TONE_HEX.teal, borderColor: TONE_HEX.teal + "55" }}
          >
            <IconLinkSpark /> {hubs} reuse hub{hubs === 1 ? "" : "s"}
          </span>
          <span className="text-[var(--color-faint)]">a method that serves &gt;1 goal is a reuse hub</span>
        </div>
      </Card>

      {pipelines.length === 0 ? (
        <Empty>No pipelines registered yet.</Empty>
      ) : (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          {pipelines.map((p) => (
            <PipelineCard key={p.id} p={p} all={pipelines} />
          ))}
        </div>
      )}
    </div>
  );
}

function PipelineCard({ p, all }: { p: Pipeline; all: Pipeline[] }) {
  const { launchSkill } = useTerminals();
  const st = GOAL_STATE[p.state] ?? { tone: "muted" as Tone, label: p.state, glyph: "·" };
  const tone = st.tone as Tone;
  const serves = p.serves ?? [];
  const servesTitles = p.serves_titles ?? {};
  const isHub = serves.length > 1;
  const isActive = OPEN_STATES.has(p.state);
  const dim = p.state === "superseded";

  const titleOf = (pid: string): string => all.find((x) => x.id === pid)?.title ?? shortId(pid);

  return (
    <Card
      className={`flex flex-col p-5 ${isHub ? "border-l-2" : ""} ${dim ? "opacity-75" : ""}`}
      style={isHub ? { borderLeftColor: TONE_HEX.teal } : undefined}
    >
      {/* header: title + state pill */}
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            {isHub && <IconLinkSpark />}
            <span className="font-mono text-[10px] text-[var(--color-faint)]">{shortId(p.id)}</span>
            {isHub && (
              <span
                className="rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider"
                style={{ backgroundColor: TONE_HEX.teal + "1e", color: TONE_HEX.teal }}
              >
                reuse hub
              </span>
            )}
          </div>
          <Link to={`/pipelines/${p.id}`} className="group/title">
            <h3 className="mt-1 inline text-[15px] font-semibold leading-snug text-[var(--color-fg)] transition-colors group-hover/title:text-[var(--color-blue)]">
              {p.title}
            </h3>
          </Link>
        </div>
        <Pill tone={tone}>
          {st.glyph} {p.state}
        </Pill>
      </div>

      {/* hypothesis — the one-line wager */}
      {p.hypothesis && (
        <p className="mb-4 text-[12px] italic leading-relaxed text-[var(--color-muted)]">
          &ldquo;{p.hypothesis}&rdquo;
        </p>
      )}

      {/* serves — goal-title chips -> /tree */}
      {serves.length > 0 && (
        <Row icon={<IconTree width={13} height={13} className="text-[var(--color-violet)]" />} label="serves">
          {serves.map((g) => (
            <Link key={g} to="/tree">
              <Chip tone="violet">{servesTitles[g] ?? shortId(g)}</Chip>
            </Link>
          ))}
        </Row>
      )}

      {/* composes — substrate component chips -> /substrate */}
      {p.composes && p.composes.length > 0 && (
        <Row icon={<IconSubstrate width={13} height={13} className="text-[var(--color-pink)]" />} label="composes">
          {p.composes.map((c) => (
            <Link key={c} to="/substrate">
              <Chip tone="pink" mono>{c}</Chip>
            </Link>
          ))}
        </Row>
      )}

      {/* tested by Nx — the run links */}
      <Row icon={<IconRuns width={13} height={13} className="text-[var(--color-blue)]" />} label={`tested by ${p.tested_by}×`}>
        {p.tested_by_runs.length === 0 ? (
          <span className="text-[11px] text-[var(--color-faint)]">untested</span>
        ) : (
          p.tested_by_runs.map((r) => (
            <Link key={r} to={`/runs/${r}`}>
              <Chip tone="blue" mono>{runShort(r)}</Chip>
            </Link>
          ))
        )}
      </Row>

      {/* supersedes / superseded_by */}
      {(p.supersedes || p.superseded_by) && (
        <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px] text-[var(--color-faint)]">
          {p.supersedes && (
            <span className="inline-flex items-center gap-1.5">
              supersedes <span className="font-mono text-[var(--color-muted)]">{titleOf(p.supersedes)}</span>
            </span>
          )}
          {p.superseded_by && (
            <span className="inline-flex items-center gap-1.5">
              <IconArrowRight width={12} height={12} /> superseded by{" "}
              <span className="font-mono text-[var(--color-muted)]">{titleOf(p.superseded_by)}</span>
            </span>
          )}
        </div>
      )}

      <div className="flex-1" />

      {/* last_measured — small bordered box (Overview style) */}
      {p.last_measured?.summary && (
        <div className="mt-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/50 px-3 py-2">
          <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
            last measured
            {p.last_measured.run && (
              <Link to={`/runs/${p.last_measured.run}`} className="font-mono text-[var(--color-blue)] hover:underline">
                {runShort(p.last_measured.run)}
              </Link>
            )}
            <span>{fmtDate(p.last_measured.at)}</span>
          </div>
          <div className="text-[12px] leading-relaxed text-[var(--color-muted)]">{p.last_measured.summary}</div>
        </div>
      )}

      {/* footer: run feed deep-link (left) + execute on active pipelines (right) */}
      <div className="mt-4 flex items-center justify-between gap-3">
        <Link
          to={`/pipelines/${p.id}`}
          className="inline-flex items-center gap-1.5 text-[11px] text-[var(--color-faint)] transition-colors hover:text-[var(--color-blue)]"
        >
          <IconRuns width={12} height={12} />
          {(p.pipeline_run_count ?? 0) > 0
            ? `ran ${p.pipeline_run_count}×`
            : "not yet run"}
          <IconArrowRight width={11} height={11} />
        </Link>
        {isActive && (
          <Button tone="teal" variant="soft" onClick={() => launchSkill("execute")}>
            <IconBolt width={13} height={13} /> execute
          </Button>
        )}
      </div>
    </Card>
  );
}

function Row({ icon, label, children }: { icon: ReactNode; label: string; children: ReactNode }) {
  return (
    <div className="mb-3 flex items-start gap-2">
      <div className="flex w-[88px] shrink-0 items-center gap-1.5 pt-1 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
        {icon}
        <span>{label}</span>
      </div>
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1.5">{children}</div>
    </div>
  );
}

function Chip({ tone, children, mono = false }: { tone: Tone; children: ReactNode; mono?: boolean }) {
  return (
    <span
      className={`inline-flex max-w-full items-center truncate rounded-md border px-2 py-0.5 text-[11px] leading-tight transition-colors hover:brightness-125 ${mono ? "font-mono" : ""}`}
      style={{ backgroundColor: TONE_HEX[tone] + "14", color: TONE_HEX[tone], borderColor: TONE_HEX[tone] + "44" }}
    >
      {children}
    </span>
  );
}

// tiny teal "reuse hub" mark — two linked nodes
function IconLinkSpark() {
  return (
    <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke={TONE_HEX.teal} strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" />
      <path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// DETAIL — /pipelines/:id  (ADR-0007 §5.3 + Slice 3)
// the static DAG + the pipeline_run feed + the materials each step shelved
// ---------------------------------------------------------------------------

// ADR-0007 §5.3 run-button: drop run_pipeline into a PTY. Compute runs in the
// terminal (visible, no daemon); run_pipeline drift-checks each step, executes the
// DAG, shelves the typed outputs, and writes the canonical pipeline_run. The
// file-watch → SSE loop then refreshes this view. The dashboard never writes the
// store (Q1 boundary intact).
const runPipelineCommand = (id: string) => `python research_os/loop/run_pipeline.py ${id} --source webapp\n`;
const dryRunPipelineCommand = (id: string) => `python research_os/loop/run_pipeline.py ${id} --dry-run\n`;

const STEP_STATUS_TONE: Record<string, Tone> = { ok: "green", error: "red" };

function PipelineDetail({ id }: { id: string }) {
  const { snap } = useStore();
  const { launchSkill } = useTerminals();
  const p = snap!.pipelines.find((x) => x.id === id);

  if (!p) {
    return (
      <div className="space-y-6">
        <BackLink />
        <Empty>No pipeline found for <span className="ml-1 font-mono">{id}</span>.</Empty>
      </div>
    );
  }

  // resolve artifact ids → labels for the threaded-materials chips
  const aiById = new Map<string, ArtifactInstance>(snap!.artifact_instances.map((a) => [a.id, a]));
  const st = GOAL_STATE[p.state] ?? { tone: "muted" as Tone, label: p.state, glyph: "·" };
  const tone = st.tone as Tone;
  const serves = p.serves ?? [];
  const servesTitles = p.serves_titles ?? {};
  const isActive = OPEN_STATES.has(p.state);
  const runs = p.pipeline_runs ?? [];

  return (
    <div className="space-y-6">
      <BackLink />

      {/* header */}
      <Card className="p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <IconPipelines width={16} height={16} className="text-[var(--color-teal)]" />
              <span className="font-mono text-[11px] text-[var(--color-faint)]">{shortId(p.id)}</span>
              <Pill tone={tone}>{st.glyph} {p.state}</Pill>
            </div>
            <h1 className="mt-2 text-[18px] font-semibold leading-snug text-[var(--color-fg)]">{p.title}</h1>
            {p.hypothesis && (
              <p className="mt-2 max-w-3xl text-[12.5px] italic leading-relaxed text-[var(--color-muted)]">
                &ldquo;{p.hypothesis}&rdquo;
              </p>
            )}
          </div>
          {isActive && (
            <RunPipelineBar id={p.id} />
          )}
        </div>

        {/* serves + composes */}
        <div className="mt-4 space-y-3">
          {serves.length > 0 && (
            <Row icon={<IconTree width={13} height={13} className="text-[var(--color-violet)]" />} label="serves">
              {serves.map((g) => (
                <Link key={g} to="/tree"><Chip tone="violet">{servesTitles[g] ?? shortId(g)}</Chip></Link>
              ))}
            </Row>
          )}
          {p.composes && p.composes.length > 0 && (
            <Row icon={<IconSubstrate width={13} height={13} className="text-[var(--color-pink)]" />} label="composes">
              {p.composes.map((c) => (
                <Link key={c} to="/substrate"><Chip tone="pink" mono>{c}</Chip></Link>
              ))}
            </Row>
          )}
        </div>
      </Card>

      {/* the static DAG */}
      {p.steps && p.steps.length > 0 && (
        <Card className="p-5">
          <SectionTitle icon={<IconPipelines width={15} height={15} className="text-[var(--color-teal)]" />} count={p.steps.length}>
            DAG · how it composes
          </SectionTitle>
          <div className="space-y-0">
            {p.steps.map((s, i) => (
              <StepDefRow key={s.id} s={s} isLast={i === p.steps!.length - 1} />
            ))}
          </div>
        </Card>
      )}

      {/* the run feed */}
      <Card className="p-5">
        <SectionTitle icon={<IconRuns width={15} height={15} className="text-[var(--color-blue)]" />} count={runs.length}>
          Run feed · pipeline_run records
        </SectionTitle>
        {runs.length === 0 ? (
          <Empty>Never run. Use the run-button above to execute the DAG.</Empty>
        ) : (
          <div className="space-y-2.5">
            {runs.map((r) => <PipelineRunRow key={r.id} r={r} aiById={aiById} />)}
          </div>
        )}
      </Card>

      {/* tested-by (the value-laden runs that cite this method) */}
      {p.tested_by_runs.length > 0 && (
        <Card className="p-5">
          <SectionTitle icon={<IconRuns width={15} height={15} className="text-[var(--color-blue)]" />} count={p.tested_by}>
            Tested by · runs that cite this method
          </SectionTitle>
          <div className="flex flex-wrap gap-1.5">
            {p.tested_by_runs.map((rid) => (
              <Link key={rid} to={`/runs/${rid}`}>
                <Chip tone="blue" mono>{runShort(rid)}</Chip>
              </Link>
            ))}
          </div>
        </Card>
      )}

      {isActive && (
        <div className="flex justify-end">
          <Button tone="teal" variant="soft" onClick={() => launchSkill("execute")}>
            <IconBolt width={13} height={13} /> execute (contracted run)
          </Button>
        </div>
      )}
    </div>
  );
}

function RunPipelineBar({ id }: { id: string }) {
  const { openTerminal } = useTerminals();
  return (
    <div className="flex shrink-0 items-center gap-2">
      <button
        onClick={() => openTerminal(dryRunPipelineCommand(id), `dry-run ${shortId(id)}`)}
        title="Drift-check + plan each step, do not run (run_pipeline --dry-run)"
        className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-border)] px-2.5 py-1 text-[11px] font-medium text-[var(--color-muted)] transition-colors hover:text-[var(--color-fg)] hover:border-[var(--color-border-2)]"
      >
        ◇ Dry-run
      </button>
      <button
        onClick={() => openTerminal(runPipelineCommand(id), `run ${shortId(id)}`)}
        title={`Run the ${id} DAG in a terminal (run_pipeline.py)`}
        className="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors"
        style={{ borderColor: TONE_HEX.green, color: TONE_HEX.green, backgroundColor: TONE_HEX.green + "12" }}
      >
        ▶ Run pipeline
      </button>
    </div>
  );
}

// one row of the static DAG definition — op badge, tool, threaded inputs
function StepDefRow({ s, isLast }: { s: NonNullable<Pipeline["steps"]>[number]; isLast: boolean }) {
  const inputs = Object.entries(s.inputs ?? {});
  return (
    <div className="relative flex gap-3 pb-4">
      {/* rail */}
      <div className="relative flex flex-col items-center pt-1">
        <span
          className="flex h-6 w-6 items-center justify-center rounded-md text-[9px] font-semibold uppercase"
          style={{ backgroundColor: TONE_HEX.teal + "1e", color: TONE_HEX.teal }}
        >
          {s.op === "map" ? "⋮" : "∘"}
        </span>
        {!isLast && <span className="mt-1 w-px flex-1 bg-[var(--color-border)]" />}
      </div>
      {/* content */}
      <div className="min-w-0 flex-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-3 py-2">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="font-mono text-[12px] font-medium text-[var(--color-fg)]">{s.id}</span>
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">{s.op}</span>
          <Link to="/substrate" className="ml-auto">
            <Chip tone="pink" mono>{s.tool}</Chip>
          </Link>
        </div>
        {inputs.length > 0 && (
          <div className="mt-1.5 space-y-1">
            {inputs.map(([k, v]) => (
              <div key={k} className="flex items-baseline gap-2 font-mono text-[10.5px]">
                <span className="text-[var(--color-faint)]">{k}</span>
                <IconArrowRight width={10} height={10} className="shrink-0 text-[var(--color-faint)]" />
                <InputRef value={v} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// render a step input value — a $steps.<id>.<port> thread, a $artifact re-feed, or a literal
function InputRef({ value }: { value: any }) {
  if (value && typeof value === "object" && "$artifact" in value) {
    return (
      <Link to="/materials" className="text-[var(--color-teal)] hover:underline">
        $artifact · {shortId(String(value.$artifact))}
      </Link>
    );
  }
  const s = typeof value === "string" ? value : JSON.stringify(value);
  if (s.startsWith("$steps.")) return <span className="text-[var(--color-blue)]">{s}</span>;
  if (s.startsWith("$artifact")) return <span className="text-[var(--color-teal)]">{s}</span>;
  return <span className="text-[var(--color-muted)]">{s}</span>;
}

function PipelineRunRow({ r, aiById }: { r: PipelineRun; aiById: Map<string, ArtifactInstance> }) {
  const [open, setOpen] = useState(false);
  const tone = STEP_STATUS_TONE[r.status] ?? "muted";
  const droveDrift = r.steps?.some((s) => s.hash_ok === false);
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40">
      {/* summary line */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full flex-wrap items-center gap-x-2.5 gap-y-1 px-3 py-2.5 text-left"
      >
        <IconChevron width={12} height={12} style={{ transform: open ? "rotate(90deg)" : "none", transition: "transform .15s" }} className="text-[var(--color-faint)]" />
        <Pill tone={tone} className="font-mono">{r.status}</Pill>
        <span className="font-mono text-[11px] text-[var(--color-fg-dim)]">{r.id.replace(/^pr_/, "")}</span>
        <span className="font-mono text-[10px] text-[var(--color-faint)]">{r.steps?.length ?? 0} steps</span>
        {r.wall_s != null && <span className="font-mono text-[10px] text-[var(--color-faint)]">{fmtWall(r.wall_s)}</span>}
        {r.source && <span className="rounded bg-[var(--color-elev)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted)]">{r.source}</span>}
        {r.oracle_clean === false && <Pill tone="amber"><IconWarn width={10} height={10} /> oracle</Pill>}
        {droveDrift && <Pill tone="amber">drift</Pill>}
        {r.reason && <span className="truncate text-[10.5px] italic text-[var(--color-faint)]">{r.reason}</span>}
        <span className="ml-auto text-[10px] text-[var(--color-faint)]">{fmtDate(r.executed_at || r.created_at)}</span>
      </button>
      {/* expanded: per-step detail */}
      {open && (
        <div className="space-y-1.5 border-t border-[var(--color-border)] px-3 py-2.5">
          {(r.steps ?? []).map((s) => <StepRunRow key={s.step} s={s} aiById={aiById} />)}
          {r.error && (
            <div className="rounded-md border-l-2 px-3 py-1.5 font-mono text-[10.5px]" style={{ borderLeftColor: TONE_HEX.red, color: TONE_HEX.red }}>
              {r.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StepRunRow({ s, aiById }: { s: PipelineStep; aiById: Map<string, ArtifactInstance> }) {
  const tone = STEP_STATUS_TONE[s.status] ?? "muted";
  const metricKeys = Object.keys(s.metrics ?? {}).filter((k) => k !== "wall_s").slice(0, 4);
  const produced = (s.artifacts_produced ?? []).map((aid) => aiById.get(aid)).filter(Boolean) as ArtifactInstance[];
  return (
    <div className="border-l-2 pl-3" style={{ borderLeftColor: TONE_HEX[tone] }}>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[11px]">
        <span className="font-mono font-medium text-[var(--color-fg)]">{s.step}</span>
        <Link to="/substrate"><span className="font-mono text-[10.5px] text-[var(--color-pink)] hover:underline">{s.tool}</span></Link>
        {s.tool_version && <span className="font-mono text-[10px] text-[var(--color-faint)]">@{s.tool_version}</span>}
        {s.hash_ok === false
          ? <Pill tone="amber">drift</Pill>
          : <span title="entry_point hash in sync" className="text-[10px] text-[var(--color-green)]">● in sync</span>}
        {s.wall_s != null && <span className="font-mono text-[10px] text-[var(--color-faint)]">{fmtWall(s.wall_s)}</span>}
        {metricKeys.length > 0 && (
          <span className="truncate font-mono text-[10px] text-[var(--color-muted)]">
            {metricKeys.map((k) => `${k}=${JSON.stringify(s.metrics![k])}`).join("  ")}
          </span>
        )}
      </div>
      {s.error && (
        <div className="mt-1 font-mono text-[10.5px]" style={{ color: TONE_HEX.red }}>{s.error}</div>
      )}
      {/* threaded materials — the jars this step shelved */}
      {produced.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          <IconMaterials width={12} height={12} className="text-[var(--color-amber)]" />
          {produced.map((a) => (
            <Link key={a.id} to="/materials" title={`${a.id} · ${a.port}`}>
              <span
                className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono text-[10px] transition-colors hover:brightness-125"
                style={{ backgroundColor: TONE_HEX.amber + "14", color: TONE_HEX.amber, borderColor: TONE_HEX.amber + "44" }}
              >
                {a.artifact_type_term ?? a.artifact_type}
                <span className="text-[var(--color-faint)]">·{a.port}</span>
              </span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function BackLink() {
  return (
    <Link to="/pipelines" className="inline-flex items-center gap-1.5 text-[12px] text-[var(--color-muted)] hover:text-[var(--color-fg)]">
      <IconArrowRight width={13} height={13} className="rotate-180" /> back to pipelines
    </Link>
  );
}
