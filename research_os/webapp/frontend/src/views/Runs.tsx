import { useMemo, useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import type { Run, Artefact, BlastRef } from "../lib/types";
import {
  Card, Pill, Dot, SectionTitle, MetricChip, Empty, Spinner, Button,
} from "../ui/kit";
import {
  RUN_STATUS, RHO_BAND, shortId, runShort, fmtDate, TONE_HEX, type Tone,
} from "../lib/ui";
import {
  IconRuns, IconBlast, IconArrowRight, IconWarn, IconDoc, IconLink,
  IconStream, IconBolt, IconExternal,
} from "../ui/icons";

// Runs — the run-record timeline (list) + the full run detail. One component
// serves both "/runs" and "/runs/:id" (PLAN §2 "run records are canonical state").
// The list mirrors Overview's RecentRuns vocabulary; the detail unfolds the whole
// record — metrics, narrative, artefacts, substrate, lineage, gates.

type StatusFilter = "all" | "confirmed" | "refuted" | "inconclusive";

/** ρ for a run — the canonical metric is `rho`; fall back to common analogues. */
function runRho(r: Run): number | undefined {
  const m = r.metrics;
  if (!m) return undefined;
  const keys = ["rho", "hifi_rho", "rho_min_hifi", "best_rho_final", "rho_surr_max", "best_polish_rho"];
  for (const k of keys) {
    const v = m[k];
    if (typeof v === "number" && isFinite(v)) return v;
  }
  return undefined;
}
function runBand(r: Run): string | undefined {
  const b = r.metrics?.rho_band;
  return typeof b === "string" ? b : undefined;
}
function runWdir(r: Run): number | undefined {
  const v = r.metrics?.w_dir_err;
  return typeof v === "number" && isFinite(v) ? v : undefined;
}

export function Runs() {
  const { id } = useParams<{ id: string }>();
  const { snap, loading } = useStore();
  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;
  if (id) return <RunDetail id={id} />;
  return <RunList />;
}

// ---------------------------------------------------------------------------
// LIST
// ---------------------------------------------------------------------------

function RunList() {
  const { snap } = useStore();
  const [status, setStatus] = useState<StatusFilter>("all");
  const [oracleOnly, setOracleOnly] = useState(false);
  const [bandOnly, setBandOnly] = useState(false);

  const runs = snap!.runs; // newest-first already
  const filtered = useMemo(
    () =>
      runs.filter((r) => {
        if (status !== "all" && r.status !== status) return false;
        if (oracleOnly && r.oracle_clean !== true) return false;
        if (bandOnly && !runBand(r)) return false;
        return true;
      }),
    [runs, status, oracleOnly, bandOnly],
  );

  const statusOpts: StatusFilter[] = ["all", "confirmed", "refuted", "inconclusive"];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-[19px] font-semibold leading-tight text-[var(--color-fg)]">Run records</h1>
        <p className="mt-1 text-[12px] text-[var(--color-muted)]">
          Canonical state — every executed experiment, newest first. {runs.length} total.
        </p>
      </div>

      {/* filter bar */}
      <Card className="flex flex-wrap items-center gap-x-5 gap-y-3 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">status</span>
          <div className="flex gap-1">
            {statusOpts.map((s) => {
              const active = status === s;
              const tone: Tone = s === "all" ? "blue" : (RUN_STATUS[s]?.tone ?? "muted");
              return (
                <button
                  key={s}
                  onClick={() => setStatus(s)}
                  className="rounded-md border px-2 py-1 text-[11px] font-medium capitalize transition-colors"
                  style={
                    active
                      ? { backgroundColor: TONE_HEX[tone] + "22", color: TONE_HEX[tone], borderColor: TONE_HEX[tone] + "55" }
                      : { borderColor: "var(--color-border)", color: "var(--color-muted)" }
                  }
                >
                  {s}
                </button>
              );
            })}
          </div>
        </div>

        <Toggle label="oracle-clean only" on={oracleOnly} onClick={() => setOracleOnly((v) => !v)} />
        <Toggle label="has ρ-band" on={bandOnly} onClick={() => setBandOnly((v) => !v)} />

        <span className="ml-auto text-[11px] text-[var(--color-faint)]">{filtered.length} shown</span>
      </Card>

      {/* table */}
      <Card className="p-2">
        {filtered.length === 0 ? (
          <Empty>No runs match these filters.</Empty>
        ) : (
          <div className="divide-y divide-[var(--color-border)]">
            {/* header row */}
            <div className="hidden grid-cols-[14px_88px_1fr_auto_92px] items-center gap-3 px-3 py-2 text-[10px] uppercase tracking-wider text-[var(--color-faint)] md:grid">
              <span />
              <span>run</span>
              <span>goal</span>
              <span className="text-right">metrics</span>
              <span className="text-right">date</span>
            </div>
            {filtered.map((r) => (
              <RunRow key={r.id} r={r} />
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function Toggle({ label, on, onClick }: { label: string; on: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 text-[11px] font-medium transition-colors"
      style={{ color: on ? TONE_HEX.blue : "var(--color-muted)" }}
    >
      <span
        className="inline-flex h-4 w-7 items-center rounded-full p-0.5 transition-colors"
        style={{ backgroundColor: on ? TONE_HEX.blue : "var(--color-elev)" }}
      >
        <span
          className="h-3 w-3 rounded-full bg-white transition-transform"
          style={{ transform: on ? "translateX(12px)" : "translateX(0)" }}
        />
      </span>
      {label}
    </button>
  );
}

function RunRow({ r }: { r: Run }) {
  const tone = (RUN_STATUS[r.status]?.tone ?? "muted") as Tone;
  const rho = runRho(r);
  const band = runBand(r);
  const wdir = runWdir(r);
  const bandTone: Tone = band ? (RHO_BAND[band] ?? "muted") : "muted";

  return (
    <Link
      to={`/runs/${r.id}`}
      className="grid grid-cols-[14px_1fr] items-center gap-3 px-3 py-2.5 hover:bg-[var(--color-elev)]/40 md:grid-cols-[14px_88px_1fr_auto_92px]"
    >
      <Dot tone={tone} size={8} />

      <span className="flex items-center gap-1.5 font-mono text-[12px] text-[var(--color-fg-dim)]">
        {runShort(r.id)}
        {r.oracle_clean === false && (
          <IconWarn width={12} height={12} className="text-[var(--color-amber)]" />
        )}
        {r.blast.length > 0 && (
          <IconBlast width={12} height={12} className="text-[var(--color-amber)]" />
        )}
      </span>

      <span className="min-w-0 truncate text-[12px] text-[var(--color-muted)]">
        {r.goal_title ?? r.question ?? shortId(r.goal_node)}
      </span>

      <span className="flex flex-wrap items-center justify-start gap-1.5 md:justify-end">
        {rho != null && <MetricChip k="ρ" v={rho.toFixed(3)} tone={bandTone} />}
        {band && (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{ backgroundColor: TONE_HEX[bandTone] + "22", color: TONE_HEX[bandTone] }}
          >
            {band}
          </span>
        )}
        {wdir != null && <MetricChip k="ω-dir" v={`${wdir.toFixed(2)}°`} />}
      </span>

      <span className="text-right font-mono text-[11px] text-[var(--color-faint)]">
        {fmtDate(r.created_at)}
      </span>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// DETAIL
// ---------------------------------------------------------------------------

function RunDetail({ id }: { id: string }) {
  const { snap } = useStore();
  const { launchSkill } = useTerminals();
  const run = snap!.runs.find((r) => r.id === id);

  if (!run) {
    return (
      <div className="space-y-6">
        <BackLink />
        <Empty>No run record found for <span className="ml-1 font-mono">{id}</span>.</Empty>
      </div>
    );
  }

  const tone = (RUN_STATUS[run.status]?.tone ?? "muted") as Tone;
  const metricEntries = Object.entries(run.metrics ?? {});
  const subVersions = Object.entries(run.substrate_versions ?? {});

  return (
    <div className="space-y-6">
      <BackLink />

      {/* header */}
      <Card className="p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <Dot tone={tone} size={9} />
              <span className="font-mono text-[15px] font-semibold text-[var(--color-fg)]">{runShort(run.id)}</span>
              <Pill tone={tone}>{RUN_STATUS[run.status]?.label ?? run.status}</Pill>
              {run.run_type && <Pill tone="muted">{run.run_type}</Pill>}
              {run.oracle_clean === false && (
                <Pill tone="amber"><IconWarn width={11} height={11} /> oracle-tainted</Pill>
              )}
              <span className="font-mono text-[11px] text-[var(--color-faint)]">{fmtDate(run.created_at)}</span>
            </div>
            <div className="mt-1 break-all font-mono text-[11px] text-[var(--color-faint)]">{run.id}</div>
            {run.goal_title && (
              <Link
                to="/tree"
                className="mt-2 inline-flex items-center gap-1.5 text-[13px] text-[var(--color-fg-dim)] hover:text-[var(--color-blue)]"
              >
                <IconLink width={13} height={13} className="text-[var(--color-faint)]" />
                {run.goal_title}
                <span className="font-mono text-[11px] text-[var(--color-faint)]">{shortId(run.goal_node)}</span>
              </Link>
            )}
          </div>
          {run.status === "refuted" && (
            <Button tone="red" variant="soft" onClick={() => launchSkill("diagnose")}>
              <IconBolt width={13} height={13} /> diagnose
            </Button>
          )}
        </div>

        {/* question + hypothesis */}
        {(run.question || run.hypothesis) && (
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {run.question && <Field label="question">{run.question}</Field>}
            {run.hypothesis && <Field label="hypothesis">{run.hypothesis}</Field>}
          </div>
        )}
      </Card>

      {/* metrics */}
      {metricEntries.length > 0 && (
        <Card className="p-5">
          <SectionTitle icon={<IconRuns width={15} height={15} className="text-[var(--color-blue)]" />} count={metricEntries.length}>
            Metrics
          </SectionTitle>
          <div className="flex flex-wrap gap-2">
            {metricEntries.map(([k, v]) => {
              const isBand = k === "rho_band" && typeof v === "string";
              const bandTone: Tone = isBand ? (RHO_BAND[v as string] ?? "muted") : "muted";
              const isRho = /(^|_)rho($|_)/.test(k);
              return (
                <MetricChip
                  key={k}
                  k={k}
                  v={fmtMetric(v)}
                  tone={isBand ? bandTone : isRho ? "teal" : undefined}
                />
              );
            })}
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* LEFT 2/3 — narrative + artefacts + blast */}
        <div className="space-y-6 lg:col-span-2">
          {run.narrative_md && (
            <Card className="p-5">
              <SectionTitle icon={<IconDoc width={15} height={15} className="text-[var(--color-muted)]" />}>
                Narrative
              </SectionTitle>
              <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/50 px-4 py-3">
                <Markdown text={run.narrative_md} />
              </div>
            </Card>
          )}

          {run.artefacts && run.artefacts.length > 0 && (
            <Card className="p-5">
              <SectionTitle count={run.artefacts.length}>Artefacts</SectionTitle>
              <div className="space-y-1.5">
                {run.artefacts.map((a, i) => (
                  <ArtefactRow key={`${a.path}-${i}`} a={a} />
                ))}
              </div>
            </Card>
          )}

          {/* blast radius */}
          {run.blast.length > 0 && (
            <Card className="border-l-2 p-5" style={{ borderLeftColor: TONE_HEX.amber }}>
              <SectionTitle icon={<IconBlast width={15} height={15} className="text-[var(--color-amber)]" />} count={run.blast.length}>
                Blast radius
              </SectionTitle>
              <p className="mb-3 text-[12px] text-[var(--color-muted)]">
                This run cites substrate that has since moved — its numbers may be stale.
              </p>
              <div className="space-y-1.5">
                {run.blast.map((b, i) => (
                  <BlastRow key={`${b.ref}-${i}`} b={b} />
                ))}
              </div>
            </Card>
          )}
        </div>

        {/* RIGHT 1/3 — substrate, lineage, gates, writeups */}
        <div className="space-y-6">
          {subVersions.length > 0 && (
            <Card className="p-5">
              <SectionTitle>Substrate</SectionTitle>
              <div className="flex flex-wrap gap-1.5">
                {subVersions.map(([comp, ver]) => (
                  <Link key={comp} to="/substrate">
                    <span className="rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-1 font-mono text-[11px] text-[var(--color-muted)] hover:border-[var(--color-border-2)]">
                      {comp}@<span className="text-[var(--color-fg-dim)]">{ver}</span>
                    </span>
                  </Link>
                ))}
              </div>
            </Card>
          )}

          {/* lineage */}
          {(hasItems(run.parents) || hasItems(run.tests) || hasItems(run.blocked_by)) && (
            <Card className="p-5">
              <SectionTitle icon={<IconLink width={15} height={15} className="text-[var(--color-violet)]" />}>
                Lineage
              </SectionTitle>
              <div className="space-y-3">
                {hasItems(run.parents) && (
                  <LineageBlock label="parents">
                    {run.parents.map((p) => (
                      <Link key={p} to={`/runs/${p}`}>
                        <span className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-1 font-mono text-[11px] text-[var(--color-blue)] hover:border-[var(--color-border-2)]">
                          {runShort(p)} <IconArrowRight width={11} height={11} />
                        </span>
                      </Link>
                    ))}
                  </LineageBlock>
                )}
                {hasItems(run.tests) && (
                  <LineageBlock label="tests pipelines">
                    {run.tests.map((t) => (
                      <Link key={t} to="/pipelines">
                        <span className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-1 font-mono text-[11px] text-[var(--color-teal)] hover:border-[var(--color-border-2)]">
                          {shortId(t)}
                        </span>
                      </Link>
                    ))}
                  </LineageBlock>
                )}
                {hasItems(run.blocked_by) && (
                  <LineageBlock label="blocked by">
                    {run.blocked_by.map((g) => (
                      <Link key={g} to="/glossary">
                        <span
                          className="inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[11px] hover:opacity-90"
                          style={{ backgroundColor: TONE_HEX.amber + "18", color: TONE_HEX.amber, borderColor: TONE_HEX.amber + "44" }}
                        >
                          {shortId(g)}
                        </span>
                      </Link>
                    ))}
                  </LineageBlock>
                )}
              </div>
            </Card>
          )}

          {/* gates */}
          {(hasItems(run.gates_passed) || hasItems(run.gates_failed)) && (
            <Card className="p-5">
              <SectionTitle>Gates</SectionTitle>
              <div className="space-y-2">
                {hasItems(run.gates_passed) && (
                  <div className="flex flex-wrap gap-1.5">
                    {run.gates_passed.map((g) => (
                      <GateChip key={g} label={g} tone="green" />
                    ))}
                  </div>
                )}
                {hasItems(run.gates_failed) && (
                  <div className="flex flex-wrap gap-1.5">
                    {run.gates_failed.map((g) => (
                      <GateChip key={g} label={g} tone="red" />
                    ))}
                  </div>
                )}
              </div>
            </Card>
          )}

          {/* writeups */}
          {hasItems(run.writeup_refs) && (
            <Card className="p-5">
              <SectionTitle icon={<IconDoc width={15} height={15} className="text-[var(--color-muted)]" />}>
                Writeups
              </SectionTitle>
              <div className="space-y-1.5">
                {run.writeup_refs.map((w) => (
                  <div key={w} className="break-all font-mono text-[11px] text-[var(--color-muted)]">{w}</div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// detail sub-components
// ---------------------------------------------------------------------------

function BackLink() {
  return (
    <Link to="/runs" className="inline-flex items-center gap-1.5 text-[12px] text-[var(--color-muted)] hover:text-[var(--color-fg)]">
      <IconArrowRight width={13} height={13} className="rotate-180" /> back to runs
    </Link>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-3 py-2.5">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">{label}</div>
      <div className="text-[13px] leading-relaxed text-[var(--color-fg-dim)]">{children}</div>
    </div>
  );
}

function LineageBlock({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">{label}</div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );
}

function GateChip({ label, tone }: { label: string; tone: Tone }) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[11px]"
      style={{ backgroundColor: TONE_HEX[tone] + "18", color: TONE_HEX[tone], borderColor: TONE_HEX[tone] + "44" }}
    >
      {tone === "green" ? "✓" : "✗"} {label}
    </span>
  );
}

const ARTEFACT_TONE: Record<Artefact["kind"], Tone> = {
  plot: "teal", anim: "violet", checkpoint: "blue", data: "muted", report: "green", log: "faint",
};

function ArtefactRow({ a }: { a: Artefact }) {
  const tone: Tone = ARTEFACT_TONE[a.kind] ?? "muted";
  const body = (
    <div className="flex items-center gap-2.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-3 py-2 hover:border-[var(--color-border-2)]">
      <Pill tone={tone}>{a.kind}</Pill>
      {a.kind === "plot" && <IconStream width={14} height={14} className="shrink-0 text-[var(--color-teal)]" />}
      <span className="min-w-0 flex-1 break-all font-mono text-[11px] text-[var(--color-muted)]">{a.path}</span>
      {a.caption && (
        <span className="hidden truncate text-[11px] text-[var(--color-faint)] lg:block lg:max-w-[35%]">{a.caption}</span>
      )}
      {a.kind === "plot"
        ? <IconArrowRight width={13} height={13} className="shrink-0 text-[var(--color-faint)]" />
        : <IconExternal width={13} height={13} className="shrink-0 text-[var(--color-faint)]" />}
    </div>
  );
  return a.kind === "plot" ? <Link to="/stream" className="block">{body}</Link> : body;
}

function BlastRow({ b }: { b: BlastRef }) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-3 py-2 text-[11px]">
      <span className="font-mono text-[var(--color-fg-dim)]">{b.component}</span>
      <span className="font-mono text-[var(--color-amber)]">{b.cited}</span>
      <IconArrowRight width={12} height={12} className="text-[var(--color-faint)]" />
      <span className="font-mono text-[var(--color-green)]">{b.current}</span>
      {b.ref && <span className="ml-auto font-mono text-[10px] text-[var(--color-faint)]">{shortId(b.ref)}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function hasItems<T>(a: T[] | undefined | null): a is T[] {
  return Array.isArray(a) && a.length > 0;
}

function fmtMetric(v: unknown): string {
  if (typeof v === "number") {
    if (!isFinite(v)) return String(v);
    if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) return v.toExponential(2);
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(3);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (v == null) return "—";
  return String(v);
}

/** Minimal markdown — paragraphs on blank lines, "- " lines become a bulleted list. */
function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let para: string[] = [];
  let list: string[] = [];
  let key = 0;

  const flushPara = () => {
    if (para.length) {
      blocks.push(
        <p key={key++} className="text-[12.5px] leading-relaxed text-[var(--color-muted)]">{para.join(" ")}</p>,
      );
      para = [];
    }
  };
  const flushList = () => {
    if (list.length) {
      blocks.push(
        <ul key={key++} className="ml-1 space-y-1">
          {list.map((li, i) => (
            <li key={i} className="flex gap-2 text-[12.5px] leading-relaxed text-[var(--color-muted)]">
              <span className="mt-[2px] text-[var(--color-faint)]">•</span>
              <span>{li}</span>
            </li>
          ))}
        </ul>,
      );
      list = [];
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (line === "") {
      flushList();
      flushPara();
    } else if (line.startsWith("- ")) {
      flushPara();
      list.push(line.slice(2));
    } else {
      flushList();
      para.push(line);
    }
  }
  flushList();
  flushPara();

  return <div className="space-y-3">{blocks}</div>;
}
