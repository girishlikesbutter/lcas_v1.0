import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import type { ArtifactInstance } from "../lib/types";
import { Card, Pill, SectionTitle, Spinner, Empty } from "../ui/kit";
import { shortId, fmtDate, TONE_HEX, type Tone } from "../lib/ui";
import {
  IconMaterials, IconArrowRight, IconPipelines, IconSubstrate, IconCheck, IconCopy,
} from "../ui/icons";

// Materials — the shelf (ADR-0007 Slice 1/2/3). Every recognised output a Tool or
// pipeline step produced becomes a labelled jar: "IA Cloud · from so3-pool-dedup ·
// step sample · made 2026-06-08". The shelf is browsable (filter by type) and you
// can COOK from it — copy the $artifact.<id> ref and pour it into a new run, no
// recompute (the working pantry). The blob lives under artifact_instances/data/;
// this card is the label. Re-feed is type-checked at the consuming port (Slice 2).

// stable tone per artifact-type so a given material reads the same colour everywhere
const TYPE_TONES: Tone[] = ["teal", "violet", "blue", "pink", "green", "amber"];
function typeTone(type: string, order: string[]): Tone {
  const i = order.indexOf(type);
  return TYPE_TONES[(i < 0 ? 0 : i) % TYPE_TONES.length];
}

export function Materials() {
  const { snap, loading } = useStore();
  const [filter, setFilter] = useState<string>("all");

  const mats = snap?.artifact_instances ?? [];
  // distinct types in stable (count-desc) order for the filter chips + tone map
  const types = useMemo(() => {
    const counts = new Map<string, number>();
    for (const m of mats) counts.set(m.artifact_type, (counts.get(m.artifact_type) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }, [mats]);
  const typeOrder = types.map(([t]) => t);

  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;

  const shown = filter === "all" ? mats : mats.filter((m) => m.artifact_type === filter);
  const labelFor = (t: string) => mats.find((m) => m.artifact_type === t)?.artifact_type_term ?? t;

  return (
    <div className="space-y-6">
      {/* page explainer */}
      <Card className="flex items-start gap-3 p-5">
        <IconMaterials width={20} height={20} className="mt-0.5 shrink-0 text-[var(--color-amber)]" />
        <div className="min-w-0 flex-1">
          <h1 className="text-[15px] font-semibold text-[var(--color-fg)]">Materials · the shelf</h1>
          <p className="mt-1 max-w-3xl text-[12px] leading-relaxed text-[var(--color-muted)]">
            Every <span className="text-[var(--color-fg-dim)]">recognised</span> output a Tool or pipeline step produces
            becomes a labelled jar — a <span className="font-mono text-[var(--color-amber)]">$artifact</span> you can pour
            into a new run with no recompute (the <span className="text-[var(--color-fg-dim)]">working pantry</span>).
            Re-feed is type-checked at the consuming port. The card is the label; the blob lives on disk.
          </p>
        </div>
        <Pill tone="amber" className="hidden shrink-0 whitespace-nowrap sm:inline-flex">{mats.length} jars</Pill>
      </Card>

      {mats.length === 0 ? (
        <Empty>No materials shelved yet. Run a pipeline or a Tool with declared output ports.</Empty>
      ) : (
        <>
          {/* type filter */}
          <Card className="flex flex-wrap items-center gap-2 px-4 py-3">
            <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">type</span>
            <FilterChip label="all" count={mats.length} active={filter === "all"} tone="muted" onClick={() => setFilter("all")} />
            {types.map(([t, n]) => (
              <FilterChip
                key={t}
                label={labelFor(t)}
                count={n}
                active={filter === t}
                tone={typeTone(t, typeOrder)}
                onClick={() => setFilter(t)}
              />
            ))}
          </Card>

          <SectionTitle icon={<IconMaterials width={15} height={15} className="text-[var(--color-amber)]" />} count={shown.length}>
            {filter === "all" ? "All materials" : labelFor(filter)}
          </SectionTitle>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {shown.map((m) => <MaterialCard key={m.id} m={m} tone={typeTone(m.artifact_type, typeOrder)} />)}
          </div>
        </>
      )}
    </div>
  );
}

function FilterChip({ label, count, active, tone, onClick }: {
  label: string; count: number; active: boolean; tone: Tone; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-medium transition-colors"
      style={
        active
          ? { backgroundColor: TONE_HEX[tone] + "22", color: TONE_HEX[tone], borderColor: TONE_HEX[tone] + "66" }
          : { borderColor: "var(--color-border)", color: "var(--color-muted)" }
      }
    >
      {label}
      <span className="rounded bg-[var(--color-elev)] px-1 text-[10px] text-[var(--color-faint)]">{count}</span>
    </button>
  );
}

function MaterialCard({ m, tone }: { m: ArtifactInstance; tone: Tone }) {
  const sample = m.sample ?? {};
  const shape = Array.isArray(sample.shape) ? `[${sample.shape.join(", ")}]` : null;
  const fields = Array.isArray(sample.fields) ? sample.fields : null;
  const keys = Array.isArray(sample.keys) ? sample.keys : null;
  const isRef = m.storage === "reference";
  const isPipeline = m.producer_kind === "pipeline_run";
  // producer_name for a pipeline_run is the pipeline id; for a tool_run, the tool id
  const producerLink = isPipeline ? `/pipelines/${m.producer_name}` : "/substrate";

  return (
    <Card className="flex flex-col overflow-hidden p-0" style={{ borderLeftWidth: 2, borderLeftColor: TONE_HEX[tone] }}>
      {/* label header */}
      <div className="border-b border-[var(--color-border)] px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] font-semibold" style={{ color: TONE_HEX[tone] }}>
            {m.artifact_type_term ?? m.artifact_type}
          </span>
          <div className="flex shrink-0 items-center gap-1.5">
            {isRef && <Pill tone="blue">ref</Pill>}
            <Pill tone={m.cardinality === "set" ? "violet" : "faint"}>
              {m.cardinality}{m.cardinality_n != null ? ` ·${m.cardinality_n}` : ""}
            </Pill>
          </div>
        </div>
        {m.caption && <p className="mt-1 text-[11.5px] leading-snug text-[var(--color-muted)]">{m.caption}</p>}
      </div>

      {/* body: lineage + digest */}
      <div className="flex-1 space-y-2.5 px-4 py-3 text-[11px]">
        {/* producer lineage */}
        <Link to={producerLink} className="flex items-center gap-1.5 text-[var(--color-fg-dim)] hover:text-[var(--color-blue)]">
          {isPipeline
            ? <IconPipelines width={13} height={13} className="text-[var(--color-teal)]" />
            : <IconSubstrate width={13} height={13} className="text-[var(--color-pink)]" />}
          <span className="font-mono">{shortId(m.producer_name ?? "—")}</span>
          {m.produced_by?.step && <span className="text-[var(--color-faint)]">· step {m.produced_by.step}</span>}
          <IconArrowRight width={11} height={11} className="text-[var(--color-faint)]" />
        </Link>

        {/* digest row */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10.5px] text-[var(--color-faint)]">
          <span>port <span className="text-[var(--color-muted)]">{m.port}</span></span>
          {shape && <span>shape <span className="text-[var(--color-muted)]">{shape}</span></span>}
          {sample.dtype && <span>{String(sample.dtype)}</span>}
          {sample.dataclass && <span className="text-[var(--color-muted)]">{String(sample.dataclass)}{sample.len != null ? `[${sample.len}]` : ""}</span>}
          {sample.file && <span className="text-[var(--color-muted)]">{String(sample.file)}{sample.exists === false ? " (missing)" : ""}</span>}
          {m.size_bytes != null && <span>{fmtBytes(m.size_bytes)}</span>}
        </div>
        {(fields || keys) && (
          <div className="font-mono text-[10px] text-[var(--color-faint)]">
            {fields ? "fields" : "keys"} <span className="text-[var(--color-muted)]">{(fields ?? keys)!.join(", ")}</span>
          </div>
        )}
        <div className="flex items-center gap-2 text-[10px] text-[var(--color-faint)]">
          <span>{fmtDate(m.created_at)}</span>
          {m.commit && <span className="font-mono">commit {m.commit}</span>}
        </div>
      </div>

      {/* cook-from-this footer */}
      <CookFromBar id={m.id} />
    </Card>
  );
}

// "cook from this" — copy the $artifact.<id> ref to paste into a run_tool --input.
// Re-feed needs the consuming Tool + kwarg (which we can't know here), so the honest
// affordance is the ref itself; the user pairs it with the port at the run site.
function CookFromBar({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  const ref = `$artifact.${id}`;
  const copy = () => {
    navigator.clipboard?.writeText(ref).then(
      () => { setCopied(true); window.setTimeout(() => setCopied(false), 1400); },
      () => {},
    );
  };
  return (
    <button
      onClick={copy}
      title={`Copy "${ref}" — pour this material into a new run (run_tool --input)`}
      className="flex items-center gap-2 border-t border-[var(--color-border)] bg-[var(--color-bg-2)]/40 px-4 py-2.5 text-left text-[11px] transition-colors hover:bg-[var(--color-elev)]/50"
    >
      {copied
        ? <IconCheck width={13} height={13} className="text-[var(--color-green)]" />
        : <IconCopy width={13} height={13} className="text-[var(--color-faint)]" />}
      <span className={copied ? "text-[var(--color-green)]" : "text-[var(--color-muted)]"}>
        {copied ? "copied ref" : "cook from this"}
      </span>
      <span className="ml-auto truncate font-mono text-[10px] text-[var(--color-faint)]">{ref}</span>
    </button>
  );
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
