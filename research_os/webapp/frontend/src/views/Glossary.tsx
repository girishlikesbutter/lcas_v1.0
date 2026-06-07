import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { Card, Pill, SectionTitle, Spinner, Empty } from "../ui/kit";
import { TONE_HEX, type Tone } from "../lib/ui";
import { IconGlossary, IconWarn, IconDoc, IconLink, IconSearch } from "../ui/icons";
import type { GlossaryTerm } from "../lib/types";

// The canon vocabulary (snap.glossary). Kills "unexplained vocab": every term is
// defined once, blocked synonyms are lint-enforced, and provisional terms graduate
// to canon through a promotion gate. Dictionary-like, alphabetical, searchable.

type StatusFilter = "all" | "canon" | "provisional";

const STATUS_TONE: Record<GlossaryTerm["status"], Tone> = {
  canon: "green",
  provisional: "amber",
};

// A coined_in value that starts with sNNN (a run id) deep-links to the run; anything
// else is a writeup path shown as a faint mono reference.
const RUN_RE = /^s\d{2,4}/;

export function Glossary() {
  const { snap, loading } = useStore();
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");

  const sorted = useMemo(() => {
    const list = snap?.glossary ?? [];
    return [...list].sort((a, b) =>
      (a.term ?? a.id).toLowerCase().localeCompare((b.term ?? b.id).toLowerCase())
    );
  }, [snap]);

  // term id -> display term, so related_terms chips can show the human label.
  const labelById = useMemo(() => {
    const m: Record<string, string> = {};
    for (const t of sorted) m[t.id] = t.term;
    return m;
  }, [sorted]);

  // the set of real run ids — only linkify coined_in when it actually resolves
  // (some terms are coined in a writeup or a short tag like "s113" with no record).
  const runIds = useMemo(() => new Set((snap?.runs ?? []).map((r) => r.id)), [snap]);

  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;

  const counts = {
    all: sorted.length,
    canon: sorted.filter((t) => t.status === "canon").length,
    provisional: sorted.filter((t) => t.status === "provisional").length,
  };

  const q = query.trim().toLowerCase();
  const filtered = sorted.filter((t) => {
    if (status !== "all" && t.status !== status) return false;
    if (!q) return true;
    const hay = [
      t.term,
      t.definition,
      ...(t.aliases ?? []),
      ...(t.synonyms_blocked ?? []),
    ]
      .join(" ")
      .toLowerCase();
    return hay.includes(q);
  });

  const tabs: { k: StatusFilter; label: string; tone: Tone }[] = [
    { k: "all", label: "all", tone: "muted" },
    { k: "canon", label: "canon", tone: "green" },
    { k: "provisional", label: "provisional", tone: "amber" },
  ];

  return (
    <div className="space-y-6">
      {/* header explainer */}
      <Card className="p-5">
        <SectionTitle
          icon={<IconGlossary width={15} height={15} className="text-[var(--color-amber)]" />}
          count={sorted.length}
        >
          Glossary · the canon vocabulary
        </SectionTitle>
        <p className="max-w-3xl text-[12px] leading-relaxed text-[var(--color-muted)]">
          One definition per term, so a name never drifts.{" "}
          <span className="text-[var(--color-amber)]">Provisional</span> terms graduate to{" "}
          <span className="text-[var(--color-green)]">canon</span> through a promotion gate. Each
          term's <span className="font-mono text-[var(--color-faint)]">blocked</span> synonyms are
          auto-flagged by the glossary-lint hook and rewritten on sight — they are the words this
          term retired.
        </p>
      </Card>

      {/* search + status filter */}
      <Card className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-1 items-center gap-2.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/60 px-3 py-2">
          <IconSearch width={15} height={15} className="text-[var(--color-faint)]" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="search term, definition, alias…"
            className="w-full bg-transparent text-[13px] text-[var(--color-fg)] placeholder:text-[var(--color-faint)] focus:outline-none"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              className="text-[11px] text-[var(--color-faint)] hover:text-[var(--color-fg)]"
            >
              clear
            </button>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {tabs.map((t) => {
            const active = status === t.k;
            return (
              <button
                key={t.k}
                onClick={() => setStatus(t.k)}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-[11px] font-medium transition-colors ${
                  active
                    ? ""
                    : "border-[var(--color-border)] text-[var(--color-muted)] hover:border-[var(--color-border-2)] hover:text-[var(--color-fg)]"
                }`}
                style={
                  active
                    ? {
                        backgroundColor: TONE_HEX[t.tone] + "22",
                        color: TONE_HEX[t.tone],
                        borderColor: TONE_HEX[t.tone] + "55",
                      }
                    : undefined
                }
              >
                {t.label}
                <span className="font-mono text-[var(--color-faint)]">{counts[t.k]}</span>
              </button>
            );
          })}
        </div>
      </Card>

      {/* term grid */}
      {filtered.length === 0 ? (
        <Empty>No terms match "{query}".</Empty>
      ) : (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {filtered.map((t) => (
            <TermCard key={t.id} t={t} labelById={labelById} onPick={setQuery} runIds={runIds} />
          ))}
        </div>
      )}
    </div>
  );
}

function TermCard({
  t,
  labelById,
  onPick,
  runIds,
}: {
  t: GlossaryTerm;
  labelById: Record<string, string>;
  onPick: (q: string) => void;
  runIds: Set<string>;
}) {
  const statusTone = STATUS_TONE[t.status];
  const blocked = t.synonyms_blocked ?? [];
  const aliases = t.aliases ?? [];
  const related = t.related_terms ?? [];
  const coined = t.coined_in;
  // link only when coined_in resolves to a real run record (RUN_RE shape + exists)
  const coinedIsRun = !!coined && RUN_RE.test(coined) && runIds.has(coined);

  return (
    <Card className="flex flex-col gap-3 p-5">
      {/* header: term + status */}
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-mono text-[14px] font-semibold leading-snug text-[var(--color-fg)]">
          {t.term}
        </h3>
        <Pill tone={statusTone}>{t.status}</Pill>
      </div>

      {/* definition */}
      <p className="text-[12px] leading-relaxed text-[var(--color-muted)]">{t.definition}</p>

      {/* coined_in */}
      {coined && (
        <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-faint)]">
          <IconDoc width={12} height={12} />
          <span className="uppercase tracking-wider">coined in</span>
          {coinedIsRun ? (
            <Link
              to={`/runs/${coined}`}
              className="font-mono text-[var(--color-blue)] hover:underline"
            >
              {coined}
            </Link>
          ) : (
            <span className="truncate font-mono text-[var(--color-faint)]" title={coined}>
              {coined}
            </span>
          )}
        </div>
      )}

      {/* blocked synonyms — lint rewrites these on sight */}
      {blocked.length > 0 && (
        <div
          className="flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-[11px]"
          style={{
            backgroundColor: TONE_HEX.red + "14",
            borderColor: TONE_HEX.red + "44",
          }}
        >
          <IconWarn width={13} height={13} className="shrink-0 text-[var(--color-red)]" />
          <span className="text-[var(--color-faint)]">blocked</span>
          <span className="font-medium text-[var(--color-red)]">{blocked.join(", ")}</span>
        </div>
      )}

      {/* aliases + related — pushed to the card bottom */}
      {(aliases.length > 0 || related.length > 0) && (
        <div className="mt-auto flex flex-wrap items-center gap-1.5 border-t border-[var(--color-border)] pt-3">
          {aliases.map((a) => (
            <span
              key={`aka-${a}`}
              className="rounded-md bg-[var(--color-elev)] px-2 py-0.5 text-[10px] text-[var(--color-faint)]"
            >
              aka {a}
            </span>
          ))}
          {related.map((r) => {
            const label = labelById[r] ?? r;
            return (
              <button
                key={`rel-${r}`}
                onClick={() => onPick(label)}
                title={`filter to ${label}`}
                className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] px-2 py-0.5 text-[10px] text-[var(--color-muted)] transition-colors hover:border-[var(--color-border-2)] hover:text-[var(--color-fg)]"
              >
                <IconLink width={10} height={10} className="text-[var(--color-faint)]" />
                {label}
              </button>
            );
          })}
        </div>
      )}
    </Card>
  );
}
