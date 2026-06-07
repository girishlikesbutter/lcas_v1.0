import { useCallback, useMemo, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { api } from "../lib/api";
import type { Claim } from "../lib/types";
import { Card, Pill, Dot, Spinner, Empty, Button } from "../ui/kit";
import {
  CLAIM_STATUS, CONFIDENCE, shortId, runShort, fmtDate, TONE_HEX, type Tone,
} from "../lib/ui";
import { IconClaims, IconBlast, IconArrowRight, IconLink, IconDoc, IconCheck } from "../ui/icons";

// Claims — the knowledge state: "what's true right now". Two decks: research
// (evidentiary, backed by runs) and preference (working preferences). Each card
// makes the trust state legible at a glance — status, confidence, scope, oracle
// cleanliness, blast-radius staleness, supporting/refuting runs, supersession.

type Deck = "research" | "preference";
type StatusKey = "all" | "live" | "needs_replication" | "superseded" | "retracted" | "draft";

// live first, then the rest in trust-priority order.
const STATUS_ORDER: Record<string, number> = {
  live: 0, needs_replication: 1, draft: 2, superseded: 3, retracted: 4,
};

const FILTERS: { key: StatusKey; label: string }[] = [
  { key: "all", label: "all" },
  { key: "live", label: "live" },
  { key: "needs_replication", label: "needs replication" },
  { key: "superseded", label: "superseded" },
  { key: "retracted", label: "retracted" },
  { key: "draft", label: "draft" },
];

export function Claims() {
  const { snap, loading } = useStore();
  const [deck, setDeck] = useState<Deck>("research");
  const [status, setStatus] = useState<StatusKey>("all");

  const all = snap?.claims ?? [];
  const deckClaims = useMemo(() => all.filter((c) => c.deck === deck), [all, deck]);

  // counts per status within the active deck (drives the filter-chip badges).
  const statusCounts = useMemo(() => {
    const m: Record<string, number> = {};
    for (const c of deckClaims) m[c.status] = (m[c.status] ?? 0) + 1;
    return m;
  }, [deckClaims]);

  const visible = useMemo(() => {
    const f = status === "all" ? deckClaims : deckClaims.filter((c) => c.status === status);
    return [...f].sort(
      (a, b) =>
        (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9) ||
        (b.scope?.N ?? 0) - (a.scope?.N ?? 0),
    );
  }, [deckClaims, status]);

  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;

  const researchN = all.filter((c) => c.deck === "research").length;
  const preferenceN = all.filter((c) => c.deck === "preference").length;
  const blastN = visible.filter((c) => c.blast_stale).length;

  return (
    <div className="space-y-6">
      {/* header + deck tabs */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <IconClaims width={18} height={18} className="text-[var(--color-green)]" />
          <h1 className="text-[16px] font-semibold text-[var(--color-fg)]">Claims</h1>
          <span className="text-[12px] text-[var(--color-faint)]">— what's true right now</span>
        </div>
        <div className="flex items-center gap-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-2)] p-1">
          <DeckTab active={deck === "research"} count={researchN} onClick={() => setDeck("research")}>research</DeckTab>
          <DeckTab active={deck === "preference"} count={preferenceN} onClick={() => setDeck("preference")}>preference</DeckTab>
        </div>
      </div>

      {/* status filter chip row */}
      <div className="flex flex-wrap items-center gap-2">
        {FILTERS.map((f) => {
          const n = f.key === "all" ? deckClaims.length : (statusCounts[f.key] ?? 0);
          const active = status === f.key;
          const tone = (f.key === "all" ? "muted" : (CLAIM_STATUS[f.key]?.tone ?? "muted")) as Tone;
          return (
            <button
              key={f.key}
              onClick={() => setStatus(f.key)}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                active ? "" : "border-[var(--color-border)] text-[var(--color-muted)] hover:border-[var(--color-border-2)] hover:text-[var(--color-fg-dim)]"
              }`}
              style={active ? { backgroundColor: TONE_HEX[tone] + "22", color: TONE_HEX[tone], borderColor: TONE_HEX[tone] + "55" } : undefined}
            >
              {f.key !== "all" && <Dot tone={tone} size={6} />}
              {f.label}
              <span className={`font-mono ${active ? "" : "text-[var(--color-faint)]"}`}>{n}</span>
            </button>
          );
        })}
      </div>

      {/* blast-radius banner (within the current filtered view) */}
      {blastN > 0 && (
        <Link to="/substrate">
          <Card hover className="flex items-center gap-3 border-l-2 px-4 py-3" style={{ borderLeftColor: TONE_HEX.amber }}>
            <IconBlast className="text-[var(--color-amber)]" width={18} height={18} />
            <div className="flex-1 text-[13px]">
              <span className="font-medium text-[var(--color-fg)]">{blastN} claim{blastN > 1 ? "s" : ""}</span>
              <span className="text-[var(--color-muted)]"> here rest on superseded substrate — review the blast radius.</span>
            </div>
            <IconArrowRight width={15} height={15} className="text-[var(--color-faint)]" />
          </Card>
        </Link>
      )}

      {/* the deck */}
      {visible.length === 0 ? (
        <Empty>No {deck} claims{status !== "all" ? ` with status "${CLAIM_STATUS[status]?.label ?? status}"` : ""}.</Empty>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {visible.map((c) => <ClaimCard key={c.id} claim={c} />)}
        </div>
      )}
    </div>
  );
}

function DeckTab({ active, count, onClick, children }: {
  active: boolean; count: number; onClick: () => void; children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[12px] font-medium transition-colors ${
        active
          ? "bg-[var(--color-elev)] text-[var(--color-fg)]"
          : "text-[var(--color-muted)] hover:text-[var(--color-fg-dim)]"
      }`}
    >
      {children}
      <span className={`rounded px-1 font-mono text-[10px] ${active ? "bg-[var(--color-bg-2)] text-[var(--color-fg-dim)]" : "text-[var(--color-faint)]"}`}>{count}</span>
    </button>
  );
}

function ClaimCard({ claim }: { claim: Claim }) {
  const c = claim;
  const isPref = c.deck === "preference";
  // Optional-field guards: an auto-drafted claim (Item #10 write-flow) may land
  // before trust_stamp/scope are filled in, so never assume their shape here.
  const trust = c.trust_stamp ?? {};
  const scope = c.scope ?? { N: 0, kind: "N1" as const };
  const statusTone = (CLAIM_STATUS[c.status]?.tone ?? "muted") as Tone;
  const confTone = (CONFIDENCE[trust.confidence ?? ""] ?? "faint") as Tone;
  const oracleClean = trust.oracle_clean;
  const scopeLabel = scope.kind === "cohort"
    ? `N=${scope.N} · cohort`
    : ((scope.N ?? 0) > 0 ? `N=${scope.N}` : "standing");
  const dimmed = c.status === "superseded" || c.status === "retracted";

  return (
    <Card className={`flex flex-col gap-3 p-5 ${dimmed ? "opacity-75" : ""}`}>
      {/* statement headline */}
      <p className="text-[13.5px] font-medium leading-relaxed text-[var(--color-fg)] line-clamp-3" title={c.statement ?? ""}>
        {c.statement ?? "(untitled claim)"}
      </p>

      {/* trust-stamp row */}
      <div className="flex flex-wrap items-center gap-1.5">
        <Pill tone={statusTone}>
          <Dot tone={statusTone} size={6} /> {CLAIM_STATUS[c.status]?.label ?? c.status}
        </Pill>
        {trust.confidence && <Pill tone={confTone}>{trust.confidence}</Pill>}
        <span className="inline-flex items-center rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-0.5 text-[11px] text-[var(--color-muted)]">
          {scopeLabel}
        </span>
        {!isPref && oracleClean !== undefined && (
          <Pill tone={oracleClean ? "green" : "amber"}>
            {oracleClean ? "oracle-clean" : "oracle-tainted"}
          </Pill>
        )}
      </div>

      {/* auto-draft → human-confirm (Q8 autonomy map: claim writes are confirmed,
          never auto-applied). The backend only enqueues the intent; the executor /
          close skill flips draft→live, keeping the store canonical. */}
      {c.status === "draft" && <ConfirmDraft claim={c} />}

      {/* blast-radius strip — rests on superseded substrate */}
      {c.blast_stale && c.blast.length > 0 && (
        <Link
          to="/substrate"
          className="flex flex-col gap-1.5 rounded-lg border border-l-2 px-3 py-2 transition-colors hover:bg-[var(--color-elev)]/40"
          style={{ borderColor: TONE_HEX.amber + "44", borderLeftColor: TONE_HEX.amber }}
        >
          <div className="flex items-center gap-2 text-[11px] font-medium text-[var(--color-amber)]">
            <IconBlast width={13} height={13} /> rests on superseded substrate
          </div>
          <div className="flex flex-wrap gap-1.5">
            {c.blast.map((b) => (
              <span key={b.ref} className="font-mono text-[11px] text-[var(--color-muted)]">
                {b.component}@<span className="text-[var(--color-red)]">{b.cited}</span>
                <span className="text-[var(--color-faint)]"> → </span>
                <span className="text-[var(--color-green)]">{b.current}</span>
              </span>
            ))}
          </div>
        </Link>
      )}

      {/* supersession lineage */}
      {(c.superseded_by || c.supersedes) && (
        <div className="flex flex-col gap-0.5 text-[11px] text-[var(--color-faint)]">
          {c.superseded_by && (
            <span>superseded by <span className="font-mono text-[var(--color-muted)]">{shortId(c.superseded_by)}</span></span>
          )}
          {c.supersedes && (
            <span>supersedes <span className="font-mono text-[var(--color-muted)]">{shortId(c.supersedes)}</span></span>
          )}
        </div>
      )}

      {/* footer — evidence (research) / links (preference) */}
      <ClaimFooter claim={c} isPref={isPref} />
    </Card>
  );
}

function ConfirmDraft({ claim }: { claim: Claim }) {
  const [state, setState] = useState<"idle" | "busy" | "queued" | "error">("idle");
  const inflight = useRef(false); // ref-guard: stale-closure-proof double-click guard
  const confirm = useCallback(async () => {
    if (inflight.current) return;
    inflight.current = true;
    setState("busy");
    try {
      await api.postIntent({
        kind: "confirm_claim",
        payload: { claim_id: claim.id, statement: claim.statement, action: "promote_to_live" },
      });
      setState("queued"); // button is replaced by the queued banner — no re-click
    } catch {
      setState("error");
      inflight.current = false; // allow retry
    }
  }, [claim.id, claim.statement]);

  if (state === "queued") {
    return (
      <div
        className="flex items-center gap-2 rounded-lg border border-l-2 px-3 py-2 text-[11px]"
        style={{ borderColor: TONE_HEX.green + "44", borderLeftColor: TONE_HEX.green }}
      >
        <IconCheck width={13} height={13} className="text-[var(--color-green)]" />
        <span className="text-[var(--color-muted)]">
          confirmation queued — the executor will promote this draft to <span className="text-[var(--color-green)]">live</span>.
        </span>
        <Link to="/control" className="ml-auto text-[var(--color-blue)] hover:underline">queue →</Link>
      </div>
    );
  }

  return (
    <div
      className="flex items-center gap-2 rounded-lg border border-l-2 px-3 py-2"
      style={{ borderColor: TONE_HEX.blue + "33", borderLeftColor: TONE_HEX.blue }}
    >
      <span className="text-[11px] text-[var(--color-muted)]">
        Auto-drafted — review, then confirm to publish.
      </span>
      <div className="ml-auto">
        <Button tone="green" variant="soft" title="Enqueue a confirm_claim intent" onClick={confirm}>
          {state === "busy" ? "queueing…" : state === "error" ? "retry" : <><IconCheck width={13} height={13} /> confirm → live</>}
        </Button>
      </div>
    </div>
  );
}

function RunChip({ run, tone }: { run: string; tone: Tone }) {
  return (
    <Link to={`/runs/${run}`}>
      <span
        className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono text-[11px] transition-opacity hover:opacity-80"
        style={{ borderColor: TONE_HEX[tone] + "44", color: TONE_HEX[tone], backgroundColor: TONE_HEX[tone] + "12" }}
      >
        {runShort(run)}
      </span>
    </Link>
  );
}

function ClaimFooter({ claim, isPref }: { claim: Claim; isPref: boolean }) {
  const c = claim;
  const supporting = c.supporting_runs ?? [];
  const refuting = c.refuting_runs ?? [];
  const deps = c.depends_on ?? [];
  const writeups = c.links_to_writeups ?? [];

  const hasContent = supporting.length > 0 || refuting.length > 0 || deps.length > 0 || writeups.length > 0;
  if (!hasContent) {
    return (
      <div className="mt-auto border-t border-[var(--color-border)] pt-3 text-[11px] text-[var(--color-faint)]">
        {fmtDate(c.created_at)}
      </div>
    );
  }

  return (
    <div className="mt-auto flex flex-col gap-2 border-t border-[var(--color-border)] pt-3">
      {/* preference cards are simpler — usually just writeup links */}
      {!isPref && supporting.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">supports</span>
          {supporting.map((r) => <RunChip key={r} run={r} tone="green" />)}
        </div>
      )}

      {!isPref && refuting.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">refutes</span>
          {refuting.map((r) => <RunChip key={r} run={r} tone="red" />)}
        </div>
      )}

      {!isPref && deps.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">depends on</span>
          {deps.map((d) => (
            <span key={d} className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-1.5 py-0.5 font-mono text-[11px] text-[var(--color-muted)]">
              <IconLink width={10} height={10} className="text-[var(--color-faint)]" /> {d}
            </span>
          ))}
        </div>
      )}

      {writeups.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          {writeups.map((w) => (
            <span key={w} className="inline-flex max-w-full items-center gap-1 truncate text-[11px] text-[var(--color-faint)]" title={w}>
              <IconDoc width={11} height={11} /> <span className="truncate">{w.split("/").pop()}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
