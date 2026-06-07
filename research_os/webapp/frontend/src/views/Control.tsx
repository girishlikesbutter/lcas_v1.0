import { useCallback, useState } from "react";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import { api } from "../lib/api";
import type { IntentRec } from "../lib/types";
import { Card, Pill, Dot, SectionTitle, Empty, Button } from "../ui/kit";
import { TONE_HEX, timeAgo, type Tone } from "../lib/ui";
import { IconQueue, IconBolt, IconArrowRight, IconCheck } from "../ui/icons";

// The control plane (PLAN §5/§11, Q11). The dashboard owns NO research state — the
// ONE thing it writes is *intent*, into research_os/queue/, for the (Phase-2)
// headless executor to consume. This view makes that queue visible: what's been
// requested (confirm a claim, pick a frontier item, run a skill, a note), its
// status, and a composer to enqueue more. Delete the queue → lose only un-consumed
// requests. Until the executor lands, intents sit here as a durable to-do for it.

const KIND_TONE: Record<string, Tone> = {
  confirm_claim: "green",
  pick_frontier: "amber",
  bless_contract: "violet",
  run_skill: "blue",
  note: "muted",
};

const KIND_LABEL: Record<string, string> = {
  confirm_claim: "confirm claim",
  pick_frontier: "pick frontier",
  bless_contract: "bless contract",
  run_skill: "run skill",
  note: "note",
};

export function Control() {
  // the queue is fetched once in the store and refreshed on the SSE intents event,
  // so this view (and the nav badge) read a single source.
  const { intents, pendingIntents: queued } = useStore();

  return (
    <div className="space-y-6">
      {/* explainer */}
      <Card className="flex items-start gap-3 p-5">
        <IconQueue width={20} height={20} className="mt-0.5 shrink-0 text-[var(--color-blue)]" />
        <div className="min-w-0 flex-1">
          <h1 className="text-[15px] font-semibold text-[var(--color-fg)]">Control plane · intent queue</h1>
          <p className="mt-1 max-w-3xl text-[12px] leading-relaxed text-[var(--color-muted)]">
            The dashboard owns no research state. The single thing it writes is <span className="text-[var(--color-fg-dim)]">intent</span> —
            approved confirmations, branch picks, skill requests — appended to{" "}
            <span className="font-mono text-[var(--color-faint)]">research_os/queue/</span> for a headless executor to consume.
            Delete the queue and you lose only un-run requests, nothing canonical.
          </p>
        </div>
        <div className="hidden shrink-0 sm:block">
          {queued > 0
            ? <Pill tone="blue">{queued} queued</Pill>
            : <Pill tone="muted">empty</Pill>}
        </div>
      </Card>

      <Composer />

      {/* executor-not-yet-here notice */}
      <Card className="flex items-center gap-3 border-l-2 px-4 py-2.5" style={{ borderLeftColor: TONE_HEX.amber }}>
        <Dot tone="amber" size={8} />
        <div className="text-[12px] text-[var(--color-muted)]">
          No headless executor is consuming this queue yet (Phase 2). Intents persist here as a durable
          work-list; for now, run the same skill in a terminal to act on one immediately.
        </div>
      </Card>

      {/* the queue */}
      <SectionTitle icon={<IconQueue width={15} height={15} className="text-[var(--color-blue)]" />} count={intents.length}>
        Queued intents
      </SectionTitle>
      {intents.length === 0 ? (
        <Empty>No intents yet. Confirm a draft claim, pick a frontier item, or post one above.</Empty>
      ) : (
        <div className="space-y-2">
          {intents.map((it) => <IntentRow key={it.id} it={it} />)}
        </div>
      )}
    </div>
  );
}

function IntentRow({ it }: { it: IntentRec }) {
  const tone = KIND_TONE[it.kind] ?? "muted";
  const payload = it.payload && Object.keys(it.payload).length > 0
    ? JSON.stringify(it.payload)
    : null;
  return (
    <Card className="flex items-start gap-3 p-3.5">
      <Pill tone={tone}>{KIND_LABEL[it.kind] ?? it.kind}</Pill>
      <div className="min-w-0 flex-1">
        {payload && (
          <div className="truncate font-mono text-[11.5px] text-[var(--color-fg-dim)]" title={payload}>{payload}</div>
        )}
        <div className="mt-0.5 flex items-center gap-2 text-[11px] text-[var(--color-faint)]">
          <span className="font-mono">{it.id}</span>
          <span>· {it.source}</span>
          <span>· {timeAgo(it.created_at) || it.created_at}</span>
        </div>
      </div>
      <Pill tone={it.status === "queued" ? "blue" : it.status === "done" ? "green" : "muted"}>{it.status}</Pill>
    </Card>
  );
}

// A minimal composer for the two free-form intent kinds. confirm_claim /
// pick_frontier / bless_contract are posted from their own objects (Claims,
// Frontier) where the id is in hand; here you can drop a note for the executor or
// queue a skill run.
function Composer() {
  const { launchSkill } = useTerminals();
  const [note, setNote] = useState("");
  const [skill, setSkill] = useState("strategize");
  const [busy, setBusy] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);

  // The store refreshes the queue on the SSE `intents` event the POST triggers, so
  // we just fire-and-flash here.
  const post = useCallback(async (body: any, msg: string) => {
    setBusy(true);
    try {
      await api.postIntent(body);
      setFlash(msg);
      setTimeout(() => setFlash(null), 2500);
    } catch {
      setFlash("failed to enqueue");
      setTimeout(() => setFlash(null), 2500);
    } finally {
      setBusy(false);
    }
  }, []);

  const SKILLS = ["orient", "strategize", "align", "execute", "close", "glossary", "dynamic-viz"];

  return (
    <Card className="p-4">
      <SectionTitle icon={<IconBolt width={15} height={15} className="text-[var(--color-amber)]" />}>
        Enqueue an intent
      </SectionTitle>
      <div className="grid gap-3 lg:grid-cols-2">
        {/* note */}
        <div className="flex flex-col gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 p-3">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">note → executor</span>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={2}
            placeholder="a freeform instruction for the headless executor…"
            className="w-full resize-none rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] px-2.5 py-2 text-[12px] text-[var(--color-fg)] placeholder:text-[var(--color-faint)] focus:border-[var(--color-blue)] focus:outline-none"
          />
          <div className="flex justify-end">
            <Button
              tone="blue"
              variant="soft"
              title="Append a note intent to the queue"
              onClick={() => { if (note.trim()) { post({ kind: "note", payload: { text: note.trim() } }, "note queued"); setNote(""); } }}
            >
              <IconArrowRight width={13} height={13} /> queue note
            </Button>
          </div>
        </div>

        {/* run skill */}
        <div className="flex flex-col gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 p-3">
          <span className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">run a skill</span>
          <select
            value={skill}
            onChange={(e) => setSkill(e.target.value)}
            className="rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] px-2.5 py-2 text-[12px] text-[var(--color-fg)] focus:border-[var(--color-blue)] focus:outline-none"
          >
            {SKILLS.map((s) => <option key={s} value={s}>/{s}</option>)}
          </select>
          <div className="flex justify-end gap-2">
            <Button
              variant="ghost"
              title="Run it now in a terminal (interactive)"
              onClick={() => launchSkill(skill)}
            >
              run in terminal
            </Button>
            <Button
              tone="blue"
              variant="soft"
              title="Queue it for the headless executor (async)"
              onClick={() => post({ kind: "run_skill", payload: { skill } }, "skill queued")}
            >
              <IconArrowRight width={13} height={13} /> queue
            </Button>
          </div>
        </div>
      </div>
      {(busy || flash) && (
        <div className="mt-2 flex items-center gap-1.5 text-[11px] text-[var(--color-green)]">
          <IconCheck width={13} height={13} /> {flash ?? "…"}
        </div>
      )}
    </Card>
  );
}
