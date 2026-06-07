import { useCallback, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import { api } from "../lib/api";
import type { Goal, Run, FrontierRankItem } from "../lib/types";
import { Card, Pill, Dot, SectionTitle, Spinner, MetricChip, Button, Empty } from "../ui/kit";
import {
  GOAL_STATE, NODE_KIND, shortId, runShort, fmtDate, timeAgo, TONE_HEX, type Tone,
} from "../lib/ui";
import {
  IconFrontier, IconBolt, IconArrowRight, IconRuns, IconTree, IconWarn,
} from "../ui/icons";

// Frontier — "what's next / promising". The open/revivable branch+question nodes,
// rendered as rich pick-cards. This is UNRANKED (recency / structure only) — the
// /strategize verb is what produces a leverage-ranked list. Matches the live-head
// "Frontier · unranked" framing in Overview.

type Lens = "ranked" | "recency" | "chapter";

interface FrontierItem {
  goal: Goal;
  chapter: Goal | null;
  lastRun: Run | null;
  runCount: number;
}

export function Frontier() {
  const { snap, loading } = useStore();
  const { launchSkill } = useTerminals();
  // null = auto: prefer the ranked lens when a /strategize ranking exists.
  const [lensChoice, setLensChoice] = useState<Lens | null>(null);

  const items = useMemo<FrontierItem[]>(() => {
    if (!snap) return [];
    const byId = new Map<string, Goal>(snap.goals.map((g) => [g.id, g]));

    // nearest enclosing chapter by walking parent links
    const chapterOf = (g: Goal): Goal | null => {
      let cur: Goal | null = g.parent ? byId.get(g.parent) ?? null : null;
      const seen = new Set<string>();
      while (cur && !seen.has(cur.id)) {
        if (cur.node_kind === "chapter") return cur;
        seen.add(cur.id);
        cur = cur.parent ? byId.get(cur.parent) ?? null : null;
      }
      return null;
    };

    const open = snap.goals.filter(
      (g) =>
        (g.node_kind === "branch" || g.node_kind === "question") &&
        (g.state === "open" || g.state === "revivable"),
    );

    return open.map((goal) => {
      const runs = snap.runs.filter((r) => r.goal_node === goal.id); // already newest-first
      return {
        goal,
        chapter: chapterOf(goal),
        lastRun: runs[0] ?? null,
        runCount: runs.length,
      };
    });
  }, [snap]);

  if (loading || !snap) return <div className="pt-10"><Spinner label="Reading the frontier…" /></div>;

  const ranking = snap.frontier_ranking ?? null;
  const rankMap = new Map<string, FrontierRankItem>((ranking?.items ?? []).map((r) => [r.goal_id, r]));
  // auto-prefer the ranked lens once a /strategize pass has produced one
  const lens: Lens = lensChoice ?? (ranking ? "ranked" : "recency");

  // recency sort: newest last-run first, nulls last
  const byRecency = (a: FrontierItem, b: FrontierItem) => {
    const ta = a.lastRun?.created_at ?? "";
    const tb = b.lastRun?.created_at ?? "";
    if (ta && tb) return tb.localeCompare(ta);
    if (ta) return -1;
    if (tb) return 1;
    return a.goal.title.localeCompare(b.goal.title);
  };

  const sorted = [...items].sort(byRecency);
  // ranked: strategize order first (by rank), un-ranked items fall to the end by recency
  const rankedSorted = [...items].sort((a, b) => {
    const ra = rankMap.get(a.goal.id)?.rank ?? Infinity;
    const rb = rankMap.get(b.goal.id)?.rank ?? Infinity;
    if (ra !== rb) return ra - rb;
    return byRecency(a, b);
  });
  const revivableCount = items.filter((i) => i.goal.state === "revivable").length;

  // group under enclosing chapter for the "by chapter" lens
  const groups = (() => {
    const m = new Map<string, { chapter: Goal | null; rows: FrontierItem[] }>();
    for (const it of sorted) {
      const key = it.chapter?.id ?? "__none__";
      if (!m.has(key)) m.set(key, { chapter: it.chapter, rows: [] });
      m.get(key)!.rows.push(it);
    }
    return Array.from(m.values()).sort((a, b) =>
      (a.chapter?.title ?? "~").localeCompare(b.chapter?.title ?? "~"),
    );
  })();

  return (
    <div className="space-y-6">
      {/* header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2.5">
            <IconFrontier width={18} height={18} className="text-[var(--color-amber)]" />
            <h1 className="text-[17px] font-semibold text-[var(--color-fg)]">Frontier</h1>
            <span className="rounded-md bg-[var(--color-elev)] px-1.5 py-0.5 text-[11px] text-[var(--color-faint)]">
              {items.length}
            </span>
          </div>
          <p className="mt-1.5 max-w-2xl text-[12px] leading-relaxed text-[var(--color-muted)]">
            Open and revivable branches &amp; questions — what&apos;s next.{" "}
            {ranking ? (
              <>
                Ranked by <span className="text-[var(--color-fg-dim)]">{ranking.lens ?? "strategize"}</span>
                {ranking.generated_at && <span className="text-[var(--color-faint)]"> · {timeAgo(ranking.generated_at) || fmtDate(ranking.generated_at)}</span>}.{" "}
                <button onClick={() => launchSkill("strategize")} className="text-[var(--color-blue)] hover:underline">
                  Re-rank →
                </button>
              </>
            ) : (
              <>
                <span className="text-[var(--color-fg-dim)]">Unranked</span>: recency / structure only, no{" "}
                <span className="font-mono text-[var(--color-blue)]">/strategize</span> pass yet.{" "}
                <button onClick={() => launchSkill("strategize")} className="text-[var(--color-blue)] hover:underline">
                  Rank it →
                </button>
              </>
            )}
          </p>
        </div>
        <LensSelector lens={lens} hasRanking={!!ranking} onChange={setLensChoice} />
      </div>

      {/* ranked-but-stale warning: the store moved since this ranking was produced */}
      {ranking?.stale && (
        <Card className="flex items-center gap-3 border-l-2 px-4 py-2.5" style={{ borderLeftColor: TONE_HEX.amber }}>
          <IconWarn width={15} height={15} className="text-[var(--color-amber)]" />
          <div className="flex-1 text-[12px] text-[var(--color-muted)]">
            This ranking predates the current store state — a run or claim has landed since.{" "}
            <button onClick={() => launchSkill("strategize")} className="text-[var(--color-blue)] hover:underline">re-rank</button> to refresh.
          </div>
        </Card>
      )}

      {/* revivable hint */}
      {revivableCount > 0 && (
        <Card
          className="flex items-center gap-3 border-l-2 px-4 py-2.5"
          style={{ borderLeftColor: TONE_HEX.violet }}
        >
          <Dot tone="violet" size={8} />
          <div className="text-[12px] text-[var(--color-muted)]">
            <span className="font-medium text-[var(--color-fg-dim)]">{revivableCount} revivable</span>{" "}
            {revivableCount === 1 ? "node" : "nodes"} — the blocker may no longer hold. Re-check before picking.
          </div>
        </Card>
      )}

      {items.length === 0 ? (
        <Empty>Frontier is clear — no open or revivable branches.</Empty>
      ) : lens === "ranked" ? (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {rankedSorted.map((it) => (
            <FrontierCard key={it.goal.id} item={it} launchSkill={launchSkill} rank={rankMap.get(it.goal.id)} />
          ))}
        </div>
      ) : lens === "recency" ? (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {sorted.map((it) => (
            <FrontierCard key={it.goal.id} item={it} launchSkill={launchSkill} />
          ))}
        </div>
      ) : (
        <div className="space-y-6">
          {groups.map((grp) => (
            <div key={grp.chapter?.id ?? "__none__"}>
              <SectionTitle
                icon={<IconTree width={14} height={14} className="text-[var(--color-blue)]" />}
                count={grp.rows.length}
                right={
                  grp.chapter && (
                    <Link to="/tree" className="font-mono text-[11px] text-[var(--color-faint)] hover:text-[var(--color-blue)]">
                      {shortId(grp.chapter.id)}
                    </Link>
                  )
                }
              >
                {grp.chapter ? grp.chapter.title : "No chapter"}
              </SectionTitle>
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                {grp.rows.map((it) => (
                  <FrontierCard key={it.goal.id} item={it} launchSkill={launchSkill} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Human gate #1 (PLAN §8 autonomy map): pick a frontier item. Records the choice
// as a pick_frontier intent so it's durable + visible in the control plane — the
// backend only writes the queue, never the goal node.
function PinNext({ goal }: { goal: Goal }) {
  const [picked, setPicked] = useState(false);
  const pick = useCallback(async () => {
    try {
      await api.postIntent({ kind: "pick_frontier", payload: { goal_id: goal.id, title: goal.title } });
      setPicked(true);
    } catch { /* leave button as-is */ }
  }, [goal.id, goal.title]);
  return (
    <Button
      variant="ghost"
      tone="violet"
      title="Pin this as the chosen next action (queues a pick_frontier intent)"
      onClick={pick}
    >
      {picked ? "pinned ✓" : "pin next"}
    </Button>
  );
}

function LensSelector({ lens, hasRanking, onChange }: { lens: Lens; hasRanking: boolean; onChange: (l: Lens) => void }) {
  const opts: { k: Lens; label: string }[] = [
    ...(hasRanking ? [{ k: "ranked" as Lens, label: "ranked" }] : []),
    { k: "recency", label: "recency" },
    { k: "chapter", label: "by chapter" },
  ];
  return (
    <div className="inline-flex shrink-0 items-center gap-0.5 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-2)] p-0.5">
      {opts.map((o) => {
        const active = lens === o.k;
        return (
          <button
            key={o.k}
            onClick={() => onChange(o.k)}
            className={`rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
              active
                ? "bg-[var(--color-elev)] text-[var(--color-fg)]"
                : "text-[var(--color-faint)] hover:text-[var(--color-fg-dim)]"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function FrontierCard({
  item, launchSkill, rank,
}: {
  item: FrontierItem;
  launchSkill: (name: string, arg?: string) => Promise<string>;
  rank?: FrontierRankItem;
}) {
  const { goal, chapter, lastRun, runCount } = item;
  const stateMeta = GOAL_STATE[goal.state];
  const kindMeta = NODE_KIND[goal.node_kind];
  const stateTone = (stateMeta?.tone ?? "muted") as Tone;
  const revivable = goal.state === "revivable";

  return (
    <Card
      className={`flex flex-col p-4 ${revivable ? "border-l-2" : ""}`}
      style={revivable ? { borderLeftColor: TONE_HEX.violet } : undefined}
    >
      {/* status row */}
      <div className="mb-2.5 flex items-center gap-2">
        {rank && (
          <span
            className="flex h-5 min-w-[20px] items-center justify-center rounded-md px-1 text-[11px] font-bold"
            style={{ backgroundColor: TONE_HEX.amber + "22", color: TONE_HEX.amber }}
            title="strategize rank"
          >
            #{rank.rank}
          </span>
        )}
        <Pill tone={stateTone}>
          {stateMeta?.glyph} {goal.state}
        </Pill>
        {kindMeta && (
          <Pill tone={kindMeta.tone}>{kindMeta.label}</Pill>
        )}
        {chapter && (
          <Link
            to="/tree"
            className="ml-auto font-mono text-[10px] text-[var(--color-faint)] hover:text-[var(--color-blue)]"
            title={chapter.title}
          >
            {shortId(chapter.id)}
          </Link>
        )}
      </div>

      {/* the question / title */}
      <div className="flex items-start gap-2">
        <Dot tone={stateTone} size={7} />
        <h3 className="min-w-0 flex-1 text-[14px] font-medium leading-snug text-[var(--color-fg-dim)]">
          {goal.title}
        </h3>
      </div>

      {/* strategize rationale (ranked lens) */}
      {rank && (rank.rationale || rank.crux || rank.cost) && (
        <div className="mt-2.5 space-y-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-3 py-2">
          {rank.rationale && <div className="text-[12px] leading-relaxed text-[var(--color-muted)]">{rank.rationale}</div>}
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
            {rank.crux && <span className="text-[var(--color-faint)]">crux: <span className="text-[var(--color-fg-dim)]">{rank.crux}</span></span>}
            {rank.cost && <span className="text-[var(--color-faint)]">cost: <span className="text-[var(--color-fg-dim)]">{rank.cost}</span></span>}
          </div>
        </div>
      )}

      {revivable && (
        <div className="mt-2 flex items-center gap-1.5 text-[11px] text-[var(--color-violet)]">
          <IconWarn width={12} height={12} />
          <span>blocker may no longer hold</span>
        </div>
      )}

      {/* metrics */}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {lastRun ? (
          <Link to={`/runs/${lastRun.id}`}>
            <MetricChip k="last" v={runShort(lastRun.id)} tone="blue" />
          </Link>
        ) : (
          <MetricChip k="last" v="never run" />
        )}
        <MetricChip k="runs" v={runCount} />
        <span className="inline-flex items-center gap-1 text-[11px] text-[var(--color-faint)]">
          <IconRuns width={11} height={11} />
          {fmtDate(lastRun?.created_at)}
        </span>
      </div>

      {/* pick actions */}
      <div className="mt-3.5 flex items-center gap-2 border-t border-[var(--color-border)] pt-3">
        <Button
          tone="amber"
          variant="soft"
          title="Pre-register this branch into a frozen contract"
          onClick={() => launchSkill("align", goal.id)}
        >
          <IconBolt width={13} height={13} /> align
          <IconArrowRight width={12} height={12} />
        </Button>
        <Button
          variant="ghost"
          title="Rank the whole frontier by leverage"
          onClick={() => launchSkill("strategize")}
        >
          strategize
        </Button>
        <PinNext goal={goal} />
        <Link
          to="/tree"
          className="ml-auto font-mono text-[10px] text-[var(--color-faint)] hover:text-[var(--color-fg-dim)]"
        >
          {shortId(goal.id)}
        </Link>
      </div>
    </Card>
  );
}
