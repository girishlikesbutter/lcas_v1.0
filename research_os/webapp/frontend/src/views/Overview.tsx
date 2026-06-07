import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import { api, streamUrl } from "../lib/api";
import type { StreamItem } from "../lib/types";
import { Card, Pill, Dot, SectionTitle, Stat, Spinner, MetricChip, Button } from "../ui/kit";
import {
  GOAL_STATE, RUN_STATUS, CLAIM_STATUS, NODE_KIND, RHO_BAND, shortId, runShort,
  fmtDate, TONE_HEX, type Tone,
} from "../lib/ui";
import {
  IconBlast, IconBolt, IconArrowRight, IconStream, IconFrontier, IconPipelines, IconClaims,
} from "../ui/icons";

// The landing glance (PLAN §2 "Live head"). Replaces "read the first 80 lines of
// PROGRESS". Everything a session needs to orient — trunk, current thread, frontier,
// what-we've-tried, trust state, recent runs, latest plots — above the fold.

export function Overview() {
  const { snap, loading } = useStore();
  const { launchSkill } = useTerminals();
  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;
  const h = snap.head;
  const blast = snap.trust.live_blast_claims;
  const ranking = snap.frontier_ranking ?? null;
  const rankOf = new Map<string, number>((ranking?.items ?? []).map((r) => [r.goal_id, r.rank]));
  const frontierPreview = ranking
    ? [...h.frontier].sort((a, b) => (rankOf.get(a.id) ?? Infinity) - (rankOf.get(b.id) ?? Infinity)).slice(0, 6)
    : h.frontier.slice(0, 6);

  return (
    <div className="space-y-6">
      {/* hero stats */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-6">
        <HeroStat to="/runs" value={snap.counts.runs} label="runs" tone="blue" />
        <HeroStat to="/pipelines" value={snap.counts.pipelines} label="pipelines" tone="teal" />
        <HeroStat to="/claims" value={snap.counts.claims} label="claims" tone="green" />
        <HeroStat to="/tree" value={snap.counts.goals} label="goal nodes" tone="violet" />
        <HeroStat to="/glossary" value={snap.counts.glossary} label="glossary" tone="amber" />
        <HeroStat to="/substrate" value={snap.counts.substrate} label="substrate" tone="pink" />
      </div>

      {/* blast-radius alert — the s001–s066 line */}
      {blast.length > 0 && (
        <Link to="/substrate">
          <Card hover className="flex items-center gap-3 border-l-2 px-4 py-3" style={{ borderLeftColor: TONE_HEX.amber }}>
            <IconBlast className="text-[var(--color-amber)]" width={18} height={18} />
            <div className="flex-1 text-[13px]">
              <span className="font-medium text-[var(--color-fg)]">{blast.length} claim{blast.length > 1 ? "s" : ""}</span>
              <span className="text-[var(--color-muted)]"> rest on superseded substrate — blast radius active.</span>
            </div>
            <span className="font-mono text-[11px] text-[var(--color-amber)]">{blast.map(shortId).join(" · ")}</span>
            <IconArrowRight width={15} height={15} className="text-[var(--color-faint)]" />
          </Card>
        </Link>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* LEFT 2/3 — the thread */}
        <div className="space-y-6 lg:col-span-2">
          {/* trunk */}
          {h.trunk && (
            <Card className="p-5">
              <div className="flex items-start gap-3">
                <Dot tone="violet" size={10} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--color-faint)]">Trunk · thesis</span>
                    <span className="font-mono text-[10px] text-[var(--color-faint)]">{shortId(h.trunk.id)}</span>
                  </div>
                  <h1 className="mt-1 text-[17px] font-semibold leading-snug text-[var(--color-fg)]">{h.trunk.title}</h1>
                  {h.trunk.last_measured?.summary && (
                    <div className="mt-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/50 px-3 py-2">
                      <div className="mb-1 flex items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
                        last measured
                        {h.trunk.last_measured.run && <span className="font-mono text-[var(--color-blue)]">{runShort(h.trunk.last_measured.run)}</span>}
                        <span>{fmtDate(h.trunk.last_measured.at)}</span>
                      </div>
                      <div className="text-[12px] leading-relaxed text-[var(--color-muted)]">{h.trunk.last_measured.summary}</div>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          )}

          {/* current thread */}
          {h.current_branch && (
            <Card className="p-5">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--color-faint)]">Current thread</span>
                  {h.current_branch.chapter && <span className="font-mono text-[10px] text-[var(--color-teal)]">{shortId(h.current_branch.chapter)}</span>}
                </div>
                <Button tone="amber" variant="soft" onClick={() => launchSkill("align", h.current_branch!.id)}>
                  <IconBolt width={13} height={13} /> align this branch
                </Button>
              </div>
              <div className="flex items-start gap-3">
                <Pill tone={(GOAL_STATE[h.current_branch.state]?.tone ?? "muted") as Tone}>
                  {GOAL_STATE[h.current_branch.state]?.glyph} {h.current_branch.state}
                </Pill>
                <div className="min-w-0 flex-1">
                  <div className="text-[15px] font-medium leading-snug text-[var(--color-fg-dim)]">{h.current_branch.title}</div>
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    {h.current_branch.latest_run && (
                      <Link to={`/runs/${h.current_branch.latest_run}`}>
                        <MetricChip k="latest" v={runShort(h.current_branch.latest_run)} />
                      </Link>
                    )}
                    {h.current_branch.latest_status && (
                      <Pill tone={(RUN_STATUS[h.current_branch.latest_status]?.tone ?? "muted") as Tone}>
                        {RUN_STATUS[h.current_branch.latest_status]?.label}
                      </Pill>
                    )}
                    <MetricChip k="spent" v={`${h.current_branch.spent_runs ?? 0} runs`} />
                  </div>
                </div>
              </div>
            </Card>
          )}

          {/* frontier — ranked if /strategize has produced an ordering */}
          <Card className="p-5">
            <SectionTitle icon={<IconFrontier width={15} height={15} className="text-[var(--color-amber)]" />} count={h.frontier_total} right={<Link to="/frontier" className="text-[11px] text-[var(--color-blue)] hover:underline">{ranking ? "frontier →" : "strategize →"}</Link>}>
              {ranking ? `Frontier · ranked${ranking.lens ? ` · ${ranking.lens}` : ""}${ranking.stale ? " (stale)" : ""}` : "Frontier · unranked"}
            </SectionTitle>
            <div className="space-y-1.5">
              {frontierPreview.map((f) => {
                const rk = rankOf.get(f.id);
                return (
                  <Link key={f.id} to="/frontier" className="flex items-center gap-3 rounded-lg px-2 py-2 hover:bg-[var(--color-elev)]/50">
                    {rk != null
                      ? <span className="w-5 shrink-0 text-center text-[11px] font-bold" style={{ color: TONE_HEX.amber }}>#{rk}</span>
                      : <Dot tone={(GOAL_STATE[f.state]?.tone ?? "muted") as Tone} size={7} />}
                    <span className="min-w-0 flex-1 truncate text-[13px] text-[var(--color-fg-dim)]">{f.title}</span>
                    {f.last_run && <span className="font-mono text-[11px] text-[var(--color-faint)]">{runShort(f.last_run)}</span>}
                    <span className="hidden w-[70px] shrink-0 text-right text-[11px] text-[var(--color-faint)] sm:block">{f.last_at ?? "—"}</span>
                  </Link>
                );
              })}
            </div>
          </Card>

          {/* tried — pipelines */}
          <Card className="p-5">
            <SectionTitle icon={<IconPipelines width={15} height={15} className="text-[var(--color-teal)]" />} count={snap.pipelines.length} right={<Link to="/pipelines" className="text-[11px] text-[var(--color-blue)] hover:underline">all →</Link>}>
              Tried · how we're trying
            </SectionTitle>
            <div className="grid gap-2 sm:grid-cols-2">
              {snap.pipelines.map((p) => (
                <Link key={p.id} to="/pipelines" className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-3 py-2.5 hover:border-[var(--color-border-2)]">
                  <div className="flex items-center justify-between">
                    <span className="truncate text-[13px] text-[var(--color-fg-dim)]">{p.title}</span>
                    <Pill tone={(GOAL_STATE[p.state]?.tone ?? "muted") as Tone}>{p.state}</Pill>
                  </div>
                  <div className="mt-1 text-[11px] text-[var(--color-faint)]">{p.tested_by}× tested · serves {p.serves?.length ?? 0}</div>
                </Link>
              ))}
            </div>
          </Card>
        </div>

        {/* RIGHT 1/3 — trust + recent + plots */}
        <div className="space-y-6">
          <TrustLedger />
          <RecentRuns />
          <PlotPeek />
        </div>
      </div>
    </div>
  );
}

function HeroStat({ to, value, label, tone }: { to: string; value: number; label: string; tone: Tone }) {
  return (
    <Link to={to}>
      <Card hover className="px-4 py-3.5">
        <Stat value={value} label={label} tone={tone} />
      </Card>
    </Link>
  );
}

function TrustLedger() {
  const { snap } = useStore();
  if (!snap) return null;
  const counts = snap.trust.status_counts;
  const order: { k: string; tone: Tone }[] = [
    { k: "live", tone: "green" }, { k: "needs_replication", tone: "amber" },
    { k: "superseded", tone: "muted" }, { k: "retracted", tone: "red" }, { k: "draft", tone: "blue" },
  ];
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return (
    <Card className="p-5">
      <SectionTitle icon={<IconClaims width={15} height={15} className="text-[var(--color-green)]" />} right={<Link to="/claims" className="text-[11px] text-[var(--color-blue)] hover:underline">deck →</Link>}>
        Trust ledger
      </SectionTitle>
      <div className="mb-3 flex h-2 overflow-hidden rounded-full bg-[var(--color-elev)]">
        {order.map((o) => counts[o.k] ? <div key={o.k} style={{ width: `${(counts[o.k] / total) * 100}%`, backgroundColor: TONE_HEX[o.tone] }} /> : null)}
      </div>
      <div className="space-y-1.5">
        {order.filter((o) => counts[o.k]).map((o) => (
          <div key={o.k} className="flex items-center gap-2 text-[12px]">
            <Dot tone={o.tone} size={7} />
            <span className="flex-1 text-[var(--color-muted)]">{CLAIM_STATUS[o.k]?.label ?? o.k}</span>
            <span className="font-mono text-[var(--color-fg-dim)]">{counts[o.k]}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 border-t border-[var(--color-border)] pt-3">
        <div className="mb-1.5 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">substrate heads</div>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(snap.trust.substrate_heads).map(([k, v]) => (
            <span key={k} className="font-mono text-[11px] text-[var(--color-muted)]">{k}@<span className="text-[var(--color-fg-dim)]">{v}</span></span>
          ))}
        </div>
      </div>
    </Card>
  );
}

function RecentRuns() {
  const { snap } = useStore();
  if (!snap) return null;
  const recent = snap.runs.slice(0, 7);
  return (
    <Card className="p-5">
      <SectionTitle right={<Link to="/runs" className="text-[11px] text-[var(--color-blue)] hover:underline">all →</Link>}>Recent runs</SectionTitle>
      <div className="space-y-1">
        {recent.map((r) => {
          const band = r.metrics?.rho_band as string | undefined;
          return (
            <Link key={r.id} to={`/runs/${r.id}`} className="flex items-center gap-2.5 rounded-md px-2 py-1.5 hover:bg-[var(--color-elev)]/50">
              <Dot tone={(RUN_STATUS[r.status]?.tone ?? "muted") as Tone} size={7} />
              <span className="font-mono text-[12px] text-[var(--color-fg-dim)]">{runShort(r.id)}</span>
              <span className="min-w-0 flex-1 truncate text-[12px] text-[var(--color-faint)]">{r.goal_title ?? r.question ?? ""}</span>
              {band && <span className="rounded px-1.5 text-[10px] font-semibold" style={{ backgroundColor: TONE_HEX[RHO_BAND[band] ?? "muted"] + "22", color: TONE_HEX[RHO_BAND[band] ?? "muted"] }}>{band}</span>}
            </Link>
          );
        })}
      </div>
    </Card>
  );
}

function PlotPeek() {
  const { streamTick } = useStore();
  const [items, setItems] = useState<StreamItem[]>([]);
  const [err, setErr] = useState(false);
  useEffect(() => {
    api.streamManifest()
      .then((m) => { setItems(m.items || []); setErr(false); })
      .catch(() => setErr(true));
  }, [streamTick]);
  const latest = items.slice(-2).reverse();
  return (
    <Card className="p-5">
      <SectionTitle icon={<IconStream width={15} height={15} className="text-[var(--color-teal)]" />} count={items.length} right={<Link to="/stream" className="text-[11px] text-[var(--color-blue)] hover:underline">stream →</Link>}>
        Latest plots
      </SectionTitle>
      {err ? (
        <div className="py-6 text-center text-[12px] text-[var(--color-red)]">couldn't load the plot manifest</div>
      ) : latest.length === 0 ? (
        <div className="py-6 text-center text-[12px] text-[var(--color-faint)]">no plots yet</div>
      ) : (
        <div className="space-y-3">
          {latest.map((it) => (
            <Link key={it.file + it.ts} to="/stream" className="block overflow-hidden rounded-lg border border-[var(--color-border)] hover:border-[var(--color-border-2)]">
              <img src={streamUrl(it.file, it.ts)} className="w-full" />
              <div className="px-2.5 py-1.5 text-[11px] text-[var(--color-muted)] line-clamp-2">{it.caption}</div>
            </Link>
          ))}
        </div>
      )}
    </Card>
  );
}
