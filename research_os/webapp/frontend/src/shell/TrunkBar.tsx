import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import { Dot, Pill } from "../ui/kit";
import { RUN_STATUS, GOAL_STATE, shortId, runShort, type Tone } from "../lib/ui";
import { IconChevron, IconBolt } from "../ui/icons";

// The pinned thread header — ALWAYS visible (PLAN §0: "at every moment keeps the
// trunk goal and current trajectory visible so neither human nor agent drifts").
// Trunk (thesis + last measured) → current chapter → current branch/question →
// latest run. This is the structural fix for "we tangent off the main thread".

export function TrunkBar({ right }: { right?: ReactNode }) {
  const { snap } = useStore();
  const { launchSkill } = useTerminals();
  const head = snap?.head;
  const trunk = head?.trunk;
  const branch = head?.current_branch;

  return (
    <header className="hairline z-10 flex items-center gap-4 bg-[var(--color-bg-2)]/80 px-7 py-3 backdrop-blur">
      {/* TRUNK */}
      <Link to="/tree" className="group flex min-w-0 items-center gap-2.5">
        <Dot tone="violet" size={9} />
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--color-faint)]">Trunk</span>
            {trunk && <span className="font-mono text-[10px] text-[var(--color-faint)]">{shortId(trunk.id)}</span>}
          </div>
          <div className="truncate text-[13px] font-medium text-[var(--color-fg)] group-hover:text-[var(--color-blue)]" style={{ maxWidth: 360 }}>
            {trunk?.title ?? "—"}
          </div>
        </div>
      </Link>

      <IconChevron className="shrink-0 text-[var(--color-faint)]" width={16} height={16} />

      {/* CURRENT THREAD */}
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--color-faint)]">Current</span>
            {branch?.chapter && (
              <Link to="/tree" className="font-mono text-[10px] text-[var(--color-teal)] hover:underline">
                {shortId(branch.chapter)}
              </Link>
            )}
            {branch && (
              <Pill tone={(GOAL_STATE[branch.state]?.tone ?? "muted") as Tone}>{branch.state}</Pill>
            )}
          </div>
          <Link to="/frontier" className="block truncate text-[13px] font-medium text-[var(--color-fg-dim)] hover:text-[var(--color-blue)]" style={{ maxWidth: 560 }}>
            {branch?.title ?? "—"}
          </Link>
        </div>

        {branch?.latest_run && (
          <Link to={`/runs/${branch.latest_run}`} className="ml-1 hidden items-center gap-1.5 md:flex">
            <span className="font-mono text-[11px] text-[var(--color-muted)]">{runShort(branch.latest_run)}</span>
            {branch.latest_status && (
              <Pill tone={(RUN_STATUS[branch.latest_status]?.tone ?? "muted") as Tone}>
                {RUN_STATUS[branch.latest_status]?.label ?? branch.latest_status}
              </Pill>
            )}
          </Link>
        )}
      </div>

      {/* last-measured ticker */}
      {trunk?.last_measured?.summary && (
        <div className="hidden max-w-[280px] items-center gap-2 border-l border-[var(--color-border)] pl-4 lg:flex">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-wider text-[var(--color-faint)]">last measured</div>
            <div className="truncate text-[11px] text-[var(--color-muted)]" title={trunk.last_measured.summary}>
              {trunk.last_measured.summary}
            </div>
          </div>
        </div>
      )}

      <div className="flex shrink-0 items-center gap-2">
        <button
          onClick={() => launchSkill("orient")}
          title="Boot Claude Code in a terminal and run /orient"
          className="flex items-center gap-1.5 rounded-lg border border-[var(--color-border)] px-2.5 py-1.5 text-[12px] text-[var(--color-muted)] hover:border-[var(--color-amber)] hover:text-[var(--color-amber)]"
        >
          <IconBolt width={14} height={14} /> orient
        </button>
        {right}
      </div>
    </header>
  );
}
