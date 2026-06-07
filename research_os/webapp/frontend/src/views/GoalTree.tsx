import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../lib/store";
import { useTerminals } from "../terminal/TerminalsProvider";
import type { GoalTreeNode } from "../lib/types";
import { Card, Pill, Dot, SectionTitle, Spinner, Meter, MetricChip, Button, Empty } from "../ui/kit";
import {
  GOAL_STATE, NODE_KIND, shortId, runShort, fmtDate, fmtWall, TONE_HEX, type Tone,
} from "../lib/ui";
import {
  IconTree, IconChevron, IconRuns, IconPipelines, IconDoc, IconBolt, IconArrowRight, IconLink,
} from "../ui/icons";

// The explorable goal tree (PLAN §2 "explore the tree for ideas"). Two panes:
// a collapsible indented thesis→chapter→branch→question tree on the left, and a
// calm detail card for the selected node on the right. Everything reads live from
// snap.goal_tree; selection + expansion are local UI state only.

export function GoalTree() {
  const { snap, loading } = useStore();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // expansion overrides: id -> open?; absent means "use the default for this depth/kind"
  const [overrides, setOverrides] = useState<Record<string, boolean>>({});

  // flat index of every node by id — drives the detail pane + default selection.
  const index = useMemo(() => {
    const m: Record<string, GoalTreeNode> = {};
    if (snap?.goal_tree) {
      const walk = (n: GoalTreeNode) => {
        m[n.id] = n;
        n.children.forEach(walk);
      };
      walk(snap.goal_tree);
    }
    return m;
  }, [snap?.goal_tree]);

  if (loading || !snap) return <div className="pt-10"><Spinner label="Indexing the store…" /></div>;
  if (!snap.goal_tree) return <Empty>No goal tree in the store yet.</Empty>;

  const root = snap.goal_tree;
  const selected: GoalTreeNode = (selectedId && index[selectedId]) || root;

  const isExpanded = (n: GoalTreeNode, depth: number): boolean => {
    if (n.id in overrides) return overrides[n.id];
    // defaults: trunk + chapters expanded; branches/questions collapsed.
    if (n.is_trunk_artifact || n.node_kind === "thesis") return true;
    return depth <= 1;
  };

  const toggle = (n: GoalTreeNode, depth: number) =>
    setOverrides((o) => ({ ...o, [n.id]: !isExpanded(n, depth) }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        {/* LEFT — the tree */}
        <Card className="p-5 lg:col-span-3">
          <SectionTitle
            icon={<IconTree width={15} height={15} className="text-[var(--color-violet)]" />}
            count={snap.counts.goals}
            right={
              <span className="hidden text-[11px] text-[var(--color-faint)] sm:inline">
                thesis · chapter · branch · question
              </span>
            }
          >
            Goal tree
          </SectionTitle>
          <div className="space-y-0.5">
            <TreeRow
              node={root}
              depth={0}
              selectedId={selected.id}
              isExpanded={isExpanded}
              onSelect={setSelectedId}
              onToggle={toggle}
            />
          </div>
        </Card>

        {/* RIGHT — detail of the selected node */}
        <div className="lg:col-span-2">
          <Detail node={selected} index={index} />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Recursive tree row. Indentation guides are left-border lines per depth.
// ---------------------------------------------------------------------------

function TreeRow({
  node, depth, selectedId, isExpanded, onSelect, onToggle,
}: {
  node: GoalTreeNode;
  depth: number;
  selectedId: string;
  isExpanded: (n: GoalTreeNode, depth: number) => boolean;
  onSelect: (id: string) => void;
  onToggle: (n: GoalTreeNode, depth: number) => void;
}) {
  const state = GOAL_STATE[node.state] ?? { tone: "muted" as Tone, label: node.state, glyph: "" };
  const kind = NODE_KIND[node.node_kind] ?? { label: node.node_kind, tone: "muted" as Tone };
  const open = isExpanded(node, depth);
  const hasChildren = node.children.length > 0;
  const selected = node.id === selectedId;
  const isTrunk = node.is_trunk_artifact || node.node_kind === "thesis";

  return (
    <div>
      <div
        onClick={() => {
          onSelect(node.id);
          if (hasChildren) onToggle(node, depth);
        }}
        className={`group flex cursor-pointer items-center gap-2 rounded-lg px-2 py-2 transition-colors ${
          selected ? "bg-[var(--color-elev)]" : "hover:bg-[var(--color-elev)]/50"
        }`}
        style={selected ? { boxShadow: `inset 2px 0 0 ${TONE_HEX[state.tone]}` } : undefined}
      >
        {/* chevron — only when there are children, else a spacer for alignment */}
        <span className="flex h-4 w-4 shrink-0 items-center justify-center text-[var(--color-faint)]">
          {hasChildren ? (
            <IconChevron
              width={13}
              height={13}
              className={`transition-transform ${open ? "rotate-90" : ""}`}
            />
          ) : (
            <span className="h-1 w-1 rounded-full bg-[var(--color-border-2)]" />
          )}
        </span>

        <Dot tone={state.tone} size={isTrunk ? 9 : 7} />

        <Pill tone={kind.tone} className="!px-1.5 !py-0 !text-[10px]">
          {kind.label}
        </Pill>

        <span
          className={`min-w-0 flex-1 truncate ${
            isTrunk
              ? "text-[14px] font-semibold text-[var(--color-fg)]"
              : selected
                ? "text-[13px] font-medium text-[var(--color-fg)]"
                : "text-[13px] text-[var(--color-fg-dim)]"
          }`}
        >
          {node.title}
        </span>

        <span className="hidden shrink-0 font-mono text-[11px] text-[var(--color-faint)] sm:inline">
          {node.run_count_total} run{node.run_count_total === 1 ? "" : "s"}
        </span>
        <span
          className="hidden w-[78px] shrink-0 text-right text-[11px] sm:inline"
          style={{ color: TONE_HEX[state.tone] }}
        >
          {state.label}
        </span>
      </div>

      {/* children with an indentation guide line */}
      {hasChildren && open && (
        <div className="ml-[15px] border-l border-[var(--color-border)] pl-2">
          {node.children.map((c) => (
            <TreeRow
              key={c.id}
              node={c}
              depth={depth + 1}
              selectedId={selectedId}
              isExpanded={isExpanded}
              onSelect={onSelect}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Detail pane for the selected node.
// ---------------------------------------------------------------------------

function Detail({ node, index }: { node: GoalTreeNode; index: Record<string, GoalTreeNode> }) {
  const { launchSkill } = useTerminals();
  const state = GOAL_STATE[node.state] ?? { tone: "muted" as Tone, label: node.state, glyph: "" };
  const kind = NODE_KIND[node.node_kind] ?? { label: node.node_kind, tone: "muted" as Tone };
  const isTrunk = node.is_trunk_artifact || node.node_kind === "thesis";

  const budgetRuns = node.budget?.expected_runs ?? 0;
  const spentRuns = node.spent?.runs ?? node.run_count_total ?? 0;
  const budgetWall = node.budget?.expected_wall_s ?? 0;
  const spentWall = node.spent?.wall_s ?? 0;

  return (
    <Card className="p-5">
      {/* header */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--color-faint)]">
          {isTrunk ? "Trunk · " : ""}{kind.label}
        </span>
        <span className="font-mono text-[10px] text-[var(--color-faint)]">{shortId(node.id)}</span>
      </div>
      <h1
        className={`mt-1 leading-snug text-[var(--color-fg)] ${
          isTrunk ? "text-[18px] font-semibold" : "text-[16px] font-semibold"
        }`}
      >
        {node.title}
      </h1>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Pill tone={kind.tone}>{kind.label}</Pill>
        <Pill tone={state.tone}>
          {state.glyph} {state.label}
        </Pill>
        <MetricChip k="runs" v={node.run_count_total} />
        {node.run_count_direct !== node.run_count_total && (
          <MetricChip k="direct" v={node.run_count_direct} />
        )}
      </div>

      {/* last measured — bordered box mirroring Overview's trunk card */}
      {node.last_measured?.summary && (
        <div className="mt-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/50 px-3 py-2.5">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-wider text-[var(--color-faint)]">
            last measured
            {node.last_measured.run && (
              <Link to={`/runs/${node.last_measured.run}`} className="font-mono text-[var(--color-blue)] hover:underline">
                {runShort(node.last_measured.run)}
              </Link>
            )}
            <span>{fmtDate(node.last_measured.at)}</span>
          </div>
          <div className="text-[12px] leading-relaxed text-[var(--color-muted)]">
            {node.last_measured.summary}
          </div>
        </div>
      )}

      {/* budget vs spent */}
      {budgetRuns > 0 && (
        <div className="mt-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]/40 px-3 py-3">
          <div className="mb-2 flex items-center justify-between text-[11px]">
            <span className="uppercase tracking-wider text-[var(--color-faint)]">budget</span>
            <span className="font-mono text-[var(--color-fg-dim)]">
              {spentRuns}/{budgetRuns} runs
            </span>
          </div>
          <Meter value={spentRuns} max={budgetRuns} tone={spentRuns > budgetRuns ? "red" : "blue"} />
          {budgetWall > 0 && (
            <div className="mt-2 flex items-center justify-between text-[11px] text-[var(--color-faint)]">
              <span>wall</span>
              <span className="font-mono">
                {fmtWall(spentWall)} / {fmtWall(budgetWall)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* direct runs */}
      <DetailList
        icon={<IconRuns width={14} height={14} className="text-[var(--color-blue)]" />}
        label="direct runs"
        count={node.direct_runs.length}
      >
        {node.direct_runs.length === 0 ? (
          <EmptyRow>no runs attached directly</EmptyRow>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {node.direct_runs.map((r) => (
              <Link key={r} to={`/runs/${r}`}>
                <span className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-1 font-mono text-[11px] text-[var(--color-fg-dim)] hover:border-[var(--color-border-2)] hover:text-[var(--color-blue)]">
                  {runShort(r)}
                </span>
              </Link>
            ))}
          </div>
        )}
      </DetailList>

      {/* pipelines */}
      <DetailList
        icon={<IconPipelines width={14} height={14} className="text-[var(--color-teal)]" />}
        label="pipelines"
        count={node.pipelines.length}
      >
        {node.pipelines.length === 0 ? (
          <EmptyRow>no pipelines serve this node</EmptyRow>
        ) : (
          <div className="space-y-1">
            {node.pipelines.map((p) => {
              const meta = index[p];
              return (
                <Link
                  key={p}
                  to="/pipelines"
                  className="flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-[var(--color-elev)]/50"
                >
                  <IconLink width={12} height={12} className="text-[var(--color-faint)]" />
                  <span className="min-w-0 flex-1 truncate text-[12px] text-[var(--color-fg-dim)]">
                    {meta?.title ?? shortId(p)}
                  </span>
                  <span className="font-mono text-[11px] text-[var(--color-faint)]">{shortId(p)}</span>
                </Link>
              );
            })}
          </div>
        )}
      </DetailList>

      {/* contract refs */}
      <DetailList
        icon={<IconDoc width={14} height={14} className="text-[var(--color-amber)]" />}
        label="contracts"
        count={node.contract_refs.length}
      >
        {node.contract_refs.length === 0 ? (
          <EmptyRow>no contracts on this node</EmptyRow>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {node.contract_refs.map((c) => (
              <span
                key={c}
                className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-1 font-mono text-[11px] text-[var(--color-muted)]"
              >
                <IconDoc width={11} height={11} className="text-[var(--color-faint)]" />
                {shortId(c)}
              </span>
            ))}
          </div>
        )}
      </DetailList>

      {/* deep-link actions */}
      <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-[var(--color-border)] pt-4">
        {node.state === "open" && (
          <Button tone="amber" variant="soft" onClick={() => launchSkill("align", node.id)}>
            <IconBolt width={13} height={13} /> align this branch
          </Button>
        )}
        <Button tone="violet" variant="soft" onClick={() => launchSkill("strategize")}>
          <IconArrowRight width={13} height={13} /> strategize
        </Button>
      </div>
    </Card>
  );
}

function DetailList({
  icon, label, count, children,
}: {
  icon: ReactNode;
  label: string;
  count: number;
  children: ReactNode;
}) {
  return (
    <div className="mt-5">
      <div className="mb-2 flex items-center gap-2">
        {icon}
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-muted)]">
          {label}
        </span>
        <span className="rounded-md bg-[var(--color-elev)] px-1.5 py-0.5 text-[10px] text-[var(--color-faint)]">
          {count}
        </span>
      </div>
      {children}
    </div>
  );
}

function EmptyRow({ children }: { children: ReactNode }) {
  return <div className="px-2 py-1 text-[12px] text-[var(--color-faint)]">{children}</div>;
}
