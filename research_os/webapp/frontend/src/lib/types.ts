// Types mirror research_os/webapp/backend/indexer.py:build_snapshot(). The backend
// is the source of truth; these are a convenience view over the JSON it returns.

export type GoalState = "open" | "blocked" | "closed" | "revivable";
export type NodeKind = "thesis" | "chapter" | "branch" | "question";
export type RunStatus = "confirmed" | "refuted" | "inconclusive";
export type ClaimStatus = "draft" | "live" | "needs_replication" | "superseded" | "retracted";
export type PipelineState = "open" | "blocked" | "closed" | "revivable" | "superseded";

export interface BlastRef {
  ref: string;
  component: string;
  cited: string;
  current: string;
}

export interface LastMeasured {
  run?: string;
  summary?: string;
  at?: string;
}

export interface GoalTreeNode {
  id: string;
  node_kind: NodeKind;
  title: string;
  state: GoalState;
  is_trunk_artifact: boolean;
  budget?: { expected_runs?: number; expected_wall_s?: number } | null;
  spent?: { runs?: number; wall_s?: number } | null;
  last_measured?: LastMeasured | null;
  contract_refs: string[];
  direct_runs: string[];
  pipelines: string[];
  children: GoalTreeNode[];
  run_count_direct: number;
  run_count_total: number;
}

export interface Goal {
  id: string;
  node_kind: NodeKind;
  parent: string | null;
  title: string;
  state: GoalState;
  is_trunk_artifact?: boolean;
  budget?: any;
  spent?: any;
  last_measured?: LastMeasured | null;
  contract_refs?: string[];
}

export interface Artefact {
  path: string;
  kind: "checkpoint" | "plot" | "anim" | "data" | "report" | "log";
  caption?: string;
}

export interface Run {
  id: string;
  goal_node: string;
  goal_title?: string | null;
  question?: string;
  hypothesis?: string;
  status: RunStatus;
  run_type?: string;
  seeds?: number[];
  N?: number;
  claim_scope?: string;
  metrics?: Record<string, any>;
  artefacts?: Artefact[];
  writeup_refs?: string[];
  parents?: string[];
  tests?: string[];
  blocked_by?: string[];
  contract_ref?: { contract: string; amendment?: string } | null;
  substrate_versions?: Record<string, string>;
  gates_passed?: string[];
  gates_failed?: string[];
  oracle_clean?: boolean;
  created_at?: string;
  narrative_md?: string;
  blast: BlastRef[];
}

export interface Claim {
  id: string;
  deck: "research" | "preference";
  statement: string;
  supporting_runs?: string[];
  refuting_runs?: string[];
  trust_stamp: {
    confidence: "low" | "medium" | "high";
    gates_passed?: string[];
    oracle_clean: boolean;
    calibrated_hit_rate?: number;
  };
  scope: { N: number; kind: "N1" | "cohort" };
  depends_on?: string[];
  status: ClaimStatus;
  superseded_by?: string | null;
  supersedes?: string | null;
  links_to_writeups?: string[];
  created_at?: string;
  blast: BlastRef[];
  blast_stale: boolean;
}

// ADR-0007 §5.3: a pipeline's runnable DAG — each step composes/maps a Tool, its
// `inputs` threading prior step outputs ($steps.<id>.<port>) or shelved materials.
export interface PipelineStepDef {
  id: string;
  op: "compose" | "map" | string;
  tool: string;
  params?: Record<string, any>;
  inputs?: Record<string, any>;
}

export interface Pipeline {
  id: string;
  title: string;
  hypothesis?: string;
  state: PipelineState;
  serves: string[];
  serves_titles: Record<string, string>;
  composes?: string[];
  steps?: PipelineStepDef[];
  supersedes?: string | null;
  superseded_by?: string | null;
  last_measured?: LastMeasured | null;
  tested_by: number;
  tested_by_runs: string[];
  // ADR-0007 §5.3 run feed (added by the indexer):
  pipeline_runs?: PipelineRun[];
  pipeline_run_count?: number;
}

// ADR-0007 §5.3: one execution of one step in a pipeline_run — value-neutral, with
// the drift check (hash_ok) and the materials it shelved (artifacts_produced).
export interface PipelineStep {
  step: string;
  tool: string;
  op?: string;
  status: "ok" | "error" | string;
  tool_version?: string;
  hash_at_run?: string | null;
  hash_ok?: boolean;
  wall_s?: number | null;
  metrics?: Record<string, any>;
  error?: string | null;
  artifacts_produced?: string[];
}

// ADR-0007 §5.3: one run of a pipeline's DAG. Canonical, value-neutral (no
// hypothesis/status verdict) — the pipeline analogue of a ToolRun.
export interface PipelineRun {
  id: string;
  pipeline: string;
  input?: Record<string, any>;
  params?: Record<string, any>;
  oracle_clean?: boolean;
  reason?: string;
  commit?: string;
  source?: "cli" | "webapp" | "skill";
  created_at: string;
  status: "ok" | "error" | string;
  wall_s?: number | null;
  steps: PipelineStep[];
  error?: string | null;
  executed_at?: string;
}

// ADR-0007 Slice 1/2 — the shelf. One produced material: a labelled jar on a shelf.
// Recognised materials only (a Tool's ports.output declared it); re-feedable via
// $artifact.<id>. The card is the label; the blob lives under artifact_instances/data/.
export interface ArtifactInstance {
  id: string;
  artifact_type: string;
  cardinality: "one" | "set";
  cardinality_n?: number | null;
  produced_by: { run: string; step?: string | null };
  port: string;
  path: string;
  // 'blob' (default, may be omitted): reify wrote a gitignored copy under data/. 'reference'
  // (Slice 1.5b): the material is a tool-written file (e.g. a SPICE kernel) the card points at.
  storage?: "blob" | "reference";
  size_bytes?: number;
  sample?: Record<string, any>;
  caption?: string;
  commit?: string;
  created_at: string;
  // enriched by the indexer:
  producer_kind?: "tool_run" | "pipeline_run";
  producer_name?: string | null;
  artifact_type_term?: string;
}

export interface SubstrateVersion {
  version: string;
  hash?: string;
  changed_on?: string;
  change_reason: string;
  is_bug_fix?: boolean;
  commit?: string;
}

export interface SubstrateFunction {
  name: string;
  signature?: string;
  role: string;
  line?: number | null;
}

// ADR-0007 laboratory: a Tool's live drift verdict (the same check the run-button
// gates on) — is the entry_point binding still in sync with the code?
export interface ToolDrift {
  verdict: "clean" | "missing" | "hash_moved" | "no_entry_point";
  drift: boolean;
  runnable: boolean;
  reason: string;
  live_hash?: string | null;
}

// ADR-0007: one invocation of one Tool on one input by the run-button. Canonical,
// value-neutral bench record (no hypothesis/status) — distinct from a Run.
export interface ToolRun {
  id: string;
  tool: string;
  tool_version: string;
  hash_at_run?: string | null;
  hash_ok?: boolean;
  status: "ok" | "drift_refused" | "error";
  input?: Record<string, any>;
  params?: Record<string, any>;
  metrics?: Record<string, any>;
  artefacts?: Artefact[];
  wall_s?: number | null;
  oracle_clean?: boolean;
  error?: string | null;
  reason?: string;
  commit?: string;
  source?: "cli" | "webapp" | "skill";
  created_at: string;
  executed_at?: string;
}

export interface Substrate {
  id: string;
  name: string;
  path: string;
  interface?: string;
  entry_point?: string | null;
  ports?: { input?: string[]; output?: string[] } | null;
  default_params?: Record<string, any> | null;
  canon?: boolean;
  variant_of?: string | null;
  current_version: string;
  current_hash?: string;
  promoted_from?: string | null;
  versions: SubstrateVersion[];
  functions?: SubstrateFunction[];
  blast_runs: string[];
  blast_claims: string[];
  // ADR-0007 run-button surface (added by the indexer):
  drift?: ToolDrift;
  bench_runs?: ToolRun[];
  bench_run_count?: number;
}

// --- Machinery map (derived render artifact, render/machinery_map.json) ----
// "Visualised pseudocode": the inversion machinery broken into ordered stages,
// each holding the load-bearing functions that implement it. The STRUCTURE
// (signature/line/doc/call-edges) is AST-derived by render/derive_machinery.py;
// the MEANING (stage, role, substrate tag) is hand-curated in machinery_overlay.json;
// `drift` is the computed cross-check between the two. Backend only READS it.
export interface MachineryStage {
  key: string;
  title: string;
  purpose?: string;
  consumes?: string;
  produces?: string;
}

export interface MachineryNode {
  name: string;
  file: string;
  stage: string;
  role: string;
  signature?: string;
  doc?: string;
  line?: number | null;
  exists: boolean;
  substrate_component?: string | null;
  calls?: string[];
}

export interface MachineryEdge {
  from: string;
  to: string;
}

export interface MachineryDrift {
  missing: { name: string; file: string }[];
  uncovered: { name: string; file: string; line?: number | null }[];
}

export interface Machinery {
  generated_at?: string;
  generated_for_code?: string;
  stale?: boolean;
  stages: MachineryStage[];
  nodes: MachineryNode[];
  edges: MachineryEdge[];
  drift: MachineryDrift;
}

export interface GlossaryTerm {
  id: string;
  term: string;
  definition: string;
  status: "provisional" | "canon";
  coined_in?: string;
  promoted_on?: string;
  synonyms_blocked?: string[];
  aliases?: string[];
  related_terms?: string[];
}

export interface Contract {
  id: string;
  goal_node: string;
  status: "draft" | "frozen" | "amended" | "closed";
  question: string;
  hypothesis: string;
  predictions: { case: string; expected_outcome: string; uncertain?: boolean }[];
  confirm_criteria: string;
  refute_criteria: string;
  budget: { expected_runs: number; expected_wall_s: number };
  required_artefacts?: string[];
  model?: string;
  reasoning?: string;
  analytical_probe?: { cheap_path: string; why_insufficient: string; probe_run?: string | null };
  amendments?: any[];
  frozen_at?: string;
}

export interface HeadFrontierRow {
  id: string;
  state: GoalState;
  title: string;
  last_run: string | null;
  last_at: string | null;
  chapter: string | null;
}

export interface Head {
  store_counts: Record<string, number>;
  newest_run?: { id: string; at: string } | null;
  trunk?: { id: string; state: string; title: string; last_measured?: LastMeasured } | null;
  current_branch?: {
    id: string; state: string; title: string; chapter?: string;
    latest_run: string; latest_status?: string; spent_runs?: number;
  } | null;
  frontier: HeadFrontierRow[];
  frontier_total: number;
  pipelines: { id: string; state: string; title: string; serves: string[]; tested_by: number; summary?: string }[];
  trust: {
    status_counts: Record<string, number>;
    needs_replication: { id: string; depends_on: string[]; stale: string[] }[];
    substrate_heads: Record<string, string>;
  };
  recent: { id: string; status: string; goal: string }[];
}

export interface FrontierRankItem {
  goal_id: string;
  rank: number;
  rationale?: string;
  action?: string;
  cost?: string;
  crux?: string;
}

export interface FrontierRanking {
  lens?: string;
  generated_at?: string;
  generated_for_rev?: string;
  stale?: boolean;
  items: FrontierRankItem[];
}

export interface Snapshot {
  rev: string;
  head: Head;
  frontier_ranking?: FrontierRanking | null;
  machinery?: Machinery | null;
  counts: Record<string, number>;
  goal_tree: GoalTreeNode;
  goals: Goal[];
  runs: Run[];
  claims: Claim[];
  pipelines: Pipeline[];
  substrate: Substrate[];
  tool_runs: ToolRun[];
  pipeline_runs: PipelineRun[];
  artifact_instances: ArtifactInstance[];
  glossary: GlossaryTerm[];
  contracts: Contract[];
  trust: {
    status_counts: Record<string, number>;
    live_blast_claims: string[];
    substrate_heads: Record<string, string>;
  };
}

export interface StreamItem {
  file: string;
  caption: string;
  run: string;
  ts: string;
}

export interface IntentRec {
  id: string;
  kind: string;
  payload: any;
  source: string;
  status: string;
  created_at: string;
}
