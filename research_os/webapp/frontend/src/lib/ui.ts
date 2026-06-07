// Shared visual vocabulary — one place every view pulls colours, glyphs and labels
// from, so the dashboard reads consistently (a goal's "open" looks the same
// everywhere). Colours echo ro_viz.py's STATUS_COLORS.

export type Tone = "green" | "amber" | "red" | "blue" | "violet" | "teal" | "muted" | "faint" | "pink";

export const TONE_HEX: Record<Tone, string> = {
  green: "#3fb950",
  amber: "#e8a33d",
  red: "#f0594f",
  blue: "#58a6ff",
  violet: "#bc8cff",
  teal: "#39c5cf",
  pink: "#f778ba",
  muted: "#8b949e",
  faint: "#6e7681",
};

// goal / pipeline node states
export const GOAL_STATE: Record<string, { tone: Tone; label: string; glyph: string }> = {
  open: { tone: "green", label: "open", glyph: "●" },
  blocked: { tone: "amber", label: "blocked", glyph: "◐" },
  closed: { tone: "faint", label: "closed", glyph: "✓" },
  revivable: { tone: "violet", label: "revivable", glyph: "○" },
  superseded: { tone: "muted", label: "superseded", glyph: "⊘" },
};

export const RUN_STATUS: Record<string, { tone: Tone; label: string }> = {
  confirmed: { tone: "green", label: "confirmed" },
  refuted: { tone: "red", label: "refuted" },
  inconclusive: { tone: "amber", label: "inconclusive" },
};

export const CLAIM_STATUS: Record<string, { tone: Tone; label: string }> = {
  live: { tone: "green", label: "live" },
  needs_replication: { tone: "amber", label: "needs replication" },
  superseded: { tone: "muted", label: "superseded" },
  retracted: { tone: "red", label: "retracted" },
  draft: { tone: "blue", label: "draft" },
};

export const CONFIDENCE: Record<string, Tone> = { high: "green", medium: "amber", low: "faint" };

export const NODE_KIND: Record<string, { label: string; tone: Tone }> = {
  thesis: { label: "thesis", tone: "violet" },
  chapter: { label: "chapter", tone: "blue" },
  branch: { label: "branch", tone: "teal" },
  question: { label: "question", tone: "amber" },
};

export const RHO_BAND: Record<string, Tone> = { A: "green", B: "teal", C: "amber", D: "red" };

export const CONTRACT_STATUS: Record<string, Tone> = {
  draft: "blue", frozen: "green", amended: "amber", closed: "faint",
};

export function toneText(t: Tone) {
  return { color: TONE_HEX[t] };
}
export function toneBg(t: Tone, alpha = "22") {
  return { backgroundColor: TONE_HEX[t] + alpha, color: TONE_HEX[t], borderColor: TONE_HEX[t] + "55" };
}

/** strip the type prefix from an id for compact display (goal_x -> x, sNNN_y -> sNNN). */
export const shortId = (id: string) =>
  id.replace(/^(goal_|pipeline_|claim_|contract_|intent_)/, "");
export const runShort = (id: string) => (id || "").split("_")[0];

export function fmtDate(s?: string | null) {
  if (!s) return "—";
  return s.slice(0, 10);
}

export function fmtWall(sec?: number | null) {
  if (sec == null) return "—";
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${(sec / 60).toFixed(0)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}

export function timeAgo(s?: string) {
  if (!s) return "";
  const d = new Date(s.replace(" ", "T"));
  const diff = (Date.now() - d.getTime()) / 1000;
  if (isNaN(diff)) return s;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
