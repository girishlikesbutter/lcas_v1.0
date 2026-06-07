import type { ReactNode, CSSProperties } from "react";
import { TONE_HEX, toneBg, type Tone } from "../lib/ui";

// ---------------------------------------------------------------------------
// Primitives shared across every view. Keep these visually consistent — they are
// the dashboard's vocabulary. (The parallel-built views import only from here +
// lib/ui.ts + lib/types.ts, so the look stays cohesive.)
// ---------------------------------------------------------------------------

export function Card({
  children, className = "", hover = false, style, onClick,
}: { children: ReactNode; className?: string; hover?: boolean; style?: CSSProperties; onClick?: () => void }) {
  return (
    <div
      onClick={onClick}
      className={`card ${hover ? "card-hover cursor-pointer" : ""} ${className}`}
      style={style}
    >
      {children}
    </div>
  );
}

/** Small coloured status chip. */
export function Pill({ tone, children, soft = true, className = "" }: {
  tone: Tone; children: ReactNode; soft?: boolean; className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium leading-none ${className}`}
      style={soft ? toneBg(tone) : { backgroundColor: TONE_HEX[tone], color: "#08090c", borderColor: "transparent" }}
    >
      {children}
    </span>
  );
}

export function Dot({ tone, pulse = false, size = 8 }: { tone: Tone; pulse?: boolean; size?: number }) {
  return (
    <span
      className={`inline-block rounded-full ${pulse ? "live-dot" : ""}`}
      style={{ width: size, height: size, backgroundColor: TONE_HEX[tone], boxShadow: `0 0 8px ${TONE_HEX[tone]}66` }}
    />
  );
}

export function MonoId({ children, tone, className = "" }: { children: ReactNode; tone?: Tone; className?: string }) {
  return (
    <span className={`font-mono text-[12px] ${className}`} style={tone ? { color: TONE_HEX[tone] } : undefined}>
      {children}
    </span>
  );
}

/** A labelled section header with optional count + right-aligned action slot. */
export function SectionTitle({ children, count, right, icon }: {
  children: ReactNode; count?: number | string; right?: ReactNode; icon?: ReactNode;
}) {
  return (
    <div className="mb-3 flex items-center justify-between">
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[var(--color-muted)]">
          {children}
        </h2>
        {count != null && (
          <span className="rounded-md bg-[var(--color-elev)] px-1.5 py-0.5 text-[11px] text-[var(--color-faint)]">
            {count}
          </span>
        )}
      </div>
      {right}
    </div>
  );
}

/** Big number stat with caption. */
export function Stat({ value, label, tone, sub }: { value: ReactNode; label: string; tone?: Tone; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-[22px] font-semibold leading-none" style={tone ? { color: TONE_HEX[tone] } : { color: "var(--color-fg)" }}>
        {value}
      </div>
      <div className="mt-1.5 text-[11px] uppercase tracking-wider text-[var(--color-faint)]">{label}</div>
      {sub && <div className="mt-0.5 text-[11px] text-[var(--color-muted)]">{sub}</div>}
    </div>
  );
}

/** key→value metric chip, e.g. ρ 0.030 · band A. */
export function MetricChip({ k, v, tone }: { k: string; v: ReactNode; tone?: Tone }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-2)] px-2 py-1 text-[11px]">
      <span className="text-[var(--color-faint)]">{k}</span>
      <span className="font-mono font-medium" style={tone ? { color: TONE_HEX[tone] } : { color: "var(--color-fg-dim)" }}>
        {v}
      </span>
    </span>
  );
}

export function Empty({ children = "Nothing here yet." }: { children?: ReactNode }) {
  return (
    <div className="flex items-center justify-center rounded-xl border border-dashed border-[var(--color-border)] py-10 text-[13px] text-[var(--color-faint)]">
      {children}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-[13px] text-[var(--color-muted)]">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-[var(--color-border-2)] border-t-[var(--color-blue)]" />
      {label || "Loading…"}
    </div>
  );
}

/** A subtle horizontal meter (budget spent vs expected). */
export function Meter({ value, max, tone = "blue" }: { value: number; max: number; tone?: Tone }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  const over = max > 0 && value > max;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-elev)]">
      <div
        className="h-full rounded-full transition-all"
        style={{ width: `${pct}%`, backgroundColor: over ? TONE_HEX.red : TONE_HEX[tone] }}
      />
    </div>
  );
}

/** Inline icon button. */
export function IconButton({ children, onClick, title, active = false }: {
  children: ReactNode; onClick?: () => void; title?: string; active?: boolean;
}) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={`flex h-7 w-7 items-center justify-center rounded-md border text-[var(--color-muted)] transition-colors hover:text-[var(--color-fg)] ${
        active ? "border-[var(--color-blue)] text-[var(--color-blue)]" : "border-[var(--color-border)] hover:border-[var(--color-border-2)]"
      }`}
    >
      {children}
    </button>
  );
}

export function Button({ children, onClick, tone = "blue", variant = "soft", title }: {
  children: ReactNode; onClick?: () => void; tone?: Tone; variant?: "soft" | "ghost" | "solid"; title?: string;
}) {
  const base = "inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12px] font-medium transition-colors";
  if (variant === "solid")
    return <button title={title} onClick={onClick} className={base} style={{ backgroundColor: TONE_HEX[tone], color: "#08090c" }}>{children}</button>;
  if (variant === "ghost")
    return <button title={title} onClick={onClick} className={`${base} text-[var(--color-muted)] hover:bg-[var(--color-elev)] hover:text-[var(--color-fg)]`}>{children}</button>;
  return <button title={title} onClick={onClick} className={`${base} border`} style={toneBg(tone)}>{children}</button>;
}
