import { useEffect, useMemo, useState } from "react";
import { useStore } from "../lib/store";
import { api, streamUrl } from "../lib/api";
import type { StreamItem } from "../lib/types";
import { Card, Pill, SectionTitle, Empty, Dot } from "../ui/kit";
import { IconStream, IconClose } from "../ui/icons";
import { timeAgo, type Tone } from "../lib/ui";

// The plot stream (PLAN §8 visual-validation). Auto-surfaced, tagged by run/branch,
// newest-first — "no path-hunting", "my eyes are a validation gate". New plots pop in
// live over SSE (streamTick) with a highlight. Subsumes render/stream/index.html.

export function Stream() {
  const { streamTick } = useStore();
  const [items, setItems] = useState<StreamItem[]>([]);
  const [filter, setFilter] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<StreamItem | null>(null);
  const [freshCount, setFreshCount] = useState(0);
  const [err, setErr] = useState(false);

  useEffect(() => {
    api.streamManifest().then((m) => {
      const next = (m.items || []).slice().reverse(); // newest first
      setItems((prev) => {
        if (prev.length && next.length > prev.length) setFreshCount(next.length - prev.length);
        return next;
      });
      setErr(false);
    }).catch(() => setErr(true));
  }, [streamTick]);

  const tags = useMemo(() => {
    const s = new Map<string, number>();
    for (const it of items) if (it.run) s.set(it.run, (s.get(it.run) || 0) + 1);
    return [...s.entries()].sort((a, b) => b[1] - a[1]);
  }, [items]);

  const shown = filter ? items.filter((i) => i.run === filter) : items;

  useEffect(() => {
    const t = setTimeout(() => setFreshCount(0), 2500);
    return () => clearTimeout(t);
  }, [freshCount]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionTitle icon={<IconStream width={16} height={16} className="text-[var(--color-teal)]" />} count={`${items.length} plots`}>
          Plot stream
        </SectionTitle>
        <div className="flex items-center gap-2 text-[11px] text-[var(--color-faint)]">
          <Dot tone="teal" pulse size={6} /> live · newest first
        </div>
      </div>

      {/* run/branch filter chips */}
      {tags.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => setFilter(null)} className={`rounded-md border px-2.5 py-1 text-[11px] ${!filter ? "border-[var(--color-blue)] text-[var(--color-blue)]" : "border-[var(--color-border)] text-[var(--color-muted)]"}`}>all</button>
          {tags.map(([tag, n]) => (
            <button key={tag} onClick={() => setFilter(tag === filter ? null : tag)} className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[11px] ${filter === tag ? "border-[var(--color-blue)] text-[var(--color-blue)]" : "border-[var(--color-border)] text-[var(--color-muted)] hover:border-[var(--color-border-2)]"}`}>
              <span className="font-mono">{tag}</span>
              <span className="text-[var(--color-faint)]">{n}</span>
            </button>
          ))}
        </div>
      )}

      {err ? (
        <Empty>Couldn't load the plot manifest — the stream endpoint returned an error.</Empty>
      ) : shown.length === 0 ? (
        <Empty>No plots yet. Instruments emit here as experiments run — try <span className="kbd mx-1">/dynamic-viz</span> in a terminal.</Empty>
      ) : (
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          {shown.map((it, i) => (
            <Card key={it.file + it.ts} hover className={`overflow-hidden ${i < freshCount && !filter ? "fadein" : ""}`} style={i < freshCount && !filter ? { boxShadow: "0 0 0 1px #39c5cf, 0 8px 30px -12px #39c5cf55" } : undefined}>
              <button onClick={() => setLightbox(it)} className="block w-full">
                <img src={streamUrl(it.file, it.ts)} className="w-full bg-[#0d1117]" />
              </button>
              <div className="px-4 py-3">
                <div className="mb-1.5 flex items-center gap-2">
                  {it.run && <Pill tone={"blue" as Tone}>{it.run}</Pill>}
                  <span className="ml-auto text-[11px] text-[var(--color-faint)]">{timeAgo(it.ts)}</span>
                </div>
                <div className="text-[12px] leading-relaxed text-[var(--color-muted)]">{it.caption}</div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {lightbox && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/80 p-8 backdrop-blur" onClick={() => setLightbox(null)}>
          <button className="absolute right-6 top-6 flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--color-border-2)] text-[var(--color-muted)] hover:text-[var(--color-fg)]"><IconClose /></button>
          <img src={streamUrl(lightbox.file, lightbox.ts)} className="max-h-[82vh] max-w-[92vw] rounded-lg" onClick={(e) => e.stopPropagation()} />
          <div className="mt-3 max-w-[800px] text-center text-[13px] text-[var(--color-muted)]">{lightbox.caption}</div>
        </div>
      )}
    </div>
  );
}
