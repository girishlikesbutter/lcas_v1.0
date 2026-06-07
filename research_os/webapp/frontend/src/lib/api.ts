import type { Snapshot, StreamItem, IntentRec } from "./types";

const J = async (r: Response) => {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

export const api = {
  snapshot: (): Promise<Snapshot> => fetch("/api/snapshot").then(J),
  streamManifest: (): Promise<{ items: StreamItem[] }> => fetch("/api/stream/manifest").then(J),
  intents: (): Promise<{ intents: IntentRec[] }> => fetch("/api/intents").then(J),
  postIntent: (body: any): Promise<IntentRec> =>
    fetch("/api/intent", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then(J),
  terminals: (): Promise<{ terminals: any[] }> => fetch("/api/terminals").then(J),
  createTerminal: (cols: number, rows: number, title = ""): Promise<{ id: string; title: string }> =>
    fetch("/api/terminals", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ cols, rows, title }),
    }).then(J),
  deleteTerminal: (id: string): Promise<any> =>
    fetch(`/api/terminals/${id}`, { method: "DELETE" }).then(J),
};

export const streamUrl = (file: string, ts?: string) =>
  `/stream/${file}${ts ? `?_=${encodeURIComponent(ts)}` : ""}`;

export type SSEEvent = { type: "store" | "stream" | "intents" | "ranking"; rev?: string };

/**
 * Subscribe to backend change events. Returns an unsubscribe fn.
 * `onStatus(connected)` reflects the live connection: true on open/message,
 * false on error (the browser auto-reconnects, and a fresh open flips it back).
 */
export function subscribeEvents(
  onEvent: (e: SSEEvent) => void,
  onStatus?: (connected: boolean) => void,
): () => void {
  const es = new EventSource("/api/events");
  es.onopen = () => onStatus?.(true);
  es.onmessage = (m) => {
    onStatus?.(true);
    try {
      onEvent(JSON.parse(m.data));
    } catch {
      /* keepalive / comment frame */
    }
  };
  es.onerror = () => {
    // readyState CONNECTING(0)/CLOSED(2) → not currently live. Auto-reconnects.
    onStatus?.(false);
  };
  return () => es.close();
}
