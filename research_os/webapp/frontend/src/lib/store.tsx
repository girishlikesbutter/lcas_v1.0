import {
  createContext, useContext, useEffect, useState, useCallback, useRef, type ReactNode,
} from "react";
import type { Snapshot, IntentRec } from "./types";
import { api, subscribeEvents } from "./api";

interface StoreCtx {
  snap: Snapshot | null;
  loading: boolean;
  error: string | null;
  /** rev id from the last SSE store event — used to flash "updated" affordances */
  lastStoreRev: string;
  /** bumps each time the plot stream changes (new plot landed) */
  streamTick: number;
  /** bumps each time the intent queue changes */
  intentTick: number;
  /** the control-plane queue (newest-first) + a count of still-queued intents */
  intents: IntentRec[];
  pendingIntents: number;
  connected: boolean;
  refetch: () => void;
}

const Ctx = createContext<StoreCtx | null>(null);

/**
 * Single source of live state for the app. Fetches the read-model snapshot, then
 * listens to the backend SSE channel and refetches on any `store` event (the
 * canonical files changed under us — a skill ran in a terminal, a record landed).
 * `stream`/`intents` events bump lightweight counters the relevant views watch.
 */
export function StoreProvider({ children }: { children: ReactNode }) {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastStoreRev, setLastStoreRev] = useState("");
  const [streamTick, setStreamTick] = useState(0);
  const [intentTick, setIntentTick] = useState(0);
  const [intents, setIntents] = useState<IntentRec[]>([]);
  const [connected, setConnected] = useState(false);
  const revRef = useRef("");

  const refetch = useCallback(() => {
    api
      .snapshot()
      .then((s) => {
        setSnap(s);
        revRef.current = s.rev;
        setError(null);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  // the control-plane queue: fetched on mount and whenever the queue changes (SSE)
  useEffect(() => {
    api.intents().then((r) => setIntents(r.intents || [])).catch(() => {});
  }, [intentTick]);

  useEffect(() => {
    refetch();
    const unsub = subscribeEvents(
      (e) => {
        if (e.type === "store") {
          setLastStoreRev(e.rev || "");
          if (e.rev && e.rev !== revRef.current) refetch();
        } else if (e.type === "stream") {
          setStreamTick((t) => t + 1);
        } else if (e.type === "intents") {
          setIntentTick((t) => t + 1);
        } else if (e.type === "ranking") {
          // a fresh /strategize ranking landed (derived; store rev unchanged) —
          // re-pull the snapshot so the frontier re-orders.
          refetch();
        }
      },
      setConnected, // live indicator tracks the real SSE connection state
    );
    return unsub;
  }, [refetch]);

  const pendingIntents = intents.filter((i) => i.status === "queued").length;

  return (
    <Ctx.Provider
      value={{ snap, loading, error, lastStoreRev, streamTick, intentTick, intents, pendingIntents, connected, refetch }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useStore() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useStore outside StoreProvider");
  return c;
}

/** Convenience: throws-free access to the snapshot (null until first load). */
export function useSnap() {
  return useStore().snap;
}
