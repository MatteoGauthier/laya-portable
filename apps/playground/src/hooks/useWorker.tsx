import React, { useEffect, useRef, useState } from 'react';

// App-lifetime worker singleton, created once per page load and intentionally
// NEVER terminated in an effect cleanup: under StrictMode (dev), React runs
// setup → cleanup → setup on mount, so a terminate-on-unmount cleanup kills
// the worker milliseconds after creation and every postMessage is then
// silently dropped (status stuck at 'starting' forever, no error anywhere).
// The browser reclaims the worker on page unload.
let shared: Worker | null = null;

export function useWorker(): Worker {
  if (!shared) {
    shared = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
  }
  return shared;
}

// Gap between expected and actual tick, i.e. main-thread stall evidence.
// Pure for tests; the component feeds it measured inter-tick gaps.
export function computeLag(gapMs: number, intervalMs: number): number {
  return Math.max(0, Math.round(gapMs - intervalMs));
}

// Isolated main-thread heartbeat (previously re-rendered the whole tab 4x/sec).
// Shows event-loop lag, not just a counter: if the main thread stalls during
// a run (memory pressure, background-tab timer clamping), max lag names it.
// A hidden tab pauses timers by browser design — the heartbeat says "paused"
// instead of freezing silently, and hidden time never counts as lag.
export function Heartbeat(): React.JSX.Element {
  const [stats, setStats] = useState({ ticks: 0, lagMs: 0, maxLagMs: 0, paused: false });
  const lastRef = useRef(0);
  useEffect(() => {
    const intervalMs = 1000;
    lastRef.current = performance.now();
    let maxLag = 0;
    const onVis = (): void => {
      if (document.hidden) {
        setStats((s) => ({ ...s, paused: true }));
      } else {
        // Returning: reset the baseline so hidden time isn't booked as lag.
        lastRef.current = performance.now();
        setStats((s) => ({ ...s, paused: false }));
      }
    };
    document.addEventListener('visibilitychange', onVis);
    const t = setInterval(() => {
      if (document.hidden) return;
      const now = performance.now();
      const lag = computeLag(now - lastRef.current, intervalMs);
      lastRef.current = now;
      if (lag > maxLag) maxLag = lag;
      setStats((s) => ({ ticks: s.ticks + 1, lagMs: lag, maxLagMs: maxLag, paused: false }));
    }, intervalMs);
    return () => {
      clearInterval(t);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, []);
  if (stats.paused) {
    return (
      <span className="ticks" title="tab hidden — timers clamp, heartbeat resumes on return" aria-live="off">
        ♥ paused (tab hidden)
      </span>
    );
  }
  return (
    <span
      className="ticks"
      title="main-thread heartbeat — lag is event-loop delay past the 1s tick (background tabs clamp timers; stalls during a run point at renderer memory pressure)"
      aria-live="off"
    >
      ♥ {stats.ticks} · lag {stats.lagMs}ms max {stats.maxLagMs}ms
    </span>
  );
}
