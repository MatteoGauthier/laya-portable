import React, { useEffect, useMemo, useRef, useState } from 'react';

// Single worker instance, created once, terminated on unmount.
// (Previous version constructed in render — leaked a second Worker under StrictMode.)
export function useWorker() {
  const ref = useRef(null);
  const worker = useMemo(() => {
    const w = new Worker(new URL('../worker.js', import.meta.url), { type: 'module' });
    ref.current = w;
    return w;
  }, []);
  useEffect(() => () => worker.terminate(), [worker]);
  return worker;
}

// Isolated main-thread heartbeat (previously re-rendered the whole tab 4x/sec).
export function Heartbeat() {
  const [ticks, setTicks] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setTicks((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <span className="ticks" title="main-thread heartbeat — proves the page stays responsive" aria-live="off">
      ♥ {ticks}
    </span>
  );
}
