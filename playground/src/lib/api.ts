// fetchJson with response.ok check + AbortSignal support.
// Replaces fire-and-forget fetch(...).then(r=>r.json()).catch(()=>{}) which
// left permanent `loading…` with no error UI.
import { useEffect, useState } from 'react';

export async function fetchJson<T>(url: string, opts: { signal?: AbortSignal } = {}): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(url, { signal: opts.signal });
  } catch (err) {
    throw new Error(`fetch ${url} failed: ${err instanceof Error ? err.message : String(err)}`, { cause: err });
  }
  if (!resp.ok) throw new Error(`fetch ${url}: HTTP ${resp.status}`);
  try {
    return (await resp.json()) as T;
  } catch (err) {
    throw new Error(`fetch ${url}: invalid JSON`, { cause: err });
  }
}

/** React helper: load JSON with loading/error states + abort on unmount. */
export function useJson<T>(url: string): [T | null, string | null] {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    setData(null);
    setError(null);
    fetchJson<T>(url, { signal: ac.signal })
      .then(setData)
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => ac.abort();
  }, [url]);
  return [data, error];
}
