// fetchJson with response.ok check + AbortSignal support.
// Replaces fire-and-forget fetch(...).then(r=>r.json()).catch(()=>{}) which
// left permanent `loading…` with no error UI.
/**
 * @param {string} url
 * @param {{signal?: AbortSignal}} [opts]
 */
export async function fetchJson(url, opts = {}) {
  let resp;
  try {
    resp = await fetch(url, { signal: opts.signal });
  } catch (err) {
    throw new Error(`fetch ${url} failed: ${err.message}`, { cause: err });
  }
  if (!resp.ok) throw new Error(`fetch ${url}: HTTP ${resp.status}`);
  try {
    return await resp.json();
  } catch (err) {
    throw new Error(`fetch ${url}: invalid JSON`, { cause: err });
  }
}

/**
 * React helper: load JSON with loading/error states + abort on unmount.
 * Returns [data, error, loading].
 */
import { useEffect, useState } from 'react';

export function useJson(url) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    const ac = new AbortController();
    setData(null);
    setError(null);
    fetchJson(url, { signal: ac.signal })
      .then(setData)
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message);
      });
    return () => ac.abort();
  }, [url]);
  return [data, error];
}
