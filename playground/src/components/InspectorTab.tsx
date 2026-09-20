import React, { useEffect, useState } from 'react';
import { loadBpeTokenizer } from '@js/laya-bpe.ts';
import { fetchJson } from '../lib/api.ts';
import { FIXTURE_NAMES } from '../lib/presets.ts';
import type { JsFixture } from '../lib/reports.ts';
import type { Tokenizer, TokenizerJson } from '@js/laya-types.ts';

export function InspectorTab(): React.JSX.Element {
  const [tok, setTok] = useState<Tokenizer | null>(null);
  const [rev, setRev] = useState<Map<number, string> | null>(null);
  const [tokErr, setTokErr] = useState<string | null>(null);
  const [text, setText] = useState('choice question: Which team should handle this?');
  const [fxName, setFxName] = useState('orig-3q');
  const [fx, setFx] = useState<JsFixture | null>(null);
  const [fxErr, setFxErr] = useState<string | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    fetchJson<TokenizerJson>('/js/tokenizer/tokenizer.json', { signal: ac.signal })
      .then((tj) => {
        setTok(loadBpeTokenizer(tj));
        setRev(new Map([...Object.entries(tj.model.vocab)].map(([k, v]) => [v, k])));
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setTokErr(err instanceof Error ? err.message : String(err));
      });
    return () => ac.abort();
  }, []);
  useEffect(() => {
    const ac = new AbortController();
    setFx(null);
    setFxErr(null);
    fetchJson<JsFixture>(`/js/fixtures/${fxName}.json`, { signal: ac.signal })
      .then(setFx)
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        setFxErr(err instanceof Error ? err.message : String(err));
      });
    return () => ac.abort();
  }, [fxName]);
  let ids: number[] = [];
  let pieces: string[] = [];
  if (tok) {
    try {
      ids = tok.encode(text);
      pieces = ids.map((id) => rev?.get(id) ?? `#${id}`);
    } catch (err) {
      pieces = [`ERROR ${err instanceof Error ? err.message : String(err)}`];
    }
  }
  return (
    <div className="grid">
      <div>
        <h3>Tokenizer (pure-JS BPE, instant)</h3>
        {tokErr ? (
          <div className="empty" role="alert">
            failed to load tokenizer: {tokErr}
          </div>
        ) : !tok ? (
          <div className="empty">loading tokenizer.json…</div>
        ) : (
          <>
            <label htmlFor="tok-text">Input text</label>
            <textarea
              id="tok-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
              spellCheck={false}
            />
            <div className="meta">{ids.length} tokens</div>
            <pre className="ids">{JSON.stringify(ids)}</pre>
            <pre className="pieces">{pieces.join(' ▏ ')}</pre>
          </>
        )}
      </div>
      <div>
        <h3>Fixtures</h3>
        <label htmlFor="fixture">Fixture</label>
        <select id="fixture" value={fxName} onChange={(e) => setFxName(e.target.value)}>
          {FIXTURE_NAMES.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        {fxErr ? (
          <div className="empty" role="alert">
            failed to load fixture: {fxErr}
          </div>
        ) : !fx ? (
          <div className="empty">loading…</div>
        ) : (
          <>
            <div className="meta">
              B={fx.batch} S={fx.seq_len} K={fx.kmax} · qids {fx.qids.join(', ')}
            </div>
            <pre>{JSON.stringify(fx.expected_answers, null, 1).slice(0, 1200)}</pre>
          </>
        )}
      </div>
    </div>
  );
}
