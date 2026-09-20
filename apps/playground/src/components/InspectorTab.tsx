import React, { useEffect, useState } from 'react';
import { loadBpeTokenizer } from '@laya/js/laya-bpe.ts';
import { FIXTURE_NAMES } from '../lib/presets.ts';
import type { JsFixture } from '../lib/reports.ts';
import type { Tokenizer, TokenizerJson } from '@laya/js/laya-types.ts';

// Lazy chunks: tokenizer.json (3.5MB) loads only when the tab mounts.
async function loadTokenizerJson(): Promise<TokenizerJson> {
  const m = (await import('@laya/js/src/tokenizer/tokenizer.json')) as unknown as { default: TokenizerJson };
  return m.default;
}

const fixtureLoaders: Record<string, () => Promise<JsFixture>> = {
  'orig-3q': () =>
    import('@laya/test-vectors/vectors/orig-3q.json').then((m) => (m.default ?? m) as unknown as JsFixture),
  'choice-3': () =>
    import('@laya/test-vectors/vectors/choice-3.json').then((m) => (m.default ?? m) as unknown as JsFixture),
  'choice-2': () =>
    import('@laya/test-vectors/vectors/choice-2.json').then((m) => (m.default ?? m) as unknown as JsFixture),
  'choice-6': () =>
    import('@laya/test-vectors/vectors/choice-6.json').then((m) => (m.default ?? m) as unknown as JsFixture),
  'mixed-batch': () =>
    import('@laya/test-vectors/vectors/mixed-batch.json').then((m) => (m.default ?? m) as unknown as JsFixture),
};

export function InspectorTab(): React.JSX.Element {
  const [tok, setTok] = useState<Tokenizer | null>(null);
  const [rev, setRev] = useState<Map<number, string> | null>(null);
  const [tokErr, setTokErr] = useState<string | null>(null);
  const [text, setText] = useState('choice question: Which team should handle this?');
  const [fxName, setFxName] = useState('orig-3q');
  const [fx, setFx] = useState<JsFixture | null>(null);
  const [fxErr, setFxErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    loadTokenizerJson()
      .then((tj) => {
        if (!live) return;
        setTok(loadBpeTokenizer(tj));
        setRev(new Map([...Object.entries(tj.model.vocab)].map(([k, v]) => [v, k])));
      })
      .catch((err: unknown) => {
        if (live) setTokErr(err instanceof Error ? err.message : String(err));
      });
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    let live = true;
    setFx(null);
    setFxErr(null);
    const load = fixtureLoaders[fxName];
    if (!load) {
      setFxErr(`unknown fixture: ${fxName}`);
      return;
    }
    load()
      .then((data) => {
        if (live) setFx(data);
      })
      .catch((err: unknown) => {
        if (live) setFxErr(err instanceof Error ? err.message : String(err));
      });
    return () => {
      live = false;
    };
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
