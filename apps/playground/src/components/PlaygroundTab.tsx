import React, { useEffect, useState } from 'react';
import { useWorker, Heartbeat } from '../hooks/useWorker.tsx';
import { PRESETS } from '../lib/presets.ts';
import { AnswerCard } from '../components/answers.tsx';
import { Waterfall } from '../components/waterfall.tsx';
import type { Questions, WorkerResponse } from '@laya/js/laya-types.ts';

export type Backend = 'auto' | 'webgpu' | 'wasm';
export type Precision = 'fp32' | 'fp16';
export type Checkpoint = 'auto' | 'english' | 'multilingual' | 'typed-decisions';

export function stageLabel(s: string): string {
  if (s === 'tokenizer') return 'Tokenizer loaded';
  if (s.startsWith('tokenizer-')) return `Tokenizer loaded (${s.slice('tokenizer-'.length)})`;
  if (s.startsWith('model-')) return `Model fetched (${s.slice('model-'.length)})`;
  if (s.startsWith('backend-')) return `Backend ready (${s.slice('backend-'.length)})`;
  if (s === 'tokenize') return 'Tokenized input';
  if (s === 'inference') return 'Ran inference';
  return s;
}

export function isSetupStage(s: string): boolean {
  return s === 'tokenizer' || s.startsWith('tokenizer-') || s.startsWith('model-') || s.startsWith('backend-');
}

export function compactJson(text: string): string {
  return JSON.stringify(JSON.parse(text) as unknown);
}

export function nodeSnippet(stateText: string, qText: string): string {
  return [
    "import { LayaClient } from '@laya/js';",
    '',
    'const laya = await LayaClient.open();',
    `const result = await laya.predict(${compactJson(stateText)}, ${compactJson(qText)});`,
    'console.log(result.answers, result.timings);',
    'await laya.close();',
    '',
  ].join('\n');
}

export function cliSnippet(stateText: string, qText: string, precision: Precision): string {
  const fp = precision === 'fp16' ? ' --fp16' : '';
  return `node bin/cli.ts${fp} --state '${compactJson(stateText)}' --questions '${compactJson(qText)}'\n`;
}

export function browserSnippet(
  stateText: string,
  qText: string,
  backend: Backend,
  precision: Precision,
  checkpoint: Checkpoint = 'auto',
): string {
  return [
    "const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });",
    'worker.onmessage = (e) => {',
    "  if (e.data.type === 'done') console.log(e.data.result.answers, e.data.timings, e.data.routing);",
    "  if (e.data.type === 'error') console.error(e.data.message);",
    '};',
    `worker.postMessage({ state: ${compactJson(stateText)}, questions: ${compactJson(qText)}, backend: '${backend}', precision: '${precision}', checkpoint: '${checkpoint}' });`,
    '',
  ].join('\n');
}

export function PlaygroundTab(): React.JSX.Element {
  const worker = useWorker();
  const [preset, setPreset] = useState('Original 3Q');
  const initial = PRESETS['Original 3Q'];
  if (!initial) throw new Error('missing Original 3Q preset');
  const [stateText, setStateText] = useState(JSON.stringify(initial.state, null, 1));
  const [qText, setQText] = useState(JSON.stringify(initial.questions, null, 1));
  const [backend, setBackend] = useState<Backend>('auto');
  const [precision, setPrecision] = useState<Precision>('fp16');
  const [checkpoint, setCheckpoint] = useState<Checkpoint>('auto');
  const [status, setStatus] = useState('idle');
  const [log, setLog] = useState<string[]>([]);
  const [pct, setPct] = useState<number | null>(null);
  const [result, setResult] = useState<Extract<WorkerResponse, { type: 'done' }> | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  const copySnippet = (label: string, build: () => string): void => {
    let text: string;
    try {
      text = build();
    } catch {
      return;
    }
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text).catch(() => undefined);
    setCopied(label);
    setTimeout(() => setCopied((c) => (c === label ? null : c)), 1500);
  };

  useEffect(() => {
    const onMsg = (e: MessageEvent<WorkerResponse>): void => {
      const m = e.data;
      if (m.type === 'progress') {
        setLog((l) => [...l, m.stage]);
        setStatus(m.stage);
      } else if (m.type === 'download') setPct(m.pct);
      else if (m.type === 'done') {
        setResult(m);
        setStatus('ready');
        setPct(null);
      } else if (m.type === 'error') {
        setStatus(`error: ${m.message}`);
        setPct(null);
      }
    };
    worker.addEventListener('message', onMsg);
    return () => worker.removeEventListener('message', onMsg);
  }, [worker]);

  const applyPreset = (name: string): void => {
    const p = PRESETS[name];
    if (!p) {
      setStatus(`unknown preset: ${name}`);
      return;
    }
    setPreset(name);
    setStateText(JSON.stringify(p.state, null, 1));
    setQText(JSON.stringify(p.questions, null, 1));
  };

  const run = (): void => {
    try {
      const state: unknown = JSON.parse(stateText) as unknown;
      const questions = JSON.parse(qText) as Questions;
      if (!questions || !Object.keys(questions).length) {
        setStatus('invalid: no questions');
        return;
      }
      setResult(null);
      setLog([]);
      setStatus('starting');
      setPct(null);
      worker.postMessage({ state, questions, backend, precision, checkpoint });
    } catch (err) {
      setStatus(`invalid JSON: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  return (
    <div className="grid">
      <div>
        <div className="row">
          <label htmlFor="preset">Preset</label>
          <select id="preset" value={preset} onChange={(e) => applyPreset(e.target.value)}>
            {Object.keys(PRESETS).map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
          <label htmlFor="backend">Backend</label>
          <select id="backend" value={backend} onChange={(e) => setBackend(e.target.value as Backend)}>
            <option value="auto">auto (webgpu→wasm)</option>
            <option value="webgpu">webgpu</option>
            <option value="wasm">wasm</option>
          </select>
          <label htmlFor="precision">Precision</label>
          <select
            id="precision"
            value={precision}
            onChange={(e) => setPrecision(e.target.value as Precision)}
            title="FP16 halves download; WebGPU FP16 is fast but 40× less accurate — see Progress"
          >
            <option value="fp16">fp16 (806MB)</option>
            <option value="fp32">fp32 (1.6GB)</option>
          </select>
          <label htmlFor="checkpoint">Checkpoint</label>
          <select
            id="checkpoint"
            value={checkpoint}
            onChange={(e) => setCheckpoint(e.target.value as Checkpoint)}
            title="auto runs JS routing (non-Latin → multilingual); explicit pins a checkpoint. Multilingual/typed need B2 export artifacts in /models."
          >
            <option value="auto">auto (router)</option>
            <option value="english">english</option>
            <option value="multilingual">multilingual</option>
            <option value="typed-decisions">typed-decisions</option>
          </select>
          <button onClick={run}>Run</button>
          <Heartbeat />
        </div>
        <label htmlFor="state">State (JSON)</label>
        <textarea
          id="state"
          value={stateText}
          onChange={(e) => setStateText(e.target.value)}
          rows={5}
          spellCheck={false}
        />
        <label htmlFor="questions">Questions (Laya format)</label>
        <textarea
          id="questions"
          value={qText}
          onChange={(e) => setQText(e.target.value)}
          rows={12}
          spellCheck={false}
        />
        <div className="status" role="status">
          status: <b>{status}</b>
          {pct !== null && (
            <>
              {' · download '}
              <progress value={pct} max={100} aria-label="model download percent">
                {pct}%
              </progress>{' '}
              {pct}%
            </>
          )}
        </div>
        {log.length > 0 && (
          <div className="log">
            {log.some(isSetupStage) && <div>Setup (once): {log.filter(isSetupStage).map(stageLabel).join(' → ')}</div>}
            <div>
              This run:{' '}
              {log
                .filter((s) => !isSetupStage(s))
                .map(stageLabel)
                .join(' → ') || '…'}
            </div>
          </div>
        )}
      </div>
      <div>
        {!result && (
          <div className="empty">
            Run to see calibrated answers here.
            <br />
            First run downloads 806MB once (localhost, cached after).
          </div>
        )}
        {result && (
          <>
            <div className="meta">
              backend {result.backend} · {result.model} · seq {result.seqLen} · K {result.kmax} · tokens{' '}
              {result.result.usage.input_tokens}
              {result.routing && (
                <>
                  {' · routed '}
                  <b>{result.routing.model}</b> ({result.routing.reason})
                </>
              )}
            </div>
            <Waterfall timings={result.timings} setup={result.setup} />
            <div className="row" aria-label="Copy this run as code">
              <button onClick={() => copySnippet('node', () => nodeSnippet(stateText, qText))}>
                {copied === 'node' ? 'Copied ✓' : 'Copy as Node'}
              </button>
              <button onClick={() => copySnippet('cli', () => cliSnippet(stateText, qText, precision))}>
                {copied === 'cli' ? 'Copied ✓' : 'Copy as CLI'}
              </button>
              <button
                onClick={() =>
                  copySnippet('browser', () => browserSnippet(stateText, qText, backend, precision, checkpoint))
                }
              >
                {copied === 'browser' ? 'Copied ✓' : 'Copy as browser'}
              </button>
            </div>
            {Object.entries(result.result.answers).map(([qid, a]) => (
              <AnswerCard key={qid} qid={qid} answer={a} />
            ))}
            <details>
              <summary>raw JSON</summary>
              <pre>{JSON.stringify(result.result, null, 1)}</pre>
            </details>
          </>
        )}
      </div>
    </div>
  );
}
