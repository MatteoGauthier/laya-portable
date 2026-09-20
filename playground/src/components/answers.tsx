import React from 'react';
import type { Answer } from '@js/laya-types.ts';

export function ProbBar({ label, value }: { label: string; value: number }): React.JSX.Element {
  return (
    <div className="prob">
      <span className="prob-label">{label}</span>
      <div
        className="prob-track"
        role="progressbar"
        aria-valuenow={Math.round(value * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div className="prob-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="prob-val">{value.toFixed(4)}</span>
    </div>
  );
}

export function AnswerCard({ qid, answer }: { qid: string; answer: Answer }): React.JSX.Element {
  return (
    <div className="card">
      <div className="card-head">
        <b>{qid}</b>
        <span className="badge">{answer.type}</span>
      </div>
      {answer.type === 'choice' && (
        <>
          <div className="choice">→ {answer.choice}</div>
          {Object.entries(answer.probabilities).map(([k, v]) => (
            <ProbBar key={k} label={k} value={v} />
          ))}
        </>
      )}
      {answer.type === 'score' && (
        <>
          <div className="choice">→ {answer.score}</div>
          {Object.entries(answer.probabilities).map(([k, v]) => (
            <ProbBar key={k} label={`${k} ${String(answer.legend[k])}`} value={v} />
          ))}
        </>
      )}
      {answer.type === 'noul' && <div className="choice">→ {answer.noul}</div>}
      <div className="meta">
        confidence {answer.confidence} · act {answer.action.act_probability}
      </div>
    </div>
  );
}
