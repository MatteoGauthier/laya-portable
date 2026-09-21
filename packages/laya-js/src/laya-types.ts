// Shared domain types for the Laya JS runtime (Node SDK, checks, browser worker).
// Imported as types only (erased at runtime) so browser bundles stay ORT-agnostic.

export type QuestionKind = 'choice' | 'score' | 'noul';

export interface ChoiceQuestion {
  type: 'choice';
  instructions: unknown;
  criteria: Record<string, unknown> | string[];
}

export interface ScoreQuestion {
  type: 'score';
  instructions: unknown;
  criteria: unknown[];
}

export interface NoulQuestion {
  type: 'noul';
  instructions: unknown;
  criteria?: unknown;
}

export type QuestionDef = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type Questions = Record<string, QuestionDef>;

export interface InternalQuestion {
  t: QuestionKind;
  ins: string;
  crit: Record<string, unknown> | unknown[];
}

export interface Tokenizer {
  encode(text: string): number[];
  maskToken: string;
  maskId: number;
  clsId: number;
  sepId: number;
  padId: number;
}

export interface BuiltItem {
  ids: number[];
  markers: number[];
  qtype: number;
}

export interface CollatedBatch {
  inputIds: number[][];
  attentionMask: number[][];
  markerPos: number[][];
  markerMask: boolean[][];
  qtype: number[];
}

export interface TemperatureConfig {
  temperature: number[];
  temperature_by_options: Record<string, number>;
}

export interface ActHeadWeights {
  w0: number[][];
  b0: number[];
  w2: number[][];
  b2: number[];
}

export interface ChoiceAnswer {
  type: 'choice';
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  action: { act_probability: number };
}

export interface ScoreAnswer {
  type: 'score';
  score: number;
  legend: Record<string, unknown>;
  probabilities: Record<string, number>;
  confidence: number;
  action: { act_probability: number };
}

export interface NoulAnswer {
  type: 'noul';
  noul: number;
  confidence: number;
  action: { act_probability: number };
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface PredictResult {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
  timings?: PredictTimings;
}

export interface PredictTimings {
  /** BPE + sequence building + feed flattening. */
  tokenize_ms: number;
  /** ORT session.run only. */
  inference_ms: number;
  /** Split outputs + CPU action head + calibration. */
  postprocess_ms: number;
  /** End-to-end wall time. */
  total_ms: number;
}

/** Minimal shape of tokenizer.json needed by the pure-JS BPE loader. */
export interface TokenizerJson {
  model: {
    vocab: Record<string, number>;
    merges: [string, string][];
    byte_fallback?: boolean;
    fuse_unk?: boolean;
    unk_token?: string | null;
  };
  added_tokens: {
    content: string;
    id: number;
    lstrip?: boolean;
    rstrip?: boolean;
  }[];
  pre_tokenizer?: {
    type: string;
    replacement?: string;
    prepend_scheme?: string;
  };
  normalizer?: {
    type: string;
    pattern?: { String?: string };
    content?: string;
  };
}

/** Worker ↔ main-thread protocol (playground). */
export type WorkerRequest = {
  state: unknown;
  questions: Questions;
  backend?: 'auto' | 'webgpu' | 'wasm';
  precision?: 'fp32' | 'fp16';
  /** B1/B2 routing: 'auto' runs JS detection, explicit name pins a checkpoint. */
  checkpoint?: 'auto' | 'english' | 'multilingual' | 'typed-decisions';
  model?: 'english' | 'multilingual' | 'typed-decisions';
  lang?: string;
};

export type WorkerResponse =
  | { type: 'progress'; stage: string }
  | { type: 'download'; pct: number }
  | {
      type: 'done';
      result: PredictResult;
      timings: PredictTimings;
      setup: { download_ms: number; session_ms: number };
      backend: string;
      model: string;
      seqLen: number;
      kmax: number;
      routing?: { model: string; reason: string };
    }
  | { type: 'error'; message: string };
