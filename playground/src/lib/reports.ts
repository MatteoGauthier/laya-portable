// Shapes of export/docs JSON reports rendered by ProgressTab.
export interface ParityFixture {
  name: string;
  max_abs_logits: number;
  max_calibrated_prob_drift: number;
  labels_match: boolean;
}

export interface ParityReport {
  fixtures: ParityFixture[];
}

export interface BaselineRun {
  device: string;
  questions: number;
  forward: { p50_ms: number };
  end_to_end: { p50_ms: number };
}

export interface BaselineReport {
  runs: BaselineRun[];
}

export interface Fp16Fixture {
  name: string;
  max_abs_logits_vs_torch: number;
  max_calibrated_prob_drift: number;
  max_confidence_drift: number;
  label_flips: number[];
}

export interface Fp16Report {
  fixtures: Fp16Fixture[];
}

export interface AccuracyRow {
  id: string;
  pass: boolean;
}

export interface AccuracyAdapter {
  name: string;
  score: number;
  rows: AccuracyRow[];
}

export interface AccuracyReport {
  cases: number;
  adapters: AccuracyAdapter[];
}

export interface JsFixture {
  batch: number;
  seq_len: number;
  kmax: number;
  qids: string[];
  expected_answers: Record<string, unknown>;
}
