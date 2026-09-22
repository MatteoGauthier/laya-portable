// Shape of the JS fixture JSON rendered by InspectorTab.
export interface JsFixture {
  batch: number;
  seq_len: number;
  kmax: number;
  qids: string[];
  expected_answers: Record<string, unknown>;
}
