import { describe, it, expect } from 'vitest';
import { stageLabel, isSetupStage, nodeSnippet, cliSnippet, browserSnippet } from './PlaygroundTab.tsx';

describe('stage labels', () => {
  it('labels known stages', () => {
    expect(stageLabel('tokenizer')).toBe('Tokenizer loaded');
    expect(stageLabel('model-auto-fp32')).toBe('Model fetched (auto-fp32)');
    expect(stageLabel('backend-webgpu')).toBe('Backend ready (webgpu)');
    expect(stageLabel('tokenize')).toBe('Tokenized input');
    expect(stageLabel('inference')).toBe('Ran inference');
  });
  it('splits setup vs per-run stages', () => {
    expect(isSetupStage('tokenizer')).toBe(true);
    expect(isSetupStage('model-auto-fp32')).toBe(true);
    expect(isSetupStage('tokenize')).toBe(false);
    expect(isSetupStage('inference')).toBe(false);
  });
});

describe('snippets', () => {
  const state = '{"subject":"hi"}';
  const questions = '{"department":{"type":"choice"}}';
  it('node snippet opens a client and predicts', () => {
    const s = nodeSnippet(state, questions);
    expect(s).toContain("from '@laya/js'");
    expect(s).toContain('LayaClient.open()');
    expect(s).toContain('laya.predict(');
    // must be syntactically plausible: JSON payloads embedded
    expect(s).toContain('{"subject":"hi"}');
  });
  it('cli snippet carries precision flag', () => {
    expect(cliSnippet(state, questions, 'fp32')).not.toContain('--fp16');
    expect(cliSnippet(state, questions, 'fp16')).toContain('--fp16');
  });
  it('browser snippet carries backend and precision', () => {
    const s = browserSnippet(state, questions, 'webgpu', 'fp16');
    expect(s).toContain("backend: 'webgpu'");
    expect(s).toContain("precision: 'fp16'");
    expect(s).toContain('postMessage');
  });
});
