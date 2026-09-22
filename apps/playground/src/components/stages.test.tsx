import { describe, it, expect } from 'vitest';
import { stageLabel, isSetupStage } from './PlaygroundTab.tsx';

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
