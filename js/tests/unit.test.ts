// Offline unit tests: pure logic, no ONNX, no network.
// Run: npm test (native type-stripping, no build step)
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBpeTokenizer } from '../laya-bpe.ts';
import {
  pyStringify,
  serializeState,
  renderOptions,
  toInternal,
  buildSequence,
  collateItems,
  QTYPES,
} from '../laya-preprocess.ts';
import { softmax, confidenceFromProbs, tempBucket, predictFromLogits } from '../laya-postprocess.ts';
import { actionLogits, gelu, erf } from '../laya-action.ts';
import { toI64, toB8, toFeedData, splitOutputs } from '../laya-feed.ts';
import { STATE, questionsFor } from '../test-helpers.ts';
import type { BuiltItem, CollatedBatch, TemperatureConfig, TokenizerJson } from '../laya-types.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const tokJson = JSON.parse(readFileSync(join(root, 'js', 'tokenizer', 'tokenizer.json'), 'utf8')) as TokenizerJson;

describe('preprocess', () => {
  it('pyStringify matches Python separators', () => {
    assert.equal(pyStringify({ a: 1, b: [1, 2] }), '{"a": 1, "b": [1, 2]}');
    assert.equal(pyStringify('x'), '"x"');
    assert.equal(serializeState({ subject: 'hi' }), '{"subject": "hi"}');
  });
  it('rejects non-finite + circular', () => {
    assert.throws(() => pyStringify({ v: Number.NaN }), /non-finite/);
    const c: Record<string, unknown> = {};
    c['self'] = c;
    assert.throws(() => pyStringify(c), /circular/);
    const arr: unknown[] = [];
    arr.push(arr);
    assert.throws(() => pyStringify(arr), /circular/);
  });
  it('toInternal validates types', () => {
    assert.throws(() => toInternal(null as never), /invalid question/);
    assert.throws(() => toInternal({ type: 'bogus' } as never), /unknown question/);
    assert.throws(() => toInternal({ type: 'score', criteria: { a: 1 } } as never), /must be an array/);
    const q = toInternal({ type: 'choice', instructions: 'i', criteria: ['a', 'b'] });
    assert.deepEqual(Object.keys(q.crit), ['a', 'b']);
  });
  it('buildSequence guards empty + collate guards empty', () => {
    const tok = loadBpeTokenizer(tokJson);
    const q = toInternal({ type: 'choice', instructions: 'Pick', criteria: { a: 'x' } });
    const { ids, markers } = buildSequence(tok, STATE, q, 512, 192);
    assert.ok(ids.length > 0 && markers.length === 1);
    assert.throws(() => collateItems([[]], tok.padId), /empty batch/);
  });
  it('optionOrder/truncateLeft params work', () => {
    const tok = loadBpeTokenizer(tokJson);
    const dept = questionsFor('choice-3')['department'];
    if (dept === undefined) throw new Error('missing fixture question');
    const q = toInternal(dept);
    const a = buildSequence(tok, STATE, q, 512, 192, null, false);
    const b = buildSequence(tok, STATE, q, 512, 192, null, true);
    assert.ok(a.ids.length > 0 && b.ids.length > 0);
  });
  it('renderOptions covers choice/score/noul', () => {
    assert.equal(renderOptions({ t: 'score', ins: '', crit: ['low', 'high'] }).length, 2);
    assert.equal(renderOptions({ t: 'noul', ins: '', crit: {} }).length, 2);
  });
});

describe('postprocess', () => {
  it('softmax stable + tempBucket', () => {
    const p = softmax([1000, 1001, 999]);
    assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    assert.equal(tempBucket(0, 2), 'choice:2');
    assert.equal(tempBucket(0, 11), 'choice:11+');
    assert.throws(() => softmax([]), /empty logits/);
  });
  it('confidence bounds', () => {
    assert.equal(confidenceFromProbs([1], 1), 1.0);
    const c = confidenceFromProbs([0.5, 0.5], 2);
    assert.ok(c >= 0 && c <= 1);
  });
  it('predictFromLogits validates + scores choice', () => {
    const questions = questionsFor('choice-2');
    const items: BuiltItem[] = [{ ids: [], markers: [1, 2], qtype: QTYPES['choice'] }];
    const temp: TemperatureConfig = { temperature: [1, 1, 1], temperature_by_options: {} };
    const out = predictFromLogits(questions, items, [[2, -2]], [[5, -5]], temp, 10);
    const ans = out.answers['dept2'];
    if (ans?.type !== 'choice') throw new Error('expected choice answer');
    assert.equal(ans.choice, 'billing');
    assert.throws(() => predictFromLogits({}, items, [[0]], [[0, 0]], temp, 0), /no questions/);
    assert.throws(
      () =>
        predictFromLogits(
          { x: { type: 'noul', instructions: '' } },
          [{ ids: [], markers: [1], qtype: 2 }],
          [[0]],
          [[0, 0]],
          temp,
          0,
        ),
      /K must be 2/,
    );
  });
});

describe('action head', () => {
  it('gelu/erf sanity + shape validation', () => {
    assert.ok(Math.abs(gelu(0)) < 1e-9);
    assert.ok(Math.abs(erf(0)) < 1e-6);
    assert.throws(() => actionLogits([[1]], [[true]], [[0]], {} as never), /malformed/);
    assert.throws(() => actionLogits([[1]], [[true]], [[0]], { w0: [], b0: [], w2: [], b2: [] }), /bad head dims/);
  });
});

describe('feed', () => {
  it('toI64/toB8 + toFeedData/splitOutputs', () => {
    assert.deepEqual([...toI64([[1, 2], [3]])], [1n, 2n, 3n]);
    assert.deepEqual([...toB8([[true, false]])], [1, 0]);
    const b: CollatedBatch = {
      inputIds: [[1, 2]],
      attentionMask: [[1, 1]],
      markerPos: [[0, 1]],
      markerMask: [[true, false]],
      qtype: [0],
    };
    const d = toFeedData(b);
    assert.deepEqual(d.dims, { B: 1, S: 2, K: 2 });
    assert.throws(() => toFeedData(null as never), /empty batch/);
    const { logits, pooled } = splitOutputs([1, 2], new Array(1024).fill(0), 1, 2);
    assert.deepEqual(logits, [[1, 2]]);
    const p0 = pooled[0];
    if (p0 === undefined) throw new Error('missing pooled row');
    assert.equal(p0.length, 1024);
  });
});

describe('bpe', () => {
  it('matches fuzz subset offline', () => {
    const tok = loadBpeTokenizer(tokJson);
    const cases = JSON.parse(readFileSync(join(root, 'js', 'bpe-fuzz.json'), 'utf8')) as {
      text: string;
      ids: number[];
    }[];
    for (const { text, ids: exp } of cases.slice(0, 50)) {
      assert.deepEqual(tok.encode(text), exp, `bpe mismatch for ${JSON.stringify(text.slice(0, 40))}`);
    }
  });
  it('surrogate pairs stay together', () => {
    const tok = loadBpeTokenizer(tokJson);
    assert.deepEqual(tok.encode('😀'), tok.encode('😀'));
    assert.ok(tok.encode('😀😀').length >= tok.encode('😀').length);
  });
});
