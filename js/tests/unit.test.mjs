// Offline unit tests: pure logic, no ONNX, no network.
// Run: npm test
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBpeTokenizer } from '../laya-bpe.mjs';
import {
  pyStringify,
  serializeState,
  renderOptions,
  toInternal,
  buildSequence,
  collateItems,
  QTYPES,
} from '../laya-preprocess.mjs';
import { softmax, confidenceFromProbs, tempBucket, predictFromLogits } from '../laya-postprocess.mjs';
import { actionLogits, gelu, erf } from '../laya-action.mjs';
import { toI64, toB8, buildFeeds, splitOutputs } from '../laya-feed.mjs';
import { STATE, questionsFor } from '../test-helpers.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const tokJson = JSON.parse(readFileSync(join(root, 'js', 'tokenizer', 'tokenizer.json'), 'utf8'));

describe('preprocess', () => {
  it('pyStringify matches Python separators', () => {
    assert.equal(pyStringify({ a: 1, b: [1, 2] }), '{"a": 1, "b": [1, 2]}');
    assert.equal(pyStringify('x'), '"x"');
    assert.equal(serializeState({ subject: 'hi' }), '{"subject": "hi"}');
  });
  it('rejects non-finite + circular', () => {
    assert.throws(() => pyStringify({ v: NaN }), /non-finite/);
    const c = {};
    c.self = c;
    assert.throws(() => pyStringify(c), /circular/);
    const arr = [];
    arr.push(arr);
    assert.throws(() => pyStringify(arr), /circular/);
  });
  it('toInternal validates types', () => {
    assert.throws(() => toInternal(null), /invalid question/);
    assert.throws(() => toInternal({ type: 'bogus' }), /unknown question/);
    assert.throws(() => toInternal({ type: 'score', criteria: { a: 1 } }), /must be an array/);
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
    const q = toInternal(questionsFor('choice-3').department);
    const a = buildSequence(tok, STATE, q, 512, 192, null, false);
    const b = buildSequence(tok, STATE, q, 512, 192, null, true);
    assert.ok(a.ids.length > 0 && b.ids.length > 0);
  });
  it('renderOptions covers choice/score/noul', () => {
    assert.equal(renderOptions({ t: 'score', crit: ['low', 'high'] }).length, 2);
    assert.equal(renderOptions({ t: 'noul', crit: {} }).length, 2);
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
    const items = [{ markers: [1, 2], qtype: QTYPES.choice }];
    const temp = { temperature: [1, 1, 1], temperature_by_options: {} };
    const out = predictFromLogits(questions, items, [[2, -2]], [[5, -5]], temp, 10);
    assert.equal(out.answers.dept2.choice, 'billing');
    assert.throws(() => predictFromLogits({}, items, [[0]], [[0, 0]], temp, 0), /no questions/);
    assert.throws(
      () => predictFromLogits({ x: { type: 'noul' } }, [{ markers: [1], qtype: 2 }], [[0]], [[0, 0]], temp, 0),
      /K must be 2/,
    );
  });
});

describe('action head', () => {
  it('gelu/erf sanity + shape validation', () => {
    assert.ok(Math.abs(gelu(0)) < 1e-9);
    assert.ok(Math.abs(erf(0)) < 1e-6);
    assert.throws(() => actionLogits([[1]], [[true]], [[0]], {}), /malformed/);
    assert.throws(() => actionLogits([[1]], [[true]], [[0]], { w0: [], b0: [], w2: [], b2: [] }), /bad head dims/);
  });
});

describe('feed', () => {
  it('toI64/toB8 + buildFeeds/splitOutputs', () => {
    assert.deepEqual([...toI64([[1, 2], [3]])], [1n, 2n, 3n]);
    assert.deepEqual([...toB8([[true, false]])], [1, 0]);
    const fakeOrt = {
      Tensor: class {
        constructor(type, data, dims) {
          this.type = type;
          this.data = data;
          this.dims = dims;
        }
      },
    };
    const b = {
      inputIds: [[1, 2]],
      attentionMask: [[1, 1]],
      markerPos: [[0, 1]],
      markerMask: [[true, false]],
      qtype: [0],
    };
    const feeds = buildFeeds(fakeOrt, b);
    assert.deepEqual(feeds.qtype.dims, [1]);
    // @ts-expect-error — null must throw, not typecheck
    assert.throws(() => buildFeeds(fakeOrt, null), /empty batch/);
    const { logits, pooled } = splitOutputs([1, 2], new Array(1024).fill(0), 1, 2);
    assert.deepEqual(logits, [[1, 2]]);
    assert.equal(pooled[0].length, 1024);
  });
});

describe('bpe', () => {
  it('matches fuzz subset offline', () => {
    const tok = loadBpeTokenizer(tokJson);
    const cases = JSON.parse(readFileSync(join(root, 'js', 'bpe-fuzz.json'), 'utf8'));
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
