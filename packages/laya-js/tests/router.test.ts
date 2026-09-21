import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { analyse, detectScript, guessLatinLanguage, isEnglish } from '../src/laya-lang.ts';
import { Router, matchTypedDecisionsWorkflow, normaliseName, route } from '../src/laya-router.ts';

describe('laya-lang', () => {
  it('detectScript covers latin/non-latin/unknown', () => {
    assert.equal(detectScript('I was charged twice, please refund.'), 'latin');
    assert.equal(detectScript('मुझसे दो बार शुल्क लिया गया'), 'devanagari');
    assert.equal(detectScript('我的账户被扣了两次费'), 'han');
    assert.equal(detectScript('123 !!!'), 'unknown');
  });
  it('analyse matches Python shape', () => {
    const en = analyse({ body: 'I was charged twice, please refund the duplicate.' });
    assert.equal(en.script, 'latin');
    assert.equal(en.is_english, true);
    const hi = analyse({ body: 'मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।' });
    assert.equal(hi.script, 'devanagari');
    assert.equal(hi.is_english, false);
  });
  it('guessLatinLanguage needs margin (short/english safe)', () => {
    assert.equal(guessLatinLanguage('hi'), null);
    assert.equal(guessLatinLanguage('I was charged twice please refund the duplicate now'), 'en');
    assert.equal(
      guessLatinLanguage('der die das und ist ein eine den dem nicht mit für auf von zu sich auch werden'),
      'de',
    );
  });
  it('isEnglish helper', () => {
    assert.equal(isEnglish('Please refund my invoice'), true);
    assert.equal(isEnglish('मुझसे दो बार शुल्क लिया गया'), false);
  });
});

describe('router route()', () => {
  const qs = { q: { type: 'choice', instructions: 'Pick', criteria: { a: 'x', b: 'y' } } } as never;
  it('normaliseName aliases', () => {
    assert.equal(normaliseName('en'), 'english');
    assert.equal(normaliseName('ml'), 'multilingual');
    assert.equal(normaliseName('typed_decisions'), 'typed-decisions');
    assert.throws(() => normaliseName('bogus'), /unknown model/);
  });
  it('english latin -> english', () => {
    const d = route({ body: 'I was charged twice, please refund.' }, qs);
    assert.equal(d.model, 'english');
  });
  it('non-latin -> multilingual with reason', () => {
    const d = route({ body: 'मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।' }, qs);
    assert.equal(d.model, 'multilingual');
    assert.match(d.reason, /non-Latin script/);
  });
  it('explicit model/lang/task win', () => {
    assert.equal(route({ body: 'hello' }, qs, { model: 'typed-decisions' }).model, 'typed-decisions');
    assert.equal(route({ body: 'hello' }, qs, { lang: 'hi' }).model, 'multilingual');
    assert.equal(route({ body: 'hello' }, qs, { lang: 'en' }).model, 'english');
    assert.equal(route({ body: 'hello' }, qs, { task: 'typed_decisions' }).model, 'typed-decisions');
  });
  it('unknown script falls back to default', () => {
    assert.equal(route({ n: 123 }, qs).model, 'english');
  });
  it('typed workflow opt-in only', () => {
    const typedQs = {
      action: { type: 'choice', instructions: 'a', criteria: { x: 'y' } },
      category: { type: 'choice', instructions: 'a', criteria: { x: 'y' } },
      churn_risk: { type: 'noul', instructions: 'a' },
      needs_human: { type: 'noul', instructions: 'a' },
      urgency: { type: 'score', instructions: 'a', criteria: ['l', 'h'] },
    } as never;
    assert.equal(matchTypedDecisionsWorkflow(typedQs), 'customer_service');
    assert.equal(route({ body: 'hello' }, typedQs).model, 'english');
    const r = new Router({ auto_task_detection: true, open: async () => ({ predict: async () => ({}) }) as never });
    assert.equal(r.routeDecision({ body: 'hello' }, typedQs).model, 'typed-decisions');
  });
});

describe('Router LRU', () => {
  it('evicts least-recently-used and preload raises cap', async () => {
    const opened: string[] = [];
    const mk = (name: string) => ({ predict: async () => ({ m: name }), close: async () => {} });
    const r = new Router({
      max_loaded: 1,
      open: async (name) => {
        opened.push(name);
        return mk(name) as never;
      },
    });
    await r.load('english');
    await r.load('multilingual');
    assert.deepEqual(r.loaded, ['multilingual']);
    assert.deepEqual(opened, ['english', 'multilingual']);
    await r.preload(['english']);
    assert.ok(r.max_loaded >= 1);
    await r.unload();
    assert.deepEqual(r.loaded, []);
  });
});
