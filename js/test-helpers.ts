// Shared fixture questions — single source of truth for checks + tests.
// Must match export/check_parity.py fixtures().
import type { ChoiceQuestion, NoulQuestion, Questions, ScoreQuestion } from './laya-types.ts';

export const STATE: Record<string, string> = {
  subject: 'Duplicate charge on invoice 4411',
  body: 'We were billed twice for March. Please refund the duplicate.',
};

export function questionsFor(name: string): Questions {
  const c3: ChoiceQuestion = {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: { billing: 'invoices, payments, refunds', technical: 'bugs and outages', sales: 'pricing' },
  };
  const s3: ScoreQuestion = {
    type: 'score',
    instructions: 'How urgent is this?',
    criteria: ['not urgent', 'soon', 'blocking'],
  };
  const n2: NoulQuestion = { type: 'noul', instructions: 'Does the user threaten to cancel?' };
  if (name === 'orig-3q') return { department: c3, urgency: s3, churn_risk: n2 };
  if (name === 'choice-3') return { department: c3 };
  if (name === 'choice-2') {
    return {
      dept2: { type: 'choice', instructions: 'Billing or tech?', criteria: { billing: 'invoices', technical: 'bugs' } },
    };
  }
  if (name === 'choice-6') {
    return {
      dept6: {
        type: 'choice',
        instructions: 'Route it',
        criteria: Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`team${i}`, `area ${i}`])),
      },
    };
  }
  if (name === 'mixed-batch') {
    return {
      a_choice5: {
        type: 'choice',
        instructions: 'Pick',
        criteria: Object.fromEntries(Array.from({ length: 5 }, (_, i) => [`o${i}`, `desc ${i}`])),
      },
      b_noul: n2,
      c_score2: { type: 'score', instructions: 'Rate', criteria: ['low', 'high'] },
    };
  }
  throw new Error(`unknown fixture: ${name}`);
}
