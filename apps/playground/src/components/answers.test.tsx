import { describe, it, expect } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { ProbBar, AnswerCard } from './answers.tsx';
import type { Answer } from '@laya/js/laya-types.ts';

describe('ProbBar', () => {
  it('renders label + value with progressbar role', () => {
    render(<ProbBar label="billing" value={0.845} />);
    expect(screen.getByText('billing')).toBeInTheDocument();
    expect(screen.getByText('0.8450')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '85');
  });
});

describe('AnswerCard', () => {
  it('renders choice with probabilities', () => {
    const answer: Answer = {
      type: 'choice',
      choice: 'billing',
      probabilities: { billing: 0.8, tech: 0.2 },
      confidence: 0.5,
      action: { act_probability: 0.9 },
    };
    render(<AnswerCard qid="dept" answer={answer} />);
    expect(screen.getByText('dept')).toBeInTheDocument();
    expect(screen.getByText('→ billing')).toBeInTheDocument();
  });
  it('renders noul', () => {
    const answer: Answer = { type: 'noul', noul: 0.1, confidence: 0.9, action: { act_probability: 0.9 } };
    render(<AnswerCard qid="risk" answer={answer} />);
    expect(screen.getByText('→ 0.1')).toBeInTheDocument();
  });
});
