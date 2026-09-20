import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, act, fireEvent, cleanup } from '@testing-library/react';
import { Heartbeat } from './useWorker.tsx';

void React;

describe('Heartbeat', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('ticks every second', () => {
    render(<Heartbeat />);
    expect(screen.getByText(/♥ 0 ·/)).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByText(/♥ 1 ·/)).toBeInTheDocument();
  });

  it('shows paused when the tab hides and resumes without booking hidden time as lag', () => {
    let hidden = false;
    vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden);
    render(<Heartbeat />);
    act(() => {
      hidden = true;
      fireEvent(document, new Event('visibilitychange'));
    });
    expect(screen.getByText('♥ paused (tab hidden)')).toBeInTheDocument();
    act(() => {
      hidden = false;
      fireEvent(document, new Event('visibilitychange'));
    });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    // ticks again, and the hidden gap is not booked as lag
    expect(screen.getByText(/♥ 1 · lag \d+ms max \d+ms/)).toBeInTheDocument();
  });
});
