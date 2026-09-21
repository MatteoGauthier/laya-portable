// Pure checkpoint routing: no node imports, safe to bundle into the browser
// worker. Decides *which* checkpoint handles a request (english /
// multilingual / typed-decisions); loading weights stays in laya-router.ts
// (Node, uses laya-paths.ts filesystem locations).
import { analyse, type AnalyseResult } from './laya-lang.ts';
import { LayaConfigError } from './laya-errors.ts';
import type { Questions } from './laya-types.ts';

export type CheckpointName = 'english' | 'multilingual' | 'typed-decisions';
export type ModelName = CheckpointName;

export interface RouteDetection extends AnalyseResult {
  [k: string]: unknown;
}

export interface RouteDecision {
  model: ModelName;
  repo: string;
  reason: string;
  detection: RouteDetection | null;
  workflow: string | null;
}

export interface RouteOptions {
  model?: string;
  task?: string;
  lang?: string;
}

const BUNDLE_REPO = 'convaiinnovations/laya';

const DEFAULT_MODELS: Record<ModelName, { repo: string; sub: string | null }> = {
  english: { repo: BUNDLE_REPO, sub: null },
  multilingual: { repo: BUNDLE_REPO, sub: 'multilingual' },
  'typed-decisions': { repo: BUNDLE_REPO, sub: 'typed-decisions' },
};

const ALIASES: Record<string, ModelName> = {
  en: 'english',
  laya: 'english',
  default: 'english',
  multi: 'multilingual',
  ml: 'multilingual',
  'laya-multilingual': 'multilingual',
  typed: 'typed-decisions',
  typed_decisions: 'typed-decisions',
  'laya-typed-decisions': 'typed-decisions',
  decisions: 'typed-decisions',
};

const TYPED_DECISION_WORKFLOWS: Record<string, ReadonlySet<string>> = {
  agent_trace_observability: new Set(['action', 'needs_review', 'outcome', 'risk', 'urgency']),
  customer_service: new Set(['action', 'category', 'churn_risk', 'needs_human', 'urgency']),
  invoice_processing: new Set(['discrepancy_severity', 'disposition', 'duplicate', 'matches_order', 'urgency']),
  security_incidents: new Set(['credential_compromise', 'disposition', 'severity', 'true_positive', 'urgency']),
};

function repoStr(repo: string, sub: string | null): string {
  return sub ? `${repo}/${sub}` : repo;
}

export function normaliseName(name: string): ModelName {
  const key = String(name).trim().toLowerCase();
  const aliased = ALIASES[key] ?? key;
  if (aliased !== 'english' && aliased !== 'multilingual' && aliased !== 'typed-decisions') {
    throw new LayaConfigError(
      `unknown model ${JSON.stringify(name)}; choose one of ${['english', 'multilingual', 'typed-decisions'].join(', ')}`,
    );
  }
  return aliased as ModelName;
}

export function matchTypedDecisionsWorkflow(questions: Questions | null | undefined): string | null {
  const ids = new Set(Object.keys(questions ?? {}));
  for (const [wf, sig] of Object.entries(TYPED_DECISION_WORKFLOWS)) {
    if (ids.size !== sig.size) continue;
    let same = true;
    for (const id of ids) {
      if (!sig.has(id)) {
        same = false;
        break;
      }
    }
    if (same) return wf;
  }
  return null;
}

export function route(
  state: unknown,
  questions: Questions | null | undefined = {},
  opts: RouteOptions = {},
  defaultName: ModelName = 'english',
  autoTaskDetection = false,
): RouteDecision {
  if (opts.model !== undefined && opts.model !== null) {
    const key = normaliseName(opts.model);
    const spec = DEFAULT_MODELS[key];
    return {
      model: key,
      repo: repoStr(spec.repo, spec.sub),
      reason: `explicit model=${JSON.stringify(opts.model)}`,
      detection: null,
      workflow: null,
    };
  }
  if (opts.task !== undefined && opts.task !== null) {
    const t = String(opts.task).toLowerCase().replace(/-/g, '_');
    const key = normaliseName(t === 'typed_decisions' ? 'typed-decisions' : opts.task);
    const spec = DEFAULT_MODELS[key];
    return {
      model: key,
      repo: repoStr(spec.repo, spec.sub),
      reason: `explicit task=${JSON.stringify(opts.task)}`,
      detection: null,
      workflow: null,
    };
  }
  const workflow = matchTypedDecisionsWorkflow(questions);
  if (workflow && autoTaskDetection) {
    const spec = DEFAULT_MODELS['typed-decisions'];
    return {
      model: 'typed-decisions',
      repo: repoStr(spec.repo, spec.sub),
      reason: `question ids match the ${JSON.stringify(workflow)} typed-decisions workflow`,
      detection: null,
      workflow,
    };
  }
  if (opts.lang !== undefined && opts.lang !== null) {
    const l = String(opts.lang).toLowerCase().split('-')[0];
    const key: ModelName = l === 'en' || l === 'eng' || l === 'english' ? 'english' : 'multilingual';
    const spec = DEFAULT_MODELS[key];
    return {
      model: key,
      repo: repoStr(spec.repo, spec.sub),
      reason: `explicit lang=${JSON.stringify(opts.lang)}`,
      detection: null,
      workflow,
    };
  }
  const det = analyse(state) as RouteDetection;
  if (det.script === 'unknown') {
    const spec = DEFAULT_MODELS[defaultName];
    return {
      model: defaultName,
      repo: repoStr(spec.repo, spec.sub),
      reason: `no letters detected in state; using default (${defaultName})`,
      detection: det,
      workflow,
    };
  }
  if (det.script !== 'latin') {
    const spec = DEFAULT_MODELS['multilingual'];
    const frac = typeof det.non_latin_fraction === 'number' ? det.non_latin_fraction : 0;
    return {
      model: 'multilingual',
      repo: repoStr(spec.repo, spec.sub),
      reason: `non-Latin script (${det.script}, ${Math.round(100 * frac)}% of letters); the English checkpoint cannot read it`,
      detection: det,
      workflow,
    };
  }
  if (!det.is_english) {
    const spec = DEFAULT_MODELS['multilingual'];
    return {
      model: 'multilingual',
      repo: repoStr(spec.repo, spec.sub),
      reason: `Latin script but language looks like ${JSON.stringify(det.language)}, not English`,
      detection: det,
      workflow,
    };
  }
  const spec = DEFAULT_MODELS['english'];
  return {
    model: 'english',
    repo: repoStr(spec.repo, spec.sub),
    reason: 'English Latin text',
    detection: det,
    workflow,
  };
}
