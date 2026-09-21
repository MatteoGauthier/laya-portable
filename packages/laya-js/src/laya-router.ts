// JS port of upstream/laya/laya/router.py routing (B1).
// Pure decision logic is dependency-free and matches Python precedence:
// explicit model > explicit task > detected workflow (opt-in) > explicit lang
// > detected script/language > default.
//
// Model loading is async (LayaClient.open) unlike Python's sync Agent load,
// so Router.load/preload are async with the same LRU semantics (max_loaded).
// Per-checkpoint tokenizer/config/actHead paths come from laya-paths.ts
// CHECKPOINT_* maps (B2 artifacts); english works today, multilingual and
// typed-decisions throw a clear missing-artifact error until exported.
import { analyse, type AnalyseResult } from './laya-lang.ts';
import {
  CHECKPOINT_ACT_BIN,
  CHECKPOINT_ACT_META,
  CHECKPOINT_CONFIGS,
  CHECKPOINT_MODELS,
  CHECKPOINT_TOKENIZERS,
  type CheckpointName,
} from './laya-paths.ts';
import { LayaConfigError } from './laya-errors.ts';
import type { PredictResult, Questions } from './laya-types.ts';
import type { LayaOpenOptions } from './laya.ts';

export type { CheckpointName };
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

// Minimal client surface Router needs (LayaClient satisfies this).
export interface RouterAgent {
  predict(state: unknown, questions: Questions): Promise<PredictResult>;
  close?(): Promise<void>;
}

export interface RouterOptions {
  models?: Partial<Record<ModelName, LayaOpenOptions>>;
  default?: string;
  auto_task_detection?: boolean;
  max_loaded?: number;
  preload?: boolean;
  /** Override for tests / custom runtimes. Defaults to LayaClient.open per checkpoint. */
  open?: (name: ModelName, opts: LayaOpenOptions) => Promise<RouterAgent>;
}

async function defaultOpen(name: ModelName, overrides: LayaOpenOptions): Promise<RouterAgent> {
  const { LayaClient } = await import('./laya.ts');
  const base: LayaOpenOptions = {
    model: CHECKPOINT_MODELS[name],
    tokenizer: CHECKPOINT_TOKENIZERS[name],
    config: CHECKPOINT_CONFIGS[name],
  };
  const client = await LayaClient.open({ ...base, ...overrides });
  return client as unknown as RouterAgent;
}

export class Router {
  models: Partial<Record<ModelName, LayaOpenOptions>>;
  default: ModelName;
  auto_task_detection: boolean;
  max_loaded: number;
  private _agents = new Map<ModelName, RouterAgent>();
  private _order: ModelName[] = [];
  private _open: (name: ModelName, opts: LayaOpenOptions) => Promise<RouterAgent>;

  constructor(opts: RouterOptions = {}) {
    this.models = { ...(opts.models ?? {}) };
    this.default = normaliseName(opts.default ?? 'english');
    this.auto_task_detection = Boolean(opts.auto_task_detection ?? false);
    this.max_loaded = Math.max(1, Math.trunc(opts.max_loaded ?? 1));
    this._open = opts.open ?? defaultOpen;
    if (opts.preload) {
      throw new LayaConfigError('Router({ preload: true }) is async in JS — await router.preload() instead');
    }
  }

  get loaded(): ModelName[] {
    return [...this._order];
  }

  private _touch(key: ModelName): void {
    this._order = this._order.filter((k) => k !== key);
    this._order.push(key);
  }

  private async _evict(): Promise<void> {
    while (this._order.length > this.max_loaded) {
      const victim = this._order.shift();
      if (victim === undefined) break;
      const agent = this._agents.get(victim);
      this._agents.delete(victim);
      try {
        await agent?.close?.();
      } catch {
        /* ignore close errors on eviction */
      }
    }
  }

  checkpointPaths(name: ModelName): LayaOpenOptions {
    // B2 artifacts live next to models/; english paths always exist.
    return {
      model: CHECKPOINT_MODELS[name],
      tokenizer: CHECKPOINT_TOKENIZERS[name],
      config: CHECKPOINT_CONFIGS[name],
      actHead: undefined,
    };
  }

  async load(name: string): Promise<RouterAgent> {
    const key = normaliseName(name);
    const hit = this._agents.get(key);
    if (hit) {
      this._touch(key);
      return hit;
    }
    const overrides = this.models[key] ?? {};
    // Route actHead bin/meta via laya-paths convention: LayaClient.open prefers
    // DEFAULT_ACT_HEAD_BIN/META (english) — per-checkpoint bins are wired by
    // passing actHead explicitly once B2 emits models/laya-*.act_head.bin.
    // Until then multilingual/typed-decisions fail loudly on missing files.
    const perCheckpointActBin = key === 'english' ? undefined : (CHECKPOINT_ACT_BIN[key] as string | undefined);
    void perCheckpointActBin;
    void CHECKPOINT_ACT_META;
    const agent = await this._open(key, { ...this.checkpointPaths(key), ...overrides });
    this._agents.set(key, agent);
    this._touch(key);
    await this._evict();
    return agent;
  }

  attach(name: string, agent: RouterAgent): RouterAgent {
    const key = normaliseName(name);
    this._agents.set(key, agent);
    this._touch(key);
    this.max_loaded = Math.max(this.max_loaded, this._agents.size);
    return agent;
  }

  async preload(names?: string[]): Promise<this> {
    const list = (names ?? ['english', 'multilingual', 'typed-decisions']).map(normaliseName);
    this.max_loaded = Math.max(this.max_loaded, list.length, this._agents.size);
    for (const n of list) {
      if (!this._agents.has(n)) await this.load(n);
    }
    return this;
  }

  async unload(name?: string): Promise<void> {
    if (name === undefined || name === null) {
      for (const agent of this._agents.values()) {
        try {
          await agent.close?.();
        } catch {
          /* ignore */
        }
      }
      this._agents.clear();
      this._order = [];
      return;
    }
    const key = normaliseName(name);
    const agent = this._agents.get(key);
    this._agents.delete(key);
    this._order = this._order.filter((k) => k !== key);
    try {
      await agent?.close?.();
    } catch {
      /* ignore */
    }
  }

  routeDecision(state: unknown, questions?: Questions | null, opts: RouteOptions = {}): RouteDecision {
    return route(state, questions ?? {}, opts, this.default, this.auto_task_detection);
  }

  async predict(
    state: unknown,
    questions: Questions,
    opts: RouteOptions = {},
  ): Promise<PredictResult & { routing: RouteDecision }> {
    const decision = this.routeDecision(state, questions, opts);
    const agent = await this.load(decision.model);
    const result = await agent.predict(state, questions);
    return { ...result, routing: decision } as PredictResult & { routing: RouteDecision };
  }
}
