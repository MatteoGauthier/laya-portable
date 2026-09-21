// JS port of upstream router.py. Precedence: explicit model > task >
// detected workflow (opt-in) > lang > detected script/language > default.
//
// Pure routing (route/normaliseName/types) lives in laya-routing.ts (no node
// imports, browser-safe). This module adds weight loading via laya-paths.ts
// (Node filesystem paths) and the multi-checkpoint Router.
import {
  matchTypedDecisionsWorkflow,
  normaliseName,
  route,
  type CheckpointName,
  type ModelName,
  type RouteDecision,
  type RouteDetection,
  type RouteOptions,
} from './laya-routing.ts';
import { CHECKPOINT_CONFIGS, CHECKPOINT_MODELS, CHECKPOINT_TOKENIZERS } from './laya-paths.ts';
import { LayaConfigError } from './laya-errors.ts';
import type { PredictResult, Questions } from './laya-types.ts';
import type { LayaOpenOptions } from './laya.ts';

export type { CheckpointName, ModelName, RouteDecision, RouteDetection, RouteOptions };
export { matchTypedDecisionsWorkflow, normaliseName, route };
export type ModelNameAlias = CheckpointName;

async function closeQuietly(agent: RouterAgent | undefined): Promise<void> {
  try {
    await agent?.close?.();
  } catch {
    // ignore close errors on eviction/unload
  }
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
      await closeQuietly(agent);
    }
  }

  checkpointPaths(name: ModelName): LayaOpenOptions {
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
        await closeQuietly(agent);
      }
      this._agents.clear();
      this._order = [];
      return;
    }
    const key = normaliseName(name);
    const agent = this._agents.get(key);
    this._agents.delete(key);
    this._order = this._order.filter((k) => k !== key);
    await closeQuietly(agent);
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
