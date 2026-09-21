// Repo locations: package moves only touch this file.
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

/** packages/laya-js. */
export const PACKAGE_DIR = resolve(here, '..');
/** Repo root. */
export const REPO_ROOT = resolve(here, '..', '..', '..');
/** Fixtures + weights shared with Python and playground. */
export const VECTORS_DIR = join(REPO_ROOT, 'packages', 'test-vectors', 'vectors');
/** Parity/accuracy reports. */
export const REPORTS_DIR = join(REPO_ROOT, 'packages', 'test-vectors', 'reports');
/** Local gitignored ONNX artifacts. */
export const MODELS_DIR = join(REPO_ROOT, 'models');

/** Default model (single-file split FP32). CLI/worker/checks must match. */
export const DEFAULT_MODEL = join(MODELS_DIR, 'laya-split-single.onnx');

/** Multi-checkpoint layout (built by export_split.py --checkpoint). */
export type CheckpointName = 'english' | 'multilingual' | 'typed-decisions';

export const CHECKPOINT_MODELS: Record<CheckpointName, string> = {
  english: DEFAULT_MODEL,
  multilingual: join(MODELS_DIR, 'laya-multilingual-split-single.onnx'),
  'typed-decisions': join(MODELS_DIR, 'laya-typed-decisions-split-single.onnx'),
};

export const CHECKPOINT_TOKENIZERS: Record<CheckpointName, string> = {
  english: join(PACKAGE_DIR, 'src', 'tokenizer', 'tokenizer.json'),
  multilingual: join(MODELS_DIR, 'laya-multilingual.tokenizer.json'),
  'typed-decisions': join(MODELS_DIR, 'laya-typed-decisions.tokenizer.json'),
};

export const CHECKPOINT_CONFIGS: Record<CheckpointName, string> = {
  english: join(PACKAGE_DIR, 'src', 'tokenizer', 'rl_agent_config.json'),
  multilingual: join(MODELS_DIR, 'laya-multilingual.rl_agent_config.json'),
  'typed-decisions': join(MODELS_DIR, 'laya-typed-decisions.rl_agent_config.json'),
};

export const CHECKPOINT_ACT_BIN: Record<CheckpointName, string> = {
  english: join(VECTORS_DIR, 'act_head.bin'),
  multilingual: join(MODELS_DIR, 'laya-multilingual.act_head.bin'),
  'typed-decisions': join(MODELS_DIR, 'laya-typed-decisions.act_head.bin'),
};

export const CHECKPOINT_ACT_META: Record<CheckpointName, string> = {
  english: join(VECTORS_DIR, 'act_head.meta.json'),
  multilingual: join(MODELS_DIR, 'laya-multilingual.act_head.meta.json'),
  'typed-decisions': join(MODELS_DIR, 'laya-typed-decisions.act_head.meta.json'),
};
export const TOKENIZER_DIR = join(PACKAGE_DIR, 'src', 'tokenizer');
export const DEFAULT_TOKENIZER = join(TOKENIZER_DIR, 'tokenizer.json');
export const DEFAULT_CONFIG = join(TOKENIZER_DIR, 'rl_agent_config.json');
export const DEFAULT_ACT_HEAD_JSON = join(VECTORS_DIR, 'act_head.json');
export const DEFAULT_ACT_HEAD_BIN = join(VECTORS_DIR, 'act_head.bin');
export const DEFAULT_ACT_HEAD_META = join(VECTORS_DIR, 'act_head.meta.json');
