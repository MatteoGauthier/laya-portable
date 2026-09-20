// Central repo-location constants. Keeps DEFAULT_MODEL / vectors / reports
// paths in one place so package moves only touch this file.
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

/** packages/laya-js (this package). */
export const PACKAGE_DIR = resolve(here, '..');
/** Repo root. */
export const REPO_ROOT = resolve(here, '..', '..', '..');
/** Generated fixtures + weights shared with Python and playground. */
export const VECTORS_DIR = join(REPO_ROOT, 'packages', 'test-vectors', 'vectors');
/** Generated parity/accuracy reports. */
export const REPORTS_DIR = join(REPO_ROOT, 'packages', 'test-vectors', 'reports');
/** Local gitignored ONNX artifacts. */
export const MODELS_DIR = join(REPO_ROOT, 'models');

/** Canonical default model (single-file split FP32). CLI/worker/checks must match. */
export const DEFAULT_MODEL = join(MODELS_DIR, 'laya-split-single.onnx');
export const TOKENIZER_DIR = join(PACKAGE_DIR, 'src', 'tokenizer');
export const DEFAULT_TOKENIZER = join(TOKENIZER_DIR, 'tokenizer.json');
export const DEFAULT_CONFIG = join(TOKENIZER_DIR, 'rl_agent_config.json');
export const DEFAULT_ACT_HEAD_JSON = join(VECTORS_DIR, 'act_head.json');
export const DEFAULT_ACT_HEAD_BIN = join(VECTORS_DIR, 'act_head.bin');
export const DEFAULT_ACT_HEAD_META = join(VECTORS_DIR, 'act_head.meta.json');
