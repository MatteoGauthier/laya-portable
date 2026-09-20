// @ts-nocheck — version gate (CI): top-level ORT must be exactly 1.30.0.
// NOTE: @huggingface/transformers@4.3.0 nests its own onnxruntime-web
// 1.31.0-dev (exact pin, unreachable from our tokenizer-only usage in
// check-tokenizer.mjs). npm overrides cannot force that nested copy, so gate
// on the copies we actually load: onnxruntime-node (SDK/checks) and
// onnxruntime-web (playground bundle).
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
function versionOf(name) {
  // Resolve via main entry (some packages don't export ./package.json);
  // walk up until package.json is found.
  let dir = dirname(require.resolve(name));
  const { existsSync } = require('node:fs');
  for (let i = 0; i < 6; i++) {
    const cand = join(dir, 'package.json');
    if (existsSync(cand)) {
      const pkg = JSON.parse(readFileSync(cand, 'utf8'));
      // Skip nested stubs (e.g. dist/cjs/package.json {"type":...}); the real
      // manifest carries our package name.
      if (pkg.name === name) return pkg.version;
    }
    dir = join(dir, '..');
  }
  throw new Error(`package.json not found for ${name}`);
}
const want = '1.30.0';
let ok = true;
for (const name of ['onnxruntime-node', 'onnxruntime-web', 'onnxruntime-common']) {
  const v = versionOf(name);
  const pass = v === want;
  ok &&= pass;
  console.log(`${name}@${v} -> ${pass ? 'OK' : `FAIL (want ${want})`}`);
}
process.exitCode = ok ? 0 : 1;
