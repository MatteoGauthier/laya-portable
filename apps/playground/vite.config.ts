import react from '@vitejs/plugin-react';
import { createReadStream, promises as fs } from 'node:fs';
import { dirname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import type { ViteDevServer } from 'vite';

const root = dirname(fileURLToPath(import.meta.url));
// Local gitignored ONNX artifacts (never bundled). Served in dev only;
// production must host models at /models.
const MODELS_DIR = resolve(root, '..', '..', 'models');

function modelsStatic(): { name: string; configureServer(server: ViteDevServer): void } {
  return {
    name: 'laya-models-static',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith('/models/')) {
          next();
          return;
        }
        const file = normalize(join(MODELS_DIR, decodeURIComponent(req.url.slice('/models/'.length))));
        if (!file.startsWith(MODELS_DIR)) {
          res.statusCode = 403;
          res.end('forbidden');
          return;
        }
        fs.stat(file)
          .then((stat) => {
            if (stat.isDirectory()) {
              res.statusCode = 404;
              res.end('not found');
              return;
            }
            res.setHeader('Content-Length', stat.size);
            res.setHeader('Accept-Ranges', 'bytes');
            createReadStream(file).pipe(res);
          })
          .catch(() => {
            res.statusCode = 404;
            res.end('not found');
          });
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), modelsStatic()],
  resolve: {
    alias: {
      '@': resolve(root, 'src'),
    },
  },
  worker: { format: 'es' },
  build: {
    target: 'baseline-widely-available',
    sourcemap: true,
  },
});
