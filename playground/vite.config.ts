import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { defineConfig } from 'vite';

const root = dirname(fileURLToPath(import.meta.url));

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(root, 'src'),
      '@js': resolve(root, '..', 'js'),
    },
  },
  server: {
    // allow importing shared ../js/*.mjs modules into the playground
    fs: { allow: ['..'] },
  },
  worker: { format: 'es' },
  build: {
    target: 'baseline-widely-available',
    sourcemap: true,
  },
});
