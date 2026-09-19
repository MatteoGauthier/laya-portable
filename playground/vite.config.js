import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // allow importing shared ../js/*.mjs modules into the playground
    fs: { allow: ['..'] },
  },
})
