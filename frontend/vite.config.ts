import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // The backend is local-only; proxying keeps the frontend origin-relative.
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
})
