import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Allow frontend dev server (5173) to talk to backend (8000) without CORS pain.
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/research': 'http://127.0.0.1:8000',
      '/research/stream': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
