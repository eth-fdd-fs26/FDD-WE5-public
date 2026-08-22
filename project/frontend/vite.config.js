import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The frontend talks to the FastAPI backend directly (CORS is enabled there).
// Override the backend location with VITE_API_BASE if you run it elsewhere.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})
