import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev server proxies the API / SSE / WS / plot-stream to the FastAPI backend on
// :8138 so the app runs single-origin in dev. In production the backend serves the
// built dist/ directly (app.py), so no proxy is needed.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5180,
    strictPort: false,
    proxy: {
      "/api": { target: "http://127.0.0.1:8138", changeOrigin: true },
      "/stream": { target: "http://127.0.0.1:8138", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8138", ws: true },
    },
  },
});
