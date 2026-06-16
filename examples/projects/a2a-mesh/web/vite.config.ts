import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite proxies API calls so the React app can hit /agent-card and /a2a/*
// without CORS gymnastics. The two A2A servers run on 8001 and 8002 — we
// route via path prefixes the UI controls.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api/research": {
        target: "http://127.0.0.1:8001",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/research/, ""),
      },
      "/api/finance": {
        target: "http://127.0.0.1:8002",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/finance/, ""),
      },
    },
  },
});
