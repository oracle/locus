import { defineConfig } from "vite";

const BFF_TARGET = process.env.BFF_TARGET ?? "http://127.0.0.1:3101";

export default defineConfig({
  server: {
    port: 5173,
    host: "0.0.0.0",
    // Vite 5+ rejects requests from hosts not on this list. localhost
    // / 127.0.0.1 covers the local-dev path and the Docker image when
    // it's accessed at http://localhost:5173. For any other host
    // (custom domain, remote tunnel, etc.) pass a comma-separated
    // VITE_ALLOWED_HOSTS env var.
    allowedHosts: [
      "localhost",
      "127.0.0.1",
      ...(process.env.VITE_ALLOWED_HOSTS ?? "").split(",").filter(Boolean),
    ],
    proxy: {
      "^/api/.*": {
        target: BFF_TARGET,
        changeOrigin: true,
      },
    },
  },
});
