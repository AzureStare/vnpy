import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";

// Build output goes directly to `flagship/app_console/site/` (served by Caddy).
// We keep `emptyOutDir=false` to avoid removing `login.html` and any other static helpers.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: path.resolve(__dirname, "../site"),
    emptyOutDir: false,
    assetsDir: "assets",
    rollupOptions: {
      input: path.resolve(__dirname, "index.html"),
    },
  },
});
