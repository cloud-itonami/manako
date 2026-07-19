import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: "../_svelte",
    emptyOutDir: true,
    target: "esnext",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  worker: {
    format: "es",
  },
});
