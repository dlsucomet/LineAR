import { defineConfig } from "vite";
import path from "path";

export default defineConfig({

  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
    target: "esnext",
  },
  resolve: {
    alias: {
      "@mediapipe/hands": path.resolve(__dirname, "src/shims/mediapipe-hands.ts"),
    },
  },
  server: {
    port: 3000,
    open: true
  },
  optimizeDeps: {
    include: ["@tensorflow/tfjs", "ml-matrix", "p5"],
    exclude: ['@mediapipe/hands', '@tensorflow-models/hand-pose-detection']
  }

});
