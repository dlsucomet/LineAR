import { defineConfig } from "vite";

export default defineConfig({

  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
    target: "esnext",
  },
  server: {
    port: 3000,
    open: true
  },
  optimizeDeps: {
    include: ["@tensorflow/tfjs", "@tensorflow-models/hand-pose-detection", "ml-matrix", "p5"],
    exclude: ['@mediapipe/hands', '@tensorflow-models/hand-pose-detection']
  }

});
