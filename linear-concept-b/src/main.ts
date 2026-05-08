// src/main.ts
// ── Static imports: lightweight only (no TensorFlow, p5, OpenCV dep chains) ──
import {
  createInitialState,
  transition,
  type AppState,
  type AppAction,
} from "./core/appstatemachine.ts";
import { TabletopUI } from "./components/tabletopui.ts";
import { camToBoard } from "./utils/helpers.ts";
import type { Matrix2x2, Point2D } from "./types/index.ts";

const CAM_W = 1280;
const CAM_H = 720;
const CAM_FPS = 30;
const BOARD_HALF = 10;
const STABLE_THRESHOLD = 15;
const ABSENT_THRESHOLD = 8;
const OPENCV_TIMEOUT = 10_000;

// ---------------------------------------------------------------------------
// UI renders immediately on load (no dependencies).
// Camera / vision / hands modules are loaded dynamically on demand.
// If any dynamic import fails the UI stays visible and functional.
// ---------------------------------------------------------------------------

window.addEventListener("load", () => {
  console.log("[LineAR] Starting UI…");

  const canvas = document.getElementById("main-canvas") as HTMLCanvasElement;
  const video = document.getElementById("video-input") as HTMLVideoElement;

  let state: AppState = createInitialState();

  function dispatch(action: AppAction): void {
    const next = transition(state, action);
    if (next === state) return;
    console.debug("[LineAR] Phase:", next.phase);
    state = next;
    ui.setAppliedMatrix(state.appliedMatrix);
    ui.setCorners(state.objectCorners);
    if (state.phase === "TRANSFORMED") {
      setTimeout(() => dispatch({ type: "TRANSFORMATION_DONE" }), 1800);
    }
  }

  // ── Canvas UI (always, no waiting) ────────────────────────────────────
  const ui = new TabletopUI(canvas);
  ui.resize(window.innerWidth, window.innerHeight);
  ui.onYes = () => dispatch({ type: "CONFIRM_YES" });
  ui.onNo = () => dispatch({ type: "CONFIRM_NO" });
  window.addEventListener("resize", () => ui.resize(window.innerWidth, window.innerHeight));

  (function renderLoop() {
    ui.draw(state);
    requestAnimationFrame(renderLoop);
  })();

  // ── Bootstrap camera/vision when OpenCV is ready (best-effort) ────────
  waitForOpenCV(OPENCV_TIMEOUT).then(() => {
    bootstrap(state, dispatch, ui, canvas, video).catch((err) =>
      console.warn("[LineAR] Camera/vision unavailable:", err),
    );
  });
});

// ---------------------------------------------------------------------------
// Wait for OpenCV.js to initialise (non-blocking)
// ---------------------------------------------------------------------------

function waitForOpenCV(maxMs: number): Promise<void> {
  return new Promise((resolve) => {
    if (window.cv?.Mat) return resolve();

    const done = (): void => {
      clearInterval(poll);
      resolve();
    };

    if (window.cv && typeof window.cv.onRuntimeInitialized === "function") {
      const orig = window.cv.onRuntimeInitialized;
      window.cv.onRuntimeInitialized = () => {
        try { orig(); } catch { /* ignore */ }
        done();
      };
    }

    const poll = setInterval(() => {
      if (window.cv?.Mat) done();
    }, 200);

    setTimeout(done, maxMs);
  });
}

// ---------------------------------------------------------------------------
// Optional camera / overlay / hands bootstrap (dynamic imports)
// ---------------------------------------------------------------------------

async function bootstrap(
  state: AppState,
  dispatch: (action: AppAction) => void,
  ui: TabletopUI,
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
): Promise<void> {
  console.log("[LineAR] Bootstrapping camera & vision…");
  let absentFrames = 0;

  // ── Dynamic imports (isolate failures) ──────────────────────────────
  const { ColorOverlay } = await safeImport<typeof import("./core/coloroverlay.ts")>(
    () => import("./core/coloroverlay.ts"),
  );
  const { CameraTracker } = await safeImport<typeof import("./core/cameratracker.ts")>(
    () => import("./core/cameratracker.ts"),
  );
  const { HandTracker } = await safeImport<typeof import("./core/handtracker.ts")>(
    () => import("./core/handtracker.ts"),
  );

  // ── Overlay (p5) ────────────────────────────────────────────────────
  const overlay = ColorOverlay ? new ColorOverlay() : null;
  if (overlay) {
    try { overlay.mount("p5-container", CAM_W, CAM_H); } catch (err) {
      console.warn("[LineAR] Overlay mount failed:", err);
    }
  }

  // ── Camera ──────────────────────────────────────────────────────────
  if (CameraTracker) {
    try {
      const processCanvas = document.createElement("canvas");
      const cameraTracker = new CameraTracker(video, processCanvas);
      await cameraTracker.start({ width: CAM_W, height: CAM_H, fps: CAM_FPS });

      cameraTracker.onFrame((objects) => {
        if (overlay) overlay.update(objects, []);

        if (objects.length === 0) {
          absentFrames++;
          if (absentFrames >= ABSENT_THRESHOLD) dispatch({ type: "OBJECTS_CLEARED" });
          return;
        }
        absentFrames = 0;
        dispatch({ type: "OBJECTS_DETECTED", payload: objects });

        if (state.phase === "OBJECT_DETECTED" && state.stableFrameCount >= STABLE_THRESHOLD) {
          const obj = objects[0]!;
          const { x, y, width, height } = obj.boundingBox;
          const tb = (p: Point2D) => camToBoard(p, CAM_W, CAM_H, BOARD_HALF);
          const corners: [Point2D, Point2D, Point2D, Point2D] = [
            tb({ x, y }),
            tb({ x: x + width, y }),
            tb({ x: x + width, y: y + height }),
            tb({ x, y: y + height }),
          ];
          dispatch({ type: "CORNERS_LOCKED", payload: { corners, center: tb(obj.center) } });
        }
      });
    } catch (err) {
      console.warn("[LineAR] Camera not available:", err);
    }
  }

  // ── Hands (best-effort) ─────────────────────────────────────────────
  if (HandTracker) {
    try {
      const handTracker = new HandTracker();
      await handTracker.init();
      handTracker.start(video);

      handTracker.onFrame((hands) => {
        if (overlay) overlay.update(state.detectedObjects, hands);
        if (!hands.length) return;
        const hand = hands[0]!;

        if (
          hand.gesture === "pinch" &&
          (state.phase === "SHOW_BASIS_VECTORS" || state.phase === "SHOW_CORNERS")
        ) {
          const lm = hand.landmarks;
          const kx = ((lm[8]!.x / CAM_W) * 2 - 1) * 1.5;
          const ky = ((lm[8]!.y / CAM_H) * 2 - 1) * 0.5;
          const m: Matrix2x2 = [1, kx, ky, 1];
          ui.setMatrix(m);
          dispatch({ type: "BASIS_ADJUSTED", payload: m });
        }
      });
    } catch (err) {
      console.warn("[LineAR] Hand tracker not available:", err);
    }
  }

  console.log("[LineAR] Ready.");
}

// ---------------------------------------------------------------------------
// Helper: try a dynamic import, return empty object on failure
// ---------------------------------------------------------------------------

async function safeImport<T>(factory: () => Promise<T>): Promise<Partial<T>> {
  try { return await factory(); } catch (err) {
    console.warn("[LineAR] Dynamic import failed:", err);
    return {};
  }
}
