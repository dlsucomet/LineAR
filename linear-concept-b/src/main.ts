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
import {
  VIRTUAL_OBJECT,
  PRESET_MATRIX,
  DRAG_TARGET,
  GHOST_ARROWS,
  PHASE_TIMINGS,
  HOVER_DWELL_MS,
  GRAB_RADIUS_PX,
  ARROW_GRAB_RADIUS_PX,
  ARROW_SNAP_PX,
  DEMO_INSTRUCTIONS,
  PRESET_CORNERS_PAYLOAD,
} from "./core/demoplayer.ts";
import type { DetectedHand, HandLandmark, Matrix2x2, Point2D } from "./types/index.ts";

const CAM_W = 1280;
const CAM_H = 720;
const CAM_FPS = 30;
const BOARD_HALF = 10;
const STABLE_THRESHOLD = 15;
const ABSENT_THRESHOLD = 8;
const OPENCV_TIMEOUT = 10_000;
const FINGER_GUIDED_DEMO = true;

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

    if (FINGER_GUIDED_DEMO) {
      if (state.phase === "OBJECT_DETECTED") {
        setTimeout(() => dispatch({ type: "CORNERS_LOCKED", payload: PRESET_CORNERS_PAYLOAD }), PHASE_TIMINGS.OBJECT_DETECTED);
      }
      if (state.phase === "SHOW_CORNERS") {
        setTimeout(() => dispatch({ type: "PHASE_ADVANCE" }), PHASE_TIMINGS.SHOW_CORNERS);
      }
    } else {
      if (state.phase === "TRANSFORMED") {
        setTimeout(() => dispatch({ type: "TRANSFORMATION_DONE" }), 1800);
      }
    }
  }

  // ── Canvas UI (always, no waiting) ────────────────────────────────────
  const ui = new TabletopUI(canvas);
  ui.resize(window.innerWidth, window.innerHeight);
  ui.onYes = () => dispatch({ type: "CONFIRM_YES" });
  ui.onNo = () => dispatch({ type: "CONFIRM_NO" });
  window.addEventListener("resize", () => ui.resize(window.innerWidth, window.innerHeight));

  // ── Demo-specific setup ──────────────────────────────────────────────
  if (FINGER_GUIDED_DEMO) {
    ui.setDemoInstructions(DEMO_INSTRUCTIONS);
    ui.setCorners(VIRTUAL_OBJECT.corners);
    ui.setGhostArrows(GHOST_ARROWS.e1, GHOST_ARROWS.e2);
    ui.setDragTarget(DRAG_TARGET.targetPos);
    ui.onStart = () => dispatch({ type: "OBJECTS_DETECTED", payload: [] });
    ui.onContinue = () => dispatch({ type: "TRANSFORMATION_DONE" });
  }

  (function renderLoop() {
    ui.draw(state);
    requestAnimationFrame(renderLoop);
  })();

  // ── Bootstrap ────────────────────────────────────────────────────────
  if (FINGER_GUIDED_DEMO) {
    startCamera(video, CAM_W, CAM_H).then(() => {
      setupDemoTracker(state, dispatch, ui, canvas, video);
    }).catch((err) =>
      console.warn("[LineAR] Demo camera unavailable:", err),
    );
  } else {
    waitForOpenCV(OPENCV_TIMEOUT).then(() => {
      bootstrap(state, dispatch, ui, canvas, video).catch((err) =>
        console.warn("[LineAR] Camera/vision unavailable:", err),
      );
    });
  }
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

// ============================================================================
// Finger-Guided Demo Mode
// ============================================================================

/** Shared mutable state for the demo interaction handlers. */
interface DemoInteractionState {
  dwellStart: number;
  isDraggingCorner: boolean;
  isDraggingE1: boolean;
  isDraggingE2: boolean;
  e1Snapped: boolean;
  e2Snapped: boolean;
  arrowTransitionFired: boolean;
  previousPhase: string | null;
  arrowMatrix: Matrix2x2;
}

function createDemoState(): DemoInteractionState {
  return {
    dwellStart: 0,
    isDraggingCorner: false,
    isDraggingE1: false,
    isDraggingE2: false,
    e1Snapped: false,
    e2Snapped: false,
    arrowTransitionFired: false,
    previousPhase: null,
    arrowMatrix: [1, 0, 0, 1],
  };
}

// ── Camera starter ───────────────────────────────────────────────────────────

async function startCamera(
  video: HTMLVideoElement,
  width: number,
  height: number,
): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width, height, frameRate: CAM_FPS },
  });
  video.srcObject = stream;
  await new Promise<void>((resolve) => {
    video.onloadedmetadata = () => resolve();
  });
  await video.play();
}

// ── Demo bootstrap ───────────────────────────────────────────────────────────

async function setupDemoTracker(
  state: AppState,
  dispatch: (action: AppAction) => void,
  ui: TabletopUI,
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
): Promise<void> {
  console.log("[LineAR] Starting demo hand tracker…");

  const { HandTracker } = await safeImport<typeof import("./core/handtracker.ts")>(
    () => import("./core/handtracker.ts"),
  );
  if (!HandTracker) {
    console.warn("[LineAR] Hand tracker not available");
    return;
  }

  const handTracker = new HandTracker({ lite: true, maxHands: 1 });
  await handTracker.init();
  handTracker.start(video);

  const ds = createDemoState();

  handTracker.onFrame((hands) => {
    handleDemoFrame(hands, state, dispatch, ui, canvas, video, ds);
  });

  console.log("[LineAR] Demo ready.");
}

// ── Frame handler ────────────────────────────────────────────────────────────

function handleDemoFrame(
  hands: DetectedHand[],
  state: AppState,
  dispatch: (action: AppAction) => void,
  ui: TabletopUI,
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  ds: DemoInteractionState,
): void {
  // Reset per-phase state on phase transitions
  if (state.phase !== ds.previousPhase) {
    ds.previousPhase = state.phase;
    if (state.phase === "SHOW_BASIS_VECTORS") {
      ds.e1Snapped = false;
      ds.e2Snapped = false;
      ds.isDraggingE1 = false;
      ds.isDraggingE2 = false;
      ds.arrowTransitionFired = false;
      ds.arrowMatrix = [1, 0, 0, 1];
      ui.setArrowSnapped(false, false);
      ui.setMatrix([1, 0, 0, 1]);
    }
  }

  // Map landmark pixel coords to mirrored canvas coords
  const vw = video.videoWidth || CAM_W;
  const vh = video.videoHeight || CAM_H;
  if (!vw || !vh) return;
  const toMirroredCanvas = (lm: HandLandmark): Point2D => ({
    x: (1 - lm.x / vw) * canvas.width,
    y: (lm.y / vh) * canvas.height,
  });

  // Pass mirrored hand data for skeleton visualization
  if (hands.length > 0) {
    const mirroredHands: DetectedHand[] = hands.map((h) => ({
      handedness: h.handedness,
      score: h.score,
      gesture: h.gesture,
      landmarks: h.landmarks.map((lm) => ({ ...lm, ...toMirroredCanvas(lm) })),
    }));
    ui.setRawHands(mirroredHands);
  } else {
    ui.setRawHands(null);
  }

  if (!hands.length) {
    ui.setFingerPosition(null);
    return;
  }

  const hand = hands[0]!;
  if (hand.score < 0.5) {
    ui.setFingerPosition(null);
    return;
  }
  if (!hand.landmarks[8]) return;

  const lm = hand.landmarks[8]!;
  const fingerCanvas = toMirroredCanvas(lm);
  ui.setFingerPosition(fingerCanvas);

  // Only skip phase-specific interactions if grid isn't available yet
  const grid = ui.getLastGridRect();
  if (!grid) return;

  switch (state.phase) {
    case "WAITING_FOR_OBJECT": {
      const r = ui.getButtonRects();
      checkDwellTap(fingerCanvas, r.start, () => dispatch({ type: "OBJECTS_DETECTED", payload: [] }), ui, ds);
      break;
    }
    case "POINTS_CALCULATED": {
      handleCornerDrag(hand, fingerCanvas, dispatch, ui, grid, ds);
      break;
    }
    case "SHOW_BASIS_VECTORS": {
      handleArrowDrag(hand, fingerCanvas, dispatch, ui, grid, ds);
      break;
    }
    case "CONFIRM_TRANSFORM": {
      const r = ui.getButtonRects();
      checkDwellTap(fingerCanvas, r.yes, () => dispatch({ type: "CONFIRM_YES" }), ui, ds);
      break;
    }
    case "TRANSFORMED": {
      const r = ui.getButtonRects();
      checkDwellTap(fingerCanvas, r.continue, () => dispatch({ type: "TRANSFORMATION_DONE" }), ui, ds);
      break;
    }
    case "CONFIRM_RESET": {
      const r = ui.getButtonRects();
      checkDwellTap(fingerCanvas, r.yes, () => dispatch({ type: "CONFIRM_YES" }), ui, ds);
      break;
    }
  }
}

// ── Dwell tap ────────────────────────────────────────────────────────────────

function checkDwellTap(
  fingerPos: Point2D,
  rect: DOMRect | null,
  callback: () => void,
  ui: TabletopUI,
  ds: DemoInteractionState,
): void {
  const now = performance.now();
  const isOver = rect !== null &&
    fingerPos.x >= rect.x && fingerPos.x <= rect.x + rect.width &&
    fingerPos.y >= rect.y && fingerPos.y <= rect.y + rect.height;

  if (!isOver) {
    ds.dwellStart = 0;
    ui.setDwellProgress(0);
    return;
  }

  if (ds.dwellStart === 0) ds.dwellStart = now;

  const elapsed = now - ds.dwellStart;
  ui.setDwellProgress(Math.min(elapsed / HOVER_DWELL_MS, 1));

  if (elapsed >= HOVER_DWELL_MS) {
    ds.dwellStart = 0;
    ui.setDwellProgress(0);
    callback();
  }
}

// ── Corner drag (Phase 3: POINTS_CALCULATED) ────────────────────────────────

function handleCornerDrag(
  hand: DetectedHand,
  fingerCanvas: Point2D,
  dispatch: (action: AppAction) => void,
  ui: TabletopUI,
  grid: DOMRect,
  ds: DemoInteractionState,
): void {
  const isPinching = hand.gesture === "pinch";
  const cornerA = VIRTUAL_OBJECT.corners[0];
  const cornerACanvas = ui.gridToCanvas(cornerA, grid);

  // Start drag: pinch near corner A
  if (!ds.isDraggingCorner && isPinching) {
    const dx = fingerCanvas.x - cornerACanvas.x;
    const dy = fingerCanvas.y - cornerACanvas.y;
    if (Math.sqrt(dx * dx + dy * dy) < GRAB_RADIUS_PX) {
      ds.isDraggingCorner = true;
    }
  }

  // During drag: corner follows finger
  if (ds.isDraggingCorner && isPinching) {
    const fingerGrid = ui.canvasToGrid(fingerCanvas, grid);
    ui.setDraggedCorner(fingerGrid, true);
    const dx = fingerGrid.x - DRAG_TARGET.targetPos.x;
    const dy = fingerGrid.y - DRAG_TARGET.targetPos.y;
    ui.setCornerInTarget(Math.sqrt(dx * dx + dy * dy) < DRAG_TARGET.snapRadius);
  }

  // Release: snap or snap back
  if (ds.isDraggingCorner && !isPinching) {
    ds.isDraggingCorner = false;
    const fingerGrid = ui.canvasToGrid(fingerCanvas, grid);
    const dx = fingerGrid.x - DRAG_TARGET.targetPos.x;
    const dy = fingerGrid.y - DRAG_TARGET.targetPos.y;
    ui.setDraggedCorner(null, false);
    ui.setCornerInTarget(false);

    if (Math.sqrt(dx * dx + dy * dy) < DRAG_TARGET.snapRadius) {
      dispatch({ type: "OBJECTS_CLEARED" });
    }
  }
}

// ── Basis arrow drag (Phase 5: SHOW_BASIS_VECTORS) ──────────────────────────

function handleArrowDrag(
  hand: DetectedHand,
  fingerCanvas: Point2D,
  dispatch: (action: AppAction) => void,
  ui: TabletopUI,
  grid: DOMRect,
  ds: DemoInteractionState,
): void {
  if (ds.arrowTransitionFired) return;

  const isPinching = hand.gesture === "pinch";
  const cx = grid.x + grid.width * 0.45;
  const cy = grid.y + grid.height * 0.52;
  const scale = Math.min(grid.width, grid.height) * 0.13;

  // Use persistent matrix so e1 position is preserved when dragging e2
  const mat = ds.arrowMatrix;

  // Ghost arrow canvas positions
  const ghostE1: Point2D = {
    x: cx + GHOST_ARROWS.e1.x * scale,
    y: cy - GHOST_ARROWS.e1.y * scale,
  };
  const ghostE2: Point2D = {
    x: cx + GHOST_ARROWS.e2.x * scale,
    y: cy - GHOST_ARROWS.e2.y * scale,
  };

  // ── Grab e1 ──
  if (!ds.isDraggingE1 && !ds.isDraggingE2 && isPinching && !ds.e1Snapped) {
    const curE1 = { x: cx + mat[0] * scale, y: cy - mat[2] * scale };
    const dx = fingerCanvas.x - curE1.x;
    const dy = fingerCanvas.y - curE1.y;
    if (Math.sqrt(dx * dx + dy * dy) < ARROW_GRAB_RADIUS_PX) {
      ds.isDraggingE1 = true;
    }
  }

  // ── Drag e1 ──
  if (ds.isDraggingE1 && isPinching && !ds.e1Snapped) {
    mat[0] = (fingerCanvas.x - cx) / scale;
    mat[2] = -(fingerCanvas.y - cy) / scale;

    const dx = fingerCanvas.x - ghostE1.x;
    const dy = fingerCanvas.y - ghostE1.y;
    if (Math.sqrt(dx * dx + dy * dy) < ARROW_SNAP_PX) {
      mat[0] = GHOST_ARROWS.e1.x;
      mat[2] = GHOST_ARROWS.e1.y;
      ds.e1Snapped = true;
      ds.isDraggingE1 = false;
      ui.setArrowSnapped(true, ds.e2Snapped);
    }
    ui.setMatrix([...mat]);
  }

  // ── Grab e2 (only after e1 is snapped) ──
  if (!ds.isDraggingE1 && !ds.isDraggingE2 && isPinching && ds.e1Snapped && !ds.e2Snapped) {
    const curE2 = { x: cx + mat[1] * scale, y: cy - mat[3] * scale };
    const dx = fingerCanvas.x - curE2.x;
    const dy = fingerCanvas.y - curE2.y;
    if (Math.sqrt(dx * dx + dy * dy) < ARROW_GRAB_RADIUS_PX) {
      ds.isDraggingE2 = true;
    }
  }

  // ── Drag e2 ──
  if (ds.isDraggingE2 && isPinching && !ds.e2Snapped) {
    mat[1] = (fingerCanvas.x - cx) / scale;
    mat[3] = -(fingerCanvas.y - cy) / scale;

    const dx = fingerCanvas.x - ghostE2.x;
    const dy = fingerCanvas.y - ghostE2.y;
    if (Math.sqrt(dx * dx + dy * dy) < ARROW_SNAP_PX) {
      mat[1] = GHOST_ARROWS.e2.x;
      mat[3] = GHOST_ARROWS.e2.y;
      ds.e2Snapped = true;
      ds.isDraggingE2 = false;
      ui.setArrowSnapped(ds.e1Snapped, true);
    }
    ui.setMatrix([...mat]);
  }

  // ── Release without snap ──
  if ((ds.isDraggingE1 || ds.isDraggingE2) && !isPinching) {
    ds.isDraggingE1 = false;
    ds.isDraggingE2 = false;
  }

  // ── Both snapped → advance ──
  if (ds.e1Snapped && ds.e2Snapped && !ds.arrowTransitionFired) {
    ds.arrowTransitionFired = true;
    ui.setMatrix([...PRESET_MATRIX]);
    setTimeout(() => {
      dispatch({ type: "BASIS_ADJUSTED", payload: PRESET_MATRIX });
    }, 300);
  }
}
