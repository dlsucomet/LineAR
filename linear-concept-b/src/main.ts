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
const FINGER_GUIDED_DEMO = false;
const EPSILON = 0.001;
const GRID_BOUNDS = 15;

// ── Corner Auto-Release & Cooldown Configuration ───────────────────────────────
// Auto-release: 3 seconds of stable position after snapping to lock corner
// Cooldown: 4 seconds before same corner can be grabbed again after snap/lock
// Move threshold: finger must stay within 10px of snap position to keep timer running
const CORNER_LOCK_DELAY_MS = 3000;
const CORNER_COOLDOWN_MS = 4000;
const CORNER_MOVE_THRESHOLD_PX = 10;

// Future consideration: Frame boundary - 50px buffer from canvas edges to prevent
// accidental grabbing when reaching near screen boundaries (not currently implemented)

// ── Centralised Input Streams for Unified Loop ─────────────────────────────
let latestHandPosition: Point2D | null = null;
let lastHandSeenTime = 0;
const HAND_TIMEOUT_MS = 400; // Hand tracking grace period before falling back to mouse

// Current app phase (updated in dispatch, used by hand tracker callback)
let currentPhase: string = "WAITING_FOR_OBJECT";
let currentStableCount = 0;

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
    currentPhase = state.phase;
    currentStableCount = state.stableFrameCount;
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
      // Auto-advance from POINTS_CALCULATED → SHOW_BASIS_VECTORS (skip corner adjustment)
      if (state.phase === "POINTS_CALCULATED") {
        setTimeout(() => dispatch({ type: "PHASE_ADVANCE" }), 1500);
      }
    }
  }

  // ── Canvas UI Setup ───────────────────────────────────────────────────
  const ui = new TabletopUI(canvas);
  ui.resize(window.innerWidth, window.innerHeight);
  ui.onYes = () => dispatch({ type: "CONFIRM_YES" });
  ui.onNo = () => dispatch({ type: "CONFIRM_NO" });
  window.addEventListener("resize", () => ui.resize(window.innerWidth, window.innerHeight));

  const ds = createDemoState();

  if (FINGER_GUIDED_DEMO) {
    ui.setDemoInstructions(DEMO_INSTRUCTIONS);
    ui.setCorners(VIRTUAL_OBJECT.corners);
    ui.setGhostArrows(GHOST_ARROWS.e1, GHOST_ARROWS.e2);
    ui.onDone = () => {
      ds.cornerCPlaced = false;
      ds.cornerDPlaced = false;
      ui.showDoneButton = false;
      dispatch({ type: "OBJECTS_CLEARED" });
    };
    ui.onContinue = () => dispatch({ type: "TRANSFORMATION_DONE" });
  }

  // ── Unified 60FPS Render & Interaction Loop ───────────────────────────
  (function renderLoop() {
    ui.draw(state);

    // 1. Track local state wipes on phase transitions
    if (state.phase !== ds.previousPhase) {
      ds.previousPhase = state.phase;
      if (state.phase === "POINTS_CALCULATED") {
        ds.isDraggingCornerA = false;
        ds.isDraggingCornerB = false;
        ds.isDraggingCornerC = false;
        ds.isDraggingCornerD = false;
        ds.cornerASnapped = false;
        ds.cornerBSnapped = false;
        ds.cornerCSnapped = false;
        ds.cornerDSnapped = false;
        ds.cornerALocked = false;
        ds.cornerBLocked = false;
        ds.cornerCLocked = false;
        ds.cornerDLocked = false;
        ds.cornerCPlaced = false;
        ds.cornerDPlaced = false;
        ds.showDonePrompt = false;
        ds.hasAdjustedCorner = false;
        ds.cornerASnapTime = 0;
        ds.cornerBSnapTime = 0;
        ds.cornerCSnapTime = 0;
        ds.cornerDSnapTime = 0;
        ds.cornerASnapPos = null;
        ds.cornerBSnapPos = null;
        ds.cornerCSnapPos = null;
        ds.cornerDSnapPos = null;
        ds.cornerAReleaseTime = 0;
        ds.cornerBReleaseTime = 0;
        ds.cornerCReleaseTime = 0;
        ds.cornerDReleaseTime = 0;
        ds.globalCooldownUntil = 0;
        ui.showDoneButton = false;
        ui.setCornerLockedStates([false, false, false, false]);
        for (let i = 0; i < 4; i++) ui.setCornerDrag(i, null, false);
        // Store initial corner positions to detect adjustments
        if (state.objectCorners) {
          ds.initialCornerPositions = [...state.objectCorners];
        }
      }
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

    // 2. Resolve Pointer Hierarchy: Hand tracking wins, Mouse acts as a smart hover fallback
    let activePointer: Point2D | null = null;
    const isHandActive = latestHandPosition && (Date.now() - lastHandSeenTime < HAND_TIMEOUT_MS);

    if (isHandActive) {
      activePointer = latestHandPosition;
    } else {
      activePointer = ui.getMouseCanvasPos();
    }

    // 3. Process pointer metrics globally
    processInteractionFrame(activePointer, state, dispatch, ui, canvas, ds);

    requestAnimationFrame(renderLoop);
  })();

  // ── Bootstrap Camera Streams ──────────────────────────────────────────
  if (FINGER_GUIDED_DEMO) {
    startCamera(video, CAM_W, CAM_H).then(() => {
      setupDemoTracker(canvas, video, ui, ds);
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

async function bootstrap(
  state: AppState,
  dispatch: (action: AppAction) => void,
  ui: TabletopUI,
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
): Promise<void> {
  const { CameraTracker } = await safeImport<typeof import("./core/cameratracker.ts")>(() => import("./core/cameratracker.ts"));
  const { HandTracker } = await safeImport<typeof import("./core/handtracker.ts")>(() => import("./core/handtracker.ts"));

  // ColorOverlay disabled - using contour-based detection, no colored overlays needed

  if (CameraTracker) {
    try {
      const processCanvas = document.createElement("canvas");
      const cameraTracker = new CameraTracker(video, processCanvas);
      await cameraTracker.start({ width: CAM_W, height: CAM_H, fps: CAM_FPS });

      let absentFrames = 0;
      cameraTracker.onFrame((objects) => {
        if (objects.length === 0) {
          absentFrames++;
          if (absentFrames >= ABSENT_THRESHOLD) {
            console.log("[LineAR] Objects cleared after", absentFrames, "absent frames");
            dispatch({ type: "OBJECTS_CLEARED" });
            ui.setDetectionOutline(null);
          }
          return;
        }
        console.log("[LineAR] Object detected:", objects.length, "object(s)");
        absentFrames = 0;
        dispatch({ type: "OBJECTS_DETECTED", payload: objects });

        // Compute detection outline for UI rendering
        if (objects.length > 0) {
          const obj = objects[0]!;
          const { x, y, width, height } = obj.boundingBox;
          const tb = (p: Point2D) => camToBoard(p, CAM_W, CAM_H, BOARD_HALF);
          const outline: Point2D[] = [
            tb({ x, y }),
            tb({ x: x + width, y }),
            tb({ x: x + width, y: y + height }),
            tb({ x, y: y + height }),
          ];
          ui.setDetectionOutline(outline);
        }

        if (currentPhase === "OBJECT_DETECTED" && currentStableCount >= STABLE_THRESHOLD) {
          console.log("[LineAR] Stability threshold reached, locking corners...");
          ui.setDetectionOutline(null);
          const obj = objects[0]!;
          console.log("[LineAR] Stable object, locking corners:", obj.boundingBox);
          const { x, y, width, height } = obj.boundingBox;
          const tb = (p: Point2D) => camToBoard(p, CAM_W, CAM_H, BOARD_HALF);
          const snapToGrid = (p: Point2D): Point2D => ({
            x: Math.round(p.x),
            y: Math.round(p.y),
          });
          const rawCorners: [Point2D, Point2D, Point2D, Point2D] = [
            tb({ x, y }),
            tb({ x: x + width, y }),
            tb({ x: x + width, y: y + height }),
            tb({ x, y: y + height }),
          ];
          const corners: [Point2D, Point2D, Point2D, Point2D] = rawCorners.map(c => snapToGrid(c)) as [Point2D, Point2D, Point2D, Point2D];
          dispatch({ type: "CORNERS_LOCKED", payload: { corners, center: tb(obj.center) } });
        }
      });
    } catch (err) {
      console.warn("[LineAR] Camera not available:", err);
    }
  }

  if (HandTracker) {
    try {
      const handTracker = new HandTracker({ lite: true, maxHands: 1 });
      await handTracker.init();
      handTracker.start(video);
      handTracker.onFrame((hands) => {
        // Only process hand tracking after object detection is complete
        // This prevents hand from interfering with object detection
        if (currentPhase !== "POINTS_CALCULATED" &&
            currentPhase !== "SHOW_CORNERS" &&
            currentPhase !== "SHOW_BASIS_VECTORS" &&
            currentPhase !== "CONFIRM_TRANSFORM" &&
            currentPhase !== "TRANSFORMED" &&
            currentPhase !== "CONFIRM_RESET") {
          // During detection phases, clear hand state
          latestHandPosition = null;
          ui.setRawHands(null);
          return;
        }

        const vw = video.videoWidth || CAM_W;
        const vh = video.videoHeight || CAM_H;
        if (!vw || !vh) return;

        const scaleX = canvas.width / vw;
        const scaleY = canvas.height / vh;
        const uniformScale = Math.min(scaleX, scaleY);
        const offsetX = (canvas.width - vw * uniformScale) / 2;
        const offsetY = (canvas.height - vh * uniformScale) / 2;

        const toMirroredCanvas = (lm: HandLandmark): Point2D => ({
          x: (vw - lm.x) * uniformScale + offsetX,
          y: lm.y * uniformScale + offsetY,
        });

        if (hands.length > 0) {
          const hand = hands[0]!;
          if (hand.score >= 0.1 && hand.landmarks[8]) {
            latestHandPosition = toMirroredCanvas(hand.landmarks[8]);
            lastHandSeenTime = Date.now();
          }

          const mirroredHands: DetectedHand[] = hands.map((h) => ({
            handedness: h.handedness,
            score: h.score,
            gesture: h.gesture,
            landmarks: h.landmarks.map((lm) => ({ ...lm, ...toMirroredCanvas(lm) })),
          }));
          ui.setRawHands(mirroredHands);
          console.log("[LineAR] Hand detected:", hands.length, "hand(s)");
        } else {
          latestHandPosition = null;
          ui.setRawHands(null);
        }
      });
    } catch (err) {
      console.warn("[LineAR] Hand tracker not available:", err);
    }
  }
}

async function safeImport<T>(factory: () => Promise<T>): Promise<Partial<T>> {
  try { return await factory(); } catch (err) {
    return {};
  }
}

// ============================================================================
// Core Interaction Engine Processing Architecture
// ============================================================================

interface DemoInteractionState {
  dwellStart: number;
  isDraggingCornerA: boolean;
  isDraggingCornerB: boolean;
  isDraggingCornerC: boolean;
  isDraggingCornerD: boolean;
  cornerASnapped: boolean;
  cornerBSnapped: boolean;
  cornerCSnapped: boolean;
  cornerDSnapped: boolean;
  // Track which corners are locked (after 3 seconds of stable snap)
  cornerALocked: boolean;
  cornerBLocked: boolean;
  cornerCLocked: boolean;
  cornerDLocked: boolean;
  cornerCPlaced: boolean;
  cornerDPlaced: boolean;
  isDraggingE1: boolean;
  isDraggingE2: boolean;
  e1Snapped: boolean;
  e2Snapped: boolean;
  arrowTransitionFired: boolean;
  previousPhase: string | null;
  arrowMatrix: Matrix2x2;
  fingerHistory: Point2D[];
  // Auto-release: timestamp when corner first snapped while being dragged
  cornerASnapTime: number;
  cornerBSnapTime: number;
  cornerCSnapTime: number;
  cornerDSnapTime: number;
  // Position where corner first snapped (to detect finger movement)
  cornerASnapPos: Point2D | null;
  cornerBSnapPos: Point2D | null;
  cornerCSnapPos: Point2D | null;
  cornerDSnapPos: Point2D | null;
  // Release time for cooldown - set when corner auto-releases after snap (kept for reference, not used for global cooldown)
  cornerAReleaseTime: number;
  cornerBReleaseTime: number;
  cornerCReleaseTime: number;
  cornerDReleaseTime: number;
  // Global cooldown - after any corner locks, NO corners can be grabbed for 4 seconds
  globalCooldownUntil: number;
  // Track initial corner positions to detect when corners are adjusted
  initialCornerPositions: [Point2D, Point2D, Point2D, Point2D] | null;
  hasAdjustedCorner: boolean;
  // Done prompt visibility
  showDonePrompt: boolean;
}

function createDemoState(): DemoInteractionState {
  return {
    dwellStart: 0,
    isDraggingCornerA: false,
    isDraggingCornerB: false,
    isDraggingCornerC: false,
    isDraggingCornerD: false,
    cornerASnapped: false,
    cornerBSnapped: false,
    cornerCSnapped: false,
    cornerDSnapped: false,
    cornerALocked: false,
    cornerBLocked: false,
    cornerCLocked: false,
    cornerDLocked: false,
    cornerCPlaced: false,
    cornerDPlaced: false,
    isDraggingE1: false,
    isDraggingE2: false,
    e1Snapped: false,
    e2Snapped: false,
    arrowTransitionFired: false,
    previousPhase: null,
    arrowMatrix: [1, 0, 0, 1],
    fingerHistory: [],
    cornerASnapTime: 0,
    cornerBSnapTime: 0,
    cornerCSnapTime: 0,
    cornerDSnapTime: 0,
    cornerASnapPos: null,
    cornerBSnapPos: null,
    cornerCSnapPos: null,
    cornerDSnapPos: null,
    cornerAReleaseTime: 0,
    cornerBReleaseTime: 0,
    cornerCReleaseTime: 0,
    cornerDReleaseTime: 0,
    globalCooldownUntil: 0,
    initialCornerPositions: null,
    hasAdjustedCorner: false,
    showDonePrompt: false,
  };
}

async function startCamera(video: HTMLVideoElement, width: number, height: number): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width, height, frameRate: CAM_FPS },
  });
  video.srcObject = stream;
  await new Promise<void>((resolve) => { video.onloadedmetadata = () => resolve(); });
  await video.play();
}

async function setupDemoTracker(canvas: HTMLCanvasElement, video: HTMLVideoElement, ui: TabletopUI, ds: DemoInteractionState): Promise<void> {
  const { HandTracker } = await safeImport<typeof import("./core/handtracker.ts")>(() => import("./core/handtracker.ts"));
  if (!HandTracker) return;

  const handTracker = new HandTracker({ lite: true, maxHands: 1 });
  await handTracker.init();
  handTracker.start(video);

  handTracker.onFrame((hands) => {
    const vw = video.videoWidth || CAM_W;
    const vh = video.videoHeight || CAM_H;
    if (!vw || !vh) return;

    const scaleX = canvas.width / vw;
    const scaleY = canvas.height / vh;
    const uniformScale = Math.min(scaleX, scaleY);
    const offsetX = (canvas.width - vw * uniformScale) / 2;
    const offsetY = (canvas.height - vh * uniformScale) / 2;

    const toMirroredCanvas = (lm: HandLandmark): Point2D => ({
      x: (vw - lm.x) * uniformScale + offsetX,
      y: lm.y * uniformScale + offsetY,
    });

    if (hands.length > 0) {
      const hand = hands[0]!;
      if (hand.score >= 0.1 && hand.landmarks[8]) {
        const rawPoint = toMirroredCanvas(hand.landmarks[8]);

        ds.fingerHistory.push(rawPoint);
        if (ds.fingerHistory.length > 4) ds.fingerHistory.shift();
        const sum = ds.fingerHistory.reduce((acc, curr) => ({ x: acc.x + curr.x, y: acc.y + curr.y }), { x: 0, y: 0 });

        latestHandPosition = { x: sum.x / ds.fingerHistory.length, y: sum.y / ds.fingerHistory.length };
        lastHandSeenTime = Date.now();
      }

      const mirroredHands: DetectedHand[] = hands.map((h) => ({
        handedness: h.handedness,
        score: h.score,
        gesture: h.gesture,
        landmarks: h.landmarks.map((lm) => ({ ...lm, ...toMirroredCanvas(lm) })),
      }));
      ui.setRawHands(mirroredHands);
    } else {
      latestHandPosition = null;
      ui.setRawHands(null);
    }
  });
}

// ── Unified interaction runner ───────────────────────────────────────────────
function processInteractionFrame(
  pointerCanvas: Point2D | null,
  state: AppState,
  dispatch: (action: AppAction) => void,
  ui: TabletopUI,
  canvas: HTMLCanvasElement,
  ds: DemoInteractionState,
): void {
  ui.setFingerPosition(pointerCanvas);

  // If no finger or mouse coordinate is active, drop grabs and bail
  if (!pointerCanvas) {
    ds.isDraggingCornerA = false;
    ds.isDraggingCornerB = false;
    ds.isDraggingCornerC = false;
    ds.isDraggingCornerD = false;
    ds.isDraggingE1 = false;
    ds.isDraggingE2 = false;
    return;
  }

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const gridLayout = ui.getLayoutProperties();
  const TEXT_STRIP_H = 40;
  const GRID_RANGE = 10;
  const gridHeight = canvas.height - TEXT_STRIP_H;
  const scale = Math.min(canvas.width, gridHeight) / (GRID_RANGE * 2);

  // ── PHASE: WAITING FOR OBJECT ──────────────────────────────────────────────
  if (state.phase === "WAITING_FOR_OBJECT") {
    const gridY = TEXT_STRIP_H;
    if (gridLayout) {
      const bx = (canvas.width - 180) / 2;
      const by = gridY + gridHeight / 2 - 24;
      if (
        pointerCanvas.x >= bx &&
        pointerCanvas.x <= bx + 180 &&
        pointerCanvas.y >= by &&
        pointerCanvas.y <= by + 48
      ) {
        if (ds.dwellStart === 0) ds.dwellStart = Date.now();
        if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
          ds.dwellStart = 0;
          dispatch({ type: "OBJECTS_DETECTED", payload: [] });
        }
      } else {
        ds.dwellStart = 0;
      }
    }
    return;
  }

  // ── PHASE: POINTS CALCULATED (CORNER TRACKING W/ GLOBAL OMNIPRESENT SNAPPING) ──
  if (state.phase === "POINTS_CALCULATED") {
    const corners = state.objectCorners;
    if (!corners) return;

    // Helper to check if corner can be grabbed (respects global cooldown after any corner locks)
    const canGrabCorner = (): boolean => {
      // No corners can be grabbed during global cooldown
      return Date.now() > ds.globalCooldownUntil;
    };

    const isValidCorner = (c: Point2D): boolean => isFinite(c.x) && isFinite(c.y) && Math.abs(c.x) < GRID_BOUNDS * 2 && Math.abs(c.y) < GRID_BOUNDS * 2;

    const getCanvasCorner = (idx: number) => {
      const corner = corners[idx]!;
      if (!isValidCorner(corner)) return { x: cx, y: cy };
      return { x: cx + corner.x * scale, y: cy - corner.y * scale };
    };

    const dist = (p1: Point2D, p2: Point2D): number => {
      if (!isFinite(p1.x) || !isFinite(p1.y) || !isFinite(p2.x) || !isFinite(p2.y)) return Infinity;
      return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
    };

    // Get current snap state for each corner
    const getCornerSnapped = (idx: number): boolean => {
      switch (idx) {
        case 0: return ds.cornerASnapped;
        case 1: return ds.cornerBSnapped;
        case 2: return ds.cornerCSnapped;
        case 3: return ds.cornerDSnapped;
        default: return false;
      }
    };

    // Check for auto-release on currently dragged corners
    // Auto-release: 3 seconds after corner first snapped while dragging
    const handleAutoRelease = (cornerIdx: number, isDragging: boolean, snapTime: number, snapPos: Point2D | null) => {
      if (!isDragging || snapTime === 0) return false;
      
      const elapsed = Date.now() - snapTime;
      if (elapsed > CORNER_LOCK_DELAY_MS) {
        // 3 seconds have passed - auto-release the corner
        switch (cornerIdx) {
          case 0:
            ds.isDraggingCornerA = false;
            ds.cornerALocked = true;
            ds.cornerAReleaseTime = Date.now();
            ds.cornerASnapTime = 0;
            ds.cornerASnapPos = null;
            ui.setCornerDrag(0, null, false);
            break;
          case 1:
            ds.isDraggingCornerB = false;
            ds.cornerBLocked = true;
            ds.cornerBReleaseTime = Date.now();
            ds.cornerBSnapTime = 0;
            ds.cornerBSnapPos = null;
            ui.setCornerDrag(1, null, false);
            break;
          case 2:
            ds.isDraggingCornerC = false;
            ds.cornerCLocked = true;
            ds.cornerCReleaseTime = Date.now();
            ds.cornerCSnapTime = 0;
            ds.cornerCSnapPos = null;
            ui.setCornerDrag(2, null, false);
            break;
          case 3:
            ds.isDraggingCornerD = false;
            ds.cornerDLocked = true;
            ds.cornerDReleaseTime = Date.now();
            ds.cornerDSnapTime = 0;
            ds.cornerDSnapPos = null;
            ui.setCornerDrag(3, null, false);
            break;
        }
        // Set global cooldown - no corners can be grabbed for 4 seconds
        ds.globalCooldownUntil = Date.now() + CORNER_COOLDOWN_MS;
        return true;
      }
      return false;
    };

    // Check for timer reset when finger moves while snapped
    const handleTimerReset = (cornerIdx: number, isDragging: boolean, snapTime: number, snapPos: Point2D | null) => {
      if (!isDragging || snapTime === 0 || !snapPos) return;
      
      const moveDist = Math.sqrt((pointerCanvas.x - snapPos.x) ** 2 + (pointerCanvas.y - snapPos.y) ** 2);
      if (moveDist > CORNER_MOVE_THRESHOLD_PX) {
        // Finger moved - reset timer
        switch (cornerIdx) {
          case 0: ds.cornerASnapTime = 0; ds.cornerASnapPos = null; break;
          case 1: ds.cornerBSnapTime = 0; ds.cornerBSnapPos = null; break;
          case 2: ds.cornerCSnapTime = 0; ds.cornerCSnapPos = null; break;
          case 3: ds.cornerDSnapTime = 0; ds.cornerDSnapPos = null; break;
        }
      }
    };

    // Process auto-release and timer reset for each corner
    handleAutoRelease(0, ds.isDraggingCornerA, ds.cornerASnapTime, ds.cornerASnapPos);
    handleTimerReset(0, ds.isDraggingCornerA, ds.cornerASnapTime, ds.cornerASnapPos);
    
    handleAutoRelease(1, ds.isDraggingCornerB, ds.cornerBSnapTime, ds.cornerBSnapPos);
    handleTimerReset(1, ds.isDraggingCornerB, ds.cornerBSnapTime, ds.cornerBSnapPos);
    
    handleAutoRelease(2, ds.isDraggingCornerC, ds.cornerCSnapTime, ds.cornerCSnapPos);
    handleTimerReset(2, ds.isDraggingCornerC, ds.cornerCSnapTime, ds.cornerCSnapPos);
    
    handleAutoRelease(3, ds.isDraggingCornerD, ds.cornerDSnapTime, ds.cornerDSnapPos);
    handleTimerReset(3, ds.isDraggingCornerD, ds.cornerDSnapTime, ds.cornerDSnapPos);

    // Update UI with locked states
    ui.setCornerLockedStates([ds.cornerALocked, ds.cornerBLocked, ds.cornerCLocked, ds.cornerDLocked]);

    if (!ds.isDraggingCornerA && !ds.isDraggingCornerB && !ds.isDraggingCornerC && !ds.isDraggingCornerD) {
      // Check if any corner has been adjusted from initial position
      if (!ds.hasAdjustedCorner && ds.initialCornerPositions) {
        for (let i = 0; i < 4; i++) {
          const init = ds.initialCornerPositions[i]!;
          const curr = corners[i]!;
          if (Math.abs(init.x - curr.x) > 0.01 || Math.abs(init.y - curr.y) > 0.01) {
            ds.hasAdjustedCorner = true;
            ds.showDonePrompt = true;
            break;
          }
        }
      }

      if (dist(pointerCanvas, getCanvasCorner(0)) < GRAB_RADIUS_PX && canGrabCorner()) {
        ds.isDraggingCornerA = true;
        ds.cornerALocked = false; // Reset locked state when grabbing
      } else if (dist(pointerCanvas, getCanvasCorner(1)) < GRAB_RADIUS_PX && canGrabCorner()) {
        ds.isDraggingCornerB = true;
        ds.cornerBLocked = false;
      } else if (dist(pointerCanvas, getCanvasCorner(2)) < GRAB_RADIUS_PX && canGrabCorner()) {
        ds.isDraggingCornerC = true;
        ds.cornerCLocked = false;
      } else if (dist(pointerCanvas, getCanvasCorner(3)) < GRAB_RADIUS_PX && canGrabCorner()) {
        ds.isDraggingCornerD = true;
        ds.cornerDLocked = false;
      }
    }

    const currentDrag = ds.isDraggingCornerA ? 0 : ds.isDraggingCornerB ? 1 : ds.isDraggingCornerC ? 2 : ds.isDraggingCornerD ? 3 : -1;

    if (currentDrag !== -1) {
      const currentCorner = corners[currentDrag]!;
      if (!isValidCorner(currentCorner)) {
        return;
      }

      let rawGridX = (pointerCanvas.x - cx) / scale;
      let rawGridY = -(pointerCanvas.y - cy) / scale;

      if (!isFinite(rawGridX) || !isFinite(rawGridY)) {
        rawGridX = currentCorner.x;
        rawGridY = currentCorner.y;
      }

      const clampedGridX = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawGridX));
      const clampedGridY = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawGridY));

      const safeRawX = Math.abs(clampedGridX) < EPSILON ? (clampedGridX >= 0 ? EPSILON : -EPSILON) : clampedGridX;
      const safeRawY = Math.abs(clampedGridY) < EPSILON ? (clampedGridY >= 0 ? EPSILON : -EPSILON) : clampedGridY;

      const nearestGridX = Math.round(safeRawX);
      const nearestGridY = Math.round(safeRawY);

      const clampedNearestX = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, nearestGridX));
      const clampedNearestY = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, nearestGridY));

      const snapCanvasX = cx + clampedNearestX * scale;
      const snapCanvasY = cy - clampedNearestY * scale;
      const distToIntersection = Math.sqrt((pointerCanvas.x - snapCanvasX) ** 2 + (pointerCanvas.y - snapCanvasY) ** 2);

      const isNowSnapped = distToIntersection < ARROW_SNAP_PX;

      // Record snap time and position when corner first snaps while dragging
      const getSnapTimeDirect = (idx: number): number => {
        switch (idx) {
          case 0: return ds.cornerASnapTime;
          case 1: return ds.cornerBSnapTime;
          case 2: return ds.cornerCSnapTime;
          case 3: return ds.cornerDSnapTime;
          default: return 0;
        }
      };

      if (isNowSnapped) {
        const currentSnapTime = getSnapTimeDirect(currentDrag);
        if (currentSnapTime === 0) {
          // First time snapping - start the timer
          switch (currentDrag) {
            case 0: ds.cornerASnapTime = Date.now(); ds.cornerASnapPos = { x: pointerCanvas.x, y: pointerCanvas.y }; break;
            case 1: ds.cornerBSnapTime = Date.now(); ds.cornerBSnapPos = { x: pointerCanvas.x, y: pointerCanvas.y }; break;
            case 2: ds.cornerCSnapTime = Date.now(); ds.cornerCSnapPos = { x: pointerCanvas.x, y: pointerCanvas.y }; break;
            case 3: ds.cornerDSnapTime = Date.now(); ds.cornerDSnapPos = { x: pointerCanvas.x, y: pointerCanvas.y }; break;
          }
        }
      } else {
        // Not snapped - reset snap timer (but only if not already locked)
        const isLocked = currentDrag === 0 ? ds.cornerALocked : currentDrag === 1 ? ds.cornerBLocked : currentDrag === 2 ? ds.cornerCLocked : ds.cornerDLocked;
        if (!isLocked) {
          switch (currentDrag) {
            case 0: ds.cornerASnapTime = 0; ds.cornerASnapPos = null; break;
            case 1: ds.cornerBSnapTime = 0; ds.cornerBSnapPos = null; break;
            case 2: ds.cornerCSnapTime = 0; ds.cornerCSnapPos = null; break;
            case 3: ds.cornerDSnapTime = 0; ds.cornerDSnapPos = null; break;
          }
        }
      }

      if (isNowSnapped) {
        corners[currentDrag] = { x: clampedNearestX, y: clampedNearestY };
        if (currentDrag === 0) ds.cornerASnapped = true;
        if (currentDrag === 1) ds.cornerBSnapped = true;
        if (currentDrag === 2) ds.cornerCSnapped = true;
        if (currentDrag === 3) ds.cornerDSnapped = true;
      } else {
        corners[currentDrag] = { x: clampedGridX, y: clampedGridY };
        if (currentDrag === 0) ds.cornerASnapped = false;
        if (currentDrag === 1) ds.cornerBSnapped = false;
        if (currentDrag === 2) ds.cornerCSnapped = false;
        if (currentDrag === 3) ds.cornerDSnapped = false;
      }

      const cornerGridPos = { 
        x: isNowSnapped ? clampedNearestX : clampedGridX, 
        y: isNowSnapped ? clampedNearestY : clampedGridY 
      };
      ui.setCornerDrag(currentDrag, cornerGridPos, currentDrag === 2 ? ds.cornerCSnapped : currentDrag === 3 ? ds.cornerDSnapped : false);
      dispatch({ type: "CORNERS_LOCKED", payload: { corners: [...corners], center: state.objectCenter ?? { x: 0, y: 0 } } });
    }

    if (ds.cornerCSnapped && !ds.isDraggingCornerC) ds.cornerCPlaced = true;
    if (ds.cornerDSnapped && !ds.isDraggingCornerD) ds.cornerDPlaced = true;

    // Handle done prompt at top (Yes/No buttons)
    if (ds.showDonePrompt) {
      ui.showDoneButton = true;
      const promptY = 60;
      const btnW = 80;
      const btnH = 36;
      const btnSpacing = 20;
      const totalW = btnW * 2 + btnSpacing;
      const startX = (canvas.width - totalW) / 2;
      
      const noBtn = { x: startX, y: promptY, w: btnW, h: btnH };
      const yesBtn = { x: startX + btnW + btnSpacing, y: promptY, w: btnW, h: btnH };
      
      // No button
      if (
        pointerCanvas.x >= noBtn.x &&
        pointerCanvas.x <= noBtn.x + noBtn.w &&
        pointerCanvas.y >= noBtn.y &&
        pointerCanvas.y <= noBtn.y + noBtn.h
      ) {
        if (ds.dwellStart === 0) ds.dwellStart = Date.now();
        if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
          ds.dwellStart = 0;
          ds.showDonePrompt = false;
        }
      } else {
        ds.dwellStart = 0;
      }

      // Yes button
      if (
        pointerCanvas.x >= yesBtn.x &&
        pointerCanvas.x <= yesBtn.x + yesBtn.w &&
        pointerCanvas.y >= yesBtn.y &&
        pointerCanvas.y <= yesBtn.y + yesBtn.h
      ) {
        if (ds.dwellStart === 0) ds.dwellStart = Date.now();
        if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
          ds.dwellStart = 0;
          ui.onDone?.();
        }
      }
    }
    return;
  }

  // ── PHASE: SHOW BASIS VECTORS (GLOBAL SNAPPING INTERSECTIONS) ──────────────
  if (state.phase === "SHOW_BASIS_VECTORS") {
    const mat = ds.arrowMatrix;
    const ghostE1 = { x: cx + GHOST_ARROWS.e1.x * scale, y: cy - GHOST_ARROWS.e1.y * scale };
    const ghostE2 = { x: cx + GHOST_ARROWS.e2.x * scale, y: cy - GHOST_ARROWS.e2.y * scale };

    if (!ds.isDraggingE1 && !ds.isDraggingE2 && !ds.e1Snapped) {
      const curE1 = { x: cx + mat[0] * scale, y: cy - mat[2] * scale };
      if (Math.sqrt((pointerCanvas.x - curE1.x) ** 2 + (pointerCanvas.y - curE1.y) ** 2) < ARROW_GRAB_RADIUS_PX) {
        ds.isDraggingE1 = true;
      }
    }

    if (ds.isDraggingE1 && !ds.e1Snapped) {
      let rawGridX = (pointerCanvas.x - cx) / scale;
      let rawGridY = -(pointerCanvas.y - cy) / scale;

      if (!isFinite(rawGridX) || !isFinite(rawGridY)) {
        rawGridX = mat[0];
        rawGridY = mat[2];
      }

      const clampedX = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawGridX));
      const clampedY = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawGridY));
      const safeX = Math.abs(clampedX) < EPSILON ? (clampedX >= 0 ? EPSILON : -EPSILON) : clampedX;
      const safeY = Math.abs(clampedY) < EPSILON ? (clampedY >= 0 ? EPSILON : -EPSILON) : clampedY;

      const nearestGridX = Math.round(safeX);
      const nearestGridY = Math.round(safeY);
      const clampedNearestX = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, nearestGridX));
      const clampedNearestY = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, nearestGridY));

      const snapCanvasX = cx + clampedNearestX * scale;
      const snapCanvasY = cy - clampedNearestY * scale;

      if (Math.sqrt((pointerCanvas.x - snapCanvasX) ** 2 + (pointerCanvas.y - snapCanvasY) ** 2) < ARROW_SNAP_PX) {
        mat[0] = clampedNearestX;
        mat[2] = clampedNearestY;
        if (Math.sqrt((pointerCanvas.x - ghostE1.x) ** 2 + (pointerCanvas.y - ghostE1.y) ** 2) < ARROW_SNAP_PX) {
          mat[0] = GHOST_ARROWS.e1.x;
          mat[2] = GHOST_ARROWS.e1.y;
          ds.e1Snapped = true;
          ds.isDraggingE1 = false;
          ui.setArrowSnapped(true, ds.e2Snapped);
        }
      } else {
        mat[0] = clampedX;
        mat[2] = clampedY;
      }
      ui.setMatrix([...mat]);
    }

    if (!ds.isDraggingE1 && !ds.isDraggingE2 && ds.e1Snapped && !ds.e2Snapped) {
      const curE2X = isFinite(mat[1]) ? mat[1] : 1;
      const curE2Y = isFinite(mat[3]) ? mat[3] : 0;
      const curE2 = { x: cx + curE2X * scale, y: cy - curE2Y * scale };
      if (Math.sqrt((pointerCanvas.x - curE2.x) ** 2 + (pointerCanvas.y - curE2.y) ** 2) < ARROW_GRAB_RADIUS_PX) {
        ds.isDraggingE2 = true;
      }
    }

    if (ds.isDraggingE2 && !ds.e2Snapped) {
      let rawGridX = (pointerCanvas.x - cx) / scale;
      let rawGridY = -(pointerCanvas.y - cy) / scale;

      if (!isFinite(rawGridX) || !isFinite(rawGridY)) {
        rawGridX = mat[1];
        rawGridY = mat[3];
      }

      const clampedX = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawGridX));
      const clampedY = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawGridY));
      const safeX = Math.abs(clampedX) < EPSILON ? (clampedX >= 0 ? EPSILON : -EPSILON) : clampedX;
      const safeY = Math.abs(clampedY) < EPSILON ? (clampedY >= 0 ? EPSILON : -EPSILON) : clampedY;

      const nearestGridX = Math.round(safeX);
      const nearestGridY = Math.round(safeY);
      const clampedNearestX = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, nearestGridX));
      const clampedNearestY = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, nearestGridY));

      const snapCanvasX = cx + clampedNearestX * scale;
      const snapCanvasY = cy - clampedNearestY * scale;

      if (Math.sqrt((pointerCanvas.x - snapCanvasX) ** 2 + (pointerCanvas.y - snapCanvasY) ** 2) < ARROW_SNAP_PX) {
        mat[1] = clampedNearestX;
        mat[3] = clampedNearestY;
        if (Math.sqrt((pointerCanvas.x - ghostE2.x) ** 2 + (pointerCanvas.y - ghostE2.y) ** 2) < ARROW_SNAP_PX) {
          mat[1] = GHOST_ARROWS.e2.x;
          mat[3] = GHOST_ARROWS.e2.y;
          ds.e2Snapped = true;
          ds.isDraggingE2 = false;
          ui.setArrowSnapped(ds.e1Snapped, true);
        }
      } else {
        mat[1] = clampedX;
        mat[3] = clampedY;
      }
      ui.setMatrix([...mat]);
    }

    if (ds.e1Snapped && ds.e2Snapped && !ds.arrowTransitionFired) {
      ds.arrowTransitionFired = true;
      dispatch({ type: "BASIS_ADJUSTED", payload: [...mat] });
    }
    return;
  }

  // ── PHASE: DIALOG CONFIRMATIONS ─────────────────────────────────────────────
  if (state.phase === "CONFIRM_TRANSFORM" || state.phase === "CONFIRM_RESET") {
    const buttons = ui.getButtonRects();
    const btnYes = buttons.yes;
    const btnNo = buttons.no;
    if (!btnYes || !btnNo) return;

    const insideBox = (p: Point2D, box: DOMRect) =>
      p.x >= box.x && p.x <= box.x + box.width && p.y >= box.y && p.y <= box.y + box.height;

    let targeted = false;
    if (insideBox(pointerCanvas, btnYes)) {
      targeted = true;
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onYes?.();
      }
    } else if (insideBox(pointerCanvas, btnNo)) {
      targeted = true;
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onNo?.();
      }
    }

    if (!targeted) ds.dwellStart = 0;
    return;
  }

  // ── PHASE: TRANSFORMED CONTINUE BUTTON ─────────────────────────────────────
  if (state.phase === "TRANSFORMED") {
    const continueBtn = { x: cx + 200, y: cy + 130, w: 150, h: 50 };
    if (
      pointerCanvas.x >= continueBtn.x &&
      pointerCanvas.x <= continueBtn.x + continueBtn.w &&
      pointerCanvas.y >= continueBtn.y &&
      pointerCanvas.y <= continueBtn.y + continueBtn.h
    ) {
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onContinue?.();
      }
    } else {
      ds.dwellStart = 0;
    }
  }
}
