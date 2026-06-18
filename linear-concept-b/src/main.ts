// src/main.ts
// ── Static imports: lightweight only (no TensorFlow, p5, OpenCV dep chains) ──
import {
  createInitialState,
  transition,
  type AppState,
  type AppAction,
} from "./core/appstatemachine.ts";
import { TabletopUI } from "./components/tabletopui.ts";
import { generateId } from "./utils/helpers.ts";
import { CoordinateMapper } from "./utils/coordinatemapper.ts";
import {
  detectProjectedMarkers,
  computeCalibrationFromMarkers,
  computeHomography,
  saveCalibration,
  loadCalibration,
  type PhysicalMarker,
} from "./core/calibration.ts";
import {
  VIRTUAL_OBJECT,
  PRESET_MATRIX,
  GHOST_ARROWS,
  PHASE_TIMINGS,
  HOVER_DWELL_MS,
  GRAB_RADIUS_PX,
  ARROW_GRAB_RADIUS_PX,
  ARROW_SNAP_PX,
  ARROW_PLACED_LOCK_MS,
  ARROW_LOCK_COOLDOWN_MS,
  ARROW_DWELL_PLACE_MS,
  DEMO_INSTRUCTIONS,
  PRESET_CORNERS_PAYLOAD,
} from "./core/demoplayer.ts";
import type { DetectedHand, DetectedObject, HandLandmark, Matrix2x2, Point2D } from "./types/index.ts";

const CAM_W = 1920;
const CAM_H = 1080;
const CAM_FPS = 30;
const BOARD_HALF = 10;
const STABLE_THRESHOLD = 10;
const ABSENT_THRESHOLD = 25;
const OPENCV_TIMEOUT = 10_000;
const FINGER_GUIDED_DEMO = false;
const EPSILON = 0.001;
const GRID_BOUNDS = 15;

// ── Coordinate Mapper (calibrated camera→grid transform) ─────────────────
let coordMapper: CoordinateMapper;

// ── Camera Flip Config ─────────────────────────────────────────────────────
// Reads public/mirror-config.txt (comma-separated: flipH,flipV).
// 0 = no flip, 1 = flip. Defaults to "0,0" if file is missing.
let flipH = true;
let flipV = true;

async function loadMirrorConfig() {
  try {
    const res = await fetch('/mirror-config.txt');
    const parts = (await res.text()).trim().split(',');
    flipH = parts[0] === '1';
    flipV = parts[1] === '1';
  } catch {
    flipH = true;
    flipV = true;
  }
}
loadMirrorConfig();

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
let handEmptyCount = 0;
const HAND_TIMEOUT_MS = 400; // Hand tracking grace period before falling back to mouse

// Current app phase (updated in dispatch, used by hand tracker callback)
let currentPhase: string = "WAITING_FOR_OBJECT";
let currentStableCount = 0;
let cameraTrackerRef: import("./core/cameratracker.ts").CameraTracker | null = null;

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
      // if (state.phase === "TRANSFORMED") {
      //   setTimeout(() => dispatch({ type: "TRANSFORMATION_DONE" }), 1800);
      // }
    }

  }

  // ── Canvas UI Setup ───────────────────────────────────────────────────
  const ui = new TabletopUI(canvas);
  ui.resize(window.innerWidth, window.innerHeight);
  ui.setVideoElement(video);
  ui.onYes = () => dispatch({ type: "CONFIRM_YES" });
  ui.onNo = () => dispatch({ type: "CONFIRM_NO" });
  ui.onContinue = () => dispatch({ type: "TRANSFORMATION_DONE" });
  window.addEventListener("resize", () => ui.resize(window.innerWidth, window.innerHeight));
  window.addEventListener("keydown", (e) => {
    if (e.key === 'h' || e.key === 'H') { flipH = !flipH; console.log("[LineAR] flipH =", flipH); }
    if (e.key === 'v' || e.key === 'V') { flipV = !flipV; console.log("[LineAR] flipV =", flipV); }
  });

  const ds = createDemoState();

  ui.setGhostArrows(GHOST_ARROWS.e1, GHOST_ARROWS.e2);
  if (FINGER_GUIDED_DEMO) {
    ui.setDemoInstructions(DEMO_INSTRUCTIONS);
    ui.setCorners(VIRTUAL_OBJECT.corners);
    // ui.onDone = () => {  // [REMOVED: Done button prompt]
    //   ds.cornerCPlaced = false;
    //   ds.cornerDPlaced = false;
    //   ui.showDoneButton = false;
    //   dispatch({ type: "OBJECTS_CLEARED" });
    // };
  }

  // ── Unified 60FPS Render & Interaction Loop ───────────────────────────
  (function renderLoop() {
    ui.setFlipH(flipH);
    ui.setFlipV(flipV);
    ui.draw(state);

    // 1. Track local state wipes on phase transitions
    if (state.phase !== ds.previousPhase) {
      console.log("[LineAR] Phase transition ->", state.phase);
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
        // ds.showDonePrompt = false;  // [REMOVED: Done button prompt]
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
        ui.setCornerLockedStates([false, false, false, false]);
        for (let i = 0; i < 4; i++) ui.setCornerDrag(i, null, false);
        ui.setPanOffset(0, 0); // Reset pan on new detection
        // Store initial corner positions to detect adjustments
        if (state.objectCorners) {
          ds.initialCornerPositions = [...state.objectCorners];
        }
      }
      if (state.phase === "SHOW_BASIS_VECTORS") {
        ds.e1Snapped = false;
        ds.e2Snapped = false;
        ds.e1Locked = false;
        ds.e2Locked = false;
        ds.isDraggingE1 = false;
        ds.isDraggingE2 = false;
        ds.arrowTransitionFired = false;
        ds.arrowMatrix = [1, 0, 0, 1];
        ds.e1Placed = false;
        ds.e2Placed = false;
        ds.e1PlaceTime = 0;
        ds.e2PlaceTime = 0;
        ds.e1LockTime = 0;
        ds.e2LockTime = 0;
        ds.e1DwellStart = 0;
        ds.e2DwellStart = 0;
        ds.e1DwellIntersection = null;
        ds.e2DwellIntersection = null;
        ds.e1ReGrabCooldown = 0;
        ds.handsAbsentSince = 0;
        ds.e2ReGrabCooldown = 0;
        ds.showBasisProceed = false;
        ui.setArrowSnapped(false, false);
        ui.setArrowLocked(false, false);
        ui.setMatrix([1, 0, 0, 1]);
      }
      if (state.phase === "TRANSFORMED") {
        ds.isPanning = false;
        ds.panStartX = 0;
        ds.panStartY = 0;
        if (state.objectCorners && state.appliedMatrix) {
          const grid = ui.getLastGridRect();
          if (grid) {
            ui.autoFrameTransformed(state.objectCorners, state.appliedMatrix, grid);
          }
        }
      }
      if (state.phase === "CONFIRM_RESET") {
        ds.isPanning = false;
        ds.panStartX = 0;
        ds.panStartY = 0;
      }
      if (state.phase === "WAITING_FOR_OBJECT") {
        ui.setPanBounds(null);
        ui.setPanOffset(0, 0);
      }
      // Auto-advance from SHOW_CORNERS → SHOW_BASIS_VECTORS after brief pause
      if (state.phase === "SHOW_CORNERS") {
        setTimeout(() => dispatch({ type: "PHASE_ADVANCE" }), 1200);
      }
    }

    // 2. Resolve Pointer Hierarchy: Hand tracking wins, Mouse acts as a smart hover fallback
    let activePointer: Point2D | null = null;
    const isHandActive = !!(latestHandPosition && (Date.now() - lastHandSeenTime < HAND_TIMEOUT_MS));

    if (isHandActive) {
      activePointer = latestHandPosition;
    } else {
      activePointer = ui.getMouseCanvasPos();
    }

    // 3. Process pointer metrics globally
    processInteractionFrame(activePointer, state, dispatch, ui, canvas, ds, isHandActive);

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
      cameraTrackerRef = cameraTracker;
      await cameraTracker.start({ width: CAM_W, height: CAM_H, fps: CAM_FPS });
      console.log("[CameraTracker] Camera started successfully with resolution:", cameraTracker.resolution);

      // Create calibrated coordinate mapper
      const res = cameraTracker.resolution;
      coordMapper = new CoordinateMapper(res.width, res.height, BOARD_HALF);

      // Check for saved calibration
      const savedMatrix = loadCalibration();
      if (savedMatrix) {
        coordMapper.setTransform({ type: "homography", matrix: savedMatrix });
        console.log("[LineAR] Loaded saved calibration");
        console.warn("[LineAR] Run localStorage.clear() in DevTools console and reload to re-calibrate");
      } else {
        console.log("[LineAR] No calibration found — running projected marker calibration");
        dispatch({ type: "CALIBRATE" });
        const calibrated = await runCalibrationSequence(
          cameraTracker, coordMapper, 10,
          (msg) => ui.setDemoInstructions({ CALIBRATING: msg }),
        );
        ui.setDemoInstructions(null);
        dispatch({ type: "CALIBRATION_DONE" });
        if (calibrated) {
          console.log("[LineAR] Physical calibration complete");
        } else {
          console.warn("[LineAR] Continuing without calibration — using direct mapping");
        }
        // Re-capture background so printed markers are no longer detected as objects
        cameraTrackerRef?.resetBackground();
      }

      let absentFrames = 0;
      const DETECTION_WINDOW_SIZE = 10;
      const DETECTION_WINDOW_THRESHOLD = 4;
      let detectionHistory: boolean[] = [];
      let windowObjects: DetectedObject[] = [];
      cameraTracker.onFrame((objects) => {
        // Only process object detection during phases that need it
        if (!["WAITING_FOR_OBJECT", "OBJECT_DETECTED", "POINTS_CALCULATED"].includes(currentPhase)) return;

        const rawDetected = objects.length > 0;
        detectionHistory.push(rawDetected);
        if (detectionHistory.length > DETECTION_WINDOW_SIZE) {
          detectionHistory.shift();
        }
        if (rawDetected) {
          windowObjects.push(objects[0]!);
          if (windowObjects.length > DETECTION_WINDOW_SIZE) {
            windowObjects.shift();
          }
        }
        const trueCount = detectionHistory.filter(Boolean).length;
        const objectPresent = trueCount >= DETECTION_WINDOW_THRESHOLD;
        if (!objectPresent) {
          if (absentFrames >= ABSENT_THRESHOLD) {
            console.log("[LineAR] Objects cleared after", absentFrames, "absent frames");
            dispatch({ type: "OBJECTS_CLEARED" });
            ui.setDetectionOutline(null);
            absentFrames = 0;
          } else {
            absentFrames++;
          }
          return;
        }
        absentFrames = 0;
        const lastObj = windowObjects[windowObjects.length - 1]!;
        const avgObj: DetectedObject = {
          ...lastObj,
          id: generateId(),
          boundingBox: {
            x: Math.round(windowObjects.reduce((s, o) => s + o.boundingBox.x, 0) / windowObjects.length),
            y: Math.round(windowObjects.reduce((s, o) => s + o.boundingBox.y, 0) / windowObjects.length),
            width: Math.round(windowObjects.reduce((s, o) => s + o.boundingBox.width, 0) / windowObjects.length),
            height: Math.round(windowObjects.reduce((s, o) => s + o.boundingBox.height, 0) / windowObjects.length),
          },
          center: {
            x: Math.round(windowObjects.reduce((s, o) => s + o.center.x, 0) / windowObjects.length),
            y: Math.round(windowObjects.reduce((s, o) => s + o.center.y, 0) / windowObjects.length),
          },
        };
        const smoothedObjects: DetectedObject[] = [avgObj];
        console.log("[LineAR] Smoothed detection (window {trueCount}/{" + DETECTION_WINDOW_SIZE + "}):", smoothedObjects.length, "object(s)");
        dispatch({ type: "OBJECTS_DETECTED", payload: smoothedObjects });

        const tb = (p: Point2D) => coordMapper.cameraToGrid(p);
        const outline: Point2D[] = avgObj.shapeCorners
          ? avgObj.shapeCorners.map(tb)
          : [
              tb({ x: avgObj.boundingBox.x, y: avgObj.boundingBox.y }),
              tb({ x: avgObj.boundingBox.x + avgObj.boundingBox.width, y: avgObj.boundingBox.y }),
              tb({ x: avgObj.boundingBox.x + avgObj.boundingBox.width, y: avgObj.boundingBox.y + avgObj.boundingBox.height }),
              tb({ x: avgObj.boundingBox.x, y: avgObj.boundingBox.y + avgObj.boundingBox.height }),
            ];
        ui.setDetectionOutline(outline);

        if (currentPhase === "OBJECT_DETECTED" && currentStableCount >= STABLE_THRESHOLD) {
          console.log("[LineAR] Stability threshold reached, locking corners...");
          ui.setDetectionOutline(null);
          console.log("[LineAR] Stable object, locking corners:", avgObj.boundingBox);
          const { x, y, width, height } = avgObj.boundingBox;
          const rawCorners: [Point2D, Point2D, Point2D, Point2D] = [
            tb({ x, y }),
            tb({ x: x + width, y }),
            tb({ x: x + width, y: y + height }),
            tb({ x, y: y + height }),
          ];
          const snapToGrid = (p: Point2D): Point2D => ({
            x: Math.round(p.x),
            y: Math.round(p.y),
          });
          const corners: [Point2D, Point2D, Point2D, Point2D] = rawCorners.map(c => snapToGrid(c)) as [Point2D, Point2D, Point2D, Point2D];
          dispatch({ type: "CORNERS_LOCKED", payload: { corners, center: tb(avgObj.center) } });
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

        if (!video.videoWidth || !video.videoHeight) return;

        const toGridCanvas = (lm: HandLandmark): Point2D => {
          const gp = coordMapper.cameraToGrid({ x: lm.x, y: lm.y });
          const r = ui.getLastGridRect();
          if (!r) return { x: -9999, y: -9999 };
          return ui.gridToCanvas(gp, r);
        };

        if (hands.length > 0) {
          const hand = hands[0]!;
          if (hand.score >= 0.1 && hand.landmarks[8]) {
        const rawPoint = toGridCanvas(hand.landmarks[8]);
        const SMOOTH_ALPHA = 0.7;
        const DEADZONE_PX = 3;
            if (latestHandPosition === null) {
              latestHandPosition = { ...rawPoint };
            } else {
              const dx = rawPoint.x - latestHandPosition.x;
              const dy = rawPoint.y - latestHandPosition.y;
              if (dx * dx + dy * dy > DEADZONE_PX * DEADZONE_PX) {
                latestHandPosition.x += SMOOTH_ALPHA * dx;
                latestHandPosition.y += SMOOTH_ALPHA * dy;
              }
            }
            lastHandSeenTime = Date.now();
          }

          const mirroredHands: DetectedHand[] = hands.map((h) => ({
            handedness: h.handedness,
            score: h.score,
            gesture: h.gesture,
            landmarks: h.landmarks.map((lm) => ({ ...lm, ...toGridCanvas(lm) })),
          }));
          ui.setRawHands(mirroredHands);
          handEmptyCount = 0;
        } else {
          handEmptyCount++;
          if (handEmptyCount >= 5) {
            latestHandPosition = null;
            ui.setRawHands(null);
          }
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Run physical ArUco marker calibration.
 * The user places 4 printed markers (IDs 1-4) at workspace corners.
 * Captures camera frames until all 4 markers are detected,
 * then computes and saves the homography.
 */
async function runCalibrationSequence(
  cameraTracker: import("./core/cameratracker.ts").CameraTracker,
  mapper: CoordinateMapper,
  retries: number,
  onStatus?: (msg: string) => void,
): Promise<boolean> {
  for (let attempt = 0; attempt < retries; attempt++) {
    onStatus?.(`Looking for markers… (attempt ${attempt + 1}/${retries})`);
    await sleep(2000);

    const canvas = cameraTracker.processCanvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) continue;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const markers = detectProjectedMarkers(imageData);

    if (markers && markers.length >= 4) {
      onStatus?.("Found 4/4 markers! Computing homography…");
      const result = computeCalibrationFromMarkers(markers);
      if (result) {
        mapper.setTransform({ type: "homography", matrix: result.matrix });
        saveCalibration(result.matrix);
        onStatus?.("Calibration complete");
        console.log(`[LineAR] Calibration succeeded with IDs: ${markers.map(m => m.id).join(",")}`);
        return true;
      }
    }

    const count = markers?.length ?? 0;
    onStatus?.(`Found ${count}/4 markers — retrying… (attempt ${attempt + 1}/${retries})`);
    console.warn(`[LineAR] Calibration attempt ${attempt + 1} failed — found: ${count}`);
  }

  console.warn("[LineAR] Calibration failed — using direct mapping");
  return false;
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
  e1Locked: boolean;
  e2Locked: boolean;
  e1SnapTime: number;
  e2SnapTime: number;
  e1SnapPos: Point2D | null;
  e2SnapPos: Point2D | null;
  e1CooldownUntil: number;
  e2CooldownUntil: number;
  arrowTransitionFired: boolean;
  // Timestamp when hand was last detected (for auto-prompt after 5 min absence)
  handsAbsentSince: number;
  previousPhase: string | null;
  arrowMatrix: Matrix2x2;
  fingerHistory: Point2D[];
  // Arrow placement state (new drag→place→lock model)
  e1Placed: boolean;
  e2Placed: boolean;
  e1PlaceTime: number;
  e2PlaceTime: number;
  e1LockTime: number;
  e2LockTime: number;
  e1DwellStart: number;
  e2DwellStart: number;
  e1DwellIntersection: Point2D | null;
  e2DwellIntersection: Point2D | null;
  e1ReGrabCooldown: number;
  e2ReGrabCooldown: number;
  showBasisProceed: boolean;

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
  // showDonePrompt: boolean;  // [REMOVED: Done button prompt]

  // Viewport pan state
  isPanning: boolean;
  panStartX: number;
  panStartY: number;
  panOffsetStartX: number;
  panOffsetStartY: number;

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
    e1Locked: false,
    e2Locked: false,
    e1SnapTime: 0,
    e2SnapTime: 0,
    e1SnapPos: null,
    e2SnapPos: null,
    e1CooldownUntil: 0,
    e2CooldownUntil: 0,
    e1Placed: false,
    e2Placed: false,
    e1PlaceTime: 0,
    e2PlaceTime: 0,
    e1LockTime: 0,
    e2LockTime: 0,
    e1DwellStart: 0,
    e2DwellStart: 0,
    e1DwellIntersection: null,
    e2DwellIntersection: null,
    e1ReGrabCooldown: 0,
    e2ReGrabCooldown: 0,
    showBasisProceed: false,
    arrowTransitionFired: false,
    handsAbsentSince: 0,
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
    // showDonePrompt: false,  // [REMOVED: Done button prompt]
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    panOffsetStartX: 0,
    panOffsetStartY: 0,
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
    if (!video.videoWidth || !video.videoHeight) return;

    const toGridCanvas = (lm: HandLandmark): Point2D => {
      const gp = coordMapper.cameraToGrid({ x: lm.x, y: lm.y });
      const r = ui.getLastGridRect();
      if (!r) return { x: -9999, y: -9999 };
      return ui.gridToCanvas(gp, r);
    };

    if (hands.length > 0) {
      const hand = hands[0]!;
      if (hand.score >= 0.1 && hand.landmarks[8]) {
            const rawPoint = toGridCanvas(hand.landmarks[8]);
            const SMOOTH_ALPHA = 0.7;
            const DEADZONE_PX = 3;
            if (latestHandPosition === null) {
              latestHandPosition = { ...rawPoint };
            } else {
              const dx = rawPoint.x - latestHandPosition.x;
              const dy = rawPoint.y - latestHandPosition.y;
              if (dx * dx + dy * dy > DEADZONE_PX * DEADZONE_PX) {
                latestHandPosition.x += SMOOTH_ALPHA * dx;
                latestHandPosition.y += SMOOTH_ALPHA * dy;
              }
            }
            lastHandSeenTime = Date.now();
      }

      const mirroredHands: DetectedHand[] = hands.map((h) => ({
        handedness: h.handedness,
        score: h.score,
        gesture: h.gesture,
        landmarks: h.landmarks.map((lm) => ({ ...lm, ...toGridCanvas(lm) })),
      }));
      ui.setRawHands(mirroredHands);
      handEmptyCount = 0;
    } else {
      handEmptyCount++;
      if (handEmptyCount >= 5) {
        latestHandPosition = null;
        ui.setRawHands(null);
      }
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
  isHandActive: boolean,
): void {
  // If no finger or mouse coordinate is active, drop grabs and bail
  if (!pointerCanvas) {
    ui.setFingerPosition(null);
    ds.isDraggingCornerA = false;
    ds.isDraggingCornerB = false;
    ds.isDraggingCornerC = false;
    ds.isDraggingCornerD = false;
    ds.isDraggingE1 = false;
    ds.isDraggingE2 = false;
    return;
  }

  // Strip pan offset from hand-tracked pointer so fixed-position
  // UI buttons (Continue, Yes/No) can be hit-tested correctly after panning.
  const pan = ui.getPanOffset();
  const pointerUI = isHandActive
    ? { x: pointerCanvas.x - pan.x, y: pointerCanvas.y - pan.y }
    : pointerCanvas;
  ui.setFingerPosition(pointerUI);

  const gridLayout = ui.getLayoutProperties();
  const TEXT_STRIP_H = 48;
  const GRID_RANGE = 10;

  // Grid center in canvas pixels, accounting for viewport pan
  const gridHeight = canvas.height - TEXT_STRIP_H;
  const gridCenterX = canvas.width / 2 + pan.x;
  const gridCenterY = TEXT_STRIP_H + gridHeight / 2 + pan.y;
  const scale = Math.min(canvas.width, gridHeight) / (GRID_RANGE * 2) * ui.getZoom();

  // ── PHASE: WAITING FOR OBJECT ──────────────────────────────────────────────
  if (state.phase === "WAITING_FOR_OBJECT") {
    const gridY = TEXT_STRIP_H;
    if (gridLayout) {
      const bx = (canvas.width - 180) / 2;
      const by = gridY + gridHeight / 2 - 24;
      if (
        pointerUI.x >= bx &&
        pointerUI.x <= bx + 180 &&
        pointerUI.y >= by &&
        pointerUI.y <= by + 48
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

  // ── Arrow Auto-Lock removed — handled in SHOW_BASIS_VECTORS section ──────────

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
      if (!isValidCorner(corner)) return { x: gridCenterX, y: gridCenterY };
      return { x: gridCenterX + corner.x * scale, y: gridCenterY - corner.y * scale };
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
      // [REMOVED: hasAdjustedCorner / showDonePrompt check — Done prompt removed]
    
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

      let rawGridX = (pointerCanvas.x - gridCenterX) / scale;
      let rawGridY = -(pointerCanvas.y - gridCenterY) / scale;

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

      const snapCanvasX = gridCenterX + clampedNearestX * scale;
      const snapCanvasY = gridCenterY - clampedNearestY * scale;
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
    return;
  }

  // ── PHASE: SHOW BASIS VECTORS (DRAG → PLACE → LOCK) ──────────────────────
  if (state.phase === "SHOW_BASIS_VECTORS") {
    const mat = ds.arrowMatrix;

    const dist = (a: Point2D, b: Point2D) =>
      Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

    const TARGET_E1 = { x: 3, y: 0 };
    const TARGET_E2 = { x: -1, y: 2 };

    const isOnTarget = (nx: number, ny: number, target: Point2D) =>
      nx === target.x && ny === target.y;

    // ── 1. Handle E1 dragging (dwell at target to lock) ─────────────────────
    if (ds.isDraggingE1) {
      let rawX = (pointerCanvas.x - gridCenterX) / scale;
      let rawY = -(pointerCanvas.y - gridCenterY) / scale;
      if (!isFinite(rawX) || !isFinite(rawY)) { rawX = mat[0]; rawY = mat[2]; }

      const clampedX = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawX));
      const clampedY = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawY));
      const safeX = Math.abs(clampedX) < EPSILON ? (clampedX >= 0 ? EPSILON : -EPSILON) : clampedX;
      const safeY = Math.abs(clampedY) < EPSILON ? (clampedY >= 0 ? EPSILON : -EPSILON) : clampedY;

      const nearestX = Math.round(safeX);
      const nearestY = Math.round(safeY);
      const snapCx = gridCenterX + nearestX * scale;
      const snapCy = gridCenterY - nearestY * scale;
      const dSnap = dist(pointerCanvas, { x: snapCx, y: snapCy });

      if (dSnap < ARROW_SNAP_PX) {
        mat[0] = nearestX;
        mat[2] = nearestY;

        const onTarget = isOnTarget(nearestX, nearestY, TARGET_E1);
        const sameInt = ds.e1DwellIntersection &&
          ds.e1DwellIntersection.x === nearestX && ds.e1DwellIntersection.y === nearestY;
        if (!sameInt) {
          ds.e1DwellStart = onTarget ? Date.now() : 0;
          ds.e1DwellIntersection = { x: nearestX, y: nearestY };
        } else if (onTarget && Date.now() - ds.e1DwellStart >= ARROW_DWELL_PLACE_MS) {
          ds.isDraggingE1 = false;
          ds.e1Locked = true;
          ds.e1DwellStart = 0;
          ds.e1DwellIntersection = null;
          ui.setArrowLocked(true, ds.e2Locked);
          ui.setArrowSnapped(true, false);
          console.log("[LineAR] e1 locked at TARGET_E1", mat[0], mat[2]);
        }
      } else {
        mat[0] = safeX;
        mat[2] = safeY;
        ds.e1DwellStart = 0;
        ds.e1DwellIntersection = null;
      }
      ui.setMatrix([...mat]);
    }

    // ── 2. Handle E2 dragging (dwell at target to lock) ─────────────────────
    if (ds.isDraggingE2) {
      let rawX = (pointerCanvas.x - gridCenterX) / scale;
      let rawY = -(pointerCanvas.y - gridCenterY) / scale;
      if (!isFinite(rawX) || !isFinite(rawY)) { rawX = mat[1]; rawY = mat[3]; }

      const clampedX = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawX));
      const clampedY = Math.max(-GRID_BOUNDS, Math.min(GRID_BOUNDS, rawY));
      const safeX = Math.abs(clampedX) < EPSILON ? (clampedX >= 0 ? EPSILON : -EPSILON) : clampedX;
      const safeY = Math.abs(clampedY) < EPSILON ? (clampedY >= 0 ? EPSILON : -EPSILON) : clampedY;

      const nearestX = Math.round(safeX);
      const nearestY = Math.round(safeY);
      const snapCx = gridCenterX + nearestX * scale;
      const snapCy = gridCenterY - nearestY * scale;
      const dSnap = dist(pointerCanvas, { x: snapCx, y: snapCy });

      if (dSnap < ARROW_SNAP_PX) {
        mat[1] = nearestX;
        mat[3] = nearestY;

        const onTarget = isOnTarget(nearestX, nearestY, TARGET_E2);
        const sameInt = ds.e2DwellIntersection &&
          ds.e2DwellIntersection.x === nearestX && ds.e2DwellIntersection.y === nearestY;
        if (!sameInt) {
          ds.e2DwellStart = onTarget ? Date.now() : 0;
          ds.e2DwellIntersection = { x: nearestX, y: nearestY };
        } else if (onTarget && Date.now() - ds.e2DwellStart >= ARROW_DWELL_PLACE_MS) {
          ds.isDraggingE2 = false;
          ds.e2Locked = true;
          ds.e2DwellStart = 0;
          ds.e2DwellIntersection = null;
          ui.setArrowLocked(ds.e1Locked, true);
          ui.setArrowSnapped(true, true);
          console.log("[LineAR] e2 locked at TARGET_E2", mat[1], mat[3]);
        }
      } else {
        mat[1] = safeX;
        mat[3] = safeY;
        ds.e2DwellStart = 0;
        ds.e2DwellIntersection = null;
      }
      ui.setMatrix([...mat]);
    }

    // ── 3. Grab detection — e1 first, then e2 only after e1 locked ──────────
    if (!ds.isDraggingE1 && !ds.isDraggingE2) {
      const e1Tip = { x: gridCenterX + mat[0] * scale, y: gridCenterY - mat[2] * scale };
      const e2Tip = { x: gridCenterX + mat[1] * scale, y: gridCenterY - mat[3] * scale };

      if (!ds.e1Locked) {
        if (dist(pointerCanvas, e1Tip) < ARROW_GRAB_RADIUS_PX) {
          ds.isDraggingE1 = true;
          console.log("[LineAR] e1 grabbed");
        }
      }
      if (ds.e1Locked && !ds.e2Locked) {
        if (dist(pointerCanvas, e2Tip) < ARROW_GRAB_RADIUS_PX) {
          ds.isDraggingE2 = true;
          console.log("[LineAR] e2 grabbed");
        }
      }
    }

    // ── 4. Update UI state ──────────────────────────────────────────────────
    const e1UISnapped = ds.e1Locked || (ds.isDraggingE1 && ds.e1DwellIntersection !== null);
    const e2UISnapped = ds.e2Locked || (ds.isDraggingE2 && ds.e2DwellIntersection !== null);
    ui.setArrowSnapped(e1UISnapped, e2UISnapped);
    ui.setDwellProgress(
      ds.isDraggingE1 && ds.e1DwellIntersection !== null
        ? Math.min(1, (Date.now() - ds.e1DwellStart) / ARROW_DWELL_PLACE_MS)
        : 0,
      ds.isDraggingE2 && ds.e2DwellIntersection !== null
        ? Math.min(1, (Date.now() - ds.e2DwellStart) / ARROW_DWELL_PLACE_MS)
        : 0,
    );

    // ── 5. Both locked → proceed ────────────────────────────────────────────
    if (ds.e1Locked && ds.e2Locked && !ds.arrowTransitionFired) {
      ds.arrowTransitionFired = true;
      dispatch({ type: "BASIS_ADJUSTED", payload: [...ds.arrowMatrix] });
    }

    return;
  }

  // ── PHASE: DIALOG CONFIRMATIONS (canvas buttons only) ─────────────────────
  if (state.phase === "CONFIRM_TRANSFORM" || state.phase === "CONFIRM_RESET") {
    const buttons = ui.getButtonRects();
    const btnYes = buttons.yes;
    const btnNo = buttons.no;
    if (!btnYes || !btnNo) return;

    const insideBox = (p: Point2D, box: DOMRect) =>
      p.x >= box.x && p.x <= box.x + box.width && p.y >= box.y && p.y <= box.y + box.height;

    if (insideBox(pointerUI, btnYes)) {
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onYes?.();
      }
    } else if (insideBox(pointerUI, btnNo)) {
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onNo?.();
      }
    } else {
      ds.dwellStart = 0;
    }
    return;
  }

  // ── PHASE: TRANSFORMED (canvas Continue button + pan) ─────────────────────
  if (state.phase === "TRANSFORMED") {
    const buttons = ui.getButtonRects();
    const btnContinue = buttons.continue;
    if (
      btnContinue &&
      pointerUI.x >= btnContinue.x &&
      pointerUI.x <= btnContinue.x + btnContinue.width &&
      pointerUI.y >= btnContinue.y &&
      pointerUI.y <= btnContinue.y + btnContinue.height
    ) {
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onContinue?.();
      }
    } else {
      ds.dwellStart = 0;
      const isPointerDown = ui.isMouseDown() || isHandActive;
      if (isPointerDown) {
        if (!ds.isPanning) {
          ds.isPanning = true;
          ds.panStartX = pointerCanvas.x;
          ds.panStartY = pointerCanvas.y;
          const offset = ui.getPanOffset();
          ds.panOffsetStartX = offset.x;
          ds.panOffsetStartY = offset.y;
        } else {
          ui.setPanOffset(
            ds.panOffsetStartX + (pointerCanvas.x - ds.panStartX),
            ds.panOffsetStartY + (pointerCanvas.y - ds.panStartY),
          );
        }
      } else {
        ds.isPanning = false;
      }
    }
    return;
  }

}
