// src/main.ts
// ── Wizard-of-oz demo: keyboard-driven (W) + hand tracking for basis vectors ──
import {
  createInitialState,
  transition,
  type AppState,
  type AppAction,
} from "./core/appstatemachine.ts";
import { TabletopUI } from "./components/tabletopui.ts";
import {
  VIRTUAL_OBJECT,
  GHOST_ARROWS,
  PHASE_TIMINGS,
  HOVER_DWELL_MS,
  ARROW_GRAB_RADIUS_PX,
  ARROW_SNAP_PX,
  ARROW_DWELL_PLACE_MS,
  ARROW_PLACED_LOCK_MS,
  ARROW_LOCK_COOLDOWN_MS,
} from "./core/demoplayer.ts";
import type { DetectedHand, HandLandmark, Matrix2x2, Point2D } from "./types/index.ts";

const CAM_W = 640;
const CAM_H = 360;
const TEXT_STRIP_H = 40;
const GRID_BOUNDS = 15;
const EPSILON = 0.001;
const HAND_TIMEOUT_MS = 400;

// ── Hand Tracking State ──────────────────────────────────────────────────────
let latestHandPosition: Point2D | null = null;
let lastHandSeenTime = 0;
let handEmptyCount = 0;

// ── Hand Tracking Mirror Config ──────────────────────────────────────────────
let flipH = false, flipV = false;

async function loadMirrorConfig() {
  try {
    const res = await fetch("/mirror-config.txt");
    const lines = (await res.text()).split("\n").filter(l => l.trim() && !l.trim().startsWith("#"));
    const parts = lines[0]?.trim().split(",") || [];
    flipH = parts[0] === "1";
    flipV = parts[1] === "1";
  } catch {
    flipH = false; flipV = false;
  }
}
loadMirrorConfig();

// ── Interaction State ────────────────────────────────────────────────────────

interface DemoInteractionState {
  dwellStart: number;
  isDraggingE1: boolean;
  isDraggingE2: boolean;
  e1Snapped: boolean;
  e2Snapped: boolean;
  e1Locked: boolean;
  e2Locked: boolean;
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
  arrowTransitionFired: boolean;
  previousPhase: string | null;
  arrowMatrix: Matrix2x2;
  isPanning: boolean;
  panStartX: number;
  panStartY: number;
  panOffsetStartX: number;
  panOffsetStartY: number;
}

function createDemoState(): DemoInteractionState {
  return {
    dwellStart: 0,
    isDraggingE1: false,
    isDraggingE2: false,
    e1Snapped: false,
    e2Snapped: false,
    e1Locked: false,
    e2Locked: false,
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
    arrowTransitionFired: false,
    previousPhase: null,
    arrowMatrix: [1, 0, 0, 1],
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    panOffsetStartX: 0,
    panOffsetStartY: 0,
  };
}

// ============================================================================

window.addEventListener("load", () => {
  const canvas = document.getElementById("main-canvas") as HTMLCanvasElement;
  const video = document.getElementById("video-input") as HTMLVideoElement;

  let state: AppState = createInitialState();
  state.objectCorners = VIRTUAL_OBJECT.corners;
  state.objectCenter = VIRTUAL_OBJECT.center;

  function dispatch(action: AppAction): void {
    const next = transition(state, action);
    if (next === state) return;
    state = next;
    ui.setAppliedMatrix(state.appliedMatrix);
    ui.setCorners(state.objectCorners);

    if (state.phase === "SHOW_CORNERS") {
      setTimeout(() => dispatch({ type: "PHASE_ADVANCE" }), PHASE_TIMINGS.SHOW_CORNERS);
    }
  }

  // ── Canvas UI Setup ────────────────────────────────────────────────────
  const ui = new TabletopUI(canvas);
  ui.resize(window.innerWidth, window.innerHeight);
  ui.onYes = () => dispatch({ type: "CONFIRM_YES" });
  ui.onNo = () => dispatch({ type: "CONFIRM_NO" });
  ui.onContinue = () => dispatch({ type: "TRANSFORMATION_DONE" });
  window.addEventListener("resize", () => ui.resize(window.innerWidth, window.innerHeight));

  ui.setGhostArrows(GHOST_ARROWS.e1, GHOST_ARROWS.e2);
  ui.setCorners(VIRTUAL_OBJECT.corners);
  ui.setVideoSource(video);
  ui.setFlipFlags(flipH, flipV);

  const ds = createDemoState();

  // ── W Key Fallback ─────────────────────────────────────────────────────
  window.addEventListener("keydown", (e) => {
    // F / V: toggle horizontal/vertical flip at runtime
    if (e.key === "f" || e.key === "F") {
      flipH = !flipH;
      ui.setFlipFlags(flipH, flipV);
      console.log("[LineAR] flipH =", flipH);
      return;
    }
    if (e.key === "v" || e.key === "V") {
      flipV = !flipV;
      ui.setFlipFlags(flipH, flipV);
      console.log("[LineAR] flipV =", flipV);
      return;
    }

    if (e.key !== "w" && e.key !== "W") return;

    if (state.phase === "SHOW_CORNERS") {
      dispatch({ type: "PHASE_ADVANCE" });
    } else if (state.phase === "SHOW_BASIS_VECTORS") {
      if (state.basisStep === 0) {
        ds.arrowMatrix[0] = GHOST_ARROWS.e1.x;
        ds.arrowMatrix[2] = GHOST_ARROWS.e1.y;
        ds.e1Placed = true;
        ds.e1Locked = true;
        ds.e1PlaceTime = 0;
        ds.e1LockTime = Date.now();
        dispatch({ type: "BASIS_STEP", payload: 1 });
        ui.setMatrix([...ds.arrowMatrix]);
        ui.setArrowSnapped(true, ds.e2Placed || ds.e2Locked);
        ui.setArrowLocked(true, ds.e2Locked);
      } else if (state.basisStep === 1) {
        ds.arrowMatrix[1] = GHOST_ARROWS.e2.x;
        ds.arrowMatrix[3] = GHOST_ARROWS.e2.y;
        ds.e2Placed = true;
        ds.e2Locked = true;
        ds.e2PlaceTime = 0;
        ds.e2LockTime = Date.now();
        dispatch({ type: "BASIS_STEP", payload: 2 });
        ui.setMatrix([...ds.arrowMatrix]);
        ui.setArrowSnapped(true, true);
        ui.setArrowLocked(true, true);
        ds.arrowTransitionFired = true;
        dispatch({ type: "BASIS_ADJUSTED", payload: [...ds.arrowMatrix] });
      }
    } else if (state.phase === "CONFIRM_TRANSFORM" || state.phase === "CONFIRM_RESET") {
      dispatch({ type: "CONFIRM_YES" });
    } else if (state.phase === "TRANSFORMED") {
      dispatch({ type: "TRANSFORMATION_DONE" });
    }
  });

  // ── Render Loop ────────────────────────────────────────────────────────
  (function renderLoop() {
    // Sync flip flags (may have been set asynchronously by loadMirrorConfig)
    ui.setFlipFlags(flipH, flipV);
    ui.draw(state);

    if (state.phase !== ds.previousPhase) {
      ds.previousPhase = state.phase;
      if (state.phase === "SHOW_BASIS_VECTORS") {
        ds.arrowMatrix = [...state.pendingMatrix];
        ui.setMatrix([...state.pendingMatrix]);
      }
      if (state.phase === "TRANSFORMED") {
        ds.isPanning = false;
        if (state.objectCorners && state.appliedMatrix) {
          const grid = ui.getLastGridRect();
          if (grid) {
            ui.autoFrameTransformed(state.objectCorners, state.appliedMatrix, grid);
          }
        }
      }
      if (state.phase === "CONFIRM_RESET") {
        ds.isPanning = false;
      }
      if (state.phase === "SHOW_CORNERS") {
        ui.setPanBounds(null);
        ui.setPanOffset(0, 0);
      }
    }

    // Resolve pointer: hand wins, mouse is fallback
    let activePointer: Point2D | null = null;
    const isHandActive = !!(latestHandPosition && (Date.now() - lastHandSeenTime < HAND_TIMEOUT_MS));
    if (isHandActive) {
      activePointer = latestHandPosition;
    } else {
      activePointer = ui.getMouseCanvasPos();
    }

    processInteractionFrame(activePointer, state, dispatch, ui, canvas, ds, isHandActive);

    requestAnimationFrame(renderLoop);
  })();

  // ── Start Camera + Hand Tracker ────────────────────────────────────────
  startCamera(video, CAM_W, CAM_H)
    .then(() => setupHandTracker(canvas, video, ui))
    .catch((err) => console.warn("[LineAR] Camera unavailable:", err));
});

// ============================================================================
// Interaction Engine
// ============================================================================

function processInteractionFrame(
  pointerCanvas: Point2D | null,
  state: AppState,
  dispatch: (action: AppAction) => void,
  ui: TabletopUI,
  canvas: HTMLCanvasElement,
  ds: DemoInteractionState,
  isHandActive: boolean,
): void {
  ui.setFingerPosition(pointerCanvas);

  if (!pointerCanvas) {
    ds.isDraggingE1 = false;
    ds.isDraggingE2 = false;
    return;
  }

  const GRID_RANGE = 10;
  const pan = ui.getPanOffset();
  const gridHeight = canvas.height - TEXT_STRIP_H;
  const gridCenterX = canvas.width / 2 + pan.x;
  const gridCenterY = TEXT_STRIP_H + gridHeight / 2 + pan.y;
  const scale = Math.min(canvas.width, gridHeight) / (GRID_RANGE * 2);

  // ── PHASE: SHOW BASIS VECTORS (DRAG → PLACE → LOCK) ─────────────────────
  if (state.phase === "SHOW_BASIS_VECTORS") {
    const mat = ds.arrowMatrix;

    const dist = (a: Point2D, b: Point2D) =>
      Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

    // ── 1. Auto-lock: placed arrow → locked after ARROW_PLACED_LOCK_MS ─────
    if (ds.e1Placed && !ds.e1Locked && ds.e1PlaceTime > 0 &&
        Date.now() - ds.e1PlaceTime >= ARROW_PLACED_LOCK_MS) {
      ds.e1Locked = true;
      ds.e1LockTime = Date.now();
      ds.e1PlaceTime = 0;
      ui.setArrowLocked(true, ds.e2Locked);
      ui.setArrowSnapped(true, ds.e2Placed || ds.e2Locked);
      if (state.basisStep === 0) dispatch({ type: "BASIS_STEP", payload: 1 });
    }
    if (ds.e2Placed && !ds.e2Locked && ds.e2PlaceTime > 0 &&
        Date.now() - ds.e2PlaceTime >= ARROW_PLACED_LOCK_MS) {
      ds.e2Locked = true;
      ds.e2LockTime = Date.now();
      ds.e2PlaceTime = 0;
      ui.setArrowLocked(ds.e1Locked, true);
      ui.setArrowSnapped(ds.e1Placed || ds.e1Locked, true);
      if (state.basisStep === 1) dispatch({ type: "BASIS_STEP", payload: 2 });
    }

    // ── Cooldown: locked arrow → grabbable after ARROW_LOCK_COOLDOWN_MS ─────
    if (ds.e1Locked && ds.e1LockTime > 0 &&
        Date.now() - ds.e1LockTime >= ARROW_LOCK_COOLDOWN_MS) {
      ds.e1Locked = false;
      ds.e1LockTime = 0;
      ds.e1Placed = true;
      ds.e1PlaceTime = Date.now();
      ds.e1ReGrabCooldown = Date.now() + 800;
      ui.setArrowLocked(false, ds.e2Locked);
      ui.setArrowSnapped(true, ds.e2Placed || ds.e2Locked);
    }
    if (ds.e2Locked && ds.e2LockTime > 0 &&
        Date.now() - ds.e2LockTime >= ARROW_LOCK_COOLDOWN_MS) {
      ds.e2Locked = false;
      ds.e2LockTime = 0;
      ds.e2Placed = true;
      ds.e2PlaceTime = Date.now();
      ds.e2ReGrabCooldown = Date.now() + 800;
      ui.setArrowLocked(ds.e1Locked, false);
      ui.setArrowSnapped(ds.e1Placed || ds.e1Locked, true);
    }

    // ── 2. Handle E1 dragging (dwell-on-snap) ─────────────────────────────
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

        const sameInt = ds.e1DwellIntersection &&
          ds.e1DwellIntersection.x === nearestX && ds.e1DwellIntersection.y === nearestY;
        if (!sameInt) {
          ds.e1DwellStart = Date.now();
          ds.e1DwellIntersection = { x: nearestX, y: nearestY };
        } else if (Date.now() - ds.e1DwellStart >= ARROW_DWELL_PLACE_MS) {
          ds.isDraggingE1 = false;
          ds.e1Placed = true;
          ds.e1PlaceTime = Date.now();
          ds.e1ReGrabCooldown = Date.now() + 800;
          ds.e1DwellStart = 0;
          ds.e1DwellIntersection = null;
        }
      } else {
        mat[0] = safeX;
        mat[2] = safeY;
        ds.e1DwellStart = 0;
        ds.e1DwellIntersection = null;
      }
      ui.setMatrix([...mat]);
    }

    // ── 3. Handle E2 dragging (dwell-on-snap) ─────────────────────────────
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

        const sameInt = ds.e2DwellIntersection &&
          ds.e2DwellIntersection.x === nearestX && ds.e2DwellIntersection.y === nearestY;
        if (!sameInt) {
          ds.e2DwellStart = Date.now();
          ds.e2DwellIntersection = { x: nearestX, y: nearestY };
        } else if (Date.now() - ds.e2DwellStart >= ARROW_DWELL_PLACE_MS) {
          ds.isDraggingE2 = false;
          ds.e2Placed = true;
          ds.e2PlaceTime = Date.now();
          ds.e2ReGrabCooldown = Date.now() + 800;
          ds.e2DwellStart = 0;
          ds.e2DwellIntersection = null;
        }
      } else {
        mat[1] = safeX;
        mat[3] = safeY;
        ds.e2DwellStart = 0;
        ds.e2DwellIntersection = null;
      }
      ui.setMatrix([...mat]);
    }

    // ── 4. Grab / re-grab detection (only when not dragging) ──────────────
    if (!ds.isDraggingE1 && !ds.isDraggingE2) {
      const e1Tip = { x: gridCenterX + mat[0] * scale, y: gridCenterY - mat[2] * scale };
      const e2Tip = { x: gridCenterX + mat[1] * scale, y: gridCenterY - mat[3] * scale };

      if (!ds.e1Placed && !ds.e1Locked) {
        if (dist(pointerCanvas, e1Tip) < ARROW_GRAB_RADIUS_PX) {
          ds.isDraggingE1 = true;
        }
      }
      if (ds.e1Placed && !ds.e1Locked && Date.now() > ds.e1ReGrabCooldown) {
        if (dist(pointerCanvas, e1Tip) < ARROW_GRAB_RADIUS_PX) {
          ds.isDraggingE1 = true;
          ds.e1Placed = false;
          ds.e1PlaceTime = 0;
        }
      }
      if (!ds.isDraggingE1 && !ds.e2Placed && !ds.e2Locked) {
        if (dist(pointerCanvas, e2Tip) < ARROW_GRAB_RADIUS_PX) {
          ds.isDraggingE2 = true;
        }
      }
      if (!ds.isDraggingE1 && ds.e2Placed && !ds.e2Locked && Date.now() > ds.e2ReGrabCooldown) {
        if (dist(pointerCanvas, e2Tip) < ARROW_GRAB_RADIUS_PX) {
          ds.isDraggingE2 = true;
          ds.e2Placed = false;
          ds.e2PlaceTime = 0;
        }
      }
    }

    // Update UI with placed/locked/dwelling state
    const e1UISnapped = ds.e1Placed || ds.e1Locked || (ds.isDraggingE1 && ds.e1DwellIntersection !== null);
    const e2UISnapped = ds.e2Placed || ds.e2Locked || (ds.isDraggingE2 && ds.e2DwellIntersection !== null);
    ui.setArrowSnapped(e1UISnapped, e2UISnapped);
    ui.setDwellProgress(
      ds.isDraggingE1 && ds.e1DwellIntersection !== null
        ? Math.min(1, (Date.now() - ds.e1DwellStart) / ARROW_DWELL_PLACE_MS)
        : ds.isDraggingE2 && ds.e2DwellIntersection !== null
        ? Math.min(1, (Date.now() - ds.e2DwellStart) / ARROW_DWELL_PLACE_MS)
        : 0,
      ds.isDraggingE2 && ds.e2DwellIntersection !== null
        ? Math.min(1, (Date.now() - ds.e2DwellStart) / ARROW_DWELL_PLACE_MS)
        : ds.isDraggingE1 && ds.e1DwellIntersection !== null
        ? Math.min(1, (Date.now() - ds.e1DwellStart) / ARROW_DWELL_PLACE_MS)
        : 0,
    );

    // ── 5. Both locked → proceed ─────────────────────────────────────────
    if (ds.e1Locked && ds.e2Locked && !ds.arrowTransitionFired) {
      ds.arrowTransitionFired = true;
      dispatch({ type: "BASIS_ADJUSTED", payload: [...ds.arrowMatrix] });
    }

    return;
  }

  // ── PHASE: DIALOG CONFIRMATIONS (canvas buttons) ────────────────────────
  if (state.phase === "CONFIRM_TRANSFORM" || state.phase === "CONFIRM_RESET") {
    const buttons = ui.getButtonRects();
    const btnYes = buttons.yes;
    const btnNo = buttons.no;
    if (!btnYes || !btnNo) return;

    const insideBox = (p: Point2D, box: DOMRect) =>
      p.x >= box.x && p.x <= box.x + box.width && p.y >= box.y && p.y <= box.y + box.height;

    if (insideBox(pointerCanvas, btnYes)) {
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      ui.setCursorDwellProgress(Math.min(1, (Date.now() - ds.dwellStart) / HOVER_DWELL_MS));
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onYes?.();
      }
    } else if (insideBox(pointerCanvas, btnNo)) {
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      ui.setCursorDwellProgress(Math.min(1, (Date.now() - ds.dwellStart) / HOVER_DWELL_MS));
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onNo?.();
      }
    } else {
      ds.dwellStart = 0;
      ui.setCursorDwellProgress(0);
    }
    return;
  }

  // ── PHASE: TRANSFORMED (Continue button + pan) ──────────────────────────
  if (state.phase === "TRANSFORMED") {
    const buttons = ui.getButtonRects();
    const btnContinue = buttons.continue;

    if (btnContinue && pointerCanvas.x >= btnContinue.x && pointerCanvas.x <= btnContinue.x + btnContinue.width &&
        pointerCanvas.y >= btnContinue.y && pointerCanvas.y <= btnContinue.y + btnContinue.height) {
      if (ds.dwellStart === 0) ds.dwellStart = Date.now();
      ui.setCursorDwellProgress(Math.min(1, (Date.now() - ds.dwellStart) / HOVER_DWELL_MS));
      if (Date.now() - ds.dwellStart > HOVER_DWELL_MS) {
        ds.dwellStart = 0;
        ui.onContinue?.();
      }
    } else {
      ds.dwellStart = 0;
      ui.setCursorDwellProgress(0);
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

// ============================================================================
// Camera + Hand Tracker
// ============================================================================

async function startCamera(video: HTMLVideoElement, width: number, height: number): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width, height, frameRate: 30 },
  });
  video.srcObject = stream;
  await new Promise<void>((resolve) => { video.onloadedmetadata = () => resolve(); });
  await video.play();
}

async function setupHandTracker(canvas: HTMLCanvasElement, video: HTMLVideoElement, ui: TabletopUI): Promise<void> {
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
      x: (flipH ? vw - lm.x : lm.x) * uniformScale + offsetX,
      y: (flipV ? vh - lm.y : lm.y) * uniformScale + offsetY,
    });

    if (hands.length > 0) {
      const hand = hands[0]!;
      if (hand.score >= 0.1 && hand.landmarks[8]) {
        const rawPoint = toMirroredCanvas(hand.landmarks[8]);
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
        landmarks: h.landmarks.map((lm) => ({ ...lm, ...toMirroredCanvas(lm) })),
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

async function safeImport<T>(factory: () => Promise<T>): Promise<Partial<T>> {
  try { return await factory(); } catch { return {}; }
}
