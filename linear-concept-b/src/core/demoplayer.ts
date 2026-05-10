import type { Matrix2x2, Point2D } from "../types/index.ts";
import type { AppPhase } from "./appstatemachine.ts";

// ─── Pre-placed virtual object (grid coordinates, -10..10) ──────────────────

export const VIRTUAL_OBJECT = {
  corners: [
    { x: 2, y: 2 },
    { x: 5, y: 2 },
    { x: 5, y: 5 },
    { x: 2, y: 5 },
  ] as [Point2D, Point2D, Point2D, Point2D],
  center: { x: 3.5, y: 3.5 },
  color: "#e63946",
};

// ─── Preset transformation matrix (shear) ───────────────────────────────────

export const PRESET_MATRIX: Matrix2x2 = [1, 0.5, 0.3, 1];

// ─── Corner drag target (phase 3: POINTS_CALCULATED) ────────────────────────

export const DRAG_TARGET = {
  cornerIndex: 0,
  targetPos: { x: 4, y: 1 } as Point2D,
  snapRadius: 1.5,
};

// ─── Ghost arrow targets (phase 5: SHOW_BASIS_VECTORS) ──────────────────────

/** Ghost arrow tip positions derived from PRESET_MATRIX columns. */
export const GHOST_ARROWS = {
  e1: { x: PRESET_MATRIX[0], y: PRESET_MATRIX[2] } as Point2D,
  e2: { x: PRESET_MATRIX[1], y: PRESET_MATRIX[3] } as Point2D,
};

// ─── Phase timings (ms) ─────────────────────────────────────────────────────

export const PHASE_TIMINGS = {
  OBJECT_DETECTED: 1500,
  SHOW_CORNERS: 1000,
  SHOW_BASIS_VECTORS: 1500,
  TRANSFORMED: 2500,
};

// ─── Interaction thresholds ─────────────────────────────────────────────────

export const HOVER_DWELL_MS = 400;
export const GRAB_RADIUS_PX = 40;
export const ARROW_GRAB_RADIUS_PX = 30;
export const ARROW_SNAP_PX = 15;
export const CORNER_SNAP_RADIUS = 1.5;

// ─── Demo instruction overrides ─────────────────────────────────────────────

export const DEMO_INSTRUCTIONS: Record<string, string> = {
  WAITING_FOR_OBJECT: "Point at ▶ Start Demo and hold still",
  POINTS_CALCULATED: "Pinch the red corner and drag it to the target",
  SHOW_BASIS_VECTORS: "Drag each arrow tip to the dotted target",
  CONFIRM_TRANSFORM: "Basis Vectors adjusted. Start Linear Transformation",
};

// ─── Preset corners payload (for CORNERS_LOCKED dispatch) ───────────────────

export const PRESET_CORNERS_PAYLOAD = {
  corners: VIRTUAL_OBJECT.corners,
  center: VIRTUAL_OBJECT.center,
};
