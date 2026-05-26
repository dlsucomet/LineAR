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

// Corner drag now snaps to nearest integer grid intersection.
// Snap radius is defined in tabletopui.ts as GRID_SNAP_RADIUS (0.5 units).
// Phase advances via a "Done ✓" button after both C and D have been placed.

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
export const GRAB_RADIUS_PX = 60;
export const ARROW_GRAB_RADIUS_PX = 70;
export const ARROW_SNAP_PX = 25;
export const ARROW_DWELL_PLACE_MS = 600;
export const ARROW_PLACED_LOCK_MS = 3000;
export const ARROW_LOCK_COOLDOWN_MS = 5000;
export const CORNER_SNAP_RADIUS = 1.5;

// ─── Demo instruction overrides ─────────────────────────────────────────────

export const DEMO_INSTRUCTIONS: Record<string, string> = {
  WAITING_FOR_OBJECT: "Point at ▶ Start Demo and hold still",
  POINTS_CALCULATED: "Drag bottom corners to grid positions, then tap Done ✓",  // [REMOVED: Done prompt — kept for demo instructions]
  SHOW_BASIS_VECTORS: "Drag each arrow tip to the dotted target",
  CONFIRM_TRANSFORM: "Basis Vectors adjusted. Start Linear Transformation",
};

// ─── Preset corners payload (for CORNERS_LOCKED dispatch) ───────────────────

export const PRESET_CORNERS_PAYLOAD = {
  corners: VIRTUAL_OBJECT.corners,
  center: VIRTUAL_OBJECT.center,
};
