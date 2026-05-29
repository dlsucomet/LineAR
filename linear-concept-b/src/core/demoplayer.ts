import type { Matrix2x2, Point2D } from "../types/index.ts";

// ─── Pre-placed virtual object (grid coordinates, -10..10) ──────────────────
// 5 grid units tall × 3 grid units wide, centered at origin

export const VIRTUAL_OBJECT = {
  corners: [
    { x: -1, y:  3 },  // top-left
    { x:  2, y:  3 },  // top-right
    { x:  2, y: -2 },  // bottom-right
    { x: -1, y: -2 },  // bottom-left
  ] as [Point2D, Point2D, Point2D, Point2D],
  center: { x: 0.5, y: 0.5 },
  color: "#e63946",
};

// ─── Preset transformation matrix (skew + scale) ────────────────────────────
// e₁ target (2,1), e₂ target (-1,2) — both at integer grid intersections

export const PRESET_MATRIX: Matrix2x2 = [2, -1, 1, 2];

// ─── Ghost arrow targets (SHOW_BASIS_VECTORS) ────────────────────────────────

/** Ghost arrow tip positions derived from PRESET_MATRIX columns. */
export const GHOST_ARROWS = {
  e1: { x: PRESET_MATRIX[0], y: PRESET_MATRIX[2] } as Point2D,
  e2: { x: PRESET_MATRIX[1], y: PRESET_MATRIX[3] } as Point2D,
};

// ─── Phase timings (ms) ─────────────────────────────────────────────────────

export const PHASE_TIMINGS = {
  SHOW_CORNERS: 1500,
};

// ─── Interaction thresholds ─────────────────────────────────────────────────

export const HOVER_DWELL_MS = 400;
export const ARROW_GRAB_RADIUS_PX = 70;
export const ARROW_SNAP_PX = 25;
export const ARROW_DWELL_PLACE_MS = 600;
export const ARROW_PLACED_LOCK_MS = 3500;
export const ARROW_LOCK_COOLDOWN_MS = 5000;

// ─── Demo instruction overrides ─────────────────────────────────────────────

export const DEMO_INSTRUCTIONS: Record<string, string> = {
  SHOW_CORNERS: "Place object on the highlighted area",
  SHOW_BASIS_VECTORS: "Drag each arrow tip to the colored target circle",
  CONFIRM_TRANSFORM: "Basis Vectors adjusted. Start Linear Transformation",
};


