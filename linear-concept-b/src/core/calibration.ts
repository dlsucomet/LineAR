// ─── src/core/calibration.ts ──────────────────────────────────────────────────
// Projector ↔ Camera homography calibration.
// Computes a 3×3 homography that maps camera pixel coordinates to projector
// pixel coordinates using four corresponding point pairs (one per corner).

import type { HomographyCalibration, Matrix3x3, Point2D } from "../types/index.js";

// ---------------------------------------------------------------------------
// 4-point DLT homography
// ---------------------------------------------------------------------------

/**
 * Compute a 3×3 homography matrix from 4 point correspondences.
 *
 * @param src - 4 points in camera pixel space.
 * @param dst - 4 corresponding points in projector pixel space.
 * @returns A HomographyCalibration with the computed matrix.
 */
export function computeHomography(
  src: [Point2D, Point2D, Point2D, Point2D],
  dst: [Point2D, Point2D, Point2D, Point2D],
): HomographyCalibration {
  const A: number[][] = [];

  for (let i = 0; i < 4; i++) {
    const { x: sx, y: sy } = src[i]!;
    const { x: dx, y: dy } = dst[i]!;

    // Two equations per point correspondence (DLT)
    A.push([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx]);
    A.push([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy]);
  }

  const h = solveDLT(A);

  const matrix: Matrix3x3 = [
    h[0]!, h[1]!, h[2]!,
    h[3]!, h[4]!, h[5]!,
    h[6]!, h[7]!, h[8]!,
  ];

  return {
    matrix,
    calibrated: true,
    calibratedAt: new Date().toISOString(),
  };
}

/**
 * Apply a 3×3 homography to a single camera-space point,
 * returning the corresponding projector-space point.
 */
export function applyHomography(h: Matrix3x3, p: Point2D): Point2D {
  const [h0, h1, h2, h3, h4, h5, h6, h7, h8] = h;
  const w = h6! * p.x + h7! * p.y + h8!;
  return {
    x: (h0! * p.x + h1! * p.y + h2!) / w,
    y: (h3! * p.x + h4! * p.y + h5!) / w,
  };
}

/**
 * Persist calibration data to localStorage.
 */
export function saveCalibration(cal: HomographyCalibration): void {
  localStorage.setItem("lineAR_calibration", JSON.stringify(cal));
}

/**
 * Load calibration data from localStorage.
 * Returns null if none is stored.
 */
export function loadCalibration(): HomographyCalibration | null {
  const raw = localStorage.getItem("lineAR_calibration");
  if (!raw) return null;
  try {
    return JSON.parse(raw) as HomographyCalibration;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// DLT solver (Gaussian elimination on the 8×9 coefficient matrix)
// ---------------------------------------------------------------------------

/**
 * Solve the homogeneous system Ah = 0 for an 8×9 matrix A using
 * simple Gaussian elimination with back-substitution.
 * Returns the 9-element solution vector (normalised so h[8]=1).
 */
function solveDLT(A: number[][]): number[] {
  const rows = A.length;   // 8
  const cols = 9;

  // Forward elimination
  for (let col = 0; col < cols - 1; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(A[col]![col]!);
    for (let row = col + 1; row < rows; row++) {
      const val = Math.abs(A[row]![col]!);
      if (val > maxVal) { maxVal = val; maxRow = row; }
    }
    [A[col], A[maxRow]] = [A[maxRow]!, A[col]!];

    for (let row = col + 1; row < rows; row++) {
      if (Math.abs(A[col]![col]!) < 1e-12) continue;
      const factor = A[row]![col]! / A[col]![col]!;
      for (let k = col; k < cols; k++) {
        A[row]![k]! -= factor * A[col]![k]!;
      }
    }
  }

  // The 9th variable h[8] is free; set it to 1
  const h = new Array<number>(cols).fill(0);
  h[8] = 1;

  // Back-substitution
  for (let row = rows - 1; row >= 0; row--) {
    let sum = A[row]![cols - 1]! * h[cols - 1]!;
    for (let col = row + 1; col < cols - 1; col++) {
      sum += A[row]![col]! * h[col]!;
    }
    const pivot = A[row]![row];
    h[row] = pivot && Math.abs(pivot) > 1e-12 ? -sum / pivot : 0;
  }

  return h;
}

// ---------------------------------------------------------------------------
// Calibration UI state machine (used by the CalibrationPanel component)
// ---------------------------------------------------------------------------

export type CalibrationCorner = "top-left" | "top-right" | "bottom-right" | "bottom-left";

export const CALIBRATION_CORNER_ORDER: CalibrationCorner[] = [
  "top-left", "top-right", "bottom-right", "bottom-left",
];

export interface CalibrationSession {
  /** Points captured from camera so far (one per corner). */
  capturedCameraPoints: Partial<Record<CalibrationCorner, Point2D>>;
  /** Points projected (known) on projector canvas. */
  projectorPoints: Record<CalibrationCorner, Point2D>;
}

/**
 * Create a new calibration session with a known projector corner layout.
 */
export function createCalibrationSession(
  projectorWidth: number,
  projectorHeight: number,
  margin = 40,
): CalibrationSession {
  return {
    capturedCameraPoints: {},
    projectorPoints: {
      "top-left": { x: margin, y: margin },
      "top-right": { x: projectorWidth - margin, y: margin },
      "bottom-right": { x: projectorWidth - margin, y: projectorHeight - margin },
      "bottom-left": { x: margin, y: projectorHeight - margin },
    },
  };
}

/**
 * Attempt to finalise the calibration once all four camera points are captured.
 * Returns null if any point is missing.
 */
export function finaliseCalibration(
  session: CalibrationSession,
): HomographyCalibration | null {
  const corners = CALIBRATION_CORNER_ORDER;
  const srcPoints: Point2D[] = [];
  const dstPoints: Point2D[] = [];

  for (const corner of corners) {
    const cam = session.capturedCameraPoints[corner];
    const proj = session.projectorPoints[corner];
    if (!cam) return null;
    srcPoints.push(cam);
    dstPoints.push(proj);
  }

  return computeHomography(
    srcPoints as [Point2D, Point2D, Point2D, Point2D],
    dstPoints as [Point2D, Point2D, Point2D, Point2D],
  );
}
