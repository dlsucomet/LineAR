import type { Point2D } from "../types/index.ts";

declare const CV: {
  Image: new (width?: number, height?: number, data?: any) => any;
};
declare const AR: {
  Detector: new () => { detect(image: any): { id: number; corners: { x: number; y: number }[] }[] };
};

// ── Marker Definitions ────────────────────────────────────────────────────

/** ArUco markers projected at these grid positions. 4×4 grid. */
export const ARUCO_MARKERS: { id: number; gridX: number; gridY: number }[] = [
  { id: 0,  gridX: -9, gridY:  9 },
  { id: 1,  gridX: -3, gridY:  9 },
  { id: 2,  gridX:  3, gridY:  9 },
  { id: 3,  gridX:  9, gridY:  9 },
  { id: 4,  gridX: -9, gridY:  3 },
  { id: 5,  gridX: -3, gridY:  3 },
  { id: 6,  gridX:  3, gridY:  3 },
  { id: 7,  gridX:  9, gridY:  3 },
  { id: 8,  gridX: -9, gridY: -3 },
  { id: 9,  gridX: -3, gridY: -3 },
  { id: 10, gridX:  3, gridY: -3 },
  { id: 11, gridX:  9, gridY: -3 },
  { id: 12, gridX: -9, gridY: -9 },
  { id: 13, gridX: -3, gridY: -9 },
  { id: 14, gridX:  3, gridY: -9 },
  { id: 15, gridX:  9, gridY: -9 },
];

const ARUCO_MARKER_PX = 120;
const STORAGE_KEY = "linear_calibration_homography";

// ── ArUco Support Check ────────────────────────────────────────────────────

export function hasAruco(): boolean {
  return !!(typeof AR !== "undefined" && AR?.Detector);
}

// ── ArUco 5x5 Marker Generator (reverses mat2id from aruco.js) ────────────────

const ROW_PATTERNS: number[][] = [
  [1, 0, 0, 0, 0],
  [1, 0, 1, 1, 1],
  [0, 1, 0, 0, 1],
  [0, 1, 1, 1, 0],
];

function id2mat(id: number): number[][] {
  const mat: number[][] = [];
  for (let row = 0; row < 5; row++) {
    const bits = (id >> (2 * (4 - row))) & 3;
    mat.push([...ROW_PATTERNS[bits]!]);
  }
  return mat;
}

// ── Marker Image Cache (ArUco) ──────────────────────────────────────────────

let arucoCache: HTMLCanvasElement[] | null = null;

function getArucoCanvas(id: number): HTMLCanvasElement | null {
  if (!arucoCache) buildArucoCache();
  return arucoCache?.[id] ?? null;
}

function buildArucoCache(): void {
  arucoCache = [];
  console.log("[Calibration] Building ArUco marker cache...");
  for (const m of ARUCO_MARKERS) {
    const matrix = id2mat(m.id);
    const canvas = document.createElement("canvas");
    canvas.width = ARUCO_MARKER_PX;
    canvas.height = ARUCO_MARKER_PX;
    const ctx = canvas.getContext("2d")!;
    if (!ctx) { console.warn("[Calibration] Failed to get 2D context for marker", m.id); continue; }

    // Black background (border)
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, ARUCO_MARKER_PX, ARUCO_MARKER_PX);

    // Draw white data cells
    ctx.fillStyle = "#fff";
    const cellSize = ARUCO_MARKER_PX / 7;
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        if (matrix[row]![col]) {
          ctx.fillRect(
            (col + 1) * cellSize,
            (row + 1) * cellSize,
            cellSize,
            cellSize,
          );
        }
      }
    }

    arucoCache.push(canvas);
  }
  console.log("[Calibration] ArUco cache built:", arucoCache.length, "markers");
}

// ── Detection ───────────────────────────────────────────────────────────────

export interface DetectionResult {
  cameraPoints: Point2D[];
  gridPoints: Point2D[];
  method: "aruco";
}

/**
 * Detect calibration markers in a camera frame using pure-JS ArUco.
 */
export function detectMarkers(
  imageData: ImageData,
  _camW: number,
  _camH: number,
): DetectionResult | null {
  if (!hasAruco()) return null;

  const cvImage = new CV.Image(imageData.width, imageData.height, imageData.data);
  return detectAruco(cvImage);
}

function detectAruco(cvImage: any): DetectionResult | null {
  try {
    const detector = new AR.Detector();
    const markers = detector.detect(cvImage);

    if (!markers || markers.length === 0) return null;

    const cameraPoints: Point2D[] = [];
    const gridPoints: Point2D[] = [];

    for (const marker of markers) {
      const markerDef = ARUCO_MARKERS.find((m) => m.id === marker.id);
      if (!markerDef) continue;

      const cx = marker.corners.reduce((s: number, c: { x: number; y: number }) => s + c.x, 0) / 4;
      const cy = marker.corners.reduce((s: number, c: { x: number; y: number }) => s + c.y, 0) / 4;

      cameraPoints.push({ x: cx, y: cy });
      gridPoints.push({ x: markerDef.gridX, y: markerDef.gridY });
    }

    if (cameraPoints.length < 4) return null;

    return { cameraPoints, gridPoints, method: "aruco" };
  } catch (e) {
    console.warn("[Calibration] ArUco detection failed:", e);
    return null;
  }
}

// ── Homography ──────────────────────────────────────────────────────────────

/**
 * Compute a 3×3 homography matrix from point correspondences.
 * Returns row-major array of 9 numbers, or null on failure.
 */
export function computeHomography(
  srcPoints: Point2D[],
  dstPoints: Point2D[],
): number[] | null {
  const cv = window.cv;
  if (!cv || srcPoints.length < 4 || dstPoints.length < 4) return null;

  try {
    const srcData: number[] = [];
    for (const p of srcPoints) srcData.push(p.x, p.y);
    const dstData: number[] = [];
    for (const p of dstPoints) dstData.push(p.x, p.y);
    const srcMat = cv.matFromArray(srcPoints.length, 1, cv.CV_32FC2, srcData);
    const dstMat = cv.matFromArray(dstPoints.length, 1, cv.CV_32FC2, dstData);

    const mask = new cv.Mat();
    const H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3, mask);

    const matrix: number[] = [];
    for (let i = 0; i < 9; i++) {
      matrix.push(H.data64F[i]);
    }

    srcMat.delete();
    dstMat.delete();
    mask.delete();
    H.delete();

    return matrix;
  } catch (e) {
    console.warn("[Calibration] Homography computation failed:", e);
    return null;
  }
}

// ── Persistence ───────────────────────────────────────────────────────────────

export function saveCalibration(matrix: number[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(matrix));
    console.log("[Calibration] Homography saved to localStorage");
  } catch (e) {
    console.warn("[Calibration] Failed to save homography:", e);
  }
}

export function loadCalibration(): number[] | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const matrix = JSON.parse(raw);
    if (!Array.isArray(matrix) || matrix.length !== 9) return null;
    if (matrix.some(v => typeof v !== 'number' || !isFinite(v))) {
      console.warn("[Calibration] Stale/invalid homography found — clearing");
      clearCalibration();
      return null;
    }
    console.log("[Calibration] Homography loaded from localStorage");
    return matrix;
  } catch {
    return null;
  }
}

export function clearCalibration(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
    console.log("[Calibration] Homography cleared");
  } catch { /* ignore */ }
}

// ── Rendering helpers (used by tabletopui) ──────────────────────────────────

/** Returns the ArUco marker canvas for a given ID, or null. */
export function getMarkerCanvas(id: number): HTMLCanvasElement | null {
  return getArucoCanvas(id);
}

export { ARUCO_MARKER_PX };
