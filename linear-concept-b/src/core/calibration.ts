import type { Point2D } from "../types/index.ts";

// ── ArUco marker definitions (physical printed markers, camera-detected) ──

export const ARUCO_MARKER_PX = 100;

export const ARUCO_MARKERS: { id: number; gridX: number; gridY: number }[] = [
  { id: 1, gridX: -9, gridY:  9 },
  { id: 2, gridX:  9, gridY:  9 },
  { id: 3, gridX: -9, gridY: -9 },
  { id: 4, gridX:  9, gridY: -9 },
];

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

let markerCanvasCache: HTMLCanvasElement[] | null = null;

function buildMarkerCanvasCache(): void {
  markerCanvasCache = [];
  for (const m of ARUCO_MARKERS) {
    const matrix = id2mat(m.id);
    const canvas = document.createElement("canvas");
    canvas.width = ARUCO_MARKER_PX;
    canvas.height = ARUCO_MARKER_PX;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, ARUCO_MARKER_PX, ARUCO_MARKER_PX);
    ctx.fillStyle = "#fff";
    const cs = ARUCO_MARKER_PX / 7;
    for (let row = 0; row < 5; row++)
      for (let col = 0; col < 5; col++)
        if (matrix[row]![col]) ctx.fillRect((col + 1) * cs, (row + 1) * cs, cs, cs);
    markerCanvasCache.push(canvas);
  }
}

export function getMarkerCanvas(id: number): HTMLCanvasElement | null {
  if (!markerCanvasCache) buildMarkerCanvasCache();
  const idx = ARUCO_MARKERS.findIndex(m => m.id === id);
  if (idx < 0 || !markerCanvasCache) return null;
  return markerCanvasCache[idx] ?? null;
}

const STORAGE_KEY = "linear_calibration_homography";

// ── Marker Detection (OpenCV.js ArUco 4×4) ──────────────────────────────
// [2024-06-03] Switched from js-aruco (5×5) to OpenCV built-in ArUco (4×4).
// Revert: restore the js-aruco detectProjectedMarkers and AR/CV declarations.

export interface PhysicalMarker {
  id: number;
  corners: Point2D[];
  center: Point2D;
  /** Grid position assigned by spatial sorting (populated after detection). */
  gridPos: Point2D;
}

/** 4×4 dictionary IDs to try (in order of preference). */
const DICT_4X4_VALUES = [0, 1, 2, 3]; // 50, 100, 250, 1000

const markerIds = new Set(ARUCO_MARKERS.map(m => m.id));

function cornersFromMat(cornersMat: any): Point2D[] {
  const pts: Point2D[] = [];
  for (let j = 0; j < 4; j++) {
    pts.push({ x: cornersMat.data32F[j * 2], y: cornersMat.data32F[j * 2 + 1] });
  }
  return pts;
}

/**
 * Detect physical 4×4 ArUco markers (IDs 1-4) using OpenCV.js.
 * Tries DICT_4X4_50 → 100 → 250 → 1000, returns first that finds ≥4 markers.
 */
export function detectProjectedMarkers(imageData: ImageData): PhysicalMarker[] | null {
  const cv = window.cv;
  if (!cv) { console.warn("[Calibration] cv not loaded"); return null; }

  if (!cv.aruco_ArucoDetector) return null;

  try {
    const img = cv.matFromImageData(imageData);
    const gray = new cv.Mat();
    cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY, 0);

    let results: PhysicalMarker[] = [];

    for (const dictId of DICT_4X4_VALUES) {
      const dict = cv.getPredefinedDictionary(dictId);
      const params = new cv.aruco_DetectorParameters();
      const detector = new cv.aruco_ArucoDetector(dict, params, new cv.aruco_RefineParameters(10, 3, true));
      const corners = new cv.MatVector();
      const ids = new cv.Mat();
      detector.detectMarkers(gray, corners, ids);

      const candidates: PhysicalMarker[] = [];
      for (let i = 0; i < ids.rows; i++) {
        const id = ids.intAt(i);
        if (!markerIds.has(id)) continue;
        const cm = corners.get(i);
        const pts = cornersFromMat(cm);
        cm.delete();
        const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
        const cy = pts.reduce((s, p) => s + p.y, 0) / 4;
        candidates.push({ id, corners: pts, center: { x: cx, y: cy }, gridPos: { x: 0, y: 0 } });
      }
      corners.delete();
      ids.delete();

      if (candidates.length >= 4) {
        assignGridPositions(candidates);
        results = candidates;
        break;
      }
    }

    img.delete();
    gray.delete();

    if (results.length < 4) return null;
    return results;
  } catch (e) {
    console.warn("[Calibration] OpenCV ArUco detection failed:", e);
    return null;
  }
}

/**
 * Assign grid positions to 4 markers by sorting them in camera space.
 * Top Y → back row (gridY = 9), left X within row → gridX = -9.
 * Removes dependency on hardcoded ID→grid mappings.
 */
function assignGridPositions(markers: PhysicalMarker[]): void {
  const sorted = [...markers].sort((a, b) => b.center.y - a.center.y);
  const top = [sorted[0]!, sorted[1]!].sort((a, b) => b.center.x - a.center.x);
  const bot = [sorted[2]!, sorted[3]!].sort((a, b) => b.center.x - a.center.x);
  top[0]!.gridPos = { x: -9, y: 9 };
  top[1]!.gridPos = { x: 9, y: 9 };
  bot[0]!.gridPos = { x: -9, y: -9 };
  bot[1]!.gridPos = { x: 9, y: -9 };
}

/**
 * Compute homography from detected markers using spatial-sorted grid positions.
 * Uses all 4 corners per marker (16 points) for a more robust RANSAC estimate.
 * Requires exactly 4 markers with gridPos populated (by detectProjectedMarkers).
 */
export function computeCalibrationFromMarkers(
  markers: PhysicalMarker[],
): { matrix: number[]; cameraPoints: Point2D[]; gridPoints: Point2D[] } | null {
  if (markers.length < 4) return null;

  const cameraPoints: Point2D[] = [];
  const gridPoints: Point2D[] = [];

  const CORNER_HALF = 0.35;
  const cornerOffsets = [
    { dx: -CORNER_HALF, dy: -CORNER_HALF },
    { dx:  CORNER_HALF, dy: -CORNER_HALF },
    { dx:  CORNER_HALF, dy:  CORNER_HALF },
    { dx: -CORNER_HALF, dy:  CORNER_HALF },
  ];

  for (const m of markers) {
    for (let j = 0; j < 4; j++) {
      const c = m.corners[j];
      const o = cornerOffsets[j];
      if (!c || !o) continue;
      cameraPoints.push(c);
      gridPoints.push({
        x: m.gridPos.x + o.dx,
        y: m.gridPos.y + o.dy,
      });
    }
  }

  const matrix = computeHomography(cameraPoints, gridPoints);
  if (!matrix) return null;

  return { matrix, cameraPoints, gridPoints };
}

// ── [DEPRECATED] Pure-JS ArUco detection ──────────────────────────────────
/*
export function hasAruco(): boolean { ... }
export function detectMarkers(...): DetectionResult | null { ... }
*/

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

// Note: getMarkerCanvas(id) is defined above in the Projected section.
