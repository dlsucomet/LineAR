import type { Point2D } from "../types/index.ts";

// ── Marker Definitions ────────────────────────────────────────────────────

/** ArUco markers (DICT_4X4_50) projected at these grid positions. 4×3 grid. */
export const ARUCO_MARKERS: { id: number; gridX: number; gridY: number }[] = [
  { id: 0,  gridX: -7, gridY:  7 },
  { id: 1,  gridX: -3, gridY:  7 },
  { id: 2,  gridX:  3, gridY:  7 },
  { id: 3,  gridX:  7, gridY:  7 },
  { id: 4,  gridX: -7, gridY:  2 },
  { id: 5,  gridX: -3, gridY:  2 },
  { id: 6,  gridX:  3, gridY:  2 },
  { id: 7,  gridX:  7, gridY:  2 },
  { id: 8,  gridX: -7, gridY: -3 },
  { id: 9,  gridX: -3, gridY: -3 },
  { id: 10, gridX:  3, gridY: -3 },
  { id: 11, gridX:  7, gridY: -3 },
];

/** Fallback corner circles at 4 corners (simpler detection). */
export const CIRCLE_MARKERS: Point2D[] = [
  { x: -7.5, y:  7.5 },
  { x:  7.5, y:  7.5 },
  { x:  7.5, y: -7.5 },
  { x: -7.5, y: -7.5 },
];

const ARUCO_MARKER_PX = 48;
const STORAGE_KEY = "linear_calibration_homography";

// ── ArUco Support Check ────────────────────────────────────────────────────

export function hasAruco(): boolean {
  const cv = window.cv;
  return !!(cv?.aruco?.detectMarkers && cv?.aruco?.drawMarker);
}

// ── Marker Image Cache (ArUco) ──────────────────────────────────────────────

let arucoCache: HTMLCanvasElement[] | null = null;

function getArucoCanvas(id: number): HTMLCanvasElement | null {
  if (!arucoCache) buildArucoCache();
  return arucoCache?.[id] ?? null;
}

function buildArucoCache(): void {
  const cv = window.cv;
  if (!cv?.aruco?.drawMarker) { arucoCache = []; return; }

  try {
    const dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50);
    arucoCache = [];
    for (const m of ARUCO_MARKERS) {
      const markerMat = new cv.Mat();
      cv.aruco.drawMarker(dict, m.id, ARUCO_MARKER_PX, markerMat, 1);
      const temp = document.createElement("canvas");
      temp.width = ARUCO_MARKER_PX;
      temp.height = ARUCO_MARKER_PX;
      cv.imshow(temp, markerMat);
      markerMat.delete();
      arucoCache.push(temp);
    }
  } catch (e) {
    console.warn("[Calibration] Failed to build ArUco cache:", e);
    arucoCache = [];
  }
}

// ── Detection ───────────────────────────────────────────────────────────────

export interface DetectionResult {
  cameraPoints: Point2D[];  // detected positions in camera pixel space
  gridPoints: Point2D[];    // corresponding known grid positions
  method: "aruco" | "circles";
}

/**
 * Attempt to detect calibration markers in a camera frame.
 * Tries ArUco first, falls back to circle blob detection.
 */
export function detectMarkers(
  imageData: ImageData,
  camW: number,
  camH: number,
): DetectionResult | null {
  const cv = window.cv;
  if (!cv) return null;

  const src = cv.matFromImageData(imageData);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

  let result: DetectionResult | null = null;

  // Try ArUco
  if (hasAruco()) {
    result = detectAruco(gray, cv);
  }

  // Fallback: corner circles
  if (!result) {
    result = detectCircles(gray, cv, camW, camH);
  }

  src.delete();
  gray.delete();
  return result;
}

function detectAruco(gray: any, cv: any): DetectionResult | null {
  try {
    const dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50);
    const corners = new cv.MatVector();
    const ids = new cv.Mat();
    cv.aruco.detectMarkers(gray, dict, corners, ids);

    if (ids.rows === 0) {
      corners.delete();
      ids.delete();
      return null;
    }

    const cameraPoints: Point2D[] = [];
    const gridPoints: Point2D[] = [];

    for (let i = 0; i < ids.rows; i++) {
      const id = ids.data32S[i];
      const marker = ARUCO_MARKERS.find((m) => m.id === id);
      if (!marker) continue;

      const cornerMat = corners.get(i);
      // cornerMat is 4×1 CV_32FC2 (or 1×4 CV_32FC2 — varies by OpenCV.js version)
      let cx = 0, cy = 0;
      const data = cornerMat.data32S ?? cornerMat.data32F;
      if (!data) continue;

      // Handle both possible layouts
      const rows = cornerMat.rows;
      const cols = cornerMat.cols;
      const step = cornerMat.channels();

      if (rows === 4 && cols === 1 && step >= 2) {
        // Layout: 4 rows × 1 col, each element is (x, y)
        for (let j = 0; j < 4; j++) {
          cx += data[j * step];
          cy += data[j * step + 1];
        }
      } else if (rows === 1 && cols >= 4 && step >= 2) {
        // Layout: 1 row × N cols
        for (let j = 0; j < Math.min(cols, 4); j++) {
          cx += data[j * step];
          cy += data[j * step + 1];
        }
      }

      cx /= 4;
      cy /= 4;

      cameraPoints.push({ x: cx, y: cy });
      gridPoints.push({ x: marker.gridX, y: marker.gridY });
    }

    corners.delete();
    ids.delete();

    if (cameraPoints.length < 4) return null;

    return { cameraPoints, gridPoints, method: "aruco" };
  } catch (e) {
    console.warn("[Calibration] ArUco detection failed:", e);
    return null;
  }
}

function detectCircles(gray: any, cv: any, camW: number, camH: number): DetectionResult | null {
  try {
    // Threshold to find bright blobs (projected circles are bright on white cartolina)
    const thresholded = new cv.Mat();
    cv.threshold(gray, thresholded, 200, 255, cv.THRESH_BINARY);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(thresholded, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const blobs: { x: number; y: number; area: number }[] = [];

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);
      if (area < 100 || area > camW * camH * 0.1) { contour.delete(); continue; }

      const perimeter = cv.arcLength(contour, true);
      if (perimeter <= 0) { contour.delete(); continue; }
      const circularity = 4 * Math.PI * area / (perimeter * perimeter);
      if (circularity < 0.5) { contour.delete(); continue; }

      const M = cv.moments(contour);
      if (M.m00 === 0) { contour.delete(); continue; }
      const cx = M.m10 / M.m00;
      const cy = M.m01 / M.m00;

      blobs.push({ x: cx, y: cy, area });
      contour.delete();
    }

    thresholded.delete();
    hierarchy.delete();
    contours.delete();

    if (blobs.length < 4) return null;

    // Sort by area descending, take top 4
    blobs.sort((a, b) => b.area - a.area);
    const top4 = blobs.slice(0, 4);

    // Sort by position: top-left, top-right, bottom-right, bottom-left
    // (matching CIRCLE_MARKERS order)
    const cx = top4.reduce((s, b) => s + b.x, 0) / top4.length;
    const cy = top4.reduce((s, b) => s + b.y, 0) / top4.length;

    const quadrant = (b: { x: number; y: number }): number => {
      if (b.x <= cx && b.y <= cy) return 0; // top-left
      if (b.x > cx && b.y <= cy) return 1;  // top-right
      if (b.x > cx && b.y > cy) return 2;   // bottom-right
      return 3;                               // bottom-left
    };

    const sorted = [...top4].sort((a, b) => quadrant(a) - quadrant(b));

    return {
      cameraPoints: sorted.map((b) => ({ x: b.x, y: b.y })),
      gridPoints: [...CIRCLE_MARKERS],
      method: "circles",
    };
  } catch (e) {
    console.warn("[Calibration] Circle detection failed:", e);
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
    const srcMat = cv.matFromArray(srcPoints.length, 1, cv.CV_32FC2);
    const dstMat = cv.matFromArray(dstPoints.length, 1, cv.CV_32FC2);

    // Fill source points
    for (let i = 0; i < srcPoints.length; i++) {
      srcMat.data32F[i * 2] = srcPoints[i]!.x;
      srcMat.data32F[i * 2 + 1] = srcPoints[i]!.y;
    }
    // Fill destination points
    for (let i = 0; i < dstPoints.length; i++) {
      dstMat.data32F[i * 2] = dstPoints[i]!.x;
      dstMat.data32F[i * 2 + 1] = dstPoints[i]!.y;
    }

    const mask = new cv.Mat();
    const H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3, mask);

    // Extract 3×3 matrix as row-major array
    // H is CV_64F (double) → H.data64F
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
