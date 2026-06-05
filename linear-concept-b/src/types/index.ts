// ─── src/types/index.ts ───────────────────────────────────────────────────────
// Central type definitions for LineAR Concept B.
// Every domain object used across the app is typed here.

// ---------------------------------------------------------------------------
// 2-D geometry primitives
// ---------------------------------------------------------------------------

/** A point in 2-D space. */
export interface Point2D {
  x: number;
  y: number;
}

/** A 2×2 matrix represented as a flat row-major array [a, b, c, d].
 *  Corresponds to:
 *    | a  b |
 *    | c  d |
 */
export type Matrix2x2 = [number, number, number, number];

/** A 3×3 homogeneous matrix (row-major, 9 elements). */
export type Matrix3x3 = [
  number, number, number,
  number, number, number,
  number, number, number,
];

// ---------------------------------------------------------------------------
// Linear Transformation types
// ---------------------------------------------------------------------------

export type TransformationType =
  | "identity"
  | "scale"
  | "rotate"
  | "shear"
  | "reflect"
  | "custom";

// ---------------------------------------------------------------------------
// Object / Marker detection
// ---------------------------------------------------------------------------

/** A detected physical object on the tabletop. */
export interface DetectedObject {
  /** Unique identifier assigned by the tracker. */
  id: string;
  /** Bounding rectangle in camera pixel coordinates. */
  boundingBox: BoundingBox;
  /** Centre point in camera pixel coordinates. */
  center: Point2D;
  /** Detected colour label (used for colour-coding with p5.js). */
  colorLabel: string;
  /** Raw HSV dominant hue (0-360). */
  hue: number;
  /** Quadrilateral corner points of the detected shape (camera pixels). */
  shapeCorners?: Point2D[];
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// ---------------------------------------------------------------------------
// Hand tracking
// ---------------------------------------------------------------------------

/** One landmark from TensorFlow HandPose. */
export interface HandLandmark {
  x: number;
  y: number;
  z: number;
  name?: string;
}

export interface DetectedHand {
  handedness: "Left" | "Right";
  score: number;
  landmarks: HandLandmark[];
  /** Derived gesture classification. */
  gesture: HandGesture;
}

export type HandGesture =
  | "open"       // all fingers extended
  | "fist"       // all fingers closed
  | "point"      // index extended, rest closed
  | "pinch"      // thumb + index close together
  | "unknown";

// ---------------------------------------------------------------------------
// Camera / projection
// ---------------------------------------------------------------------------

export interface CameraConfig {
  /** Desired capture width (pixels). */
  width: number;
  /** Desired capture height (pixels). */
  height: number;
  /** Desired frame rate. */
  fps: number;
  /** deviceId from MediaDeviceInfo. */
  deviceId?: string;
}

/** Homography that maps camera pixels → projector pixels. */
export interface HomographyCalibration {
  /** 3×3 homography matrix (row-major). */
  matrix: Matrix3x3;
  /** Whether calibration has been performed. */
  calibrated: boolean;
  /** ISO timestamp of last calibration. */
  calibratedAt?: string;
}

// ---------------------------------------------------------------------------
// Grid / JSXGraph
// ---------------------------------------------------------------------------

export interface GridConfig {
  /** Number of grid cells along each axis (e.g. 10 → –5 to 5). */
  gridSize: number;
  /** Whether to show axis labels. */
  showLabels: boolean;
  /** Whether to animate transformation transitions. */
  animateTransition: boolean;
  /** Transition duration in milliseconds. */
  transitionMs: number;
}


