// ─── src/core/linearTransform.ts ─────────────────────────────────────────────
// Pure math helpers for 2-D linear transformations using ml-matrix.
// All functions are stateless and fully typed.

import { Matrix } from "ml-matrix";
import type { Matrix2x2, Point2D, TransformationType } from "../types/index.js";

// ---------------------------------------------------------------------------
// Matrix factory helpers
// ---------------------------------------------------------------------------

/** Wrap a flat 2×2 tuple into an ml-matrix Matrix. */
export function toMatrix(m: Matrix2x2): Matrix {
  return new Matrix([
    [m[0], m[1]],
    [m[2], m[3]],
  ]);
}

/** Unwrap an ml-matrix 2×2 Matrix into a flat tuple. */
export function fromMatrix(m: Matrix): Matrix2x2 {
  return [
    m.get(0, 0), m.get(0, 1),
    m.get(1, 0), m.get(1, 1),
  ];
}

/** Identity matrix. */
export function identity(): Matrix2x2 {
  return [1, 0, 0, 1];
}

// ---------------------------------------------------------------------------
// Canonical transformation matrices
// ---------------------------------------------------------------------------

/**
 * Uniform scale matrix.
 * @param sx - Scale factor along x-axis.
 * @param sy - Scale factor along y-axis (defaults to sx for uniform scale).
 */
export function scaleMatrix(sx: number, sy: number = sx): Matrix2x2 {
  return [sx, 0, 0, sy];
}

/**
 * Counter-clockwise rotation matrix.
 * @param angleDeg - Rotation angle in degrees.
 */
export function rotationMatrix(angleDeg: number): Matrix2x2 {
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return [cos, -sin, sin, cos];
}

/**
 * Shear matrix.
 * @param kx - Horizontal shear factor.
 * @param ky - Vertical shear factor.
 */
export function shearMatrix(kx: number, ky: number = 0): Matrix2x2 {
  return [1, kx, ky, 1];
}

/** Reflection across the x-axis. */
export function reflectXMatrix(): Matrix2x2 {
  return [1, 0, 0, -1];
}

/** Reflection across the y-axis. */
export function reflectYMatrix(): Matrix2x2 {
  return [-1, 0, 0, 1];
}

/** Reflection across the line y = x. */
export function reflectYeqXMatrix(): Matrix2x2 {
  return [0, 1, 1, 0];
}

// ---------------------------------------------------------------------------
// Core transform operations
// ---------------------------------------------------------------------------

/**
 * Apply a 2×2 transformation matrix to a single 2-D point.
 * Returns the transformed point.
 */
export function applyTransform(matrix: Matrix2x2, point: Point2D): Point2D {
  const [a, b, c, d] = matrix;
  return {
    x: a * point.x + b * point.y,
    y: c * point.x + d * point.y,
  };
}

/**
 * Apply a 2×2 transformation to an array of points.
 * Useful for transforming every vertex of a shape.
 */
export function applyTransformToPoints(
  matrix: Matrix2x2,
  points: Point2D[],
): Point2D[] {
  return points.map((p) => applyTransform(matrix, p));
}

/**
 * Compose two 2×2 transformation matrices (B ∘ A — applies A first).
 */
export function composeTransforms(a: Matrix2x2, b: Matrix2x2): Matrix2x2 {
  const ma = toMatrix(a);
  const mb = toMatrix(b);
  return fromMatrix(mb.mmul(ma));
}

/**
 * Compute the inverse of a 2×2 transformation matrix.
 * Throws if the matrix is singular (det = 0).
 */
export function invertTransform(m: Matrix2x2): Matrix2x2 {
  const [a, b, c, d] = m;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-10) {
    throw new Error("Matrix is singular – inverse does not exist.");
  }
  return [d / det, -b / det, -c / det, a / det];
}

/**
 * Compute the determinant of a 2×2 matrix.
 * The determinant tells us how area scales under the transformation:
 *   |det| > 1 → area expands
 *   |det| < 1 → area shrinks
 *   det < 0   → orientation flips
 */
export function determinant(m: Matrix2x2): number {
  return m[0] * m[3] - m[1] * m[2];
}

/**
 * Linearly interpolate between two 2×2 matrices.
 * Useful for animating a transformation step-by-step.
 * @param a - Start matrix.
 * @param b - End matrix.
 * @param t - Interpolation factor in [0, 1].
 */
export function lerpMatrix(a: Matrix2x2, b: Matrix2x2, t: number): Matrix2x2 {
  const clampedT = Math.max(0, Math.min(1, t));
  return [
    a[0] + (b[0] - a[0]) * clampedT,
    a[1] + (b[1] - a[1]) * clampedT,
    a[2] + (b[2] - a[2]) * clampedT,
    a[3] + (b[3] - a[3]) * clampedT,
  ];
}

// ---------------------------------------------------------------------------
// Eigen / geometric properties
// ---------------------------------------------------------------------------

/**
 * Compute eigenvalues of a 2×2 matrix.
 * Returns the two (possibly complex) eigenvalues as { real, imag } pairs.
 */
export interface Eigenvalue {
  real: number;
  imag: number;
}

export function eigenvalues(m: Matrix2x2): [Eigenvalue, Eigenvalue] {
  const [a, b, c, d] = m;
  const trace = a + d;
  const det = a * d - b * c;
  const discriminant = trace * trace - 4 * det;

  if (discriminant >= 0) {
    const sqrtDisc = Math.sqrt(discriminant);
    return [
      { real: (trace + sqrtDisc) / 2, imag: 0 },
      { real: (trace - sqrtDisc) / 2, imag: 0 },
    ];
  } else {
    const real = trace / 2;
    const imag = Math.sqrt(-discriminant) / 2;
    return [
      { real, imag },
      { real, imag: -imag },
    ];
  }
}

// ---------------------------------------------------------------------------
// Type-guard / label utilities
// ---------------------------------------------------------------------------

/**
 * Heuristically detect what kind of transformation a 2×2 matrix represents.
 */
export function classifyTransform(m: Matrix2x2): TransformationType {
  const [a, b, c, d] = m;
  const eps = 1e-6;

  // Identity
  if (
    Math.abs(a - 1) < eps &&
    Math.abs(b) < eps &&
    Math.abs(c) < eps &&
    Math.abs(d - 1) < eps
  ) {
    return "identity";
  }

  // Pure scale (diagonal)
  if (Math.abs(b) < eps && Math.abs(c) < eps) {
    return "scale";
  }

  // Rotation: a === d && b === -c && a²+b²≈1
  if (
    Math.abs(a - d) < eps &&
    Math.abs(b + c) < eps &&
    Math.abs(a * a + b * b - 1) < eps
  ) {
    return "rotate";
  }

  // Shear: diagonal entries are 1
  if (Math.abs(a - 1) < eps && Math.abs(d - 1) < eps) {
    return "shear";
  }

  // Reflection: det === -1
  if (Math.abs(determinant(m) + 1) < eps) {
    return "reflect";
  }

  return "custom";
}
