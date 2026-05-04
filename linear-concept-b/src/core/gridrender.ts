// ─── src/core/gridRenderer.ts ─────────────────────────────────────────────────
// Manages the JSXGraph board that renders the coordinate grid, basis vectors,
// and transformed objects on top of the projector canvas.

/// <reference types="jsxgraph" />
import type { GridConfig, Matrix2x2, Point2D } from "../types/index.js";
import { applyTransform, lerpMatrix, identity } from "./linearTransform.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GridPoint extends Point2D {
  label?: string;
}

type JXGPoint = JXG.Point;
type JXGLine = JXG.Line;
type JXGArrow = JXG.Line;

// ---------------------------------------------------------------------------
// GridRenderer
// ---------------------------------------------------------------------------

export class GridRenderer {
  private board: JXG.Board;
  private config: GridConfig;

  /** The two basis vector arrows (e1, e2). */
  private basisArrows: [JXGArrow, JXGArrow] | null = null;

  /** Tracked points placed on the board by the user / detection. */
  private trackedPoints: Map<string, JXGPoint> = new Map();

  /** Whether a transition animation is in progress. */
  private animating = false;

  constructor(containerId: string, config: GridConfig) {
    this.config = config;
    const half = config.gridSize / 2;

    this.board = JXG.JSXGraph.initBoard(containerId, {
      boundingbox: [-half, half, half, -half],
      axis: true,
      grid: true,
      showCopyright: false,
      showNavigation: false,
      keepAspectRatio: true,
      pan: { enabled: false },
      zoom: { enabled: false },
    });

    this.drawBasisVectors(identity());
  }

  // -------------------------------------------------------------------------
  // Basis vectors
  // -------------------------------------------------------------------------

  /**
   * Draw (or update) the standard basis vectors e₁ and e₂ under the given
   * transformation matrix.
   */
  drawBasisVectors(matrix: Matrix2x2): void {
    const e1 = applyTransform(matrix, { x: 1, y: 0 });
    const e2 = applyTransform(matrix, { x: 0, y: 1 });

    if (this.basisArrows) {
      // Update existing arrows
      this.basisArrows[0].point2.moveTo([e1.x, e1.y]);
      this.basisArrows[1].point2.moveTo([e2.x, e2.y]);
      this.board.update();
      return;
    }

    const origin: [number, number] = [0, 0];
    const arrowStyle = (color: string) => ({
      strokeColor: color,
      strokeWidth: 3,
      firstArrow: false,
      lastArrow: { type: 1, size: 8 },
      fixed: false,
    });

    this.basisArrows = [
      this.board.create("arrow", [origin, [e1.x, e1.y]], {
        ...arrowStyle("#e63946"),
        name: "e₁",
        withLabel: this.config.showLabels,
        label: { offset: [5, -5], color: "#e63946" },
      }) as unknown as JXGArrow,

      this.board.create("arrow", [origin, [e2.x, e2.y]], {
        ...arrowStyle("#2a9d8f"),
        name: "e₂",
        withLabel: this.config.showLabels,
        label: { offset: [5, 5], color: "#2a9d8f" },
      }) as unknown as JXGArrow,
    ];
  }

  // -------------------------------------------------------------------------
  // Animated transformation
  // -------------------------------------------------------------------------

  /**
   * Animate the grid from one transformation matrix to another.
   * Interpolates the basis vectors over `transitionMs` milliseconds.
   */
  animateTo(
    fromMatrix: Matrix2x2,
    toMatrix: Matrix2x2,
    onComplete?: () => void,
  ): void {
    if (!this.config.animateTransition || this.animating) {
      this.drawBasisVectors(toMatrix);
      this.transformTrackedPoints(toMatrix);
      onComplete?.();
      return;
    }

    this.animating = true;
    const startTime = performance.now();
    const duration = this.config.transitionMs;

    const frame = (now: number): void => {
      const t = Math.min((now - startTime) / duration, 1);
      const interpolated = lerpMatrix(fromMatrix, toMatrix, easeInOut(t));

      this.drawBasisVectors(interpolated);
      this.transformTrackedPoints(interpolated);

      if (t < 1) {
        requestAnimationFrame(frame);
      } else {
        this.animating = false;
        onComplete?.();
      }
    };

    requestAnimationFrame(frame);
  }

  // -------------------------------------------------------------------------
  // Tracked / detected points
  // -------------------------------------------------------------------------

  /** Add or move a tracked point (e.g. from object detection). */
  upsertTrackedPoint(id: string, canvasPoint: Point2D, boardPoint: Point2D, color: string): void {
    if (this.trackedPoints.has(id)) {
      this.trackedPoints.get(id)!.moveTo([boardPoint.x, boardPoint.y]);
    } else {
      const pt = this.board.create("point", [boardPoint.x, boardPoint.y], {
        name: id,
        fixed: true,
        strokeColor: color,
        fillColor: color,
        size: 6,
        withLabel: false,
      }) as unknown as JXGPoint;
      this.trackedPoints.set(id, pt);
    }
    void canvasPoint; // reserved for homography mapping
    this.board.update();
  }

  /** Remove a tracked point. */
  removeTrackedPoint(id: string): void {
    const pt = this.trackedPoints.get(id);
    if (pt) {
      this.board.removeObject(pt);
      this.trackedPoints.delete(id);
    }
  }

  /** Apply transformation matrix to all tracked points. */
  private transformTrackedPoints(matrix: Matrix2x2): void {
    for (const [, pt] of this.trackedPoints) {
      const coords = pt.coords.usrCoords;
      const original: Point2D = { x: coords[1], y: coords[2] };
      const transformed = applyTransform(matrix, original);
      pt.moveTo([transformed.x, transformed.y]);
    }
    this.board.update();
  }

  // -------------------------------------------------------------------------
  // Misc helpers
  // -------------------------------------------------------------------------

  /** Highlight a specific grid point for the instruction panel. */
  highlightPoint(p: GridPoint, color = "#f4a261"): JXGPoint {
    return this.board.create("point", [p.x, p.y], {
      name: p.label ?? "",
      withLabel: !!p.label,
      strokeColor: color,
      fillColor: color,
      size: 8,
      fixed: true,
    }) as unknown as JXGPoint;
  }

  /** Draw a polygon (e.g. the shape of the physical object on the grid). */
  drawPolygon(vertices: Point2D[], color = "#457b9d"): void {
    const pts = vertices.map((v, i) =>
      this.board.create("point", [v.x, v.y], {
        name: `v${i}`,
        visible: false,
        fixed: true,
      }),
    );
    this.board.create("polygon", pts, {
      fillColor: color,
      fillOpacity: 0.25,
      strokeColor: color,
      strokeWidth: 2,
    });
  }

  /** Remove all user-added elements and reset to identity. */
  reset(): void {
    this.board.suspendUpdate();
    for (const [id, pt] of this.trackedPoints) {
      this.board.removeObject(pt);
      this.trackedPoints.delete(id);
    }
    if (this.basisArrows) {
      this.board.removeObject(this.basisArrows[0]);
      this.board.removeObject(this.basisArrows[1]);
      this.basisArrows = null;
    }
    this.board.unsuspendUpdate();
    this.drawBasisVectors(identity());
  }

  get jxgBoard(): JXG.Board {
    return this.board;
  }
}

// ---------------------------------------------------------------------------
// Easing
// ---------------------------------------------------------------------------

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}
