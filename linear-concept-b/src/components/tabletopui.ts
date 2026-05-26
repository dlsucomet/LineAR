// ─── src/components/tabletopUI.ts ────────────────────────────────────────────
// Renders the projected grid display: white background, blue grid,
// instruction text, confirmation buttons, corner markers, basis vectors,
// and 2×2 matrix label. No chrome/title bar — pure projection content.

import type { AppPhase, AppState } from "../core/appstatemachine.ts";
import {
  getInstructionText,
  showBasisVectors,
  showConfirmButtons,
  showCorners,
} from "../core/appstatemachine.ts";
import type { DetectedHand, Matrix2x2, Point2D } from "../types/index.ts";

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------
const C = {
  gridLine: "rgba(100, 160, 220, 0.55)",
  gridBg: "#eef4fb",
  instrBlue: "#1a3a6b",
  instrCyan: "#1a7a9a",
  instrOrange: "#c05000",
  btnYesBg: "#3a5a7a",
  btnNoBg: "#555",
  btnText: "#fff",
  e1: "#d53a3a",
  e2: "#3ad56b",
  corner: "#222",
  cornerFill: "#fff",
  matrixText: "#111",
  hl: "#d53a3a",
  hlBg: "rgba(213, 58, 58, 0.4)",
  hlStroke: "#d53a3a",
  hlActive: "#3ad56b",
  hlActiveBg: "rgba(58, 213, 107, 0.4)",
  cursor: "#3a7bd5",
  cursorBg: "rgba(58, 123, 213, 0.4)",
  cursorStroke: "#3a7bd5",
};

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const GRID_RANGE = 10;
const GRID_SNAP_RADIUS = 2.0;
const TEXT_STRIP_H = 40;

// ---------------------------------------------------------------------------
// TabletopUI
// ---------------------------------------------------------------------------

export class TabletopUI {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private W = 0;
  private H = 0;
  private lastGridRect: DOMRect | null = null;

  panX = 0;
  panY = 0;

  private animStartTime = 0;
  private readonly ANIM_DURATION = 10000;

  private matrix: Matrix2x2 = [1, 0, 0, 1];
  private appliedMatrix: Matrix2x2 = [1, 0, 0, 1];
  private corners: [Point2D, Point2D, Point2D, Point2D] | null = null;
  private detectionOutline: Point2D[] | null = null;
  private btnYes: DOMRect | null = null;
  private btnNo: DOMRect | null = null;
  private btnContinue: DOMRect | null = null;

  // Demo-specific state
  private fingerPos: Point2D | null = null;
  private dwellProgress = 0;
  private demoInstructions: Record<string, string> | null = null;

  // Drag state (phase 3: POINTS_CALCULATED) — all 4 corners snap to grid intersections
  private cornerDragPos: [Point2D | null, Point2D | null, Point2D | null, Point2D | null] = [null, null, null, null];
  private cornerSnapped: [boolean, boolean, boolean, boolean] = [false, false, false, false];
  private cornerLocked: [boolean, boolean, boolean, boolean] = [false, false, false, false];

  // [REMOVED: Done button prompt — corner adjustment phase removed]
  // showDoneButton = false;
  // btnDone: DOMRect | null = null;
  // onDone: (() => void) | null = null;

  // Arrow drag state (phase 5: SHOW_BASIS_VECTORS)
  private ghostArrows: { e1: Point2D; e2: Point2D } | null = null;
  private e1Snapped = false;
  private e2Snapped = false;
  private e1Locked = false;
  private e2Locked = false;
  private e1Dwelling = false;
  private e2Dwelling = false;

  // Mouse tracking for corner drag fallback
  private mouseDown = false;
  private mouseCanvasPos: Point2D | null = null;

  // Raw hand data for skeleton visualization
  private rawHands: DetectedHand[] | null = null;

  onYes: (() => void) | null = null;
  onNo: (() => void) | null = null;
  onContinue: (() => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not get 2D canvas context.");
    this.ctx = ctx;
    canvas.addEventListener("click", (e) => this.handleClick(e));
    canvas.addEventListener("mousedown", (e) => this.handleMouseDown(e));
    canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));
    canvas.addEventListener("mouseup", () => this.handleMouseUp());
    canvas.addEventListener("mouseleave", () => this.handleMouseUp());
  }

  resize(w: number, h: number): void {
    this.W = w; this.H = h;
    this.canvas.width = w;
    this.canvas.height = h;
  }

  setMatrix(m: Matrix2x2): void { this.matrix = m; }
  setAppliedMatrix(m: Matrix2x2): void { this.appliedMatrix = m; }
  setCorners(c: [Point2D, Point2D, Point2D, Point2D] | null): void { this.corners = c; }

  setDetectionOutline(outline: Point2D[] | null): void { this.detectionOutline = outline; }

  // Demo setters
  setFingerPosition(pos: Point2D | null): void { this.fingerPos = pos; }
  setDwellProgress(v: number): void { this.dwellProgress = v; }
  setDemoInstructions(overrides: Record<string, string> | null): void { this.demoInstructions = overrides; }

  setCornerDrag(index: number, pos: Point2D | null, snapped: boolean): void {
    this.cornerDragPos[index] = pos;
    this.cornerSnapped[index] = snapped;
  }

  setCornerLockedStates(states: [boolean, boolean, boolean, boolean]): void {
    this.cornerLocked = states;
  }

  getDragPos(index: number): Point2D | null { return this.cornerDragPos[index] as Point2D | null; }

  setGhostArrows(e1: Point2D, e2: Point2D): void { this.ghostArrows = { e1, e2 }; }
  setArrowSnapped(e1: boolean, e2: boolean): void { this.e1Snapped = e1; this.e2Snapped = e2; }
  setArrowLocked(e1: boolean, e2: boolean): void { this.e1Locked = e1; this.e2Locked = e2; }
  setDwellingIndicators(e1: boolean, e2: boolean): void { this.e1Dwelling = e1; this.e2Dwelling = e2; }
  setRawHands(hands: DetectedHand[] | null): void { this.rawHands = hands; }

  getButtonRects(): { yes: DOMRect | null; no: DOMRect | null; continue: DOMRect | null } {
    return {
      yes: this.btnYes,
      no: this.btnNo,
      continue: this.btnContinue,
    };
  }

  getPanOffset(): { x: number; y: number } {
    return { x: this.panX, y: this.panY };
  }

  setPanOffset(x: number, y: number): void {
    this.panX = x;
    this.panY = y;
  }

  autoFrameTransformed(corners: Point2D[], matrix: Matrix2x2, grid: DOMRect): void {
    if (!corners || corners.length < 4) return;
    const [a, b, c, d] = matrix;
    const pts = corners.map(p => ({
      x: a * p.x + b * p.y,
      y: c * p.x + d * p.y,
    }));
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const scale = Math.min(grid.width, grid.height) / 20;
    this.panX = -centerX * scale;
    this.panY = centerY * scale;
  }

  getLayoutProperties(): { gridX: number; gridY: number; gridWidth: number; gridHeight: number; scale: number } | null {
    if (this.W === 0 || this.H === 0) return null;
    const gridY = TEXT_STRIP_H;
    const gridHeight = this.H - TEXT_STRIP_H;
    const scale = Math.min(this.W, gridHeight) / (GRID_RANGE * 2);
    return {
      gridX: 0,
      gridY,
      gridWidth: this.W,
      gridHeight,
      scale,
    };
  }

  /**
   * Compute shared animation progress for the TRANSFORMED phase.
   * Animation plays once: triggers on first TRANSFORMED frame (animStartTime === 0),
   * runs for ANIM_DURATION ms, then sets animStartTime = -1 (done) so it never
   * retriggers. All other phases return animT = 1 (no interpolation).
   */
  private getAnimationProgress(phase: AppPhase): { animT: number; eased: number } {
    if (phase === "TRANSFORMED" && this.animStartTime === 0) {
      this.animStartTime = Date.now();
    }

    let animT = 1;
    if (phase === "TRANSFORMED" && this.animStartTime > 0) {
      animT = Math.min(1, (Date.now() - this.animStartTime) / this.ANIM_DURATION);
      if (animT >= 1) this.animStartTime = -1;
    }
    const eased = 1 - Math.pow(1 - animT, 3);
    return { animT, eased };
  }

  /** Main draw call — renders every frame. */
  draw(state: AppState): void {
    const { ctx, W, H } = this;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);

    const grid = this.computeGridRect();

    // Grid background
    ctx.fillStyle = C.gridBg;
    ctx.fillRect(grid.x, grid.y, grid.width, grid.height);

    // ── Pannable content (clipped to grid) ──
    ctx.save();
    ctx.beginPath();
    ctx.rect(grid.x, grid.y, grid.width, grid.height);
    ctx.clip();

    this.drawGrid(grid);
    this.drawDetectionOutline(grid, state.phase);
    const anim = this.getAnimationProgress(state.phase);
    this.drawVirtualObject(grid, state.phase, state.appliedMatrix, anim);

    if (state.phase === "POINTS_CALCULATED") {
      this.drawSnapPreview(grid);
      this.drawDragFeedback(grid);
    }
    if (state.phase === "SHOW_BASIS_VECTORS" && this.ghostArrows) {
      this.drawGhostArrows(grid, state.phase);
      this.drawArrowDragFeedback(grid);
    }
    if (showCorners(state.phase) && this.corners) this.drawCorners(grid, state.phase, state.appliedMatrix, anim);
    if (showBasisVectors(state.phase)) {
      const mat = state.phase === "TRANSFORMED" || state.phase === "CONFIRM_RESET"
        ? state.appliedMatrix
        : this.matrix;
      this.drawBasisArrows(grid, mat);
    }

    ctx.restore();

    // Grid border (fixed)
    ctx.strokeStyle = "rgba(100,160,220,0.8)";
    ctx.lineWidth = 1.2;
    ctx.strokeRect(grid.x, grid.y, grid.width, grid.height);

    // Fixed position elements
    this.drawInstruction(state.phase, grid);
    if (showConfirmButtons(state.phase)) this.drawButtons(grid, state.phase);
    if (state.phase === "TRANSFORMED") this.drawContinueButton(grid);
    // [REMOVED: Done button prompt during POINTS_CALCULATED]
    // if (state.phase === "POINTS_CALCULATED") {
    //   if (this.showDoneButton) this.drawDoneButton(grid);
    // }
    this.drawHandSkeleton();
    this.drawHandStatus();
    this.drawCursor();
    this.drawDwellIndicator();
  }

  private computeGridRect(): DOMRect {
    const { W, H } = this;
    this.lastGridRect = new DOMRect(0, TEXT_STRIP_H, W, H - TEXT_STRIP_H);
    return this.lastGridRect;
  }

  /** Get the most recently computed grid rect (for external coordinate conversions). */
  getLastGridRect(): DOMRect | null { return this.lastGridRect; }

  private drawGrid(r: DOMRect): void {
    const { ctx, W, H } = this;
    const cx = r.x + r.width * 0.5 + this.panX;
    const cy = r.y + r.height * 0.5 + this.panY;
    const scale = Math.min(r.width, r.height) / (GRID_RANGE * 2);

    // Vertical lines at every integer grid x that falls within the canvas
    const firstCol = Math.ceil((0 - cx) / scale);
    const lastCol = Math.floor((W - cx) / scale);
    ctx.strokeStyle = C.gridLine;
    ctx.lineWidth = 0.8;
    for (let i = firstCol; i <= lastCol; i++) {
      const x = cx + i * scale;
      ctx.beginPath();
      ctx.moveTo(x, r.y);
      ctx.lineTo(x, H);
      ctx.stroke();
    }

    // Horizontal lines at every integer grid y that falls within the canvas
    const firstRow = Math.ceil((cy - H) / scale);
    const lastRow = Math.floor((cy - r.y) / scale);
    for (let i = firstRow; i <= lastRow; i++) {
      const y = cy - i * scale;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }
  }

  private drawInstruction(phase: AppPhase, _grid: DOMRect): void {
    const { ctx, W } = this;
    const text = this.demoInstructions?.[phase] ?? getInstructionText(phase);
    if (!text) return;

    let color = C.instrBlue;
    if (phase === "OBJECT_DETECTED") color = C.instrOrange;
    if (phase === "TRANSFORMED") color = C.instrCyan;
    if (phase === "WAITING_FOR_OBJECT") color = C.instrBlue;
    if (phase === "POINTS_CALCULATED") color = C.instrCyan;
    if (phase === "CONFIRM_TRANSFORM") color = C.instrBlue;
    if (phase === "CONFIRM_RESET") color = C.instrBlue;

    ctx.fillStyle = color;
    ctx.font = "18px 'Courier New', monospace";

    // For phases with buttons, left-align text as part of a centered group (text + buttons)
    if (phase === "CONFIRM_TRANSFORM" || phase === "CONFIRM_RESET" || phase === "TRANSFORMED") {
      const buttonWidth = phase === "TRANSFORMED" ? 100 : 64 * 2 + 10;
      const tw = ctx.measureText(text).width;
      const gap = 10;
      const groupWidth = tw + gap + buttonWidth;
      const groupStartX = (W - groupWidth) / 2;
      ctx.textAlign = "left";
      ctx.fillText(text, groupStartX, TEXT_STRIP_H - 12);
    } else {
      ctx.textAlign = "center";
      ctx.fillText(text, W / 2, TEXT_STRIP_H - 12);
    }
    ctx.textAlign = "left";
  }

  private drawButtons(_grid: DOMRect, phase: AppPhase): void {
    const { ctx, W } = this;
    const bw = 64;
    const bh = 28;
    const gap = 10;

    ctx.font = "18px 'Courier New', monospace";
    const text = this.demoInstructions?.[phase] ?? getInstructionText(phase);
    const tw = ctx.measureText(text).width;
    const buttonGroupWidth = bw * 2 + gap;
    const groupWidth = tw + gap + buttonGroupWidth;
    const groupStartX = (W - groupWidth) / 2;

    const yesX = groupStartX + tw + gap;
    const noX = yesX + bw + gap;
    const btnY = 6;

    // Yes button (darker blue in frames)
    ctx.fillStyle = C.btnYesBg;
    ctx.beginPath();
    roundRect(ctx, yesX, btnY, bw, bh, 5);
    ctx.fill();
    ctx.fillStyle = C.btnText;
    ctx.font = "14px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Yes", yesX + bw / 2, btnY + 18);

    // No button
    ctx.fillStyle = C.btnNoBg;
    ctx.beginPath();
    roundRect(ctx, noX, btnY, bw, bh, 5);
    ctx.fill();
    ctx.fillStyle = C.btnText;
    ctx.fillText("No", noX + bw / 2, btnY + 18);

    ctx.textAlign = "left";

    // Store hit areas
    this.btnYes = new DOMRect(yesX, btnY, bw, bh);
    this.btnNo = new DOMRect(noX, btnY, bw, bh);
  }

  /**
   * Draw the four corner markers.
   * During TRANSFORMED / CONFIRM_RESET, corners are transformed by appliedMatrix.
   * During POINTS_CALCULATED, any dragged corner uses its temporary position.
   */
  private drawCorners(grid: DOMRect, phase: AppPhase, appliedMatrix: Matrix2x2, anim: { animT: number; eased: number }): void {
    const { ctx } = this;
    if (!this.corners) return;

    let drawCorners = this.corners;
    if (phase === "POINTS_CALCULATED") {
      drawCorners = [...this.corners];
      for (let i = 0; i < 4; i++) {
        if (this.cornerDragPos[i]) drawCorners[i] = this.cornerDragPos[i]!;
      }
    }

    const applyMat = (phase === "TRANSFORMED" || phase === "CONFIRM_RESET");
    const { animT, eased } = anim;

    const pts = drawCorners.map((c) => {
      if (!applyMat) return this.gridToCanvas(c, grid);

      const tx = appliedMatrix[0] * c.x + appliedMatrix[1] * c.y;
      const ty = appliedMatrix[2] * c.x + appliedMatrix[3] * c.y;

      if (animT < 1) {
        return this.gridToCanvas({
          x: c.x + (tx - c.x) * eased,
          y: c.y + (ty - c.y) * eased,
        }, grid);
      }
      return this.gridToCanvas({ x: tx, y: ty }, grid);
    });

    // Draw connecting rectangle lines
    ctx.strokeStyle = C.corner;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
    ctx.closePath();
    ctx.stroke();

    // Draw corner circles
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i]!;
      const isDragged = phase === "POINTS_CALCULATED" && this.cornerDragPos[i] !== null;
      const isLocked = phase === "POINTS_CALCULATED" && this.cornerLocked[i];
      ctx.beginPath();
      ctx.arc(p.x, p.y, isDragged ? 14 : 10, 0, Math.PI * 2);
      if (isDragged) {
        ctx.fillStyle = C.hlBg;
        ctx.strokeStyle = C.hlStroke;
        ctx.lineWidth = 3;
      } else if (isLocked) {
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = C.hl;
        ctx.lineWidth = 3;
      } else {
        ctx.fillStyle = C.cornerFill;
        ctx.strokeStyle = C.corner;
        ctx.lineWidth = 2;
      }
      ctx.fill();
      ctx.stroke();

      // Coordinate labels during transformed phases
      if (applyMat) {
        const orig = drawCorners[i]!;
        const tx = appliedMatrix[0] * orig.x + appliedMatrix[1] * orig.y;
        const ty = appliedMatrix[2] * orig.x + appliedMatrix[3] * orig.y;
        const fmt = (n: number) => Math.round(n * 10) / 10;
        ctx.font = "10px 'Courier New', monospace";
        ctx.fillStyle = `rgba(51, 51, 51, ${animT})`;
        ctx.textAlign = "left";
        ctx.fillText(`(${fmt(orig.x)},${fmt(orig.y)}) → (${fmt(tx)},${fmt(ty)})`, p.x + 14, p.y + 4);
      }
    }
  }

  /**
   * Draw basis vector arrows + matrix label at the grid centre.
   * Arrow length = ~15% of grid width per unit.
   */
  private drawBasisArrows(grid: DOMRect, mat: Matrix2x2): void {
    const { ctx } = this;
    const cx = grid.x + grid.width * 0.5 + this.panX;
    const cy = grid.y + grid.height * 0.5 + this.panY;
    const scale = Math.min(grid.width, grid.height) / 20;

    // e1 = first column of matrix → (mat[0], mat[2])
    const e1x = cx + mat[0] * scale;
    const e1y = cy - mat[2] * scale;   // flip Y

    // e2 = second column → (mat[1], mat[3])
    const e2x = cx + mat[1] * scale;
    const e2y = cy - mat[3] * scale;

    drawArrow(ctx, cx, cy, e1x, e1y, C.e1, 3);
    drawArrow(ctx, cx, cy, e2x, e2y, C.e2, 3);

    // Matrix label — positioned just to the right of arrow origin
    const lx = cx + 12;
    const ly = cy - 20;
    this.drawMatrixLabel(lx, ly, mat);
  }

  private drawMatrixLabel(x: number, y: number, mat: Matrix2x2): void {
    const { ctx } = this;
    ctx.font = "bold 15px 'Courier New', monospace";
    ctx.fillStyle = C.matrixText;

    const fmt = (n: number): string => {
      // Show as integer if clean, else 1 decimal
      const v = Math.round(n * 10) / 10;
      return Number.isInteger(v) ? String(v) : v.toFixed(1);
    };

    const a = fmt(mat[0]), b = fmt(mat[1]);
    const c = fmt(mat[2]), d = fmt(mat[3]);

    ctx.fillText(`[ ${a}  ${b} ]`, x, y);
    ctx.fillText(`[ ${c}  ${d} ]`, x, y + 20);
  }

  // ─── Detection Outline ─────────────────────────────────────────────────

  /** Draw the detected object outline (semi-transparent fill + solid border, black). */
  private drawDetectionOutline(grid: DOMRect, phase: AppPhase): void {
    if (!this.detectionOutline || this.detectionOutline.length < 3) return;
    if (phase !== "WAITING_FOR_OBJECT" && phase !== "OBJECT_DETECTED") return;

    const { ctx } = this;
    const pts = this.detectionOutline.map(p => this.gridToCanvas(p, grid));

    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
    ctx.closePath();

    ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
    ctx.fill();
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // ─── Demo rendering: Virtual Object ───────────────────────────────────────

  /** Draw the pre-placed virtual quadrilateral. Transformed during phases 7-8. */
  private drawVirtualObject(grid: DOMRect, phase: AppPhase, appliedMatrix: Matrix2x2, anim: { animT: number; eased: number }): void {
    if (!this.corners) return;
    if (
      phase !== "WAITING_FOR_OBJECT" &&
      phase !== "OBJECT_DETECTED" &&
      phase !== "POINTS_CALCULATED" &&
      phase !== "TRANSFORMED" &&
      phase !== "CONFIRM_RESET"
    ) return;

    // During corner-drag phase, substitute temporary positions for any dragged corner
    let drawCorners = this.corners;
    if (phase === "POINTS_CALCULATED") {
      drawCorners = [...this.corners];
      for (let i = 0; i < 4; i++) {
        if (this.cornerDragPos[i]) drawCorners[i] = this.cornerDragPos[i]!;
      }
    }

    const shouldTransform = phase === "TRANSFORMED" || phase === "CONFIRM_RESET";
    const [a, b, c, d] = shouldTransform ? appliedMatrix : [1, 0, 0, 1];
    const { animT, eased } = anim;

    const pts = drawCorners.map(p => {
      const tx = a * p.x + b * p.y;
      const ty = c * p.x + d * p.y;
      if (shouldTransform && animT < 1) {
        return this.gridToCanvas({
          x: p.x + (tx - p.x) * eased,
          y: p.y + (ty - p.y) * eased,
        }, grid);
      }
      return this.gridToCanvas({ x: tx, y: ty }, grid);
    });

    const { ctx } = this;
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
    ctx.closePath();
    ctx.fillStyle = "rgba(58, 123, 213, 0.35)";
    ctx.fill();
    ctx.strokeStyle = "rgba(58, 123, 213, 0.5)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // ─── Demo rendering: Continue Button ──────────────────────────────────────

  /** Draw "Continue →" button during TRANSFORMED phase. */
  private drawContinueButton(_grid: DOMRect): void {
    const ctx = this.ctx;
    ctx.font = "18px 'Courier New', monospace";
    const text = this.demoInstructions?.["TRANSFORMED"] ?? getInstructionText("TRANSFORMED");
    const tw = ctx.measureText(text).width;
    const gap = 10;
    const bw = 100;
    const bh = 28;
    const groupWidth = tw + gap + bw;
    const groupStartX = (this.W - groupWidth) / 2;
    const x = groupStartX + tw + gap;
    const y = 6;

    ctx.fillStyle = "#3a5a7a";
    ctx.beginPath();
    roundRect(ctx, x, y, bw, bh, 5);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.font = "14px Arial";
    ctx.textAlign = "center";
    ctx.fillText("Continue →", x + bw / 2, y + 18);
    ctx.textAlign = "left";

    this.btnContinue = new DOMRect(x, y, bw, bh);
  }

  /** Highlight dragged corners during POINTS_CALCULATED — show drag circle, filled dot when snapped. */
  private drawDragFeedback(grid: DOMRect): void {
    const ctx = this.ctx;

    for (let i = 0; i < 4; i++) {
      const pos = this.cornerDragPos[i];
      const snapped = this.cornerSnapped[i];
      if (!pos) continue;
      const p = this.gridToCanvas(pos, grid);

      if (snapped) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = C.hl;
        ctx.fill();
        continue;
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
      ctx.fillStyle = C.hlBg;
      ctx.fill();
      ctx.strokeStyle = C.hlStroke;
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  }

  // ─── Demo rendering: Snap Preview (Phase 3) ────────────────────────────────

  /** Pulsing highlight ring at the nearest grid intersection while dragging a corner. */
  private drawSnapPreview(grid: DOMRect): void {
    const ctx = this.ctx;
    const pulse = 0.4 + 0.3 * Math.sin(Date.now() * 0.005);

    for (let i = 0; i < 4; i++) {
      const pos = this.cornerDragPos[i];
      if (!pos) continue;
      const snap = this.nearestGridIntersection(pos);
      if (!snap) continue;
      const p = this.gridToCanvas(snap, grid);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 16, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(58, 123, 213, ${pulse})`;
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(58, 123, 213, 0.1)`;
      ctx.fill();
    }
  }

  // [REMOVED: drawDoneButton — corner adjustment prompt removed]
  // private drawDoneButton(_grid: DOMRect): void { ... }

  // ─── Demo rendering: Ghost Arrows (Phase 5) ───────────────────────────────

  /** Semi-transparent dashed ghost arrows at the preset matrix target positions. */
  private drawGhostArrows(grid: DOMRect, _phase: AppPhase): void {
    if (!this.ghostArrows) return;
    const ctx = this.ctx;
    const cx = grid.x + grid.width * 0.5 + this.panX;
    const cy = grid.y + grid.height * 0.5 + this.panY;
    const scale = Math.min(grid.width, grid.height) / 20;

    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.setLineDash([6, 4]);

    const e1x = cx + this.ghostArrows.e1.x * scale;
    const e1y = cy - this.ghostArrows.e1.y * scale;
    drawArrow(ctx, cx, cy, e1x, e1y, C.hl, 2);

    const e2x = cx + this.ghostArrows.e2.x * scale;
    const e2y = cy - this.ghostArrows.e2.y * scale;
    drawArrow(ctx, cx, cy, e2x, e2y, C.hlActive, 2);

    ctx.restore();
  }

  /** Draw snap/lock indicators on basis arrows. */
  private drawArrowDragFeedback(grid: DOMRect): void {
    const ctx = this.ctx;
    const cx = grid.x + grid.width * 0.5 + this.panX;
    const cy = grid.y + grid.height * 0.5 + this.panY;
    const scale = Math.min(grid.width, grid.height) / 20;

    const mat = this.matrix;

    const drawTip = (tx: number, ty: number, snapped: boolean, locked: boolean, snapColor: string, snapBg: string) => {
      if (locked) {
        ctx.beginPath();
        ctx.arc(tx, ty, 10, 0, Math.PI * 2);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 3;
        ctx.stroke();
      } else if (snapped) {
        ctx.beginPath();
        ctx.arc(tx, ty, 8, 0, Math.PI * 2);
        ctx.fillStyle = snapBg;
        ctx.fill();
        ctx.strokeStyle = snapColor;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    };

    drawTip(
      cx + mat[0] * scale, cy - mat[2] * scale,
      this.e1Snapped, this.e1Locked, C.hl, C.hlBg,
    );
    drawTip(
      cx + mat[1] * scale, cy - mat[3] * scale,
      this.e2Snapped, this.e2Locked, C.hlActive, C.hlActiveBg,
    );

    // Pulsing dwell ring
    const pulseAlpha = 0.3 + 0.4 * Math.sin(Date.now() * 0.008);
    if (this.e1Dwelling) {
      const tx = cx + mat[0] * scale, ty = cy - mat[2] * scale;
      ctx.beginPath();
      ctx.arc(tx, ty, 14, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(58, 123, 213, ${pulseAlpha})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (this.e2Dwelling) {
      const tx = cx + mat[1] * scale, ty = cy - mat[3] * scale;
      ctx.beginPath();
      ctx.arc(tx, ty, 14, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(91, 156, 245, ${pulseAlpha})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // ─── Demo rendering: Cursor & Dwell ───────────────────────────────────────

  /** Draw the finger cursor at the tracked fingertip position. */
  private drawCursor(): void {
    if (!this.fingerPos) return;
    const { x, y } = this.fingerPos;
    if (!isFinite(x) || !isFinite(y)) return;
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(x, y, 14, 0, Math.PI * 2);
    ctx.fillStyle = C.cursorBg;
    ctx.fill();
    ctx.strokeStyle = C.cursorStroke;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = C.cursor;
    ctx.fill();
  }

  /** Draw the dwell progress arc around the cursor when hovering a button. */
  private drawDwellIndicator(): void {
    if (!this.fingerPos || this.dwellProgress <= 0) return;
    const ctx = this.ctx;
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + Math.PI * 2 * this.dwellProgress;

    ctx.beginPath();
    ctx.arc(this.fingerPos.x, this.fingerPos.y, 14, startAngle, endAngle);
    ctx.strokeStyle = "#3a5a7a";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // ─── Demo rendering: Hand visualization ─────────────────────────────────────

  /** Draw hand skeleton (bones + landmarks) on the main canvas. */
  private drawHandSkeleton(): void {
    if (!this.rawHands?.length) return;
    const ctx = this.ctx;
    const fingerChains = [
      [0, 1, 2, 3, 4],
      [0, 5, 6, 7, 8],
      [0, 9, 10, 11, 12],
      [0, 13, 14, 15, 16],
      [0, 17, 18, 19, 20],
    ];

    for (const hand of this.rawHands) {
      const lms = hand.landmarks;
      if (lms.length < 21) continue;

      // Bones
      ctx.strokeStyle = "rgba(200, 200, 200, 0.5)";
      ctx.lineWidth = 1.5;
      for (const chain of fingerChains) {
        for (let i = 0; i < chain.length - 1; i++) {
          const a = lms[chain[i]!]!;
          const b = lms[chain[i + 1]!]!;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }

      // Landmark dots
      for (const lm of lms) {
        ctx.beginPath();
        ctx.arc(lm.x, lm.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(168, 218, 220, 0.7)";
        ctx.fill();
      }

      // Gesture label above wrist
      const wrist = lms[0]!;
      ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
      ctx.font = "11px monospace";
      ctx.textAlign = "left";
      ctx.fillText(`[${hand.gesture}]`, wrist.x + 10, wrist.y - 8);
      ctx.textAlign = "left";
    }
  }

  /** Draw a small hand status indicator in the top-right corner. */
  private drawHandStatus(): void {
    const ctx = this.ctx;
    const hasHands = this.rawHands !== null && this.rawHands.length > 0;
    ctx.font = "13px monospace";
    ctx.textAlign = "right";
    ctx.fillStyle = hasHands ? "rgba(80, 200, 80, 0.7)" : "rgba(150, 150, 150, 0.5)";
    ctx.fillText(hasHands ? "Hand: \u2713" : "Hand: \u2014", this.W - 12, 20);
    ctx.textAlign = "left";
  }

  // ─── Coordinate helpers ────────────────────────────────────────────────────

  /**
   * Map a point in grid space (–10..10) to canvas pixel coords.
   * The grid is centred at (0.5, 0.5) of the grid rect.
   */
  gridToCanvas(p: Point2D, grid: DOMRect): Point2D {
    const cx = grid.x + grid.width * 0.5 + this.panX;
    const cy = grid.y + grid.height * 0.5 + this.panY;
    const scale = Math.min(grid.width, grid.height) / 20;
    return {
      x: cx + p.x * scale,
      y: cy - p.y * scale,
    };
  }

  /** Inverse of gridToCanvas: map canvas pixel coords back to grid space (–10..10). */
  canvasToGrid(p: Point2D, grid: DOMRect): Point2D {
    const cx = grid.x + grid.width * 0.5 + this.panX;
    const cy = grid.y + grid.height * 0.5 + this.panY;
    const scale = Math.min(grid.width, grid.height) / 20;
    return {
      x: (p.x - cx) / scale,
      y: -(p.y - cy) / scale,
    };
  }

  /** Snap a grid-space point to the nearest integer grid intersection. */
  snapToGrid(p: Point2D): Point2D {
    return { x: Math.round(p.x), y: Math.round(p.y) };
  }

  /** Return the nearest grid intersection if within snap radius, or null. */
  nearestGridIntersection(p: Point2D): Point2D | null {
    const snapped = this.snapToGrid(p);
    const dx = p.x - snapped.x;
    const dy = p.y - snapped.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    return dist <= GRID_SNAP_RADIUS ? snapped : null;
  }

  // ─── Mouse tracking (corner drag fallback) ─────────────────────────────────

  isMouseDown(): boolean { return this.mouseDown; }
  getMouseCanvasPos(): Point2D | null { return this.mouseCanvasPos; }

  private handleMouseDown(e: MouseEvent): void {
    this.mouseDown = true;
    const rect = this.canvas.getBoundingClientRect();
    this.mouseCanvasPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  private handleMouseMove(e: MouseEvent): void {
    if (!this.mouseDown) return;
    const rect = this.canvas.getBoundingClientRect();
    this.mouseCanvasPos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  private handleMouseUp(): void {
    this.mouseDown = false;
  }

  // ─── Input handling ────────────────────────────────────────────────────────

  private handleClick(e: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (this.btnYes && hitTest(mx, my, this.btnYes)) {
      this.btnYes = null; this.btnNo = null;
      this.onYes?.();
    } else if (this.btnNo && hitTest(mx, my, this.btnNo)) {
      this.btnYes = null; this.btnNo = null;
      this.onNo?.();
    } else if (this.btnContinue && hitTest(mx, my, this.btnContinue)) {
      this.btnContinue = null;
      this.onContinue?.();
    }
    // [REMOVED: Done button hit test]
    // } else if (this.btnDone && hitTest(mx, my, this.btnDone)) {
    //   this.btnDone = null;
    //   this.onDone?.();
    // }
  }
}

// ---------------------------------------------------------------------------
// Canvas draw utilities
// ---------------------------------------------------------------------------

function drawArrow(
  ctx: CanvasRenderingContext2D,
  fromX: number, fromY: number,
  toX: number, toY: number,
  color: string,
  lineWidth: number,
): void {
  const headLen = 12;
  const angle = Math.atan2(toY - fromY, toX - fromX);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;

  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  // Arrow head
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(
    toX - headLen * Math.cos(angle - Math.PI / 6),
    toY - headLen * Math.sin(angle - Math.PI / 6),
  );
  ctx.lineTo(
    toX - headLen * Math.cos(angle + Math.PI / 6),
    toY - headLen * Math.sin(angle + Math.PI / 6),
  );
  ctx.closePath();
  ctx.fill();
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  w: number, h: number,
  r: number,
): void {
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function hitTest(mx: number, my: number, rect: DOMRect): boolean {
  return mx >= rect.x && mx <= rect.x + rect.width &&
    my >= rect.y && my <= rect.y + rect.height;
}
