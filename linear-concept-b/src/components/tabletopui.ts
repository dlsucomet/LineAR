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
  e1: "#e63030",
  e2: "#22aa22",
  corner: "#222",
  cornerFill: "#fff",
  matrixText: "#111",
};

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const INSET = 28;
const INSTR_H = 52;

// ---------------------------------------------------------------------------
// TabletopUI
// ---------------------------------------------------------------------------

export class TabletopUI {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private W = 0;
  private H = 0;
  private lastGridRect: DOMRect | null = null;

  private matrix: Matrix2x2 = [1, 0, 0, 1];
  private appliedMatrix: Matrix2x2 = [1, 0, 0, 1];
  private corners: [Point2D, Point2D, Point2D, Point2D] | null = null;
  private btnYes: DOMRect | null = null;
  private btnNo: DOMRect | null = null;
  private btnStart: DOMRect | null = null;
  private btnContinue: DOMRect | null = null;

  // Demo-specific state
  private fingerPos: Point2D | null = null;
  private dwellProgress = 0;
  private demoInstructions: Record<string, string> | null = null;

  // Drag state (phase 3: POINTS_CALCULATED)
  private draggingCorner = false;
  private draggedCornerPos: Point2D | null = null;
  private targetPos: Point2D | null = null;
  private cornerInTarget = false;

  // Arrow drag state (phase 5: SHOW_BASIS_VECTORS)
  private ghostArrows: { e1: Point2D; e2: Point2D } | null = null;
  private e1Snapped = false;
  private e2Snapped = false;

  // Raw hand data for skeleton visualization
  private rawHands: DetectedHand[] | null = null;

  onYes: (() => void) | null = null;
  onNo: (() => void) | null = null;
  onStart: (() => void) | null = null;
  onContinue: (() => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not get 2D canvas context.");
    this.ctx = ctx;
    canvas.addEventListener("click", (e) => this.handleClick(e));
  }

  resize(w: number, h: number): void {
    this.W = w; this.H = h;
    this.canvas.width = w;
    this.canvas.height = h;
  }

  setMatrix(m: Matrix2x2): void { this.matrix = m; }
  setAppliedMatrix(m: Matrix2x2): void { this.appliedMatrix = m; }
  setCorners(c: [Point2D, Point2D, Point2D, Point2D] | null): void { this.corners = c; }

  // Demo setters
  setFingerPosition(pos: Point2D | null): void { this.fingerPos = pos; }
  setDwellProgress(v: number): void { this.dwellProgress = v; }
  setDemoInstructions(overrides: Record<string, string> | null): void { this.demoInstructions = overrides; }

  setDragTarget(pos: Point2D | null): void { this.targetPos = pos; }
  setDraggedCorner(pos: Point2D | null, dragging: boolean): void {
    this.draggedCornerPos = pos;
    this.draggingCorner = dragging;
  }
  setCornerInTarget(v: boolean): void { this.cornerInTarget = v; }

  setGhostArrows(e1: Point2D, e2: Point2D): void { this.ghostArrows = { e1, e2 }; }
  setArrowSnapped(e1: boolean, e2: boolean): void { this.e1Snapped = e1; this.e2Snapped = e2; }
  setRawHands(hands: DetectedHand[] | null): void { this.rawHands = hands; }

  getButtonRects(): { start: DOMRect | null; yes: DOMRect | null; no: DOMRect | null; continue: DOMRect | null } {
    return {
      start: this.btnStart,
      yes: this.btnYes,
      no: this.btnNo,
      continue: this.btnContinue,
    };
  }

  /** Main draw call — renders every frame. */
  draw(state: AppState): void {
    const { ctx, W, H } = this;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);

    const grid = this.computeGridRect();
    this.drawGrid(grid);

    // Virtual object (pre-placed quadrilateral)
    this.drawVirtualObject(grid, state.phase, state.appliedMatrix);

    this.drawInstruction(state.phase, grid);

    // Start button during WAITING_FOR_OBJECT
    if (state.phase === "WAITING_FOR_OBJECT") this.drawStartButton(grid);

    // Yes/No buttons during confirm phases
    if (showConfirmButtons(state.phase)) this.drawButtons(grid, state.phase);

    // Continue button during TRANSFORMED
    if (state.phase === "TRANSFORMED") this.drawContinueButton(grid);

    // Drag target highlight during POINTS_CALCULATED
    if (state.phase === "POINTS_CALCULATED") {
      this.drawDragTarget(grid);
      if (this.draggingCorner) this.drawDragFeedback(grid);
    }

    // Ghost arrows + drag feedback during SHOW_BASIS_VECTORS
    if (state.phase === "SHOW_BASIS_VECTORS" && this.ghostArrows) {
      this.drawGhostArrows(grid, state.phase);
      this.drawArrowDragFeedback(grid);
    }

    if (showCorners(state.phase) && this.corners) this.drawCorners(grid, state.phase, state.appliedMatrix);
    if (showBasisVectors(state.phase)) {
      const mat = state.phase === "TRANSFORMED" || state.phase === "CONFIRM_RESET"
        ? state.appliedMatrix
        : this.matrix;
      this.drawBasisArrows(grid, mat);
    }

    // Hand skeleton visualization (debug overlay)
    this.drawHandSkeleton();
    this.drawHandStatus();

    // Finger cursor (always on top)
    this.drawCursor();
    this.drawDwellIndicator();
  }

  private computeGridRect(): DOMRect {
    const { W, H } = this;
    const gx = INSET;
    const gy = INSET + INSTR_H;
    const gw = W - INSET * 2;
    const gh = H - INSET * 2 - INSTR_H;
    this.lastGridRect = new DOMRect(gx, gy, gw, gh);
    return this.lastGridRect;
  }

  /** Get the most recently computed grid rect (for external coordinate conversions). */
  getLastGridRect(): DOMRect | null { return this.lastGridRect; }

  private drawGrid(r: DOMRect): void {
    const { ctx } = this;
    // Grid background
    ctx.fillStyle = C.gridBg;
    ctx.fillRect(r.x, r.y, r.width, r.height);

    const cols = 40;
    const rows = 28;
    const cellW = r.width / cols;
    const cellH = r.height / rows;

    ctx.strokeStyle = C.gridLine;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (let c = 0; c <= cols; c++) {
      const x = r.x + c * cellW;
      ctx.moveTo(x, r.y);
      ctx.lineTo(x, r.y + r.height);
    }
    for (let rr = 0; rr <= rows; rr++) {
      const y = r.y + rr * cellH;
      ctx.moveTo(r.x, y);
      ctx.lineTo(r.x + r.width, y);
    }
    ctx.stroke();

    // Thin border around grid
    ctx.strokeStyle = "rgba(100,160,220,0.8)";
    ctx.lineWidth = 1.2;
    ctx.strokeRect(r.x, r.y, r.width, r.height);
  }

  private drawInstruction(phase: AppPhase, grid: DOMRect): void {
    const { ctx, W } = this;
    const override = this.demoInstructions?.[phase];
    const text = override ?? getInstructionText(phase);
    if (!text) return;

    const iy = grid.y - 18;   // centred in the gap above the grid

    // Pick text colour from frames:
    let color = C.instrBlue;
    if (phase === "OBJECT_DETECTED") color = C.instrOrange;
    if (phase === "TRANSFORMED") color = C.instrCyan;
    if (phase === "WAITING_FOR_OBJECT") color = C.instrBlue;
    if (phase === "POINTS_CALCULATED") color = C.instrCyan;
    if (phase === "CONFIRM_TRANSFORM") color = C.instrBlue;
    if (phase === "CONFIRM_RESET") color = C.instrBlue;

    ctx.fillStyle = color;
    ctx.font = "18px 'Courier New', monospace";
    ctx.textAlign = "center";
    ctx.fillText(text, W / 2, iy);
    ctx.textAlign = "left";
  }

  private drawButtons(grid: DOMRect, phase: AppPhase): void {
    const { ctx, W } = this;
    const iy = grid.y - 26;
    const bw = 64;
    const bh = 28;
    const gap = 10;

    // Calculate text width to position buttons after it
    ctx.font = "18px 'Courier New', monospace";
    const text = getInstructionText(phase);
    const tw = ctx.measureText(text).width;
    const startX = W / 2 - tw / 2 + tw + gap + 4;

    const yesX = startX;
    const noX = yesX + bw + gap;
    const btnY = iy - bh + 4;

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
   */
  private drawCorners(grid: DOMRect, phase: AppPhase, appliedMatrix: Matrix2x2): void {
    const { ctx } = this;
    if (!this.corners) return;

    const applyMat = (phase === "TRANSFORMED" || phase === "CONFIRM_RESET");
    const pts = this.corners.map((c) => {
      const p = applyMat
        ? { x: appliedMatrix[0] * c.x + appliedMatrix[1] * c.y, y: appliedMatrix[2] * c.x + appliedMatrix[3] * c.y }
        : c;
      return this.gridToCanvas(p, grid);
    });

    // Apply dragged corner position (phase 3 override)
    if (phase === "POINTS_CALCULATED" && this.draggedCornerPos && this.corners) {
      const draggedCanvas = this.gridToCanvas(this.draggedCornerPos, grid);
      pts[0] = draggedCanvas;
    }

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
      const isDraggedCorner = i === 0 && phase === "POINTS_CALCULATED" && this.draggedCornerPos !== null;
      ctx.beginPath();
      ctx.arc(p.x, p.y, isDraggedCorner ? 14 : 10, 0, Math.PI * 2);
      ctx.fillStyle = isDraggedCorner ? "rgba(230, 120, 30, 0.4)" : C.cornerFill;
      ctx.fill();
      ctx.strokeStyle = isDraggedCorner ? "#e6781e" : C.corner;
      ctx.lineWidth = isDraggedCorner ? 3 : 2;
      ctx.stroke();
    }
  }

  /**
   * Draw basis vector arrows + matrix label at the grid centre.
   * Arrow length = ~15% of grid width per unit.
   */
  private drawBasisArrows(grid: DOMRect, mat: Matrix2x2): void {
    const { ctx } = this;
    const cx = grid.x + grid.width * 0.45;
    const cy = grid.y + grid.height * 0.52;
    const scale = Math.min(grid.width, grid.height) * 0.13;

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

  // ─── Demo rendering: Virtual Object ───────────────────────────────────────

  /** Draw the pre-placed virtual quadrilateral. Transformed during phases 7-8. */
  private drawVirtualObject(grid: DOMRect, phase: AppPhase, appliedMatrix: Matrix2x2): void {
    if (!this.corners) return;
    if (
      phase !== "WAITING_FOR_OBJECT" &&
      phase !== "OBJECT_DETECTED" &&
      phase !== "POINTS_CALCULATED" &&
      phase !== "TRANSFORMED" &&
      phase !== "CONFIRM_RESET"
    ) return;

    const shouldTransform = phase === "TRANSFORMED" || phase === "CONFIRM_RESET";
    const [a, b, c, d] = shouldTransform ? appliedMatrix : [1, 0, 0, 1];

    const pts = this.corners.map(p => {
      const tx = a * p.x + b * p.y;
      const ty = c * p.x + d * p.y;
      return this.gridToCanvas({ x: tx, y: ty }, grid);
    });

    const { ctx } = this;
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
    ctx.closePath();
    ctx.fillStyle = "rgba(230, 49, 70, 0.35)";
    ctx.fill();
    ctx.strokeStyle = "rgba(230, 49, 70, 0.5)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // ─── Demo rendering: Start Button ─────────────────────────────────────────

  /** Draw "▶ Start Demo" button centred in the grid during WAITING_FOR_OBJECT. */
  private drawStartButton(grid: DOMRect): void {
    const ctx = this.ctx;
    const bw = 180;
    const bh = 48;
    const bx = (this.W - bw) / 2;
    const by = grid.y + grid.height / 2 - bh / 2;

    ctx.fillStyle = "#3a5a7a";
    ctx.beginPath();
    roundRect(ctx, bx, by, bw, bh, 8);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.font = "bold 20px Arial";
    ctx.textAlign = "center";
    ctx.fillText("▶  Start Demo", bx + bw / 2, by + 32);
    ctx.textAlign = "left";

    this.btnStart = new DOMRect(bx, by, bw, bh);
  }

  // ─── Demo rendering: Continue Button ──────────────────────────────────────

  /** Draw "Continue →" button during TRANSFORMED phase. */
  private drawContinueButton(grid: DOMRect): void {
    const ctx = this.ctx;
    const text = "Object has been linearly transformed";
    ctx.font = "18px 'Courier New', monospace";
    const tw = ctx.measureText(text).width;
    const gap = 10;
    const bw = 100;
    const bh = 28;
    const x = this.W / 2 - tw / 2 + tw + gap + 4;
    const y = grid.y - 26 - bh + 4;

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

  // ─── Demo rendering: Drag Target (Phase 3) ────────────────────────────────

  /** Pulsing target circle at the drag destination during POINTS_CALCULATED. */
  private drawDragTarget(grid: DOMRect): void {
    if (!this.targetPos) return;
    const ctx = this.ctx;
    const p = this.gridToCanvas(this.targetPos, grid);
    const pulse = 0.4 + 0.3 * Math.sin(Date.now() * 0.004);
    const color = this.cornerInTarget
      ? `rgba(50, 200, 50, ${0.5 + 0.3 * Math.sin(Date.now() * 0.005)})`
      : `rgba(230, 120, 30, ${pulse})`;

    ctx.beginPath();
    ctx.arc(p.x, p.y, 20, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();

    if (this.cornerInTarget) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 18, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(50, 200, 50, 0.15)";
      ctx.fill();
    }
  }

  /** Highlight the dragged corner during POINTS_CALCULATED drag. */
  private drawDragFeedback(grid: DOMRect): void {
    if (!this.draggedCornerPos) return;
    const ctx = this.ctx;
    const p = this.gridToCanvas(this.draggedCornerPos, grid);

    ctx.beginPath();
    ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = this.cornerInTarget ? "rgba(50, 200, 50, 0.4)" : "rgba(230, 120, 30, 0.4)";
    ctx.fill();
    ctx.strokeStyle = this.cornerInTarget ? "#32c832" : "#e6781e";
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  // ─── Demo rendering: Ghost Arrows (Phase 5) ───────────────────────────────

  /** Semi-transparent dashed ghost arrows at the preset matrix target positions. */
  private drawGhostArrows(grid: DOMRect, _phase: AppPhase): void {
    if (!this.ghostArrows) return;
    const ctx = this.ctx;
    const cx = grid.x + grid.width * 0.45;
    const cy = grid.y + grid.height * 0.52;
    const scale = Math.min(grid.width, grid.height) * 0.13;

    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.setLineDash([6, 4]);

    const e1x = cx + this.ghostArrows.e1.x * scale;
    const e1y = cy - this.ghostArrows.e1.y * scale;
    drawArrow(ctx, cx, cy, e1x, e1y, "#e63030", 2);

    const e2x = cx + this.ghostArrows.e2.x * scale;
    const e2y = cy - this.ghostArrows.e2.y * scale;
    drawArrow(ctx, cx, cy, e2x, e2y, "#22aa22", 2);

    ctx.restore();
  }

  /** Draw snap indicators on arrows that have been aligned to ghost targets. */
  private drawArrowDragFeedback(grid: DOMRect): void {
    const ctx = this.ctx;
    const cx = grid.x + grid.width * 0.45;
    const cy = grid.y + grid.height * 0.52;
    const scale = Math.min(grid.width, grid.height) * 0.13;

    const mat = this.matrix;
    if (this.e1Snapped) {
      const tx = cx + mat[0] * scale;
      const ty = cy - mat[2] * scale;
      ctx.beginPath();
      ctx.arc(tx, ty, 8, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(50, 200, 50, 0.5)";
      ctx.fill();
      ctx.strokeStyle = "#32c832";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (this.e2Snapped) {
      const tx = cx + mat[1] * scale;
      const ty = cy - mat[3] * scale;
      ctx.beginPath();
      ctx.arc(tx, ty, 8, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(50, 200, 50, 0.5)";
      ctx.fill();
      ctx.strokeStyle = "#32c832";
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
    ctx.fillStyle = "rgba(60, 180, 255, 0.5)";
    ctx.fill();
    ctx.strokeStyle = "rgba(60, 180, 255, 0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(60, 180, 255, 0.9)";
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
    const cx = grid.x + grid.width * 0.5;
    const cy = grid.y + grid.height * 0.5;
    const scale = Math.min(grid.width, grid.height) / 20;
    return {
      x: cx + p.x * scale,
      y: cy - p.y * scale,
    };
  }

  /** Inverse of gridToCanvas: map canvas pixel coords back to grid space (–10..10). */
  canvasToGrid(p: Point2D, grid: DOMRect): Point2D {
    const cx = grid.x + grid.width * 0.5;
    const cy = grid.y + grid.height * 0.5;
    const scale = Math.min(grid.width, grid.height) / 20;
    return {
      x: (p.x - cx) / scale,
      y: -(p.y - cy) / scale,
    };
  }

  // ─── Input handling ────────────────────────────────────────────────────────

  private handleClick(e: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (this.btnStart && hitTest(mx, my, this.btnStart)) {
      this.btnStart = null;
      this.onStart?.();
    } else if (this.btnYes && hitTest(mx, my, this.btnYes)) {
      this.btnYes = null; this.btnNo = null;
      this.onYes?.();
    } else if (this.btnNo && hitTest(mx, my, this.btnNo)) {
      this.btnYes = null; this.btnNo = null;
      this.onNo?.();
    } else if (this.btnContinue && hitTest(mx, my, this.btnContinue)) {
      this.btnContinue = null;
      this.onContinue?.();
    }
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
