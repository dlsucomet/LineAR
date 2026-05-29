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
  locked: "#3a7bd5",
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
const PAN_PADDING = 4;

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
  private panBounds: { minX: number; maxX: number; minY: number; maxY: number } | null = null;

  private animStartTime = 0;
  private readonly ANIM_DURATION = 10000;

  private matrix: Matrix2x2 = [1, 0, 0, 1];
  private appliedMatrix: Matrix2x2 = [1, 0, 0, 1];
  private corners: [Point2D, Point2D, Point2D, Point2D] | null = null;
  private btnYes: DOMRect | null = null;
  private btnNo: DOMRect | null = null;
  private btnContinue: DOMRect | null = null;

  // Demo-specific state
  private fingerPos: Point2D | null = null;
  private cursorDwellProgress = 0;
  private demoInstructions: Record<string, string> | null = null;

  // Arrow drag state (SHOW_BASIS_VECTORS)
  private ghostArrows: { e1: Point2D; e2: Point2D } | null = null;
  private e1Snapped = false;
  private e2Snapped = false;
  private e1Locked = false;
  private e2Locked = false;
  private e1DwellProgress = 0;
  private e2DwellProgress = 0;

  // Mouse tracking for corner drag fallback
  private mouseDown = false;
  private mouseCanvasPos: Point2D | null = null;

  // Raw hand data for skeleton visualization
  private rawHands: DetectedHand[] | null = null;

  // Camera background
  private videoEl: HTMLVideoElement | null = null;
  private flipH = false;
  private flipV = false;

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

  // Demo setters
  setFingerPosition(pos: Point2D | null): void { this.fingerPos = pos; }
  setCursorDwellProgress(v: number): void { this.cursorDwellProgress = v; }
  setDemoInstructions(overrides: Record<string, string> | null): void { this.demoInstructions = overrides; }

  setGhostArrows(e1: Point2D, e2: Point2D): void { this.ghostArrows = { e1, e2 }; }
  setArrowSnapped(e1: boolean, e2: boolean): void { this.e1Snapped = e1; this.e2Snapped = e2; }
  setArrowLocked(e1: boolean, e2: boolean): void { this.e1Locked = e1; this.e2Locked = e2; }
  setDwellProgress(e1: number, e2: number): void { this.e1DwellProgress = e1; this.e2DwellProgress = e2; }
  setRawHands(hands: DetectedHand[] | null): void { this.rawHands = hands; }
  setVideoSource(video: HTMLVideoElement | null): void { this.videoEl = video; }
  setFlipFlags(h: boolean, v: boolean): void { this.flipH = h; this.flipV = v; }

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

  setPanBounds(bounds: { minX: number; maxX: number; minY: number; maxY: number } | null): void {
    this.panBounds = bounds;
  }

  setPanOffset(x: number, y: number): void {
    this.panX = x;
    this.panY = y;
    if (!this.panBounds) return;

    const gridH = this.H - TEXT_STRIP_H;
    const scale = Math.min(this.W, gridH) / (GRID_RANGE * 2);
    const halfW = (this.W / 2) / scale;
    const halfH = (gridH / 2) / scale;

    const pkMin = -(this.panBounds.maxX * scale - this.W / 2);
    const pkMax = -(this.panBounds.minX * scale + this.W / 2);
    this.panX = pkMin > pkMax
      ? (pkMin + pkMax) / 2
      : Math.max(pkMin, Math.min(pkMax, this.panX));

    const pyMin = this.panBounds.minY * scale + gridH / 2;
    const pyMax = this.panBounds.maxY * scale - gridH / 2;
    this.panY = pyMin > pyMax
      ? (pyMin + pyMax) / 2
      : Math.max(pyMin, Math.min(pyMax, this.panY));
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

    this.panBounds = {
      minX: minX - PAN_PADDING,
      maxX: maxX + PAN_PADDING,
      minY: minY - PAN_PADDING,
      maxY: maxY + PAN_PADDING,
    };
    this.setPanOffset(this.panX, this.panY);
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

  /** Draw the camera feed as the canvas background, with flipH/flipV applied. */
  private drawCameraBackground(): void {
    if (!this.videoEl) return;
    const { ctx, W, H } = this;
    const vw = this.videoEl.videoWidth;
    const vh = this.videoEl.videoHeight;
    if (!vw || !vh) return;

    const scale = Math.min(W / vw, H / vh); // letterbox (same as toMirroredCanvas)
    const dx = (W - vw * scale) / 2;
    const dy = (H - vh * scale) / 2;

    ctx.save();
    ctx.translate(W / 2, H / 2);
    ctx.scale(this.flipH ? -1 : 1, this.flipV ? -1 : 1);
    ctx.translate(-W / 2, -H / 2);
    ctx.drawImage(this.videoEl, dx, dy, vw * scale, vh * scale);
    ctx.restore();
  }

  /** Main draw call — renders every frame. */
  draw(state: AppState): void {
    const { ctx, W, H } = this;
    ctx.clearRect(0, 0, W, H);

    // Camera background (behind everything)
    this.drawCameraBackground();
    // Semi-transparent overlay so grid stays readable on top of the feed
    ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
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
    const anim = this.getAnimationProgress(state.phase);

    if (this.corners) {
      this.drawObjectHighlight(grid, state.phase);
      this.drawCorners(grid, state.phase, state.appliedMatrix, anim);
    }
    this.drawVirtualObject(grid, state.phase, state.appliedMatrix, anim);

    if (state.phase === "SHOW_BASIS_VECTORS" && this.ghostArrows) {
      this.drawBasisTargetCircles(grid);
      this.drawArrowDragFeedback(grid);
    }
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
    if (phase === "TRANSFORMED") color = C.instrCyan;
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
   */
  private drawCorners(grid: DOMRect, phase: AppPhase, appliedMatrix: Matrix2x2, anim: { animT: number; eased: number }): void {
    const { ctx } = this;
    if (!this.corners) return;

    const applyMat = (phase === "TRANSFORMED" || phase === "CONFIRM_RESET");
    const { animT, eased } = anim;

    const pts = this.corners.map((c) => {
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

    if (phase !== "SHOW_CORNERS") {
      // Draw connecting rectangle lines
      ctx.strokeStyle = C.corner;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pts[0]!.x, pts[0]!.y);
      for (let i = 1; i < 4; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
      ctx.closePath();
      ctx.stroke();
    }

    // Draw corner circles + coordinate labels
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i]!;

      if (phase !== "SHOW_CORNERS") {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = C.cornerFill;
        ctx.strokeStyle = C.corner;
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
      }

      // Coordinate labels during transformed phases
      if (applyMat) {
        const orig = this.corners[i]!;
        const tx = appliedMatrix[0] * orig.x + appliedMatrix[1] * orig.y;
        const ty = appliedMatrix[2] * orig.x + appliedMatrix[3] * orig.y;
        const fmt = (n: number) => Math.round(n * 10) / 10;
        ctx.font = "10px 'Courier New', monospace";
        ctx.fillStyle = `rgba(51, 51, 51, ${animT})`;
        const label = `(${fmt(orig.x)},${fmt(orig.y)}) → (${fmt(tx)},${fmt(ty)})`;
        if (i === 0 || i === 3) {
          ctx.textAlign = "right";
          ctx.fillText(label, p.x - 14, p.y + 4);
        } else {
          ctx.textAlign = "left";
          ctx.fillText(label, p.x + 14, p.y + 4);
        }
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

    drawArrow(ctx, cx, cy, e1x, e1y, C.e1, 3, this.e1Locked ? C.locked : undefined);
    drawArrow(ctx, cx, cy, e2x, e2y, C.e2, 3, this.e2Locked ? C.locked : undefined);

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

  private dwellColor(baseHex: string, progress: number): string {
    if (progress <= 0 || progress >= 1) return baseHex;
    const t = Math.sin(progress * Math.PI);
    const r = parseInt(baseHex.slice(1, 3), 16);
    const g = parseInt(baseHex.slice(3, 5), 16);
    const b = parseInt(baseHex.slice(5, 7), 16);
    const blend = (ch: number, w: number) => Math.round(ch + (255 - ch) * w);
    return `rgb(${blend(r, t)},${blend(g, t)},${blend(b, t)})`;
  }

  // ─── Demo rendering: Virtual Object ───────────────────────────────────────

  /** Draw the transformed filled object rect (after basis vectors confirmed). */
  private drawVirtualObject(grid: DOMRect, phase: AppPhase, appliedMatrix: Matrix2x2, anim: { animT: number; eased: number }): void {
    if (!this.corners) return;
    if (phase !== "TRANSFORMED" && phase !== "CONFIRM_RESET") return;

    const shouldTransform = true;
    const [a, b, c, d] = appliedMatrix;
    const { animT, eased } = anim;

    const pts = this.corners.map(p => {
      const tx = a * p.x + b * p.y;
      const ty = c * p.x + d * p.y;
      if (animT < 1) {
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

  /** Dashed highight rect for the 5×3 object placement zone (pre-transformation). */
  private drawObjectHighlight(grid: DOMRect, phase: AppPhase): void {
    if (phase === "TRANSFORMED" || phase === "CONFIRM_RESET") return;
    if (!this.corners) return;
    const { ctx } = this;

    const pts = this.corners.map(c => this.gridToCanvas(c, grid));
    ctx.save();
    ctx.setLineDash([8, 6]);
    ctx.strokeStyle = "rgba(58, 123, 213, 0.7)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
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

  // ─── Demo rendering: Basis Target Circles ──────────────────────────────────

  /** Draw encircled grid intersections at the preset arrow target positions. */
  private drawBasisTargetCircles(grid: DOMRect): void {
    if (!this.ghostArrows) return;
    const { ctx } = this;
    const cx = grid.x + grid.width * 0.5 + this.panX;
    const cy = grid.y + grid.height * 0.5 + this.panY;
    const scale = Math.min(grid.width, grid.height) / 20;

    const drawTarget = (gx: number, gy: number, color: string) => {
      const px = cx + gx * scale;
      const py = cy - gy * scale;
      const pulse = 0.5 + 0.3 * Math.sin(Date.now() * 0.004);

      ctx.beginPath();
      ctx.arc(px, py, 18, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.globalAlpha = pulse;
      ctx.stroke();
      ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.25;
      ctx.fill();
      ctx.globalAlpha = 1;
    };

    drawTarget(this.ghostArrows.e1.x, this.ghostArrows.e1.y, C.hl);
    drawTarget(this.ghostArrows.e2.x, this.ghostArrows.e2.y, C.hlActive);
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
        const lockPulse = 0.4 + 0.6 * Math.sin(Date.now() * 0.004);
        ctx.beginPath();
        ctx.arc(tx, ty, 14, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255, 255, 255, ${lockPulse})`;
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
    if (this.e1DwellProgress > 0) {
      const tx = cx + mat[0] * scale, ty = cy - mat[2] * scale;
      ctx.beginPath();
      ctx.arc(tx, ty, 14, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(58, 123, 213, ${pulseAlpha})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (this.e2DwellProgress > 0) {
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
    if (!this.fingerPos || this.cursorDwellProgress <= 0) return;
    const ctx = this.ctx;
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + Math.PI * 2 * this.cursorDwellProgress;

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
  outlineColor?: string,
): void {
  const headLen = 12;
  const angle = Math.atan2(toY - fromY, toX - fromX);

  if (outlineColor) {
    ctx.strokeStyle = outlineColor;
    ctx.fillStyle = outlineColor;
    ctx.lineWidth = lineWidth + 3;

    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.stroke();
  }

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;

  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
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
