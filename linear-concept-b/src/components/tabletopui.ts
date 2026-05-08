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
import type { Matrix2x2, Point2D } from "../types/index.ts";

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

  private matrix: Matrix2x2 = [1, 0, 0, 1];
  private appliedMatrix: Matrix2x2 = [1, 0, 0, 1];
  private corners: [Point2D, Point2D, Point2D, Point2D] | null = null;
  private btnYes: DOMRect | null = null;
  private btnNo: DOMRect | null = null;

  onYes: (() => void) | null = null;
  onNo: (() => void) | null = null;

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

  /** Main draw call — renders every frame. */
  draw(state: AppState): void {
    const { ctx, W, H } = this;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, W, H);

    const grid = this.computeGridRect();
    this.drawGrid(grid);
    this.drawInstruction(state.phase, grid);
    if (showConfirmButtons(state.phase)) this.drawButtons(grid, state.phase);
    if (showCorners(state.phase) && this.corners) this.drawCorners(grid);
    if (showBasisVectors(state.phase)) {
      const mat = state.phase === "TRANSFORMED" || state.phase === "CONFIRM_RESET"
        ? state.appliedMatrix
        : this.matrix;
      this.drawBasisArrows(grid, mat);
    }
  }

  private computeGridRect(): DOMRect {
    const { W, H } = this;
    const gx = INSET;
    const gy = INSET + INSTR_H;
    const gw = W - INSET * 2;
    const gh = H - INSET * 2 - INSTR_H;
    return new DOMRect(gx, gy, gw, gh);
  }

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
    const text = getInstructionText(phase);
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
   * Corners are stored in normalised grid units (–1..1 ish), so map to canvas.
   */
  private drawCorners(grid: DOMRect): void {
    const { ctx } = this;
    if (!this.corners) return;

    const pts = this.corners.map((c) => this.gridToCanvas(c, grid));

    // Draw connecting rectangle lines
    ctx.strokeStyle = C.corner;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0]!.x, pts[0]!.y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i]!.x, pts[i]!.y);
    ctx.closePath();
    ctx.stroke();

    // Draw corner circles
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.fillStyle = C.cornerFill;
      ctx.fill();
      ctx.strokeStyle = C.corner;
      ctx.lineWidth = 2;
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

  // ─── Coordinate helpers ────────────────────────────────────────────────────

  /**
   * Map a point in normalised grid space (–1..1) to canvas pixel coords.
   * The grid is centred at (0.5, 0.5) of the grid rect.
   */
  private gridToCanvas(p: Point2D, grid: DOMRect): Point2D {
    const cx = grid.x + grid.width * 0.5;
    const cy = grid.y + grid.height * 0.5;
    const scale = Math.min(grid.width, grid.height) * 0.22;
    return {
      x: cx + p.x * scale,
      y: cy - p.y * scale,   // flip Y
    };
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
