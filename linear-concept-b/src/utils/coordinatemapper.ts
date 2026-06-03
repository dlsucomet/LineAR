import type { Point2D } from "../types/index.ts";

export type CameraTransform =
  | { type: "direct" }
  | { type: "homography"; matrix: number[] };

export class CoordinateMapper {
  private transform: CameraTransform = { type: "direct" };
  private boardHalf: number;
  private camW: number;
  private camH: number;
  private flipH = false;
  private flipV = false;

  constructor(camW: number, camH: number, boardHalf: number = 10) {
    this.camW = camW;
    this.camH = camH;
    this.boardHalf = boardHalf;
  }

  setCameraResolution(w: number, h: number): void {
    this.camW = w;
    this.camH = h;
  }

  setFlips(h: boolean, v: boolean): void {
    this.flipH = h;
    this.flipV = v;
  }

  setTransform(t: CameraTransform): void {
    this.transform = t;
  }

  getTransform(): CameraTransform {
    return this.transform;
  }

  isCalibrated(): boolean {
    return this.transform.type === "homography";
  }

  /** Convert a camera-space pixel to grid coordinates. */
  cameraToGrid(pixel: Point2D): Point2D {
    const fx = this.flipH ? this.camW - pixel.x : pixel.x;
    const fy = this.flipV ? this.camH - pixel.y : pixel.y;
    const t = this.transform;
    if (t.type === "direct") {
      return {
        x: ((fx / this.camW) * 2 - 1) * this.boardHalf,
        y: -((fy / this.camH) * 2 - 1) * this.boardHalf,
      };
    }

    const H = t.matrix;
    const w = H[6]! * fx + H[7]! * fy + H[8]!;
    return {
      x: (H[0]! * fx + H[1]! * fy + H[2]!) / w,
      y: (H[3]! * fx + H[4]! * fy + H[5]!) / w,
    };
  }

  /** Convert a canvas pixel to grid coordinates (for UI interactions, not affected by camera calibration). */
  canvasToGrid(pixel: Point2D, gridCenterX: number, gridCenterY: number, scale: number): Point2D {
    return {
      x: (pixel.x - gridCenterX) / scale,
      y: -(pixel.y - gridCenterY) / scale,
    };
  }
}
