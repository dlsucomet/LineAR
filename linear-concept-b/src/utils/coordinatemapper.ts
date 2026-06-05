import type { Point2D } from "../types/index.ts";

export type CameraTransform =
  | { type: "direct" }
  | { type: "homography"; matrix: number[] };

export class CoordinateMapper {
  private transform: CameraTransform = { type: "direct" };
  private boardHalf: number;
  private camW: number;
  private camH: number;
  constructor(camW: number, camH: number, boardHalf: number = 10) {
    this.camW = camW;
    this.camH = camH;
    this.boardHalf = boardHalf;
  }

  setCameraResolution(w: number, h: number): void {
    this.camW = w;
    this.camH = h;
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
    const t = this.transform;
    if (t.type === "direct") {
      return {
        x: ((pixel.x / this.camW) * 2 - 1) * this.boardHalf,
        y: -((pixel.y / this.camH) * 2 - 1) * this.boardHalf,
      };
    }

    const H = t.matrix;
    const w = H[6]! * pixel.x + H[7]! * pixel.y + H[8]!;
    return {
      x: (H[0]! * pixel.x + H[1]! * pixel.y + H[2]!) / w,
      y: (H[3]! * pixel.x + H[4]! * pixel.y + H[5]!) / w,
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
