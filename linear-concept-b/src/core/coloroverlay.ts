// ─── src/core/colorOverlay.ts ─────────────────────────────────────────────────
// Uses p5.js in instance mode to draw colour-coded overlays on detected objects
// and decorative visual feedback on the projector canvas.

import p5 from "p5";
import type { DetectedObject, DetectedHand, Point2D } from "../types/index.ts";

// ---------------------------------------------------------------------------
// Colour map (label → p5 colour string)
// ---------------------------------------------------------------------------

const COLOR_MAP: Record<string, string> = {
  red: "#e63946",
  green: "#2a9d8f",
  blue: "#457b9d",
  yellow: "#e9c46a",
  default: "#f4a261",
};

function colorFor(label: string): string {
  return COLOR_MAP[label] ?? COLOR_MAP["default"]!;
}

// ---------------------------------------------------------------------------
// ColorOverlay
// ---------------------------------------------------------------------------

export class ColorOverlay {
  private p5Instance: p5 | null = null;

  /** Shared state pushed from the app every frame. */
  private objects: DetectedObject[] = [];
  private hands: DetectedHand[] = [];
  private width = 0;
  private height = 0;

  /** Whether to show hand skeleton overlay. */
  showHandSkeleton = true;

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  /**
   * Mount the p5 sketch onto the given DOM container.
   * The canvas is sized to match `width` × `height`.
   */
  mount(containerId: string, width: number, height: number): void {
    this.width = width;
    this.height = height;

    const sketch = (p: p5): void => {
      p.setup = (): void => {
        p.createCanvas(width, height).parent(containerId);
        p.frameRate(30);
        p.noLoop();
      };

      p.draw = (): void => {
        p.clear();
        this.drawObjects(p);
        if (this.showHandSkeleton) this.drawHands(p);
      };
    };

    this.p5Instance = new p5(sketch);
  }

  /** Unmount and destroy the p5 instance. */
  unmount(): void {
    this.p5Instance?.remove();
    this.p5Instance = null;
  }

  // -------------------------------------------------------------------------
  // Per-frame update
  // -------------------------------------------------------------------------

  /** Push new frame data and trigger a redraw. */
  update(objects: DetectedObject[], hands: DetectedHand[]): void {
    this.objects = objects;
    this.hands = hands;
    this.p5Instance?.redraw();
  }

  // -------------------------------------------------------------------------
  // Drawing routines
  // -------------------------------------------------------------------------

  private drawObjects(p: p5): void {
    for (const obj of this.objects) {
      const col = colorFor(obj.colorLabel);
      p.stroke(col);
      p.strokeWeight(3);
      p.noFill();

      // Bounding rectangle
      p.rect(obj.boundingBox.x, obj.boundingBox.y, obj.boundingBox.width, obj.boundingBox.height, 8);

      // Centre cross
      p.fill(col);
      p.noStroke();
      p.ellipse(obj.center.x, obj.center.y, 10, 10);

      // Colour label text
      p.fill(col);
      p.noStroke();
      p.textSize(14);
      p.textFont("monospace");
      p.text(obj.colorLabel, obj.boundingBox.x + 6, obj.boundingBox.y - 6);
    }
  }

  private drawHands(p: p5): void {
    for (const hand of this.hands) {
      const lms = hand.landmarks;
      if (lms.length < 21) continue;

      const handColor = hand.handedness === "Left" ? "#e9c46a" : "#a8dadc";
      p.stroke(handColor);
      p.strokeWeight(2);

      // Draw finger bones
      const fingerChains = [
        [0, 1, 2, 3, 4],       // thumb
        [0, 5, 6, 7, 8],       // index
        [0, 9, 10, 11, 12],    // middle
        [0, 13, 14, 15, 16],   // ring
        [0, 17, 18, 19, 20],   // pinky
      ];

      for (const chain of fingerChains) {
        for (let i = 0; i < chain.length - 1; i++) {
          const a = lms[chain[i]!]!;
          const b = lms[chain[i + 1]!]!;
          p.line(a.x, a.y, b.x, b.y);
        }
      }

      // Draw landmarks
      p.fill(handColor);
      p.noStroke();
      for (const lm of lms) {
        p.ellipse(lm.x, lm.y, 6, 6);
      }

      // Gesture label above wrist
      const wrist = lms[0]!;
      p.fill(255);
      p.noStroke();
      p.textSize(12);
      p.textFont("monospace");
      p.text(`[${hand.gesture}]`, wrist.x + 10, wrist.y - 10);
    }
  }

  // -------------------------------------------------------------------------
  // Visual feedback helpers (called from instruction panel)
  // -------------------------------------------------------------------------

  /** Flash a point on the overlay (e.g. to confirm an interaction). */
  flashPoint(p5ref: p5, point: Point2D, color: string): void {
    p5ref.fill(color);
    p5ref.noStroke();
    p5ref.ellipse(point.x, point.y, 30, 30);
  }
}
