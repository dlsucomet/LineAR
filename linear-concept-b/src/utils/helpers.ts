// ─── src/utils/helpers.ts ─────────────────────────────────────────────────────

let _idCounter = 0;

/** Generate a short unique ID string. */
export function generateId(): string {
  return `obj_${Date.now()}_${_idCounter++}`;
}

/**
 * Map a camera-space pixel coordinate into JSXGraph board coordinates.
 * Assumes the board's bounding box is symmetric around the origin.
 *
 * @param camPx       - Point in camera pixel space { x, y }
 * @param camW        - Camera frame width in pixels
 * @param camH        - Camera frame height in pixels
 * @param boardHalf   - Half the board's grid size (e.g. 10 for a -10..10 board)
 */
export function camToBoard(
  camPx: { x: number; y: number },
  camW: number,
  camH: number,
  boardHalf: number,
): { x: number; y: number } {
  return {
    x: ((camPx.x / camW) * 2 - 1) * boardHalf,
    // Flip Y: camera Y increases downward, board Y increases upward
    y: -((camPx.y / camH) * 2 - 1) * boardHalf,
  };
}

/** Clamp a number between min and max. */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Simple debounce wrapper. */
export function debounce<T extends (...args: unknown[]) => void>(
  fn: T,
  delayMs: number,
): (...args: Parameters<T>) => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  return (...args: Parameters<T>) => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delayMs);
  };
}
