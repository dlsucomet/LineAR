# Pan vs Continue Button Hit-Test Fix

## Date
2026-06-11

## Problem
When hand tracking is active, the finger cursor position (`latestHandPosition`) is computed
through `gridToCanvas()` in `main.ts:464-469`, which **includes the pan offset** (`panX`/`panY`).

UI buttons (Continue, Yes/No) are rendered at **fixed canvas pixel positions** (e.g. `y = 10`)
and their hit-test rects are stored in raw canvas space.

After the user pans the canvas, the hand cursor position is offset by `(panX, panY)` relative
to the UI buttons. The cursor and the buttons live in different coordinate spaces, so hovering
over the button after panning becomes impossible.

Mouse input does not have this problem because `getMouseCanvasPos()` returns raw canvas
coordinates (no pan offset).

## Files Changed
- `src/main.ts` — `processInteractionFrame()` function

## Before (Bug)
In `processInteractionFrame()`:
- `pointerCanvas` is the active pointer in grid-canvas space (includes pan when hand tracking)
- `ui.setFingerPosition(pointerCanvas)` — cursor drawn at pan-offset position
- All UI button hit-tests use `pointerCanvas` directly

## After (Fix)
In `processInteractionFrame()`:
- Compute a separate `pointerUI` that strips the pan offset from hand-tracked positions:
  ```ts
  const pan = ui.getPanOffset();
  const pointerUI = pointerCanvas && isHandActive
    ? { x: pointerCanvas.x - pan.x, y: pointerCanvas.y - pan.y }
    : pointerCanvas;
  ```
- `ui.setFingerPosition(pointerUI)` — cursor drawn at correct screen position
- UI button hit-tests use `pointerUI`:
  - WAITING_FOR_OBJECT "Start Demo" button
  - CONFIRM_TRANSFORM / CONFIRM_RESET Yes/No buttons
  - TRANSFORMED Continue button
- Grid interactions (corner drag, arrow drag, panning) continue using `pointerCanvas`

## Revert Instructions
To revert this change, run:
```bash
git checkout src/main.ts
rm docs/PAN_BUTTON_FIX.md
```
