# Interaction Guide

## Pointer Input

Two input sources, resolved each frame in `main.ts:processInteractionFrame`:

1. **Hand tracking** (camera + MediaPipe) — index fingertip mapped to canvas coordinates. Active when `isHandActive` is true (hand seen within last 400ms).
2. **Mouse / touch** — fallback via `ui.getMouseCanvasPos()`.

Hand tracking uses TF.js WASM backend (`@tensorflow/tfjs-backend-wasm`) for inference. The `lite` model is used with `maxHands: 1`.

## Dwell Interaction

All buttons and interactions use dwell (hover-and-hold) instead of click:

- **HOVER_DWELL_MS** (400ms) — time to dwell on a button/prompt to activate
- Dwell progress resets when pointer leaves the target area

## Canvas Panning

Pan is available only during **TRANSFORMED** phase. Enabled via pointer-down drag anywhere outside the Continue button.

- Pan offsets stored in `TabletopUI.panX` / `panY`
- Grid center = `(canvas.width/2 + panX, TEXT_STRIP_H + gridHeight/2 + panY)`
- All grid-drawn content (grid lines, corners, basis arrows) respects pan offset

Pan not available during CONFIRM_TRANSFORM, CONFIRM_RESET, or SHOW_BASIS_VECTORS to avoid competing with button/arrow interaction.

## Basis Vectors (SHOW_BASIS_VECTORS Phase)

Two arrow tips (e1, e2) represent the 2×2 transformation matrix columns:

| Arrow | Matrix column | Initial position |
|---|---|---|
| e1 (blue) | `[mat[0], mat[2]]` | (1, 0) |
| e2 (light blue) | `[mat[1], mat[3]]` | (0, 1) |

### Arrow Interaction Sequence

```
1. GRAB   → pointer within ARROW_GRAB_RADIUS_PX (70px) of arrow tip
2. DRAG   → move pointer; tip snaps to nearest grid intersection within ARROW_SNAP_PX (25px)
3. DWELL  → hold still on intersection for ARROW_DWELL_PLACE_MS (600ms) → arrow is "placed"
4. LOCK   → after ARROW_PLACED_LOCK_MS (3000ms) → arrow auto-locks in position
5. COOLDOWN → after ARROW_LOCK_COOLDOWN_MS (5000ms) → arrow becomes re-grabbable
```

- While dragging, the dwell timer resets if the pointer leaves the snap zone
- Placed arrows can be re-grabbed after a short cooldown (800ms)
- When BOTH arrows are locked → `BASIS_ADJUSTED` dispatched → advances to CONFIRM_TRANSFORM
- Ghost arrow targets (dashed, semi-transparent) show the preset target positions

### Coordinate System

Arrow tip positions are computed in grid space then mapped to canvas:

```
canvasX = gridCenterX + mat[i] * scale
canvasY = gridCenterY - mat[j] * scale    (Y is flipped)
```

Where `scale = min(canvas.width, gridHeight) / 20` and grid range is -10..10.

## Corner Interaction (POINTS_CALCULATED Phase)

Four corner circles of the detected object. Drag to adjust, snap to nearest integer grid intersection. Auto-release after `CORNER_LOCK_DELAY_MS` (3s) of stable snap.

## Transformed Phase

- Continue button at top-right of instruction text (dwell to advance to CONFIRM_RESET)
- Pan available anywhere outside Continue button
- Corner labels show `(origX,origY) → (transformedX,transformedY)` for each corner
- Basis arrows show the applied matrix; matrix label displayed next to arrow origin

## Coordinate Mapping

| Function | Purpose | Location |
|---|---|---|
| `gridToCanvas(p, grid)` | Grid coords → canvas pixels | `tabletopui.ts:843` |
| `canvasToGrid(p, grid)` | Canvas pixels → grid coords | `tabletopui.ts:854` |

Both use: center = `grid center + pan`, scale = `min(grid.width, grid.height) / 20`, Y-flip.
