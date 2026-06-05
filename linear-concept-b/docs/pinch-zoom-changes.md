# Pinch-to-Pan & Pinch-to-Zoom

**Date:** 2026-06-05

Adds two-finger pinch gestures for panning and zooming the grid canvas in
TRANSFORMED and CONFIRM_RESET phases.

## Changeset 1 – CSS

### `index.html`
- Added `touch-action: none` to `#main-canvas` to prevent the browser from
  intercepting two-finger gestures (page zoom / scroll).

## Changeset 2 – Zoom state + touch handlers

### `src/components/tabletopui.ts`

| Location | What | Why |
|----------|------|-----|
| `~74` | New field `zoom = 1` | Multiplicative zoom factor |
| `~75` | New field `currentPhase` | Track phase for gesture gating |
| `~76-78` | Touch tracking fields | `activeTouches` map, `lastPinchDist`, `lastPinchMid` |
| Constructor `~141` | Touch listeners | `touchstart`/`touchmove` (passive:false) + `touchend` |
| `draw()` | `this.currentPhase = state.phase` | Keep phase in sync |
| After mouse helpers | `getZoom()`, `handleTouchStart/Move/End`, `getTouchDist`, `getTouchMid`, `startPinch`, `updatePinch` | Core gesture logic |
| `setPanOffset` | Scale → `effScale = scale * zoom` | Bounds clamping respects zoom |
| `autoFrameTransformed` | `scale * this.zoom` | Initial framing respects current zoom |
| `getLayoutProperties` | `scale * this.zoom` | External callers see zoomed scale |
| `drawGrid` (×1) | `scale * this.zoom` | Grid lines zoom |
| `drawBasisArrows` (×1) | `scale * this.zoom` | Arrow tips zoom |
| `drawGhostArrows` (×1) | `scale * this.zoom` | Ghost arrows zoom |
| `drawTargetCircles` (×1) | `scale * this.zoom` | Target circles zoom |
| `drawArrowDragFeedback` (×2) | `scale * this.zoom` | Arrow feedback rings zoom |
| `gridToCanvas` | `scale * this.zoom` | Grid→canvas conversion zooms |
| `canvasToGrid` | `scale * this.zoom` | Canvas→grid conversion zooms |

#### Gesture behaviour

- **Two fingers** in TRANSFORMED / CONFIRM_RESET: `preventDefault()` stops
  browser zoom/scroll; midpoint tracks panning, distance tracks zooming.
- **Single finger** any phase: no `preventDefault`, existing mouse/corner-drag
  logic works unmodified.
- **Other phases**: two-finger gestures are ignored (no `preventDefault`);
  browser handles them normally.

#### Zoom formula

Zoom toward the pinch midpoint:
```
G = grid point under OLD midpoint at (oldZoom, oldPan)
pan = NEW midpoint − canvasCenter − G · baseScale · newZoom
```

This keeps the same grid point under your fingers as you pinch — natural
tablet feel. Pure pan (midpoint movement) and pure zoom (distance change)
both fall out correctly.

#### Clamping

- Zoom range: 0.3× .. 5×
- Pan bounds: scaled by zoom so content doesn't go off-screen

## Changeset 3 – main.ts

### `src/main.ts:828`
```typescript
const scale = Math.min(canvas.width, gridHeight) / (GRID_RANGE * 2) * ui.getZoom();
```

All downstream coordinate math (corner hit-testing, arrow snapping, etc.)
automatically uses the zoomed scale.

## Revert

```bash
git revert <commit-hash>
```

Or manually:

1. `index.html` – remove `touch-action: none` from `#main-canvas`
2. `src/components/tabletopui.ts` – remove `zoom` field, touch fields,
   touch listeners, touch handlers. Revert all `scale * this.zoom` to
   plain `scale`. Revert `setPanOffset` to use `scale` only.
3. `src/main.ts` – remove `* ui.getZoom()` from line 828.
