# Basis Vector Target Circles

## Overview

In the `SHOW_BASIS_VECTORS` phase, two colored target circles appear on the grid. The user must drag each basis arrow to its matching circle and hold still for 1.5 s to lock it. Red arrow (e1) must be locked first; green arrow (e2) becomes grabbable only after e1 is locked.

## Configuration

| Constant | File | Value | Purpose |
|----------|------|-------|---------|
| `TARGET_E1` | `tabletopui.ts` | `{x:3, y:0}` | Grid position of the red target circle |
| `TARGET_E2` | `tabletopui.ts` | `{x:0, y:3}` | Grid position of the green target circle |
| `ARROW_DWELL_PLACE_MS` | `demoplayer.ts` | `1500` | Milliseconds the arrow tip must dwell on the target to lock |
| `ARROW_GRAB_RADIUS_PX` | `demoplayer.ts` | `70` | Pixel distance from arrow tip to start a drag |
| `ARROW_SNAP_PX` | `demoplayer.ts` | `25` | Pixel distance from grid intersection to snap |

## Interaction Flow

```
SHOW_BASIS_VECTORS
  │
  ├─ Red circle (TARGET_E1) + Green circle (TARGET_E2) rendered on grid
  │
  ├─ User grabs red arrow (e1) ── drags freely, snaps to grid intersections
  │    └─ Arrow tip at TARGET_E1? → dwell timer starts (1.5 s)
  │        └─ Timer expires → e1Locked = true, blue outline on arrow tip
  │
  ├─ User grabs green arrow (e2) — only possible if e1Locked === true
  │    └─ Arrow tip at TARGET_E2? → dwell timer starts (1.5 s)
  │        └─ Timer expires → e2Locked = true, blue outline on arrow tip
  │
  └─ Both locked → dispatch BASIS_ADJUSTED with matrix [3, 0, 0, 3]
```

## Visual Indicators

| State | Arrow tip appearance |
|-------|---------------------|
| Dragging, no snap | follows finger |
| Snapped to grid | highlight ring (red/green) |
| Snapped on target circle + dwelling | pulsing dwell ring |
| **Locked** | **blue** (`#3a7bd5`) solid outline, radius 12, lineWidth 4 |

## Relevant Files

| File | Role |
|------|------|
| `src/components/tabletopui.ts` | `drawTargetCircles()` renders the red/green circles; `drawArrowDragFeedback()` draws blue lock ring |
| `src/main.ts` | `SHOW_BASIS_VECTORS` handler: grab → drag → dwell → lock → proceed |
| `src/core/demoplayer.ts` | Interaction timing constants |
