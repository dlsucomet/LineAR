# Header Strip Layout — Before & After

## Goal
Center the entire group (instruction text + buttons) as one unit inside the 40px header strip, so it looks like a single row spanning the canvas width:
```
┌──────────────────────────────────────────────────────────┐
│         Basis Vectors adjusted. [Yes] [No]               │
├──────────────────────────────────────────────────────────┤
│                          (grid)                          │
└──────────────────────────────────────────────────────────┘
```

---

## Before

### drawInstruction (all phases)
- **y**: `TEXT_STRIP_H - 12 = 28` ✓ (inside strip)
- **x**: `W / 2` (centered)
- **textAlign**: `center`

### drawButtons (CONFIRM_TRANSFORM, CONFIRM_RESET)
- **y**: `TEXT_STRIP_H + 6 = 46` — ✗ **6px into the grid**, outside the strip
- **x**: `W/2 - tw/2 + tw + gap + 4` — dangling off the right of the centered text
- Group is NOT centered as a unit — text centers independently, buttons hang off its right edge

### drawContinueButton (TRANSFORMED)
- **y**: `TEXT_STRIP_H + 6 = 46` — ✗ same bug, overlapping grid
- **x**: same dangling formula as drawButtons

---

## After

### Common layout (for CONFIRM_TRANSFORM, CONFIRM_RESET, TRANSFORMED)
All three phases use the same centered-group formula:

```
btnY         = 6                              (vertically centered in 40px strip)
textWidth    = measureText(instruction).width
buttonWidth  = phase has 2 buttons ? 64*2+10 : 100
groupWidth   = textWidth + 10 + buttonWidth
groupStartX  = (W - groupWidth) / 2
```

### drawInstruction
- **y**: unchanged at `TEXT_STRIP_H - 12 = 28`
- For CONFIRM_TRANSFORM, CONFIRM_RESET, TRANSFORMED:
  - `textAlign`: `left`
  - **x**: `groupStartX`
- For other phases: unchanged (`textAlign: center`, `W/2`)

### drawButtons (CONFIRM_TRANSFORM, CONFIRM_RESET)
- **y**: `TEXT_STRIP_H - 28 = 6` — moved UP into the strip
- **x**: `yesX = groupStartX + textWidth + 10`, `noX = yesX + 74`

### drawContinueButton (TRANSFORMED)
- **y**: `TEXT_STRIP_H - 28 = 6` — moved UP into the strip
- **x**: `groupStartX + textWidth + 10`

---

## Files Changed
- `src/components/tabletopui.ts` — 3 methods: `drawInstruction`, `drawButtons`, `drawContinueButton`

## Verification
- `npx tsc --noEmit` — must pass with no errors
- Buttons should render fully inside the 40px strip, not overlapping the grid
