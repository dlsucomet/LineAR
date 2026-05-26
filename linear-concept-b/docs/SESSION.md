# Session Memory — May 24 2026

## Goal
Combine infinite pan with basis vector lock system and canvas buttons, while making hand tracking smooth on a laptop.

## Constraints & Preferences
- TFJS runtime must be used (research paper requirement), not MediaPipe WASM (bundling issues with `@mediapipe/hands`).
- Pan only during TRANSFORMED phase — no pan during CONFIRM_TRANSFORM/CONFIRM_RESET.
- Continue button is the only way to advance from TRANSFORMED (auto-transition removed).
- All buttons on canvas with dwell interaction — no HTML header strip.
- Origin 0,0 at center of canvas grid area.

## Progress

### Done
- Fixed basis vectors not at grid intersection: `drawBasisArrows`, `drawGhostArrows`, `drawArrowDragFeedback` now use `gridCenter + pan` and correct scale `/20` (tabletopui.ts).
- Fixed phase not switching: added `BASIS_ADJUSTED` dispatch when both arrows locked (main.ts).
- Fixed camera cycling: removed camera pause during interactive phases, reset `absentFrames=0` after `OBJECTS_CLEARED` (main.ts).
- Guarded detection handler: camera detection only processed during `WAITING_FOR_OBJECT`, `OBJECT_DETECTED`, `POINTS_CALCULATED` — stops blinking in later phases (main.ts).
- Removed verbose `console.log` calls from `detectObjects()` in cameratracker.ts + `[Hand detected]` from main.ts.
- Adjusted interaction thresholds: `ARROW_DWELL_PLACE_MS=600`, `ARROW_GRAB_RADIUS_PX=70`, `ARROW_SNAP_PX=25` (demoplayer.ts).
- Removed auto-transition `setTimeout` in TRANSFORMED (main.ts) — Continue button only.
- Added corner coordinate labels `(origX,origY) → (transformedX,transformedY)` during TRANSFORMED/CONFIRM_RESET (tabletopui.ts).
- Added 8px deadzone + EMA smoothing (`alpha=0.5`) on hand tracking in both callbacks (main.ts).
- Set `WEBGL_CPU_FORWARD=false` to force GPU path (handtracker.ts).
- Switched to MediaPipe runtime then reverted to TFJS — `@mediapipe/hands` real package is a Closure-compiled UMD that Vite can't bundle; shim restored in vite.config.ts.
- Created `docs/` with `ARCHITECTURE.md`, `INTERACTION.md`, `CONFIGURATION.md`, `SESSION.md`.
- Set `FRAME_SKIP=2` (handtracker.ts) — halved inference calls per frame.
- Unified both hand tracking callbacks to `DEADZONE_PX=8` (bootstrap was 4, setupDemoTracker was 8).
- Fixed corner animation (tabletopui.ts): extracted `getAnimationProgress()` method, animated `drawVirtualObject()` fill, fixed perpetual restart bug (completion sentinel `0` → `-1`), increased `ANIM_DURATION` from 1500ms → 10000ms.

### Known Issues
- Hand tracking still jittery and slow on laptop — TFJS WebGL inference struggles on integrated GPU even with GPU force, deadzone, smoothing, and FRAME_SKIP=2.
- `@mediapipe/hands` cannot be used with Vite — its Closure-compiled UMD output doesn't work as an ESM dependency. The shim at `src/shims/mediapipe-hands.ts` exports `Hands: undefined`.

## Next Steps
- Verify corner animation visually — check that fill and outline both animate smoothly over 5s, once per entry to TRANSFORMED.
- Diagnose TFJS backend — open browser console to check `[HandTracker] TF.js backend:` log to see if WebGL is actually active.
- If backend is `"cpu"` — investigate why `WEBGL_CPU_FORWARD=false` didn't force WebGL.
- Deadzone tradeoff — 8px is somewhat aggressive, monitor if fine-grained positioning becomes frustrating.

## Removed

### Done Button Prompt (corner adjustment phase)
- Removed the "Are you done adjusting the corners?" prompt during POINTS_CALCULATED — corner adjustment phase is not used.
- **`src/components/tabletopui.ts`**: `showDoneButton`, `btnDone`, `onDone` fields; `drawDoneButton` method; render call and hit test for Done button.
- **`src/main.ts`**: `ui.onDone` callback; `showDonePrompt` field; `hasAdjustedCorner` detection block.
- All removed code is commented in-place with `// [REMOVED: Done button prompt]` tags for traceability.

## Relevant Files

| File | Purpose |
|---|---|
| `src/core/handtracker.ts` | TFJS runtime, `WEBGL_CPU_FORWARD=false`, `FRAME_SKIP=2`, logs backend |
| `src/main.ts` | Smoothing+deadzone in both callbacks, detection phase guard, no camera pause, auto-transition removed, pan only in TRANSFORMED, `BASIS_ADJUSTED` dispatch, `absentFrames` reset |
| `src/components/tabletopui.ts` | Grid rendering, basis arrows/ghosts/feedback, corner labels with orig→transformed coords, corner animation (5s ease-out cubic, fill+outline) |
| `src/core/demoplayer.ts` | Interaction thresholds, phase timings, presets |
| `src/core/cameratracker.ts` | OpenCV.js detection (verbose logs removed) |
| `src/core/appstatemachine.ts` | Phase state machine reducer |
| `vite.config.ts` | Shim alias for `@mediapipe/hands` |
| `src/shims/mediapipe-hands.ts` | Stub exporting `Hands: undefined` |
| `docs/ARCHITECTURE.md` | File structure, phase state machine, actions |
| `docs/INTERACTION.md` | Input handling, dwell, basis vectors, pan, coordinate mapping |
| `docs/CONFIGURATION.md` | All thresholds, timings, constants |
| `docs/SESSION.md` | This file — session continuity memory |
