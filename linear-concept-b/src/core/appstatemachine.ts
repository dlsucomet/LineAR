// ─── src/core/appStateMachine.ts ─────────────────────────────────────────────
// Drives the 8-frame screen flow observed in the design frames.
//
// State transitions:
//
//  WAITING_FOR_OBJECT
//    → (object detected)         → OBJECT_DETECTED
//  OBJECT_DETECTED
//    → (stable for N frames)     → POINTS_CALCULATED
//  POINTS_CALCULATED
//    → (object removed)          → SHOW_CORNERS
//  SHOW_CORNERS
//    → (auto, short delay)       → SHOW_BASIS_VECTORS
//  SHOW_BASIS_VECTORS
//    → (user clicks Yes)         → CONFIRM_TRANSFORM
//  CONFIRM_TRANSFORM             (prompt: "Start linear transformation? Yes/No")
//    → Yes                       → TRANSFORMED
//    → No                        → SHOW_BASIS_VECTORS
//  TRANSFORMED
//    → (auto / user action)      → CONFIRM_RESET
//  CONFIRM_RESET                 (prompt: "Reset the Grid? Yes/No")
//    → Yes                       → WAITING_FOR_OBJECT
//    → No                        → TRANSFORMED

import type { DetectedObject, Matrix2x2, Point2D } from "../types/index.ts";

// ---------------------------------------------------------------------------
// State definitions
// ---------------------------------------------------------------------------

export type AppPhase =
  | "WAITING_FOR_OBJECT"    // Frame 8 – empty grid, waiting for object
  | "OBJECT_DETECTED"       // Frame 7 – object on grid, measuring
  | "POINTS_CALCULATED"     // Frame 6 – corners found, "please remove object"
  | "SHOW_CORNERS"          // Frame 5 – object removed, corners shown
  | "SHOW_BASIS_VECTORS"    // Frame 4 – identity basis vectors overlaid
  | "CONFIRM_TRANSFORM"     // Frame 3 – adjusted basis, yes/no prompt
  | "TRANSFORMED"           // Frame 2 – grid/object transformed
  | "CONFIRM_RESET"         // Frame 1 – yes/no reset prompt
  | "CALIBRATING";          // Projector-camera calibration — projecting markers

export interface AppState {
  phase: AppPhase;

  /** Corner points of detected object in board (JSXGraph) coordinates. */
  objectCorners: [Point2D, Point2D, Point2D, Point2D] | null;

  /** Centre of detected object in board coordinates. */
  objectCenter: Point2D | null;

  /** The 2×2 matrix the user has adjusted the basis vectors to. */
  pendingMatrix: Matrix2x2;

  /** The matrix that was confirmed and applied to the grid. */
  appliedMatrix: Matrix2x2;

  /** Raw detected objects from the camera tracker (current frame). */
  detectedObjects: DetectedObject[];

  /** Detection outline points for UI rendering (grid coordinates). */
  detectionOutline: Point2D[] | null;

  /** How many consecutive frames an object has been detected. */
  stableFrameCount: number;
}

// Frames an object must be stable before we consider it locked-in
const STABLE_FRAME_THRESHOLD = 10;

// Identity matrix shorthand
const IDENTITY: Matrix2x2 = [1, 0, 0, 1];

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createInitialState(): AppState {
  return {
    phase: "WAITING_FOR_OBJECT",
    objectCorners: null,
    objectCenter: null,
    pendingMatrix: [...IDENTITY],
    appliedMatrix: [...IDENTITY],
    detectedObjects: [],
    detectionOutline: null,
    stableFrameCount: 0,
  };
}

// ---------------------------------------------------------------------------
// Reducer-style transition function
// ---------------------------------------------------------------------------

export type AppAction =
  | { type: "OBJECTS_DETECTED"; payload: DetectedObject[] }
  | { type: "OBJECTS_CLEARED" }
  | { type: "CORNERS_LOCKED"; payload: { corners: [Point2D, Point2D, Point2D, Point2D]; center: Point2D } }
  | { type: "OBJECT_REMOVED" }
  | { type: "BASIS_ADJUSTED"; payload: Matrix2x2 }
  | { type: "CONFIRM_YES" }
  | { type: "CONFIRM_NO" }
  | { type: "TRANSFORMATION_DONE" }
  | { type: "RESET_CONFIRMED" }
  | { type: "PHASE_ADVANCE" }
  | { type: "CALIBRATE" }
  | { type: "CALIBRATION_DONE" };

export function transition(state: AppState, action: AppAction): AppState {
  switch (action.type) {

    // Camera sees objects
    case "OBJECTS_DETECTED": {
      if (state.phase !== "WAITING_FOR_OBJECT" && state.phase !== "OBJECT_DETECTED") {
        // Ignore spurious detections in later phases
        return { ...state, detectedObjects: action.payload };
      }
      const newCount = state.stableFrameCount + 1;
      if (newCount >= STABLE_FRAME_THRESHOLD && state.phase === "OBJECT_DETECTED") {
        // Already in OBJECT_DETECTED and stable enough → do nothing, wait for
        // explicit CORNERS_LOCKED action from the grid renderer
        return { ...state, detectedObjects: action.payload, stableFrameCount: newCount };
      }
      return {
        ...state,
        detectedObjects: action.payload,
        stableFrameCount: newCount,
        phase: "OBJECT_DETECTED",
      };
    }

    // Camera loses all objects
    case "OBJECTS_CLEARED": {
      if (state.phase === "OBJECT_DETECTED") {
        return { ...state, phase: "WAITING_FOR_OBJECT", stableFrameCount: 0, detectedObjects: [] };
      }
      if (state.phase === "POINTS_CALCULATED") {
        // User removed the object as instructed → show clean corners
        return { ...state, phase: "SHOW_CORNERS", detectedObjects: [] };
      }
      return { ...state, detectedObjects: [] };
    }

    // Grid renderer finished computing corners
    case "CORNERS_LOCKED": {
      if (state.phase !== "OBJECT_DETECTED") return state;
      return {
        ...state,
        phase: "POINTS_CALCULATED",
        objectCorners: action.payload.corners,
        objectCenter: action.payload.center,
      };
    }

    // Transition from SHOW_CORNERS to SHOW_BASIS_VECTORS (auto or triggered)
    case "OBJECT_REMOVED": {
      if (state.phase !== "POINTS_CALCULATED") return state;
      return { ...state, phase: "SHOW_CORNERS" };
    }

    // User finishes adjusting basis vectors
    case "BASIS_ADJUSTED": {
      if (state.phase === "SHOW_BASIS_VECTORS" || state.phase === "SHOW_CORNERS") {
        return {
          ...state,
          phase: "CONFIRM_TRANSFORM",
          pendingMatrix: action.payload,
        };
      }
      return { ...state, pendingMatrix: action.payload };
    }

    // User presses "Yes" on any confirmation prompt
    case "CONFIRM_YES": {
      if (state.phase === "CONFIRM_TRANSFORM") {
        return { ...state, phase: "TRANSFORMED", appliedMatrix: state.pendingMatrix };
      }
      if (state.phase === "CONFIRM_RESET") {
        return {
          ...createInitialState(),
          // preserve nothing – full reset
        };
      }
      return state;
    }

    // User presses "No"
    case "CONFIRM_NO": {
      if (state.phase === "CONFIRM_TRANSFORM") {
        return { ...state, phase: "SHOW_BASIS_VECTORS" };
      }
      if (state.phase === "CONFIRM_RESET") {
        return { ...state, phase: "TRANSFORMED" };
      }
      return state;
    }

    // Transformation animation finished
    case "TRANSFORMATION_DONE": {
      if (state.phase === "TRANSFORMED") {
        return { ...state, phase: "CONFIRM_RESET" };
      }
      return state;
    }

    // Auto-advance from SHOW_CORNERS → SHOW_BASIS_VECTORS
    case "PHASE_ADVANCE": {
      if (state.phase !== "SHOW_CORNERS") return state;
      return { ...state, phase: "SHOW_BASIS_VECTORS" };
    }

    case "CALIBRATE": {
      if (state.phase !== "WAITING_FOR_OBJECT") return state;
      return { ...state, phase: "CALIBRATING" };
    }

    case "CALIBRATION_DONE": {
      if (state.phase !== "CALIBRATING") return state;
      return { ...createInitialState() };
    }

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Selector helpers
// ---------------------------------------------------------------------------

/** Human-readable instruction text for each phase (matches frame designs). */
export function getInstructionText(phase: AppPhase): string {
  switch (phase) {
    case "WAITING_FOR_OBJECT": return "Waiting for Object. Please place object on the Grid";
    case "OBJECT_DETECTED": return "Object Detected";
    case "POINTS_CALCULATED": return "Vector Points Calculated. Please remove object";
    case "SHOW_CORNERS": return ""; // No text, just corners
    case "SHOW_BASIS_VECTORS": return "Basis Vectors displayed. Please adjust to your liking.";
    case "CONFIRM_TRANSFORM": return "Basis vectors adjusted. Start linear transformation?";
    case "TRANSFORMED": return "Object has been linearly transformed";
    case "CONFIRM_RESET": return "Reset the Grid?";
    case "CALIBRATING": return "Calibrating — projected markers visible on screen";
  }
}

/** Whether the current phase should show the Yes/No button pair. */
export function showConfirmButtons(phase: AppPhase): boolean {
  return phase === "CONFIRM_TRANSFORM" || phase === "CONFIRM_RESET";
}

/** Whether the current phase should show the basis vector arrows. */
export function showBasisVectors(phase: AppPhase): boolean {
  return (
    phase === "SHOW_BASIS_VECTORS" ||
    phase === "CONFIRM_TRANSFORM" ||
    phase === "TRANSFORMED" ||
    phase === "CONFIRM_RESET"
  );
}

/** Whether the corner markers should be drawn on the grid. */
export function showCorners(phase: AppPhase): boolean {
  return (
    phase === "POINTS_CALCULATED" ||
    phase === "SHOW_CORNERS" ||
    phase === "SHOW_BASIS_VECTORS" ||
    phase === "CONFIRM_TRANSFORM" ||
    phase === "TRANSFORMED" ||
    phase === "CONFIRM_RESET"
  );
}
