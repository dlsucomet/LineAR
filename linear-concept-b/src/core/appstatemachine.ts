// ─── src/core/appStateMachine.ts ─────────────────────────────────────────────
// Drives the wizard-of-oz linear transformation demo.
//
// Phases:
//   SHOW_CORNERS           — dashed 5×3 rect + 4 corner markers, auto-advances
//   SHOW_BASIS_VECTORS     — basis arrows + target circles, hand-drag to place
//   CONFIRM_TRANSFORM      — "Start transformation? Yes/No"
//   TRANSFORMED            — grid/object transformed + Continue button
//   CONFIRM_RESET          — "Reset the Grid? Yes/No"
//
// The W key advances through all phases with a fallback during
// SHOW_BASIS_VECTORS (skips arrow placement sub-steps).

import type { Matrix2x2, Point2D } from "../types/index.ts";

// ---------------------------------------------------------------------------
// State definitions
// ---------------------------------------------------------------------------

export type AppPhase =
  | "SHOW_CORNERS"          // Frame 1 — highlight area + corners
  | "SHOW_BASIS_VECTORS"    // Frame 2 — basis arrows + target circles
  | "CONFIRM_TRANSFORM"     // Frame 3 — yes/no prompt
  | "TRANSFORMED"           // Frame 4 — grid fully transformed
  | "CONFIRM_RESET";        // Frame 5 — yes/no reset prompt

export interface AppState {
  phase: AppPhase;

  /** Corner points of the virtual object in grid coordinates. */
  objectCorners: [Point2D, Point2D, Point2D, Point2D] | null;

  /** Centre of the virtual object in grid coordinates. */
  objectCenter: Point2D | null;

  /** The 2×2 matrix the user has adjusted the basis vectors to. */
  pendingMatrix: Matrix2x2;

  /** The matrix that was confirmed and applied to the grid. */
  appliedMatrix: Matrix2x2;

  /** Sub-step within SHOW_BASIS_VECTORS: 0=place e₁, 1=place e₂, 2=done. */
  basisStep: 0 | 1 | 2;
}

// Identity matrix shorthand
const IDENTITY: Matrix2x2 = [1, 0, 0, 1];

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createInitialState(): AppState {
  return {
    phase: "SHOW_CORNERS",
    objectCorners: null,
    objectCenter: null,
    pendingMatrix: [...IDENTITY],
    appliedMatrix: [...IDENTITY],
    basisStep: 0,
  };
}

// ---------------------------------------------------------------------------
// Reducer-style transition function
// ---------------------------------------------------------------------------

export type AppAction =
  | { type: "PHASE_ADVANCE" }
  | { type: "BASIS_STEP"; payload: 0 | 1 | 2 }
  | { type: "BASIS_ADJUSTED"; payload: Matrix2x2 }
  | { type: "CONFIRM_YES" }
  | { type: "CONFIRM_NO" }
  | { type: "TRANSFORMATION_DONE" };

export function transition(state: AppState, action: AppAction): AppState {
  switch (action.type) {

    // Advance from SHOW_CORNERS → SHOW_BASIS_VECTORS
    case "PHASE_ADVANCE": {
      if (state.phase !== "SHOW_CORNERS") return state;
      return { ...state, phase: "SHOW_BASIS_VECTORS", basisStep: 0 };
    }

    // Update sub-step within SHOW_BASIS_VECTORS
    case "BASIS_STEP": {
      if (state.phase !== "SHOW_BASIS_VECTORS") return state;
      return { ...state, basisStep: action.payload };
    }

    // User finished adjusting basis vectors → CONFIRM_TRANSFORM
    case "BASIS_ADJUSTED": {
      if (state.phase !== "SHOW_BASIS_VECTORS") return state;
      return {
        ...state,
        phase: "CONFIRM_TRANSFORM",
        pendingMatrix: action.payload,
      };
    }

    // User presses "Yes" on any confirmation prompt
    case "CONFIRM_YES": {
      if (state.phase === "CONFIRM_TRANSFORM") {
        return { ...state, phase: "TRANSFORMED", appliedMatrix: state.pendingMatrix };
      }
      if (state.phase === "CONFIRM_RESET") {
        return { ...createInitialState() };
      }
      return state;
    }

    // User presses "No"
    case "CONFIRM_NO": {
      if (state.phase === "CONFIRM_TRANSFORM") {
        return { ...state, phase: "SHOW_BASIS_VECTORS", basisStep: 0 };
      }
      if (state.phase === "CONFIRM_RESET") {
        return { ...state, phase: "TRANSFORMED" };
      }
      return state;
    }

    // Transformation animation finished → show Continue button / reset prompt
    case "TRANSFORMATION_DONE": {
      if (state.phase !== "TRANSFORMED") return state;
      return { ...state, phase: "CONFIRM_RESET" };
    }

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Selector helpers
// ---------------------------------------------------------------------------

/** Human-readable instruction text for each phase. */
export function getInstructionText(phase: AppPhase): string {
  switch (phase) {
    case "SHOW_CORNERS": return "Place object on the highlighted area";
    case "SHOW_BASIS_VECTORS": return "Drag arrow tips to the colored targets";
    case "CONFIRM_TRANSFORM": return "Basis vectors adjusted. Start linear transformation?";
    case "TRANSFORMED": return "Object has been linearly transformed";
    case "CONFIRM_RESET": return "Reset the Grid?";
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
export function showCorners(_phase: AppPhase): boolean {
  return true;
}
