

function setBoxState(ids, state) {

  if (!Array.isArray(ids)) {
    ids = [ids];
  }

  ids.forEach(id => {

    const box = document.getElementById(id);

    box.classList.remove(
      "copy-box",
      "correct-box",
      "wrong-box"
    );

    box.classList.add(state);

  });

}

const emptyInstruction = "";

const topInstructionText = [
  "Place paper on the blue area",
  "Represent the linear transformation as a sum of the given vectors",
  "Substitute the given vectors by their linear transformations",
  "Perform scalar multiplication on the highlighted vector",
  "Perform vector addition on the highlighted vector",
  "Now, try solving a problem on your own",
  "Turn the paper over",
  "Solve the linear transformation problem"
];

const sideInstructionText = [
  "Copy the blue element and blue vector on the empty highlighted area",
  "Copy the highlighted vector on the empty highlighted area",
  "Scalar multiplication is done this way:",
  "Vector addition is done this way:"
];

function setTopInstruction(text) {
  document.getElementById("top-instruction").textContent = text;
}

function setSideInstruction(text) {
  document.getElementById("side-instruction").textContent = text;
}


/*
  States

  0. Empty (Transitioning and whatever)
  1. Initialization
  2. First step (1)
  3. First step (1 correct)
  4. First step (1 wrong)
  5. First step (2)
  6. First step (2 correct)
  7. First step (2 wrong)
  8. Second step (1)
  9. Second step (1 correct)
  10. Second step (1 wrong)
  11. Second step (2)
  12. Second step (2 correct)
  13. Second step (2 wrong)
  14. Third step (1)
  15. Third step (1 correct)
  16. Third step (1 wrong)
  17. Third step (2)
  18. Third step (2 correct)
  19. Third step (2 wrong)
  20. Fourth step 
  21. Fourth step (correct)
  22. Fourth step (wrong)
  23. Now, try solving [5]
  24. Turn paper over [6]
  25. Solve on own [7]
*/

const steps = [
  { /* 0 */
    topText: emptyInstruction,

    sideText: emptyInstruction,

    highlights: []
  },

  { /* 1 */
    topText: topInstructionText[0],

    sideText: emptyInstruction,

    highlights: []
  },

  { /* 2 */
    topText: topInstructionText[1],

    sideText: sideInstructionText[0],

    highlights: [
      {
        ids: [
          "top-left-fifth",
          "top-left-ninth-second",
          "top-left-ninth-third",
          "top-mid-seventh-sixth",
          "top-mid-eighth-third",
          "top-mid-eighth-fourth",
          "top-mid-eighth-sixth",
          "top-mid-eighth-ninth",
          "top-mid-ninth-first",
          "top-mid-ninth-second",
          "top-mid-ninth-fourth",
          "top-mid-ninth-fifth",
          "top-mid-ninth-seventh",
          "top-mid-ninth-eighth",
        ],

        state: "copy-box"
      }
    ]
  },

  { /* 3 */
    topText: topInstructionText[1],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "top-mid-seventh-third",
          "top-mid-seventh-sixth",
          "top-mid-seventh-ninth",
          "top-mid-eighth-first",
          "top-mid-eighth-second",
          "top-mid-eighth-third",
          "top-mid-eighth-fourth",
          "top-mid-eighth-fifth",
          "top-mid-eighth-sixth",
          "top-mid-eighth-seventh",
          "top-mid-eighth-eighth",
          "top-mid-eighth-ninth",
          "top-mid-ninth-first",
          "top-mid-ninth-second",
          "top-mid-ninth-fourth",
          "top-mid-ninth-fifth",
          "top-mid-ninth-seventh",
          "top-mid-ninth-eighth",
        ],

        state: "correct-box"
      }
    ]
  },

  { /* 4 */
    topText: topInstructionText[1],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "top-mid-seventh-third",
          "top-mid-seventh-sixth",
          "top-mid-seventh-ninth",
          "top-mid-eighth-first",
          "top-mid-eighth-second",
          "top-mid-eighth-third",
          "top-mid-eighth-fourth",
          "top-mid-eighth-fifth",
          "top-mid-eighth-sixth",
          "top-mid-eighth-seventh",
          "top-mid-eighth-eighth",
          "top-mid-eighth-ninth",
          "top-mid-ninth-first",
          "top-mid-ninth-second",
          "top-mid-ninth-fourth",
          "top-mid-ninth-fifth",
          "top-mid-ninth-seventh",
          "top-mid-ninth-eighth",
        ],

        state: "wrong-box"
      }
    ]
  },

  { /* 5 */
    topText: topInstructionText[1],

    sideText: sideInstructionText[0],

    highlights: [
      {
        ids: [
          "top-left-ninth-eighth",
          "top-left-ninth-ninth",
          "top-mid-fifth",
          "top-right-seventh-sixth",
          "top-right-eighth-third",
          "top-right-eighth-fourth",
          "top-right-eighth-sixth",
          "top-right-eighth-ninth",
          "top-right-ninth-first",
          "top-right-ninth-second",
          "top-right-ninth-fourth",
          "top-right-ninth-fifth",
          "top-right-ninth-seventh",
          "top-right-ninth-eighth",
        ],

        state: "copy-box"
      }
    ]
  },

  { /* 6 */
    topText: topInstructionText[1],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "top-right-seventh-third",
          "top-right-seventh-sixth",
          "top-right-seventh-ninth",
          "top-right-eighth-first",
          "top-right-eighth-second",
          "top-right-eighth-third",
          "top-right-eighth-fourth",
          "top-right-eighth-fifth",
          "top-right-eighth-sixth",
          "top-right-eighth-seventh",
          "top-right-eighth-eighth",
          "top-right-eighth-ninth",
          "top-right-ninth-first",
          "top-right-ninth-second",
          "top-right-ninth-fourth",
          "top-right-ninth-fifth",
          "top-right-ninth-seventh",
          "top-right-ninth-eighth"
        ],

        state: "correct-box"
      }
    ]
  },

  { /* 7 */
    topText: topInstructionText[1],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "top-right-seventh-third",
          "top-right-seventh-sixth",
          "top-right-seventh-ninth",
          "top-right-eighth-first",
          "top-right-eighth-second",
          "top-right-eighth-third",
          "top-right-eighth-fourth",
          "top-right-eighth-fifth",
          "top-right-eighth-sixth",
          "top-right-eighth-seventh",
          "top-right-eighth-eighth",
          "top-right-eighth-ninth",
          "top-right-ninth-first",
          "top-right-ninth-second",
          "top-right-ninth-fourth",
          "top-right-ninth-fifth",
          "top-right-ninth-seventh",
          "top-right-ninth-eighth"
        ],

        state: "wrong-box"
      }
    ]
  },

  { /* 8 */
    topText: topInstructionText[2],

    sideText: sideInstructionText[1],

    highlights: [
      {
        ids: [
          "top-left-sixth",
          "mid-mid-second-third",
          "mid-mid-second-sixth",
          "mid-mid-second-ninth",
          "mid-mid-third-first",
          "mid-mid-third-second",
          "mid-mid-third-fourth",
          "mid-mid-third-fifth",
          "mid-mid-third-seventh",
          "mid-mid-third-eighth"
        ],

        state: "copy-box"
      }
    ]
  },

  { /* 9 */
    topText: topInstructionText[2],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-mid-second-third",
          "mid-mid-second-sixth",
          "mid-mid-second-ninth",
          "mid-mid-third-first",
          "mid-mid-third-second",
          "mid-mid-third-fourth",
          "mid-mid-third-fifth",
          "mid-mid-third-seventh",
          "mid-mid-third-eighth"
        ],

        state: "correct-box"
      }
    ]
  },

  { /* 10 */
    topText: topInstructionText[2],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-mid-second-third",
          "mid-mid-second-sixth",
          "mid-mid-second-ninth",
          "mid-mid-third-first",
          "mid-mid-third-second",
          "mid-mid-third-fourth",
          "mid-mid-third-fifth",
          "mid-mid-third-seventh",
          "mid-mid-third-eighth"
        ],

        state: "wrong-box"
      }
    ]
  },

  { /* 11 */
    topText: topInstructionText[2],

    sideText: sideInstructionText[1],

    highlights: [
      {
        ids: [
          "top-mid-sixth",
          "mid-right-second-third",
          "mid-right-second-sixth",
          "mid-right-second-ninth",
          "mid-right-third-first",
          "mid-right-third-second",
          "mid-right-third-fourth",
          "mid-right-third-fifth",
          "mid-right-third-seventh",
          "mid-right-third-eighth",
        ],

        state: "copy-box"
      }
    ]
  },

  { /* 12 */
    topText: topInstructionText[2],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-right-second-third",
          "mid-right-second-sixth",
          "mid-right-second-ninth",
          "mid-right-third-first",
          "mid-right-third-second",
          "mid-right-third-fourth",
          "mid-right-third-fifth",
          "mid-right-third-seventh",
          "mid-right-third-eighth",
        ],

        state: "correct-box"
      }
    ]
  },

  { /* 13 */
    topText: topInstructionText[2],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-right-second-third",
          "mid-right-second-sixth",
          "mid-right-second-ninth",
          "mid-right-third-first",
          "mid-right-third-second",
          "mid-right-third-fourth",
          "mid-right-third-fifth",
          "mid-right-third-seventh",
          "mid-right-third-eighth",
        ],

        state: "wrong-box"
      }
    ]
  },

  { /* 14 */
    topText: topInstructionText[3],

    sideText: sideInstructionText[2],

    highlights: [
      {
        ids: [
          "mid-mid-first-third",
          "mid-mid-first-sixth",
          "mid-mid-first-ninth",
          "mid-mid-second-first",
          "mid-mid-second-second",
          "mid-mid-second-third",
          "mid-mid-second-fourth",
          "mid-mid-second-fifth",
          "mid-mid-second-sixth",
          "mid-mid-second-seventh",
          "mid-mid-second-eighth",
          "mid-mid-second-ninth",
          "mid-mid-third-first",
          "mid-mid-third-second",
          "mid-mid-third-fourth",
          "mid-mid-third-fifth",
          "mid-mid-third-seventh",
          "mid-mid-third-eighth",
          "mid-mid-fourth"
        ],

        state: "copy-box"
      }
    ]
  },

  { /* 15 */
    topText: topInstructionText[3],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-mid-fourth"
        ],

        state: "correct-box"
      }
    ]
  },

  { /* 16 */
    topText: topInstructionText[3],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-mid-fourth"
        ],

        state: "wrong-box"
      }
    ]
  },

  { /* 17 */
    topText: topInstructionText[3],

    sideText: sideInstructionText[2],

    highlights: [
      {
        ids: [
          "mid-right-first-third",
          "mid-right-first-sixth",
          "mid-right-first-ninth",
          "mid-right-second-first",
          "mid-right-second-second",
          "mid-right-second-third",
          "mid-right-second-fourth",
          "mid-right-second-fifth",
          "mid-right-second-sixth",
          "mid-right-second-seventh",
          "mid-right-second-eighth",
          "mid-right-second-ninth",
          "mid-right-third-first",
          "mid-right-third-second",
          "mid-right-third-fourth",
          "mid-right-third-fifth",
          "mid-right-third-seventh",
          "mid-right-third-eighth",
          "mid-mid-sixth"
        ],

        state: "copy-box"
      }
    ]
  },

  { /* 18 */
    topText: topInstructionText[3],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-mid-sixth"
        ],

        state: "correct-box"
      }
    ]
  },

  { /* 19 */
    topText: topInstructionText[3],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-mid-sixth"
        ],

        state: "wrong-box"
      }
    ]
  },

  { /* 20 */
    topText: topInstructionText[4],

    sideText: sideInstructionText[3],

    highlights: [
      {
        ids: [
          "mid-mid-fourth",
          "mid-mid-fifth",
          "mid-mid-sixth",
          "mid-mid-seventh"
        ],

        state: "copy-box"
      }
    ]
  },

  { /* 21 */
    topText: topInstructionText[4],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-mid-seventh"
        ],

        state: "correct-box"
      }
    ]
  },

  { /* 22 */
    topText: topInstructionText[4],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "mid-mid-seventh"
        ],

        state: "wrong-box"
      }
    ]
  },

  { /* 23 */
    topText: topInstructionText[5],

    sideText: emptyInstruction,

    highlights: []
  },

  { /* 24 */
    topText: topInstructionText[6],

    sideText: emptyInstruction,

    highlights: []
  },

  { /* 25 */
    topText: topInstructionText[7],

    sideText: emptyInstruction,

    highlights: []
  },

];

function renderStep(stepIndex) {

  const step = steps[stepIndex];

  setTopInstruction(step.topText);

  setSideInstruction(step.sideText);

  step.highlights.forEach(highlight => {

    setBoxState(
      highlight.ids,
      highlight.state
    );

  });

}

/*
document
  .getElementById("checker")
  .addEventListener("click", () => {
    renderStep(25)
  });
*/

const boxes = document.querySelectorAll(".box");

boxes.forEach(box => {
  box.addEventListener("click", () => {

    if (box.classList.contains("copy-box")) {
      box.classList.remove("copy-box");
      box.classList.add("correct-box");

    } else if (box.classList.contains("correct-box")) {
      box.classList.remove("correct-box");
      box.classList.add("wrong-box");

    } else if (box.classList.contains("wrong-box")) {
      box.classList.remove("wrong-box");

    } else {
      box.classList.add("copy-box");
    }

  });
});

const bigBoxes = document.querySelectorAll(".big-box");

bigBoxes.forEach(box => {
  box.addEventListener("click", () => {

    if (box.classList.contains("copy-box")) {
      box.classList.remove("copy-box");
      box.classList.add("correct-box");

    } else if (box.classList.contains("correct-box")) {
      box.classList.remove("correct-box");
      box.classList.add("wrong-box");

    } else if (box.classList.contains("wrong-box")) {
      box.classList.remove("wrong-box");

    } else {
      box.classList.add("copy-box");
    }

  });
});