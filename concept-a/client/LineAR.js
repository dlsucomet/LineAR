const emptyInstruction = "";

const topInstructionText = [
  "Place paper on the blue area",
  "",
  "Represent the linear transformation as a sum of the given vectors",
  "Perform scalar multiplication on the highlighted vector",
  "Perform vector addition on the highlighted vector",
  "Now, try solving a problem on your own",
  "Turn the paper over",
  "Solve the linear transformation problem"
];

const sideInstructionText = [
  "",
  "Copy the blue element and blue vector on the empty highlighted area",
  `Scalar multiplication is done this way:
  \\[
  a
  \\begin{bmatrix}
  b \\\\
  c
  \\end{bmatrix}
  =
  \\begin{bmatrix}
  a \\cdot b \\\\
  a \\cdot c
  \\end{bmatrix}
  \\]
  `,
  `Vector addition is done this way:
  \\[
  \\begin{bmatrix}
  a \\\\
  b
  \\end{bmatrix}
  +
  \\begin{bmatrix}
  c \\\\
  d
  \\end{bmatrix}
  =
  \\begin{bmatrix}
  a + c \\\\
  b + d
  \\end{bmatrix}
  \\]
  `
];

/*
  States

  0. Empty (Transitioning and whatever)
  1. Initialization
  
  --Removed
  2. First step (1)
  3. First step (1 correct) 
  4. First step (1 wrong)
  5. First step (2)
  6. First step (2 correct)
  7. First step (2 wrong)
  --Removed

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

  /*
  { 2 
    topText: topInstructionText[1],

    sideText: sideInstructionText[0],

    highlights: [
      {
        ids: [
          "top-left-fifth-sixth",
          "top-left-fifth-ninth",
          "top-left-sixth-fourth",
          "top-left-sixth-seventh",
          "top-left-ninth-third",
          "top-mid-seventh-fifth",
          "top-mid-eighth-first",
          "top-mid-eighth-second",
          "top-mid-eighth-fourth",
          "top-mid-eighth-fifth",
        ],

        state: "copy-box"
      }
    ]
  },

  { 3
    topText: topInstructionText[1],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "top-mid-seventh-second",
          "top-mid-seventh-third",
          "top-mid-seventh-fifth",
          "top-mid-seventh-sixth",
          "top-mid-eighth-first",
          "top-mid-eighth-second",
          "top-mid-eighth-fourth",
          "top-mid-eighth-fifth",
        ],

        state: "correct-box"
      }
    ]
  },

  { 4
    topText: topInstructionText[1],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "top-mid-seventh-second",
          "top-mid-seventh-third",
          "top-mid-seventh-fifth",
          "top-mid-seventh-sixth",
          "top-mid-eighth-first",
          "top-mid-eighth-second",
          "top-mid-eighth-fourth",
          "top-mid-eighth-fifth",
        ],

        state: "wrong-box"
      }
    ]
  },

  { 5 
    topText: topInstructionText[1],

    sideText: sideInstructionText[0],

    highlights: [
      {
        ids: [
          "top-left-ninth-sixth",
          "top-mid-fourth-fifth",
          "top-mid-fourth-sixth",
          "top-mid-fourth-eighth",
          "top-mid-fourth-ninth",
          "top-mid-ninth-sixth",
          "top-right-seventh-second",
          "top-right-seventh-third",
          "top-right-seventh-fifth",
          "top-right-seventh-sixth",
        ],

        state: "copy-box"
      }
    ]
  },

  { 6 
    topText: topInstructionText[1],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "top-mid-ninth-third",
          "top-mid-ninth-sixth",
          "top-right-seventh-first",
          "top-right-seventh-second",
          "top-right-seventh-third",
          "top-right-seventh-fourth",
          "top-right-seventh-fifth",
          "top-right-seventh-sixth",
        ],

        state: "correct-box"
      }
    ]
  },

  {  7 
    topText: topInstructionText[1],

    sideText: emptyInstruction,

    highlights: [
      {
        ids: [
          "top-mid-ninth-third",
          "top-mid-ninth-sixth",
          "top-right-seventh-first",
          "top-right-seventh-second",
          "top-right-seventh-third",
          "top-right-seventh-fourth",
          "top-right-seventh-fifth",
          "top-right-seventh-sixth",
        ],

        state: "wrong-box"
      }
    ]
  },
  */

  { /* 8 */
    topText: topInstructionText[2],

    sideText: sideInstructionText[1],

    highlights: [
      {
        ids: [
          /*
          "top-left-sixth-sixth",
          "top-left-sixth-ninth",
          "top-mid-seventh-ninth",
          "top-mid-eighth-seventh",
          "mid-mid-first-third",
          "mid-mid-second-first",
          */
          "top-mid-fourth-fourth",
          "top-mid-fourth-fifth",
          "top-mid-fourth-seventh",
          "top-mid-fourth-eighth",
          "top-mid-seventh-first",
          "top-mid-seventh-second",

          "top-mid-seventh-fourth",

          "mid-left-third-fifth",
          "mid-left-third-sixth",
          "mid-left-third-eighth",
          "mid-left-third-ninth",
          
          "mid-mid-first-second",
          "mid-mid-first-third",
          "mid-mid-first-fifth",
          "mid-mid-first-sixth",
          "mid-mid-first-eighth",
          "mid-mid-first-ninth",
          "mid-mid-fourth-second",
          "mid-mid-fourth-third"
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
          /*
          "top-mid-seventh-ninth",
          "top-mid-eighth-seventh",
          "mid-mid-first-third",
          "mid-mid-second-first",
          */
          "mid-left-third-second",
          "mid-left-third-third",
          "mid-left-third-fifth",
          "mid-left-third-sixth",
          "mid-left-third-eighth",
          "mid-left-third-ninth",
          "mid-left-sixth-second",
          "mid-left-sixth-third",
        
          "mid-mid-first-first",
          "mid-mid-first-second",
          "mid-mid-first-third",
          "mid-mid-first-fourth",
          "mid-mid-first-fifth",
          "mid-mid-first-sixth",
          "mid-mid-first-seventh",
          "mid-mid-first-eighth",
          "mid-mid-first-ninth",
          "mid-mid-fourth-first",
          "mid-mid-fourth-second",
          "mid-mid-fourth-third"
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
          /*
          "top-mid-seventh-ninth",
          "top-mid-eighth-seventh",
          "mid-mid-first-third",
          "mid-mid-second-first",
          */
          "mid-left-third-second",
          "mid-left-third-third",
          "mid-left-third-fifth",
          "mid-left-third-sixth",
          "mid-left-third-eighth",
          "mid-left-third-ninth",
          "mid-left-sixth-second",
          "mid-left-sixth-third",
        
          "mid-mid-first-first",
          "mid-mid-first-second",
          "mid-mid-first-third",
          "mid-mid-first-fourth",
          "mid-mid-first-fifth",
          "mid-mid-first-sixth",
          "mid-mid-first-seventh",
          "mid-mid-first-eighth",
          "mid-mid-first-ninth",
          "mid-mid-fourth-first",
          "mid-mid-fourth-second",
          "mid-mid-fourth-third"
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
          /*
          "top-mid-fifth-fifth",
          "top-mid-fifth-eighth",
          "top-mid-ninth-eighth",
          "top-mid-ninth-ninth",
          "mid-mid-third-second",
          "mid-mid-third-third",
          */
          "top-mid-sixth-sixth",
          "top-mid-sixth-ninth",
          "top-right-fourth-fourth",
          "top-right-fourth-seventh",

          "top-mid-seventh-seventh",
          
          "mid-mid-second-fifth",
          "mid-mid-second-sixth",
          "mid-mid-second-eighth",
          "mid-mid-second-ninth",

          "mid-mid-third-second",
          "mid-mid-third-third",
          "mid-mid-third-fifth",
          "mid-mid-third-sixth",
          "mid-mid-third-eighth",
          "mid-mid-third-ninth",
          "mid-mid-sixth-second",
          "mid-mid-sixth-third",
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
          /*
          "top-mid-ninth-eighth",
          "top-mid-ninth-ninth",
          "mid-mid-third-second",
          "mid-mid-third-third",
          */
          "mid-mid-second-second",
          "mid-mid-second-third",
          "mid-mid-second-fifth",
          "mid-mid-second-sixth",
          "mid-mid-second-eighth",
          "mid-mid-second-ninth",
          "mid-mid-fifth-second",
          "mid-mid-fifth-third",

          "mid-mid-third-first",
          "mid-mid-third-second",
          "mid-mid-third-third",
          "mid-mid-third-fourth",
          "mid-mid-third-fifth",
          "mid-mid-third-sixth",
          "mid-mid-third-seventh",
          "mid-mid-third-eighth",
          "mid-mid-third-ninth",
          "mid-mid-sixth-first",
          "mid-mid-sixth-second",
          "mid-mid-sixth-third",
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
          /*
          "top-mid-ninth-eighth",
          "top-mid-ninth-ninth",
          "mid-mid-third-second",
          "mid-mid-third-third",
          */
          "mid-mid-second-second",
          "mid-mid-second-third",
          "mid-mid-second-fifth",
          "mid-mid-second-sixth",
          "mid-mid-second-eighth",
          "mid-mid-second-ninth",
          "mid-mid-fifth-second",
          "mid-mid-fifth-third",

          "mid-mid-third-first",
          "mid-mid-third-second",
          "mid-mid-third-third",
          "mid-mid-third-fourth",
          "mid-mid-third-fifth",
          "mid-mid-third-sixth",
          "mid-mid-third-seventh",
          "mid-mid-third-eighth",
          "mid-mid-third-ninth",
          "mid-mid-sixth-first",
          "mid-mid-sixth-second",
          "mid-mid-sixth-third",
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
          "mid-left-third-second",
          "mid-left-third-third",
          "mid-left-third-fifth",
          "mid-left-third-sixth",
          "mid-left-third-eighth",
          "mid-left-third-ninth",
          "mid-left-sixth-second",
          "mid-left-sixth-third",
        
          "mid-mid-first-first",
          "mid-mid-first-second",
          "mid-mid-first-third",
          "mid-mid-first-fourth",
          "mid-mid-first-fifth",
          "mid-mid-first-sixth",
          "mid-mid-first-seventh",
          "mid-mid-first-eighth",
          "mid-mid-first-ninth",
          "mid-mid-fourth-first",
          "mid-mid-fourth-second",
          "mid-mid-fourth-third",

          "mid-mid-fourth-seventh",
          "mid-mid-fourth-eighth",
          "mid-mid-seventh-first",
          "mid-mid-seventh-second",
          "mid-mid-seventh-fourth",
          "mid-mid-seventh-fifth",
          "mid-mid-seventh-seventh",
          "mid-mid-seventh-eighth",
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
          "mid-mid-fourth-seventh",
          "mid-mid-fourth-eighth",
          "mid-mid-seventh-first",
          "mid-mid-seventh-second",
          "mid-mid-seventh-fourth",
          "mid-mid-seventh-fifth",
          "mid-mid-seventh-seventh",
          "mid-mid-seventh-eighth",
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
          "mid-mid-fourth-seventh",
          "mid-mid-fourth-eighth",
          "mid-mid-seventh-first",
          "mid-mid-seventh-second",
          "mid-mid-seventh-fourth",
          "mid-mid-seventh-fifth",
          "mid-mid-seventh-seventh",
          "mid-mid-seventh-eighth",
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
          "mid-mid-second-second",
          "mid-mid-second-third",
          "mid-mid-second-fifth",
          "mid-mid-second-sixth",
          "mid-mid-second-eighth",
          "mid-mid-second-ninth",
          "mid-mid-fifth-second",
          "mid-mid-fifth-third",

          "mid-mid-third-first",
          "mid-mid-third-second",
          "mid-mid-third-third",
          "mid-mid-third-fourth",
          "mid-mid-third-fifth",
          "mid-mid-third-sixth",
          "mid-mid-third-seventh",
          "mid-mid-third-eighth",
          "mid-mid-third-ninth",
          "mid-mid-sixth-first",
          "mid-mid-sixth-second",
          "mid-mid-sixth-third",

          "mid-mid-fifth-seventh",
          "mid-mid-fifth-eighth",
          "mid-mid-eighth-first",
          "mid-mid-eighth-second",
          "mid-mid-eighth-fourth",
          "mid-mid-eighth-fifth",
          "mid-mid-eighth-seventh",
          "mid-mid-eighth-eighth",
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
          "mid-mid-fifth-seventh",
          "mid-mid-fifth-eighth",
          "mid-mid-eighth-first",
          "mid-mid-eighth-second",
          "mid-mid-eighth-fourth",
          "mid-mid-eighth-fifth",
          "mid-mid-eighth-seventh",
          "mid-mid-eighth-eighth",
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
          "mid-mid-fifth-seventh",
          "mid-mid-fifth-eighth",
          "mid-mid-eighth-first",
          "mid-mid-eighth-second",
          "mid-mid-eighth-fourth",
          "mid-mid-eighth-fifth",
          "mid-mid-eighth-seventh",
          "mid-mid-eighth-eighth",
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
          "mid-mid-fourth-seventh",
          "mid-mid-fourth-eighth",
          "mid-mid-fourth-ninth",
          "mid-mid-seventh-first",
          "mid-mid-seventh-second",
          "mid-mid-seventh-third",
          "mid-mid-seventh-fourth",
          "mid-mid-seventh-fifth",
          "mid-mid-seventh-sixth",
          "mid-mid-seventh-seventh",
          "mid-mid-seventh-eighth",
          "mid-mid-seventh-ninth",

          "mid-mid-fifth-seventh",
          "mid-mid-fifth-eighth",
          "mid-mid-eighth-first",
          "mid-mid-eighth-second",
          "mid-mid-eighth-fourth",
          "mid-mid-eighth-fifth",
          "mid-mid-eighth-seventh",
          "mid-mid-eighth-eighth",

          "bot-mid-first-fourth",
          "bot-mid-first-fifth",
          "bot-mid-first-seventh",
          "bot-mid-first-eighth",
          "bot-mid-fourth-first",
          "bot-mid-fourth-second",
          "bot-mid-fourth-fourth",
          "bot-mid-fourth-fifth",
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
          "bot-mid-first-fourth",
          "bot-mid-first-fifth",
          "bot-mid-first-seventh",
          "bot-mid-first-eighth",
          "bot-mid-fourth-first",
          "bot-mid-fourth-second",
          "bot-mid-fourth-fourth",
          "bot-mid-fourth-fifth",
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
          "bot-mid-first-fourth",
          "bot-mid-first-fifth",
          "bot-mid-first-seventh",
          "bot-mid-first-eighth",
          "bot-mid-fourth-first",
          "bot-mid-fourth-second",
          "bot-mid-fourth-fourth",
          "bot-mid-fourth-fifth",
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


function setBoxState(ids, state) {

  if (!Array.isArray(ids)) {
    ids = [ids];
  }

  ids.forEach(id => {

    const box = document.getElementById(id);

    if (!box) return;

    box.classList.remove(
      "copy-box",
      "correct-box",
      "wrong-box"
    );

    if (state !== "") {
      box.classList.add(state);
    }

  });

}

function clearAllBoxes() {
  const boxes = document.querySelectorAll(".box");

  boxes.forEach(box => {
    box.classList.remove(
      "copy-box",
      "correct-box",
      "wrong-box"
    );
  });

  const bigBox = document.querySelectorAll(".big-box");

  bigBox.forEach(box => {
    box.classList.remove(
      "copy-box",
      "correct-box",
      "wrong-box"
    );
  });

}

function setTopInstruction(text) {
  document.getElementById("top-instruction").textContent = text;
}

function setSideInstruction(text) {
  const element = document.getElementById("side-instruction");
  element.innerHTML = text;

  MathJax.typesetPromise([element]);
}

let currentStep = 1;

function renderStep(stepIndex) {

  clearAllBoxes();

  const step = steps[stepIndex];

  if (!step) return;

  setTopInstruction(step.topText);

  setSideInstruction(step.sideText);

  step.highlights.forEach(highlight => {

    setBoxState(
      highlight.ids,
      highlight.state
    );

  });

  console.log("State changed to " + stepIndex);
}

/*
                                                                    Webcam stuff                                            
*/
const video = document.createElement("video");

video.autoplay = true;
video.playsInline = true;

const canvas = document.createElement("canvas");

const ctx = canvas.getContext("2d");

async function initializeCamera() {
  const stream =
    await navigator.mediaDevices.getUserMedia({
      video: {
        width: 1920,
        height: 1080
      }
    });
  video.srcObject = stream;
  console.log("Camera initialized");
}

/*
                                                                       Main Loop
*/
let previousFrame = null;
let isProcessing = false;

const delays = {
  correct: 7000,
  wrong: 5000,
  transition: 7000,
  initialization: 5000,
  final: 200000
};

function goToStep(step) {
  currentStep = step;
  console.log("Current Step:", currentStep);
  renderStep(currentStep);
}

function delayedStep(step, delay) {
  setTimeout(() => {
    goToStep(step);
    isProcessing = false;
  }, delay);
}

function handleValidation(result, correctStep, wrongStep, retryStep) {
  if (result.state === "incomplete") {
    console.log("Still writing...");
    setTimeout(() => {
      isProcessing = false;
    }, delays.scanCooldown);
    return;
  }

  if (result.state === "correct") {
    goToStep(correctStep);
    delayedStep(correctStep + 2, delays.correct);
    return;
  }

  if (result.state === "wrong") {
    goToStep(wrongStep);
    delayedStep(retryStep, delays.wrong);
    return;
  }

  isProcessing = false;
}

async function recognitionLoop() {

  requestAnimationFrame(recognitionLoop);
  
  if (isProcessing) {
    return;
  }

  if (video.videoWidth === 0 || video.videoHeight === 0) {
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  const currentFrame = canvas.toDataURL("image/png");

  if (currentFrame === previousFrame) {
    return;
  }

  previousFrame = currentFrame;
  console.log("Paper changed");


  isProcessing = true;

  try {
    console.log("Processing image");

    const response = await fetch("http://localhost:3000/frame",
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json"
          },
          body: JSON.stringify({
            image:currentFrame,
            step: currentStep
          })
        }
      );

    const result = await response.json();
    console.log(result);

    switch(currentStep) {
      case 1:
        delayedStep(2, delays.initialization);
        break;
      case 2:
        handleValidation(result, 3, 4, 2);
        break;
      case 5:
        handleValidation(result, 6, 7, 5);
        break;
      case 8:
        handleValidation(result, 9, 10, 8);
        break;
      case 11:
        handleValidation(result, 12, 13, 11);
        break;
      case 14:
        handleValidation(result, 15, 16, 14);
        break;
      case 17:
        goToStep(18);
        delayedStep(19, delays.transition);
        break;
      case 19:
        delayedStep(1, delays.final);
        break;
      default:
        isProcessing = false;
        break;
    }
  } catch(error) {
    console.error(error);
    isProcessing = false;
  }

}

async function initializeSystem() {
  let override = 0;

  if(override == 0) {
    await initializeCamera();

    await new Promise(resolve => setTimeout(resolve, 3000));

    renderStep(currentStep);

    recognitionLoop();
  }
};

initializeSystem();

const buttonStepMap = {
  init: 1,
  /*
  L1: 2,
  L1c: 3,
  L1w: 4,
  R1: 5,
  R1c: 6,
  R1w: 7,
  */
  L2: 2,
  L2c: 3,
  L2w: 4,
  R2: 5,
  R2c: 6,
  R2w: 7,
  L3: 8,
  L3c: 9,
  L3w: 10,
  R3: 11,
  R3c: 12,
  R3w: 13,
  b4: 14,
  b4c: 15,
  b4w: 16,
  b5: 17,
  b6: 18,
  b7: 19
};


window.addEventListener("DOMContentLoaded",() => {
    Object.keys(buttonStepMap).forEach(id => {
      const button = document.getElementById(id);

      if (!button) {
        console.log("Missing button:", id);
        return;
      }

      button.addEventListener("click",() => {
          const step = buttonStepMap[id];
          goToStep(step);
        }
      );
    });
  }
);


