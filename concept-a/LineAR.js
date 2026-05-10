/*

function setBoxState(ids, state) {

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
]

function setTopInstruction(text) {
  document.getElementById("top-instruction").textContent = text;
}

function setSideInstruction() {
    document.getElementById("side-instruction").textContext = text;
}

const steps = [
  {
    topText: topInstructionText[0],

    sideText: sideInstructionText[0],

    highlights: [
      {
        id: "mid-mid-fifth",
        state: "copy-box"
      }
    ]
  },

  {
    topText: topInstructionText[1],

    sideText: sideInstructionText[1],

    highlights: [
      {
        id: "mid-left-first",
        state: "copy-box"
      },

      {
        id: "mid-left-second",
        state: "copy-box"
      }
    ]
  },

  {
    topText: topInstructionText[2],

    sideText: sideInstructionText[2],

    highlights: [
      {
        id: "bot-right-third",
        state: "correct-box"
      },

      {
        id: "bot-right-fourth",
        state: "wrong-box"
      }
    ]
  }
];

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