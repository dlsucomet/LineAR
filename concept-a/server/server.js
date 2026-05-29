const express = require("express");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const sharp = require("sharp");
const Tesseract = require("tesseract.js");
const app = express();

app.use(cors());

app.use(express.json({
  limit: "50mb"
}));

/* Change values here according to setup */
const cropRegions = {
  firstStepLeft: {
    one: {
      top: 400,
      left: 840,
      width: 70,
      height: 70
    },

    two: {
      top: 400,
      left: 910,
      width: 70,
      height: 70
    },

    three: {
      top: 400,
      left: 910,
      width: 70,
      height: 70
    }
  },

  firstStepRight: {
    one: {
      top: 200,
      left: 900,
      width: 300,
      height: 300
    },

    two: {
      top: 200,
      left: 200,
      width: 300,
      height: 300
    },

    three: {
      top: 400,
      left: 910,
      width: 70,
      height: 70
    }
  },

  secondStepLeft: {
    one: {
      top: 550,
      left: 200,
      width: 300,
      height: 300
    }
  },

  secondStepRight: {
    one: {
      top: 550,
      left: 900,
      width: 300,
      height: 300
    }
  },

  thirdStepLeft: {
    one: {
      top: 900,
      left: 200,
      width: 300,
      height: 300
    }
  },

  thirdStepRight: {
    one: {
      top: 900,
      left: 900,
      width: 300,
      height: 300
    }
  },

  fourthStep: {
    one: {
      top: 200,
      left: 550,
      width: 300,
      height: 300
    }
  }
};

let currentStep = "firstStepLeft";

/* Hardcoded answers for prototype */
const expectedAnswers = {
  firstStepLeft: {
    one: ["3"],
    two: ["1"],
    three: ["2"]
  },

  firstStepRight: {
    one: ["2"],
    two: ["0"],
    three: ["1"]
  },

  secondStepLeft: {
    one: ["4"],
    two: ["-1"]
  },

  secondStepRight: {
    one: ["-2"],
    two: ["-1"]
  },

  thirdStepLeft: {
    one: ["12"],
    two: ["-3"]
  },

  thirdStepRight: {
    one: ["-2"],
    two: ["4"]
  },

  fourthStep: {
    one: ["10"],
    two: ["1"]
  }

};

const stepOrder = [
  "firstStepLeft",
  "firstStepRight",
  "secondStepLeft",
  "secondStepRight",
  "thirdStepLeft",
  "thirdStepRight",
  "fourthStep"
];


app.post("/frame", async (req, res) => {
    try {
        const image = req.body.image;
        const base64 = image.replace(/^data:image\/png;base64,/,"");
        const filename = `frame-${Date.now()}.png`;
        const filepath = path.join(__dirname, "../captures", filename);

        const imageBuffer = Buffer.from(base64,"base64");

        fs.writeFileSync(filepath,imageBuffer);

        console.log("Saved:", filename);


        const currentRegions = cropRegions[currentStep];
        const recognizedResults = {};
        recognizedResults[currentStep] = {};

        for (const regionName in currentRegions) {
          const region = currentRegions[regionName];
          const croppedPath = path.join(__dirname, "../captures", `${currentStep}-${regionName}-${filename}`);

          await sharp(imageBuffer)
            .extract({
              left: region.left,
              top: region.top,
              width: region.width,
              height: region.height
            })
            .resize(800, 800)
            .toFile(croppedPath);

          console.log("Cropped:", croppedPath);

          const result = await Tesseract.recognize(
              croppedPath,
              "eng",
              {
                config: {
                  tessedit_char_whitelist: "0123456789-",
                  tessedit_pageseg_mode: 7
                }
              }
            );

          const rawText = result.data.text;
          const recognizedValues =
            rawText
              .split(/\s+/)
              .map(
                value =>
                  value.replace(
                    /[^0-9-]/g,
                    ""
                  )
              )
              .filter(
                value =>
                  value !== ""
              );

          console.log(`${currentStep} ${regionName}`);
          console.log("OCR:",rawText);
          console.log("Recognized:", recognizedValues);

          recognizedResults[currentStep][regionName] = recognizedValues;
        }

        let state = "correct";

        const expectedStep = expectedAnswers[currentStep];
        const recognizedStep = recognizedResults[currentStep];

        for (const regionName in expectedStep) {
          const expectedValues = expectedStep[regionName];
          const recognizedValues = recognizedStep[regionName];

          if (recognizedValues.length < expectedValues.length) {
            state = "incomplete";
            console.log(`${regionName} incomplete`);
            break;
          }

          const matches = JSON.stringify(expectedValues) === JSON.stringify(recognizedValues);

          if (!matches) {
            state = "wrong";
            console.log(`${regionName} incorrect`);
            break;
          }
          console.log(`${regionName} correct`);
        }

        res.json({success: true, filename, currentStep, recognizedResults, state});

        } catch(error) {
            console.error(error);
            res.status(500).json({success: false});
        }
    }
);

app.listen(3000, () => {
    console.log("Server running on port 3000");
  }
);