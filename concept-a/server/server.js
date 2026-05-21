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
const cropRegions = [
    {
        firstStepLeft: {
            one: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
            two: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
        },
    },

    {
        firstStepRight: {
            one: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
            two: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
        },
    },

    {
        secondStepLeft: {
            one: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
        },
    },

    {
        secondStepRight: {
            one: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
        },
    },

    {
        thirdStepLeft: {
            one: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
        },
    },

    {
        thirdStepRight: {
            one: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
        },
    },

    {
        fourthStep: {
            one: {
                top: 200,
                left: 200,
                width: 300,
                height: 300,
            },
        },
    },
];

/* Hardcoded answers for prototype */
const expectedAnswers = {
  firstStep: {
    leftMatrix: [
      "3",
      "1",
      "2"
    ],
    rightMatrix: [
      "4",
      "-1"
    ]
  },

  secondStep: {
    left: [
      "3",
      "1",
      "2"
    ],
    right: [
      "4",
      "-1"
    ]
  },

  thirdStep: {
    left: [
      "3",
      "1",
      "2"
    ],
    right: [
      "4",
      "-1"
    ]
  },

  fourthStep: {
    left: [
      "3",
      "1",
      "2"
    ],
  }
};


app.post("/frame", async (req, res) => {
    try {
        const image = req.body.image;
        const base64 = image.replace(/^data:image\/png;base64,/,"");
        const filename = `frame-${Date.now()}.png`;
        const filepath = path.join(__dirname, "../captures", filename);

        const imageBuffer = Buffer.from(base64,"base64");

        fs.writeFileSync(filepath,imageBuffer);

        console.log("Saved:", filename);

        const croppedPath = path.join(__dirname, "../captures", `cropped-${filename}`);

        const metadata = await sharp(imageBuffer).metadata();
        const imageWidth = metadata.width;
        const imageHeight = metadata.height;

        await sharp(imageBuffer)
        .extract({
            left: 300,
            top: 0,
            width: 500,
            height: 500
            })
            .sharpen()
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
        const replacedText = rawText.replace(/[^0-9-]/g,"");

        console.log("OCR:", rawText);
        console.log("Trimmed:", replacedText);

        let state = "correct";

        /*
            Example validation
            state = "correct";
        */
        
        res.json({success: true, filename, rawText, state});
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