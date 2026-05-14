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

        await sharp(imageBuffer)
            .extract({
            left: 100,
            top: 200,
            width: 300,
            height: 150
            })
            .toFile(croppedPath);

        console.log("Cropped:", croppedPath);

        const result = await Tesseract.recognize(croppedPath, "eng");

        const recognizedText = result.data.text;
        console.log("OCR:", recognizedText);


        let state = "wrong";

        /*
            Example validation
        */

        
        state = "correct";
        

        res.json({success: true, filename, recognizedText, state});
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