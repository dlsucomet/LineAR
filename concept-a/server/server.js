const express = require("express");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const { exec } = require("child_process");
const util = require("util");

const execPromise = util.promisify(exec);
const app = express();

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.static(path.join(__dirname, "../client")));
app.use("/captures", express.static(path.join(__dirname, "../captures")));

let currentStep = "secondStepLeft"; 

const expectedAnswers = {
  secondStepLeft:  { one: ["3"],    two: ["4", "-1", "-4", "1"] },
  secondStepRight: { one: ["-2", "2"],   two: ["-1", "2"] },
  thirdStepLeft:   { one: ["12", "1", "2", "3", "-3"]},
  thirdStepRight:  { one: ["2", "-4", "-2", "4"]},
  fourthStep:      { one: ["14", "7", "-7"]}
};

app.post("/frame", async (req, res) => {
    try {
        const image = req.body.image;
        if (!image) return res.status(400).json({ success: false, error: "Missing image payload" });

        const base64 = image.replace(/^data:image\/\w+;base64,/, "");
        const filename = `frame-${Date.now()}.png`;
        const filepath = path.join(__dirname, "../captures", filename);
        const warpedPath = path.join(__dirname, "../captures", `warped-${filename}`);

        fs.writeFileSync(filepath, Buffer.from(base64, "base64"));

        let pythonData;

        try {
            const flaskResponse = await fetch("http://127.0.0.1:5000/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image_path: filepath,
                    warped_out_path: warpedPath,
                    current_step: currentStep
                })
            });

            // Extract the json data matrix output directly
            pythonData = await flaskResponse.json();

        } catch (error) {
            console.error("Pipeline Communication Error:", error.message || error);
            return res.status(500).json({ 
                success: false, 
                error: "Flask API pipeline communication failed" 
            });
        }

        if (pythonData.status === "paper_not_found") {
            console.log("ArUco Verification Failure: Markers could not be located.");
            return res.json({ success: true, state: "paper_not_found" });
        }

        if (pythonData.status === "error") {
            console.error("Python processing runtime error:", pythonData.message);
            return res.status(500).json({ success: false, error: pythonData.message });
        }

        console.log(`Python execution succeeded for ${currentStep}. Validating answers via OR rules...`);
        
        const recognizedResults = { [currentStep]: pythonData.recognized };
        let state = "correct";
        const expectedStep = expectedAnswers[currentStep];
        const recognizedStep = recognizedResults[currentStep];

        for (const regionName in expectedStep) {
            const expectedValues = expectedStep[regionName];
            const recognizedValues = recognizedStep[regionName] || [];

            console.log(`Region [${regionName}] | Expected: ${expectedValues} | Found: ${recognizedValues}`);

            if (recognizedValues.length === 0) {
                state = "incomplete";
                break;
            }

            const containsValidAnswer = expectedValues.some(val => {
                const escapedVal = val.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
                
                const regex = new RegExp(`${escapedVal}`);
                return recognizedValues.some(ocrStr => regex.test(ocrStr));
            });

            if (!containsValidAnswer) {
                state = "wrong";
                break;
            }
        }

        if (state === "correct") {
            console.log(`Step ${currentStep} successfully cleared!`);
            if (currentStep === "secondStepLeft") {
                currentStep = "secondStepRight";
            } else if (currentStep === "secondStepRight") {
                currentStep = "thirdStepLeft";
            } else if (currentStep === "thirdStepLeft") {
                currentStep = "thirdStepRight";
            } else if (currentStep === "thirdStepRight") {
                currentStep = "fourthStep";
            } else if (currentStep === "fourthStep") {
                console.log("Sheet completed!");
            }
            
            console.log(`Next frame payload will request Python to look at: ${currentStep}`);
        }

        return res.json({ success: true, filename, currentStep, recognizedResults, state });

    } catch (error) {
        console.error("Server Pipeline Error:", error);
        return res.status(500).json({ success: false, error: "Internal Server Error" });
    }
});

app.listen(3000, () => {
    console.log("Server running on port 3000");
});