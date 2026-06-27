STEP_SEQUENCE = ["firstStepLeft", "firstStepRight", "secondStepLeft", "secondStepRight", "thirdStep", "complete"]

PROJECTION_REGIONS = {
    
}

CROP_REGIONS = {
    "secondStepLeft": {
        "one": {"top": 400, "left": 200, "width": 90, "height": 190},
        "two": {"top": 380, "left": 350, "width": 90, "height": 190}
    },
    "secondStepRight": {
        "one": {"top": 400, "left": 500, "width": 90, "height": 90},
        "two": {"top": 380, "left": 635, "width": 90, "height": 190}
    },
    "thirdStepLeft": {
        "one": {"top": 600, "left": 300, "width": 90, "height": 190}
    },
    "thirdStepRight": {
        "one": {"top": 600, "left": 450, "width": 90, "height": 190}
    }
}

STEP_GUIDANCE = {
    "secondStepLeft": {
        "title": "Vector Addition Property (Left Component)",
        "desc": "Isolate the linear combination matrix scaling variables for step (i). Write your vector factors in the dashed zones.",
        "math": "[ a ] + [ c ] = [ a + c ]"
    },
    "secondStepRight": {
        "title": "Vector Addition Property (Right Component)",
        "desc": "Compute the right scalar transformation parameters to compute the balanced equation values.",
        "math": "L(v) = c1*L(v1) + c2*L(v2)"
    },
    "thirdStepLeft": {
        "title": "Linear Mapping Composition",
        "desc": "Multiply out the linear map values to transform your standard basis coordinates into spatial projections.",
        "math": "A * x = b"
    },
    "thirdStepRight": {
        "title": "Final Combination Calculation",
        "desc": "Sum up the remaining transformed vector outputs to solve for the target vector mapping.",
        "math": "Final Vector Mapping Check"
    }
}