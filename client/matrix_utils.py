import re
import numpy as np

class VariableMatrix:
    def __init__(self):
        """
        Maintains a base (Identity) matrix and a custom transformation target matrix.
        Allows real-time interpolation between them for fluid UI transitions.
        """
        self.base = [[1.0, 0.0], 
                     [0.0, 1.0]]
        
        self.target = [[1.0, 0.0], 
                       [0.0, 1.0]]
        
        self.target_vector = None     # Stores the vector to find (e.g., [2.0, 1.0])
        self.expected_output = None   # Stores the correct answer vector (e.g., [-16.0, 15.0])
        
    def set_target(self, a, b, c, d):
        """
        Manually adjust the target 2x2 matrix elements:
        [ a  b ]
        [ c  d ]
        """
        self.target = [[float(a), float(b)], 
                       [float(c), float(d)]]

    def parse_from_text(self, math_string):
        """
        Extracts values from strings containing matrix notations like '[2/-1] + [1/1]'
        and sets them as the target matrix.
        """
        matches = re.findall(r'\[(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)\]', math_string)
        if len(matches) >= 2:
            a, c = map(float, matches[0])
            b, d = map(float, matches[1])
            self.set_target(a, b, c, d)
            return True
        elif len(matches) == 1:
            mx, my = map(float, matches[0])
            self.set_target(mx, 0.0, 0.0, my)
            return True
        return False

    def parse_from_qr(self, qr_data):
        """
        Parses QR string format: "x1,y1; tx1,ty1|x2,y2; tx2,ty2|target_x,target_y; ans_x,ans_y"
        1. Solves the 2x2 matrix transformation matrix.
        2. Saves the target vector and correct answer for downstream OCR verification.
        """
        try:
            pairs = qr_data.split("|")
            if len(pairs) < 2:
                print("QR Data Error: Missing required vector pairs.")
                return False
                
            # --- 1. Extract and solve the matrix transformation ---
            inputs = []
            outputs = []
            for pair in pairs[:2]: 
                input_str, output_str = pair.split(";")
                inputs.append([float(x) for x in input_str.split(",")])
                outputs.append([float(x) for x in output_str.split(",")])
            
            X = np.array(inputs).T
            Y = np.array(outputs).T
            M = np.linalg.solve(X.T, Y.T).T
            
            self.set_target(M[0][0], M[0][1], M[1][0], M[1][1])
            
            # --- 2. Extract and store OCR target data if present ---
            if len(pairs) >= 3:
                target_str, answer_str = pairs[2].split(";")
                self.target_vector = [float(x) for x in target_str.split(",")]
                self.expected_output = [float(x) for x in answer_str.split(",")]
                print(f"OCR targets locked -> Find L{self.target_vector} = {self.expected_output}")
            
            return True
            
        except Exception as e:
            print(f"Error parsing QR data: {e}")
            return False

    def get_interpolated_matrix(self, progress):
        """
        Linearly interpolates (lerps) between the base matrix and target matrix.
        """
        t = max(0.0, min(1.0, progress))
        a = self.base[0][0] + (self.target[0][0] - self.base[0][0]) * t
        b = self.base[0][1] + (self.target[0][1] - self.base[0][1]) * t
        c = self.base[1][0] + (self.target[1][0] - self.base[1][0]) * t
        d = self.base[1][1] + (self.target[1][1] - self.base[1][1]) * t
        return [[a, b], [c, d]]

    def transform_shape(self, vertices, progress=0.0):
        """
        Transforms a whole set of coordinates [[x1, y1], [x2, y2], ...] 
        """
        matrix = self.get_interpolated_matrix(progress)
        transformed = []
        for x, y in vertices:
            tx = matrix[0][0] * x + matrix[0][1] * y
            ty = matrix[1][0] * x + matrix[1][1] * y
            transformed.append([tx, ty])
        return transformed
        
    @staticmethod
    def verify_ocr_solution(scanned_ocr_text, expected_output):
        """
        Extracts numbers from OCR text and verifies if the final pair matches 
        the expected output vector.
        """
        if not expected_output:
            return False
            
        numbers = re.findall(r'-?\d+(?:\.\d+)?', scanned_ocr_text)
        
        if len(numbers) < 2:
            print("OCR Error: Could not find enough numbers to form a vector coordinate.")
            return False
            
        detected_final_x = float(numbers[-2])
        detected_final_y = float(numbers[-1])
        
        target_x, target_y = expected_output[0], expected_output[1]
        
        if abs(detected_final_x - target_x) < 0.01 and abs(detected_final_y - target_y) < 0.01:
            print(f"Match found! Detected: [{detected_final_x}, {detected_final_y}]")
            return True
        else:
            print(f"Mismatched Answer. Expected: {expected_output}, Got: [{detected_final_x}, {detected_final_y}]")
            return False


if __name__ == "__main__":
    expected = [-16.0, 15.0]

    print("Running validation tests...")
    VariableMatrix.verify_ocr_solution("The answer is -16 15", expected)      
    VariableMatrix.verify_ocr_solution("Result: (-16, 15)", expected)        
    VariableMatrix.verify_ocr_solution("= [ -16 , 15 ]", expected)           
    VariableMatrix.verify_ocr_solution("Step 3: [4 10] Final: -16\n15", expected)