import re

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

    def get_interpolated_matrix(self, progress):
        """
        Linearly interpolates (lerps) between the base matrix and target matrix.
        progress: float from 0.0 (fully base/untransformed) to 1.0 (fully transformed).
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
        based on the transformation's animation progress.
        """
        matrix = self.get_interpolated_matrix(progress)
        transformed = []
        for x, y in vertices:
            tx = matrix[0][0] * x + matrix[0][1] * y
            ty = matrix[1][0] * x + matrix[1][1] * y
            transformed.append([tx, ty])
        return transformed