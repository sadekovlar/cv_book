import numpy as np

class Vector3:
    """Трёхмерный вектор (X, Y, Z) с набором вспомогательных функций"""
    def __init__(self, x, y : float = None, z : float = None):
        self.x, self.y, self.z = tuple(i for i in x[:3]) if (isinstance(x, tuple) or isinstance(x, list)) else (x, y, z)
    def __str__(self): return "vector3(" + str((self.x, self.y, self.z)) + ")"

    @property
    def vec(self): return np.array([ [self.x], [self.y], [self.z] ])