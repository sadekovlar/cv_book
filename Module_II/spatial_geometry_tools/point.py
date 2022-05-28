import numpy as np


class Point3d:
    """Трёхмерная точка с координатами (x, y, z)"""
    def __init__(self, coordinates: tuple):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        self.vec = np.array([[self.x], [self.y], [self.z]])
