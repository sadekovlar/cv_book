import numpy as np


class Point3d:
    def __init__(self, coords: tuple):
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]
        self.vec = np.array([[self.x],[self.y],[self.z]])