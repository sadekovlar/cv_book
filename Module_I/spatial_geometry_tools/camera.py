import numpy as np

from .point import Point3d
from .calib import Calib


class Camera:
    EPS = 1e-3

    def __init__(self, calib: Calib):
        self.calib = calib

    def project_point_3d_to_2d(self, point3d: Point3d):
        rotated = np.array(self.calib.r) @ np.array(point3d.vec)
        rotated = rotated - self.calib.t

        vr_cs = self.calib.cam_to_vr @ rotated
        res = self.calib.K @ vr_cs
        w = res[-1]

        res_2d = (0, 0)
        if abs(w) > Camera.EPS:
            res_2d = (int(res[0] / w), int(res[1] / w))
        return res_2d
