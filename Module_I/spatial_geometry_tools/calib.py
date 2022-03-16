import numpy as np


class Calib:
    """
    Параметры из файла yaml.

    K - интринсики, D - дисторсия, r - поворот, t - смещение.
    """
    def __init__(self, calib_dict):
        self.cam_to_vr = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        self.K = calib_dict['K']
        self.D = calib_dict['D']
        roll, pitch, yaw = calib_dict['r']
        self.r = self.rotation_matrix_from([roll, pitch, yaw]).T
        self.t = calib_dict['t']

    @staticmethod
    def rotation_matrix_from(angles):
        sinuses = np.sin(angles)
        cosines = np.cos(angles)
        Rx = np.array([
            [1, 0, 0],
            [0, cosines[0], -sinuses[0]],
            [0, sinuses[0], cosines[0]],
        ], dtype=object)
        Ry = np.array([
            [cosines[1], 0, sinuses[1]],
            [0, 1, 0],
            [-sinuses[1], 0, cosines[1]],
        ], dtype=object)
        Rz = np.array([
            [cosines[2], -sinuses[2], 0],
            [sinuses[2], cosines[2], 0],
            [0, 0, 1],
        ], dtype=object)

        return Rz @ Ry @ Rx
