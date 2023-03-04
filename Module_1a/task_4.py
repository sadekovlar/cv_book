from srccam.object3d import Object3d
from srccam.point import Point3d

import cv2
import numpy as np

from srccam.load_calib import CalibReader
from srccam.season_reader import SeasonReader

from srccam.calib import Calib
from srccam.camera import Camera
from srccam.point import Point3d as Point

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class Quaternion:
    def __init__(self, w=0, x=0, y=0, z=1):
        self._w = w
        self._vx = x
        self._vy = y
        self._vz = z
        self._rotation_vector = [x, y, z]

    def normalized(self):
        x = self.vx()
        y = self.vy()
        z = self.vz()
        quat = Quaternion(self.w(), x, y, z)

        length = np.sqrt(x * x + y * y + z * z)
        if length > 0:
            np.divide(quat._rotation_vector, length)

        return quat

    @staticmethod
    def make_quaternion(angle, vx=0, vy=0, vz=1):
        return Quaternion(
            np.cos(angle / 2),
            vx * np.sin(angle / 2),
            vy * np.sin(angle / 2),
            vz * np.sin(angle / 2),
        )

    def inversed(self):
        quat = self.normalized()
        return Quaternion(
            +quat.w(),
            -quat.vx(),
            -quat.vy(),
            -quat.vz(),
        )

    def vx(self):
        """Параметр кватерниона: x-составляющая вектора поворота"""
        return self._vx

    def vy(self):
        """Параметр кватерниона: y-составляющая вектора поворота"""
        return self._vy

    def vz(self):
        """Параметр кватерниона: z-составляющая вектора поворота"""
        return self._vz

    def w(self):
        """Параметр кватерниона: косинус половинного угла"""
        return self._w

    def multiply(self, quat):

        w1, w2 = self.w(), quat.w()
        x1, x2 = self.vx(), quat.vx()
        y1, y2 = self.vy(), quat.vy()
        z1, z2 = self.vz(), quat.vz()

        res = Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            x1*w2 + w1*x2 + y1*z2 - z1*y2,
            w1*y2 + w2*y1 + z1*x2 - x1*z2,
            z1*w2 + w1*z2 + x1*y2 - y1*x2
        )
        return res


class QuaternionRotate:

    @staticmethod
    def quat_rotate(cube: Object3d,
                    x_angle: float,
                    y_angle: float,
                    z_angle: float,
                ):
        quat_x = Quaternion.make_quaternion(x_angle, 1, 0, 0).normalized()
        quat_y = Quaternion.make_quaternion(y_angle, 0, 1, 0).normalized()
        quat_z = Quaternion.make_quaternion(z_angle, 0, 0, 1).normalized()

        quat = quat_x.multiply(quat_y).multiply(quat_z).normalized()

        radiusVectors = []
        rotatedVectors = []
        center = [cube.pos.x, cube.pos.y, cube.pos.z]
        for point in cube.points:
            radiusVectors.append(np.subtract(center, point)[0])

        for vector in radiusVectors:
            qVector = Quaternion(0, vector[0], vector[1], vector[2])
            final_quat = quat.multiply(qVector).multiply(quat.inversed())
            rotatedVector = [final_quat.vx(),
                             final_quat.vy(),
                             final_quat.vz()]

            rotatedVectors.append(rotatedVector)

        for i in range(len(cube.points)):
            cube.points[i] = [np.add(center, rotatedVectors[i])]


BLACK = (0, 0, 0)
BLUE = (100, 0, 0)
GREEN = (100, 200, 100)
RED = (0, 0, 200)
LINE_WIDTH = 5


class Reader(SeasonReader):
    """Обработка видеопотока."""
    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/city/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        calib = Calib(calib_dict)
        self.camera = Camera(calib)
        self.cube1 = Object3d(Point((0, 10, 1)), np.array([0, 0, 0]), 1, 3, 2, GREEN, BLUE)
        self.cube2 = Object3d(Point((-4.5, 10, 1)), np.array([0, 0, 0]), 1, 3, 2, GREEN, RED)
        self.cube3 = Object3d(Point((4.5, 10, 1)), np.array([0, 0, 0]), 1, 3, 2, GREEN, BLACK)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        QuaternionRotate.quat_rotate(self.cube1, 0.03, 0.02, 0.05)
        QuaternionRotate.quat_rotate(self.cube2, 0.0, 0.0, 0.05)
        QuaternionRotate.quat_rotate(self.cube3, 0.03, 0.0, 0.0)

        self.cube1.draw(self.frame, self.camera, drawVertex=True)
        self.cube2.draw(self.frame, self.camera, drawVertex=True)
        self.cube3.draw(self.frame, self.camera, drawVertex=True)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        return True

    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]
        return True


if __name__ == '__main__':
    init_args = {
        'path_to_data_root': '../data/city/'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')







