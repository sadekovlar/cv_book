import math

import cv2
import numpy as np

from srccam.load_calib import CalibReader
from srccam.season_reader import SeasonReader
from srccam.calib import Calib
from srccam.point import Point3d as Point

class Vertical:

    def __init__(self, calib_dict):
        self.calib = Calib(calib_dict)
        self.A_inv = self.count_matrix_for_2d_3d_projection()


    def reproject_point_2d_to_3d_on_floor(self, point2d: None):
        """
        Проецирование 2d точек в 3d координаты.

        (Только для точек, принадлежащих плоскости земли.)
        """
        if point2d is None:
            point2d = []
        h = 0  # Процекция по земле, следовательно высота нулевая
        p_ = self.A_inv @ Point((point2d[0], point2d[1], 1)).vec
        error = abs(self.calib.t)/2 # Погрешность перевода между С.К.
        x = p_[0] / p_[2] + error[0]
        y = p_[1] / p_[2] + error[1]
        z = h
        return Point((x, y, z))

    def count_matrix_for_2d_3d_projection(self):
        """Предрасчет матриц, необходимых для проецирования точек из 2d в 3d"""
        R = self.calib.cam_to_vr @ self.calib.r  # Изменение порядка осей
        affine_matrix = np.concatenate((R, -R @ self.calib.t), 1)
        P = self.calib.K @ affine_matrix
        A = self.get_A_from_P_on_floor(P)
        A_inv = np.linalg.inv(A)
        return A_inv

    def get_A_from_P_on_floor(self, P: np.ndarray) -> np.ndarray:
        """
        Значения первых двух столбцов неизменны,
        последние два столбца складываются
        """
        h = 0  # Процекция по земле, следовательно высота нулевая
        A = np.zeros((3, 3))
        A[0, 0], A[0, 1], A[0, 2] = P[0, 0], P[0, 1], h * P[0, 2] + P[0, 3]
        A[1, 0], A[1, 1], A[1, 2] = P[1, 0], P[1, 1], h * P[1, 2] + P[1, 3]
        A[2, 0], A[2, 1], A[2, 2] = P[2, 0], P[2, 1], h * P[2, 2] + P[2, 3]
        return A

    def find_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Выделение границ детектором Канни
        edges = cv2.Canny(gray, 0, 500)

        # Поиск линий с помощью Хаффа
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            100,
            minLineLength=10,
            maxLineGap=250
        ).squeeze()


        map_points = []
        for x1, y1, x2, y2 in lines:

            if x1 == x2:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                a = self.reproject_point_2d_to_3d_on_floor([x1, y1])
                print(a.x, a.y, a.z)
                map_points.append([a.x, a.y])

        #print(map_points)
        #print(len(map_points))


        return map_points



class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.vertical = Vertical(calib_dict)
        self.img = np.zeros((500, 500, 1), np.uint8)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        map_points = self.vertical.find_lines(self.frame)
        for point in map_points:
            self.img[int(point[0]), int(point[1])] = 255
        cv2.imshow('map', self.img)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        return True

    def on_imu_frame(self):
        return True


if __name__ == '__main__':
    init_args = {
        'path_to_data_root': '../data/tram/'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')