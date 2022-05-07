import cv2
import numpy as np

from Module_I.load_calib import CalibReader
from Module_I.season_reader import SeasonReader
from Module_I.spatial_geometry_tools.calib import Calib
from Module_I.spatial_geometry_tools.camera import Camera
from Module_I.spatial_geometry_tools.point import Point3d as Point


class PointsCounter:

    def __init__(self, calib_dict, point_importance):
        self.prev_points = np.array([])
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.gps = []
        self.speed = 0
        self.left_2d_far = self.camera.project_point_3d_to_2d(Point((-1.5, 12, 0)))
        self.right_2d_far = self.camera.project_point_3d_to_2d(Point((1.5, 12, 0)))
        self.A_inv = self.count_matrix_for_2d_3d_projection()
        self.points_importance = point_importance  # Уровень отсечения точек
        self.yaw = 0;

    def count_point_moving(self, Ri, t_ii1, yaw_i: int = 0):
        """
        Рассмотрение точек в плоскости земли.

        Аргументы:
        Ri -- точки на исходном кадре,
        Ti -- начальная точка пути,
        tii1 -- пройденный путь по GPS,
        yawi -- угол yaw по GPS (default 0)
        """

        Rz = np.array([  # Матрица преобразований при повороте на угол yaw из gps
            [np.cos(yaw_i), -np.sin(yaw_i), 0],
            [np.sin(yaw_i), np.cos(yaw_i), 0],
            [0, 0, 1],
        ])

        RR_i = []
        t_ii1 = np.array([[t_ii1[0]], [t_ii1[1]], [t_ii1[2]]])  # Вектор смещения
        for i in Ri:
            i.vec = Rz @ (i.vec + t_ii1)  # Смещение текущих точек
            a = self.camera.project_point_3d_to_2d(i)

            # Проверка, что погрешность не выходит за рамки массива
            if a[0] < 540 and a[1] < 540:
                RR_i.append(a)
        return RR_i

    def perv_points_projection_to_new(self, img):
        """Отрисовка точкек на изображении - старых и новых"""
        # Применение детектора Хариса для текущего изображения
        # И обрезка лишних точек
        new_Harris = self.apply_Harris(img)
        new_Harris[:self.left_2d_far[1], :] = 0
        new_Harris[:, :self.left_2d_far[0]] = 0
        new_Harris[:, self.right_2d_far[0]:] = 0

        time = 0.02 * 10 / 36  # Время для расчета пути (с переводом в м/с)

        # Работа с точками из предыдущего кадра
        if self.prev_points.size > 0:
            a = np.array(
                self.count_point_moving(
                    self.get_3d_points_on_land(self.prev_points),
                    [0, time * self.speed, 0.]))
            if a.size != 0:
                # Отрисовка рассчитанных точек синим
                img[a[:, 0], a[:, 1]] = [255, 0, 0]
            # Отрисовка предыдущих точек зеленым
            img[self.prev_points
                > self.points_importance * self.prev_points.max()] = [0, 255, 0]
        # Отрисовка текущих точек красным
        img[new_Harris > self.points_importance * new_Harris.max()] = [0, 0, 255]

        # Сохранение текущих точек для следующего кадра
        self.prev_points = new_Harris
        return img

    def get_3d_points_on_land(self, new_Harris):
        """Функция отсечения точек не принадлежащих выбранному участку земли"""
        # Получение координат точек проходящих по уровню
        points = np.argwhere(new_Harris
            > self.points_importance * new_Harris.max())
        # Получение 3d координат точек
        points = np.apply_along_axis(
            self.reproject_point_2d_to_3d_on_floor, 1, points)
        return points

    @staticmethod
    def get_A_from_P_on_floor(P: np.ndarray) -> np.ndarray:
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

    def apply_Harris(self, img):
        """
        Детектор Харриса.

        Принимает BGR изображение,
        возвращает изображение в GrayScale с отметкой углов.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.15)
        dst = cv2.dilate(dst, None)
        return dst


class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.counter = PointsCounter(calib_dict, 0.20)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.counter.perv_points_projection_to_new(self.frame)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        self.counter.speed = self.shot[self._gps_name]['senseData']['speed']
        self.counter.yaw = self.shot[self._gps_name]['senseData']['yaw']
        return True

    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]
        return True


if __name__ == '__main__':
    video_name = 'klt.427.003.mp4'
    init_args = {
        'path_to_data_root': '../data/tram/'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')
