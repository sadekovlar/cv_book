import math

import cv2
from Module_I.season_reader import SeasonReader
from Module_I.load_calib import CalibReader
import numpy as np


class Horizon:
    """Построение линии горизонта на кадре."""

    @staticmethod
    def find_intersection(line_1, line_2):
        x_diff = (line_1[0] - line_1[2], line_2[0] - line_2[2])
        y_diff = (line_1[1] - line_1[3], line_2[1] - line_2[3])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(x_diff, y_diff)
        if div == 0:
            raise Exception('lines do not intersect')

        det_line_1 = det((line_1[0], line_1[1]), (line_1[2], line_1[3]))
        det_line_2 = det((line_2[0], line_2[1]), (line_2[2], line_2[3]))
        d = (det_line_1, det_line_2)

        x = det(d, x_diff) / div
        y = det(d, y_diff) / div
        return math.ceil(x), math.ceil(y)

    @staticmethod
    def is_belong_to_line(x1, y1, x2, y2, x3, y3):
        is_x_same = x1 == x2 == x3
        is_y_same = y1 == y2 == y3
        if is_x_same or is_y_same:
            return True
        return (x1 - x3) * (y2 - y1) == (x2 - x1) * (y1 - y3)

    def get_intersection_points(self, roads, img):
        intersection_points = {}
        for i in roads:
            for j in roads:
                if i.all() == j.all():
                    continue
                # Поиск точки пересечения линий
                x, y = self.find_intersection(i, j)
                # Отсеивание точек, не принадлежащих изображению
                is_x_belong_to_image = 0 <= x < img.shape[1]
                is_y_belong_to_image = 0 <= y < img.shape[0]
                if not (is_x_belong_to_image and is_y_belong_to_image):
                    continue
                # Подсчёт количества прямых, пересекающихся в этой точке
                intersection_points[(x, y)] =\
                    intersection_points.setdefault((x, y), 0) + 1
                # Отрисовка линий
                cv2.line(img, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 3)
                cv2.line(img, (j[0], j[1]), (j[2], j[3]), (255, 0, 0), 3)
        return intersection_points

    def find_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Выделение границ детектором Канни
        edges = cv2.Canny(gray, 50, 200)

        # Поиск линий с помощью Хаффа
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            200,
            minLineLength=10,
            maxLineGap=250
        ).squeeze()

        # Отсеивание линий, не принадлежащих рельсам
        left_threshold = 300
        right_threshold = 650
        roads = lines[(lines[:, 0] < right_threshold) &\
                      (lines[:, 0] > left_threshold) &\
                      (lines[:, 2] < right_threshold) &\
                      (lines[:, 2] > 300)]

        # Нахождение точек пересечения линий с линией потенциального горизонта
        intersection_points = self.get_intersection_points(roads, img)

        if not len(intersection_points):
            return

        # Поиск точки, в которой больше всего пересечений
        point = max(intersection_points, key=intersection_points.get)
        # Отрисовка точки, где пересеклось больше всего линий
        cv2.circle(img, (point[0], point[1]), 5, (0, 255, 0))
        # Отрисовка линии горизонта
        cv2.line(img, (0, point[1]), (img.shape[1], point[1]), (0, 255, 0), 2)


class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        self.horizont = Horizon()

        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.horizont.find_lines(self.frame)
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
        'path_to_data_root': '../data/tram/'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')