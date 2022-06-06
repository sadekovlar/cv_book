import math

import cv2
import numpy as np

from Module_I.load_calib import CalibReader
from Module_I.season_reader import SeasonReader


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
    def is_point_belong_to_image(x, y, img):
        is_x_belong_to_image = 0 <= x < img.shape[1]
        is_y_belong_to_image = 0 <= y < img.shape[0]
        return is_x_belong_to_image and is_y_belong_to_image

    def get_intersection_points(self, roads, img):
        intersection_points = {}
        for i in range(0, len(roads) - 1):
            for j in range(i + 1, len(roads)):
                elem_i, elem_j = roads[i], roads[j]
                if elem_i.all() == elem_j.all():
                    continue
                # Поиск точки пересечения линий
                x, y = self.find_intersection(elem_i, elem_j)
                # Отсеивание точек, не принадлежащих изображению
                if not self.is_point_belong_to_image(x, y, img):
                    continue
                # Подсчёт количества прямых, пересекающихся в этой точке
                intersection_points[(x, y)] = \
                    intersection_points.setdefault((x, y), 0) + 1
                # Отрисовка линий
                cv2.line(img, (elem_i[0], elem_i[1]), (elem_i[2], elem_i[3]), (255, 0, 0), 3)
                cv2.line(img, (elem_j[0], elem_j[1]), (elem_j[2], elem_j[3]), (255, 0, 0), 3)
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
        roads = lines[(lines[:, 0] < right_threshold) &
                      (lines[:, 0] > left_threshold) &
                      (lines[:, 2] < right_threshold) &
                      (lines[:, 2] > left_threshold)]

        # Массив точек пересечения линий, оставшихся после отсеивания
        intersection_points = self.get_intersection_points(roads, img)

        if not intersection_points:
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
        self.horizon = Horizon()

        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.horizon.find_lines(self.frame)
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
