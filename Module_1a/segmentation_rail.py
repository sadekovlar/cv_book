from typing import Optional

import cv2
import numpy as np
from srccam.load_calib import CalibReader
from srccam.season_reader import SeasonReader
from srccam.calib import Calib
from srccam.camera import Camera
from srccam.point import Point3d as Point
import os

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class WayEstimator:
    """Класс для построения процекции пути движения ТС."""

    def __init__(self, calib_dict: dict, ways_length: int):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.left_3d_near = Point((-0.76, 5.0, 0))
        self.left_3d_far = Point((-0.76, ways_length, 0))
        self.right_3d_near = Point((0.76, 5.0, 0))
        self.right_3d_far = Point((0.76, ways_length, 0))

    def draw_way(self, img: np.array):
        left_2d_near = self.camera.project_point_3d_to_2d(self.left_3d_near)
        left_2d_far = self.camera.project_point_3d_to_2d(self.left_3d_far)
        right_2d_near = self.camera.project_point_3d_to_2d(self.right_3d_near)
        right_2d_far = self.camera.project_point_3d_to_2d(self.right_3d_far)
        cv2.line(img, right_2d_near, right_2d_far, BLACK, LINE_WIDTH)
        cv2.line(img, left_2d_near, left_2d_far, BLACK, LINE_WIDTH)

        # Получаем опорные точки изображения для рисования маски
        left_2d_near = (left_2d_near[0] - 100, left_2d_near[1] + 100)
        right_2d_near = (right_2d_near[0] + 100, right_2d_near[1] + 100)
        right_2d_far = (right_2d_far[0], right_2d_far[1] - 97)
        left_2d_far = (left_2d_far[0], left_2d_far[1] - 100)

        points = np.array([left_2d_near, left_2d_far, right_2d_far, right_2d_near])
        figure = np.zeros_like(img)
        cv2.fillPoly(figure, pts=[points], color=(255, 255, 255))
        # Сохраняем в переменную img результат побитового 'и'
        # Функция draw_way может ничего не возвращать
        img[::] = cv2.bitwise_and(img, figure)


class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: Optional[str] = None) -> bool:
        par = ["K", "D", "r", "t"]
        calib_reader = CalibReader(file_name="../data/city/leftImage.yml", param=par)
        calib_dict = calib_reader.read()
        self.way_estimator = WayEstimator(calib_dict, 10)

        return True

    def on_shot(self) -> bool:
        return True

    def on_frame(self):
        cv2.putText(
            self.frame,
            text=f"GrabMsec: {self.frame_grab_msec}",
            org=(15, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
        )
        self.way_estimator.draw_way(self.frame)
        return True

    def on_gps_frame(self) -> bool:
        shot: dict = self.shot[self._gps_name]["senseData"]
        shot["grabMsec"] = self.shot[self._gps_name]["grabMsec"]
        return True

    def on_imu_frame(self) -> bool:
        shot: dict = self.shot[self._imu_name]  # noqa: F841
        return True


if __name__ == '__main__':

    init_args = {"path_to_data_root": "../data/city/"}
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')
