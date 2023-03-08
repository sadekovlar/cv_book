from typing import Any, Optional

import cv2
import numpy as np

from srccam.calib import Calib
from srccam.camera import Camera
from srccam.load_calib import CalibReader
from srccam.point import Point3d as Point
from srccam.season_reader import SeasonReader

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class WayEstimator:
    """Класс для построения процекции пути движения ТС."""

    def __init__(self, calib_dict: dict[str, Any], ways_length: int):
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
        cv2.line(img, pt1=right_2d_near, pt2=right_2d_far, color=BLACK, thickness=LINE_WIDTH)
        cv2.line(img, pt1=left_2d_near, pt2=left_2d_far, color=BLACK, thickness=LINE_WIDTH)
        return img


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


if __name__ == "__main__":
    init_args = {"path_to_data_root": "../data/city/"}
    reader = Reader()
    reader.initialize(**init_args)
    reader.run()
    print("Done!")
