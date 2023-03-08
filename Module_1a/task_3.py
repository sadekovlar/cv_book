import cv2
import numpy as np

from srccam.calib import Calib
from srccam.camera import Camera
from srccam.load_calib import CalibReader
from srccam.object3d import Object3d
from srccam.point import Point3d as Point
from srccam.season_reader import SeasonReader

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ["K", "D", "r", "t"]
        calib_reader = CalibReader()
        calib_reader.initialize(file_name="../data/city/leftImage.yml", param=par)
        calib_dict = calib_reader.read()
        calib = Calib(calib_dict)
        self.camera = Camera(calib)

        self.myCube = Object3d(Point((0, 10, 0.8)), np.array([0, 0, 0]), 1.6, 1.6, 1.6)
        self.myParall1 = Object3d(Point((-5, 12, 0)), np.array([0, 0, 0]), 3, 1, 1, GREEN, RED)
        self.myParall2 = Object3d(Point((5, 12, 0)), np.array([0, 0, 0]), 1.2, 4, 1, RED, GREEN)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(
            self.frame, f"GrabMsec: {self.frame_grab_msec}", (15, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2
        )
        # Устанавливаем повороты для фигур:
        self.myCube.add_rotation(np.array([0.0, 0.01, 0.0]))
        self.myParall1.add_rotation(np.array([0.02, 0.0, 0.00]))
        self.myParall2.add_rotation(np.array([0.0, 0.0, 0.05]))
        # Отрисовка созданных фигур:
        self.myCube.draw(self.frame, self.camera)
        self.myParall1.draw(self.frame, self.camera, drawVertex=False)
        self.myParall2.draw(self.frame, self.camera, drawEdges=False)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]["senseData"]
        shot["grabMsec"] = self.shot[self._gps_name]["grabMsec"]
        return True

    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]  # noqa: F841
        return True


if __name__ == "__main__":
    init_args = {"path_to_data_root": "../data/city/"}
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print("Done!")
