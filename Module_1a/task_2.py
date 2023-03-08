import cv2
import numpy as np

from srccam.calib import Calib
from srccam.camera import Camera
from srccam.load_calib import CalibReader
from srccam.point import Point3d as Point
from srccam.season_reader import SeasonReader

BLUE = (255, 0, 0)
LINE_WIDTH = 3


class ObjectOnWaysEstimator:
    """
    Задаем калиб, высоту, ширину, глубину параллелограмма и расстояние от камеры.
    """

    def __init__(self, calib_dict: dict, height: float, width: float, length: float, depth: int):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.points_3d = self.get_cube_vertices(height=height, width=width, length=length, depth=depth)

    @staticmethod
    def get_cube_vertices(height: float, width: float, length: float, depth: int) -> list:
        """
         5 ___6
        4/__7/|
         |___|/2
         0   3
        """
        points = []
        Xs = [x for sublist in [[-width / 2] * 2, [width / 2] * 2] * 2 for x in sublist]
        Ys = [depth, depth + length, depth + length, depth] * 2
        Zs = [z for sublist in [[0] * 4, [height] * 4] for z in sublist]
        for x, y, z in zip(Xs, Ys, Zs):
            points.append(Point((x, y, z)))
        return points

    def draw_object(self, img):
        points_2d = []
        for pt in self.points_3d:
            points_2d.append(self.camera.project_point_3d_to_2d(pt))

        lower_pts_2d = np.array([[list(pt) for pt in points_2d[:4]]])
        upper_pts_2d = np.array([[list(pt) for pt in points_2d[4:]]])
        for pts in [lower_pts_2d, upper_pts_2d]:
            pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, BLUE, LINE_WIDTH)

        half = 4
        for i in range(half):
            cv2.line(img, points_2d[i], points_2d[i + half], BLUE, LINE_WIDTH)

        return img


class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ["K", "D", "r", "t"]
        calib_reader = CalibReader()
        calib_reader.initialize(file_name="../data/city/leftImage.yml", param=par)
        calib_dict = calib_reader.read()
        self.obj_estimator = ObjectOnWaysEstimator(calib_dict=calib_dict, height=1.6, width=1.6, length=10, depth=8)
        return True

    def on_shot(self):
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
        self.obj_estimator.draw_object(self.frame)
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
