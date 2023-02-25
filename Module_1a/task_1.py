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
        return img


class Reader(SeasonReader):
    """Обработка видеопотока."""
    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/city/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.way_estimator = WayEstimator(calib_dict, 10)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.way_estimator.draw_way(self.frame)
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
