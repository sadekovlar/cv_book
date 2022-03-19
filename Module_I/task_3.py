import cv2
import numpy as np
from load_calib import CalibReader
from season_reader import SeasonReader
from spatial_geometry_tools.calib import Calib
from spatial_geometry_tools.camera import Camera
from spatial_geometry_tools.point import Point3d as Point


BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class WayEstimator:
    """Класс для построения процекции пути движения ТС."""
    def __init__(self, calib_dict: np.array, ways_length: int):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.left_3d_near = Point((-0.8, 1, 0))
        self.left_3d_far = Point((-0.8, ways_length-1, 0))
        self.right_3d_near = Point((0.8, 1, 0))
        self.right_3d_far = Point((0.8, ways_length-1, 0))

    def dray_way(self, img):
        left_2d_near = self.camera.project_point_3d_to_2d(self.left_3d_near)
        left_2d_far = self.camera.project_point_3d_to_2d(self.left_3d_far)
        right_2d_near = self.camera.project_point_3d_to_2d(self.right_3d_near)
        right_2d_far = self.camera.project_point_3d_to_2d(self.right_3d_far)
        cv2.line(img, right_2d_near, right_2d_far, BLACK, LINE_WIDTH)
        cv2.line(img, left_2d_near, left_2d_far, BLACK, LINE_WIDTH)
        return img

    def draw_coordinate_system(self, img):
        center3d = Point((0, 0, 0))
        center2d = self.camera.project_point_3d_to_2d(center3d)
        cv2.circle(img, center2d, 5, BLACK, 5)
        end = 10
        for i in range(1, end):
            x3d = Point((i, 0, 0))
            y3d = Point((0, i, 0))
            z3d = Point((0, 0, i))
            x2d = self.camera.project_point_3d_to_2d(x3d)
            y2d = self.camera.project_point_3d_to_2d(y3d)
            z2d = self.camera.project_point_3d_to_2d(z3d)
            cv2.circle(img, x2d, 5, BLUE, 5)
            cv2.circle(img, y2d, 5, GREEN, 5)
            cv2.circle(img, z2d, 5, RED, 5)
            if i == end-1:
                cv2.line(img, x2d, center2d, BLUE, LINE_WIDTH)
                cv2.line(img, y2d, center2d, GREEN, LINE_WIDTH)
                cv2.line(img, z2d, center2d, RED, LINE_WIDTH)
        return img


class Reader(SeasonReader):
    """Обработка видеопотока."""
    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.way_estimator = WayEstimator(calib_dict, 10)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        cv2.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        self.way_estimator.dray_way(self.frame)
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
