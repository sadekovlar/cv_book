import numpy as np
import cv2

from Module_I.spatial_geometry_tools.calib import Calib
from Module_I.spatial_geometry_tools.camera import Camera
from Module_I.spatial_geometry_tools.point import Point3d as Point


BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class WayEstimator:
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
        print('center2d:', center2d)
        cv2.circle(img, center2d, 5, BLACK, 5)

        for i in range(1, 20):
            x3d = Point((i, 0, 0))
            x2d = self.camera.project_point_3d_to_2d(x3d)
            print("x2d:", x2d)

            y3d = Point((0, i, 0))
            y2d = self.camera.project_point_3d_to_2d(y3d)
            print("y2d:", y2d)

            z3d = Point((0, 0, i))
            z2d = self.camera.project_point_3d_to_2d(z3d)
            print("z2d:", z2d)

            # x
            cv2.line(img, x2d, center2d, BLUE, LINE_WIDTH)
            cv2.circle(img, x2d, 5, BLUE, 5)

            # green y
            cv2.line(img, y2d, center2d, GREEN, LINE_WIDTH)
            cv2.circle(img, y2d, 5, GREEN, 5)

            # red z
            cv2.line(img, z2d, center2d, RED, LINE_WIDTH)
            cv2.circle(img, z2d, 5, RED, 5)

        return img