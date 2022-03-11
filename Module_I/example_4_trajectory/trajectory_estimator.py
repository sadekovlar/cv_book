import numpy as np
import cv2

from Module_I.spatial_geometry_tools.calib import Calib
from Module_I.spatial_geometry_tools.camera import Camera
from Module_I.spatial_geometry_tools.point import Point3d as Point


BLUE = (255, 0, 0)
LINE_WIDTH = 3


class TrajectoryEstimator:
    '''
    Задаем калиб, высоту, ширину, глубину параллелограмма и расстояние от камеры.
    '''
    def __init__(self, calib_dict: np.array, height, wight, length, depth: int):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.left_3d_near1 = Point((-wight / 2, depth, 0))
        self.left_3d_far1 = Point((-wight / 2, length, 0))
        self.right_3d_near1 = Point((wight / 2, depth, 0))
        self.right_3d_far1 = Point((wight / 2, length, 0))
        self.left_3d_near_up1 = Point((-wight / 2, depth, height))
        self.left_3d_far_up1 = Point((-wight / 2, length, height))
        self.right_3d_near_up1 = Point((wight / 2, depth, height))
        self.right_3d_far_up1 = Point((wight / 2, length, height))

    def dray_trajectory(self, img):
        left_2d_near1 = self.camera.project_point_3d_to_2d(self.left_3d_near1)
        left_2d_far1 = self.camera.project_point_3d_to_2d(self.left_3d_far1)
        right_2d_near1 = self.camera.project_point_3d_to_2d(self.right_3d_near1)
        right_2d_far1 = self.camera.project_point_3d_to_2d(self.right_3d_far1)
        left_2d_near_up1 = self.camera.project_point_3d_to_2d(self.left_3d_near_up1)
        left_2d_far_up1 = self.camera.project_point_3d_to_2d(self.left_3d_far_up1)
        right_2d_near_up1 = self.camera.project_point_3d_to_2d(self.right_3d_near_up1)
        right_2d_far_up1 = self.camera.project_point_3d_to_2d(self.right_3d_far_up1)

        cv2.rectangle(img, left_2d_near1, right_2d_near1, BLUE, LINE_WIDTH)
        cv2.rectangle(img, left_2d_far1, right_2d_far1, BLUE, LINE_WIDTH)
        cv2.rectangle(img, left_2d_far_up1, right_2d_far_up1, BLUE, LINE_WIDTH)
        cv2.rectangle(img, left_2d_near_up1, right_2d_near_up1, BLUE, LINE_WIDTH)
        cv2.rectangle(img, left_2d_far_up1, left_2d_far1, BLUE, LINE_WIDTH)
        cv2.rectangle(img, right_2d_far_up1, right_2d_far1, BLUE, LINE_WIDTH)
        cv2.rectangle(img, left_2d_near_up1, left_2d_near1, BLUE, LINE_WIDTH)
        cv2.rectangle(img, right_2d_near_up1, right_2d_near1, BLUE, LINE_WIDTH)
        cv2.line(img, left_2d_near1, left_2d_far1, BLUE, LINE_WIDTH)
        cv2.line(img, left_2d_near_up1, left_2d_far_up1, BLUE, LINE_WIDTH)
        cv2.line(img, right_2d_near1, right_2d_far1, BLUE, LINE_WIDTH)
        cv2.line(img, right_2d_near_up1, right_2d_far_up1, BLUE, LINE_WIDTH)

        return img