from math import pi
import cv2
import numpy as np
from load_calib import CalibReader
from season_reader import SeasonReader
from spatial_geometry_tools.calib import Calib
from spatial_geometry_tools.camera import Camera
from src.object3d import Object3D
from src.vector import Vector3 as vector
from src.constants import *

class WayAnalyzer:
    """Класс для анализа и рисования фигур."""
    def __init__(self, calib_dict: np.array):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.drawCointer = 0

    def radians(deg): return deg * (pi/180)
    def degrees(rad): return rad * (180/pi)

    def get2d(self, vec : vector):
        return self.camera.project_point_3d_to_2d(vec)

    def draw_figure(self, img, obj : Object3D):
        self.drawCointer += 1
        obj.setRotation( vector(self.drawCointer/100, 0, 0) ) # Демонстративное вращение по оси X

        p, l = obj.get()
        for i in l: cv2.line(img, self.get2d(p[i[0]]),self.get2d(p[i[1]]), GREEN, 1)
        for i in p: cv2.circle(img, self.get2d(i), 1, RED, 1)

class Reader(SeasonReader):
    """Обработка видеопотока."""
    def on_init(self, _file_name: str = None): # Инициализация
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(file_name='../data/tram/leftImage.yml', param=par)
        calib_dict = calib_reader.read()

        self.way_analyzer = WayAnalyzer(calib_dict)
        self.parall = Object3D( vector(0,18,0), vector(0,0,0), Object3D.OBJ_PARALL, 4, 0.4, 1.6)
        self.cube = Object3D( vector(0,18,1), vector(0,0,0), Object3D.OBJ_PARALL, 1.6, 1.6, 1.6)
        self.misis = Object3D( vector(0,8,0), vector(0,0,0), Object3D.OBJ_MISiS, 2, 1.8, 0)
        return True

    def on_shot(self):
        return True

    def on_frame(self): # События, применяемые к КАЖДОМУ кадру
        self.way_analyzer.draw_figure(self.frame, self.parall)
        self.way_analyzer.draw_figure(self.frame, self.cube)
        self.way_analyzer.draw_figure(self.frame, self.misis)
        return True

    def on_gps_frame(self): # События, происходящие при получении данных от GPS
        return True

    def on_imu_frame(self): # События, происходящие при получении данных от IMU
        return True

if __name__ == '__main__':
    init_args = { 'path_to_data_root': '../data/tram/' }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')