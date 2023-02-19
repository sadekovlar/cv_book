import os
import gzip
import yaml
import math

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import cv2
import numpy as np

from srccam import CalibReader, Calib, Camera, SeasonReader, Point3d as Point


DATA_PATHS = ['../data/city/trm.169.007.info.yml.gz',
              '../data/city/trm.169.008.info.yml.gz']
GPS_TAG = 'emlidLeft'

RED = (0, 0, 255)


def get_gps_data(file_paths: list) -> list:
    """Получение информации пути трамвая."""

    data = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError('File {} not found.'.format(file_path))

        try:
            with gzip.open(file_path, 'rt') as file:
                file.readline()
                file = file.read()
                file = file.replace(':', ': ')
                yml = yaml.load(file, Loader=Loader)
                for shot in yml['shots']:
                    if GPS_TAG in shot:
                        sense_data = shot[GPS_TAG]['senseData']
                        data.append({'nord': sense_data['nord'],
                                     'east': sense_data['east'],
                                     'yaw': sense_data['yaw']})

        except Exception as e:
            print('Error while reading file {}: {}.'.format(file_path, e))
            raise

    return data


def get_dist_and_angle(lat1: float, long1: float, lat2: float, long2: float) -> (float, float):
    """Вычисление расстояния между точками и начального азимута по их координатам."""

    # Радиус Земли
    rad = 6372795

    # Перевод координат в радианы
    lat1 = lat1 * math.pi / 180
    lat2 = lat2 * math.pi / 180
    long1 = long1 * math.pi / 180
    long2 = long2 * math.pi / 180

    # Косинусы и синусы широт и разницы долгот
    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    c_delta = math.cos(delta)
    s_delta = math.sin(delta)

    # Вычисление длины большого круга
    y = math.sqrt(math.pow(cl2 * s_delta, 2) + math.pow(cl1 * sl2 - sl1 * cl2 * c_delta, 2))
    x = sl1 * sl2 + cl1 * cl2 * c_delta
    ad = math.atan2(y, x)
    dist = ad * rad

    # Вычисление начального азимута
    x = (cl1 * sl2) - (sl1 * cl2 * c_delta)
    y = s_delta * cl2
    z = math.degrees(math.atan(-y / x))

    if x < 0:
        z = z + 180

    z2 = (z + 180) % 360 - 180
    z2 = - math.radians(z2)
    angle_rad = z2 - ((2 * math.pi) * math.floor((z2 / (2 * math.pi))))
    # angle_deg = (angle_rad * 180) / math.pi

    return dist, angle_rad


class PathCreator:
    """Вычисление и отображение пути следования трамвая из данных GPS."""

    POINT_CNT = 10
    COLOR = RED
    LINE_WIDTH = 10

    def __init__(self, calib):
        self.camera = Camera(calib)
        self.data = get_gps_data(DATA_PATHS)
        self.dists, self.angles = np.zeros(len(self.data) - 1), np.zeros(len(self.data) - 1)
        self.process_data()
        # индекс текущей точки пути
        self.current_index = 0
        # текущий участок пути
        self.path = [(self.dists[i], self.angles[i]) for i in range(self.POINT_CNT)]

    def process_data(self):
        """Нахождение расстояний и углов для всех известных точек."""

        for i in range(len(self.data) - 1):
            p1, p2 = self.data[i], self.data[i + 1]
            self.dists[i], self.angles[i] = get_dist_and_angle(p1['nord'], p1['east'], p2['nord'], p2['east'])

    def next_point(self):
        """Добавление следующей точки в маршрут."""

        self.current_index += 1
        self.path = self.path[1:]
        point_index = self.current_index + self.POINT_CNT
        if point_index < len(self.dists):
            self.path.append((self.dists[point_index], self.angles[point_index]))
        else:
            self.path.append(self.path[-1])

    def render_path(self, frame):
        """Отображение пути на экране."""

        points_3d = [Point((0, 0, 0))]

        # TODO: calculate positions and add points
        for (dist, angle) in self.path:
            pass
            points_3d.append(Point((0, 10, 0)))

        points_2d = [self.camera.project_point_3d_to_2d(point) for point in points_3d]

        for i in range(len(points_2d) - 1):
            cv2.line(frame, points_2d[i], points_2d[i + 1], self.COLOR, self.LINE_WIDTH)


# X2:=L*sin((U*PI)/180)+X;
# Y2:=L*cos((U*PI)/180)+Y;


class PathPredictor(SeasonReader):
    """Обработка видеопотока."""

    def __init__(self):
        self.path_creator = None

    def on_init(self) -> bool:
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(file_name='../data/city/leftImage.yml', param=par)
        calib_dict = calib_reader.read()
        calib = Calib(calib_dict)
        self.path_creator = PathCreator(calib)
        return True

    def on_shot(self) -> bool:
        return True

    def on_frame(self) -> bool:
        self.path_creator.render_path(self.frame)
        return True

    def on_gps_frame(self) -> bool:
        self.path_creator.next_point()
        return True

    def on_imu_frame(self) -> bool:
        return True


if __name__ == '__main__':
    predictor = PathPredictor()
    predictor.initialize(path_to_data_root='../data/city/')
    predictor.run()
