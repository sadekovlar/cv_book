import cv2 as cv
import numpy as np
import os
import yaml
from load_calib import CalibReader
from season_reader import SeasonReader
from spatial_geometry_tools.calib import Calib
from spatial_geometry_tools.camera import Camera
from spatial_geometry_tools.point import Point3d as Point
from src.gps_data import GpsData
import math
import matplotlib.pyplot as plt
import pandas as pd

BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class DirectionPrediction:
    def __init__(self, calib_dict: dict, ways_length: int):
        self.calib = Calib(calib_dict)
        self.camera = Camera(self.calib)
        self.center_near = Point((0.3, 5, 0))
        self.current_point = 0
        self.main_point = ()
        self.df = []
        self.x_values_new = []
        self.y_values_new = []
        self.two_d_coords = []
        self.gps_data = GpsData()
        self.GPS_DIR = "../data/tram/"  # This is your Project Root
        self.files = [f for f in os.listdir(self.GPS_DIR) if f.endswith("info.yml")]
        self.points = []
        self.files.sort()

    @staticmethod
    def mercator(data, origin):
        R_equator = 6378137.000
        lat0Rad = np.pi * origin[0] / 180.0
        lon0Rad = np.pi * origin[1] / 180.0
        latRad = np.pi * data[0] / 180.0
        lonRad = np.pi * data[1] / 180.0
        # print("Forward mercator")
        scale = np.cos(lat0Rad)
        # print("scale:", scale)
        # print("origin:", lat0Rad, lon0Rad)
        R = R_equator * (0.99832407 + 0.00167644 * np.cos(2.0 * lat0Rad) - 0.00000352 * np.cos(4.0 * lat0Rad))
        x = scale * R * lonRad
        y = scale * R * np.log(np.tan(np.pi / 4.0 + latRad / 2.0))
        x0 = scale * R * lon0Rad
        y0 = scale * R * np.log(np.tan(np.pi / 4.0 + lat0Rad / 2.0))
        x = np.copy(x - x0)
        y = np.copy(y - y0)
        return [x, y]

    def draw_point(self, img: np.array, data: dict):
        x_cur, y_cur = self.mercator([data['nord'], data['east']], [55.88677827, 37.642756055])
        flag = self.df.grabMsec > data['grabMsec']
        dFrame = self.df[flag]
        dFrame.x = dFrame.x - x_cur
        dFrame.y = dFrame.y - y_cur
        yaw = math.atan2(dFrame['y'].values[1] - dFrame['y'].values[0],
                         dFrame['x'].values[1] - dFrame['x'].values[0]) - np.pi/2
        x = dFrame['x'] * math.cos(-yaw) - dFrame['y'] * math.sin(-yaw)
        y = dFrame['x'] * math.sin(-yaw) + dFrame['y'] * math.cos(-yaw)

        uv_image = list()
        for el in range(0, len(x)):
            center_near = self.camera.project_point_3d_to_2d(Point((x.values[el], y.values[el], 0)))
            uv_image.append(center_near)
        return uv_image

    def convert_gps_to_xy(self):
        df = pd.read_csv(self.GPS_DIR + "gps.csv", sep=';', index_col=False)
        [x, y] = self.mercator([df['nord'].values, df['east'].values], [55.88677827, 37.642756055])
        df['x'] = x
        df['y'] = y
        self.df = df


class Reader(SeasonReader):
    """Обработка видеопотока."""

    def on_init(self, _file_name: str = None):
        par = ['K', 'D', 'r', 't']
        calib_reader = CalibReader()
        calib_reader.initialize(
            file_name='../data/tram/leftImage.yml',
            param=par)
        calib_dict = calib_reader.read()
        self.uv = list()
        self.direction_prediction = DirectionPrediction(calib_dict, 10)
<<<<<<< HEAD
        self.direction_prediction.convert_gps_to_xy()
=======
        self.direction_prediction.read_files()
        # self.direction_prediction.convert_gps_to_xy()

>>>>>>> 32f3d95fe53437fb582ab7dcffa1621f8a0f4670
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        # 960 x 540 pixels.
        cv.putText(self.frame, f'GrabMsec: {self.frame_grab_msec}', (15, 50),
                   cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        return True

    def on_gps_frame(self):
        shot: dict = self.shot[self._gps_name]['senseData']
        shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
        return self.direction_prediction.draw_point(self.frame, shot)

    def on_imu_frame(self):
        shot: dict = self.shot[self._imu_name]
        return True

    def run(self) -> bool:
        data_files: List[str] = []
        for _, _, files in os.walk(self._data_path):
            data_files = files
            break

        video_iter = sorted([x for x in data_files if self._video_ext in x])

        i_episode = self._start_episode
        print("->run")
        while i_episode <= self._finish_episode:
            if i_episode > len(video_iter):
                x = self.direction_prediction.x_values
                y = self.direction_prediction.y_values
                plt.plot(x, y, label="origin")
                x_new = self.direction_prediction.x_values_new
                y_new = self.direction_prediction.y_values_new
                plt.plot(x_new, y_new, label='new')
                plt.legend()
                plt.show()
                break

            it = video_iter[i_episode - 1]
            episode_title = ''.join(e + '.' for e in it.split('.')[:3])
            video_path = os.path.join(self._data_path, it)
            info_path = os.path.join(self._data_path, episode_title + 'info.yml.gz')
            info_file = self._read_info_file(info_path)
            cap = cv.VideoCapture(video_path)
            # self.read_storage("../data/tram/"+episode_title+'info.yml')
            print("-> " + episode_title)
            for shot in info_file['shots']:
                self.shot = shot
                self.shot_grab_msec = shot['grabMsec']
                did_on_shot: bool = self.on_shot()
                if not did_on_shot:
                    return False

                if self._gps_name in shot:
                    uv = self.on_gps_frame()
                    self.uv = uv

                if self._imu_name in shot:
                    did_on_frame: bool = self.on_imu_frame()
                    if not did_on_frame:
                        return False

                if self._camera_name in shot:
                    _, self.frame = cap.read()
                    self.frame_grab_msec = self.shot_grab_msec

                    did_on_frame: bool = self.on_frame()
                    if not did_on_frame:
                        return False

                    for u_v_ in self.uv:
                        cv.circle(self.frame, u_v_, 5, RED, LINE_WIDTH)
                    cv.imshow(self._window_name, self.frame)
                    keyboard = cv.waitKey(10)
                    if keyboard == ord('q'):
                        cap.release()
                        return True

            i_episode += 1
            cap.release()
        return True


if __name__ == '__main__':
    init_args = {
        'path_to_data_root': '../data/tram/'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')
