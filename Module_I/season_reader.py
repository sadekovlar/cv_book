import os
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import gzip
import cv2
import numpy as np
import pandas as pd
from typing import Optional, List

class SeasonReader:
    _window_name: str = 'Frame'
    _start_episode: int = 1
    _finish_episode: int = 10
    _camera_name: str = 'leftImage'
    _gps_name: str = 'ubloxGps'
    _imu_name: str = 'minsEth'
    _data_path: str = '/data/'
    _video_ext: str = 'avi'

    frame: Optional[np.ndarray] = None
    frame_grab_msec: int = 0
    shot_grab_msec: int = 0
    shot: Optional[dict] = None

    def initialize(self, path_to_data_root: str = '', window_name: str = 'Frame',
                   camera_name: str = 'leftImage', start_episode: int = 1, finish_episode: int = 999, video_ext: str = 'avi',
                   gps_name: str = 'emlidLeft', imu_name: str = 'minsEth') -> bool:
        self._data_path = path_to_data_root
        self._camera_name = camera_name
        self._window_name = window_name
        self._start_episode = start_episode
        self._finish_episode = finish_episode
        self._video_ext = video_ext
        self._gps_name = gps_name
        self._imu_name = imu_name

        did_on_init: bool = self.on_init()
        if not did_on_init:
            return False

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
                break

            it = video_iter[i_episode - 1]
            episode_title = ''.join(e + '.' for e in it.split('.')[:3])
            video_path = os.path.join(self._data_path,  it)
            info_path = os.path.join(self._data_path, episode_title + 'info.yml.gz')
            info_file = self._read_info_file(info_path)
            cap = cv2.VideoCapture(video_path) 

            print("-> " + episode_title)
            for shot in info_file['shots']:
                self.shot = shot
                self.shot_grab_msec = shot['grabMsec']
                did_on_shot: bool = self.on_shot()
                if not did_on_shot:
                    return False

                if self._gps_name in shot:
                    did_on_frame: bool = self.on_gps_frame()
                    if not did_on_frame:
                        return False

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

                    cv2.imshow(self._window_name, self.frame)
                    keyboard = cv2.waitKey(10)
                    if keyboard == ord('q'):
                        cap.release()
                        return True

            i_episode += 1
            cap.release()
        return True

    def on_init(self) -> bool:
        raise NotImplementedError()

    def on_shot(self) -> bool:
        raise NotImplementedError()

    def on_frame(self) -> bool:
        raise NotImplementedError()

    def on_gps_frame(self) -> bool:
        raise NotImplementedError()
        
    def on_imu_frame(self) -> bool:
        raise NotImplementedError()
        
    @staticmethod
    def _read_info_file(file_path) -> dict:
        if not os.path.isfile(file_path):
            raise FileNotFoundError('error opening info file ' + file_path)

        try:
            with gzip.open(file_path, 'rt') as f:
                f.readline()
                f = f.read()
                f = f.replace(':', ': ')
                yml = yaml.load(f, Loader=Loader)

                return yml
        except OSError:
            print(f"Error in {file_path}")
            raise