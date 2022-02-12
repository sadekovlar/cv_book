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
    _finish_episode: int = 999
    _camera_name: str = 'central60'
    _camera_calib_name: str = 'central60'
    _gps_name: str = 'ubloxGps'
    _imu_name: str = 'minsEth'
    _can_data: str = 'dbwFbVehicleCan'
    _data_path: str = '/testdata'
    _video_ext: str = 'avi'
    _serial: str = 'trm'

    frame: Optional[np.ndarray] = None
    frame_grab_msec: int = 0
    shot_grab_msec: int = 0
    shot: Optional[dict] = None

    def initialize(self, serial: str = "trm", season: int = 169,
                   path_to_data_root: str = '/testdata', window_name: str = 'Frame',
                   camera_name: str = 'central60', camera_calib_name: Optional[str] = None, 
                   start_episode: int = 1, finish_episode: int = 999, video_ext: str = 'avi',
                   gps_name: str = 'emlidLeft', imu_name: str = 'minsEth',
                   can_name: str = 'dbwFbTram') -> bool:
        self._data_path = os.path.join(path_to_data_root, serial, "inp", f"{serial}.{season:03d}")
        self._camera_name = camera_name
        self._camera_calib_name = camera_calib_name if camera_calib_name else camera_name
        self._window_name = window_name
        self._start_episode = start_episode
        self._finish_episode = finish_episode
        self._video_ext = video_ext
        self._gps_name = gps_name
        self._imu_name = imu_name
        self._can_data = can_name
        self._serial = serial

        did_on_init: bool = self.on_init()
        if not did_on_init:
            return False

        return True

    def run(self) -> bool:
        data_files: List[str] = []
        for _, _, files in os.walk(self._data_path):
            data_files = files
            break

        video_iter = sorted([x for x in data_files if self._camera_name + "." + self._video_ext in x])
        
        i_episode = self._start_episode
        print("->run")
        while i_episode <= self._finish_episode:
            if i_episode > len(video_iter):
                break

            it = video_iter[i_episode - 1]
            episode_title = ''.join(e + '.' for e in it.split('.')[:3])
            video_path = os.path.join(self._data_path, episode_title + self._camera_name + f'.{self._video_ext}')
            info_path = os.path.join(self._data_path, episode_title + 'info.yml.gz')
            info_file = self._read_info_file(info_path)
            print("-> " + episode_title)
            for shot in info_file['shots']:
                self.shot = shot
                self.shot_grab_msec = shot['grabMsec']
                did_on_shot: bool = self.on_shot()
                if not did_on_shot:
                    return False      
            i_episode += 1
        return True

    def on_init(self) -> bool:
        raise NotImplementedError()

    def on_shot(self) -> bool:
        raise NotImplementedError()

    
    def get_gps_frame(self) -> pd.DataFrame:
        try:
            shot: dict = self.shot[self._gps_name]['senseData']
            shot['grabMsec'] = self.shot[self._gps_name]['grabMsec']
            return shot
        except:
            return list()
        
    def get_imu_frame(self) -> pd.DataFrame:
        try:
            shot: dict = self.shot[self._imu_name]
            return shot
        except:
            return list()
        
    def get_can_frame(self) -> pd.DataFrame:
        try:
            if self._serial == 'trm':
                shot: dict = {}
            else:
                shot: dict = self.shot[self._can_data]['dbw_feedback_data']
            shot['steeringAngle'] = self.shot[self._can_data]['vehicleCanData']['vehicleCanDetection0']['steeringAngle']
            shot['speed'] = self.shot[self._can_data]['vehicleCanData']['vehicleCanDetection0']['speed']
            shot['grabMsec'] = self.shot[self._can_data]['grabMsec']
            odoFrame = pd.DataFrame.from_dict(shot, orient="index")
            return shot
        except:
            return list()
        
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