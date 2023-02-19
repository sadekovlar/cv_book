import os
import gzip
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import cv2

from srccam import SeasonReader, SenseData


GPS_TAG = 'emlidLeft'


def get_info(file_paths: list):
    """Получение информации о положении трамвая."""

    info = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError('File {} not found.'.format(file_path))

        try:
            with gzip.open(file_path, 'rt') as file:
                file.readline()
                file = file.read()
                file = file.replace(':', ': ')
                data = yaml.load(file, Loader=Loader)
                for shot in data['shots']:
                    if GPS_TAG in shot:
                        info.append(shot[GPS_TAG])

        except Exception as e:
            print('Error while reading file {}: {}.'.format(file_path, e))
            raise

    return info


class PathPredictor:
    """Аппроксимация пути следования трамвая из данных GPS."""

    def __init__(self):
        self.info = get_info(['../data/city/trm.169.007.info.yml.gz',
                              '../data/city/trm.169.008.info.yml.gz'])
        print(len(self.info))
        print('emlidLeft' in self.info[6])

    def predict(self):
        pass


class Reader(SeasonReader):
    """Обработка видеопотока."""
    pass


if __name__ == '__main__':
    predictor = PathPredictor()
