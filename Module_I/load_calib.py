import cv2
from typing import Optional, List

class CalibReader:
    _file_name: str = 'leftImage.yaml'

    def initialize(self, file_name: str = '', param = list()) -> bool:
        self._file_name = file_name
        self._param = param
        return True

    def read(self) -> str:
        file_name = self._file_name
        list_param = self._param
        fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            param = dict()
            for index in list_param:
                param[index] = fs.getNode(index).mat()
        fs.release()
        return param 

if __name__ == "__main__":
    par = ["K", "D", "r", "t" ]
    calib = CalibReader()
    calib.initialize(file_name = '../data/tram/leftImage.yml', param = par)
    matrix = calib.read()
    print(matrix)

    