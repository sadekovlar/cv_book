from typing import Any, Optional

import cv2


class CalibReader:
    _file_name: str = "leftImage.yaml"

    def __init__(self, file_name: str = "", param: Optional[list[str]] = None):
        if param is None:
            param = []
        self._file_name = file_name
        self._param = param

    @classmethod
    def initialize(cls, *args, **kwargs):
        cls(*args, **kwargs)

    def read(self) -> dict[str, Any]:
        file_name = self._file_name
        list_param = self._param
        fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        param = dict()
        if fs.isOpened():
            for index in list_param:
                param[index] = fs.getNode(index).mat()
        fs.release()
        return param


if __name__ == "__main__":
    par = ["K", "D", "r", "t"]
    calib = CalibReader(file_name="../data/tram/leftImage.yml", param=par)
    matrix = calib.read()
    print(matrix)
