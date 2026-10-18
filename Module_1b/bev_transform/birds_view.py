#!/usr/bin/env python3
"""BEV через фиксированную гомографию по углам трапеции в кадре.

Не использует калиб r/t — углы площадки задаются вручную в set_params
(доля ширины/высоты кадра).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "srccam"))

from srccam.season_reader import SeasonReader


class BirdsView(SeasonReader):
    """Преобразование кадра к виду птичьего полёта и врезка в угол изображения."""

    corner_points_array = None
    img_params = None
    height = 0
    width = 0
    matrix = None

    def on_init(self):
        it = [x for x in os.listdir(self._data_path) if self._video_ext in x][0]
        video_path = os.path.join(self._data_path, it)
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.set_params()
        self.matrix = cv2.getPerspectiveTransform(self.corner_points_array, self.img_params)
        print(self.width, " ", self.height)
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        bv = cv2.warpPerspective(self.frame, self.matrix, (self.width, self.height))
        scale_percent = 30
        width_s = int(self.width * scale_percent / 100)
        height_s = int(self.height * scale_percent / 100)
        dim = (width_s, height_s)
        bv = cv2.resize(bv, dim)
        x_offset = 5
        y_offset = 5
        self.frame[y_offset : y_offset + bv.shape[0], x_offset : x_offset + bv.shape[1]] = bv
        return True

    def on_gps_frame(self) -> bool:
        return True

    def on_imu_frame(self) -> bool:
        return True

    def set_params(self):
        bl = [0, self.height]
        br = [self.width, self.height]
        tr = [self.width * 3 / 5, self.height / 2.25]
        tl = [self.width * 2 / 5, self.height / 2.25]
        self.corner_points_array = np.float32([tl, tr, br, bl])
        imgtl = [0, 0]
        imgtr = [self.width, 0]
        imgbr = [self.width, self.height]
        imgbl = [0, self.height]
        self.img_params = np.float32([imgtl, imgtr, imgbr, imgbl])


if __name__ == "__main__":
    city = ROOT / "data" / "city"
    init_args = {"path_to_data_root": str(city) + os.sep}
    reader = BirdsView()
    reader.initialize(**init_args)
    reader.run()
    print("Done!")
