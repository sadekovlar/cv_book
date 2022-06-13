import numpy as np
import cv2
import os
import sys
from season_reader import SeasonReader


class BirdsView(SeasonReader):
    corner_points_array = None
    img_params = None
    height = 0
    width = 0

    def on_init(self):
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        if self.img_params is None:
            self.set_params()
        matrix = cv2.getPerspectiveTransform(self.corner_points_array, self.img_params)
        bv = cv2.warpPerspective(self.frame, matrix, (self.width, self.height))
        scale_percent = 30
        width_s = int(self.width * scale_percent / 100)
        height_s = int(self.height * scale_percent / 100)
        dim = (width_s, height_s)
        bv = cv2.resize(bv, dim)
        x_offset = 5
        y_offset = 5
        self.frame[y_offset:y_offset + bv.shape[0], x_offset:x_offset + bv.shape[1]] = bv
        return True

    def on_gps_frame(self) -> bool:
        return True

    def set_params(self):
        self.height, self.width, _ = self.frame.shape
        bl = [0, self.height]
        br = [self.width, self.height]
        tr = [self.width * 3 / 5, self.height / 2.25]
        tl = [self.width * 2 / 5, self.height / 2.25]
        self.corner_points_array = np.float32([tl, tr, br, bl])
        imgTl = [0, 0]
        imgTr = [self.width, 0]
        imgBr = [self.width, self.height]
        imgBl = [0, self.height]
        self.img_params = np.float32([imgTl, imgTr, imgBr, imgBl])


if __name__ == "__main__":
    for number in range(235, 236):
        init_args = {
            'path_to_data_root': './data/tram/'
        }
        s = BirdsView()
        s.initialize(**init_args)
        s.run()
    print("Done!")
