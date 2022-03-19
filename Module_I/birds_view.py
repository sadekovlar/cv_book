import numpy as np
import cv2
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from example import MyReader


class BirdsView(MyReader):
    corner_points_array = None
    img_params = None
    height = 0
    width = 0

    def on_frame(self):
        if self.img_params is None:
            self.set_params()
        super().on_frame()
        matrix = cv2.getPerspectiveTransform(self.corner_points_array, self.img_params)
        bv = cv2.warpPerspective(self.frame, matrix, (self.width, self.height))
        scale_percent = 30
        width_s = int(self.width * scale_percent / 100)
        height_s = int(self.height * scale_percent / 100)
        dim = (width_s, height_s)
        bv = cv2.resize(bv, dim)
        x_offset = 1
        y_offset = 80
        self.frame[y_offset:y_offset + bv.shape[0], x_offset:x_offset + bv.shape[1]] = bv
        return True

    def set_params(self):
        self.height, self.width, _ = self.frame.shape
        bl = [self.width/4, self.height]
        br = [self.width/4*3, self.height]
        tr = [self.width/12*8, self.height/4*3]
        tl = [self.width/12*4, self.height/4*3]
        self.corner_points_array = np.float32([tl, tr, br, bl])
        imgTl = [0, 0]
        imgTr = [self.width, 0]
        imgBr = [self.width, self.height]
        imgBl = [0, self.height]
        self.img_params = np.float32([imgTl, imgTr, imgBr, imgBl])


if __name__ == "__main__":
    for number in range(235, 236):
        init_args = {
            'path_to_data_root' : './data/tram/'
        }
        s = BirdsView()
        s.initialize(**init_args)
        s.run()
    print("Done!")
