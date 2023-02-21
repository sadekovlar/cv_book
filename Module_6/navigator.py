import numpy as np
import cv2 as cv

from srccam.season_reader import SeasonReader


class MovingObjects(SeasonReader):

    def on_init(self):
        self.prvs = None
        self.hsv = None
        self.first_frame = True
        self.label_move = ''
        self.label_steer = ''
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        if self.first_frame:
            self.first_frame = False
            self.prvs = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            self.hsv = np.zeros_like(self.frame)
            self.hsv[..., 1] = 255
        else:
            next = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(self.prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            avg_mag = np.average(mag)
            avg_ang = np.average(ang)
            self.hsv[..., 0] = ang * 180 / np.pi / 2
            self.hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            prev_steer = self.label_steer
            if avg_mag < 0.5:
                self.label_move = 'STOPPED'
                self.label_steer = ''
            else:
                self.label_move = 'MOVING'
                if np.pi*7/8 < avg_ang % (2*np.pi) < 4/3*np.pi:
                    self.label_steer = 'RIGHT'
                elif np.pi*5/3 < avg_ang % (2*np.pi) < 1/8*np.pi:
                    self.label_steer = 'LEFT'
                else:
                    self.label_steer = ''
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(self.frame, self.label_move, (50, 50), font, 1, (255, 0, 0), 2, cv.LINE_4)
            if prev_steer == self.label_steer:
                cv.putText(self.frame, self.label_steer, (50, 100), font, 1, (255, 0, 0), 2, cv.LINE_4)
            self.prvs = next
        return True

    def on_gps_frame(self) -> bool:
        return True


if __name__ == "__main__":
    for number in range(235, 236):
        init_args = {
            'path_to_data_root': '../data/city/'
        }
        s = MovingObjects()
        s.initialize(**init_args)
        s.run()
    print("Done!")
