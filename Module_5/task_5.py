import numpy as np
import cv2
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from srccam.season_reader import SeasonReader


class TrafficLightDetector(SeasonReader):

    def on_init(self):
        return True

    def on_shot(self):
        return True

    def on_frame(self):
        img = self.frame
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        low_green = np.array([68, 85, 0])
        high_green = np.array([102, 255, 255])
        green_mask = cv2.inRange(hsv, low_green, high_green)
        gray = cv2.bitwise_and(gray, gray, mask=green_mask)

        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                  param1=100, param2=15,
                                  minRadius=1, maxRadius=15)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(img, center, 1, (0, 100, 100), 3)
                radius = i[2]
                cv2.circle(img, center, radius, (255, 0, 255), 3)

        self.frame = img
        return True

    def on_gps_frame(self) -> bool:
        return True


if __name__ == "__main__":
    for number in range(235, 236):
        init_args = {
            'path_to_data_root': '../data/tram/'
        }
        s = TrafficLightDetector()
        s.initialize(**init_args)
        s.run()
    print("Done!")
