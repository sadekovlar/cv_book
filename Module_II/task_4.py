import cv2
import numpy as np


class ReduceGlare:
    """Обработка видеопотока."""

    def __init__(self, path_to_video):
        self.cap = cv2.VideoCapture(path_to_video)
        self.frame = None

    def run(self):
        if not self.cap.isOpened():
            print("Файл не открылся :(")
        else:
            while self.cap.isOpened():
                ret, self.frame = self.cap.read()
                if not ret: break
                self.on_frame()
                cv2.imshow('VideoWithReducedGlare', self.frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()

    def on_frame(self):
        self.create_mask()
        return True

    def create_mask(self):
        img = self.frame
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([15, 24, 196])
        upper = np.array([72, 194, 255])
        mask = cv2.inRange(hsv, lower, upper)
        pos = np.where(mask > 0)
        h, s, v = cv2.split(hsv)
        v[pos] -= 30
        hsv2 = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
        self.frame = result
        return True


if __name__ == '__main__':
    init_args = {
        'path_to_video': '../data/processing/trm.168.090.avi'
    }
    s = ReduceGlare(**init_args)
    s.run()
    print('Done!')
