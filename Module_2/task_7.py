import cv2
import numpy as np


class LightsReducer:

    @staticmethod
    def get_map_mask(img):
        img_blur_bil = cv2.bilateralFilter(img,  9, 75, 75, cv2.BORDER_DEFAULT)
        blurred = cv2.GaussianBlur(img_blur_bil, (9, 9), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        overexposed_mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]
        return overexposed_mask

    @staticmethod
    def reduce_lights_struck(img: np.ndarray) -> np.ndarray:
        res = img.copy()
        overexposed_mask = LightsStruckReducer.get_map_mask(res)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([15, 24, 196])
        upper = np.array([72, 194, 255])
        mask = cv2.inRange(hsv, lower, upper)
        pos = np.where(overexposed_mask > 0)
        h, s, v = cv2.split(hsv)
        v[pos] -= 30
        hsv2 = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
        return result


if __name__ == '__main__':
    video_name = '../data/processing/trm.179.003.mp4'
    cap = cv2.VideoCapture(video_name)
    while cap.isOpened():
        succeed, frame = cap.read()
        if succeed:
            frame = LightsStruckReducer.reduce_lights_struck(frame)
            cv2.imshow(video_name, frame)
        else:
            cv2.destroyAllWindows()
            cap.release()
        cv2.waitKey(1)