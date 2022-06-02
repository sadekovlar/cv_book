import cv2
import numpy as np


class LightsStruckReducer:
    """Класс для приглушения засветки от фар"""

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
        look_up_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(img, look_up_table)
        return res

    @staticmethod
    def get_map_mask(img):
        """Получение маски на основе карты расстояний до источника засветки"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        overexposed_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
        overexposed_mask_inv = cv2.bitwise_not(overexposed_mask)
        dist = cv2.distanceTransform(overexposed_mask_inv,
                                     cv2.DIST_L2,
                                     cv2.DIST_MASK_3,
                                     overexposed_mask_inv)
        cv2.normalize(dist, dist, 0.0, 20.0, cv2.NORM_MINMAX)
        map_mask_inv = np.zeros(dist.shape, dtype=np.uint8)
        map_mask_inv[np.where(dist > 1)] = 255
        map_mask = np.zeros(dist.shape, dtype=np.uint8)
        map_mask[np.where(map_mask_inv == 0)] = 255
        return map_mask

    @staticmethod
    def reduce_lights_struck(img: np.ndarray) -> np.ndarray:
        res = img.copy()
        overexposed_mask = LightsStruckReducer.get_map_mask(res)
        mask_inv = cv2.bitwise_not(overexposed_mask)
        non_overexposed_part = cv2.bitwise_and(res, res, mask=mask_inv)
        overexposed_part = cv2.bitwise_and(res, res, mask=overexposed_mask)
        corrected = LightsStruckReducer.gamma_correction(overexposed_part, 1.06)
        res = cv2.add(non_overexposed_part, corrected)
        return res


if __name__ == '__main__':
    video_name = 'klt.427.003.mp4'
    cap = cv2.VideoCapture('../data/processing/' + video_name)
    while cap.isOpened():
        succeed, frame = cap.read()
        if succeed:
            frame = LightsStruckReducer.reduce_lights_struck(frame)
            cv2.imshow(video_name, frame)
        else:
            cv2.destroyAllWindows()
            cap.release()
        cv2.waitKey(1)
