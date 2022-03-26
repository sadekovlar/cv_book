import cv2
import numpy as np

class CarLightsStruckReducer:
    '''Класс для уменьшения засветки от фар встречного ТС'''
    @staticmethod
    def __gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
        look_up_table = np.empty((1,256), np.uint8)
        for i in range(256):
            look_up_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(img, look_up_table)
        return res

    @staticmethod
    def reduce_light_struck(img: np.ndarray) -> np.ndarray:
        res = CarLightsStruckReducer.__gamma_correction(img, 1.4)
        return res


if __name__ == '__main__':
    video_name = 'klt.427.003.mp4'
    cap = cv2.VideoCapture('../data/processing/' + video_name)
    while cap.isOpened():
        succeed, frame = cap.read()
        if succeed:
            frame = CarLightsStruckReducer.reduce_light_struck(frame)
            cv2.imshow(video_name, frame)
        else:
            cv2.destroyAllWindows()
            cap.release()
        cv2.waitKey(10)
