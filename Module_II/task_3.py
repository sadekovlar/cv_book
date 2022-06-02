import cv2 as cv
import numpy as np

class Reader:
    """Обработка видеопотока."""
    def initialize(self, path_to_video):
        self.cap = cv.VideoCapture(path_to_video)
        self.prevFrame = np.array([])

    def run(self):
        if not self.cap.isOpened():
            print("Ошибка! Не удалось открыть файл.")
        else:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break
                self.on_frame(frame)
                cv.imshow('VideoPlayer', frame)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv.destroyAllWindows()

    def on_frame(self, frame):
        if self.prevFrame.size != 0:                            # Если в истории есть предыдущий кадр, то..
            mask = cv.absdiff(frame, self.prevFrame)            # .. Вычисляем разницу между кадрами
            cv.normalize(mask, mask, 0, 120, cv.NORM_MINMAX)    # .. Нормализуем яркость разницы, чтобы исключить пересветы в местах замены
            cv.addWeighted(frame, 1.0, mask, 1.0, 0.0, frame)   # .. Производим наложение нормализованной разницы на кадр
        self.prevFrame = frame

if __name__ == '__main__':
    init_args = {
        'path_to_video': '../data/processing/trm.168.005.avi'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')