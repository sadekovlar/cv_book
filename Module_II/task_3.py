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
                if ret == True:
                    self.on_frame(frame)
                    cv.imshow('VideoPlayer', frame)
                    if cv.waitKey(10) & 0xFF == ord('q'):
                        break
                else:
                    break
            self.cap.release()
            cv.destroyAllWindows()

    def on_frame(self, frame):
        if self.prevFrame.size != 0: # Если предшеуствующий кадр зафиксирован, то ...
            # Реализация варианта с пропуском кадров, на которых дворники загораживают обзор:
            h1 = cv.calcHist([frame], [0], None, [100], [0,100])           # Гистограмма ТЕКУЩЕГО кадра в диапазоне [0,100]
            h2 = cv.calcHist([self.prevFrame], [0], None, [100], [0,100])  # Гистограмма ПРЕДЫДУЩЕГО кадра в диапазоне [0,100]
            if (cv.compareHist(h1, h2, cv.HISTCMP_CORREL) < 0.95):         # Сравнение гистограмм в режиме КОРРЕЛЯЦИИ
                cv.addWeighted(self.prevFrame, 1, frame, 0, 0, frame)      # Если резкое различие - пропускаем кадр
        self.prevFrame = frame.copy() # Фиксирует кадр как предшествующий для дальнейшего сравнения и постобработки
        # Повышение резкости текущего кадра для частичного избавления от размытия:
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        bluredFrame = cv.blur(frame, (3,3))
        filter = cv.filter2D(bluredFrame, -1, kernel)
        cv.addWeighted(frame, 1.5, filter, -0.5, 0, frame)
        return True


if __name__ == '__main__':
    init_args = {
        'path_to_video': '../data/processing/trm.168.005.avi'
    }
    s = Reader()
    s.initialize(**init_args)
    s.run()
    print('Done!')