import math

import cv2
import numpy as np

VIDEO_PATH = '../data/processing/trm.179.003.avi'


class GlareReduction:
    """Класс для уменьшения эффекта засветки от ярких источников света на видео."""

    def __init__(self, path: str) -> None:
        self.path = path  # путь к видео для обработки
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX  # шрифт для надписей
        self.font_scale = 1.2  # размер шрифта для надписей
        self.image = None  # текущее обрабатываемое изображение

    def put_text(self, image: np.ndarray, text: str) -> None:
        """Размещение надписи `text` на изображении `image`."""

        text_size = cv2.getTextSize(text, self.font_face, self.font_scale, 2)[0]

        # вычисление координат для размещения текста
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = int(image.shape[0] * 0.9)

        cv2.putText(image, text, (text_x, text_y), self.font_face, self.font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    def _apply_polynomial_function(self, *args: float) -> None:
        """Применение полиномиального преобразования с коэффициентами `args` к изображению `self.image`."""

        table = np.array([args[0] + args[1] * i + args[2] * (i ** 2) + args[3] * (i ** 3)
                          for i in np.arange(0, 256)], dtype='uint8')
        cv2.LUT(self.image, table, self.image)

    def _apply_gamma_correction(self, gamma=1.0) -> None:
        """Применение гамма-коррекции с коэффициентом `gamma` к изображению `self.image`."""

        table = np.array([((i / 255) ** gamma) * 255 for i in np.arange(0, 256)], dtype='uint8')
        cv2.LUT(self.image, table, self.image)

    def reduce_glare(self, image: np.ndarray) -> np.ndarray:
        """Уменьшение эффекта засветки на изображении `image`."""

        self.image = image.copy()

        # * ПЕРВАЯ ПОЛИНОМИАЛЬНАЯ ФУНКЦИЯ
        # (коэффициенты подобраны таким образом, чтобы функция приближала
        # интенсивность пикселей к пороговому значению 100)
        # * результат: уменьшение эффекта засветки и уменьшение контрастности
        self._apply_polynomial_function(0, 1.657766, -0.009157128, 0.00002579473)

        # * ГАММА-КОРРЕКЦИЯ С `gamma` == 4 / 3 == 1.(3)
        # (значение > 1 для увеличения контрастности и уменьшения яркости)
        self._apply_gamma_correction(4 / 3)

        # * ВТОРАЯ ПОЛИНОМИАЛЬНАЯ ФУНКЦИЯ
        # (коэффициенты подобраны таким образом, чтобы функция приближала
        # интенсивность пикселей к пороговому значению 160)
        # * результат: уменьшение эффекта засветки и уменьшение контрастности
        # (эффект менее ярко выраженный, чем от первой полиномиальной функции)
        self._apply_polynomial_function(-4.263256 * math.exp(-14), 1.546429, -0.005558036, 0.00001339286)

        # * ГАММА-КОРРЕКЦИЯ С `gamma` == 5 / 4 == 1.25
        # (значение > 1 для увеличения контрастности и уменьшения яркости)
        # `gamma` ближе к 1 для менее явного эффекта, чем при первой гамма-коррекции
        self._apply_gamma_correction(5 / 4)

        return self.image

    def run(self) -> None:
        """Обработка видео и отображение результата."""

        cap = cv2.VideoCapture(self.path)

        while cap.isOpened():
            frame_read, frame = cap.read()

            if not frame_read:
                cv2.destroyAllWindows()
                cap.release()
                break

            original_image = frame.copy()
            self.put_text(original_image, 'ORIGINAL IMAGE')

            processed_image = self.reduce_glare(frame)
            self.put_text(processed_image, 'REDUCED GLARE')

            result = np.hstack((original_image, processed_image))

            cv2.imshow('Reducing glare', result)
            cv2.waitKey(10)


if __name__ == '__main__':
    gr = GlareReduction(VIDEO_PATH)
    gr.run()
