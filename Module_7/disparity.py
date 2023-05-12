import numpy as np
import cv2
from pathlib import Path
import time

def _load_images(filepath):
        """
        Функция создает массив изображений из видеозаписи
        Параметры:
        filepath (str): путь к каталогу с видео
        Возвращает:
        images (list): изображения
        """
        images = list()
        cap = cv2.VideoCapture(filepath)
        i=0 #переменная отслеживает каждый третий кадр
        while cap.isOpened():
            succeed, frame = cap.read()
            if succeed:
                # if i == 2: #в массив изображений добавляется лишь каждый третий кадр
                #     images.append(frame)
                #     i=0
                # else: i+=1
                images.append(frame)
            else:
                cap.release()       
        return np.array(images)

def calculate_disparity(imagesL, imagesR):
    num_frames = len(imagesL)

    # Создаем объект StereoBM для расчета диспаритета
    # Чем больше значение numDisparities, тем больше диапазон глубин будет рассматриваться
    # blockSize  размер окна, используемого для сопоставления блоков между левым и правым изображениями
    stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    disparities = []

    for i in range(num_frames):
        # Загружаем изображения с левой и правой камеры
        imgL = imagesL[i]
        imgR = imagesR[i]

        # Переводим изображения в оттенки серого
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Вычисляем диспаритет
        disparity = stereo_bm.compute(grayL, grayR)

        # Нормализуем диспаритет для визуализации
        normalized_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        disparities.append(normalized_disparity)

    return disparities


def main():
    # Загружаем видео с левой и правой камеры
    pathL = str(Path('data','kem.011.001.left.avi'))
    pathR = str(Path('data','kem.011.001.right.avi'))
    imagesL = _load_images(pathL)
    imagesR = _load_images(pathR)
    
    # Рассчитываем диспаритет
    disparities = calculate_disparity(imagesL, imagesR)
    
    # Визуализируем диспаритет
    for disparity in disparities:
        cv2.imshow('Disparity', disparity)
        if cv2.waitKey(200) & 0xFF == ord('q'):  # Ожидаем 200 миллисекунд (5 кадров в секунду)
            break
        time.sleep(0.2)  # Задержка между кадрами

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    pass
