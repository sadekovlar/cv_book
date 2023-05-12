import numpy as np
import cv2
from pathlib import Path
import time
import calib

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

def preprocess_image(image):
    # Произведите необходимую предварительную обработку изображения
    # (например, сглаживание, устранение шума и т. д.)
    
    # Сглаживание изображения с помощью фильтра Гаусса
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # В этом примере просто преобразуем изображение в оттенки серого
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Улучшение контраста изображения
    image = cv2.equalizeHist(image)
    return image


def main():
    # Загружаем видео с левой и правой камеры
    pathL = str(Path('data','kem.011.001.left.avi'))
    pathR = str(Path('data','kem.011.001.right.avi'))
    imagesL = _load_images(pathL)
    imagesR = _load_images(pathR)

    # Параметры алгоритма стереозрения
    num_disparities = 112
    block_size = 5
    
    # Создание объекта для вычисления карты глубины
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    # Обработка изображений и вычисление карты глубины
    depth_maps = []
    for imgL, imgR in zip(imagesL, imagesR):
        grayL = preprocess_image(imgL)
        grayR = preprocess_image(imgR)

        disparity = stereo.compute(grayL, grayR)
        depth_map = np.zeros_like(disparity, dtype=np.float32)
        depth_map[disparity > 0] = (num_disparities * block_size) / disparity[disparity > 0]

        depth_maps.append(depth_map)

    # Отображение карты глубины
    for depth_map in depth_maps:
        cv2.imshow('Depth Map', depth_map)
        if cv2.waitKey(200) & 0xFF == ord('q'):  # Ожидаем 200 миллисекунд (5 кадров в секунду)
            break
        time.sleep(0.2)  # Задержка между кадрами

    cv2.destroyAllWindows()

    pass

if __name__ == '__main__':
    main()
    pass
