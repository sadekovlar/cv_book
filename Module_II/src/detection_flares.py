import cv2
import numpy as np


# Получение маски бликов (glares) с использованием цветовой модели HLS (https://en.wikipedia.org/wiki/HSL_and_HSV)
# img - изображение подается в BGR
# horizon_line - линия горизонта, передается в пикселях
# lightness - по умолчанию на 90%, т.е. мы отсекаем слишком яркие объекты, которые скорее всего являются бликами
def get_mask_of_glares(img, horizon_line=0, lightness=0.9):
    low_threshold = int(255 * lightness)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Применяем пороговую бинаризацию по каналу яркости (Lightness)
    _, mask = cv2.threshold(hls[:, :, 1], low_threshold, 255, cv2.THRESH_BINARY)
    # Убираем маску выше линии горизонта
    if 0 < horizon_line <= img.shape[1]:
        mask[0:horizon_line, 0:mask.shape[1]] = 0
    # белым цветом - блики
    return mask


# Рисуем блики на изображении
def draw_glares(img, mask, color=(0, 255, 0)):
    colored_background = np.zeros(img.shape, np.uint8)
    colored_background[::] = color
    bk = cv2.bitwise_or(colored_background, colored_background, mask=mask)
    mask = cv2.bitwise_not(mask)
    fg = cv2.bitwise_or(img, img, mask=mask)
    result = cv2.bitwise_or(fg, bk)
    return result


# Подавляет яркость пикселей. Чисто визуальный эффект
def suppress_lightness(img, mask, quality=0.9):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    non_zero = cv2.findNonZero(mask)
    for idx in non_zero:
        y, x = idx[0][0], idx[0][1]
        new_value = hls.item(x, y, 1) * quality
        hls.itemset(x, y, 1, new_value)
    result = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    return result
