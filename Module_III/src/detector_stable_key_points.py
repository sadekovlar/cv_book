import copy
import cv2
import numpy as np

COLORS = [(255, 100, 0), (30, 100, 150), (255, 0, 255), (0, 100, 250), (255, 255, 0), (32, 255, 32)]
DISPLAY_SPEC = {
    'font': cv2.FONT_HERSHEY_PLAIN,
    'scale': 1.0,
    'color': (0, 255, 0),  # Green
    'thickness': 2
}
SHOW_RES = True


# Построение Гауссовой пирамиды масштабов с отличными от 2 масштабами
# На основе стандартной функции cv2.pyrDown
# https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff
# @src - исходное изображение в любом цветовом формате
# @scale - может быть как массивом, тогда на каждое понижение будет брать свой масштаб,
# либо просто число, тогда везде будет применяться одно и тоже значение
# @amount - количество изображений в пирамиде
# @gaussian_kernal_size - размер окна при blur
# return results где results = [src, pyr1, pyr2, ..., pyrn] где n = amount
def get_down_pyramids(src, scale=1.2, amount=3, gaussian_kernal_size=(3, 3)):
    # Результаты пирамиды масштабов
    results = [src]
    scales = generate_list_of_scales(scale, amount)
    if len(scales) == 0:
        return []
    for i in range(amount):
        rows, cols, _ = results[i].shape
        scale = scales[i]
        blur = cv2.GaussianBlur(results[i], gaussian_kernal_size, 0)
        pyr = cv2.resize(blur, dsize=(int(cols // scale), int(rows // scale)),
                         interpolation=cv2.INTER_LINEAR)
        results.append(pyr)
    return results


# Детектирование ключевых точек с помощью метода OBR
# https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
def get_key_points(imgs):
    orb = cv2.ORB_create()
    key_points = []
    for img in imgs:
        kp = orb.detect(img, None)
        key_points.append(kp)
    return key_points


# Поиск стабильных точек на пирамиде масштабов
def find_stable_points_on_pyr_scales(imgs_shape, key_points_, scale, search_radius=3):
    key_points = key_points_.copy()
    scales = generate_list_of_scales(scale, len(key_points) - 1)
    if len(scales) == 0:
        return []
    for pos in reversed(range(1, len(key_points))):
        # Делаем маску из точек
        mask = np.zeros((imgs_shape[pos][0], imgs_shape[pos][1], 1), np.uint8)
        r = search_radius
        # генерация r -радиуса поиска
        for i in range(pos):
            r /= scales[i]
        r = int(r)
        for point in key_points[pos]:
            pt = (int(point.pt[0]), int(point.pt[1]))
            cv2.circle(mask, pt, r, 255, -1)
        # Смотрим на уровень выше, какие точки попали у нас
        index_of_stable_points = []
        for id, point in enumerate(key_points[pos - 1]):
            y, x = int(point.pt[0] / scales[pos - 1]), int(point.pt[1] / scales[pos - 1])
            value = mask.item(x, y, 0)
            if value > 0:
                index_of_stable_points.append(id)
                cv2.circle(mask, (y, x), r, (0, 0, 0),
                           -1)  # TODO(Заменить на KD-дерево или другой поиск. Issue: Некорректно работает с увеличением радиуса поиска)

        stable_key_points = [point for id, point in enumerate(key_points[pos - 1]) if id in index_of_stable_points]
        # Перезаписываем на стабильные точки
        key_points[pos - 1] = stable_key_points

    return stable_key_points


def upscale_points(key_points, scale):
    upscale_key_points = key_points.copy()
    scales = generate_list_of_scales(scale, len(key_points) - 1)
    multipliers = []
    for pos in reversed(range(0, len(key_points))):
        # коэффициент на который необходимо домножить координаты
        multiplier = 1
        for i in range(pos):
            multiplier *= scales[i]
        multipliers.append(multiplier)
        if not multiplier == 1:
            for point in upscale_key_points[pos]:
                point.pt = (point.pt[0] * multiplier, point.pt[1] * multiplier)
    multipliers.reverse()
    return upscale_key_points, multipliers


# Вспомогательные функции
def generate_list_of_scales(scale, amount):
    scales = scale
    # Проверка, что есть scale на каждое изображение
    if type(scale) == type(list()):
        if not len(scale) == amount:
            print('Массив scale должен совпадать с количеством запрошенным количеством изображений')
            return []
    else:
        scales = [scale] * amount
    return scales


def draw_points(frame, upscaled_points, stable_key_points, mults=[]):
    res_imgs = []
    frame_with_all_key_points = copy.deepcopy(frame)
    frame_with_stable_points = copy.deepcopy(frame)
    # 1. Отрисовываем по отдельности
    for id, kp in enumerate(upscaled_points):
        img = copy.deepcopy(frame)
        if mults != []:
            cv2.putText(img, f"Pyr {id}. Scale - x{round(mults[id], 3)} Amount: {len(kp)}", (630, 520),
                        DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'], COLORS[id], DISPLAY_SPEC['thickness'])
        res_imgs.append(cv2.drawKeypoints(img, kp, None, color=COLORS[id], flags=0))

        # 2. Отрисовываем все Key points на одном frame
        frame_with_all_key_points = cv2.drawKeypoints(frame_with_all_key_points, kp, None, color=COLORS[id], flags=0)

    cv2.putText(frame_with_all_key_points, f"All key points", (630, 520),
                DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'], (0, 255, 0), DISPLAY_SPEC['thickness'])
    res_imgs.append(frame_with_all_key_points)

    # 3. Stable key points
    frame_with_stable_points = cv2.drawKeypoints(frame_with_stable_points, stable_key_points, None, color=(0, 255, 0),
                                                 flags=0)
    cv2.putText(frame_with_stable_points, f"Stable points. Amount: {len(stable_key_points)}", (630, 520),
                DISPLAY_SPEC['font'], DISPLAY_SPEC['scale'], (0, 255, 0), DISPLAY_SPEC['thickness'])
    res_imgs.append(frame_with_stable_points)

    return res_imgs
